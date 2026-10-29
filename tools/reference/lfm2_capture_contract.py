"""Independent saved-operand LFM2 dispatch. Never executes a forward graph.

Private draft: the native event/state reader is a separate, fail-closed caller.
Every operand below is an authenticated captured array or original weight range.
No missing native operand is substituted with a reference/source value.
"""
import hashlib
import json
import math
import mmap
import os
import stat
import struct
from dataclasses import dataclass

import numpy as np
import lfm2_conformance_math as op
import lfm2_exact_affine as affine

F = np.float32
ATTENTION = (2, 5, 8, 10, 12, 14)
CONVOLUTION = tuple(i for i in range(16) if i not in ATTENTION)
MODEL_BYTES = 199527820
MODEL_SHA = '558353ad098c1230a8e2ea508cd34ef311a1b97a2bc4f47498f4df8229dc5c0f'
HEADER_SHA = '7c5e241d377d693fac694426ed10fb79df470864a9ec31d0ca82f3044171506d'
SOURCE_SHA = '9db9e7cd82a1a15ef60f75b2cd91b04f20d30553bb64cb55be72d2c201d0837a'
SOURCE_MANIFEST_SHA = '2654f475aedb37024b66f6c9a9988369d8ea8138a34e078505adf6fe16635133'
GATES_SHA = '7b1879ee54fd5a421d0b5f17f2d9192d291742bdc078ef2f5e90885eb293fbdf'
PROTOCOL_SHA = op.PROTOCOL_SHA256
HALF_PAD, WORD_PAD = 0x7e55, 0x5a170031
MAX_NUMERIC = 62 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, low, high, name):
    require(type(value) is int and low <= value <= high, name)
    return value


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def parse_json(raw):
    def invalid_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def strict_equal(a, b):
    """JSON equality that never treats booleans or floats as integer IDs."""
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(strict_equal(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b


class File:
    """One held regular descriptor; capped streamed identity and endpoint checks."""
    def __init__(self, path, cap, expected=None, *, allow_empty=False):
        self.path = os.path.abspath(os.fspath(path))
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        self.file = os.fdopen(fd, 'rb')
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and (0 if allow_empty else 1) <= info.st_size <= cap,
                    'regular nonempty bounded file')
            self.identity = (info.st_dev, info.st_ino, info.st_size)
            self.bytes = info.st_size
            self.sha256 = self.hash()
            if expected is not None:
                require(expected == {'bytes': self.bytes, 'sha256': self.sha256}, 'file identity')
        except BaseException:
            self.file.close()
            raise

    def hash(self):
        self.file.seek(0)
        h = hashlib.sha256()
        count = 0
        while count <= self.bytes:
            block = self.file.read(min(65536, self.bytes + 1 - count))
            if not block:
                break
            count += len(block)
            require(count <= self.bytes, 'saved file grew while hashing')
            h.update(block)
        info = os.fstat(self.file.fileno())
        require(count == self.bytes and (info.st_dev, info.st_ino, info.st_size) == self.identity,
                'saved file truncated/replaced while hashing')
        return h.hexdigest()

    def read(self, offset, count):
        integer(offset, 0, self.bytes, 'read offset')
        integer(count, 0, min(self.bytes - offset, 8 * 1024**2), 'bounded read length')
        self.file.seek(offset)
        raw = self.file.read(count)
        require(len(raw) == count, 'short saved file read')
        return raw

    def json(self, cap):
        require(self.bytes <= cap, 'JSON cap')
        self.file.seek(0)
        raw = self.file.read(self.bytes + 1)
        require(len(raw) == self.bytes and hashlib.sha256(raw).hexdigest() == self.sha256,
                'consumed JSON identity/extent')
        return parse_json(raw)

    def endpoint(self):
        info = os.stat(self.path, follow_symlinks=False)
        require(stat.S_ISREG(info.st_mode) and
                (info.st_dev, info.st_ino, info.st_size) == self.identity,
                'file path replaced during audit')
        held = os.fstat(self.file.fileno())
        require((held.st_dev, held.st_ino, held.st_size) == self.identity and
                self.hash() == self.sha256, 'held file changed during audit')

    def close(self):
        self.file.close()


@dataclass(frozen=True)
class Stage:
    name: str
    width: int
    operation: str
    inputs: tuple = ()
    tensor: str = ''
    rows: int = 0
    columns: int = 0
    layer: int = -1

    @property
    def coupled(self):
        return self.operation == 'coupled'

    @property
    def dtype(self):
        return 'f32le' if self.name == 'logits' or self.name.endswith(
            ('attention_scores', 'attention_probabilities')) else 'f16le'


def stage_inventory():
    """Concrete selected LFM2 routes; cold construction, no graph evaluation."""
    stages = [Stage('embedding_lookup', 1024, 'gather', tensor='model.embed_tokens')]
    previous = 'embedding_lookup'
    for layer in range(16):
        p = f'layers.{layer}.'
        w = f'model.layers.{layer}.'

        def add(name, width, operation, inputs=(), tensor='', rows=0, columns=0):
            stages.append(Stage(p + name, width, operation, inputs,
                                w + tensor if tensor else '', rows, columns, layer))

        add('operator_norm', 1024, 'rms', (previous,), 'operator_norm.weight')
        if layer in ATTENTION:
            for name, width, member in (('q_projection', 1024, 'q_proj'),
                                        ('k_projection', 512, 'k_proj'),
                                        ('v_projection', 512, 'v_proj')):
                add(name, width, 'projection', (p + 'operator_norm',),
                    'self_attn.' + member, width, 1024)
            add('q_norm', 1024, 'head_rms', (p + 'q_projection',), 'self_attn.q_layernorm.weight')
            add('k_norm', 512, 'head_rms', (p + 'k_projection',), 'self_attn.k_layernorm.weight')
            add('q_rotary', 1024, 'rotary', (p + 'q_norm',))
            add('k_rotary', 512, 'rotary', (p + 'k_norm',))
            add('attention_scores', 16, 'coupled')
            add('attention_probabilities', 16, 'probabilities', (p + 'attention_scores',))
            add('attention_context', 1024, 'coupled')
            add('operator_projection', 1024, 'projection', (p + 'attention_context',),
                'self_attn.out_proj', 1024, 1024)
        else:
            add('bcx_projection', 3072, 'projection', (p + 'operator_norm',),
                'conv.in_proj', 3072, 1024)
            for name in ('b', 'c', 'x'):
                add(name, 1024, 'view', (p + 'bcx_projection',))
            add('u', 1024, 'product', (p + 'b', p + 'x'))
            add('convolution', 1024, 'conv', tensor='conv.conv.weight')
            add('gated_conv', 1024, 'product', (p + 'c', p + 'convolution'))
            add('operator_projection', 1024, 'projection', (p + 'gated_conv',),
                'conv.out_proj', 1024, 1024)
        add('operator_residual', 1024, 'residual', (previous, p + 'operator_projection'))
        add('ffn_norm', 1024, 'rms', (p + 'operator_residual',), 'ffn_norm.weight')
        for name, member in (('gate_projection', 'w1'), ('up_projection', 'w3')):
            add(name, 4608, 'projection', (p + 'ffn_norm',), 'feed_forward.' + member, 4608, 1024)
        add('gate_silu', 4608, 'silu', (p + 'gate_projection',))
        add('ffn_gated', 4608, 'product', (p + 'gate_silu', p + 'up_projection'))
        add('down_projection', 1024, 'projection', (p + 'ffn_gated',), 'feed_forward.w2', 1024, 4608)
        add('output', 1024, 'residual', (p + 'operator_residual', p + 'down_projection'))
        previous = p + 'output'
    stages.extend((Stage('final_norm', 1024, 'rms', (previous,), 'model.embedding_norm.weight'),
                   Stage('logits', 65536, 'coupled')))
    require(len(stages) == 293 and sum(s.coupled for s in stages) == 13, 'fixed stage coverage')
    return tuple(stages)


STAGES = stage_inventory()
SLOTS = {s.name: i for i, s in enumerate(STAGES)}


class Weights:
    """Original bytes once; at most one <=128-row decoded stripe is owned."""
    def __init__(self, path):
        self.input = File(path, MODEL_BYTES, {'bytes': MODEL_BYTES, 'sha256': MODEL_SHA})
        try:
            require(struct.unpack('<Q', self.input.read(0, 8))[0] == 36740, 'original header size')
            raw = self.input.read(8, 36740)
            require(hashlib.sha256(raw).hexdigest() == HEADER_SHA, 'original header SHA')
            self.header = parse_json(raw)
            self.base = 36748
            spans = []
            for name, d in self.header.items():
                if name == '__metadata__':
                    continue
                require(set(d) == {'dtype', 'shape', 'data_offsets'}, 'tensor descriptor schema')
                require(d['dtype'] in ('BF16', 'U32') and d['shape'], 'tensor dtype/shape')
                size = math.prod(integer(x, 1, MODEL_BYTES, 'tensor dimension') for x in d['shape'])
                a, b = d['data_offsets']
                integer(a, 0, MODEL_BYTES - self.base, 'tensor begin')
                integer(b, a, MODEL_BYTES - self.base, 'tensor end')
                require(b - a == size * (2 if d['dtype'] == 'BF16' else 4), 'tensor extent')
                spans.append((a, b))
            spans.sort()
            require(len(spans) == 334 and spans[0][0] == 0 and
                    spans[-1][1] == MODEL_BYTES - self.base and
                    all(a[1] == b[0] for a, b in zip(spans, spans[1:])), 'complete original extents')
        except BaseException:
            self.input.close()
            raise

    def descriptor(self, name, shape, dtype):
        d = self.header[name]
        require(d['shape'] == list(shape) and d['dtype'] == dtype, 'original tensor layout')
        return d

    def layout(self, name, rows, columns):
        offsets = []
        for suffix, shape, dtype in (('.weight', (rows, columns // 8), 'U32'),
                                     ('.scales', (rows, columns // 64), 'BF16'),
                                     ('.biases', (rows, columns // 64), 'BF16')):
            d = self.descriptor(name + suffix, shape, dtype)
            offsets.append(self.base + d['data_offsets'][0])
        return affine.Layout(rows, columns, *offsets, MODEL_BYTES).validate()

    def stripes(self, name, rows, columns, start=0, stop=None):
        yield from affine.read_stripes(self.input.file, self.layout(name, rows, columns), start, stop)

    def coefficient(self, name, shape):
        d = self.descriptor(name, shape, 'BF16')
        a, b = d['data_offsets']
        words = np.frombuffer(self.input.read(self.base + a, b - a), dtype='<u2')
        value = (words.astype(np.uint32) << 16).view(F).reshape(shape)
        require(np.isfinite(value).all(), 'original nonfinite coefficient')
        return value

    def gather(self, token):
        integer(token, 0, 65535, 'literal input token')
        row, values = next(self.stripes('model.embed_tokens', 65536, 1024, token, token + 1))
        require(row == token, 'original gather range')
        return op.gather_row(values[0])

    def model_owners(self):
        """Expected335 physical upload hashes, streamed from original ranges."""
        result = []
        matrices = [('model.embed_tokens', 65536, 1024)]
        for layer in range(16):
            p = f'model.layers.{layer}.feed_forward.'
            matrices.extend(((p + 'w1', 4608, 1024), (p + 'w3', 4608, 1024),
                             (p + 'w2', 1024, 4608)))
        for layer in ATTENTION:
            p = f'model.layers.{layer}.self_attn.'
            matrices.extend(((p + 'q_proj', 1024, 1024), (p + 'k_proj', 512, 1024),
                             (p + 'v_proj', 512, 1024), (p + 'out_proj', 1024, 1024)))
        for layer in CONVOLUTION:
            p = f'model.layers.{layer}.conv.'
            matrices.extend(((p + 'in_proj', 3072, 1024), (p + 'out_proj', 1024, 1024)))
        require(len(matrices) == 93, 'original matrix inventory')
        for name, rows, columns in matrices:
            self.layout(name, rows, columns)
            for role in ('weight', 'scales', 'biases'):
                d = self.header[name + '.' + role]
                a, z = d['data_offsets']; h = hashlib.sha256()
                for offset in range(a, z, 65536):
                    h.update(self.input.read(self.base + offset, min(65536, z - offset)))
                size = z - a
                padded = size if role == 'weight' else ((size + 3) // 4) * 4
                h.update(bytes(padded - size))
                result.append(dict(index=len(result), name=name, role=role,
                                   bytes=padded, sha256=h.hexdigest()))
        gains = [('model.embedding_norm.weight', (1024,))]
        for layer in range(16):
            gains.extend((f'model.layers.{layer}.{kind}.weight', (1024,))
                         for kind in ('operator_norm', 'ffn_norm'))
        for layer in ATTENTION:
            gains.extend((f'model.layers.{layer}.self_attn.{kind}.weight', (64,))
                         for kind in ('q_layernorm', 'k_layernorm'))
        require(len(gains) == 45, 'original direct-gain inventory')
        for name, shape, role in (
                [(name, shape, 'direct_gain') for name, shape in gains] +
                [(f'model.layers.{layer}.conv.conv.weight', (1024, 3, 1), 'convolution_tap')
                 for layer in CONVOLUTION]):
            values = self.coefficient(name, shape)
            raw = values.tobytes() + struct.pack('<7I', *([WORD_PAD] * 7))
            result.append(dict(index=len(result), name=name, role=role,
                               bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        zero_bytes = (65536 + 7) * 2
        result.append(dict(index=334, name='zero_bias', role='zero_bias', bytes=zero_bytes,
                           sha256=hashlib.sha256(bytes(zero_bytes)).hexdigest()))
        require(len(result) == 335 and sum(v['bytes'] for v in result) == 199754258,
                'unique original model physical byte ledger')
        return result

    def close(self):
        self.input.close()


class Source:
    """Frozen MAIN/WIDE descriptor map and literal pre-target coefficient table."""
    def __init__(self, manifest, arrays, gates):
        self.manifest_file = self.input = self.gate_file = None
        try:
            self.manifest_file = File(manifest, 4 * 1024**2,
                                      {'bytes': 2247239, 'sha256': SOURCE_MANIFEST_SHA})
            self.input = File(arrays, 48049152, {'bytes': 48049152, 'sha256': SOURCE_SHA})
            self.gate_file = File(gates, 2 * 1024**2, {'bytes': 1046647, 'sha256': GATES_SHA})
            self.manifest = self.manifest_file.json(4 * 1024**2)
            self.groups = self.manifest['arrays']
            public = self.gate_file.json(2 * 1024**2)
            require(len(self.groups) == 3102 and set(public['groups']) == set(self.groups), 'source group inventory')
            spans = []
            for name, d in self.groups.items():
                require(public['groups'][name] == {k: d[k] for k in
                        ('dtype', 'shape', 'route', 'envelope', 'gate', 'gate_usage')}, 'literal source gate changed')
                require(d['dtype'] in ('f16', 'f32'), 'source dtype')
                size = math.prod(d['shape']) * (2 if d['dtype'] == 'f16' else 4)
                for profile in ('main', 'wide'):
                    r = d[profile]
                    require(r['bytes'] == size and 0 <= r['offset'] <= self.input.bytes - size,
                            'source descriptor range')
                    spans.append((r['offset'], r['offset'] + size))
            spans.sort()
            require(len(spans) == 6204 and spans[0][0] == 0 and spans[-1][1] == self.input.bytes and
                    all(a[1] == b[0] for a, b in zip(spans, spans[1:])), 'complete source ranges')
        except BaseException:
            for file in (self.manifest_file, self.input, self.gate_file):
                if file is not None:
                    file.close()
            raise

    def values(self, name, profile, keys=None):
        d = self.groups[name]
        r = d[profile]
        raw = self.input.read(r['offset'], r['bytes'])
        require(hashlib.sha256(raw).hexdigest() == r['sha256'], 'source range hash')
        value = np.frombuffer(raw, dtype='<f2' if d['dtype'] == 'f16' else '<f4').reshape(d['shape'])
        if keys is not None:
            require(value.ndim == 3 and 0 < keys <= value.shape[-1], 'source absolute key subset')
            value = value[..., :keys]
        result = value.astype(F)
        require(np.isfinite(result).all(), 'source finite')
        return result

    def close(self):
        self.input.close(); self.manifest_file.close(); self.gate_file.close()


class Totals:
    def __init__(self):
        self.groups = self.values = self.failures = 0
        self.max_error = self.max_fraction = 0.0
        self.first = None

    def add(self, actual, center, gate, name, semantic=None):
        require(actual.shape == center.shape and np.isfinite(actual).all() and
                np.isfinite(center).all(), 'numeric comparison shape/finite')
        require(set(gate) == {'atol', 'rtol'} and math.isfinite(gate['atol']) and
                gate['atol'] >= 1e-4 and gate['rtol'] == .001, 'literal comparison coefficients')
        error = np.abs(actual.astype(np.float64) - center.astype(np.float64))
        bound = gate['atol'] + gate['rtol'] * np.abs(center.astype(np.float64))
        ok = error <= bound
        if semantic is not None:
            require(semantic.shape == ok.shape and semantic.dtype == np.bool_, 'semantic comparison shape')
            ok &= semantic
        self.groups += 1; self.values += actual.size
        count = int(np.count_nonzero(~ok)); self.failures += count
        self.max_error = max(self.max_error, float(error.max(initial=0)))
        self.max_fraction = max(self.max_fraction, float((error / bound).max(initial=0)))
        if count and self.first is None:
            i = int(np.flatnonzero(~ok)[0])
            self.first = {'name': name, 'index': i, 'actual_word': int(actual.view(np.uint32).flat[i]),
                          'center_word': int(center.view(np.uint32).flat[i]), 'gate': dict(gate)}
        return ok

    def report(self):
        return dict(groups=self.groups, values=self.values, failures=self.failures,
                    maximum_error=self.max_error, maximum_fraction=self.max_fraction, first_failure=self.first)


class Dispatcher:
    """All 293 selected stages, evaluated only on saved native operands.

    `captured(slot)` returns exactly the same selected native row. `u(layer)`
    and `old(layer)` return actual current-chunk producer and pre-chunk bank.
    These callbacks may only retrieve authenticated records; no computations.
    """
    def __init__(self, weights, source):
        self.weights, self.source = weights, source
        self.main, self.wide, self.coupled = Totals(), Totals(), Totals()
        self.local = Totals()
        self.local_semantic_failures = self.local_numeric_failures = 0
        self.projection = {'stripes': 0, 'exact_bracket_checks': 0, 'equal_center_certificates': 0}

    def selected(self, case, absolute, token, start, end, captured, u, old):
        integer(absolute, start, end - 1, 'selected actual query')
        for slot, stage in enumerate(STAGES):
            actual = captured(slot)
            attention = stage.name.endswith(('attention_scores', 'attention_probabilities'))
            shape = (16, 1, end) if attention else (stage.width,)
            require(actual.shape == shape and actual.dtype == np.dtype('<f4'), 'selected native stage shape')
            op.array(actual, 65536)
            if stage.dtype == 'f16le':
                op.stored(actual, 65536)
            name = f'cases.{case}.row_{absolute}.{stage.name}'
            d = self.source.groups[name]
            gate = d['gate']
            main = self.source.values(name, 'main', end if attention else None)
            wide = self.source.values(name, 'wide', end if attention else None)
            self.main.add(actual, main, gate, name)
            self.wide.add(actual, wide, gate, name)
            if stage.coupled:
                self.coupled.add(actual, main, gate, name)
                continue

            def operand(i=0):
                return captured(SLOTS[stage.inputs[i]])

            operation = stage.operation
            if operation == 'projection':
                inputs = np.ascontiguousarray(operand().reshape(1, stage.columns))
                center = np.empty_like(actual); semantic = np.empty(actual.shape, dtype=bool)
                for row, weights in self.weights.stripes(stage.tensor, stage.rows, stage.columns):
                    stop = row + len(weights)
                    c, s, stats = op.projection(inputs, weights,
                                                np.ascontiguousarray(actual[row:stop].reshape(1, -1)))
                    center[row:stop] = c[0]; semantic[row:stop] = s[0]
                    self.projection['stripes'] += 1
                    for key in ('exact_bracket_checks', 'equal_center_certificates'):
                        self.projection[key] += stats[key]
                    del weights, c, s, stats
            elif operation == 'gather':
                center = self.weights.gather(token); semantic = actual.view(np.uint32) == center.view(np.uint32)
            elif operation in ('rms', 'head_rms'):
                width = 64 if operation == 'head_rms' else 1024
                gains = self.weights.coefficient(stage.tensor, (width,))
                center = op.rms(operand().reshape(-1, width), gains).reshape(actual.shape)
                semantic = actual == center
            elif operation == 'view':
                center = op.bcx_view(operand(), stage.name.rsplit('.', 1)[-1]); semantic = actual.view(np.uint32) == center.view(np.uint32)
            elif operation in ('product', 'residual'):
                center = op.binary(operand(0), operand(1), operation); semantic = actual == center
            elif operation == 'conv':
                taps = self.weights.coefficient(stage.tensor, (1024, 3, 1)).reshape(1024, 3)
                center = op.conv_row(u(stage.layer), old(stage.layer), taps, absolute - start)
                semantic = actual == center
            elif operation == 'silu':
                center, semantic = op.silu(operand(), actual)
            elif operation == 'rotary':
                center, semantic = op.rotary(operand().reshape(-1, 64), actual.reshape(-1, 64), absolute)
                center = center.reshape(actual.shape); semantic = semantic.reshape(actual.shape)
            elif operation == 'probabilities':
                center, semantic = op.probabilities(operand(), actual, absolute)
            else:
                raise ValueError('unimplemented selected stage: ' + stage.name)
            numeric, _, _ = op.numeric_gate(actual, center, gate)
            self.local_numeric_failures += int(np.count_nonzero(~numeric))
            self.local_semantic_failures += int(np.count_nonzero(~semantic))
            self.local.add(actual, center, gate, name, semantic)

    def teacher(self, index, actual):
        name = f'teacher.prefix_{index}.logits'
        require(actual.shape == (65536,) and actual.dtype == np.dtype('<f4'), 'forced teacher head')
        d = self.source.groups[name]
        main = self.source.values(name, 'main').reshape(-1)
        wide = self.source.values(name, 'wide').reshape(-1)
        self.main.add(actual, main, d['gate'], name)
        self.wide.add(actual, wide, d['gate'], name)
        self.coupled.add(actual, main, d['gate'], name)

    def finish(self):
        require((self.coupled.groups, self.coupled.values) == (303, 1965024), 'strict coupled coverage')
        require((self.local.groups, self.local.values) == (6440, 11971552), 'actual-input coverage')
        require(self.main.groups == self.wide.groups == 6743 and
                self.main.values == self.wide.values == 13936576, 'complete retained trajectory coverage')
        return {'strict_coupled': self.coupled.report(), 'actual_input': self.local.report(),
                'local_semantic_failures': self.local_semantic_failures,
                'local_numeric_failures': self.local_numeric_failures,
                'trajectory_main_diagnostic': self.main.report(), 'trajectory_wide_diagnostic': self.wide.report(),
                'projection_certificates': self.projection,
                'fatal_numerical_passed': self.coupled.failures == self.local.failures == 0}
