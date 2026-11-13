"""Bounded independent supplied-NumPy Ministral graph; no numerical imports.

The caller admits NumPy and authenticates conversion/fixture/protocol inputs.
References stream one stage at a time and never retain a full decoded model.
"""
from decimal import localcontext
from contextlib import ExitStack
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import ministral3_graph_protocol as protocol
import ministral3_rotary_fixture as rotary
import ministral3_quantize as quant
import granite_reference_weights as saved_weights
import ministral3_convert_audit as input_bytes
import ministral3_convert_layout as layout
import minicpm5_context_runtime as runtime

need = protocol.need
f32 = protocol.f32
ROOT = Path(__file__).resolve().parents[2]
FLOOR = 2 * 1024**3

def tile_bindings(planned, rows, norms):
    """Selected 224-row receipts with exact rich component layout identities."""
    need(type(rows) is list and type(norms) is list, 'Explicit complete tile/norm receipts required')
    expected = {}
    for name, matrix in planned['matrices'].items():
        count, columns = matrix['shape']
        for first in range(0, count, quant.ROWS):
            expected[name, first] = min(quant.ROWS, count - first), columns, matrix
    tiles = {}
    for row in rows:
        need(type(row) is dict and type(row.get('name')) is str
             and type(row.get('first_row')) is int, 'Typed conversion tile identity required')
        key = row['name'], row['first_row']
        need(key in expected and key not in tiles, 'Unique selected conversion tile required')
        count, columns, matrix = expected[key]
        for field, value in [('rows', count), ('columns', columns), ('bits', matrix['bits'])]:
            need(type(row.get(field)) is int and row[field] == value, 'Exact conversion geometry required: ' + field)
        fields = row.get('fields')
        need(type(fields) is dict and set(fields) == set(matrix['fields']), 'Complete three-component receipt required')
        pins = {}
        for field, declaration in matrix['fields'].items():
            pin = fields[field]
            spec = planned['header'][declaration['name']]
            start = planned['origin'] + spec['data_offsets'][0] + key[1] * declaration['row_stride']
            need(type(pin) is dict and set(pin) == {'name', 'dtype', 'shape', 'offset', 'bytes', 'sha256'}
                 and pin['name'] == declaration['name'] and pin['dtype'] == spec['dtype']
                 and pin['shape'] == [count, spec['shape'][1]]
                 and all(type(v) is int for v in pin['shape'])
                 and type(pin['offset']) is int and pin['offset'] == start
                 and type(pin['bytes']) is int and pin['bytes'] == count * declaration['row_stride'],
                 'Exact component name/dtype/shape/offset/extent required')
            pins[field] = dict(bytes=pin['bytes'], sha256=saved_weights.sha(pin['sha256']))
        tiles[key] = dict(rows=count, fields=pins)
    need(set(tiles) == set(expected), 'Complete selected matrix tile coverage required')
    expected_norms = {name for name, shape in planned['logical'].items() if len(shape) == 1}
    norm_pins = {}
    for row in norms:
        need(type(row) is dict and type(row.get('name')) is str and row['name'] in expected_norms
             and row['name'] not in norm_pins, 'Unique complete norm receipt required')
        name, pin = row['name'], row.get('child')
        spec = planned['header'][name]
        start, end = spec['data_offsets']
        need(type(pin) is dict and set(pin) == {'name', 'dtype', 'shape', 'offset', 'bytes', 'sha256'}
             and pin['name'] == name and pin['dtype'] == spec['dtype'] == 'BF16'
             and pin['shape'] == spec['shape'] and all(type(v) is int for v in pin['shape'])
             and type(pin['offset']) is int and pin['offset'] == planned['origin'] + start
             and type(pin['bytes']) is int and pin['bytes'] == end - start
             and type(row.get('bytes')) is int and row['bytes'] == end - start
             and row.get('copied_BF16_sha256') == pin['sha256'], 'Exact saved norm identity required')
        norm_pins[name] = dict(bytes=end - start, sha256=saved_weights.sha(pin['sha256']))
    need(set(norm_pins) == expected_norms, 'Complete selected norm coverage required')
    return tiles, norm_pins


class Child(saved_weights.Child):
    """Reuse the reviewed exact-FD/whole-SHA/endpoint reader, with 224-row plans.

    The complete conversion manifest/journal identities must be pinned externally.
    This reader independently binds and authenticates every consumed component;
    it does not replace saved-byte source/journal qualification.
    """
    def __init__(self, path, planned, expected_file, tiles, norms):
        need(type(expected_file) is dict and set(expected_file) == {'bytes', 'sha256'}
             and type(expected_file['bytes']) is int
             and 0 < expected_file['bytes'] == planned['file_bytes'] <= 6 * 1024**3,
             'Exact bounded complete child file identity required')
        self.expected_sha = saved_weights.sha(expected_file['sha256'])
        self.tiles, self.norms = tile_bindings(planned, tiles, norms)
        self.path, self.planned = Path(path), planned
        self.fd = self.open_identity = None
        self.component_reads = self.norm_reads = self.consumed_bytes = 0


class PackedWeights:
    """One authenticated child and at most one 224-row reconstruction stripe."""
    def __init__(self, child):
        self.child = child

    def physical(self, name):
        need(type(name) is str and name.startswith('model.'), 'Logical text weight name required')
        physical = 'language_model.' + name
        need(physical in self.child.planned['logical'], 'Exact physical text weight required')
        return physical

    def shape(self, name):
        return tuple(self.child.planned['logical'][self.physical(name)])

    def rows(self, name, first, count, np):
        physical = self.physical(name)
        matrix = self.child.planned['matrices'][physical]
        need(type(first) is int and first % quant.ROWS == 0
             and type(count) is int and count == self.child.tiles[physical, first]['rows'],
             'Exact selected complete row stripe required')
        raw = quant.reconstruct_f32(self.child.tile(physical, first), count,
                                   matrix['shape'][1], matrix['bits'], np)
        return np.frombuffer(raw, dtype='<f4').reshape(count, matrix['shape'][1])

    def norm(self, name, np):
        raw = np.frombuffer(self.child.norm(self.physical(name)), dtype='<u2')
        need(not np.any((raw & 0x7f80) == 0x7f80), 'Finite BF16 norm required')
        values = (raw.astype('<u4') << 16).view('<f4').astype(np.float16)
        need(np.isfinite(values).all(), 'Norm exceeds Half storage')
        return values

    def gather(self, tokens, np):
        name = 'model.embed_tokens.weight'
        rows, columns = self.shape(name)
        output = np.empty((len(tokens), columns), dtype=np.float16)
        previous, tile = -1, None
        for index, token in enumerate(tokens):
            first = token // quant.ROWS * quant.ROWS
            if first != previous:
                tile = self.rows(name, first, min(quant.ROWS, rows - first), np)
                previous = first
            output[index] = tile[token - first]
        need(np.isfinite(output).all(), 'Embedding exceeds Half storage')
        return output


def project(np, owner, name, x, precision, output_dtype=None):
    rows, columns = owner.shape(name)
    need(x.ndim == 2 and x.dtype == np.float16 and x.shape[1] == columns
         and 1 <= len(x) <= 512 and rows <= 131072 and columns <= 9216,
         'Explicit bounded Half projection geometry required')
    output = np.empty((len(x), rows), dtype=np.float16 if output_dtype is None else output_dtype)
    lhs = x.astype(precision)
    for first in range(0, rows, quant.ROWS):
        count = min(quant.ROWS, rows - first)
        weight = owner.rows(name, first, count, np)
        need(weight.shape == (count, columns) and weight.dtype == np.float32
             and np.isfinite(weight).all(), 'Finite independent FP32 weight stripe required')
        reduced = lhs @ weight.astype(precision, copy=False).T
        need(np.isfinite(reduced).all(), 'Finite projection reduction required')
        output[:, first:first + count] = reduced
    need(np.isfinite(output).all(), 'Projection exceeds its storage precision')
    return output


def table_cache(count, offset, wide):
    """Same stored coefficient/gain boundary; no native coefficient substitution."""
    periods = rotary.periods_wide(rotary.GEOMETRY) if wide else rotary.periods_main(rotary.GEOMETRY)
    coefficients, gains = [], []
    with localcontext() as context:
        context.prec = 80
        for position in range(offset, offset + count):
            gains.append(rotary.from_half(rotary.gain_wide(position, rotary.GEOMETRY)
                                         if wide else rotary.gain_main(position, rotary.GEOMETRY)))
            row = []
            for word in periods:
                period = rotary.from_f32(word)
                if wide:
                    cosine, sine = rotary.decimal_sincos(rotary.Decimal(position) / rotary.widen(period))
                    row.append((rotary.from_f32(rotary.d32(cosine)), rotary.from_f32(rotary.d32(sine))))
                else:
                    angle = f32(f32(position) / period)
                    row.append((f32(math.cos(angle)), f32(math.sin(angle))))
            coefficients.append(row)
    return coefficients, gains


def numpy_rotary(np, x, tables, queries, precision):
    coefficients, gains = tables
    need(type(queries) is bool and x.dtype == np.float16 and x.shape[-1] == 128,
         'Explicit query/key split-half input required')
    c = np.asarray(coefficients, dtype=precision)[:, None]
    first, second = x[..., :64].astype(precision), x[..., 64:].astype(precision)
    cosine, sine = c[..., 0], c[..., 1]
    # Separate products/adds, then Half rotation store BEFORE the Half gain.
    rotated = np.concatenate((first * cosine - second * sine,
                              second * cosine + first * sine), axis=-1).astype(np.float16)
    if queries:
        gain = np.asarray(gains, dtype=np.float16)[:, None, None]
        rotated = (rotated.astype(precision) * gain.astype(precision)).astype(np.float16)
    return rotated


def numpy_norm(np, x, weight, precision, epsilon):
    wide = x.astype(precision)
    mean = np.mean(wide * wide, axis=-1, keepdims=True, dtype=precision)
    inverse = precision(1) / np.sqrt(mean + precision(np.float32(epsilon)))
    return (wide * inverse * weight.astype(precision)).astype(np.float16)


def numpy_attention(np, q, k, v, precision):
    count, heads, dimension = q.shape
    need(dimension == 128 and heads % k.shape[1] == 0 and k.shape == v.shape,
         'Exact independent grouped-query geometry required')
    output = np.empty((count, heads, dimension), dtype=np.float16)
    causal = np.tri(count, dtype=bool)
    scale = precision(np.float32(1 / math.sqrt(dimension)))
    for head in range(heads):
        kv = head // (heads // k.shape[1])
        score = (q[:, head].astype(precision) @ k[:, kv].astype(precision).T * scale).astype(np.float32)
        valid = np.where(causal, score.astype(precision), -np.inf)
        exponential = np.exp(valid - np.max(valid, axis=-1, keepdims=True))
        probability = (exponential / np.sum(exponential, axis=-1, keepdims=True, dtype=precision)).astype(np.float32)
        output[:, head] = probability.astype(precision) @ v[:, kv].astype(precision)
    return output.reshape(count, heads * dimension)


def forward(np, owner, config, tokens, offset, profile, save, diagnostic_offset=False):
    """Stream independent full/tiny stages; caller admits NumPy and retains pins.

    No model, output list or decoded weights are retained by this function. The
    supplied callback must consume each stage immediately. Inputs are independent
    reference activations throughout; substituting native inputs is forbidden.
    """
    protocol.geometry(config, tokens, offset, diagnostic_offset)
    need(type(profile) is str and profile in ('main', 'wide'), 'Explicit main/wide profile required')
    precision = np.float64 if profile == 'wide' else np.float32
    expected, seen = protocol.contracts(config, len(tokens)), set()
    tables = table_cache(len(tokens), offset, profile == 'wide')

    def emit(name, value):
        value = value.reshape(len(tokens), -1)
        need(name in expected and name not in seen and list(value.shape) == expected[name]
             and value.dtype == (np.float32 if name == 'logits' else np.float16)
             and np.isfinite(value).all(), 'Complete finite stored graph stage required: ' + name)
        save(name, value)
        seen.add(name)
        return value

    x = emit('embedding', owner.gather(tokens, np))
    for layer in range(config['layers']):
        prefix, stage = f'model.layers.{layer}.', f'layers.{layer}.'
        normalized = emit(stage + 'input_norm', numpy_norm(np, x, owner.norm(prefix + 'input_layernorm.weight', np), precision, config['rms_eps']))
        q = emit(stage + 'q', project(np, owner, prefix + 'self_attn.q_proj.weight', normalized, precision))
        k = emit(stage + 'k', project(np, owner, prefix + 'self_attn.k_proj.weight', normalized, precision))
        v = emit(stage + 'v', project(np, owner, prefix + 'self_attn.v_proj.weight', normalized, precision))
        q = emit(stage + 'q_rotary', numpy_rotary(np, q.reshape(len(tokens), config['query_heads'], 128), tables, True, precision))
        k = emit(stage + 'k_rotary', numpy_rotary(np, k.reshape(len(tokens), config['kv_heads'], 128), tables, False, precision))
        context = emit(stage + 'attention_context', numpy_attention(np, q.reshape(len(tokens), config['query_heads'], 128),
                      k.reshape(len(tokens), config['kv_heads'], 128), v.reshape(len(tokens), config['kv_heads'], 128), precision))
        branch = emit(stage + 'attention_projection', project(np, owner, prefix + 'self_attn.o_proj.weight', context, precision))
        x = emit(stage + 'attention_residual', (x.astype(precision) + branch.astype(precision)).astype(np.float16))
        del normalized, q, k, v, context, branch
        normalized = emit(stage + 'post_attention_norm', numpy_norm(np, x, owner.norm(prefix + 'post_attention_layernorm.weight', np), precision, config['rms_eps']))
        gate = emit(stage + 'gate', project(np, owner, prefix + 'mlp.gate_proj.weight', normalized, precision))
        up = emit(stage + 'up', project(np, owner, prefix + 'mlp.up_proj.weight', normalized, precision))
        values = gate.astype(precision)
        exponential = np.exp(-np.abs(values))
        denominator = precision(1) + exponential
        sigmoid = np.where(values >= 0, precision(1) / denominator, exponential / denominator)
        activated = emit(stage + 'silu', (values * sigmoid).astype(np.float16))
        gated = emit(stage + 'gated', (activated.astype(precision) * up.astype(precision)).astype(np.float16))
        down = emit(stage + 'down', project(np, owner, prefix + 'mlp.down_proj.weight', gated, precision))
        x = emit(stage + 'output', (x.astype(precision) + down.astype(precision)).astype(np.float16))
        del normalized, gate, up, values, exponential, denominator, sigmoid, activated, gated, down
    final = emit('final_norm', numpy_norm(np, x, owner.norm('model.norm.weight', np), precision, config['rms_eps']))
    emit('logits', project(np, owner, 'model.embed_tokens.weight', final, precision, np.float32))
    need(seen == set(expected), 'Complete graph stage coverage required')


def controls(np, fixture):
    """Both supplied-NumPy profiles against every frozen independent tiny array."""
    import ministral3_graph_fixture as tiny
    tiny.validate_fixture(fixture)
    decoded = tiny.decode_weights(fixture['weights'])

    class ArrayWeights:
        def __init__(self):
            self.values = {name: np.asarray(values, dtype=np.float32) for name, values in decoded.items()}

        def shape(self, name):
            return self.values[name].shape

        def rows(self, name, first, count, _):
            return self.values[name][first:first + count]

        def norm(self, name, _):
            return self.values[name].astype(np.float16)

        def gather(self, tokens, _):
            return self.values['model.embed_tokens.weight'][tokens].astype(np.float16)

    owner, records = ArrayWeights(), []
    failures, comparisons = 0, 0
    for case in fixture['cases']:
        for profile in ('main', 'wide'):
            def emit(stage, array):
                nonlocal failures, comparisons
                spec = case['arrays'][stage]
                dtype, word = ('<f4', '<u4') if stage == 'logits' else ('<f2', '<u2')
                actual = array.astype(np.dtype(dtype), copy=False).reshape(-1).astype(np.float64)
                outcomes = {}
                for field in ('main_bits', 'wide_bits'):
                    expected = np.asarray(spec[field], dtype=word).view(dtype).astype(np.float64)
                    difference = np.abs(actual - expected)
                    bound = spec['gate']['atol'] + spec['gate']['rtol'] * np.abs(expected)
                    bad = ~np.isfinite(actual) | (difference > bound)
                    count = int(np.count_nonzero(bad))
                    failures += count
                    comparisons += len(actual)
                    outcomes[field] = dict(coordinates=len(actual), failures=count,
                                           maximum_absolute_error=float(np.max(difference)),
                                           first_failure_indices=np.flatnonzero(bad)[:8].tolist())
                records.append(dict(case=case['name'], profile=profile, stage=stage, comparisons=outcomes))
            forward(np, owner, fixture['config'], case['tokens'], case['position_offset'], profile, emit,
                    case['diagnostic_offset'])
    need(len(records) == len(fixture['cases']) * 2 * 19, 'Complete tiny numerical control inventory required')
    return dict(passed=failures == 0, profile_comparisons=comparisons, failures=failures,
                stage_profile_records=records,
                scope='Independent tiny supplied-NumPy oracle controls; no native/model qualification')


class InputLease:
    """Keep each external metadata FD and exact bytes stable through execution."""
    def __init__(self):
        self.stack, self.files = ExitStack(), []

    def read(self, path, expected, maximum):
        saved = self.stack.enter_context(input_bytes.SavedFile(Path(path).absolute()))
        need(saved.before[2] <= maximum and saved.digest() == saved_weights.sha(expected),
             'Pinned bounded input metadata differs')
        self.files.append((saved, expected))
        return saved

    def document(self, path, expected, maximum=32 * 1024**2):
        saved = self.read(path, expected, maximum)
        raw = b''.join(saved.read(min(1024**2, saved.before[2] - offset), offset)
                       for offset in range(0, saved.before[2], 1024**2))
        value = input_bytes.loads(raw)
        need(type(value) is dict, 'Pinned metadata object required')
        return value

    def endpoint(self):
        result = []
        for saved, expected in self.files:
            need(saved.digest() == expected, 'External input bytes changed during execution')
            result.append(dict(path=str(saved.path), bytes=saved.before[2], sha256=expected,
                               before_identity=saved.before, after_identity=saved.check()))
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stack.__exit__(*args)


def journal_tiles(saved):
    """Stream bounded lines from the separately pinned complete journal."""
    tiles, carry, records = [], b'', 0
    for offset in range(0, saved.before[2], 1024**2):
        raw = carry + saved.read(min(1024**2, saved.before[2] - offset), offset)
        lines = raw.split(b'\n')
        carry = lines.pop()
        need(len(carry) <= 64 * 1024, 'Conversion journal line exceeds bound')
        for line in lines:
            need(0 < len(line) <= 64 * 1024, 'Exact bounded complete journal record required')
            value = input_bytes.loads(line)
            records += 1
            if value.get('event') == 'matrix_tile':
                tiles.append(value)
    need(not carry and records == 4712 and len(tiles) == 4252, 'Complete selected conversion journal required')
    saved.check()
    return tiles


def sources():
    """Explicit cold numerical driver closure plus all frozen tiny inputs."""
    names = ('ministral3_graph_reference.py', 'ministral3_graph_check.py',
             'ministral3_graph_protocol.py', 'ministral3_graph_fixture.py',
             'ministral3_rotary_fixture.py', 'ministral3_quantize.py',
             'ministral3_convert_layout.py', 'ministral3_convert_audit.py',
             'granite_reference_weights.py', 'granite_quantize.py',
             'granite_convert_layout.py', 'minicpm5_parent.py', 'minicpm5_quantize.py',
             'check_granite_reference.py', 'minicpm5_context_runtime.py',
             'reference_runtime.py', 'text_reference_runtime.py')
    selected = {ROOT / 'tools/reference' / name for name in names}
    selected.update({ROOT / 'tests/fixtures/ministral3-graph-v1' / name for name in ('fixture.json', 'manifest.json')})
    selected.add(ROOT / rotary.PROFILE_FILE)
    return sorted(selected)


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_array(path, value, dtype, np):
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    raw = memoryview(contiguous).cast('B')
    digest = hashlib.sha256()
    with Path(path).open('xb') as stream:
        for first in range(0, len(raw), 1024**2):
            block = raw[first:first + 1024**2]
            need(stream.write(block) == len(block), 'Exact reference array write required')
            digest.update(block)
        stream.flush()
        os.fsync(stream.fileno())
    with input_bytes.SavedFile(Path(path).absolute(), len(raw)) as saved:
        need(saved.digest() == digest.hexdigest(), 'Saved reference array readback differs')
    return dict(bytes=len(raw), sha256=digest.hexdigest(), shape=list(value.shape), dtype=dtype)


def final_child_hash(child):
    digest = hashlib.sha256()
    size = child.planned['file_bytes']
    for offset in range(0, size, 1024**2):
        digest.update(saved_weights.read_exact(child.fd, min(1024**2, size - offset), offset))
    need(os.pread(child.fd, 1, size) == b'' and digest.hexdigest() == child.expected_sha,
         'Complete child bytes changed after reference execution')
    return dict(**child.endpoint(), after_sha256=digest.hexdigest())


def capture(args, output, np):
    import ministral3_graph_check as checker
    with InputLease() as lease:
        prospective = checker.validate_protocol(lease.document(args.protocol, args.protocol_sha256))
        contract = lease.document(args.contract, args.contract_sha256)
        conversion = lease.document(args.conversion_manifest, args.conversion_manifest_sha256)
        need(conversion.get('format') == 'toklet-ministral3-mixed-affine-conversion-v1'
             and conversion.get('passed') is True and conversion.get('contract_sha256') == args.contract_sha256
             and conversion.get('matrix_count') == 183 and conversion.get('norm_count') == 53
             and conversion['child']['sha256'] == prospective['model']['data_sha256']
             and conversion['child']['config_sha256'] == prospective['model']['config_sha256'],
             'Exact successful selected mixed child conversion required')
        recipe = contract['recipe']
        matrices, norms = layout.selected_recipe()
        input_bytes.same(recipe['matrices'], matrices, 'Frozen selected matrix recipe differs')
        input_bytes.same(recipe['norms'], norms, 'Frozen selected norm recipe differs')
        planned = layout.output_layout(matrices, norms)
        need(planned['file_bytes'] == conversion['child']['bytes']
             and hashlib.sha256(planned['prefix']).hexdigest() == conversion['child']['header_sha256'],
             'Complete selected physical child layout differs')
        journal = conversion['journal']
        saved_journal = lease.read(args.conversion_journal, journal['sha256'], 16 * 1024**2)
        need(saved_journal.before[2] == journal['bytes'], 'Exact conversion journal extent required')
        tiles = journal_tiles(saved_journal)
        model = Path(args.model)
        config = lease.document(model / 'config.json', prospective['model']['config_sha256'], 1024**2)
        text = config['text_config']
        declared = dict(hidden=text['hidden_size'], intermediate=text['intermediate_size'], layers=text['num_hidden_layers'],
                        vocabulary=text['vocab_size'], query_heads=text['num_attention_heads'], kv_heads=text['num_key_value_heads'],
                        head_dim=text['head_dim'], theta=float(text['rope_parameters']['rope_theta']),
                        rms_eps=float(text['rms_norm_eps']), tied_head=text['tie_word_embeddings'])
        input_bytes.same(declared, protocol.REAL, 'Exact selected real reference geometry required')
        # Hash-pinned config bytes bind all non-geometric source semantics too;
        # the recipe and previous conversion remain separate mandatory gates.
        one_raw = sum(sum(math.prod(shape) * (4 if stage == 'logits' else 2)
                          for stage, shape in protocol.contracts(protocol.REAL, len(case['tokens'])).items())
                      for case in prospective['cases'])
        need(shutil.disk_usage(output).free >= 2 * one_raw + FLOOR, 'Complete reference reservation exceeds disk floor')
        arrays, cases, published = {}, [], 0
        expected_file = {key: conversion['child'][key] for key in ('bytes', 'sha256')}
        with Child(model / 'model.safetensors', planned, expected_file, tiles, conversion['exact_norms']) as child:
            owner = PackedWeights(child)
            for case in prospective['cases']:
                for profile in ('main', 'wide'):
                    folder = output / case['name'] / profile
                    folder.mkdir(parents=True)
                    started, seen, last = time.monotonic_ns(), 0, None

                    def emit(stage, value):
                        nonlocal published, seen, last
                        need(shutil.disk_usage(output).free >= 2 * one_raw - published + FLOOR,
                             'Remaining reference reservation exceeds disk floor')
                        key = case['name'] + '.' + profile + '.' + stage
                        need(key not in arrays, 'Unique full reference stage required')
                        path = folder / (stage + '.bin')
                        entry = write_array(path, value, '<f4' if stage == 'logits' else '<f2', np)
                        entry['file'] = str(path.relative_to(output))
                        arrays[key] = entry
                        published += entry['bytes']
                        seen += 1
                        if stage == 'logits':
                            row = value[-1]
                            order = np.lexsort((np.arange(len(row)), -row))[:2]
                            last = dict(final_logits_sha256=hashlib.sha256(row.astype('<f4', copy=False).tobytes()).hexdigest(),
                                        top_two_ids=order.tolist(), margin=float(row[order[0]]) - float(row[order[1]]))
                        if stage.endswith('.output'):
                            print(json.dumps(dict(event='reference_layer', case=case['name'], profile=profile, stage=stage)), flush=True)

                    forward(np, owner, protocol.REAL, case['tokens'], case['offset'], profile, emit)
                    need(seen == 419 and last is not None, 'Every selected full-model native-visible stage required')
                    cases.append(dict(name=case['name'], profile=profile, arrays=seen,
                                      diagnostic_graph_and_publication_ns=time.monotonic_ns() - started, **last))
            consumed = final_child_hash(child)
        need(len(arrays) == 4190 and published == 2 * one_raw, 'Complete independent profile byte publication required')
        return dict(passed=True, protocol_sha256=args.protocol_sha256, contract_sha256=args.contract_sha256,
                    conversion_manifest_sha256=args.conversion_manifest_sha256,
                    conversion_journal_sha256=journal['sha256'], model=prospective['model'],
                    geometry=protocol.REAL, numerical_policy=protocol.POLICY,
                    cases=cases, arrays=arrays, published_bytes=published,
                    source_consumption=consumed, input_endpoints=lease.endpoint(),
                    native_compared=False, graph_qualified=False, quality_qualified=False,
                    performance_qualified=False, managed_adoption_qualified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inventory', 'controls', 'capture', 'compare'), required=True)
    for name in ('environment-root', 'fresh-output'):
        parser.add_argument('--' + name, required=True)
    for name in ('runtime-manifest', 'runtime-sha256', 'protocol', 'protocol-sha256', 'contract', 'contract-sha256',
                 'conversion-manifest', 'conversion-manifest-sha256', 'conversion-journal', 'model',
                 'reference', 'reference-sha256', 'native', 'native-sha256'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    required = {'inventory': (), 'controls': ('runtime_manifest', 'runtime_sha256'),
                'capture': ('runtime_manifest', 'runtime_sha256', 'protocol', 'protocol_sha256',
                            'contract', 'contract_sha256', 'conversion_manifest', 'conversion_manifest_sha256',
                            'conversion_journal', 'model'),
                'compare': ('runtime_manifest', 'runtime_sha256', 'protocol', 'protocol_sha256',
                            'reference', 'reference_sha256', 'native', 'native_sha256')}[args.mode]
    need(all(getattr(args, name) for name in required), 'Complete externally pinned reference arguments required')
    output = Path(args.fresh_output)
    need(not output.exists() and not output.is_symlink(), 'Fresh reference output required')
    for candidate in (args.model, args.reference, args.native, args.conversion_manifest):
        if candidate:
            source = Path(candidate).resolve()
            source = source if source.is_dir() else source.parent
            need(not output.resolve().is_relative_to(source) and not source.is_relative_to(output.resolve()),
                 'Reference output must not overlap immutable input assets')
    need(os.environ.get('MallocLargeCache') == os.environ.get('MallocMediumZone') == '0',
         'Bounded reference requires launch-time MallocLargeCache=0 and MallocMediumZone=0')
    output.mkdir()
    runtime.bind_environment(args.environment_root)
    if args.mode == 'inventory':
        files = runtime.paths(sources())
        pins = {}
        for path in files:
            with input_bytes.SavedFile(path) as saved:
                pins[str(path)] = dict(bytes=saved.before[2], sha256=saved.digest())
        value = dict(format=runtime.FORMAT, versions=runtime.VERSIONS, files=pins)
        publish(output / 'runtime.json', value)
        result = dict(format='toklet-ministral3-reference-inventory-v1', passed=True, files=len(files),
                      runtime= dict(file='runtime.json', bytes=(output / 'runtime.json').stat().st_size,
                                    sha256=hashlib.sha256((output / 'runtime.json').read_bytes()).hexdigest()),
                      numerical_packages_imported=False)
    else:
        admitted, endpoint = runtime.admit(Path(args.runtime_manifest), args.runtime_sha256, sources())
        import numpy as np
        if args.mode == 'controls':
            import ministral3_graph_fixture as tiny
            result = controls(np, tiny.load_fixture(ROOT / 'tests/fixtures/ministral3-graph-v1'))
        elif args.mode == 'capture':
            result = capture(args, output, np)
        else:
            import ministral3_graph_check as checker
            with InputLease() as lease:
                prospective = checker.validate_protocol(lease.document(args.protocol, args.protocol_sha256))
                reference = lease.document(args.reference, args.reference_sha256)
                native = lease.document(args.native, args.native_sha256)
                result = checker.compare(np, prospective, args.protocol_sha256, reference, native,
                                         Path(args.reference).absolute().parent, Path(args.native).absolute().parent, output)
                result['input_endpoints'] = lease.endpoint()
        result.update(format='toklet-ministral3-independent-reference-v1', mode=args.mode,
                      admitted_runtime=admitted, endpoint_origins=endpoint(),
                      allocator_policy=dict(launch_environment={'MallocLargeCache': '0', 'MallocMediumZone': '0'},
                                            counters_reset_or_discounted=False), whole_goal_complete=False)
    publish(output / 'manifest.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in
                     ('arrays', 'cases', 'stage_profile_records', 'admitted_runtime', 'endpoint_origins', 'input_endpoints')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
