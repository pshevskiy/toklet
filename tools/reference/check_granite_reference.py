"""Bounded offline Granite graph comparison; never executes a model or GPU.

All three input documents require caller-supplied SHA256 pins. Original native
Half/F32 baselines and every retained patch are authenticated from descriptors;
repeats are replayed before their hashes or exact-schedule counts are trusted.
NumPy is imported only by main(), after the unchanged cold runtime admission.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys

import minicpm5_context_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]
MAX_INTEGER = (1 << 63) - 1
METADATA_LIMIT = 32 * 1024**2
BLOCK = 65536
FLOOR = 2 * 1024**3
DTYPES = {'<f2': 2, '<f4': 4}
STATE_CHECKS = {'exact_chunk_schedule', 'diagnostic_last_row_matches_logits',
                'full_logit_greedy_selection', 'schedule_all_stage_bytes_equal',
                'fully_cached_prompt_zero_work_unchanged_logits',
                'capacity_rejection_preserves_prefix', 'reset_clears_history_KV_prefix_and_distribution',
                'reset_replay_exact_schedule_and_final_logits',
                'greedy_token_readback_equals_full_logit_selection'}
STAGES = ('input_norm', 'q', 'k', 'v', 'q_rotary', 'k_rotary',
          'attention_context', 'attention_projection', 'attention_residual',
          'post_attention_norm', 'gate', 'silu', 'up', 'gated', 'down', 'output')
GEOMETRY = dict(hidden=2560, intermediate=8192, layers=40, vocabulary=100352,
                query_heads=40, kv_heads=8, head_dim=64, theta=10000000,
                rms_eps=1e-5, attention_scale=.015625, embedding_scale=1,
                residual_scale=1, logits_divisor=1, tied_head=False)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, low=0, high=MAX_INTEGER):
    need(type(value) is int and low <= value <= high, 'Bounded concrete integer required')
    return value


def sha(value):
    need(type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None,
         'External lowercase SHA256 identity required')
    return value


def unique(items):
    result = {}
    for key, value in items:
        need(key not in result, 'Duplicate JSON field: ' + key)
        result[key] = value
    return result


def json_integer(value):
    return integer(int(value), -MAX_INTEGER)


def json_float(value):
    number = float(value)
    need(math.isfinite(number), 'Nonfinite or overflowing JSON number')
    return number


def bad_number(value):
    raise ValueError('Invalid JSON numeric constant: ' + value)


def parsed(raw):
    return json.loads(raw, object_pairs_hook=unique, parse_int=json_integer,
                      parse_float=json_float, parse_constant=bad_number)


def relative(value):
    need(type(value) is str and 0 < len(value) <= 256 and '\\' not in value
         and '\x00' not in value, 'Bounded relative POSIX path required')
    parts = value.split('/')
    need(all(part not in ('', '.', '..') for part in parts)
         and not Path(value).is_absolute(), 'Canonical safe relative path required')
    return parts


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


class Files:
    """Root/ancestor identities and unique regular-file inodes, without aliases.

    Each path component is opened with O_NOFOLLOW, including the caller's root.
    Descriptor-relative traversal prevents a replaced parent from redirecting a
    read. Endpoint checks bind every opened directory back to its original name.
    """
    def __init__(self, root):
        self.root = Path(os.path.abspath(root))
        self.directories = {}
        self.claimed = set()
        self.inodes = set()
        try:
            self._directory(self.root)
        except BaseException:
            self.close()
            raise

    def _directory(self, path):
        if path in self.directories:
            return self.directories[path][0]
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
        if path == path.parent:
            fd = os.open(path, flags)
        else:
            fd = os.open(path.name, flags, dir_fd=self._directory(path.parent))
        value = os.fstat(fd)
        need(stat.S_ISDIR(value.st_mode), 'Regular directory required')
        self.directories[path] = (fd, identity(value))
        return fd

    def open(self, name, bytes, sha256):
        size, expected = bytes, sha256
        parts = relative(name)
        path = self.root.joinpath(*parts)
        parent = self._directory(path.parent)
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        fd = os.open(parts[-1], flags, dir_fd=parent)
        try:
            value = os.fstat(fd)
            need(stat.S_ISREG(value.st_mode) and value.st_nlink == 1
                 and value.st_size == integer(size), 'Exact unaliased regular file extent required')
            key = value.st_dev, value.st_ino
            need(name not in self.claimed and key not in self.inodes,
                 'Duplicate path or file inode alias')
            self.claimed.add(name)
            self.inodes.add(key)
            return Raw(self, path, parent, fd, value, sha(expected))
        except BaseException:
            os.close(fd)
            raise

    def endpoint(self):
        for path, (fd, original) in self.directories.items():
            named = (os.stat(path.name, dir_fd=self.directories[path.parent][0], follow_symlinks=False)
                     if path != path.parent else path.lstat())
            # Directory mtimes may change when an unrelated file is written;
            # inode/type identity still binds the path used by every descriptor.
            need(stat.S_ISDIR(named.st_mode) and (named.st_dev, named.st_ino) == original[:2]
                 and identity(os.fstat(fd))[:2] == original[:2], 'Capture directory changed')

    def close(self):
        for fd, _ in self.directories.values():
            os.close(fd)
        self.directories.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Raw:
    """One bounded immutable descriptor, with streaming hash/endpoint checks."""
    def __init__(self, owner, path, parent, fd, value, expected):
        self.owner, self.path, self.parent, self.fd = owner, path, parent, fd
        self.original, self.size, self.expected = identity(value), value.st_size, expected

    def read(self, offset, count):
        integer(offset, 0, self.size)
        integer(count, 0, BLOCK)
        need(count <= self.size - offset, 'Read exceeds exact array extent')
        result = bytearray()
        while len(result) < count:
            part = os.pread(self.fd, count - len(result), offset + len(result))
            need(bool(part), 'Truncated capture file')
            result.extend(part)
        return bytes(result)

    def endpoint(self):
        named = os.stat(self.path.name, dir_fd=self.parent, follow_symlinks=False)
        need(stat.S_ISREG(named.st_mode) and named.st_nlink == 1
             and identity(named) == self.original and identity(os.fstat(self.fd)) == self.original,
             'Capture file changed during consumption')

    def verify(self):
        digest = hashlib.sha256()
        for offset in range(0, self.size, BLOCK):
            digest.update(self.read(offset, min(BLOCK, self.size - offset)))
        need(os.pread(self.fd, 1, self.size) == b'' and digest.hexdigest() == self.expected,
             'Actual capture file SHA256 differs')
        self.endpoint()

    def close(self):
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def decoded(path, expected, maximum=METADATA_LIMIT):
    path = Path(os.path.abspath(path))
    with Files(path.parent) as files:
        size = path.lstat().st_size
        integer(size, 1, maximum)
        with files.open(path.name, size, expected) as source:
            raw = bytearray()
            for offset in range(0, size, BLOCK):
                raw.extend(source.read(offset, min(BLOCK, size - offset)))
            need(hashlib.sha256(raw).hexdigest() == sha(expected), 'Pinned JSON identity differs')
            value = parsed(raw)
            source.endpoint()
        files.endpoint()
    need(type(value) is dict, 'JSON document object required')
    return value


def exact(value, expected, message):
    # Python considers True == 1; structural equality must preserve JSON types.
    need(type(value) is type(expected), message)
    if type(expected) is dict:
        need(set(value) == set(expected), message)
        for key in expected:
            exact(value[key], expected[key], message)
    elif type(expected) is list:
        need(len(value) == len(expected), message)
        for left, right in zip(value, expected, strict=True):
            exact(left, right, message)
    elif type(expected) in (int, float):
        need(value == expected, message)
    else:
        need(value == expected, message)


def contracts(count):
    integer(count, 1, 512)
    widths = dict.fromkeys(STAGES, GEOMETRY['hidden'])
    widths.update(k=512, v=512, k_rotary=512)
    widths.update({name: 8192 for name in ('gate', 'silu', 'up', 'gated')})
    result = {name: [count, width] for name, width in
              [('embedding', 2560), ('final_norm', 2560), ('logits', 100352)]}
    for layer in range(40):
        result.update({f'layers.{layer}.{name}': [count, width] for name, width in widths.items()})
    return result


def fixed_cases():
    sequence = lambda count: [(7 + 17 * i) % 100352 for i in range(count)]
    return [dict(name=name, tokens=tokens, offset=offset) for name, tokens, offset in
            [('single', [100283], 0), ('short', [100283, 1, 7, 13], 0),
             ('future_changed', [100283, 1, 7, 17], 0), ('offset5', [7, 13, 3], 5),
             ('tile17', ([100283, 1, 7, 13] * 5)[:17], 0), ('offset509', [7, 13, 3], 509),
             ('context128', sequence(128), 0), ('context256', sequence(256), 0),
             ('context512', sequence(512), 0), ('context128_offset384', sequence(128), 384)]]


def validate_protocol(value):
    need(value.get('format') == 'toklet-granite-full-graph-prospective-v1', 'Frozen Granite protocol required')
    exact(value.get('cases'), fixed_cases(), 'Exact frozen ten-case token/offset inventory required')
    # Source configuration uses integer scalar literals except the two fractions.
    exact(value.get('geometry'), GEOMETRY, 'Exact Granite geometry required')
    policy = value.get('numerical_policy')
    need(type(policy) is dict and type(policy.get('atol_floor')) is float
         and policy['atol_floor'] == 1e-4 and type(policy.get('envelope_factor')) is int
         and policy['envelope_factor'] == 2 and type(policy.get('rtol')) is float
         and policy['rtol'] == 1e-3, 'Frozen per-array numerical policy required')
    native = value.get('native')
    need(type(native) is dict, 'Native protocol required')
    for key, expected in dict(capacity=512, projection='MlxAffine', normalization='Compensated',
                              attention='Reference', chunks=[128, 32, 1],
                              effective_chunk='min(requested,512-position_offset)').items():
        exact(native.get(key), expected, 'Fixed native workload policy differs: ' + key)
    model = value.get('model')
    need(type(model) is dict, 'Frozen model identities required')
    for field in ('data_sha256', 'config_sha256', 'conversion_manifest_sha256', 'conversion_journal_sha256'):
        sha(model.get(field))
    inputs = value.get('source_inputs_sha256')
    need(type(inputs) is dict and bool(inputs), 'Frozen source identity inventory required')
    for path, pin in inputs.items():
        relative(path)
        sha(pin)
    limits = value.get('memory_and_disk')
    need(type(limits) is dict, 'Fixed limits required')
    for field, expected in dict(engine_limit_bytes=6 * 1024**3,
                                reference_setup_limit_bytes=512 * 1024**2,
                                helper_reserve_bytes=32 * 1024**2,
                                disk_floor_bytes=FLOOR).items():
        exact(limits.get(field), expected, 'Frozen memory/disk limit differs')
    total = sum(sum(shape[0] * shape[1] * DTYPES['<f4' if stage == 'logits' else '<f2']
                    for stage, shape in contracts(len(case['tokens'])).items()) for case in value['cases'])
    exact(limits.get('raw_bytes_per_reference_profile_or_lossless_native_capture'), total,
          'Complete full-model byte inventory differs')
    return value


def array_spec(value, shape, reference=False):
    required = ({'file', 'shape', 'dtype', 'bytes', 'sha256'} if reference else
                {'shape', 'dtype', 'bytes', 'sha256', 'baseline_file', 'mismatching_bytes', 'patches'})
    need(type(value) is dict and set(value) == required, 'Exact array fields required')
    exact(value['shape'], shape, 'Complete stage shape differs')
    dtype = '<f4' if shape[1] == 100352 else '<f2'
    exact(value['dtype'], dtype, 'Native stage storage dtype differs')
    size = shape[0] * shape[1] * DTYPES[dtype]
    exact(value['bytes'], size, 'Exact stage byte extent differs')
    sha(value['sha256'])
    return size


def validate_reference(value, protocol, protocol_sha):
    need(value.get('format') == 'toklet-granite-independent-reference-v1'
         and value.get('mode') == 'capture' and value.get('passed') is True
         and value.get('protocol_sha256') == sha(protocol_sha), 'Successful exact reference capture required')
    for flag in ('native_compared', 'graph_qualified', 'quality_qualified', 'performance_qualified',
                 'managed_adoption_qualified', 'whole_goal_complete'):
        need(value.get(flag) is False, 'Independent reference qualification boundary differs: ' + flag)
    for field in ('numerical_policy', 'geometry'):
        exact(value.get(field), protocol[field], 'Reference protocol association differs')
    need(value.get('conversion_manifest_sha256') == protocol['model']['conversion_manifest_sha256'],
         'Reference conversion identity differs')
    sha(value.get('contract_sha256'))
    consumed = value.get('source_consumption')
    need(type(consumed) is dict and consumed.get('file_sha256') == protocol['model']['data_sha256'],
         'Reference consumed source identity differs')
    for field in ('component_reads', 'norm_reads', 'consumed_bytes'):
        integer(consumed.get(field), 1)
    arrays = value.get('arrays')
    need(type(arrays) is dict, 'Complete reference arrays mapping required')
    pending, total = set(), 0
    for case in protocol['cases']:
        for profile in ('main', 'wide'):
            for stage, shape in contracts(len(case['tokens'])).items():
                key = case['name'] + '.' + profile + '.' + stage
                pending.add(key)
                need(key in arrays, 'Missing reference array: ' + key)
                spec = arrays[key]
                total += array_spec(spec, shape, True)
                exact(spec['file'], case['name'] + '/' + profile + '/' + stage + '.bin',
                      'Canonical reference stage path required')
    need(set(arrays) == pending and len(pending) == 12860, 'Exact complete reference coverage required')
    exact(value.get('published_bytes'), total, 'Reference publication byte count differs')
    cases = value.get('cases')
    need(type(cases) is list and len(cases) == 20, 'Complete reference profile receipts required')
    seen = set()
    for entry in cases:
        need(type(entry) is dict and type(entry.get('name')) is str and type(entry.get('profile')) is str,
             'Concrete reference case receipt required')
        key = entry.get('name'), entry.get('profile')
        need(key in {(c['name'], p) for c in protocol['cases'] for p in ('main', 'wide')}
             and key not in seen, 'Unique reference profile receipt required')
        seen.add(key)
        exact(entry.get('arrays'), 643, 'Complete reference case stage count required')
        sha(entry.get('last_logits_sha256'))
    return value


def validate_patches(value, size):
    need(type(value) is list and len(value) <= size, 'Bounded concrete patch list required')
    end = 0
    for patch in value:
        need(type(patch) is dict and set(patch) == {'offset', 'bytes', 'file', 'sha256'},
             'Exact retained patch fields required')
        offset, count = integer(patch['offset'], 0, size), integer(patch['bytes'], 1, size)
        need(offset >= end and count <= size - offset, 'Sorted nonoverlapping in-bounds patches required')
        end = offset + count
        parts = relative(patch['file'])
        need(len(parts) == 1, 'Retained patch leaf required')
        sha(patch['sha256'])
    return value


def native_arrays(arrays, shapes, baseline=False):
    need(type(arrays) is dict and set(arrays) == set(shapes), 'All 643 native stages required')
    for stage, shape in shapes.items():
        spec = arrays[stage]
        size = array_spec(spec, shape)
        exact(spec['baseline_file'], stage + '.bin', 'Canonical native baseline leaf required')
        integer(spec['mismatching_bytes'], 0, size)
        validate_patches(spec['patches'], size)
        if baseline:
            need(spec['patches'] == [] and spec['mismatching_bytes'] == 0,
                 'Full original baseline cannot contain repeat patches')


def trace_receipts(entry):
    required = {'final_logits_sha256', 'final_greedy_token', 'diagnostic_last_row_matches_logits'}
    need(required <= set(entry), 'Complete final-logit SHA256/lowest-ID/state receipts required')
    sha(entry['final_logits_sha256'])
    integer(entry['final_greedy_token'], 0, 100351)
    need(type(entry['diagnostic_last_row_matches_logits']) is bool,
         'Concrete diagnostic/logit state boolean required')


def required_state_checks(name, count, offset, chunk):
    required = set(STATE_CHECKS)
    if count < 512 - offset:
        required.add('pending_sampled_token_is_uncommitted_and_discardable')
    if name == 'short':
        required.add('changed_prefix_replay_equals_fresh_session')
    if chunk == 128:
        required.add('reset_replay_all_stage_bytes_equal')
    return required


def check_trace_receipt(entry, actual):
    """Compare every mandatory receipt to the hash/argmax derived from raw data."""
    failures = [dict(field=field, reported=entry[field], actual=actual[field])
                for field in ('final_logits_sha256', 'final_greedy_token')
                if entry[field] != actual[field]]
    if entry['diagnostic_last_row_matches_logits'] is False:
        failures.append(dict(field='diagnostic_last_row_matches_logits', reported=False))
    return failures


def validate_native(value, protocol, protocol_sha):
    need(value.get('format') == 'toklet-granite-native-graph-v1'
         and value.get('protocol_sha256') == sha(protocol_sha)
         and value.get('model_data_sha256') == protocol['model']['data_sha256']
         and value.get('model_config_sha256') == protocol['model']['config_sha256'],
         'Native source/protocol identities differ')
    need(value.get('complete') is True, 'Explicit complete native capture required')
    need('error' in value and value['error'] is None, 'Explicit null native capture error required')
    need(type(value.get('state_checks_passed')) is bool, 'Concrete native state summary required')
    cases = value.get('cases')
    need(type(cases) is list and len(cases) == 30, 'All thirty native case/schedule entries required')
    expected = {(case['name'], chunk): case for case in protocol['cases'] for chunk in (128, 32, 1)}
    indexed = {}
    for entry in cases:
        need(type(entry) is dict and type(entry.get('name')) is str
             and type(entry.get('requested_chunk')) is int, 'Concrete native case/schedule required')
        key = entry['name'], entry['requested_chunk']
        need(key in expected and key not in indexed, 'Unique expected native case/schedule required')
        case = expected[key]
        exact(entry.get('position_offset'), case['offset'], 'Exact native offset required')
        exact(entry.get('tokens'), case['tokens'], 'Exact native token IDs required')
        folder = f'{key[0]}/chunk{key[1]}'
        exact(entry.get('folder'), folder, 'Canonical native schedule folder required')
        exact(entry.get('baseline_folder'), key[0] + '/chunk128', 'Canonical native baseline required')
        effective = min(key[1], 512 - case['offset'])
        exact(entry.get('effective_chunk'), effective, 'Native effective chunk differs')
        rows = [min(effective, len(case['tokens']) - first)
                for first in range(0, len(case['tokens']), effective)]
        exact(entry.get('effective_chunks'), rows, 'Actual native call row schedule differs')
        checks = entry.get('state_checks')
        need(type(checks) is dict and bool(checks)
             and all(type(name) is str and re.fullmatch('[a-zA-Z][a-zA-Z0-9_]{0,127}', name)
                     and type(verdict) is bool for name, verdict in checks.items()),
             'Nonempty concrete native state-check booleans required')
        required = required_state_checks(key[0], len(case['tokens']), case['offset'], key[1])
        need(required <= set(checks), 'Missing required native state checks: ' +
             ','.join(sorted(required - set(checks))))
        if 'state_evidence' in entry:
            need(type(entry['state_evidence']) is dict, 'Concrete native state evidence object required')
        trace_receipts(entry)
        arrays = entry.get('arrays')
        shapes = contracts(len(case['tokens']))
        native_arrays(arrays, shapes, key[1] == 128)
        need(key[1] != 128 or 'reset_replay' in entry, 'Required full-stage baseline reset trace is missing')
        if 'reset_replay' in entry:
            reset = entry['reset_replay']
            need(key[1] == 128 and type(reset) is dict
                 and set(reset) == {'folder', 'baseline_folder', 'arrays', 'effective_chunks',
                                   'effective_chunk', 'final_logits_sha256', 'final_greedy_token',
                                   'diagnostic_last_row_matches_logits'},
                 'Concrete reset replay belongs only to the baseline schedule')
            relative(reset['folder'])
            need(reset['folder'] != folder and reset['folder'].startswith(key[0] + '/'),
                 'Separate canonical case reset folder required')
            exact(reset['baseline_folder'], entry['baseline_folder'], 'Reset baseline association differs')
            exact(reset['effective_chunks'], rows, 'Reset row schedule differs')
            exact(reset['effective_chunk'], effective, 'Reset effective chunk differs')
            trace_receipts(reset)
            native_arrays(reset['arrays'], shapes)
        indexed[key] = entry
    need(set(indexed) == set(expected), 'Complete native schedule coverage required')
    return indexed


class Replay:
    """Stream a repeat using a verified baseline and at most one patch FD."""
    def __init__(self, files, baseline, folder, spec):
        self.files, self.baseline, self.folder, self.spec = files, baseline, folder, spec
        validate_patches(spec['patches'], baseline.size)

    def blocks(self):
        digest = hashlib.sha256()
        mismatches, cursor = 0, 0
        for patch in self.spec['patches']:
            while cursor < patch['offset']:
                count = min(BLOCK, patch['offset'] - cursor)
                raw = self.baseline.read(cursor, count)
                digest.update(raw)
                yield cursor, raw
                cursor += count
            with self.files.open(self.folder + '/' + patch['file'], patch['bytes'], patch['sha256']) as source:
                patch_hash = hashlib.sha256()
                changed = 0
                for offset in range(0, patch['bytes'], BLOCK):
                    count = min(BLOCK, patch['bytes'] - offset)
                    raw = source.read(offset, count)
                    previous = self.baseline.read(cursor, count)
                    difference = sum(a != b for a, b in zip(raw, previous, strict=True))
                    changed += difference
                    mismatches += difference
                    patch_hash.update(raw)
                    digest.update(raw)
                    yield cursor, raw
                    cursor += count
                need(changed > 0, 'Retained patch must contain an actual changed byte')
                need(patch_hash.hexdigest() == patch['sha256'], 'Actual retained patch SHA256 differs')
                source.endpoint()
        while cursor < self.baseline.size:
            count = min(BLOCK, self.baseline.size - cursor)
            raw = self.baseline.read(cursor, count)
            digest.update(raw)
            yield cursor, raw
            cursor += count
        need(digest.hexdigest() == self.spec['sha256'], 'Reconstructed complete native SHA256 differs')
        need(mismatches == self.spec['mismatching_bytes'], 'Exact replay mismatching-byte count differs')
        self.baseline.endpoint()


def original_blocks(source):
    digest = hashlib.sha256()
    for offset in range(0, source.size, BLOCK):
        raw = source.read(offset, min(BLOCK, source.size - offset))
        digest.update(raw)
        yield offset, raw
    need(digest.hexdigest() == source.expected, 'Actual complete baseline SHA256 differs')
    source.endpoint()


def prefix_difference(left, right, count, width, columns):
    """Exact authenticated native prefix observation, separate from all gates."""
    integer(count, 1, min(left.size, right.size))
    need(width in (2, 4) and count % width == 0, 'Complete scalar prefix extent required')
    integer(columns, 1, 100352)
    failed, examples = 0, []
    for offset in range(0, count, BLOCK):
        size = min(BLOCK, count - offset)
        a, b = left.read(offset, size), right.read(offset, size)
        failed += sum(x != y for x, y in zip(a, b, strict=True))
        if len(examples) < 8:
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                if x != y:
                    scalar = (offset + index) // width
                    examples.append(dict(byte_offset=offset + index,
                                         coordinate=[scalar // columns, scalar % columns], left=x, right=y))
                    if len(examples) == 8:
                        break
    left.endpoint()
    right.endpoint()
    return dict(bytes=count, mismatching_bytes=failed, examples=examples)


def causal_prefix(indexed, native_folder, output):
    records, failed, compared = 0, 0, 0
    with Files(native_folder) as files, (output / 'causal-prefix.jsonl').open('x') as report:
        for stage, shape in contracts(4).items():
            a = indexed['short', 128]['arrays'][stage]
            b = indexed['future_changed', 128]['arrays'][stage]
            with files.open('short/chunk128/' + a['baseline_file'], a['bytes'], a['sha256']) as left, \
                    files.open('future_changed/chunk128/' + b['baseline_file'], b['bytes'], b['sha256']) as right:
                left.verify()
                right.verify()
                width = DTYPES[a['dtype']]
                row = prefix_difference(left, right, 3 * shape[1] * width, width, shape[1])
                failed += row['mismatching_bytes']
                compared += row['bytes']
                records += 1
                row.update(stage=stage, cases=['short', 'future_changed'], rows=3, requested_chunk=128)
                report.write(json.dumps(row, separators=(',', ':')) + '\n')
        files.endpoint()
    need(records == 643, 'Complete exact causal-prefix stage coverage required')
    return dict(arrays=records, compared_bytes=compared, mismatching_bytes=failed,
                baseline_equal=failed == 0,
                scope='First three original baseline rows of short/future_changed; repeats separately require complete exact schedule replay.')


def immutable_gate(maximum):
    need(type(maximum) in (int, float) and math.isfinite(maximum) and maximum >= 0,
         'Finite independent profile difference required')
    return dict(atol=max(1e-4, 2 * maximum), rtol=1e-3)


def scalar_member(actual, reference, gate):
    """Portable gate oracle, also used by no-NumPy boundary controls."""
    return (math.isfinite(actual) and math.isfinite(reference)
            and abs(actual - reference) <= gate['atol'] + gate['rtol'] * abs(reference))


def gate_from_files(main, wide, dtype, np):
    maximum = 0.0
    main_hash, wide_hash = hashlib.sha256(), hashlib.sha256()
    for offset in range(0, main.size, BLOCK):
        count = min(BLOCK, main.size - offset)
        a, b = main.read(offset, count), wide.read(offset, count)
        main_hash.update(a)
        wide_hash.update(b)
        left = np.frombuffer(a, dtype=dtype).astype(np.float64)
        right = np.frombuffer(b, dtype=dtype).astype(np.float64)
        need(np.isfinite(left).all() and np.isfinite(right).all(), 'Nonfinite independent reference coordinate')
        maximum = max(maximum, float(np.max(np.abs(left - right))))
    need(main_hash.hexdigest() == main.expected and wide_hash.hexdigest() == wide.expected,
         'Actual independent reference array SHA256 differs')
    main.endpoint()
    wide.endpoint()
    return immutable_gate(maximum), maximum


def compare_blocks(blocks, main, wide, spec, gate, np, logits=False):
    """Float64 predicates implement the frozen formula for both profiles.

    Patch boundaries need not align to scalar words. Retain at most one word of
    carry, so arbitrary legal changed-byte blocks reconstruct the original data.
    """
    dtype, width = spec['dtype'], DTYPES[spec['dtype']]
    outcomes = {p: dict(values=0, failed_values=0, max_absolute_error=0.,
                        max_gate_fraction=0., examples=[]) for p in ('main', 'wide')}
    carry, consumed, nonfinite = b'', 0, 0
    last_start = (spec['shape'][0] - 1) * spec['shape'][1]
    last_hash, top = hashlib.sha256(), []
    for offset, raw in blocks:
        need(offset == consumed, 'Complete ordered replay stream required')
        if logits and offset + len(raw) > last_start * width:
            last_hash.update(raw[max(0, last_start * width - offset):])
        consumed += len(raw)
        raw = carry + raw
        complete = len(raw) // width * width
        carry = raw[complete:]
        if not complete:
            continue
        start = consumed - len(raw)
        values = np.frombuffer(raw[:complete], dtype=dtype).astype(np.float64)
        finite = np.isfinite(values)
        nonfinite += int(np.count_nonzero(~finite))
        if logits and start // width + len(values) > last_start:
            first = max(0, last_start - start // width)
            tail = values[first:]
            if np.isfinite(tail).all():
                order = np.lexsort((np.arange(len(tail)), -tail))[:2]
                top.extend((float(tail[i]), start // width + first + int(i) - last_start) for i in order)
                top = sorted(top, key=lambda item: (-item[0], item[1]))[:2]
        for profile, source in (('main', main), ('wide', wide)):
            # complete can be BLOCK + width - 1 after a byte-aligned patch.
            expected_raw = b''.join(source.read(start + at, min(BLOCK, complete - at))
                                    for at in range(0, complete, BLOCK))
            expected = np.frombuffer(expected_raw, dtype=dtype).astype(np.float64)
            need(np.isfinite(expected).all(), 'Nonfinite independent reference coordinate')
            with np.errstate(invalid='ignore'):
                error = np.abs(values - expected)
                limit = gate['atol'] + gate['rtol'] * np.abs(expected)
                bad = (~finite) | (error > limit)
            row = outcomes[profile]
            row['values'] += len(values)
            row['failed_values'] += int(np.count_nonzero(bad))
            if finite.any():
                row['max_absolute_error'] = max(row['max_absolute_error'], float(np.max(error[finite])))
                row['max_gate_fraction'] = max(row['max_gate_fraction'], float(np.max(error[finite] / limit[finite])))
            for index in np.flatnonzero(bad)[:max(0, 8 - len(row['examples']))]:
                position = start // width + int(index)
                row['examples'].append(dict(coordinate=[position // spec['shape'][1], position % spec['shape'][1]],
                    native=float(values[index]) if finite[index] else str(float(values[index])),
                    reference=float(expected[index]), limit=float(limit[index])))
    need(not carry and consumed == spec['bytes'], 'Complete native scalar byte coverage required')
    need(all(row['values'] == spec['shape'][0] * spec['shape'][1] for row in outcomes.values()),
         'Every native coordinate must compare against both profiles')
    main.endpoint()
    wide.endpoint()
    result = dict(references=outcomes, nonfinite_native_values=nonfinite)
    if logits:
        result.update(final_logits_sha256=last_hash.hexdigest(),
                      final_greedy_token=top[0][1] if top and nonfinite == 0 else None)
    return result


def sources():
    return sorted(ROOT / 'tools/reference' / name for name in
                  ('check_granite_reference.py', 'test_check_granite_reference.py',
                   'minicpm5_context_runtime.py', 'reference_runtime.py', 'text_reference_runtime.py'))


def comparison_status(result):
    """Required campaign coverage is independent of numerical pass/fail counts."""
    names = {case['name'] for case in fixed_cases()}
    resets = result['reset_replayed_cases']
    complete = (result['arrays'] == 19290 and result['required_native_state_checks'] == 304
                and result['unavailable_native_state_checks'] == 0
                and result['reset_replayed_arrays'] == 6430 and len(resets) == 10 and set(resets) == names
                and result['reset_unavailable_cases'] == []
                and result['final_logit_receipts_checked'] == 40
                and result['final_logit_receipts_verified'] == 40)
    passed = (complete and result['failed_profile_coordinates'] == 0
              and result['exact_schedule_mismatching_bytes'] == 0 and result['reset_mismatching_bytes'] == 0
              and result['failed_native_state_checks'] == 0 and not result['native_receipt_failures']
              and result['native_state_summary'] is True
              and result['exact_native_causal_prefix']['mismatching_bytes'] == 0)
    return dict(completed=complete, passed=passed)


def compare(reference, native, indexed, protocol, reference_folder, native_folder, output, np):
    compared, failed, nonfinite, array_count, schedule_mismatches = 0, 0, 0, 0, 0
    reset_arrays, reset_mismatches, reset_cases = 0, 0, []
    state, receipt_failures = [], []
    required_check_count, receipts_checked, receipts_verified = 0, 0, 0
    for key, entry in indexed.items():
        required = required_state_checks(key[0], len(entry['tokens']), entry['position_offset'], key[1])
        required_check_count += len(required)
        state.append(dict(case=key[0], requested_chunk=key[1], checks=entry['state_checks'],
                          failed=[k for k, v in entry['state_checks'].items() if not v],
                          unavailable=sorted(required - set(entry['state_checks'])),
                          evidence=entry.get('state_evidence')))
    reference_receipts = {(case['name'], case['profile']): case for case in reference['cases']}
    with Files(reference_folder) as refs, Files(native_folder) as actuals, (output / 'arrays.jsonl').open('x') as report:
        for case in protocol['cases']:
            name = case['name']
            for stage in contracts(len(case['tokens'])):
                a = reference['arrays'][name + '.main.' + stage]
                b = reference['arrays'][name + '.wide.' + stage]
                baseline_entry = indexed[name, 128]
                base = baseline_entry['arrays'][stage]
                with refs.open(a['file'], a['bytes'], a['sha256']) as main, \
                        refs.open(b['file'], b['bytes'], b['sha256']) as wide, \
                        actuals.open(baseline_entry['baseline_folder'] + '/' + base['baseline_file'],
                                     base['bytes'], base['sha256']) as baseline:
                    gate, spread = gate_from_files(main, wide, a['dtype'], np)
                    # Authenticate full baseline first, before any repeat replay.
                    baseline.verify()
                    if stage == 'logits':
                        for profile, source in (('main', main), ('wide', wide)):
                            digest = hashlib.sha256()
                            start = (len(case['tokens']) - 1) * 100352 * 4
                            for offset in range(start, source.size, BLOCK):
                                digest.update(source.read(offset, min(BLOCK, source.size - offset)))
                            need(digest.hexdigest() == reference_receipts[name, profile]['last_logits_sha256'],
                                 'Reference final-logit receipt differs from actual raw row')
                    for chunk in (128, 32, 1):
                        entry = indexed[name, chunk]
                        spec = entry['arrays'][stage]
                        blocks = (original_blocks(baseline) if chunk == 128 else
                                  Replay(actuals, baseline, entry['folder'], spec).blocks())
                        row = compare_blocks(blocks, main, wide, spec, gate, np, stage == 'logits')
                        if stage == 'logits':
                            problems = check_trace_receipt(entry, row)
                            receipts_checked += 1
                            receipts_verified += not problems
                            receipt_failures.extend(dict(case=name, requested_chunk=chunk, **problem)
                                                    for problem in problems)
                        mismatch = 0 if chunk == 128 else spec['mismatching_bytes']
                        schedule_mismatches += mismatch
                        count = sum(item['values'] for item in row['references'].values())
                        bad = sum(item['failed_values'] for item in row['references'].values())
                        compared += count
                        failed += bad
                        nonfinite += row['nonfinite_native_values']
                        array_count += 1
                        row.update(case=name, stage=stage, requested_chunk=chunk, shape=spec['shape'],
                                   dtype=spec['dtype'], gate=gate, independent_max_abs=spread,
                                   sha256=spec['sha256'], exact_schedule_mismatching_bytes=mismatch)
                        report.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
                    if 'reset_replay' in baseline_entry:
                        reset = baseline_entry['reset_replay']
                        spec = reset['arrays'][stage]
                        replay = Replay(actuals, baseline, reset['folder'], spec)
                        row = compare_blocks(replay.blocks(), main, wide, spec, gate, np, stage == 'logits')
                        if stage == 'logits':
                            problems = check_trace_receipt(reset, row)
                            receipts_checked += 1
                            receipts_verified += not problems
                            receipt_failures.extend(dict(case=name, requested_chunk=128, reset_replay=True, **problem)
                                                    for problem in problems)
                        reset_arrays += 1
                        reset_mismatches += spec['mismatching_bytes']
                        compared += sum(item['values'] for item in row['references'].values())
                        failed += sum(item['failed_values'] for item in row['references'].values())
                        nonfinite += row['nonfinite_native_values']
                        row.update(case=name, stage=stage, requested_chunk=128, reset_replay=True,
                                   shape=spec['shape'], dtype=spec['dtype'], gate=gate,
                                   independent_max_abs=spread, sha256=spec['sha256'],
                                   exact_schedule_mismatching_bytes=spec['mismatching_bytes'])
                        report.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
                if stage == 'logits':
                    print(json.dumps(dict(event='compared_case', case=name, arrays=array_count,
                                          failed_profile_coordinates=failed)), flush=True)
            if 'reset_replay' in indexed[name, 128]:
                reset_cases.append(name)
        refs.endpoint()
        actuals.endpoint()
    need(array_count == 19290, 'Complete thirty-schedule all-stage comparisons required')
    prefix = causal_prefix(indexed, native_folder, output)
    failed_checks = sum(len(row['failed']) for row in state)
    unavailable_checks = sum(len(row['unavailable']) for row in state)
    result = dict(arrays=array_count, profile_coordinate_comparisons=compared,
                failed_profile_coordinates=failed, nonfinite_native_values=nonfinite,
                exact_schedule_mismatching_bytes=schedule_mismatches,
                exact_native_schedule_equivalence=schedule_mismatches == 0,
                exact_native_causal_prefix=prefix,
                reset_replayed_arrays=reset_arrays, reset_replayed_cases=reset_cases,
                reset_unavailable_cases=[case['name'] for case in protocol['cases'] if case['name'] not in reset_cases],
                reset_mismatching_bytes=reset_mismatches,
                exact_native_reset_equivalence=reset_mismatches == 0 if reset_arrays else None,
                native_state_checks=state, failed_native_state_checks=failed_checks,
                required_native_state_checks=required_check_count,
                unavailable_native_state_checks=unavailable_checks, native_state_summary=native['state_checks_passed'],
                native_receipt_failures=receipt_failures,
                final_logit_receipts_checked=receipts_checked, final_logit_receipts_verified=receipts_verified,
                state_case_availability='All applicable frozen campaign checks are required. Cancellation/device-failure injection hooks are outside this contract, unavailable and unclaimed.',
                unchanged_independent_per_array_gates=True, original_raw_evidence_retained=True,
                graph_qualified=False, quality_qualified=False, performance_qualified=False,
                observer_overhead_qualified=False, managed_adoption_qualified=False, whole_goal_complete=False)
    result.update(comparison_status(result))
    return result


def publish(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('environment-root', 'runtime-manifest', 'runtime-sha256', 'protocol', 'protocol-sha256',
                 'reference', 'reference-sha256', 'native', 'native-sha256', 'fresh-output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = Path(os.path.abspath(args.fresh_output))
    need(not output.exists() and not output.is_symlink(), 'Fresh absent comparison output required')
    for source in (Path(args.reference).resolve(), Path(args.native).resolve()):
        need(not output.resolve().is_relative_to(source) and not source.is_relative_to(output.resolve()),
             'Comparison output must not overlap original raw evidence')
    output.mkdir()
    result = dict(format='toklet-granite-offline-comparison-v1', passed=False, completed=False,
                  protocol_sha256=args.protocol_sha256, reference_sha256=args.reference_sha256,
                  native_sha256=args.native_sha256, graph_qualified=False, quality_qualified=False,
                  performance_qualified=False, observer_overhead_qualified=False,
                  managed_adoption_qualified=False, whole_goal_complete=False)
    endpoint = None
    try:
        need(shutil.disk_usage(output).free >= FLOOR, 'Comparison disk floor is unavailable')
        protocol = validate_protocol(decoded(args.protocol, args.protocol_sha256, 2 * 1024**2))
        reference = validate_reference(decoded(Path(args.reference) / 'manifest.json', args.reference_sha256),
                                       protocol, args.protocol_sha256)
        native = decoded(Path(args.native) / 'manifest.json', args.native_sha256)
        indexed = validate_native(native, protocol, args.protocol_sha256)
        runtime.bind_environment(args.environment_root)
        selected = sources()
        admitted, endpoint = runtime.admit(Path(args.runtime_manifest), args.runtime_sha256, selected)
        result['admitted_runtime'] = admitted
        import numpy as np
        result.update(compare(reference, native, indexed, protocol, args.reference, args.native, output, np))
        need(sources() == selected, 'Selected checker source inventory changed')
        result['endpoint_origins'] = endpoint()
    except (ValueError, OSError, KeyError, TypeError, OverflowError, KeyboardInterrupt) as error:
        result.update(passed=False, completed=False, failure=dict(type=type(error).__name__, message=str(error)))
        if endpoint is not None:
            try:
                result['endpoint_origins'] = endpoint()
            except (ValueError, OSError) as final_error:
                result['endpoint_failure'] = str(final_error)
    publish(output / 'manifest.json', result)
    print(json.dumps({key: value for key, value in result.items()
                      if key in ('passed', 'completed', 'arrays', 'failed_profile_coordinates',
                                 'exact_schedule_mismatching_bytes', 'failure')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
