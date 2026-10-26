"""Freeze target schedules and replay saved BF16-reader outputs using stdlib.

Imports neither the reader under test nor its cold worker/numerical runtime.
Expected read spans and row bits derive only from the pre-existing oracle.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct
import check_gemma4_bf16_rows_oracle as oracle

ORACLE_SHA = 'a17dede82620277d77149134750aa7e51abc72cf8995d6481799291e140113cc'


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def plan(directory):
    manifest, assets = oracle.load(directory, ORACLE_SHA)
    captures = []; reads = []; chain = hashlib.sha256(b'toklet-bf16-source-reads-v1\0')
    read_bytes = 0
    def read(offset, body):
        nonlocal read_bytes
        digest = hashlib.sha256(body).digest(); sequence = len(reads) + 1
        chain.update(struct.pack('<QQQ', sequence, offset, len(body))); chain.update(digest)
        reads.append(dict(offset=offset, bytes=len(body), sha256=digest.hex())); read_bytes += len(body)
    def capture(label, shape, dtype, body=None, references=None):
        item = dict(label=label, shape=shape, dtype=dtype, bytes=math.prod(shape) * (4 if dtype == '<f4' else 8))
        if body is not None: item['expected_sha256'] = oracle.digest(body)
        if references is not None: item['references'] = references
        captures.append(item)
    def span(source):
        start = source['raw_offset']; return assets['raw'][start:start + source['bytes']]
    for case in manifest['cases']:
        columns = case['shape'][1]; sources = case['sources']; vector = len(case['tensor_shape']) == 1
        start = case['offset']; body = assets['arrays'][start:start + case['bytes']]
        if vector: read(sources[0]['file_offset'], span(sources[0]))
        elif case['id'] == 'maximum_tile': read(sources[0]['file_offset'], b''.join(span(s) for s in sources))
        else:
            read(sources[0]['file_offset'], span(sources[0]) + span(sources[1]))
            for source in sources[2:]: read(source['file_offset'], span(source))
        capture(case['id'] + ':grouped', case['shape'], '<f4', body)
        if not vector:
            if case['id'] == 'maximum_tile':
                for group in (sources[:64], sources[64:]): read(group[0]['file_offset'], b''.join(span(s) for s in group))
            else:
                for source in sources: read(source['file_offset'], span(source))
            capture(case['id'] + ':split', case['shape'], '<f4', body)
        order = case['gather_order']; stride = columns * 4
        for index in order: read(sources[index]['file_offset'], span(sources[index]))
        gathered = b''.join(body[index * stride:(index + 1) * stride] for index in order)
        capture(case['id'] + ':gather', [len(order), columns], '<f4', gathered)
    wide = 'model.language_model.layers.15.mlp.down_proj.weight'
    first = next(case for case in manifest['cases'] if case['id'] == 'sample:' + wide)
    source = first['sources'][0]; read(source['file_offset'], span(source))
    start = first['offset']; columns = first['shape'][1]
    capture('after rejected capacity', [1, columns], '<f4', assets['arrays'][start:start + columns * 4])
    for projection in manifest['projections']:
        rows, columns = projection['matrix_shape']; raw_start = projection['raw_offset']
        # The original header, independently authenticated by oracle replay,
        # fixes the complete matrix start even if its sample list ever changes.
        sample = next(case for case in manifest['cases'] if case['tensor'] == projection['tensor'])
        file_start = sample['sources'][0]['file_offset']
        for dtype in ('<f4', '<f8'):
            for start in range(0, rows, 128):
                count = min(128, rows - start); offset = start * columns * 2
                read(file_start + offset, assets['raw'][raw_start + offset:raw_start + offset + count * columns * 2])
            capture(projection['tensor'] + ':' + dtype, [2, rows], dtype, references=projection['references'])
    audit = dict(**oracle.MODEL_ID, reads=len(reads), read_bytes=read_bytes,
                 read_chain_sha256=chain.hexdigest(), poisoned=False, poison_reason=None, closed=True)
    oracle.same(len(captures), 1366, 'fixed target schedule count differs')
    oracle.same(len(reads), 4004, 'fixed source read count differs')
    oracle.same(sum(c['bytes'] for c in captures), 56288676, 'fixed capture size differs')
    return dict(format='toklet-gemma-bf16-target-plan-v1', oracle_manifest_sha256=ORACLE_SHA,
                captures=captures, ordered_reads=reads, source_audit=audit,
                projection_policy=manifest['projection_policy'], admission_controls=65)


def replay(directory, plan_path, plan_sha, target_directory):
    frozen = oracle.safe.read(plan_path, 4 * 1024**2)
    oracle.same(oracle.digest(frozen), plan_sha, 'external target plan differs')
    expected = plan(directory); oracle.same(frozen, encoded(expected), 'pre-target schedule differs')
    result = oracle.safe.document(oracle.safe.read(target_directory.parent / (target_directory.name + '-result.json'), 4 * 1024**2))
    oracle.same(result['passed'], True, 'target worker failed'); oracle.same(result['primary_error'], None, 'target primary failure')
    oracle.same(result['endpoint_errors'], [], 'target runtime endpoint failure')
    actual = result['observation']; oracle.same(actual['passed'], True, 'incomplete target observation')
    oracle.same(actual['source_audit'], expected['source_audit'], 'ordered target read audit differs')
    oracle.same(actual['row_cases_completed'], 541, 'incomplete target row cases')
    oracle.same(actual['projection_calls_completed'], 4, 'incomplete target projections')
    for key in ('official_graph_executed', 'model_fit_qualified', 'quality_scored', 'performance_qualified', 'app_eligibility_changed', 'full_model_window_qualified'):
        oracle.same(actual[key], False, 'unsupported target qualification')
    labels = ['129 contiguous rows', '129 gather rows', 'layout bytes']
    m, assets = oracle.load(directory, ORACLE_SHA)
    labels += [*m['excluded_shared_kv'], 'model.language_model.unknown.weight', 'different authenticated source']
    messages = ['bounded integer required', 'bounded aggregate gather rows required', 'exact official source/layout identity required']
    messages += ['selected official text tensor required'] * 61 + ['exact official source/layout identity required']
    oracle.same(len(actual['admission_controls']), expected['admission_controls'], 'admission control count differs')
    for control, label, message in zip(actual['admission_controls'], labels, messages, strict=True):
        oracle.same(control, dict(label=label, rejected=True, expected_error=message, actual_error=message), 'intended admission rejection differs')
    oracle.same(len(actual['captures']), len(expected['captures']), 'target capture count differs')
    raw = oracle.safe.read(target_directory / 'arrays.bin', 64 * 1024**2)
    cursor = exact = projections = 0
    for wanted, capture in zip(expected['captures'], actual['captures'], strict=True):
        keys = {'label', 'shape', 'dtype', 'offset', 'bytes', 'sha256', 'captured', 'compared'}
        if 'references' in wanted: keys.add('comparisons')
        oracle.same(set(capture), keys, 'target capture schema differs')
        for key in ('label', 'shape', 'dtype', 'bytes'): oracle.same(capture[key], wanted[key], 'target array metadata differs')
        oracle.same(capture['offset'], cursor, 'target output coverage differs')
        body = raw[cursor:cursor + wanted['bytes']]; cursor += len(body)
        oracle.same(len(body), wanted['bytes'], 'target array truncated')
        oracle.same(capture['sha256'], oracle.digest(body), 'target saved array differs')
        oracle.same(capture['captured'], True, 'uncaptured target output'); oracle.same(capture['compared'], True, 'uncompared target output')
        if 'expected_sha256' in wanted:
            oracle.same(oracle.digest(body), wanted['expected_sha256'], 'target row bits differ'); exact += len(body) // 4
        else:
            dtype = '<f' if wanted['dtype'] == '<f4' else '<d'
            values = [v for (v,) in struct.iter_unpack(dtype, body)]
            oracle.safe.require(all(math.isfinite(v) for v in values), 'nonfinite target projection')
            comparisons = []
            for kind in ('fsum', 'ordered'):
                descriptor = wanted['references'][kind]; start = descriptor['offset']
                reference = [v for (v,) in struct.iter_unpack('<d', assets['arrays'][start:start + descriptor['bytes']])]
                prefix = 'fp32' if dtype == '<f' else 'fp64'; policy = expected['projection_policy']
                atol, rtol = policy[prefix + '_atol'], policy[prefix + '_rtol']
                errors = [abs(a - b) for a, b in zip(values, reference, strict=True)]
                allowed = [atol + rtol * abs(v) for v in reference]
                comparison = dict(reference=kind, atol=atol, rtol=rtol, compared_values=len(values),
                                  maximum_absolute_error=max(errors), maximum_tolerance_ratio=max(a / b for a, b in zip(errors, allowed, strict=True)),
                                  failures=sum(a > b for a, b in zip(errors, allowed, strict=True)))
                oracle.same(comparison['failures'], 0, 'saved target projection differs'); comparisons.append(comparison)
            oracle.same(capture['comparisons'], comparisons, 'target projection report differs'); projections += 1
    oracle.same(cursor, len(raw), 'complete target output coverage differs')
    oracle.same(actual['exact_float32_values'], exact, 'target row value count differs')
    return dict(passed=True, captures=len(expected['captures']), exact_float32_values=exact,
                projection_calls=projections, source_audit=expected['source_audit'], admission_controls=65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--prepare', type=Path); parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha'); parser.add_argument('--target', type=Path)
    args = parser.parse_args()
    if args.prepare is not None:
        body = encoded(plan(args.oracle))
        with args.prepare.open('xb') as file: oracle.same(file.write(body), len(body), 'complete target plan write required')
        print(json.dumps(dict(passed=True, bytes=len(body), sha256=oracle.digest(body))))
    else:
        print(json.dumps(replay(args.oracle, args.plan, args.plan_sha, args.target)))
