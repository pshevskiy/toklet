#!/usr/bin/env python3
"""Streaming necessary-envelope audit of the fixed Huihui4B reference capture.

Disjoint conservative intervals reject the unchanged coupled gate. Overlap is
only a necessary condition: it proves neither an FP32 native candidate nor model
fidelity, quality, physical admission, performance or adoption.
"""
import argparse
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys

import check_qwen35_real_reference as geometry
from check_qwen35_context256 import runtime_origins

FORMAT = 'toklet-qwen35-huihui4b-envelope-v1'
COLLECTOR_SHA = '8321089893e0176503e1d5596efc9830aaafa99a72237ad7d7537b5d0232aab0'
ROWS_BYTES = 461893632
COUNTS = {128: 18180608, 256: 20310528}
PAIRS = 38491136
BLOCK = 65536
DOCUMENT_LIMIT = 2 * 1024**2
PROFILES = ('ideal_fp64', 'staged_fp32')
FILE_SHA = '244b47c46b759664c223a996046857720826a388bdf3969620a8f12ebb6e14e0'
CONFIG_SHA = '13a514e981196ba683d8d53312789384ff5d9b9e73c7bfa137010397212c4af3'
HEADER_SHA = '7284f68df36a7761569b48321066e92d616306e33b4a9487029673a0ff344cb2'
TOKENS = (1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10, 19, 8, 21, 23, 29)
GEOMETRY = dict(hidden=2560, intermediate=9216, vocab=248320, layers=32,
    query_heads=16, kv_heads=4, head_dim=256, rotated_dims=64,
    key_heads=16, value_heads=32, key_dim=128, value_dim=128,
    max_positions=262144, rope_theta=10000000.0, rms_epsilon=1e-6)
LEDGER = dict(python_state_and_temporaries=2*1024**3,
    interpreter_and_runtime=512*1024**2, small_weight_vectors=64*1024**2,
    tile_and_source_buffers=128*1024**2, capture=512*1024**2,
    guard_helper=32*1024**2, controller_and_logs=64*1024**2)
PROVENANCE = dict(original_sha256='faa580d347df630267e8082377eb5555d89de76c9f7b149fa424239f88c46582',
    vector_sha256='fd7d173a829c2db1ca04032d039094a4546cd237b88f78870f2f641e0c92943c',
    helper_sha256='425d121dd27dd08d3f7baee95a41a446fbfcd8500ae4df192e0286777ca6b792',
    builder_sha256='932e0acf2cd658a17bd70a80a88c9febf7ffa2e9ad0945427607b406ad69c506',
    inverse_patch_exact=True)
EQUIVALENCE = dict(
    source_proof='build_qwen35_vector_graph.validate: exact inverse recurrence-only patch',
    portable_controls='tools/reference/test_qwen35_recurrent_vector.py: real-shape repeated recurrence, signed zero/subnormal/cancellation, complete tied/untied traces in both original profiles',
    historical_checkpoint_proof='artifacts/qwen35-context256-2026-11-09/reference-graph-vector-experiment/full256-parity.json',
    historical_checkpoint_proof_sha256='96153cd566f2c98d7ef62275d0b80632050bca65f6444f5d6e98a09034768386',
    historical_scope='Original-source staged_fp32 checkpoint bytes only; not an ideal_fp64 full-context proof, new4 capture, native fidelity or quality qualification')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, low=0, high=2**63-1):
    return type(value) is int and low <= value <= high


def sha(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
         'External lowercase SHA256 required')
    return value


def interval_bounds(reference):
    """Conservative outward enclosure; four radius ULPs retain rounding slack."""
    need(type(reference) is float and math.isfinite(reference), 'Exact finite Python float required')
    tolerance = 1e-5 + 1e-4 * abs(reference)
    for _ in range(4):
        tolerance = math.nextafter(tolerance, math.inf)
    low = math.nextafter(reference - tolerance, -math.inf)
    high = math.nextafter(reference + tolerance, math.inf)
    need(math.isfinite(tolerance) and math.isfinite(low) and math.isfinite(high),
         'Envelope bound overflow')
    return low, high


def disjoint(ideal, staged):
    a, b = interval_bounds(ideal), interval_bounds(staged)
    return a[1] < b[0] or b[1] < a[0]


def validate_inventory(descriptors):
    """Validate all4204 arrays using only neutral stage geometry/key helpers."""
    need(type(descriptors) is list and len(descriptors) == 4204, 'Complete4204-array inventory required')
    index, cursor = {}, 0
    fields = {'checkpoint', 'profile', 'token', 'layer', 'stage', 'count', 'offset', 'bytes', 'sha256'}
    for block_index, (profile, width, checkpoint) in enumerate(
            (p, w, n) for p, w in [('ideal_fp64', 8), ('staged_fp32', 4)] for n in COUNTS):
        wanted, seen, count = geometry.expected(checkpoint), set(), 0
        need(len(wanted) == 1051, 'Neutral geometry differs')
        for row in descriptors[block_index*1051:(block_index+1)*1051]:
            need(type(row) is dict and set(row) == fields, 'Exact descriptor fields required')
            need(type(row['checkpoint']) is int and row['checkpoint'] == checkpoint
                 and type(row['profile']) is str and row['profile'] == profile,
                 'Frozen profile/checkpoint order differs')
            key = geometry.key(row, checkpoint-1)
            need(key in wanted and key not in seen, 'Unknown or duplicate stage')
            need(integer(row['count'], 1, ROWS_BYTES) and row['count'] == wanted[key]
                 and integer(row['offset'], 0, ROWS_BYTES) and row['offset'] == cursor
                 and integer(row['bytes'], 1, ROWS_BYTES) and row['bytes'] == row['count']*width,
                 'Array count/offset/width differs')
            sha(row['sha256'])
            seen.add(key)
            index[profile, checkpoint, key] = row
            cursor += row['bytes']
            count += row['count']
        need(seen == set(wanted) and count == COUNTS[checkpoint], 'Incomplete checkpoint geometry')
    need(cursor == ROWS_BYTES and len(index) == 4204, 'Body coverage differs')
    return index


def _endpoint(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _open(path, size=None, bound=None):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        value = os.fstat(fd)
        need(stat.S_ISREG(value.st_mode) and (size is None or value.st_size == size)
             and (bound is None or value.st_size <= bound), 'Bounded regular input required')
        return os.fdopen(fd, 'rb', buffering=0)
    except BaseException:
        os.close(fd)
        raise


def _stable(handle, path, endpoint):
    value = os.stat(path, follow_symlinks=False)
    need(stat.S_ISREG(value.st_mode) and _endpoint(value) == endpoint
         and _endpoint(os.fstat(handle.fileno())) == endpoint, 'Input descriptor/path changed')


def _hash(handle):
    handle.seek(0)
    checksum, count = hashlib.sha256(), 0
    while data := handle.read(BLOCK):
        checksum.update(data)
        count += len(data)
    return count, checksum.hexdigest()


def _document(path, expected):
    with _open(path, bound=DOCUMENT_LIMIT) as handle:
        endpoint = _endpoint(os.fstat(handle.fileno()))
        raw = handle.read(DOCUMENT_LIMIT+1)
        need(len(raw) == endpoint[2] <= DOCUMENT_LIMIT
             and hashlib.sha256(raw).hexdigest() == sha(expected), 'Pinned JSON identity differs')
        _stable(handle, path, endpoint)
    value = geometry.decode(raw.decode('utf-8'))
    need(type(value) is dict, 'JSON object required')
    return value, dict(path=str(path), bytes=len(raw), sha256=expected, endpoint=endpoint)


def _files(manifest):
    files = manifest.get('files')
    need(type(files) is dict and 0 < len(files) <= 40000, 'Bounded manifest files required')
    endpoints = {}
    for name, pin in files.items():
        need(type(name) is str and Path(name).is_absolute() and type(pin) is dict
             and set(pin) == {'bytes', 'sha256'} and integer(pin['bytes'], 0, 512*1024**2),
             'Exact runtime file pin required')
        sha(pin['sha256'])
        with _open(Path(name), size=pin['bytes']) as handle:
            endpoint = _endpoint(os.fstat(handle.fileno()))
            need(_hash(handle) == (pin['bytes'], pin['sha256']), 'Actual runtime/source bytes differ')
            _stable(handle, Path(name), endpoint)
            endpoints[name] = endpoint
    return files, endpoints


def _validate_result(result, runtime, native, runtime_sha, native_sha):
    fixed = dict(format='toklet-qwen35-huihui4b-context256-reference-v1',
        reference_variant='ordered_vector_recurrence_context256', graph_provenance=PROVENANCE,
        arithmetic_equivalence=EQUIVALENCE,
        source_pins=dict(container_bytes=2367237149, container_sha256=FILE_SHA,
            config_bytes=3024, config_sha256=CONFIG_SHA, header_bytes=119829,
            header_sha256=HEADER_SHA, payload_origin=119837),
        deadline_seconds=21600, process_limit_bytes=6*1024**3,
        repository='TheWirelessPhoenix/Qwen3.5-4B-abliterated_4bit',
        revision='916d9f7c2e27ed41e2b9ed39f3a2badccb2bbc7c', source_sha256=FILE_SHA,
        tokens=[TOKENS[i % len(TOKENS)] for i in range(256)], position_offset=0,
        checkpoints=[128, 256], geometry=GEOMETRY, gate=dict(atol=1e-5, rtol=1e-4),
        planned_rows_bytes=ROWS_BYTES, ledger=LEDGER, engine_executed=False,
        contextual_numerical_qualified=False, request_metrics_qualified=False,
        quality_qualified=False, managed_adoption_qualified=False)
    dynamic = {'completed', 'arrays', 'runtime_admission', 'source', 'primary_error',
               'endpoint_errors', 'process', 'execution_environment', 'source_endpoint_sha256',
               'rows_sha256', 'runtime_endpoint'}
    need(set(result) == set(fixed) | dynamic and result['completed'] is True
         and result['primary_error'] is None and result['endpoint_errors'] == [], 'Complete fixed capture required')
    for key, expected in fixed.items():
        need(geometry.exact(result[key], expected), 'Frozen reference scope differs: '+key)
    sha(result['rows_sha256'])
    source = result['source']
    need(type(source) is dict and set(source) == {'bytes', 'sha256', 'reads', 'read_bytes',
         'read_chain_sha256', 'poisoned', 'poison_reason', 'closed'}
         and type(source['bytes']) is int and source['bytes'] == 2367237149
         and source['sha256'] == result['source_endpoint_sha256'] == FILE_SHA
         and source['closed'] is True and source['poisoned'] is False
         and source['poison_reason'] is None and integer(source['reads'], 1)
         and integer(source['read_bytes'], 2367237149), 'Closed fixed source audit required')
    sha(source['read_chain_sha256'])
    process = result['process']
    need(type(process) is dict and set(process) == {'pid', 'group', 'parent_pid'}
         and all(integer(v, 1) for v in process.values())
         and process['pid'] != process['parent_pid'], 'Typed capture process identity required')
    environment = result['execution_environment']
    keys = {'PYTHONNOUSERSITE', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'OMP_NUM_THREADS', 'MallocLargeCache', 'MallocNanoZone', 'MallocAggressiveMadvise'}
    need(type(environment) is dict and set(environment) == keys
         and all(v is None or type(v) is str for v in environment.values()), 'Execution environment differs')
    need(runtime.get('format') == 'toklet-mlx-reference-runtime-v1'
         and geometry.exact(runtime.get('versions'), {'numpy': '2.3.2', 'mlx': '0.32.3'})
         and set(runtime) == {'format', 'versions', 'files'}
         and native.get('format') == 'toklet-python-native-runtime-v1'
         and set(native) == {'format', 'files'}, 'Actual Python reference manifests required')
    (files, runtime_endpoints), (native_files, native_endpoints) = _files(runtime), _files(native)
    need(all(geometry.exact(files[k], native_files[k]) for k in files.keys() & native_files.keys()),
         'Conflicting runtime/native file identity')
    admission = result['runtime_admission']
    need(type(admission) is dict and set(admission) == {'admitted', 'native_manifest_sha256', 'initial_origins'},
         'Exact runtime admission required')
    admitted = admission['admitted']
    need(type(admitted) is dict and set(admitted) == {'manifest_sha256', 'files', 'pycache_prefix', 'imports'}
         and admitted['manifest_sha256'] == runtime_sha
         and integer(admitted['files'], 1) and admitted['files'] == len(files)
         and type(admitted['pycache_prefix']) is str and Path(admitted['pycache_prefix']).is_absolute()
         and admission['native_manifest_sha256'] == native_sha, 'External runtime admission differs')
    imports = admitted['imports']
    need(type(imports) is dict and set(imports) == {'numpy', 'mlx.core', 'mlx.nn'}
         and all(type(path) is str and path in files for path in imports.values()), 'Admitted imports differ')
    for endpoint in (admission['initial_origins'], result['runtime_endpoint']):
        runtime_origins(endpoint, imports, runtime, native)
    for name, expected in [('qwen35_huihui4b_reference.py', COLLECTOR_SHA),
            ('qwen35_cpu_graph.py', PROVENANCE['original_sha256']),
            ('qwen35_vector_graph.py', PROVENANCE['vector_sha256']),
            ('qwen35_recurrent_vector.py', PROVENANCE['helper_sha256']),
            ('build_qwen35_vector_graph.py', PROVENANCE['builder_sha256'])]:
        matching = [pin for path, pin in files.items() if Path(path).name == name]
        need(len(matching) == 1 and matching[0]['sha256'] == expected, 'Pinned collector/graph source differs')
    return runtime_endpoints, native_endpoints


def _certified_gap(ideal, staged, bounds):
    pairs = []
    for reference, (low, high) in zip((ideal, staged), bounds):
        exact = Fraction.from_float(reference)
        for radius in (Fraction(1, 100000)+abs(exact)/10000,
                       Fraction.from_float(1e-5)+Fraction.from_float(1e-4)*abs(exact)):
            need(Fraction.from_float(low) <= exact-radius
                 and Fraction.from_float(high) >= exact+radius, 'Outward enclosure certificate failed')
        pairs.append(tuple(Fraction.from_float(v) for v in (low, high)))
    gap = max(pairs[1][0]-pairs[0][1], pairs[0][0]-pairs[1][1])
    need(gap > 0, 'Exact positive gap certificate required')
    return str(gap)


def audit(result_path, result_sha, rows_path, runtime_path, runtime_sha, native_path, native_sha, output):
    output = Path(output)
    need(output.parent.is_dir() and not output.exists(), 'Exclusive fresh report path required')
    report = dict(format=FORMAT, status='incomplete', completed=False, failure=None,
        paired_coordinates=0, disjoint_coordinates=0, per_stage=[], witnesses=[], input_pins=[],
        feasible_envelope_only=False, native_candidate_exists=None, native_numerical_qualified=False,
        quality_qualified=False, memory_fit_qualified=False, performance_qualified=False,
        adoption_qualified=False, whole_goal_complete=False, original_rejections_unchanged=True,
        scope='Conservative necessary envelope only; overlap does not prove an FP32 candidate exists.',
        radius_guard_ULPs=4, IO_window_bytes=BLOCK, owned_window_limit_bytes=16*1024**2,
        external_collection_guard_receipt_required=True)
    pinned, body, body_endpoint, runtime_endpoints = [], None, None, None
    with ExitStack() as stack:
        try:
            result, a = _document(Path(result_path), sha(result_sha))
            runtime, b = _document(Path(runtime_path), sha(runtime_sha))
            native, c = _document(Path(native_path), sha(native_sha))
            pinned.extend((a, b, c))
            runtime_endpoints = _validate_result(result, runtime, native, runtime_sha, native_sha)
            index = validate_inventory(result['arrays'])
            body = stack.enter_context(_open(Path(rows_path), size=ROWS_BYTES))
            body_endpoint = _endpoint(os.fstat(body.fileno()))
            whole = hashlib.sha256()
            for row in result['arrays']:
                body.seek(row['offset'])
                checksum, left = hashlib.sha256(), row['bytes']
                width = 8 if row['profile'] == 'ideal_fp64' else 4
                while left:
                    data = body.read(min(BLOCK, left))
                    need(data and len(data) % width == 0, 'Short or unaligned array window')
                    left -= len(data)
                    checksum.update(data)
                    whole.update(data)
                    need(all(math.isfinite(v) for v, in struct.iter_unpack('<d' if width == 8 else '<f', data)),
                         'Nonfinite raw reference value')
                need(checksum.hexdigest() == row['sha256'], 'Raw array SHA differs')
            need(whole.hexdigest() == result['rows_sha256'], 'Whole reference body SHA differs')
            _stable(body, Path(rows_path), body_endpoint)
            for checkpoint in COUNTS:
                for key in geometry.expected(checkpoint):
                    ideal = index['ideal_fp64', checkpoint, key]
                    staged = index['staged_fp32', checkpoint, key]
                    count, rejected = 0, 0
                    for first in range(0, ideal['count'], BLOCK//8):
                        n = min(BLOCK//8, ideal['count']-first)
                        body.seek(ideal['offset']+first*8)
                        x = body.read(n*8)
                        body.seek(staged['offset']+first*4)
                        y = body.read(n*4)
                        need(len(x) == n*8 and len(y) == n*4, 'Truncated paired reference window')
                        for (u,), (v,) in zip(struct.iter_unpack('<d', x), struct.iter_unpack('<f', y), strict=True):
                            bounds = interval_bounds(u), interval_bounds(v)
                            separated = bounds[0][1] < bounds[1][0] or bounds[1][1] < bounds[0][0]
                            if separated:
                                rejected += 1
                                if len(report['witnesses']) < 32:
                                    report['witnesses'].append(dict(checkpoint=checkpoint, layer=key[0],
                                        stage=key[1], head=key[2], index=count, ideal=u, staged=v,
                                        ideal_bounds=bounds[0], staged_bounds=bounds[1],
                                        certified_exact_gap=_certified_gap(u, v, bounds)))
                            count += 1
                    need(count == ideal['count'], 'Incomplete paired coordinate coverage')
                    report['per_stage'].append(dict(checkpoint=checkpoint, layer=key[0], stage=key[1],
                        head=key[2], count=count, disjoint=rejected))
                    report['paired_coordinates'] += count
                    report['disjoint_coordinates'] += rejected
            need(report['paired_coordinates'] == PAIRS and len(report['per_stage']) == 2102,
                 'Full paired envelope coverage required')
            report.update(completed=True, feasible_envelope_only=report['disjoint_coordinates'] == 0,
                          status='reject' if report['disjoint_coordinates'] else 'feasible_envelope_only')
        except (Exception, KeyboardInterrupt) as error:
            report.update(failure=f'{type(error).__name__}: {str(error)[:2048]}')
        finally:
            try:
                if body is not None:
                    _stable(body, Path(rows_path), body_endpoint)
                    size, digest = _hash(body)
                    need(size == ROWS_BYTES and digest == result['rows_sha256'], 'Body endpoint SHA differs')
                    _stable(body, Path(rows_path), body_endpoint)
                    report['body_pin'] = dict(path=str(rows_path), bytes=size, sha256=digest, endpoint=body_endpoint)
                for pin in pinned:
                    _, again = _document(Path(pin['path']), pin['sha256'])
                    need(again == pin, 'Input JSON changed during audit')
                if runtime_endpoints is not None:
                    _, first = _files(runtime)
                    _, second = _files(native)
                    need((first, second) == runtime_endpoints, 'Runtime/source endpoint changed during audit')
            except (Exception, KeyboardInterrupt) as error:
                report.update(status='incomplete', completed=False, feasible_envelope_only=False,
                              failure=f'Endpoint closure: {type(error).__name__}: {str(error)[:2048]}')
    report['input_pins'] = pinned
    data = (json.dumps(report, allow_nan=False, separators=(',', ':'))+'\n').encode()
    need(len(data) <= DOCUMENT_LIMIT, 'Envelope report exceeds bound')
    partial = output.with_name(output.name+'.partial')
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, output)
    partial.unlink()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('result', 'rows', 'runtime-manifest', 'native-manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('result-sha256', 'runtime-sha256', 'native-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    result = audit(args.result, args.result_sha256, args.rows, args.runtime_manifest,
                   args.runtime_sha256, args.native_manifest, args.native_sha256, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'completed', 'paired_coordinates', 'disjoint_coordinates')}))
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
