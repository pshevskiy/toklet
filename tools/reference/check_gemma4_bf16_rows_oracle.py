"""Independent stdlib replay of frozen official BF16 excerpts and scalar dots.

No producer, target row adapter, NumPy, Torch or Toklet import is used.
"""
from pathlib import Path
import argparse
import functools
import hashlib
import json
import math
import operator
import os
import signal
import stat
import struct
import time
import check_gemma_hybrid_tiny as safe

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'models/Gemma4-E2B-it-BF16-70af34e/model.safetensors'
CANDIDATE = ROOT / 'models/MLX-Gemma4-E2B-4bit-2387675/model.safetensors'
MODEL_ID = dict(bytes=10246621918, sha256='2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550')
CANDIDATE_ID = dict(bytes=3550670554, sha256='038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e')
PREFIX = 'model.language_model.'


def same(actual, expected, label):
    try: safe.same(actual, expected)
    except ValueError as error: raise ValueError(label) from error


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def expanded(raw):
    safe.require(len(raw) % 2 == 0, 'complete BF16 words required')
    result = bytearray(2 * len(raw))
    for offset in range(0, len(raw), 2):
        word = int.from_bytes(raw[offset:offset + 2], 'little')
        safe.require(word & 0x7f80 != 0x7f80, 'finite oracle word required')
        result[2 * offset:2 * offset + 4] = (word * 65536).to_bytes(4, 'little')
    return bytes(result)


def fields(value, expected, label):
    same(set(value), set(expected), label)


def load(directory, expected_sha):
    raw = safe.read(directory / 'manifest.json', 4 * 1024**2)
    same(digest(raw), expected_sha, 'external oracle manifest differs')
    m = safe.document(raw)
    fields(m, ['format', 'passed', 'primary_error', 'endpoint_errors', 'model', 'cases', 'projections',
               'candidate_vectors', 'arrays', 'raw', 'official_graph_executed', 'arithmetic_profile_qualified',
               'model_fit_qualified', 'quality_scored', 'app_eligibility_changed', 'full_model_window_qualified',
               'projection_policy', 'header', 'inventory_sha256', 'excluded_shared_kv', 'candidate_model'], 'oracle schema differs')
    same(m['format'], 'toklet-gemma-bf16-row-oracle-v1', 'oracle format differs')
    same(m['passed'], True, 'incomplete oracle'); same(m['primary_error'], None, 'failed oracle')
    same(m['endpoint_errors'], [], 'oracle endpoint failure')
    same(m['model'], MODEL_ID, 'official source identity differs')
    same(m['candidate_model'], CANDIDATE_ID, 'candidate source identity differs')
    same(m['header'], dict(bytes=263952, sha256='12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e'), 'official header identity differs')
    same(m['inventory_sha256'], 'a3dff3b9d825fcf4fa6f391262732c5f429345682d683d6a39bc551ddacd68e2', 'geometry inventory differs')
    for flag in ('official_graph_executed', 'arithmetic_profile_qualified', 'model_fit_qualified',
                 'quality_scored', 'app_eligibility_changed', 'full_model_window_qualified'):
        same(m[flag], False, 'unsupported oracle qualification')
    same(m['projection_policy'], dict(fp32_atol=1e-5, fp32_rtol=1e-4, fp64_atol=1e-13, fp64_rtol=1e-13), 'projection gates differ')
    assets = {}
    for key, name in [('arrays', 'arrays.bin'), ('raw', 'raw.bin')]:
        value = safe.read(directory / name, 64 * 1024**2)
        same(m[key], dict(file=name, bytes=len(value), sha256=digest(value)), 'complete oracle binary differs')
        assets[key] = value
    return m, assets


def identity(value):
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def open_source(path, expected):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    file = os.fdopen(fd, 'rb')
    try:
        original = identity(os.fstat(fd))
        safe.require(stat.S_ISREG(original[2]) and original[3] == expected['bytes'], 'regular replay source required')
        safe.same(identity(path.lstat()), original)
        sha = hashlib.sha256(); count = 0; deadline = time.monotonic() + 900
        while True:
            safe.require(time.monotonic() < deadline, 'source replay deadline')
            raw = file.read(min(1024**2, expected['bytes'] - count + 1))
            if not raw: break
            count += len(raw); safe.require(count <= expected['bytes'], 'replay source grew'); sha.update(raw)
        same(count, expected['bytes'], 'replay source truncated'); same(sha.hexdigest(), expected['sha256'], 'replay full source SHA differs')
        same(identity(os.fstat(fd)), original, 'replay FD changed'); same(identity(path.lstat()), original, 'replay path changed')
        return file, path, original
    except BaseException as error:
        try: file.close()
        except BaseException as cleanup: error.add_note('replay authentication cleanup: ' + str(cleanup))
        raise


def source_bytes(source, offset, count):
    file, path, original = source
    safe.require(type(offset) is int and type(count) is int and 0 <= offset
                 and 0 < count <= 128 * 12288 * 2 and offset + count <= original[3], 'bounded replay span')
    same(identity(os.fstat(file.fileno())), original, 'replay FD changed')
    same(identity(path.lstat()), original, 'replay path changed')
    raw = os.pread(file.fileno(), count, offset)
    same(len(raw), count, 'replay row truncated')
    same(identity(os.fstat(file.fileno())), original, 'replay FD changed')
    same(identity(path.lstat()), original, 'replay path changed')
    return raw


def verify(directory, expected_sha, official, candidate):
    m, assets = load(directory, expected_sha)
    inv_raw = safe.read(ROOT / 'tests/fixtures/gemma-resident-storage-reference-v1/inventory.json', 2 * 1024**2)
    same(digest(inv_raw), m['inventory_sha256'], 'prior geometry input differs')
    inventory = safe.document(inv_raw)
    header_raw = source_bytes(official, 8, 263952)
    same(source_bytes(official, 0, 8), struct.pack('<Q', 263952), 'official header length differs')
    same(digest(header_raw), m['header']['sha256'], 'official consumed header differs')
    header = safe.document(header_raw); origin = 263960
    geometry = {}; prior_vectors = {}
    for value in inventory['matrices']:
        name = PREFIX + value['name'][len('language_model.model.'):]
        geometry[name] = [value['rows'], value['columns']]
    for value in inventory['vectors']:
        name = PREFIX + value['name'][len('language_model.model.'):]
        geometry[name] = [value['count']]; prior_vectors[name] = value['source']
    same(len(geometry), 540, 'all selected geometry required')
    unused = sorted(f'{PREFIX}layers.{i}.self_attn.{suffix}' for i in range(15, 35)
                    for suffix in ('k_norm.weight', 'k_proj.weight', 'v_proj.weight'))
    same(m['excluded_shared_kv'], unused, 'shared consumer exclusions differ')
    same({name for name in header if name.startswith(PREFIX)}, set(geometry) | set(unused), 'official text inventory differs')
    for name, shape in geometry.items():
        same(header[name]['dtype'], 'BF16', 'official tensor dtype differs')
        same(header[name]['data_offsets'][1] - header[name]['data_offsets'][0], 2 * math.prod(shape), 'official tensor extent differs')
    for name in unused:
        layer = int(name.split('.')[3]); dim = 512 if (layer + 1) % 5 == 0 else 256
        shape = [dim] if name.endswith('k_norm.weight') else [dim, 1536]
        same(header[name]['dtype'], 'BF16', 'excluded tensor dtype differs')
        same(header[name]['shape'], shape, 'excluded tensor shape differs')
        same(header[name]['data_offsets'][1] - header[name]['data_offsets'][0], 2 * math.prod(shape), 'excluded tensor extent differs')
    identifiers = ['sample:' + name for name in sorted(geometry)] + ['maximum_tile']
    same([case['id'] for case in m['cases']], identifiers, 'frozen tensor selection differs')
    array_cursor = raw_cursor = values_compared = 0; vector_relations = []
    wide = PREFIX + 'layers.15.mlp.down_proj.weight'
    for case, identifier in zip(m['cases'], identifiers, strict=True):
        fields(case, ['id', 'tensor', 'tensor_shape', 'rows', 'gather_order', 'shape', 'dtype',
                      'offset', 'bytes', 'sha256', 'sources', 'completed'], 'row case schema differs')
        name = identifier.removeprefix('sample:') if identifier != 'maximum_tile' else wide
        same(case['tensor'], name, 'frozen tensor selection differs')
        shape = geometry[name]; same(header[name]['shape'], shape, 'header geometry differs')
        columns = shape[-1]
        rows = [0] if len(shape) == 1 else [0, 1, shape[0] // 2, shape[0] - 1]
        gather = [0, 0] if len(shape) == 1 else [3, 0, 3, 2]
        if identifier == 'maximum_tile': rows, gather = list(range(1408, 1536)), [127, 0, 127, 64] + list(range(124))
        same(case['rows'], rows, 'frozen source row order differs')
        same(case['gather_order'], gather, 'frozen gather order differs')
        same(case['tensor_shape'], shape, 'frozen tensor shape differs')
        same(case['shape'], [len(rows), columns], 'frozen tensor shape differs')
        same(case['dtype'], '<f4', 'frozen array dtype differs'); same(case['completed'], True, 'incomplete row case')
        same(case['offset'], array_cursor, 'complete array coverage differs')
        same(case['bytes'], len(rows) * columns * 4, 'complete array extent differs')
        array = assets['arrays'][array_cursor:array_cursor + case['bytes']]
        same(digest(array), case['sha256'], 'case array hash differs')
        same(len(case['sources']), len(rows), 'complete source rows required')
        for index, row in enumerate(rows):
            offset = origin + header[name]['data_offsets'][0] + row * columns * 2
            raw = assets['raw'][raw_cursor:raw_cursor + columns * 2]
            same(case['sources'][index], dict(file_offset=offset, bytes=len(raw), raw_offset=raw_cursor, sha256=digest(raw)), 'official source span differs')
            same(raw, source_bytes(official, offset, columns * 2), 'official source bytes differ')
            same(array[index * columns * 4:(index + 1) * columns * 4], expanded(raw), 'frozen F32 words differ')
            values_compared += columns; raw_cursor += len(raw)
            if len(shape) == 1:
                prior = prior_vectors[name]
                converted = assets['raw'][raw_cursor:raw_cursor + prior['bytes']]
                same(converted, source_bytes(candidate, prior['offset'], prior['bytes']), 'candidate vector excerpt differs')
                same(digest(converted), prior['sha256'], 'candidate vector component differs')
                safe.require(raw == converted, 'candidate BF16 vector lineage differs')
                vector_relations.append(dict(tensor=name, bytes=len(raw), official_sha256=digest(raw),
                                             candidate_sha256=digest(converted), candidate_file_offset=prior['offset'],
                                             candidate_raw_offset=raw_cursor, prior_component_sha256=prior['sha256'],
                                             component_hash_matches=True, equal=True))
                raw_cursor += len(converted)
        array_cursor += len(array)
    same(m['candidate_vectors'], vector_relations, 'complete vector lineage records differ')
    same([p['tensor'] for p in m['projections']], [PREFIX + 'layers.0.self_attn.k_proj.weight', wide], 'projection selection differs')
    for projection in m['projections']:
        fields(projection, ['tensor', 'matrix_shape', 'input', 'raw_offset', 'raw_bytes',
                            'raw_sha256', 'references', 'completed'], 'projection schema differs')
        fields(projection['references'], ['fsum', 'ordered'], 'projection reference set differs')
        name = projection['tensor']; rows, columns = geometry[name]
        same(projection['matrix_shape'], [rows, columns], 'projection shape differs')
        inputs = [[((i % 7) - 3) * .125 for i in range(columns)], [(1 if i % 2 == 0 else -1) * (i % 5) * .0625 for i in range(columns)]]
        packed = b''.join(struct.pack('<f', value) for row in inputs for value in row)
        same(projection['input'], dict(shape=[2, columns], dtype='<f4', offset=array_cursor, bytes=len(packed), sha256=digest(packed)), 'projection input descriptor differs')
        same(assets['arrays'][array_cursor:array_cursor + len(packed)], packed, 'projection input words differ'); array_cursor += len(packed)
        same(projection['raw_offset'], raw_cursor, 'complete raw coverage differs')
        raw_bytes = rows * columns * 2
        same(projection['raw_bytes'], raw_bytes, 'projection source extent differs')
        raw_matrix = assets['raw'][raw_cursor:raw_cursor + raw_bytes]
        same(digest(raw_matrix), projection['raw_sha256'], 'projection source hash differs')
        fsum = [[0.] * rows for _ in range(2)]; ordered = [[0.] * rows for _ in range(2)]
        for row in range(rows):
            raw = raw_matrix[row * columns * 2:(row + 1) * columns * 2]
            same(raw, source_bytes(official, origin + header[name]['data_offsets'][0] + row * columns * 2, columns * 2), 'projection official source differs')
            weights = [value for (value,) in struct.iter_unpack('<f', expanded(raw))]
            for token in range(2):
                products = [a * b for a, b in zip(weights, inputs[token], strict=True)]
                fsum[token][row] = math.fsum(products)
                ordered[token][row] = functools.reduce(operator.add, products, 0.)
        raw_cursor += raw_bytes
        for label, expected in [('fsum', fsum), ('ordered', ordered)]:
            packed = b''.join(struct.pack('<d', value) for row in expected for value in row)
            actual = assets['arrays'][array_cursor:array_cursor + len(packed)]
            same(projection['references'][label], dict(shape=[2, rows], dtype='<f8', offset=array_cursor, bytes=len(packed), sha256=digest(actual)), 'scalar projection descriptor differs')
            same(actual, packed, 'scalar projection oracle differs'); array_cursor += len(packed)
        same(projection['completed'], True, 'incomplete projection oracle')
    same(array_cursor, len(assets['arrays']), 'complete array coverage differs')
    same(raw_cursor, len(assets['raw']), 'complete raw coverage differs')
    return dict(passed=True, row_cases=541, selected_matrices=278, complete_vectors=262,
                vector_lineage_matches=262, float32_values_compared=values_compared,
                maximum_tile=[128, 12288], scalar_projection_cases=2,
                official_graph_executed=False, arithmetic_profile_qualified=False, quality_scored=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--oracle', type=Path, required=True); parser.add_argument('--manifest-sha', required=True)
    args = parser.parse_args(); handles = []
    report = dict(passed=False, primary_error=None, endpoint_errors=[])
    def stop(number, _):
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(str(number))
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        official = open_source(MODEL, MODEL_ID); handles.append(official)
        candidate = open_source(CANDIDATE, CANDIDATE_ID); handles.append(candidate)
        report['replay'] = verify(args.oracle, args.manifest_sha, official, candidate)
        report['passed'] = True
    except BaseException as error:
        report['primary_error'] = type(error).__name__ + ': ' + str(error)
    finally:
        for sig in old: signal.signal(sig, signal.SIG_IGN)
        for file, path, original in handles:
            try:
                same(identity(os.fstat(file.fileno())), original, 'replay source endpoint differs')
                same(identity(path.lstat()), original, 'replay source path endpoint differs')
            except BaseException as error: report['endpoint_errors'].append(str(error))
            try: file.close()
            except BaseException as error: report['endpoint_errors'].append('source close: ' + str(error))
        if report['endpoint_errors']: report['passed'] = False
        for sig, handler in old.items(): signal.signal(sig, handler)
    print(json.dumps(report))
    raise SystemExit(0 if report['passed'] else 1)
