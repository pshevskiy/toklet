"""Freeze official BF16 row/projection oracles without NumPy or target imports.

Only selected rows and complete small vectors are expanded. Two projection
oracles use scalar FP64 accumulation over exact BF16 values. No graph executes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import signal
import stat
import struct
import time
import check_gemma_hybrid_tiny as safe

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'models/Gemma4-E2B-it-BF16-70af34e/model.safetensors'
CANDIDATE = ROOT / 'models/MLX-Gemma4-E2B-4bit-2387675/model.safetensors'
SIZE = 10246621918
SHA = '2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550'
HEADER_SHA = '12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e'
INVENTORY_SHA = 'a3dff3b9d825fcf4fa6f391262732c5f429345682d683d6a39bc551ddacd68e2'
PREFIX = 'model.language_model.'
MAX_BINARY = 64 * 1024**2
MAX_READ = 128 * 12288 * 2


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def guard(file, path, original):
    safe.require(identity(os.fstat(file.fileno())) == original == identity(path.lstat()),
                 'oracle source identity changed')


def read_exact(file, path, original, offset, count, deadline):
    safe.require(type(offset) is int and type(count) is int and 0 <= offset
                 and 0 < count <= MAX_READ and offset + count <= original[3], 'bounded oracle read')
    safe.require(time.monotonic() < deadline, 'oracle deadline')
    guard(file, path, original)
    raw = os.pread(file.fileno(), count, offset)
    safe.require(len(raw) == count, 'oracle row truncated')
    guard(file, path, original)
    return raw


def authenticate(path, size, digest, deadline):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    file = os.fdopen(fd, 'rb')
    try:
        value = os.fstat(fd)
        safe.require(stat.S_ISREG(value.st_mode) and value.st_size == size, 'oracle regular exact-size source')
        original = identity(value); guard(file, path, original)
        total = 0; actual = hashlib.sha256()
        while True:
            safe.require(time.monotonic() < deadline, 'oracle authentication deadline')
            raw = file.read(min(1024**2, size - total + 1))
            if not raw: break
            total += len(raw); safe.require(total <= size, 'oracle source grew'); actual.update(raw)
        guard(file, path, original)
        safe.require(total == size and actual.hexdigest() == digest, 'oracle full source SHA differs')
        return file, original
    except BaseException as error:
        try: file.close()
        except BaseException as cleanup: error.add_note('oracle authentication cleanup: ' + str(cleanup))
        raise


def selected_geometry(header, inventory):
    """Map prior independent geometry to original names, not target row views."""
    result = {}
    for value in inventory['matrices'] + inventory['vectors']:
        old = value['name']; safe.require(old.startswith('language_model.model.'), 'prior text tensor name')
        name = PREFIX + old[len('language_model.model.'):]
        shape = [value['rows'], value['columns']] if value['kind'] == 'matrix' else [value['count']]
        tensor = header[name]
        safe.same(tensor['dtype'], 'BF16'); safe.same(tensor['shape'], shape)
        safe.same(tensor['data_offsets'][1] - tensor['data_offsets'][0], 2 * math.prod(shape))
        safe.require(name not in result, 'unique selected tensor')
        result[name] = dict(shape=shape, candidate=value if value['kind'] == 'vector' else None)
    safe.same(len(result), 540)
    unused = {f'{PREFIX}layers.{layer}.self_attn.{suffix}' for layer in range(15, 35)
              for suffix in ('k_norm.weight', 'k_proj.weight', 'v_proj.weight')}
    safe.same({name for name in header if name.startswith(PREFIX)}, set(result) | unused)
    for name in unused:
        layer = int(name.split('.')[3]); dim = 512 if layer % 5 == 4 else 256
        shape = [dim] if name.endswith('k_norm.weight') else [dim, 1536]
        safe.same(header[name]['dtype'], 'BF16'); safe.same(header[name]['shape'], shape)
        safe.same(header[name]['data_offsets'][1] - header[name]['data_offsets'][0], 2 * math.prod(shape))
    safe.same(sum(len(item['shape']) == 2 for item in result.values()), 278)
    return result, sorted(unused)


def expand(raw):
    """Independent IEEE bit expansion, one bounded row; rejects every nonfinite."""
    safe.require(len(raw) % 2 == 0, 'complete BF16 words')
    words = [value for (value,) in struct.iter_unpack('<H', raw)]
    safe.require(all(value & 0x7f80 != 0x7f80 for value in words), 'nonfinite official BF16 word')
    return b''.join(struct.pack('<I', value << 16) for value in words)


def prepare(output):
    output.mkdir()
    report = dict(format='toklet-gemma-bf16-row-oracle-v1', passed=False, primary_error=None,
                  endpoint_errors=[], model=dict(bytes=SIZE, sha256=SHA), cases=[], projections=[],
                  candidate_vectors=[], arrays=None, raw=None, official_graph_executed=False,
                  arithmetic_profile_qualified=False, model_fit_qualified=False, quality_scored=False,
                  app_eligibility_changed=False, full_model_window_qualified=False,
                  projection_policy=dict(fp32_atol=1e-5, fp32_rtol=1e-4,
                                         fp64_atol=1e-13, fp64_rtol=1e-13))
    resources = []; sources = []; deadline = time.monotonic() + 900
    old = {}
    def interrupted(number, _):
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(str(number))
    try:
        old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        inventory_raw = safe.read(ROOT / 'tests/fixtures/gemma-resident-storage-reference-v1/inventory.json', 2 * 1024**2)
        safe.same(sha(inventory_raw), INVENTORY_SHA); inventory = safe.document(inventory_raw)
        model, model_id = authenticate(MODEL, SIZE, SHA, deadline); sources.append((model, MODEL, model_id))
        candidate, candidate_id = authenticate(CANDIDATE, 3550670554, '038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e', deadline)
        sources.append((candidate, CANDIDATE, candidate_id))
        first = read_exact(model, MODEL, model_id, 0, 8, deadline)
        length = struct.unpack('<Q', first)[0]; safe.same(length, 263952)
        header_raw = read_exact(model, MODEL, model_id, 8, length, deadline)
        safe.same(sha(header_raw), HEADER_SHA); header = safe.document(header_raw); origin = 8 + length
        safe.same(header['__metadata__'], dict(format='pt')); safe.same(len(header), 2012)
        spans = sorted(tuple(value['data_offsets']) for name, value in header.items() if name != '__metadata__')
        safe.same(spans[0][0], 0); safe.same(spans[-1][1], SIZE - origin)
        safe.require(all(a[1] == b[0] for a, b in zip(spans, spans[1:])), 'complete header extent coverage')
        geometry, unused = selected_geometry(header, inventory)
        report.update(header=dict(bytes=length, sha256=HEADER_SHA), inventory_sha256=INVENTORY_SHA,
                      excluded_shared_kv=unused, candidate_model=dict(bytes=3550670554, sha256='038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e'))
        arrays = (output / 'arrays.bin').open('xb'); resources.append(arrays)
        raw_file = (output / 'raw.bin').open('xb'); resources.append(raw_file)
        journal = (output / 'capture.jsonl').open('x'); resources.append(journal)
        def event(value):
            journal.write(json.dumps(value, sort_keys=True) + '\n'); journal.flush()
        def emit(file, raw):
            start = file.tell(); safe.require(start + len(raw) <= MAX_BINARY, 'bounded oracle binary')
            safe.same(file.write(raw), len(raw)); file.flush(); return start
        def capture_rows(identifier, name, ids, gather):
            shape = geometry[name]['shape']; columns = shape[-1]
            entry = header[name]; base = origin + entry['data_offsets'][0]
            case = dict(id=identifier, tensor=name, tensor_shape=shape, rows=ids, gather_order=gather,
                        shape=[len(ids), columns], dtype='<f4', offset=arrays.tell(), bytes=0,
                        sha256=None, sources=[], completed=False)
            report['cases'].append(case); event(dict(kind='case_started', id=identifier))
            digest = hashlib.sha256()
            for index in ids:
                source_offset = base + index * columns * 2
                raw = read_exact(model, MODEL, model_id, source_offset, columns * 2, deadline)
                raw_offset = emit(raw_file, raw)
                case['sources'].append(dict(file_offset=source_offset, bytes=len(raw), raw_offset=raw_offset, sha256=sha(raw)))
                expanded = expand(raw); emit(arrays, expanded); digest.update(expanded); case['bytes'] += len(expanded)
                if len(shape) == 1:
                    prior = geometry[name]['candidate']['source']
                    converted = read_exact(candidate, CANDIDATE, candidate_id, prior['offset'], prior['bytes'], deadline)
                    candidate_offset = emit(raw_file, converted)
                    relation = dict(tensor=name, bytes=len(raw), official_sha256=sha(raw),
                                    candidate_sha256=sha(converted), candidate_file_offset=prior['offset'],
                                    candidate_raw_offset=candidate_offset, prior_component_sha256=prior['sha256'],
                                    component_hash_matches=sha(converted) == prior['sha256'], equal=raw == converted)
                    report['candidate_vectors'].append(relation); event(dict(kind='vector_attempt', **relation))
                    safe.require(relation['component_hash_matches'], 'candidate vector component differs')
                    safe.require(relation['equal'], 'unquantized candidate vector differs')
            case.update(sha256=digest.hexdigest(), completed=True); event(dict(kind='case_completed', id=identifier, sha256=case['sha256']))
        for name, item in sorted(geometry.items()):
            if len(item['shape']) == 1: ids, gather = [0], [0, 0]
            else:
                rows = item['shape'][0]; ids, gather = [0, 1, rows // 2, rows - 1], [3, 0, 3, 2]
            capture_rows('sample:' + name, name, ids, gather)
        wide = PREFIX + 'layers.15.mlp.down_proj.weight'
        capture_rows('maximum_tile', wide, list(range(1408, 1536)), [127, 0, 127, 64] + list(range(124)))
        for name in [PREFIX + 'layers.0.self_attn.k_proj.weight', wide]:
            rows, columns = geometry[name]['shape']; base = origin + header[name]['data_offsets'][0]
            inputs = [[(i % 7 - 3) / 8 for i in range(columns)], [(-1)**i * (i % 5) / 16 for i in range(columns)]]
            packed_input = b''.join(struct.pack('<f', value) for row in inputs for value in row)
            projection = dict(tensor=name, matrix_shape=[rows, columns], input=dict(shape=[2, columns], dtype='<f4', offset=emit(arrays, packed_input), bytes=len(packed_input), sha256=sha(packed_input)), raw_offset=raw_file.tell(), raw_bytes=0, raw_sha256=None, references={}, completed=False)
            report['projections'].append(projection); event(dict(kind='projection_started', tensor=name))
            fsum = [[0.] * rows for _ in range(2)]; ordered = [[0.] * rows for _ in range(2)]; raw_digest = hashlib.sha256()
            for row in range(rows):
                raw = read_exact(model, MODEL, model_id, base + row * columns * 2, columns * 2, deadline)
                emit(raw_file, raw); raw_digest.update(raw); projection['raw_bytes'] += len(raw)
                values = [value for (value,) in struct.iter_unpack('<f', expand(raw))]
                for token in range(2):
                    products = [a * b for a, b in zip(inputs[token], values, strict=True)]
                    fsum[token][row] = math.fsum(products)
                    total = 0.
                    for value in products: total += value
                    ordered[token][row] = total
            projection['raw_sha256'] = raw_digest.hexdigest()
            for label, values in [('fsum', fsum), ('ordered', ordered)]:
                raw = b''.join(struct.pack('<d', value) for row in values for value in row)
                safe.require(all(math.isfinite(value) for row in values for value in row), 'finite projection oracle')
                projection['references'][label] = dict(shape=[2, rows], dtype='<f8', offset=emit(arrays, raw), bytes=len(raw), sha256=sha(raw))
            projection['completed'] = True; event(dict(kind='projection_completed', tensor=name))
        safe.same(len(report['cases']), 541); safe.same(len(report['candidate_vectors']), 262)
    except BaseException as error:
        report['primary_error'] = type(error).__name__ + ': ' + str(error)
    finally:
        for sig in old: signal.signal(sig, signal.SIG_IGN)
        for file, path, original in sources:
            try: guard(file, path, original)
            except BaseException as error: report['endpoint_errors'].append('source: ' + str(error))
            try: file.close()
            except BaseException as error: report['endpoint_errors'].append('source close: ' + str(error))
        for file in resources:
            try: file.close()
            except BaseException as error: report['endpoint_errors'].append('output close: ' + str(error))
        for key, filename in [('arrays', 'arrays.bin'), ('raw', 'raw.bin')]:
            path = output / filename
            if path.exists():
                try:
                    raw = safe.read(path, MAX_BINARY); report[key] = dict(file=filename, bytes=len(raw), sha256=sha(raw))
                except BaseException as error: report['endpoint_errors'].append('capture ' + key + ': ' + str(error))
        report['passed'] = report['primary_error'] is None and not report['endpoint_errors']
        try:
            with (output / 'manifest.json').open('x') as file: json.dump(report, file, indent=2, allow_nan=False); file.write('\n')
        except BaseException as error:
            report['endpoint_errors'].append('manifest capture: ' + str(error)); report['passed'] = False
        finally:
            for sig, handler in old.items(): signal.signal(sig, handler)
    print(json.dumps({key: report.get(key) for key in ['passed', 'primary_error', 'endpoint_errors', 'arrays', 'raw']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(prepare(parser.parse_args().output))
