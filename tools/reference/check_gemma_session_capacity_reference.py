"""Independent stdlib reconstruction of the pre-target KV capacity witness."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import struct
import tempfile
import check_gemma_hybrid_tiny as safe


def digest(raw): return hashlib.sha256(raw).hexdigest()


def frozen(directory, manifest_sha):
    raw = safe.read(directory / 'manifest.json', 64 * 1024)
    safe.require(digest(raw) == manifest_sha, 'external capacity oracle differs')
    data = safe.document(raw); binary = safe.read(directory / 'kv.bin', 32 * 1024)
    safe.same(set(data), {'format', 'window', 'original_global_capacity', 'original_max_positions', 'layers',
                         'producer_count', 'routes', 'producers', 'cases', 'binary', 'real_model',
                         'numerical_graph_executed', 'larger_context_qualified', 'full_model_window_qualified', 'app_eligibility_changed'})
    for key, expected in [('format', 'toklet-session-capacity-oracle-v1'), ('window', 512), ('original_global_capacity', 32),
                          ('original_max_positions', 512), ('layers', 35), ('producer_count', 15)]: safe.same(data[key], expected)
    safe.same(data['binary'], dict(file='kv.bin', bytes=len(binary), sha256=digest(binary)))
    safe.require(len(binary) == 16896, 'complete capacity binary required')
    expected_routes = list(range(15)) + [14 if (layer + 1) % 5 == 0 else 13 for layer in range(15, 35)]
    safe.require(data['routes'] == expected_routes, 'capacity producer routes differ')
    safe.require(len(data['producers']) == 15, 'complete capacity producers required')
    cursor = 0
    for producer, actual in enumerate(data['producers']):
        full = (producer + 1) % 5 == 0; dims = 3 if full else 2
        safe.same(set(actual), {'producer', 'kind', 'heads', 'head_dim', 'k', 'v'})
        safe.same({k: actual[k] for k in ('producer', 'kind', 'heads', 'head_dim')},
                  dict(producer=producer, kind='full' if full else 'sliding', heads=2, head_dim=dims))
        for kind in ('k', 'v'):
            count = 2 * 32 * dims * 4; body = binary[cursor:cursor + count]
            expected = bytearray()
            for head in range(2):
                for position in range(32):
                    for column in range(dims):
                        if kind == 'k': value = (4096 * producer + 512 * head + 32 * position + 4 * column + 1) / 4
                        else: value = -(1024 * producer + 128 * head + 8 * position + 2 * column + 1) / 2
                        expected.extend(struct.pack('<f', value))
            safe.require(body == expected, 'independent capacity word recipe differs')
            safe.same(actual[kind], dict(shape=[2, 32, dims], dtype='<f4', offset=cursor, bytes=count, sha256=digest(body)))
            cursor += count
    safe.same(cursor, len(binary))
    cases = []
    for capacity in (1, 2, 3, 4, 5, 17, 32):
        for rows in sorted({1, min(capacity, 3), min(capacity, 8), capacity}):
            steps = len(range(0, capacity, rows))
            cases.append(dict(capacity=capacity, chunk_rows=rows, steps=steps, view_records=70 * steps))
    safe.require(data['cases'] == cases, 'fixed capacity schedules differ')
    for key in ('real_model', 'numerical_graph_executed', 'larger_context_qualified', 'full_model_window_qualified', 'app_eligibility_changed'):
        safe.same(data[key], False)
    return data, binary


def controls(directory, manifest_sha):
    original, binary = frozen(directory, manifest_sha); results = []
    for mode, message in [('route', 'capacity producer routes differ'), ('count', 'complete capacity producers required'),
                          ('schedule', 'fixed capacity schedules differ'), ('word', 'independent capacity word recipe differs'),
                          ('head_order', 'independent capacity word recipe differs')]:
        data = copy.deepcopy(original); body = bytearray(binary)
        if mode == 'route': data['routes'][-1] = 13
        elif mode == 'count': data['producers'].pop()
        elif mode == 'schedule': data['cases'][-1]['capacity'] = 33
        elif mode == 'word': body[0] ^= 1
        else:
            stride = 32 * 2 * 4; body[:2 * stride] = body[stride:2 * stride] + body[:stride]
        data['binary'].update(bytes=len(body), sha256=digest(body))
        for producer in data['producers']:
            for kind in ('k', 'v'):
                item = producer[kind]; start = item['offset']; item['sha256'] = digest(body[start:start + item['bytes']])
        raw = (json.dumps(data, indent=2, allow_nan=False) + '\n').encode()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder); (output / 'manifest.json').write_bytes(raw); (output / 'kv.bin').write_bytes(body)
            try: frozen(output, digest(raw))
            except ValueError as error:
                safe.same(str(error), message); results.append(dict(control=mode, intended_error=message, rejected=True))
            else: raise ValueError('coherent capacity control accepted: ' + mode)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True); parser.add_argument('--controls', action='store_true'); args = parser.parse_args()
    data, _ = frozen(args.fixture, args.manifest_sha)
    result = dict(passed=True, cases=len(data['cases']), view_records=sum(case['view_records'] for case in data['cases']))
    if args.controls: result['controls'] = controls(args.fixture, args.manifest_sha)
    print(json.dumps(result))
