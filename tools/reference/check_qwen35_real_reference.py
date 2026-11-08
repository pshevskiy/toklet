"""Strict full-array comparison of closed independent and native one-input runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import check_qwen35_graph_capture as native_check
import check_qwen35_payload as pin
import qwen35_cpu_graph as graph
import qwen35_real_reference as producer
import qwen35_real_weights as adapter

need = adapter.need


def read(path, bound):
    need(not path.is_symlink() and path.is_file() and path.stat().st_size <= bound, 'Bounded regular evidence file required')
    with path.open('rb') as stream: raw = stream.read(bound+1)
    need(len(raw) <= bound, 'Evidence grew beyond bound')
    return raw


def decode(raw):
    def finite(value):
        number = float(value)
        need(math.isfinite(number), 'Nonfinite JSON number')
        return number
    def constant(value): raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=adapter.unique,
        parse_float=finite, parse_constant=constant)


def document(path, bound=2*1024**2): return decode(read(path,bound))


def exact(got, expected):
    if type(got) is not type(expected): return False
    if type(expected) is dict:
        return got.keys() == expected.keys() and all(exact(got[k], v) for k, v in expected.items())
    if type(expected) is list: return len(got) == len(expected) and all(exact(a,b) for a,b in zip(got,expected,strict=True))
    return got == expected


def key(item, token=0):
    need(type(item['token']) is int and item['token'] == token, 'Frozen token index differs')
    need(item['layer'] is None or type(item['layer']) is int and 0 <= item['layer'] < 32, 'Layer type/range differs')
    need(type(item['stage']) is str, 'Stage string required')
    name = item['stage']; match = re.fullmatch(r'(attention_(?:k_cache|v_cache|scores|probabilities))\.(\d+)', name)
    if match: need(match[2] == str(int(match[2])), 'Canonical head number required')
    return (item['layer'], match[1], int(match[2])) if match else (item['layer'], name, None)


def expected(tokens=1):
    need(type(tokens) is int and 1 <= tokens <= 512, 'Bounded integer context required')
    wanted = native_check.expected()
    for k in wanted:
        if k[1] in ['attention_k_cache', 'attention_v_cache', 'attention_scores', 'attention_probabilities']:
            wanted[k] *= tokens
    return wanted


def reference(folder, phase='one'):
    import numpy as np
    tokens, position, values = producer.plan(phase)
    receipt = document(folder/'checks.json')
    need(receipt['collection_passed'] is True and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
         and receipt['error'] is None and receipt['group_absent'] is True and receipt['changed_sources'] == [],
         'Closed successful owned independent collection required')
    peak = receipt['lifetime_peak_physical_bytes']
    need(type(peak) is int and 0 < peak <= 6*1024**3, 'Measured physical peak required')
    wanted_files = {'console.log', 'time-and-errors.log', 'inputs.json', 'runtime.json', 'native.json',
                    'capture/result.json', 'capture/rows.bin'}
    need(set(receipt['files_sha256']) == wanted_files, 'Sealed independent file inventory differs')
    for name, digest in receipt['files_sha256'].items():
        need(hashlib.sha256(read(folder/name, producer.MAX_CAPTURE)).hexdigest() == digest, 'Sealed independent evidence changed')
    result = document(folder/'capture/result.json')
    need(result['completed'] is True and result['primary_error'] is None and result['endpoint_errors'] == []
         and result['format'] == 'toklet-qwen35-real-reference-v1', 'Complete numerical result required')
    for name, value in dict(repository=adapter.REPOSITORY, revision=adapter.REVISION, source_sha256=pin.FILE_SHA,
        tokens=tokens, position_offset=position, geometry=adapter.GEOMETRY, gate=graph.GATE,
        planned_rows_bytes=values*12, ledger=producer.LEDGER, engine_executed=False,
        contextual_numerical_qualified=False, quality_qualified=False, managed_adoption_qualified=False).items():
        need(exact(result[name], value), 'Frozen independent contract differs: ' + name)
    need(result.get('phase', 'one') == phase, 'Named reference phase differs')
    source = result['source']
    need(source['closed'] is True and source['poisoned'] is False and source['poison_reason'] is None
         and source['bytes'] == pin.FILE_BYTES and type(source['bytes']) is int
         and source['sha256'] == result['source_endpoint_sha256'] == pin.FILE_SHA, 'Closed authenticated source endpoint required')
    need(type(source['reads']) is int and source['reads'] > 0 and type(source['read_bytes']) is int
         and source['read_bytes'] >= pin.FILE_BYTES, 'Source read audit differs')
    inputs = document(folder/'inputs.json')
    freeze_raw = read(folder.parent/'adapter-freeze.json', 2*1024**2)
    freeze = json.loads(freeze_raw, object_pairs_hook=adapter.unique)
    need(hashlib.sha256(freeze_raw).hexdigest() == inputs['freeze_sha256']
         and freeze['frozen_before_real_reference'] is True
         and exact(freeze['gate'], graph.GATE), 'Pre-execution producer freeze differs')
    need(hashlib.sha256(read(folder.parent/'acceptance-contract.json', 2*1024**2)).hexdigest()
         == freeze['contract_sha256'], 'Frozen acceptance contract differs')
    runtime = document(folder/'runtime.json')
    expected_sources = {p: h for p, h in freeze['files'].items()
        if p not in ['tools/reference/reference_runtime.py', 'tools/reference/test_qwen35_real_reference.py']}
    need(len(expected_sources) == len(inputs['source_pins']) == 7, 'Exact producer source inventory required')
    for relative, digest in expected_sources.items():
        candidates = [p for p in inputs['source_pins'] if Path(p).parts[-len(Path(relative).parts):] == Path(relative).parts]
        need(len(candidates) == 1 and inputs['source_pins'][candidates[0]] == digest
             and runtime['files'][candidates[0]]['sha256'] == digest, 'Frozen producer runtime source differs')
    for name in ['runtime', 'native']:
        need(hashlib.sha256(read(folder/(name+'.json'), 2*1024**2)).hexdigest() == inputs[name+'_manifest_sha256'],
             'Pinned runtime manifest differs')
    need(result['runtime_admission']['admitted']['manifest_sha256'] == inputs['runtime_manifest_sha256']
         and result['runtime_admission']['native_manifest_sha256'] == inputs['native_manifest_sha256']
         and result['runtime_endpoint']['numpy_modules'] and result['runtime_endpoint']['non_system_native_images'],
         'Runtime closure admission/endpoint missing')
    allowed = {**runtime['files'], **document(folder/'native.json')['files']}
    for name, value in result['runtime_endpoint']['non_system_native_images'].items():
        need(name in allowed and exact(value, allowed[name]), 'Loaded native image pin differs')
    need(all(p in runtime['files'] for p in result['runtime_endpoint']['numpy_modules'].values()),
         'Loaded NumPy origin outside admitted runtime')
    raw = read(folder/'capture/rows.bin', producer.MAX_CAPTURE)
    need(len(raw) == values*12 and hashlib.sha256(raw).hexdigest() == result['rows_sha256'], 'Independent body identity differs')
    need(type(result['arrays']) is list and len(result['arrays']) == producer.ARRAYS*2, 'Complete independent array inventory required')
    wanted = expected(len(tokens)); rows = {}; offset = 0
    for profile, width, dtype in [('ideal_fp64', 8, '<f8'), ('staged_fp32', 4, '<f4')]:
        selected = result['arrays'][len(rows)*producer.ARRAYS:(len(rows)+1)*producer.ARRAYS]
        arrays = {}
        for item in selected:
            need(set(item) == {'profile', 'token', 'layer', 'stage', 'count', 'offset', 'bytes', 'sha256'}, 'Exact array descriptor fields required')
            k = key(item, len(tokens)-1)
            need(item['profile'] == profile and k in wanted and k not in arrays, 'Array profile/owner/uniqueness differs')
            for field in ['count', 'offset', 'bytes']: need(type(item[field]) is int, 'Strict integer array extents required')
            need(item['count'] == wanted[k] and item['offset'] == offset and item['bytes'] == wanted[k]*width, 'Complete array geometry differs')
            block = memoryview(raw)[offset:offset+item['bytes']]
            need(len(block) == item['bytes'] and hashlib.sha256(block).hexdigest() == item['sha256'], 'Array span hash differs')
            data = np.frombuffer(block, dtype=dtype)
            need(np.isfinite(data).all(), 'Nonfinite independent array')
            arrays[k] = data; offset += len(block)
        need(set(arrays) == set(wanted), 'Missing independent stages')
        rows[profile] = arrays
    need(offset == len(raw), 'Trailing independent body bytes')
    return rows, peak


def statistics(actual, expected):
    import numpy as np
    need(actual.shape == expected.shape and actual.size > 0 and np.isfinite(actual).all()
         and np.isfinite(expected).all(), 'Finite complete equal-shaped arrays required')
    error = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    ratio = error/(1e-5+1e-4*np.abs(expected.astype(np.float64)))
    return dict(values=int(actual.size), failed_values=int(np.count_nonzero(ratio > 1)),
        maximum_gate_ratio=float(ratio.max()), maximum_absolute_error=float(error.max()))


def check(native, independent, model):
    import numpy as np
    need(np.__version__ == '2.3.2', 'Pinned NumPy required for offline comparison')
    native_receipt = native_check.check(native, model)
    expected, peak = reference(independent)
    log = read(native/'console.log', 2*1024**2).decode()
    descriptors = [json.loads(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith('QWEN35_GRAPH_ARRAY ')]
    raw = read(native/'capture/rows.f32le', 128*1024**2)
    rows = []; greedy = []
    for item in descriptors:
        k = native_check.key(item)
        actual = np.frombuffer(memoryview(raw)[item['offset']:item['offset']+item['bytes']], dtype='<f4')
        for profile, arrays in expected.items():
            want = arrays[k]
            rows.append(dict(profile=profile, layer=k[0], stage=k[1], head=k[2], **statistics(actual,want)))
            if k == (None, 'logits', None):
                greedy.append(dict(profile=profile, native_id=int(actual.argmax()), reference_id=int(want.argmax())))
    mismatches = [v for v in greedy if v['native_id'] != v['reference_id']]
    passed = all(v['failed_values'] == 0 for v in rows) and not mismatches
    return dict(one_input_coupled_fidelity_passed=passed, compared_values=sum(v['values'] for v in rows),
        arrays=len(rows), rows=rows, greedy=greedy, unresolved_greedy_mismatches=mismatches,
        gate=graph.GATE, native_capture_sha256=native_receipt['capture_sha256'],
        native_peak_bytes=native_receipt['lifetime_peak_physical_bytes'], independent_peak_bytes=peak,
        contextual_numerical_qualified=False, quality_qualified=False, managed_adoption_qualified=False,
        scope='All stages/state/full logits at rawinput1 offset0 only; not nonzero recurrence, contextual/generation/codec/quality/GPU/performance/adoption qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path); parser.add_argument('independent', type=Path)
    parser.add_argument('model', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = check(args.native, args.independent, args.model)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'greedy']}))
    raise SystemExit(0 if result['one_input_coupled_fidelity_passed'] else 1)
