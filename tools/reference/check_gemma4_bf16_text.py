"""Stdlib plan/replay for original-BF16 mathematical graph preparation.

Does not import the producer, weight adapter, graph or numerical runtime.
The old packed fixture supplies geometry/order only, never expected values.
"""
from array import array
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import signal
import struct
import sys
import check_gemma4_bf16_rows_oracle as source

ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'original-bf16-weights-fp32-storage-fp64-reductions-decimal80-rotary-v1'
TOKENS = {'original': [2, 100, 257, 100], 'future': [2, 100, 257, 101]}
LEDGER_MIB = dict(runtime_blas=320, tile_conversion=64, graph_state=32,
                  head_capture=48, comparison=24, vectors_views=4,
                  metadata_authentication=8, margin=12)
GATE = dict(atol=1e-5, rtol=1e-4)


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def layout():
    raw = source.safe.read(ROOT / 'tests/fixtures/gemma-unquantized-header-v1/layout.json', 2 * 1024**2)
    source.same(source.digest(raw), '2419f70e7232fd75a6d361e0ef8e8acaf37f87f52124e63e1e04e814d1040840', 'official graph layout differs')
    return source.safe.document(raw)['selected_text']


def specifications():
    raw = source.safe.read(ROOT / 'tests/fixtures/gemma4-text-math-v1/math.json', 2 * 1024**2)
    source.same(source.digest(raw), 'f60c9f355b9008cfba6acd5d39c89c108de2d1ed5a3c8013a3a595576d8f910b', 'frozen graph inventory differs')
    previous = source.safe.document(raw)['arrays']; result = []
    for sequence in TOKENS:
        cursor = 0
        for item in previous:
            shape = item['source_shape']; count = math.prod(shape) * (8 if item['dtype'] == '<f8' else 4)
            result.append(dict(sequence=sequence, contract=item['contract'], name=item['name'],
                               shape=shape, dtype=item['dtype'], offset=cursor, bytes=count))
            cursor += count
        source.same(cursor, 103403520, 'complete graph file plan differs')
    source.same(len(result), 2852, 'complete graph inventory differs')
    return result


def close_source(handle, primary):
    file, path, original = handle; errors = []
    try:
        source.same(source.identity(os.fstat(file.fileno())), original, 'graph source endpoint differs')
        source.same(source.identity(path.lstat()), original, 'graph source path differs')
    except BaseException as error: errors.append(str(error))
    try: file.close()
    except BaseException as error: errors.append('graph source close: ' + str(error))
    if errors:
        if primary is not None: primary.add_note('; '.join(errors))
        else: raise ValueError('; '.join(errors))


def read_spans(specs, selected):
    for name, item in sorted(selected.items()):
        if len(item['shape']) == 1: yield item['file_offset'], 2 * item['shape'][0]
    for spec in specs:
        name = spec['name']; ids = TOKENS[spec['sequence']]
        if name in ('embedding', 'ple_lookup'):
            full = 'model.language_model.' + ('embed_tokens.weight' if name == 'embedding' else 'embed_tokens_per_layer.weight')
            item = selected[full]; width = item['shape'][1] * 2
            for token in ids: yield item['file_offset'] + token * width, width
            continue
        name = {'ple_projection': 'per_layer_model_projection.weight', 'head': 'embed_tokens.weight'}.get(name, name.replace('layer.', 'layers.', 1) + '.weight')
        item = selected.get('model.language_model.' + name)
        if item is None or len(item['shape']) != 2: continue
        rows, columns = item['shape']
        for start in range(0, rows, 128): yield item['file_offset'] + start * columns * 2, min(128, rows - start) * columns * 2


def prepare():
    specs = specifications(); selected = layout()
    source.same(sum(LEDGER_MIB.values()), 512, 'lifetime memory allowance differs')
    handle = source.open_source(source.MODEL, source.MODEL_ID); primary = None
    chain = hashlib.sha256(b'toklet-bf16-source-reads-v1\0'); reads = count = 0
    try:
        for offset, size in read_spans(specs, selected):
            body = source.source_bytes(handle, offset, size); reads += 1; count += size
            chain.update(struct.pack('<QQQ', reads, offset, size)); chain.update(hashlib.sha256(body).digest())
    except BaseException as error:
        primary = error; raise
    finally: close_source(handle, primary)
    return dict(format='toklet-gemma-bf16-text-plan-v1', arithmetic_profile=PROFILE, tokens=TOKENS,
                captures=specs, file_bytes=103403520, total_array_bytes=206807040,
                memory_allowance_bytes=512 * 1024**2, memory_ledger_mib=LEDGER_MIB, gate=GATE,
                source_audit=dict(**source.MODEL_ID, reads=reads, read_bytes=count,
                                  read_chain_sha256=chain.hexdigest(), poisoned=False, poison_reason=None, closed=True),
                independent_full_graph_agreement=False, upstream_bf16_runtime_executed=False,
                model_fit_qualified=False, quality_scored=False, app_eligibility_changed=False,
                full_model_window_qualified=False)


def load_plan(path, expected_sha):
    raw = source.safe.read(path, 4 * 1024**2); source.same(source.digest(raw), expected_sha, 'external graph plan differs')
    result = source.safe.document(raw)
    source.same(result['format'], 'toklet-gemma-bf16-text-plan-v1', 'graph plan format differs')
    source.same(result['arithmetic_profile'], PROFILE, 'graph arithmetic profile differs')
    source.same(result['tokens'], TOKENS, 'graph tokens differ')
    source.same(result['captures'], specifications(), 'graph capture plan differs')
    source.same(result['file_bytes'], 103403520, 'graph file bound differs')
    source.same(result['total_array_bytes'], 206807040, 'graph array bound differs')
    source.same(result['memory_ledger_mib'], LEDGER_MIB, 'graph lifetime ledger differs')
    source.same(result['memory_allowance_bytes'], 512 * 1024**2, 'graph memory bound differs')
    source.same(result['gate'], GATE, 'graph gates differ')
    source.same({key: result['source_audit'][key] for key in ('bytes', 'sha256')}, source.MODEL_ID, 'planned official source differs')
    count = total = 0
    for _, size in read_spans(result['captures'], layout()): count += 1; total += size
    audit = result['source_audit']
    source.same(set(audit), {'bytes', 'sha256', 'reads', 'read_bytes', 'read_chain_sha256', 'poisoned', 'poison_reason', 'closed'}, 'planned audit schema differs')
    source.same(audit['reads'], count, 'planned graph read count differs'); source.same(audit['read_bytes'], total, 'planned graph read bytes differ')
    source.same(audit['poisoned'], False, 'planned poisoned source'); source.same(audit['poison_reason'], None, 'planned poison reason')
    source.same(audit['closed'], True, 'planned source close differs')
    source.safe.require(type(audit['read_chain_sha256']) is str and len(audit['read_chain_sha256']) == 64
                        and all(c in '0123456789abcdef' for c in audit['read_chain_sha256']), 'planned graph read-chain identity required')
    for key in ('independent_full_graph_agreement', 'upstream_bf16_runtime_executed', 'model_fit_qualified',
                'quality_scored', 'app_eligibility_changed', 'full_model_window_qualified'):
        source.same(result[key], False, 'unsupported graph qualification')
    return result


def values(handle, spec):
    file, path, original = handle; dtype = 'd' if spec['dtype'] == '<f8' else 'f'; result = array(dtype)
    digest = hashlib.sha256(); offset = spec['offset']; remaining = spec['bytes']
    while remaining:
        count = min(1024**2, remaining); body = source.source_bytes(handle, offset, count)
        digest.update(body); result.frombytes(body); offset += len(body); remaining -= len(body)
    if sys.byteorder != 'little': result.byteswap()
    source.same(digest.hexdigest(), spec['sha256'], 'saved graph state differs')
    source.safe.require(all(math.isfinite(v) for v in result), 'nonfinite saved graph state')
    return result


def compare(actual, expected):
    source.same(len(actual), len(expected), 'graph comparison extent differs')
    maximum = 0.; failures = 0
    for a, b in zip(actual, expected, strict=True):
        fraction = abs(a - b) / (GATE['atol'] + GATE['rtol'] * abs(b))
        maximum = max(maximum, fraction); failures += fraction > 1.
    return dict(values=len(actual), failed_values=failures, maximum_gate_fraction=maximum, passed=failures == 0)


def predictions(state, vocab):
    result = []
    for start in range(0, len(state), vocab):
        winner = 0; first = second = -math.inf
        for index in range(vocab):
            value = state[start + index]
            if value > first: second, first, winner = first, value, index
            elif value > second: second = value
        result.append(dict(token=winner, margin=first - second))
    return result


def boundaries(plan, specs, files, official, raw_file, report):
    selected = layout()
    def read(name, row):
        item = selected['model.language_model.' + name]; width = item['shape'][-1] * 2
        offset = item['file_offset'] + row * width; body = source.source_bytes(official, offset, width)
        record = dict(tensor=name, row=row, file_offset=offset, offset=raw_file.tell(), bytes=len(body), sha256=source.digest(body), captured=False)
        report['boundary_sources'].append(record)
        source.safe.require(raw_file.tell() + len(body) <= 4 * 1024**2, 'bounded graph boundary sources')
        source.same(raw_file.write(body), len(body), 'complete boundary source write required'); raw_file.flush(); record['captured'] = True
        return [v for (v,) in struct.iter_unpack('<f', source.expanded(body))]
    def get(sequence, contract, name):
        return values(files[sequence], specs[(sequence, contract, name)])
    def check(label, actual, expected):
        item = dict(label=label, **compare(actual, expected)); report['boundaries'].append(item)
        source.safe.require(item['passed'], 'independent graph boundary differs: ' + label)
    for sequence, ids in TOKENS.items():
        for contract in ('math_fp64', 'staged_fp32'):
            def rounded(value): return struct.unpack('<f', struct.pack('<f', value))[0] if contract == 'staged_fp32' else value
            label = sequence + '/' + contract + '/'
            expected = [v for token in ids for v in read('embed_tokens.weight', token)]
            check(label + 'embedding', get(sequence, contract, 'embedding'), expected)
            expected = [rounded(v * rounded(math.sqrt(1536))) for v in expected]
            check(label + 'scaled_embedding', get(sequence, contract, 'scaled_embedding'), expected)
            scale = read('layers.0.input_layernorm.weight', 0); normalized = []
            epsilon = struct.unpack('<f', struct.pack('<f', 1e-6))[0]
            for start in range(0, len(expected), 1536):
                row = expected[start:start + 1536]; divisor = math.sqrt(math.fsum(v * v for v in row) / 1536 + epsilon)
                normalized.extend(rounded(v / divisor * s) for v, s in zip(row, scale, strict=True))
            check(label + 'layer.0.input_layernorm', get(sequence, contract, 'layer.0.input_layernorm'), normalized)
            projected = [0.] * (4 * 256)
            for row in range(256):
                weights = read('layers.0.self_attn.k_proj.weight', row)
                for token in range(4):
                    projected[token * 256 + row] = rounded(math.fsum(a * b for a, b in zip(normalized[token * 1536:(token + 1) * 1536], weights, strict=True)))
            check(label + 'layer.0.self_attn.k_proj', get(sequence, contract, 'layer.0.self_attn.k_proj'), projected)
            expected = [v for token in ids for v in read('embed_tokens_per_layer.weight', token)]
            check(label + 'ple_lookup', get(sequence, contract, 'ple_lookup'), expected)
            head = get(sequence, contract, 'head')
            expected = array('d', (rounded(rounded(math.tanh(rounded(v / 30.))) * 30.) for v in head)); del head
            check(label + 'logits', get(sequence, contract, 'logits'), expected)


def replay(directory, plan, manifest_sha, report):
    raw = source.safe.read(directory / 'observation.json', 4 * 1024**2)
    source.same(source.digest(raw), manifest_sha, 'external graph capture manifest differs')
    capture = source.safe.document(raw)
    source.same(set(capture), {'format', 'passed', 'primary_error', 'endpoint_errors', 'arithmetic_profile', 'tokens',
                              'runtime_admission', 'memory_ledger_mib', 'capture_complete', 'causal_prefix_checked',
                              'cross_precision_compared', 'independent_full_graph_agreement', 'upstream_bf16_runtime_executed',
                              'model_fit_qualified', 'quality_scored', 'app_eligibility_changed', 'full_model_window_qualified',
                              'arrays', 'files', 'predictions', 'loaded_weights', 'source_audit', 'runtime_endpoint'}, 'graph capture schema differs')
    source.same(capture['passed'], True, 'graph worker failed')
    source.same(capture['format'], 'toklet-gemma-bf16-text-capture-v1', 'graph capture format differs')
    source.same(capture['tokens'], TOKENS, 'captured graph tokens differ')
    source.same(capture['memory_ledger_mib'], LEDGER_MIB, 'captured lifetime ledger differs')
    source.same(capture['loaded_weights'], dict(matrices=278, vectors=262, vector_bytes=1143948), 'captured weight inventory differs')
    for key in ('causal_prefix_checked', 'cross_precision_compared', 'independent_full_graph_agreement',
                'upstream_bf16_runtime_executed', 'model_fit_qualified', 'quality_scored',
                'app_eligibility_changed', 'full_model_window_qualified'):
        source.same(capture[key], False, 'unsupported capture qualification')
    source.same(capture['primary_error'], None, 'graph primary failure'); source.same(capture['endpoint_errors'], [], 'graph endpoint failure')
    source.same(capture['capture_complete'], True, 'incomplete graph capture')
    source.same(capture['arithmetic_profile'], PROFILE, 'captured arithmetic profile differs')
    source.same(capture['source_audit'], plan['source_audit'], 'complete graph read audit differs')
    source.same(len(capture['arrays']), 2852, 'full graph capture count differs')
    source.same(set(capture['files']), set(TOKENS), 'captured graph file inventory differs')
    source.same(set(capture['predictions']), {sequence + '/' + contract for sequence in TOKENS for contract in ('math_fp64', 'staged_fp32')}, 'graph prediction inventory differs')
    specs = {}; handles = {}; primary = None; raw_file = None
    try:
        for sequence in TOKENS:
            description = capture['files'][sequence]
            source.same(description['bytes'], plan['file_bytes'], 'captured graph file extent differs')
            handles[sequence] = source.open_source(directory / (sequence + '.bin'), description)
        for wanted, actual in zip(plan['captures'], capture['arrays'], strict=True):
            source.same(set(actual), set(wanted) | {'sha256', 'captured'}, 'graph state schema differs')
            source.same({k: actual[k] for k in wanted}, wanted, 'ordered graph state differs')
            source.same(actual['captured'], True, 'graph state not captured')
            key = tuple(actual[k] for k in ('sequence', 'contract', 'name')); source.safe.require(key not in specs, 'unique graph state required'); specs[key] = actual
            state = values(handles[actual['sequence']], actual)
            if actual['name'] == 'logits':
                source.same(capture['predictions'][actual['sequence'] + '/' + actual['contract']], predictions(state, 262144), 'graph predictions differ')
            del state
        for wanted in plan['captures'][:1426]:
            contract, name = wanted['contract'], wanted['name']; original = values(handles['original'], specs[('original', contract, name)])
            future = values(handles['future'], specs[('future', contract, name)])
            axis = 2 if name.endswith(('rotary', 'context')) else 1; stride = math.prod(wanted['shape'][axis + 1:])
            maximum = 0.; failures = count = 0
            for index, (a, b) in enumerate(zip(future, original, strict=True)):
                if index // stride % 4 < 3:
                    fraction = abs(a - b) / (GATE['atol'] + GATE['rtol'] * abs(b))
                    maximum = max(maximum, fraction); failures += fraction > 1.; count += 1
            item = dict(contract=contract, name=name, values=count, failed_values=failures,
                        maximum_gate_fraction=maximum, passed=failures == 0); report['prefix'].append(item)
            source.safe.require(item['passed'], 'future graph changed causal prefix')
            del original, future
        for sequence in TOKENS:
            for wanted in plan['captures'][:713]:
                name = wanted['name']; ideal = values(handles[sequence], specs[(sequence, 'math_fp64', name)])
                staged = values(handles[sequence], specs[(sequence, 'staged_fp32', name)])
                report['cross_precision'].append(dict(sequence=sequence, name=name, **compare(staged, ideal)))
                del ideal, staged
        official = source.open_source(source.MODEL, source.MODEL_ID); handles['official'] = official
        raw_file = (directory.parent / (directory.name + '-boundary-source.bin')).open('xb')
        boundaries(plan, specs, handles, official, raw_file, report)
        source.same(len(report['boundaries']), 24, 'complete selected graph boundaries required')
        source.same(len(report['boundary_sources']), 1060, 'complete boundary source records required')
        source.same(raw_file.tell(), 3493888, 'complete boundary source bytes required')
        report['capture_validated'] = True; report['causal_prefix_checked'] = True
        report['selected_independent_boundaries_passed'] = True
        report['cross_precision_failed_arrays'] = sum(not item['passed'] for item in report['cross_precision'])
        report['cross_precision_failed_values'] = sum(item['failed_values'] for item in report['cross_precision'])
        report['all_cross_precision_gates_passed'] = report['cross_precision_failed_values'] == 0
    except BaseException as error:
        primary = error; raise
    finally:
        errors = []
        for handle in handles.values():
            try: close_source(handle, primary)
            except BaseException as error: errors.append(str(error))
        if raw_file is not None:
            try: raw_file.close()
            except BaseException as error: errors.append('boundary output close: ' + str(error))
        if errors:
            if primary is not None: primary.add_note('; '.join(errors))
            else: raise ValueError('; '.join(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', type=Path)
    parser.add_argument('--plan', type=Path); parser.add_argument('--plan-sha')
    parser.add_argument('--capture', type=Path); parser.add_argument('--manifest-sha'); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); report = dict(passed=False, primary_error=None, endpoint_errors=[], prefix=[], cross_precision=[], boundaries=[], boundary_sources=[],
                                            independent_full_graph_agreement=False, upstream_bf16_runtime_executed=False,
                                            model_fit_qualified=False, quality_scored=False, app_eligibility_changed=False, full_model_window_qualified=False)
    def stop(number, _):
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(str(number))
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.prepare is not None:
            plan = prepare(); body = encoded(plan)
            with args.prepare.open('xb') as file: source.same(file.write(body), len(body), 'complete graph plan write required')
            report['plan'] = dict(bytes=len(body), sha256=source.digest(body), source_audit=plan['source_audit'])
        else: replay(args.capture, load_plan(args.plan, args.plan_sha), args.manifest_sha, report)
        report['passed'] = True
    except BaseException as error:
        report['primary_error'] = type(error).__name__ + ': ' + str(error); report['cleanup_notes'] = getattr(error, '__notes__', [])
    finally:
        for sig in old: signal.signal(sig, signal.SIG_IGN)
        try:
            with args.output.open('x') as file: file.write(encoded(report).decode())
        finally:
            for sig, handler in old.items(): signal.signal(sig, handler)
    print(json.dumps({key: report.get(key) for key in ('passed', 'primary_error', 'plan', 'capture_validated', 'cross_precision_failed_arrays', 'cross_precision_failed_values')}))
    raise SystemExit(0 if report['passed'] else 1)
