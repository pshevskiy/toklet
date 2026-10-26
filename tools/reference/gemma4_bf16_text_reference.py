"""Stream complete short E2B math using original BF16 row-backed weights.

Both arithmetic contracts reuse existing operators. This prepares references;
it does not implement upstream BF16 arithmetic or grant engine/app eligibility.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import stat
import sys
import time
import bf16_source
import bf16_source_reference as runtime
import check_gemma4_bf16_text as contract
import gemma4_bf16_weights as adapter
import gemma4_text_math as graph

ROOT = Path(__file__).resolve().parents[2]


def extras():
    return [Path(__file__).resolve(), Path(adapter.__file__).resolve(), Path(graph.__file__).resolve(),
            Path(contract.__file__).resolve(), Path(contract.source.__file__).resolve()]


def sources():
    return runtime.sources(extras())


def file_identity(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as file:
        original = contract.source.identity(os.fstat(fd)); digest = hashlib.sha256(); count = 0
        contract.source.safe.require(stat.S_ISREG(original[2]) and original[3] <= maximum, 'bounded regular graph output required')
        while True:
            body = file.read(1024**2)
            if not body: break
            count += len(body); contract.source.safe.require(count <= maximum, 'graph output grew'); digest.update(body)
        contract.source.same(count, original[3], 'complete graph output required')
        contract.source.same(contract.source.identity(os.fstat(fd)), original, 'graph output descriptor changed')
        contract.source.same(contract.source.identity(path.lstat()), original, 'graph output path changed')
    return dict(bytes=count, sha256=digest.hexdigest())


def run(output, plan, admission, endpoint):
    import numpy as np
    record = dict(format='toklet-gemma-bf16-text-capture-v1', passed=False, primary_error=None,
                  endpoint_errors=[], arithmetic_profile=contract.PROFILE, tokens=contract.TOKENS,
                  runtime_admission=admission, memory_ledger_mib=contract.LEDGER_MIB,
                  capture_complete=False, causal_prefix_checked=False, cross_precision_compared=False,
                  independent_full_graph_agreement=False, upstream_bf16_runtime_executed=False,
                  model_fit_qualified=False, quality_scored=False, app_eligibility_changed=False,
                  full_model_window_qualified=False, arrays=[], files={}, predictions={})
    output.mkdir(); resources = {}; owner = None
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        config = runtime.safe.read(ROOT / 'tests/fixtures/gemma-unquantized-header-v1/config.json', 8192)
        runtime.safe.same(hashlib.sha256(config).hexdigest(), '1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330')
        layout = runtime.safe.read(ROOT / 'tests/fixtures/gemma-unquantized-header-v1/layout.json', 2 * 1024**2)
        deadline = time.monotonic() + 900
        owner = bf16_source.Source(contract.source.MODEL, contract.source.MODEL_ID['bytes'], contract.source.MODEL_ID['sha256'], deadline=deadline)
        with owner:
            model = adapter.weights(owner, layout, deadline)
            record['loaded_weights'] = dict(matrices=len(model.matrices), vectors=len(model.vectors),
                                             vector_bytes=sum(value.nbytes for value in model.vectors.values()))
            runtime.safe.same(record['loaded_weights'], dict(matrices=278, vectors=262, vector_bytes=1143948))
            for sequence, ids in contract.TOKENS.items():
                file = (output / (sequence + '.bin')).open('xb'); resources[sequence] = file
                for name, staged in [('math_fp64', False), ('staged_fp32', True)]:
                    def write(state_name, value):
                        wanted = plan['captures'][len(record['arrays'])]; body = value.tobytes(order='C')
                        actual = dict(sequence=sequence, contract=name, name=state_name, shape=list(value.shape),
                                      dtype=value.dtype.str, offset=file.tell(), bytes=len(body),
                                      sha256=hashlib.sha256(body).hexdigest(), captured=False)
                        record['arrays'].append(actual)
                        runtime.safe.require(file.tell() + len(body) <= plan['file_bytes'], 'bounded graph capture')
                        runtime.safe.same(file.write(body), len(body)); file.flush(); actual['captured'] = True
                        runtime.safe.same({key: actual[key] for key in wanted}, wanted)
                        runtime.safe.require(np.isfinite(value).all(), 'finite graph state required')
                        if state_name == 'logits':
                            predictions = []
                            for row in value[0]:
                                winner = int(np.argmax(row)); top = np.partition(row, -2)[-2:]
                                predictions.append(dict(token=winner, margin=float(max(top)) - float(min(top))))
                            record['predictions'][sequence + '/' + name] = predictions
                    graph.evaluate(model, ids, staged, sink=write)
                runtime.safe.same(file.tell(), plan['file_bytes'])
                file.close()
            runtime.safe.same(len(record['arrays']), len(plan['captures']))
            del model
            record['capture_complete'] = True
    except BaseException as error:
        record['primary_error'] = type(error).__name__ + ': ' + str(error)
        record['cleanup_notes'] = getattr(error, '__notes__', [])
    finally:
        for sig in old: signal.signal(sig, signal.SIG_IGN)
        if owner is not None: record['source_audit'] = owner.audit()
        for name, file in resources.items():
            try: file.close()
            except BaseException as error: record['endpoint_errors'].append(name + ' close: ' + str(error))
        try: record['runtime_endpoint'] = endpoint()
        except BaseException as error: record['endpoint_errors'].append('runtime: ' + type(error).__name__ + ': ' + str(error))
        for name in contract.TOKENS:
            if os.path.lexists(output / (name + '.bin')):
                try: record['files'][name] = file_identity(output / (name + '.bin'), plan['file_bytes'])
                except BaseException as error: record['endpoint_errors'].append(name + ' capture: ' + str(error))
        try:
            runtime.safe.same(record.get('source_audit'), plan['source_audit'])
        except BaseException as error: record['endpoint_errors'].append('read plan: ' + str(error))
        record['passed'] = record['capture_complete'] and record['primary_error'] is None and not record['endpoint_errors']
        try:
            with (output / 'observation.json').open('x') as file: json.dump(record, file, indent=2, allow_nan=False); file.write('\n')
        finally:
            for sig, handler in old.items(): signal.signal(sig, handler)
    print(json.dumps({key: record[key] for key in ('passed', 'primary_error', 'endpoint_errors', 'capture_complete', 'files')}))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--runtime-manifest', type=Path, required=True); parser.add_argument('--runtime-sha', required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    plan = contract.load_plan(args.plan, args.plan_sha)
    selected = runtime.runtime.runtime_paths(sources())
    admission = runtime.runtime.admit_runtime(args.runtime_manifest, args.runtime_sha, sources())
    manifest = runtime.safe.document(runtime.safe.read(args.runtime_manifest, 4 * 1024**2))
    def stop(number, _):
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(str(number))
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try: code = run(args.output, plan, admission, lambda: runtime.endpoint(manifest, selected, extras()))
    finally:
        for sig, handler in old.items(): signal.signal(sig, handler)
    raise SystemExit(code)
