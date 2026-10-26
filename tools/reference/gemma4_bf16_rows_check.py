"""Cold official BF16 reader/projection checks against pre-existing references.

No model graph executes. Full-file identity and fixed official layout are
required independently of the sampled-row oracle. Outputs precede their gates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import signal
import sys
import time
import affine4_tiles as tiles
import bf16_source as target
import bf16_source_reference as witness
import check_gemma4_bf16_rows_oracle as oracle

ROOT = Path(__file__).resolve().parents[2]
ORACLE_SHA = 'a17dede82620277d77149134750aa7e51abc72cf8995d6481799291e140113cc'
MAX_BINARY = 64 * 1024**2


def extras():
    return [Path(__file__).resolve(), Path(oracle.__file__).resolve()]


def sources():
    return witness.sources(extras())


def check(directory, output, observation):
    import numpy as np
    witness.safe.same(np.__version__, '2.3.2')
    manifest, assets = oracle.load(directory, ORACLE_SHA)
    arrays = assets['arrays']; del assets
    layout = witness.safe.read(ROOT / 'tests/fixtures/gemma-unquantized-header-v1/layout.json', 2 * 1024**2)
    deadline = time.monotonic() + 900
    captures = observation['captures']; output_file = (output / 'arrays.bin').open('xb')

    def capture(label, value, shape, dtype):
        body = value.tobytes(order='C'); offset = output_file.tell()
        witness.safe.require(offset + len(body) <= MAX_BINARY, 'bounded target capture')
        record = dict(label=label, shape=list(value.shape), dtype=value.dtype.str,
                      offset=offset, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                      captured=False, compared=False)
        captures.append(record)
        witness.safe.same(output_file.write(body), len(body)); output_file.flush()
        record['captured'] = True
        witness.safe.same(record['shape'], shape); witness.safe.same(record['dtype'], dtype)
        return body, record

    def row_capture(label, value, shape, expected):
        body, record = capture(label, value, shape, '<f4')
        witness.safe.same(body, expected); record['compared'] = True
        observation['exact_float32_values'] += len(body) // 4

    def reject(label, function, message):
        attempt = dict(label=label, rejected=False, expected_error=message, actual_error=None)
        observation['admission_controls'].append(attempt)
        try: function()
        except ValueError as error:
            attempt['actual_error'] = str(error); witness.safe.same(str(error), message)
            attempt['rejected'] = True
        else: raise ValueError('target admitted invalid official binding: ' + label)

    primary = None
    try:
        with target.Source(oracle.MODEL, oracle.MODEL_ID['bytes'], oracle.MODEL_ID['sha256'], deadline=deadline) as owner:
            for case in manifest['cases']:
                view = target.official_rows(owner, case['tensor'], layout, deadline=deadline)
                shape = case['tensor_shape']; columns = shape[-1]; rows = case['rows']
                witness.safe.same([view.rows, view.columns], [shape[0], columns] if len(shape) == 2 else [1, columns])
                start = case['offset']; expected = arrays[start:start + case['bytes']]
                if len(shape) == 1:
                    full = view.read_rows(0, 1)
                elif case['id'] == 'maximum_tile':
                    full = view.read_rows(rows[0], rows[-1] + 1)
                else:
                    full = np.concatenate([view.read_rows(0, 2), view.read_rows(rows[2], rows[2] + 1),
                                           view.read_rows(rows[3], rows[3] + 1)], axis=0)
                row_capture(case['id'] + ':grouped', full, case['shape'], expected)
                del full
                if len(shape) == 2:
                    if case['id'] == 'maximum_tile':
                        split = np.concatenate([view.read_rows(rows[0], rows[64]), view.read_rows(rows[64], rows[-1] + 1)], axis=0)
                    else:
                        split = np.concatenate([view.read_rows(row, row + 1) for row in rows], axis=0)
                    row_capture(case['id'] + ':split', split, case['shape'], expected)
                    del split
                order = case['gather_order']; ids = [rows[index] for index in order]
                stride = columns * 4
                expected_gather = b''.join(expected[index * stride:(index + 1) * stride] for index in order)
                gathered = view.gather(ids)
                row_capture(case['id'] + ':gather', gathered, [len(order), columns], expected_gather)
                del gathered
                observation['row_cases_completed'] += 1
            view_name = case['tensor']; before = owner.audit()
            reject('129 contiguous rows', lambda: view.read_rows(0, 129), 'bounded integer required')
            witness.safe.same(owner.audit(), before)
            reject('129 gather rows', lambda: view.gather(list(range(129))), 'bounded aggregate gather rows required')
            witness.safe.same(owner.audit(), before)
            first = next(case for case in manifest['cases'] if case['id'] == 'sample:' + view_name)
            offset = first['offset']
            row_capture('after rejected capacity', view.read_rows(0, 1), [1, view.columns], arrays[offset:offset + view.columns * 4])
            for projection in manifest['projections']:
                view = target.official_rows(owner, projection['tensor'], layout, deadline=deadline)
                descriptor = projection['input']; offset = descriptor['offset']
                inputs = np.frombuffer(arrays[offset:offset + descriptor['bytes']], dtype='<f4').reshape(descriptor['shape'])
                for dtype, name in [(np.float32, '<f4'), (np.float64, '<f8')]:
                    value = tiles.project_rows(view, inputs, dtype)
                    _, record = capture(projection['tensor'] + ':' + name, value, [2, view.rows], name)
                    record['comparisons'] = []
                    for kind in ('fsum', 'ordered'):
                        reference = projection['references'][kind]; offset = reference['offset']
                        expected = np.frombuffer(arrays[offset:offset + reference['bytes']], dtype='<f8').reshape(reference['shape'])
                        prefix = 'fp32' if dtype is np.float32 else 'fp64'
                        atol = manifest['projection_policy'][prefix + '_atol']; rtol = manifest['projection_policy'][prefix + '_rtol']
                        error = np.abs(value.astype(np.float64) - expected); allowed = atol + rtol * np.abs(expected)
                        comparison = dict(reference=kind, atol=atol, rtol=rtol, compared_values=int(error.size),
                                          maximum_absolute_error=float(error.max()),
                                          maximum_tolerance_ratio=float((error / allowed).max()),
                                          failures=int(np.count_nonzero(error > allowed)))
                        record['comparisons'].append(comparison)
                        witness.safe.require(np.isfinite(value).all() and comparison['failures'] == 0, 'BF16 bounded projection differs')
                    record['compared'] = True; observation['projection_calls_completed'] += 1
            reject('layout bytes', lambda: target.official_rows(owner, 'model.language_model.norm.weight', layout + b' ', deadline=deadline),
                   'exact official source/layout identity required')
            for name in [*manifest['excluded_shared_kv'], 'model.language_model.unknown.weight']:
                reject(name, lambda name=name: target.official_rows(owner, name, layout, deadline=deadline),
                       'selected official text tensor required')
        observation['source_audit'] = owner.audit()
        with target.Source(witness.FIXTURE / 'source.bin', 40, 'cd1f0d6dd1061d41d1d6f53109fb850962398bacc97332e00ae8cad91c1f6fc3', deadline=deadline) as other:
            reject('different authenticated source', lambda: target.official_rows(other, 'model.language_model.norm.weight', layout, deadline=deadline),
                   'exact official source/layout identity required')
        witness.safe.same(observation['row_cases_completed'], 541)
        witness.safe.same(observation['projection_calls_completed'], 4)
        witness.safe.same(len(observation['admission_controls']), 65)
        witness.safe.same(len(captures), 1366)
        witness.safe.same(output_file.tell(), 56288676)
        observation['passed'] = True
    except BaseException as error:
        primary = error
        raise
    finally:
        if 'owner' in locals(): observation['source_audit'] = owner.audit()
        try: output_file.close()
        except BaseException as error:
            if primary is None: raise
            primary.add_note('target capture cleanup: ' + str(error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--runtime-sha', required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selected = witness.runtime.runtime_paths(sources())
    admission = witness.runtime.admit_runtime(args.runtime_manifest, args.runtime_sha, sources())
    runtime_manifest = witness.safe.document(witness.safe.read(args.runtime_manifest, 4 * 1024**2))
    result = dict(passed=False, primary_error=None, endpoint_errors=[], admission=admission,
                  observation=dict(passed=False, captures=[], exact_float32_values=0,
                                   row_cases_completed=0, projection_calls_completed=0, admission_controls=[],
                                   official_graph_executed=False, model_fit_qualified=False, quality_scored=False,
                                   performance_qualified=False, app_eligibility_changed=False, full_model_window_qualified=False))
    def stop(number, _):
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(str(number))
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        args.output.mkdir()
        check(args.oracle, args.output, result['observation']); result['passed'] = True
    except BaseException as error:
        result['primary_error'] = type(error).__name__ + ': ' + str(error)
        result['cleanup_notes'] = getattr(error, '__notes__', [])
    finally:
        for sig in old: signal.signal(sig, signal.SIG_IGN)
        try: result['runtime_endpoint'] = witness.endpoint(runtime_manifest, selected, extras())
        except BaseException as error:
            result['endpoint_errors'].append(type(error).__name__ + ': ' + str(error)); result['passed'] = False
        try:
            with (args.output.parent / (args.output.name + '-result.json')).open('x') as file:
                json.dump(result, file, indent=2, allow_nan=False); file.write('\n')
        finally:
            for sig, handler in old.items(): signal.signal(sig, handler)
    print(json.dumps({key: result.get(key) for key in ('passed', 'primary_error', 'endpoint_errors', 'observation')}))
    raise SystemExit(0 if result['passed'] and not result['endpoint_errors'] else 1)
