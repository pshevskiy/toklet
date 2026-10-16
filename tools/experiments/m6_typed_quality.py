"""Explicit producer-typed M6 quality collection with the unchanged native worker.

The old collector treated f32 shortest JSON round trips as exact f64 values.
This separate format restores only winner-pair elements in a copied validation
view. Native requests, raw events, f64 metrics, gates and lifecycle are unchanged.
"""
import argparse
import json
from pathlib import Path

import m6_context_long_budget as wait
import m6_fp32_values as values

base = wait.base
ROOT = base.ROOT
FORMAT = 'toklet-m6-typed-quality-collection-v1'
POLICY = 'Only four declared f32 winner values per row restored before unchanged comparison; raw phases and f64 metrics unchanged'
MODULES = (base, base.shared, base.loss, base.loss.shared, base.loss.large, base.loss.large.quality_rows,
    base.shared.frozen, base.shared.common, base.shared.prepare, base.shared.metal_engine,
    base.shared.native_usage, base.shared.metal_prefill, base.shared.metal_continue,
    wait, values, values.quality, values.quality.audit)


def comparison(events, ids, memories, model, depth):
    records, reconstruction = values.restore([dict(model=model, depth=depth, plan='quality',
        worker=dict(phases=[dict(events=events)]))])
    restored = records[0]['worker']['phases'][0]['events']
    quality = base.comparison(restored, ids, memories)
    if not quality['gate_passed']:
        raise ValueError('Unchanged paired loss gate failed')
    return quality, reconstruction


def run(output, model, depth, request_timeout=43200):
    wait.budget(request_timeout)
    if model not in ('small', 'larger') or type(depth) is not int or depth not in (4096,8192,16384,32768):
        raise ValueError('Frozen typed quality model/depth required')
    paths = [output, output.with_suffix('.raw.jsonl'), output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):
        raise ValueError('Exclusive typed quality evidence exists')
    binary, pins, compiled_check = base.compiled_admission()
    _, arrays, source, packed, fixed = base.inputs(model)
    manifest, _ = base.shared.contract()
    pins.update(fixed); pins.update(base.controlled_pins(manifest))
    files = set(pins) | {Path(__file__).resolve(), *(Path(m.__file__).resolve() for m in MODULES)}
    identities = base.capture_identities(files, pins)
    ids = arrays[depth]
    result = dict(format=FORMAT, passed=False, collection_completed=False, model=model,
        plan='quality', depth=depth, capacity=depth, profiling=False, collector='m6_typed_quality',
        fp32_policy=POLICY, event_read_timeout_seconds=request_timeout, lifetime_timeout_seconds=wait.LIFETIME,
        identities=identities, revision=base.shared.metal_continue.observation(['git','rev-parse','HEAD']),
        compiler=base.shared.metal_continue.observation(['rustc','-Vv']),
        platform=base.shared.metal_continue.observation(['/usr/bin/sw_vers']),
        power_before=base.shared.metal_continue.endpoints(), energy_joules=None, continuous_memory_pressure=None,
        scope='Same admitted worker and frozen quality requests/gates; typed FP32 validation only; no timing or overall M6 acceptance')
    def check():
        compiled_check()
        if any(base.shared.metal_engine.digest(ROOT/p) != sha for p, sha in identities.items()):
            raise ValueError('Typed quality consumed inputs changed before launch/at endpoint')
    with output.open('x') as report, paths[1].open('x') as raw, paths[2].open('xb') as stderr:
        worker = wait.Worker(binary, source, packed, depth, 128, False, model, raw, stderr, check,
                             request_timeout=request_timeout)
        result['worker'] = worker.record
        try:
            with worker:
                events = worker.call(dict(action='compare',tokens=ids[:-128],continuation=ids[-128:]), 'comparison_done')
                result['quality'], result['fp32_reconstruction'] = comparison(events, ids,
                    [worker.admission[f'{mode}_memory'] for mode in base.MODES], model, depth)
            result.update(passed=worker.record['passed'], collection_completed=worker.record['passed'])
        except BaseException as error:
            result.update(passed=False, error=f'{type(error).__name__}: {error}')
        finally:
            result['endpoint_errors'] = []
            try:
                check(); result['identities_unchanged'] = True
            except BaseException as error:
                result['identities_unchanged'] = False; result['endpoint_errors'].append(str(error))
            try:
                result['power_after'] = base.shared.metal_continue.endpoints()
            except BaseException as error:
                result['endpoint_errors'].append(str(error))
            if result['endpoint_errors']:
                result.update(passed=False, collection_completed=False)
            json.dump(result, report, indent=2, allow_nan=False); report.write('\n')
    if not result['passed']:
        raise ValueError('Typed quality collection failed; failed evidence retained')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('small','larger'), required=True)
    parser.add_argument('--depth', type=int, choices=(4096,8192,16384,32768), required=True)
    parser.add_argument('--event-read-timeout', type=int, choices=wait.BUDGETS, default=43200)
    args = parser.parse_args()
    run(args.output.resolve(), args.model, args.depth, args.event_read_timeout)
