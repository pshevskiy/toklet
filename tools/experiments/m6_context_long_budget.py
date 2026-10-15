"""M6 long collection with an explicit larger bounded IPC wait allowance.

Reuse the accepted numerical worker and frozen collection validators unchanged.
Keep the original collector/source identities reproducible for completed runs.
The allowance is per blocking JSON event read, bounded by the 24-hour lifetime.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import m6_context as base

ROOT = base.ROOT
LIFETIME = 86400
BUDGETS = (7200, 21600, 43200)
PLANS = ('probe', 'quality', 'profile', 'long', 'long-confirmation')


def budget(value):
    if type(value) is not int or value not in BUDGETS:
        raise ValueError('IPC wait must be an explicit bounded 2/6/12-hour choice')
    return value


class Worker(base.Worker):
    def __init__(self, *args, request_timeout, **kwargs):
        self.request_timeout = budget(request_timeout)
        super().__init__(*args, **kwargs)
        self.record.update(event_read_timeout_seconds=self.request_timeout, lifetime_timeout_seconds=LIFETIME)

    def __enter__(self):
        try:
            self.check()
            self.child = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            self.record['worker_pid'] = self.child.pid
            self.transport = base.shared.metal_engine.Transport(self.child, self.raw, self.stderr, stdout_limit=16*1024**2,
                                                               event_limit=32768, request_timeout=self.request_timeout,
                                                               lifetime_timeout=LIFETIME)
            self.admission = self.transport.read(); self.record['ready'] = self.admission
            base.ready(self.admission, self.model, self.capacity, self.chunk, self.profiling)
            self.usage('ready')
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise


def run(output, model, plan, depth, request_timeout=21600):
    budget(request_timeout)
    if model not in ('small', 'larger') or plan not in PLANS or type(depth) is not int or depth not in (4096, 8192, 16384, 32768):
        raise ValueError('Frozen long collection model/plan/depth required')
    paths = [output, output.with_suffix('.raw.jsonl'), output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):
        raise ValueError('Exclusive extended-budget context evidence exists')
    binary, pins, check = base.compiled_admission()
    _, long_arrays, source, packed, fixed = base.inputs(model)
    manifest, arrays = base.shared.contract()
    pins.update(fixed); pins.update(base.controlled_pins(manifest))
    modules = (base, base.shared, base.loss, base.loss.shared, base.loss.large, base.loss.large.quality_rows,
               base.shared.frozen, base.shared.common, base.shared.prepare, base.shared.metal_engine,
               base.shared.native_usage, base.shared.metal_prefill, base.shared.metal_continue)
    files = set(pins) | {Path(__file__), *(Path(m.__file__) for m in modules)}
    identities = base.capture_identities(files, pins)
    capacity = 2048 if plan == 'probe' else depth
    result = dict(format='toklet-m6-context-collection-v1', passed=False, collection_completed=False,
                  model=model, plan=plan, depth=depth, capacity=capacity, profiling=plan == 'profile',
                  collector='m6_context_long_budget', event_read_timeout_seconds=request_timeout, lifetime_timeout_seconds=LIFETIME,
                  scope='Same frozen numerical inputs/gates/worker; explicit bounded IPC allowance only. Long timing descriptive; no natural long-document ability.',
                  identities=identities, revision=base.shared.metal_continue.observation(['git', 'rev-parse', 'HEAD']),
                  compiler=base.shared.metal_continue.observation(['rustc', '-Vv']),
                  platform=base.shared.metal_continue.observation(['/usr/bin/sw_vers']),
                  power_before=base.shared.metal_continue.endpoints(), energy_joules=None, continuous_memory_pressure=None)
    def before_launch():
        check()
        if any(base.shared.metal_engine.digest(ROOT/p) != s for p, s in identities.items()):
            raise ValueError('Consumed extended-budget collection inputs changed before launch')
    with paths[0].open('x') as report, paths[1].open('x') as raw, paths[2].open('xb') as stderr:
        worker = Worker(binary, source, packed, capacity, 128, plan == 'profile', model, raw, stderr,
                        before_launch, request_timeout=request_timeout)
        result['worker'] = worker.record
        try:
            with worker:
                base.collect(worker, plan, long_arrays[depth], manifest, arrays, result)
            result.update(passed=worker.record['passed'], collection_completed=worker.record['passed'])
        except BaseException as error:
            result.update(passed=False, error=f'{type(error).__name__}: {error}')
        finally:
            result['endpoint_errors'] = []
            try:
                result['identities_unchanged'] = all(base.shared.metal_engine.digest(ROOT/p) == s for p, s in identities.items())
                check()
            except BaseException as error:
                result['identities_unchanged'] = False; result['endpoint_errors'].append(str(error))
            try:
                result['power_after'] = base.shared.metal_continue.endpoints()
            except BaseException as error:
                result['endpoint_errors'].append(str(error))
            if not result['identities_unchanged'] or result['endpoint_errors']:
                result.update(passed=False, collection_completed=False)
            json.dump(result, report, indent=2, allow_nan=False); report.write('\n')
    if not result['passed']:
        raise ValueError('Extended-budget M6 collection failed; see retained evidence')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=('small', 'larger'), required=True)
    p.add_argument('--plan', choices=PLANS, required=True)
    p.add_argument('--depth', type=int, choices=(4096, 8192, 16384, 32768), default=4096)
    p.add_argument('--event-read-timeout', dest='request_timeout', type=int, choices=BUDGETS, default=21600,
                   help='Maximum seconds per blocking JSON event read, subject to the 24-hour worker lifetime')
    a = p.parse_args(); run(a.output.resolve(), a.model, a.plan, a.depth, a.request_timeout)
