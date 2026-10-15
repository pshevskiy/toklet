"""Replay saved M6 collections without launching numerical work.

This audit verifies recorded requests against the qualified collection plan,
raw event coverage, derived metrics, native charges and consumed file identities.
It does not turn incomplete milestone coverage into acceptance.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import m6_context as base

ROOT = base.ROOT
BOUND = 64 * 1024**2


def read_bytes(path):
    with path.open('rb') as stream:
        data = stream.read(BOUND + 1)
    if len(data) > BOUND:
        raise ValueError('Saved evidence exceeds audit bound')
    return data, hashlib.sha256(data).hexdigest()


def read_json(path, lines=False):
    data, sha = read_bytes(path)
    def nonfinite(value):
        raise ValueError(f'Nonfinite saved JSON: {value}')
    if lines:
        return [json.loads(line, parse_constant=nonfinite) for line in data.splitlines()], sha
    return json.loads(data, parse_constant=nonfinite), sha


def verify_bindings(pins):
    for relative, sha in pins.items():
        if base.shared.metal_engine.digest(ROOT/relative) != sha:
            raise ValueError(f'Consumed bytes changed: {relative}')


def replay(record, raw, manifest, arrays, ids):
    """Reissue the frozen plan to a saved-event reader, never a native worker."""
    if record.get('format') != 'toklet-m6-context-collection-v1':
        raise ValueError('Unknown collection format')
    for key in ('passed', 'collection_completed', 'identities_unchanged'):
        if record.get(key) is not True:
            raise ValueError(f'Collection not complete: {key}')
    if record.get('endpoint_errors') != []:
        raise ValueError('Collection endpoint failures')
    model, plan, depth = record['model'], record['plan'], record['depth']
    if model not in ('small', 'larger') or plan not in ('probe', 'quality', 'profile', 'matrix', 'primary', 'confirmation', 'long', 'long-confirmation'):
        raise ValueError('Unknown frozen collection plan')
    if type(depth) is not int or depth not in (4096, 8192, 16384, 32768):
        raise ValueError('Unknown frozen context depth')
    capacity = 2048 if plan in ('probe', 'matrix', 'primary', 'confirmation') else depth
    if record.get('capacity') != capacity or type(record.get('capacity')) is not int or record.get('profiling') is not (plan == 'profile'):
        raise ValueError('Collection capacity/profile selection differs')
    worker = record['worker']
    if worker.get('passed') is not True or type(worker.get('exit_code')) is not int or worker['exit_code'] != 0 or worker.get('cleanup_errors') != []:
        raise ValueError('Owned worker did not complete cleanly')
    base.ready(worker['ready'], model, capacity, 128, plan == 'profile')
    phases = worker['phases']
    expected_raw = [worker['ready'], *(event for phase in phases for event in phase['events'])]
    if raw != expected_raw:
        raise ValueError('Raw event coverage or values differ')
    usage = worker['usage']
    if len(usage) != len(phases) + 1:
        raise ValueError('Native charge phase coverage differs')
    for i, row in enumerate(usage):
        if row['phase'] != ('ready' if i == 0 else f'phase_{i}'):
            raise ValueError('Native charge ordering differs')
        values = [row[owner].get('lifetime_max_phys_footprint') for owner in ('worker', 'driver')]
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError('Invalid native lifetime charge')
        joint = row.get('joint_native_lifetime_peak_bytes')
        if type(joint) is not int or joint != sum(values) or joint > base.LIMIT:
            raise ValueError('Native joint charge differs or exceeds budget')

    class SavedWorker:
        def __init__(self):
            self.model = model
            self.admission = worker['ready']
            self.index = 0

        def call(self, request, final):
            if self.index >= len(phases):
                raise ValueError('Saved request coverage is incomplete')
            phase = phases[self.index]
            self.index += 1
            if phase['request'] != request:
                raise ValueError('Saved request differs from frozen plan')
            events = phase['events']
            if not events or events[-1].get('event') not in (final, 'error'):
                raise ValueError('Saved request final event differs')
            if request['action'] not in ('compare', 'profile') and len(events) != 1:
                raise ValueError('Ordinary phase must have exactly one event')
            if any(event.get('event') in (final, 'error') for event in events[:-1]):
                raise ValueError('Early terminal event in saved phase')
            return events

    saved = SavedWorker()
    derived = {}
    with contextlib.redirect_stdout(io.StringIO()):
        base.collect(saved, plan, ids, manifest, arrays, derived)
    if saved.index != len(phases):
        raise ValueError('Unconsumed saved requests')
    if any(record.get(key) != value for key, value in derived.items()):
        raise ValueError('Saved derived metrics differ from exact replay')
    if plan == 'quality' and not derived['quality']['gate_passed']:
        raise ValueError('Frozen paired loss gate failed')
    return dict(model=model, plan=plan, depth=depth, raw_events=len(raw), phases=len(phases),
                admission_bytes=worker['ready']['unique_memory_admitted_bytes'],
                max_joint_native_bytes=max(row['joint_native_lifetime_peak_bytes'] for row in usage),
                derived=derived)


def audit(paths):
    source_path = Path(__file__).resolve()
    _, source_sha = read_bytes(source_path)
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Duplicate collection evidence')
    manifest, arrays = base.shared.contract()
    _, long_arrays, _, _, _ = base.inputs('small')  # Only reads bounded contract/configuration metadata.
    _, compiled, check = base.compiled_admission()
    pins = {str(p.relative_to(ROOT)): sha for p, sha in compiled.items()}
    pins[str(source_path.relative_to(ROOT))] = source_sha
    pins.update({str(p.relative_to(ROOT)): sha for p, sha in base.controlled_pins(manifest).items()})
    rows = []
    for path in paths:
        path = path.resolve()
        record, record_sha = read_json(path)
        raw_path = path.with_suffix('.raw.jsonl')
        raw, raw_sha = read_json(raw_path, lines=True)
        row = replay(record, raw, manifest, arrays, long_arrays[record['depth']])
        _, _, _, _, fixed = base.inputs(record['model'])
        required = {**compiled, **fixed, **base.controlled_pins(manifest)}
        modules = (base, base.shared, base.loss, base.loss.shared, base.loss.large, base.loss.large.quality_rows,
                   base.shared.frozen, base.shared.common, base.shared.prepare, base.shared.metal_engine,
                   base.shared.native_usage, base.shared.metal_prefill, base.shared.metal_continue)
        expected_paths = set(required) | {Path(module.__file__) for module in modules}
        collector = record.get('collector')
        if collector == 'm6_context_long_budget':
            expected_paths.add(ROOT/'tools/experiments/m6_context_long_budget.py')
        elif collector is not None:
            raise ValueError('Unknown collector identity')
        if set(record['identities']) != {str(p.relative_to(ROOT)) for p in expected_paths}:
            raise ValueError('Consumed identity inventory differs')
        for p, sha in required.items():
            if record['identities'][str(p.relative_to(ROOT))] != sha:
                raise ValueError('Frozen consumed identity differs')
        expected_command = [str(ROOT/'artifacts/m6-context-worker-native-v1/metal_context'),
                            str(ROOT/'models'/base.loss.SPECS[record['model']]['name']),
                            str(ROOT/'artifacts/m5-larger-q4-v2') if record['model'] == 'larger' else '-',
                            str(record['capacity']), '128'] + (['profile'] if record['profiling'] else [])
        if record['worker']['command'] != expected_command:
            raise ValueError('Recorded numerical command differs')
        if record.get('collector') == 'm6_context_long_budget':
            timeout = record.get('event_read_timeout_seconds')
            if type(timeout) is not int or timeout not in (7200, 21600, 43200) or type(record.get('lifetime_timeout_seconds')) is not int or record['lifetime_timeout_seconds'] != 86400:
                raise ValueError('Extended wait metadata differs')
            for key in ('event_read_timeout_seconds', 'lifetime_timeout_seconds'):
                if record['worker'].get(key) != record[key]:
                    raise ValueError('Worker wait metadata differs')
        for key, sha in record['identities'].items():
            if key in pins and pins[key] != sha:
                raise ValueError('Conflicting consumed file identity')
            pins[key] = sha
        stderr_path = path.with_suffix('.stderr')
        _, stderr_sha = read_bytes(stderr_path)
        for evidence, sha in ((path, record_sha), (raw_path, raw_sha), (stderr_path, stderr_sha)):
            pins[str(evidence.relative_to(ROOT))] = sha
        row['path'] = str(path.relative_to(ROOT))
        rows.append(row)
    verify_bindings(pins)
    check()
    return dict(format='toklet-m6-context-audit-v1', passed=True,
                scope='Exact replay of listed collections only; milestone coverage/retention is separate',
                records=rows, bindings=pins, audit_source_sha256=source_sha, m6_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('records', type=Path, nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Exclusive saved audit evidence already exists')
    result = audit(args.records)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
