"""Strict saved replay for the separate producer-typed quality collection.

Raw native events are verified before making a copied typed validation view.
The original qualified replay and its numerical/state/memory checks stay intact.
Old failed attempts cannot enter this route, regardless of emitted row counts.
"""
from pathlib import Path

import m6_context_audit as original
import m6_typed_quality as collector

base, ROOT = original.base, original.ROOT
read_bytes, read_json, verify_bindings = original.read_bytes, original.read_json, original.verify_bindings


def replay(record, raw, manifest, arrays, ids):
    if record.get('format') != collector.FORMAT or record.get('collector') != 'm6_typed_quality':
        raise ValueError('Separate typed quality collection required')
    if record.get('plan') != 'quality' or record.get('fp32_policy') != collector.POLICY:
        raise ValueError('Exact typed quality plan/policy required')
    worker = record['worker']
    expected = [worker['ready'], *(event for phase in worker['phases'] for event in phase['events'])]
    if raw != expected:
        raise ValueError('Untouched native raw coverage differs before restoration')
    normalized, precision = collector.values.restore([record])
    view = normalized[0]
    view['format'] = 'toklet-m6-context-collection-v1'
    if record.get('fp32_reconstruction') != precision:
        raise ValueError('Saved producer-type reconstruction differs')
    typed_raw = [view['worker']['ready'], *(event for phase in view['worker']['phases'] for event in phase['events'])]
    result = original.replay(view, typed_raw, manifest, arrays, ids)
    result.update(typed_fp32_policy=collector.POLICY, original_raw_verified_before_restoration=True)
    return result


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
    pins[str(Path(original.__file__).resolve().relative_to(ROOT))] = read_bytes(Path(original.__file__).resolve())[1]
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
        modules = collector.MODULES
        expected_paths = set(required) | {Path(module.__file__) for module in modules}
        expected_paths.add(Path(collector.__file__).resolve())
        if record.get('collector') != 'm6_typed_quality':
            raise ValueError('Unknown typed quality collector identity')
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
        if record.get('collector') == 'm6_typed_quality':
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
    return dict(format='toklet-m6-typed-quality-audit-v1', passed=True,
                scope='Exact replay of listed collections only; milestone coverage/retention is separate',
                records=rows, bindings=pins, audit_source_sha256=source_sha, m6_complete=False)
