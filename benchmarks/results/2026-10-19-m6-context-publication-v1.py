"""Publish the independently replayed M6 context packet without changing raw data.

This creates a publication draft. Astra's publication acceptance is recorded
separately; KV retention and overall M6 acceptance stay outside this packet.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'Cargo.toml').is_file())

PROOF = 'artifacts/m6-context-reviewed-closure-v1.json'
ASTRA = 'artifacts/m6-context-reviewed-closure-astra-v1.json'
PROOF_SHA = '9873fe34fb887a6dac2936f5ea505ee57bc2bbe7ae0da742b6e1cfc52ddb9320'
CHECKS = 'artifacts/m6-context-checks-v1.json'
CHECKS_SHA = 'f8f22b6184c1a5c4d0a8ff578571284312ae3d7869c6b621992afd471cc330cd'
OUTPUT = 'benchmarks/results/2026-10-19-m6-context-qualification-v1.json'
RESULTS = ROOT / 'benchmarks/results'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def admit_bytes(relative, admitted):
    if relative not in admitted:
        with (ROOT / relative).open('rb') as stream:
            data = stream.read(16 * 1024**2 + 1)
        if len(data) > 16 * 1024**2:
            raise ValueError('Publication metadata exceeds16MiB: ' + relative)
        admitted[relative] = (data, hashlib.sha256(data).hexdigest())
    return admitted[relative]


def pinned_json(relative, admitted, expected=None):
    data, actual = admit_bytes(relative, admitted)
    if expected is not None and actual != expected:
        raise ValueError('Pinned input differs: ' + relative)
    return json.loads(data)


def merge(*groups):
    result = {}
    for group in groups:
        for path, expected in group.items():
            if path in result and result[path] != expected:
                raise ValueError('Conflicting publication binding: ' + path)
            result[path] = expected
    return result


def main():
    if (ROOT / OUTPUT).exists():
        raise ValueError('Exclusive publication draft already exists')
    admitted = {}
    source_paths = ['tools/experiments/m6_context_reviewed_closure.py',
                    'tools/experiments/test_m6_context_reviewed_closure.py',
                    str(Path(__file__).resolve().relative_to(ROOT)),
                    'artifacts/m6-remaining-native-queue-v2.py']
    sources = {path: admit_bytes(path, admitted)[1] for path in source_paths}
    sys.path.insert(0, str(ROOT / 'tools/experiments'))
    import m6_context_reviewed_closure as context
    proof = pinned_json(PROOF, admitted, PROOF_SHA)
    pinned_json(ASTRA, admitted, PROOF_SHA)
    checks = pinned_json(CHECKS, admitted, CHECKS_SHA)
    if any(proof.get(k) is not True for k in ('passed', 'complete_frozen_context_coverage',
                                             'frozen_nll_gates_passed', 'all_argmax_changes_investigated')):
        raise ValueError('Complete context gates required')
    if proof['m6_complete'] is not False or proof['kv_retention_qualified'] is not False:
        raise ValueError('Context-only scope required')
    audit = proof['automatic_audit']
    if audit['passed'] is not False or audit['derived']['long_quality']['argmax_investigations_complete'] is not False:
        raise ValueError('Original automatic unresolved flags must be preserved')
    records = audit['replay']['records']
    if len(records) != 32 or sum(r['raw_events'] for r in records) != 22752 or len(proof['bindings']) != 429:
        raise ValueError('Reviewed replay inventory differs')
    if checks.get('passed') is not True or checks['command_group_exit_code'] != 0:
        raise ValueError('Completed successful checks required')
    check_pins = merge(checks['logs'], checks['protected_cpu_binaries'])
    for path, expected in check_pins.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Check log or protected binary differs: ' + path)
    context.typed.audit.verify_bindings(proof['bindings'])

    dependencies = [context.RECEIPT,
        'benchmarks/results/2026-10-16-m6-full-long-quality-qualification-v1.json',
        'benchmarks/results/2026-10-16-m6-supervision-qualification-v1.json']
    dependency_values = {}
    for path in dependencies:
        value = pinned_json(path, admitted, context.RECEIPT_SHA if path == context.RECEIPT else None)
        dependency_values[path] = value
        if value.get('passed') is not True or value.get('review', {}).get('accepted') is not True:
            raise ValueError('Accepted prerequisite required: ' + path)
    supervision = dependency_values[dependencies[-1]]
    queue_source = 'artifacts/m6-remaining-native-queue-v2.py'
    if sources[queue_source] != supervision['source_pins'][queue_source]:
        raise ValueError('Reviewed serial supervisor source differs')
    status = 'artifacts/m6-remaining-native-queue-v2.status.jsonl'
    events = [json.loads(line) for line in admit_bytes(status, admitted)[0].splitlines()]
    if events[-1]['event'] != 'queue_complete' or [e['index'] for e in events if e['event'] == 'role_complete'] != list(range(10)):
        raise ValueError('Actual ten-role queue completion differs')

    existing = {}
    for path in sorted(RESULTS.glob('*m6*qualification-v*.json')):
        value = pinned_json(str(path.relative_to(ROOT)), admitted)
        for published, original in value.get('published_copies', {}).items():
            existing.setdefault(original, []).append(published)
    copies, evidence, new_copies = {}, {}, []

    def publish(original, expected, reuse=True):
        source = ROOT / original
        if digest(source) != expected:
            raise ValueError('Publication input differs: ' + original)
        candidates = existing.get(original, []) if reuse else []
        published = candidates[0] if candidates else 'benchmarks/results/2026-10-19-' + source.name
        destination = ROOT / published
        if not destination.resolve().is_relative_to(RESULTS.resolve()):
            raise ValueError('Publication destination escapes results')
        if destination.exists():
            if digest(destination) != expected:
                raise ValueError('Immutable published copy differs: ' + published)
        else:
            with source.open('rb') as incoming, destination.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            new_copies.append(published)
        if digest(destination) != expected:
            raise ValueError('Published bytes differ: ' + published)
        if published in copies and copies[published] != original:
            raise ValueError('Conflicting published source: ' + published)
        copies[published] = original
        evidence.update(merge(evidence, {original: expected}))

    for record in records:
        path = Path(record['path'])
        for original in (path, path.with_suffix('.raw.jsonl'), path.with_suffix('.stderr')):
            relative = str(original)
            publish(relative, proof['bindings'][relative])
    extras = [PROOF, ASTRA, CHECKS, *checks['logs'], status,
        'artifacts/m6-remaining-native-queue-v2.owner.json',
        'artifacts/m6-remaining-native-queue-v2.log',
        'artifacts/m6-remaining-native-queue-v2.supervisor.log',
        str(Path(__file__).resolve().relative_to(ROOT))]
    expected_extras = merge({PROOF: PROOF_SHA, ASTRA: PROOF_SHA, CHECKS: CHECKS_SHA},
                            checks['logs'], sources)
    for path in extras:
        actual = admit_bytes(path, admitted)[1]
        expected_extras = merge(expected_extras, {path: actual})
        publish(path, expected_extras[path], reuse=False)
    admitted_pins = {path: value[1] for path, value in admitted.items()}
    evidence = merge(evidence, admitted_pins, check_pins)
    for path in sources:
        evidence.pop(path, None)
    models = {}
    for model, values in audit['derived']['models'].items():
        own = [r for r in records if r['model'] == model]
        models[model] = dict(depths=values['depths'],
            numerical_quality=audit['derived']['long_quality']['models'][model],
            maximum_admission_bytes=max(r['admission_bytes'] for r in own),
            maximum_native_charge_bytes=max(r['max_joint_native_bytes'] for r in own))
    context.typed.audit.verify_bindings(merge(proof['bindings'], sources, evidence))
    result = dict(format='toklet-m6-context-qualification-v1', passed=True,
        review=dict(reviewer='Astra', accepted=False, code_and_data_accepted=True,
                    publication='pending'),
        source_parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_pins=sources, evidence_pins=evidence, published_copies=copies,
        new_published_copies=new_copies, records=32, raw_events=22752, replay_bindings=429,
        checks=checks, models=models, exact_manual_cases=proof['reviewed_cases'],
        automatic_unresolved_flags_preserved=True, complete_frozen_context_coverage=True,
        full_long_numerical_quality_qualified=True, context_boundary_coverage_complete=True,
        kv_retention_qualified=False, m6_complete=False,
        report_path='docs/experiments/2026-10-19-m6-context.md',
        scope='All32 exact context roles plus only the two reviewed original winner cases; long timings descriptive, final FP16 retention and M6 synthesis separate')
    with (ROOT / OUTPUT).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=OUTPUT, sources=len(sources), evidence=len(evidence),
                         copies=len(copies), new_copies=len(new_copies), publication_review_pending=True)))


if __name__ == '__main__':
    main()
