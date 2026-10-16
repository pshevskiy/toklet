"""Restore typed FP32 winner values before the unchanged saved-data checks.

The Rust producer serializes winner-pair elements as f32 shortest round trips,
but computes margins/error/NLL as f64. A Python JSON float is f64. Restoring the
declared f32 type avoids inventing a margin discrepancy from decimal spelling.
Raw records, old qualified auditors and every numerical gate stay unchanged.
Changed predictions still require a separate reviewed investigation.
"""
import argparse
import json
from pathlib import Path

import m6_context_audit as audit
import m6_fp32_values as values
import m6_typed_quality_audit as typed_audit

import m6_context_scope_audit as scope
import m6_long_quality_audit as quality

fp32, restore = values.fp32, values.restore


def replay(paths, loaded):
    """Route original and separately typed native formats without relabeling."""
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Duplicate mixed native evidence')
    ordinary, typed = [], []
    for path, record, _ in loaded:
        if record.get('format') == 'toklet-m6-context-collection-v1':
            ordinary.append(path)
        elif record.get('format') == typed_audit.collector.FORMAT:
            typed.append(path)
        else:
            raise ValueError('Unknown mixed native collection format')
    pins, rows = {}, []
    for subset, runner in ((ordinary, audit.audit), (typed, typed_audit.audit)):
        if not subset:
            continue
        proof = runner(subset)
        if proof.get('passed') is not True:
            raise ValueError('Mixed native replay did not pass')
        rows.extend(proof['records'])
        for path, sha in proof['bindings'].items():
            if path in pins and pins[path] != sha:
                raise ValueError('Conflicting mixed native replay identity')
            pins[path] = sha
    return dict(format='toklet-m6-mixed-native-replay-v1', passed=True, records=rows, bindings=pins,
                m6_complete=False, scope='Each original/typed collection strictly replays through its own format; failed originals never relabeled')


def run(paths, kind):
    expected = 8 if kind == 'quality' else 32 if kind == 'context' else None
    if expected is None or len(paths) != expected:
        raise ValueError('Require exact quality8/context32 saved collections')
    sources = [Path(__file__).resolve(), Path(quality.__file__).resolve(), Path(scope.__file__).resolve(), Path(values.__file__).resolve(),
               Path(typed_audit.__file__).resolve(), Path(typed_audit.collector.__file__).resolve()]
    pins = {str(p.relative_to(audit.ROOT)): audit.read_bytes(p)[1] for p in sources}
    loaded = [(p.resolve(), *audit.read_json(p)) for p in paths]
    proof = replay(paths, loaded)
    for path, _, sha in loaded:
        if proof['bindings'].get(str(path.relative_to(audit.ROOT))) != sha:
            raise ValueError('Parsed FP32-audit input differs from replayed bytes')
    normalized, reconstruction = restore([record for _, record, _ in loaded])
    derived = quality.summarize(normalized) if kind == 'quality' else scope.summarize(normalized)
    for path, sha in proof['bindings'].items():
        if path in pins and pins[path] != sha:
            raise ValueError('Conflicting FP32-audit source identity')
        pins[path] = sha
    audit.verify_bindings(pins)
    return dict(format='toklet-m6-fp32-context-audit-v1', kind=kind, passed=derived['passed'],
                m6_complete=False, frozen_numerical_gates_unchanged=True,
                fp32_reconstruction=reconstruction, derived=derived, replay=proof, bindings=pins,
                scope='Exact native replay followed by producer-typed FP32 restoration; changed predictions remain open for separate review')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('quality', 'context'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('records', nargs='+', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Exclusive typed-FP32 audit evidence exists')
    result = run(args.records, args.kind)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if not result['passed']:
        raise SystemExit('Quality failed or changed predictions require investigation; see saved audit')
