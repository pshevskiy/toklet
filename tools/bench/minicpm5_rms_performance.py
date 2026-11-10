"""Strict request comparison for legacy, serial and parallel-store RMS workers."""
import argparse
import json
from pathlib import Path

import minicpm5_performance as common
from minicpm5_profile import timestamp_grain_ns


def validate(report, owned, contract, variant):
    common.need(variant in ['legacy', 'serial', 'parallel-store'], 'Unknown RMS benchmark control')
    expected = contract['expected_session_memory'][variant]
    if variant == 'legacy':
        result = common.validate(report, owned, 'full-logits', contract['teacher_tokens'],
                                 expected, contract['source_manifest_sha256'])
        setup = None
    else:
        implementation = 'serial' if variant == 'serial' else 'serial-reduction-parallel-store'
        common.need(type(report) is dict and report.get('format') == 'toklet-minicpm5-rms-request-bench-v1'
                    and report.get('normalization_implementation') == implementation
                    and 'session_setup_seconds' in report, 'RMS version/implementation/setup differs')
        setup = common.duration_ns(report['session_setup_seconds'])
        # Explicit version adapter: numerical work/state/phase fields retain the
        # complete v1 contract. No raw bytes, receipt or source gate is rewritten.
        numerical = {k: v for k, v in report.items()
                     if k not in ['normalization_implementation', 'session_setup_seconds']}
        numerical['format'] = 'toklet-minicpm5-raw-request-bench-v1'
        result = common.validate(numerical, owned, 'full-logits', contract['teacher_tokens'],
                                 expected, contract['source_manifest_sha256'])
    elapsed = common.interval(owned)
    load = common.duration_ns(report['load_timing']['load_seconds'])
    common.need(load + (0 if setup is None else setup) + result['total_request_ns'] <= elapsed + 1,
                'Disjoint load/setup/requests exceed owned lifetime')
    first, previous = None, 0.
    for sample in report['samples']:
        for step in sample['steps']:
            clock = step['completed_timing']
            begin, end = clock['gpu_start_seconds'], clock['gpu_end_seconds']
            common.need(previous <= begin < end, 'Serial benchmark GPU commands overlap or reorder')
            first = begin if first is None else first
            previous = end
    common.need(load + (0 if setup is None else setup) + common.duration_ns(previous - first)
                <= elapsed + timestamp_grain_ns(first, previous),
                'Disjoint load/setup/GPU series exceed owned lifetime')
    result.update(normalization_implementation=variant, session_setup_nanoseconds=setup,
                  session_setup_available=setup is not None,
                  ordinary_telemetry_overhead_qualified=False, model_quality_adoption_qualified=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['report', 'owned', 'contract', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('variant', choices=['legacy', 'serial', 'parallel-store'])
    args = parser.parse_args()
    def metadata(path):
        with path.open('rb') as stream:
            raw = stream.read(1024 ** 2 + 1)
        common.need(len(raw) <= 1024 ** 2, 'Oversize RMS request metadata')
        return json.loads(raw, object_pairs_hook=common.unique)
    owned = metadata(args.owned)
    result = validate(common.read_report(args.report, owned), owned, metadata(args.contract), args.variant)
    result['environment_evaluated'] = False
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(validated_requests=13, variant=args.variant, environment_evaluated=False,
                          model_quality_adoption_qualified=False)))
