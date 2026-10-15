"""Offline partial-data compatibility check; does not close M6 long quality."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/experiments'))
import m6_long_quality_audit as quality


def run():
    source_paths = [Path(__file__).resolve(), ROOT/'tools/experiments/m6_long_quality_audit.py',
                    ROOT/'tools/experiments/test_m6_long_quality_audit.py']
    sources = {str(path.relative_to(ROOT)): quality.audit.read_bytes(path)[1] for path in source_paths}
    paths = [ROOT/f'artifacts/m6-context-small-quality-{depth}-v1.json' for depth in (4096, 8192)]
    paths.append(ROOT/'artifacts/m6-context-small-profile-32768-v1.json')
    loaded = [(path, *quality.audit.read_json(path)) for path in paths]
    proof = quality.audit.audit(paths)
    for path, _, sha in loaded:
        if proof['bindings'].get(str(path.relative_to(ROOT))) != sha:
            raise ValueError('Parsed compatibility input differs from replayed bytes')
    depths = []
    for expected_depth, (_, record, _) in zip((4096, 8192), loaded[:2]):
        if (record['model'], record['plan'], record['depth']) != ('small', 'quality', expected_depth):
            raise ValueError('Existing small quality depth identity differs')
        rows = record['worker']['phases'][0]['events'][1:-1]
        changes = [value for row in rows if (value := quality.winner_accounting(row)) is not None]
        if len(rows) != 128 or changes:
            raise ValueError('Expected existing128-row/no-winner-change compatibility record')
        depths.append(dict(model=record['model'], depth=record['depth'],
                           checked_winner_rows=len(rows), argmax_changes=len(changes)))
    profile = proof['records'][-1]
    if (profile['model'], profile['plan'], profile['depth']) != ('small', 'profile', 32768):
        raise ValueError('Expected completed small maximum-depth profile')
    if len(profile['derived']['profiles']) != 6:
        raise ValueError('Expected three profiles per mode')
    pins = {**proof['bindings'], **sources}
    quality.audit.verify_bindings(pins)
    return dict(format='toklet-m6-long-quality-audit-compatibility-v1', passed=True,
                checked_winner_rows=sum(row['checked_winner_rows'] for row in depths), depths=depths,
                profile_medians=profile['derived']['profile_medians'],
                scope='Two previously accepted small quality depths and one completed small32Kprofile only; not full512-target/model long quality, application qualification or a KV decision',
                bindings=pins, replay=proof, m6_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Exclusive offline compatibility evidence exists')
    result = run()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
