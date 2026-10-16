"""Exact eight-record typed replay; changed winners remain unresolved here."""
import argparse
import json
from pathlib import Path
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'Cargo.toml').is_file() and (p/'tools/experiments').is_dir())
sys.path.insert(0, str(ROOT/'tools/experiments'))
import m6_fp32_context_audit as typed


def run():
    source = Path(__file__).resolve()
    source_sha = typed.audit.read_bytes(source)[1]
    paths = [ROOT/'artifacts'/f'm6-context-{model}-quality-{depth}-v{2 if (model == "small" and depth == 16384) or (model == "larger" and depth == 32768) else 1}.json'
             for model in typed.quality.MODELS for depth in typed.quality.DEPTHS]
    result = typed.run(paths, 'quality')
    q = result['derived']
    if q['complete_depth_target_coverage'] is not True or q['frozen_nll_gates_passed'] is not True:
        raise ValueError('Complete unchanged eight-record quality gates required')
    for model in typed.quality.MODELS:
        row = q['models'][model]
        if row['scored_targets'] != 512 or row['pooled_gate_passed'] is not True or row['each_depth_gate_passed'] is not True:
            raise ValueError('Full per-depth and pooled quality coverage required')
    changed = sum(row['argmax_change_count'] for row in q['models'].values())
    if len(result['fp32_reconstruction']['changed_prediction_values']) != changed:
        raise ValueError('Typed diagnostic changed-winner coverage differs')
    if changed and (result['passed'] is not False or q['argmax_investigations_complete'] is not False):
        raise ValueError('Changed predictions must remain unresolved in this audit')
    key = str(source.relative_to(ROOT))
    if key in result['bindings'] and result['bindings'][key] != source_sha:
        raise ValueError('Qualification helper source identity conflict')
    result['bindings'][key] = source_sha
    typed.audit.verify_bindings(result['bindings'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Exclusive typed qualification evidence exists')
    result = run()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(complete_quality_coverage=result['derived']['complete_depth_target_coverage'],
        frozen_nll_gates_passed=result['derived']['frozen_nll_gates_passed'],
        automatic_quality_passed=result['passed'], m6_complete=result['m6_complete'],
        changed_predictions=len(result['fp32_reconstruction']['changed_prediction_values']),
        records=len(result['replay']['records']),
        raw_events=sum(row['raw_events'] for row in result['replay']['records']), bindings=len(result['bindings']))))
