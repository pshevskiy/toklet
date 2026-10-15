"""Online attention responses for the original100 held-out tasks; grading separate."""
import argparse
import copy
import json
from pathlib import Path

import m6_context as context
import metal_tasks as shared
import metal_cli_v2 as cli_v2

ROOT = shared.ROOT
QUAL_REL = 'benchmarks/results/2026-10-15-m6-long-admission-component-v1.json'
QUAL_SHA = 'fe50a340de7ada257479aa0c68ff841028965cf50ee183bde4e5a2c2eebe7c02'
LOSS_REL = 'benchmarks/results/2026-10-15-m6-online-full-loss-qualification-v1.json'
LOSS_SHA = '7b87646805ac5c6053150b3d7f02a06ea37daf46c967222401d6735d3a64def9'


def qualified_cli():
    q = json.loads(context.bounded(ROOT/QUAL_REL, 2*1024**2, QUAL_SHA))
    if q.get('format') != 'toklet-m6-long-admission-component-v1' or q.get('astra_review', {}).get('status') != 'accepted' or q.get('astra_review', {}).get('remaining_findings') != [] or q.get('checks', {}).get('native_cli_workers') != 6:
        raise ValueError('Reviewed M6 CLI prerequisite required')
    binary = ROOT/q['cli_binary']['path']
    expected = {str(p.relative_to(ROOT)) for p in [ROOT/'Cargo.toml', ROOT/'Cargo.lock', *(ROOT/'src').rglob('*.rs'), *(ROOT/'src').rglob('*.metal')]}
    compiled = {p: s for p, s in q['source_pins'].items() if p.startswith('src/') or p in ('Cargo.toml', 'Cargo.lock')}
    if set(compiled) != expected:
        raise ValueError('CLI compiled closure differs')
    pins = {ROOT/QUAL_REL: QUAL_SHA, binary: q['cli_binary']['sha256']}
    pins.update({ROOT/p: s for p, s in compiled.items()})
    pins.update({ROOT/q['source_snapshot']/p: s for p, s in compiled.items()})
    return binary, pins


def loss_prerequisite(model):
    q = json.loads(context.bounded(ROOT/LOSS_REL, 2*1024**2, LOSS_SHA))
    selected = [m for m in q.get('models', []) if m.get('model') == model]
    if q.get('passed') is not True or q.get('review', {}).get('accepted') is not True or len(selected) != 1 or selected[0].get('loss_gate_passed') is not True or selected[0].get('scored_targets') != 10000:
        raise ValueError('Reviewed unchanged full-loss gate required')


def ready(event, model, capacity=2048):
    execution = event.get('execution', {})
    if execution.get('attention_mode') != 'Online':
        raise ValueError('Actual Online CLI attention required')
    legacy = copy.deepcopy(event)
    del legacy['execution']['attention_mode']
    if model == 'small':
        memory = shared.cli.ready(legacy, 'HalfCompensated', capacity, 128)
        weights = 988069596
    elif model == 'larger':
        weights = context.loss.SPECS[model]['weights']
        memory = cli_v2.ready(legacy, cli_v2.MODE, capacity, 128, weights=weights, external=247772624)
    else:
        raise ValueError('Only selected models admitted')
    if memory['weights_bytes'] != weights or memory['tokenizer_application_allowance_bytes'] != 247772624:
        raise ValueError('Complete Online model/tokenizer residency differs')
    return memory


def arguments(model, packed):
    return ('q4-v2-compensated-tiled', ('--q4-weights', str(packed), '--metal-attention', 'online')) if model == 'larger' else ('half-compensated', ('--metal-attention', 'online'))


def provenance(model):
    binary, pins = qualified_cli()
    loss_prerequisite(model)
    _, _, source, packed, fixed = context.inputs(model)
    pins.update(fixed)
    pins[ROOT/LOSS_REL] = LOSS_SHA
    manifest, _ = shared.controlled.contract()
    pins.update(context.controlled_pins(manifest))
    pins[ROOT/manifest['quality']['tasks_file']] = shared.tasks.TASK_SHA
    pins.update({source/p: s for p, s in shared.cli.ASSET_PINS.items()})
    modules = (context, shared, shared.cli, cli_v2, context.loss, context.loss.shared, context.loss.large,
               context.loss.large.quality_rows, shared.controlled, shared.reference, shared.tasks,
               shared.tasks.prepare, shared.controlled.frozen, shared.controlled.common,
               shared.controlled.metal_prefill, shared.controlled.metal_continue, shared.metal_engine, shared.native_usage)
    files = set(pins) | {Path(__file__), *(Path(m.__file__) for m in modules)}
    identities = context.capture_identities(files, pins)
    return binary, source, packed, identities


def check_identities(identities):
    if any(shared.metal_engine.digest(ROOT/p) != s for p, s in identities.items()):
        raise ValueError('Online task source/input identities changed')


def run(output, model, check=False):
    paths = [output, output.with_suffix('.raw.jsonl'), output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):
        raise ValueError('Exclusive Online task evidence exists')
    binary, source, packed, identities = provenance(model)
    cases = shared.tasks.load_cases()
    projection, extra = arguments(model, packed)
    record = dict(format='toklet-m6-online-task-collection-v1', passed=False, check_only=check,
                  collection_completed=False, quality_completed=False, model=model, backend='metal',
                  attention_mode='Online', projection_mode='Q4CompensatedTiled' if model == 'larger' else 'HalfCompensated',
                  normalization_mode='Compensated' if model == 'larger' else 'Original', readback_mode='FullLogits',
                  task_ids=[c['id'] for c in cases], task_payload_sha256=shared.tasks.TASK_SHA,
                  identities=identities, revision=shared.controlled.metal_continue.observation(['git', 'rev-parse', 'HEAD']),
                  scope='Frozen100 greedy task responses; literal grading and latency acceptance separate')
    with paths[0].open('x') as summary, paths[1].open('x') as raw, paths[2].open('xb') as stderr:
        try:
            record['run'] = shared.child(binary, source, model, cases, raw, stderr, check,
                                         admission_check=ready, projection_argument=projection, extra_args=extra,
                                         before_launch=lambda: check_identities(identities))
            if not record['run']['passed']:
                raise ValueError('Online task worker failed')
            record.update(passed=True, collection_completed=not check)
        except BaseException as error:
            record['error'] = f'{type(error).__name__}: {error}'
        finally:
            record['endpoint_errors'] = []
            try:
                check_identities(identities); record['identities_unchanged'] = True
            except BaseException as error:
                record['identities_unchanged'] = False; record['endpoint_errors'].append(str(error))
            if not record['identities_unchanged'] or record['endpoint_errors']:
                record.update(passed=False, collection_completed=False)
            json.dump(record, summary, indent=2, allow_nan=False); summary.write('\n')
    if not record['passed']:
        raise ValueError('Online task collection failed; see retained evidence')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=('small', 'larger'), required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args(); run(a.output.resolve(), a.model, a.check)
