"""Read-only replay of completed M6 responses/literal windows; no model/code execution."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/Users/coloner/work/rust/toklet')
sys.path.insert(0, str(ROOT/'tools/experiments'))
import m6_task_admission as admission
import task_grading_window as window

producer = admission.producer
evidence = admission.evidence
bindings = {}


def bind(path, expected=None):
    path = Path(path)
    digest = producer.shared.metal_engine.digest(path)
    if expected is not None and digest != expected:
        raise ValueError(f'Endpoint identity differs: {path}')
    name = str(path.relative_to(ROOT))
    if name in bindings and bindings[name]['sha256'] != digest:
        raise ValueError(f'Conflicting identity: {name}')
    bindings[name] = {'sha256': digest}
    return digest


def record(path, maximum=32*1024**2):
    bind(path)
    return evidence.decode_event(admission.legacy.bounded(path, maximum).decode())


def main():
    result = dict(format='toklet-m6-online-task-quality-audit-v1', passed=False, quality_completed=False,
                  scope='Unchanged full100-task gate for both selected Online configurations; no performance or long-context acceptance',
                  source_parent='bf2dd9c', models=[], native_code_runs=0, resolved_outcomes=0,
                  infrastructure_unresolved=0, bindings=bindings)
    bind(Path(__file__))
    baselines = {}
    for name, sha in [('2026-10-14-half-task-quality-qualification-v1.json', '9ad3e14409b0154b5ac9b73175fc7a5f6594e81f6f44c40b3cc448a11c3a0c92'),
                      ('2026-10-14-q4-v2-task-quality-qualification-v1.json', 'ecee8abce37d56171c730a980344dd737764d3fa6aa87c26ac794d79bff7b217')]:
        p = ROOT/'benchmarks/results'/name; bind(p, sha); baselines[name] = record(p)
    assert baselines['2026-10-14-half-task-quality-qualification-v1.json']['small']['fp32_passed'] == 2
    assert baselines['2026-10-14-q4-v2-task-quality-qualification-v1.json']['fp32_passed'] == 11
    for model, baseline in [('small', 2), ('larger', 11)]:
        path = ROOT/f'artifacts/m6-tasks-{model}-full-v1.json'
        accepted = admission.validate_collection(path)
        collection = accepted['record']; run = collection['run']
        for p, sha in collection['identities'].items():
            bind(ROOT/p, sha)
        for suffix in ('.json', '.raw.jsonl', '.stderr'):
            bind(path.with_suffix(suffix))
        idle = ROOT/f'artifacts/m6-tasks-{model}-idle-v1.json'
        proof = record(idle)
        assert proof == {'format': 'toklet-m6-online-task-idle-v1', 'collections': [str(path)]}
        binding = dict(proof_path=str(idle), proof_sha256=bindings[str(idle.relative_to(ROOT))]['sha256'],
                       collections=[dict(path=str(path), sha256=accepted['collection_sha256'], raw_sha256=accepted['raw_sha256'],
                                         identity=accepted['identity'], worker_pid=accepted['worker_pid'])],
                       protected_binaries={
                           'target/release/toklet': '5250694e814ea112148a52cfdeaf5bc6286ff2804b3acfd8f89e80bc73122a59',
                           'target/release/examples/cpu_bench': 'e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea'})
        window.check_idle(binding)
        for p, sha in binding['protected_binaries'].items(): bind(ROOT/p, sha)
        prefix = ROOT/f'benchmarks/raw/2026-10-14-backend-task-m6-online-{model}-v1'
        w = record(Path(str(prefix)+'-window.json'))
        g = record(Path(str(prefix)+'-probe.json'))
        bind(ROOT/f'artifacts/m6-task-{model}-window-v1.log')
        assert all(w.get(k) is True for k in ('completed', 'quality_completed', 'started_by_this_task', 'restored_stopped', 'pipe_lifetime_finished'))
        assert not w.get('failure') and not w.get('restoration_failure') and w['running_containers_before_stop'] == []
        assert w['collection_sha256'] == accepted['collection_sha256'] and (ROOT/w['collection_path']).resolve() == path
        assert w['identity'] == accepted['identity'] and w['idle_collections'] == binding['collections']
        assert w['idle_sha256'] == binding['proof_sha256']
        assert all(row['Running'] is False and row['Starting'] is False for row in w['machine_after'])
        literal = window.pinned_record(ROOT/'benchmarks/results/2026-10-11-literal-native-v3-probe.json', window.LITERAL_SHA)
        prior = window.pinned_record(window.PRIOR, window.PRIOR_SHA)
        assert literal['completed'] and prior['completed']
        assert w['image'] == prior['image'] and w['connection_binding'] == prior['connection_binding']
        assert w['literal_qualification_sha256'] == window.LITERAL_SHA and w['prior_qualification_sha256'] == window.PRIOR_SHA
        bind(ROOT/'benchmarks/results/2026-10-11-literal-native-v3-probe.json', window.LITERAL_SHA)
        bind(window.PRIOR, window.PRIOR_SHA)
        assert g['completed'] is True and g['quality_completed'] is True and g['unresolved_case_ids'] == []
        assert g['collection_sha256'] == accepted['collection_sha256'] and g['native_raw_sha256'] == accepted['raw_sha256']
        assert g['identity'] == accepted['identity'] and g['task_payload_sha256'] == producer.shared.tasks.TASK_SHA
        assert g['runtime'] == prior['admitted_runtime'] and g['python_version'] == prior['admitted_python_version']
        for names in (w['window_source_sha256'], g['source_sha256'], literal['source_sha256']):
            for p, sha in names.items(): bind(ROOT/p, sha)
        programs = iter(g['runs']); outcomes = []; reasons = {}
        for case, response in zip(accepted['cases'], accepted['responses'], strict=True):
            if case['category'] == 'code':
                native = next(programs)
                assert native['id'] == case['id'] and native['completion_sha256'] == response['visible_text_sha256']
                program = producer.shared.tasks.human_eval_program(case, response['visible_text'])
                payload = dict(case_id=case['id'], program=program, program_sha256=hashlib.sha256(program.encode()).hexdigest())
                assert native['program_sha256'] == payload['program_sha256']
                resolved = evidence.classify(native['raw'], payload, g['runtime'], g['python_version'])
                assert resolved == native['resolved']
                passed = resolved['passed']; reason = resolved.get('reason', resolved['outcome'])
                reasons[reason] = reasons.get(reason, 0)+1
            else:
                passed = producer.shared.tasks.grade_text(case, response['visible_text'])
            outcomes.append(dict(id=case['id'], category=case['category'], passed=passed,
                                 status='model_pass' if passed else 'model_fail'))
        assert next(programs, None) is None and outcomes == g['outcomes']
        aggregate = producer.shared.tasks.aggregate(accepted['cases'], outcomes)
        assert aggregate == g['aggregate'] and aggregate['cases'] == 100 and len(g['runs']) == 40
        drop = baseline-aggregate['passed']
        assert drop <= 2
        result['models'].append(dict(model=model, cases=100, online_passed=aggregate['passed'], fp32_passed=baseline,
                                     drop_percentage_points=drop, unchanged_allowed_drop_percentage_points=2,
                                     aggregate=aggregate, code_reasons=reasons, code_runs=40, resolved_outcomes=100,
                                     collection=str(path.relative_to(ROOT)), grading=str(Path(str(prefix)+'-probe.json').relative_to(ROOT)),
                                     window=str(Path(str(prefix)+'-window.json').relative_to(ROOT)),
                                     admission_bytes=run['ready']['memory_admitted_bytes'],
                                     max_joint_native_bytes=max(u['joint_native_lifetime_peak_bytes'] for u in run['idle_usage'])))
        result['native_code_runs'] += 40; result['resolved_outcomes'] += 100
    final_path = ROOT/'artifacts/m6-task-final-machine-v1.json'
    final = record(final_path)
    assert final['exit_code'] == 0 and not final.get('failure') and final['cleanup_errors'] == [] and final['surviving_group_members'] == []
    inventory = json.loads(window.boundary.output(final))
    assert inventory and all(v['Running'] is False and v['Starting'] is False for v in inventory)
    result.update(passed=True, quality_completed=True, all_windows_restored_stopped=True,
                  all_helper_pipe_lifetimes_finished=True, final_machine_stopped=True)
    with (ROOT/'artifacts/m6-task-quality-audit-v2.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print('Replayed200raw responses/200outcomes/80literal programs; unchanged2/100and11/100 gates pass; bothwindows/freshVMstopped')


if __name__ == '__main__': main()
