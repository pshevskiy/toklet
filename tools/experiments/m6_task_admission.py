"""Admit complete Online100-task collections to the existing literal grader."""
import hashlib
from pathlib import Path

import m6_tasks as producer
import task_grade_admission as legacy

ROOT = producer.ROOT
evidence = legacy.evidence


def validate_records(record, cases, events):
    model = record.get('model')
    if model not in ('small', 'larger'):
        raise ValueError('Selected Online model required')
    expected = dict(format='toklet-m6-online-task-collection-v1', passed=True, collection_completed=True,
                    check_only=False, quality_completed=False, identities_unchanged=True, endpoint_errors=[],
                    task_payload_sha256=producer.shared.tasks.TASK_SHA, backend='metal', attention_mode='Online',
                    projection_mode='Q4CompensatedTiled' if model == 'larger' else 'HalfCompensated',
                    normalization_mode='Compensated' if model == 'larger' else 'Original', readback_mode='FullLogits')
    if len(cases) != 100 or any(type(record.get(k)) is not type(v) or record[k] != v for k, v in expected.items()) or record.get('task_ids') != [c['id'] for c in cases]:
        raise ValueError('Complete frozen Online100-case collection required')
    run = record.get('run', {})
    if run.get('passed') is not True or run.get('model_exited') is not True or type(run.get('exit_code')) is not int or run['exit_code'] != 0 or run.get('cleanup_errors') != [] or len(run.get('cases', [])) != 100 or type(run.get('worker_pid')) is not int or run['worker_pid'] <= 1:
        raise ValueError('Naturally exited owned Online model required')
    producer.ready(run['ready'], model)
    responses, expected_events = [], [run['ready']]
    for case, item in zip(cases, run['cases'], strict=True):
        if type(item.get('max_output_tokens')) is not int or item['max_output_tokens'] != case['max_output_tokens']:
            raise ValueError('Original full task cap differs')
        response = producer.shared.collect_response(case, item['native_events'], run['ready'])
        if response != item['response']:
            raise ValueError('Saved Online response differs from raw events')
        responses.append(response); expected_events.extend(item['native_events'])
    if events != expected_events or [u.get('phase') for u in run.get('idle_usage', [])] != ['ready']+[f'case_{i}' for i in range(1, 101)]:
        raise ValueError('Complete raw/idle ordered coverage differs')
    for usage in run['idle_usage']:
        a, b = (usage[k].get('lifetime_max_phys_footprint') for k in ('worker', 'driver'))
        if type(a) is not int or type(b) is not int or a <= 0 or b <= 0 or type(usage.get('joint_native_lifetime_peak_bytes')) is not int or usage['joint_native_lifetime_peak_bytes'] != a+b or a+b > producer.shared.LIMIT:
            raise ValueError('Online task native charge differs')
    identity = {k: record[k] for k in ('backend', 'model', 'attention_mode', 'projection_mode', 'normalization_mode', 'readback_mode')}
    return dict(identity=identity, responses=responses, worker_pid=run['worker_pid'], record=record)


def validate_collection(path):
    path = Path(path).resolve()
    payload = legacy.bounded(path, legacy.MAX_SUMMARY)
    record = evidence.decode_event(payload.decode())
    raw = legacy.bounded(path.with_suffix('.raw.jsonl'), legacy.MAX_RAW)
    events = [evidence.decode_event(line) for line in raw.decode().splitlines()]
    cases = producer.shared.tasks.load_cases()
    valid = validate_records(record, cases, events)
    binary, source, packed, expected_identities = producer.provenance(record['model'])
    if record.get('identities') != expected_identities:
        raise ValueError('Complete Online producer inventory differs')
    projection, extra = producer.arguments(record['model'], packed)
    command = [str(binary), '--model', str(source), '--backend', 'metal', '--metal-projection', projection,
               '--interactive', '--json', '--context-capacity', '2048', '--prefill-chunk', '128', *extra]
    if record['run'].get('command') != command:
        raise ValueError('Exact qualified Online command differs')
    producer.shared.finalized(valid['responses'], source/'tokenizer.json', model_exited=True)
    valid.update(collection_path=str(path), collection_sha256=hashlib.sha256(payload).hexdigest(),
                 raw_sha256=hashlib.sha256(raw).hexdigest(), cases=cases)
    return valid


def admit_idle(path, selected):
    payload = legacy.bounded(path, 65536)
    proof = evidence.decode_event(payload.decode())
    if proof.get('format') != 'toklet-m6-online-task-idle-v1' or proof.get('collections') != [str(Path(selected).resolve())]:
        raise ValueError('Exactly the completed Online collection must be idle')
    accepted = validate_collection(selected)
    # Actual process/group absence is checked by the existing window before VM
    # startup; never run generated programs in the model or host process.
    return dict(proof_path=str(Path(path).resolve()), proof_sha256=hashlib.sha256(payload).hexdigest(),
                collections=[dict(path=accepted['collection_path'], sha256=accepted['collection_sha256'],
                                  raw_sha256=accepted['raw_sha256'], identity=accepted['identity'], worker_pid=accepted['worker_pid'])],
                protected_binaries={
                    'target/release/toklet': '5250694e814ea112148a52cfdeaf5bc6286ff2804b3acfd8f89e80bc73122a59',
                    'target/release/examples/cpu_bench': 'e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea'})
