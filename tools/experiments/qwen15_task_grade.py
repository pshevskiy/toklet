"""Admit the larger MLX8 native collection to the qualified literal grader."""
import argparse
import hashlib
from pathlib import Path
import sys

import qwen15_tasks as producer
import task_grade_admission as legacy
import task_grading_window as window

ROOT = producer.ROOT
EVIDENCE = legacy.evidence
RECEIPT = ROOT/'benchmarks/results/2026-11-06-qwen15-task-collection-v1.json'
LOSS = ROOT/'benchmarks/results/2026-11-06-qwen15-full-loss-v1.json'
PROTECTED = {
    'target/release/toklet': '5250694e814ea112148a52cfdeaf5bc6286ff2804b3acfd8f89e80bc73122a59',
    'target/release/examples/cpu_bench': 'e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea',
}


def validate_collection(path):
    path = Path(path).resolve()
    payload = legacy.bounded(path, legacy.MAX_SUMMARY)
    record = EVIDENCE.decode_event(payload.decode())
    raw = legacy.bounded(path.with_suffix('.raw.jsonl'), legacy.MAX_RAW)
    receipt = EVIDENCE.decode_event(legacy.bounded(RECEIPT, 65536).decode())
    loss = EVIDENCE.decode_event(legacy.bounded(LOSS, 65536).decode())
    payload_sha = hashlib.sha256(payload).hexdigest()
    raw_sha = hashlib.sha256(raw).hexdigest()
    if (loss.get('passed') is not True or loss.get('scored') != 10000
            or receipt.get('passed') is not True or receipt['loss_sha256'] != producer.digest(LOSS)
            or path != ROOT/receipt['collection'] or payload_sha != receipt['collection_sha256']
            or raw_sha != receipt['raw_sha256']):
        raise ValueError('Reviewed larger MLX8 collection and full loss required')
    cases = producer.shared.tasks.load_cases()
    producer.reference(cases)
    if (record.get('format') != 'toklet-qwen15-mlx8-task-collection-v1'
            or record.get('passed') is not True or record.get('check_only') is not False
            or record.get('identity') != producer.IDENTITY
            or record.get('task_payload_sha256') != producer.shared.tasks.TASK_SHA
            or record.get('task_ids') != [case['id'] for case in cases]):
        raise ValueError('Complete frozen larger MLX8 task collection required')
    binary = ROOT/receipt['binary']
    if producer.digest(binary) != receipt['binary_sha256'] or producer.files(binary) != record['identities']:
        raise ValueError('Larger MLX8 exact producer/source inventory differs')
    if ({p.name for p in producer.SOURCE.iterdir() if p.name != 'README.md'} != set(producer.MODEL_PINS)
            or {p.name for p in producer.RAW_SOURCE.iterdir()} != producer.RAW_ASSETS):
        raise ValueError('Larger MLX8 source inventory changed')
    run = record['run']
    if (run.get('passed') is not True or run.get('model_exited') is not True
            or type(run.get('exit_code')) is not int or run['exit_code'] != 0
            or type(run.get('worker_pid')) is not int or run['worker_pid'] <= 1
            or run.get('cleanup_errors') != [] or len(run.get('cases', [])) != 100
            or run['command'] != producer.command(binary)):
        raise ValueError('Naturally exited exact native task owner required')
    producer.ready(run['ready'], producer.IDENTITY['model'])
    producer.absent(run['worker_pid'])
    expected_events = [run['ready']]
    responses = []
    for case, item in zip(cases, run['cases'], strict=True):
        if item.get('max_output_tokens') != case['max_output_tokens']:
            raise ValueError('Original full task output cap differs')
        response = producer.shared.collect_response(case, item['native_events'], run['ready'])
        if response['visible_text']:
            raise ValueError('Raw native task worker emitted text')
        expected_events.extend(item['native_events'])
        responses.append(response)
    if [EVIDENCE.decode_event(line) for line in raw.decode().splitlines()] != expected_events:
        raise ValueError('Complete raw Metal journal differs')
    if [item.get('phase') for item in run['idle_usage']] != ['ready']+[f'case_{i}' for i in range(1,101)]:
        raise ValueError('Ordered native peak coverage differs')
    for item in run['idle_usage']:
        worker, driver = (item[name]['lifetime_max_phys_footprint'] for name in ('worker','driver'))
        if (type(worker) is not int or type(driver) is not int or worker <= 0 or driver <= 0
                or worker+driver != item['joint_native_lifetime_peak_bytes']
                or worker+driver > producer.shared.LIMIT):
            raise ValueError('Joint native model/driver memory differs')
    producer.shared.reference.finalize(responses, producer.SOURCE/'tokenizer.json', model_exited=True)
    if any(response != item['response'] for response, item in zip(responses, run['cases'], strict=True)):
        raise ValueError('Finalized raw native task response differs')
    return dict(collection_path=str(path), collection_sha256=payload_sha, raw_sha256=raw_sha,
                identity=producer.IDENTITY, worker_pid=run['worker_pid'], cases=cases, responses=responses)


def admit_idle(path, selected):
    data = legacy.bounded(path, 65536)
    proof = EVIDENCE.decode_event(data.decode())
    if proof != {'format':'toklet-qwen15-task-idle-v1','collections':[str(Path(selected).resolve())]}:
        raise ValueError('Exactly the completed larger MLX8 collection must be idle')
    accepted = validate_collection(selected)
    return dict(proof_path=str(Path(path).resolve()), proof_sha256=hashlib.sha256(data).hexdigest(),
                collections=[dict(path=accepted['collection_path'], sha256=accepted['collection_sha256'],
                                  raw_sha256=accepted['raw_sha256'], identity=accepted['identity'],
                                  worker_pid=accepted['worker_pid'])], protected_binaries=PROTECTED)


def run(args):
    module = sys.modules[__name__]
    return window.run(args, collection_admission=module, idle_admission=admit_idle,
                      extra_source_paths=(Path(__file__).resolve(), Path(producer.__file__).resolve(), RECEIPT, LOSS),
                      extra_grading_modules=(module, producer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--idle-proof', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    sys.exit(run(parser.parse_args()))
