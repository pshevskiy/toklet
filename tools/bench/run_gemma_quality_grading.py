"""Own the Gemma grading window through worker exit and stopped-VM verification."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import gemma_quality_grading as adapter
import paired_task_grading as runner
import grading_support as support
import linux_code_isolation_probe as boundary
from run_llama3_quality_grading import acceptable, group_alive, read_inner_result

ROOT = adapter.ROOT


def run(collection_file, collection_sha, output, timeout=3600):
    output = Path(output).resolve()
    support.require(output.is_relative_to(ROOT/'artifacts'), 'Fresh outer artifact folder required')
    output.mkdir()
    bindings = {}
    record = dict(completed=False, generated_code_runs_on_host=False)
    workers = []
    try:
        adapter.bind(bindings, collection_file, collection_sha)
        collection, _ = adapter.read_json(collection_file)
        support.require(collection['format'] == 'toklet-gemma-paired-task-collection-v1'
                        and collection['completed'] is True and collection['cases'] == 100
                        and collection['manifest_sha256'] == adapter.MANIFEST_SHA,
                        'Complete paired Gemma collection required')
        for role in ['original', 'native']:
            item = collection['outputs'][role]
            path = ROOT/item['worker_receipt']
            adapter.bind(bindings, path, item['worker_receipt_sha256'])
            worker, _ = adapter.read_json(path)
            workers.append(worker)
            adapter.bind(bindings, ROOT/item['journal'], item['journal_sha256'])
            support.require(worker['tasks_sha256'] == item['journal_sha256'], 'Worker journal binding differs')
        adapter.results.require_idle(workers)
        for name in [__file__, adapter.__file__, runner.__file__, support.__file__,
                     'tools/bench/run_llama3_quality_grading.py', 'tools/bench/llama3_quality_grading.py',
                     'tools/bench/gemma_task_results.py', 'tools/bench/check_gemma_quality.py',
                     'tools/bench/task_cases.py', 'tools/bench/linux_task_controller.py',
                     'tools/bench/linux_task_evidence.py', 'tools/bench/linux_code_isolation_probe.py',
                     'tools/bench/podman_machine_lifecycle.py', 'tools/bench/held_out_task_window.py',
                     runner.PRIOR, runner.LITERAL]:
            adapter.bind(bindings, ROOT/name)
        prior = runner.checked_json(runner.PRIOR, runner.PRIOR_SHA, 16*1024**2)
        client_pin = dict(bytes=Path(runner.PODMAN).stat().st_size, sha256=prior['podman_sha256'])

        def guard():
            support.verify_file(Path(runner.PODMAN), client_pin)
            for name, pin in bindings.items():
                support.verify_file(ROOT/name, pin)
            adapter.results.require_idle(workers)

        def observe():
            item = boundary.capture([runner.PODMAN, 'machine', 'list', '--format=json'], 20, 262144)
            support.require(not item.get('failure') and item['exit_code'] == 0 and not item['wall_timed_out']
                            and not item['output_limit_exceeded'] and item['cleanup_errors'] == []
                            and item['surviving_group_members'] == [], 'Outer machine query failed')
            return json.loads(boundary.output(item))

        command = [sys.executable, str(ROOT/'tools/bench/gemma_quality_grading.py'),
                   str(collection_file), collection_sha, str(output/'inner')]
        # Preserve the qualified Podman JSON's local-time rendering.
        environment = dict(os.environ, TZ='Europe/Moscow')
        plan = dict(bindings=bindings, podman_client_pin=client_pin, command=command,
                    collection_sha256=collection_sha, manifest_sha256=adapter.MANIFEST_SHA,
                    timezone=environment['TZ'], policy='TERM only; retain owner through exit and stopped VM')
        support.write_json(output/'execution-plan.json', plan)
        guard()
        window = support.supervise_window(command, output/'child.stdout', output/'child.stderr',
                                          environment, observe, group_alive, timeout)
        record['window'] = window
        guard()
        pid = window.get('pid')
        support.require(type(pid) is int and pid > 1 and not group_alive(pid), 'Grader group remains live')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('Grader process remains live')
        inner_path = output/'inner/result.json'
        inner, pin = read_inner_result(inner_path)
        support.require(acceptable(window, inner), 'Outer or inner restoration/decision guard failed')
        support.require(inner['format'] == 'toklet-gemma-full-task-grading-v1'
                        and inner['collection_sha256'] == collection_sha
                        and inner['manifest_sha256'] == adapter.MANIFEST_SHA,
                        'Inner grading identity differs from outer plan')
        decision = adapter.results.decision(adapter.task_cases.load_cases(), inner['outcomes'])
        support.require(decision == inner['decision'], 'Inner decision differs from complete resolved outcomes')
        adapter.bind(bindings, inner_path, pin['sha256'])
        record.update(completed=True, decision=decision, inner_result_sha256=pin['sha256'],
                      independent_group_absence_verified=True,
                      plan_sha256=hashlib.sha256((output/'execution-plan.json').read_bytes()).hexdigest())
    except BaseException as error:
        record['failure'] = dict(type=type(error).__name__, message=str(error)[:4096])
    record['bindings'] = bindings
    support.write_json(output/'result.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path)
    parser.add_argument('collection_sha256')
    parser.add_argument('fresh_output', type=Path)
    args = parser.parse_args()
    report = run(args.collection, args.collection_sha256, args.fresh_output)
    print(json.dumps({k: v for k, v in report.items() if k != 'bindings'}, indent=2))
    raise SystemExit(0 if report['completed'] else 1)
