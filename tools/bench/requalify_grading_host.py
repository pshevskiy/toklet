"""Repeat fixed evaluator controls after changing host process-group discovery.

This owns the existing stopped 2 GiB VM for one window. It executes only the
original 14 isolation probes and 28 authored literal programs. Dataset programs
and model responses are never accepted; guest code and limits stay unchanged.
Run under the separately bounded host/guest resource supervisor.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

import grading_support as support
from held_out_task_window import running_containers
import linux_code_isolation_child as child
import linux_code_isolation_probe as boundary
import linux_task_controller as controller
import linux_task_native_probe as literal
import podman_machine_lifecycle as lifecycle

ROOT = boundary.ROOT
PODMAN = '/opt/podman/bin/podman'
IMAGE = 'eb7d16fec874f15b584063275012ac719885a2b87574d124c0da9c156d504629'
PRIORS = (
    ('2026-10-11-linux-code-isolation-probe-v8.json',
     '0736ec7d8cf7481c5327c01c5f0c8555b6e3c8669c6c63e7a267f0044873d3f1'),
    ('2026-10-11-literal-native-v3-probe.json',
     '77cdda26d80a079bc56da9a80d52f7fa09d7b2942ce768d464128f07798a9453'),
)


def prior_records():
    records = []
    for name, sha in PRIORS:
        path = ROOT / 'benchmarks/results' / name
        support.require(path.stat().st_size <= 16 * 1024**2, 'Bounded prior proof required')
        support.verify_file(path, dict(bytes=path.stat().st_size, sha256=sha))
        records.append(json.loads(path.read_bytes()))
    first, second = records
    support.require(first['completed'] is True and second['completed'] is True
                    and len(first['runs']) == 14 and len(second['runs']) == 28
                    and all(row['expectation_passed'] is True for row in second['runs']),
                    'Complete original qualification required')
    support.require(first['policy'] == child.POLICY
                    and first['bundle_sha256'] == hashlib.sha256(boundary.bundled_source().encode()).hexdigest()
                    and second['bundle_sha256'] == hashlib.sha256(controller.bundled_source().encode()).hexdigest(),
                    'Guest program or original policy changed')
    return records


def run(output):
    output = output.resolve()
    support.require(output.is_relative_to(ROOT / 'artifacts') and not output.exists(),
                    'Fresh artifact output required')
    priors = prior_records()
    # The two host wrappers are the only original source bindings being replaced.
    replaced = {'tools/bench/code_isolation_probe.py', 'tools/bench/task_cases_probe.py'}
    for prior in priors:
        for name, sha in prior['source_sha256'].items():
            if name not in replaced:
                support.require(boundary.digest(ROOT / name) == sha, 'Unchanged evaluator source differs')
    names = set().union(*(prior['source_sha256'] for prior in priors))
    names.update({'tools/bench/process_groups.py', 'tools/bench/requalify_grading_host.py',
                  'tools/bench/podman_machine_lifecycle.py', 'tools/bench/held_out_task_window.py',
                  'tools/bench/grading_support.py'})
    identities = {name: boundary.digest(ROOT / name) for name in sorted(names)}
    literal_path = ROOT / 'benchmarks/raw' / (output.name + '-literal.json')
    support.require(not literal_path.exists(), 'Fresh literal qualification output required')
    output.mkdir()
    record = dict(completed=False, runs=[], commands=[], started_by_this_task=False,
                  generated_code_runs_on_host=False, started_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=identities, predecessor_sha256=dict(PRIORS))
    owner = None
    api = None
    old = {}

    def guard():
        support.require(all(boundary.digest(ROOT / name) == sha for name, sha in identities.items()),
                        'Evaluator sources changed during qualification')
        support.require(boundary.digest(PODMAN) == priors[0]['podman_sha256'], 'Podman binary changed')

    def command(argv, seconds=120):
        item = boundary.capture(argv, seconds, 262144)
        record['commands'].append(item)
        support.require(not item.get('failure') and item['exit_code'] == 0
                        and not item['wall_timed_out'] and not item['output_limit_exceeded']
                        and item['cleanup_errors'] == [] and item['surviving_group_members'] == [],
                        'Bounded qualification control failed')
        return boundary.output(item)

    def observe():
        return json.loads(command([PODMAN, 'machine', 'list', '--format=json'], 20))

    def stop(number, _frame):
        raise InterruptedError(str(number))

    try:
        guard()
        before = observe()
        machine = support.selected(before)
        support.require(machine.get('Running') is False and machine.get('Starting') is False
                        and machine.get('Memory') in ('2147483648', 2147483648),
                        'Existing stopped 2 GiB VM required')
        record['machine_before'] = before
        inspected = json.loads(command([PODMAN, 'machine', 'inspect', lifecycle.MACHINE]))
        connections = json.loads(command([PODMAN, 'system', 'connection', 'list', '--format=json']))
        support.require(len(inspected) == 1, 'Unique machine inspection required')
        binding = boundary.bound_machine_connection(inspected[0], connections)
        support.require(binding == priors[0]['connection_binding'], 'Prior VM/API identity changed')
        record['connection_binding'] = binding
        api = (PODMAN, '--url', binding['URI'], '--identity', binding['Identity'])
        old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        record['started_by_this_task'] = True
        startup, owner = lifecycle.start_machine(PODMAN)
        record['startup'] = startup
        support.require(startup['client_succeeded'] and not startup.get('helper_capture_failure'),
                        'Owned VM startup failed')
        deadline = time.monotonic() + 60
        while True:
            item = boundary.capture([*api, 'info', '--format=json'], 20, 262144)
            record['commands'].append(item)
            if (not item.get('failure') and item['exit_code'] == 0 and not item['wall_timed_out']
                    and not item['output_limit_exceeded'] and not item['cleanup_errors']
                    and item['surviving_group_members'] == []):
                break
            support.require(time.monotonic() < deadline, 'Owned API readiness failed')
            time.sleep(2)
        record['podman_version'] = command([*api, 'version', '--format=json'])
        support.require(json.loads(record['podman_version']) == json.loads(priors[0]['podman_version']),
                        'Qualified Podman version changed')
        image = json.loads(command([*api, 'image', 'inspect', 'sha256:' + IMAGE]))
        support.require(len(image) == 1 and priors[0]['image']
                        == {key: image[0].get(key) for key in priors[0]['image']},
                        'Qualified guest image changed')
        record['image'] = {key: image[0].get(key) for key in priors[0]['image']}
        record.update(podman_binary=PODMAN, podman_sha256=boundary.digest(PODMAN),
                      policy=child.POLICY, admitted_runtime=priors[0]['admitted_runtime'],
                      admitted_python_version=priors[0]['admitted_python_version'],
                      bundle_sha256=hashlib.sha256(boundary.bundled_source().encode()).hexdigest())
        support.require(running_containers(api, record) == [], 'Unexpected running container')
        source = boundary.bundled_source()
        for mode in child.MODES:
            guard()
            item = boundary.execute(api, IMAGE, '/usr/local/bin/python3', source, mode)
            record['runs'].append(item)
            support.require(not item.get('failure') and not item['container_cleanup_errors']
                            and item.get('container_removed') is True, 'Isolation control infrastructure failed')
            active = boundary.check_record(item)
            support.require(active['runtime'] == priors[0]['admitted_runtime']
                            and active['python'] == priors[0]['admitted_python_version'],
                            'Original guest runtime changed')
            item['expectation_passed'] = True
            print(json.dumps(dict(isolation_mode=mode, passed=True)), flush=True)
        support.require(len(record['runs']) == 14, 'All original isolation modes required')
        native = literal.run(api, IMAGE, priors[0]['admitted_runtime'],
                             priors[0]['admitted_python_version'], literal_path)
        record['literal_result'] = str(literal_path.relative_to(ROOT))
        record['literal_result_sha256'] = boundary.digest(literal_path)
        support.require(native['completed'] is True and len(native['runs']) == 28
                        and all(item['expectation_passed'] is True for item in native['runs']),
                        'All original authored literal cases required')
        guard()
        support.require(running_containers(api, record) == [], 'Container survived qualification')
        final = json.loads(command([*api, 'image', 'inspect', 'sha256:' + IMAGE]))
        support.require(len(final) == 1 and record['image']
                        == {key: final[0].get(key) for key in record['image']}, 'Guest image changed')
        inspected = json.loads(command([PODMAN, 'machine', 'inspect', lifecycle.MACHINE]))
        connections = json.loads(command([PODMAN, 'system', 'connection', 'list', '--format=json']))
        support.require(len(inspected) == 1
                        and boundary.bound_machine_connection(inspected[0], connections) == binding,
                        'VM/API identity changed')
        record['controls_passed'] = True
    except BaseException as error:
        record['failure'] = dict(type=type(error).__name__, message=str(error)[:4096])
    finally:
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        if record['started_by_this_task']:
            try:
                machine = support.selected(observe())
                if machine['Running'] is True or machine['Starting'] is True:
                    support.require(running_containers(api, record) == [], 'Unknown container prevents stop')
                    command([PODMAN, 'machine', 'stop', lifecycle.MACHINE])
                after = observe()
                machine = support.selected(after)
                support.require(machine['Running'] is False and machine['Starting'] is False,
                                'VM restoration unverified')
                record['machine_after'] = after
                record['restored_stopped'] = True
                if owner is not None:
                    owner.finish(after)
                    support.require(not record['startup'].get('helper_capture_failure'), 'Helper capture failed')
                    record['pipe_lifetime_finished'] = True
            except BaseException as error:
                record['restoration_failure'] = dict(type=type(error).__name__, message=str(error)[:4096])
        try:
            guard()
            record['completed'] = (record.get('controls_passed') is True
                                   and record.get('restored_stopped') is True
                                   and record.get('pipe_lifetime_finished') is True
                                   and not record.get('failure') and not record.get('restoration_failure'))
            support.write_json(output / 'result.json', record)
        finally:
            try:
                lifecycle.await_stopped_owner(owner, observe)
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({key: result.get(key) for key in ('completed', 'failure', 'restoration_failure')}))
    raise SystemExit(0 if result['completed'] else 1)
