"""Frozen held-out grading window after completed collection.

Only run after all three matched full task collections naturally exit.
Preserve failed attempts; never resume the original held orchestration.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, 'tools/bench')
import linux_code_isolation_probe as boundary
import podman_machine_lifecycle as lifecycle
import task_grading as probe
import task_grade_admission as admission
import hashlib
import argparse

ROOT = boundary.ROOT
PODMAN = '/opt/podman/bin/podman'
IMAGE = 'eb7d16fec874f15b584063275012ac719885a2b87574d124c0da9c156d504629'
PREFIX = None
IDLE = None
PRIOR = ROOT / 'benchmarks/results/2026-10-11-linux-code-isolation-probe-v8.json'


LITERAL_SHA = '77cdda26d80a079bc56da9a80d52f7fa09d7b2942ce768d464128f07798a9453'
PRIOR_SHA = '0736ec7d8cf7481c5327c01c5f0c8555b6e3c8669c6c63e7a267f0044873d3f1'


def pinned_record(path, expected):
    data = admission.bounded(path, 8*1024**2)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Qualified literal/runtime receipt differs')
    return admission.evidence.decode_event(data.decode())


def check_idle(binding):
    if boundary.digest(binding['proof_path']) != binding['proof_sha256']:
        raise ValueError('Task idle proof changed')
    pids = [item['worker_pid'] for item in binding['collections']]
    absent = subprocess.run(['ps', '-p', ','.join(map(str, pids)), '-o', 'pid='], capture_output=True, text=True)
    if absent.returncode != 1 or absent.stdout or absent.stderr or any(boundary.live_group_members(pid) for pid in pids):
        raise ValueError('Task owned model process/group remains live')
    for item in binding['collections']:
        if (boundary.digest(item['path']) != item['sha256']
                or boundary.digest(Path(item['path']).with_suffix('.raw.jsonl')) != item['raw_sha256']):
            raise ValueError('Task idle collection changed')
    for name, expected in binding['protected_binaries'].items():
        if boundary.digest(ROOT/name) != expected:
            raise ValueError('Protected CPU binary differs')


def admit_idle(path, selected):
    data = admission.bounded(path, 65536)
    proof = admission.evidence.decode_event(data.decode())
    paths = proof.get('collections')
    if proof.get('format') != 'toklet-backend-task-idle-input-v1' or type(paths) is not list or len(paths) != 3:
        raise ValueError('All three full task collections required')
    paths = [Path(p).resolve() for p in paths]
    if len(set(paths)) != 3 or Path(selected).resolve() not in paths:
        raise ValueError('Distinct full task collection paths required')
    collections = []
    for collection_path in paths:
        accepted = admission.validate_collection(collection_path)
        collections.append({'path':str(collection_path), 'sha256':accepted['collection_sha256'], 'raw_sha256':accepted['raw_sha256'], 'identity':accepted['identity'], 'worker_pid':accepted['worker_pid']})
    roles = {(v['identity']['backend'],v['identity']['model']) for v in collections}
    if roles != {('metal','small'),('metal','larger'),('cpu','larger')} or len({v['worker_pid'] for v in collections}) != 3:
        raise ValueError('Matched full task backends/models and distinct owned workers required')
    binding = {'proof_path':str(Path(path).resolve()),'proof_sha256':hashlib.sha256(data).hexdigest(),'collections':collections,'protected_binaries':{
        'target/release/toklet':'5250694e814ea112148a52cfdeaf5bc6286ff2804b3acfd8f89e80bc73122a59',
        'target/release/examples/cpu_bench':'e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea'}}
    check_idle(binding)
    return binding


def running_containers(api, record):
    commands = [[*api, 'ps', '--format=json'],
                [PODMAN, 'machine', 'ssh', lifecycle.MACHINE, 'podman', 'ps', '--format=json']]
    for argv in commands:
        item = boundary.capture(argv, 20, 262144)
        record['commands'].append(item)
        if (not item.get('failure') and item['exit_code'] == 0 and not item['wall_timed_out']
                and not item['output_limit_exceeded'] and item['cleanup_errors'] == []
                and item['surviving_group_members'] == []):
            inventory = json.loads(boundary.output(item))
            if not isinstance(inventory, list):
                raise ValueError('Running container inventory is not a list')
            return inventory
    raise ValueError('Running container inventory unavailable; machine retained')


def run(args,*,collection_admission=None,idle_admission=None,extra_source_paths=(),extra_grading_modules=()):
    global PREFIX, IDLE
    collection_validator=(collection_admission or admission).validate_collection
    idle_validator=idle_admission or admit_idle
    PREFIX = args.prefix
    IDLE = args.idle_proof.resolve()
    if not PREFIX.startswith("2026-10-14-backend-task-") or not PREFIX.replace("-", "").isalnum():
        raise ValueError("Fresh controlled grading prefix is required")
    attempt = ROOT / 'benchmarks/raw' / (PREFIX + '-window.json')
    recovery = ROOT / 'benchmarks/raw' / (PREFIX + '-restoration.json')
    output = ROOT / 'benchmarks/raw' / (PREFIX + '-probe.json')
    if any(path.exists() for path in (attempt, recovery, output)):
        raise ValueError('Fresh authored literal window paths are required')
    record = {'format': 'toklet-backend-task-native-window-v1', 'completed': False,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'producer_sha256': boundary.digest(__file__), 'commands': [],
              'started_by_this_task': False, 'restored_stopped': False,
              'pipe_lifetime_finished': False,
              'window_source_sha256': {str(path.relative_to(ROOT)): boundary.digest(path) for path in
                                      (Path(__file__).resolve(),
                                       ROOT / 'tools/bench/podman_machine_lifecycle.py',
                                       ROOT/'tools/experiments/task_grading.py', ROOT/'tools/experiments/task_grade_admission.py',*extra_source_paths)}}
    owner = None
    api = None

    def command(argv, seconds=120):
        item = boundary.capture(argv, seconds, 262144)
        record['commands'].append(item)
        if (item.get('failure') or item['exit_code'] != 0 or item['wall_timed_out']
                or item['output_limit_exceeded'] or item['cleanup_errors']
                or item['surviving_group_members'] != []):
            raise ValueError('Authored literal window control failed: ' + str(argv[:6]))
        return boundary.output(item)

    idle_binding = None

    def idle():
        nonlocal idle_binding
        if idle_binding is None:
            idle_binding = idle_validator(IDLE, args.collection)
            record['idle_collections'] = idle_binding['collections']
        check_idle(idle_binding)
        return idle_binding['proof_sha256']

    def observe():
        item = boundary.capture([PODMAN, 'machine', 'list', '--format=json'], 20, 262144)
        record['last_restoration_inventory'] = item
        if (item.get('failure') or item['exit_code'] != 0 or item['wall_timed_out']
                or item['output_limit_exceeded'] or item['cleanup_errors']
                or item['surviving_group_members'] != []):
            raise ValueError('Owned machine restoration inventory unavailable')
        return json.loads(boundary.output(item))

    def selected(inventory):
        matches = [item for item in inventory if item.get('Name') == lifecycle.MACHINE]
        if len(matches) != 1:
            raise ValueError('Unique owned machine inventory is required')
        return matches[0]

    try:
        try:
            record['idle_sha256'] = idle()
            literal_path=ROOT/'benchmarks/results/2026-10-11-literal-native-v3-probe.json'
            literal=pinned_record(literal_path, LITERAL_SHA)
            if (literal['completed'] is not True or len(literal['runs']) != 28
                    or any(item.get('expectation_passed') is not True for item in literal['runs'])):
                raise ValueError('Accepted literal controller qualification is required')
            for name, expected in literal['source_sha256'].items():
                if boundary.digest(ROOT/name) != expected:
                    raise ValueError('Literal controller qualification source differs: '+name)
            record['literal_qualification_sha256']=LITERAL_SHA
            collection=collection_validator(args.collection)
            record['collection_path']=str(args.collection)
            record['collection_sha256']=collection['collection_sha256']
            probe.check_collection(args.collection, collection)
            record['identity']=collection['identity']
            prior = pinned_record(PRIOR, PRIOR_SHA)
            if prior['completed'] is not True or len(prior['runs']) != 14:
                raise ValueError('Qualified authored runtime receipt is required')
            for item in prior['runs']:
                boundary.check_record(item)
            record['prior_qualification_sha256'] = PRIOR_SHA
            record['admitted_runtime'] = prior['admitted_runtime']
            record['admitted_python_version'] = prior['admitted_python_version']
            record['podman_sha256'] = boundary.digest(PODMAN)
            record['host_python_sha256'] = boundary.digest(Path(sys.executable).resolve())
            before = json.loads(command([PODMAN, 'machine', 'list', '--format=json']))
            record['machine_before'] = before
            machine = selected(before)
            if machine['Running'] is not False or machine['Starting'] is not False:
                raise ValueError('Qualification requires an initially stopped machine')
            machines = json.loads(command([PODMAN, 'machine', 'inspect', lifecycle.MACHINE]))
            connections = json.loads(command([PODMAN, 'system', 'connection', 'list', '--format=json']))
            if len(machines) != 1:
                raise ValueError('Unique machine inspection is required')
            binding = boundary.bound_machine_connection(machines[0], connections)
            if binding != prior['connection_binding']:
                raise ValueError('Qualified machine/API identity changed')
            record['connection_binding'] = binding
            api = (PODMAN, '--url', binding['URI'], '--identity', binding['Identity'])
            probe.check_collection(args.collection, collection)
            idle()
            record['started_by_this_task'] = True
            startup, owner = lifecycle.start_machine(PODMAN)
            record['startup'] = startup
            if not startup['client_succeeded'] or startup.get('helper_capture_failure'):
                raise ValueError('Owned machine startup failed')
            deadline = time.monotonic() + 60
            while True:
                ready = boundary.capture([*api, 'info', '--format=json'], 20, 262144)
                record['commands'].append(ready)
                if (not ready.get('failure') and ready['exit_code'] == 0
                        and not ready['wall_timed_out'] and not ready['output_limit_exceeded']
                        and ready['cleanup_errors'] == [] and ready['surviving_group_members'] == []):
                    record['api_readiness'] = json.loads(boundary.output(ready))
                    break
                if time.monotonic() >= deadline:
                    raise ValueError('Owned machine API readiness failed')
                time.sleep(2)
            image = json.loads(command([*api, 'image', 'inspect', 'sha256:' + IMAGE]))
            if (len(image) != 1 or boundary.normalized_id(image[0].get('Id')) != IMAGE
                    or image[0].get('Architecture') != 'arm64' or image[0].get('Os') != 'linux'):
                raise ValueError('Pinned authored Linux image differs')
            record['image'] = {key: image[0].get(key) for key in
                               ('Id', 'Digest', 'RepoDigests', 'Architecture', 'Os', 'RootFS')}
            if record['image'] != prior['image']:
                raise ValueError('Qualified authored image metadata changed')
            idle()
            result = probe.run(api, IMAGE, record['admitted_runtime'],
                               record['admitted_python_version'], args.collection, output,
                               collection_validator=collection_validator,extra_identity_modules=extra_grading_modules)
            record['probe_path'] = str(output.relative_to(ROOT))
            record['probe_sha256'] = boundary.digest(output)
            record['quality_completed']=result.get('quality_completed',False)
            record['unresolved_case_ids']=result.get('unresolved_case_ids')
            if result['completed'] is not True:
                raise ValueError('Held-out grading attempt failed; raw evidence retained')
            if record['podman_sha256'] != boundary.digest(PODMAN):
                raise ValueError('Podman client changed')
            if record['host_python_sha256'] != boundary.digest(Path(sys.executable).resolve()):
                raise ValueError('Host interpreter changed')
            final = json.loads(command([*api, 'image', 'inspect', 'sha256:' + IMAGE]))
            if len(final) != 1 or record['image'] != {key: final[0].get(key) for key in record['image']}:
                raise ValueError('Authored image identity changed')
            machines = json.loads(command([PODMAN, 'machine', 'inspect', lifecycle.MACHINE]))
            connections = json.loads(command([PODMAN, 'system', 'connection', 'list', '--format=json']))
            if len(machines) != 1 or boundary.bound_machine_connection(machines[0], connections) != binding:
                raise ValueError('Owned machine/API connection changed')
            idle()
            record['completed'] = True
        except BaseException as error:
            record['failure'] = {'type': type(error).__name__, 'message': str(error)[:4096]}
        finally:
            if record['started_by_this_task']:
                try:
                    inventory = json.loads(command([PODMAN, 'machine', 'list', '--format=json']))
                    machine = selected(inventory)
                    if machine['Running'] is True or machine['Starting'] is True:
                        # Never stop a VM containing an unexplained live container.
                        listing = running_containers(api, record)
                        record['running_containers_before_stop'] = listing
                        if listing != []:
                            raise ValueError('Machine retained: running container ownership needs resolution')
                        command([PODMAN, 'machine', 'stop', lifecycle.MACHINE])
                    after = json.loads(command([PODMAN, 'machine', 'list', '--format=json']))
                    machine = selected(after)
                    if machine['Running'] is not False or machine['Starting'] is not False:
                        raise ValueError('Owned machine shutdown is unverified')
                    record['machine_after'] = after
                    record['restored_stopped'] = True
                    if owner is not None:
                        owner.finish(after)
                        record['pipe_lifetime_finished'] = True
                        if record['startup'].get('helper_capture_failure'):
                            raise ValueError('Owned helper pipe capture failed')
                except BaseException as error:
                    record['restoration_failure'] = {'type': type(error).__name__, 'message': str(error)[:4096]}
            if record['completed'] and record['restored_stopped']:
                try:
                    pinned_record(literal_path, LITERAL_SHA)
                    pinned_record(PRIOR, PRIOR_SHA)
                    if record['window_source_sha256'] != {
                            name: boundary.digest(ROOT / name) for name in record['window_source_sha256']}:
                        raise ValueError('Window or lifecycle source changed')
                    if record['idle_sha256'] != idle():
                        raise ValueError('Task collection idle receipt changed')
                except BaseException as error:
                    record['completed'] = False
                    record['failure'] = {'type': type(error).__name__, 'message': str(error)[:4096]}
            record['finished_utc'] = datetime.now(timezone.utc).isoformat()
            with attempt.open('x') as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.write('\n')
    finally:
        needed = owner is not None and not record['pipe_lifetime_finished']
        lifecycle.await_stopped_owner(owner, observe)
        if needed:
            restored = {'format': 'toklet-backend-task-helper-restoration-v1',
                        'verified_utc': datetime.now(timezone.utc).isoformat(),
                        'attempt_sha256': boundary.digest(attempt) if attempt.exists() else None,
                        'machine_inventory': record.get('last_restoration_inventory'),
                        'startup_final': record['startup'], 'helper_pipe_lifetime_completed': True}
            with recovery.open('x') as stream:
                json.dump(restored, stream, indent=2, allow_nan=False)
                stream.write('\n')
    return 0 if (record['completed'] and record['restored_stopped']
                 and record['pipe_lifetime_finished'] and not record.get('restoration_failure')) else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--idle-proof',type=Path,required=True)
    parser.add_argument('--prefix',required=True)
    sys.exit(run(parser.parse_args()))
