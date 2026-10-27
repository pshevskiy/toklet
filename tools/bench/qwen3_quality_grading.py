"""Qwen3 adapter around the existing qualified Linux literal evaluator/window.

Model processes must finish first. No containment, source-test construction,
text scorer, resource policy or unresolved-case aggregation rule changes.
"""
from pathlib import Path
import hashlib
import json
import os
import signal
import stat
import subprocess
import time

import qwen3_quality_results as results
import linux_code_isolation_probe as boundary
import linux_task_controller as controller
import linux_task_evidence as evidence
import podman_machine_lifecycle as lifecycle
from held_out_task_window import running_containers

ROOT = results.contract.ROOT
PODMAN = '/opt/podman/bin/podman'
IMAGE = 'eb7d16fec874f15b584063275012ac719885a2b87574d124c0da9c156d504629'
CONTRACT_SHA = '908e037de5b399d9e9a8dd09aa9fcc568f67abb714b06508672deac6bb8ac529'
PRIOR = ROOT/'benchmarks/results/2026-10-11-linux-code-isolation-probe-v8.json'
PRIOR_SHA = '0736ec7d8cf7481c5327c01c5f0c8555b6e3c8669c6c63e7a267f0044873d3f1'
LITERAL = ROOT/'benchmarks/results/2026-10-11-literal-native-v3-probe.json'
LITERAL_SHA = '77cdda26d80a079bc56da9a80d52f7fa09d7b2942ce768d464128f07798a9453'


def verify_file(path, expected):
    def fields(v):return v.st_dev,v.st_ino,v.st_mode,v.st_size,v.st_mtime_ns,v.st_ctime_ns
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as file:
        before = os.fstat(file.fileno());digest = hashlib.sha256()
        size = expected['bytes']
        results.require(type(size) is int and 0<=size<=4*1024**3
            and stat.S_ISREG(before.st_mode) and before.st_size==size, 'regular admitted evidence file required')
        remaining = size
        while remaining:
            body = file.read(min(1024**2,remaining))
            results.require(body, 'admitted evidence truncated during hashing')
            remaining -= len(body);digest.update(body)
        results.require(not file.read(1) and digest.hexdigest()==expected['sha256'] and fields(os.fstat(file.fileno()))==fields(before)
            and fields(Path(path).lstat())==fields(before), 'collection/source evidence changed')


def require_idle(record):
    results.require(record.get('passed') is True and record.get('primary_error') is None
        and record.get('endpoint_errors')==[] and record.get('observations',{}).get('scope',{}).get('full_model_collections_complete') is True,
        'complete successful model collections required before grading')
    commands = record.get('commands')
    results.require(isinstance(commands,list) and len(commands)==7, 'all seven owned collection phases required')
    pids = []
    for command in commands:
        pid = command.get('pid')
        results.require(type(pid) is int and pid>1 and pid not in pids and command.get('exit_code')==0
            and command.get('primary_error') is None and command.get('cleanup_errors')==[], 'natural owned collection exit required')
        for target in (pid,-pid):
            try:os.kill(target,0)
            except ProcessLookupError:pass
            else:raise ValueError('model process or owned group remains live')
        pids.append(pid)
    return pids


def selected(inventory):
    matches = [v for v in inventory if v.get('Name')==lifecycle.MACHINE]
    results.require(len(matches)==1, 'unique existing grader machine required')
    return matches[0]


def write_json(path, value):
    with Path(path).open('x') as file:json.dump(value,file,indent=2,ensure_ascii=False,allow_nan=False);file.write('\n')


def finish_decision(record, pending, guard):
    if (pending is None or record.get('failure') or record.get('restoration_failure')
            or record.get('restored_stopped') is not True or record.get('pipe_lifetime_finished') is not True):
        return
    try:
        guard();record['decision'] = pending;record['completed'] = True
    except BaseException as error:record['failure'] = dict(type=type(error).__name__,message=str(error)[:4096])


def supervise_window(command, stdout, stderr, env, observe, group_alive, timeout=3600):
    """Retain the lifetime owner through graceful cancellation and VM shutdown.

    The deadline requests SIGTERM, never SIGKILL. Continue waiting for the whole
    worker group, including cleanup after the direct time-wrapper process exits.
    Independently verify the original stopped VM state after the owner exits.
    """
    record = dict(command=command,exit_code=None,pid=None,primary_error=None,cleanup_errors=[],
        signals=[],lifetime_policy='graceful TERM only; retain owner until group exit; independently verify VM stopped')
    process = None;old = {}
    try:
        before = observe();record['machine_before'] = before;machine = selected(before)
        results.require(machine.get('Running') is False and machine.get('Starting') is False, 'outer window requires stopped VM')
        with Path(stdout).open('xb') as out,Path(stderr).open('xb') as err:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
            record['pid'] = process.pid
            try:
                record['exit_code'] = process.wait(timeout=timeout)
                if record['exit_code']!=0:record['primary_error'] = 'tool exit '+str(record['exit_code'])
            except BaseException as error:
                record['primary_error'] = type(error).__name__+': '+str(error)
                old = {s:signal.signal(s,signal.SIG_IGN)for s in(signal.SIGINT,signal.SIGTERM)}
                try:os.killpg(process.pid,signal.SIGTERM);record['signals'].append('SIGTERM')
                except ProcessLookupError:pass
            finally:
                # No bounded forced-drain helper here: it could kill the VM
                # lifetime owner while its restoration finally block is active.
                if not old:old = {s:signal.signal(s,signal.SIG_IGN)for s in(signal.SIGINT,signal.SIGTERM)}
                while process.poll() is None or group_alive(process.pid):time.sleep(0.1)
                record['exit_code'] = process.wait()
    except BaseException as error:
        if record['primary_error'] is None:record['primary_error'] = type(error).__name__+': '+str(error)
    finally:
        try:
            after = observe();record['machine_after'] = after;machine = selected(after)
            results.require(machine.get('Running') is False and machine.get('Starting') is False,
                            'outer VM restoration is unverified')
            record['independent_stopped_verification'] = True
        except BaseException as error:record['cleanup_errors'].append(type(error).__name__+': '+str(error))
        for sig,handler in old.items():signal.signal(sig,handler)
        record.update(stdout=str(Path(stdout).relative_to(ROOT)),stderr=str(Path(stderr).relative_to(ROOT)))
    return record


def run(checks_path, checks_sha, output, endpoint):
    checks_path = Path(checks_path).resolve();output = Path(output).resolve()
    results.require(output.is_relative_to(ROOT/'artifacts'), 'fresh artifact grading destination required')
    output.mkdir()
    record = dict(format='toklet-qwen3-full-quality-grading-v1',completed=False,commands=[],runs=[],outcomes={},
        started_by_this_task=False,restored_stopped=False,pipe_lifetime_finished=False,
        collection_sha256=checks_sha,generated_code_runs_on_host=False,unresolved_case_ids=[],
        scope='Literal source-test outcomes, no adversarial shared-interpreter grading attestation or timing acceptance')
    owner = None;api = None;pending_decision = None;bindings = {};decoded = {};collections = {};cases = []
    old = {}

    def command(argv, seconds=120):
        item = boundary.capture(argv,seconds,262144);record['commands'].append(item)
        results.require(not item.get('failure') and item['exit_code']==0 and not item['wall_timed_out']
            and not item['output_limit_exceeded'] and item['cleanup_errors']==[] and item['surviving_group_members']==[],
            'qualified bounded VM control failed')
        return boundary.output(item)

    def observe():
        return json.loads(command([PODMAN,'machine','list','--format=json'],20))

    def guard():
        for name,expected in bindings.items():verify_file(ROOT/name,expected)
        require_idle(collection)
        results.require(boundary.digest(checks_path)==checks_sha, 'collection receipt changed')
        endpoint()

    def stop(number,_):raise InterruptedError(str(number))
    try:
        collection = results.contract.safe.document(results.contract.checked(checks_path,checks_sha,4*1024**2))
        record['exited_owned_model_pids'] = require_idle(collection)
        bindings = {**collection['bindings'],**collection['outputs']}
        m = results.contract.safe.document(results.contract.checked(ROOT/'benchmarks/qwen3-quality-v1/manifest.json',CONTRACT_SHA,2*1024**2))
        for name,sha in m['candidate']['numerical_source_sha256'].items():
            results.require(bindings[name]['sha256']==sha, 'collected numerical candidate differs from frozen contract')
        guard()
        inputs = ROOT/'artifacts/m8-qwen3-quality-inputs-v1/inputs'
        payload = results.contract.checked(inputs/'tasks.jsonl',m['files']['tasks.jsonl']['sha256'],4*1024**2)
        cases = [results.contract.safe.document(line) for line in payload.splitlines()]
        loss_raw = results.contract.checked(inputs/'held_out_loss.u32le',m['files']['held_out_loss.u32le']['sha256'],40004)
        import struct
        ids = list(struct.unpack('<10001I',loss_raw))
        for profile in ('baseline','native'):
            prefix = 'source' if profile=='baseline' else 'native'
            def rows(mode):
                path = checks_path.parent/(prefix+'-'+mode+('.jsonl'if profile=='baseline'else'.stdout'))
                return results.events(path,collection['outputs'][str(path.relative_to(ROOT))]['sha256'])
            collections[profile] = results.loss(rows('loss'),profile,m,ids)
            responses = results.tasks(rows('tasks'),profile,m,cases)
            decoded[profile] = results.decode(cases,responses)
            write_json(output/(profile+'-decoded.json'),decoded[profile])
        record['losses'] = collections
        prior = results.contract.safe.document(results.contract.checked(PRIOR,PRIOR_SHA,16*1024**2))
        literal = results.contract.safe.document(results.contract.checked(LITERAL,LITERAL_SHA,16*1024**2))
        results.require(prior['completed'] is True and len(prior['runs'])==14 and literal['completed'] is True
            and len(literal['runs'])==28 and all(v['expectation_passed'] for v in literal['runs']), 'existing isolated evaluator qualification required')
        for v in prior['runs']:boundary.check_record(v)
        for proof in (prior,literal):
            for name,sha in proof['source_sha256'].items():
                results.require(boundary.digest(ROOT/name)==sha, 'qualified evaluator source changed')
                bindings[name] = dict(bytes=(ROOT/name).stat().st_size,sha256=sha)
        record['qualification'] = dict(prior_sha256=PRIOR_SHA,literal_sha256=LITERAL_SHA,
            runtime=prior['admitted_runtime'],python_version=prior['admitted_python_version'])
        record['podman_sha256'] = boundary.digest(PODMAN)
        results.require(record['podman_sha256']==prior['podman_sha256'], 'qualified Podman client binary changed')
        before = observe();record['machine_before'] = before;machine = selected(before)
        results.require(machine.get('Running') is False and machine.get('Starting') is False, 'grader requires initially stopped existing VM')
        results.require(machine.get('Memory') in ('2147483648',2147483648), 'existing grader VM must retain its2GiB memory reservation')
        machines = json.loads(command([PODMAN,'machine','inspect',lifecycle.MACHINE]))
        connections = json.loads(command([PODMAN,'system','connection','list','--format=json']))
        results.require(len(machines)==1, 'unique machine inspect required')
        binding = boundary.bound_machine_connection(machines[0],connections)
        results.require(binding==prior['connection_binding'], 'qualified existing VM/API identity changed')
        record['connection_binding'] = binding;api = (PODMAN,'--url',binding['URI'],'--identity',binding['Identity'])
        guard();old = {s:signal.signal(s,stop)for s in(signal.SIGINT,signal.SIGTERM)}
        record['started_by_this_task'] = True
        startup,owner = lifecycle.start_machine(PODMAN);record['startup'] = startup
        results.require(startup['client_succeeded'] and not startup.get('helper_capture_failure'), 'owned VM startup failed')
        deadline = time.monotonic()+60
        while True:
            item = boundary.capture([*api,'info','--format=json'],20,262144);record['commands'].append(item)
            if not item.get('failure') and item['exit_code']==0 and not item['wall_timed_out'] and not item['output_limit_exceeded'] and not item['cleanup_errors'] and not item['surviving_group_members']:break
            results.require(time.monotonic()<deadline, 'owned VM API readiness failed');time.sleep(2)
        actual_version = json.loads(command([*api,'version','--format=json']))
        results.require(actual_version==json.loads(prior['podman_version']), 'qualified client/server version changed')
        record['podman_version'] = actual_version
        image = json.loads(command([*api,'image','inspect','sha256:'+IMAGE]))
        results.require(len(image)==1 and boundary.normalized_id(image[0].get('Id'))==IMAGE
            and image[0].get('Architecture')=='arm64' and image[0].get('Os')=='linux', 'qualified Linux image required')
        observed_image = {k:image[0].get(k)for k in ('Id','Digest','RepoDigests','Architecture','Os','RootFS')}
        results.require(observed_image==prior['image'], 'qualified image metadata changed');record['image'] = observed_image
        results.require(running_containers(api,record)==[], 'unexplained existing running container')
        trusted = controller.bundled_source();controller.admit_bootstrap(trusted)
        with (output/'grading.jsonl').open('x') as journal:
            for profile in ('baseline','native'):
                outcomes = [];record['outcomes'][profile] = outcomes
                for case,response in zip(cases,decoded[profile],strict=True):
                    if case['category']!='code':outcome = response['outcome']
                    else:
                        payload = response['program_payload']
                        expected_program = results.task_cases.human_eval_program(results.legacy_case(case),response['visible_text'])
                        results.require(payload==dict(case_id=case['id'],program=expected_program,
                            program_sha256=hashlib.sha256(expected_program.encode()).hexdigest()), 'literal program adapter changed source')
                        raw = controller.execute(api,IMAGE,trusted,payload)
                        run_record = dict(profile=profile,id=case['id'],raw=raw,program_sha256=payload['program_sha256'],completion_sha256=response['visible_text_sha256'])
                        record['runs'].append(run_record)
                        # Preserve full container/transport evidence before classification.
                        journal.write(json.dumps(run_record,separators=(',',':'),allow_nan=False)+'\n');journal.flush()
                        try:resolved = evidence.classify(raw,payload,prior['admitted_runtime'],prior['admitted_python_version'])
                        except ValueError:
                            record['unresolved_case_ids'].append(dict(profile=profile,id=case['id']))
                            raise
                        passed = resolved['passed'];outcome = dict(id=case['id'],category='code',passed=passed,
                            status='model_pass'if passed else'model_fail',resolved=resolved)
                    outcomes.append(outcome)
                results.task_cases.aggregate([results.legacy_case(c)for c in cases],outcomes)
        results.require(len(record['runs'])==80, 'all80 isolated code cases required')
        pending_decision = results.decision(collections['native'],collections['baseline'],cases,record['outcomes']['native'],record['outcomes']['baseline'])
        final = json.loads(command([*api,'image','inspect','sha256:'+IMAGE]))
        results.require(len(final)==1 and observed_image=={k:final[0].get(k)for k in observed_image}, 'graded image changed')
        machines = json.loads(command([PODMAN,'machine','inspect',lifecycle.MACHINE]));connections = json.loads(command([PODMAN,'system','connection','list','--format=json']))
        results.require(len(machines)==1 and boundary.bound_machine_connection(machines[0],connections)==binding, 'owned VM/API identity changed')
        results.require(record['podman_sha256']==boundary.digest(PODMAN), 'Podman client changed')
        guard()
    except BaseException as error:
        record['failure'] = dict(type=type(error).__name__,message=str(error)[:4096])
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        if record['started_by_this_task']:
            try:
                machines = observe();machine = selected(machines)
                if machine['Running'] is True or machine['Starting'] is True:
                    listing = running_containers(api,record);record['running_containers_before_stop'] = listing
                    results.require(listing==[], 'machine retained: unexplained live container')
                    command([PODMAN,'machine','stop',lifecycle.MACHINE])
                after = observe();machine = selected(after)
                results.require(machine['Running'] is False and machine['Starting'] is False, 'owned VM shutdown unverified')
                record['machine_after'] = after;record['restored_stopped'] = True
                if owner is not None:
                    owner.finish(after);record['pipe_lifetime_finished'] = True
                    results.require(not record['startup'].get('helper_capture_failure'), 'VM helper capture failed')
            except BaseException as error:record['restoration_failure'] = dict(type=type(error).__name__,message=str(error)[:4096])
        finish_decision(record,pending_decision,guard)
        record['source_and_evidence_bindings'] = bindings
        try:write_json(output/'result.json',record)
        finally:
            # Keep inherited VM-helper pipes alive until the existing lifecycle
            # observer verifies shutdown, even if evidence publication fails.
            try:lifecycle.await_stopped_owner(owner,observe)
            finally:
                for sig,handler in old.items():signal.signal(sig,handler)
    return record
