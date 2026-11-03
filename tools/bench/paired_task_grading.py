"""Shared qualified isolated runner for two complete literal task collections.

Model-specific collection/decoding stays in adapters. This runner owns VM/client/
image/bootstrap admission, all80 program runs, evidence classification, shutdown,
and the final guarded publication. Generated code never executes on the host.
"""
from pathlib import Path
import hashlib
import json
import signal
import time
import task_cases
import linux_code_isolation_probe as boundary
import linux_task_controller as controller
import linux_task_evidence as evidence
import podman_machine_lifecycle as lifecycle
from held_out_task_window import running_containers
from grading_support import require,selected,write_json,finish_decision,verify_file
ROOT=Path(__file__).resolve().parents[2]
PODMAN='/opt/podman/bin/podman'
IMAGE='eb7d16fec874f15b584063275012ac719885a2b87574d124c0da9c156d504629'
PRIOR=ROOT/'benchmarks/results/2026-10-11-linux-code-isolation-probe-v8.json'
PRIOR_SHA='0736ec7d8cf7481c5327c01c5f0c8555b6e3c8669c6c63e7a267f0044873d3f1'
LITERAL=ROOT/'benchmarks/results/2026-10-11-literal-native-v3-probe.json'
LITERAL_SHA='77cdda26d80a079bc56da9a80d52f7fa09d7b2942ce768d464128f07798a9453'


def checked_json(path,expected,limit):
    # These fixed qualification records are immutable repository evidence.
    size=path.lstat().st_size;require(size<=limit,'oversized qualification evidence')
    verify_file(path,dict(bytes=size,sha256=expected))
    with path.open('rb')as stream:raw=stream.read(limit+1)
    require(len(raw)==size and hashlib.sha256(raw).hexdigest()==expected,'qualification changed while reading')
    return json.loads(raw)


def grade_paired_cases(cases,decoded,guard,output,record,bindings,decision):
    output=Path(output).resolve();require(output.is_relative_to(ROOT/'artifacts')and output.is_dir(),'owned artifact output required')
    record.update(started_by_this_task=False,restored_stopped=False,pipe_lifetime_finished=False,completed=False)
    owner=None;api=None;pending_decision=None;old={}
    def command(argv,seconds=120):
        item=boundary.capture(argv,seconds,262144);record['commands'].append(item)
        require(not item.get('failure')and item['exit_code']==0 and not item['wall_timed_out']
                and not item['output_limit_exceeded']and item['cleanup_errors']==[]and item['surviving_group_members']==[],
                'qualified bounded VM control failed')
        return boundary.output(item)
    def observe():return json.loads(command([PODMAN,'machine','list','--format=json'],20))
    def stop(number,_):raise InterruptedError(str(number))
    try:
        require(len(cases)==100 and set(decoded)=={'baseline','native'},'complete paired task collections required')
        for case in cases:task_cases.validate_case(case)
        require(len({case['id']for case in cases})==100,'duplicate task cases')
        for profile in ['baseline','native']:
            require(len(decoded[profile])==100,'all100 decoded responses required')
            for case,response in zip(cases,decoded[profile],strict=True):
                require(response['id']==case['id']and response['category']==case['category']and isinstance(response['visible_text'],str)
                        and len(response['visible_text'].encode())<=65536
                        and response['visible_text_sha256']==hashlib.sha256(response['visible_text'].encode()).hexdigest(),
                        'literal completion identity differs')
        guard()
        prior = checked_json(PRIOR,PRIOR_SHA,16*1024**2)
        literal = checked_json(LITERAL,LITERAL_SHA,16*1024**2)
        require(prior['completed'] is True and len(prior['runs'])==14 and literal['completed'] is True
            and len(literal['runs'])==28 and all(v['expectation_passed'] for v in literal['runs']), 'existing isolated evaluator qualification required')
        for v in prior['runs']:boundary.check_record(v)
        for proof in (prior,literal):
            for name,sha in proof['source_sha256'].items():
                require(boundary.digest(ROOT/name)==sha, 'qualified evaluator source changed')
                bindings[name] = dict(bytes=(ROOT/name).stat().st_size,sha256=sha)
        record['qualification'] = dict(prior_sha256=PRIOR_SHA,literal_sha256=LITERAL_SHA,
            runtime=prior['admitted_runtime'],python_version=prior['admitted_python_version'])
        record['podman_sha256'] = boundary.digest(PODMAN)
        require(record['podman_sha256']==prior['podman_sha256'], 'qualified Podman client binary changed')
        before = observe();record['machine_before'] = before;machine = selected(before)
        require(machine.get('Running') is False and machine.get('Starting') is False, 'grader requires initially stopped existing VM')
        require(machine.get('Memory') in ('2147483648',2147483648), 'existing grader VM must retain its2GiB memory reservation')
        machines = json.loads(command([PODMAN,'machine','inspect',lifecycle.MACHINE]))
        connections = json.loads(command([PODMAN,'system','connection','list','--format=json']))
        require(len(machines)==1, 'unique machine inspect required')
        binding = boundary.bound_machine_connection(machines[0],connections)
        require(binding==prior['connection_binding'], 'qualified existing VM/API identity changed')
        record['connection_binding'] = binding;api = (PODMAN,'--url',binding['URI'],'--identity',binding['Identity'])
        guard();old = {s:signal.signal(s,stop)for s in(signal.SIGINT,signal.SIGTERM)}
        record['started_by_this_task'] = True
        startup,owner = lifecycle.start_machine(PODMAN);record['startup'] = startup
        require(startup['client_succeeded'] and not startup.get('helper_capture_failure'), 'owned VM startup failed')
        deadline = time.monotonic()+60
        while True:
            item = boundary.capture([*api,'info','--format=json'],20,262144);record['commands'].append(item)
            if not item.get('failure') and item['exit_code']==0 and not item['wall_timed_out'] and not item['output_limit_exceeded'] and not item['cleanup_errors'] and not item['surviving_group_members']:break
            require(time.monotonic()<deadline, 'owned VM API readiness failed');time.sleep(2)
        actual_version = json.loads(command([*api,'version','--format=json']))
        require(actual_version==json.loads(prior['podman_version']), 'qualified client/server version changed')
        record['podman_version'] = actual_version
        image = json.loads(command([*api,'image','inspect','sha256:'+IMAGE]))
        require(len(image)==1 and boundary.normalized_id(image[0].get('Id'))==IMAGE
            and image[0].get('Architecture')=='arm64' and image[0].get('Os')=='linux', 'qualified Linux image required')
        observed_image = {k:image[0].get(k)for k in ('Id','Digest','RepoDigests','Architecture','Os','RootFS')}
        require(observed_image==prior['image'], 'qualified image metadata changed');record['image'] = observed_image
        require(running_containers(api,record)==[], 'unexplained existing running container')
        trusted = controller.bundled_source();controller.admit_bootstrap(trusted)
        with (output/'grading.jsonl').open('x') as journal:
            for profile in ('baseline','native'):
                outcomes = [];record['outcomes'][profile] = outcomes
                for case,response in zip(cases,decoded[profile],strict=True):
                    if case['category']!='code':
                        passed=task_cases.grade_text(case,response['visible_text'])
                        outcome=dict(id=case['id'],category=case['category'],passed=passed,status='model_pass'if passed else'model_fail')
                    else:
                        program=task_cases.human_eval_program(case,response['visible_text'])
                        payload=dict(case_id=case['id'],program=program,program_sha256=hashlib.sha256(program.encode()).hexdigest())
                        if 'program_payload'in response:require(response['program_payload']==payload,'literal program adapter changed source')
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
                task_cases.aggregate(cases,outcomes)
        require(len(record['runs'])==80, 'all80 isolated code cases required')
        pending_decision = decision(record['outcomes'])
        final = json.loads(command([*api,'image','inspect','sha256:'+IMAGE]))
        require(len(final)==1 and observed_image=={k:final[0].get(k)for k in observed_image}, 'graded image changed')
        machines = json.loads(command([PODMAN,'machine','inspect',lifecycle.MACHINE]));connections = json.loads(command([PODMAN,'system','connection','list','--format=json']))
        require(len(machines)==1 and boundary.bound_machine_connection(machines[0],connections)==binding, 'owned VM/API identity changed')
        require(record['podman_sha256']==boundary.digest(PODMAN), 'Podman client changed')
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
                    require(listing==[], 'machine retained: unexplained live container')
                    command([PODMAN,'machine','stop',lifecycle.MACHINE])
                after = observe();machine = selected(after)
                require(machine['Running'] is False and machine['Starting'] is False, 'owned VM shutdown unverified')
                record['machine_after'] = after;record['restored_stopped'] = True
                if owner is not None:
                    owner.finish(after);record['pipe_lifetime_finished'] = True
                    require(not record['startup'].get('helper_capture_failure'), 'VM helper capture failed')
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
