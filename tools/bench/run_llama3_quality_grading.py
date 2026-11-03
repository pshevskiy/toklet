"""Sealed outer owner: accept grading only after independent stopped/exit checks."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import llama3_quality_grading as adapter
import paired_task_grading as runner
import grading_support as support
import linux_code_isolation_probe as boundary
ROOT=adapter.ROOT


def group_alive(pid):
    try:os.kill(-pid,0)
    except ProcessLookupError:return False
    return True


def acceptable(window,inner):
    return (window.get('exit_code')==0 and window.get('primary_error')is None
            and window.get('cleanup_errors')==[]and window.get('independent_stopped_verification')is True
            and inner.get('completed')is True and not inner.get('failure')and not inner.get('restoration_failure')
            and inner.get('restored_stopped')is True and inner.get('pipe_lifetime_finished')is True
            and not inner.get('unresolved_case_ids'))


def read_inner_result(path):
    # An80-program transport/evidence receipt exceeds the journal's8MiB cap.
    inner,raw=adapter.results.document(path,64*1024**2)
    pin=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    support.verify_file(Path(path),pin)
    return inner,pin


def run(collection_file,collection_sha,input_folder,manifest_sha,output,timeout=3600):
    output=Path(output).resolve();support.require(output.is_relative_to(ROOT/'artifacts'),'fresh outer artifact folder required');output.mkdir()
    bindings={};record=dict(completed=False,generated_code_runs_on_host=False)
    adapter.bind(bindings,collection_file,collection_sha)
    collection,_=adapter.results.document(collection_file);adapter.require_idle(collection)
    adapter.bind(bindings,Path(input_folder)/'manifest.json',manifest_sha)
    manifest,_=adapter.inputs.load(input_folder,manifest_sha)
    adapter.bind(bindings,Path(input_folder)/'execution.jsonl',manifest['tasks']['sha256'])
    for path in [Path(__file__),Path(adapter.__file__),Path(runner.__file__),Path(support.__file__),
                 ROOT/'tools/bench/llama3_task_results.py',ROOT/'tools/bench/task_cases.py',
                 ROOT/'tools/bench/linux_task_controller.py',ROOT/'tools/bench/linux_task_evidence.py',
                 ROOT/'tools/bench/linux_code_isolation_probe.py',ROOT/'tools/bench/podman_machine_lifecycle.py',
                 ROOT/'tools/bench/held_out_task_window.py',runner.PRIOR,runner.LITERAL]:adapter.bind(bindings,path)
    prior=runner.checked_json(runner.PRIOR,runner.PRIOR_SHA,16*1024**2)
    client_pin=dict(bytes=Path(runner.PODMAN).stat().st_size,sha256=prior['podman_sha256'])
    def guard():
        support.verify_file(Path(runner.PODMAN),client_pin)
        for name,pin in bindings.items():support.verify_file(ROOT/name,pin)
        adapter.require_idle(collection)
    def observe():
        item=boundary.capture([runner.PODMAN,'machine','list','--format=json'],20,262144)
        support.require(not item.get('failure')and item['exit_code']==0 and not item['wall_timed_out']and not item['output_limit_exceeded']
                        and item['cleanup_errors']==[]and item['surviving_group_members']==[],'outer machine query failed')
        return json.loads(boundary.output(item))
    command=[sys.executable,str(ROOT/'tools/bench/llama3_quality_grading.py'),str(collection_file),collection_sha,str(input_folder),manifest_sha,str(output/'inner')]
    # Podman renders BuiltTime in local time; retain exact qualified JSON.
    child_env=dict(os.environ, TZ='Europe/Moscow')
    plan=dict(timezone=child_env['TZ'],bindings=bindings,podman_client_pin=client_pin,command=command,collection_sha256=collection_sha,manifest_sha256=manifest_sha,
              initial_state='existing stopped VM; both model processes/groups absent',policy='TERM only; retain lifetime owner until group exit and independently stopped VM')
    support.write_json(output/'execution-plan.json',plan)
    try:
        guard()
        window=support.supervise_window(command,output/'child.stdout',output/'child.stderr',child_env,observe,group_alive,timeout)
        record['window']=window;guard()
        pid=window.get('pid')
        support.require(type(pid)is int and pid>1 and not group_alive(pid),'grading worker group exit unverified')
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise ValueError('grading worker remains live')
        inner_path=output/'inner/result.json';inner,inner_pin=read_inner_result(inner_path)
        support.require(acceptable(window,inner),'outer or inner restoration/decision guard failed')
        adapter.bind(bindings,inner_path,inner_pin['sha256'])
        record.update(completed=True,decision=inner['decision'],inner_result_sha256=bindings[str(inner_path.relative_to(ROOT))]['sha256'],
                      independent_group_absence_verified=True,plan_sha256=hashlib.sha256((output/'execution-plan.json').read_bytes()).hexdigest())
    except BaseException as error:record['failure']=dict(type=type(error).__name__,message=str(error)[:4096])
    record['bindings']=bindings;support.write_json(output/'result.json',record);return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection',type=Path);parser.add_argument('collection_sha')
    parser.add_argument('inputs',type=Path);parser.add_argument('manifest_sha');parser.add_argument('new_output',type=Path)
    args=parser.parse_args();result=run(args.collection,args.collection_sha,args.inputs,args.manifest_sha,args.new_output)
    print(json.dumps({k:v for k,v in result.items()if k!='bindings'},indent=2));raise SystemExit(0 if result['completed']else 1)
