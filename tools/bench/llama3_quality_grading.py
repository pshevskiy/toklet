"""Llama collection/decoding adapter for the shared qualified isolated runner."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import prepare_llama3_tasks as inputs
import llama3_task_results as results
import task_cases
from grading_support import require,verify_file,write_json
import paired_task_grading as isolated
ROOT=inputs.ROOT


def require_idle(record):
    require(record.get('passed')is True and record.get('primary_error')is None
            and record.get('endpoint_errors')==[]
            and record.get('observations',{}).get('scope',{}).get('full_model_collections_complete')is True,
            'complete qualified Llama collections required')
    commands=record.get('commands')
    require(isinstance(commands,list)and len(commands)==2 and [c.get('name')for c in commands]==['native','original'],
            'both named model phases required')
    pids=[]
    for command in commands:
        pid=command.get('pid')
        require(type(pid)is int and pid>1 and pid not in pids and command.get('pgid')==pid
                and command.get('exit_code')==0 and command.get('natural_exit')is True and command.get('reaped')is True
                and command.get('primary_error')is None and command.get('cleanup_errors')==[],
                'natural clean model exits required')
        for target in [pid,-pid]:
            try:os.kill(target,0)
            except ProcessLookupError:pass
            else:raise ValueError('model process or group remains live')
        pids.append(pid)
    return pids


def bind(bindings,path,expected=None):
    path=Path(path).absolute();require(path.resolve().is_relative_to(ROOT)and not path.is_symlink(),'evidence outside repository or symlink')
    size=path.lstat().st_size;require(0<=size<=4*1024**3,'bounded evidence required')
    if expected is None:
        descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(descriptor,'rb')as stream:
            digest=hashlib.sha256();remaining=size
            while remaining:
                body=stream.read(min(1024**2,remaining));require(body,'evidence truncated');digest.update(body);remaining-=len(body)
            require(not stream.read(1),'evidence grew');expected=digest.hexdigest()
    value=dict(bytes=size,sha256=expected)
    verify_file(path,value);bindings[str(path.relative_to(ROOT))]=value


def decode(cases,records,manifest):
    require(importlib.metadata.version('tokenizers')=='0.22.2','pinned tokenizer runtime required')
    from tokenizers import Tokenizer
    assets={}
    for name,sha in manifest['tokenizer_pins'].items():
        descriptor=os.open(inputs.codec_policy.MODEL/name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(descriptor,'rb')as stream:raw=stream.read(16*1024**2+1)
        require(len(raw)<=16*1024**2 and hashlib.sha256(raw).hexdigest()==sha,'codec asset changed')
        assets[name]=raw
    require(inputs.codec_policy.closure()==manifest['preparation_provider_sha256'],'text provider changed')
    tokenizer=Tokenizer.from_buffer(assets['tokenizer.json'])
    require(len(cases)==len(records)==100,'all100 task responses required')
    decoded=[]
    for case,record in zip(cases,records):
        require(case['id']==record['id']and case['category']==record['category'],'decoded task order differs')
        text=tokenizer.decode(record['visible_token_ids'],skip_special_tokens=True)
        require(len(text.encode())<=65536,'decoded completion exceeds transport bound')
        decoded.append(dict(id=case['id'],category=case['category'],visible_token_ids=record['visible_token_ids'],
                            visible_text=text,visible_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    return decoded


def decision(cases,outcomes):
    baseline=task_cases.aggregate(cases,outcomes['baseline']);native=task_cases.aggregate(cases,outcomes['native'])
    return dict(complete=True,baseline_tasks=baseline,native_tasks=native,
                task_pass_rate_loss_percentage_points=baseline['passed']-native['passed'],
                task_limit_percentage_points=2,task_gate_passed=native['passed']+2>=baseline['passed'],
                full_quality_qualified=False,app_discovery_qualified=False,device_qualified=False)


def run(collection_file,collection_sha,input_folder,manifest_sha,output,endpoint=lambda:None):
    output=Path(output).resolve();require(output.is_relative_to(ROOT/'artifacts'),'fresh artifact output required');output.mkdir()
    record=dict(format='toklet-llama3-full-task-grading-v1',completed=False,commands=[],runs=[],outcomes={},
                started_by_this_task=False,restored_stopped=False,pipe_lifetime_finished=False,
                collection_sha256=collection_sha,manifest_sha256=manifest_sha,generated_code_runs_on_host=False,unresolved_case_ids=[],
                scope='Literal source-test outcomes; no adversarial shared-interpreter grading attestation or timing acceptance')
    bindings={}
    def guard():
        for name,pin in bindings.items():verify_file(ROOT/name,pin)
        require_idle(collection);endpoint()
    try:
        bind(bindings,collection_file,collection_sha)
        collection,_=results.document(collection_file);record['exited_owned_model_pids']=require_idle(collection)
        plan_file=Path(collection_file).parent/'execution-plan.json';bind(bindings,plan_file,collection['plan_sha256'])
        plan,_=results.document(plan_file)
        require(plan['manifest_sha256']==manifest_sha,'collection task contract differs')
        for name,sha in plan['source_sha256'].items():bind(bindings,ROOT/name,sha)
        binary=ROOT/'artifacts/new-goal-b0-checks-v1/target/release/examples/llama3_quality_tasks'
        bind(bindings,binary,plan['binary_sha256'])
        manifest,frozen=inputs.load(input_folder,manifest_sha)
        bind(bindings,Path(input_folder)/'manifest.json',manifest_sha);bind(bindings,Path(input_folder)/'execution.jsonl',manifest['tasks']['sha256'])
        cases=task_cases.load_cases();require([c['id']for c in cases]==[c['id']for c in frozen],'grading case order differs')
        bind(bindings,ROOT/manifest['source_tasks']['path'],manifest['source_tasks']['sha256'])
        decoded={}
        for role,profile in [('original','baseline'),('native','native')]:
            item=collection['outputs'][role];folder=(ROOT/item['folder']).resolve()
            require(folder.is_relative_to(ROOT/'artifacts'),'model outputs outside artifact scope')
            bind(bindings,folder/'result.json',item['report_sha256']);bind(bindings,folder/'tasks.jsonl',item['journal_sha256'])
            report,records=results.collection(folder,input_folder,manifest_sha,role,False)
            if role=='original':
                pins=folder/report['runtime_pins']['file'];bind(bindings,pins,report['runtime_pins']['sha256'])
                provider,_=results.document(pins)
                for name,sha in provider.items():bind(bindings,name,sha)
            decoded[profile]=decode(cases,records,manifest)
            write_json(output/(profile+'-decoded.json'),decoded[profile]);bind(bindings,output/(profile+'-decoded.json'))
        for name,sha in manifest['tokenizer_pins'].items():bind(bindings,inputs.codec_policy.MODEL/name,sha)
        for name,sha in manifest['preparation_provider_sha256'].items():bind(bindings,ROOT/name,sha)
        for path in [Path(__file__),Path(isolated.__file__),ROOT/'tools/bench/grading_support.py']:
            bind(bindings,path)
        guard()
    except BaseException as error:
        record['failure']=dict(type=type(error).__name__,message=str(error)[:4096])
        record['source_and_evidence_bindings']=bindings;write_json(output/'result.json',record);return record
    return isolated.grade_paired_cases(cases,decoded,guard,output,record,bindings,lambda outcomes:decision(cases,outcomes))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection',type=Path);parser.add_argument('collection_sha')
    parser.add_argument('inputs',type=Path);parser.add_argument('manifest_sha');parser.add_argument('new_output',type=Path)
    args=parser.parse_args();report=run(args.collection,args.collection_sha,args.inputs,args.manifest_sha,args.new_output)
    print(json.dumps({k:report.get(k)for k in ['completed','decision','failure','restoration_failure','unresolved_case_ids']},indent=2))
    raise SystemExit(0 if report['completed']else 1)
