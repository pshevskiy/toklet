"""Frozen Half task responses with actual Metal admission; no score inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import metal_cli as cli
import metal_controlled as controlled
import metal_engine
import native_usage
import cpu_task_reference as reference
sys.path.insert(0,str(controlled.ROOT/'tools/bench'))
import task_cases as tasks
ROOT=controlled.ROOT
QUALIFICATION=ROOT/'benchmarks/results/2026-10-14-metal-cli-half-qualification-v1.json'
QUALIFICATION_SHA='cb8a8f7117a46af6f81b340e2d92255e706b003beae62dc94330bc1766d52367'
validate_case=tasks.validate_case
VOCAB=tasks.VOCAB
EOS_IDS=tasks.EOS_IDS
MAX_RESPONSE_BYTES=tasks.MAX_RESPONSE_BYTES
LIMIT=controlled.LIMIT


def qualified_cli():
    with QUALIFICATION.open('rb') as stream:payload=stream.read(2*1024**2+1)
    if len(payload)>2*1024**2 or hashlib.sha256(payload).hexdigest()!=QUALIFICATION_SHA:raise ValueError('Frozen CLI qualification differs')
    record=json.loads(payload)
    if record.get('format')!='toklet-metal-cli-half-qualification-v1' or record.get('passed') is not True or record.get('native_runs')!=6 or record.get('raw_event_replay_count')!=79:raise ValueError('CLI functional prerequisite incomplete')
    bindings=record['bindings'];binary=ROOT/'artifacts/m5-cli-half-native-binary-v1/toklet'
    if metal_engine.digest(binary)!=bindings[str(binary.relative_to(ROOT))]['sha256']:raise ValueError('Qualified CLI binary differs')
    snapshot='artifacts/m5-cli-half-source-v1/'
    for name,value in bindings.items():
        if name.startswith(snapshot):
            if metal_engine.digest(ROOT/name)!=value['sha256']:raise ValueError('Compiled CLI snapshot differs')
            if name.startswith(snapshot+'src/') and metal_engine.digest(ROOT/name[len(snapshot):])!=value['sha256']:raise ValueError('Current numerical/CLI source differs from qualified binary')
    return binary


def ready(event,model):
    cli.ready(event,'HalfCompensated',2048,128)
    if event['logical_memory']['weights_bytes']!=controlled.MODELS[model][5] or event['logical_memory']['tokenizer_application_allowance_bytes']!=247772624:raise ValueError('Actual task model/tokenizer admission differs')

def collect_response(case, events, admission, probe_cap=None):
    """Validate one empty-session generation followed by explicit reset.

    The caller supplies bounded JSON events through the reset acknowledgement.
    Finalization text belongs to this response even when it follows `done`.
    Transport/admission errors raise and cannot count as failed model answers.
    """
    validate_case(case)
    projection = "none"
    cap = case["max_output_tokens"] if probe_cap is None else probe_cap
    if probe_cap is not None and (type(probe_cap) is not int or not 1 <= probe_cap <= 8):
        raise ValueError("Metal task probe cap outside1..8")
    ids, text, elapsed = [], [], []
    final_texts = reset_texts = 0
    size = 0
    phase = "generate"
    statistics = None
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Task event must be an object")
        kind = event.get("event")
        if kind == "token" and phase == "generate":
            token = event.get("token_id")
            if (type(event.get("index")) is not int or event["index"] != len(ids)
                    or type(token) is not int or not 0 <= token < VOCAB or token in EOS_IDS
                    or len(ids) >= cap
                    or type(event.get("elapsed_seconds")) not in (int, float)
                    or not math.isfinite(event["elapsed_seconds"]) or event["elapsed_seconds"] < 0
                    or elapsed and event["elapsed_seconds"] < elapsed[-1]):
                raise ValueError("Task visible token event differs")
            ids.append(token)
            elapsed.append(event["elapsed_seconds"])
        elif kind == "text" and phase in ("generate", "finalize", "reset"):
            if phase in ("generate", "finalize"):
                final_texts += 1
                phase = "finalize"
            else:
                reset_texts += 1
            if final_texts > 1 or reset_texts > 1:
                raise ValueError("Task repeats finalization text events")
        elif kind == "done" and phase in ("generate", "finalize"):
            statistics = event.get("statistics")
            phase = "reset"
        elif kind == "reset" and phase == "reset":
            phase = "complete"
        else:
            raise ValueError("Task generation/reset event order differs")
        if kind in ("token", "text"):
            chunk = event.get("text")
            if not isinstance(chunk, str):
                raise ValueError("Task text payload differs")
            size += len(chunk.encode("utf-8"))
            if size > MAX_RESPONSE_BYTES:
                raise ValueError("Task response exceeds transport byte bound")
            text.append(chunk)
    if phase != "complete" or not isinstance(statistics, dict) or statistics.get("cpu_projection") != projection:
        raise ValueError("Task response lacks completed reset or actual Metal mode")
    if any(type(statistics.get(key)) not in (int, float) or statistics[key] != 0
           for key in ("input_formatting_seconds", "tokenization_seconds")):
        raise ValueError("Prepared task IDs were formatted/tokenized again")
    cli.stats({"event":"done","statistics":statistics}, admission)
    generation = statistics.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("Task generation statistics must be an object")
    reason = generation.get("reason", {})
    if not isinstance(reason, dict):
        raise ValueError("Task stop reason must be an object")
    if reason == {"kind": "output_limit"}:
        sampled = cap
        if len(ids) != cap:
            raise ValueError("Task output-limit coverage differs")
        if final_texts:
            raise ValueError("Output-limit text must finalize at reset")
    elif (reason.get("kind") == "stop_token" and set(reason) == {"kind", "token_id"}
          and type(reason["token_id"]) is int and reason["token_id"] in EOS_IDS):
        sampled = len(ids) + 1
        if sampled > cap:
            raise ValueError("Task sampled stop exceeds output admission")
        if reset_texts:
            raise ValueError("EOS text must finalize before done")
    else:
        raise ValueError("Task stop policy differs")
    prompt = len(case["prompt_tokens"])
    expected = {"reused_tokens": 0, "prefill_tokens": prompt, "sampled_tokens": sampled,
                "emitted_tokens": len(ids), "history_tokens": prompt + sampled,
                "evaluated_tokens": prompt + sampled - 1, "pending_tokens": 1}
    if any(type(generation.get(key)) is not int or generation[key] != value for key, value in expected.items()):
        raise ValueError("Task committed/pending/sample state differs")
    timers = ("prefill_seconds", "decode_seconds", "sampling_seconds", "total_seconds")
    first = generation.get("first_token_seconds")
    if ("first_token_seconds" not in generation
            or any(type(generation.get(key)) not in (int, float) or not math.isfinite(generation[key])
            or generation[key] < 0 for key in timers)
            or any(generation[key] > generation["total_seconds"] for key in timers[:-1])
            or elapsed and type(first) not in (int, float)
            or first != (elapsed[0] if elapsed else None)
            or elapsed and elapsed[-1] > generation["total_seconds"]):
        raise ValueError("Task generation timing evidence differs")
    visible = "".join(text)
    return {"id": case["id"], "category": case["category"], "visible_token_ids": ids,
            "visible_text": visible, "visible_text_sha256": hashlib.sha256(visible.encode()).hexdigest(),
            "stop_reason": reason.copy(), "statistics": statistics}



def finalized(responses,path,*,model_exited):
    if model_exited is not True:raise ValueError('Independent decoder requires model exit/reaping')
    copies=reference.finalize([{'visible_token_ids':r['visible_token_ids']} for r in responses],path,model_exited=True)
    if any(a['visible_text']!=b['visible_text'] or a['visible_text_sha256']!=b['visible_text_sha256'] for a,b in zip(copies,responses,strict=True)):raise ValueError('Native finalized task text differs from independent batch decode')


def child(binary,source,model,cases,raw,stderr,check,*,admission_check=None,projection_argument="half-compensated",extra_args=()):
    run={'passed':False,'model_exited':False,'cases':[],'idle_usage':[],'cleanup_errors':[]};worker=None;transport=None
    try:
        command=[str(binary),'--model',str(source),'--backend','metal','--metal-projection',projection_argument,'--interactive','--json','--context-capacity','2048','--prefill-chunk','128',*extra_args];run['command']=command
        worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True);run['worker_pid']=worker.pid
        transport=metal_engine.Transport(worker,raw,stderr,stdout_limit=8*1024**2,event_limit=16000,request_timeout=600,lifetime_timeout=24*3600)
        admission=transport.read();(admission_check or ready)(admission,model);run['ready']=admission
        def usage(label):
            own=native_usage.usage(worker.pid);driver=native_usage.usage();joint=own['lifetime_max_phys_footprint']+driver['lifetime_max_phys_footprint']
            if not 0<joint<=LIMIT:raise ValueError('Task model-phase joint native charge exceeds6GiB')
            run['idle_usage'].append({'phase':label,'worker':own,'driver':driver,'joint_native_lifetime_peak_bytes':joint})
        usage('ready')
        def send(value):worker.stdin.write((json.dumps(value)+'\n').encode());worker.stdin.flush()
        selected=[cases[i] for i in (0,40,70)] if check else cases
        for case in selected:
            request=tasks.request(case)
            if check:request['max_new_tokens']=min(8,request['max_new_tokens'])
            send(request);events=[]
            while True:
                e=transport.read();events.append(e)
                if e.get('event')=='done':break
                if e.get('event') not in ('token','text'):raise ValueError('Metal task generation failed')
            send({'action':'reset'})
            while True:
                e=transport.read();events.append(e)
                if e.get('event')=='reset':break
                if e.get('event')!='text':raise ValueError('Metal task reset/finalization failed')
            response=collect_response(case,events,admission,request['max_new_tokens'] if check else None)
            run['cases'].append({'response':response,'native_events':events,'max_output_tokens':request['max_new_tokens']});usage(f'case_{len(run["cases"])}');print(f'{model} {case["id"]}: {len(response["visible_token_ids"])} emitted',flush=True)
        worker.stdin.close();run['exit_code']=transport.finish()
        if run['exit_code']!=0 or transport.pending or transport.events!=1+sum(len(v['native_events']) for v in run['cases']):raise ValueError('Metal task finalization coverage differs')
        worker.wait(timeout=5);run['model_exited']=True;finalized([v['response'] for v in run['cases']],source/'tokenizer.json',model_exited=True);run['passed']=True
    except BaseException as error:run['error']=f'{type(error).__name__}: {error}'
    finally:
        def cleanup(label,action):
            try:action()
            except BaseException as error:run['cleanup_errors'].append(f'{label}: {type(error).__name__}: {error}')
        if worker:
            cleanup('kill owned group',lambda:metal_engine.kill_owned_group(worker));cleanup('reap',lambda:worker.wait(timeout=5))
            for stream in (worker.stdin,worker.stdout,worker.stderr):cleanup('close pipe',stream.close)
        if transport:cleanup('close transport',transport.close)
        if run['cleanup_errors']:run['passed']=False
    return run


def run(output,model,check=False):
    paths=[output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):raise ValueError('Exclusive task evidence exists')
    binary=qualified_cli();manifest,_=controlled.contract();cases=tasks.load_cases();spec=controlled.MODELS[model];source=ROOT/'models'/spec[0]
    modules=(cli,controlled,reference,tasks,tasks.prepare,controlled.frozen,controlled.common,controlled.metal_prefill,controlled.metal_continue,metal_engine,native_usage)
    files=[binary,Path(__file__),QUALIFICATION,ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/manifest['quality']['tasks_file'],ROOT/'benchmarks/sources.json',ROOT/'benchmarks/result.schema.json',ROOT/'tools/bench/requirements.lock',controlled.prepare.CONTRACT/'manifest.json']+[Path(v.__file__) for v in modules]+list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))+[controlled.prepare.CONTRACT/v['file'] for v in manifest['arrays'].values()]+[source/name for name in ('model.safetensors','config.json',*cli.ASSET_PINS)]
    identities={str(p):metal_engine.digest(p) for p in files};controlled.initial_pins(identities,manifest,source,spec)
    if identities[str(QUALIFICATION)]!=QUALIFICATION_SHA:raise ValueError('CLI qualification changed after parsed admission')
    if identities[str(ROOT/manifest['quality']['tasks_file'])]!=tasks.TASK_SHA or any(identities[str(source/name)]!=sha for name,sha in cli.ASSET_PINS.items()):raise ValueError('Frozen task payload/tokenizer assets differ')
    record={'format':'toklet-metal-task-collection-v1','passed':False,'check_only':check,'collection_completed':False,'quality_completed':False,'model':model,'backend':'metal','projection_mode':'HalfCompensated','normalization_mode':'Original','readback_mode':'FullLogits','task_ids':[v['id'] for v in cases],'task_payload_sha256':tasks.TASK_SHA,'identities':identities,'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scope':'Frozen greedy Half task responses; grading separate; no latency acceptance'}
    with paths[0].open('x') as summary,paths[1].open('x') as raw,paths[2].open('xb') as stderr:
        try:
            record['run']=child(binary,source,model,cases,raw,stderr,check)
            if not record['run']['passed']:raise ValueError('Metal task worker failed')
            record['passed']=True;record['collection_completed']=not check
        except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
        finally:
            record['endpoint_errors']=[];record['identities_unchanged']=False
            try:record['identities_unchanged']=all(metal_engine.digest(Path(p))==sha for p,sha in identities.items())
            except BaseException as error:record['endpoint_errors'].append(f'{type(error).__name__}: {error}')
            if record['endpoint_errors'] or not record['identities_unchanged']:record['passed']=record['collection_completed']=False
            json.dump(record,summary,indent=2,allow_nan=False);summary.write('\n')
    if not record['passed']:raise ValueError('Metal task collection failed; see retained record')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',choices=controlled.MODELS,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();run(a.output.resolve(),a.model,a.check)
