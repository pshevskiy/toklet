"""Frozen authored Half application trace; probes do not establish latency acceptance."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import task_grade_admission as admission
import metal_tasks as tasks
import metal_cli as cli
import metal_controlled as controlled
import metal_engine
import native_usage
ROOT=controlled.ROOT
VOCAB=tasks.VOCAB
EOS_IDS=tasks.EOS_IDS
MAX_RESPONSE_BYTES=65536


def trace_inputs(source,manifest,cap):
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    import tokenizers
    if jinja2.__version__!='3.1.6' or tokenizers.__version__!='0.22.2':raise ValueError('Pinned application oracle versions differ')
    path=controlled.prepare.CONTRACT/'application_trace.json'
    data=admission.bounded(path,65536)
    if hashlib.sha256(data).hexdigest()!=manifest['authored_text_sha256'][path.name]:raise ValueError('Frozen application transcript differs')
    messages=json.loads(data,object_pairs_hook=admission.evidence.unique_object,parse_constant=admission.evidence.reject_constant,parse_float=admission.evidence.finite_float)
    if [v['role'] for v in messages]!=['system','user','assistant','user','assistant','user']:raise ValueError('Frozen application roles differ')
    config=admission.bounded(source/'tokenizer_config.json',1024**2)
    if hashlib.sha256(config).hexdigest()!=cli.ASSET_PINS['tokenizer_config.json']:raise ValueError('Frozen application template config differs')
    config=admission.evidence.decode_event(config.decode());template=config['chat_template']
    if hashlib.sha256(template.encode()).hexdigest()!='cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f':raise ValueError('Frozen application template differs')
    environment=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols'])
    def reject(message):raise ValueError(message)
    environment.globals['raise_exception']=reject
    if metal_engine.digest(source/'tokenizer.json')!=cli.ASSET_PINS['tokenizer.json']:raise ValueError('Frozen application tokenizer differs')
    tokenizer=tokenizers.Tokenizer.from_file(str(source/'tokenizer.json'))
    requests=[];prompts=[]
    for end in (2,4,6):
        subset=messages[:end];rendered=environment.from_string(template).render(messages=subset,add_generation_prompt=True,tools=None,bos_token=config.get('bos_token'),eos_token=config.get('eos_token'))
        prompts.append(tokenizer.encode(rendered,add_special_tokens=False).ids)
        requests.append({'action':'generate','messages':subset,'max_new_tokens':cap})
    if list(map(len,prompts))!=[36,76,117]:raise ValueError('Frozen application prompt counts differ')
    del tokenizer;gc.collect()
    return requests,prompts


def collect_response(prompt_ids, cap, events, admission):
    """Validate one empty-session generation followed by explicit reset.

    The caller supplies bounded JSON events through the reset acknowledgement.
    Finalization text belongs to this response even when it follows `done`.
    Transport/admission errors raise and cannot count as failed model answers.
    """
    if type(cap) is not int or cap not in (2,128) or not prompt_ids:
        raise ValueError('Application cap/prompt differs')
    projection = 'none'
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
    for key in ('input_formatting_seconds','tokenization_seconds','detokenization_seconds','publication_seconds','load_seconds','conversion_seconds','tokenizer_load_seconds'):
        if type(statistics.get(key)) not in (int,float) or not math.isfinite(statistics[key]) or statistics[key]<0:
            raise ValueError('Application attributed timing differs')
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
    prompt = len(prompt_ids)
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
    return {"visible_token_ids": ids,
            "visible_text": visible, "visible_text_sha256": hashlib.sha256(visible.encode()).hexdigest(),
            "stop_reason": reason.copy(), "statistics": statistics}




def received_timing(events, received):
    if len(events)!=len(received) or any(type(t) not in (int,float) or not math.isfinite(t) or t<0 for t in received) or any(b<a for a,b in zip(received,received[1:])):raise ValueError('Application receipt timestamps differ')
    tokens=[t for e,t in zip(events,received,strict=True) if e['event']=='token']
    visible=[t for e,t in zip(events,received,strict=True) if e.get('event') in ('token','text') and e['text']]
    done=[t for e,t in zip(events,received,strict=True) if e['event']=='done']
    if len(done)!=1 or events[-1]['event']!='reset':raise ValueError('Application receipt completion differs')
    return {'submission_to_first_token_seconds':tokens[0] if tokens else None,'submission_to_first_visible_text_seconds':visible[0] if visible else None,'submission_to_done_seconds':done[0],'submission_to_finalized_seconds':received[-1],'received_token_event_seconds':tokens}


def child(binary,source,model,requests,prompts,raw,stderr,check,*,admission_check=None,projection_argument="half-compensated",extra_args=()):
    run={'passed':False,'model_exited':False,'samples':[],'warmups':[],'idle_usage':[],'cleanup_errors':[]};worker=None;transport=None
    try:
        command=[str(binary),'--model',str(source),'--backend','metal','--metal-projection',projection_argument,'--interactive','--json','--context-capacity','2048','--prefill-chunk','128',*extra_args];run['command']=command
        started=time.perf_counter();worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True);run['worker_pid']=worker.pid
        transport=metal_engine.Transport(worker,raw,stderr,stdout_limit=16*1024**2,event_limit=16000,request_timeout=600,lifetime_timeout=86400)
        ready=transport.read();ready_at=time.perf_counter();(admission_check or tasks.ready)(ready,model);run['ready']=ready;run['submission_to_ready_seconds']=ready_at-started
        def usage(label):
            own=native_usage.usage(worker.pid);driver=native_usage.usage();joint=own['lifetime_max_phys_footprint']+driver['lifetime_max_phys_footprint']
            if not 0<joint<=controlled.LIMIT:raise ValueError('Application native joint charge exceeds6GiB')
            run['idle_usage'].append({'phase':label,'worker':own,'driver':driver,'joint_native_lifetime_peak_bytes':joint})
        usage('ready');warmups,repetitions=(0,1) if check else (3,10)
        for iteration in range(warmups+repetitions):
            scenario_start=time.perf_counter();turns=[]
            for index,(request,prompt) in enumerate(zip(requests,prompts,strict=True)):
                submitted=time.perf_counter();worker.stdin.write((json.dumps(request)+'\n').encode());worker.stdin.flush();events=[];received=[]
                phase='generate'
                while True:
                    event=transport.read();at=time.perf_counter();events.append(event);received.append(at-submitted)
                    kind=event.get('event')
                    if kind=='done' and phase=='generate':
                        phase='reset';worker.stdin.write(b'{"action":"reset"}\n');worker.stdin.flush()
                    elif kind=='reset' and phase=='reset':break
                    elif kind not in (('token','text') if phase=='generate' else ('text',)):
                        raise ValueError('Application generation/reset event order differs')
                response=collect_response(prompt,request['max_new_tokens'],events,ready)
                turns.append({'request':request,'prompt_tokens':prompt,'native_events':events,'response':response,**received_timing(events,received)})
                usage(f'iteration_{iteration}_turn_{index}')
            sample={'turns':turns,'scenario_seconds':time.perf_counter()-scenario_start};run['warmups' if iteration<warmups else 'samples'].append(sample)
            print(f'{model} application {iteration+1}/{warmups+repetitions}: {sample["scenario_seconds"]:.3f}s',flush=True)
        worker.stdin.close();transport.finish();run['exit_code']=worker.poll()
        if run['exit_code']!=0 or transport.pending or transport.events!=1+sum(len(t['native_events']) for s in run['warmups']+run['samples'] for t in s['turns']):raise ValueError('Application finalization coverage/exit differs')
        worker.wait(timeout=5);run['model_exited']=True
        responses=[t['response'] for s in run['warmups']+run['samples'] for t in s['turns']]
        tasks.finalized(responses,source/'tokenizer.json',model_exited=True)
        run['passed']=True
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
    output=Path(output).resolve();paths=[output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):raise ValueError('Exclusive application evidence exists')
    manifest=admission.frozen_manifest();binary,binary_sha=admission.qualified_binary('toklet-metal-task-collection-v1');source=ROOT/'models'/controlled.MODELS[model][0]
    cap=2 if check else 128;requests,prompts=trace_inputs(source,manifest,cap)
    files=admission.inventory_paths('toklet-metal-task-collection-v1',binary,source,manifest)|{str(Path(__file__)),str(Path(admission.__file__)),str(controlled.prepare.CONTRACT/'application_trace.json')}
    identities={p:metal_engine.digest(Path(p)) for p in files};controlled.initial_pins(identities,manifest,source,controlled.MODELS[model])
    if identities[str(binary)]!=binary_sha or identities[str(tasks.QUALIFICATION)]!=tasks.QUALIFICATION_SHA or identities[str(controlled.prepare.CONTRACT/'application_trace.json')]!=manifest['authored_text_sha256']['application_trace.json'] or any(identities[str(source/name)]!=sha for name,sha in cli.ASSET_PINS.items()):raise ValueError('Application pinned binary/assets/transcript differ')
    observation=controlled.metal_prefill.observation
    environment={'hardware':observation(['/usr/sbin/sysctl','-n','hw.model']),'chip':observation(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string']),'physical_memory_bytes':int(observation(['/usr/sbin/sysctl','-n','hw.memsize'])),'os':observation(['/usr/bin/sw_vers','-productVersion']),'os_build':observation(['/usr/bin/sw_vers','-buildVersion']),'compiler':observation(['rustc','--version']),'power':observation(['/usr/bin/pmset','-g','batt'])+'\n'+observation(['/usr/bin/pmset','-g','custom']),'thermal_start':observation(['/usr/bin/pmset','-g','therm'])}
    record={'format':'toklet-metal-application-v1','passed':False,'check_only':check,'measurements_completed':False,'quality_completed':False,'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'identities':identities,'model':model,'backend':'metal','projection_mode':'HalfCompensated','normalization_mode':'Original','readback_mode':'FullLogits','environment':environment,'policy':{'warmups':0 if check else 3,'repetitions':1 if check else 10,'turns':3,'output_cap':cap,'greedy':True,'cache':'empty per turn; scripted assistant history; resident model'},'timing_scope':'Driver submission through receipt of complete JSON token/nonempty text/done/reset events; completed native numerical/sampling/formatting/tokenization/text/publication components overlap and are not summed; scenario includes per-turn driver validation/memory instrumentation. Model setup and independent oracle decode outside warm request timers; no cold storage/energy claim.'}
    with paths[0].open('x') as summary,paths[1].open('x') as raw,paths[2].open('xb') as stderr:
        try:
            record['run']=child(binary,source,model,requests,prompts,raw,stderr,check)
            if not record['run']['passed']:raise ValueError('Application owned worker failed')
            record['passed']=True;record['measurements_completed']=not check
            if not check:
                values=[s['scenario_seconds'] for s in record['run']['samples']];record['summary']={'scenario_median_seconds':statistics.median(values),'scenario_min_seconds':min(values),'scenario_max_seconds':max(values)}
        except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
        finally:
            record['endpoint_errors']=[];record['identities_unchanged']=False
            try:record['identities_unchanged']=all(metal_engine.digest(Path(p))==sha for p,sha in identities.items())
            except BaseException as error:record['endpoint_errors'].append(f'{type(error).__name__}: {error}')
            try:record['environment']['thermal_end']=observation(['/usr/bin/pmset','-g','therm'])
            except BaseException as error:record['endpoint_errors'].append(f'thermal_end: {type(error).__name__}: {error}')
            if record['endpoint_errors'] or not record['identities_unchanged']:record['passed']=record['measurements_completed']=False
            json.dump(record,summary,indent=2,allow_nan=False);summary.write('\n')
    if not record['passed']:raise ValueError('Application collection failed; evidence retained')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=tuple(controlled.MODELS),required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--check',action='store_true');args=parser.parse_args();run(args.output,args.model,args.check)
