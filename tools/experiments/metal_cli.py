"""Explicit bounded native CLI functional qualification; no comparative timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

import metal_engine
import native_usage

ROOT=Path(__file__).resolve().parents[2]
LIMIT=6*1024**3
FIELDS=('weights_bytes','kv_bytes','scratch_bytes','history_bytes','logits_bytes','metadata_bytes','metadata_allowance_bytes','tokenizer_application_allowance_bytes')
ASSET_PINS={
    'tokenizer.json':'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json':'5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
    'generation_config.json':'e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6',
}
CHAT_SHA='27988589738912067d64f25fd24217fd61e578c8981cd827fd3befdd8cdd6851'

def chat_case(path):
    with path.open('rb') as stream:data=stream.read(1024**2+1)
    if len(data)>1024**2 or hashlib.sha256(data).hexdigest()!=CHAT_SHA:raise ValueError('Frozen chat manifest differs')
    value=json.loads(data)['chat_cases'][0]
    if value['add_generation_prompt'] is not True or len(value['ids'])!=31:raise ValueError('Frozen first chat case differs')
    return value

def execution(mode,chunk):
    return {'weight_dtype':'FP16','kv_dtype':'FP16','projection_mode':mode,'normalization_mode':'Original','readback_mode':'FullLogits','prefill_chunk_size':chunk}

def memory(record):
    m=record.get('logical_memory')
    if type(m) is not dict or set(m)!=set(FIELDS) or any(type(m[k]) is not int or m[k]<0 for k in FIELDS):raise ValueError('CLI complete logical memory differs')
    if type(record.get('memory_admitted_bytes')) is not int or sum(m.values())!=record['memory_admitted_bytes'] or not 0<sum(m.values())<=LIMIT:raise ValueError('CLI admission differs')
    return m

def ready(event,mode,capacity,chunk):
    if event.get('event')!='ready' or event.get('backend')!='metal' or event.get('cpu_projection')!='none' or type(event.get('context_capacity')) is not int or event['context_capacity']!=capacity or event.get('execution')!=execution(mode,chunk) or type(event.get('execution',{}).get('prefill_chunk_size')) is not int:raise ValueError('Actual CLI execution metadata differs')
    return memory(event)

def stats(event,initial):
    s=event.get('statistics',{})
    if event.get('event')!='done' or s.get('backend')!='metal' or s.get('cpu_projection')!='none' or s.get('execution')!=initial['execution'] or memory(s)!=initial['logical_memory']:raise ValueError('CLI statistics/readiness differ')
    g=s['generation']
    for key in ('reused_tokens','prefill_tokens','sampled_tokens','emitted_tokens','history_tokens','evaluated_tokens','pending_tokens'):
        if type(g.get(key)) is not int or g[key]<0:raise ValueError('CLI generation counts differ')
    if g['history_tokens']!=g['evaluated_tokens']+g['pending_tokens'] or g['emitted_tokens']>g['sampled_tokens'] or g['pending_tokens'] not in (0,1):raise ValueError('CLI generation state differs')
    return g

def child(binary,source,mode,raw,stderr,script,*,capacity=32,chunk=3,extra=(),idle_exit=False,admission=ready,projection_argument=None):
    command=[str(binary),'--model',str(source),'--backend','metal','--interactive','--context-capacity',str(capacity),'--prefill-chunk',str(chunk),*extra]
    if mode!='Default':command+=['--metal-projection',projection_argument or ('half-compensated' if mode=='HalfCompensated' else 'original')]
    actual='Original' if mode=='Default' else mode
    worker=None;transport=None;record={'command':command,'passed':False,'phases':[],'idle_usage':[],'cleanup_errors':[]}
    try:
        worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        record['worker_pid']=worker.pid;transport=metal_engine.Transport(worker,raw,stderr,request_timeout=120,lifetime_timeout=900,event_limit=4096,stdout_limit=1024**2)
        initial=transport.read();admission(initial,actual,capacity,chunk);record['ready']=initial
        def usage(label):
            left=native_usage.usage(worker.pid);right=native_usage.usage();charge=left['lifetime_max_phys_footprint']+right['lifetime_max_phys_footprint']
            if not 0<charge<=LIMIT:raise ValueError('CLI joint native charge exceeds6GiB')
            record['idle_usage'].append({'phase':label,'worker':left,'driver':right,'joint_native_lifetime_peak_bytes':charge})
        usage('ready')
        def request(value,*,cancel=False):
            worker.stdin.write((json.dumps(value)+'\n').encode());worker.stdin.flush();events=[];cancelled=False
            while True:
                e=transport.read();events.append(e)
                if e.get('event')=='token' and cancel and not cancelled:os.killpg(worker.pid,signal.SIGINT);cancelled=True
                if e.get('event') in ('done','error','reset'):
                    if e['event']=='done':
                        g=stats(e,initial);tokens=[v for v in events if v.get('event')=='token']
                        if len(tokens)!=g['emitted_tokens'] or [v['index'] for v in tokens]!=list(range(len(tokens))):raise ValueError('CLI emitted event coverage differs')
                    record['phases'].append({'request':value,'events':events,'cancel_requested':cancel});usage(f'phase_{len(record["phases"])}');return events
        script(request,record)
        if idle_exit:os.killpg(worker.pid,signal.SIGINT)
        else:worker.stdin.close()
        while not transport.stdout_closed or b'\n' in transport.pending:
            if b'\n' in transport.pending:
                e=transport.read()
                if e.get('event')!='text':raise ValueError('Unexpected EOF finalization event')
                record.setdefault('finalization',[]).append(e)
            else:transport.poll(.1)
            if worker.poll() is not None and not transport.pending and transport.stdout_closed:break
            if transport.started+900<time.monotonic():raise TimeoutError('CLI finalization watchdog')
        record['exit_code']=worker.wait(timeout=5)
        if record['exit_code']!=(130 if idle_exit else 0):raise ValueError('CLI unexpected exit')
        if transport.pending:raise ValueError('CLI truncated finalization')
        record['passed']=True
    except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
    finally:
        def cleanup(label,action):
            try:action()
            except BaseException as error:record['cleanup_errors'].append(f'{label}: {type(error).__name__}: {error}')
        if worker:
            cleanup('kill owned group',lambda:metal_engine.kill_owned_group(worker));cleanup('reap',lambda:worker.wait(timeout=5))
            for stream in (worker.stdin,worker.stdout,worker.stderr):cleanup('close pipe',stream.close)
        if transport:cleanup('close transport',transport.close)
        if record['cleanup_errors']:record['passed']=False
    return record

def raw_script(request,record):
    e=request({'tokens':[1,7,13,3],'max_new_tokens':4});g=e[-1]['statistics']['generation']
    if [v['token_id'] for v in e if v['event']=='token']!=[47]*4 or (g['reused_tokens'],g['prefill_tokens'],g['evaluated_tokens'],g['pending_tokens'])!=(0,4,7,1):raise ValueError('Tiny frozen greedy/state differs')
    g=request({'tokens':[1,7,13,3,47,47,47],'max_new_tokens':1})[-1]['statistics']['generation']
    if (g['reused_tokens'],g['prefill_tokens'],g['evaluated_tokens'],g['pending_tokens'])!=(7,0,7,1):raise ValueError('Tiny cached prefix differs')
    g=request({'action':'continue','max_new_tokens':2})[-1]['statistics']['generation']
    if (g['reused_tokens'],g['prefill_tokens'],g['evaluated_tokens'],g['pending_tokens'])!=(7,1,9,1):raise ValueError('Pending continuation differs')
    if request({'tokens':[64],'max_new_tokens':1})[-1]['event']!='error':raise ValueError('Invalid request accepted')
    g=request({'action':'continue','max_new_tokens':1})[-1]['statistics']['generation']
    if (g['reused_tokens'],g['prefill_tokens'],g['evaluated_tokens'],g['pending_tokens'])!=(9,1,10,1):raise ValueError('Invalid input changed prefix')
    if request({'action':'reset'})[-1]['event']!='reset':raise ValueError('Reset missing')
    g=request({'tokens':[11],'max_new_tokens':1})[-1]['statistics']['generation']
    if (g['reused_tokens'],g['prefill_tokens'])!=(0,1):raise ValueError('Reset reused state')

def stop_script(request,record):
    e=request({'tokens':[1,7,13,3],'max_new_tokens':4});g=e[-1]['statistics']['generation']
    if (g['sampled_tokens'],g['emitted_tokens'],g['evaluated_tokens'],g['pending_tokens'],g['reason'])!=(1,0,4,1,{'kind':'stop_token','token_id':47}):raise ValueError('Hidden stop semantics differ')

def cancel_script(request,record):
    e=request({'tokens':[1,7,13,3],'max_new_tokens':2000},cancel=True);before=e[-1]['statistics']['generation']
    if before['reason']['kind']!='cancelled':raise ValueError('Active cancellation missing')
    after=request({'action':'continue','max_new_tokens':1})[-1]['statistics']['generation']
    if (after['reused_tokens'],after['prefill_tokens'],after['reason']['kind'])!=(before['evaluated_tokens'],before['pending_tokens'],'output_limit'):raise ValueError('Cancellation resume differs')

def text_script(request,record,raw_count,chat_case):
    values=[{'prompt':'你好，🙂 café.','raw':True,'max_new_tokens':8},
            {'messages':chat_case['messages'],'tools':chat_case['tools'],'max_new_tokens':8}]
    record['expected_text_prompt_tokens']=[raw_count,len(chat_case['ids'])]
    for value,count in zip(values,record['expected_text_prompt_tokens'],strict=True):
        events=request(value)
        if not events or events[-1].get('event')!='done' or any(e.get('event')=='error' for e in events):raise ValueError('Text generation did not complete successfully')
        g=events[-1]['statistics']['generation'];reason=g['reason']['kind']
        if (g['reused_tokens']!=0 or g['prefill_tokens']!=count or not 1<=g['sampled_tokens']<=8 or g['pending_tokens']!=1
                or g['evaluated_tokens']!=count+g['sampled_tokens']-1 or reason not in ('output_limit','stop_token')
                or (reason=='output_limit' and g['sampled_tokens']!=8)):raise ValueError('Text prompt/reset generation state differs')
        reset=request({'action':'reset'})
        if not reset or reset[-1].get('event')!='reset' or any(e.get('event')=='error' for e in reset):raise ValueError('Text reset not acknowledged')

def run(binary,output):
    paths=[output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):raise ValueError('Exclusive CLI evidence destination exists')
    tiny=ROOT/'artifacts/m5-tiny-source-v1';source=ROOT/'models/Qwen2.5-0.5B-Instruct'
    manifest=ROOT/'tests/fixtures/chat-v1/manifest.json';frozen_chat=chat_case(manifest)
    files=[binary,Path(__file__),Path(metal_engine.__file__),Path(native_usage.__file__),ROOT/'Cargo.toml',ROOT/'Cargo.lock',manifest]+list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))+list(tiny.glob('*'))+list(source.glob('*.json'))+[source/'model.safetensors']
    identities={str(p):metal_engine.digest(p) for p in files if p.is_file()}
    if identities[str(source/'model.safetensors')]!='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe' or identities[str(source/'config.json')]!='18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45':raise ValueError('Fixed CLI source/config differs')
    if identities[str(manifest)]!=CHAT_SHA or any(identities.get(str(source/name))!=sha for name,sha in ASSET_PINS.items()):raise ValueError('Fixed CLI tokenizer/chat assets differ')
    record={'format':'toklet-metal-cli-functional-v1','passed':False,'scope':'Authored functional CLI only; no task quality/latency acceptance','identities':identities,'text_chat_case':frozen_chat,'runs':[]}
    with paths[0].open('x') as summary,paths[1].open('x') as raw,paths[2].open('xb') as stderr:
        try:
            for mode in ('Default','Original','HalfCompensated'):record['runs'].append(child(binary,tiny,mode,raw,stderr,raw_script))
            record['runs'].append(child(binary,tiny,'HalfCompensated',raw,stderr,stop_script,extra=('--stop-token','47')))
            with tempfile.TemporaryDirectory(prefix='toklet-cli-cancel-') as directory:
                cap=Path(directory);config=json.loads((tiny/'config.json').read_text());config['max_position_embeddings']=2048;(cap/'config.json').write_text(json.dumps(config));(cap/'model.safetensors').symlink_to(tiny/'model.safetensors')
                record['cancel_config']=config;record['runs'].append(child(binary,cap,'HalfCompensated',raw,stderr,cancel_script,capacity=2048,idle_exit=True))
            # Independent batch detokenization; load before GPU worker startup.
            from tokenizers import Tokenizer
            tokenizer=Tokenizer.from_file(str(source/'tokenizer.json'))
            raw_count=len(tokenizer.encode('你好，🙂 café.',add_special_tokens=False).ids)
            result=child(binary,source,'HalfCompensated',raw,stderr,lambda request,current:text_script(request,current,raw_count,frozen_chat),capacity=256,chunk=3);record['runs'].append(result)
            # Resets finalize each response; token text plus text events must equal
            # independent batch decode, including any deferred output-limit tail.
            ids=[];text=''
            for phase in result.get('phases',[]):
                for e in phase['events']:
                    if e['event']=='token':ids.append(e['token_id']);text+=e['text']
                    elif e['event']=='text':text+=e['text']
                    elif e['event']=='reset':
                        if text!=tokenizer.decode(ids,skip_special_tokens=True):raise ValueError('Finalized CLI text differs from batch decode')
                        ids=[];text=''
            if ids or text:raise ValueError('Text response not finalized')
            if not all(r['passed'] for r in record['runs']):raise ValueError('Native CLI functional worker failed')
            record['passed']=True
        except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
        finally:
            record['endpoint_errors']=[];record['identities_unchanged']=False
            try:record['identities_unchanged']=all(metal_engine.digest(Path(p))==sha for p,sha in identities.items())
            except BaseException as error:record['endpoint_errors'].append(f'{type(error).__name__}: {error}')
            if record['endpoint_errors'] or not record['identities_unchanged']:record['passed']=False
            json.dump(record,summary,indent=2,allow_nan=False);summary.write('\n')
    if not record['passed']:raise ValueError('CLI qualification failed; see retained record')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.binary.resolve(),a.output.resolve())
