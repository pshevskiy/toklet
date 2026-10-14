"""Explicit v2 CLI function/memory qualification; no task quality or latency."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess

import metal_cli as shared
import metal_quality_v2 as loss

ROOT=shared.ROOT
MODE='Q4CompensatedTiled'
ARG='q4-v2-compensated-tiled'
GPU_REL='benchmarks/results/2026-10-14-q4-v2-gpu-qualification-v1.json'
GPU_SHA='e084d9c12a564f3979b220c7e597c2d503f223d2706e1f86a9b9519f64ade1f1'
TINY_WEIGHTS=38504


def execution(chunk):
    return dict(weight_dtype='Q4 v2 + FP16 exclusions',kv_dtype='FP16',projection_mode=MODE,
                normalization_mode='Compensated',readback_mode='FullLogits',prefill_chunk_size=chunk,
                packed_format_version=2,quantization_group_size=8,
                quantizer='toklet-q4-affine-f32-source-half-metadata-rne-g8-v2')


def ready(event,mode,capacity,chunk,*,weights,external=None):
    if (mode!=MODE or event.get('event')!='ready' or event.get('backend')!='metal'
            or event.get('cpu_projection')!='none' or type(event.get('context_capacity')) is not int
            or event['context_capacity']!=capacity or event.get('execution')!=execution(chunk)
            or any(type(event.get('execution',{}).get(k)) is not type(v) for k,v in execution(chunk).items())):
        raise ValueError('Actual v2 CLI metadata differs')
    memory=shared.memory(event)
    if memory['weights_bytes']!=weights or external is not None and memory['tokenizer_application_allowance_bytes']!=external:
        raise ValueError('Actual v2 model/application residency differs')
    return memory


def call(binary,source,packed,raw,stderr,script,*,weights,capacity=32,chunk=3,extra=(),idle_exit=False,external=None):
    return shared.child(binary,source,MODE,raw,stderr,script,capacity=capacity,chunk=chunk,
                        extra=('--q4-weights',str(packed),*extra),idle_exit=idle_exit,
                        admission=partial(ready,weights=weights,external=external),projection_argument=ARG)


def cancel_script(request,record):
    events=request({'tokens':[1,7,13,3],'max_new_tokens':28},cancel=True)
    before=events[-1]['statistics']['generation']
    if before['reason']['kind']!='cancelled':raise ValueError('V2 active cancellation missing')
    after=request({'action':'continue','max_new_tokens':1})[-1]['statistics']['generation']
    if (after['reused_tokens'],after['prefill_tokens'],after['reason']['kind'])!=(before['evaluated_tokens'],before['pending_tokens'],'output_limit'):
        raise ValueError('V2 cancellation prefix replay differs')


def raw_real_script(request,record,expected):
    tokens=[9707,11,358,498]
    events=request({'tokens':tokens,'max_new_tokens':1})
    emitted=[v['token_id'] for v in events if v['event']=='token']
    if emitted!=[expected]:raise ValueError('Frozen stored-FP32 real-model greedy differs')
    cached=request({'tokens':tokens,'max_new_tokens':1})[-1]['statistics']['generation']
    if (cached['reused_tokens'],cached['prefill_tokens'],cached['evaluated_tokens'],cached['pending_tokens'])!=(4,0,4,1):
        raise ValueError('V2 fully cached real prompt differs')
    continued=request({'action':'continue','max_new_tokens':2})
    first=[v['token_id'] for v in continued if v['event']=='token']
    if len(first)!=2:raise ValueError('V2 real continuation incomplete')
    replay=request({'tokens':tokens+[expected],'max_new_tokens':2})
    if [v['token_id'] for v in replay if v['event']=='token']!=first:raise ValueError('V2 real pending continuation differs from replay')
    if request({'tokens':[151936],'max_new_tokens':1})[-1]['event']!='error':raise ValueError('V2 invalid ID accepted')
    if request({'action':'reset'})[-1]['event']!='reset':raise ValueError('V2 real reset missing')


def v1_command(binary,tiny,packed):
    # JSON raw-ID mode needs no tokenizer assets in the tiny source fixture.
    return [str(binary),'--model',str(tiny),'--backend','metal','--q4-weights',str(packed),
            '--metal-projection',ARG,'--tokens','1','--json','--context-capacity','32']


def check_v1_rejection(completed):
    if (completed.returncode!=1 or completed.stdout
            or completed.stderr.decode().strip()!='error: --q4-weights requires a Q4 v2 artifact'):
        raise ValueError('V1 artifact did not reject at explicit v2 boundary')


def run(binary,output):
    paths=[output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):raise ValueError('Exclusive v2 CLI evidence exists')
    gpu_path=ROOT/GPU_REL
    gpu=json.loads(loss.shared.frozen_bytes(gpu_path,2*1024**2,GPU_SHA))
    geometry_path=ROOT/loss.GEOMETRY_REL
    geometry=json.loads(loss.shared.frozen_bytes(geometry_path,1024**2,loss.GEOMETRY_SHA))
    if not gpu['passed'] or not gpu['review']['accepted'] or not geometry['passed'] or not geometry['review']['accepted']:
        raise ValueError('Reviewed v2 numerical prerequisites required')
    artifact_pins={p:v['sha256'] for p,v in gpu['bindings'].items() if p.startswith(('artifacts/m5-tiny-source-v1/','artifacts/m5-tiny-q4-v2/'))}
    artifact_pins.update({p:v['sha256'] for p,v in geometry['bindings'].items() if p.startswith(('artifacts/m5-small-q4-v2/','artifacts/m5-larger-q4-v2/','tests/fixtures/qwen-small-q4-v2-v1/','tests/fixtures/qwen-larger-q4-v2-v1/'))})
    tiny=ROOT/'artifacts/m5-tiny-source-v1';tiny_packed=ROOT/'artifacts/m5-tiny-q4-v2'
    # The v1 rejection only needs a valid authenticated legacy artifact. Its
    # loaded format must fail the explicit v2 CLI boundary before any event.
    v1=ROOT/'artifacts/m5-tiny-q4-v1'
    files=[binary,Path(__file__),Path(shared.__file__),Path(loss.__file__),Path(shared.metal_engine.__file__),Path(shared.native_usage.__file__),gpu_path,geometry_path,
           ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'tests/fixtures/chat-v1/manifest.json']
    files+=list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))+[ROOT/p for p in artifact_pins]+list(v1.glob('*'))
    pins={gpu_path:GPU_SHA,geometry_path:loss.GEOMETRY_SHA,ROOT/'tests/fixtures/chat-v1/manifest.json':shared.CHAT_SHA}
    pins.update({ROOT/p:sha for p,sha in artifact_pins.items()})
    for spec in loss.SPECS.values():
        source=ROOT/'models'/spec['name']
        files += [source/p for p in ('model.safetensors','config.json',*shared.ASSET_PINS)]
        pins.update({source/'model.safetensors':spec['source'],source/'config.json':spec['config']})
        pins.update({source/name:sha for name,sha in shared.ASSET_PINS.items()})
    identities={str(p):shared.metal_engine.digest(p) for p in files if p.is_file()}
    if any(identities.get(str(p))!=sha for p,sha in pins.items()):raise ValueError('Initial frozen v2 CLI inputs differ')
    chat=shared.chat_case(ROOT/'tests/fixtures/chat-v1/manifest.json')
    record={'format':'toklet-metal-cli-v2-functional-v1','passed':False,'scope':'Explicit v2 CLI function/memory only; small quality remains rejected, no task/performance/M5 acceptance','identities':identities,'runs':[]}
    with paths[0].open('x') as summary,paths[1].open('x') as raw,paths[2].open('xb') as stderr:
        try:
            command=v1_command(binary,tiny,v1)
            rejected=subprocess.run(command,capture_output=True,timeout=120)
            record['v1_rejection']={'command':command,'exit_code':rejected.returncode,'stdout':rejected.stdout.decode(),'stderr':rejected.stderr.decode()}
            check_v1_rejection(rejected)
            # Exact legacy metadata/default arithmetic remains independently checked.
            record['runs'].append(shared.child(binary,tiny,'HalfCompensated',raw,stderr,shared.raw_script))
            record['runs'].append(call(binary,tiny,tiny_packed,raw,stderr,shared.raw_script,weights=TINY_WEIGHTS,external=128*1024**2))
            record['runs'].append(call(binary,tiny,tiny_packed,raw,stderr,shared.stop_script,weights=TINY_WEIGHTS,extra=('--stop-token','47')))
            record['runs'].append(call(binary,tiny,tiny_packed,raw,stderr,cancel_script,weights=TINY_WEIGHTS,idle_exit=True))
            for label,spec in loss.SPECS.items():
                source=ROOT/'models'/spec['name'];packed=ROOT/f'artifacts/m5-{label}-q4-v2'
                probe=json.loads(loss.shared.frozen_bytes(ROOT/f'tests/fixtures/qwen-{label}-q4-v2-v1/manifest.json',1024**2,artifact_pins[f'tests/fixtures/qwen-{label}-q4-v2-v1/manifest.json']))
                expected=probe['argmax']['future_changed']['stored_fp32'][-1]
                record['runs'].append(call(binary,source,packed,raw,stderr,lambda request,current:raw_real_script(request,current,expected),weights=spec['weights'],extra=('--no-eos',),external=247772624))
                # Driver tokenizer loading happens between exited workers. Its
                # ownership is accounted by the joint native observations.
                from tokenizers import Tokenizer
                tokenizer=Tokenizer.from_file(str(source/'tokenizer.json'))
                raw_count=len(tokenizer.encode('你好，🙂 café.',add_special_tokens=False).ids)
                text=call(binary,source,packed,raw,stderr,lambda request,current:shared.text_script(request,current,raw_count,chat),weights=spec['weights'],capacity=256,external=247772624)
                record['runs'].append(text);ids=[];visible=''
                for phase in text.get('phases',[]):
                    for event in phase['events']:
                        if event['event']=='token':ids.append(event['token_id']);visible+=event['text']
                        elif event['event']=='text':visible+=event['text']
                        elif event['event']=='reset':
                            if visible!=tokenizer.decode(ids,skip_special_tokens=True):raise ValueError('V2 finalized text differs from independent decode')
                            ids=[];visible=''
                if ids or visible:raise ValueError('V2 text not finalized')
                del tokenizer
            if not all(v['passed'] for v in record['runs']):raise ValueError('V2 CLI functional worker failed')
            record['passed']=True
        except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
        finally:
            record['endpoint_errors']=[]
            try:record['identities_unchanged']=all(shared.metal_engine.digest(Path(p))==sha for p,sha in identities.items())
            except BaseException as error:record['identities_unchanged']=False;record['endpoint_errors'].append(str(error))
            if not record['identities_unchanged'] or record['endpoint_errors']:record['passed']=False
            json.dump(record,summary,indent=2,allow_nan=False);summary.write('\n')
    if not record['passed']:raise ValueError('V2 CLI qualification failed; see retained record')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.binary.resolve(),args.output.resolve())
