"""Independent four-step raw E2B greedy witness, frozen before native generation.

Staged FP32 chooses the next prefix token. FP64 remains a separate diagnostic;
no tokenizer, stop policy, native engine, source-BF16 or quality claim is made.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys

import gemma4_text_reference as seed

ROOT=Path(__file__).resolve().parents[2]
PROMPT=[2,100,257,100]
STEPS=4
PLANNED_BYTES=STEPS*(1536+2*262144)*12
MAX_BINARY=64*1024**2
ALLOWANCE=512*1024**2
CAPTURE={'final_norm':1536,'head':262144,'logits':262144}


def workload(profile):
    if profile=='raw':
        if PLANNED_BYTES!=25239552 or len(PROMPT)+STEPS-1!=7:
            raise ValueError('Fixed raw greedy workload differs')
        return dict(prompt=list(PROMPT),steps=STEPS,stops=[],planned_bytes=PLANNED_BYTES,
                    maximum_binary_bytes=MAX_BINARY,largest_prefix=7)
    if profile!='basic-hello':raise ValueError('Unknown fixed greedy profile')
    receipt=ROOT/'benchmarks/results/2026-10-25-gemma4-text-codec-reference-v1.json'
    authenticated=json.loads(seed.source.runtime.captured(receipt,2*1024**2,
        '1c71bf5eea1be0b87400a49220929be3f77566d2d8b2b3db7daaf4abfb5c7b78'))
    folder=ROOT/'tests/fixtures/gemma4-text-codec-v1'
    manifest=folder/'manifest.json';fixture=folder/'witness.jsonl'
    if seed.source.sha(manifest)!='4760a6d6cc8f4a63cc07f4da5f2f51c6fa3d1a3bd179b8a56d49481bb25f56ac':
        raise ValueError('Frozen basic-text manifest differs')
    with fixture.open('rb') as source:raw=source.read(1024**2+1)
    if len(raw)!=36960 or hashlib.sha256(raw).hexdigest()!='391077f928d1e288c227bc981f42ae505f7b407f9ee854332f9d2d9c3a410c76':
        raise ValueError('Frozen basic-text records differ')
    if (authenticated['fixture_manifest_sha256']!='4760a6d6cc8f4a63cc07f4da5f2f51c6fa3d1a3bd179b8a56d49481bb25f56ac'
            or authenticated['fixture_records_sha256']!=hashlib.sha256(raw).hexdigest()):
        raise ValueError('Basic-text receipt fixture identities differ')
    rows=[json.loads(line) for line in raw.splitlines()]
    prompts=[row for row in rows if row['kind']=='chat' and row['name']=='hello']
    stops=[row for row in rows if row['kind']=='stop_policy']
    if (len(prompts)!=1 or len(stops)!=1
            or prompts[0]['ids']!=[2,105,2364,107,10979,106,107,105,4368,107]
            or stops[0]['selected_artifact_stops']!=[1,106,50]):
        raise ValueError('Exact frozen basic-text case differs')
    return dict(prompt=prompts[0]['ids'],steps=16,stops=stops[0]['selected_artifact_stops'],
                planned_bytes=16*(1536+2*262144)*12,maximum_binary_bytes=128*1024**2,
                largest_prefix=25,fixture_sha256=hashlib.sha256(raw).hexdigest(),
                rendered_prompt=prompts[0]['rendered'])


def sample(values):
    import numpy as np
    if values.ndim!=1 or values.size<2 or not np.isfinite(values).all():
        raise ValueError('Finite complete vocabulary row required')
    winner=int(np.argmax(values));top=np.partition(values,-2)[-2:]
    return dict(token=winner,margin=float(max(top))-float(min(top)),maximum_ties=int(np.count_nonzero(values==values[winner])))


def capture(binary,arrays,step,contract,name,values,*,steps=STEPS,planned_bytes=PLANNED_BYTES,maximum=MAX_BINARY):
    import numpy as np
    if (steps,planned_bytes,maximum) not in ((4,25239552,64*1024**2),(16,100958208,128*1024**2)):
        raise ValueError('Only fixed capture profiles admitted')
    if contract not in ('math_fp64','staged_fp32') or type(step) is not int or not 0<=step<steps:
        raise ValueError('Fixed greedy step/precision required')
    dtype=np.dtype('<f4' if contract=='staged_fp32' else '<f8')
    if name not in CAPTURE or values.shape!=(CAPTURE[name],) or values.dtype!=dtype:
        raise ValueError('Fixed complete final-row shape/dtype required')
    raw=values.tobytes(order='C')
    item=dict(step=step,contract=contract,name=name,shape=list(values.shape),dtype=dtype.str,
              offset=binary.tell(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),completed=False)
    arrays.append(item)
    if item['offset']+len(raw)>planned_bytes or planned_bytes>maximum:
        raise ValueError('Greedy reference capture exceeds admitted plan')
    if binary.write(raw)!=len(raw):raise OSError('Short greedy array write')
    binary.flush();item['completed']=True
    if not np.isfinite(values).all():raise ValueError('Nonfinite greedy reference values')
    return item


def run(output,admission,runtime_guard,*,profile='raw'):
    import numpy as np
    spec=workload(profile)
    if len(spec['prompt'])+spec['steps']-1!=spec['largest_prefix'] or spec['largest_prefix']>32:
        raise ValueError('Fixed greedy position bound differs')
    # Raw N7 and basic-text N25 head-plus-logits pairs remain within128MiB.
    # Last-row capture/sample/readback overlap is bounded below64MiB. Reserve
    # 128MiB each for runtime/BLAS,tiled projection and head/output plus64MiB
    # each for capture overlap and graph/metadata, matching the512MiB process gate.
    peak_head=2*spec['largest_prefix']*262144*8
    if peak_head>128*1024**2 or 3*128*1024**2+2*64*1024**2!=ALLOWANCE:
        raise ValueError('Greedy reference reservation differs')
    output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('New local artifact output required')
    output.mkdir()
    record=dict(format='toklet-gemma4-greedy-math-v1',passed=False,repository=seed.source.REPO,
        revision=seed.source.REVISION,prompt=spec['prompt'],requested_tokens=spec['steps'],sampled_ids=[],steps=[],arrays=[],
        primary_error=None,endpoint_errors=[],runtime_admission=admission,
        planned_array_bytes=spec['planned_bytes'],process_allowance_bytes=ALLOWANCE,
        largest_prefix=spec['largest_prefix'],peak_fp64_head_and_logits_bytes=peak_head,
        gate=dict(atol=1e-5,rtol=1e-4),selection_contract='staged_fp32; first index of maximum capped logit',
        fp64_history_policy='Teacher-forced on the staged-FP32-selected prefix; not an independent FP64 rollout',
        native_engine_executed=False,original_bf16_source_executed=False,source_graph_qualified=False,
        tokenizer_chat_or_stop_policy_qualified=False,quality_qualified=False,device_fit_qualified=False,
        app_eligibility_changed=False)
    if profile=='basic-hello':
        record.update(format='toklet-gemma4-basic-chat-math-v1',profile=profile,
            prompt_fixture_sha256=spec['fixture_sha256'],rendered_prompt=spec['rendered_prompt'],
            stop_ids=spec['stops'],visible_ids=[],finish_reason=None,
            formatted_prompt_from_frozen_fixture=True,maximum_binary_bytes=spec['maximum_binary_bytes'])
    stream=None;binary=None
    def stop(number,_):
        for value in (signal.SIGINT,signal.SIGTERM):signal.signal(value,signal.SIG_IGN)
        raise InterruptedError('Received signal '+str(number))
    old={v:signal.signal(v,stop) for v in (signal.SIGINT,signal.SIGTERM)}
    try:
        binary=(output/'arrays.bin').open('xb')
        for name,pin in seed.source.PINS.items():
            if seed.source.sha(seed.source.MODEL/name)!=pin:raise ValueError('Pinned checkpoint changed')
        stream=(seed.source.MODEL/'model.safetensors').open('rb');model=seed.weights(stream)
        ids=list(spec['prompt'])
        for step in range(spec['steps']):
            expected=seed.graph.stage_shapes(seed.graph.E2B,len(ids))
            if len(expected)!=713:raise ValueError('Exact complete graph inventory required')
            row=dict(step=step,prefix=list(ids),predictions={},cross_precision=[],completed=False)
            record['steps'].append(row);ledger={}
            for contract,staged in [('math_fp64',False),('staged_fp32',True)]:
                seen=set()
                def sink(name,value):
                    if name in seen or name not in expected or tuple(value.shape)!=expected[name] or value.dtype!=np.dtype('<f4' if staged else '<f8'):
                        raise ValueError('Complete greedy graph shape/dtype differs')
                    seen.add(name)
                    if name in CAPTURE:
                        values=value[0,-1]
                        ledger[(contract,name)]=capture(binary,record['arrays'],step,contract,name,values,
                            steps=spec['steps'],planned_bytes=spec['planned_bytes'],maximum=spec['maximum_binary_bytes'])
                    if not np.isfinite(value).all():raise ValueError('Nonfinite complete greedy graph state')
                    if name=='logits':row['predictions'][contract]=sample(values)
                seed.graph.evaluate(model,list(ids),staged,sink=sink)
                if seen!=set(expected):raise ValueError('Incomplete greedy graph traversal')
            with (output/'arrays.bin').open('rb') as saved:
                def read(contract,name):
                    item=ledger[(contract,name)];saved.seek(item['offset']);body=saved.read(item['bytes'])
                    if len(body)!=item['bytes'] or hashlib.sha256(body).hexdigest()!=item['sha256']:raise ValueError('Saved greedy array changed')
                    return np.frombuffer(body,dtype=item['dtype'])
                for name in CAPTURE:
                    ideal=read('math_fp64',name);actual=read('staged_fp32',name).astype(np.float64)
                    fraction=np.abs(actual-ideal)/(1e-5+1e-4*np.abs(ideal));failures=int(np.count_nonzero(fraction>1.))
                    row['cross_precision'].append(dict(name=name,values=int(actual.size),failed_values=failures,
                        maximum_gate_fraction=float(np.max(fraction)),passed=failures==0))
            token=row['predictions']['staged_fp32']['token']
            if not 0<=token<262144:raise ValueError('Sample outside fixed vocabulary')
            row['selection_disagrees']=row['predictions']['math_fp64']['token']!=token
            row['completed']=True;record['sampled_ids'].append(token);ids.append(token)
            if profile=='basic-hello':
                if token in spec['stops']:
                    record['finish_reason']=dict(kind='stop_token',token_id=token);break
                record['visible_ids'].append(token)
        completed=len(record['sampled_ids'])
        if binary.tell()!=completed*(1536+2*262144)*12 or not 1<=completed<=spec['steps']:
            raise ValueError('Incomplete greedy capture byte count')
        if profile=='raw' and binary.tell()!=PLANNED_BYTES:raise ValueError('Incomplete raw greedy capture')
        if profile=='basic-hello' and record['finish_reason'] is None:
            record['finish_reason']=dict(kind='output_limit')
        record['committed_ids']=ids[:-1];record['pending_id']=ids[-1]
        record['all_cross_precision_gates_passed']=all(c['passed'] for row in record['steps'] for c in row['cross_precision'])
        if profile=='basic-hello':
            record['selection_disagreement_count']=sum(row['selection_disagrees'] for row in record['steps'])
            record['stop_membership_disagreement_count']=sum(
                (row['predictions']['staged_fp32']['token'] in spec['stops']) !=
                (row['predictions']['math_fp64']['token'] in spec['stops']) for row in record['steps'])
            record['policy_diagnostics_passed']=record['selection_disagreement_count']==0 and record['stop_membership_disagreement_count']==0
    except BaseException as error:record['primary_error']=type(error).__name__+': '+str(error)
    finally:
        for value in old:signal.signal(value,signal.SIG_IGN)
        try:record['runtime_endpoint_origins']=runtime_guard()
        except BaseException as error:record['endpoint_errors'].append('runtime: '+type(error).__name__+': '+str(error))
        if profile=='basic-hello':
            try:
                if workload(profile)!=spec:raise ValueError('Prompt/stop witness changed')
            except BaseException as error:record['endpoint_errors'].append('prompt source: '+type(error).__name__+': '+str(error))
        try:
            if stream is not None and seed.source.descriptor_sha(stream)!=seed.source.PINS['model.safetensors']:raise ValueError('Consumed checkpoint descriptor changed')
            for name,pin in seed.source.PINS.items():
                if seed.source.sha(seed.source.MODEL/name)!=pin:raise ValueError('Pinned checkpoint path changed')
        except BaseException as error:record['endpoint_errors'].append(type(error).__name__+': '+str(error))
        try:
            for name,resource in [('source',stream),('binary',binary)]:
                if resource is not None:
                    try:resource.close()
                    except BaseException as error:record['endpoint_errors'].append(name+' close: '+type(error).__name__+': '+str(error))
            try:
                path=output/'arrays.bin'
                if path.exists():record['binary']=dict(bytes=path.stat().st_size,sha256=seed.source.sha(path))
            except BaseException as error:record['endpoint_errors'].append('binary capture: '+type(error).__name__+': '+str(error))
            record['passed']=record['primary_error'] is None and not record['endpoint_errors']
            with (output/'observation.json').open('x') as file:json.dump(record,file,indent=2,allow_nan=False);file.write('\n')
        finally:
            for value,handler in old.items():signal.signal(value,handler)
    print(json.dumps(dict(passed=record['passed'],sampled_ids=record['sampled_ids'],arrays=len(record['arrays']),primary_error=record['primary_error'],endpoint_errors=record['endpoint_errors'])))
    return 0 if record['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True);parser.add_argument('--expected-runtime-sha256',required=True)
    parser.add_argument('--native-manifest',type=Path,required=True);parser.add_argument('--expected-native-sha256',required=True)
    parser.add_argument('--profile',choices=['raw','basic-hello'],default='raw')
    args=parser.parse_args()
    admission,guard=seed.source.runtime_context(args.runtime_manifest,args.expected_runtime_sha256,args.native_manifest,args.expected_native_sha256,
        extra_sources=[Path(__file__).resolve(),Path(seed.__file__).resolve(),Path(seed.graph.__file__).resolve(),ROOT/'tools/reference/test_gemma4_greedy_reference.py'])
    sys.exit(run(args.output,admission,guard,profile=args.profile))
