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


def sample(values):
    import numpy as np
    if values.ndim!=1 or values.size<2 or not np.isfinite(values).all():
        raise ValueError('Finite complete vocabulary row required')
    winner=int(np.argmax(values));top=np.partition(values,-2)[-2:]
    return dict(token=winner,margin=float(max(top))-float(min(top)),maximum_ties=int(np.count_nonzero(values==values[winner])))


def capture(binary,arrays,step,contract,name,values):
    import numpy as np
    if contract not in ('math_fp64','staged_fp32') or type(step) is not int or not 0<=step<STEPS:
        raise ValueError('Fixed greedy step/precision required')
    dtype=np.dtype('<f4' if contract=='staged_fp32' else '<f8')
    if name not in CAPTURE or values.shape!=(CAPTURE[name],) or values.dtype!=dtype:
        raise ValueError('Fixed complete final-row shape/dtype required')
    raw=values.tobytes(order='C')
    item=dict(step=step,contract=contract,name=name,shape=list(values.shape),dtype=dtype.str,
              offset=binary.tell(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),completed=False)
    arrays.append(item)
    if item['offset']+len(raw)>PLANNED_BYTES or PLANNED_BYTES>MAX_BINARY:
        raise ValueError('Greedy reference capture exceeds admitted plan')
    if binary.write(raw)!=len(raw):raise OSError('Short greedy array write')
    binary.flush();item['completed']=True
    if not np.isfinite(values).all():raise ValueError('Nonfinite greedy reference values')
    return item


def run(output,admission,runtime_guard):
    import numpy as np
    if PLANNED_BYTES!=25239552 or len(PROMPT)+STEPS-1>7:
        raise ValueError('Fixed greedy workload differs')
    # N7 FP64 head plus capped logits:2*7*262144*8 =29,360,128B.
    # Last-row capture/sample/readback overlap is bounded below64MiB. Reserve
    # 128MiB each for runtime/BLAS,tiled projection and head/output plus64MiB
    # each for capture overlap and graph/metadata, matching the512MiB process gate.
    peak_head=2*(len(PROMPT)+STEPS-1)*262144*8
    if peak_head>128*1024**2 or 3*128*1024**2+2*64*1024**2!=ALLOWANCE:
        raise ValueError('Greedy reference reservation differs')
    output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('New local artifact output required')
    output.mkdir()
    record=dict(format='toklet-gemma4-greedy-math-v1',passed=False,repository=seed.source.REPO,
        revision=seed.source.REVISION,prompt=PROMPT,requested_tokens=STEPS,sampled_ids=[],steps=[],arrays=[],
        primary_error=None,endpoint_errors=[],runtime_admission=admission,
        planned_array_bytes=PLANNED_BYTES,process_allowance_bytes=ALLOWANCE,
        largest_prefix=7,peak_fp64_head_and_logits_bytes=peak_head,
        gate=dict(atol=1e-5,rtol=1e-4),selection_contract='staged_fp32; first index of maximum capped logit',
        fp64_history_policy='Teacher-forced on the staged-FP32-selected prefix; not an independent FP64 rollout',
        native_engine_executed=False,original_bf16_source_executed=False,source_graph_qualified=False,
        tokenizer_chat_or_stop_policy_qualified=False,quality_qualified=False,device_fit_qualified=False,
        app_eligibility_changed=False)
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
        ids=list(PROMPT)
        for step in range(STEPS):
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
                        ledger[(contract,name)]=capture(binary,record['arrays'],step,contract,name,values)
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
        if binary.tell()!=PLANNED_BYTES:raise ValueError('Incomplete greedy capture byte count')
        record['committed_ids']=ids[:-1];record['pending_id']=ids[-1]
        record['all_cross_precision_gates_passed']=all(c['passed'] for row in record['steps'] for c in row['cross_precision'])
    except BaseException as error:record['primary_error']=type(error).__name__+': '+str(error)
    finally:
        for value in old:signal.signal(value,signal.SIG_IGN)
        try:record['runtime_endpoint_origins']=runtime_guard()
        except BaseException as error:record['endpoint_errors'].append('runtime: '+type(error).__name__+': '+str(error))
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
    args=parser.parse_args()
    admission,guard=seed.source.runtime_context(args.runtime_manifest,args.expected_runtime_sha256,args.native_manifest,args.expected_native_sha256,
        extra_sources=[Path(__file__).resolve(),Path(seed.__file__).resolve(),Path(seed.graph.__file__).resolve(),ROOT/'tools/reference/test_gemma4_greedy_reference.py'])
    sys.exit(run(args.output,admission,guard))
