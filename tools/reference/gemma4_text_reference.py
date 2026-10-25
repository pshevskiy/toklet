"""Four-token real E2B mathematical reference, with bounded streamed evidence.

This executes independent NumPy/libm math over FP32-reconstructed packed rows.
It does not execute Toklet, upstream original-BF16 code, tokenization or Chat.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import sys

import affine4_tiles as tiles
import gemma4_affine_reference as source
import gemma4_text_math as graph

ROOT=Path(__file__).resolve().parents[2]
TOKENS=[2,100,257,100]
FUTURE_CHANGED=[2,100,257,101]
ALLOWANCE=512*1024**2
MAX_BINARY=128*1024**2
MAX_PROBES=128*1024
MAX_FUTURE_BINARY=77552640 # three unchanged positions out of each four-position state


def weights(stream):
    """Exact pinned text tensors; immutable packed rows and direct FP32 vectors."""
    import numpy as np
    packed=source.matrices(stream)
    matrices={name.removeprefix(source.PREFIX):matrix for name,matrix in packed.items()}
    stream.seek(8);raw=stream.read(source.HEADER_BYTES)
    if hashlib.sha256(raw).hexdigest()!=source.HEADER_SHA:raise ValueError('Consumed header changed')
    header=json.loads(raw,object_pairs_hook=source.unique);vectors={};physical=set()
    for name,shape in graph.shapes(graph.E2B).items():
        full=source.PREFIX+name
        if len(shape)==2:
            physical.update([full,full.removesuffix('.weight')+'.scales',full.removesuffix('.weight')+'.biases'])
            continue
        physical.add(full);item=header[full]
        if item['dtype']!='BF16' or tuple(item['shape'])!=shape:raise ValueError('Direct graph vector differs')
        start,end=item['data_offsets']
        if not 0<=start<end<=source.CONTAINER_BYTES-8-source.HEADER_BYTES or end-start!=2*shape[0]:
            raise ValueError('Direct vector extent differs')
        stream.seek(8+source.HEADER_BYTES+start);value=stream.read(end-start)
        if len(value)!=end-start:raise ValueError('Direct vector truncated')
        vectors[name]=tiles.bf16(np.frombuffer(value,dtype='<u2'))
    if physical!={name for name in header if name.startswith(source.PREFIX)}:
        raise ValueError('Exact physical text tensor inventory required')
    return graph.Weights(graph.E2B,matrices,vectors)


def capture_future_prefix(contract,name,value,reference,record,binary):
    """Keep exact-prefix hashes, or raw typed values before any numerical gate."""
    import numpy as np
    if value.shape!=reference.shape or value.dtype!=reference.dtype:
        raise ValueError('Future graph state dtype/shape differs')
    def prefix(x):return x[:,:,:3] if name.endswith(('rotary','context')) else x[:,:3]
    actual=prefix(value);reference=prefix(reference)
    raw=actual.tobytes(order='C');reference_raw=reference.tobytes(order='C')
    actual_sha=hashlib.sha256(raw).hexdigest();reference_sha=hashlib.sha256(reference_raw).hexdigest()
    exact=raw==reference_raw
    item=dict(contract=contract,name=name,shape=list(actual.shape),dtype=actual.dtype.str,
              values=int(actual.size),actual_sha256=actual_sha,reference_sha256=reference_sha,
              exact_prefix=exact,raw=None,capture_completed=False,comparison_completed=False,
              maximum_gate_fraction=None,passed=False)
    record['future_prefix'].append(item)
    if not exact:
        descriptor=dict(offset=binary.tell(),bytes=len(raw),sha256=actual_sha,completed=False)
        item['raw']=descriptor
        if binary.tell()+len(raw)>MAX_FUTURE_BINARY:raise ValueError('Future-prefix capture exceeds admitted plan')
        if binary.write(raw)!=len(raw):raise OSError('Short future-prefix write')
        binary.flush();descriptor['completed']=True
    item['capture_completed']=True
    actual=actual.astype(np.float64);reference=reference.astype(np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Non-finite future-prefix comparison')
    fraction=np.abs(actual-reference)/(1e-5+1e-4*np.abs(reference))
    maximum=float(np.max(fraction));passed=maximum<=1.
    item.update(maximum_gate_fraction=maximum,passed=passed,comparison_completed=True)
    if not passed:raise ValueError('Future token altered prefix: '+name)


def run(output,admission,runtime_guard):
    import numpy as np
    expected=graph.stage_shapes(graph.E2B,len(TOKENS))
    planned=sum(math.prod(shape) for shape in expected.values())*12
    if len(expected)!=713 or planned!=103403520 or planned>MAX_BINARY:
        raise ValueError('Independent graph array plan differs')
    #128MiB runtime/BLAS +128MiB tile/output +128MiB head/logits +64MiB
    # capture copy +64MiB graph/metadata. This is reference-only admission.
    if 128*1024**2*3+64*1024**2*2>ALLOWANCE:raise ValueError('Reference reservation exceeds allowance')
    output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('New local artifact output required')
    output.mkdir()
    record=dict(format='toklet-gemma4-text-math-v1',passed=False,repository=source.REPO,revision=source.REVISION,
        tokens=TOKENS,future_changed_tokens=FUTURE_CHANGED,primary_error=None,endpoint_errors=[],arrays=[],
        cross_precision=[],future_prefix=[],next_token_predictions={},runtime_admission=admission,
        precision='FP32 reconstructed weights; separate FP64 activations and staged FP32, FP64 reductions/Decimal80 rotary',
        gate=dict(atol=1e-5,rtol=1e-4),process_allowance_bytes=ALLOWANCE,planned_array_bytes=planned,
        states_per_contract=len(expected),native_engine_executed=False,original_bf16_source_executed=False,
        source_graph_qualified=False,device_fit_qualified=False,quality_qualified=False,app_eligibility_changed=False)
    stream=None;binary=None;probes=None;future_binary=None
    def stop(number,_):
        for value in (signal.SIGINT,signal.SIGTERM):signal.signal(value,signal.SIG_IGN)
        raise InterruptedError('Received signal '+str(number))
    old={v:signal.signal(v,stop) for v in (signal.SIGINT,signal.SIGTERM)}
    try:
        binary=(output/'arrays.bin').open('xb');probes=(output/'probes.bin').open('xb')
        future_binary=(output/'future.bin').open('xb')
        for name,pin in source.PINS.items():
            if source.sha(source.MODEL/name)!=pin:raise ValueError('Pinned checkpoint path changed')
        stream=(source.MODEL/'model.safetensors').open('rb');model=weights(stream)
        ledger={}
        def write(contract,name,value):
            if name not in expected or tuple(value.shape)!=expected[name] or value.dtype!=np.dtype('<f4' if contract=='staged_fp32' else '<f8') or not np.isfinite(value).all():
                raise ValueError('Expected finite graph state differs: '+name)
            key=(contract,name)
            if key in ledger:raise ValueError('Duplicate graph state')
            raw=value.tobytes(order='C');start=binary.tell()
            if start+len(raw)>planned:raise ValueError('Graph capture exceeds admitted array plan')
            item=dict(contract=contract,name=name,shape=list(value.shape),dtype=value.dtype.str,
                      offset=start,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),completed=False)
            record['arrays'].append(item);ledger[key]=item
            if binary.write(raw)!=len(raw):raise OSError('Short graph array write')
            binary.flush();item['completed']=True
            indices=sorted({i*(value.size-1)//7 for i in range(8)})
            selected=value.reshape(-1)[indices].tobytes()
            probe=dict(indices=indices,offset=probes.tell(),bytes=len(selected),sha256=hashlib.sha256(selected).hexdigest(),completed=False)
            item['probe']=probe
            if probes.tell()+len(selected)>MAX_PROBES or probes.write(selected)!=len(selected):raise OSError('Probe capture bound/write failure')
            probes.flush();probe['completed']=True
            if name=='logits':
                predictions=[]
                for row in value[0]:
                    winner=int(np.argmax(row));top=np.partition(row,-2)[-2:]
                    predictions.append(dict(token=winner,margin=float(max(top)-min(top))))
                record['next_token_predictions'][contract]=predictions
        for contract,staged in [('math_fp64',False),('staged_fp32',True)]:
            graph.evaluate(model,TOKENS,staged,sink=lambda n,x:write(contract,n,x))
            if {name for c,name in ledger if c==contract}!=set(expected):raise ValueError('Incomplete graph capture')
        if binary.tell()!=planned:raise ValueError('Captured graph byte count differs')
        binary.flush()
        with (output/'arrays.bin').open('rb') as saved:
            def read(contract,name):
                item=ledger[(contract,name)];saved.seek(item['offset']);raw=saved.read(item['bytes'])
                if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Saved graph state changed')
                return np.frombuffer(raw,dtype=item['dtype']).reshape(item['shape'])
            # Cross-precision comparisons are diagnostics, never gate waivers.
            for name in expected:
                ideal=read('math_fp64',name);staged=read('staged_fp32',name).astype(np.float64)
                fraction=np.abs(staged-ideal)/(1e-5+1e-4*np.abs(ideal))
                failures=int(np.count_nonzero(fraction>1.))
                record['cross_precision'].append(dict(name=name,values=int(ideal.size),failed_values=failures,
                    maximum_gate_fraction=float(np.max(fraction)),passed=failures==0))
            for contract,staged in [('math_fp64',False),('staged_fp32',True)]:
                seen=set()
                def future(name,value):
                    if name in seen or name not in expected or tuple(value.shape)!=expected[name]:raise ValueError('Unexpected future graph state')
                    seen.add(name);reference=read(contract,name)
                    capture_future_prefix(contract,name,value,reference,record,future_binary)
                graph.evaluate(model,FUTURE_CHANGED,staged,sink=future)
                if seen!=set(expected):raise ValueError('Incomplete future-prefix comparison')
        record['all_cross_precision_gates_passed']=all(x['passed'] for x in record['cross_precision'])
        record['cross_precision_failed_arrays']=sum(not x['passed'] for x in record['cross_precision'])
        record['cross_precision_failed_values']=sum(x['failed_values'] for x in record['cross_precision'])
    except BaseException as error:record['primary_error']=type(error).__name__+': '+str(error)
    finally:
        for value in old:signal.signal(value,signal.SIG_IGN)
        try:record['runtime_endpoint_origins']=runtime_guard()
        except BaseException as error:record['endpoint_errors'].append('runtime: '+type(error).__name__+': '+str(error))
        try:
            if stream is not None and source.descriptor_sha(stream)!=source.PINS['model.safetensors']:raise ValueError('Consumed checkpoint descriptor changed')
            for name,pin in source.PINS.items():
                if source.sha(source.MODEL/name)!=pin:raise ValueError('Pinned checkpoint path changed')
        except BaseException as error:record['endpoint_errors'].append(type(error).__name__+': '+str(error))
        try:
            for name,resource in [('source',stream),('binary',binary),('probes',probes),('future',future_binary)]:
                if resource is not None:
                    try:resource.close()
                    except BaseException as error:record['endpoint_errors'].append(name+' close: '+type(error).__name__+': '+str(error))
            for name in ('arrays.bin','probes.bin','future.bin'):
                try:
                    path=output/name
                    if path.exists():record[name]=dict(bytes=path.stat().st_size,sha256=source.sha(path))
                except BaseException as error:record['endpoint_errors'].append(name+' capture: '+type(error).__name__+': '+str(error))
            record['passed']=record['primary_error'] is None and not record['endpoint_errors']
            with (output/'observation.json').open('x') as file:json.dump(record,file,indent=2,allow_nan=False);file.write('\n')
        finally:
            for value,handler in old.items():signal.signal(value,handler)
    print(json.dumps(dict(passed=record['passed'],arrays=len(record['arrays']),primary_error=record['primary_error'],endpoint_errors=record['endpoint_errors'],cross_precision_failed_arrays=record.get('cross_precision_failed_arrays'),cross_precision_failed_values=record.get('cross_precision_failed_values'))))
    return 0 if record['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True);parser.add_argument('--expected-runtime-sha256',required=True)
    parser.add_argument('--native-manifest',type=Path,required=True);parser.add_argument('--expected-native-sha256',required=True)
    args=parser.parse_args()
    admission,guard=source.runtime_context(args.runtime_manifest,args.expected_runtime_sha256,args.native_manifest,args.expected_native_sha256,
        extra_sources=[Path(__file__).resolve(),Path(graph.__file__).resolve(),ROOT/'tools/reference/test_gemma4_text_math.py'])
    sys.exit(run(args.output,admission,guard))
