"""Independent logical KV/staging/ring expectations and pinned cache comparison.

The required attention union, query ranges and final fixed-slot ring are separate.
Upstream normal-update parity does not qualify native transaction/rollback.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import sys

import reference_runtime as runtime

FORMAT='toklet-gemma-kv-math-v1'
MAX_JSON,MAX_RAW=4*1024*1024,4*1024*1024
REVISION='0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
SOURCES={
    'base.py':'63d8e58a4b6f8c0902719499c693647fbf486bbd5d22b09868a07d37e08c83d6',
    'cache.py':'c96cb169fd5e0ec4584a162dd61dc62ab08ccdca03a9850f8a491ce6809145e6',
}
SPECS=(
    ('empty',3,(),4,2,3),
    ('single_wrap',3,(1,1,1,1,1,1,1),4,2,3),
    ('prefill_wrap',3,(5,1,1,2,1),4,2,3),
    ('concat_wrap',3,(2,2,1,3,1),4,2,3),
    ('window_one',1,(3,1,2,1),2,1,2),
    ('global',None,(2,1,3,1),4,2,3),
    ('window512',512,(511,1,1,2,1),1,1,1),
)
ATOL,RTOL=1e-5,1e-4


def encoded(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def layout(offset,count,window,capacity):
    if (any(type(x) is not int for x in (offset,count,capacity))
            or not 0<=offset<=516 or not 1<=count<=511 or offset+count>516
            or not 1<=capacity<=516 or (window is not None and (type(window) is not int or not 1<=window<=512 or window!=capacity))
            or (window is None and offset+count>capacity)):
        raise ValueError('Bounded KV staging geometry required')
    after=offset+count
    start=0 if window is None else max(0,offset-window+1)
    old=offset-start
    spans=[]
    if old:
        slot=start%capacity;first=min(old,capacity-slot);spans.append([slot,first])
        if first<old:spans.append([0,old-first])
    def positions(end):
        result=[None]*capacity
        first=0 if window is None else max(0,end-window)
        for position in range(first,end):result[position%capacity]=position
        return result
    return dict(before=offset,after=after,count=count,required_positions=list(range(start,after)),
        visible_ranges=[[0 if window is None else max(0,q-window+1),q+1] for q in range(offset,after)],
        old_spans=spans,positions_before=positions(offset),positions_after=positions(after))


def data_values(positions,heads,dims,kind):
    import numpy as np
    out=np.zeros((heads,len(positions),dims),np.float32)
    for head in range(heads):
        for row,pos in enumerate(positions):
            if pos is None:continue
            for feature in range(dims):
                if kind=='k':value=1+pos*8+head*2+feature*.25
                elif kind=='v':value=((pos*5+head*3+feature)%23-11)*.125
                elif kind=='q':value=((pos*3+head*7+feature*2)%11-5)*.125
                else:raise ValueError('Known KV operand required')
                out[head,row,feature]=value
    return out


def context(q,k,v,offset,window):
    import numpy as np
    q,k,v=[np.asarray(x) for x in (q,k,v)]
    if (type(offset) is not int or not 0<=offset<=516 or q.ndim!=3 or k.ndim!=3 or v.shape!=k.shape
            or not 1<=q.shape[0]<=4 or not 1<=k.shape[0]<=2 or q.shape[0]%k.shape[0]
            or not 1<=q.shape[1]<=511 or not 1<=q.shape[2]<=3 or q.shape[2]!=k.shape[2]
            or k.shape[1]!=offset+q.shape[1] or k.shape[1]>516
            or (window is not None and (type(window) is not int or not 1<=window<=512))
            or any(x.dtype!=np.dtype('float32') or not np.isfinite(x).all() for x in (q,k,v))):
        raise ValueError('Bounded finite FP32 logical KV context required')
    result=np.zeros(q.shape,np.float64)
    for head in range(q.shape[0]):
        owner=head//(q.shape[0]//k.shape[0])
        scores=q[head].astype(np.float64) @ k[owner].astype(np.float64).T
        for row in range(q.shape[1]):
            end=offset+row+1;start=0 if window is None else max(0,end-window)
            included=scores[row,start:end];weights=np.exp(included-included.max());weights/=weights.sum()
            result[head,row]=weights @ v[owner,start:end].astype(np.float64)
    return result


def describe(value):
    return dict(dtype='<f4' if value.dtype.itemsize==4 else '<f8',shape=list(value.shape),values=value.reshape(-1).tolist())


def export():
    import numpy as np
    cases=[]
    for name,window,chunks,h,hk,d in SPECS:
        capacity=window if window is not None else sum(chunks)
        abort=next((i for i in range(len(chunks)) if sum(chunks[:i])>0 and (window is None or sum(chunks[:i+1])>window)),None)
        initial=data_values([None]*capacity,hk,d,'k')
        case=dict(name=name,window=window,chunks=list(chunks),query_heads=h,kv_heads=hk,dims=d,capacity=capacity,
            abort_step=abort,initial_k=describe(initial),initial_v=describe(initial.copy()),steps=[])
        offset=0
        for index,count in enumerate(chunks):
            item=layout(offset,count,window,capacity)
            q=data_values(range(offset,offset+count),h,d,'q');k=data_values(range(offset+count),hk,d,'k');v=data_values(range(offset+count),hk,d,'v')
            arrays=dict(q=q,new_k=k[:,offset:].copy(),new_v=v[:,offset:].copy(),logical_k=k,logical_v=v,
                ring_before_k=data_values(item['positions_before'],hk,d,'k'),ring_before_v=data_values(item['positions_before'],hk,d,'v'),
                ring_after_k=data_values(item['positions_after'],hk,d,'k'),ring_after_v=data_values(item['positions_after'],hk,d,'v'),
                context=context(q,k,v,offset,window))
            case['steps'].append(dict(index=index,generation_before=index,generation_after=index+1,abort_expected=index==abort,**item,
                arrays={key:describe(value) for key,value in arrays.items()}))
            offset+=count
        cases.append(case)
    return encoded(dict(format=FORMAT,source_revision=REVISION,source_sha256=SOURCES,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime=dict(numpy=np.__version__),
        precision='FP32 operands; independent FP64 logical-prefix attention; exact fixed-slot committed state',
        rollback_is_required_policy_not_upstream_qualification=True,native_graph_implemented=False,device_qualified=False,cases=cases))


def same(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,dict):return set(a)==set(b) and all(same(a[key],b[key]) for key in a)
    return a==b


def verify(path,expected_sha):
    raw=runtime.captured(path,MAX_JSON,runtime.sha(expected_sha))
    data=json.loads(raw,object_pairs_hook=runtime.json_unique,parse_constant=runtime.bad_json_number,parse_float=runtime.finite_json_number)
    if (not isinstance(data,dict) or set(data)!={'format','source_revision','source_sha256','producer_sha256','runtime','precision','rollback_is_required_policy_not_upstream_qualification','native_graph_implemented','device_qualified','cases'}
            or data['format']!=FORMAT or data['source_revision']!=REVISION or data['source_sha256']!=SOURCES or data['runtime']!=dict(numpy=runtime.VERSIONS['numpy'])
            or data['precision']!='FP32 operands; independent FP64 logical-prefix attention; exact fixed-slot committed state'
            or data['rollback_is_required_policy_not_upstream_qualification'] is not True or data['native_graph_implemented'] is not False or data['device_qualified'] is not False
            or not isinstance(data['cases'],list) or len(data['cases'])!=len(SPECS)):
        raise ValueError('KV mathematical identity/scope differs')
    runtime.sha(data['producer_sha256'])
    for case,(name,window,chunks,h,hk,d) in zip(data['cases'],SPECS):
        capacity=window if window is not None else sum(chunks)
        abort=next((i for i in range(len(chunks)) if sum(chunks[:i])>0 and (window is None or sum(chunks[:i+1])>window)),None)
        metadata=dict(name=name,window=window,chunks=list(chunks),query_heads=h,kv_heads=hk,dims=d,capacity=capacity,abort_step=abort)
        if not isinstance(case,dict) or set(case)!={*metadata,'initial_k','initial_v','steps'} or not same({key:case[key] for key in metadata},metadata) or not isinstance(case['steps'],list) or len(case['steps'])!=len(chunks):raise ValueError('KV case inventory differs')
        rows=[(case['initial_k'],[hk,capacity,d],'<f4'),(case['initial_v'],[hk,capacity,d],'<f4')]
        offset=0
        for index,(step,count) in enumerate(zip(case['steps'],chunks)):
            expected=dict(index=index,generation_before=index,generation_after=index+1,abort_expected=index==abort,**layout(offset,count,window,capacity))
            if not isinstance(step,dict) or set(step)!={*expected,'arrays'} or not same({key:step[key] for key in expected},expected):raise ValueError('KV step metadata differs')
            shapes=dict(q=[h,count,d],new_k=[hk,count,d],new_v=[hk,count,d],logical_k=[hk,offset+count,d],logical_v=[hk,offset+count,d],context=[h,count,d],
                **{key:[hk,capacity,d] for key in ('ring_before_k','ring_before_v','ring_after_k','ring_after_v')})
            if not isinstance(step['arrays'],dict) or set(step['arrays'])!=set(shapes):raise ValueError('KV array inventory differs')
            rows.extend((step['arrays'][key],shape,'<f8' if key=='context' else '<f4') for key,shape in shapes.items());offset+=count
        for item,shape,dtype in rows:
            if (not isinstance(item,dict) or set(item)!={'dtype','shape','values'} or item['dtype']!=dtype or not same(item['shape'],shape)
                    or not isinstance(item['values'],list) or len(item['values'])!=math.prod(shape)):
                raise ValueError('KV array descriptor differs')
            for value in item['values']:
                if type(value) not in (float,int) or not -sys.float_info.max<=value<=sys.float_info.max or not math.isfinite(value):raise ValueError('Finite KV array value required')
                if dtype=='<f4':
                    try:
                        if struct.unpack('<f',struct.pack('<f',value))[0]!=value:raise ValueError('Exactly representable FP32 KV input required')
                    except OverflowError as error:raise ValueError('FP32 KV input overflow') from error
    return data


def source_nodes(sources):
    chosen={'base.py':{'create_causal_mask','scaled_dot_product_attention'},'cache.py':{'create_attention_mask','_BaseCache','KVCache','RotatingKVCache'}}
    result=[]
    for filename,names in chosen.items():
        nodes=[node for node in ast.parse(sources[filename]).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
        if len(nodes)!=len(names) or {node.name for node in nodes}!=names:raise ValueError('Unique pinned KV source definitions required')
        result.extend((filename,node) for node in nodes)
    return result


def compare(reference,reference_sha,sources,device,destination,runtime_manifest,runtime_sha):
    if device not in ('cpu','gpu'):raise ValueError('Explicit KV source reference device required')
    data=verify(reference,reference_sha)
    if (set(sources)!=set(SOURCES) or any(not isinstance(raw,bytes) or len(raw)>128*1024
            or hashlib.sha256(raw).hexdigest()!=SOURCES[name] for name,raw in sources.items())):
        raise ValueError('Authenticated KV source buffers differ')
    nodes=source_nodes(sources)
    admitted=runtime.admit_runtime(runtime_manifest,runtime_sha,[Path(__file__)])
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    if (np.__file__!=admitted['imports']['numpy'] or mx.__file__!=admitted['imports']['mlx.core']
            or nn.__file__!=admitted['imports']['mlx.nn']):raise ValueError('Actual KV imports differ from admitted files')
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):raise ValueError('Empty regular KV output required')
    previous_device=mx.default_device();previous_signals={};blob=bytearray()
    rows,comparisons,cleanup=[],[],[];primary=None

    def interrupted(number,_frame):
        for value in (signal.SIGTERM,signal.SIGINT):signal.signal(value,signal.SIG_IGN)
        raise InterruptedError('Received signal '+str(number))

    def array(step,name):
        item=step['arrays'][name]
        return np.array(item['values'],dtype=item['dtype']).reshape(item['shape'])

    def observe(step,name,value):
        mx.eval(value)
        actual=np.asarray(value)
        if actual.dtype.hasobject or actual.dtype.kind not in 'biufc' or actual.nbytes>MAX_RAW:
            step['arrays'][name]=dict(raw_unavailable='Unsupported numeric observation',dtype=str(actual.dtype),shape=list(actual.shape))
            raise ValueError('Bounded numeric KV observation required')
        dtype=actual.dtype.newbyteorder('<');raw=np.ascontiguousarray(actual,dtype=dtype).tobytes();padding=(-len(blob))%8
        if len(blob)+padding+len(raw)>MAX_RAW:raise ValueError('KV observation buffer exceeds bound')
        blob.extend(b'\0'*padding)
        step['arrays'][name]=dict(dtype=dtype.str,shape=list(actual.shape),offset=len(blob),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        blob.extend(raw)
        if not np.isfinite(actual).all():raise ValueError('Finite KV source observation required')
        return actual

    try:
        for value in (signal.SIGTERM,signal.SIGINT):previous_signals[value]=signal.signal(value,interrupted)
        mx.set_default_device(mx.cpu if device=='cpu' else mx.gpu)
        environment=dict(mx=mx,nn=nn)
        # Only annotations are deferred. Every selected executable body is unchanged.
        deferred=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        module=ast.fix_missing_locations(ast.Module(body=[deferred,*[node for _,node in nodes]],type_ignores=[]))
        exec(compile(module,'authenticated-kv-selected-source','exec'),environment)
        for case in data['cases']:
            cache=environment['KVCache']() if case['window'] is None else environment['RotatingKVCache'](max_size=case['window'],keep=0)
            row=dict(name=case['name'],completed=False,steps=[]);rows.append(row)
            if cache.offset!=0 or cache.size()!=0 or cache.keys is not None or cache.values is not None:raise ValueError('Initial source KV state differs')
            for step in case['steps']:
                trial=dict(index=step['index'],completed=False,arrays={});row['steps'].append(trial)
                if cache.offset!=step['before']:raise ValueError('Source KV offset before update differs')
                q,new_k,new_v=[mx.array(array(step,name))[None] for name in ('q','new_k','new_v')]
                mask=cache.make_mask(step['count'],window_size=case['window'],return_array=True)
                trial['mask_kind']='none' if mask is None else 'array'
                # Retain the completed mask before an update that can fail.
                actual_mask=None if mask is None else observe(trial,'mask',mask)
                fetched_k,fetched_v=cache.update_and_fetch(new_k,new_v)
                actual_k=observe(trial,'k',fetched_k)
                actual_v=observe(trial,'v',fetched_v)
                if (actual_k.dtype!=np.dtype('float32') or actual_v.dtype!=np.dtype('float32')
                        or actual_k.ndim!=4 or actual_k.shape[:2]!=(1,case['kv_heads']) or actual_k.shape[3]!=case['dims']
                        or actual_v.shape!=actual_k.shape or actual_k.shape[2]!=len(step['required_positions'])):
                    raise ValueError('Fetched source KV dtype/shape differs')
                encoded_positions=(actual_k[0,0,:,0].astype(np.float64)-1)/8
                if not np.all(encoded_positions==np.floor(encoded_positions)):raise ValueError('Source KV position markers differ')
                positions=encoded_positions.astype(np.int64).tolist();trial['physical_positions']=positions
                if len(set(positions))!=len(positions) or sorted(positions)!=step['required_positions']:raise ValueError('Source KV logical attention union differs')
                for actual,kind in ((actual_k,'k'),(actual_v,'v')):
                    if not np.array_equal(actual[0],data_values(positions,case['kv_heads'],case['dims'],kind)):
                        raise ValueError('Fetched source KV values differ from independent position encoding')
                expected_mask=np.array([[first<=position<last for position in positions] for first,last in step['visible_ranges']],np.bool_)
                if actual_mask is None:
                    if not expected_mask.all():raise ValueError('Missing source mask would include expired/future keys')
                elif actual_mask.dtype!=np.dtype('bool') or actual_mask.shape!=expected_mask.shape or not np.array_equal(actual_mask,expected_mask):
                    raise ValueError('Source KV query visibility differs')
                value=environment['scaled_dot_product_attention'](q,fetched_k,fetched_v,cache=cache,scale=1.,mask=mask)
                actual=observe(trial,'context',value[0]);expected=array(step,'context')
                if actual.dtype!=np.dtype('float32') or actual.shape!=expected.shape:raise ValueError('Source KV context dtype/shape differs')
                difference=np.abs(actual.astype(np.float64)-expected);ratio=difference/(ATOL+RTOL*np.abs(expected))
                comparisons.append(dict(case=case['name'],step=step['index'],passed=bool(np.all(ratio<=1)),maximum_absolute_error=float(difference.max()),maximum_error_to_policy_ratio=float(ratio.max())))
                for actual,name in zip((q,new_k,new_v),('q','new_k','new_v')):
                    if not np.array_equal(np.asarray(actual)[0],array(step,name)):raise ValueError('Source KV mutated immutable step inputs')
                expected_size=step['after'] if case['window'] is None else min(step['after'],case['window'])
                if cache.offset!=step['after'] or cache.size()!=expected_size:raise ValueError('Source KV committed offset/size differs')
                trial.update(completed=True,offset=cache.offset,size=cache.size(),allocated_rows=cache.keys.shape[2],physical_cursor=getattr(cache,'_idx',None))
            row['completed']=True
    except (Exception,KeyboardInterrupt) as error:
        primary=type(error).__name__+': '+str(error)
    finally:
        for value in previous_signals:signal.signal(value,signal.SIG_IGN)
        try:mx.set_default_device(previous_device)
        except Exception as error:cleanup.append('Restore device: '+type(error).__name__+': '+str(error))
    passed=(primary is None and not cleanup and len(rows)==len(SPECS) and all(row['completed'] for row in rows)
        and len(comparisons)==sum(len(spec[2]) for spec in SPECS) and all(item['passed'] for item in comparisons))
    report=dict(format='toklet-gemma-kv-observation-v1',passed=passed,mathematical_reference_sha256=reference_sha,
        source_revision=REVISION,source_sha256=SOURCES,admitted_runtime=admitted,requested_device=device,source_policy=dict(atol=ATOL,rtol=RTOL),
        annotation_adaptation='from __future__ import annotations; selected executable bodies unchanged',
        selected_source_definitions=[dict(file=filename,name=node.name,line=node.lineno,source=ast.unparse(node)) for filename,node in nodes],
        cases=rows,comparisons=comparisons,binary=dict(path='observed.bin',bytes=len(blob),alignment=8,sha256=hashlib.sha256(blob).hexdigest()),
        primary_error=primary,cleanup_errors=cleanup,native_graph_implemented=False,device_qualified=False,
        rollback_qualified=False,shared_kv_qualified=False,
        scope='Normal source KV updates and logical attention only; fixed-slot native transaction/rollback remains required policy')
    try:
        for name,raw in [('observed.bin',bytes(blob)),('observation.json',encoded(report))]:
            with (destination/name).open('xb') as stream:stream.write(raw)
    finally:
        for value,handler in previous_signals.items():signal.signal(value,handler)
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['export', 'compare'])
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--runtime-manifest', required=True, type=Path)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--expected-reference-sha256')
    parser.add_argument('--device', choices=['cpu', 'gpu'])
    options = parser.parse_args()
    sources = {name: runtime.captured(options.sources / name, 128 * 1024, expected) for name, expected in SOURCES.items()}
    source_nodes(sources)
    if options.mode == 'compare':
        if options.reference is None or options.expected_reference_sha256 is None or options.device is None:
            parser.error('Comparison requires explicit reference identity and device')
        verify(options.reference, options.expected_reference_sha256)
    if options.mode == 'export':
        admitted = runtime.admit_runtime(options.runtime_manifest, options.expected_runtime_sha256, [Path(__file__)])
        import numpy as np
        if np.__file__ != admitted['imports']['numpy']:
            raise ValueError('Imported NumPy differs from admitted producer')
        with options.output.open('xb') as stream:
            stream.write(export())
    else:
        passed = compare(options.reference, options.expected_reference_sha256, sources, options.device, options.output, options.runtime_manifest, options.expected_runtime_sha256)
        print(json.dumps(dict(passed=passed, mathematical_reference_only=True)))
        raise SystemExit(0 if passed else 1)
