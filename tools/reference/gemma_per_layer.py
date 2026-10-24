"""Independent Gemma per-layer lookup/projection witnesses, before composition.

FP64 mathematics and an explicitly staged FP32 contract are separate. Captured
helper outputs use real FP32 MLX operators, never BF16 checkpoint qualification.
The nearest-token input-embedding fallback is deliberately outside this scope.
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

FORMAT='toklet-gemma-per-layer-math-v1'
MAX_JSON,MAX_RAW=16*1024*1024,4*1024*1024
REVISION='0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
SOURCES={'gemma4_text.py':'73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d'}
SPECS=(
    ('empty',1,0,4,3,2,True),
    ('hand',1,2,4,2,2,True),
    ('batch',2,3,5,3,3,True),
    ('projected_only',1,2,5,3,3,False),
    ('per_layer_width_one',1,2,3,2,1,True),
    ('actual_hidden_scale',1,2,1536,2,3,True),
    ('actual_layer_input_shape',1,1,4,35,256,True),
)
ATOL,RTOL=1e-5,1e-4
EPSILON=struct.unpack('<f',struct.pack('<f',1e-6))[0]


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def geometry(batch,tokens,hidden,layers,width,with_lookup):
    if (any(type(x) is not int for x in (batch,tokens,hidden,layers,width)) or type(with_lookup) is not bool
            or not 1<=batch<=2 or not 0<=tokens<=3 or not 1<=hidden<=1536 or not 1<=layers<=35 or not 1<=width<=256
            or layers*width*hidden>40000 or batch*tokens*layers*width>18000):
        raise ValueError('Bounded per-layer input geometry required')


def inputs(batch,tokens,hidden,layers,width,with_lookup):
    geometry(batch,tokens,hidden,layers,width,with_lookup)
    import numpy as np
    def values(shape,multiplier,shift,denominator):
        return (((np.arange(math.prod(shape))*multiplier+shift)%17)-8).astype(np.float32).reshape(shape)/np.float32(denominator)
    ids=((np.arange(batch*tokens)*3+1)%4).astype(np.int32).reshape(batch,tokens)
    raw=values((batch,tokens,hidden),5,3,16)
    table=values((4,layers*width),7,2,8)
    projection=values((layers*width,hidden),3,4,32)
    norm=np.array([(.75,-.5,1.25,.375)[i%4] for i in range(width)],np.float32)
    return dict(ids=ids,raw_embedding=raw,table=table,projection_weight=projection,norm_weight=norm)


def mathematics(operands,batch,tokens,hidden,layers,width,with_lookup,staged):
    geometry(batch,tokens,hidden,layers,width,with_lookup)
    import numpy as np
    expected={'ids':((batch,tokens),np.dtype('int32')),'raw_embedding':((batch,tokens,hidden),np.dtype('float32')),
        'table':((4,layers*width),np.dtype('float32')),'projection_weight':((layers*width,hidden),np.dtype('float32')),'norm_weight':((width,),np.dtype('float32'))}
    if set(operands)!=set(expected) or type(staged) is not bool:raise ValueError('Exact per-layer operand inventory required')
    for name,(shape,dtype) in expected.items():
        value=operands[name]
        if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=dtype or not np.isfinite(value).all():raise ValueError('Finite exact per-layer operand shapes/dtypes required')
    if np.any(operands['ids']<0) or np.any(operands['ids']>=4):raise ValueError('Harness token-ID admission rejects outside lookup vocabulary')
    def rounded(value):return np.asarray(value,dtype=np.float32 if staged else np.float64)
    def scale(value):return np.float32(value) if staged else value
    raw=operands['raw_embedding'].astype(np.float64)
    h=rounded(raw*scale(math.sqrt(hidden)))
    lookup_raw=operands['table'][operands['ids']].copy()
    lookup_scaled=rounded(lookup_raw.astype(np.float64)*scale(math.sqrt(width)))
    lookup=lookup_scaled.reshape(batch,tokens,layers,width)
    projection=rounded(h.astype(np.float64) @ operands['projection_weight'].astype(np.float64).T)
    projection_scaled=rounded(projection.astype(np.float64)*scale(hidden**-.5))
    reshaped=projection_scaled.reshape(batch,tokens,layers,width)
    x=reshaped.astype(np.float64)
    normalized=rounded(x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+EPSILON)*operands['norm_weight'].astype(np.float64))
    result=dict(scaled_embedding=h,lookup_raw=lookup_raw.astype(np.float32 if staged else np.float64),lookup_scaled=lookup_scaled,lookup=lookup,
        projection=projection,projection_scaled=projection_scaled,projection_reshaped=reshaped,projection_norm=normalized)
    if with_lookup:
        combined=rounded(normalized.astype(np.float64)+lookup.astype(np.float64))
        result['combined']=combined
        output=rounded(combined.astype(np.float64)*scale(2**-.5))
    else:output=normalized.copy()
    result['output']=output
    return result


def describe(value):
    dtype='<i4' if value.dtype.kind=='i' else '<f4' if value.dtype.itemsize==4 else '<f8'
    return dict(dtype=dtype,shape=list(value.shape),values=value.reshape(-1).tolist())


def export():
    import numpy as np
    cases=[]
    for name,b,n,h,l,p,lookup in SPECS:
        operands=inputs(b,n,h,l,p,lookup)
        cases.append(dict(name=name,batch=b,tokens=n,hidden=h,layers=l,width=p,with_lookup=lookup,epsilon=EPSILON,
            inputs={key:describe(value) for key,value in operands.items()},
            math={key:describe(value) for key,value in mathematics(operands,b,n,h,l,p,lookup,False).items()},
            staged_fp32={key:describe(value) for key,value in mathematics(operands,b,n,h,l,p,lookup,True).items()}))
    return encoded(dict(format=FORMAT,source_revision=REVISION,source_sha256=SOURCES,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(numpy=np.__version__),input_boundary='Raw main embeddings; explicit sqrt(H) before model projection, without algebraic scale cancellation',
        staged_fp32_scope='FP64 products/reductions with FP32 storage/scale boundaries; separate from ideal FP64 math and BF16 checkpoint execution',
        nearest_token_fallback_qualified=False,native_composition_implemented=False,device_qualified=False,bf16_checkpoint_qualified=False,cases=cases))


def source_nodes(sources):
    raw=sources['gemma4_text.py']
    classes=[node for node in ast.parse(raw).body if isinstance(node,ast.ClassDef) and node.name=='Gemma4TextModel']
    if len(classes)!=1:raise ValueError('Unique pinned per-layer helper owner required')
    names={'_get_per_layer_inputs','_project_per_layer_inputs'}
    nodes=[node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(nodes)!=2 or {node.name for node in nodes}!=names:raise ValueError('Unique pinned per-layer helper definitions required')
    return {node.name:node for node in nodes}


def same(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,dict):return set(a)==set(b) and all(same(a[key],b[key]) for key in a)
    return a==b


def verify(path,expected_sha):
    raw=runtime.captured(path,MAX_JSON,runtime.sha(expected_sha))
    data=json.loads(raw,object_pairs_hook=runtime.json_unique,parse_constant=runtime.bad_json_number,parse_float=runtime.finite_json_number)
    metadata=dict(format=FORMAT,source_revision=REVISION,source_sha256=SOURCES,runtime=dict(numpy=runtime.VERSIONS['numpy']),
        input_boundary='Raw main embeddings; explicit sqrt(H) before model projection, without algebraic scale cancellation',
        staged_fp32_scope='FP64 products/reductions with FP32 storage/scale boundaries; separate from ideal FP64 math and BF16 checkpoint execution',
        nearest_token_fallback_qualified=False,native_composition_implemented=False,device_qualified=False,bf16_checkpoint_qualified=False)
    if (not isinstance(data,dict) or set(data)!={*metadata,'producer_sha256','cases'} or not same({key:data[key] for key in metadata},metadata)
            or not isinstance(data['cases'],list) or len(data['cases'])!=len(SPECS)):raise ValueError('Per-layer mathematical identity/scope differs')
    runtime.sha(data['producer_sha256'])
    for case,(name,b,n,h,l,p,lookup) in zip(data['cases'],SPECS):
        metadata=dict(name=name,batch=b,tokens=n,hidden=h,layers=l,width=p,with_lookup=lookup,epsilon=EPSILON)
        if not isinstance(case,dict) or set(case)!={*metadata,'inputs','math','staged_fp32'} or not same({key:case[key] for key in metadata},metadata):raise ValueError('Per-layer case inventory differs')
        geometry(b,n,h,l,p,lookup)
        shapes=dict(ids=[b,n],raw_embedding=[b,n,h],table=[4,l*p],projection_weight=[l*p,h],norm_weight=[p])
        stages=dict(scaled_embedding=[b,n,h],lookup_raw=[b,n,l*p],lookup_scaled=[b,n,l*p],lookup=[b,n,l,p],projection=[b,n,l*p],projection_scaled=[b,n,l*p],projection_reshaped=[b,n,l,p],projection_norm=[b,n,l,p],output=[b,n,l,p])
        if lookup:stages['combined']=[b,n,l,p]
        for group,expected in [('inputs',shapes),('math',stages),('staged_fp32',stages)]:
            if not isinstance(case[group],dict) or set(case[group])!=set(expected):raise ValueError('Per-layer array inventory differs')
            for key,shape in expected.items():
                item=case[group][key];dtype='<i4' if group=='inputs' and key=='ids' else '<f8' if group=='math' else '<f4'
                if not isinstance(item,dict) or set(item)!={'dtype','shape','values'} or item['dtype']!=dtype or not same(item['shape'],shape) or not isinstance(item['values'],list) or len(item['values'])!=math.prod(shape):raise ValueError('Per-layer descriptor differs')
                for value in item['values']:
                    if dtype=='<i4':
                        if type(value) is not int or not 0<=value<4:raise ValueError('Admitted integer per-layer token IDs required')
                    else:
                        if type(value) not in (int,float) or not -sys.float_info.max<=value<=sys.float_info.max or not math.isfinite(value):raise ValueError('Finite per-layer value required')
                        if dtype=='<f4':
                            try:
                                if struct.unpack('<f',struct.pack('<f',value))[0]!=value:raise ValueError('Exact FP32 per-layer value required')
                            except OverflowError as error:raise ValueError('Per-layer FP32 value overflow') from error
    return data


def compare(reference,reference_sha,sources,device,destination,runtime_manifest,runtime_sha):
    if device not in ('cpu','gpu'):raise ValueError('Explicit per-layer source device required')
    data=verify(reference,reference_sha)
    if set(sources)!=set(SOURCES) or any(not isinstance(raw,bytes) or len(raw)>128*1024 or hashlib.sha256(raw).hexdigest()!=SOURCES[name] for name,raw in sources.items()):raise ValueError('Authenticated per-layer sources differ')
    nodes=source_nodes(sources)
    admitted=runtime.admit_runtime(runtime_manifest,runtime_sha,[Path(__file__)])
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from types import SimpleNamespace
    if (np.__file__!=admitted['imports']['numpy'] or mx.__file__!=admitted['imports']['mlx.core'] or nn.__file__!=admitted['imports']['mlx.nn']):raise ValueError('Actual per-layer imports differ from admitted files')
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):raise ValueError('Empty regular per-layer output required')
    previous_device=mx.default_device();previous_signals={};blob=bytearray();rows=[];comparisons=[];cleanup=[];primary=None
    def interrupted(number,_):
        for value in (signal.SIGTERM,signal.SIGINT):signal.signal(value,signal.SIG_IGN)
        raise InterruptedError('Received signal '+str(number))
    def array(case,group,name):
        item=case[group][name];return np.array(item['values'],dtype=item['dtype']).reshape(item['shape'])
    def observe(row,case,name,value,expected_name):
        mx.eval(value);actual=np.asarray(value)
        if actual.dtype.hasobject or actual.dtype.kind not in 'biufc' or actual.nbytes>MAX_RAW:
            row['arrays'][name]=dict(raw_unavailable='Unsupported numeric observation',dtype=str(actual.dtype),shape=list(actual.shape))
            raise ValueError('Bounded numeric per-layer observation required')
        dtype=actual.dtype.newbyteorder('<');raw=np.ascontiguousarray(actual,dtype=dtype).tobytes();padding=(-len(blob))%8
        if len(blob)+padding+len(raw)>MAX_RAW:raise ValueError('Per-layer raw observation bound exceeded')
        blob.extend(b'\0'*padding);row['arrays'][name]=dict(dtype=dtype.str,shape=list(actual.shape),offset=len(blob),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());blob.extend(raw)
        expected=array(case,'math',expected_name)
        if actual.dtype!=np.dtype('float32') or actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('Per-layer dtype/shape/finite observation differs')
        for contract in ('math','staged_fp32'):
            expected=array(case,contract,expected_name).astype(np.float64);difference=np.abs(actual.astype(np.float64)-expected);ratio=difference/(ATOL+RTOL*np.abs(expected))
            comparisons.append(dict(case=case['name'],name=name,contract=contract,passed=bool(np.all(ratio<=1)),maximum_absolute_error=float(difference.max()),maximum_error_to_policy_ratio=float(ratio.max())))
        return value
    try:
        for value in (signal.SIGTERM,signal.SIGINT):previous_signals[value]=signal.signal(value,interrupted)
        mx.set_default_device(mx.cpu if device=='cpu' else mx.gpu)
        environment=dict(mx=mx)
        deferred=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[deferred,*nodes.values()],type_ignores=[])),'authenticated-per-layer-selected-source','exec'),environment)
        for case in data['cases']:
            row=dict(name=case['name'],completed=False,upstream_helpers_evaluated=False,arrays={});rows.append(row)
            if case['tokens']==0:
                row.update(completed=True,reason='Empty sequence is mathematical no-call policy');continue
            b,n,h,l,p=[case[key] for key in ('batch','tokens','hidden','layers','width')]
            obj=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=l),hidden_size_per_layer_input=p,
                embed_tokens_per_layer=nn.Embedding(4,l*p),embed_tokens_per_layer_scale=math.sqrt(p),
                per_layer_model_projection=nn.Linear(h,l*p,bias=False),per_layer_projection_scale=h**-.5,
                per_layer_projection_norm=nn.RMSNorm(p,eps=EPSILON),per_layer_input_scale=2**-.5)
            obj.embed_tokens_per_layer.weight=mx.array(array(case,'inputs','table'))
            obj.per_layer_model_projection.weight=mx.array(array(case,'inputs','projection_weight'))
            obj.per_layer_projection_norm.weight=mx.array(array(case,'inputs','norm_weight'))
            ids=mx.array(array(case,'inputs','ids'));raw=mx.array(array(case,'inputs','raw_embedding'))
            # These independently evaluated stage probes are deliberately separate
            # from the unchanged helper calls below. They do not qualify lazy
            # graph fusion or represent internal captures of those helper calls.
            scaled=observe(row,case,'scaled_embedding',raw*math.sqrt(h),'scaled_embedding')
            lookup_raw=observe(row,case,'lookup_raw',obj.embed_tokens_per_layer(ids),'lookup_raw')
            lookup_scaled=observe(row,case,'lookup_scaled',lookup_raw*obj.embed_tokens_per_layer_scale,'lookup_scaled')
            lookup=observe(row,case,'lookup',mx.unflatten(lookup_scaled,-1,(l,p)),'lookup')
            projected=observe(row,case,'projection',obj.per_layer_model_projection(scaled),'projection')
            projection_scaled=observe(row,case,'projection_scaled',projected*obj.per_layer_projection_scale,'projection_scaled')
            reshaped=observe(row,case,'projection_reshaped',mx.unflatten(projection_scaled,-1,(l,p)),'projection_reshaped')
            normalized=observe(row,case,'projection_norm',obj.per_layer_projection_norm(reshaped),'projection_norm')
            if case['with_lookup']:
                combined=observe(row,case,'combined',normalized+lookup,'combined')
                observe(row,case,'output',combined*obj.per_layer_input_scale,'output')
            else:observe(row,case,'output',normalized,'output')
            source_lookup=observe(row,case,'source_lookup',environment['_get_per_layer_inputs'](obj,ids),'lookup')
            observe(row,case,'source_output',environment['_project_per_layer_inputs'](obj,scaled,source_lookup if case['with_lookup'] else None),'output')
            for value,name in [(ids,'ids'),(raw,'raw_embedding'),(obj.embed_tokens_per_layer.weight,'table'),(obj.per_layer_model_projection.weight,'projection_weight'),(obj.per_layer_projection_norm.weight,'norm_weight')]:
                if not np.array_equal(np.asarray(value),array(case,'inputs',name)):raise ValueError('Per-layer source mutated immutable operands')
            row.update(completed=True,upstream_helpers_evaluated=True)
    except (Exception,KeyboardInterrupt) as error:primary=type(error).__name__+': '+str(error)
    finally:
        for value in previous_signals:signal.signal(value,signal.SIG_IGN)
        try:mx.set_default_device(previous_device)
        except Exception as error:cleanup.append('Restore device: '+type(error).__name__+': '+str(error))
    expected_comparisons=sum(2*(len(case['math'])+2) for case in data['cases'] if case['tokens'])
    passed=primary is None and not cleanup and len(rows)==len(SPECS) and all(row['completed'] for row in rows) and len(comparisons)==expected_comparisons and all(item['passed'] for item in comparisons)
    report=dict(format='toklet-gemma-per-layer-observation-v1',passed=passed,mathematical_reference_sha256=reference_sha,
        source_revision=REVISION,source_sha256=SOURCES,admitted_runtime=admitted,requested_device=device,source_policy=dict(atol=ATOL,rtol=RTOL),
        selected_source_definitions={name:dict(line=node.lineno,source=ast.unparse(node)) for name,node in nodes.items()},
        annotation_adaptation='from __future__ import annotations; selected helper executable bodies unchanged',
        observation_scope='Separately evaluated real FP32 operator stages plus unchanged helper outputs; no internal-stage or fusion qualification of helper execution',
        cases=rows,comparisons=comparisons,binary=dict(path='observed.bin',bytes=len(blob),alignment=8,sha256=hashlib.sha256(blob).hexdigest()),
        primary_error=primary,cleanup_errors=cleanup,nearest_token_fallback_qualified=False,native_composition_implemented=False,device_qualified=False,bf16_checkpoint_qualified=False)
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
