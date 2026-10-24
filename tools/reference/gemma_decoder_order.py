"""Independent Gemma dense decoder order with an external attention boundary.

Two mathematical contracts precede native composition. Real FP32 source calls
use a supplied attention result: no attention, KV, rotary or model qualification.
"""
import argparse
import ast
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import sys

import reference_runtime as runtime

FORMAT='toklet-gemma-decoder-order-math-v1'
MAX_JSON,MAX_RAW=16*1024*1024,4*1024*1024
REVISION='0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
SOURCES={'gemma4_text.py':'73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d'}
SPECS=(
    ('empty',1,0,4,6,2,'full',True),
    ('hand',1,1,4,6,2,'full',True),
    ('batch',2,3,5,7,3,'full',True),
    ('double_width_shape',1,2,4,12,3,'full',True),
    ('input_absent',1,2,5,7,3,'no_input',True),
    ('gate_absent',1,2,5,7,3,'no_gate',True),
    ('projection_absent',1,2,5,7,3,'no_projection',True),
    ('norm_absent',1,2,5,7,3,'no_norm',True),
    ('ple_disabled',1,2,5,7,3,'disabled',True),
    ('scalar_absent_synthetic',1,2,5,7,3,'full',False),
    ('actual_hidden_norm',1,1,1536,3,2,'full',True),
    ('actual_ple_width',1,2,4,5,256,'full',True),
)
ATOL,RTOL=1e-5,1e-4
EPSILON=struct.unpack('<f',struct.pack('<f',1e-6))[0]
MODES={'full','no_input','no_gate','no_projection','no_norm','disabled'}
NORMS=('input_norm','post_attention_norm','pre_ff_norm','post_ff_norm','post_ple_norm')
SCOPE='Dense DecoderLayer/MLP/geglu calls with real FP32 modules and externally supplied attention; separate stage probes do not capture helper internals or fusion'


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def geometry(batch,tokens,hidden,intermediate,width,ple_mode,with_scalar):
    if (any(type(x) is not int for x in (batch,tokens,hidden,intermediate,width)) or type(with_scalar) is not bool
            or type(ple_mode) is not str or ple_mode not in MODES
            or not 1<=batch<=2 or not 0<=tokens<=3 or not 1<=hidden<=1536 or not 1<=intermediate<=12 or not 1<=width<=256
            or hidden*(intermediate+width)>40000 or batch*tokens*hidden>18000):
        raise ValueError('Bounded decoder-order geometry required')


def shapes(b,n,h,i,p):
    result=dict(hidden=[b,n,h],attention_output=[b,n,h],per_layer_input=[b,n,p],gate_weight=[i,h],up_weight=[i,h],down_weight=[h,i],ple_gate_weight=[p,h],ple_projection_weight=[h,p],layer_scalar=[1])
    result.update({key:[h] for key in NORMS});return result


def stages(b,n,h,i,p,mode):
    result={key:[b,n,h] for key in ('attention_input','attention_norm','attention_residual','ff_input','ff_down','ff_norm','ff_residual','output')}
    result.update({key:[b,n,i] for key in ('ff_gate','ff_up','ff_activated','ff_product')})
    if mode=='full':
        result.update({key:[b,n,p] for key in ('ple_gate','ple_activated','ple_product')})
        result.update({key:[b,n,h] for key in ('ple_projected','ple_norm','ple_residual')})
    return result


def inputs(batch,tokens,hidden,intermediate,width,ple_mode,with_scalar):
    geometry(batch,tokens,hidden,intermediate,width,ple_mode,with_scalar)
    import numpy as np
    result={}
    for j,(key,shape) in enumerate(shapes(batch,tokens,hidden,intermediate,width).items()):
        result[key]=((((np.arange(math.prod(shape))*(2*j+3)+j+1)%19)-9).astype(np.float32)/np.float32(16 if 'weight' in key else 8)).reshape(shape)
    for j,key in enumerate(NORMS):result[key]=np.array([(.75,-.5,1.25,.375)[(v+j)%4] for v in range(hidden)],np.float32)
    result['layer_scalar']=np.array([-1.25],np.float32)
    return result


def mathematics(operands,batch,tokens,hidden,intermediate,width,ple_mode,with_scalar,staged):
    geometry(batch,tokens,hidden,intermediate,width,ple_mode,with_scalar)
    import numpy as np
    expected=shapes(batch,tokens,hidden,intermediate,width)
    if set(operands)!=set(expected) or type(staged) is not bool:raise ValueError('Exact decoder-order operand inventory required')
    for key,shape in expected.items():
        x=operands[key]
        if not isinstance(x,np.ndarray) or x.dtype!=np.dtype('float32') or list(x.shape)!=shape or not np.isfinite(x).all():raise ValueError('Finite exact decoder-order shapes/dtypes required')
    def rounded(x):return np.asarray(x,dtype=np.float32 if staged else np.float64)
    def linear(x,key):return rounded(x.astype(np.float64)@operands[key].astype(np.float64).T)
    def norm(x,key):
        y=x.astype(np.float64);return rounded(y/np.sqrt(np.mean(y*y,-1,keepdims=True)+EPSILON)*operands[key].astype(np.float64))
    def gelu(x):
        y=x.astype(np.float64);return rounded(.5*y*(1+np.tanh(math.sqrt(2/math.pi)*(y+.044715*y**3))))
    def add(x,y):return rounded(x.astype(np.float64)+y.astype(np.float64))
    def multiply(x,y):return rounded(x.astype(np.float64)*y.astype(np.float64))
    r={};r['attention_input']=norm(operands['hidden'],'input_norm')
    r['attention_norm']=norm(operands['attention_output'],'post_attention_norm')
    r['attention_residual']=add(operands['hidden'],r['attention_norm'])
    r['ff_input']=norm(r['attention_residual'],'pre_ff_norm')
    r['ff_gate']=linear(r['ff_input'],'gate_weight');r['ff_up']=linear(r['ff_input'],'up_weight')
    r['ff_activated']=gelu(r['ff_gate']);r['ff_product']=multiply(r['ff_activated'],r['ff_up'])
    r['ff_down']=linear(r['ff_product'],'down_weight');r['ff_norm']=norm(r['ff_down'],'post_ff_norm')
    r['ff_residual']=add(r['attention_residual'],r['ff_norm']);x=r['ff_residual']
    if ple_mode=='full':
        r['ple_gate']=linear(x,'ple_gate_weight');r['ple_activated']=gelu(r['ple_gate'])
        r['ple_product']=multiply(r['ple_activated'],operands['per_layer_input'])
        r['ple_projected']=linear(r['ple_product'],'ple_projection_weight');r['ple_norm']=norm(r['ple_projected'],'post_ple_norm')
        r['ple_residual']=add(x,r['ple_norm']);x=r['ple_residual']
    r['output']=multiply(x,operands['layer_scalar']) if with_scalar else x.copy()
    return r


def describe(value):return dict(dtype='<f4' if value.dtype.itemsize==4 else '<f8',shape=list(value.shape),values=value.reshape(-1).tolist())


def metadata():
    return dict(format=FORMAT,source_revision=REVISION,source_sha256=SOURCES,runtime=dict(numpy=runtime.VERSIONS['numpy']),
        input_boundary='Hidden state plus supplied attention output and already composed per-layer input',
        staged_fp32_scope='FP64 linear/reduction/tanh arithmetic; FP32 storage after each operator and residual/product boundary',
        observation_scope=SCOPE,native_composition_implemented=False,device_qualified=False,bf16_checkpoint_qualified=False,
        attention_kv_rotary_qualified=False,constructor_or_double_width_selection_qualified=False,scalar_absent_is_synthetic=True)


def export():
    cases=[]
    for name,b,n,h,i,p,mode,scalar in SPECS:
        x=inputs(b,n,h,i,p,mode,scalar)
        cases.append(dict(name=name,batch=b,tokens=n,hidden=h,intermediate=i,width=p,ple_mode=mode,with_scalar=scalar,epsilon=EPSILON,
            inputs={k:describe(v) for k,v in x.items()},math={k:describe(v) for k,v in mathematics(x,b,n,h,i,p,mode,scalar,False).items()},
            staged_fp32={k:describe(v) for k,v in mathematics(x,b,n,h,i,p,mode,scalar,True).items()}))
    return encoded(dict(**metadata(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases))


def source_nodes(sources):
    tree=ast.parse(sources['gemma4_text.py']);result={}
    for name in ('MLP','DecoderLayer'):
        classes=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==name]
        if len(classes)!=1:raise ValueError('Unique pinned decoder-order source owner required')
        methods=[x for x in classes[0].body if isinstance(x,ast.FunctionDef) and x.name=='__call__']
        if len(methods)!=1:raise ValueError('Unique pinned decoder-order call required')
        result[name]=methods[0]
    functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='geglu']
    if len(functions)!=1:raise ValueError('Unique pinned geglu source required')
    result['geglu']=functions[0];return result


def same(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,dict):return set(a)==set(b) and all(same(a[key],b[key]) for key in a)
    return a==b


def verify(path,expected_sha):
    raw=runtime.captured(path,MAX_JSON,runtime.sha(expected_sha))
    data=json.loads(raw,object_pairs_hook=runtime.json_unique,parse_constant=runtime.bad_json_number,parse_float=runtime.finite_json_number)
    expected=metadata()
    if (not isinstance(data,dict) or set(data)!={*expected,'producer_sha256','cases'} or not same({k:data[k] for k in expected},expected)
            or not isinstance(data['cases'],list) or len(data['cases'])!=len(SPECS)):raise ValueError('Decoder-order mathematical identity/scope differs')
    runtime.sha(data['producer_sha256'])
    for case,(name,b,n,h,i,p,mode,scalar) in zip(data['cases'],SPECS):
        wanted=dict(name=name,batch=b,tokens=n,hidden=h,intermediate=i,width=p,ple_mode=mode,with_scalar=scalar,epsilon=EPSILON)
        if not isinstance(case,dict) or set(case)!={*wanted,'inputs','math','staged_fp32'} or not same({k:case[k] for k in wanted},wanted):raise ValueError('Decoder-order case inventory differs')
        geometry(b,n,h,i,p,mode,scalar)
        for group,inventory in [('inputs',shapes(b,n,h,i,p)),('math',stages(b,n,h,i,p,mode)),('staged_fp32',stages(b,n,h,i,p,mode))]:
            if not isinstance(case[group],dict) or set(case[group])!=set(inventory):raise ValueError('Decoder-order array inventory differs')
            for key,shape in inventory.items():
                item=case[group][key];dtype='<f8' if group=='math' else '<f4'
                if not isinstance(item,dict) or set(item)!={'dtype','shape','values'} or item['dtype']!=dtype or not same(item['shape'],shape) or not isinstance(item['values'],list) or len(item['values'])!=math.prod(shape):raise ValueError('Decoder-order descriptor differs')
                for value in item['values']:
                    if type(value) not in (int,float) or not -sys.float_info.max<=value<=sys.float_info.max or not math.isfinite(value):raise ValueError('Finite decoder-order value required')
                    if dtype=='<f4':
                        try:
                            if struct.unpack('<f',struct.pack('<f',value))[0]!=value:raise ValueError('Exact FP32 decoder-order value required')
                        except OverflowError as error:raise ValueError('Decoder-order FP32 overflow') from error
    return data


def compare(reference,reference_sha,sources,device,destination,runtime_manifest,runtime_sha):
    if device not in ('cpu','gpu'):raise ValueError('Explicit decoder-order source device required')
    data=verify(reference,reference_sha)
    if set(sources)!=set(SOURCES) or any(not isinstance(raw,bytes) or len(raw)>128*1024 or hashlib.sha256(raw).hexdigest()!=SOURCES[name] for name,raw in sources.items()):raise ValueError('Authenticated decoder-order sources differ')
    nodes=source_nodes(sources)
    admitted=runtime.admit_runtime(runtime_manifest,runtime_sha,[Path(__file__)])
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from types import SimpleNamespace
    if (np.__file__!=admitted['imports']['numpy'] or mx.__file__!=admitted['imports']['mlx.core'] or nn.__file__!=admitted['imports']['mlx.nn']):raise ValueError('Actual decoder-order imports differ from admitted files')
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):raise ValueError('Empty regular decoder-order output required')
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
            raise ValueError('Bounded numeric decoder-order observation required')
        dtype=actual.dtype.newbyteorder('<');raw=np.ascontiguousarray(actual,dtype=dtype).tobytes();padding=(-len(blob))%8
        if len(blob)+padding+len(raw)>MAX_RAW:raise ValueError('Decoder-order raw observation bound exceeded')
        blob.extend(b'\0'*padding);row['arrays'][name]=dict(dtype=dtype.str,shape=list(actual.shape),offset=len(blob),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());blob.extend(raw)
        expected=array(case,'math',expected_name)
        if actual.dtype!=np.dtype('float32') or actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('Decoder-order dtype/shape/finite observation differs')
        for contract in ('math','staged_fp32'):
            expected=array(case,contract,expected_name).astype(np.float64);difference=np.abs(actual.astype(np.float64)-expected);ratio=difference/(ATOL+RTOL*np.abs(expected))
            comparisons.append(dict(case=case['name'],name=name,contract=contract,passed=bool(np.all(ratio<=1)),maximum_absolute_error=float(difference.max()),maximum_error_to_policy_ratio=float(ratio.max())))
        return value
    try:
        for value in (signal.SIGTERM,signal.SIGINT):previous_signals[value]=signal.signal(value,interrupted)
        mx.set_default_device(mx.cpu if device=='cpu' else mx.gpu)
        environment=dict(mx=mx,nn=nn,partial=partial)
        deferred=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        for key in ('geglu','MLP','DecoderLayer'):
            node=nodes[key]
            exec(compile(ast.fix_missing_locations(ast.Module(body=[deferred,node],type_ignores=[])),'authenticated-decoder-order-selected-source','exec'),environment)
            environment[key+'_call']=environment[node.name]
        for case in data['cases']:
            row=dict(name=case['name'],completed=False,upstream_helpers_evaluated=False,attention_forwarding_checked=False,arrays={});rows.append(row)
            if case['tokens']==0:
                row.update(completed=True,reason='Empty sequence is mathematical no-call policy');continue
            b,n,h,i,p=[case[key] for key in ('batch','tokens','hidden','intermediate','width')]
            mode=case['ple_mode']
            def linear(key):
                weight=mx.array(array(case,'inputs',key));obj=nn.Linear(weight.shape[1],weight.shape[0],bias=False);obj.weight=weight;return obj
            def norm(key):
                obj=nn.RMSNorm(h,eps=EPSILON);obj.weight=mx.array(array(case,'inputs',key));return obj
            mlp=SimpleNamespace(gate_proj=linear('gate_weight'),up_proj=linear('up_weight'),down_proj=linear('down_weight'))
            def mlp_call(x):return environment['MLP_call'](mlp,x)
            x=mx.array(array(case,'inputs','hidden'));attention=mx.array(array(case,'inputs','attention_output'));ple=mx.array(array(case,'inputs','per_layer_input'))
            mask,cache,shared,offset=object(),object(),(object(),object()),object()
            returned_shared,returned_offset=(object(),object()),object();calls=[]
            def attention_call(value,actual_mask,actual_cache,shared_kv,offset):
                if actual_mask is not mask or actual_cache is not cache or shared_kv is not shared or offset is not incoming_offset:raise ValueError('Attention arguments not forwarded unchanged')
                calls.append(value);observe(row,case,'source_attention_input',value,'attention_input')
                return attention,returned_shared,returned_offset
            incoming_offset=offset
            obj=SimpleNamespace(enable_moe=False,self_attn=attention_call,mlp=mlp_call,
                input_layernorm=norm('input_norm'),post_attention_layernorm=norm('post_attention_norm'),
                pre_feedforward_layernorm=norm('pre_ff_norm'),post_feedforward_layernorm=norm('post_ff_norm'),
                per_layer_input_gate=None if mode in ('no_gate','disabled') else linear('ple_gate_weight'),
                per_layer_projection=None if mode in ('no_projection','disabled') else linear('ple_projection_weight'),
                post_per_layer_input_norm=None if mode in ('no_norm','disabled') else norm('post_ple_norm'),
                layer_scalar=mx.array(array(case,'inputs','layer_scalar')) if case['with_scalar'] else None)
            # Probes below are independently evaluated; helper lazy/fused
            # execution remains separate and no internal stage capture is claimed.
            observe(row,case,'attention_input',obj.input_layernorm(x),'attention_input')
            a=observe(row,case,'attention_norm',obj.post_attention_layernorm(attention),'attention_norm')
            a=observe(row,case,'attention_residual',x+a,'attention_residual')
            z=observe(row,case,'ff_input',obj.pre_feedforward_layernorm(a),'ff_input')
            gate=observe(row,case,'ff_gate',mlp.gate_proj(z),'ff_gate');up=observe(row,case,'ff_up',mlp.up_proj(z),'ff_up')
            active=observe(row,case,'ff_activated',nn.gelu_approx(gate),'ff_activated')
            product=observe(row,case,'ff_product',active*up,'ff_product')
            down=observe(row,case,'ff_down',mlp.down_proj(product),'ff_down')
            normalized=observe(row,case,'ff_norm',obj.post_feedforward_layernorm(down),'ff_norm')
            z=observe(row,case,'ff_residual',a+normalized,'ff_residual')
            if mode=='full':
                gate=observe(row,case,'ple_gate',obj.per_layer_input_gate(z),'ple_gate')
                gate=observe(row,case,'ple_activated',nn.gelu_approx(gate),'ple_activated')
                gate=observe(row,case,'ple_product',mx.multiply(gate,ple),'ple_product')
                gate=observe(row,case,'ple_projected',obj.per_layer_projection(gate),'ple_projected')
                gate=observe(row,case,'ple_norm',obj.post_per_layer_input_norm(gate),'ple_norm')
                z=observe(row,case,'ple_residual',z+gate,'ple_residual')
            observe(row,case,'output',z*obj.layer_scalar if obj.layer_scalar is not None else z,'output')
            observe(row,case,'source_mlp_output',environment['MLP_call'](mlp,obj.pre_feedforward_layernorm(a)),'ff_down')
            output,actual_shared,actual_offset=environment['DecoderLayer_call'](obj,x,mask,cache,per_layer_input=None if mode=='no_input' else ple,shared_kv=shared,offset=offset)
            observe(row,case,'source_output',output,'output')
            if len(calls)!=1 or actual_shared is not returned_shared or actual_offset is not returned_offset:raise ValueError('Attention result tuple not forwarded unchanged')
            # All numerical inputs/weights are immutable, including unused guard operands.
            observed=dict(hidden=x,attention_output=attention,per_layer_input=ple,gate_weight=mlp.gate_proj.weight,up_weight=mlp.up_proj.weight,down_weight=mlp.down_proj.weight,
                input_norm=obj.input_layernorm.weight,post_attention_norm=obj.post_attention_layernorm.weight,pre_ff_norm=obj.pre_feedforward_layernorm.weight,post_ff_norm=obj.post_feedforward_layernorm.weight)
            if obj.per_layer_input_gate is not None:observed['ple_gate_weight']=obj.per_layer_input_gate.weight
            if obj.per_layer_projection is not None:observed['ple_projection_weight']=obj.per_layer_projection.weight
            if obj.post_per_layer_input_norm is not None:observed['post_ple_norm']=obj.post_per_layer_input_norm.weight
            if obj.layer_scalar is not None:observed['layer_scalar']=obj.layer_scalar
            for key,value in observed.items():
                if not np.array_equal(np.asarray(value),array(case,'inputs',key)):raise ValueError('Decoder-order immutable operand changed')
            row.update(completed=True,upstream_helpers_evaluated=True,attention_forwarding_checked=True)
    except (Exception,KeyboardInterrupt) as error:primary=type(error).__name__+': '+str(error)
    finally:
        for value in previous_signals:signal.signal(value,signal.SIG_IGN)
        try:mx.set_default_device(previous_device)
        except Exception as error:cleanup.append('Restore device: '+type(error).__name__+': '+str(error))
    expected_comparisons=sum(2*(len(case['math'])+3) for case in data['cases'] if case['tokens'])
    passed=primary is None and not cleanup and len(rows)==len(SPECS) and all(row['completed'] for row in rows) and len(comparisons)==expected_comparisons and all(item['passed'] for item in comparisons)
    report=dict(format='toklet-gemma-decoder-order-observation-v1',passed=passed,mathematical_reference_sha256=reference_sha,
        source_revision=REVISION,source_sha256=SOURCES,admitted_runtime=admitted,requested_device=device,source_policy=dict(atol=ATOL,rtol=RTOL),
        selected_source_definitions={name:dict(line=node.lineno,source=ast.unparse(node)) for name,node in nodes.items()},
        annotation_adaptation='Deferred annotations; selected executable bodies and compiled geglu decorator unchanged',
        observation_scope=SCOPE,cases=rows,comparisons=comparisons,binary=dict(path='observed.bin',bytes=len(blob),alignment=8,sha256=hashlib.sha256(blob).hexdigest()),
        primary_error=primary,cleanup_errors=cleanup,native_composition_implemented=False,device_qualified=False,bf16_checkpoint_qualified=False,
        attention_kv_rotary_qualified=False,constructor_or_double_width_selection_qualified=False,scalar_absent_is_synthetic=True)
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
