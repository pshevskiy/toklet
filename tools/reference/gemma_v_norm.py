"""Independent Gemma V normalization without learned scale.

Each [B,T,Hkv,D] row normalizes only its final D dimension. This witness
precedes native mapping, with no V projection, KV storage or attention claim.
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

FORMAT='toklet-gemma-v-norm-math-v1'
MAX_JSON,MAX_RAW=1024*1024,1024*1024
REVISION='0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
SOURCES={'gemma4_text.py':'73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d'}
SPECS=(('empty',1,0,2,4),('hand',1,1,2,4),('batch',2,3,3,5),('width_one',1,2,2,1),('zeros',1,2,2,7),('epsilon_dominated',1,2,2,4),('local_width',1,3,1,256),('global_width',1,3,1,512))
ATOL,RTOL=1e-5,1e-4
EPSILON=struct.unpack('<f',struct.pack('<f',1e-6))[0]
SCOPE='Unchanged RMSNormNoScale call with real FP32 mx.fast.rms_norm(x,None,eps), per final D dimension; no constructor/attention/KV qualification'


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def geometry(batch,tokens,heads,width):
    if (any(type(x) is not int for x in (batch,tokens,heads,width)) or not 1<=batch<=2 or not 0<=tokens<=3 or not 1<=heads<=3 or not 1<=width<=512 or batch*tokens*heads*width>10000):raise ValueError('Bounded V-norm geometry required')


def inputs(name,batch,tokens,heads,width):
    geometry(batch,tokens,heads,width)
    import numpy as np
    if name not in {row[0] for row in SPECS}:raise ValueError('Known V-norm pattern required')
    x=((((np.arange(batch*tokens*heads*width)*7+3)%23)-11).astype(np.float32)/np.float32(8)).reshape(batch,tokens,heads,width)
    if name=='hand':
        if (batch,tokens,heads,width)!=(1,1,2,4):raise ValueError('Exact hand V geometry required')
        x=np.array([[[[2.,0.,0.,0.],[0.,0.,-3.,0.]]]],np.float32)
    elif name=='zeros':x.fill(0)
    elif name=='epsilon_dominated':x=x/np.float32(1048576)
    return x


def mathematics(x,batch,tokens,heads,width,staged):
    geometry(batch,tokens,heads,width)
    import numpy as np
    if type(staged) is not bool or not isinstance(x,np.ndarray) or x.dtype!=np.dtype('float32') or x.shape!=(batch,tokens,heads,width) or not np.isfinite(x).all():raise ValueError('Finite exact V-norm input required')
    y=x.astype(np.float64);output=y/np.sqrt(np.mean(y*y,-1,keepdims=True)+EPSILON)
    return output.astype(np.float32 if staged else np.float64)


def describe(x):return dict(dtype='<f4' if x.dtype.itemsize==4 else '<f8',shape=list(x.shape),values=x.reshape(-1).tolist())


def metadata():
    return dict(format=FORMAT,source_revision=REVISION,source_sha256=SOURCES,runtime=dict(numpy=runtime.VERSIONS['numpy']),
        precision='FP32 inputs; independent FP64 per-D mean-square RMS, staged contract rounds output once to FP32',observation_scope=SCOPE,
        native_mapping_qualified=False,device_qualified=False,bf16_checkpoint_qualified=False,attention_or_kv_qualified=False,constructor_qualified=False)


def export():
    cases=[]
    for name,b,n,h,d in SPECS:
        x=inputs(name,b,n,h,d)
        cases.append(dict(name=name,batch=b,tokens=n,heads=h,width=d,epsilon=EPSILON,input=describe(x),math=describe(mathematics(x,b,n,h,d,False)),staged_fp32=describe(mathematics(x,b,n,h,d,True))))
    return encoded(dict(**metadata(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases))


def source_nodes(sources):
    classes=[x for x in ast.parse(sources['gemma4_text.py']).body if isinstance(x,ast.ClassDef) and x.name=='RMSNormNoScale']
    if len(classes)!=1:raise ValueError('Unique pinned V-norm owner required')
    calls=[x for x in classes[0].body if isinstance(x,ast.FunctionDef) and x.name=='__call__']
    if len(calls)!=1:raise ValueError('Unique pinned V-norm call required')
    return {'RMSNormNoScale':calls[0]}


def same(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,dict):return set(a)==set(b) and all(same(a[key],b[key]) for key in a)
    return a==b


def verify(path,expected_sha):
    raw=runtime.captured(path,MAX_JSON,runtime.sha(expected_sha))
    data=json.loads(raw,object_pairs_hook=runtime.json_unique,parse_constant=runtime.bad_json_number,parse_float=runtime.finite_json_number)
    wanted=metadata()
    if (not isinstance(data,dict) or set(data)!={*wanted,'producer_sha256','cases'} or not same({k:data[k] for k in wanted},wanted)
            or not isinstance(data['cases'],list) or len(data['cases'])!=len(SPECS)):raise ValueError('V-norm mathematical identity/scope differs')
    runtime.sha(data['producer_sha256'])
    for case,(name,b,n,h,d) in zip(data['cases'],SPECS):
        wanted=dict(name=name,batch=b,tokens=n,heads=h,width=d,epsilon=EPSILON)
        if not isinstance(case,dict) or set(case)!={*wanted,'input','math','staged_fp32'} or not same({k:case[k] for k in wanted},wanted):raise ValueError('V-norm case inventory differs')
        geometry(b,n,h,d)
        for key in ('input','math','staged_fp32'):
            item=case[key];dtype='<f8' if key=='math' else '<f4'
            if not isinstance(item,dict) or set(item)!={'dtype','shape','values'} or item['dtype']!=dtype or not same(item['shape'],[b,n,h,d]) or not isinstance(item['values'],list) or len(item['values'])!=b*n*h*d:raise ValueError('V-norm descriptor differs')
            for value in item['values']:
                if type(value) not in (int,float) or not -sys.float_info.max<=value<=sys.float_info.max or not math.isfinite(value):raise ValueError('Finite V-norm value required')
                if dtype=='<f4':
                    try:
                        if struct.unpack('<f',struct.pack('<f',value))[0]!=value:raise ValueError('Exact FP32 V-norm value required')
                    except OverflowError as error:raise ValueError('V-norm FP32 overflow') from error
    return data


def compare(reference,reference_sha,sources,device,destination,runtime_manifest,runtime_sha):
    if device not in ('cpu','gpu'):raise ValueError('Explicit V-norm source device required')
    data=verify(reference,reference_sha)
    if set(sources)!=set(SOURCES) or any(not isinstance(raw,bytes) or len(raw)>128*1024 or hashlib.sha256(raw).hexdigest()!=SOURCES[name] for name,raw in sources.items()):raise ValueError('Authenticated V-norm sources differ')
    nodes=source_nodes(sources)
    admitted=runtime.admit_runtime(runtime_manifest,runtime_sha,[Path(__file__)])
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from types import SimpleNamespace
    if (np.__file__!=admitted['imports']['numpy'] or mx.__file__!=admitted['imports']['mlx.core'] or nn.__file__!=admitted['imports']['mlx.nn']):raise ValueError('Actual V-norm imports differ from admitted files')
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):raise ValueError('Empty regular V-norm output required')
    previous_device=mx.default_device();previous_signals={};blob=bytearray();rows=[];comparisons=[];cleanup=[];primary=None
    def interrupted(number,_):
        for value in (signal.SIGTERM,signal.SIGINT):signal.signal(value,signal.SIG_IGN)
        raise InterruptedError('Received signal '+str(number))
    def array(case,group):
        item=case[group];return np.array(item['values'],dtype=item['dtype']).reshape(item['shape'])
    def observe(row,case,name,value):
        mx.eval(value);actual=np.asarray(value)
        if actual.dtype.hasobject or actual.dtype.kind not in 'biufc' or actual.nbytes>MAX_RAW:
            row['arrays'][name]=dict(raw_unavailable='Unsupported numeric observation',dtype=str(actual.dtype),shape=list(actual.shape))
            raise ValueError('Bounded numeric V-norm observation required')
        dtype=actual.dtype.newbyteorder('<');raw=np.ascontiguousarray(actual,dtype=dtype).tobytes();padding=(-len(blob))%8
        if len(blob)+padding+len(raw)>MAX_RAW:raise ValueError('V-norm raw observation bound exceeded')
        blob.extend(b'\0'*padding);row['arrays'][name]=dict(dtype=dtype.str,shape=list(actual.shape),offset=len(blob),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());blob.extend(raw)
        expected=array(case,'math')
        if actual.dtype!=np.dtype('float32') or actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('V-norm dtype/shape/finite observation differs')
        for contract in ('math','staged_fp32'):
            expected=array(case,contract).astype(np.float64);difference=np.abs(actual.astype(np.float64)-expected);ratio=difference/(ATOL+RTOL*np.abs(expected))
            comparisons.append(dict(case=case['name'],name=name,contract=contract,passed=bool(np.all(ratio<=1)),maximum_absolute_error=float(difference.max()),maximum_error_to_policy_ratio=float(ratio.max())))
        return value
    try:
        for value in (signal.SIGTERM,signal.SIGINT):previous_signals[value]=signal.signal(value,interrupted)
        mx.set_default_device(mx.cpu if device=='cpu' else mx.gpu)
        environment=dict(mx=mx)
        deferred=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[deferred,*nodes.values()],type_ignores=[])),'authenticated-V-norm-selected-source','exec'),environment)
        for case in data['cases']:
            row=dict(name=case['name'],completed=False,upstream_helper_evaluated=False,arrays={});rows.append(row)
            if case['tokens']==0:
                row.update(completed=True,reason='Empty sequence is mathematical no-call policy');continue
            x=mx.array(array(case,'input'));obj=SimpleNamespace(eps=EPSILON)
            observe(row,case,'source_output',environment['__call__'](obj,x))
            if not np.array_equal(np.asarray(x),array(case,'input')):raise ValueError('V-norm source mutated input')
            row.update(completed=True,upstream_helper_evaluated=True)
    except (Exception,KeyboardInterrupt) as error:primary=type(error).__name__+': '+str(error)
    finally:
        for value in previous_signals:signal.signal(value,signal.SIG_IGN)
        try:mx.set_default_device(previous_device)
        except Exception as error:cleanup.append('Restore device: '+type(error).__name__+': '+str(error))
    expected_comparisons=sum(2 for case in data['cases'] if case['tokens'])
    passed=primary is None and not cleanup and len(rows)==len(SPECS) and all(row['completed'] for row in rows) and len(comparisons)==expected_comparisons and all(item['passed'] for item in comparisons)
    report=dict(format='toklet-gemma-v-norm-observation-v1',passed=passed,mathematical_reference_sha256=reference_sha,
        source_revision=REVISION,source_sha256=SOURCES,admitted_runtime=admitted,requested_device=device,source_policy=dict(atol=ATOL,rtol=RTOL),
        selected_source_definitions={name:dict(line=node.lineno,source=ast.unparse(node)) for name,node in nodes.items()},
        annotation_adaptation='Deferred annotations; selected RMSNormNoScale executable body unchanged',
        observation_scope=SCOPE,cases=rows,comparisons=comparisons,binary=dict(path='observed.bin',bytes=len(blob),alignment=8,sha256=hashlib.sha256(blob).hexdigest()),
        primary_error=primary,cleanup_errors=cleanup,native_mapping_qualified=False,device_qualified=False,bf16_checkpoint_qualified=False,attention_or_kv_qualified=False,constructor_qualified=False)
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
