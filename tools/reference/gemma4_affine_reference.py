"""Explicit real-checkpoint row/projection witness, before E2B graph execution.

Never imports Toklet. This validates bounded reference helpers on actual data;
it is not a graph, generation, original-BF16, performance or device-fit receipt.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import sysconfig

import reference_runtime as runtime

import affine4_tiles as tiles

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT/'models/MLX-Gemma4-E2B-4bit-2387675'
REPO = 'mlx-community/gemma-4-e2b-it-4bit'
REVISION = '238767527555cb75a05732a84dff5d6ba0dd6809'
PINS = {'config.json':'6397cb6eca41b911d1dcab74e17941351057bd759284052a2331918ff6f9246c',
        'model.safetensors.index.json':'edb157dbf495e23f37377af4a628a9ad13c4ee7937f93ccb36ec9e9a19940f16',
        'model.safetensors':'038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e'}
CONTAINER_BYTES = 3550670554
HEADER_BYTES = 318220
HEADER_SHA = '4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7'
PREFIX = 'language_model.model.'
PROJECTIONS = {PREFIX+'layers.0.self_attn.k_proj.weight':(256,1536),
               PREFIX+'layers.15.mlp.down_proj.weight':(1536,12288)}
PROCESS_ALLOWANCE = 256 * 1024**2


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def descriptor_sha(stream):
    position = stream.tell()
    try:
        stream.seek(0)
        return hashlib.file_digest(stream,'sha256').hexdigest()
    finally:stream.seek(position)


def unique(items):
    out = {}
    for key,value in items:
        if key in out:raise ValueError('Duplicate tensor field')
        out[key] = value
    return out


def scalar_row(matrix, row):
    """Independent word/nibble decoding and scalar reconstruction, one row."""
    raw = []
    for offset,stride in matrix.components():
        matrix.stream.seek(offset+row*stride)
        value = matrix.stream.read(stride)
        if len(value)!=stride:raise ValueError('Scalar row truncated')
        raw.append(value)
    words=struct.unpack('<'+'I'*(matrix.columns//8),raw[0])
    scales=[struct.unpack('<f',struct.pack('<I',v[0]<<16))[0] for v in struct.iter_unpack('<H',raw[1])]
    biases=[struct.unpack('<f',struct.pack('<I',v[0]<<16))[0] for v in struct.iter_unpack('<H',raw[2])]
    values=[]
    for column in range(matrix.columns):
        code=(words[column//8]//(16**(column%8)))%16
        value=float(code)*scales[column//64]+biases[column//64]
        values.append(struct.unpack('<f',struct.pack('<f',value))[0])
    return values


def matrices(stream):
    if os.fstat(stream.fileno()).st_size!=CONTAINER_BYTES or descriptor_sha(stream)!=PINS['model.safetensors']:
        raise ValueError('Pinned consumed weight descriptor differs')
    stream.seek(0)
    if stream.read(8)!=struct.pack('<Q',HEADER_BYTES):raise ValueError('Header extent differs')
    raw=stream.read(HEADER_BYTES)
    if hashlib.sha256(raw).hexdigest()!=HEADER_SHA:raise ValueError('Pinned header differs')
    header=json.loads(raw,object_pairs_hook=unique);origin=8+HEADER_BYTES;result={}
    for name,item in header.items():
        if not name.startswith(PREFIX) or item.get('dtype')!='U32':continue
        if not name.endswith('.weight') or len(item['shape'])!=2:raise ValueError('Unexpected packed text tensor')
        rows,words=item['shape'];columns=words*8
        offsets=[]
        for key,dtype,shape,size in [(name,'U32',[rows,words],rows*words*4),
                (name.removesuffix('.weight')+'.scales','BF16',[rows,columns//64],rows*columns//64*2),
                (name.removesuffix('.weight')+'.biases','BF16',[rows,columns//64],rows*columns//64*2)]:
            value=header[key]
            if value['dtype']!=dtype or value['shape']!=shape or value['data_offsets'][1]-value['data_offsets'][0]!=size:
                raise ValueError('Pinned affine descriptor shape differs')
            offsets.append(origin+value['data_offsets'][0])
        result[name]=tiles.MatrixRows(stream,CONTAINER_BYTES,rows,columns,*offsets)
    if len(result)!=278:raise ValueError('Expected278 text matrices')
    return result


def runtime_context(manifest_path, manifest_sha, native_path, native_sha, *, extra_sources=()):
    """Authenticate the cold runtime before NumPy import, then inspect origins."""
    if not sys.flags.no_user_site:
        raise ValueError('User-site imports must be disabled')
    selected=[Path(__file__).resolve(),Path(tiles.__file__).resolve(),*(Path(q).resolve() for q in extra_sources)]
    admitted=runtime.admit_runtime(manifest_path,manifest_sha,selected)
    raw=runtime.captured(native_path,2*1024**2,runtime.sha(native_sha))
    native=json.loads(raw,object_pairs_hook=runtime.json_unique)
    if set(native)!={'format','files'} or native['format']!='toklet-python-native-runtime-v1':
        raise ValueError('Explicit Python native runtime closure required')
    base=Path(sys.base_prefix).resolve()
    expected={base/'Python',base/'Resources/Python.app/Contents/MacOS/Python',*base.rglob('*.dylib'),*(Path(sysconfig.get_path('stdlib'))/'lib-dynload').glob('*.so')}
    if set(native['files'])!={str(q.resolve()) for q in expected}:
        raise ValueError('Python native runtime inventory differs')
    for name,pin in native['files'].items():
        if set(pin)!={'bytes','sha256'} or type(pin['bytes']) is not int or not 0<=pin['bytes']<=512*1024**2:
            raise ValueError('Exact bounded native file identity required')
        runtime.captured(Path(name),pin['bytes'],runtime.sha(pin['sha256']))
    manifest=json.loads(runtime.captured(manifest_path,2*1024**2,manifest_sha))
    allowed={**manifest['files'],**native['files']}
    def origins():
        import ctypes
        imported={}
        for name,module in list(sys.modules.items()):
            if name=='numpy' or name.startswith('numpy.'):
                origin=getattr(module,'__file__',None)
                if not origin:raise ValueError('NumPy module has no admitted origin')
                path=str(Path(origin).resolve())
                if path not in manifest['files']:raise ValueError('Unadmitted NumPy module origin: '+name)
                runtime.captured(Path(path),manifest['files'][path]['bytes'],manifest['files'][path]['sha256'])
                imported[name]=path
        root=Path(sysconfig.get_path('purelib')).resolve()/'numpy'
        if imported.get('numpy')!=str(root/'__init__.py') or 'numpy._core._multiarray_umath' not in imported:
            raise ValueError('Actual NumPy root/native imports differ')
        library=ctypes.CDLL(None)
        count=library._dyld_image_count;count.argtypes=[];count.restype=ctypes.c_uint32
        image=library._dyld_get_image_name;image.argtypes=[ctypes.c_uint32];image.restype=ctypes.c_char_p
        non_system={};system=[]
        for index in range(count()):
            name=image(index).decode();path=Path(name).resolve();key=str(path)
            if key.startswith(('/usr/lib/','/System/')):system.append(name);continue
            if key not in allowed:raise ValueError('Unadmitted loaded native image: '+key)
            pin=allowed[key];runtime.captured(path,pin['bytes'],pin['sha256']);non_system[key]=pin
        return dict(numpy_modules=imported,non_system_native_images=non_system,system_native_images=sorted(system))
    import numpy
    initial=origins()
    return dict(admitted=admitted,native_manifest_sha256=native_sha,initial_origins=initial),origins


def run(output, *, admission=None, runtime_guard=None):
    import numpy as np
    if np.__version__!='2.3.2':raise ValueError('Pinned NumPy2.3.2 required')
    output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Local artifact output required')
    output.mkdir() # Exclusive: failed observations must remain independently visible.
    report=dict(format='toklet-gemma4-affine-reference-v1',passed=False,repository=REPO,revision=REVISION,
                primary_error=None,endpoint_errors=[],rows=[],projections=[],arrays=[],
                weights='Packed4/group64 BF16 metadata, FP64 affine then one FP32 reconstruction',
                accumulation='NumPy/BLAS FP64 over reconstructed FP32 weights; scalar fsum/sequential checks',
                gate=dict(atol=1e-5,rtol=1e-4),numpy=np.__version__,process_allowance_bytes=PROCESS_ALLOWANCE,
                inference_executed=False,original_bf16_graph_qualified=False,device_fit_qualified=False,
                performance_qualified=False,app_eligibility_changed=False)
    report['runtime_admission']=admission
    stream=None;binary=None
    def stop(number,_):
        for value in (signal.SIGINT,signal.SIGTERM):signal.signal(value,signal.SIG_IGN)
        raise InterruptedError('Received signal '+str(number))
    old={v:signal.signal(v,stop) for v in (signal.SIGINT,signal.SIGTERM)}
    try:
        binary=(output/'arrays.bin').open('xb')
        for name,digest in PINS.items():
            if sha(MODEL/name)!=digest:raise ValueError('Checkpoint path changed: '+name)
        stream=(MODEL/'model.safetensors').open('rb');admitted=matrices(stream)
        def save(name,value):
            raw=value.tobytes(order='C');item=dict(name=name,shape=list(value.shape),dtype=value.dtype.str,
                offset=binary.tell(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),completed=False)
            report['arrays'].append(item)
            if binary.write(raw)!=len(raw):raise OSError('Short array write')
            binary.flush();item['completed']=True
        for name,matrix in sorted(admitted.items()):
            for row in [0,matrix.rows//2,matrix.rows-1]:
                actual=matrix.read_rows(row,row+1)
                expected=struct.pack('<'+'f'*matrix.columns,*scalar_row(matrix,row))
                if actual.tobytes()!=expected:raise ValueError('Scalar row mismatch: '+name)
                report['rows'].append(dict(name=name,row=row,columns=matrix.columns,
                    sha256=hashlib.sha256(expected).hexdigest(),passed=True))
        for name,shape in PROJECTIONS.items():
            matrix=admitted[name]
            if (matrix.rows,matrix.columns)!=shape:raise ValueError('Projection shape differs')
            x=np.array([[(i%7-3)/8 for i in range(matrix.columns)],
                        [(-1)**i*(i%5)/16 for i in range(matrix.columns)]],dtype=np.float32)
            # Array overlap + output,128MiB runtime/BLAS allowance,16MiB metadata,
            # 1MiB hashing/read reserve and16MiB capture reserve fit256MiB.
            reservation=tiles.planned_buffer_bytes(matrix,2,np.float64)+161*1024**2
            if reservation>PROCESS_ALLOWANCE:raise ValueError('Reference resource allowance exceeded')
            fsum=np.empty((2,matrix.rows),dtype=np.float64);ordered=np.empty_like(fsum)
            for row in range(matrix.rows):
                w=scalar_row(matrix,row)
                for token in range(2):
                    products=[float(a)*b for a,b in zip(x[token],w)]
                    fsum[token,row]=math.fsum(products)
                    total=0.
                    for value in products:total+=value
                    ordered[token,row]=total
            save(name+'.input',x);save(name+'.scalar_fsum',fsum);save(name+'.scalar_ordered',ordered)
            for dtype in (np.float32,np.float64):
                actual=tiles.project_rows(matrix,x,dtype)
                save(name+'.numpy_'+np.dtype(dtype).name,actual)
                for label,source in [('fsum',fsum),('ordered',ordered)]:
                    expected=source.astype(dtype)
                    reference=expected.astype(np.float64)
                    fraction=np.abs(actual.astype(np.float64)-reference)/(1e-5+1e-4*np.abs(reference))
                    maximum=float(np.max(fraction))
                    row=dict(name=name,dtype=np.dtype(dtype).name,contract=label,
                             values=int(actual.size),maximum_gate_fraction=maximum,passed=bool(np.isfinite(actual).all() and maximum<=1.))
                    report['projections'].append(row)
                    if not row['passed']:raise ValueError('Projection contract mismatch: '+name)
            report.setdefault('buffer_reservations',{})[name]=reservation
    except BaseException as error:report['primary_error']=type(error).__name__+': '+str(error)
    finally:
        for value in old:signal.signal(value,signal.SIG_IGN)
        try:
            if runtime_guard is not None:report['runtime_endpoint_origins']=runtime_guard()
        except BaseException as error:report['endpoint_errors'].append('runtime origins: '+type(error).__name__+': '+str(error))
        try:
            if stream is not None and descriptor_sha(stream)!=PINS['model.safetensors']:
                report['endpoint_errors'].append('Consumed descriptor changed')
            for name,digest in PINS.items():
                if sha(MODEL/name)!=digest:report['endpoint_errors'].append('Checkpoint path changed: '+name)
        except BaseException as error:report['endpoint_errors'].append(type(error).__name__+': '+str(error))
        try:
            for label,resource in [('source',stream),('binary',binary)]:
                if resource is not None:
                    try:resource.close()
                    except BaseException as error:report['endpoint_errors'].append(label+' close: '+type(error).__name__+': '+str(error))
            path=output/'arrays.bin'
            try:
                if path.exists():report['binary']=dict(bytes=path.stat().st_size,sha256=sha(path))
            except BaseException as error:report['endpoint_errors'].append('binary capture: '+type(error).__name__+': '+str(error))
            report['passed']=report['primary_error'] is None and not report['endpoint_errors']
            with (output/'observation.json').open('x') as file:json.dump(report,file,indent=2,allow_nan=False);file.write('\n')
        finally:
            for value,handler in old.items():signal.signal(value,handler)
    print(json.dumps(dict(passed=report['passed'],rows=len(report['rows']),comparisons=len(report['projections']),primary_error=report['primary_error'],endpoint_errors=report['endpoint_errors'])))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--runtime-manifest',required=True,type=Path)
    parser.add_argument('--expected-runtime-sha256',required=True)
    parser.add_argument('--native-manifest',required=True,type=Path)
    parser.add_argument('--expected-native-sha256',required=True)
    args=parser.parse_args()
    admission,guard=runtime_context(args.runtime_manifest,args.expected_runtime_sha256,args.native_manifest,args.expected_native_sha256)
    sys.exit(run(args.output,admission=admission,runtime_guard=guard))
