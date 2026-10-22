"""CPU-only, bounded references for the immutable M8 public Qwen0.5B MLX trial.

The reader independently unpacks eight-bit/group64 Half metadata into FP32.
One tied FP32 table plus one current matrix are retained, never a full model.
"""
import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
REPO = 'mlx-community/Qwen2.5-0.5B-Instruct-8bit'
REVISION = '6474d4c1a05bd7bd634a9c993560e4e7272eda3d'
MODEL = ROOT / 'models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c'
PINS = {'config.json': '4dae7c9622e41e50752ba422149df0694044266c06cd851aff630799701c83c6',
        'model.safetensors': '3dd0b6c2983ac5fe35f60ba260b1c7c35e4c38f17f3d5139d0bb477924e7aef4',
        'model.safetensors.index.json': 'e48752ffebac15ee35d1d86e96c9b87f2efd7783b19ebf82a26955dbd43e0476'}
WEIGHT_BYTES = 525045902
STRIPE = 128
LAYERS = [0, 1, 23]
CASES = {'single': {'tokens': [9707], 'offset': 0},
         'short': {'tokens': [9707, 11, 358, 2776], 'offset': 0},
         'future_changed': {'tokens': [9707, 11, 358, 498], 'offset': 0},
         'offset': {'tokens': [11, 358, 2776], 'offset': 5},
         'tile17': {'tokens': ([9707, 11, 358, 2776]*5)[:17], 'offset': 0},
         'tile17_offset5': {'tokens': ([9707, 11, 358, 2776]*5)[:17], 'offset': 5}}
POLICY = {'atol_floor': 1e-4, 'envelope_multiplier': 2, 'rtol': 1e-3,
          'absolute_limit': 'max(1e-4,2*pooled_FP32_FP64_max_abs,2*equivalent_prefix_max_abs)',
          'case_scope': 'same named stage/layer across six authored cases and four final greedy distributions',
          'reference': 'Independent NumPy CPU FP32 and FP64 arithmetic; original FP32 affine weights, Half gather/exclusions/activations/KV, FP32 logits.'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def descriptor_sha(stream):
    position = stream.tell()
    try:
        stream.seek(0)
        return hashlib.file_digest(stream, 'sha256').hexdigest()
    finally:
        stream.seek(position)


def unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate JSON field: '+name)
        result[name] = value
    return result


def bounded(path, maximum, expected):
    with Path(path).open('rb') as stream:
        data = stream.read(maximum+1)
    if len(data) > maximum or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Admitted bytes differ: '+str(path))
    return json.loads(data, object_pairs_hook=unique)


def decode(packed, scales, biases, columns):
    import numpy as np
    if packed.dtype != np.dtype('<u4') or scales.dtype != np.dtype('<f2') or biases.dtype != np.dtype('<f2'):
        raise ValueError('Selected affine dtypes differ')
    if (type(columns) is not int or columns <= 0 or columns % 64 or packed.ndim != 2
            or len(packed) == 0 or packed.shape[1] != columns//4 or scales.shape != (len(packed), columns//64) or biases.shape != scales.shape):
        raise ValueError('Selected affine shapes differ')
    codes = np.stack([(packed >> shift) & 255 for shift in range(0, 32, 8)], axis=-1).reshape(len(packed), columns)
    value = (codes.astype(np.float64)*np.repeat(scales.astype(np.float64), 64, axis=-1)
             + np.repeat(biases.astype(np.float64), 64, axis=-1)).astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite independent affine reconstruction')
    return value


class TiedEmbedding:
    """Token-list indexing is Half gather; row slices preserve FP32 output weights."""
    def __init__(self, table):
        self.table = table

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        import numpy as np
        if isinstance(key, list):
            return self.table[key].astype(np.float16)
        if isinstance(key, slice):
            return self.table[key]
        raise ValueError('Independent tied access must be tokens or rows')


class Weights:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.stream = None
        self.mapping = None
        self.embedding = None
        self.config = bounded(self.folder/'config.json', 1024**2, PINS['config.json'])
        self.index = bounded(self.folder/'model.safetensors.index.json', 1024**2, PINS['model.safetensors.index.json'])
        if self.config['quantization'] != {'group_size': 64, 'bits': 8}:
            raise ValueError('Selected quantization differs')

    def __enter__(self):
        import numpy as np
        try:
            self.stream = (self.folder/'model.safetensors').open('rb')
            if os.fstat(self.stream.fileno()).st_size != WEIGHT_BYTES or descriptor_sha(self.stream) != PINS['model.safetensors']:
                raise ValueError('Selected weight bytes differ')
            size = struct.unpack('<Q', self.stream.read(8))[0]
            if size > 1024**2:
                raise ValueError('Oversize tensor header')
            self.header = json.loads(self.stream.read(size), object_pairs_hook=unique)
            self.origin = 8+size
            self.header.pop('__metadata__', None)
            shapes = {'model.embed_tokens.weight': [151936, 896], 'model.norm.weight': [896]}
            for i in range(24):
                prefix = f'model.layers.{i}'
                for norm in ['input_layernorm', 'post_attention_layernorm']:
                    shapes[f'{prefix}.{norm}.weight'] = [896]
                for name, rows, columns in [('q',896,896),('k',128,896),('v',128,896),('o',896,896),
                                             ('gate',4864,896),('up',4864,896),('down',896,4864)]:
                    block = 'self_attn' if name in ['q','k','v','o'] else 'mlp'
                    shapes[f'{prefix}.{block}.{name}_proj.weight'] = [rows,columns]
                    if name in ['q','k','v']:
                        shapes[f'{prefix}.{block}.{name}_proj.bias'] = [rows]
            expected = {}
            self.logical = shapes
            for name, shape in shapes.items():
                if len(shape) == 2:
                    rows, columns = shape;module = name.removesuffix('.weight')
                    expected[name] = ('U32',[rows,columns//4])
                    expected[module+'.scales'] = ('F16',[rows,columns//64])
                    expected[module+'.biases'] = ('F16',[rows,columns//64])
                else:
                    expected[name] = ('F16',shape)
            if set(self.header) != set(expected) or set(self.index['weight_map']) != set(expected):
                raise ValueError('Selected tensor inventory differs')
            spans = []
            for name, (dtype, shape) in expected.items():
                item = self.header[name];begin,end = item['data_offsets']
                if (item['dtype'] != dtype or item['shape'] != shape or type(begin) is not int or type(end) is not int
                        or begin < 0 or end-begin != math.prod(shape)*(4 if dtype == 'U32' else 2)
                        or end > WEIGHT_BYTES-self.origin or self.index['weight_map'][name] != 'model.safetensors'):
                    raise ValueError('Selected tensor metadata differs: '+name)
                spans.append((begin,end))
            cursor = 0
            for begin,end in sorted(spans):
                if begin != cursor:
                    raise ValueError('Tensor spans overlap or contain gaps')
                cursor = end
            if cursor != WEIGHT_BYTES-self.origin:
                raise ValueError('Unexpected tensor payload tail')
            self.mapping = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
            table = self.matrix('model.embed_tokens.weight')
            self.embedding = TiedEmbedding(table)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def raw(self, name, start=0, end=None):
        import numpy as np
        item = self.header[name];shape = item['shape'];dtype = '<u4' if item['dtype'] == 'U32' else '<f2'
        value = np.ndarray(shape, dtype=dtype, buffer=self.mapping, offset=self.origin+item['data_offsets'][0])
        return value[start:end]

    def matrix(self, name):
        import numpy as np
        rows,columns = self.logical[name];module = name.removesuffix('.weight')
        out = np.empty((rows,columns),dtype=np.float32)
        for start in range(0,rows,STRIPE):
            end = min(start+STRIPE,rows)
            out[start:end] = decode(self.raw(name,start,end),self.raw(module+'.scales',start,end),
                                    self.raw(module+'.biases',start,end),columns)
        return out

    def __getitem__(self, key):
        import numpy as np
        if key == 'embed_tokens':
            return self.embedding
        if key == 'norm':
            name = 'model.norm.weight'
        elif key.endswith(('input_layernorm','post_attention_layernorm')):
            name = 'model.'+key+'.weight'
        else:
            parts = key.split('.')
            block = 'self_attn' if parts[2] in ['q_proj','k_proj','v_proj','o_proj'] else 'mlp'
            name = '.'.join(['model',*parts[:2],block,*parts[2:]])
        if len(self.logical[name]) == 2:
            return self.matrix(name)
        out = self.raw(name).copy()
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite floating exclusion')
        return out

    def verify(self):
        if descriptor_sha(self.stream) != PINS['model.safetensors']:
            raise ValueError('Consumed weight descriptor changed')
        for name,digest in PINS.items():
            if sha(self.folder/name) != digest:
                raise ValueError('Checkpoint path changed: '+name)

    def __exit__(self, *_):
        self.embedding = None
        if self.mapping is not None:
            self.mapping.close();self.mapping = None
        if self.stream is not None:
            self.stream.close();self.stream = None


def expected_arrays():
    """Literal graph boundaries; independent of exported/target array inventories."""
    arrays = {}
    for case, spec in CASES.items():
        n = len(spec['tokens'])
        stages = {'embedding': ([n,896],'<f2')}
        for layer in LAYERS:
            for stage in ['input_norm','q_projection','k_projection','v_projection','q_rotary','k_rotary',
                          'attention_scores','attention_probabilities','attention_context','attention_projection',
                          'attention_residual','post_attention_norm','gate_projection','up_projection',
                          'gate_silu','gated','down_projection','output']:
                shape = [n,896];dtype = '<f2'
                if stage in ['k_projection','v_projection']:shape = [n,128]
                elif stage == 'q_rotary':shape = [n,14,64]
                elif stage == 'k_rotary':shape = [n,2,64]
                elif stage in ['attention_scores','attention_probabilities']:shape = [14,n,n];dtype = '<f4'
                elif stage in ['gate_projection','up_projection','gate_silu','gated']:shape = [n,4864]
                stages[f'layers.{layer}.{stage}'] = (shape,dtype)
        stages['final_norm'] = ([n,896],'<f2')
        stages['logits'] = ([n,151936],'<f4')
        arrays.update({f'cases.{case}.{stage}': value for stage,value in stages.items()})
    arrays.update({f'greedy.step_{i}.logits': ([151936],'<f4') for i in range(4)})
    return arrays


def validate_manifest(manifest):
    if (manifest['format'] != 'toklet-mlx-qwen-real-reference-v1' or manifest['policy'] != POLICY
            or manifest['cases'] != CASES or manifest['layers'] != LAYERS
            or manifest['model'] != dict(repository=REPO,revision=REVISION,file_pins=PINS)
            or manifest['numpy'] != '2.3.2'
            or any(manifest[key] is not False for key in ['target_gpu_executed','held_out_quality_qualified','physical_device_qualified'])):
        raise ValueError('Selected reference identity differs')
    producer_names = {'tools/reference/mlx_qwen_real8.py','tools/reference/qwen_fp16.py','tools/reference/qwen.py',
                      'tools/bench/prepare.py','tools/experiments/native_usage.py'}
    def digest(value):
        return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    if (set(manifest['producer_pins']) != producer_names or not manifest['numpy_binary_pins']
            or any(not digest(value) for value in [manifest['array_sha256'],*manifest['producer_pins'].values(),*manifest['numpy_binary_pins'].values()])
            or any(Path(name).is_absolute() or '..' in Path(name).parts or not name.startswith('.venv/')
                   or not name.endswith(('.so','.dylib')) for name in manifest['numpy_binary_pins'])):
        raise ValueError('Reference producer identity differs')
    expected = expected_arrays()
    if set(manifest['arrays']) != set(expected):raise ValueError('Reference coverage differs')
    cursor = 0;gates = {}
    for name,item in manifest['arrays'].items():
        shape,dtype = expected[name];gate = item['gate'];stage = name.split('.',2)[2]
        if (set(gate) != {'atol','rtol','pooled_fp32_fp64_max_abs','equivalent_prefix_max_abs'}
                or not digest(item['sha256'])):raise ValueError('Reference gate/hash fields differ: '+name)
        if (not isinstance(item['shape'],list) or any(type(dimension) is not int or dimension <= 0 for dimension in item['shape'])
                or item['shape'] != shape or item['dtype'] != dtype or type(item['offset']) is not int
                or type(item['bytes']) is not int or item['offset'] != cursor
                or item['bytes'] != math.prod(shape)*(2 if dtype == '<f2' else 4)):
            raise ValueError('Reference array layout differs: '+name)
        cursor += item['bytes']
        errors = [gate['pooled_fp32_fp64_max_abs'],gate['equivalent_prefix_max_abs']]
        if (any(type(value) not in [int,float] or not math.isfinite(value) or value < 0 for value in gate.values())
                or not math.isfinite(max(1e-4,2*errors[0],2*errors[1]))
                or gate['rtol'] != 1e-3 or gate['atol'] != max(1e-4,2*errors[0],2*errors[1])
                or (stage in gates and gates[stage] != gate)):
            raise ValueError('Reference numerical policy differs: '+name)
        gates[stage] = gate
    if type(manifest['array_bytes']) is not int or manifest['array_bytes'] != cursor:
        raise ValueError('Reference total bytes differ')
    greedy = manifest['greedy']
    if (greedy['prompt'] != CASES['short']['tokens']
            or greedy['teacher_forcing'] != 'Independent FP64-accumulation greedy history'
            or any(len(greedy[key]) != 4 for key in ['fp32_tokens','fp64_tokens','fp64_logit_margins'])
            or any(type(token) is not int or not 0 <= token < 151936 for key in ['fp32_tokens','fp64_tokens'] for token in greedy[key])
            or any(type(value) not in [int,float] or not math.isfinite(value) or value < 0 for value in greedy['fp64_logit_margins'])):
        raise ValueError('Reference greedy contract differs')


def verify(folder, expected_sha=None):
    with (folder/'manifest.json').open('rb') as stream:data = stream.read(2*1024**2+1)
    if len(data) > 2*1024**2:raise ValueError('Reference manifest exceeds limit')
    if expected_sha is not None and hashlib.sha256(data).hexdigest() != expected_sha:raise ValueError('Frozen manifest SHA differs')
    manifest = json.loads(data, object_pairs_hook=unique)
    validate_manifest(manifest)
    path = folder/'arrays.bin';whole = hashlib.sha256()
    with path.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size != manifest['array_bytes']:raise ValueError('Reference size differs')
        for name,item in manifest['arrays'].items():
            digest = hashlib.sha256();remaining = item['bytes'];code = '<e' if item['dtype'] == '<f2' else '<f'
            while remaining:
                raw = stream.read(min(65536,remaining))
                if not raw:raise ValueError('Reference array truncated: '+name)
                remaining -= len(raw);digest.update(raw);whole.update(raw)
                if any(not math.isfinite(value[0]) for value in struct.iter_unpack(code,raw)):
                    raise ValueError('Reference contains nonfinite values: '+name)
            if digest.hexdigest() != item['sha256']:raise ValueError('Reference array differs: '+name)
        if stream.read(1) or whole.hexdigest() != manifest['array_sha256']:raise ValueError('Reference arrays differ')
    return manifest


def export(folder):
    if os.path.lexists(folder):
        raise FileExistsError('Never overwrite a frozen reference')
    for variable in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:
        os.environ[variable] = '1'
    producer_paths = [Path(__file__).resolve(),ROOT/'tools/reference/qwen_fp16.py',ROOT/'tools/reference/qwen.py',
                      ROOT/'tools/bench/prepare.py',ROOT/'tools/experiments/native_usage.py']
    producers = {str(path.relative_to(ROOT)):sha(path) for path in producer_paths}
    import importlib.metadata
    distribution = importlib.metadata.distribution('numpy')
    if distribution.version != '2.3.2':
        raise ValueError('Pinned NumPy differs')
    binaries = {str(Path(distribution.locate_file(f)).resolve().relative_to(ROOT)):sha(distribution.locate_file(f))
                for f in distribution.files if str(f).endswith(('.so','.dylib'))}
    if not binaries:
        raise ValueError('NumPy binary inventory absent')
    import numpy as np
    import qwen_fp16
    sys.path.insert(0,str(ROOT/'tools/experiments'))
    import native_usage
    if np.__version__ != '2.3.2':
        raise ValueError('Imported NumPy version differs')
    numerical = dict(num_layers=24,num_query_heads=14,num_kv_heads=2,head_dim=64,rms_norm_eps=1e-6,rope_theta=1e6)
    admission = dict(tied_fp32_bytes=151936*896*4,source_mapping_bytes=WEIGHT_BYTES,
                     current_matrix_and_blas_bytes=128*1024**2,retained_trace_bytes=256*1024**2,
                     framework_metadata_allowance_bytes=512*1024**2)
    admission['admitted_bytes'] = sum(admission.values())
    if admission['admitted_bytes'] > 6*1024**3:
        raise ValueError('Reference admission exceeds Mac budget')
    snapshots = {};greedy={};usage=[dict(phase='before',usage=native_usage.usage())]
    with Weights(MODEL) as weights:
        for case,spec in CASES.items():
            main = qwen_fp16.forward(spec['tokens'],spec['offset'],weights,numerical,np.float32,LAYERS)
            wide = qwen_fp16.forward(spec['tokens'],spec['offset'],weights,numerical,np.float64,LAYERS)
            for stage,value in main.items():
                snapshots[f'cases.{case}.{stage}'] = (value,wide[stage])
            usage.append(dict(phase=case,usage=native_usage.usage()))
        history = list(CASES['short']['tokens']);tokens=[];wide_tokens=[];margins=[]
        for step in range(4):
            main = qwen_fp16.forward(history,0,weights,numerical,np.float32,[])['logits'][-1]
            wide = qwen_fp16.forward(history,0,weights,numerical,np.float64,[])['logits'][-1]
            snapshots[f'greedy.step_{step}.logits'] = (main,wide)
            tokens.append(int(np.argmax(main)));token=int(np.argmax(wide));wide_tokens.append(token)
            ordered=np.sort(wide);margins.append(float(ordered[-1]-ordered[-2]));history.append(token)
        greedy=dict(prompt=CASES['short']['tokens'],teacher_forcing='Independent FP64-accumulation greedy history',
                    fp32_tokens=tokens,fp64_tokens=wide_tokens,fp64_logit_margins=margins)
        weights.verify()
    pooled={};equivalent={}
    for key,(main,wide) in snapshots.items():
        stage=key.split('.',2)[2];delta=float(np.max(np.abs(main.astype(np.float64)-wide)))
        pooled[stage]=max(pooled.get(stage,0),delta)
    for first,a in CASES.items():
        for second,b in CASES.items():
            if first>=second or a['offset']!=b['offset']:
                continue
            common=0
            for x,y in zip(a['tokens'],b['tokens']):
                if x!=y:break
                common+=1
            if not common:continue
            for stage in pooled:
                left_key=f'cases.{first}.{stage}';right_key=f'cases.{second}.{stage}'
                if left_key not in snapshots or right_key not in snapshots:continue
                for precision in [0,1]:
                    left,right=snapshots[left_key][precision],snapshots[right_key][precision]
                    if stage.endswith(('attention_scores','attention_probabilities')):
                        left,right=left[:,:common,:common],right[:,:common,:common]
                    else:left,right=left[:common],right[:common]
                    delta=float(np.max(np.abs(left.astype(np.float64)-right)))
                    equivalent[stage]=max(equivalent.get(stage,0),delta)
    arrays={};raw=bytearray()
    for key,(main,wide) in snapshots.items():
        if not np.isfinite(main).all() or not np.isfinite(wide).all():raise ValueError('Nonfinite reference')
        dtype='<f2' if main.dtype==np.float16 else '<f4';data=main.astype(dtype).tobytes();stage=key.split('.',2)[2]
        arrays[key]=dict(offset=len(raw),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),dtype=dtype,shape=list(main.shape),
                         gate=dict(atol=max(1e-4,2*pooled[stage],2*equivalent.get(stage,0)),rtol=1e-3,
                                   pooled_fp32_fp64_max_abs=pooled[stage],equivalent_prefix_max_abs=equivalent.get(stage,0)))
        raw.extend(data)
    if len(arrays)!=346:raise ValueError('Independent reference coverage differs')
    for name,digest in {**producers,**binaries}.items():
        if sha(ROOT/name)!=digest:raise ValueError('Oracle producer changed: '+name)
    for name,digest in PINS.items():
        if sha(MODEL/name)!=digest:raise ValueError('Input changed before publication')
    folder.mkdir()
    (folder/'arrays.bin').write_bytes(raw)
    manifest=dict(format='toklet-mlx-qwen-real-reference-v1',model=dict(repository=REPO,revision=REVISION,file_pins=PINS),
                  policy=POLICY,cases=CASES,layers=LAYERS,producer_pins=producers,numpy_binary_pins=binaries,numpy=np.__version__,
                  admission=admission,greedy=greedy,array_bytes=len(raw),array_sha256=hashlib.sha256(raw).hexdigest(),arrays=arrays,
                  native_samples=usage,target_gpu_executed=False,held_out_quality_qualified=False,physical_device_qualified=False)
    pending=folder/'.manifest.pending';pending.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    final_usage=native_usage.usage()
    if not 0<final_usage['lifetime_max_phys_footprint']<=6*1024**3:raise ValueError('Completed reference exceeds Mac budget')
    manifest['native_samples'].append(dict(phase='completed-before-publication',usage=final_usage))
    pending.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n');pending.rename(folder/'manifest.json')
    verify(folder)
    print(json.dumps(dict(arrays=len(arrays),bytes=len(raw),manifest_sha256=sha(folder/'manifest.json'),greedy=greedy,completed_native_usage=final_usage)))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--export',type=Path);group.add_argument('--check',type=Path)
    parser.add_argument('--manifest-sha256');args=parser.parse_args()
    if args.check and not args.manifest_sha256:parser.error('--check requires the externally frozen --manifest-sha256')
    if args.export:export(args.export)
    else:print(json.dumps(dict(verified=True,arrays=len(verify(args.check,args.manifest_sha256)['arrays']))))
