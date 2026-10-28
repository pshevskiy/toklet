"""CPU-only, bounded references for the immutable M8 public Llama3.2 Llama-family MLX trial.

The reader independently unpacks eight-bit/group64 Half metadata into FP32.
One tied FP32 table plus one current matrix are retained, never a full model.
NumPy/BLAS reductions are independently pinned profiles, not scalar-order or
GPU emulation. Both main/wide raw stored arrays are preserved for audit.
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
import sysconfig

ROOT = Path(__file__).resolve().parents[2]
REPO = 'mlx-community/Llama-3.2-1B-Instruct-8bit'
REVISION = 'd48cdf0a4ea22d893b7c63a99d6a693e24822795'
MODEL = ROOT / 'models/MLX-Llama-3.2-1B-Instruct-8bit-d48cdf0'
# The full advertised payload hash is verified only when an explicitly admitted
# future export opens the file. This producer performs no downloading.
PINS = {'config.json':'847794a29e25927b4b98f35f5e66829e1b8bfd78f7743a9582780b4fd4a8ef9d',
        'model.safetensors':'cd3582a78766654e3e2107e3b4d51d9cb18e6754b659bfb44a34328ef9d7fa52',
        'model.safetensors.index.json':'c444df144fe9d86b42f8b1d797432c1f8379225316e41b1b9551e90d63136155'}
WEIGHT_BYTES = 1313157436
STRIPE = 128
LAYERS = [0, 1, 15]
# Numerical probes deliberately use fixed IDs, not a claim about rendered text.
CASES = {'single': {'tokens': [128000], 'offset': 0},
         'short': {'tokens': [128000,1,7,13], 'offset': 0},
         'future_changed': {'tokens': [128000,1,7,17], 'offset': 0},
         'offset5': {'tokens': [7,13,3], 'offset': 5},
         'tile17': {'tokens': ([128000,1,7,13]*5)[:17], 'offset': 0},
         'offset509': {'tokens': [7,13,3], 'offset': 509}}
NUMERICAL = dict(num_layers=16,num_query_heads=32,num_kv_heads=8,head_dim=64,
                 rms_norm_eps=1e-5,rope_theta=500000.)
SCALING = dict(factor=32.,low=1.,high=4.,original=8192)
NATIVE_PROFILE = dict(projection='MlxAffine',norm='Compensated',attention='Reference',
                      schedule='Serial',activation='Half',kv='Half',capacity=512,
                      chunks=[32,128],readbacks=['FullLogits','GreedyToken'])
POLICY = {'atol_floor':1e-4,'envelope_multiplier':2,'rtol':1e-3,
          'absolute_limit':'max(1e-4,2*pooled_FP32_FP64_max_abs)',
          'case_scope':'same named stage/layer across six authored cases, four greedy and three scoring distributions',
          'reference':'Independent NumPy BLAS FP32 and FP64 arithmetic; original FP32 affine weights, Half gather/exclusions/activations/KV, FP32 scores/probabilities/logits; canonical Llama3 rotary. Not scalar-order or exact Metal emulation.',
          'cross_profile_greedy':'diagnostic only; native schedule/readback selection consistency remains exact',
          'equivalent_prefix':'reported only; never enlarges this gate'}
PRODUCER_NAMES = ('tools/reference/mlx_llama3_real.py','tools/reference/llama3_fp16.py',
                  'tools/reference/llama3_rotary.py','tools/reference/qwen_fp16.py',
                  'tools/reference/qwen.py','tools/bench/prepare.py','tools/experiments/native_usage.py')
MEMORY_LIMIT = 6*1024**3


def admission():
    # Separate simultaneous owners; map includes full file even if pages are clean.
    # Largest ordinary reconstructed matrix is8192x2048 FP32 (64MiB).
    # qwen_fp16 linear widens <=1024 rows, <=8192 columns (64MiB FP64).
    # Dequant128-row stripes retain codes/repeat/product temporaries;256MiB
    # workspace covers these and graph intermediates without pretending BLAS is free.
    result = dict(tied_fp32_bytes=128256*2048*4,source_mapping_bytes=WEIGHT_BYTES,
                  current_matrix_bytes=8192*2048*4,linear_wide_stripe_bytes=1024*8192*8,
                  dequant_graph_blas_workspace_bytes=256*1024**2,
                  retained_dual_traces_and_publication_bytes=512*1024**2,
                  framework_metadata_allowance_bytes=1024*1024**2)
    result['admitted_bytes']=sum(result.values())
    if result['admitted_bytes']>MEMORY_LIMIT:raise ValueError('Reference admission exceeds Mac budget')
    return result


def package_pins():
    site=Path(sysconfig.get_path('purelib')).resolve()
    if not site.is_relative_to(ROOT): raise ValueError('Repository virtual environment required')
    paths=[]
    for name in ['numpy']:
        folder=site/name
        if not folder.is_dir(): raise ValueError('Producer package absent')
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.so','.dylib'])
    for folder in site.glob('numpy*.libs'):
        paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.so','.dylib'])
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


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
            shapes = {'model.embed_tokens.weight': [128256, 2048], 'model.norm.weight': [2048]}
            for i in range(16):
                prefix = f'model.layers.{i}'
                for norm in ['input_layernorm', 'post_attention_layernorm']:
                    shapes[f'{prefix}.{norm}.weight'] = [2048]
                for name, rows, columns in [('q',2048,2048),('k',512,2048),('v',512,2048),('o',2048,2048),
                                             ('gate',8192,2048),('up',8192,2048),('down',2048,8192)]:
                    block = 'self_attn' if name in ['q','k','v','o'] else 'mlp'
                    shapes[f'{prefix}.{block}.{name}_proj.weight'] = [rows,columns]
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
        except BaseException as primary:
            self.__exit__(type(primary), primary, primary.__traceback__)
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

    def __exit__(self, _kind, primary, _trace):
        self.embedding=None
        failures=[]
        for name in ['mapping','stream']:
            value=getattr(self,name)
            setattr(self,name,None)
            if value is not None:
                try:value.close()
                except BaseException as error:failures.append(error)
        if failures:
            if primary is not None:
                for error in failures:primary.add_note('Weight cleanup: '+repr(error))
            else:raise failures[0]


def expected_arrays():
    """Literal graph boundaries; independent of exported/target array inventories."""
    arrays = {}
    for case, spec in CASES.items():
        n = len(spec['tokens'])
        stages = {'embedding': ([n,2048],'<f2')}
        for layer in LAYERS:
            for stage in ['input_norm','q_projection','k_projection','v_projection','q_rotary','k_rotary',
                          'attention_scores','attention_probabilities','attention_context','attention_projection',
                          'attention_residual','post_attention_norm','gate_projection','up_projection',
                          'gate_silu','gated','down_projection','output']:
                shape = [n,2048];dtype = '<f2'
                if stage in ['k_projection','v_projection']:shape = [n,512]
                elif stage == 'q_rotary':shape = [n,32,64]
                elif stage == 'k_rotary':shape = [n,8,64]
                elif stage in ['attention_scores','attention_probabilities']:shape = [32,n,n];dtype = '<f4'
                elif stage in ['gate_projection','up_projection','gate_silu','gated']:shape = [n,8192]
                stages[f'layers.{layer}.{stage}'] = (shape,dtype)
        stages['final_norm'] = ([n,2048],'<f2')
        stages['logits'] = ([n,128256],'<f4')
        arrays.update({f'cases.{case}.{stage}': value for stage,value in stages.items()})
    arrays.update({f'greedy.step_{i}.logits': ([128256],'<f4') for i in range(4)})
    arrays.update({f'scoring.prefix_{i}.logits': ([128256],'<f4') for i in range(1,4)})
    return arrays


def validate_manifest(manifest):
    if (manifest['format'] != 'toklet-mlx-llama3-real-reference-v1' or manifest['policy'] != POLICY
            or manifest['cases'] != CASES or manifest['layers'] != LAYERS
            or manifest['model'] != dict(repository=REPO,revision=REVISION,file_pins=PINS)
            or manifest['numpy'] != '2.3.2' or manifest['numerical'] != NUMERICAL
            or manifest['scaling'] != SCALING or manifest['native_profile'] != NATIVE_PROFILE
            or manifest['admission'] != admission()
            or any(manifest[key] is not False for key in ['target_gpu_executed','held_out_quality_qualified','physical_device_qualified'])):
        raise ValueError('Selected reference identity differs')
    producer_names = set(PRODUCER_NAMES)
    def digest(value):
        return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    if (set(manifest['producer_pins']) != producer_names or not manifest['package_pins']
            or any(not digest(value) for value in [manifest['array_sha256'],manifest['wide_array_sha256'],*manifest['producer_pins'].values(),*manifest['package_pins'].values()])
            or any(Path(name).is_absolute() or '..' in Path(name).parts or not name.startswith('.venv/')
                   or not name.endswith(('.py','.so','.dylib')) for name in manifest['package_pins'])):
        raise ValueError('Reference producer identity differs')
    for marker in ['/numpy/__init__.py']:
        if not any(name.endswith(marker) for name in manifest['package_pins']):raise ValueError('Producer package source closure absent')
    for marker in ['/numpy/']:
        if not any(marker in name and name.endswith('.so') for name in manifest['package_pins']):raise ValueError('Producer native closure absent')
    expected = expected_arrays()
    if set(manifest['arrays']) != set(expected):raise ValueError('Reference coverage differs')
    cursor = 0;gates = {}
    for name,item in manifest['arrays'].items():
        shape,dtype = expected[name];gate = item['gate'];stage = name.split('.',2)[2]
        if (set(gate) != {'atol','rtol','pooled_fp32_fp64_max_abs','equivalent_prefix_max_abs'}
                or not digest(item['sha256']) or not digest(item['wide_sha256'])):raise ValueError('Reference gate/hash fields differ: '+name)
        if (not isinstance(item['shape'],list) or any(type(dimension) is not int or dimension <= 0 for dimension in item['shape'])
                or item['shape'] != shape or item['dtype'] != dtype or type(item['offset']) is not int
                or type(item['bytes']) is not int or item['offset'] != cursor
                or item['bytes'] != math.prod(shape)*(2 if dtype == '<f2' else 4)):
            raise ValueError('Reference array layout differs: '+name)
        cursor += item['bytes']
        errors = [gate['pooled_fp32_fp64_max_abs'],gate['equivalent_prefix_max_abs']]
        if (any(type(value) not in [int,float] or not math.isfinite(value) or value < 0 for value in gate.values())
                or not math.isfinite(max(1e-4,2*errors[0]))
                or gate['rtol'] != 1e-3 or gate['atol'] != max(1e-4,2*errors[0])
                or (stage in gates and gates[stage] != gate)):
            raise ValueError('Reference numerical policy differs: '+name)
        gates[stage] = gate
    if type(manifest['array_bytes']) is not int or manifest['array_bytes'] != cursor:
        raise ValueError('Reference total bytes differ')
    greedy = manifest['greedy']
    if (greedy['prompt'] != CASES['short']['tokens']
            or greedy['teacher_forcing'] != 'Independent FP64-accumulation greedy history'
            or any(len(greedy[key]) != 4 for key in ['fp32_tokens','fp64_tokens','fp64_logit_margins'])
            or any(type(token) is not int or not 0 <= token < 128256 for key in ['fp32_tokens','fp64_tokens'] for token in greedy[key])
            or any(type(value) not in [int,float] or not math.isfinite(value) or value < 0 for value in greedy['fp64_logit_margins'])):
        raise ValueError('Reference greedy contract differs')


def verify(folder, expected_sha=None, manifest_name='manifest.json'):
    with (folder/manifest_name).open('rb') as stream:data = stream.read(2*1024**2+1)
    if len(data)>2*1024**2:raise ValueError('Reference manifest exceeds limit')
    if expected_sha is not None and hashlib.sha256(data).hexdigest()!=expected_sha:
        raise ValueError('Frozen manifest SHA differs')
    manifest=json.loads(data,object_pairs_hook=unique);validate_manifest(manifest)
    main_path,wide_path=folder/'arrays.bin',folder/'wide.bin'
    whole=hashlib.sha256();wide_whole=hashlib.sha256();selected={};wide_selected={};pooled={}
    with main_path.open('rb') as main,wide_path.open('rb') as wide:
        if any(os.fstat(s.fileno()).st_size!=manifest['array_bytes'] for s in [main,wide]):
            raise ValueError('Reference size differs')
        for name,item in manifest['arrays'].items():
            digest=hashlib.sha256();wide_digest=hashlib.sha256();remaining=item['bytes']
            code='<e' if item['dtype']=='<f2' else '<f';error=0.;saved=[];wide_saved=[]
            while remaining:
                size=min(65536,remaining);a,b=main.read(size),wide.read(size)
                if len(a)!=size or len(b)!=size:raise ValueError('Reference array truncated: '+name)
                remaining-=size;digest.update(a);wide_digest.update(b);whole.update(a);wide_whole.update(b)
                for (x,),(y,) in zip(struct.iter_unpack(code,a),struct.iter_unpack(code,b),strict=True):
                    if not math.isfinite(x) or not math.isfinite(y):raise ValueError('Nonfinite reference: '+name)
                    error=max(error,abs(x-y))
                    if name.startswith('greedy.'):saved.append(x);wide_saved.append(y)
            if digest.hexdigest()!=item['sha256'] or wide_digest.hexdigest()!=item['wide_sha256']:
                raise ValueError('Reference array hash differs: '+name)
            if saved:selected[name]=saved;wide_selected[name]=wide_saved
            stage=name.split('.',2)[2];pooled[stage]=max(pooled.get(stage,0.),error)
        if (main.read(1) or wide.read(1) or whole.hexdigest()!=manifest['array_sha256']
                or wide_whole.hexdigest()!=manifest['wide_array_sha256']):raise ValueError('Reference payload differs')
    for name,item in manifest['arrays'].items():
        if item['gate']['pooled_fp32_fp64_max_abs']!=pooled[name.split('.',2)[2]]:
            raise ValueError('Envelope does not match preserved main/wide bits: '+name)
    saved_greedy(selected,manifest)
    saved_greedy(wide_selected,manifest,wide=True)
    return manifest


def saved_greedy(selected_logits,manifest,wide=False):
    ids=[max(range(128256),key=lambda i:selected_logits[f'greedy.step_{step}.logits'][i]) for step in range(4)]
    if ids!=manifest['greedy']['fp64_tokens' if wide else 'fp32_tokens']:raise ValueError('Saved greedy IDs differ')
    if wide:
        margins=[]
        for step in range(4):
            ordered=sorted(selected_logits[f'greedy.step_{step}.logits'])
            margins.append(ordered[-1]-ordered[-2])
        if margins!=manifest['greedy']['fp64_logit_margins']:raise ValueError('Saved wide greedy margins differ')



def export(folder):
    if os.path.lexists(folder):raise FileExistsError('Never overwrite a frozen reference')
    folder.mkdir()
    try:
        return export_into(folder)
    except BaseException as error:
        try:(folder/'failure.json').write_text(json.dumps(dict(completed=False,error_type=type(error).__name__,error=str(error),notes=getattr(error,'__notes__',[])),indent=2)+'\n')
        except BaseException as cleanup:error.add_note('Failure receipt write failed: '+repr(cleanup))
        raise


def export_into(folder):
    for variable in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:
        os.environ[variable]='1'
    producers={name:sha(ROOT/name) for name in PRODUCER_NAMES}
    import importlib.metadata
    if importlib.metadata.version('numpy')!='2.3.2':raise ValueError('Pinned NumPy differs')
    binaries=package_pins()
    import numpy as np
    import llama3_fp16
    sys.path.insert(0,str(ROOT/'tools/experiments'))
    import native_usage
    if np.__version__!='2.3.2':raise ValueError('Imported NumPy differs')
    admitted=admission()
    def record(phase):
        value=dict(phase=phase,usage=native_usage.usage())
        with (folder/'progress.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        if not 0<value['usage']['lifetime_max_phys_footprint']<=MEMORY_LIMIT:
            raise ValueError('Source process exceeded6GiB')
        return value
    snapshots = {};greedy={};usage=[record('before')]
    with Weights(MODEL) as weights:
        for case,spec in CASES.items():
            main = llama3_fp16.forward(spec['tokens'],spec['offset'],weights,NUMERICAL,np.float32,LAYERS,SCALING)
            wide = llama3_fp16.forward(spec['tokens'],spec['offset'],weights,NUMERICAL,np.float64,LAYERS,SCALING)
            for stage,value in main.items():
                snapshots[f'cases.{case}.{stage}'] = (value,wide[stage])
            usage.append(record(case))
        history = list(CASES['short']['tokens']);tokens=[];wide_tokens=[];margins=[]
        for step in range(4):
            main = llama3_fp16.forward(history,0,weights,NUMERICAL,np.float32,[],SCALING)['logits'][-1].copy()
            wide = llama3_fp16.forward(history,0,weights,NUMERICAL,np.float64,[],SCALING)['logits'][-1].copy()
            snapshots[f'greedy.step_{step}.logits'] = (main,wide)
            tokens.append(int(np.argmax(main)));token=int(np.argmax(wide));wide_tokens.append(token)
            ordered=np.sort(wide);margins.append(float(ordered[-1])-float(ordered[-2]));history.append(token);usage.append(record('greedy.'+str(step)))
        greedy=dict(prompt=CASES['short']['tokens'],teacher_forcing='Independent FP64-accumulation greedy history',
                    fp32_tokens=tokens,fp64_tokens=wide_tokens,fp64_logit_margins=margins)
        for count in range(1,4):
            ts=CASES['short']['tokens'][:count]
            main=llama3_fp16.forward(ts,0,weights,NUMERICAL,np.float32,[],SCALING)['logits'][-1].copy()
            wide=llama3_fp16.forward(ts,0,weights,NUMERICAL,np.float64,[],SCALING)['logits'][-1].copy()
            snapshots[f'scoring.prefix_{count}.logits']=(main,wide);usage.append(record('scoring.'+str(count)))
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
    for count in range(1,4):
        for precision in [0,1]:
            full=snapshots['cases.short.logits'][precision][count-1]
            prefix=snapshots[f'scoring.prefix_{count}.logits'][precision]
            equivalent['logits']=max(equivalent.get('logits',0),float(np.max(np.abs(full.astype(np.float64)-prefix))))
    arrays={};raw=bytearray();wide_raw=bytearray()
    for key,(main,wide) in snapshots.items():
        if not np.isfinite(main).all() or not np.isfinite(wide).all():raise ValueError('Nonfinite reference')
        dtype='<f2' if main.dtype==np.float16 else '<f4';data=main.astype(dtype).tobytes();wide_data=wide.astype(dtype).tobytes();stage=key.split('.',2)[2]
        arrays[key]=dict(offset=len(raw),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),wide_sha256=hashlib.sha256(wide_data).hexdigest(),dtype=dtype,shape=list(main.shape),
                         gate=dict(atol=max(1e-4,2*pooled[stage]),rtol=1e-3,
                                   pooled_fp32_fp64_max_abs=pooled[stage],equivalent_prefix_max_abs=equivalent.get(stage,0)))
        raw.extend(data);wide_raw.extend(wide_data)
    if len(arrays)!=349:raise ValueError('Independent reference coverage differs')
    for name,digest in {**producers,**binaries}.items():
        if sha(ROOT/name)!=digest:raise ValueError('Oracle producer changed: '+name)
    for name,digest in PINS.items():
        if sha(MODEL/name)!=digest:raise ValueError('Input changed before publication')
    if package_pins()!=binaries:raise ValueError('Producer package closure changed')
    (folder/'arrays.bin').write_bytes(raw)
    (folder/'wide.bin').write_bytes(wide_raw)
    manifest=dict(format='toklet-mlx-llama3-real-reference-v1',model=dict(repository=REPO,revision=REVISION,file_pins=PINS),
                  policy=POLICY,cases=CASES,layers=LAYERS,producer_pins=producers,package_pins=binaries,numpy=np.__version__,numerical=NUMERICAL,scaling=SCALING,native_profile=NATIVE_PROFILE,
                  admission=admitted,greedy=greedy,array_bytes=len(raw),array_sha256=hashlib.sha256(raw).hexdigest(),wide_array_sha256=hashlib.sha256(wide_raw).hexdigest(),arrays=arrays,
                  native_samples=usage,target_gpu_executed=False,held_out_quality_qualified=False,physical_device_qualified=False)
    pending=folder/'.manifest.pending';pending.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    final_usage=native_usage.usage()
    if not 0<final_usage['lifetime_max_phys_footprint']<=6*1024**3:raise ValueError('Completed reference exceeds Mac budget')
    manifest['native_samples'].append(dict(phase='completed-before-publication',usage=final_usage))
    pending.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    verify(folder,manifest_name='.manifest.pending')
    manifest['native_samples'].append(record('verified-before-publication'))
    pending.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    pending.rename(folder/'manifest.json')
    print(json.dumps(dict(arrays=len(arrays),bytes=len(raw),manifest_sha256=sha(folder/'manifest.json'),greedy=greedy,completed_native_usage=final_usage)))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--export',type=Path);group.add_argument('--check',type=Path)
    parser.add_argument('--manifest-sha256');args=parser.parse_args()
    if args.check and not args.manifest_sha256:parser.error('--check requires the externally frozen --manifest-sha256')
    if args.export:export(args.export)
    else:print(json.dumps(dict(verified=True,arrays=len(verify(args.check,args.manifest_sha256)['arrays']))))
