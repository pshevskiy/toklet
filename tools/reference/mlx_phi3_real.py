"""Bounded independent Phi3 Half profiles; no native/MLX/remote-code execution.

Only original packed rows are decoded. One temporary FP32 logical matrix is
owned by project(); embeddings gather selected rows directly to Half. NumPy
BLAS profiles are independent arithmetic, not exact GPU or Torch emulation.
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
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPO = 'mlx-community/Phi-3-mini-4k-instruct-4bit'
REVISION = '5b3819ed6317784fb20eddeae9bed984f778d0d0'
MODEL = ROOT / 'models/MLX-Phi-3-mini-4k-instruct-4bit-5b3819e'
PINS = {'config.json':'0e2e43bc4358b4cabbcc33c496f34e170fdfe04612a47428f1691d1e9ec5a568',
        'model.safetensors':'8d75680621a09474f6601e9176f2f61f92a5e4c079d68d583901f51699fda50a',
        'model.safetensors.index.json':'4ef199a5c6c29fa54d84d3710bfad4b5424b75bbd68872f9e1e12234bd3fd9c3'}
WEIGHT_BYTES = 2149696167
HEADER_BYTES = 51871
HEADER_SHA = '32e5cd4fd8939775e11c655e29013d42a5f9294d461a2064d01540550e30b5c3'
DESIGN_PROTOCOL_SHA256 = '1d8bfa6eb27fee9a35c5ea4bbe3bccdf6957008576aa97ec05cf3a6e1f22ce5e'
STRIPE = 128
LAYERS = [0, 1, 31]
CASES = {'single': {'tokens': [1], 'offset': 0},
         'short': {'tokens': [1,7,13,3], 'offset': 0},
         'future_changed': {'tokens': [1,7,13,17], 'offset': 0},
         'offset5': {'tokens': [7,13,3], 'offset': 5},
         'tile17': {'tokens': ([1,7,13,3]*5)[:17], 'offset': 0},
         'offset509': {'tokens': [7,13,3], 'offset': 509}}
NUMERICAL = dict(num_layers=32,num_query_heads=32,num_kv_heads=32,head_dim=96,
                 rms_norm_eps=1e-5,rope_theta=10000.)
NATIVE_PROFILE = dict(projection='MlxAffine',norm='Compensated',attention='Reference',
                     activation='Half',kv='Half',capacity=512,requested_chunks=[32,128,3],
                     effective_chunk='min(requested,512-position_offset)',
                     readbacks=['FullLogits','GreedyToken'])
POLICY = {'atol_floor':1e-4,'envelope_multiplier':2,'rtol':1e-3,
          'absolute_limit':'max(1e-4,2*exact_array_FP32_FP64_max_abs)',
          'case_scope':'each exact case/stage and each greedy/scoring distribution separately',
          'comparison':'native versus independent main only; wide comparison diagnostic',
          'reference':'NumPy BLAS FP32/FP64 arithmetic; reconstructed FP32 affine weights; Half gather/norms/activations/KV; FP32 scores/probabilities/logits; ordinary split-half rotary. Not scalar-order or exact Metal/Torch/MLX emulation.',
          'cross_profile_greedy':'diagnostic only; native schedules/readback selection remain exact numeric',
          'equivalent_prefix':'diagnostic only; never enlarges this gate'}
SOURCE_PROFILE = dict(quantization_bits=4,group_size=64,metadata='F16',packed_word='little-endian U32; eight low-to-high nibbles',
                      tied_head=False,declared_positions=4096,declared_sliding_window=2047,implemented_positions=512,
                      main='NumPy2.3.2 FP32 BLAS',wide='NumPy2.3.2 FP64 BLAS',
                      main_rotary='FP32 exponent, pow, angle, trig and separate products/adds -> Half',
                      wide_rotary='Decimal80 frequency -> FP64 angle/trig/products -> Half',
                      embedding='selected original rows -> FP32 reconstruction -> Half gather',head='distinct original lm_head reconstructed FP32')
PRODUCER_NAMES = ('tools/reference/mlx_phi3_real.py','tools/reference/phi3_fp16.py',
                  'tools/reference/qwen_fp16.py','tools/reference/qwen.py',
                  'tools/bench/prepare.py','tools/experiments/native_usage.py')
MEMORY_LIMIT = 6*1024**3
MODEL_CONFIG = {'architectures': ['Phi3ForCausalLM'], 'attention_bias': False, 'attention_dropout': 0.0, 'auto_map': {'AutoConfig': 'configuration_phi3.Phi3Config', 'AutoModelForCausalLM': 'modeling_phi3.Phi3ForCausalLM'}, 'bos_token_id': 1, 'embd_pdrop': 0.0, 'eos_token_id': 32000, 'hidden_act': 'silu', 'hidden_size': 3072, 'initializer_range': 0.02, 'intermediate_size': 8192, 'max_position_embeddings': 4096, 'model_type': 'phi3', 'num_attention_heads': 32, 'num_hidden_layers': 32, 'num_key_value_heads': 32, 'original_max_position_embeddings': 4096, 'pad_token_id': 32000, 'quantization': {'group_size': 64, 'bits': 4}, 'resid_pdrop': 0.0, 'rms_norm_eps': 1e-05, 'rope_scaling': None, 'rope_theta': 10000.0, 'sliding_window': 2047, 'tie_word_embeddings': False, 'torch_dtype': 'bfloat16', 'transformers_version': '4.40.2', 'use_cache': True, 'vocab_size': 32064}


def admission():
    result = dict(source_mapping_bytes=WEIGHT_BYTES,current_reconstructed_matrix_bytes=32064*3072*4,
                  wide_linear_stripe_bytes=1024*8192*8,dequant_graph_blas_workspace_bytes=256*1024**2,
                  retained_dual_traces_publication_bytes=512*1024**2,framework_metadata_allowance_bytes=1024**3)
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
    for folder in site.glob('numpy-*.dist-info'):
        paths.extend(p for p in folder.rglob('*') if p.is_file())
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}


def runtime_identity():
    executable=Path(sys.executable).resolve()
    config=Path(sys.prefix)/'pyvenv.cfg'
    if not config.is_file() or not config.resolve().is_relative_to(ROOT):
        raise ValueError('Pinned repository virtual environment required')
    return dict(python_version=sys.version,executable=dict(path=str(executable),bytes=executable.stat().st_size,sha256=sha(executable)),
                venv_config=dict(path=str(config.resolve()),bytes=config.stat().st_size,sha256=sha(config)))


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
            or len(packed) == 0 or packed.shape[1] != columns//8 or scales.shape != (len(packed), columns//64)
            or biases.shape != scales.shape or not np.isfinite(scales).all() or not np.isfinite(biases).all()):
        raise ValueError('Selected affine shapes/metadata differ')
    # A Half scale times a four-bit integer is exactly representable in FP32.
    # FP64 reconstruction followed by one FP32 cast preserves the intended affine result.
    codes = np.stack([(packed >> shift) & 15 for shift in range(0,32,4)],axis=-1).reshape(len(packed),columns)
    value = (codes.astype(np.float64)*np.repeat(scales.astype(np.float64),64,axis=-1)
             + np.repeat(biases.astype(np.float64),64,axis=-1)).astype(np.float32)
    if not np.isfinite(value).all():raise ValueError('Nonfinite independent affine reconstruction')
    return value


def specifications(config):
    """Original physical header shapes plus independently authored logical spans."""
    h=config['hidden_size'];i=config['intermediate_size'];v=config['vocab_size']
    d=h//config['num_attention_heads'];kv=config['num_key_value_heads']*d
    physical={'model.embed_tokens.weight':[v,h],'lm_head.weight':[v,h],'model.norm.weight':[h]}
    selections={'lm_head.weight':('lm_head.weight',0,v,h)}
    for layer in range(config['num_hidden_layers']):
        p=f'model.layers.{layer}';short=f'layers.{layer}'
        for norm in ['input_layernorm','post_attention_layernorm']:physical[f'{p}.{norm}.weight']=[h]
        for name,rows,columns in [('self_attn.qkv_proj',h+2*kv,h),('self_attn.o_proj',h,h),
                                  ('mlp.gate_up_proj',2*i,h),('mlp.down_proj',h,i)]:
            physical[f'{p}.{name}.weight']=[rows,columns]
        for name,begin,end in [('q',0,h),('k',h,h+kv),('v',h+kv,h+2*kv)]:
            selections[f'{short}.{name}_proj.weight']=(f'{p}.self_attn.qkv_proj.weight',begin,end,h)
        for name,begin,end in [('gate',0,i),('up',i,2*i)]:
            selections[f'{short}.{name}_proj.weight']=(f'{p}.mlp.gate_up_proj.weight',begin,end,h)
        selections[f'{short}.o_proj.weight']=(f'{p}.self_attn.o_proj.weight',0,h,h)
        selections[f'{short}.down_proj.weight']=(f'{p}.mlp.down_proj.weight',0,h,i)
    expected={}
    for name,shape in physical.items():
        if len(shape)==1:expected[name]=('F16',shape)
        else:
            rows,columns=shape;module=name.removesuffix('.weight')
            if columns%64:raise ValueError('Original input columns must use complete group64')
            expected[name]=('U32',[rows,columns//8])
            expected[module+'.scales']=('F16',[rows,columns//64])
            expected[module+'.biases']=('F16',[rows,columns//64])
    return physical,selections,expected


def validate_header(header,index,expected,origin,file_bytes):
    if set(header)!=set(expected) or set(index.get('weight_map',{}))!=set(expected):
        raise ValueError('Selected tensor inventory differs')
    spans=[]
    for name,(dtype,shape) in expected.items():
        item=header[name]
        if (not isinstance(item,dict) or set(item)!={'dtype','shape','data_offsets'}
                or item['dtype']!=dtype or item['shape']!=shape
                or any(type(n) is not int for n in item['shape'])
                or not isinstance(item['data_offsets'],list) or len(item['data_offsets'])!=2
                or any(type(n) is not int for n in item['data_offsets'])):
            raise ValueError('Selected tensor metadata differs: '+name)
        begin,end=item['data_offsets']
        if (begin<0 or end-begin!=math.prod(shape)*(4 if dtype=='U32' else 2)
                or end>file_bytes-origin or index['weight_map'][name]!='model.safetensors'):
            raise ValueError('Selected tensor extent/index differs: '+name)
        spans.append((begin,end))
    cursor=0
    for begin,end in sorted(spans):
        if begin!=cursor:raise ValueError('Tensor spans overlap or contain gaps')
        cursor=end
    if cursor!=file_bytes-origin:raise ValueError('Unexpected tensor payload tail')


class Weights:
    """Only project() owns an expanded matrix; no public matrix/view escapes."""
    def __init__(self,folder):
        self.folder=Path(folder);self.stream=None;self.mapping=None;self.active=False
        self.config=bounded(self.folder/'config.json',1024**2,PINS['config.json'])
        self.index=bounded(self.folder/'model.safetensors.index.json',1024**2,PINS['model.safetensors.index.json'])
        if self.config!=MODEL_CONFIG:raise ValueError('Selected math/config differs')
        self.physical,self.selections,self.expected=specifications(self.config)

    def __enter__(self):
        try:
            self.stream=(self.folder/'model.safetensors').open('rb')
            if os.fstat(self.stream.fileno()).st_size!=WEIGHT_BYTES or descriptor_sha(self.stream)!=PINS['model.safetensors']:
                raise ValueError('Selected weight bytes differ')
            prefix=self.stream.read(8)
            if len(prefix)!=8:raise ValueError('Truncated tensor prefix')
            size=struct.unpack('<Q',prefix)[0]
            if size!=HEADER_BYTES or size>1024**2:raise ValueError('Selected tensor header extent differs')
            raw=self.stream.read(size)
            if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=HEADER_SHA:raise ValueError('Selected header bytes differ')
            self.header=json.loads(raw,object_pairs_hook=unique);self.origin=8+size
            self.header.pop('__metadata__',None)
            validate_header(self.header,self.index,self.expected,self.origin,WEIGHT_BYTES)
            self.mapping=mmap.mmap(self.stream.fileno(),0,access=mmap.ACCESS_READ)
            # All original floating metadata, including unselected embedding rows,
            # is checked in bounded row stripes. No expanded table is retained.
            import numpy as np
            for name,item in self.header.items():
                if item['dtype']=='F16':
                    for begin in range(0,item['shape'][0],STRIPE):
                        if not np.isfinite(self._raw_copy(name,begin,min(begin+STRIPE,item['shape'][0]))).all():
                            raise ValueError('Nonfinite original floating metadata: '+name)
            return self
        except BaseException as primary:
            self.__exit__(type(primary),primary,primary.__traceback__)
            raise

    def _raw_copy(self,name,start,end):
        """Byte-safe NumPy copy. Never return or retain a mapping-backed view.

        Copying the bounded packed/coefficient stripe also prevents traceback
        frames on downstream exceptions from keeping a live mmap export.
        """
        import numpy as np
        if self.mapping is None:raise ValueError('Closed source reader')
        item=self.header[name];shape=item['shape']
        if type(start) is not int or type(end) is not int or not 0<=start<end<=shape[0]:
            raise ValueError('Invalid original row span')
        dtype=np.dtype('<u4' if item['dtype']=='U32' else '<f2')
        stride=math.prod(shape[1:])*dtype.itemsize
        begin=self.origin+item['data_offsets'][0]+start*stride
        count=(end-start)*stride
        if begin<self.origin or begin+count>self.origin+item['data_offsets'][1] or begin+count>len(self.mapping):
            raise ValueError('Original row extent outside payload')
        # mmap byte slicing copies, then frombuffer views those private bytes.
        # The returned array owns independent storage; unaligned input is legal.
        return np.frombuffer(self.mapping[begin:begin+count],dtype=dtype).reshape([end-start,*shape[1:]]).copy()

    def _rows(self,name,begin,end,columns):
        module=name.removesuffix('.weight')
        return decode(self._raw_copy(name,begin,end),self._raw_copy(module+'.scales',begin,end),
                      self._raw_copy(module+'.biases',begin,end),columns)

    def _matrix(self,key):
        import numpy as np
        name,begin,end,columns=self.selections[key]
        result=np.empty((end-begin,columns),dtype=np.float32)
        for start in range(begin,end,STRIPE):
            stop=min(start+STRIPE,end)
            result[start-begin:stop-begin]=self._rows(name,start,stop,columns)
        return result

    def lookup(self,tokens):
        import numpy as np
        if (not isinstance(tokens,list) or not 0<len(tokens)<=512
                or any(type(i) is not int or not 0<=i<self.config['vocab_size'] for i in tokens)):
            raise ValueError('Invalid embedding IDs')
        h=self.config['hidden_size'];out=np.empty((len(tokens),h),dtype=np.float16)
        for row,token in enumerate(tokens):out[row]=self._rows('model.embed_tokens.weight',token,token+1,h)[0]
        if not np.isfinite(out).all():raise ValueError('Nonfinite gathered Half embedding')
        return out

    def norm(self,key):
        name='model.norm.weight' if key=='norm' else 'model.'+key+'.weight'
        if name not in self.physical or len(self.physical[name])!=1:raise ValueError('Unknown floating exclusion')
        result=self._raw_copy(name,0,self.physical[name][0])
        import numpy as np
        if not np.isfinite(result).all():raise ValueError('Nonfinite floating exclusion')
        return result

    def project(self,key,x,precision,output_dtype=None):
        import numpy as np
        from qwen_fp16 import linear
        if self.active:raise ValueError('Overlapping reconstructed matrix owners')
        self.active=True;matrix=None
        try:
            matrix=self._matrix(key)
            return linear(x,matrix,None,precision,np.float16 if output_dtype is None else output_dtype)
        except BaseException as error:
            matrix=None
            # Preserve exception/stack locations while releasing completed inner
            # frames' matrix/BLAS temporaries. The active project frame is skipped
            # by clear_frames and its own matrix reference is already empty.
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            matrix=None;self.active=False

    def verify(self):
        if self.stream is None or descriptor_sha(self.stream)!=PINS['model.safetensors']:
            raise ValueError('Consumed weight descriptor changed')
        for name,digest in PINS.items():
            if sha(self.folder/name)!=digest:raise ValueError('Checkpoint path changed: '+name)

    def __exit__(self,_kind,primary,_trace):
        failures=[]
        for name in ['mapping','stream']:
            value=getattr(self,name);setattr(self,name,None)
            if value is not None:
                try:value.close()
                except BaseException as error:failures.append(error)
        if failures:
            if primary is not None:
                for error in failures:primary.add_note('Weight cleanup: '+repr(error))
            else:raise failures[0]


def last_distribution(forward,tokens,offset,weights,precision):
    # copy severs the view into a complete token-by-vocabulary logits matrix.
    return forward(tokens,offset,weights,NUMERICAL,precision,[])['logits'][-1].copy()


def expected_arrays():
    """Literal graph boundaries; independent of exported/target array inventories."""
    arrays = {}
    for case, spec in CASES.items():
        n = len(spec['tokens'])
        stages = {'embedding': ([n,3072],'<f2')}
        for layer in LAYERS:
            for stage in ['input_norm','q_projection','k_projection','v_projection','q_rotary','k_rotary',
                          'attention_scores','attention_probabilities','attention_context','attention_projection',
                          'attention_residual','post_attention_norm','gate_projection','up_projection',
                          'gate_silu','gated','down_projection','output']:
                shape = [n,3072];dtype = '<f2'
                if stage in ['k_projection','v_projection']:shape = [n,3072]
                elif stage == 'q_rotary':shape = [n,32,96]
                elif stage == 'k_rotary':shape = [n,32,96]
                elif stage in ['attention_scores','attention_probabilities']:shape = [32,n,n];dtype = '<f4'
                elif stage in ['gate_projection','up_projection','gate_silu','gated']:shape = [n,8192]
                stages[f'layers.{layer}.{stage}'] = (shape,dtype)
        stages['final_norm'] = ([n,3072],'<f2')
        stages['logits'] = ([n,32064],'<f4')
        arrays.update({f'cases.{case}.{stage}': value for stage,value in stages.items()})
    arrays.update({f'greedy.step_{i}.logits': ([32064],'<f4') for i in range(4)})
    arrays.update({f'scoring.prefix_{i}.logits': ([32064],'<f4') for i in range(1,4)})
    return arrays


def validate_manifest(manifest):
    if (manifest['format'] != 'toklet-mlx-phi3-real-reference-v1' or manifest['policy'] != POLICY
            or manifest['cases'] != CASES or manifest['layers'] != LAYERS
            or manifest['model'] != dict(repository=REPO,revision=REVISION,file_pins=PINS)
            or manifest['numpy'] != '2.3.2' or manifest['numerical'] != NUMERICAL
            or manifest['source_profile'] != SOURCE_PROFILE or manifest['native_profile'] != NATIVE_PROFILE
            or manifest['design_protocol_sha256'] != DESIGN_PROTOCOL_SHA256 or manifest['source_cpu_graph_executed'] is not True
            or manifest['admission'] != admission()
            or any(manifest[key] is not False for key in ['target_gpu_executed','held_out_quality_qualified','physical_device_qualified'])):
        raise ValueError('Selected reference identity differs')
    producer_names = set(PRODUCER_NAMES)
    def digest(value):
        return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    if (set(manifest['producer_pins']) != producer_names or not manifest['package_pins']
            or any(not digest(value) for value in [manifest['array_sha256'],manifest['wide_array_sha256'],*manifest['producer_pins'].values(),*manifest['package_pins'].values()])
            or any(Path(name).is_absolute() or '..' in Path(name).parts or not name.startswith('.venv/')
                    for name in manifest['package_pins'])):
        raise ValueError('Reference producer identity differs')
    for marker in ['/numpy/__init__.py']:
        if not any(name.endswith(marker) for name in manifest['package_pins']):raise ValueError('Producer package source closure absent')
    for marker in ['/numpy/']:
        if not any(marker in name and name.endswith('.so') for name in manifest['package_pins']):raise ValueError('Producer native closure absent')
    runtime=manifest['runtime_identity']
    if not isinstance(runtime.get('python_version'),str) or not runtime['python_version']:
        raise ValueError('Missing Python runtime version')
    for name in ['executable','venv_config']:
        identity=runtime.get(name,{})
        if (set(identity)!={'path','bytes','sha256'} or not Path(identity['path']).is_absolute()
                or type(identity['bytes']) is not int or identity['bytes']<=0 or not digest(identity['sha256'])):
            raise ValueError('Missing Python runtime identity: '+name)
    expected = expected_arrays()
    if set(manifest['arrays']) != set(expected):raise ValueError('Reference coverage differs')
    cursor = 0
    for name,item in manifest['arrays'].items():
        shape,dtype = expected[name];gate = item['gate']
        if (set(gate) != {'atol','rtol','exact_array_fp32_fp64_max_abs'}
                or not digest(item['sha256']) or not digest(item['wide_sha256'])):raise ValueError('Reference gate/hash fields differ: '+name)
        if (not isinstance(item['shape'],list) or any(type(dimension) is not int or dimension <= 0 for dimension in item['shape'])
                or item['shape'] != shape or item['dtype'] != dtype or type(item['offset']) is not int
                or type(item['bytes']) is not int or item['offset'] != cursor
                or item['bytes'] != math.prod(shape)*(2 if dtype == '<f2' else 4)):
            raise ValueError('Reference array layout differs: '+name)
        cursor += item['bytes']
        errors = [gate['exact_array_fp32_fp64_max_abs']]
        if (any(type(value) not in [int,float] or not math.isfinite(value) or value < 0 for value in gate.values())
                or not math.isfinite(max(1e-4,2*errors[0]))
                or gate['rtol'] != 1e-3 or gate['atol'] != max(1e-4,2*errors[0])):
            raise ValueError('Reference numerical policy differs: '+name)
    if type(manifest['array_bytes']) is not int or manifest['array_bytes'] != cursor:
        raise ValueError('Reference total bytes differ')
    greedy = manifest['greedy']
    if (greedy['prompt'] != CASES['short']['tokens']
            or greedy['teacher_forcing'] != 'Independent FP64-accumulation greedy history'
            or any(len(greedy[key]) != 4 for key in ['fp32_tokens','fp64_tokens','fp64_logit_margins'])
            or any(type(token) is not int or not 0 <= token < 32064 for key in ['fp32_tokens','fp64_tokens'] for token in greedy[key])
            or any(type(value) not in [int,float] or not math.isfinite(value) or value < 0 for value in greedy['fp64_logit_margins'])):
        raise ValueError('Reference greedy contract differs')


def verify(folder, expected_sha=None, manifest_name='manifest.json'):
    with (folder/manifest_name).open('rb') as stream:data = stream.read(2*1024**2+1)
    if len(data)>2*1024**2:raise ValueError('Reference manifest exceeds limit')
    if expected_sha is not None and hashlib.sha256(data).hexdigest()!=expected_sha:
        raise ValueError('Frozen manifest SHA differs')
    manifest=json.loads(data,object_pairs_hook=unique);validate_manifest(manifest)
    main_path,wide_path=folder/'arrays.bin',folder/'wide.bin'
    whole=hashlib.sha256();wide_whole=hashlib.sha256();selected={};wide_selected={};envelopes={}
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
            envelopes[name]=error
        if (main.read(1) or wide.read(1) or whole.hexdigest()!=manifest['array_sha256']
                or wide_whole.hexdigest()!=manifest['wide_array_sha256']):raise ValueError('Reference payload differs')
    for name,item in manifest['arrays'].items():
        if item['gate']['exact_array_fp32_fp64_max_abs']!=envelopes[name]:
            raise ValueError('Envelope does not match preserved main/wide bits: '+name)
    saved_greedy(selected,manifest)
    saved_greedy(wide_selected,manifest,wide=True)
    return manifest


def saved_greedy(selected_logits,manifest,wide=False):
    ids=[max(range(32064),key=lambda i:selected_logits[f'greedy.step_{step}.logits'][i]) for step in range(4)]
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
        try:(folder/'failure.json').write_text(json.dumps(dict(completed=False,source_execution_status='incomplete; source graph may have run; inspect progress.jsonl',target_gpu_executed=False,error_type=type(error).__name__,error=str(error),notes=getattr(error,'__notes__',[])),indent=2)+'\n')
        except BaseException as cleanup:error.add_note('Failure receipt write failed: '+repr(cleanup))
        raise


def export_into(folder):
    for variable in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:
        os.environ[variable]='1'
    producers={name:sha(ROOT/name) for name in PRODUCER_NAMES}
    import importlib.metadata
    if importlib.metadata.version('numpy')!='2.3.2':raise ValueError('Pinned NumPy differs')
    binaries=package_pins();runtime=runtime_identity()
    import numpy as np
    import phi3_fp16
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
            main = phi3_fp16.forward(spec['tokens'],spec['offset'],weights,NUMERICAL,np.float32,LAYERS)
            wide = phi3_fp16.forward(spec['tokens'],spec['offset'],weights,NUMERICAL,np.float64,LAYERS)
            for stage,value in main.items():
                snapshots[f'cases.{case}.{stage}'] = (value,wide[stage])
            usage.append(record(case))
        history = list(CASES['short']['tokens']);tokens=[];wide_tokens=[];margins=[]
        for step in range(4):
            main = last_distribution(phi3_fp16.forward,history,0,weights,np.float32)
            wide = last_distribution(phi3_fp16.forward,history,0,weights,np.float64)
            snapshots[f'greedy.step_{step}.logits'] = (main,wide)
            tokens.append(int(np.argmax(main)));token=int(np.argmax(wide));wide_tokens.append(token)
            ordered=np.sort(wide);margins.append(float(ordered[-1])-float(ordered[-2]));history.append(token);usage.append(record('greedy.'+str(step)))
        greedy=dict(prompt=CASES['short']['tokens'],teacher_forcing='Independent FP64-accumulation greedy history',
                    fp32_tokens=tokens,fp64_tokens=wide_tokens,fp64_logit_margins=margins)
        for count in range(1,4):
            ts=CASES['short']['tokens'][:count]
            main=last_distribution(phi3_fp16.forward,ts,0,weights,np.float32)
            wide=last_distribution(phi3_fp16.forward,ts,0,weights,np.float64)
            snapshots[f'scoring.prefix_{count}.logits']=(main,wide);usage.append(record('scoring.'+str(count)))
        weights.verify()
    arrays={};raw=bytearray();wide_raw=bytearray()
    for key,(main,wide) in snapshots.items():
        if not np.isfinite(main).all() or not np.isfinite(wide).all():raise ValueError('Nonfinite reference')
        dtype='<f2' if main.dtype==np.float16 else '<f4';data=main.astype(dtype).tobytes();wide_data=wide.astype(dtype).tobytes()
        envelope=float(np.max(np.abs(main.astype(np.float64)-wide.astype(np.float64))))
        arrays[key]=dict(offset=len(raw),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),wide_sha256=hashlib.sha256(wide_data).hexdigest(),dtype=dtype,shape=list(main.shape),
                         gate=dict(atol=max(1e-4,2*envelope),rtol=1e-3,exact_array_fp32_fp64_max_abs=envelope))
        raw.extend(data);wide_raw.extend(wide_data)
    if len(arrays)!=349 or len(raw)!=19025664:raise ValueError('Independent reference coverage differs')
    for name,digest in {**producers,**binaries}.items():
        if sha(ROOT/name)!=digest:raise ValueError('Oracle producer changed: '+name)
    for name,digest in PINS.items():
        if sha(MODEL/name)!=digest:raise ValueError('Input changed before publication')
    if package_pins()!=binaries or runtime_identity()!=runtime:raise ValueError('Producer runtime closure changed')
    (folder/'arrays.bin').write_bytes(raw)
    (folder/'wide.bin').write_bytes(wide_raw)
    manifest=dict(format='toklet-mlx-phi3-real-reference-v1',model=dict(repository=REPO,revision=REVISION,file_pins=PINS),
                  policy=POLICY,cases=CASES,layers=LAYERS,producer_pins=producers,package_pins=binaries,runtime_identity=runtime,numpy=np.__version__,numerical=NUMERICAL,source_profile=SOURCE_PROFILE,native_profile=NATIVE_PROFILE,
                  design_protocol_sha256=DESIGN_PROTOCOL_SHA256,source_cpu_graph_executed=True,
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
