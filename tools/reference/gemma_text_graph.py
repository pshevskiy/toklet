"""Complete tiny token-only Gemma4 text graph, before native graph integration.

The independent ideal and staged contracts share explicit FP32 inputs, not an
upstream implementation. The source comparison executes authenticated class
bodies and real MLX modules. It grants no checkpoint, long-context or device fit.
"""
import argparse
import ast
from decimal import Decimal, localcontext
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import sys
from types import SimpleNamespace

import reference_runtime as runtime

FORMAT = 'toklet-gemma-text-graph-math-v1'
REVISION = '0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
SOURCES = {
    'gemma4_text.py': '73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d',
    'base.py': '63d8e58a4b6f8c0902719499c693647fbf486bbd5d22b09868a07d37e08c83d6',
    'cache.py': 'c96cb169fd5e0ec4584a162dd61dc62ab08ccdca03a9850f8a491ce6809145e6',
    'rope_utils.py': 'ccba2f0c7b246c008a1188dc3760c162bdd906b15a1c490461dee4be64bc62df',
}
MAX_JSON, MAX_RAW = 16 * 1024 * 1024, 16 * 1024 * 1024
MAX_OBSERVATION_JSON = 8 * 1024 * 1024
ATOL, RTOL = 1e-5, 1e-4
LOCAL, FULL = 'sliding_attention', 'full_attention'
CONFIG = dict(model_type='gemma4_text', hidden_size=8, num_hidden_layers=4,
    intermediate_size=12, num_attention_heads=2, head_dim=8, global_head_dim=16,
    global_partial_rotary_factor=.25, rms_norm_eps=1e-6, vocab_size=17,
    vocab_size_per_layer_input=17, num_key_value_heads=1,
    num_global_key_value_heads=None, num_kv_shared_layers=2, pad_token_id=0,
    hidden_size_per_layer_input=2, rope_traditional=False, partial_rotary_factor=1.,
    rope_parameters={LOCAL: dict(partial_rotary_factor=1., rope_theta=10000., rope_type='default'),
                     FULL: dict(partial_rotary_factor=.25, rope_theta=1e6, rope_type='proportional')},
    sliding_window=4, sliding_window_pattern=2, max_position_embeddings=32,
    attention_k_eq_v=False, final_logit_softcapping=30., use_double_wide_mlp=True,
    enable_moe_block=False, num_experts=None, top_k_experts=None,
    moe_intermediate_size=None, layer_types=[LOCAL, FULL, LOCAL, FULL], tie_word_embeddings=True)
SCHEDULES = (('prefill32', (32,)), ('decode32', (1,) * 32),
             ('mixed32', (3, 1, 4, 2, 7, 1, 8, 6)))
NORMS = ('input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm',
         'post_feedforward_layernorm', 'post_per_layer_input_norm')
EXCLUSIONS = ['ModelArgs/dataclass parsing', 'sanitize/quant_predicate and checkpoint loading',
    'BF16 or quantized weights/activations/KV', 'multimodal inputs and embedding-only lookup recovery',
    'MoE, K-equals-V, untied head, missing PLE and alternate config branches',
    'native session atomicity, cancellation, prefix reuse and sampling',
    'positions above 31, real-model quality, performance and device memory fit']
SCOPE = 'Tiny dense FP32 token-only graph; exact selected source bodies, constructors for this explicit configuration, normal cache updates and every absolute position 0..31'


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def same(a, b):
    if type(a) is not type(b): return False
    if isinstance(a, dict): return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def weight_shapes():
    c = CONFIG; h, p, n = c['hidden_size'], c['hidden_size_per_layer_input'], c['num_hidden_layers']
    result = {'model.embed_tokens.weight': [17, h], 'model.embed_tokens_per_layer.weight': [17, n*p],
        'model.per_layer_model_projection.weight': [n*p, h], 'model.per_layer_projection_norm.weight': [p],
        'model.norm.weight': [h]}
    for i, kind in enumerate(c['layer_types']):
        prefix = f'model.layers.{i}.'; d = 8 if kind == LOCAL else 16; intermediate = 12 if i < 2 else 24
        result[prefix + 'self_attn.q_proj.weight'] = [2*d, h]
        result[prefix + 'self_attn.q_norm.weight'] = [d]
        result[prefix + 'self_attn.o_proj.weight'] = [h, 2*d]
        if i < 2:
            for key in ('k_proj', 'v_proj'): result[prefix + 'self_attn.' + key + '.weight'] = [d, h]
            result[prefix + 'self_attn.k_norm.weight'] = [d]
        for key in ('gate_proj', 'up_proj'): result[prefix + 'mlp.' + key + '.weight'] = [intermediate, h]
        result[prefix + 'mlp.down_proj.weight'] = [h, intermediate]
        for key in NORMS: result[prefix + key + '.weight'] = [h]
        result[prefix + 'per_layer_input_gate.weight'] = [p, h]
        result[prefix + 'per_layer_projection.weight'] = [h, p]
        result[prefix + 'layer_scalar'] = [1]
    return result


def operands():
    import numpy as np
    result = {}
    for name, shape in weight_shapes().items():
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little'); values = []
        for _ in range(math.prod(shape)):
            seed = (1664525 * seed + 1013904223) % (1 << 32)
            values.append(((seed >> 16) % 257 - 128) / 512)
        if name.endswith('layer_scalar'):
            values = [(1., .875, 1.125, -.75)[int(name.split('.')[2])]]
        elif len(shape) == 1:
            values = [1. + value for value in values]
        result[name] = np.array(values, np.float32).reshape(shape)
    return result


def token_ids():
    # Repeats and non-monotonic IDs make an accidental position/ID substitution visible.
    return [((position * position + 3 * position + 7) % 17) for position in range(32)]


def mathematics(weights, ids, staged):
    import numpy as np
    if type(staged) is not bool or not isinstance(ids,list) or len(ids)!=32 or any(type(x) is not int or not 0<=x<17 for x in ids):
        raise ValueError('Exact bounded text-graph IDs and precision contract required')
    shapes = weight_shapes()
    if not isinstance(weights, dict) or set(weights) != set(shapes): raise ValueError('Exact text-graph weights required')
    for name, shape in shapes.items():
        value = weights[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype('float32') or list(value.shape) != shape or not np.isfinite(value).all():
            raise ValueError('Finite FP32 text-graph weight shape required')
    r = {}; dtype = np.float32 if staged else np.float64
    def rounded(x): return np.asarray(x, dtype=dtype)
    def save(name, x): r[name] = rounded(x); return r[name]
    def linear(x, key):
        w = weights[key].astype(np.float64); flat = x.astype(np.float64).reshape(-1, x.shape[-1])
        out = [[math.fsum(float(a)*float(b) for a,b in zip(row,weight)) for weight in w] for row in flat]
        return rounded(np.array(out).reshape(*x.shape[:-1], w.shape[0]))
    def norm(x, key=None):
        flat = x.astype(np.float64).reshape(-1, x.shape[-1]); w = np.ones(x.shape[-1]) if key is None else weights[key].astype(np.float64)
        # The native scalar normalizer admits the FP32 epsilon value.
        eps = struct.unpack('<f', struct.pack('<f', CONFIG['rms_norm_eps']))[0]
        out = [[float(v)/math.sqrt(math.fsum(float(y)*float(y) for y in row)/len(row)+eps)*float(scale) for v,scale in zip(row,w)] for row in flat]
        return rounded(np.array(out).reshape(x.shape))
    def gelu(x):
        return rounded(np.array([.5*float(v)*(1+math.tanh(math.sqrt(2/math.pi)*(float(v)+.044715*float(v)**3))) for v in x.flat]).reshape(x.shape))
    def rotate(x, base, rotated):
        out = x.astype(np.float64).copy(); dims = x.shape[-1]
        with localcontext() as ctx:
            ctx.prec = 80
            for pair in range(rotated//2):
                freq = Decimal.from_float(base) ** (Decimal(2*pair)/Decimal(dims))
                for pos in range(32):
                    angle = float(Decimal(pos)/freq); cosine, sine = math.cos(angle), math.sin(angle)
                    if staged: cosine, sine = float(np.float32(cosine)), float(np.float32(sine))
                    a, b = x[0,:,pos,pair].astype(np.float64), x[0,:,pos,pair+dims//2].astype(np.float64)
                    if staged:
                        # Match the existing scalar rotary's explicit FP32 products.
                        out[0,:,pos,pair] = rounded(a*cosine).astype(np.float64)-rounded(b*sine).astype(np.float64)
                        out[0,:,pos,pair+dims//2] = rounded(a*sine).astype(np.float64)+rounded(b*cosine).astype(np.float64)
                    else:
                        out[0,:,pos,pair] = a*cosine-b*sine; out[0,:,pos,pair+dims//2] = a*sine+b*cosine
        return rounded(out)
    def attention(q,k,v,local):
        out = np.empty(q.shape, dtype=dtype)
        for head in range(2):
            for pos in range(32):
                start = max(0,pos-3) if local else 0
                score = rounded([math.fsum(float(a)*float(b) for a,b in zip(q[0,head,pos],key)) for key in k[0,0,start:pos+1]])
                exp = [math.exp(float(s)-float(max(score))) for s in score]; total = math.fsum(exp)
                prob = rounded([s/total for s in exp])
                out[0,head,pos] = rounded([math.fsum(float(prob[t])*float(v[0,0,start+t,d]) for t in range(len(prob))) for d in range(q.shape[-1])])
        return out
    x = save('embedding', weights['model.embed_tokens.weight'][ids][None])
    x = save('scaled_embedding', x.astype(np.float64)*float(np.float32(math.sqrt(8))) if staged else x*math.sqrt(8))
    lookup = save('ple_lookup', weights['model.embed_tokens_per_layer.weight'][ids][None])
    lookup = rounded(lookup.astype(np.float64)*(float(np.float32(math.sqrt(2))) if staged else math.sqrt(2))).reshape(1,32,4,2)
    projection = save('ple_projection', linear(x,'model.per_layer_model_projection.weight'))
    projection = rounded(projection.astype(np.float64)*(float(np.float32(8**-.5)) if staged else 8**-.5)).reshape(1,32,4,2)
    projection = save('ple_norm', norm(projection,'model.per_layer_projection_norm.weight'))
    ple = rounded(rounded(projection.astype(np.float64)+lookup.astype(np.float64)).astype(np.float64)*(float(np.float32(2**-.5)) if staged else 2**-.5))
    kvs = {}
    for i, kind in enumerate(CONFIG['layer_types']):
        p = f'layer.{i}.'; w = f'model.layers.{i}.'; d = 8 if kind == LOCAL else 16
        save(p+'input',x); save(p+'ple_input',ple[:,:,i,:])
        y = save(p+'input_layernorm',norm(x,w+'input_layernorm.weight'))
        q = save(p+'self_attn.q_proj',linear(y,w+'self_attn.q_proj.weight'))
        q = save(p+'self_attn.q_norm',norm(q.reshape(1,32,2,d),w+'self_attn.q_norm.weight'))
        q = save(p+'q_rotary',rotate(q.transpose(0,2,1,3),1e4 if kind==LOCAL else 1e6,d if kind==LOCAL else 4))
        if i < 2:
            k = save(p+'self_attn.k_proj',linear(y,w+'self_attn.k_proj.weight'))
            v = save(p+'self_attn.v_proj',linear(y,w+'self_attn.v_proj.weight'))
            k = save(p+'self_attn.k_norm',norm(k.reshape(1,32,1,d),w+'self_attn.k_norm.weight'))
            v = save(p+'self_attn.v_norm',norm(v.reshape(1,32,1,d)))
            k = save(p+'k_rotary',rotate(k.transpose(0,2,1,3),1e4 if kind==LOCAL else 1e6,d if kind==LOCAL else 4))
            v = v.transpose(0,2,1,3); kvs[kind] = (k,v)
        k,v = kvs[kind]
        y = save(p+'context',attention(q,k,v,kind==LOCAL))
        y = save(p+'self_attn.o_proj',linear(y.transpose(0,2,1,3).reshape(1,32,2*d),w+'self_attn.o_proj.weight'))
        y = save(p+'post_attention_layernorm',norm(y,w+'post_attention_layernorm.weight'))
        x = rounded(x.astype(np.float64)+y.astype(np.float64)); residual=x
        y = save(p+'pre_feedforward_layernorm',norm(x,w+'pre_feedforward_layernorm.weight'))
        gate=save(p+'mlp.gate_proj',linear(y,w+'mlp.gate_proj.weight')); up=save(p+'mlp.up_proj',linear(y,w+'mlp.up_proj.weight'))
        y=rounded(gelu(gate).astype(np.float64)*up.astype(np.float64))
        y=save(p+'mlp.down_proj',linear(y,w+'mlp.down_proj.weight'))
        y=save(p+'post_feedforward_layernorm',norm(y,w+'post_feedforward_layernorm.weight'))
        x=rounded(residual.astype(np.float64)+y.astype(np.float64));residual=x
        y=save(p+'per_layer_input_gate',linear(x,w+'per_layer_input_gate.weight'))
        y=rounded(gelu(y).astype(np.float64)*ple[:,:,i,:].astype(np.float64))
        y=save(p+'per_layer_projection',linear(y,w+'per_layer_projection.weight'))
        y=save(p+'post_per_layer_input_norm',norm(y,w+'post_per_layer_input_norm.weight'))
        x=rounded(residual.astype(np.float64)+y.astype(np.float64))
        x=save(p+'output',x.astype(np.float64)*weights[w+'layer_scalar'].astype(np.float64))
    x=save('final_norm',norm(x,'model.norm.weight'))
    x=save('head',linear(x,'model.embed_tokens.weight'))
    # This oracle keeps the explicitly staged division and tanh boundaries.
    y=rounded(x.astype(np.float64)/30.)
    y=rounded(np.array([math.tanh(float(v)) for v in y.flat]).reshape(y.shape))
    save('logits',y.astype(np.float64)*30.)
    return r


def describe(x):
    return dict(dtype='<f4' if x.dtype.itemsize==4 else '<f8',shape=list(x.shape),values=x.reshape(-1).tolist())


def metadata():
    return dict(format=FORMAT,source_revision=REVISION,source_sha256=SOURCES,config=CONFIG,
        producer_runtime=dict(python='3.12.8',numpy=runtime.VERSIONS['numpy']),
        observation_scope=SCOPE,exclusions=EXCLUSIONS,
        precision='Exact FP32 weights; FP64 fsum reductions/Decimal80 rotary frequencies and angles; separate ideal and operator-staged FP32 contracts, including separate FP32 rotary products',
        native_graph_implemented=False,checkpoint_qualified=False,device_qualified=False,
        upstream_long_rotary_gate_closed=False,
        schedules=[dict(name=name,chunks=list(chunks)) for name,chunks in SCHEDULES])


def export():
    weights=operands();ids=token_ids()
    return encoded(dict(**metadata(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        token_ids=ids,weights={k:describe(v) for k,v in weights.items()},
        math={k:describe(v) for k,v in mathematics(weights,ids,False).items()},
        staged_fp32={k:describe(v) for k,v in mathematics(weights,ids,True).items()}))


def stage_shapes():
    r=dict(embedding=[1,32,8],scaled_embedding=[1,32,8],ple_lookup=[1,32,8],
        ple_projection=[1,32,8],ple_norm=[1,32,4,2],final_norm=[1,32,8],head=[1,32,17],logits=[1,32,17])
    for i,kind in enumerate(CONFIG['layer_types']):
        p=f'layer.{i}.';d=8 if kind==LOCAL else 16;intermediate=12 if i<2 else 24
        r[p+'ple_input']=[1,32,2]
        for key in ('input','output',*NORMS,'self_attn.o_proj','mlp.down_proj','per_layer_projection'):
            r[p+key]=[1,32,8]
        r[p+'self_attn.q_proj']=[1,32,2*d];r[p+'self_attn.q_norm']=[1,32,2,d]
        for key in ('q_rotary','context'):r[p+key]=[1,2,32,d]
        for key in ('mlp.gate_proj','mlp.up_proj'):r[p+key]=[1,32,intermediate]
        r[p+'per_layer_input_gate']=[1,32,2]
        if i<2:
            for key in ('k_proj','v_proj'):r[p+'self_attn.'+key]=[1,32,d]
            for key in ('k_norm','v_norm'):r[p+'self_attn.'+key]=[1,32,1,d]
            r[p+'k_rotary']=[1,1,32,d]
    return r


def verify(path,expected_sha):
    data=json.loads(runtime.captured(path,MAX_JSON,runtime.sha(expected_sha)),object_pairs_hook=runtime.json_unique,
        parse_constant=runtime.bad_json_number,parse_float=runtime.finite_json_number)
    wanted=metadata()
    if (not isinstance(data,dict) or set(data)!={*wanted,'producer_sha256','token_ids','weights','math','staged_fp32'}
            or not same({k:data[k] for k in wanted},wanted) or not same(data['token_ids'],token_ids())):
        raise ValueError('Exact tiny text-graph identity required')
    runtime.sha(data['producer_sha256'])
    for group,shapes in [('weights',weight_shapes()),('math',stage_shapes()),('staged_fp32',stage_shapes())]:
        if not isinstance(data[group],dict) or set(data[group])!=set(shapes):raise ValueError('Exact graph array inventory required')
        for key,shape in shapes.items():
            item=data[group][key];dtype='<f8' if group=='math' else '<f4'
            if (not isinstance(item,dict) or set(item)!={'dtype','shape','values'} or item['dtype']!=dtype
                    or not same(item['shape'],shape) or not isinstance(item['values'],list) or len(item['values'])!=math.prod(shape)):
                raise ValueError('Exact graph array descriptor required')
            for value in item['values']:
                if type(value) not in (float,int) or not -sys.float_info.max<=value<=sys.float_info.max or not math.isfinite(value):
                    raise ValueError('Finite graph array value required')
                if dtype=='<f4':
                    try:
                        if struct.unpack('<f',struct.pack('<f',value))[0]!=value:raise ValueError('Exact FP32 graph value required')
                    except OverflowError as error:raise ValueError('FP32 graph value overflow') from error
    return data


def source_nodes(sources):
    selected={
        'base.py':('create_causal_mask','create_attention_mask','scaled_dot_product_attention'),
        'cache.py':('create_attention_mask','_BaseCache','KVCache','RotatingKVCache'),
        'rope_utils.py':('ProportionalRoPE','initialize_rope'),
        'gemma4_text.py':('RMSNormNoScale','logit_softcap','geglu','MLP','Attention','DecoderLayer','Gemma4TextModel','Model')}
    result={}
    for filename,names in selected.items():
        tree=ast.parse(sources[filename]);nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        if len(nodes)!=len(names) or {n.name for n in nodes}!=set(names):raise ValueError('Unique selected graph source definitions required')
        result[filename]=nodes
    return result


def compare(reference,reference_sha,sources,device,destination,runtime_manifest,runtime_sha):
    if device not in ('cpu','gpu'):raise ValueError('Explicit graph source device required')
    data=verify(reference,reference_sha)
    if set(sources)!=set(SOURCES) or any(not isinstance(raw,bytes) or len(raw)>128*1024 or hashlib.sha256(raw).hexdigest()!=SOURCES[name] for name,raw in sources.items()):
        raise ValueError('Authenticated graph sources differ')
    nodes=source_nodes(sources)
    admitted=runtime.admit_runtime(runtime_manifest,runtime_sha,[Path(__file__)])
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    if any(module.__file__!=admitted['imports'][name] for name,module in [('numpy',np),('mlx.core',mx),('mlx.nn',nn)]):
        raise ValueError('Actual graph imports differ from admitted files')
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):raise ValueError('Empty regular graph output required')
    previous_device=mx.default_device();previous_signals={};rows=[];comparisons=[];blob=bytearray();primary=None;cleanup=[]
    active=dict(row=None,before=0,count=0,layer=None,returned={},rope_calls={})
    def interrupted(number,_):
        for value in (signal.SIGTERM,signal.SIGINT):signal.signal(value,signal.SIG_IGN)
        raise InterruptedError('Received signal '+str(number))
    def array(group,name):
        item=data[group][name];return np.array(item['values'],dtype=item['dtype']).reshape(item['shape'])
    def observe(name,value,expected_name=None,axis=1):
        row=active['row']
        if row is None:return value
        if name in row['arrays']:raise ValueError('Duplicate source graph observation: '+name)
        mx.eval(value);actual=np.asarray(value)
        if actual.dtype.hasobject or actual.dtype.kind not in 'biuf' or actual.nbytes>MAX_RAW:
            row['arrays'][name]=dict(raw_unavailable='Unsupported or oversized numeric observation',dtype=str(actual.dtype),shape=list(actual.shape))
            raise ValueError('Bounded numeric graph observation required')
        dtype=actual.dtype.newbyteorder('<');raw=np.ascontiguousarray(actual,dtype=dtype).tobytes();padding=(-len(blob))%8
        if len(blob)+padding+len(raw)>MAX_RAW:raise ValueError('Raw graph observation bound exceeded')
        blob.extend(b'\0'*padding);row['arrays'][name]=dict(dtype=dtype.str,shape=list(actual.shape),offset=len(blob),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());blob.extend(raw)
        if expected_name is None:expected_name=name
        expected=array('math',expected_name);selection=[slice(None)]*expected.ndim
        selection[axis]=slice(active['before'],active['before']+active['count']);selection=tuple(selection)
        if actual.dtype!=np.dtype('float32') or actual.shape!=expected[selection].shape or not np.isfinite(actual).all():raise ValueError('Source graph dtype/shape/finite result differs')
        for contract in ('math','staged_fp32'):
            wanted=array(contract,expected_name)[selection].astype(np.float64);difference=np.abs(actual.astype(np.float64)-wanted)
            ratio=difference/(ATOL+RTOL*np.abs(wanted))
            comparisons.append(dict(schedule=row['schedule'],trace=row['trace'],step=row['step'],name=name,expected_name=expected_name,
                contract=contract,axis=axis,passed=bool(np.all(ratio<=1)),maximum_absolute_error=float(difference.max()),maximum_error_to_policy_ratio=float(ratio.max())))
        return value
    def execute(traced):
        deferred=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        def compile_nodes(filename,namespace):
            module=ast.fix_missing_locations(ast.Module(body=[deferred,*nodes[filename]],type_ignores=[]))
            exec(compile(module,'authenticated-gemma-text-graph/'+filename,'exec'),namespace)
        base=dict(mx=mx);compile_nodes('base.py',base)
        cache=dict(mx=mx,create_causal_mask=base['create_causal_mask']);compile_nodes('cache.py',cache)
        rope=dict(mx=mx,nn=nn);compile_nodes('rope_utils.py',rope)
        env=dict(mx=mx,nn=nn,partial=partial,create_attention_mask=base['create_attention_mask'],
            scaled_dot_product_attention=base['scaled_dot_product_attention'],initialize_rope=rope['initialize_rope'],
            KVCache=cache['KVCache'],RotatingKVCache=cache['RotatingKVCache'])
        compile_nodes('gemma4_text.py',env)
        if traced:
            original_decoder,original_attention=env['DecoderLayer'],env['Attention']
            class TraceDecoder(original_decoder):
                def __call__(self,x,mask=None,cache=None,per_layer_input=None,shared_kv=None,offset=None):
                    p=f'layer.{self.layer_idx}.'
                    observe(p+'input',x);observe(p+'ple_input',per_layer_input)
                    result=super().__call__(x,mask,cache,per_layer_input=per_layer_input,shared_kv=shared_kv,offset=offset)
                    observe(p+'output',result[0]);return result
            class TraceAttention(original_attention):
                def __call__(self,x,mask=None,cache=None,shared_kv=None,offset=None):
                    i=self.layer_idx;active['layer']=i;active['rope_calls'][i]=0
                    producer=i if i<2 else i-2
                    if i<2:
                        if cache is None or cache.offset!=active['before'] or shared_kv is not None or offset is not None:
                            raise ValueError('Producer source cache/offset forwarding differs')
                    else:
                        expected,expected_offset=active['returned'][producer]
                        if cache is not None or shared_kv is None or any(a is not b for a,b in zip(shared_kv,expected)) or offset is not expected_offset:
                            raise ValueError('Shared source KV/offset identity differs')
                    result=super().__call__(x,mask,cache,shared_kv=shared_kv,offset=offset)
                    if int(result[2])!=active['before']:raise ValueError('Absolute source rotary offset differs')
                    active['returned'][i]=(result[1],result[2])
                    active['row']['routes'].append(dict(layer=i,producer=producer,offset=int(result[2]),cache_present=cache is not None,
                        kv_shape=list(result[1][0].shape),normal_update_only=True))
                    return result
            env['DecoderLayer'],env['Attention']=TraceDecoder,TraceAttention
            source_sdpa=env['scaled_dot_product_attention']
            def sdpa(q,k,v,cache,scale,mask):
                if scale!=1.:raise ValueError('Gemma source attention scale changed')
                result=source_sdpa(q,k,v,cache=cache,scale=scale,mask=mask)
                observe(f"layer.{active['layer']}.context",result,axis=2);return result
            env['scaled_dot_product_attention']=sdpa
        model=env['Model'](SimpleNamespace(**CONFIG))
        def learned_parameters():
            found={}
            def walk(tree,prefix):
                if isinstance(tree,mx.array):found[prefix]=tree
                elif isinstance(tree,(dict,list,tuple)):
                    entries=tree.items() if isinstance(tree,dict) else enumerate(tree)
                    for key,value in entries:walk(value,(prefix+'.' if prefix else '')+str(key))
                else:raise ValueError('Unexpected learned parameter tree leaf')
            walk(model.parameters(),'')
            if set(found)!=set(weight_shapes()):raise ValueError('Complete learned parameter inventory differs')
            for name,value in found.items():
                if list(value.shape)!=weight_shapes()[name] or value.dtype!=mx.float32:raise ValueError('Learned parameter shape/dtype differs')
            return found
        learned_parameters()
        for name in weight_shapes():
            parts=name.split('.');obj=model
            for part in parts[:-1]:obj=obj[int(part)] if part.isdigit() else getattr(obj,part)
            setattr(obj,parts[-1],mx.array(array('weights',name)))
        handles=learned_parameters()
        for name,value in handles.items():
            mx.eval(value)
            if not np.array_equal(np.asarray(value),array('weights',name)):raise ValueError('Learned parameter value differs from admitted weight')
        # Private derived RoPE frequencies are not learned checkpoint tensors;
        # their source constructor and executed rotary outputs are checked separately.
        if model.model.previous_kvs!=[0,1,0,1] or hasattr(model,'lm_head'):raise ValueError('Concrete tied/shared constructor choices differ')
        if [layer.mlp.gate_proj.weight.shape[0] for layer in model.model.layers]!=[12,12,24,24]:raise ValueError('Concrete double-width MLP selection differs')
        for i,layer in enumerate(model.model.layers):
            if layer.self_attn.has_kv!=(i<2) or hasattr(layer.self_attn,'k_proj')!=(i<2) or hasattr(layer.self_attn,'v_proj')!=(i<2):
                raise ValueError('Concrete KV projection ownership differs')
        if traced:
            class Tap(nn.Module):
                def __init__(self,module,name):super().__init__();self.module,self.name=module,name
                def __call__(self,*args,**kwargs):
                    if self.name=='ple_projection':observe('scaled_embedding',args[0])
                    return observe(self.name,self.module(*args,**kwargs))
                def as_linear(self,x):return observe('head',self.module.as_linear(x))
            class RopeTap(nn.Module):
                def __init__(self,module,index):super().__init__();self.module,self.index=module,index
                def __call__(self,x,offset=0):
                    call=active['rope_calls'][self.index];active['rope_calls'][self.index]+=1
                    role='k_rotary' if self.index<2 and call==0 else 'q_rotary'
                    return observe(f'layer.{self.index}.'+role,self.module(x,offset=offset),axis=2)
            text=model.model
            text.embed_tokens=Tap(text.embed_tokens,'embedding')
            text.embed_tokens_per_layer=Tap(text.embed_tokens_per_layer,'ple_lookup')
            text.per_layer_model_projection=Tap(text.per_layer_model_projection,'ple_projection')
            text.per_layer_projection_norm=Tap(text.per_layer_projection_norm,'ple_norm')
            text.norm=Tap(text.norm,'final_norm')
            for i,layer in enumerate(text.layers):
                for parent,keys in [(layer,NORMS+('per_layer_input_gate','per_layer_projection')),
                        (layer.mlp,('gate_proj','up_proj','down_proj')),
                        (layer.self_attn,('q_proj','q_norm','o_proj')+(('k_proj','v_proj','k_norm','v_norm') if i<2 else ()) )]:
                    for key in keys:
                        sub='mlp.' if parent is layer.mlp else 'self_attn.' if parent is layer.self_attn else ''
                        setattr(parent,key,Tap(getattr(parent,key),f'layer.{i}.'+sub+key))
                layer.self_attn.rope=RopeTap(layer.self_attn.rope,i)
        return model,handles
    try:
        for value in (signal.SIGTERM,signal.SIGINT):previous_signals[value]=signal.signal(value,interrupted)
        mx.set_default_device(mx.cpu if device=='cpu' else mx.gpu)
        for traced in (False,True):
            for schedule,chunks in SCHEDULES:
                model,handles=execute(traced);caches=model.make_cache();before=0
                if len(caches)!=2 or [type(cache).__name__ for cache in caches]!=['RotatingKVCache','KVCache']:
                    raise ValueError('Concrete source cache construction differs')
                for index,count in enumerate(chunks):
                    row=dict(schedule=schedule,trace=traced,step=index,before=before,count=count,completed=False,routes=[],arrays={});rows.append(row)
                    active.update(row=row,before=before,count=count,returned={},rope_calls={})
                    ids=mx.array([data['token_ids'][before:before+count]],dtype=mx.int32)
                    result=model(ids,cache=caches);observe('logits',result)
                    expected_names=set(stage_shapes()) if traced else {'logits'}
                    if set(row['arrays'])!=expected_names:raise ValueError('Complete source graph observation inventory differs')
                    if traced and len(row['routes'])!=4:raise ValueError('Complete source layer routing not observed')
                    if any(cache.offset!=before+count for cache in caches):raise ValueError('Source cache normal committed offset differs')
                    row['cache_offsets']=[cache.offset for cache in caches];row['cache_sizes']=[cache.size() for cache in caches]
                    if row['cache_sizes']!=[min(before+count,4),before+count]:raise ValueError('Source cache logical size differs')
                    for name,value in handles.items():
                        mx.eval(value)
                        if not np.array_equal(np.asarray(value),array('weights',name)):raise ValueError('Immutable learned weight changed during source graph call')
                    row['completed']=True;before+=count
                if before!=32:raise ValueError('Every short position must be exercised')
    except (Exception,KeyboardInterrupt) as error:
        primary=type(error).__name__+': '+str(error)
    finally:
        for value in previous_signals:signal.signal(value,signal.SIG_IGN)
        try:mx.set_default_device(previous_device)
        except (Exception,KeyboardInterrupt) as error:cleanup.append(type(error).__name__+': '+str(error))
        raw=bytes(blob)
        report=dict(format='toklet-gemma-text-graph-observation-v1',source_revision=REVISION,source_sha256=SOURCES,
            mathematical_reference_sha256=reference_sha,requested_device=device,admitted_runtime=admitted,
            observation_scope=SCOPE,exclusions=EXCLUSIONS,atol=ATOL,rtol=RTOL,
            binary=dict(path='observed.bin',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),alignment=8),
            rows=rows,comparisons=comparisons,primary_error=primary,cleanup_errors=cleanup,
            native_graph_implemented=False,checkpoint_qualified=False,device_qualified=False,upstream_long_rotary_gate_closed=False,
            passed=primary is None and not cleanup and len(rows)==2*sum(len(c) for _,c in SCHEDULES)
                and all(r['completed'] for r in rows) and all(c['passed'] for c in comparisons))
        try:
            with (destination/'observed.bin').open('xb') as stream:stream.write(raw)
            with (destination/'observation.json').open('xb') as stream:stream.write(encoded(report))
        finally:
            for value,handler in previous_signals.items():signal.signal(value,handler)
    return report['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('export','compare'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True);parser.add_argument('--runtime-manifest',type=Path,required=True)
    parser.add_argument('--expected-runtime-sha256',required=True);parser.add_argument('--reference',type=Path)
    parser.add_argument('--expected-reference-sha256');parser.add_argument('--device',choices=('cpu','gpu'))
    options=parser.parse_args()
    if options.action=='export':
        runtime.admit_runtime(options.runtime_manifest,options.expected_runtime_sha256,[Path(__file__)])
        with options.output.open('xb') as stream:stream.write(export())
    else:
        sources={name:runtime.captured(options.sources/name,128*1024,expected) for name,expected in SOURCES.items()}
        raise SystemExit(0 if compare(options.reference,options.expected_reference_sha256,sources,options.device,options.output,
            options.runtime_manifest,options.expected_runtime_sha256) else 1)
