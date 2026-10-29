"""Independent stdlib LFM2 scalar tiny graph, original packed owners and states.

Only generic byte/packing helpers are reused from the frozen Gemma3 producer;
no forward, norm, attention, rotary or target implementation is imported/called.
FP64 math reductions cross declared FP32 stores; this is not bit emulation.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct
import gemma3_cpu_graph as packing

FORMAT = 'toklet-lfm2-scalar-tiny-graph-v1'
PROTOCOL_SHA256 = 'e6cfd973db5392e8f1cd684230a26dce1af12508392df9405319b2b496e0ef83'
H, I, V, Q, KV, D, CAPACITY = 256, 256, 64, 4, 2, 64, 12
CPU_GATE = dict(atol=1e-5, rtol=1e-4)
LONG = (1,7,13,3,9,5,11,2,17,4,6,10)
CASES = (('single',(1,)),('short',(1,7,13)),('context12',LONG),('future_changed12',LONG[:-1]+(19,)))
MAX_ARTIFACT, MAX_ARRAYS, MAX_MANIFEST = 4*1024**2, 8*1024**2, 2*1024**2
# Concrete helper dependencies only. None of packing's graph/math operations are used.
bf16_bits, bf16_value = packing.bf16_bits, packing.bf16_value
encoded, identity, strict_json = packing.encoded, packing.identity, packing.strict_json


def f32(value):
    try:
        result=struct.unpack('<f',struct.pack('<f',value))[0]
    except (OverflowError,struct.error) as e:
        raise ValueError('FP32 store overflow') from e
    if not math.isfinite(result): raise ValueError('Nonfinite FP32 store')
    return result


def flat(value):
    return [v for part in value for v in flat(part)] if isinstance(value,(list,tuple)) else [value]


def finite(values):
    if any(not math.isfinite(v) for v in flat(values)): raise ValueError('Nonfinite operand')


def configuration():
    return dict(model_type='lfm2',fixture_only=True,hidden_size=H,intermediate_size=I,
        vocab_size=V,num_hidden_layers=3,num_attention_heads=Q,num_key_value_heads=KV,
        head_dim=D,full_attn_idxs=[1],conv_L_cache=3,conv_bias=False,
        max_position_embeddings=CAPACITY,rope_theta=1000000.0,norm_eps=1e-5,
        tie_word_embeddings=True,torch_dtype='bfloat16',quantization=dict(bits=4,group_size=64))


def shapes():
    result={'model.embed_tokens.weight':[V,H],'model.embedding_norm.weight':[H]}
    for layer in range(3):
        p=f'model.layers.{layer}'
        for name in ('operator_norm','ffn_norm'): result[p+'.'+name+'.weight']=[H]
        for name in ('w1','w3','w2'): result[p+'.feed_forward.'+name+'.weight']=[I,H]
        if layer==1:
            for name in ('q','k'): result[p+'.self_attn.'+name+'_layernorm.weight']=[D]
            for name,rows in [('q_proj',Q*D),('k_proj',KV*D),('v_proj',KV*D),('out_proj',H)]:
                result[p+'.self_attn.'+name+'.weight']=[rows,H]
        else:
            result[p+'.conv.in_proj.weight']=[3*H,H]
            result[p+'.conv.out_proj.weight']=[H,H]
            result[p+'.conv.conv.weight']=[H,3,1]
    return result


def make_artifact():
    tensors={}
    for name,shape in shapes().items():
        seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
        if len(shape)==2:
            codes,scales,biases=packing.packed_matrix(*shape,seed)
            base=name.removesuffix('.weight')
            tensors[name]=('U32',[shape[0],shape[1]//8],codes)
            for suffix,raw in [('scales',scales),('biases',biases)]:
                tensors[base+'.'+suffix]=('BF16',[shape[0],shape[1]//64],raw)
        else:
            if len(shape)==1:
                values=[1+((seed+7*j)%17-8)/32 for j in range(shape[0])]
                values[:3]=[0.0,-0.5,1.5] # direct gamma, distinguish +1 preparation
            else:
                values=[((seed+5*channel+11*tap)%19-9)/32 for channel in range(H) for tap in range(3)]
            tensors[name]=('BF16',shape,b''.join(struct.pack('<H',bf16_bits(x)) for x in values))
    header={'__metadata__':{'format':'mlx'}};payload=bytearray()
    for name,(dtype,shape,raw) in sorted(tensors.items()):
        start=len(payload);payload.extend(raw)
        header[name]=dict(dtype=dtype,shape=shape,data_offsets=[start,len(payload)])
    raw=bytearray(json.dumps(header,sort_keys=True,separators=(',',':')).encode())
    while (8+len(raw))%4!=2:raw.append(32)
    result=struct.pack('<Q',len(raw))+raw+payload
    if len(result)>MAX_ARTIFACT:raise ValueError('Artifact bound')
    return bytes(result)


def decode(raw):
    if not 8<len(raw)<=MAX_ARTIFACT:raise ValueError('Artifact length')
    n=struct.unpack('<Q',raw[:8])[0]
    if not 0<n<=65536 or 8+n>len(raw):raise ValueError('Header length')
    header=strict_json(raw[8:8+n]);expected={}
    for name,shape in shapes().items():
        if len(shape)==2:
            rows,cols=shape;expected[name]=('U32',[rows,cols//8])
            for suffix in ('scales','biases'):expected[name.removesuffix('.weight')+'.'+suffix]=('BF16',[rows,cols//64])
        else:expected[name]=('BF16',shape)
    if not isinstance(header,dict) or set(header)!=set(expected)|{'__metadata__'} or header['__metadata__']!={'format':'mlx'}:
        raise ValueError('Exact original tensor inventory')
    arrays={};cursor=0
    items=[]
    for name,(dtype,shape) in expected.items():
        spec=header[name]
        if not isinstance(spec,dict) or set(spec)!={'dtype','shape','data_offsets'} or spec['dtype']!=dtype or spec['shape']!=shape:
            raise ValueError('Original tensor descriptor')
        offsets=spec['data_offsets']
        if not isinstance(offsets,list) or len(offsets)!=2 or any(type(x)is not int for x in offsets+spec['shape']):raise ValueError('Integer extents')
        begin,end=offsets
        if begin<0 or end<begin or end-begin!=math.prod(shape)*(4 if dtype=='U32' else 2) or 8+n+end>len(raw):raise ValueError('Original tensor extent')
        items.append((begin,end,name))
    for begin,end,name in sorted(items):
        if begin!=cursor:raise ValueError('Original extent gap/overlap')
        arrays[name]=raw[8+n+begin:8+n+end];cursor=end
    if 8+n+cursor!=len(raw):raise ValueError('Trailing payload')
    matrices,norms,convs={},{},{}
    for name,shape in shapes().items():
        if len(shape)!=2:
            values=[bf16_value(x[0]) for x in struct.iter_unpack('<H',arrays[name])]
            if len(shape)==1:norms[name]=values # direct; no +1, no narrowing
            else:convs[name]=[values[i:i+3] for i in range(0,len(values),3)]
            continue
        rows,cols=shape;base=name.removesuffix('.weight')
        words=[x[0] for x in struct.iter_unpack('<I',arrays[name])]
        scales=[bf16_value(x[0]) for x in struct.iter_unpack('<H',arrays[base+'.scales'])]
        biases=[bf16_value(x[0]) for x in struct.iter_unpack('<H',arrays[base+'.biases'])]
        result=[]
        for row in range(rows):
            values=[]
            for col in range(cols):
                q=(words[row*(cols//8)+col//8]>>(4*(col%8)))&15
                group=row*(cols//64)+col//64
                # These bounded BF16×Q4 products are exact finite FP32.
                values.append(f32(f32(q*scales[group])+biases[group]))
            result.append(values)
        matrices[name]=result
    return matrices,norms,convs,8+n


def linear(x,weights):
    if not x or not weights or any(len(row)!=len(weights[0]) for row in x+weights):raise ValueError('Linear shape')
    finite(x);finite(weights)
    return [[f32(math.fsum(a*b for a,b in zip(row,w,strict=True))) for w in weights] for row in x]


def norm(x,gamma):
    if not x or not gamma or any(len(row)!=len(gamma) for row in x):raise ValueError('Norm shape')
    finite(x);finite(gamma);out=[]
    for row in x:
        denominator=math.sqrt(math.fsum(v*v for v in row)/len(row)+f32(1e-5))
        out.append([f32(v/denominator*g) for v,g in zip(row,gamma,strict=True)])
    return out


def head_norm(x,gamma,heads):
    d=len(gamma)
    if heads<=0 or any(len(row)!=heads*d for row in x):raise ValueError('Head norm shape')
    return [[v for h in range(heads) for v in norm([row[h*d:(h+1)*d]],gamma)[0]] for row in x]


def rotary(x,heads,dim,offset=0):
    if not x or heads<=0 or dim<=0 or dim%2 or offset<0 or offset+len(x)>CAPACITY or any(len(row)!=heads*dim for row in x):raise ValueError('Rotary geometry')
    finite(x);result=[row[:] for row in x]
    with localcontext() as ctx:
        ctx.prec=80
        inverses=[float((Decimal(1000000).ln()*Decimal(-2*j)/dim).exp()) for j in range(dim//2)]
    for token,row in enumerate(x):
        for j,frequency in enumerate(inverses):
            angle=(offset+token)*frequency;co=f32(math.cos(angle));si=f32(math.sin(angle))
            for head in range(heads):
                a=head*dim+j;b=a+dim//2
                result[token][a]=f32(f32(row[a]*co)-f32(row[b]*si))
                result[token][b]=f32(f32(row[b]*co)+f32(row[a]*si))
    return result


def silu(value):
    value=f32(value);exponential=f32(math.exp(-abs(value)));denominator=f32(1+exponential)
    sigmoid=f32(1/denominator) if value>=0 else f32(exponential/denominator)
    return f32(value*sigmoid)


def multiply(a,b):
    if len(a)!=len(b) or any(len(x)!=len(y) for x,y in zip(a,b)):raise ValueError('Product shape')
    return [[f32(x*y) for x,y in zip(row,other,strict=True)] for row,other in zip(a,b,strict=True)]


def residual(a,b):
    if len(a)!=len(b) or any(len(x)!=len(y) for x,y in zip(a,b)):raise ValueError('Residual shape')
    return [[f32(x+y) for x,y in zip(row,other,strict=True)] for row,other in zip(a,b,strict=True)]


def convolution(u,coefficients):
    if not u or not coefficients or any(len(row)!=len(coefficients) for row in u) or any(len(taps)!=3 for taps in coefficients):raise ValueError('Convolution shape')
    finite(u);finite(coefficients);channels=len(coefficients);zero=[0.0]*channels
    joined=[zero[:],zero[:]]+[row[:] for row in u]
    before=[[row[:] for row in joined[t:t+2]] for t in range(len(u))]
    after=[[row[:] for row in joined[t+1:t+3]] for t in range(len(u))]
    out=[[f32(math.fsum(joined[t+tap][c]*coefficients[c][tap] for tap in range(3))) for c in range(channels)] for t in range(len(u))]
    return out,before,after


def attention(q,k,v,heads=Q,kv_heads=KV,dim=D):
    n=len(q)
    if not n or len(k)!=n or len(v)!=n or heads<=0 or kv_heads<=0 or heads%kv_heads or dim<=0 or any(len(row)!=heads*dim for row in q) or any(len(row)!=kv_heads*dim for row in k+v):raise ValueError('Attention geometry')
    finite(q);finite(k);finite(v);scale=f32(1/math.sqrt(dim));scores=[];probs=[];context=[[0.0]*(heads*dim) for _ in q]
    for h in range(heads):
        hs,ks=h*dim,(h//(heads//kv_heads))*dim;hscores=[];hprobs=[]
        for t,row in enumerate(q):
            sr=[f32(math.fsum(row[hs+d]*key[ks+d] for d in range(dim))*scale) for key in k]
            maximum=max(sr[:t+1]);ex=[math.exp(x-maximum) for x in sr[:t+1]];mass=math.fsum(ex)
            pr=[f32(x/mass) for x in ex]+[0.0]*(n-t-1)
            for d in range(dim):context[t][hs+d]=f32(math.fsum(pr[j]*v[j][ks+d] for j in range(t+1)))
            hscores.append(sr);hprobs.append(pr)
        scores.append(hscores);probs.append(hprobs)
    return scores,probs,context


def kv_snapshots(rows):
    return [[[[rows[k][head*D+d] if k<=t else 0.0 for d in range(D)] for k in range(CAPACITY)] for head in range(KV)] for t in range(len(rows))]


def forward(matrices,norms,convs,tokens):
    if not tokens or len(tokens)>CAPACITY or any(type(t)is not int or not 0<=t<V for t in tokens):raise ValueError('Tiny token request')
    captures={}
    def save(name,values):
        if name in captures:raise ValueError('Duplicate stage')
        values_flat=flat(values);finite(values_flat);captures[name]=values_flat;return values
    table=matrices['model.embed_tokens.weight'];x=save('embedding_lookup',[table[t][:] for t in tokens])
    for layer in range(3):
        p=f'model.layers.{layer}';s=f'layers.{layer}.'
        normalized=save(s+'operator_norm',norm(x,norms[p+'.operator_norm.weight']))
        if layer!=1:
            bcx=save(s+'bcx_projection',linear(normalized,matrices[p+'.conv.in_proj.weight']))
            b=save(s+'b',[row[:H] for row in bcx]);c=save(s+'c',[row[H:2*H] for row in bcx]);xx=save(s+'x',[row[2*H:] for row in bcx])
            u=save(s+'u',multiply(b,xx));z,before,after=convolution(u,convs[p+'.conv.conv.weight'])
            save(s+'history_before',before);save(s+'history_after',after)
            z=save(s+'convolution',z);gated=save(s+'gated_conv',multiply(c,z))
            projected=save(s+'operator_projection',linear(gated,matrices[p+'.conv.out_proj.weight']))
        else:
            q=save(s+'q_projection',linear(normalized,matrices[p+'.self_attn.q_proj.weight']))
            k=save(s+'k_projection',linear(normalized,matrices[p+'.self_attn.k_proj.weight']))
            v=save(s+'v_projection',linear(normalized,matrices[p+'.self_attn.v_proj.weight']))
            q=save(s+'q_norm',head_norm(q,norms[p+'.self_attn.q_layernorm.weight'],Q))
            k=save(s+'k_norm',head_norm(k,norms[p+'.self_attn.k_layernorm.weight'],KV))
            q=save(s+'q_rotary',rotary(q,Q,D));k=save(s+'k_rotary',rotary(k,KV,D))
            scores,probabilities,context=attention(q,k,v)
            save(s+'attention_scores',scores);save(s+'attention_probabilities',probabilities)
            save(s+'kv_keys_after',kv_snapshots(k));save(s+'kv_values_after',kv_snapshots(v))
            context=save(s+'attention_context',context)
            projected=save(s+'operator_projection',linear(context,matrices[p+'.self_attn.out_proj.weight']))
        x=save(s+'operator_residual',residual(x,projected))
        normalized=save(s+'ffn_norm',norm(x,norms[p+'.ffn_norm.weight']))
        gate=save(s+'gate_projection',linear(normalized,matrices[p+'.feed_forward.w1.weight']))
        up=save(s+'up_projection',linear(normalized,matrices[p+'.feed_forward.w3.weight']))
        active=save(s+'gate_silu',[[silu(v) for v in row] for row in gate])
        gated=save(s+'ffn_gated',multiply(active,up));down=save(s+'down_projection',linear(gated,matrices[p+'.feed_forward.w2.weight']))
        x=save(s+'output',residual(x,down))
    final=save('final_norm',norm(x,norms['model.embedding_norm.weight']))
    save('logits',linear(final,table))
    return captures


def contracts(n):
    result={'embedding_lookup':[n,H],'final_norm':[n,H],'logits':[n,V]}
    common={name:[n,H] for name in ('operator_norm','operator_projection','operator_residual','ffn_norm','gate_projection','up_projection','gate_silu','ffn_gated','down_projection','output')}
    for layer in range(3):
        spec=dict(common)
        if layer!=1:
            spec.update(bcx_projection=[n,3*H],**{name:[n,H] for name in ('b','c','x','u','convolution','gated_conv')},history_before=[n,2,H],history_after=[n,2,H])
        else:
            spec.update(q_projection=[n,Q,D],k_projection=[n,KV,D],v_projection=[n,KV,D],q_norm=[n,Q,D],k_norm=[n,KV,D],q_rotary=[n,Q,D],k_rotary=[n,KV,D],attention_scores=[Q,n,n],attention_probabilities=[Q,n,n],attention_context=[n,Q*D],kv_keys_after=[n,KV,CAPACITY,D],kv_values_after=[n,KV,CAPACITY,D])
        result.update({f'layers.{layer}.{name}':shape for name,shape in spec.items()})
    if len(result)!=63:raise ValueError('Frozen stage inventory')
    return result


def causal_equal(first,changed,n):
    expected=contracts(n)
    if set(first)!=set(expected) or set(changed)!=set(expected):raise ValueError('Causal inventory')
    for name,shape in expected.items():
        a,b=first[name],changed[name]
        if name.endswith('attention_scores'):
            for h in range(Q):
                for query in range(n-1):
                    start=(h*n+query)*n
                    if a[start:start+query+1]!=b[start:start+query+1]:raise ValueError('Causal visible scores')
        elif name.endswith('attention_probabilities'):
            for h in range(Q):
                if a[h*n*n:h*n*n+(n-1)*n]!=b[h*n*n:h*n*n+(n-1)*n]:raise ValueError('Causal probabilities')
        else:
            count=(n-1)*math.prod(shape[1:])
            if a[:count]!=b[:count]:raise ValueError('Causal earlier stage '+name)


def components():
    u=[[2.,-3.],[6.,-2.],[12.,4.],[20.,-4.],[30.,-6.]];w=[[1.,10.,100.],[3.,-2.,5.]]
    z,before,after=convolution(u,w)
    scores,probabilities,context=attention([[0.]*4],[[0.]*2],[[2.,7.]],4,2,1)
    return dict(conv=dict(u=u,coefficients=w,z=z,before=before,after=after),
        direct_gamma=dict(input=[[1.,2.,3.]],raw_bf16_bits=[0x0000,0xbf00,0x3fc0],output=norm([[1.,2.,3.]],[0.,-.5,1.5])),
        gqa_one_key=dict(scores=scores,probabilities=probabilities,context=context),
        rotary_zero=rotary([[1.,2.,3.,4.]],1,4),
        no_mask_support=True,no_half_equivalence=True)


def export(directory):
    directory=Path(directory);directory.mkdir(parents=False,exist_ok=False)
    dependencies={str(Path(__file__).resolve()):identity(Path(__file__).read_bytes()),str(Path(packing.__file__).resolve()):identity(Path(packing.__file__).read_bytes())}
    try:
        artifact=make_artifact();matrices,norms,convs,start=decode(artifact);files={}
        for name,raw in [('config.json',encoded(configuration())),('model.safetensors',artifact),('components.json',encoded(components()))]:
            (directory/name).write_bytes(raw);files[name]=identity(raw)
        rows=[];previous=None
        with (directory/'arrays.bin').open('xb') as binary,(directory/'progress.jsonl').open('x') as progress:
            for name,tokens in CASES:
                progress.write(json.dumps(dict(case=name,started=True,offset=binary.tell()))+'\n');progress.flush()
                values=forward(matrices,norms,convs,tokens);expected=contracts(len(tokens));descriptors={}
                if set(values)!=set(expected):raise ValueError('Graph stage inventory')
                if values['layers.0.history_after']==values['layers.2.history_after']:raise ValueError('Degenerate convolution-owner fixture')
                kv=values['layers.1.k_projection'];vv=values['layers.1.v_projection']
                if all(kv[t*128:t*128+64]==kv[t*128+64:(t+1)*128] for t in range(len(tokens))) or all(vv[t*128:t*128+64]==vv[t*128+64:(t+1)*128] for t in range(len(tokens))):raise ValueError('Degenerate KV-head fixture')
                if name=='future_changed12':causal_equal(previous,values,len(tokens))
                if name=='context12':previous=values
                for stage,shape in expected.items():
                    numbers=values[stage]
                    if len(numbers)!=math.prod(shape):raise ValueError('Graph stage shape')
                    raw=b''.join(struct.pack('<f',v) for v in numbers);begin=binary.tell()
                    if begin+len(raw)>MAX_ARRAYS:raise ValueError('Array bound')
                    binary.write(raw)
                    descriptors[stage]=dict(file='arrays.bin',offset=begin,dtype='f32le',shape=shape,**identity(raw))
                binary.flush();rows.append(dict(name=name,tokens=list(tokens),arrays=descriptors))
                progress.write(json.dumps(dict(case=name,complete=True,stage_groups=len(descriptors),offset=binary.tell()))+'\n');progress.flush()
        for name in ('arrays.bin','progress.jsonl'):files[name]=identity((directory/name).read_bytes())
        for path,pin in dependencies.items():
            if identity(Path(path).read_bytes())!=pin:raise ValueError('Producer dependency changed')
        manifest=dict(format=FORMAT,protocol_sha256=PROTOCOL_SHA256,gate=CPU_GATE,configuration=configuration(),
            precision='independent FP64 reductions with declared FP32 operation stores; scalar CPU only',
            helper_scope=['packed_matrix','bf16_bits','bf16_value','encoded','identity','strict_json'],dependencies=dependencies,
            cases=rows,files=files,physical_tensors=65,logical_owners=29,affine_owners=18,direct_norm_owners=9,depthwise3_owners=2,
            data_start=start,unaligned_u32_payload=(start%4==2),stage_groups=252,mathematical_groups=228,state_groups=24,
            original_tied_table_owners=1,causal_prefix_exact=True,source_graph_executed=True,target_executed=False,
            target_state_comparison='active KV prefix only; source inactive positive-zero is not a target tail-clearing requirement',
            gpu_executed=False,real_model_executed=False,original_quality_qualified=False)
        raw=encoded(manifest)
        if len(raw)>MAX_MANIFEST:raise ValueError('Manifest bound')
        (directory/'manifest.json').write_bytes(raw)
    except BaseException as error:
        (directory/'failure.json').write_bytes(encoded(dict(passed=False,error=repr(error),target_executed=False)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args().output)
