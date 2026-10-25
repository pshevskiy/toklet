"""Independent short token-to-logit math over bounded, FP32-reconstructed weights.

Two explicit profiles exist: the frozen H64 packed witness for offline checks,
and the inspected real E2B graph. Neither grants engine or source-BF16 support.
Weights stay outside the graph; projections and gathers use bounded file rows.
"""
from dataclasses import dataclass
from decimal import Decimal, localcontext
import math

import affine4_tiles as tiles


@dataclass(frozen=True)
class Profile:
    hidden: int
    vocab: int
    layers: int
    producers: int
    heads: int
    ple: int
    local_dim: int
    full_dim: int
    full_rotated: int
    intermediate: int
    window: int
    pattern: int

    def full(self, layer):return layer % self.pattern == self.pattern-1
    def dimension(self, layer):return self.full_dim if self.full(layer) else self.local_dim


WITNESS = Profile(64,17,4,2,2,64,64,128,32,64,4,2)
E2B = Profile(1536,262144,35,15,8,256,256,512,128,6144,512,5)
NORMS = ('input_layernorm','post_attention_layernorm','pre_feedforward_layernorm',
         'post_feedforward_layernorm','post_per_layer_input_norm')


def shapes(profile):
    if profile not in (WITNESS,E2B):raise ValueError('Named graph profile required')
    p=profile;h=p.hidden
    out={'embed_tokens.weight':(p.vocab,h),'embed_tokens_per_layer.weight':(p.vocab,p.layers*p.ple),
         'per_layer_model_projection.weight':(p.layers*p.ple,h),
         'per_layer_projection_norm.weight':(p.ple,),'norm.weight':(h,)}
    for i in range(p.layers):
        base=f'layers.{i}.';d=p.dimension(i);ff=p.intermediate*(1 if i<p.producers else 2)
        for name in NORMS:out[base+name+'.weight']=(h,)
        out[base+'layer_scalar']=(1,);out[base+'self_attn.q_norm.weight']=(d,)
        for key,shape in [('self_attn.q_proj',(p.heads*d,h)),('self_attn.o_proj',(h,p.heads*d)),
                          ('mlp.gate_proj',(ff,h)),('mlp.up_proj',(ff,h)),('mlp.down_proj',(h,ff)),
                          ('per_layer_input_gate',(p.ple,h)),('per_layer_projection',(h,p.ple))]:
            out[base+key+'.weight']=shape
        if i<p.producers:
            for key in ('k_proj','v_proj'):out[base+'self_attn.'+key+'.weight']=(d,h)
            out[base+'self_attn.k_norm.weight']=(d,)
    return out


def stage_shapes(profile,tokens):
    shapes(profile) # Named profile admission, independent of execution output.
    tiles.integer(tokens,1,32);p=profile;n=tokens
    out={name:(1,n,p.hidden) for name in ('embedding','scaled_embedding','final_norm')}
    out.update({name:(1,n,p.layers*p.ple) for name in ('ple_lookup','ple_projection')})
    out['ple_norm']=(1,n,p.layers,p.ple)
    out.update({name:(1,n,p.vocab) for name in ('head','logits')})
    for i in range(p.layers):
        base=f'layer.{i}.';d=p.dimension(i);ff=p.intermediate*(1 if i<p.producers else 2)
        out[base+'ple_input']=(1,n,p.ple)
        for name in ('input','output',*NORMS,'self_attn.o_proj','mlp.down_proj','per_layer_projection'):
            out[base+name]=(1,n,p.hidden)
        out[base+'self_attn.q_proj']=(1,n,p.heads*d);out[base+'self_attn.q_norm']=(1,n,p.heads,d)
        for name in ('q_rotary','context'):out[base+name]=(1,p.heads,n,d)
        for name in ('mlp.gate_proj','mlp.up_proj'):out[base+name]=(1,n,ff)
        out[base+'per_layer_input_gate']=(1,n,p.ple)
        if i<p.producers:
            for name in ('self_attn.k_proj','self_attn.v_proj'):out[base+name]=(1,n,d)
            for name in ('self_attn.k_norm','self_attn.v_norm'):out[base+name]=(1,n,1,d)
            out[base+'k_rotary']=(1,1,n,d)
    return out


def softcap(values,staged):
    """Bound Python/libm temporaries to1024 values, retaining one output tensor."""
    import numpy as np
    dtype=np.float32 if staged else np.float64
    output=np.empty(values.shape,dtype=dtype);source=values.reshape(-1);target=output.reshape(-1)
    for start in range(0,len(source),1024):
        end=min(start+1024,len(source))
        divided=(source[start:end].astype(np.float64)/30.).astype(dtype)
        capped=np.array([math.tanh(float(v)) for v in divided],dtype=dtype)
        target[start:end]=(capped.astype(np.float64)*30.).astype(dtype)
    return output


class Weights:
    """Resolved file matrices and small direct FP32 vectors, separate from math."""
    def __init__(self,profile,matrices,vectors):
        import numpy as np
        expected=shapes(profile)
        if set(matrices)!={n for n,s in expected.items() if len(s)==2} or set(vectors)!={n for n,s in expected.items() if len(s)==1}:
            raise ValueError('Exact graph tensor inventory required')
        for name,matrix in matrices.items():
            if (matrix.rows,matrix.columns)!=expected[name]:raise ValueError('Graph matrix shape differs')
        for name,vector in vectors.items():
            if not isinstance(vector,np.ndarray) or vector.dtype!=np.dtype('float32') or vector.shape!=expected[name] or not np.isfinite(vector).all():
                raise ValueError('Finite direct FP32 graph vector required')
        self.profile=profile;self.matrices=matrices;self.vectors=vectors
    def gather(self,name,ids):return tiles.gather_rows(self.matrices[name],ids)
    def linear(self,name,x,dtype):
        shape=x.shape;out=tiles.project_rows(self.matrices[name],x.reshape(-1,shape[-1]),dtype)
        return out.reshape(*shape[:-1],self.matrices[name].rows)


def evaluate(weights,ids,staged,*,sink=None):
    """Batch-one causal prefill at positions0..N-1, N<=32; no tokenizer or KV API.

    The ideal path has FP64 activations/reductions over reconstructed FP32
    weights. The staged path rounds each explicit operator boundary to FP32,
    with FP64 reductions and explicitly rounded rotary products. Decimal80
    defines rotary angles independently of the target implementation.
    """
    import numpy as np
    p=weights.profile
    if type(staged) is not bool or type(ids) is not list or not 1<=len(ids)<=32 or any(type(t) is not int or not 0<=t<p.vocab for t in ids):
        raise ValueError('Bounded explicit token IDs and precision contract required')
    if sink is not None and not callable(sink):raise ValueError('Callable graph sink required')
    if p==E2B and sink is None:raise ValueError('E2B graph requires a streaming sink')
    n=len(ids);dtype=np.float32 if staged else np.float64;states={}
    def rounded(x):return np.asarray(x,dtype=dtype)
    def scalar(x):return float(np.float32(x)) if staged else float(x)
    def save(name,x):
        value=rounded(x)
        if not np.isfinite(value).all():raise ValueError('Nonfinite graph state: '+name)
        if sink is None:states[name]=value
        else:sink(name,value)
        return value
    def linear(x,name):return weights.linear(name,x,dtype)
    def norm(x,name=None):
        flat=x.astype(np.float64).reshape(-1,x.shape[-1]);scale=np.ones(x.shape[-1]) if name is None else weights.vectors[name].astype(np.float64)
        epsilon=float(np.float32(1e-6))
        out=[]
        for row in flat:
            denominator=math.sqrt(math.fsum(float(y)*float(y) for y in row)/len(row)+epsilon)
            out.append([float(v)/denominator*float(s) for v,s in zip(row,scale)])
        return rounded(np.array(out).reshape(x.shape))
    def gelu(x):
        return rounded(np.array([.5*float(v)*(1+math.tanh(math.sqrt(2/math.pi)*(float(v)+.044715*float(v)**3))) for v in x.flat]).reshape(x.shape))
    def rotate(x,layer):
        out=x.astype(np.float64).copy();d=p.dimension(layer)
        rotated=p.full_rotated if p.full(layer) else d;base=1e6 if p.full(layer) else 1e4
        with localcontext() as context:
            context.prec=80
            for pair in range(rotated//2):
                frequency=Decimal.from_float(base)**(Decimal(2*pair)/Decimal(d))
                for position in range(n):
                    angle=float(Decimal(position)/frequency);cosine=scalar(math.cos(angle));sine=scalar(math.sin(angle))
                    a=x[0,:,position,pair].astype(np.float64);b=x[0,:,position,pair+d//2].astype(np.float64)
                    if staged:
                        out[0,:,position,pair]=rounded(a*cosine).astype(np.float64)-rounded(b*sine).astype(np.float64)
                        out[0,:,position,pair+d//2]=rounded(a*sine).astype(np.float64)+rounded(b*cosine).astype(np.float64)
                    else:
                        out[0,:,position,pair]=a*cosine-b*sine;out[0,:,position,pair+d//2]=a*sine+b*cosine
        return rounded(out)
    def attention(q,k,v,local):
        out=np.empty(q.shape,dtype=dtype)
        for head in range(p.heads):
            for position in range(n):
                start=max(0,position-p.window+1) if local else 0
                scores=rounded([math.fsum(float(a)*float(b) for a,b in zip(q[0,head,position],key)) for key in k[0,0,start:position+1]])
                exponent=[math.exp(float(value)-float(max(scores))) for value in scores];total=math.fsum(exponent)
                probability=rounded([value/total for value in exponent])
                out[0,head,position]=rounded([math.fsum(float(probability[t])*float(v[0,0,start+t,d]) for t in range(len(probability))) for d in range(q.shape[-1])])
        return out
    x=save('embedding',weights.gather('embed_tokens.weight',ids)[None])
    x=save('scaled_embedding',x.astype(np.float64)*scalar(math.sqrt(p.hidden)))
    lookup=save('ple_lookup',weights.gather('embed_tokens_per_layer.weight',ids)[None])
    lookup=rounded(lookup.astype(np.float64)*scalar(math.sqrt(p.ple))).reshape(1,n,p.layers,p.ple)
    projection=save('ple_projection',linear(x,'per_layer_model_projection.weight'))
    projection=rounded(projection.astype(np.float64)*scalar(p.hidden**-.5)).reshape(1,n,p.layers,p.ple)
    projection=save('ple_norm',norm(projection,'per_layer_projection_norm.weight'))
    ple=rounded(rounded(projection.astype(np.float64)+lookup.astype(np.float64)).astype(np.float64)*scalar(2**-.5))
    shared={}
    for i in range(p.layers):
        name=f'layer.{i}.';base=f'layers.{i}.';d=p.dimension(i)
        save(name+'input',x);save(name+'ple_input',ple[:,:,i,:])
        y=save(name+'input_layernorm',norm(x,base+'input_layernorm.weight'))
        q=save(name+'self_attn.q_proj',linear(y,base+'self_attn.q_proj.weight'))
        q=save(name+'self_attn.q_norm',norm(q.reshape(1,n,p.heads,d),base+'self_attn.q_norm.weight'))
        q=save(name+'q_rotary',rotate(q.transpose(0,2,1,3),i))
        if i<p.producers:
            k=save(name+'self_attn.k_proj',linear(y,base+'self_attn.k_proj.weight'))
            v=save(name+'self_attn.v_proj',linear(y,base+'self_attn.v_proj.weight'))
            k=save(name+'self_attn.k_norm',norm(k.reshape(1,n,1,d),base+'self_attn.k_norm.weight'))
            v=save(name+'self_attn.v_norm',norm(v.reshape(1,n,1,d)))
            k=save(name+'k_rotary',rotate(k.transpose(0,2,1,3),i));shared[p.full(i)]=(k,v.transpose(0,2,1,3))
        k,v=shared[p.full(i)]
        y=save(name+'context',attention(q,k,v,not p.full(i)))
        y=save(name+'self_attn.o_proj',linear(y.transpose(0,2,1,3).reshape(1,n,p.heads*d),base+'self_attn.o_proj.weight'))
        y=save(name+'post_attention_layernorm',norm(y,base+'post_attention_layernorm.weight'))
        x=rounded(x.astype(np.float64)+y.astype(np.float64));residual=x
        y=save(name+'pre_feedforward_layernorm',norm(x,base+'pre_feedforward_layernorm.weight'))
        gate=save(name+'mlp.gate_proj',linear(y,base+'mlp.gate_proj.weight'));up=save(name+'mlp.up_proj',linear(y,base+'mlp.up_proj.weight'))
        y=rounded(gelu(gate).astype(np.float64)*up.astype(np.float64))
        y=save(name+'mlp.down_proj',linear(y,base+'mlp.down_proj.weight'))
        y=save(name+'post_feedforward_layernorm',norm(y,base+'post_feedforward_layernorm.weight'))
        x=rounded(residual.astype(np.float64)+y.astype(np.float64));residual=x
        y=save(name+'per_layer_input_gate',linear(x,base+'per_layer_input_gate.weight'))
        y=rounded(gelu(y).astype(np.float64)*ple[:,:,i,:].astype(np.float64))
        y=save(name+'per_layer_projection',linear(y,base+'per_layer_projection.weight'))
        y=save(name+'post_per_layer_input_norm',norm(y,base+'post_per_layer_input_norm.weight'))
        x=rounded(residual.astype(np.float64)+y.astype(np.float64))
        x=save(name+'output',x.astype(np.float64)*weights.vectors[base+'layer_scalar'].astype(np.float64))
    x=save('final_norm',norm(x,'norm.weight'))
    x=save('head',linear(x,'embed_tokens.weight'))
    save('logits',softcap(x,staged))
    return states
