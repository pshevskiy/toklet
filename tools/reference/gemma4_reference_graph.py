"""Full-sequence reference candidate changing only the attention dot reduction.

Historical scalar batch sources remain frozen. Norms/GELU/rotary and graph order
are explicit arithmetic copies, checked against complete frozen traces. E2B is
limited to the saved four-token comparison until larger composition qualifies.
"""
import math
from decimal import Decimal, localcontext
import affine4_tiles as tiles
import gemma4_text_math as frozen
import gemma_reference_attention as dot

WITNESS, E2B = frozen.WITNESS, frozen.E2B
MAX_WITNESS_POSITIONS = 1025
MAX_E2B_ROWS = 4
softcap = frozen.softcap
PROFILE = dot.PROFILE


class BatchedWeights:
    """Keep existing file-row projection/gather bounds independently of N."""
    def __init__(self, weights):
        self.inner = weights
        self.profile = weights.profile
        self.vectors = weights.vectors

    def gather(self, name, ids):
        import numpy as np
        blocks = [self.inner.gather(name, ids[start:start+tiles.MAX_TOKENS])
                  for start in range(0, len(ids), tiles.MAX_TOKENS)]
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)

    def linear(self, name, x, dtype):
        import numpy as np
        shape = x.shape; source = x.reshape(-1, shape[-1])
        blocks = [self.inner.linear(name, source[start:start+tiles.MAX_TOKENS], dtype)
                  for start in range(0, len(source), tiles.MAX_TOKENS)]
        output = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)
        return output.reshape(*shape[:-1], self.inner.matrices[name].rows)


def evaluate(weights,ids,staged,*,sink=None):
    """Batch-one reference prefill; WITNESS<=1025, E2B<=4, no tokenizer or KV API.

    The ideal path has FP64 activations/reductions over reconstructed FP32
    weights. The staged path rounds each explicit operator boundary to FP32,
    with FP64 reductions and explicitly rounded rotary products. Decimal80
    defines rotary angles independently of the target implementation.
    """
    import numpy as np
    if type(weights) is not frozen.Weights or weights.profile not in (WITNESS, E2B):
        raise ValueError('Named reference graph weights required')
    p=weights.profile
    limit=MAX_WITNESS_POSITIONS if p==WITNESS else MAX_E2B_ROWS
    if type(staged) is not bool or type(ids) is not list or not 1<=len(ids)<=limit or any(type(t) is not int or not 0<=t<p.vocab for t in ids):
        raise ValueError('Bounded explicit token IDs and precision contract required')
    if sink is not None and not callable(sink):raise ValueError('Callable graph sink required')
    if (len(ids)>32 or p==E2B)and sink is None:raise ValueError('Bounded reference graph requires a streaming sink')
    weights=BatchedWeights(weights)
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
        return dot.attention(q,k,v,staged=staged,window=p.window if local else None)
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
