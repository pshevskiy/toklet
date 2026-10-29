"""Independent bounded NumPy NativeHalf Gemma3 graph; no native/MLX imports.

FP32/FP64 BLAS reductions are named source profiles, not scalar-order or GPU
compensation emulation. Every named activation is stored Half; scores,
probabilities and final logits are stored FP32 in both profiles.
"""
from decimal import Decimal, localcontext
import math
import numpy as np


def finite(x):
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite numerical operand/store')
    return x


def half(x):
    with np.errstate(over='ignore', invalid='ignore'):
        return finite(np.asarray(x).astype(np.float16))


def residual(x, branch):
    """MLX Gemma3 FP16 residual policy, at both post-normalized branch sites.

    Invalid active operands remain errors: clamp is only defined for finite Half
    inputs. Sum in FP32, saturate to finite Half bounds, then narrow. WIDE uses
    this same explicitly FP32 operation; no other operation is saturated.
    """
    if x.dtype != np.float16 or branch.dtype != np.float16 or x.shape != branch.shape:
        raise ValueError('Clipped residual Half storage/shape')
    finite(x);finite(branch)
    total = x.astype(np.float32) + branch.astype(np.float32)
    return half(np.clip(total, np.float32(-65504), np.float32(65504)))


def dtype(profile):
    if profile not in ('main', 'wide'):
        raise ValueError('Unknown arithmetic profile')
    return np.float32 if profile == 'main' else np.float64


def bf16(words):
    if words.dtype != np.dtype('<u2'):
        raise ValueError('BF16 word storage required')
    return finite((words.astype(np.uint32) << 16).view(np.float32))


def decode(code, scale_words, bias_words, columns):
    if (code.dtype != np.dtype('<u4') or code.ndim != 2 or columns <= 0
            or columns % 64 or code.shape[1] != columns // 8
            or scale_words.shape != (len(code), columns // 64)
            or bias_words.shape != scale_words.shape):
        raise ValueError('Original Q4/group64 descriptor mismatch')
    scales, biases = bf16(scale_words), bf16(bias_words)
    codes = np.stack([(code >> shift) & 15 for shift in range(0, 32, 4)], axis=-1).reshape(len(code), columns)
    # BF16 significand times a four-bit integer fits FP32 precision. FP64
    # expression followed by FP32 gives the canonical reconstruction, even when
    # separate FP32 multiply would overflow before cancellation by the bias.
    result = (codes.astype(np.float64) * np.repeat(scales.astype(np.float64), 64, axis=1)
              + np.repeat(biases.astype(np.float64), 64, axis=1)).astype(np.float32)
    return finite(result)


def projection(x, matrix, profile, output_float=False):
    p = dtype(profile)
    if x.dtype != np.float16 or matrix.dtype != np.float32 or x.ndim != 2 or matrix.ndim != 2 or x.shape[1] != matrix.shape[1]:
        raise ValueError('Projection storage/shape')
    # Only this operation owns the widened matrix. Returned output never aliases it.
    converted = matrix.astype(p, copy=False)
    try:
        out = x.astype(p) @ converted.T
        return finite(out.astype(np.float32)) if output_float else half(out)
    finally:
        del converted


def norm(x, scale, profile):
    p = dtype(profile)
    if x.dtype != np.float16 or scale.dtype != np.float32 or x.shape[-1] != len(scale):
        raise ValueError('RMS storage/width')
    original = x.shape
    rows = finite(x).reshape(-1, len(scale)).astype(p)
    finite(scale)
    if profile == 'main':
        total = np.zeros(len(rows), dtype=np.float32)
        correction = np.zeros_like(total)
        for j in range(len(scale)):
            value = rows[:, j] * rows[:, j]
            following = total + value
            error = np.where(np.abs(total) >= np.abs(value), (total-following)+value, (value-following)+total)
            correction = correction + error
            total = following
        square_sum = total + correction
    else:
        square_sum = np.sum(rows*rows, axis=1, dtype=np.float64)
    mean = square_sum / p(len(scale))
    root = np.sqrt(mean + p(np.float32(1e-6)))
    inverse = p(1) / root
    return half((rows * inverse[:, None]) * scale.astype(p)[None, :]).reshape(original)


def rotary(x, theta, profile):
    p = dtype(profile)
    if x.dtype != np.float16 or x.ndim != 3 or x.shape[1] not in (1,4) or x.shape[2] not in (32,256) or not 1 <= len(x) <= 544 or theta not in (10000,1000000):
        raise ValueError('Bounded ordinary RoPE shape')
    d=x.shape[2];n=len(x);split=d//2
    if profile == 'main':
        exponent=(np.float32(-2)*np.arange(split,dtype=np.float32))/np.float32(d)
        inverse=np.power(np.float32(theta),exponent).astype(np.float32)
        angles=np.arange(n,dtype=np.float32)[:,None]*inverse[None,:]
    else:
        with localcontext() as ctx:
            ctx.prec=80
            inverse=[(Decimal(theta).ln()*Decimal(-2*j)/Decimal(d)).exp() for j in range(split)]
            angles=np.array([[float(Decimal(position)*f) for f in inverse] for position in range(n)],dtype=np.float64)
    co=np.cos(angles)[:,None,:];si=np.sin(angles)[:,None,:]
    first=x[...,:split].astype(p);second=x[...,split:].astype(p)
    return half(np.concatenate((first*co-second*si,second*co+first*si),axis=-1))


def gelu(x, profile):
    p=dtype(profile);a=x.astype(p)
    if x.dtype != np.float16:raise ValueError('GELU Half storage')
    cube=(a*a)*a
    z=p(math.sqrt(2/math.pi))*(a+p(.044715)*cube)
    return half((p(.5)*a)*(p(1)+np.tanh(z)))


def visibility(tokens,window):
    if type(tokens) is not int or not 1<=tokens<=544 or window not in (None,4,512):
        raise ValueError('Bounded visibility geometry')
    query=np.arange(tokens)[:,None];key=np.arange(tokens)[None,:]
    mask=key<=query
    if window is not None:mask &= key>=query-window+1
    return mask


def attention(q,k,v,window,profile):
    p=dtype(profile);n=len(q)
    if (q.dtype!=np.float16 or k.dtype!=np.float16 or v.dtype!=np.float16
            or q.ndim!=3 or q.shape[1]!=4 or k.shape!=(n,1,q.shape[2]) or v.shape!=k.shape
            or not 1<=n<=544 or window not in (None,4,512)):
        raise ValueError('Bounded GQA attention shape')
    finite(q);finite(k);finite(v)
    # No expanded KV heads. All raw dot scores are retained, including masked
    # cells; only probabilities are masked to zero. Stores precede later use.
    scores=finite(((q.astype(p).transpose(1,0,2) @ k[:,0,:].astype(p).T) / p(math.sqrt(q.shape[2]))).astype(np.float32))
    mask=visibility(n,window)
    widened=scores.astype(p)
    masked=np.where(mask[None,:,:],widened,p(-np.inf))
    maximum=np.max(masked,axis=-1,keepdims=True)
    weights=np.exp(masked-maximum)
    mass=np.sum(weights,axis=-1,keepdims=True,dtype=p)
    probability=finite((weights/mass).astype(np.float32))
    context=half((probability.astype(p) @ v[:,0,:].astype(p)).transpose(1,0,2).reshape(n,-1))
    return scores,probability,context


def forward(weights,tokens,selected_rows,profile,emit,logits_only=False):
    """One bounded full causal graph. Head sees selected rows ONLY.

    emit(row,stage,array) must consume synchronously; every selected slice is a
    detached copy. No producer object retains full-layer activations/traces.
    """
    dtype(profile)
    if (not tokens or len(tokens)>544 or any(type(t) is not int or not 0<=t<weights.vocab for t in tokens)
            or selected_rows!=sorted(set(selected_rows)) or not selected_rows
            or any(type(row) is not int or not 0<=row<len(tokens) for row in selected_rows)):
        raise ValueError('Bounded token/selected-row request')
    n=len(tokens)
    def save(name,a):
        finite(a)
        if not logits_only:
            scores=name.endswith(('attention_scores','attention_probabilities'))
            for row in selected_rows:
                emit(row,name,(a[:,row:row+1,:] if scores else a[row:row+1]).copy())
        return a
    table=weights.matrices['model.embed_tokens']
    x=save('embedding_lookup',half(table[np.asarray(tokens,dtype=np.intp)]))
    x=save('embedding',half(x.astype(dtype(profile))*dtype(profile)(25.25)))
    for layer in range(weights.layers):
        prefix=f'model.layers.{layer}';trace=f'layers.{layer}'
        def store(name,a):return save(trace+'.'+name,a)
        def normalized(name,value,field):return store(name,norm(value,weights.norms[prefix+'.'+field+'.weight'],profile))
        z=normalized('input_norm',x,'input_layernorm')
        q=store('q_projection',projection(z,weights.matrices[prefix+'.self_attn.q_proj'],profile).reshape(n,4,weights.dim))
        k=store('k_projection',projection(z,weights.matrices[prefix+'.self_attn.k_proj'],profile).reshape(n,1,weights.dim))
        v=store('v_projection',projection(z,weights.matrices[prefix+'.self_attn.v_proj'],profile).reshape(n,1,weights.dim))
        q=normalized('q_norm',q,'self_attn.q_norm');k=normalized('k_norm',k,'self_attn.k_norm')
        local=(layer+1)%6!=0;theta=10000 if local else 1000000
        q=store('q_rotary',rotary(q,theta,profile));k=store('k_rotary',rotary(k,theta,profile))
        scores,probabilities,context=attention(q,k,v,512 if local else None,profile)
        store('attention_scores',scores);store('attention_probabilities',probabilities);store('attention_context',context)
        del scores,probabilities,q,k,v
        branch=store('attention_projection',projection(context,weights.matrices[prefix+'.self_attn.o_proj'],profile))
        branch=normalized('post_attention_norm',branch,'post_attention_layernorm')
        x=store('attention_residual',residual(x,branch))
        z=normalized('pre_ff_norm',x,'pre_feedforward_layernorm')
        gate=store('gate_projection',projection(z,weights.matrices[prefix+'.mlp.gate_proj'],profile))
        up=store('up_projection',projection(z,weights.matrices[prefix+'.mlp.up_proj'],profile))
        activated=store('gate_gelu',gelu(gate,profile))
        gated=store('gated',half(activated.astype(dtype(profile))*up.astype(dtype(profile))))
        down=store('down_projection',projection(gated,weights.matrices[prefix+'.mlp.down_proj'],profile))
        branch=normalized('post_ff_norm',down,'post_feedforward_layernorm')
        x=store('output',residual(x,branch))
    z=save('final_norm',norm(x,weights.norms['model.norm.weight'],profile))
    result=projection(z[selected_rows].copy(),table,profile,output_float=True)
    for index,row in enumerate(selected_rows):emit(row,'logits',result[index:index+1].copy())
    return result
