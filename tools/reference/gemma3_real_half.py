"""Independent bounded NumPy NativeHalf Gemma3 graph; no native/MLX imports.

MAIN projections use certified binary64 stripes with exact dyadic fallback;
MAIN attention follows declared serial FP32 operations. WIDE keeps its original
binary64 BLAS profile. Neither path emulates native lane/SIMD compensation. Every named activation is stored Half; scores,
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



# At most16MiB of binary64 weight conversion at the real maximum input width2048.
# No full abs(weight) copy, expanded Half model, or native-output lookup table.
PROJECTION_STRIPE=1024
PROJECTION_AUDIT={}


def reset_projection_audit():
    PROJECTION_AUDIT.clear()
    PROJECTION_AUDIT.update(calls=0,values=0,certified=0,exact_fallback=0,
                            stripes=0,max_converted_bytes=0,witnesses=[])


reset_projection_audit()


def dyadic_units(word,half_input):
    """Exact signed integer units: Half2^-24 or binary32 coefficient2^-149."""
    fraction_bits,exponent_bits=(10,5) if half_input else (23,8)
    fraction=word & ((1<<fraction_bits)-1)
    exponent=(word>>fraction_bits)&((1<<exponent_bits)-1)
    if exponent==(1<<exponent_bits)-1:raise ValueError('Nonfinite exact dot operand')
    magnitude=fraction if exponent==0 else ((1<<fraction_bits)|fraction)<<(exponent-1)
    return -magnitude if word>>(fraction_bits+exponent_bits) else magnitude


def exact_store_word(total,output_float):
    """Round exact integer*2^-173 directly to binary16/32, nearest ties-to-even.

    Zero exact sums canonicalize to+0. Nonzero sums preserve sign through zero.
    Overflow is an error, never a clamp. Residual saturation remains separate.
    """
    precision,minimum,bias,exponent_bits=(24,-126,127,8) if output_float else (11,-14,15,5)
    sign=int(total<0);value=abs(total)
    if not value:return 0
    exponent=value.bit_length()-1-173
    step=max(minimum-(precision-1),exponent-(precision-1))
    shift=173+step
    if shift>0:
        rounded,remainder=divmod(value,1<<shift)
        midpoint=1<<(shift-1)
        rounded+=int(remainder>midpoint or (remainder==midpoint and rounded&1))
    else:rounded=value<<(-shift)
    if rounded>=(1<<precision):rounded>>=1;step+=1
    sign_word=sign<<(precision-1+exponent_bits)
    if rounded<(1<<(precision-1)):return sign_word|rounded
    stored_exponent=step+(precision-1)+bias
    if stored_exponent>=(1<<exponent_bits)-1:raise ValueError('Nonfinite exact projection store')
    return sign_word|(stored_exponent<<(precision-1))|(rounded-(1<<(precision-1)))


def exact_dot_word(x,weight,output_float):
    # Each finite Half*FP32 product is dyadic and exact in units2^-173.
    # Use integer arithmetic only for analytically ambiguous rounding cells.
    xx=x.view(np.uint16);ww=weight.view(np.uint32)
    total=sum(dyadic_units(int(a),True)*dyadic_units(int(b),False) for a,b in zip(xx,ww))
    return exact_store_word(total,output_float),total


def certified_projection(x,matrix,output_float):
    """Accurate original-FP32 dot -> declared storage, independent of GPU order.

    Every product of finite Half and FP32 fits binary64 exactly (<=35bits and
    exponent range safely inside binary64). For audited ordinary binary64 BLAS,
    gamma_n bounds any summation tree's rounding error. A conservative sumabs
    upper bound uses exact Half row sums and maxabs of each FP32 weight row.
    Endpoint nextafter operations expand all bound arithmetic outwards. Only
    intervals whose two ends round to the same finite destination word certify;
    midpoint/cancellation/underflow ambiguity uses an exact integer dot instead.
    This does not claim binary64 BLAS alone is correctly rounded.
    """
    import hashlib
    if not 1<=x.shape[1]<=2048 or not 1<=len(x)<=544 or not 1<=len(matrix)<=262144:
        raise ValueError('Certified projection bounded shape')
    finite(x)
    rows,columns=len(x),len(matrix);width=x.shape[1]
    store=np.float32 if output_float else np.float16
    word=np.uint32 if output_float else np.uint16
    result=np.empty((rows,columns),dtype=store)
    xx=x.astype(np.float64)
    # Half values in units2^-24, at width<=2048, fit an exact <=51bit sum.
    sumabs=np.sum(np.abs(xx),axis=1,dtype=np.float64)
    unit=2.0**-53
    gamma=np.nextafter((width*unit)/(1-width*unit),np.inf)
    PROJECTION_AUDIT['calls']+=1;call=PROJECTION_AUDIT['calls']
    for start in range(0,columns,PROJECTION_STRIPE):
        stop=min(columns,start+PROJECTION_STRIPE)
        converted=matrix[start:stop].astype(np.float64)
        try:
            finite(converted)
            approximate=xx@converted.T
            maxabs=np.maximum(np.abs(np.min(converted,axis=1)),np.abs(np.max(converted,axis=1)))
            bound=np.nextafter(sumabs[:,None]*maxabs[None,:],np.inf)
            bound=np.nextafter(bound*gamma,np.inf)
            low=np.nextafter(approximate-bound,-np.inf)
            high=np.nextafter(approximate+bound,np.inf)
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                low_store=low.astype(store);high_store=high.astype(store)
            certain=(low_store.view(word)==high_store.view(word)) & np.isfinite(low_store)
            target=result[:,start:stop];target[:]=low_store
            uncertain=np.argwhere(~certain)
            PROJECTION_AUDIT['stripes']+=1
            PROJECTION_AUDIT['max_converted_bytes']=max(PROJECTION_AUDIT['max_converted_bytes'],converted.nbytes)
            PROJECTION_AUDIT['values']+=certain.size
            PROJECTION_AUDIT['certified']+=int(np.count_nonzero(certain))
            PROJECTION_AUDIT['exact_fallback']+=len(uncertain)
            for row,col in uncertain:
                try:scalar,exact=exact_dot_word(x[row],matrix[start+col],output_float)
                except BaseException as exc:
                    PROJECTION_AUDIT['failed_exact_store']=dict(call=call,row=int(row),column=int(start+col),
                        input_sha256=hashlib.sha256(x[row].tobytes()).hexdigest(),
                        weight_sha256=hashlib.sha256(matrix[start+col].tobytes()).hexdigest(),error=repr(exc))
                    raise
                target.view(word)[row,col]=scalar
                if len(PROJECTION_AUDIT['witnesses'])<32:
                    PROJECTION_AUDIT['witnesses'].append(dict(call=call,row=int(row),column=int(start+col),
                        input_sha256=hashlib.sha256(x[row].tobytes()).hexdigest(),
                        weight_sha256=hashlib.sha256(matrix[start+col].tobytes()).hexdigest(),
                        exact_units_2neg173=str(exact),destination_word=scalar,output_float=bool(output_float)))
            finite(target)
        finally:del converted
    return result


def projection(x, matrix, profile, output_float=False):
    p = dtype(profile)
    if x.dtype != np.float16 or matrix.dtype != np.float32 or x.ndim != 2 or matrix.ndim != 2 or x.shape[1] != matrix.shape[1]:
        raise ValueError('Projection storage/shape')
    if profile == 'main':return certified_projection(x,matrix,output_float)
    # WIDE unchanged: only this operation owns its widened matrix.
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



def serial_attention(q,k,v,window):
    """Independent serial FP32 Reference contract, vectorized only over outputs.

    Reduction loops preserve increasing feature/key order. Products and additions
    are separate NumPy ufuncs, never BLAS or fused multiply-add. CPU exp/sqrt are
    FP32 operations, not a claim of exact Metal transcendental implementation.
    Raw masked scores remain observable; masked probabilities are exact+0.
    """
    n=len(q);dim=q.shape[2]
    qq=q.astype(np.float32).transpose(1,0,2)
    kk=k[:,0,:].astype(np.float32);vv=v[:,0,:].astype(np.float32)
    scores=np.zeros((4,n,n),dtype=np.float32)
    for feature in range(dim):
        product=qq[:,:,feature,None]*kk[None,None,:,feature]
        scores=scores+product
    scores=finite(scores/np.sqrt(np.float32(dim)))
    mask=visibility(n,window)
    maximum=np.full((4,n),-np.inf,dtype=np.float32)
    for key in range(n):maximum=np.maximum(maximum,np.where(mask[None,:,key],scores[:,:,key],np.float32(-np.inf)))
    weights=np.exp(np.where(mask[None,:,:],scores-maximum[:,:,None],np.float32(-np.inf))).astype(np.float32)
    mass=np.zeros((4,n),dtype=np.float32)
    for key in range(n):mass=mass+weights[:,:,key]
    if not np.isfinite(mass).all() or np.any(mass<=0):raise ValueError('Invalid Reference attention mass')
    probability=finite(weights/mass[:,:,None])
    total=np.zeros((4,n,dim),dtype=np.float32)
    for key in range(n):
        product=probability[:,:,key,None]*vv[key,None,None,:]
        # MSL skips masked context cells rather than adding zero products.
        np.add(total,product,out=total,where=mask[None,:,key,None])
    context=half(total.transpose(1,0,2).reshape(n,-1))
    return scores,probability,context


def attention(q,k,v,window,profile):
    p=dtype(profile);n=len(q)
    if (q.dtype!=np.float16 or k.dtype!=np.float16 or v.dtype!=np.float16
            or q.ndim!=3 or q.shape[1]!=4 or k.shape!=(n,1,q.shape[2]) or v.shape!=k.shape
            or not 1<=n<=544 or window not in (None,4,512)):
        raise ValueError('Bounded GQA attention shape')
    finite(q);finite(k);finite(v)
    if profile == 'main':return serial_attention(q,k,v,window)
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
