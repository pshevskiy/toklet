"""Bias-free Llama3 Half-storage CPU graph; no Rust/Metal/MLX execution.

Reuse unchanged independently specified numerical primitives from qwen_fp16;
weight mapping, bias absence and graph admission are Llama-specific.
"""
import numpy as np
from qwen_fp16 import linear, rmsnorm, attention
from decimal import Decimal, localcontext
import math
from llama3_rotary import scaled_frequency


def frequencies(dimension,theta,precision,scaling):
    if dimension<=0 or dimension%2 or not math.isfinite(theta) or theta<1:
        raise ValueError('Unsupported native Llama positional domain')
    factor,low,high,original=(scaling[k] for k in ['factor','low','high','original'])
    if not (all(math.isfinite(v) for v in [factor,low,high]) and factor>=1 and 0<low<high
            and type(original) is int and 0<original<=2**53):raise ValueError('Invalid scaling')
    result=[]
    for pair in range(dimension//2):
        if precision==np.float32:
            ordinary=float(theta)**(-2.0*pair/dimension)
            wavelength=math.tau/ordinary
            if wavelength<original/high:value=ordinary
            elif wavelength>original/low:value=ordinary/factor
            else:
                smooth=(original/wavelength-low)/(high-low)
                value=(1-smooth)*ordinary/factor+smooth*ordinary
            result.append(np.float32(value))
        elif precision==np.float64:
            with localcontext() as ctx:
                ctx.prec=80
                ordinary=(Decimal(theta).ln()*Decimal(-2*pair)/dimension).exp()
                result.append(float(scaled_frequency(ordinary,Decimal(factor),Decimal(low),Decimal(high),Decimal(original))))
        else:raise ValueError('Unsupported arithmetic profile')
    values=np.array(result,dtype=precision)
    if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Invalid frequency')
    if precision==np.float32 and np.any(values<np.finfo(np.float32).tiny):raise ValueError('Subnormal inverse')
    return values


def rotary(x,offset,theta,precision,scaling):
    if x.ndim!=3 or x.dtype!=np.float16 or type(offset) is not int or offset<0 or offset+len(x)>512:
        raise ValueError('Invalid rotary input/offset')
    dim=x.shape[-1];half=dim//2
    inverse=frequencies(dim,theta,precision,scaling)
    angles=np.arange(offset,offset+len(x),dtype=precision)[:,None,None]*inverse[None,None]
    # NumPy operations make each FP32 product and add separate stores. Trig
    # implementation remains the pinned CPU profile, not a GPU exact oracle.
    first,second=x[...,:half].astype(precision),x[...,half:].astype(precision)
    cosine,sine=np.cos(angles),np.sin(angles)
    left=first*cosine-second*sine;right=second*cosine+first*sine
    return np.concatenate([left,right],axis=-1).astype(np.float16)

def forward(tokens, offset, weights, config, precision, layers, scaling):
    """Round every stored named activation; retain FP32 scores/probabilities/logits."""
    if isinstance(weights,dict) and any(name.endswith('.bias') for name in weights):
        raise ValueError('Bias-free Llama reference rejects projection biases')
    if not tokens or type(offset) is not int or offset<0 or offset+len(tokens)>512:
        raise ValueError("Invalid bounded graph tokens/offset")
    n, dimension = len(tokens), config["head_dim"]
    x = weights["embed_tokens"][tokens].copy()
    values = {"embedding": x}
    for index in range(config["num_layers"]):
        prefix = f"layers.{index}"

        def save(name, tensor):
            if not np.isfinite(tensor).all():
                raise ValueError(f"Nonfinite FP16 reconstruction: {prefix}.{name}")
            if index in layers:
                values[f"{prefix}.{name}"] = tensor
            return tensor

        normalized = save("input_norm", rmsnorm(x, weights[f"{prefix}.input_layernorm"], config["rms_norm_eps"], precision))
        projected = {}
        for name, heads in (("q", config["num_query_heads"]), ("k", config["num_kv_heads"]), ("v", config["num_kv_heads"])):
            projected[name] = save(f"{name}_projection", linear(normalized, weights[f"{prefix}.{name}_proj.weight"],
                                     None, precision)).reshape(n, heads, dimension)
        query = save("q_rotary", rotary(projected["q"], offset, config["rope_theta"], precision, scaling))
        key = save("k_rotary", rotary(projected["k"], offset, config["rope_theta"], precision, scaling))
        scores, probabilities, context = attention(query, key, projected["v"], precision)
        save("attention_scores", scores)
        save("attention_probabilities", probabilities)
        save("attention_context", context)
        projected_attention = save("attention_projection", linear(context, weights[f"{prefix}.o_proj.weight"], None, precision))
        x = save("attention_residual", (x.astype(precision) + projected_attention).astype(np.float16))
        normalized = save("post_attention_norm", rmsnorm(x, weights[f"{prefix}.post_attention_layernorm"], config["rms_norm_eps"], precision))
        gate = save("gate_projection", linear(normalized, weights[f"{prefix}.gate_proj.weight"], None, precision))
        up = save("up_projection", linear(normalized, weights[f"{prefix}.up_proj.weight"], None, precision))
        wide = gate.astype(precision)
        exp = np.exp(-np.abs(wide))
        sigmoid = np.where(wide >= 0, precision(1) / (precision(1) + exp), exp / (precision(1) + exp))
        activated = save("gate_silu", (wide * sigmoid).astype(np.float16))
        gated = save("gated", (activated.astype(precision) * up).astype(np.float16))
        down = save("down_projection", linear(gated, weights[f"{prefix}.down_proj.weight"], None, precision))
        x = save("output", (x.astype(precision) + down).astype(np.float16))
    values["final_norm"] = rmsnorm(x, weights["norm"], config["rms_norm_eps"], precision)
    values["logits"] = linear(values["final_norm"], weights["embed_tokens"], None, precision, np.float32)
    if not all(np.isfinite(array).all() for array in values.values()):
        raise ValueError("Nonfinite final FP16 reconstruction")
    return values

