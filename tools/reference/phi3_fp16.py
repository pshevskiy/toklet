"""Independent ordinary-RoPE Phi3 Half graph, using original-row reader owners.

Reuse only unchanged generic norm/attention helpers. The concrete reader owns
one temporary projection matrix at a time; no tied-head or scaled-RoPE path.
"""
from decimal import Decimal, localcontext
import math
import numpy as np
from qwen_fp16 import rmsnorm, attention


def frequencies(dimension,theta,precision):
    if type(dimension) is not int or dimension<=0 or dimension%2 or not math.isfinite(theta) or theta<1:
        raise ValueError('Unsupported ordinary rotary domain')
    if precision==np.float32:
        exponent=np.arange(dimension//2,dtype=np.float32)*np.float32(2)/np.float32(dimension)
        result=np.power(np.float32(theta),-exponent).astype(np.float32)
    elif precision==np.float64:
        with localcontext() as ctx:
            ctx.prec=80
            result=np.array([float((Decimal(theta).ln()*Decimal(-2*j)/dimension).exp())
                             for j in range(dimension//2)],dtype=np.float64)
    else:raise ValueError('Unsupported arithmetic profile')
    if not np.isfinite(result).all() or np.any(result<=0):raise ValueError('Invalid ordinary frequencies')
    return result


def rotary(x,offset,theta,precision):
    if x.ndim!=3 or x.dtype!=np.float16 or type(offset) is not int or offset<0 or offset+len(x)>512:
        raise ValueError('Invalid rotary input/offset')
    half=x.shape[-1]//2;inverse=frequencies(x.shape[-1],theta,precision)
    angles=np.arange(offset,offset+len(x),dtype=precision)[:,None,None]*inverse[None,None]
    first,second=x[...,:half].astype(precision),x[...,half:].astype(precision)
    cosine,sine=np.cos(angles),np.sin(angles)
    return np.concatenate([first*cosine-second*sine,second*cosine+first*sine],axis=-1).astype(np.float16)


def forward(tokens, offset, weights, config, precision, layers):
    """Round every stored named activation; retain FP32 scores/probabilities/logits."""
    if not tokens or type(offset) is not int or offset<0 or offset+len(tokens)>512:
        raise ValueError("Invalid bounded graph tokens/offset")
    n, dimension = len(tokens), config["head_dim"]
    x = weights.lookup(tokens)
    values = {"embedding": x}
    for index in range(config["num_layers"]):
        prefix = f"layers.{index}"

        def save(name, tensor):
            if not np.isfinite(tensor).all():
                raise ValueError(f"Nonfinite FP16 reconstruction: {prefix}.{name}")
            if index in layers:
                values[f"{prefix}.{name}"] = tensor
            return tensor

        normalized = save("input_norm", rmsnorm(x, weights.norm(f"{prefix}.input_layernorm"), config["rms_norm_eps"], precision))
        projected = {}
        for name, heads in (("q", config["num_query_heads"]), ("k", config["num_kv_heads"]), ("v", config["num_kv_heads"])):
            projected[name] = save(f"{name}_projection", weights.project(f"{prefix}.{name}_proj.weight", normalized, precision)).reshape(n, heads, dimension)
        query = save("q_rotary", rotary(projected["q"], offset, config["rope_theta"], precision))
        key = save("k_rotary", rotary(projected["k"], offset, config["rope_theta"], precision))
        scores, probabilities, context = attention(query, key, projected["v"], precision)
        save("attention_scores", scores)
        save("attention_probabilities", probabilities)
        save("attention_context", context)
        projected_attention = save("attention_projection", weights.project(f"{prefix}.o_proj.weight", context, precision))
        x = save("attention_residual", (x.astype(precision) + projected_attention).astype(np.float16))
        normalized = save("post_attention_norm", rmsnorm(x, weights.norm(f"{prefix}.post_attention_layernorm"), config["rms_norm_eps"], precision))
        gate = save("gate_projection", weights.project(f"{prefix}.gate_proj.weight", normalized, precision))
        up = save("up_projection", weights.project(f"{prefix}.up_proj.weight", normalized, precision))
        wide = gate.astype(precision)
        exp = np.exp(-np.abs(wide))
        sigmoid = np.where(wide >= 0, precision(1) / (precision(1) + exp), exp / (precision(1) + exp))
        activated = save("gate_silu", (wide * sigmoid).astype(np.float16))
        gated = save("gated", (activated.astype(precision) * up).astype(np.float16))
        down = save("down_projection", weights.project(f"{prefix}.down_proj.weight", gated, precision))
        x = save("output", (x.astype(precision) + down).astype(np.float16))
    values["final_norm"] = rmsnorm(x, weights.norm("norm"), config["rms_norm_eps"], precision)
    values["logits"] = weights.project("lm_head.weight", values["final_norm"], precision, np.float32)
    if not all(np.isfinite(array).all() for array in values.values()):
        raise ValueError("Nonfinite final FP16 reconstruction")
    return values
