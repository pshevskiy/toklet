"""Bias-free Llama Half-storage CPU graph; no Rust/Metal/MLX execution.

Reuse unchanged independently specified numerical primitives from qwen_fp16;
weight mapping, bias absence and graph admission are Llama-specific.
"""
import numpy as np
from qwen_fp16 import linear, rmsnorm, rotary, attention

def forward(tokens, offset, weights, config, precision, layers):
    """Round every stored named activation; retain FP32 scores/probabilities/logits."""
    if isinstance(weights,dict) and any(name.endswith('.bias') for name in weights):
        raise ValueError('Bias-free Llama reference rejects projection biases')
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
        query = save("q_rotary", rotary(projected["q"], offset, config["rope_theta"], precision))
        key = save("k_rotary", rotary(projected["k"], offset, config["rope_theta"], precision))
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

