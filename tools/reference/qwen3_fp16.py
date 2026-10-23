"""Independent dense Qwen3 Half-storage graph, using CPU numerical primitives.

Q/K projection widths may differ from hidden. Learned head-wise RMSNorm precedes
RoPE; V is unchanged. No Rust, Metal, MLX or repository model code executes.
"""
import numpy as np
from qwen_fp16 import linear, rmsnorm, rotary, attention


def forward(tokens, offset, weights, config, precision, layers):
    if isinstance(weights, dict) and any(name.endswith('.bias') for name in weights):
        raise ValueError('Bias-free Qwen3 reference rejects projection biases')
    n, dimension = len(tokens), config['head_dim']
    x = weights['embed_tokens'][tokens].copy()
    values = {'embedding': x}
    for index in range(config['num_layers']):
        prefix = f'layers.{index}'

        def save(name, value):
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite Qwen3 activation: ' + prefix + '.' + name)
            if index in layers:
                values[prefix + '.' + name] = value
            return value

        normalized = save('input_norm', rmsnorm(x, weights[prefix + '.input_layernorm'], config['rms_norm_eps'], precision))
        projected = {}
        for name, heads in [('q', config['num_query_heads']), ('k', config['num_kv_heads']), ('v', config['num_kv_heads'])]:
            projected[name] = save(name + '_projection', linear(normalized, weights[prefix + '.' + name + '_proj.weight'],
                                    None, precision)).reshape(n, heads, dimension)
        normalized_heads = {}
        for name in ['q', 'k']:
            shape = projected[name].shape
            normalized_heads[name] = save(name + '_norm', rmsnorm(projected[name].reshape(-1, dimension),
                weights[prefix + '.' + name + '_norm'], config['rms_norm_eps'], precision).reshape(shape))
        query = save('q_rotary', rotary(normalized_heads['q'], offset, config['rope_theta'], precision))
        key = save('k_rotary', rotary(normalized_heads['k'], offset, config['rope_theta'], precision))
        scores, probabilities, context = attention(query, key, projected['v'], precision)
        save('attention_scores', scores);save('attention_probabilities', probabilities);save('attention_context', context)
        projection = save('attention_projection', linear(context, weights[prefix + '.o_proj.weight'], None, precision))
        x = save('attention_residual', (x.astype(precision) + projection).astype(np.float16))
        normalized = save('post_attention_norm', rmsnorm(x, weights[prefix + '.post_attention_layernorm'], config['rms_norm_eps'], precision))
        gate = save('gate_projection', linear(normalized, weights[prefix + '.gate_proj.weight'], None, precision))
        up = save('up_projection', linear(normalized, weights[prefix + '.up_proj.weight'], None, precision))
        wide = gate.astype(precision);exponential = np.exp(-np.abs(wide))
        sigmoid = np.where(wide >= 0, precision(1) / (precision(1) + exponential), exponential / (precision(1) + exponential))
        activated = save('gate_silu', (wide * sigmoid).astype(np.float16))
        gated = save('gated', (activated.astype(precision) * up).astype(np.float16))
        down = save('down_projection', linear(gated, weights[prefix + '.down_proj.weight'], None, precision))
        x = save('output', (x.astype(precision) + down).astype(np.float16))
    values['final_norm'] = rmsnorm(x, weights['norm'], config['rms_norm_eps'], precision)
    values['logits'] = linear(values['final_norm'], weights['embed_tokens'], None, precision, np.float32)
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError('Nonfinite Qwen3 final output')
    return values
