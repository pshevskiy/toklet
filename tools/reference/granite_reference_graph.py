"""Bounded independent Granite Half-storage graph; numerical imports are explicit.

FP32 and FP64 NumPy BLAS profiles share the same reconstructed FP32 weights,
Half norm/gather/activation/KV boundaries and FP32 score/probability/logit stores.
They are independent arithmetic profiles, not scalar-order or GPU emulation.
The caller admits NumPy before supplying it, and publishes each stage immediately.
"""
from decimal import Decimal, localcontext
import math

import granite_quantize as quant

need = quant.need
STAGES = ('input_norm', 'q', 'k', 'v', 'q_rotary', 'k_rotary',
          'attention_context', 'attention_projection', 'attention_residual',
          'post_attention_norm', 'gate', 'silu', 'up', 'gated', 'down', 'output')


def geometry(config, tokens, offset):
    integer_limits = dict(hidden=2560, intermediate=8192, layers=40, vocabulary=100352,
                          query_heads=40, kv_heads=8, head_dim=64)
    scalars = ('theta', 'rms_eps', 'attention_scale', 'embedding_scale',
               'residual_scale', 'logits_divisor')
    need(type(config) is dict and set(config) == set(integer_limits) | set(scalars) | {'tied_head'},
         'Complete explicit Granite reference geometry required')
    for name, limit in integer_limits.items():
        need(type(config[name]) is int and 0 < config[name] <= limit,
             'Bounded Granite dimension required: ' + name)
    need(config['hidden'] == config['query_heads'] * config['head_dim']
         and config['query_heads'] % config['kv_heads'] == 0
         and config['head_dim'] % 2 == 0 and type(config['tied_head']) is bool,
         'Explicit split-half rotary and grouped-query geometry required')
    for name in scalars:
        value = config[name]
        need(type(value) in (int, float) and math.isfinite(value)
             and 2**-126 <= value <= (2 - 2**-23) * 2**127,
             'Positive normal FP32 source scalar required: ' + name)
    need(type(tokens) is list and 1 <= len(tokens) <= 512
         and all(type(t) is int and 0 <= t < config['vocabulary'] for t in tokens)
         and type(offset) is int and 0 <= offset and offset + len(tokens) <= 512,
         'Exact bounded nonempty tokens and absolute positions required')


def contracts(config, count):
    h, kv = config['hidden'], config['kv_heads'] * config['head_dim']
    result = {name: (count, width) for name, width in
              [('embedding', h), ('final_norm', h), ('logits', config['vocabulary'])]}
    widths = dict.fromkeys(STAGES, h)
    widths.update(k=kv, v=kv, k_rotary=kv)
    widths.update({name: config['intermediate'] for name in ('gate', 'silu', 'up', 'gated')})
    for layer in range(config['layers']):
        result.update({f'layers.{layer}.{name}': (count, width) for name, width in widths.items()})
    return result


class PackedWeights:
    """One authenticated child reader and at most one decoded row stripe."""
    def __init__(self, child):
        self.child = child

    def shape(self, name):
        return tuple(self.child.planned['logical'][name])

    def rows(self, name, first, count, np):
        matrix = self.child.planned['matrices'][name]
        need(type(first) is int and first % quant.ROWS == 0
             and type(count) is int and count == self.child.tiles[name, first]['rows'],
             'Exact authenticated row stripe required')
        raw = quant.reconstruct_f32(self.child.tile(name, first), count,
                                    matrix['shape'][1], matrix['bits'], np)
        return np.frombuffer(raw, dtype='<f4').reshape(count, matrix['shape'][1])

    def norm(self, name, np):
        raw = np.frombuffer(self.child.norm(name), dtype='<u2')
        need(not np.any((raw & 0x7f80) == 0x7f80), 'Nonfinite BF16 norm')
        values = (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
        need(np.isfinite(values).all(), 'Norm exceeds native Half storage')
        return values

    def gather(self, tokens, np):
        name = 'model.embed_tokens.weight'
        rows, columns = self.shape(name)
        output = np.empty((len(tokens), columns), dtype=np.float16)
        previous, tile = -1, None
        for index, token in enumerate(tokens):
            first = token // quant.ROWS * quant.ROWS
            if first != previous:
                tile = self.rows(name, first, min(quant.ROWS, rows - first), np)
                previous = first
            output[index] = tile[token - first]
        need(np.isfinite(output).all(), 'Embedding exceeds native Half storage')
        return output


def project(owner, name, x, precision, np, output_dtype=None):
    rows, columns = owner.shape(name)
    need(x.ndim == 2 and x.dtype == np.float16 and x.shape[1] == columns
         and 0 < len(x) <= 512 and rows <= 100352 and columns <= 8192,
         'Bounded Half input and row-major projection required')
    output = np.empty((len(x), rows), dtype=np.float16 if output_dtype is None else output_dtype)
    lhs = x.astype(precision)
    for first in range(0, rows, quant.ROWS):
        count = min(quant.ROWS, rows - first)
        weight = owner.rows(name, first, count, np)
        values = lhs @ weight.astype(precision, copy=False).T
        need(np.isfinite(values).all(), 'Nonfinite independent matrix reduction')
        output[:, first:first + count] = values
    need(np.isfinite(output).all(), 'Projection exceeds its storage precision')
    return output


def norm(x, weights, epsilon, precision, np):
    wide = x.astype(precision)
    mean = np.mean(wide * wide, axis=-1, keepdims=True, dtype=precision)
    inverse = precision(1) / np.sqrt(mean + precision(np.float32(epsilon)))
    return (wide * inverse * weights.astype(precision)).astype(np.float16)


def rotary(x, offset, theta, wide, np):
    dimension = x.shape[-1]
    half = dimension // 2
    if wide:
        with localcontext() as context:
            context.prec = 80
            logarithm = Decimal(str(float(np.float32(theta)))).ln()
            frequency = [(logarithm * Decimal(-2 * i) / dimension).exp() for i in range(half)]
            angles = np.asarray([[float(Decimal(offset + row) * value) for value in frequency]
                                 for row in range(len(x))], dtype=np.float64)[:, None, :]
        precision = np.float64
    else:
        precision = np.float32
        frequency = precision(theta) ** (-np.arange(half, dtype=precision) * precision(2 / dimension))
        angles = np.arange(offset, offset + len(x), dtype=precision)[:, None, None] * frequency[None, None]
    first, second = x[..., :half].astype(precision), x[..., half:].astype(precision)
    cosine, sine = np.cos(angles), np.sin(angles)
    return np.concatenate([first * cosine - second * sine,
                           second * cosine + first * sine], axis=-1).astype(np.float16)


def attention(q, k, v, multiplier, precision, np):
    count, heads, dimension = q.shape
    group = heads // k.shape[1]
    context = np.empty((count, heads, dimension), dtype=np.float16)
    causal = np.tri(count, dtype=bool)
    # Scores and probabilities are stored FP32 per head. Neither all-head
    # square tensors nor expanded K/V heads need to be retained.
    for head in range(heads):
        kv = head // group
        scores = (q[:, head].astype(precision) @ k[:, kv].astype(precision).T
                  * precision(np.float32(multiplier))).astype(np.float32)
        valid = np.where(causal, scores.astype(precision), -np.inf)
        exponential = np.exp(valid - np.max(valid, axis=-1, keepdims=True))
        probability = (exponential / np.sum(exponential, axis=-1, keepdims=True,
                                           dtype=precision)).astype(np.float32)
        context[:, head] = probability.astype(precision) @ v[:, kv].astype(precision)
    return context.reshape(count, heads * dimension)


def forward(owner, config, tokens, offset, wide, np, emit):
    geometry(config, tokens, offset)
    need(type(wide) is bool, 'Explicit FP32 or FP64 reference profile required')
    precision = np.float64 if wide else np.float32
    expected = contracts(config, len(tokens))
    seen = set()

    def save(name, value):
        value = value.reshape(len(tokens), -1)
        need(name in expected and name not in seen and value.shape == expected[name]
             and value.dtype == (np.float32 if name == 'logits' else np.float16)
             and np.isfinite(value).all(), 'Complete finite stored graph stage required: ' + name)
        emit(name, value)
        seen.add(name)
        return value

    def residual(x, branch):
        coefficient = precision(np.float16(config['residual_scale']))
        scaled = (branch.astype(precision) * coefficient).astype(np.float16)
        return (x.astype(precision) + scaled.astype(precision)).astype(np.float16)

    x = owner.gather(tokens, np)
    x = save('embedding', (x.astype(precision) * precision(np.float16(config['embedding_scale'])))
             .astype(np.float16))
    for layer in range(config['layers']):
        prefix, stage = f'model.layers.{layer}', f'layers.{layer}.'
        normalized = save(stage + 'input_norm', norm(x, owner.norm(prefix + '.input_layernorm.weight', np),
                                                     config['rms_eps'], precision, np))
        q = save(stage + 'q', project(owner, prefix + '.self_attn.q_proj.weight', normalized, precision, np))
        k = save(stage + 'k', project(owner, prefix + '.self_attn.k_proj.weight', normalized, precision, np))
        v = save(stage + 'v', project(owner, prefix + '.self_attn.v_proj.weight', normalized, precision, np))
        q = save(stage + 'q_rotary', rotary(q.reshape(len(tokens), config['query_heads'], -1),
                                          offset, config['theta'], wide, np))
        k = save(stage + 'k_rotary', rotary(k.reshape(len(tokens), config['kv_heads'], -1),
                                          offset, config['theta'], wide, np))
        context = save(stage + 'attention_context', attention(
            q.reshape(len(tokens), config['query_heads'], -1),
            k.reshape(len(tokens), config['kv_heads'], -1),
            v.reshape(len(tokens), config['kv_heads'], -1), config['attention_scale'], precision, np))
        branch = save(stage + 'attention_projection', project(owner, prefix + '.self_attn.o_proj.weight',
                                                             context, precision, np))
        x = save(stage + 'attention_residual', residual(x, branch))
        del normalized, q, k, v, context, branch
        normalized = save(stage + 'post_attention_norm', norm(
            x, owner.norm(prefix + '.post_attention_layernorm.weight', np), config['rms_eps'], precision, np))
        gate = save(stage + 'gate', project(owner, prefix + '.mlp.gate_proj.weight', normalized, precision, np))
        up = save(stage + 'up', project(owner, prefix + '.mlp.up_proj.weight', normalized, precision, np))
        values = gate.astype(precision)
        exponential = np.exp(-np.abs(values))
        denominator = precision(1) + exponential
        sigmoid = np.where(values >= 0, precision(1) / denominator, exponential / denominator)
        activated = save(stage + 'silu', (values * sigmoid).astype(np.float16))
        gated = save(stage + 'gated', (activated.astype(precision) * up.astype(precision)).astype(np.float16))
        down = save(stage + 'down', project(owner, prefix + '.mlp.down_proj.weight', gated, precision, np))
        x = save(stage + 'output', residual(x, down))
        del normalized, gate, up, values, exponential, denominator, sigmoid, activated, gated, down
    final = save('final_norm', norm(x, owner.norm('model.norm.weight', np), config['rms_eps'], precision, np))
    name = 'model.embed_tokens.weight' if config['tied_head'] else 'lm_head.weight'
    logits = project(owner, name, final, precision, np, np.float32)
    logits /= np.float32(config['logits_divisor'])
    save('logits', logits)
    need(seen == set(expected), 'Complete graph stage coverage required')
    return logits[-1].copy()
