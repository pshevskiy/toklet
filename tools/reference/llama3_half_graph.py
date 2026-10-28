"""Independent tiny Llama3 Half-storage graphs; stdlib only, no target execution.

Frozen FP32-ordered and FP64-wide arithmetic use the SAME stored weights and
Half activation boundaries. Scores, probabilities and logits store FP32 in both.
This is an independent tolerance-envelope reference, not exact Metal emulation.
No existing producer or fixture is rewritten. Export requires a fresh file.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/reference'))
import llama3_graph as source
import llama3_rotary as positional

FORMAT = 'toklet-llama3-half-graph-v1'
POLICY = dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3)
CASES = (
    ('prompt', [1, 7, 13, 3], 0),
    ('future_changed', [1, 7, 13, 17], 0),
    ('offset5', [7, 13, 3], 5),
    ('offset509', [7, 13, 3], 509),
)
LAYER_STAGES = (
    'input_norm', 'q_projection', 'k_projection', 'v_projection',
    'q_rotary', 'k_rotary', 'attention_scores', 'attention_probabilities',
    'attention_context', 'attention_projection', 'attention_residual',
    'post_attention_norm', 'gate_projection', 'up_projection', 'gate_silu',
    'gated', 'down_projection', 'output',
)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def half(value):
    return struct.unpack('<e', struct.pack('<e', value))[0]


def bits(value, dtype):
    value_code, bit_code = ('e', 'H') if dtype == 'f16le' else ('f', 'I')
    return struct.unpack('<' + bit_code, struct.pack('<' + value_code, value))[0]


def flatten(values):
    if isinstance(values, list):
        return [item for value in values for item in flatten(value)]
    return [values]


def pin(path, limit=4 * 1024**2):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Expected bounded regular source file: ' + str(path))
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError('Source grew beyond bound')
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read_weights(variant, manifest):
    weights, config = source.matrices(variant, manifest)
    data = (source.SOURCE / variant / 'model.safetensors').read_bytes()
    size = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8 + size])
    packed = {name for name in weights if name.removesuffix('.weight') + '.scales' in header}
    for name, values in weights.items():
        if name not in packed:
            weights[name] = ([[half(v) for v in row] for row in values]
                             if isinstance(values[0], list) else [half(v) for v in values])
    # Packed matrices stay FP32 reconstructed in BOTH profiles, including the
    # tied vocabulary head. Only actual embedding gathers round to Half below.
    return weights, config, sorted(packed)


def dot(a, b, precision):
    if len(a) != len(b):
        raise ValueError('Dot shape differs')
    total = precision(0)
    for x, y in zip(a, b, strict=True):
        total = precision(total + precision(precision(x) * precision(y)))
    return total


def linear(x, weights, precision, store=half):
    return [[store(dot(row, weight, precision)) for weight in weights] for row in x]


def norm(x, weights, epsilon, precision):
    output = []
    for row in x:
        mean = precision(dot(row, row, precision) / precision(len(row)))
        root = precision(math.sqrt(precision(mean + precision(epsilon))))
        inverse = precision(precision(1) / root)
        output.append([half(precision(precision(precision(v) * inverse) * precision(w)))
                       for v, w in zip(row, weights, strict=True)])
    return output


def canonical_inverse(pair, dim, theta):
    ordinary = math.pow(float(f32(theta)), -2.0 * pair / dim)
    factor, low, high, original = 8.0, 1.0, 4.0, 64.0
    wavelength = math.tau / ordinary
    if wavelength < original / high:
        return ordinary
    if wavelength > original / low:
        return ordinary / factor
    smooth = (original / wavelength - low) / (high - low)
    return (1.0 - smooth) * ordinary / factor + smooth * ordinary


def rotary(x, heads, offset, wide):
    dim = 8
    output = [list(row) for row in x]
    with localcontext() as context:
        context.prec = 80
        for pair in range(dim // 2):
            if wide:
                inverse = (Decimal(10000).ln() * Decimal(-2 * pair) / dim).exp()
                inverse = positional.scaled_frequency(
                    inverse, Decimal(8), Decimal(1), Decimal(4), Decimal(64))
            else:
                inverse = f32(canonical_inverse(pair, dim, 10000))
            for token, row in enumerate(x):
                if wide:
                    angle = float(Decimal(offset + token) * inverse)
                    sine, cosine = math.sin(angle), math.cos(angle)
                else:
                    angle = f32(f32(offset + token) * inverse)
                    sine, cosine = f32(math.sin(angle)), f32(math.cos(angle))
                for head in range(heads):
                    a = head * dim + pair
                    b = a + dim // 2
                    if wide:
                        left = row[a] * cosine - row[b] * sine
                        right = row[b] * cosine + row[a] * sine
                    else:
                        left = f32(f32(row[a] * cosine) - f32(row[b] * sine))
                        right = f32(f32(row[b] * cosine) + f32(row[a] * sine))
                    output[token][a], output[token][b] = half(left), half(right)
    return output


def attention(query, key, value, precision):
    n, heads, kvheads, dim = len(query), 4, 2, 8
    scores = [[[0.0] * n for _ in range(n)] for _ in range(heads)]
    probabilities = [[[0.0] * n for _ in range(n)] for _ in range(heads)]
    result = [[0.0] * (heads * dim) for _ in range(n)]
    scale = precision(math.sqrt(precision(dim)))
    for head in range(heads):
        qs, ks = head * dim, (head // (heads // kvheads)) * dim
        for token in range(n):
            # Capture all scores, even future positions; only probabilities mask.
            for position in range(n):
                score = precision(dot(query[token][qs:qs + dim], key[position][ks:ks + dim], precision) / scale)
                scores[head][token][position] = f32(score)
            valid = scores[head][token][:token + 1]
            maximum = max(valid)
            exponential = [precision(math.exp(precision(precision(v) - precision(maximum)))) for v in valid]
            mass = precision(0)
            for v in exponential:
                mass = precision(mass + v)
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError('Invalid attention mass')
            for position, v in enumerate(exponential):
                probabilities[head][token][position] = f32(precision(v / mass))
            for feature in range(dim):
                result[token][qs + feature] = half(dot(
                    probabilities[head][token][:token + 1],
                    [value[position][ks + feature] for position in range(token + 1)], precision))
    return scores, probabilities, result


def forward(weights, config, tokens, offset, wide):
    precision = float if wide else f32
    captured = {}

    def save(name, values):
        values_flat = flatten(values)
        if not all(math.isfinite(v) for v in values_flat):
            raise ValueError('Nonfinite stage: ' + name)
        captured[name] = values_flat
        return values

    x = save('embedding', [[half(v) for v in weights['model.embed_tokens.weight'][token]] for token in tokens])
    epsilon = f32(config['rms_norm_eps'])
    for layer in range(2):
        p, stage = f'model.layers.{layer}', f'layers.{layer}.'
        normalized = save(stage + 'input_norm', norm(x, weights[p + '.input_layernorm.weight'], epsilon, precision))
        q = save(stage + 'q_projection', linear(normalized, weights[p + '.self_attn.q_proj.weight'], precision))
        k = save(stage + 'k_projection', linear(normalized, weights[p + '.self_attn.k_proj.weight'], precision))
        v = save(stage + 'v_projection', linear(normalized, weights[p + '.self_attn.v_proj.weight'], precision))
        q = save(stage + 'q_rotary', rotary(q, 4, offset, wide))
        k = save(stage + 'k_rotary', rotary(k, 2, offset, wide))
        scores, probabilities, value = attention(q, k, v, precision)
        save(stage + 'attention_scores', scores)
        save(stage + 'attention_probabilities', probabilities)
        value = save(stage + 'attention_context', value)
        projected = save(stage + 'attention_projection', linear(value, weights[p + '.self_attn.o_proj.weight'], precision))
        x = save(stage + 'attention_residual', [[half(precision(a + b)) for a, b in zip(row, delta, strict=True)]
                 for row, delta in zip(x, projected, strict=True)])
        normalized = save(stage + 'post_attention_norm', norm(x, weights[p + '.post_attention_layernorm.weight'], epsilon, precision))
        gate = save(stage + 'gate_projection', linear(normalized, weights[p + '.mlp.gate_proj.weight'], precision))
        up = save(stage + 'up_projection', linear(normalized, weights[p + '.mlp.up_proj.weight'], precision))
        active = []
        for row in gate:
            out = []
            for v in row:
                exponential = precision(math.exp(precision(-abs(v))))
                denominator = precision(precision(1) + exponential)
                sigmoid = precision(precision(1) / denominator) if v >= 0 else precision(exponential / denominator)
                out.append(half(precision(v * sigmoid)))
            active.append(out)
        save(stage + 'gate_silu', active)
        gated = save(stage + 'gated', [[half(precision(a * b)) for a, b in zip(row, other, strict=True)]
                     for row, other in zip(active, up, strict=True)])
        down = save(stage + 'down_projection', linear(gated, weights[p + '.mlp.down_proj.weight'], precision))
        x = save(stage + 'output', [[half(precision(a + b)) for a, b in zip(row, delta, strict=True)]
                 for row, delta in zip(x, down, strict=True)])
    final = save('final_norm', norm(x, weights['model.norm.weight'], epsilon, precision))
    head = weights['model.embed_tokens.weight'] if config['tie_word_embeddings'] else weights['lm_head.weight']
    save('logits', linear(final, head, precision, f32))
    return captured


def contracts(n):
    result = {'embedding': [n, 32], 'final_norm': [n, 32], 'logits': [n, 64]}
    layer = dict(input_norm=[n, 32], q_projection=[n, 4, 8], k_projection=[n, 2, 8],
                 v_projection=[n, 2, 8], q_rotary=[n, 4, 8], k_rotary=[n, 2, 8],
                 attention_scores=[4, n, n], attention_probabilities=[4, n, n],
                 attention_context=[n, 32], attention_projection=[n, 32], attention_residual=[n, 32],
                 post_attention_norm=[n, 32], gate_projection=[n, 64], up_projection=[n, 64],
                 gate_silu=[n, 64], gated=[n, 64], down_projection=[n, 32], output=[n, 32])
    if set(layer) != set(LAYER_STAGES):
        raise ValueError('Stage contract differs')
    for index in range(2):
        result.update({f'layers.{index}.{name}': shape for name, shape in layer.items()})
    return result


def export():
    inputs = {}
    manifest_path = source.SOURCE / 'manifest.json'
    inputs[str(manifest_path.relative_to(ROOT))] = pin(manifest_path)
    if inputs[str(manifest_path.relative_to(ROOT))]['sha256'] != source.SOURCE_SHA:
        raise ValueError('Frozen old source manifest differs')
    manifest = json.loads(manifest_path.read_bytes())
    for variant in ('tied', 'untied'):
        for leaf in ('config.json', 'model.safetensors'):
            key = variant + '/' + leaf
            actual = pin(source.SOURCE / key)
            if actual != manifest['files'][key]:
                raise ValueError('Frozen source file differs: ' + key)
            inputs[str((source.SOURCE / key).relative_to(ROOT))] = actual
    cases, templates, packed_owners = [], {}, {}
    for variant in ('tied', 'untied'):
        weights, config, packed = read_weights(variant, manifest)
        if (config['hidden_size'], config['num_hidden_layers'], config['num_attention_heads'],
            config['num_key_value_heads'], config['vocab_size']) != (32, 2, 4, 2, 64):
            raise ValueError('Tiny geometry differs')
        templates[variant] = dict(config, rope_scaling=source.SCALING, max_position_embeddings=512)
        packed_owners[variant] = packed
        for name, tokens, offset in CASES:
            main = forward(weights, config, tokens, offset, False)
            wide = forward(weights, config, tokens, offset, True)
            contract = contracts(len(tokens))
            if set(main) != set(contract) or set(wide) != set(contract) or len(contract) != 39:
                raise ValueError('Exact stage coverage differs')
            arrays = {}
            for stage, shape in contract.items():
                if len(main[stage]) != math.prod(shape) or len(wide[stage]) != math.prod(shape):
                    raise ValueError('Exact stage extents differ')
                dtype = 'f32le' if stage.endswith(('attention_scores', 'attention_probabilities', 'logits')) else 'f16le'
                envelope = max(abs(a - b) for a, b in zip(main[stage], wide[stage], strict=True))
                arrays[stage] = dict(shape=shape, dtype=dtype,
                    main_bits=[bits(v, dtype) for v in main[stage]],
                    wide_bits=[bits(v, dtype) for v in wide[stage]],
                    precision_envelope=envelope,
                    gate=dict(atol=max(POLICY['atol_floor'], POLICY['envelope_factor'] * envelope), rtol=POLICY['rtol']))
            cases.append(dict(variant=variant, name=name, tokens=tokens, position_offset=offset,
                              initial_kv_tokens=0, arrays=arrays))
    # Verify causal prefix in BOTH independent profiles before publication.
    for variant in ('tied', 'untied'):
        normal = next(c for c in cases if c['variant'] == variant and c['name'] == 'prompt')
        changed = next(c for c in cases if c['variant'] == variant and c['name'] == 'future_changed')
        for profile in ('main_bits', 'wide_bits'):
            if normal['arrays']['logits'][profile][:3 * 64] != changed['arrays']['logits'][profile][:3 * 64]:
                raise ValueError('Future token changed earlier logits')
    for name, before in inputs.items():
        if pin(ROOT / name) != before:
            raise ValueError('Source endpoint changed')
    producers = {str(path.relative_to(ROOT)): pin(path)
                 for path in (Path(__file__).resolve(), Path(source.__file__), Path(positional.__file__))}
    return dict(format=FORMAT, policy=POLICY, inputs=inputs, producers=producers,
                runtime=dict(python=sys.version, implementation=sys.implementation.name,
                             executable=str(Path(sys.executable).resolve())),
                scaling=source.SCALING, config_templates=templates, packed_owners=packed_owners,
                cases=cases, stage_count=312,
                scope='Tiny independent Half-storage envelopes; no exact Metal, upstream MLX, real-model, quality or device qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Complete independent graph first; an exception cannot publish partial JSON.
    data = (json.dumps(export(), indent=2, allow_nan=False) + '\n').encode()
    if len(data) > 8 * 1024**2:
        raise ValueError('Bounded fixture output exceeded')
    with args.output.open('xb') as stream:
        stream.write(data)
