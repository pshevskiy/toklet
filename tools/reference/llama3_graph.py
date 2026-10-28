"""Independent tiny scaled-Llama graphs; stdlib only, no target execution.

Reads the unchanged frozen tied/untied MLX fixture, reconstructs its tiny
weights, and computes an explicit mathematical dense graph. FP64 arithmetic
rounds to FP32 at operation/storage boundaries. RoPE frequencies use Decimal80.
The existing 1e-5+1e-4 relative FP32 gate is fixed before target outcomes.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama3_rotary as positional

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tests/fixtures/mlx-llama-v1'
SOURCE_SHA = '824d2ee2ae9dd5b623c5aa9db10c13b2e4dee32647ea5fa9d78274fb3529a1f5'
SCALING = dict(rope_type='llama3', factor=8, low_freq_factor=1,
               high_freq_factor=4, original_max_position_embeddings=64)


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def pin(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def matrices(variant, manifest):
    config = json.loads((SOURCE / variant / 'config.json').read_bytes())
    data = (SOURCE / variant / 'model.safetensors').read_bytes()
    size = struct.unpack('<Q', data[:8])[0]
    assert size < 1024 * 1024 and len(data) < 4 * 1024 * 1024
    header = json.loads(data[8:8 + size])
    arrays = {}
    for name, item in header.items():
        if name == '__metadata__':
            continue
        begin, end = item['data_offsets']
        raw = data[8 + size + begin:8 + size + end]
        kind = {'F16': 'e', 'F32': 'f', 'U32': 'I'}[item['dtype']]
        values = [v[0] for v in struct.iter_unpack('<' + kind, raw)]
        assert len(values) == math.prod(item['shape'])
        arrays[name] = (item['shape'], values)
    result = {}
    for name, (shape, values) in arrays.items():
        if not name.endswith('.weight'):
            continue
        module = name.removesuffix('.weight')
        if module + '.scales' in arrays:
            quantization = config['quantization'].get(module, config['quantization'])
            bits, group = quantization['bits'], quantization['group_size']
            rows = shape[0]
            columns = shape[1] * (32 // bits)
            scales = arrays[module + '.scales'][1]
            biases = arrays[module + '.biases'][1]
            decoded = []
            for row in range(rows):
                for col in range(columns):
                    word = values[row * shape[1] + col // (32 // bits)]
                    code = (word >> (col % (32 // bits) * bits)) & ((1 << bits) - 1)
                    meta = row * (columns // group) + col // group
                    decoded.append(f32(scales[meta] * code + biases[meta]))
            values = decoded
            shape = [rows, columns]
        result[name] = [values[i:i + shape[1]] for i in range(0, len(values), shape[1])] if len(shape) == 2 else values
    return result, config


def linear(x, w):
    return [[f32(sum(a * b for a, b in zip(row, weight))) for weight in w] for row in x]


def norm(x, w, epsilon):
    return [[f32(a / math.sqrt(sum(v * v for v in row) / len(row) + epsilon) * b)
             for a, b in zip(row, w)] for row in x]


def rotary(x, heads, offset, scaled):
    output = [list(row) for row in x]
    with localcontext() as context:
        context.prec = 80
        for pair in range(4):
            inverse = (Decimal(10000).ln() * Decimal(-2 * pair) / 8).exp()
            if scaled:
                inverse = positional.scaled_frequency(inverse, Decimal(8), Decimal(1), Decimal(4), Decimal(64))
            for token, row in enumerate(x):
                angle = float(Decimal(offset + token) * inverse)
                sine, cosine = math.sin(angle), math.cos(angle)
                for head in range(heads):
                    a = head * 8 + pair
                    b = a + 4
                    output[token][a] = f32(row[a] * cosine - row[b] * sine)
                    output[token][b] = f32(row[b] * cosine + row[a] * sine)
    return output


def forward(weights, config, offset, scaled):
    x = [list(weights['model.embed_tokens.weight'][token]) for token in [1, 7, 13, 3]]
    captures = {}
    epsilon = f32(config['rms_norm_eps'])
    for layer in range(2):
        p = f'model.layers.{layer}'
        normalized = norm(x, weights[p + '.input_layernorm.weight'], epsilon)
        q = rotary(linear(normalized, weights[p + '.self_attn.q_proj.weight']), 4, offset, scaled)
        k = rotary(linear(normalized, weights[p + '.self_attn.k_proj.weight']), 2, offset, scaled)
        v = linear(normalized, weights[p + '.self_attn.v_proj.weight'])
        captures[f'layer{layer}.q_rotary'] = sum(q, [])
        captures[f'layer{layer}.k_rotary'] = sum(k, [])
        context = [[None] * 32 for _ in x]
        for token in range(4):
            for head in range(4):
                q_start, kv_start = head * 8, head // 2 * 8
                scores = [f32(sum(a * b for a, b in zip(q[token][q_start:q_start + 8], k[position][kv_start:kv_start + 8])) / math.sqrt(8)) for position in range(token + 1)]
                mass = [math.exp(value - max(scores)) for value in scores]
                probabilities = [f32(value / sum(mass)) for value in mass]
                for feature in range(8):
                    context[token][q_start + feature] = f32(sum(probabilities[position] * v[position][kv_start + feature] for position in range(token + 1)))
        attention = linear(context, weights[p + '.self_attn.o_proj.weight'])
        x = [[f32(a + b) for a, b in zip(row, delta)] for row, delta in zip(x, attention)]
        normalized = norm(x, weights[p + '.post_attention_layernorm.weight'], epsilon)
        gate = linear(normalized, weights[p + '.mlp.gate_proj.weight'])
        up = linear(normalized, weights[p + '.mlp.up_proj.weight'])
        active = [[f32(a / (1 + math.exp(-a))) for a in row] for row in gate]
        gated = [[f32(a * b) for a, b in zip(row, other)] for row, other in zip(active, up)]
        down = linear(gated, weights[p + '.mlp.down_proj.weight'])
        x = [[f32(a + b) for a, b in zip(row, delta)] for row, delta in zip(x, down)]
    final = norm(x, weights['model.norm.weight'], epsilon)
    head = weights['model.embed_tokens.weight'] if config['tie_word_embeddings'] else weights['lm_head.weight']
    captures['logits'] = sum(linear(final, head), [])
    assert all(math.isfinite(value) for values in captures.values() for value in values)
    return captures


def export():
    assert pin(SOURCE / 'manifest.json')['sha256'] == SOURCE_SHA
    manifest = json.loads((SOURCE / 'manifest.json').read_bytes())
    inputs = {str((SOURCE / 'manifest.json').relative_to(ROOT)): pin(SOURCE / 'manifest.json')}
    for variant in ['tied', 'untied']:
        for leaf in ['config.json', 'model.safetensors']:
            name = variant + '/' + leaf
            actual = pin(SOURCE / name)
            assert actual == manifest['files'][name]
            inputs[str((SOURCE / name).relative_to(ROOT))] = actual
    cases = []
    for variant in ['tied', 'untied']:
        weights, config = matrices(variant, manifest)
        for offset in [0, 5]:
            captures = forward(weights, config, offset, True)
            ordinary = forward(weights, config, offset, False)
            difference = max(abs(a - b) for a, b in zip(captures['logits'], ordinary['logits']))
            assert difference > 1e-5
            cases.append(dict(variant=variant, offset=offset, tokens=[1, 7, 13, 3],
                              arrays=captures, ordinary_logit_difference=difference))
    return dict(format='toklet-llama3-tiny-graph-v1', scaling=SCALING,
                producer=pin(Path(__file__).resolve()), positional_producer=pin(Path(positional.__file__)),
                inputs=inputs, atol=1e-5, rtol=1e-4, cases=cases,
                scope='Independent tiny scalar wiring only; no MLX/native GPU or real-model qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with args.output.open('xb') as stream:
        stream.write((json.dumps(export(), indent=2, allow_nan=False) + '\n').encode())
