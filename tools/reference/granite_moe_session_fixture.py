"""Independent two-layer Granite MoE session with uniform causal attention."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from moe_experts_fixture import dot, f32, silu

ROOT = Path(__file__).resolve().parents[2]
HIDDEN, INTERMEDIATE, VOCAB = 32, 8, 32
TOKENS = [0, 1, 0, 1, 1]


def half(value):
    return struct.unpack('<e', struct.pack('<e', value))[0]


def norm(row):
    divisor = math.sqrt(math.fsum(x * x for x in row) / HIDDEN + 1e-6)
    return [half(x / divisor) for x in row]


def residual(row, delta):
    return [half(x + half(y * half(0.22))) for x, y in zip(row, delta, strict=True)]


def diagonal(rows, columns, scale):
    return [scale if i == j else 0 for i in range(rows) for j in range(columns)]


def export(directory):
    config = json.loads((ROOT / 'tests/fixtures/granite-moe-source-v1/real-config.json').read_text())
    config.update(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, vocab_size=VOCAB,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                  num_local_experts=2, num_experts_per_tok=1, max_position_embeddings=64)
    embeddings = [[1] * HIDDEN, [-1] * HIDDEN, [8192] * HIDDEN] + [[0] * HIDDEN for _ in range(VOCAB - 3)]
    tensors = {
        'model.embed_tokens.weight': ([VOCAB, HIDDEN], sum(embeddings, [])),
        'model.norm.weight': ([HIDDEN], [1] * HIDDEN),
    }
    for layer in range(2):
        fused = [0] * (2 * 2 * INTERMEDIATE * HIDDEN)
        for expert, sign in enumerate([1, -1]):
            base = expert * 2 * INTERMEDIATE * HIDDEN
            fused[base] = 2 * sign
            fused[base + INTERMEDIATE * HIDDEN] = sign
        down = [0] * (2 * HIDDEN * INTERMEDIATE)
        down[0], down[HIDDEN * INTERMEDIATE] = 1, 2
        for name, shape, data in [
            ('input_layernorm.weight', [HIDDEN], [1] * HIDDEN),
            ('post_attention_layernorm.weight', [HIDDEN], [1] * HIDDEN),
            ('self_attn.q_proj.weight', [HIDDEN, HIDDEN], [0] * (HIDDEN * HIDDEN)),
            ('self_attn.k_proj.weight', [16, HIDDEN], [0] * (16 * HIDDEN)),
            ('self_attn.v_proj.weight', [16, HIDDEN], diagonal(16, HIDDEN, 1)),
            ('self_attn.o_proj.weight', [HIDDEN, HIDDEN], diagonal(HIDDEN, HIDDEN, 0.25)),
            ('block_sparse_moe.router.layer.weight', [2, HIDDEN], [1] + [0] * 31 + [-1] + [0] * 31),
            ('block_sparse_moe.input_linear.weight', [2, 2 * INTERMEDIATE, HIDDEN], fused),
            ('block_sparse_moe.output_linear.weight', [2, HIDDEN, INTERMEDIATE], down),
        ]:
            tensors[f'model.layers.{layer}.{name}'] = (shape, data)
    header, payload = {}, bytearray()
    for name, (shape, values) in sorted(tensors.items()):
        start = len(payload)
        for value in values:
            word, = struct.unpack('<I', struct.pack('<f', value))
            assert word & 0xffff == 0
            payload.extend(struct.pack('<H', word >> 16))
        header[name] = dict(dtype='BF16', shape=shape, data_offsets=[start, len(payload)])
    raw_header = json.dumps(header, separators=(',', ':')).encode()
    raw_header += b' ' * (-len(raw_header) % 8)
    model = struct.pack('<Q', len(raw_header)) + raw_header + payload
    config_bytes = (json.dumps(config, indent=2) + '\n').encode()
    rows = [[12 * v for v in embeddings[token]] for token in TOKENS]
    routes = []
    for _ in range(2):
        values = [norm(row)[:16] for row in rows]
        next_rows, selected = [], []
        for i, row in enumerate(rows):
            probability = f32(1 / (i + 1))
            # Both query heads see the same KV head. Zero Q and K yield
            # uniform causal attention at every rotary offset.
            context = [half(dot(list(column), [probability] * (i + 1)))
                       for column in zip(*values[:i + 1], strict=True)]
            attended = residual(row, [half(v * 0.25) for v in context] * 2)
            x = norm(attended)[0]
            expert = int(x < 0)
            selected.append(expert)
            gate = half(2 * abs(x))
            activated = half(half(silu(gate)) * abs(x))
            delta = [half((expert + 1) * activated)] + [0] * (HIDDEN - 1)
            next_rows.append(residual(attended, delta))
        routes.append(selected)
        rows = next_rows
    expected = [[f32(dot(norm(row), table) / 6) for table in embeddings] for row in rows]
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in [('config.json', config_bytes), ('model.safetensors', model)]:
        (directory / name).write_bytes(data)
    (directory / 'oracle.json').write_text(json.dumps(dict(tokens=TOKENS, routes=routes, logits=expected,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    export(Path(sys.argv[1]))
