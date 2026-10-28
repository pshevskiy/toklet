"""Independent tiny Gemma3 CPU mathematics and original packed BF16 fixture.

Stdlib only. No target, Gemma4 forward, MLX, Torch or Half graph is imported.
FP64 mathematical reductions cross explicit FP32 operation boundaries. This is
an independent reference, not an assertion of scalar or upstream bit emulation.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

FORMAT = 'toklet-gemma3-cpu-tiny-graph-v1'
PROTOCOL_SHA256 = '38d183309748bde1652ef6a5a088960a10ff8b3d143dc9847d09ccf0a15a4156'
CPU_GATE = dict(atol=1e-5, rtol=1e-4)
WRAP = (1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10)
CASES = (('single', (1,)), ('short', (1, 7, 13, 3)),
         ('wrap12', WRAP), ('future_changed12', WRAP[:-1] + (19,)))
MAX_ARTIFACT = 2 * 1024**2
MAX_ARRAY_BYTES = 4 * 1024**2
MAX_MANIFEST = 1024**2


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16_bits(x):
    if not math.isfinite(x):
        raise ValueError('BF16 input must be finite')
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    result = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    if result & 0x7f80 == 0x7f80:
        raise ValueError('BF16 overflow')
    return result


def bf16_value(bits):
    value = struct.unpack('<f', struct.pack('<I', bits << 16))[0]
    if not math.isfinite(value):
        raise ValueError('Nonfinite BF16 source')
    return value


def embedding_scale(hidden):
    if type(hidden) is not int or hidden <= 0:
        raise ValueError('Embedding width')
    return bf16_value(bf16_bits(math.sqrt(hidden)))


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def configuration():
    return dict(model_type='gemma3_text', architectures=['Gemma3ForCausalLM'],
        hidden_size=64, intermediate_size=128, vocab_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=1, head_dim=32,
        sliding_window=4, _sliding_window_pattern=2, max_position_embeddings=12,
        layer_types=['sliding_attention', 'full_attention'],
        rope_local_base_freq=10000.0, rope_theta=1000000.0, rope_scaling=None,
        rms_norm_eps=1e-6, query_pre_attn_scalar=32, hidden_activation='gelu_pytorch_tanh',
        tie_word_embeddings=True, attention_bias=False, attention_dropout=0.0,
        torch_dtype='bfloat16', quantization=dict(bits=4, group_size=64))


def shapes():
    result = {'model.embed_tokens.weight': [64, 64], 'model.norm.weight': [64]}
    for layer in range(2):
        p = f'model.layers.{layer}'
        for name in ('input_layernorm', 'post_attention_layernorm',
                     'pre_feedforward_layernorm', 'post_feedforward_layernorm'):
            result[p + '.' + name + '.weight'] = [64]
        for name in ('q_norm', 'k_norm'):
            result[p + '.self_attn.' + name + '.weight'] = [32]
        for name, rows, columns in [('q_proj', 128, 64), ('k_proj', 32, 64),
                                    ('v_proj', 32, 64), ('o_proj', 64, 128)]:
            result[p + '.self_attn.' + name + '.weight'] = [rows, columns]
        for name, rows, columns in [('gate_proj', 128, 64), ('up_proj', 128, 64),
                                    ('down_proj', 64, 128)]:
            result[p + '.mlp.' + name + '.weight'] = [rows, columns]
    return result


def packed_matrix(rows, columns, seed):
    if rows <= 0 or columns <= 0 or columns % 64:
        raise ValueError('Packed group64 shape')
    codes, scales, biases = bytearray(), bytearray(), bytearray()
    for row in range(rows):
        for column in range(0, columns, 8):
            word = sum(((seed + 11 * row + 7 * (column + lane)
                + 3 * ((column + lane) // 8)) % 16) << (4 * lane) for lane in range(8))
            codes.extend(struct.pack('<I', word))
        for group in range(columns // 64):
            scale = bf16_value(0x3b00 + (seed + 3 * row + 5 * group) % 97)
            bias = -7.5 * scale + ((row + group + seed) % 7 - 3) / 4096
            scales.extend(struct.pack('<H', bf16_bits(scale)))
            biases.extend(struct.pack('<H', bf16_bits(bias)))
    return bytes(codes), bytes(scales), bytes(biases)


def make_artifact():
    tensors = {}
    for name, shape in shapes().items():
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
        if len(shape) == 1:
            gamma = [((seed + 7 * i) % 33 - 16) / 64 for i in range(shape[0])]
            tensors[name] = ('BF16', shape, b''.join(struct.pack('<H', bf16_bits(v)) for v in gamma))
        else:
            rows, columns = shape
            codes, scales, biases = packed_matrix(rows, columns, seed)
            base = name.removesuffix('.weight')
            tensors[name] = ('U32', [rows, columns // 8], codes)
            tensors[base + '.scales'] = ('BF16', [rows, columns // 64], scales)
            tensors[base + '.biases'] = ('BF16', [rows, columns // 64], biases)
    header, payload = {}, bytearray()
    for name, (dtype, shape, raw) in sorted(tensors.items()):
        begin = len(payload)
        payload.extend(raw)
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[begin, len(payload)])
    raw = bytearray(json.dumps(header, sort_keys=True, separators=(',', ':')).encode())
    while (8 + len(raw)) % 4 != 2:
        raw.append(32)
    result = struct.pack('<Q', len(raw)) + raw + payload
    if len(result) > MAX_ARTIFACT:
        raise ValueError('Artifact bound')
    return bytes(result)


def decode(raw):
    if not 8 < len(raw) <= MAX_ARTIFACT:
        raise ValueError('Artifact length')
    size = struct.unpack('<Q', raw[:8])[0]
    if not 0 < size <= 65536 or 8 + size > len(raw):
        raise ValueError('Header length')
    header = strict_json(raw[8:8 + size])
    expected = {}
    for name, shape in shapes().items():
        if len(shape) == 1:
            expected[name] = ('BF16', shape)
        else:
            rows, columns = shape
            expected[name] = ('U32', [rows, columns // 8])
            for suffix in ('scales', 'biases'):
                expected[name.removesuffix('.weight') + '.' + suffix] = ('BF16', [rows, columns // 64])
    if not isinstance(header, dict) or set(header) != set(expected):
        raise ValueError('Exact original inventory')
    arrays, cursor = {}, 0
    for name, spec in sorted(header.items(), key=lambda item: item[1]['data_offsets'][0]):
        dtype, shape = expected[name]
        if set(spec) != {'dtype', 'shape', 'data_offsets'} or spec['dtype'] != dtype or spec['shape'] != shape:
            raise ValueError('Tensor metadata')
        if any(type(v) is not int for v in spec['shape'] + spec['data_offsets']):
            raise ValueError('Integer shape/extents required')
        begin, end = spec['data_offsets']
        width = 4 if dtype == 'U32' else 2
        if begin != cursor or end - begin != math.prod(shape) * width:
            raise ValueError('Contiguous exact tensor extent')
        data = raw[8 + size + begin:8 + size + end]
        if len(data) != end - begin:
            raise ValueError('Truncated tensor')
        arrays[name] = data
        cursor = end
    if 8 + size + cursor != len(raw):
        raise ValueError('Final payload extent')
    matrices, gamma, effective = {}, {}, {}
    for name, shape in shapes().items():
        if len(shape) == 1:
            gamma[name] = [bf16_value(v[0]) for v in struct.iter_unpack('<H', arrays[name])]
            effective[name] = [f32(1.0 + v) for v in gamma[name]]
            continue
        rows, columns = shape
        base = name.removesuffix('.weight')
        words = [v[0] for v in struct.iter_unpack('<I', arrays[name])]
        scales = [bf16_value(v[0]) for v in struct.iter_unpack('<H', arrays[base + '.scales'])]
        biases = [bf16_value(v[0]) for v in struct.iter_unpack('<H', arrays[base + '.biases'])]
        matrix = []
        for row in range(rows):
            values = []
            for column in range(columns):
                code = (words[row * (columns // 8) + column // 8] >> (4 * (column % 8))) & 15
                group = row * (columns // 64) + column // 64
                values.append(f32(f32(scales[group] * code) + biases[group]))
            matrix.append(values)
        matrices[name] = matrix
    if any(not math.isfinite(v) for matrix in matrices.values() for row in matrix for v in row):
        raise ValueError('Nonfinite reconstructed matrix')
    return matrices, gamma, effective, 8 + size


def linear(x, matrix):
    return [[f32(math.fsum(a * b for a, b in zip(row, weight, strict=True)))
             for weight in matrix] for row in x]


def norm(x, weights):
    result = []
    for row in x:
        inverse = 1 / math.sqrt(math.fsum(v * v for v in row) / len(row) + f32(1e-6))
        result.append([f32(v * inverse * w) for v, w in zip(row, weights, strict=True)])
    return result


def head_norm(x, weights, heads):
    dim = len(weights)
    if any(len(row) != heads * dim for row in x):
        raise ValueError('Head norm width')
    return [[v for head in range(heads) for v in norm([row[head * dim:(head + 1) * dim]], weights)[0]] for row in x]


def rotary(x, heads, dim, theta):
    if heads <= 0 or dim <= 0 or dim % 2 or theta <= 0 or any(len(row) != heads * dim for row in x):
        raise ValueError('Rotary shape')
    result = [row[:] for row in x]
    with localcontext() as context:
        context.prec = 80
        inverses = [(Decimal(theta).ln() * Decimal(-2 * pair) / dim).exp() for pair in range(dim // 2)]
        for token, row in enumerate(x):
            for pair, inverse in enumerate(inverses):
                angle = float(Decimal(token) * inverse)
                cosine, sine = math.cos(angle), math.sin(angle)
                for head in range(heads):
                    a, b = head * dim + pair, head * dim + pair + dim // 2
                    result[token][a] = f32(row[a] * cosine - row[b] * sine)
                    result[token][b] = f32(row[b] * cosine + row[a] * sine)
    return result


def visible(query, key, window):
    return key <= query and (window is None or query - key < window)


def attention(q, k, v, heads, kvheads, dim, window, scale, query_start=0):
    n, keys = len(q), len(k)
    if not n or not keys or len(v) != keys or heads <= 0 or kvheads <= 0 or heads % kvheads or dim <= 0:
        raise ValueError('Attention dimensions')
    if window is not None and (type(window) is not int or window <= 0):
        raise ValueError('Attention window')
    if query_start < 0 or query_start + n > keys or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Attention positions/scale')
    if any(len(row) != heads * dim for row in q) or any(len(row) != kvheads * dim for row in k + v):
        raise ValueError('Attention widths')
    scores, probabilities = [], []
    output = [[0.0] * (heads * dim) for _ in q]
    for head in range(heads):
        qs, ks = head * dim, (head // (heads // kvheads)) * dim
        head_scores, head_probs = [], []
        for t, row in enumerate(q):
            values = [f32(math.fsum(a * b for a, b in zip(row[qs:qs + dim], key[ks:ks + dim], strict=True)) * scale) for key in k]
            included = [i for i in range(keys) if visible(query_start + t, i, window)]
            maximum = max(values[i] for i in included)
            exp = [math.exp(values[i] - maximum) for i in included]
            mass = math.fsum(exp)
            probs = [0.0] * keys
            for i, e in zip(included, exp, strict=True):
                probs[i] = f32(e / mass)
            for feature in range(dim):
                output[t][qs + feature] = f32(math.fsum(probs[i] * v[i][ks + feature] for i in included))
            head_scores.append(values)
            head_probs.append(probs)
        scores.append(head_scores)
        probabilities.append(head_probs)
    return scores, probabilities, output


def flatten(x):
    return [v for item in x for v in flatten(item)] if isinstance(x, (list, tuple)) else [x]


def residual(a, b):
    return [[f32(x + y) for x, y in zip(row, delta, strict=True)] for row, delta in zip(a, b, strict=True)]


def forward(matrices, effective, tokens):
    if not tokens or len(tokens) > 12 or any(type(t) is not int or not 0 <= t < 64 for t in tokens):
        raise ValueError('Fixed tiny request')
    captured = {}
    def save(name, values):
        raw = flatten(values)
        if any(not math.isfinite(v) for v in raw):
            raise ValueError('Nonfinite stage: ' + name)
        captured[name] = raw
        return values
    table = matrices['model.embed_tokens.weight']
    x = save('embedding_lookup', [table[t][:] for t in tokens])
    x = save('embedding', [[f32(v * embedding_scale(64)) for v in row] for row in x])
    for layer in range(2):
        p, s = f'model.layers.{layer}', f'layers.{layer}.'
        xn = save(s + 'input_norm', norm(x, effective[p + '.input_layernorm.weight']))
        q = save(s + 'q_projection', linear(xn, matrices[p + '.self_attn.q_proj.weight']))
        k = save(s + 'k_projection', linear(xn, matrices[p + '.self_attn.k_proj.weight']))
        v = save(s + 'v_projection', linear(xn, matrices[p + '.self_attn.v_proj.weight']))
        q = save(s + 'q_norm', head_norm(q, effective[p + '.self_attn.q_norm.weight'], 4))
        k = save(s + 'k_norm', head_norm(k, effective[p + '.self_attn.k_norm.weight'], 1))
        theta = 10000 if layer == 0 else 1000000
        q = save(s + 'q_rotary', rotary(q, 4, 32, theta))
        k = save(s + 'k_rotary', rotary(k, 1, 32, theta))
        scores, probs, context = attention(q, k, v, 4, 1, 32, 4 if layer == 0 else None, f32(1 / math.sqrt(32)))
        save(s + 'attention_scores', scores)
        save(s + 'attention_probabilities', probs)
        save(s + 'attention_context', context)
        projected = save(s + 'attention_projection', linear(context, matrices[p + '.self_attn.o_proj.weight']))
        projected = save(s + 'post_attention_norm', norm(projected, effective[p + '.post_attention_layernorm.weight']))
        x = save(s + 'attention_residual', residual(x, projected))
        xn = save(s + 'pre_ff_norm', norm(x, effective[p + '.pre_feedforward_layernorm.weight']))
        gate = save(s + 'gate_projection', linear(xn, matrices[p + '.mlp.gate_proj.weight']))
        up = save(s + 'up_projection', linear(xn, matrices[p + '.mlp.up_proj.weight']))
        active = save(s + 'gate_gelu', [[f32(.5 * z * (1 + math.tanh(math.sqrt(2 / math.pi) * (z + .044715 * z * z * z)))) for z in row] for row in gate])
        gated = save(s + 'gated', [[f32(a * b) for a, b in zip(row, other, strict=True)] for row, other in zip(active, up, strict=True)])
        down = save(s + 'down_projection', linear(gated, matrices[p + '.mlp.down_proj.weight']))
        down = save(s + 'post_ff_norm', norm(down, effective[p + '.post_feedforward_layernorm.weight']))
        x = save(s + 'output', residual(x, down))
    final = save('final_norm', norm(x, effective['model.norm.weight']))
    save('logits', linear(final, table))
    return captured


def contracts(n):
    result = {name: [n, 64] for name in ('embedding_lookup', 'embedding', 'final_norm', 'logits')}
    layer = dict(input_norm=[n, 64], q_projection=[n, 4, 32], k_projection=[n, 1, 32], v_projection=[n, 1, 32],
        q_norm=[n, 4, 32], k_norm=[n, 1, 32], q_rotary=[n, 4, 32], k_rotary=[n, 1, 32],
        attention_scores=[4, n, n], attention_probabilities=[4, n, n], attention_context=[n, 128],
        attention_projection=[n, 64], post_attention_norm=[n, 64], attention_residual=[n, 64],
        pre_ff_norm=[n, 64], gate_projection=[n, 128], up_projection=[n, 128], gate_gelu=[n, 128],
        gated=[n, 128], down_projection=[n, 64], post_ff_norm=[n, 64], output=[n, 64])
    for i in range(2):
        result.update({f'layers.{i}.{name}': shape for name, shape in layer.items()})
    return result


def components():
    gamma_bits = [0xbf80, 0x0000, 0x3f80, 0xbf7f, 0xbf81]
    gamma = [bf16_value(b) for b in gamma_bits]
    effective = [f32(1 + v) for v in gamma]
    # Uniform Q/K makes visibility and values independently hand-computable.
    q, k, v = [[0.0] * 256], [[0.0] * 256 for _ in range(513)], [[0.0] * 256 for _ in range(513)]
    for index, value in ((0, 8.0), (1, 4.0), (512, 2.0)):
        v[index][0] = value
    local = attention(q, k, v, 1, 1, 256, 512, 1 / 16, 512)
    global_ = attention(q, k, v, 1, 1, 256, None, 1 / 16, 512)
    return dict(gamma=dict(raw_bf16_bits=gamma_bits, widened=gamma, effective_f32=effective,
                          input=[1., 2., 3., 4., 5.], normalized=norm([[1., 2., 3., 4., 5.]], effective)[0]),
        embedding=dict(hidden=640, multiplier=embedding_scale(640), bf16_bits=bf16_bits(math.sqrt(640))),
        visibility=dict(dim=256, window=512, query=512, local_included=list(range(1, 513)),
            global_included=list(range(513)), local_probabilities=local[1][0][0], global_probabilities=global_[1][0][0],
            local_context=local[2][0], global_context=global_[2][0],
            expected_local_first=f32(6 / 512),
            expected_global_first=f32(f32(1 / 513) * 14)))


def export(directory):
    before = identity(Path(__file__).read_bytes())
    directory.mkdir(parents=False, exist_ok=False)
    try:
        files = {}
        artifact = make_artifact()
        matrices, gamma, effective, start = decode(artifact)
        for name, raw in [('config.json', encoded(configuration())), ('model.safetensors', artifact),
                          ('components.json', encoded(components()))]:
            (directory / name).write_bytes(raw)
            files[name] = identity(raw)
        cases, prefix = [], None
        with (directory / 'arrays.bin').open('xb') as binary, (directory / 'progress.jsonl').open('x') as progress:
            for name, tokens in CASES:
                progress.write(json.dumps(dict(case=name, started=True, arrays_end=binary.tell())) + '\n')
                progress.flush()
                arrays, descriptors = forward(matrices, effective, tokens), {}
                expected = contracts(len(tokens))
                if set(arrays) != set(expected) or len(arrays) != 48:
                    raise ValueError('Stage inventory')
                for stage, shape in expected.items():
                    values = arrays[stage]
                    if len(values) != math.prod(shape):
                        raise ValueError('Stage shape')
                    raw = b''.join(struct.pack('<f', value) for value in values)
                    begin = binary.tell()
                    if begin + len(raw) > MAX_ARRAY_BYTES:
                        raise ValueError('Array byte bound')
                    binary.write(raw)
                    descriptors[stage] = dict(file='arrays.bin', offset=begin, shape=shape, dtype='f32le', **identity(raw))
                if name == 'wrap12':
                    prefix = arrays['logits'][:11 * 64]
                if name == 'future_changed12' and arrays['logits'][:11 * 64] != prefix:
                    raise ValueError('Source causal prefix')
                cases.append(dict(name=name, tokens=list(tokens), position_offset=0, initial_kv_tokens=0, arrays=descriptors))
                binary.flush()
                progress.write(json.dumps(dict(case=name, complete=True, arrays_end=binary.tell())) + '\n')
                progress.flush()
        files['arrays.bin'] = identity((directory / 'arrays.bin').read_bytes())
        for name, pin in files.items():
            if identity((directory / name).read_bytes()) != pin:
                raise ValueError('Output endpoint changed')
        if identity(Path(__file__).read_bytes()) != before:
            raise ValueError('Producer endpoint changed')
        manifest = dict(format=FORMAT, producer=before, protocol_sha256=PROTOCOL_SHA256, cpu_gate=CPU_GATE,
            source_cpu_graph_executed=True, target_graph_executed=False,
            files=files, cases=cases, stage_groups=192, stored_arrays=192,
            profile='Independent FP64 mathematical reductions with FP32 operation stores; not exact scalar/upstream emulation',
            source=dict(matrix_owners=len(matrices), raw_gamma_vectors=len(gamma), effective_norm_elements=sum(map(len, effective.values())),
                tied_head='model.embed_tokens.weight', physical_lm_head=False, payload_start=start,
                gamma_semantics='Raw BF16 widened then FP32(1+gamma) once; not BF16 addition or Half profile'),
            runtime=dict(python=sys.version, executable=str(Path(sys.executable).resolve())),
            scope='Tiny CPU only; no real model, Half, GPU, quality, app or performance qualification')
        raw = encoded(manifest)
        if len(raw) > MAX_MANIFEST:
            raise ValueError('Manifest bound')
        (directory / 'manifest.json').write_bytes(raw)
        return identity(raw)
    except BaseException as error:
        (directory / 'failure.json').write_bytes(encoded(dict(complete=False, error=type(error).__name__ + ': ' + str(error))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir)))
