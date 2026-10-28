"""Independent tiny fused-source Phi3 graph fixtures, stdlib only.

CPU reference uses ordered FP64 math with FP32 storage; it is not exact scalar
engine emulation. Half main uses ordered FP32 math, wide uses ordered FP64;
all activation stores are Half except scores/probabilities/logits (FP32).
Ordinary split-half head96 RoPE is implemented here independently. No Toklet,
MLX, NumPy, model download or old fixture producer is imported/executed.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

FORMAT = 'toklet-phi3-tiny-graph-v1'
HALF_POLICY = dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3)
CPU_GATE = dict(atol=1e-5, rtol=1e-4)
PROMPT = (1, 7, 13, 3, 5, 11, 17, 23, 2, 19, 29, 31, 37, 41, 43, 47, 53)
CASES = (
    ('untied', 'prompt', PROMPT, 0),
    ('untied', 'future_changed', PROMPT[:-1] + (59,), 0),
    ('untied', 'offset5', PROMPT, 5),
    ('untied', 'offset495', PROMPT, 495),
    ('tied', 'tied_control', (1, 7, 13), 0),
)
PROFILES = ('cpu', 'main', 'wide')
MAX_ARTIFACT = 2 * 1024**2
MAX_ARRAY_BYTES = 16 * 1024**2
MAX_MANIFEST_BYTES = 1024**2
LAYER_STAGES = (
    'input_norm', 'q_projection', 'k_projection', 'v_projection', 'q_rotary', 'k_rotary',
    'attention_scores', 'attention_probabilities', 'attention_context', 'attention_projection',
    'attention_residual', 'post_attention_norm', 'gate_projection', 'up_projection',
    'gate_silu', 'gated', 'down_projection', 'output',
)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def half(value):
    return struct.unpack('<e', struct.pack('<e', value))[0]


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def configuration(tied):
    return dict(model_type='phi3', architectures=['Phi3ForCausalLM'], hidden_act='silu',
        hidden_size=192, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=4096, original_max_position_embeddings=4096,
        sliding_window=2047, rms_norm_eps=1e-5, rope_theta=10000.0,
        rope_scaling=None, partial_rotary_factor=1.0, tie_word_embeddings=tied,
        attention_bias=False, attention_dropout=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
        torch_dtype='float16', quantization=dict(bits=4, group_size=64))


def physical_shapes(tied):
    result = {'model.embed_tokens.weight': [64, 192], 'model.norm.weight': [192]}
    if not tied:
        result['lm_head.weight'] = [64, 192]
    for layer in range(2):
        p = f'model.layers.{layer}'
        result.update({p + '.input_layernorm.weight': [192],
            p + '.post_attention_layernorm.weight': [192],
            p + '.self_attn.qkv_proj.weight': [576, 192],
            p + '.self_attn.o_proj.weight': [192, 192],
            p + '.mlp.gate_up_proj.weight': [512, 192],
            p + '.mlp.down_proj.weight': [192, 256]})
    return result


def packed_matrix(rows, columns, seed):
    """Literal MLX4 little-endian words; coefficient groups follow input columns."""
    if rows <= 0 or columns <= 0 or columns % 64:
        raise ValueError('Positive group64 dimensions required')
    codes, scales, biases = bytearray(), bytearray(), bytearray()
    for row in range(rows):
        for column in range(0, columns, 8):
            word = 0
            for lane in range(8):
                c = column + lane
                code = (seed + 11 * row + 7 * c + 3 * (c // 8) + 5 * (c // 64)) % 16
                word |= code << (4 * lane)
            codes.extend(struct.pack('<I', word))
        for group in range(columns // 64):
            scale_bits = 0x1801 + 2 * ((seed + 3 * row + 5 * group) % 127)
            scale = struct.unpack('<e', struct.pack('<H', scale_bits))[0]
            bias = half(-7.5 * scale + ((row + group + seed) % 7 - 3) / 65536)
            scales.extend(struct.pack('<e', scale))
            biases.extend(struct.pack('<e', bias))
    return bytes(codes), bytes(scales), bytes(biases)


def make_artifact(tied):
    tensors = {}
    # Name-derived deterministic seeds keep all shared weights identical across
    # tied/untied artifacts. The untied head is a separate original owner.
    for name, shape in physical_shapes(tied).items():
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
        if len(shape) == 1:
            raw = b''.join(struct.pack('<e', 1 + ((seed + 7 * i) % 17 - 8) / 512) for i in range(shape[0]))
            tensors[name] = ('F16', shape, raw)
        else:
            rows, columns = shape
            codes, scales, biases = packed_matrix(rows, columns, seed)
            module = name.removesuffix('.weight')
            tensors[name] = ('U32', [rows, columns // 8], codes)
            tensors[module + '.scales'] = ('F16', [rows, columns // 64], scales)
            tensors[module + '.biases'] = ('F16', [rows, columns // 64], biases)
    header, payload = {}, bytearray()
    for name, (dtype, shape, raw) in sorted(tensors.items()):
        begin = len(payload)
        payload.extend(raw)
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[begin, len(payload)])
    raw_header = bytearray(json.dumps(header, sort_keys=True, separators=(',', ':')).encode())
    while (8 + len(raw_header)) % 4 != 1:
        raw_header.append(32)
    result = struct.pack('<Q', len(raw_header)) + raw_header + payload
    if len(result) > MAX_ARTIFACT:
        raise ValueError('Tiny source exceeds artifact bound')
    return bytes(result)


def decode_physical(raw, tied):
    """Independent scalar decode of ORIGINAL fused matrices; no Rust map input."""
    if len(raw) < 8 or len(raw) > MAX_ARTIFACT:
        raise ValueError('Artifact extent invalid')
    header_size = struct.unpack('<Q', raw[:8])[0]
    if not 0 < header_size <= 64 * 1024 or 8 + header_size > len(raw):
        raise ValueError('Header extent invalid')
    header = json.loads(raw[8:8 + header_size])
    offset, arrays = 0, {}
    expected = physical_shapes(tied)
    expected_names = set()
    for name, shape in expected.items():
        expected_names.add(name)
        if len(shape) == 2:
            module = name.removesuffix('.weight')
            expected_names.update((module + '.scales', module + '.biases'))
    if set(header) != expected_names:
        raise ValueError('Exact physical inventory differs')
    for name, spec in sorted(header.items(), key=lambda item: item[1]['data_offsets'][0]):
        begin, end = spec['data_offsets']
        width = {'F16': 2, 'U32': 4}.get(spec['dtype'])
        if width is None or begin != offset or end <= begin or end - begin != width * math.prod(spec['shape']):
            raise ValueError('Tensor shape/extent differs')
        data = raw[8 + header_size + begin:8 + header_size + end]
        if len(data) != end - begin:
            raise ValueError('Truncated selected payload')
        arrays[name] = (spec['shape'], spec['dtype'], data)
        offset = end
    if 8 + header_size + offset != len(raw):
        raise ValueError('Artifact final extent differs')
    weights = {}
    for name, shape in expected.items():
        actual_shape, dtype, data = arrays[name]
        if len(shape) == 1:
            if dtype != 'F16' or actual_shape != shape:
                raise ValueError('Norm dtype/shape differs')
            weights[name] = [v[0] for v in struct.iter_unpack('<e', data)]
            continue
        rows, columns = shape
        module = name.removesuffix('.weight')
        sc_shape, sc_dtype, sc_raw = arrays[module + '.scales']
        bi_shape, bi_dtype, bi_raw = arrays[module + '.biases']
        if (dtype, actual_shape, sc_dtype, bi_dtype, sc_shape, bi_shape) != (
                'U32', [rows, columns // 8], 'F16', 'F16', [rows, columns // 64], [rows, columns // 64]):
            raise ValueError('Affine shape/dtype differs')
        words = [v[0] for v in struct.iter_unpack('<I', data)]
        scales = [v[0] for v in struct.iter_unpack('<e', sc_raw)]
        biases = [v[0] for v in struct.iter_unpack('<e', bi_raw)]
        matrix = []
        for row in range(rows):
            values = []
            for column in range(columns):
                word = words[row * (columns // 8) + column // 8]
                code = (word >> (4 * (column % 8))) & 15
                group = row * (columns // 64) + column // 64
                values.append(f32(f32(scales[group] * code) + biases[group]))
            matrix.append(values)
        weights[name] = matrix
    if any(not math.isfinite(v) for array in weights.values() for v in flatten(array)):
        raise ValueError('Nonfinite reconstructed source')
    return weights, 8 + header_size


def split_original(physical):
    """Literal source graph slices, independent of the native logical planner."""
    weights = dict(physical)
    for layer in range(2):
        p = f'model.layers.{layer}'
        qkv = weights.pop(p + '.self_attn.qkv_proj.weight')
        gate_up = weights.pop(p + '.mlp.gate_up_proj.weight')
        if len(qkv) != 576 or len(gate_up) != 512:
            raise ValueError('Original fused row extent differs')
        for kind, matrix in [('q', qkv[:192]), ('k', qkv[192:384]), ('v', qkv[384:576])]:
            weights[p + f'.self_attn.{kind}_proj.weight'] = matrix
        weights[p + '.mlp.gate_proj.weight'] = gate_up[:256]
        weights[p + '.mlp.up_proj.weight'] = gate_up[256:512]
    return weights


def flatten(values):
    if isinstance(values, (list, tuple)):
        return [item for value in values for item in flatten(value)]
    return [values]


def dot(a, b, precision):
    if len(a) != len(b):
        raise ValueError('Dot shape differs')
    total = 0.0
    # All operands already are exactly representable FP32 or Half values.
    for x, y in zip(a, b, strict=True):
        total = precision(total + precision(x * y))
    return total


def linear(x, weights, precision, store):
    return [[store(dot(row, weight, precision)) for weight in weights] for row in x]


def norm(x, weights, epsilon, precision, store):
    output = []
    for row in x:
        mean = precision(dot(row, row, precision) / len(row))
        inverse = precision(1 / precision(math.sqrt(precision(mean + epsilon))))
        output.append([store(precision(precision(v * inverse) * w)) for v, w in zip(row, weights, strict=True)])
    return output


def rotary(x, heads, offset, profile, dim=96, theta=10000.0):
    if profile not in PROFILES or dim <= 0 or dim % 2 or heads <= 0 or any(len(row) != heads * dim for row in x):
        raise ValueError('Rotary profile/shape differs')
    output = [list(row) for row in x]
    store = f32 if profile == 'cpu' else half
    with localcontext() as context:
        context.prec = 80
        for pair in range(dim // 2):
            if profile == 'main':
                exponent = f32(f32(-2 * pair) / f32(dim))
                inverse = f32(math.pow(f32(theta), exponent))
            else:
                inverse = (Decimal.from_float(f32(theta)).ln() * Decimal(-2 * pair) / dim).exp()
            for token, row in enumerate(x):
                if profile == 'main':
                    angle = f32(f32(offset + token) * inverse)
                    sine, cosine = f32(math.sin(angle)), f32(math.cos(angle))
                else:
                    angle = float(Decimal(offset + token) * inverse)
                    sine, cosine = math.sin(angle), math.cos(angle)
                for head in range(heads):
                    a, b = head * dim + pair, head * dim + pair + dim // 2
                    if profile == 'main':
                        left = f32(f32(row[a] * cosine) - f32(row[b] * sine))
                        right = f32(f32(row[b] * cosine) + f32(row[a] * sine))
                    else:
                        left = row[a] * cosine - row[b] * sine
                        right = row[b] * cosine + row[a] * sine
                    output[token][a], output[token][b] = store(left), store(right)
    return output


def attention(q, k, v, precision, store, heads=2, kvheads=2, dim=96):
    n = len(q)
    if n == 0 or heads <= 0 or kvheads <= 0 or heads % kvheads or dim <= 0 or len(k) != n or len(v) != n:
        raise ValueError('Attention shape differs')
    if any(len(row) != heads * dim for row in q) or any(len(row) != kvheads * dim for row in k + v):
        raise ValueError('Attention row width differs')
    scores = [[[0.0] * n for _ in range(n)] for _ in range(heads)]
    probabilities = [[[0.0] * n for _ in range(n)] for _ in range(heads)]
    output = [[0.0] * (heads * dim) for _ in range(n)]
    scale = precision(math.sqrt(dim))
    for head in range(heads):
        qs, ks = head * dim, (head // (heads // kvheads)) * dim
        for token in range(n):
            for position in range(n):
                scores[head][token][position] = f32(precision(dot(q[token][qs:qs + dim], k[position][ks:ks + dim], precision) / scale))
            valid = scores[head][token][:token + 1]
            maximum = max(valid)
            exponential = [precision(math.exp(precision(value - maximum))) for value in valid]
            mass = 0.0
            for value in exponential:
                mass = precision(mass + value)
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError('Invalid softmax mass')
            for position, value in enumerate(exponential):
                probabilities[head][token][position] = f32(precision(value / mass))
            for feature in range(dim):
                output[token][qs + feature] = store(dot(probabilities[head][token][:token + 1],
                    [v[position][ks + feature] for position in range(token + 1)], precision))
    return scores, probabilities, output


def forward(weights, tied, tokens, offset, profile):
    if profile not in PROFILES or not tokens or len(tokens) > 17 or offset < 0 or offset + len(tokens) > 512 or any(t < 0 or t >= 64 for t in tokens):
        raise ValueError('Fixed tiny graph request differs')
    precision, store = (f32 if profile == 'main' else float), (f32 if profile == 'cpu' else half)
    captured = {}
    def save(name, values):
        raw = flatten(values)
        if not all(math.isfinite(v) for v in raw):
            raise ValueError('Nonfinite stage: ' + name)
        captured[name] = raw
        return values
    x = save('embedding', [[store(v) for v in weights['model.embed_tokens.weight'][t]] for t in tokens])
    epsilon = f32(1e-5)
    for layer in range(2):
        p, stage = f'model.layers.{layer}', f'layers.{layer}.'
        normalized = save(stage + 'input_norm', norm(x, weights[p + '.input_layernorm.weight'], epsilon, precision, store))
        q = save(stage + 'q_projection', linear(normalized, weights[p + '.self_attn.q_proj.weight'], precision, store))
        k = save(stage + 'k_projection', linear(normalized, weights[p + '.self_attn.k_proj.weight'], precision, store))
        v = save(stage + 'v_projection', linear(normalized, weights[p + '.self_attn.v_proj.weight'], precision, store))
        q = save(stage + 'q_rotary', rotary(q, 2, offset, profile))
        k = save(stage + 'k_rotary', rotary(k, 2, offset, profile))
        scores, probabilities, value = attention(q, k, v, precision, store)
        save(stage + 'attention_scores', scores)
        save(stage + 'attention_probabilities', probabilities)
        value = save(stage + 'attention_context', value)
        projected = save(stage + 'attention_projection', linear(value, weights[p + '.self_attn.o_proj.weight'], precision, store))
        x = save(stage + 'attention_residual', [[store(precision(a + b)) for a, b in zip(row, delta, strict=True)] for row, delta in zip(x, projected, strict=True)])
        normalized = save(stage + 'post_attention_norm', norm(x, weights[p + '.post_attention_layernorm.weight'], epsilon, precision, store))
        gate = save(stage + 'gate_projection', linear(normalized, weights[p + '.mlp.gate_proj.weight'], precision, store))
        up = save(stage + 'up_projection', linear(normalized, weights[p + '.mlp.up_proj.weight'], precision, store))
        active = []
        for row in gate:
            out = []
            for value in row:
                exponential = precision(math.exp(-abs(value)))
                denominator = precision(1 + exponential)
                sigmoid = precision(1 / denominator) if value >= 0 else precision(exponential / denominator)
                out.append(store(precision(value * sigmoid)))
            active.append(out)
        save(stage + 'gate_silu', active)
        gated = save(stage + 'gated', [[store(precision(a * b)) for a, b in zip(row, other, strict=True)] for row, other in zip(active, up, strict=True)])
        down = save(stage + 'down_projection', linear(gated, weights[p + '.mlp.down_proj.weight'], precision, store))
        x = save(stage + 'output', [[store(precision(a + b)) for a, b in zip(row, delta, strict=True)] for row, delta in zip(x, down, strict=True)])
    final = save('final_norm', norm(x, weights['model.norm.weight'], epsilon, precision, store))
    head = weights['model.embed_tokens.weight'] if tied else weights['lm_head.weight']
    save('logits', linear(final, head, precision, f32))
    return captured


def contracts(n):
    result = {'embedding': [n, 192], 'final_norm': [n, 192], 'logits': [n, 64]}
    stage = dict(input_norm=[n, 192], q_projection=[n, 2, 96], k_projection=[n, 2, 96], v_projection=[n, 2, 96],
        q_rotary=[n, 2, 96], k_rotary=[n, 2, 96], attention_scores=[2, n, n], attention_probabilities=[2, n, n],
        attention_context=[n, 192], attention_projection=[n, 192], attention_residual=[n, 192],
        post_attention_norm=[n, 192], gate_projection=[n, 256], up_projection=[n, 256], gate_silu=[n, 256],
        gated=[n, 256], down_projection=[n, 192], output=[n, 192])
    if set(stage) != set(LAYER_STAGES):
        raise ValueError('Stage inventory differs')
    for layer in range(2):
        result.update({f'layers.{layer}.{name}': shape for name, shape in stage.items()})
    return result


def export(directory):
    # Fresh output only; incomplete exports retain progress and never publish an
    # accepted manifest. No target outputs enter weights, inputs or gate formula.
    before = identity(Path(__file__).read_bytes())
    directory.mkdir(parents=False, exist_ok=False)
    files, cases, prefix, source_details = {}, [], {}, {}
    try:
        for variant in ('untied', 'tied'):
            tied = variant == 'tied'
            (directory / variant).mkdir()
            for leaf, raw in [('config.json', encoded(configuration(tied))), ('model.safetensors', make_artifact(tied))]:
                (directory / variant / leaf).write_bytes(raw)
                files[f'{variant}/{leaf}'] = identity(raw)
        with (directory / 'arrays.bin').open('xb') as binary, (directory / 'progress.jsonl').open('x') as progress:
            for variant in ('untied', 'tied'):
                artifact = (directory / variant / 'model.safetensors').read_bytes()
                physical, data_start = decode_physical(artifact, variant == 'tied')
                weights = split_original(physical)
                head_name = 'model.embed_tokens.weight' if variant == 'tied' else 'lm_head.weight'
                nonhalf = sum(v != half(v) for v in flatten(weights[head_name]))
                if nonhalf == 0 or data_start % 4 != 1:
                    raise ValueError('Packed-head or unaligned-source witness absent')
                source_details[variant] = dict(data_start=data_start, original_head=head_name,
                    head_reconstructed_values_not_half=nonhalf, physical_modules=len(physical),
                    logical_tensors=len(weights))
                for case_variant, name, tokens, offset in CASES:
                    if case_variant != variant:
                        continue
                    arrays = {p: forward(weights, variant == 'tied', tokens, offset, p) for p in PROFILES}
                    shape_contract = contracts(len(tokens))
                    if len(shape_contract) != 39 or any(set(a) != set(shape_contract) for a in arrays.values()):
                        raise ValueError('Exact case stage inventory differs')
                    stages = {}
                    for stage, shape in shape_contract.items():
                        desc = {}
                        for profile in PROFILES:
                            values = arrays[profile][stage]
                            if len(values) != math.prod(shape):
                                raise ValueError('Stage shape differs')
                            kind = 'f' if profile == 'cpu' or stage.endswith(('attention_scores', 'attention_probabilities', 'logits')) else 'e'
                            raw = b''.join(struct.pack('<' + kind, v) for v in values)
                            begin = binary.tell()
                            if begin + len(raw) > MAX_ARRAY_BYTES:
                                raise ValueError('Fixture array bytes exceed fixed bound')
                            binary.write(raw)
                            desc[profile] = dict(file='arrays.bin', offset=begin, shape=shape,
                                dtype='f32le' if kind == 'f' else 'f16le', **identity(raw))
                        envelope = max(abs(a - b) for a, b in zip(arrays['main'][stage], arrays['wide'][stage], strict=True))
                        stages[stage] = dict(shape=shape, profiles=desc, precision_envelope=envelope,
                            half_gate=dict(atol=max(HALF_POLICY['atol_floor'], HALF_POLICY['envelope_factor'] * envelope), rtol=HALF_POLICY['rtol']))
                    if variant == 'untied' and name in ('prompt', 'future_changed'):
                        for profile in PROFILES:
                            raw_prefix = struct.pack('<' + 'f' * (16 * 64), *arrays[profile]['logits'][:16 * 64])
                            if name == 'prompt':
                                prefix[profile] = raw_prefix
                            elif prefix[profile] != raw_prefix:
                                raise ValueError('Future token changed earlier source logits')
                    case = dict(variant=variant, name=name, tokens=list(tokens), position_offset=offset,
                        initial_kv_tokens=0, arrays=stages)
                    cases.append(case)
                    binary.flush()
                    progress.write(json.dumps(dict(case=name, variant=variant, complete=True, arrays_end=binary.tell())) + '\n')
                    progress.flush()
                    del arrays
                del weights, physical
        raw_arrays = (directory / 'arrays.bin').read_bytes()
        files['arrays.bin'] = identity(raw_arrays)
        del raw_arrays
        if len(cases) != 5 or sum(len(case['arrays']) for case in cases) != 195:
            raise ValueError('Fixed case count differs')
        for relative, pin in files.items():
            if identity((directory / relative).read_bytes()) != pin:
                raise ValueError('Output endpoint changed')
        if identity(Path(__file__).read_bytes()) != before:
            raise ValueError('Producer endpoint changed')
        manifest = dict(format=FORMAT, producer=before, files=files, source_details=source_details,
            runtime=dict(python=sys.version, implementation=sys.implementation.name, executable=str(Path(sys.executable).resolve())),
            half_policy=HALF_POLICY, cpu_gate=CPU_GATE,
            prospective_policy_sha256='f2eba6a4a8aa5c00020ddde373cdc4adc9365686cf62f54f6949359a593e1613',
            profiles=dict(cpu='ordered FP64 arithmetic with FP32 operation/storage boundaries; independent reference, not scalar emulation',
                main='ordered FP32 arithmetic and Half activation stores',
                wide='ordered FP64 arithmetic and Half activation stores',
                shared='FP32 reconstructed packed matrices including output head; FP32 scores/probabilities/logits; source float norms F16',
                rotary_main='FP32 exponent, FP32-rounded pow result, FP32 angle/trig/separate products/addition then Half',
                rotary_wide_cpu='Decimal80 ordinary frequency and angle, FP64 sin/cos and products, then profile store'),
            cases=cases, stage_groups=195, stored_arrays=585,
            oracle_stages_per_case=39, native_diagnostic_stages_per_case=35,
            implemented_context=512, declared_positions=4096, declared_sliding_window=2047,
            scope='Independent tiny original-row Phi composition; no exact Metal/CPU emulation, real checkpoint, longer-window, quality, app or performance qualification')
        raw_manifest = encoded(manifest)
        if len(raw_manifest) > MAX_MANIFEST_BYTES:
            raise ValueError('Manifest exceeds fixed bound')
        (directory / 'manifest.json').write_bytes(raw_manifest)
        return identity(raw_manifest)
    except BaseException as error:
        try:
            (directory / 'failure.json').write_bytes(encoded(dict(complete=False, error=type(error).__name__ + ': ' + str(error))))
        except Exception as record_error:
            error.add_note('Failure receipt could not be written: ' + repr(record_error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir)))
