"""Bounded independent Gemma3 Half graph and ordinary-RoPE references.

Stdlib only. Generic frozen source helpers are reused; no CPU forward, target
Rust/Metal or Gemma4 graph is executed. Main serial FP32 projections do not
emulate the native compensated SIMD reduction or midpoint-aware store.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import gemma3_cpu_graph as source
import gemma3_half_rms as rms_source
import gemma3_half_pointwise as pointwise
import gemma3_staged_attention as staged

FORMAT = 'toklet-gemma3-half-tiny-graph-v1'
PROTOCOL = '902e5938625c58fa53116ccb8950162ad7cc657dbf688f212a24c8a53769a17b'
POLICY = dict(fatal_reference='main', atol_floor=1e-4, envelope_factor=2, rtol=1e-3)
CASES = (('single', (1,)), ('short', (1, 7, 13, 3)),
         ('wrap12', (1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10)),
         ('future_changed12', (1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 19)))
POSITIONS = (0, 511, 512, 513)
POSITION_CAPACITY = 544  # Component domain [0,544); tiny graph remains [0,12).
MAX_BINARY, MAX_JSON = 8 * 1024**2, 2 * 1024**2
INPUT_PINS = {'config.json': {'bytes': 773, 'sha256': 'fa631e8e9476e2e75e0946e8724ec529d75122813d9acebb2717a6a1fa7a2690'}, 'model.safetensors': {'bytes': 60206, 'sha256': '5ea7ae502d8d1b385155d62f252bdcc82c637f60324c3937fa42caae53a5989d'}, 'components.json': {'bytes': 41785, 'sha256': '23a9677d3961f4e092eb5b9507c75174a2abe13af0c9a36ac252d681e1da380c'}}
DEPENDENCY_PINS = {'gemma3_cpu_graph.py': {'bytes': 21859, 'sha256': '6586a0e38d50f15b71c2bec8f087a6083330b67207e948f1e9b16e1f13e67754'}, 'gemma3_half_rms.py': {'bytes': 10205, 'sha256': '9368ba31ee169c0d527e6462df60f0ebcaacc09ac33bab98d4c5929f795a4ca7'}, 'gemma3_half_pointwise.py': {'bytes': 11191, 'sha256': '0e2086db4c2aacf786b117f3a320304c8248ffefe7877b2a5e1576b2704ba1b5'}, 'gemma3_staged_attention.py': {'bytes': 17609, 'sha256': 'a3a37f5460d6566e83200c0a29f5eaf19ab3eadbe52b1576170f7d4f75aabaab'}}
f32, half_word, half_value = source.f32, rms_source.half_word, rms_source.half_value
encoded, identity = source.encoded, source.identity


def half(value):
    value = half_value(half_word(value))
    if not math.isfinite(value):
        raise ValueError('Nonfinite Half store')
    return value


def profile_check(profile):
    if profile not in ('main', 'wide'):
        raise ValueError('Unknown profile')


def authenticate(directory):
    """Read immutable existing fixture; never generate a replacement artifact."""
    result = {}
    for name, pin in INPUT_PINS.items():
        path = directory / name
        if path.stat().st_size != pin['bytes']:
            raise ValueError('Input size differs: ' + name)
        raw = path.read_bytes()
        if identity(raw) != pin:
            raise ValueError('Input identity differs: ' + name)
        result[name] = raw
    if source.strict_json(result['config.json']) != source.configuration():
        raise ValueError('Fixed tiny config')
    return result


def dependencies():
    result = {}
    for name, pin in DEPENDENCY_PINS.items():
        path = Path(__file__).with_name(name)
        if identity(path.read_bytes()) != pin:
            raise ValueError('Source dependency differs: ' + name)
        result[name] = pin
    return result


def linear(x, matrix, profile, float_output=False):
    profile_check(profile)
    store = f32 if float_output else half
    output = []
    for row in x:
        values = []
        for weights in matrix:
            if len(row) != len(weights):
                raise ValueError('Projection width')
            if profile == 'main':
                total = 0.0
                for a, b in zip(row, weights, strict=True):
                    total = f32(total + f32(a * b))
            else:
                total = math.fsum(a * b for a, b in zip(row, weights, strict=True))
            value = store(total)
            if not math.isfinite(value):
                raise ValueError('Nonfinite projection')
            values.append(value)
        output.append(values)
    return output


def norm(x, scales, profile, heads=1):
    width = len(scales)
    if not x or any(len(row) != heads * width for row in x):
        raise ValueError('Per-head RMS shape')
    words = [half_word(v) for row in x for v in row]
    values = rms_source.rms(words, scales, len(x) * heads, width, width, profile)
    return [[half_value(v) for v in values[i * heads * width:(i + 1) * heads * width]] for i in range(len(x))]


def rotary(x, heads, dim, theta, positions, profile):
    profile_check(profile)
    if type(heads) is not int or heads not in (1, 4) or type(dim) is not int or dim not in (32, 256):
        raise ValueError('Bounded ordinary rotary shape')
    if theta not in (10000, 1000000) or not x or len(x) != len(positions):
        raise ValueError('Rotary theta/rows')
    if any(type(p) is not int or not 0 <= p < POSITION_CAPACITY for p in positions):
        raise ValueError('Rotary position domain [0,544)')
    if any(len(row) != heads * dim for row in x) or any(not math.isfinite(v) or half(v) != v for row in x for v in row):
        raise ValueError('Rotary finite Half input/width')
    output = [row[:] for row in x]
    with localcontext() as context:
        context.prec = 80
        for pair in range(dim // 2):
            if profile == 'main':
                exponent = f32(f32(-2.0 * f32(pair)) / f32(dim))
                inverse = f32(math.pow(f32(theta), exponent))
            else:
                inverse = (Decimal(theta).ln() * Decimal(-2 * pair) / Decimal(dim)).exp()
            for token, (row, position) in enumerate(zip(x, positions, strict=True)):
                angle = f32(f32(position) * inverse) if profile == 'main' else float(Decimal(position) * inverse)
                cosine, sine = math.cos(angle), math.sin(angle)
                if profile == 'main':
                    cosine, sine = f32(cosine), f32(sine)
                for head in range(heads):
                    a, b = head * dim + pair, head * dim + pair + dim // 2
                    if profile == 'main':
                        first = f32(f32(row[a] * cosine) - f32(row[b] * sine))
                        second = f32(f32(row[b] * cosine) + f32(row[a] * sine))
                    else:
                        first, second = row[a] * cosine - row[b] * sine, row[b] * cosine + row[a] * sine
                    output[token][a], output[token][b] = half(first), half(second)
    return output


def attention(q, k, v, window, profile):
    n, d = len(q), 32
    if not 1 <= n <= 12 or len(k) != n or len(v) != n:
        raise ValueError('Full tiny attention rows')
    if any(len(row) != 128 for row in q) or any(len(row) != d for row in k + v):
        raise ValueError('Tiny attention widths')
    c = dict(committed=0, tokens=n, dim=d, capacity=12, window=window,
             kind='local' if window else 'full', pattern='dyadic')
    old = {kind: [staged.PAD] * ((window or 12) * d + staged.GUARD) for kind in ('k', 'v')}
    new = {kind: [half_word(z) for row in rows for z in row] + [staged.PAD] * ((32 - n) * d + staged.GUARD)
           for kind, rows in (('k', k), ('v', v))}
    query = [half_word(z) for row in q for z in row] + [staged.PAD] * ((32 - n) * 128 + staged.GUARD)
    result = staged.attention(c, old, new, query, profile)
    context = [[half_value(z) for z in result['context'][i * 128:(i + 1) * 128]] for i in range(n)]
    return result['scores'], result['probabilities'], context


def residual(x, delta):
    return [[half(f32(a + b)) for a, b in zip(row, other, strict=True)]
            for row, other in zip(x, delta, strict=True)]


def contracts(n):
    result = {'embedding_lookup': [n, 64], 'embedding': [n, 64]}
    layer = dict(input_norm=[n, 64], q_projection=[n, 4, 32], k_projection=[n, 1, 32], v_projection=[n, 1, 32],
        q_norm=[n, 4, 32], k_norm=[n, 1, 32], q_rotary=[n, 4, 32], k_rotary=[n, 1, 32],
        attention_scores=[4, n, n], attention_probabilities=[4, n, n], attention_context=[n, 128],
        attention_projection=[n, 64], post_attention_norm=[n, 64], attention_residual=[n, 64],
        pre_ff_norm=[n, 64], gate_projection=[n, 128], up_projection=[n, 128], gate_gelu=[n, 128],
        gated=[n, 128], down_projection=[n, 64], post_ff_norm=[n, 64], output=[n, 64])
    for layer_index in range(2):
        result.update({f'layers.{layer_index}.{name}': shape for name, shape in layer.items()})
    result.update(final_norm=[n, 64], logits=[n, 64])
    return result


def forward(matrices, effective, tokens, profile):
    profile_check(profile)
    if not 1 <= len(tokens) <= 12 or any(type(t) is not int or not 0 <= t < 64 for t in tokens):
        raise ValueError('Fixed tiny request')
    captured = {}
    def save(name, values):
        values_flat = source.flatten(values)
        if name in captured or any(not math.isfinite(v) for v in values_flat):
            raise ValueError('Duplicate or nonfinite stage')
        captured[name] = values_flat
        return values
    table = matrices['model.embed_tokens.weight']
    x = save('embedding_lookup', [[half(v) for v in table[t]] for t in tokens])
    x = save('embedding', [[half_value(pointwise.scale_word(half_word(v), 8.0, profile)) for v in row] for row in x])
    for layer in range(2):
        p, s = f'model.layers.{layer}', f'layers.{layer}.'
        xn = save(s + 'input_norm', norm(x, effective[p + '.input_layernorm.weight'], profile))
        q = save(s + 'q_projection', linear(xn, matrices[p + '.self_attn.q_proj.weight'], profile))
        k = save(s + 'k_projection', linear(xn, matrices[p + '.self_attn.k_proj.weight'], profile))
        v = save(s + 'v_projection', linear(xn, matrices[p + '.self_attn.v_proj.weight'], profile))
        q = save(s + 'q_norm', norm(q, effective[p + '.self_attn.q_norm.weight'], profile, 4))
        k = save(s + 'k_norm', norm(k, effective[p + '.self_attn.k_norm.weight'], profile))
        theta = 10000 if layer == 0 else 1000000
        q = save(s + 'q_rotary', rotary(q, 4, 32, theta, list(range(len(tokens))), profile))
        k = save(s + 'k_rotary', rotary(k, 1, 32, theta, list(range(len(tokens))), profile))
        scores, probabilities, context = attention(q, k, v, 4 if layer == 0 else None, profile)
        save(s + 'attention_scores', scores)
        save(s + 'attention_probabilities', probabilities)
        save(s + 'attention_context', context)
        branch = save(s + 'attention_projection', linear(context, matrices[p + '.self_attn.o_proj.weight'], profile))
        branch = save(s + 'post_attention_norm', norm(branch, effective[p + '.post_attention_layernorm.weight'], profile))
        x = save(s + 'attention_residual', residual(x, branch))
        xn = save(s + 'pre_ff_norm', norm(x, effective[p + '.pre_feedforward_layernorm.weight'], profile))
        gate = save(s + 'gate_projection', linear(xn, matrices[p + '.mlp.gate_proj.weight'], profile))
        up = save(s + 'up_projection', linear(xn, matrices[p + '.mlp.up_proj.weight'], profile))
        pairs = [[pointwise.gelu_product(half_word(a), half_word(b), profile) for a, b in zip(row, other, strict=True)]
                 for row, other in zip(gate, up, strict=True)]
        save(s + 'gate_gelu', [[half_value(a) for a, _ in row] for row in pairs])
        gated = save(s + 'gated', [[half_value(b) for _, b in row] for row in pairs])
        branch = save(s + 'down_projection', linear(gated, matrices[p + '.mlp.down_proj.weight'], profile))
        branch = save(s + 'post_ff_norm', norm(branch, effective[p + '.post_feedforward_layernorm.weight'], profile))
        x = save(s + 'output', residual(x, branch))
    final = save('final_norm', norm(x, effective['model.norm.weight'], profile))
    save('logits', linear(final, table, profile, float_output=True))
    expected = contracts(len(tokens))
    if list(captured) != list(expected) or len(captured) != 48:
        raise ValueError('Ordered 48-stage inventory')
    if any(len(captured[name]) != math.prod(shape) for name, shape in expected.items()):
        raise ValueError('Stage shape')
    return captured


def rotary_components():
    """Independent mathematical output remains unrounded for envelope checks."""
    for dim in (32, 256):
        for theta in (10000, 1000000):
            for heads in (4, 1):
                x = [[half(((i * heads * dim + j) * 13 % 31) * .125 - 1.875)
                      for j in range(heads * dim)] for i in range(len(POSITIONS))]
                math_values, bounds = [row[:] for row in x], [row[:] for row in x]
                with localcontext() as context:
                    context.prec = 80
                    for pair in range(dim // 2):
                        inverse = (Decimal(theta).ln() * Decimal(-2 * pair) / Decimal(dim)).exp()
                        for token, position in enumerate(POSITIONS):
                            angle = float(Decimal(position) * inverse)
                            cosine, sine = math.cos(angle), math.sin(angle)
                            for head in range(heads):
                                a, b = head * dim + pair, head * dim + pair + dim // 2
                                av, bv = x[token][a], x[token][b]
                                math_values[token][a] = av * cosine - bv * sine
                                math_values[token][b] = bv * cosine + av * sine
                                bound = (abs(av) + abs(bv)) * (8 * 2**-23 * (abs(angle) + 1) + .0005) + 1e-5
                                bounds[token][a] = bounds[token][b] = bound
                yield dict(name=f'd{dim}_theta{theta}_heads{heads}', dim=dim, theta=theta, heads=heads,
                    positions=list(POSITIONS), shape=[4, heads, dim], input=source.flatten(x),
                    main=source.flatten(rotary(x, heads, dim, theta, POSITIONS, 'main')),
                    wide=source.flatten(rotary(x, heads, dim, theta, POSITIONS, 'wide')),
                    mathematical=source.flatten(math_values), math_bound=source.flatten(bounds))


def gate(main, wide):
    if not main or any(not math.isfinite(v) for v in main + wide):
        raise ValueError('Gate requires finite nonempty arrays')
    envelope = max(abs(a - b) for a, b in zip(main, wide, strict=True))
    return dict(envelope=envelope, gate=dict(atol=max(1e-4, 2 * envelope), rtol=1e-3))


def export(directory, fixture):
    before, deps = identity(Path(__file__).read_bytes()), dependencies()
    inputs = authenticate(fixture)
    # For BF16 x nibble, the product is exactly representable in FP32; the
    # frozen decoder's two operations therefore equal one canonical FP32 affine
    # reconstruction. Both profiles share these weights, never Half weights.
    matrices, gamma, effective, payload_start = source.decode(inputs['model.safetensors'])
    directory.mkdir(parents=False, exist_ok=False)
    try:
        cases, components, causal = [], [], {}
        with (directory / 'arrays.bin').open('xb') as binary, (directory / 'progress.jsonl').open('x') as progress:
            def save(values, dtype, shape):
                code = {'u16le': 'H', 'f32le': 'f', 'f64le': 'd'}[dtype]
                data = [half_word(v) for v in values] if dtype == 'u16le' else values
                if len(data) != math.prod(shape) or any(not math.isfinite(v) for v in values):
                    raise ValueError('Array shape/finite contract')
                raw = b''.join(struct.pack('<' + code, v) for v in data)
                offset = binary.tell()
                if offset + len(raw) > MAX_BINARY:
                    raise ValueError('Bounded binary extent')
                binary.write(raw)
                return dict(file='arrays.bin', offset=offset, dtype=dtype, shape=shape, **identity(raw))
            for name, tokens in CASES:
                progress.write(json.dumps(dict(case=name, started=True)) + '\n')
                progress.flush()
                profiles = {p: forward(matrices, effective, tokens, p) for p in ('main', 'wide')}
                arrays = {}
                for stage, shape in contracts(len(tokens)).items():
                    dtype = 'f32le' if stage == 'logits' or stage.endswith(('attention_scores', 'attention_probabilities')) else 'u16le'
                    a, b = profiles['main'][stage], profiles['wide'][stage]
                    arrays[stage] = dict(main=save(a, dtype, shape), wide=save(b, dtype, shape), **gate(a, b))
                for profile in profiles:
                    if name == 'wrap12':
                        causal[profile] = profiles[profile]['logits'][:11 * 64]
                    if name == 'future_changed12' and causal[profile] != profiles[profile]['logits'][:11 * 64]:
                        raise ValueError('Source causal logits prefix')
                cases.append(dict(name=name, tokens=list(tokens), position_offset=0, initial_kv_tokens=0,
                    stage_order=list(contracts(len(tokens))), arrays=arrays))
                progress.write(json.dumps(dict(case=name, complete=True, arrays_end=binary.tell())) + '\n')
                progress.flush()
            for component in rotary_components():
                arrays = {}
                for kind in ('input', 'main', 'wide', 'mathematical', 'math_bound'):
                    dtype = 'f64le' if kind in ('mathematical', 'math_bound') else 'u16le'
                    arrays[kind] = save(component[kind], dtype, component['shape'])
                for profile in ('main', 'wide'):
                    if any(abs(a - b) > bound for a, b, bound in zip(component[profile], component['mathematical'], component['math_bound'], strict=True)):
                        raise ValueError('Source rotary mathematical envelope')
                components.append(dict(**{k: component[k] for k in ('name', 'dim', 'theta', 'heads', 'positions', 'shape')},
                    arrays=arrays, **gate(component['main'], component['wide'])))
        files = {'arrays.bin': identity((directory / 'arrays.bin').read_bytes())}
        if identity(Path(__file__).read_bytes()) != before or dependencies() != deps or authenticate(fixture) != inputs:
            raise ValueError('Source endpoint changed')
        manifest = dict(format=FORMAT, protocol_sha256=PROTOCOL, producer=before, dependencies=deps,
            input_files=INPUT_PINS, files=files, cases=cases, rotary_components=components, policy=POLICY,
            graph_stage_groups=192, graph_stored_arrays=384, rotary_stored_arrays=40, total_stored_arrays=424,
            source_reference_executed=True, target_gpu_executed=False,
            source=dict(matrix_owners=len(matrices), raw_gamma_vectors=len(gamma), payload_start=payload_start,
                        tied_head='model.embed_tokens.weight', source_position_domain=[0, POSITION_CAPACITY]),
            native_schedule_controls=dict(full_tokens=12, chunks=[1, 3, 7], exact_numeric_logits=True,
                signed_zero_bits_required=False, source_schedule_emulation=False),
            runtime=dict(python=sys.version, executable=str(Path(sys.executable).resolve())),
            scope='Tiny independent Half graph and bounded ordinary RoPE; no native SIMD emulation, device, real-model, admission, quality or performance qualification')
        raw = encoded(manifest)
        if len(raw) > MAX_JSON:
            raise ValueError('Manifest bound')
        (directory / 'manifest.json').write_bytes(raw)
        return identity(raw)
    except BaseException as error:
        (directory / 'failure.json').write_bytes(encoded(dict(complete=False, error=type(error).__name__ + ': ' + str(error))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--fixture-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures/gemma3-cpu-v1')
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir, args.fixture_dir)))
