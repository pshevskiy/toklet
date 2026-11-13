"""Independent stdlib tiny mixed-affine Ministral graph and supplied-NumPy oracle.

Packed weights are authored words, never native outputs or quantizer captures.
Both profiles reconstruct identical FP32 weights and keep Half storage at actual
graph boundaries. The FP32-ordered and FP64-wide profiles are independent of the
Metal reduction tree. Existing YaRN period/cache witnesses are reused unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import ministral3_graph_protocol as protocol
import ministral3_rotary_fixture as rotary

ROOT = Path(__file__).resolve().parents[2]
FORMAT = 'toklet-ministral3-half-graph-v1'
FIXTURE_LIMIT = 4 * 1024**2
need = protocol.need
f32 = protocol.f32


def half(value):
    return rotary.from_half(rotary.half_word(value))


def bf16_word(value):
    word = rotary.f32_word(value)
    return (word + 0x7fff + ((word >> 16) & 1)) >> 16


def bf16(word):
    need(type(word) is int and 0 <= word <= 65535 and word & 0x7f80 != 0x7f80,
         'Finite BF16 word required')
    return rotary.from_f32(word << 16)


def flatten(rows):
    return [value for row in rows for value in row]


def dot(left, right, precision):
    need(len(left) == len(right), 'Dot shape differs')
    total = precision(0)
    for a, b in zip(left, right, strict=True):
        total = precision(total + precision(precision(a) * precision(b)))
    return total


def linear(x, weight, precision, store=half):
    return [[store(dot(row, output, precision)) for output in weight] for row in x]


def norm(x, weight, precision):
    output = []
    for row in x:
        mean = precision(dot(row, row, precision) / precision(len(row)))
        inverse = precision(precision(1) / precision(math.sqrt(precision(mean + precision(f32(1e-5))))))
        output.append([half(precision(precision(precision(value) * inverse) * precision(scale)))
                       for value, scale in zip(row, weight, strict=True)])
    return output


def weight_specs():
    declarations = [('model.embed_tokens.weight', 64, 64, 8)]
    prefix = 'model.layers.0.'
    for role, rows, columns, bits in (
            ('self_attn.q_proj', 256, 64, 4), ('self_attn.k_proj', 128, 64, 4),
            ('self_attn.v_proj', 128, 64, 4), ('self_attn.o_proj', 64, 256, 8),
            ('mlp.gate_proj', 128, 64, 4), ('mlp.up_proj', 128, 64, 4),
            ('mlp.down_proj', 64, 128, 8)):
        declarations.append((prefix + role + '.weight', rows, columns, bits))
    return declarations


def norm_names():
    return ('model.norm.weight', 'model.layers.0.input_layernorm.weight',
            'model.layers.0.post_attention_layernorm.weight')


def authored_weights():
    """Tight little-endian low-code-first words; explicit separate BF16 metadata."""
    c = protocol.TINY
    output = {}
    for seed, (name, rows, columns, bits) in enumerate(weight_specs(), 1):
        maximum, lanes = (1 << bits) - 1, 32 // bits
        words, scales, biases = [], [], []
        for row in range(rows):
            for group in range(columns // 64):
                exponent = -7 if bits == 4 else (-10 if seed == 1 else -11)
                scale = 2.0**exponent * (1 + ((row + group + seed) % 3) / 4)
                if (row + group + seed) % 5 == 0:
                    scale = -scale
                scales.append(bf16_word(scale))
                biases.append(bf16_word(-maximum * scale / 2))
            for first in range(0, columns, lanes):
                word = 0
                for lane in range(lanes):
                    column = first + lane
                    code = (row * 73 + column * 29 + seed * 41
                            + ((row + 11) * (column + 7) % 257) * 13) & maximum
                    word |= code << (lane * bits)
                words.append(word)
        output[name] = dict(shape=[rows, columns], bits=bits, group_size=64,
                            weight_words=words, scales_bf16=scales, biases_bf16=biases)
    for seed, name in enumerate(norm_names()):
        output[name] = dict(shape=[c['hidden']], bf16_words=[
            bf16_word(1 + (((index * 11 + seed) % 17) - 8) / 128) for index in range(c['hidden'])])
    return output


def decode_weights(records):
    """Separate FP32 multiply/add; no entire packed weight is cast to Half."""
    specs = {name: (rows, columns, bits) for name, rows, columns, bits in weight_specs()}
    need(type(records) is dict and set(records) == set(specs) | set(norm_names()), 'Complete authored weight inventory required')
    output = {}
    for name, item in records.items():
        need(type(item) is dict, 'Typed authored weight record required')
        if name in norm_names():
            need(set(item) == {'shape', 'bf16_words'} and item['shape'] == [64]
                 and len(item['bf16_words']) == 64, 'Exact tiny norm extent required')
            output[name] = [half(bf16(word)) for word in item['bf16_words']]
            continue
        need(set(item) == {'shape', 'bits', 'group_size', 'weight_words', 'scales_bf16', 'biases_bf16'},
             'Complete packed record required')
        rows, columns, expected_bits = specs[name]
        bits = item['bits']
        need(item['shape'] == [rows, columns] and all(type(v) is int for v in item['shape'])
             and type(bits) is int and bits == expected_bits and type(item['group_size']) is int
             and item['group_size'] == 64 and columns % 64 == 0
             and all(type(v) is int and 0 <= v <= 0xffffffff for v in item['weight_words'])
             and len(item['weight_words']) == rows * columns * bits // 32
             and len(item['scales_bf16']) == len(item['biases_bf16']) == rows * columns // 64,
             'Exact authored packed extents required')
        lanes, maximum = 32 // bits, (1 << bits) - 1
        values = []
        for row in range(rows):
            decoded = []
            for column in range(columns):
                word = item['weight_words'][row * (columns // lanes) + column // lanes]
                code = (word >> ((column % lanes) * bits)) & maximum
                group = row * (columns // 64) + column // 64
                decoded.append(f32(f32(code * bf16(item['scales_bf16'][group]))
                                   + bf16(item['biases_bf16'][group])))
            values.append(decoded)
        output[name] = values
    return output


def rotate(q, k, offset, wide):
    case = dict(tokens=len(q), offset=offset, query_heads=2, kv_heads=1)
    periods = rotary.periods_wide(rotary.GEOMETRY) if wide else rotary.periods_main(rotary.GEOMETRY)
    expected = rotary.reference(case, rotary.GEOMETRY, periods,
                               [rotary.half_word(v) for v in flatten(q)],
                               [rotary.half_word(v) for v in flatten(k)], wide)
    def rows(name, width):
        raw = expected[name]
        return [[rotary.from_half(v) for v in raw[first:first + width]]
                for first in range(0, len(raw), width)]
    return rows('queries', 256), rows('keys', 128)


def attention(q, k, v, precision):
    count, dimension = len(q), 128
    scale = precision(f32(1 / math.sqrt(dimension)))
    output = [[0.0] * 256 for _ in range(count)]
    for head in range(2):
        first = head * dimension
        for token in range(count):
            scores = [f32(precision(dot(q[token][first:first + dimension], k[key], precision) * scale))
                      for key in range(token + 1)]
            maximum = max(scores)
            exponentials = [precision(math.exp(precision(value - maximum))) for value in scores]
            total = precision(0)
            for value in exponentials:
                total = precision(total + value)
            probability = [f32(precision(value / total)) for value in exponentials]
            for feature in range(dimension):
                output[token][first + feature] = half(dot(probability, [v[key][feature] for key in range(token + 1)], precision))
    return output


def tiny_forward(weights, tokens, offset, wide=False, diagnostic_offset=False):
    protocol.geometry(protocol.TINY, tokens, offset, diagnostic_offset)
    need(type(wide) is bool, 'Explicit independent profile required')
    precision = float if wide else f32
    captured = {}
    def save(name, values):
        flat = flatten(values)
        need(name not in captured and all(math.isfinite(v) for v in flat), 'Unique finite stage required')
        captured[name] = flat
        return values
    x = save('embedding', [[half(value) for value in weights['model.embed_tokens.weight'][token]] for token in tokens])
    prefix, stage = 'model.layers.0.', 'layers.0.'
    normalized = save(stage + 'input_norm', norm(x, weights[prefix + 'input_layernorm.weight'], precision))
    q = save(stage + 'q', linear(normalized, weights[prefix + 'self_attn.q_proj.weight'], precision))
    k = save(stage + 'k', linear(normalized, weights[prefix + 'self_attn.k_proj.weight'], precision))
    v = save(stage + 'v', linear(normalized, weights[prefix + 'self_attn.v_proj.weight'], precision))
    qr, kr = rotate(q, k, offset, wide)
    q, k = save(stage + 'q_rotary', qr), save(stage + 'k_rotary', kr)
    context = save(stage + 'attention_context', attention(q, k, v, precision))
    branch = save(stage + 'attention_projection', linear(context, weights[prefix + 'self_attn.o_proj.weight'], precision))
    x = save(stage + 'attention_residual', [[half(precision(a + b)) for a, b in zip(row, delta, strict=True)]
                                         for row, delta in zip(x, branch, strict=True)])
    normalized = save(stage + 'post_attention_norm', norm(x, weights[prefix + 'post_attention_layernorm.weight'], precision))
    gate = save(stage + 'gate', linear(normalized, weights[prefix + 'mlp.gate_proj.weight'], precision))
    up = save(stage + 'up', linear(normalized, weights[prefix + 'mlp.up_proj.weight'], precision))
    activated = []
    for row in gate:
        values = []
        for value in row:
            exponential = precision(math.exp(precision(-abs(value))))
            denominator = precision(1 + exponential)
            sigmoid = precision(1 / denominator) if value >= 0 else precision(exponential / denominator)
            values.append(half(precision(value * sigmoid)))
        activated.append(values)
    activated = save(stage + 'silu', activated)
    gated = save(stage + 'gated', [[half(precision(a * b)) for a, b in zip(row, other, strict=True)]
                                  for row, other in zip(activated, up, strict=True)])
    down = save(stage + 'down', linear(gated, weights[prefix + 'mlp.down_proj.weight'], precision))
    x = save(stage + 'output', [[half(precision(a + b)) for a, b in zip(row, delta, strict=True)]
                              for row, delta in zip(x, down, strict=True)])
    final = save('final_norm', norm(x, weights['model.norm.weight'], precision))
    save('logits', linear(final, weights['model.embed_tokens.weight'], precision, f32))
    need(set(captured) == set(protocol.contracts(protocol.TINY, len(tokens))), 'Complete tiny graph stages required')
    return captured


def record(path):
    raw = rotary.read_regular(path, FIXTURE_LIMIT)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def export():
    profile_path = ROOT / rotary.PROFILE_FILE
    profile, pin = rotary.load_profile(profile_path)
    weights = authored_weights()
    decoded = decode_weights(weights)
    cases = []
    for name, tokens, offset in protocol.TINY_CASES:
        diagnostic = name.startswith('diagnostic_')
        main = tiny_forward(decoded, tokens, offset, False, diagnostic)
        wide = tiny_forward(decoded, tokens, offset, True, diagnostic)
        arrays = {}
        for stage, shape in protocol.contracts(protocol.TINY, len(tokens)).items():
            dtype = 'f32le' if stage == 'logits' else 'f16le'
            encode = rotary.f32_word if stage == 'logits' else rotary.half_word
            envelope, gate = protocol.gate(main[stage], wide[stage])
            arrays[stage] = dict(shape=shape, dtype=dtype,
                                 main_bits=[encode(value) for value in main[stage]],
                                 wide_bits=[encode(value) for value in wide[stage]],
                                 precision_envelope=envelope, gate=gate)
        cases.append(dict(name=name, tokens=tokens, position_offset=offset,
                          initial_kv_tokens=0, diagnostic_offset=diagnostic, arrays=arrays))
    prompt, changed = (next(case for case in cases if case['name'] == name) for name in ('prompt', 'future_changed'))
    for stage, spec in prompt['arrays'].items():
        width = spec['shape'][1]
        for field in ('main_bits', 'wide_bits'):
            need(spec[field][:3 * width] == changed['arrays'][stage][field][:3 * width], 'Causal prefix changed: ' + stage)
    high = next(case for case in cases if case['diagnostic_offset'])
    composition = dict(case=high['name'], source_stage='layers.0.q', target_stage='layers.0.q_rotary',
                       diagnostic_only=True, wrong_keys_kind={})
    for profile_name in ('main_bits', 'wide_bits'):
        is_wide = profile_name == 'wide_bits'
        periods = rotary.periods_wide(rotary.GEOMETRY) if is_wide else rotary.periods_main(rotary.GEOMETRY)
        source = high['arrays']['layers.0.q'][profile_name]
        wrong = rotary.reference(dict(tokens=2, offset=16384, query_heads=2, kv_heads=2),
                                 rotary.GEOMETRY, periods, source, source, is_wide)['keys']
        need(high['arrays']['layers.0.q_rotary'][profile_name] != wrong,
             'High-position query/key composition witness is ineffective')
        composition['wrong_keys_kind'][profile_name] = wrong
    producers = {str(path.relative_to(ROOT)): record(path) for path in
                 (Path(__file__), Path(protocol.__file__), Path(rotary.__file__))}
    return dict(format=FORMAT, scope='Independent tiny graph/composition only; no model, quality, fit, performance or ordinary admission',
                config=protocol.TINY, policy=protocol.POLICY, inputs={rotary.PROFILE_FILE: {key: pin[key] for key in ('bytes', 'sha256')}},
                producers=producers, weights=weights, cases=cases, composition_control=composition,
                schedules=dict(chunks=[1, 2, 3, 4], replay_Half_bytes_exact=True,
                               reset_Half_bytes_exact=True, causal_prefix_tokens=3,
                               absolute_offsets_do_not_allocate_missing_prefix=True),
                comparison=dict(both_profiles_required=True, pooling_forbidden=True,
                                native_derived_enlargement_forbidden=True,
                                high_position_is_diagnostic_only=True))


def serialized(value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    need(len(raw) <= FIXTURE_LIMIT, 'Tiny fixture exceeds bound')
    return raw


def validate_fixture(value):
    need(type(value) is dict and value.get('format') == FORMAT
         and value.get('config') == protocol.TINY and value.get('policy') == protocol.POLICY,
         'Frozen tiny graph identity/geometry/policy required')
    protocol.geometry(value['config'], [1], 0)
    for field, expected in protocol.POLICY.items():
        need(type(value['policy'][field]) is type(expected), 'Typed immutable numerical policy required')
    decode_weights(value['weights'])
    cases = value.get('cases')
    need(type(cases) is list and len(cases) == len(protocol.TINY_CASES), 'Complete ordered tiny case inventory required')
    for case, (name, tokens, offset) in zip(cases, protocol.TINY_CASES, strict=True):
        need(type(case) is dict and case.get('name') == name and case.get('tokens') == tokens
             and type(case.get('position_offset')) is int and case['position_offset'] == offset
             and type(case.get('initial_kv_tokens')) is int and case['initial_kv_tokens'] == 0
             and type(case.get('diagnostic_offset')) is bool
             and case['diagnostic_offset'] == name.startswith('diagnostic_'), 'Exact case/token/position identity required')
        protocol.geometry(value['config'], case['tokens'], offset, case['diagnostic_offset'])
        shapes = protocol.contracts(value['config'], len(tokens))
        need(type(case.get('arrays')) is dict and set(case['arrays']) == set(shapes), 'Complete graph array inventory required')
        for stage, shape in shapes.items():
            array = case['arrays'][stage]
            dtype = 'f32le' if stage == 'logits' else 'f16le'
            decode = rotary.from_f32 if stage == 'logits' else rotary.from_half
            need(type(array) is dict and set(array) == {'shape', 'dtype', 'main_bits', 'wide_bits', 'precision_envelope', 'gate'}
                 and array['shape'] == shape and all(type(v) is int for v in array['shape'])
                 and array['dtype'] == dtype, 'Exact stored graph shape/dtype required')
            profiles = []
            for field in ('main_bits', 'wide_bits'):
                need(type(array[field]) is list and len(array[field]) == math.prod(shape), 'Complete profile coordinates required')
                profiles.append([decode(word) for word in array[field]])
            envelope, gate = protocol.gate(*profiles)
            need(type(array['precision_envelope']) is float and array['precision_envelope'] == envelope
                 and array['gate'] == gate and all(type(array['gate'][key]) is float for key in ('atol', 'rtol')),
                 'Reference-only unchanged per-array envelope/gate required')
    need(value.get('schedules') == dict(chunks=[1, 2, 3, 4], replay_Half_bytes_exact=True,
                                      reset_Half_bytes_exact=True, causal_prefix_tokens=3,
                                      absolute_offsets_do_not_allocate_missing_prefix=True)
         and value.get('comparison') == dict(both_profiles_required=True, pooling_forbidden=True,
                                             native_derived_enlargement_forbidden=True,
                                             high_position_is_diagnostic_only=True), 'Exact frozen state/comparison policy required')
    control = value.get('composition_control')
    need(type(control) is dict and control.get('case') == 'diagnostic_gain16384'
         and control.get('source_stage') == 'layers.0.q' and control.get('target_stage') == 'layers.0.q_rotary'
         and control.get('diagnostic_only') is True
         and type(control.get('wrong_keys_kind')) is dict
         and set(control['wrong_keys_kind']) == {'main_bits', 'wide_bits'}, 'Explicit high-position query-kind control required')
    high = cases[-1]['arrays']['layers.0.q_rotary']
    for field in ('main_bits', 'wide_bits'):
        wrong = control['wrong_keys_kind'][field]
        need(type(wrong) is list and len(wrong) == 512, 'Complete wrong-kind composition coordinates required')
        wrong = [rotary.from_half(word) for word in wrong]
        reference = [rotary.from_half(word) for word in high[field]]
        need(any(not protocol.member(a, b, high['gate']) for a, b in zip(wrong, reference, strict=True)),
             'Wrong query kind must fail its frozen profile gate')
    return value


def load_fixture(folder):
    folder = Path(folder)
    manifest = rotary.parse_json(rotary.read_regular(folder / 'manifest.json', 64 * 1024))
    need(manifest.get('format') == 'toklet-ministral3-half-graph-manifest-v1'
         and manifest.get('fixture', {}).get('file') == 'fixture.json', 'Exact tiny graph manifest required')
    raw = rotary.read_regular(folder / 'fixture.json', FIXTURE_LIMIT)
    need(manifest['fixture'] == dict(file='fixture.json', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
         'Complete fixture byte identity differs')
    fixture = validate_fixture(rotary.parse_json(raw))
    need(manifest.get('stages') == 19 and manifest.get('cases') == 6
         and manifest.get('coordinates_per_profile') == sum(math.prod(a['shape']) for case in fixture['cases'] for a in case['arrays'].values())
         and manifest.get('inputs') == fixture['inputs'] and manifest.get('producers') == fixture['producers'],
         'Complete manifest coverage and source identities required')
    for name, pin in {**fixture['inputs'], **fixture['producers']}.items():
        path = Path(name)
        need(not path.is_absolute() and '..' not in path.parts
             and (ROOT / path).resolve().is_relative_to(ROOT), 'Safe local fixture source required')
        need(record(ROOT / path) == pin, 'Fixture source identity differs: ' + name)
    return fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-directory', required=True)
    args = parser.parse_args(argv)
    output = Path(args.fresh_directory)
    need(not output.exists() and not output.is_symlink(), 'Fresh fixture directory required')
    fixture = validate_fixture(export())
    raw = serialized(fixture)
    output.mkdir(parents=False, exist_ok=False)
    with (output / 'fixture.json').open('xb') as stream:
        stream.write(raw)
    manifest = dict(format='toklet-ministral3-half-graph-manifest-v1',
                    fixture=dict(file='fixture.json', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                    stages=len(protocol.STAGES) + 3, cases=len(fixture['cases']),
                    coordinates_per_profile=sum(math.prod(value['shape']) for case in fixture['cases'] for value in case['arrays'].values()),
                    inputs=fixture['inputs'], producers=fixture['producers'],
                    scope=fixture['scope'])
    with (output / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: manifest[key] for key in ('fixture', 'stages', 'cases', 'coordinates_per_profile')}))


if __name__ == '__main__':
    main()
