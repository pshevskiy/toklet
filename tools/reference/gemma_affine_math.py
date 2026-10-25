#!/usr/bin/env python3
"""Pre-target packed4/group64/BF16 projection witnesses; exact Fraction dots.

No engine, MLX, NumPy or model imports. Ordered FP64 and mathematical dots remain
separate. Rejected cases preserve oracle disagreement instead of weakening gates.
"""
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import struct
import sys

ATOL, RTOL = 1e-5, 1e-4
MAXIMUM = struct.unpack('<f', bytes.fromhex('ffff7f7f'))[0]
MINIMUM = 2.0 ** -126
MAX_WEIGHT = 2.0 ** 32
UNIT = 2.0 ** -53
RAW_LIMIT = 8 * 1024 ** 2


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bf16_bits(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    if bits & 0x7f800000 == 0x7f800000:
        raise ValueError('finite BF16 operand required')
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    if rounded & 0x7f80 == 0x7f80:
        raise ValueError('BF16 rounding overflow')
    return rounded


def bf16(value):
    return struct.unpack('<f', struct.pack('<I', bf16_bits(value) << 16))[0]


def up_product(a, b):
    return 0.0 if a == 0 or b == 0 else math.nextafter(a * b, math.inf)


def admission(width, max_weight, inputs):
    if type(width) is not int or not 64 <= width <= 12288 or width % 64 or type(max_weight) not in (int, float) or not math.isfinite(max_weight) or not 0 <= max_weight <= MAX_WEIGHT or not 1 <= len(inputs) <= 32 * width:
        raise ValueError('bounded reference envelope required')
    if any(type(v) not in (int, float) or not math.isfinite(v) or f32(v) != v for v in inputs):
        raise ValueError('finite FP32 inputs required')
    maximum = max((abs(v) for v in inputs), default=0.0)
    bound = up_product(up_product(float(width), max_weight), maximum)
    error = up_product(up_product(up_product(2.0, float(width)), UNIT), bound)
    reason = ('input-absolute-products' if bound > MAXIMUM / 4 else
              'input-summation' if error > ATOL / 16 else None)
    return dict(admitted=reason is None, reason=reason,
                absolute_products_bound=bound, ordered_error_bound=error)


def codebook(scales, biases):
    if not 1 <= len(scales) == len(biases) <= 16384 * 192:
        raise ValueError('bounded paired codebook required')
    maximum = 0.0
    for scale, bias in zip(scales, biases):
        if any(type(v) not in (int, float) or not math.isfinite(v) or (v != 0 and abs(v) < MINIMUM) or bf16(v) != v for v in (scale, bias)):
            return dict(admitted=False, reason='metadata-domain', max_weight=None)
        for code in range(16):
            try:
                value = f32(float(scale) * code + float(bias))
            except OverflowError:
                return dict(admitted=False, reason='reconstruction-finite', max_weight=None)
            if not math.isfinite(value):
                return dict(admitted=False, reason='reconstruction-finite', max_weight=None)
            if value != 0 and (abs(value) < MINIMUM or abs(value) > MAX_WEIGHT):
                return dict(admitted=False, reason='weight-domain', max_weight=None)
            maximum = max(maximum, abs(value))
    return dict(admitted=True, reason=None, max_weight=maximum)


def operands(width, outputs, batch):
    codes = [(row * 11 + col * 7 + (col // 64) * 3) % 16
             for row in range(outputs) for col in range(width)]
    groups = width // 64
    scales = [bf16(0.0 if (g + row) % 11 == 0 else
                   (-1 if (g + row) % 3 == 0 else 1) * (1 + (g * 3 + row) % 9) / 173)
              for row in range(outputs) for g in range(groups)]
    biases = [bf16((.1, -.3, .0078125, .51, -.0625, 0.)[(g * 2 + row) % 6])
              for row in range(outputs) for g in range(groups)]
    first = [f32(((i * 37 + 11) % 257 - 128) / 997) for i in range(width)]
    inputs = [first[(i + token * 13) % width] for token in range(batch) for i in range(width)]
    return codes, scales, biases, inputs


@lru_cache(maxsize=1)
def cases():
    result = []

    def add(name, width, outputs, batch, values=None):
        codes, scales, biases, inputs = values or operands(width, outputs, batch)
        assert len(codes) == outputs * width and all(type(c) is int and 0 <= c < 16 for c in codes)
        assert len(scales) == len(biases) == outputs * width // 64
        assert len(inputs) == batch * width
        weights = [f32(float(scales[(r * width + c) // 64]) * codes[r * width + c] +
                       float(biases[(r * width + c) // 64]))
                   for r in range(outputs) for c in range(width)]
        policy = codebook(scales, biases)
        input_policy = admission(width, policy['max_weight'], inputs) if policy['admitted'] else None
        ordered, mathematical, rational = [], [], []
        for token in range(batch):
            for row in range(outputs):
                total = 0.0
                exact = Fraction(0)
                for col in range(width):
                    weight = weights[row * width + col]
                    value = inputs[token * width + col]
                    total += float(weight) * float(value)
                    exact += Fraction(weight) * Fraction(value)
                ordered.append(total)
                mathematical.append(float(exact))
                rational.append([str(exact.numerator), str(exact.denominator)])
        result.append(dict(name=name, width=width, outputs=outputs, batch=batch,
                           codes=codes, scales=scales, biases=biases, input=inputs,
                           constructor=policy, input_admission=input_policy,
                           reconstructed_sha256=hashlib.sha256(struct.pack('<' + 'f' * len(weights), *weights)).hexdigest(),
                           ordered_fp64=ordered, math_fp64=mathematical, exact_dot_rationals=rational))

    for width, outputs in [(64, 35), (128, 7), (256, 9), (512, 7), (1536, 9),
                           (4096, 5), (6144, 3), (8960, 5), (12288, 3)]:
        add('width-' + str(width), width, outputs, 3)
    # Identical terms in different lane/group orders; retain near-zero residual.
    for name, positions in [('adjacent', (0, 1)), ('different-lanes', (3, 27)),
                            ('different-groups', (3, 131))]:
        width = 256
        inputs = [0.] * width
        inputs[positions[0]], inputs[positions[1]] = 1., f32(-1. + 2. ** -20)
        add('cancel-' + name, width, 5, 1, ([0] * (5 * width), [0.] * 20,
            [bf16((r + 1) / 8) for r in range(5) for _ in range(4)], inputs))
    codes, scales, biases, inputs = operands(512, 7, 2)
    add('repeated-token-rows', 512, 7, 4,
        (codes, scales, biases, inputs + inputs))
    for name, positions in [('within-lane', (0, 32, 64)),
                            ('across-lanes', (0, 1, 31)),
                            ('cross-group', (0, 65, 127))]:
        inputs = [0.] * 128
        inputs[positions[0]], inputs[positions[1]], inputs[positions[2]] = 2. ** 17, 2. ** -8, -2. ** 17
        add('compensated-' + name, 128, 1, 1, ([0] * 128, [0.] * 2, [1.] * 2, inputs))
    # Near the summation admission cutoff at the deepest input width. A small
    # residual survives long positive/negative blocks but naive FP32 loses it.
    for position in (0, 67, 6143):
        inputs = [16.] * 6144 + [-16.] * 6144
        inputs[position] = f32(16. + 2. ** -14)
        add('deep-cancel-' + str(position), 12288, 1, 1,
            ([0] * 12288, [0.] * 192, [1.] * 192, inputs))
    width = 64
    for name, inputs in [('ordered-exact-conflict', [2. ** 100, 1., -2. ** 100] + [0.] * 61),
                         ('finite-product-overflow', [MAXIMUM, -MAXIMUM] + [0.] * 62),
                         ('zero-weights-max-input', [MAXIMUM, -MAXIMUM] + [1.] * 62)]:
        weight = 0. if name.startswith('zero-') else 2. if name.startswith('finite-') else 1.
        add(name, width, 1, 1, ([0] * width, [0.], [weight], inputs))
    # Fix both sides of the exact conservative U cutoff before any target exists.
    estimate = f32((ATOL / 16) / (2 * width * UNIT * width))
    bits = struct.unpack('<I', struct.pack('<f', estimate))[0]
    candidates = [struct.unpack('<f', struct.pack('<I', bits + i))[0] for i in range(-4, 5)]
    lower = max(v for v in candidates if admission(width, 1., [v])['admitted'])
    upper = min(v for v in candidates if not admission(width, 1., [v])['admitted'])
    assert struct.unpack('<I', struct.pack('<f', upper))[0] == struct.unpack('<I', struct.pack('<f', lower))[0] + 1
    for name, value in [('boundary-below', lower), ('boundary-above', upper)]:
        add(name, width, 1, 1, ([0] * width, [0.], [1.], [value] * width))
    for name, scale, bias, code in [('subnormal-metadata', 0., 2. ** -133, 0),
                                   ('subnormal-weight', 2. ** -126, -bf16(129 * 2. ** -133), 1),
                                   ('large-weight', 0., 2. ** 40, 0)]:
        add(name, width, 1, 1, ([code] * width, [scale], [bias], [MAXIMUM if name.startswith('subnormal-') else 1.] * width))
    return result


def document():
    raw = bytearray()
    descriptions = []
    for case in cases():
        entry = {key: value for key, value in case.items() if key not in ('codes', 'scales', 'biases', 'input')}
        arrays = {}
        payloads = [('weight', 'U32', [case['outputs'], case['width'] // 8],
                     bytes(a | (b << 4) for a, b in zip(case['codes'][::2], case['codes'][1::2]))),
                    ('scales', 'BF16', [case['outputs'], case['width'] // 64], struct.pack('<' + 'H' * len(case['scales']), *(bf16_bits(v) for v in case['scales']))),
                    ('biases', 'BF16', [case['outputs'], case['width'] // 64], struct.pack('<' + 'H' * len(case['biases']), *(bf16_bits(v) for v in case['biases']))),
                    ('input', 'F32', [case['batch'], case['width']], struct.pack('<' + 'f' * len(case['input']), *case['input']))]
        for name, dtype, shape, body in payloads:
            arrays[name] = dict(dtype=dtype, shape=shape, offset=len(raw), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            raw.extend(body)
        entry['arrays'] = arrays
        descriptions.append(entry)
    assert len(raw) <= RAW_LIMIT
    return dict(format='toklet-gemma-metal-affine-math-v1', producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                precision='once-reconstructed FP32 weights; separate ordered FP64 and exact Fraction dot converted to FP64',
                bits=4, group_size=64, gate=dict(absolute=ATOL, relative=RTOL),
                admission=dict(maximum_weight=MAX_WEIGHT, minimum_normal=MINIMUM,
                               ordered_unit=UNIT, ordered_error_limit=ATOL / 16, upward_positive_products=True),
                cases=descriptions, binary=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                source_or_device_qualified=False, whole_graph_qualified=False, performance_qualified=False), bytes(raw)


def write(directory):
    data, raw = document()
    body = (json.dumps(data, indent=2, allow_nan=False) + '\n').encode()
    assert len(body) <= 1024 ** 2
    directory.mkdir()
    for name, payload in [('math.bin', raw), ('math.json', body)]:
        with (directory / name).open('xb') as stream:
            assert stream.write(payload) == len(payload)
    manifest = dict(format='toklet-gemma-metal-affine-manifest-v1', producer_sha256=data['producer_sha256'],
                    files={name: dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                           for name, payload in [('math.json', body), ('math.bin', raw)]},
                    cases=len(data['cases']), native_target_executed=False)
    with (directory / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    return manifest


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: gemma_affine_math.py NEW_CAPTURE_DIRECTORY')
    print(json.dumps(write(Path(sys.argv[1]))))
