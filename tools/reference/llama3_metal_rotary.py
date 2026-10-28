"""Independent predeclared math and rounded-profile witnesses for B1c1.

No target/GPU execution. FP32 operations and Half nearest-even are explicit
stdlib struct conversions; mathematical frequencies use Decimal80 separately.
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

POLICY_SHA = 'f5f6e91726d3dbe32ba8fda9b7f949c27c9aeceb9f5e86d3953e924019549a46'
SPECS = (
    ('zero', 1, 1, 8, 0, 10000, 8, 1, 4, 64),
    ('tiny_all_bands', 3, 3, 8, 5, 10000, 8, 1, 4, 64),
    ('actual_short', 2, 2, 64, 0, 500000, 32, 1, 4, 8192),
    ('actual_k_heads_last', 3, 8, 64, 509, 500000, 32, 1, 4, 8192),
    ('actual_q_heads', 1, 32, 64, 5, 500000, 32, 1, 4, 8192),
    ('identity_last', 3, 1, 64, 509, 500000, 1, 1, 4, 8192),
    ('irregular', 5, 3, 16, 31, 65536, 8, 0.5, 2, 1024),
)
KEYS = ('tokens', 'heads', 'head_dim', 'offset', 'theta', 'factor', 'low', 'high', 'original')


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def half_bits(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def half_value(bits):
    return struct.unpack('<e', struct.pack('<H', bits))[0]


def spacing(bits):
    magnitude = bits & 0x7fff
    assert magnitude < 0x7bff
    center = half_value(magnitude)
    below = half_value(magnitude - 1) if magnitude else -2**-24
    above = half_value(magnitude + 1)
    return max(center - below, above - center)


def export():
    cases = []
    with localcontext() as context:
        context.prec = 80
        for name, *spec in SPECS:
            n, heads, dim, offset, theta, factor, low, high, original = spec
            values = [(((i * 7 + 4) % 31) - 15) / 8 for i in range(n * heads * dim)]
            values[0] = -0.0
            frequencies = [positional.scaled_frequency(
                (Decimal(theta).ln() * Decimal(-2 * pair) / dim).exp(),
                Decimal(factor), Decimal(low), Decimal(high), Decimal(original)) for pair in range(dim // 2)]
            for token in range(n):
                angle = float(Decimal(offset + token) * frequencies[0])
                if abs(math.sin(angle)) > 0.1:
                    ratio = math.cos(angle) / math.sin(angle)
                    if abs(ratio) <= 2:
                        first = token * heads * dim
                        values[first] = 1.0
                        values[first + dim // 2] = round(ratio * 1024) / 1024
            assert all(half_value(half_bits(v)) == v for v in values)
            count = len(values)
            mathematical, profile, math_bound, profile_bound = ([None] * count for _ in range(4))
            for pair, inverse in enumerate(frequencies):
                for token in range(n):
                    angle = float(Decimal(offset + token) * inverse)
                    sine, cosine = math.sin(angle), math.cos(angle)
                    rounded_angle = f32((offset + token) * f32(float(inverse)))
                    ps, pc = f32(math.sin(rounded_angle)), f32(math.cos(rounded_angle))
                    for head in range(heads):
                        a = (token * heads + head) * dim + pair
                        b = a + dim // 2
                        av, bv = values[a], values[b]
                        scale = abs(av) + abs(bv)
                        mathematical[a] = av * cosine - bv * sine
                        mathematical[b] = bv * cosine + av * sine
                        x = f32(f32(av * pc) - f32(bv * ps))
                        y = f32(f32(bv * pc) + f32(av * ps))
                        for index, result in ((a, x), (b, y)):
                            profile[index] = half_bits(result)
                            math_bound[index] = scale * (8 * 2**-23 * (abs(angle) + 1) + 0.0005) + 1e-5
                            profile_bound[index] = 8 * 2**-23 * scale + 2 * spacing(profile[index])
            cases.append(dict(name=name, **dict(zip(KEYS, spec)), input=values,
                              inverse_bits=[struct.unpack('<I', struct.pack('<f', float(v)))[0] for v in frequencies],
                              mathematical=mathematical, profile_half_bits=profile,
                              mathematical_bound=math_bound, profile_bound=profile_bound))
    return dict(format='toklet-llama3-metal-rotary-v1', astra_policy_sha256=POLICY_SHA,
                producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                positional_producer_sha256=hashlib.sha256(Path(positional.__file__).read_bytes()).hexdigest(),
                scope='Isolated Half rotary positions0..511; not model or MLX source parity', cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with args.output.open('xb') as stream:
        stream.write((json.dumps(export(), indent=2, allow_nan=False) + '\n').encode())
