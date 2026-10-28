"""Freeze an independent Decimal80 mathematical Llama3 rotary witness.

Uses frequency-space cutoffs and effective wavelength stretch, rather than the
Rust wavelength-branch expression. Trig and final pair arithmetic use FP64.
This is not a capture of an MLX or Transformers device execution profile.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path

PI = Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628')
SPECS = (
    ('position_zero', 1, 1, 8, 0, 500000, 32, 1, 4, 8192),
    ('actual_short', 3, 2, 64, 5, 500000, 32, 1, 4, 8192),
    ('original_context_edge', 3, 1, 64, 8191, 500000, 32, 1, 4, 8192),
    ('declared_context_end', 3, 2, 64, 131069, 500000, 32, 1, 4, 8192),
    ('high_position_math_only', 2, 1, 64, 1048575, 500000, 32, 1, 4, 8192),
    ('identity_factor', 3, 2, 16, 8191, 500000, 1, 1, 4, 8192),
    ('other_parameters', 2, 1, 16, 1023, 65536, 8, 0.5, 2, 1024),
)
KEYS = ('tokens', 'heads', 'head_dim', 'offset', 'theta', 'factor', 'low', 'high', 'original')


def scaled_frequency(frequency, factor, low, high, original):
    tau = 2 * PI
    lower = tau * low / original
    upper = tau * high / original
    if frequency <= lower:
        stretch = factor
    elif frequency >= upper:
        stretch = Decimal(1)
    else:
        fraction = (frequency - lower) / (upper - lower)
        stretch = factor / (1 + (factor - 1) * fraction)
    return frequency / stretch


def export():
    cases = []
    controls = []
    with localcontext() as context:
        context.prec = 80
        for name, *spec in SPECS:
            params = dict(zip(KEYS, spec))
            n, heads, dim, offset, theta, factor, low, high, original = spec
            count = n * heads * dim
            values = [(((i * 7 + 4) % 31) - 15) / 8 for i in range(count)]
            output = [None] * count
            frequencies = []
            for pair in range(dim // 2):
                ordinary = (Decimal(theta).ln() * Decimal(-2 * pair) / dim).exp()
                inverse = scaled_frequency(ordinary, Decimal(factor), Decimal(low), Decimal(high), Decimal(original))
                frequencies.append(float(inverse))
                for token in range(n):
                    angle = float(Decimal(offset + token) * inverse)
                    sine, cosine = math.sin(angle), math.cos(angle)
                    for head in range(heads):
                        a = (token * heads + head) * dim + pair
                        b = a + dim // 2
                        output[a] = values[a] * cosine - values[b] * sine
                        output[b] = values[b] * cosine + values[a] * sine
            cases.append(dict(name=name, **params, input=values,
                              inverse_frequencies=frequencies, output=output))
        # Both exact transition boundaries and their two neighboring regions.
        for wavelength in (1024, 2048, 4096, 8192, 16384):
            ordinary = 2 * PI / wavelength
            inverse = scaled_frequency(ordinary, Decimal(32), Decimal(1), Decimal(4), Decimal(8192))
            controls.append(dict(wavelength=wavelength, ordinary=float(ordinary), scaled=float(inverse)))
    return dict(
        format='toklet-llama3-rotary-math-v1',
        precision='Decimal80 frequencies and angles, FP64 trig and output; exact FP32 inputs',
        source=dict(url='https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/modeling_rope_utils.py',
                    sha256='81432edf7d4974e202e850798b034cdb463b0e6a9d5503d043f26cdaca79ce95', symbol='_compute_llama3_parameters'),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_execution_qualified=False, metal_qualified=False,
        controls=controls, cases=cases,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = (json.dumps(export(), indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as stream:
        stream.write(raw)
