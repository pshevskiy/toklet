"""Pinned LFM3 source-formula fixtures; no remote model code is executed.

Torch CPU supplies the declared FP32/BF16 source arithmetic. A separate scalar
implementation checks every BF16 result and the original FP32 mathematical
diagnostic. Inputs, source bytes, runtime and gates are fixed before Rust work.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

RUNTIME_SHA = 'fe30f7eb810f6ca3637a3c60020904a13b0b7fcd16bc0505483dd231c5a1b5c4'
CONTRACT_SHA = 'a62409af216c61c7b5752b15570fa27716595a6703899b3962ea11a95ed7505e'


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def bf16(value):
    word = bits(value)
    need(word & 0x7f800000 != 0x7f800000, 'Finite BF16 conversion required')
    result = ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 0xffff
    need(result & 0x7f80 != 0x7f80, 'BF16 conversion overflow')
    return result


def widen(value):
    need(type(value) is int and 0 <= value < 65536 and value & 0x7f80 != 0x7f80,
         'Finite BF16 bits required')
    return struct.unpack('<f', struct.pack('<I', value << 16))[0]


def input_pool():
    return [0, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x80, 0x8080,
            0x3f80, 0xbf80, 0x3f01, 0xbf01, 0x0081, 0x8081] + [
                bf16(((index * 29) % 251 - 125) / 16) for index in range(128)]


def inputs(pool, offset, tokens, heads):
    return [pool[(index * 73 + token * 11 + head * 7 + offset * 3) % len(pool)]
            for token in range(tokens) for head in range(heads) for dim in range(128)
            for index in [token * heads * 128 + head * 128 + dim]]


def scalar_pair(a, b, cosine, sine):
    """Separate FP32 products/additions, including the source signed-zero rules."""
    a, b, cosine, sine = map(widen, (a, b, cosine, sine))
    first = f32(f32(a * cosine) + f32((-b) * sine))
    second = f32(f32(b * cosine) + f32(a * sine))
    return bf16(first), bf16(second)


def verify_runtime(runtime):
    for name, pin in runtime['files'].items():
        need(digest(Path(name)) == pin, 'Admitted runtime file changed: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['contract', 'metadata', 'runtime', 'fresh_fixture', 'fresh_raw_directory']:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    need(not args.fresh_fixture.exists() and args.fresh_fixture.parent.is_dir()
         and not args.fresh_raw_directory.exists(), 'Fresh fixture/raw outputs required')
    need(digest(args.contract) == CONTRACT_SHA, 'Frozen source contract differs')
    contract = json.loads(args.contract.read_text())
    for name, pin in contract['source_code_sha256'].items():
        need(digest(args.metadata / name) == pin, 'Pinned source metadata differs')
    need(digest(args.runtime) == RUNTIME_SHA, 'Pinned admitted runtime differs')
    runtime = json.loads(args.runtime.read_text())
    verify_runtime(runtime)
    import numpy as np
    import torch
    need(torch.__version__ == '2.9.1' and np.__version__ == '2.3.2', 'Admitted versions differ')
    torch.set_num_threads(1)
    args.fresh_raw_directory.mkdir()
    pool = input_pool()
    profiles, cases = [], []
    scalar_failures = diagnostic_failures = values = 0
    diagnostic_maximum = 0.0
    positions = list(range(2048)) + contract['sparse_positions']

    def raw(tensor):
        return tensor.contiguous().view(torch.int16).numpy().astype('<u2').tobytes()

    for name, profile in contract['profiles'].items():
        pairs, theta = profile['active_pairs'], profile['theta']
        exponent = torch.arange(0, 2 * pairs, 2, dtype=torch.float32) / 128
        frequency = torch.cat((1.0 / (theta ** exponent), torch.zeros(64 - pairs, dtype=torch.float32)))
        angle = torch.tensor(positions, dtype=torch.float32)[:, None] * frequency
        cos32, sin32 = angle.cos(), angle.sin()
        cosine, sine = cos32.to(torch.bfloat16), sin32.to(torch.bfloat16)
        table = torch.stack((cosine, sine), dim=-1)
        prefix_bytes = raw(table[:2048])
        (args.fresh_raw_directory / (name + '-coefficients.u16le')).write_bytes(prefix_bytes)
        sparse = []
        for row, position in enumerate(contract['sparse_positions'], 2048):
            sparse.append(dict(position=position, coefficient_bits=list(struct.unpack('<128H', raw(table[row])))))
        profiles.append(dict(name=name, frequency_bits=frequency.view(torch.int32).tolist(),
            prefix_coefficient_sha256=hashlib.sha256(prefix_bytes).hexdigest(), sparse=sparse))
        for offset in contract['fixture_offsets']:
            for tokens in contract['fixture_lengths']:
                for heads in contract['nominal_heads']:
                    source = inputs(pool, offset, tokens, heads)
                    x = torch.from_numpy(np.array(source, dtype=np.uint16)).view(torch.bfloat16)
                    xf = x.reshape(tokens, heads, 128).float()
                    c = cosine[offset:offset + tokens].float()
                    s = sine[offset:offset + tokens].float()
                    c = torch.cat((c, c), dim=-1)[:, None, :]
                    s = torch.cat((s, s), dim=-1)[:, None, :]
                    rotated = torch.cat((-xf[..., 64:], xf[..., :64]), dim=-1)
                    output = (xf * c + rotated * s).to(torch.bfloat16)
                    result = raw(output)
                    expected = list(struct.unpack('<' + str(len(source)) + 'H', result))
                    c_words = cosine[offset:offset + tokens].view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()
                    s_words = sine[offset:offset + tokens].view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()
                    # The mathematical diagnostic keeps the contract's FP32
                    # angles but evaluates trig/rotation independently in FP64.
                    c32 = torch.cat((cos32[offset:offset + tokens], cos32[offset:offset + tokens]), dim=-1)[:, None, :]
                    s32 = torch.cat((sin32[offset:offset + tokens], sin32[offset:offset + tokens]), dim=-1)[:, None, :]
                    diagnostic = (xf * c32 + rotated * s32).reshape(-1).tolist()
                    for token in range(tokens):
                        for head in range(heads):
                            start = (token * heads + head) * 128
                            for pair in range(64):
                                first, second = start + pair, start + 64 + pair
                                got = scalar_pair(source[first], source[second],
                                                  c_words[token][pair], s_words[token][pair])
                                scalar_failures += got != (expected[first], expected[second])
                                a = widen(source[first]); b = widen(source[second])
                                phase = angle[offset + token, pair].item()
                                cs, sn = math.cos(phase), math.sin(phase)
                                for index, want in [(first, a * cs + (-b) * sn), (second, b * cs + a * sn)]:
                                    error = abs(diagnostic[index] - want)
                                    diagnostic_maximum = max(diagnostic_maximum, error)
                                    diagnostic_failures += error > 1e-5 + 1e-4 * abs(want)
                    case_name = f'{name}-{offset}-{tokens}-{heads}'
                    (args.fresh_raw_directory / (case_name + '.u16le')).write_bytes(result)
                    values += len(source)
                    cases.append(dict(name=case_name, profile=name, offset=offset, tokens=tokens, heads=heads,
                        input_sha256=hashlib.sha256(struct.pack('<' + str(len(source)) + 'H', *source)).hexdigest(),
                        output_sha256=hashlib.sha256(result).hexdigest(), first_output_bits=expected[:16],
                        last_output_bits=expected[-16:]))
    verify_runtime(runtime)
    need(scalar_failures == diagnostic_failures == 0, 'Independent scalar/source comparison rejected')
    result = dict(fixture_completed=True, format='toklet-lfm3-rotary-bf16-v1', contract_sha256=CONTRACT_SHA,
        source_code_sha256=contract['source_code_sha256'], runtime_sha256=digest(args.runtime),
        producer_sha256=digest(Path(__file__)), torch_version=torch.__version__, numpy_version=np.__version__,
        device='cpu', head_dim=128, coefficient_layout='[position,64,cosine/sine] BF16 little-endian',
        input_pool=pool, input_rule='pool[(flat_index*73+token*11+head*7+offset*3)%pool.len]',
        profiles=profiles, cases=cases, compared_output_values=values, scalar_bit_failures=scalar_failures,
        fp32_mathematical_diagnostic=dict(abs=1e-5, rel=1e-4, failures=diagnostic_failures,
            maximum_absolute_error=diagnostic_maximum, angle='contract FP32 position*frequency',
            reference='independent FP64 trig/rotation, before BF16 coefficient/store rounding'),
        whole_model_adoption_qualified=False)
    with args.fresh_fixture.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(fixture_completed=True, cases=len(cases), output_values=values,
        scalar_bit_failures=scalar_failures, diagnostic_failures=diagnostic_failures,
        fixture_sha256=digest(args.fresh_fixture))))


if __name__ == '__main__':
    main()
