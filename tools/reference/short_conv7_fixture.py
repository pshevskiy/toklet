"""Independent exact-rational seven-tap fixtures; no Toklet/NumPy/MLX/Torch math.

Only scalar FP32 convolution/history semantics are specified here. B*x/C gates,
BF16 activation rounding, complete LFM3 execution and adoption remain separate.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct

CHILD_SHA = 'b6ff17ac993dddf0f6f51bd872b12ae1437f4453a54a0b4878cedc7864fbc07e'
HEADER_SHA = '491ddbe9714ec2907c4d444294301198ae6eed8e1152af3b90237332b860b30b'
CHANNELS = [0, 1, 1023, 2047]


def need(value, message):
    if not value:
        raise ValueError(message)


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def rational(value):
    need(type(value) is int and 0 <= value < 2**32 and value >> 23 & 255 != 255,
         'Finite FP32 bits required')
    exponent, fraction = value >> 23 & 255, value & ((1 << 23)-1)
    significand = fraction if exponent == 0 else fraction + (1 << 23)
    shift = -149 if exponent == 0 else exponent-150
    sign = -1 if value >> 31 else 1
    return Fraction(sign*significand*(1 << max(shift, 0)), 1 << max(-shift, 0))


def rounded_ratio(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (2*remainder > denominator or (2*remainder == denominator and quotient % 2 == 1))


def rounded(value):
    """Direct exact-rational to IEEE FP32, nearest-even; no intermediate float."""
    need(type(value) is Fraction, 'Exact rational output required')
    if not value:
        return 0
    sign = (1 << 31) if value < 0 else 0
    value = abs(value)
    n, d = value.numerator, value.denominator
    exponent = n.bit_length()-d.bit_length()
    if value < Fraction(1 << max(exponent, 0), 1 << max(-exponent, 0)):
        exponent -= 1
    if exponent < -126:
        units = rounded_ratio(n << 149, d)
        return sign | units
    shift = 23-exponent
    units = rounded_ratio(n << max(shift, 0), d << max(-shift, 0))
    if units == 1 << 24:
        units >>= 1
        exponent += 1
    need(exponent <= 127, 'Exact convolution output overflows FP32')
    return sign | ((exponent+127) << 23) | (units-(1 << 23))


def oracle(inputs, weights, initial, tokens, channels):
    need(type(tokens) is int and 0 <= tokens <= 32 and type(channels) is int and 1 <= channels <= 8
         and len(inputs) == tokens*channels and len(weights) == 7*channels and len(initial) == 6*channels,
         'Fixed small seven-tap geometry required')
    joined = initial+inputs
    output = []
    histories = []
    for token in range(tokens):
        for channel in range(channels):
            dot = sum((rational(joined[(token+tap)*channels+channel])
                       * rational(weights[channel*7+tap]) for tap in range(7)), Fraction(0))
            output.append(rounded(dot))
        histories.extend(joined[(token+1)*channels:(token+7)*channels])
    final = joined[-6*channels:]
    return output, histories, final


def coefficients(path, header_path):
    header = header_path.read_bytes()
    need(hashlib.sha256(header).hexdigest() == HEADER_SHA, 'Pinned child header differs')
    schema = json.loads(header)
    row = schema['model.layers.0.conv.conv.weight']
    need(row['dtype'] == 'BF16' and row['shape'] == [2048, 7, 1]
         and row['data_offsets'][1]-row['data_offsets'][0] == 2048*7*2, 'Pinned convolution layout differs')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        need(before.st_size == 3457635448, 'Pinned candidate size differs')
        with os.fdopen(os.dup(fd), 'rb') as stream:
            need(hashlib.file_digest(stream, 'sha256').hexdigest() == CHILD_SHA, 'Pinned candidate hash differs')
        origin = 8+len(header)
        need(os.pread(fd, origin, 0) == len(header).to_bytes(8, 'little')+header, 'Actual candidate header differs')
        raw = b''.join(os.pread(fd, 14, origin+row['data_offsets'][0]+14*channel) for channel in CHANNELS)
        need(len(raw) == 56, 'Short candidate coefficient read')
        after = path.lstat()
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Candidate changed during fixture capture')
    finally:
        os.close(fd)
    values = list(struct.unpack('<28H', raw))
    for value in values:
        rational(value << 16)
    return [value << 16 for value in values], dict(object_sha256=CHILD_SHA,
        header_sha256=HEADER_SHA, tensor='model.layers.0.conv.conv.weight', shape=[2048, 7, 1],
        channels=CHANNELS, tensor_data_offset=row['data_offsets'][0], payload_origin=origin,
        selected_bf16_u16=values, selected_bytes_sha256=hashlib.sha256(raw).hexdigest())


def make_case(name, tokens, channels, inputs, weights, initial, all_partitions=False):
    output, histories, final = oracle(inputs, weights, initial, tokens, channels)
    return dict(name=name, tokens=tokens, channels=channels, kernel=7,
        all_chunk_partitions=all_partitions, input_bits=inputs, weight_bits=weights,
        initial_history_bits=initial, output_bits=output, history_after_each_bits=histories,
        final_history_bits=final)


def fixture(source_weights, provenance, contract_sha):
    inputs = [bits(((i*7+4) % 31-15)/8) for i in range(17*3)]
    weights = [bits(((i*11+3) % 31-15)/16) for i in range(7*3)]
    initial = [bits(((i*13+8) % 31-15)/4) for i in range(6*3)]
    cases = [make_case('asymmetric-nonzero-history', 17, 3, inputs, weights, initial)]
    impulse = [0]*28
    for index, value in [(0, 1.), (16, 2.), (7, -1.), (23, .5)]:
        impulse[index] = bits(value)
    taps = [bits((1 << i)/64) for i in range(7)] + [bits((-1 if i % 2 else 1)*(i+1)/128) for i in range(7)]
    cases.append(make_case('tap-order-impulses', 14, 2, impulse, taps, [0]*12))
    cases.append(make_case('authenticated-bf16-coefficients', 9, 4,
        [bits(((i*13+2) % 37-18)/32) for i in range(36)], source_weights, [0]*24, True))
    cases.append(make_case('empty-retains-history', 0, 4, [], source_weights,
        [bits((i-11)/16) for i in range(24)]))
    values = [.1, -.7, .2, 2**-20, -3.1, .75, 2**-12]
    cases.append(make_case('fp32-rounding', 17, 1, [bits(values[i % 7]) for i in range(17)],
        [0x3f800001, 0xbf000001, 0x3eaaaaab, 0xbe800001, 0x3f200001, 0xbf400001, 0x3dcccccd], [0]*6))
    return dict(format='toklet-short-conv7-exact-v1', kernel=7,
        contract_sha256=contract_sha, producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numerical_gate=dict(abs=1e-5, rel=1e-4), precision='Exact rational FP32 inputs/products/sums rounded directly to FP32 nearest-even',
        source_coefficients=provenance, cases=cases, complete_lfm3_graph_qualified=False,
        bf16_activation_profile_qualified=False, model_adoption_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'header', 'contract', 'fresh_output']:
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    need(not args.fresh_output.exists() and args.fresh_output.parent.is_dir(), 'Fresh fixture path required')
    contract = json.loads(args.contract.read_text())
    need(contract['base_commit'] == '4a3015d264c2547fae462d08b361e67d880d3495'
         and contract['before_new_implementation_and_fixture_outcomes'] is True
         and contract['numerical_gate'] == dict(abs=1e-5, rel=1e-4), 'Prospective seven-tap contract differs')
    weights, provenance = coefficients(args.model, args.header)
    result = fixture(weights, provenance, hashlib.sha256(args.contract.read_bytes()).hexdigest())
    with args.fresh_output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True, cases=len(result['cases']), source_coefficients=len(weights),
        fixture_sha256=hashlib.sha256(args.fresh_output.read_bytes()).hexdigest(), numerical_execution_qualified=False)))


if __name__ == '__main__':
    main()
