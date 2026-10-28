"""Bounded independent Gemma3 Half RMS references; stdlib, no GPU/imported graph.

Main explicitly rounds each Neumaier/scalar step to FP32. Wide evaluates the
same Half inputs and FP32 effective scales in FP64. Both store Half outputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

FORMAT = 'toklet-gemma3-half-rms-v1'
PROTOCOL = '01d5b70be1a408b03dd849bd7b166a108b25489a55300bd779fa0e30d855e845'
POLICY = dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3, fatal_reference='main')
CASES = (('coefficient_rounding', 1, 32, 'rounding'), ('mixed32', 3, 32, 'dyadic'),
         ('hidden64', 4, 64, 'dyadic'), ('heads256', 128, 256, 'dyadic'),
         ('hidden640', 32, 640, 'dyadic'), ('large640', 1, 640, 'large'))
GUARD = 7
INPUT_PADDING = 0x7e55
OUTPUT_CANARY = 0x5a17
COEFFICIENT_GUARD = 0x5a170031
MAX_BINARY = 1024**2
MAX_JSON = 256 * 1024


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def half_word(x):
    return struct.unpack('<H', struct.pack('<e', x))[0]


def half_value(word):
    return struct.unpack('<e', struct.pack('<H', word))[0]


def float_word(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def float_value(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def gamma_value(word):
    x = float_value(word << 16)
    if not math.isfinite(x):
        raise ValueError('Nonfinite raw gamma')
    return x


def effective(gamma):
    return [f32(1 + gamma_value(word)) for word in gamma]


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def operands(rows, width, pattern):
    if type(rows) is not int or not 1 <= rows <= 128 or width not in (32, 64, 256, 640):
        raise ValueError('Fixed RMS shape')
    if pattern not in ('rounding', 'dyadic', 'large'):
        raise ValueError('Fixed input pattern')
    # -1,0,1,2^-12,-.5, BF16 neighbors immediately above/below -1.
    cycle = (0xbf80, 0, 0x3f80, 0x3980, 0xbf00, 0xbf7f, 0xbf81)
    gamma = [cycle[i % len(cycle)] for i in range(width)]
    if pattern == 'rounding':
        gamma = [0x3980] + [0] * (width - 1)
    stride = width + 3
    words = [INPUT_PADDING] * (rows * stride + GUARD)
    for row in range(rows):
        for column in range(width):
            if pattern == 'rounding':
                x = 2.0 if column == 0 else 1.0
            elif pattern == 'large':
                x = 65504.0 if column % 2 == 0 else -65504.0
            else:
                x = 0.0 if row == 0 else ((17 * row + 11 * column) % 33 - 16) / 8
            words[row * stride + column] = half_word(x)
    return words, gamma, effective(gamma), stride


def neumaier(values):
    total, correction = 0.0, 0.0
    for value in values:
        value = f32(value)
        next_ = f32(total + value)
        if abs(total) >= abs(value):
            error = f32(f32(total - next_) + value)
        else:
            error = f32(f32(value - next_) + total)
        correction = f32(correction + error)
        total = next_
    return f32(total + correction)


def rms(words, scales, rows, width, stride, profile):
    if profile not in ('main', 'wide') or rows <= 0 or width <= 0 or stride < width:
        raise ValueError('RMS profile/shape')
    if len(scales) != width or len(words) < (rows - 1) * stride + width:
        raise ValueError('RMS extent')
    if any(not math.isfinite(x) for x in scales):
        raise ValueError('Nonfinite effective coefficient')
    result = []
    epsilon = f32(1e-6)
    for row in range(rows):
        x = [half_value(word) for word in words[row * stride:row * stride + width]]
        if any(not math.isfinite(v) for v in x):
            raise ValueError('Nonfinite active input')
        if profile == 'main':
            total = neumaier([f32(v * v) for v in x])
            root = f32(math.sqrt(f32(f32(total / f32(width)) + epsilon)))
            inverse = f32(1 / root)
            values = [f32(f32(v * inverse) * scale) for v, scale in zip(x, scales, strict=True)]
        else:
            inverse = 1 / math.sqrt(math.fsum(v * v for v in x) / width + epsilon)
            values = [(v * inverse) * scale for v, scale in zip(x, scales, strict=True)]
        for value in values:
            if not math.isfinite(value):
                raise ValueError('Nonfinite RMS intermediate')
            result.append(half_word(value))  # OverflowError is an expected invalid-control outcome.
    return result


def errors():
    words, gamma, _, stride = operands(1, 32, 'rounding')
    result = []
    for name, word in [('active_nan', 0x7e00), ('active_infinity', 0x7c00)]:
        input_ = words[:]
        input_[0] = word
        result.append(dict(name=name, input_words=input_, gamma_words=gamma,
                           rows=1, width=32, input_stride=stride, status_nonzero=True))
    overflow_gamma = gamma[:]
    overflow_gamma[0] = 0x4780  # BF16 65536; FP32 effective scale65537.
    result.append(dict(name='half_output_overflow', input_words=words, gamma_words=overflow_gamma,
                       rows=1, width=32, input_stride=stride, status_nonzero=True))
    return result


def export(directory):
    before = identity(Path(__file__).read_bytes())
    directory.mkdir(parents=False, exist_ok=False)
    cases = []
    try:
        with (directory / 'arrays.bin').open('xb') as binary, (directory / 'progress.jsonl').open('x') as progress:
            def save(values, kind, shape):
                raw = b''.join(struct.pack('<' + kind, v) for v in values)
                begin = binary.tell()
                if begin + len(raw) > MAX_BINARY:
                    raise ValueError('Fixed binary bound')
                binary.write(raw)
                return dict(file='arrays.bin', offset=begin, shape=shape,
                            dtype={'H': 'u16le', 'I': 'u32le'}[kind], **identity(raw))
            for name, rows, width, pattern in CASES:
                progress.write(json.dumps(dict(case=name, started=True)) + '\n')
                progress.flush()
                words, gamma, scales, stride = operands(rows, width, pattern)
                main = rms(words, scales, rows, width, stride, 'main')
                wide = rms(words, scales, rows, width, stride, 'wide')
                envelope = max(abs(half_value(a) - half_value(b)) for a, b in zip(main, wide, strict=True))
                exact = []
                if name == 'coefficient_rounding':
                    wrong_scales = [half_value(half_word(v)) for v in scales]
                    wrong = rms(words, wrong_scales, rows, width, stride, 'main')
                    if (main[0], wide[0], wrong[0]) != (0x3fa7, 0x3fa7, 0x3fa6):
                        raise ValueError('Prospective exact coefficient witness differs')
                    exact.append(dict(row=0, column=0, expected_word=0x3fa7, wrong_half_scale_word=0x3fa6))
                descriptors = dict(input=save(words, 'H', [len(words)]), gamma=save(gamma, 'H', [width]),
                    scales=save([float_word(v) for v in scales] + [COEFFICIENT_GUARD] * GUARD, 'I', [width + GUARD]),
                    main=save(main, 'H', [rows, width]), wide=save(wide, 'H', [rows, width]))
                cases.append(dict(name=name, rows=rows, width=width, input_stride=stride,
                    output_stride=width + 5, output_elements=rows * (width + 5) + GUARD,
                    arrays=descriptors, envelope=envelope,
                    gate=dict(atol=max(POLICY['atol_floor'], POLICY['envelope_factor'] * envelope), rtol=POLICY['rtol']),
                    exact=exact))
                binary.flush()
                progress.write(json.dumps(dict(case=name, complete=True, arrays_end=binary.tell())) + '\n')
                progress.flush()
        invalid = errors()
        for control in invalid:
            for profile in ('main', 'wide'):
                try:
                    rms(control['input_words'], effective(control['gamma_words']), control['rows'],
                        control['width'], control['input_stride'], profile)
                except (ValueError, OverflowError):
                    continue
                raise ValueError('Invalid control unexpectedly finite')
        error_raw = encoded(dict(controls=invalid, output_bits_not_a_gate=True))
        (directory / 'errors.json').write_bytes(error_raw)
        files = {'arrays.bin': identity((directory / 'arrays.bin').read_bytes()), 'errors.json': identity(error_raw)}
        if identity(Path(__file__).read_bytes()) != before:
            raise ValueError('Producer endpoint changed')
        for leaf, pin in files.items():
            if identity((directory / leaf).read_bytes()) != pin:
                raise ValueError('Output endpoint changed')
        manifest = dict(format=FORMAT, protocol_sha256=PROTOCOL, producer=before, policy=POLICY,
            epsilon_bits=float_word(f32(1e-6)), guard=GUARD, input_padding_word=INPUT_PADDING,
            output_canary_word=OUTPUT_CANARY, coefficient_guard_word=COEFFICIENT_GUARD,
            cases=cases, files=files, finite_output_values=sum(c['rows'] * c['width'] for c in cases),
            source_reference_executed=True, target_gpu_executed=False,
            runtime=dict(python=sys.version, executable=str(Path(sys.executable).resolve())),
            scope='Isolated independent Half RMS; exact chosen scale witness plus main-only per-array gate; no whole graph or upstream runtime parity')
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
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir)))
