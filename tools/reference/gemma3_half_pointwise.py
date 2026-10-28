"""Independent Half GELU/gated and embedding-scale fixtures; stdlib only.

This fixture does not import target code or execute an inference graph. Main
rounds every prescribed scalar step to FP32; wide uses independent FP64 math.
Both preserve the explicit Half activation store before the gated product.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

FORMAT = 'toklet-gemma3-half-pointwise-v1'
PROTOCOL = 'fd84f903bc1f078b3413e4a6ff1a195f404ab3eac723fd111761276c3265e45b'
POLICY = dict(fatal_reference='main', atol_floor=1e-4, envelope_factor=2, rtol=1e-3)
CASES = (('gelu_tiny', 'gelu_gated', 3, 128, None),
         ('gelu_real_width', 'gelu_gated', 32, 2048, None),
         ('scale_tiny', 'scale', 3, 64, 8.0),
         ('scale_real_width', 'scale', 32, 640, 25.25))
GUARD, INPUT_PADDING, OUTPUT_CANARY = 7, 0x7e55, 0x5a17
MAX_BINARY, MAX_JSON = 2 * 1024**2, 128 * 1024


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def half_word(x):
    return struct.unpack('<H', struct.pack('<e', x))[0]


def half_value(word):
    return struct.unpack('<e', struct.pack('<H', word))[0]


def float_word(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def gelu_unstored(x, profile):
    if not math.isfinite(x) or profile not in ('main', 'wide'):
        raise ValueError('GELU finite/profile contract')
    if profile == 'main':
        cube = f32(f32(x * x) * x)
        z = f32(f32(math.sqrt(2 / math.pi)) * f32(x + f32(f32(.044715) * cube)))
        return f32(f32(.5 * x) * f32(1 + f32(math.tanh(z))))
    return .5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + .044715 * x * x * x)))


def gelu_product(x_word, up_word, profile):
    x, up = half_value(x_word), half_value(up_word)
    if not math.isfinite(up):
        raise ValueError('Nonfinite up operand')
    activation = half_word(gelu_unstored(x, profile))
    product = half_value(activation) * up
    if profile == 'main':
        product = f32(product)
    return activation, half_word(product)


def scale_word(x_word, factor, profile):
    x = half_value(x_word)
    if not math.isfinite(x) or factor not in (8.0, 25.25) or profile not in ('main', 'wide'):
        raise ValueError('Scale finite/factor/profile contract')
    value = x * f32(factor)
    return half_word(f32(value) if profile == 'main' else value)


def operands(operation, rows, width):
    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError('Rows bound')
    allowed = {'gelu_gated': (128, 2048), 'scale': (64, 640)}
    if operation not in allowed or width not in allowed[operation]:
        raise ValueError('Operation width')
    xs = (-8., -4., -2., -1., -.5, -.25, -0.0, 0.0, .25, .5, 1., 2., 4., 8., 2**-24, -2**-24, 2**-14, -2**-14)
    ups = (-2., -1., .5, 1., 1.5, 3., 5., 7.)
    stride = width + 3
    words = [INPUT_PADDING] * (rows * stride + GUARD)
    up = [INPUT_PADDING] * (rows * width + GUARD) if operation == 'gelu_gated' else None
    for row in range(rows):
        for col in range(width):
            words[row * stride + col] = half_word(xs[(row * 17 + col * 7) % len(xs)])
            if up is not None:
                up[row * width + col] = half_word(ups[(row * 3 + col) % len(ups)])
    words[0:3] = [0x3400, 0x0002, 0x8002] if operation == 'gelu_gated' else [0x3c00, 0x0001, 0x8001]
    if up is not None:
        up[0:3] = [0x4500] * 3
    return words, up, stride


def evaluate(operation, rows, width, words, up, stride, factor, profile):
    if stride < width or len(words) < (rows - 1) * stride + width:
        raise ValueError('Input extent')
    if operation == 'gelu_gated' and (up is None or len(up) < rows * width):
        raise ValueError('Up extent')
    arrays = {'activation': [], 'product': []} if operation == 'gelu_gated' else {'scaled': []}
    for row in range(rows):
        for col in range(width):
            word = words[row * stride + col]
            if operation == 'gelu_gated':
                activation, product = gelu_product(word, up[row * width + col], profile)
                arrays['activation'].append(activation)
                arrays['product'].append(product)
            elif operation == 'scale':
                arrays['scaled'].append(scale_word(word, factor, profile))
            else:
                raise ValueError('Unknown operation')
    return arrays


def exact_controls(operation, factor):
    if operation == 'gelu_gated':
        return [dict(row=0, column=i, activation=a, product=p) for i, a, p in
                [(0, 0x30ca, 0x39fc), (1, 0x0001, 0x0005), (2, 0x8001, 0x8005)]]
    sub = 8 if factor == 8 else 25
    return [dict(row=0, column=i, scaled=value) for i, value in
            [(0, half_word(factor)), (1, sub), (2, 0x8000 | sub)]]


def error_controls():
    return [dict(operation='gelu_gated', input_word=x, up_word=0x3c00, status_nonzero=True)
            for x in (0x7e00, 0x7c00)] + [
        dict(operation='gelu_gated', input_word=0x7bff, up_word=0x4000, status_nonzero=True),
        dict(operation='gelu_gated', input_word=0x3c00, up_word=0x7e00, status_nonzero=True),
        *[dict(operation='scale', input_word=x, factor=25.25, status_nonzero=True)
          for x in (0x7e00, 0x7c00, 0x7bff)]]


def ordering_controls():
    x = f32(1 + 2**-11)
    gathered = half_word(x)
    correct = scale_word(gathered, 25.25, 'main')
    wrong_scale = half_word(f32(x * 25.25))
    wrong_gelu = half_word(f32(gelu_unstored(.25, 'main') * 5))
    if (gathered, correct, wrong_scale, wrong_gelu) != (0x3c00, 0x4e50, 0x4e51, 0x39fd):
        raise ValueError('Prospectively frozen ordering witnesses differ')
    return dict(gather_input_f32_word=float_word(x), gathered_half_word=gathered,
        scaled_half_word=correct, wrong_scale_before_gather_half_word=wrong_scale,
        wrong_delayed_gelu_half_product=wrong_gelu,
        scope='Scalar source ordering witness; leaf Half API alone does not prove actual gather implementation')


def export(directory):
    before = identity(Path(__file__).read_bytes())
    directory.mkdir(parents=False, exist_ok=False)
    cases = []
    try:
        with (directory / 'arrays.bin').open('xb') as binary, (directory / 'progress.jsonl').open('x') as progress:
            def save(words, shape):
                raw = b''.join(struct.pack('<H', word) for word in words)
                offset = binary.tell()
                if offset + len(raw) > MAX_BINARY:
                    raise ValueError('Binary bound')
                binary.write(raw)
                return dict(file='arrays.bin', dtype='u16le', offset=offset, shape=shape, **identity(raw))
            for name, operation, rows, width, factor in CASES:
                progress.write(json.dumps(dict(case=name, started=True)) + '\n')
                progress.flush()
                words, up, stride = operands(operation, rows, width)
                main = evaluate(operation, rows, width, words, up, stride, factor, 'main')
                wide = evaluate(operation, rows, width, words, up, stride, factor, 'wide')
                exact = exact_controls(operation, factor)
                for point in exact:
                    index = point['row'] * width + point['column']
                    for stage, expected in point.items():
                        if stage not in ('row', 'column') and (main[stage][index] != expected or wide[stage][index] != expected):
                            raise ValueError('Exact prospective word differs: ' + name + '/' + stage)
                inputs = dict(input=save(words, [len(words)]))
                if up is not None:
                    inputs['up'] = save(up, [len(up)])
                arrays = {}
                for stage in main:
                    envelope = max(abs(half_value(a) - half_value(b)) for a, b in zip(main[stage], wide[stage], strict=True))
                    arrays[stage] = dict(main=save(main[stage], [rows, width]), wide=save(wide[stage], [rows, width]),
                        envelope=envelope, gate=dict(atol=max(POLICY['atol_floor'], 2 * envelope), rtol=POLICY['rtol']))
                cases.append(dict(name=name, operation=operation, rows=rows, width=width, factor=factor,
                    input_stride=stride, output_stride=width + 5, output_elements=rows * (width + 5) + GUARD,
                    inputs=inputs, arrays=arrays, exact=exact))
                binary.flush()
                progress.write(json.dumps(dict(case=name, complete=True, arrays_end=binary.tell())) + '\n')
                progress.flush()
        errors = error_controls()
        for item in errors:
            for profile in ('main', 'wide'):
                try:
                    if item['operation'] == 'scale':
                        scale_word(item['input_word'], item['factor'], profile)
                    else:
                        gelu_product(item['input_word'], item['up_word'], profile)
                except (ValueError, OverflowError):
                    continue
                raise ValueError('Invalid control unexpectedly finite')
        controls = encoded(dict(errors=errors, ordering=ordering_controls(), error_output_words_not_a_gate=True))
        (directory / 'controls.json').write_bytes(controls)
        files = {'arrays.bin': identity((directory / 'arrays.bin').read_bytes()), 'controls.json': identity(controls)}
        if identity(Path(__file__).read_bytes()) != before:
            raise ValueError('Producer endpoint changed')
        for leaf, pin in files.items():
            if identity((directory / leaf).read_bytes()) != pin:
                raise ValueError('Output endpoint changed')
        manifest = dict(format=FORMAT, protocol_sha256=PROTOCOL, producer=before, policy=POLICY,
            constants_f32_bits=dict(c=float_word(f32(math.sqrt(2 / math.pi))), k=float_word(f32(.044715))),
            cases=cases, files=files, guard=GUARD, input_padding_word=INPUT_PADDING, output_canary_word=OUTPUT_CANARY,
            values_per_profile=sum(c['rows'] * c['width'] * len(c['arrays']) for c in cases),
            runtime=dict(python=sys.version, executable=str(Path(sys.executable).resolve())),
            source_reference_executed=True, target_gpu_executed=False,
            scope='Isolated Half GELU then Half product and Half scale; no complete graph/HF runtime parity/performance')
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
