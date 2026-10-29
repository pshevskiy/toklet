"""Independent bounded direct-gain Half RMS; no source model or target imports.

Only generic word/Neumaier/serialization helpers are reused unchanged. This new
profile uses direct raw BF16 gains and EPS1e-5; old Gemma sources stay untouched.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

_helper_path = Path(__file__).with_name('gemma3_half_rms.py')
_spec = importlib.util.spec_from_file_location('_lfm_rms_word_helpers', _helper_path)
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
f32, half_word, half_value = helper.f32, helper.half_word, helper.half_value
float_word, float_value = helper.float_word, helper.float_value
neumaier, encoded, identity = helper.neumaier, helper.encoded, helper.identity
FORMAT = 'toklet-lfm2-half-rms-v1'
PROTOCOL = 'f2518543fc49139d4d982bcad3f0aad5bf4a41b99267aca75cf0cacea1cb3476'
EPSILON = f32(1e-5)
POLICY = dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3, fatal_reference='main')
CASES = (('direct_zero_gain',1,64,'zero_gain'),('coefficient_underflow',1,64,'coefficient'),
         ('epsilon_subnormal',2,64,'subnormal'),('heads64',512,64,'dyadic'),
         ('hidden256',32,256,'dyadic'),('hidden1024',32,1024,'dyadic'),('large1024',1,1024,'large'))
GUARD, INPUT_PADDING, OUTPUT_CANARY, COEFFICIENT_GUARD = 7, 0x7e55, 0x5a17, 0x5a170031
MAX_BINARY, MAX_JSON = 1024**2, 256*1024


def shape(rows, width):
    if type(rows) is not int or type(width) is not int or rows < 1 or not ((width == 64 and rows <= 512) or (width in (256,1024) and rows <= 32)):
        raise ValueError('LFM RMS shape profile')


def direct(gamma):
    values = [float_value(word << 16) for word in gamma]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite direct BF16 gain')
    return values


def operands(rows, width, pattern):
    shape(rows,width)
    if pattern not in ('zero_gain','coefficient','subnormal','dyadic','large'):
        raise ValueError('Fixed input pattern')
    cycle = (0xbf80,0,0x3f80,0x3980,0xbf00,0x3f81,0x3f7f)
    gamma = [cycle[i % len(cycle)] for i in range(width)]
    if pattern == 'zero_gain': gamma = [0]*width
    if pattern in ('coefficient','subnormal'): gamma = [0x3f80]*width
    if pattern == 'coefficient': gamma[0] = 0x3300  # Exact BF16 2^-25; Half narrowing becomes0.
    stride = width+3
    words = [INPUT_PADDING]*(rows*stride+GUARD)
    for row in range(rows):
        for column in range(width):
            if pattern == 'zero_gain': x = 1.
            elif pattern == 'coefficient': x = 1. if column == 0 else 0.
            elif pattern == 'subnormal': x = (2**-24 if row%2 == 0 else -2**-24) if column == 0 else 0.
            elif pattern == 'large': x = 65504. if column%2 == 0 else -65504.
            else: x = 0. if row == 0 else ((17*row+11*column)%33-16)/8
            words[row*stride+column] = half_word(x)
    return words,gamma,direct(gamma),stride


def rms(words, scales, rows, width, stride, profile):
    shape(rows,width)
    if profile not in ('main','wide') or stride < width:
        raise ValueError('RMS profile/stride')
    if len(scales) != width or len(words) < (rows-1)*stride+width:
        raise ValueError('RMS extent')
    if any(not math.isfinite(v) for v in scales):
        raise ValueError('Nonfinite direct coefficient')
    result=[]
    for row in range(rows):
        x=[half_value(w) for w in words[row*stride:row*stride+width]]
        if any(not math.isfinite(v) for v in x): raise ValueError('Nonfinite active input')
        if profile == 'main':
            total=neumaier([f32(v*v) for v in x])
            root=f32(math.sqrt(f32(f32(total/f32(width))+EPSILON)))
            inverse=f32(1/root)
            values=[f32(f32(v*inverse)*s) for v,s in zip(x,scales,strict=True)]
        else:
            inverse=1/math.sqrt(math.fsum(v*v for v in x)/width+EPSILON)
            values=[(v*inverse)*s for v,s in zip(x,scales,strict=True)]
        for value in values:
            if not math.isfinite(value): raise ValueError('Nonfinite RMS intermediate')
            result.append(half_word(value))
    return result


def errors():
    words,gamma,_,stride=operands(1,64,'coefficient')
    result=[]
    for name,word in [('active_nan',0x7e00),('active_infinity',0x7c00)]:
        input_=words[:];input_[0]=word
        result.append(dict(name=name,input_words=input_,gamma_words=gamma,rows=1,width=64,input_stride=stride,status_nonzero=True))
    for name,word in [('scale_nan',0x7fc0),('scale_infinity',0x7f80)]:
        invalid=gamma[:];invalid[0]=word
        result.append(dict(name=name,input_words=words,gamma_words=invalid,rows=1,width=64,input_stride=stride,status_nonzero=True))
    overflow=gamma[:];overflow[0]=0x4780
    result.append(dict(name='half_output_overflow',input_words=words,gamma_words=overflow,rows=1,width=64,input_stride=stride,status_nonzero=True))
    return result


def export(directory):
    before = identity(Path(__file__).read_bytes())
    helper_before = identity(Path(helper.__file__).read_bytes())
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
                if name == 'direct_zero_gain':
                    if any(main) or any(wide):
                        raise ValueError('Direct-zero gain witness differs')
                    exact.append(dict(row=0, column=0, expected_word=0, wrong_unit_offset_word=0x3c00))
                elif name == 'coefficient_underflow':
                    wrong = rms(words, [half_value(half_word(v)) for v in scales], rows, width, stride, 'main')
                    if (main[0], wide[0], wrong[0]) != (4, 4, 0):
                        raise ValueError('Direct gain narrowing witness differs')
                    exact.append(dict(row=0, column=0, expected_word=4, wrong_half_scale_word=0))
                elif name == 'epsilon_subnormal':
                    if (main[0], main[width], wide[0], wide[width]) != (0x013c, 0x813c, 0x013c, 0x813c):
                        raise ValueError('Epsilon/subnormal witness differs')
                    exact.extend([dict(row=0, column=0, expected_word=0x013c), dict(row=1, column=0, expected_word=0x813c)])
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
                    rms(control['input_words'], direct(control['gamma_words']), control['rows'],
                        control['width'], control['input_stride'], profile)
                except (ValueError, OverflowError):
                    continue
                raise ValueError('Invalid control unexpectedly finite')
        error_raw = encoded(dict(controls=invalid, output_bits_not_a_gate=True))
        (directory / 'errors.json').write_bytes(error_raw)
        files = {'arrays.bin': identity((directory / 'arrays.bin').read_bytes()), 'errors.json': identity(error_raw)}
        if identity(Path(__file__).read_bytes()) != before or identity(Path(helper.__file__).read_bytes()) != helper_before:
            raise ValueError('Producer endpoint changed')
        for leaf, pin in files.items():
            if identity((directory / leaf).read_bytes()) != pin:
                raise ValueError('Output endpoint changed')
        manifest = dict(format=FORMAT, protocol_sha256=PROTOCOL, producer=before, helper=dict(path=str(Path(helper.__file__).resolve()), **helper_before), policy=POLICY,
            epsilon_bits=float_word(EPSILON), guard=GUARD, input_padding_word=INPUT_PADDING,
            output_canary_word=OUTPUT_CANARY, coefficient_guard_word=COEFFICIENT_GUARD,
            cases=cases, files=files, finite_output_values=sum(c['rows'] * c['width'] for c in cases),
            source_reference_executed=True, target_gpu_executed=False,
            runtime=dict(python=sys.version, executable=str(Path(sys.executable).resolve())),
            scope='Isolated direct-gain LFM NativeHalf RMS, EPS1e-5; main-only fixed per-array gate plus exact witnesses; no whole graph, upstream intermediate-cast parity or quality qualification')
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
