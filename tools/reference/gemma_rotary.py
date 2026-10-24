"""Independent mathematical proportional rotary witness, not upstream parity.

Decimal80 constructs frequencies and angles from exact FP32 base/factor inputs.
Angles then round to FP64 for libm trig and FP64 pair arithmetic. Inactive
coordinates copy exactly. Long-position upstream disagreements stay separate.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys

import reference_runtime as runtime

FORMAT = 'toklet-gemma-rotary-math-v1'
MAX_JSON = 1024 * 1024
MAX_ELEMENTS = 8192
MAX_POSITION = (1 << 53)
PRECISION = 'Exact FP32 inputs; Decimal80 frequency/angle; FP64 trig and pair output'
SOURCE_REVISION = '0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
SOURCE_SHA256 = 'ccba2f0c7b246c008a1188dc3760c162bdd906b15a1c490461dee4be64bc62df'
KEYS = ('tokens', 'heads', 'dims', 'rotated_dims', 'offset', 'base', 'factor')
SPECS = (
    ('empty', (0, 2, 8, 2, 0, 10000., 1.)),
    ('position_zero', (1, 1, 8, 2, 0, 1e6, 1.)),
    ('small_half_pairs', (3, 2, 16, 4, 2, 256., 1.)),
    ('full_rotation', (4, 3, 8, 8, 7, 16., 2.)),
    ('zero_rotation', (2, 2, 8, 0, 5, 10000., 1.)),
    ('actual_short', (3, 2, 512, 128, 123, 1e6, 1.)),
    ('actual_context_end', (3, 2, 512, 128, 131069, 1e6, 1.)),
    ('frequency_factor', (5, 1, 16, 4, 11, 10000., 2.5)),
)


def encoded(data):
    return (json.dumps(data, indent=2, allow_nan=False) + '\n').encode()


def fp32(value):
    if type(value) not in (float, int):
        raise ValueError('Finite non-Boolean number required')
    try:
        if not math.isfinite(value):
            raise ValueError('Finite number required')
        result = struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError('Number exceeds FP32') from error
    if result != value:
        raise ValueError('Exactly representable FP32 value required')
    return result


def geometry(tokens, heads, dims, rotated_dims, offset, base, factor):
    if (any(type(v) is not int for v in (tokens, heads, dims, rotated_dims, offset))
            or not 0 <= tokens <= 8 or not 1 <= heads <= 4 or not 2 <= dims <= 512
            or dims % 2 or not 0 <= rotated_dims <= dims or rotated_dims % 2
            or tokens * heads * dims > MAX_ELEMENTS):
        raise ValueError('Bounded even proportional rotary geometry required')
    if offset < 0 or offset > MAX_POSITION or (tokens and offset + tokens - 1 > MAX_POSITION):
        raise ValueError('Rotary positions exceed exact FP64 integer range')
    base, factor = fp32(base), fp32(factor)
    if not 1 <= base <= 1e6 or not 0.125 <= factor <= 8:
        raise ValueError('Bounded positive reference base/factor required')
    return base, factor


def math_values(input_values, tokens, heads, dims, rotated_dims, offset, base, factor):
    base, factor = geometry(tokens, heads, dims, rotated_dims, offset, base, factor)
    if not isinstance(input_values, list) or len(input_values) != tokens * heads * dims:
        raise ValueError('Exact contiguous [tokens,heads,dims] input required')
    values = [fp32(value) for value in input_values]
    output = list(values)
    frequencies = []
    with localcontext() as context:
        context.prec = 80
        for pair in range(rotated_dims // 2):
            frequency = Decimal.from_float(factor) * Decimal.from_float(base) ** (Decimal(2 * pair) / Decimal(dims))
            frequencies.append(float(frequency))
            for token in range(tokens):
                # Construct the absolute integer position before division.
                angle = float(Decimal(offset + token) / frequency)
                sine, cosine = math.sin(angle), math.cos(angle)
                for head in range(heads):
                    first = (token * heads + head) * dims + pair
                    second = first + dims // 2
                    a, b = values[first], values[second]
                    output[first] = a * cosine - b * sine
                    output[second] = a * sine + b * cosine
    if not all(math.isfinite(value) for value in frequencies + output):
        raise ValueError('Nonfinite mathematical rotary result')
    return frequencies, output


def export():
    cases = []
    for name, spec in SPECS:
        params = dict(zip(KEYS, spec))
        count = params['tokens'] * params['heads'] * params['dims']
        values = [(((i * 7 + 4) % 31) - 15) * 0.125 for i in range(count)]
        frequencies, output = math_values(values, **params)
        shape = [params[key] for key in ('tokens', 'heads', 'dims')]
        arrays = {
            'input': dict(dtype='<f4', shape=shape, values=values),
            'frequencies': dict(dtype='<f8', shape=[params['rotated_dims']//2], values=frequencies),
            'output': dict(dtype='<f8', shape=shape, values=output),
        }
        cases.append(dict(name=name, **params, arrays=arrays))
    return encoded(dict(format=FORMAT, precision=PRECISION,
        source_revision=SOURCE_REVISION, source_sha256=SOURCE_SHA256,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(python=platform.python_version()), cases=cases,
        upstream_parity_qualified=False, native_graph_implemented=False, device_qualified=False))


def verify(path, expected_sha):
    raw = runtime.captured(path, MAX_JSON, runtime.sha(expected_sha))
    data = json.loads(raw, object_pairs_hook=runtime.json_unique,
        parse_constant=runtime.bad_json_number, parse_float=runtime.finite_json_number)
    fields = {'format', 'precision', 'source_revision', 'source_sha256', 'producer_sha256',
        'runtime', 'cases', 'upstream_parity_qualified', 'native_graph_implemented', 'device_qualified'}
    if (not isinstance(data, dict) or set(data) != fields or data['format'] != FORMAT
            or data['precision'] != PRECISION or data['source_revision'] != SOURCE_REVISION
            or data['source_sha256'] != SOURCE_SHA256
            or data['runtime'] != dict(python='3.12.8')
            or any(data[key] is not False for key in ('upstream_parity_qualified', 'native_graph_implemented', 'device_qualified'))
            or not isinstance(data['cases'], list) or len(data['cases']) != len(SPECS)):
        raise ValueError('Rotary mathematical reference identity/scope differs')
    runtime.sha(data['producer_sha256'])
    for row, (name, spec) in zip(data['cases'], SPECS):
        if (not isinstance(row, dict) or set(row) != {'name', 'arrays', *KEYS}
                or row['name'] != name or tuple(row[key] for key in KEYS) != spec):
            raise ValueError('Rotary case geometry differs')
        geometry(*(row[key] for key in KEYS))
        if not isinstance(row['arrays'], dict) or set(row['arrays']) != {'input', 'frequencies', 'output'}:
            raise ValueError('Rotary array inventory differs')
        for key, item in row['arrays'].items():
            shape = [row['rotated_dims']//2] if key == 'frequencies' else [row[k] for k in ('tokens', 'heads', 'dims')]
            if (not isinstance(item, dict) or set(item) != {'dtype', 'shape', 'values'}
                    or item['dtype'] != ('<f4' if key == 'input' else '<f8')
                    or item['shape'] != shape or not isinstance(item['shape'], list)
                    or any(type(n) is not int for n in item['shape'])
                    or not isinstance(item['values'], list) or len(item['values']) != math.prod(shape)):
                raise ValueError('Rotary array descriptor differs')
            for value in item['values']:
                if type(value) not in (float, int):
                    raise ValueError('Finite non-Boolean rotary value required')
                try:
                    valid = math.isfinite(value)
                except OverflowError:
                    valid = False
                if not valid:
                    raise ValueError('Finite rotary value required')
                if key == 'input':
                    fp32(value)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with args.output.open('xb') as stream:
        stream.write(export())
