"""Check literal pipeline probes; subnormal outputs stay descriptive."""
import json
import math
import struct
from fractions import Fraction
from check_gemma_affine_admission import rounded

SENTINEL = 0x5a170031
CASES = [
    ('subnormal-positive', 2.**24, [2.**-149, 0.]),
    ('subnormal-negative', 2.**24, [-2.**-149, 0.]),
    ('two-subnormal-products', .5, [2.**-126, 2.**-126]),
    ('single-subnormal-product', .5, [2.**-126, 0.]),
    ('normal-residual', 1.0078125, [1+2.**-23, -1.]),
    ('positive-zero', 1., [0., 0.]),
]


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def verify(events):
    if len(events) != 14 or events[0][0] != 'META' or events[-1] != ('COMPLETE', dict(cases=6, repeats=2, coupled_graph_qualified=False)):
        raise ValueError('Complete ordered pipeline probe inventory required')
    meta = events[0][1]
    width = meta['simd_width']
    if (type(width) is not int or not 1 <= width <= 64 or width & (width-1)
            or not isinstance(meta['device'], str) or not 1 <= len(meta['device']) <= 256
            or {k: v for k, v in meta.items() if k not in ['device', 'simd_width']} !=
            dict(math_mode='Safe', functions='Precise', language='2.3', cases=6, coupled_graph_qualified=False)):
        raise ValueError('Queried precise pipeline environment required')
    observations = []
    owned = None
    for index, (name, weight, first) in enumerate(CASES):
        inputs = first+[0.]*62
        ieee = bits(rounded(sum((Fraction(weight)*Fraction(v) for v in inputs), Fraction())))
        baseline = None
        for repeat in range(2):
            kind, record = events[1+index*2+repeat]
            expected = dict(name=name, repeat=repeat, completed=True, error=None,
                            weight_bits=bits(weight), input_bits=[bits(v) for v in inputs])
            if kind != 'CASE' or any(record[k] != value for k, value in expected.items()):
                raise ValueError('Literal projection probe operands/order differ')
            output = record['output_bits']
            if (type(output) is not list or len(output) != 1 or type(output[0]) is not int
                    or not 0 <= output[0] <= 0xffffffff
                    or not math.isfinite(struct.unpack('<f', struct.pack('<I', output[0]))[0])):
                raise ValueError('Completed finite projection output required')
            expected_device = dict(codes=[0]*8+[SENTINEL]*7, scales=[0]+[SENTINEL]*7,
                biases=[bits(weight) >> 16]+[SENTINEL]*7, input=[bits(v) for v in inputs]+[SENTINEL]*7,
                output=output+[SENTINEL]*7, status=[0]+[SENTINEL]*7)
            if record['device'] != expected_device:
                raise ValueError('Projection buffers or guards differ')
            if type(record['owned_bytes']) is not int or record['owned_bytes'] < 472:
                raise ValueError('Projection owner accounting missing')
            if owned is not None and record['owned_bytes'] != owned:
                raise ValueError('Projection owner accounting changed')
            owned = record['owned_bytes']
            if baseline is not None and baseline != output:
                raise ValueError('Projection repeat changed')
            baseline = output
        if index >= 4 and baseline != [ieee]:
            raise ValueError('Normal residual/zero control differs from exact result')
        observations.append(dict(name=name, output_bits=baseline[0], ieee_bits=ieee,
            observation='ieee_output' if baseline[0] == ieee else
                        'zero_output' if baseline[0] & 0x7fffffff == 0 else 'other_finite_output'))
    return dict(passed=True, simd_width=width, device=meta['device'], observations=observations,
                normal_controls_exact=True, repeats=2, guards_exact=True, owned_bytes=owned,
                general_flush_semantics_qualified=False, coupled_graph_qualified=False)


def transcript(raw):
    if type(raw) is not bytes or len(raw) > 1024**2:
        raise ValueError('Bounded pipeline transcript required')
    events = []
    for line in raw.decode().splitlines():
        if line.startswith('GEMMA_NATIVE_PROJECTION_'):
            tag, body = line.removeprefix('GEMMA_NATIVE_PROJECTION_').split(' ', 1)
            events.append((tag, json.loads(body)))
    return verify(events)
