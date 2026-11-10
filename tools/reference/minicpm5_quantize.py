"""Bounded pinned-MLX affine8 reconstruction; no model loading or inference."""
import hashlib
import struct

ROWS = 256
MAX_COLUMNS = 6144
GROUP = 64


def need(value, message):
    if not value:
        raise ValueError(message)


def geometry(raw, rows, columns):
    need(type(raw) is bytes and type(rows) is int and 1 <= rows <= ROWS
         and type(columns) is int and GROUP <= columns <= MAX_COLUMNS
         and columns % GROUP == 0 and len(raw) == rows*columns*2,
         'Bounded complete BF16 rows required')
    return rows*columns


def quantize(raw, rows, columns, device, np, mx):
    """BF16 [rows,cols] -> U32 codes and BF16 [rows,cols/64] metadata.

    The caller admits the numerical runtime before obtaining np/mx. Array.view
    preserves every source bit, including values not representable as FP16.
    Output byte buffers are returned only after completed evaluation/readback.
    """
    geometry(raw, rows, columns)
    need(device in ['cpu', 'gpu'], 'Explicit quantization device required')
    values = np.frombuffer(raw, dtype='<u2').reshape(rows, columns)
    need(not np.any((values & 0x7f80) == 0x7f80), 'Nonfinite BF16 input')
    target = mx.cpu if device == 'cpu' else mx.gpu
    source = mx.array(values).view(mx.bfloat16, stream=target)
    packed, scales, biases = mx.quantize(source, group_size=GROUP, bits=8,
                                        mode='affine', stream=target)
    mx.eval(packed, scales, biases)
    mx.synchronize(target)
    need(packed.dtype == mx.uint32 and scales.dtype == biases.dtype == mx.bfloat16
         and tuple(packed.shape) == (rows, columns//4)
         and tuple(scales.shape) == tuple(biases.shape) == (rows, columns//GROUP),
         'Upstream affine8 output representation differs')
    result = dict(weight=np.asarray(packed).astype('<u4', copy=False).tobytes(),
                  scales=np.asarray(scales.view(mx.uint16, stream=target)).astype('<u2', copy=False).tobytes(),
                  biases=np.asarray(biases.view(mx.uint16, stream=target)).astype('<u2', copy=False).tobytes())
    for key in ['scales', 'biases']:
        values = np.frombuffer(result[key], dtype='<u2')
        need(not np.any((values & 0x7f80) == 0x7f80), 'Nonfinite affine metadata')
    return result


def bf16_exact(values):
    """Literal controls use only exactly BF16-representable finite values."""
    encoded = []
    for value in values:
        bits = struct.unpack('<I', struct.pack('<f', value))[0]
        need(bits & 0xffff == 0 and bits & 0x7f800000 != 0x7f800000,
             'Literal control is not exactly finite BF16')
        encoded.append(bits >> 16)
    return struct.pack('<'+'H'*len(encoded), *encoded)


def controls():
    """Hand-derived controls, fixed before either upstream device executes.

    The exact scale2 tie case separates GPU round-away from CPU rint-even.
    Negative-only groups expose the GPU maximum's initial zero. Metadata is
    literal little-endian BF16; codes are literal low-to-high packed bytes.
    """
    zero = bytes(64)
    ramp = list(range(-128, -96)) + list(range(96, 128))
    ties = [-256] + [-255+2*i for i in range(62)] + [254]
    cases = [
        ('zeros', [0]*64, zero, 'd7b3', '0000', zero, 'd7b3'),
        ('ones', [1]*64, zero, 'd7b3', '803f', zero, 'd7b3'),
        ('negative_ones', [-1]*64, zero, '813b', '80bf', zero, 'd7b3'),
        ('exact_packing', ramp, bytes(list(range(32))+list(range(224, 256))),
         '803f', '00c3', bytes(list(range(32))+list(range(224, 256))), '803f'),
        ('round_ties', ties, bytes([0]+list(range(1, 63))+[255]), '0040', '80c3',
         bytes([0]+[i+i%2 for i in range(62)]+[255]), '0040'),
        ('above_fp16_range', [65536]*64, zero, 'd7b3', '8047', zero, 'd7b3'),
    ]
    return [dict(name=name, rows=1, columns=64, input_bf16=bf16_exact(values).hex(),
                 gpu=dict(weight=codes.hex(), scales=scale, biases=bias),
                 cpu=dict(weight=cpu_codes.hex(), scales=cpu_scale, biases=bias))
            for name, values, codes, scale, bias, cpu_codes, cpu_scale in cases]


def compare(actual, expected):
    need(type(actual) is dict and set(actual) == {'weight', 'scales', 'biases'},
         'All three affine buffers required')
    rows = {}
    for key in ['weight', 'scales', 'biases']:
        got, want = actual[key], expected[key]
        need(type(got) is bytes and type(want) is bytes and len(got) == len(want),
             'Affine buffer byte extent differs')
        mismatches = 0 if got == want else sum(a != b for a, b in zip(got, want))
        rows[key] = dict(bytes=len(got), mismatching_bytes=mismatches,
                         reconstructed_sha256=hashlib.sha256(got).hexdigest(),
                         expected_sha256=hashlib.sha256(want).hexdigest())
    return rows
