"""Bounded Granite affine4/8 conversion controls; no model loading or imports.

BF16 complete rows -> low-bit-first U32 codes and BF16 group-64 metadata.
The caller must admit NumPy/MLX and its device before calling numerical code.
The existing MiniCPM5 affine8/6144-column contract stays unchanged.
"""
import struct

from minicpm5_quantize import bf16_exact, compare

ROWS = 256
MAX_COLUMNS = 8192
GROUP = 64


def need(value, message):
    if not value:
        raise ValueError(message)


def shape(rows, columns, bits):
    need(type(rows) is int and 1 <= rows <= ROWS
         and type(columns) is int and GROUP <= columns <= MAX_COLUMNS
         and columns % GROUP == 0 and type(bits) is int and bits in (4, 8),
         'Bounded complete group-64 affine4/8 rows required')
    return rows * columns


def geometry(raw, rows, columns, bits):
    count = shape(rows, columns, bits)
    need(type(raw) is bytes and len(raw) == count * 2,
         'Exact complete BF16 byte extent required')
    return count


def quantize(raw, rows, columns, bits, device, np, mx):
    """Return completed, little-endian buffers without a full-model owner."""
    geometry(raw, rows, columns, bits)
    need(device in ('cpu', 'gpu'), 'Explicit quantization device required')
    values = np.frombuffer(raw, dtype='<u2').reshape(rows, columns)
    need(not np.any((values & 0x7f80) == 0x7f80), 'Nonfinite BF16 input')
    target = mx.cpu if device == 'cpu' else mx.gpu
    source = mx.array(values).view(mx.bfloat16, stream=target)
    packed, scales, biases = mx.quantize(source, group_size=GROUP, bits=bits,
                                        mode='affine', stream=target)
    mx.eval(packed, scales, biases)
    mx.synchronize(target)
    need(packed.dtype == mx.uint32 and scales.dtype == biases.dtype == mx.bfloat16
         and tuple(packed.shape) == (rows, columns * bits // 32)
         and tuple(scales.shape) == tuple(biases.shape) == (rows, columns // GROUP),
         'Upstream affine output representation differs')
    result = dict(weight=np.asarray(packed).astype('<u4', copy=False).tobytes(),
                  scales=np.asarray(scales.view(mx.uint16, stream=target)).astype('<u2', copy=False).tobytes(),
                  biases=np.asarray(biases.view(mx.uint16, stream=target)).astype('<u2', copy=False).tobytes())
    for key in ('scales', 'biases'):
        words = np.frombuffer(result[key], dtype='<u2')
        need(not np.any((words & 0x7f80) == 0x7f80), 'Nonfinite affine metadata')
    return result


def reconstruct_f32(buffers, rows, columns, bits, np):
    """Independent bounded packing/metadata decoder; never calls MLX.

    Integer code extraction is explicit. A BF16 scale times an unsigned <=8-bit
    code is exact in FP32 for the qualified finite range; addition rounds to
    FP32 separately. Nonfinite reconstruction rejects this witness.
    """
    count = shape(rows, columns, bits)
    need(type(buffers) is dict and set(buffers) == {'weight', 'scales', 'biases'},
         'All three affine buffers required')
    for key, expected in [('weight', count * bits // 8),
                          ('scales', count // GROUP * 2),
                          ('biases', count // GROUP * 2)]:
        need(type(buffers[key]) is bytes and len(buffers[key]) == expected,
             'Exact affine buffer extent required')
    words = np.frombuffer(buffers['weight'], dtype='<u4').reshape(rows, -1)
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    codes = ((words[:, :, None] >> shifts) & ((1 << bits) - 1)).reshape(rows, columns)
    metadata = {}
    for key in ('scales', 'biases'):
        raw = np.frombuffer(buffers[key], dtype='<u2')
        need(not np.any((raw & 0x7f80) == 0x7f80), 'Nonfinite affine metadata')
        values = (raw.astype(np.uint32) << 16).view(np.float32).reshape(rows, -1)
        metadata[key] = np.repeat(values, GROUP, axis=1)
    out = np.empty((rows, columns), dtype=np.float32)
    np.multiply(codes, metadata['scales'], out=out, dtype=np.float32)
    np.add(out, metadata['biases'], out=out, dtype=np.float32)
    need(np.isfinite(out).all(), 'Nonfinite reconstructed FP32 value')
    return out.astype('<f4', copy=False).tobytes()


def controls():
    """Prospective literal bytes, before either device executes this helper.

    MLX v0.32.3 CPU quantized.cpp uses rint; Metal quantized.h uses round and
    initializes the group maximum at zero. The rational ramp/tie cases have
    exact scale/bias. Negative-only input exposes the device-specific maximum.
    Sources: https://github.com/ml-explore/mlx/tree/v0.32.3/mlx/backend
    """
    from minicpm5_quantize import controls as affine8_controls
    cases = [dict(case, bits=8) for case in affine8_controls()]
    zero = bytes(32)
    ramp = list(range(-8, 8)) * 4
    ties = [-16] + [-15 + 2 * i for i in range(15)] + [14] + [-16] * 47
    gpu_ties = bytes.fromhex('1032547698badcfe0f') + bytes(23)
    cpu_ties = bytes.fromhex('0022446688aaccee0f') + bytes(23)
    literals = [
        ('zeros', [0] * 64, zero, 'd7b3', '0000', zero, 'd7b3'),
        ('ones', [1] * 64, zero, 'd7b3', '803f', zero, 'd7b3'),
        ('negative_ones', [-1] * 64, zero, '893d', '80bf', zero, 'd7b3'),
        ('exact_packing', ramp, bytes.fromhex('1032547698badcfe') * 4,
         '803f', '00c1', bytes.fromhex('1032547698badcfe') * 4, '803f'),
        ('round_ties', ties, gpu_ties, '0040', '80c1', cpu_ties, '0040'),
        ('above_fp16_range', [65536] * 64, zero, 'd7b3', '8047', zero, 'd7b3'),
    ]
    cases.extend(dict(name=name, bits=4, rows=1, columns=64,
                      input_bf16=bf16_exact(values).hex(),
                      gpu=dict(weight=codes.hex(), scales=scale, biases=bias),
                      cpu=dict(weight=cpu_codes.hex(), scales=cpu_scale, biases=bias))
                 for name, values, codes, scale, bias, cpu_codes, cpu_scale in literals)
    return cases


def literal_reconstruction(case, device):
    """Small stdlib-only oracle for literal control buffers."""
    bits = case['bits']
    encoded = case[device]
    packed = bytes.fromhex(encoded['weight'])
    scale = struct.unpack('<f', bytes(2) + bytes.fromhex(encoded['scales']))[0]
    bias = struct.unpack('<f', bytes(2) + bytes.fromhex(encoded['biases']))[0]
    values = [((byte >> shift) & ((1 << bits) - 1))
              for byte in packed for shift in range(0, 8, bits)]
    return b''.join(struct.pack('<f', scale * code + bias) for code in values)
