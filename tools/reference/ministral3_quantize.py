"""Bounded Ministral affine4/8 tiles; no numerical imports or model loading.

Complete BF16 rows become low-bit-first U32 codes and separate BF16 group-64
scale/bias arrays. Callers admit the numerical runtime before supplying np/mx.
The existing Granite and MiniCPM5 geometry contracts remain unchanged.
"""
import struct

import granite_quantize as legacy

ROWS = 224
MAX_COLUMNS = 9216
GROUP = 64
need = legacy.need
compare = legacy.compare


def shape(rows, columns, bits):
    need(type(rows) is int and 1 <= rows <= ROWS
         and type(columns) is int and GROUP <= columns <= MAX_COLUMNS
         and columns % GROUP == 0 and type(bits) is int and bits in (4, 8),
         'Bounded Ministral complete group-64 affine4/8 rows required')
    count = rows * columns
    need(count * 2 <= 4 * 1024**2, 'BF16 tile exceeds 4 MiB')
    return count


def geometry(raw, rows, columns, bits):
    count = shape(rows, columns, bits)
    need(type(raw) is bytes and len(raw) == count * 2,
         'Exact complete Ministral BF16 byte extent required')
    return count


def buffer_geometry(buffers, rows, columns, bits):
    """Validate extents and finite BF16 metadata before numerical access."""
    count = shape(rows, columns, bits)
    need(type(buffers) is dict and set(buffers) == {'weight', 'scales', 'biases'},
         'All three Ministral affine buffers required')
    for key, expected in [('weight', count * bits // 8),
                          ('scales', count // GROUP * 2),
                          ('biases', count // GROUP * 2)]:
        need(type(buffers[key]) is bytes and len(buffers[key]) == expected,
             'Exact Ministral affine buffer extent required')
    for key in ('scales', 'biases'):
        need(all(word & 0x7f80 != 0x7f80
                 for (word,) in struct.iter_unpack('<H', buffers[key])),
             'Nonfinite Ministral affine metadata')
    return count


def quantize(raw, rows, columns, bits, device, np, mx):
    """Return completed little-endian buffers; no full-model owner survives."""
    geometry(raw, rows, columns, bits)
    need(device in ('cpu', 'gpu'), 'Explicit Ministral quantization device required')
    if columns <= legacy.MAX_COLUMNS:
        return legacy.quantize(raw, rows, columns, bits, device, np, mx)
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
         'Upstream Ministral affine representation differs')
    buffers = dict(weight=np.asarray(packed).astype('<u4', copy=False).tobytes(),
                   scales=np.asarray(scales.view(mx.uint16, stream=target)).astype('<u2', copy=False).tobytes(),
                   biases=np.asarray(biases.view(mx.uint16, stream=target)).astype('<u2', copy=False).tobytes())
    buffer_geometry(buffers, rows, columns, bits)
    return buffers


def reconstruct_f32(buffers, rows, columns, bits, np):
    """Independent extraction and separate FP32 multiply/add, never MLX."""
    buffer_geometry(buffers, rows, columns, bits)
    if columns <= legacy.MAX_COLUMNS:
        return legacy.reconstruct_f32(buffers, rows, columns, bits, np)
    words = np.frombuffer(buffers['weight'], dtype='<u4').reshape(rows, -1)
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    codes = ((words[:, :, None] >> shifts) & ((1 << bits) - 1)).reshape(rows, columns)
    metadata = {}
    for key in ('scales', 'biases'):
        raw = np.frombuffer(buffers[key], dtype='<u2')
        values = (raw.astype(np.uint32) << 16).view(np.float32).reshape(rows, -1)
        metadata[key] = np.repeat(values, GROUP, axis=1)
    out = np.empty((rows, columns), dtype=np.float32)
    np.multiply(codes, metadata['scales'], out=out, dtype=np.float32)
    np.add(out, metadata['biases'], out=out, dtype=np.float32)
    need(np.isfinite(out).all(), 'Nonfinite reconstructed FP32 value')
    return out.astype('<f4', copy=False).tobytes()


def controls():
    """The same twelve prospective literal cases and device-specific words."""
    return legacy.controls()


def literal_reconstruction(case, device):
    return legacy.literal_reconstruction(case, device)
