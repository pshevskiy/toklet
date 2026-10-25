"""Bounded file-row helpers for independent four-bit numerical references.

Only little-endian packed4/group64 with BF16 affine metadata is implemented.
Callers authenticate the source descriptor/container/schema and own its lifetime.
These helpers validate component extents and numerical values, but grant no
checkpoint, model, device or architecture eligibility. No target code imports.
"""
from dataclasses import dataclass

MAX_FILE_BYTES = 4 * 1024**3
MAX_ROWS = 262144
MAX_COLUMNS = 12288
MAX_TOKENS = 32
TILE_ROWS = 128
# Conservative NumPy-array overlap: old/new tile, raw bytes, BF16 conversions,
# reconstructed FP32, FP64 dot operands/results and source conversion. BLAS
# private workspace and interpreter/package residency need caller admission.
TILE_SCRATCH_RESERVATION = 64 * 1024**2
MAX_BUFFER_RESERVATION = 128 * 1024**2


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Bounded integer required')
    return value


def bf16(words):
    import numpy as np
    if words.dtype != np.dtype('<u2'):
        raise ValueError('Little-endian BF16 bits required')
    values = (words.astype('<u4') << 16).view('<f4')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite BF16 metadata')
    return values


def decode_rows(packed, scales, biases, columns):
    """FP64 affine reconstruction rounded once to FP32, including signed zero."""
    import numpy as np
    integer(columns, 64, MAX_COLUMNS)
    if columns % 64 or packed.dtype != np.dtype('uint8') or packed.ndim != 2:
        raise ValueError('Packed4/group64 rows required')
    rows = integer(len(packed), 1, TILE_ROWS)
    if (packed.shape != (rows, columns // 2)
            or scales.shape != (rows, columns // 64) or biases.shape != scales.shape):
        raise ValueError('Affine component shapes differ')
    scale = bf16(scales).astype(np.float64)
    bias = bf16(biases).astype(np.float64)
    # Check both code endpoints, including unused codes, before reconstruction.
    # All affine interior values lie between them, even for negative scales.
    with np.errstate(over='ignore', invalid='ignore'):
        endpoints = (bias.astype(np.float32), (15 * scale + bias).astype(np.float32))
    if any(not np.isfinite(value).all() for value in endpoints):
        raise ValueError('Affine code domain overflows FP32')
    codes = np.empty((rows, columns), dtype=np.float64)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    grouped = codes.reshape(rows, columns // 64, 64)
    grouped *= scale[:, :, None]
    grouped += bias[:, :, None]
    values = codes.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite affine reconstruction')
    return values


@dataclass(frozen=True)
class MatrixRows:
    """Borrow a caller-authenticated, seekable binary descriptor; never close it.

    Offsets are absolute file offsets. All three components are contiguous
    row-major spans. There is no retained tensor expansion or memory mapping.
    A single reader must own descriptor positioning during every operation.
    """
    stream: object
    file_bytes: int
    rows: int
    columns: int
    weight_offset: int
    scale_offset: int
    bias_offset: int

    def __post_init__(self):
        integer(self.file_bytes, 1, MAX_FILE_BYTES)
        integer(self.rows, 1, MAX_ROWS)
        integer(self.columns, 64, MAX_COLUMNS)
        if self.columns % 64:
            raise ValueError('Complete group64 rows required')
        spans = []
        for offset, stride in self.components():
            integer(offset, 0, self.file_bytes)
            end = offset + self.rows * stride
            if end > self.file_bytes:
                raise ValueError('Affine component exceeds source extent')
            spans.append((offset, end))
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError('Affine component spans overlap')

    def components(self):
        return ((self.weight_offset, self.columns // 2),
                (self.scale_offset, self.columns // 64 * 2),
                (self.bias_offset, self.columns // 64 * 2))

    def read_rows(self, start, end):
        import numpy as np
        integer(start, 0, self.rows - 1)
        integer(end, start + 1, min(start + TILE_ROWS, self.rows))
        count = end - start
        raw = []
        for offset, stride in self.components():
            self.stream.seek(offset + start * stride)
            value = self.stream.read(count * stride)
            if type(value) is not bytes or len(value) != count * stride:
                raise ValueError('Affine row read truncated or oversized')
            raw.append(value)
        return decode_rows(np.frombuffer(raw[0], dtype=np.uint8).reshape(count, self.columns // 2),
                           np.frombuffer(raw[1], dtype='<u2').reshape(count, self.columns // 64),
                           np.frombuffer(raw[2], dtype='<u2').reshape(count, self.columns // 64),
                           self.columns)


def gather_rows(matrix, ids):
    """Repeated/nonmonotonic token IDs return independent contiguous FP32 rows."""
    import numpy as np
    if type(ids) is not list or not 1 <= len(ids) <= MAX_TOKENS:
        raise ValueError('Bounded explicit row IDs required')
    for value in ids:
        integer(value, 0, matrix.rows - 1)
    out = np.empty((len(ids), matrix.columns), dtype=np.float32)
    for index, value in enumerate(ids):
        out[index] = matrix.read_rows(value, value + 1)[0]
    return out


def planned_buffer_bytes(matrix, tokens, output_dtype):
    import numpy as np
    integer(tokens, 1, MAX_TOKENS)
    if output_dtype is not np.float32 and output_dtype is not np.float64:
        raise ValueError('Explicit FP32 or FP64 projection contract required')
    # Output peaks at32*262144*8=64MiB. The independent64MiB scratch
    # reservation includes conversion overlap, not merely the final weight tile.
    return TILE_SCRATCH_RESERVATION + tokens * matrix.rows * np.dtype(output_dtype).itemsize


def project_rows(matrix, activations, output_dtype, *, buffer_limit_bytes=MAX_BUFFER_RESERVATION):
    """Tile weights only; accumulate each dot in NumPy/BLAS FP64.

    BLAS may reassociate FP64 sums. This is checked against independent scalar
    fsum and sequential witnesses; it does not assert bitwise scalar equivalence.
    Inputs can be FP64 or stage-rounded FP32, output explicitly one of them.
    Both always use FP32-reconstructed weights, unlike an exact-affine FP64
    projection contract. The buffer allowance covers NumPy arrays, not process fit.
    No full input-size-by-output-size floating matrix is allocated.
    """
    import numpy as np
    if (not isinstance(activations, np.ndarray) or activations.ndim != 2
            or activations.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not 1 <= len(activations) <= MAX_TOKENS
            or activations.shape[1] != matrix.columns or not np.isfinite(activations).all()
            or (output_dtype is not np.float32 and output_dtype is not np.float64)):
        raise ValueError('Finite bounded projection operands required')
    required = planned_buffer_bytes(matrix, len(activations), output_dtype)
    integer(buffer_limit_bytes, 1, MAX_BUFFER_RESERVATION)
    if required > buffer_limit_bytes:
        raise ValueError('Projection array buffer allowance exceeded')
    source = activations.astype(np.float64, copy=False)
    out = np.empty((len(source), matrix.rows), dtype=output_dtype)
    for start in range(0, matrix.rows, TILE_ROWS):
        end = min(start + TILE_ROWS, matrix.rows)
        # read_rows releases its FP64 reconstruction before returning FP32 rows.
        weights = matrix.read_rows(start, end)
        value = source @ weights.astype(np.float64).T
        with np.errstate(over='ignore', invalid='ignore'):
            rounded = value.astype(output_dtype)
        if not np.isfinite(rounded).all():
            raise ValueError('Nonfinite projection result')
        out[:, start:end] = rounded
    return out
