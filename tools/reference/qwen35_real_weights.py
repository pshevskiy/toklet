"""Independent bounded rows for one authenticated, already-sanitized Qwen source.

Little-endian U32 nibbles, group64, BF16 scales/biases; affine FP64 then FP32.
The payload origin is odd. BF16 bytes are decoded without alignment assumptions.
No tokenizer, target code, MLX graph or whole floating-point matrix is loaded.
"""
import hashlib
import json
import math
import struct
import affine4_tiles as tiles
import bf16_source
import check_qwen35_payload as pin
import qwen35_cpu_graph as graph

REPOSITORY = 'avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit'
REVISION = 'fb98fbc59c71a567f3e23435386c9e93951261db'
HEADER_BYTES = 119829
ORIGIN = HEADER_BYTES + 8
GEOMETRY = dict(hidden=2560, intermediate=9216, vocab=248320, layers=32,
    query_heads=16, kv_heads=4, head_dim=256, rotated_dims=64,
    key_heads=16, value_heads=32, key_dim=128, value_dim=128,
    max_positions=262144, rope_theta=10000000.0, rms_epsilon=1e-6)


def need(ok, message):
    if not ok: raise ValueError(message)


def unique(items):
    result = {}
    for name, value in items:
        need(name not in result, 'Duplicate Qwen metadata field')
        result[name] = value
    return result


class Cursor:
    """Single-caller positioning only; the immutable Source owns the descriptor."""
    def __init__(self, owner, deadline):
        self.owner, self.deadline, self.offset = owner, deadline, 0

    def seek(self, offset):
        bf16_source.integer(offset, 0, self.owner.size - 1)
        self.offset = offset

    def read(self, count):
        raw = self.owner.read(self.offset, count, deadline=self.deadline)
        self.offset += len(raw)
        return raw


class Matrix:
    """Borrow packed rows; indexed access is only the embedding gather."""
    def __init__(self, rows): self.rows = rows

    def __getitem__(self, row):
        bf16_source.integer(row, 0, self.rows.rows - 1)
        return self.rows.read_rows(row, row + 1)[0].tolist()


class Math(graph.Math):
    def linear(self, x, matrix):
        import numpy as np
        need(type(matrix) is Matrix, 'Explicit bounded packed matrix required')
        dtype = np.float32 if self.staged else np.float64
        source = np.asarray([x], dtype=dtype)
        return tiles.project_rows(matrix.rows, source, dtype)[0].tolist()


def component(table, origin, size, name, dtype, shape):
    need(type(origin) is int and 0 <= origin < size, 'Payload origin differs')
    item = table.get(name)
    need(type(item) is dict and set(item) == {'dtype', 'shape', 'data_offsets'},
         'Exact component descriptor required: ' + name)
    need(item['dtype'] == dtype and item['shape'] == shape
         and all(type(v) is int and v > 0 for v in item['shape']),
         'Component representation differs: ' + name)
    span = item['data_offsets']
    need(type(span) is list and len(span) == 2
         and all(type(v) is int for v in span)
         and 0 <= span[0] < span[1] <= size - origin
         and span[1] - span[0] == math.prod(shape) * (4 if dtype == 'U32' else 2),
         'Component span differs: ' + name)
    return origin + span[0]


def bind(owner, table, origin, geometry, deadline):
    """Checked row binding; checkpoint eligibility belongs to weights(), below."""
    import numpy as np
    cursor = Cursor(owner, deadline)
    matrices, vectors, consumed = {}, {}, set()
    for name, (kind, shape) in graph.specs(geometry).items():
        if kind == 'matrix':
            rows, columns = shape
            stem = name.removesuffix('.weight')
            offsets = []
            for key, dtype, sizes in [(name, 'U32', [rows, columns//8]),
                (stem+'.scales', 'BF16', [rows, columns//64]),
                (stem+'.biases', 'BF16', [rows, columns//64])]:
                offsets.append(component(table, origin, owner.size, key, dtype, sizes))
                consumed.add(key)
            matrices[name] = Matrix(tiles.MatrixRows(cursor, owner.size, rows, columns, *offsets))
        else:
            offset = component(table, origin, owner.size, name, 'BF16', shape)
            count = math.prod(shape)
            need(count <= 32768, 'Small unquantized owner exceeds bound')
            values = []
            for start in range(0, count, 8192):
                n = min(8192, count-start)
                raw = owner.read(offset + 2*start, 2*n, deadline=deadline)
                need(len(raw) == 2*n, 'Truncated unaligned BF16 component')
                values.extend(tiles.bf16(np.frombuffer(raw, dtype='<u2')).tolist())
            vectors[name] = values
            consumed.add(name)
    need(consumed == set(table), 'Unconsumed or missing physical components')
    return matrices, vectors


def weights(owner, config, deadline):
    need(type(owner) is bf16_source.Source and owner.size == pin.FILE_BYTES
         and owner.sha256 == pin.FILE_SHA, 'Authenticated pinned Qwen source required')
    need(type(config) is bytes and len(config) == 3024
         and hashlib.sha256(config).hexdigest() == pin.CONFIG_SHA, 'Pinned Qwen config differs')
    need(owner.read(0, 8, deadline=deadline) == struct.pack('<Q', HEADER_BYTES), 'Header prefix differs')
    raw = owner.read(8, HEADER_BYTES, deadline=deadline)
    need(hashlib.sha256(raw).hexdigest() == pin.HEADER_SHA, 'Pinned Qwen header differs')
    table = json.loads(raw, object_pairs_hook=unique)
    table.pop('__metadata__')
    need(len(table) == 924, 'Physical Qwen inventory differs')
    spans = sorted(item['data_offsets'] for item in table.values())
    need(spans[0][0] == 0 and spans[-1][1] == owner.size - ORIGIN
         and all(a[1] == b[0] for a, b in zip(spans, spans[1:])),
         'Complete physical payload coverage differs')
    matrices, vectors = bind(owner, table, ORIGIN, GEOMETRY, deadline)
    need(len(matrices) == 249 and len(vectors) == 177, 'Logical Qwen inventory differs')
    return matrices, vectors
