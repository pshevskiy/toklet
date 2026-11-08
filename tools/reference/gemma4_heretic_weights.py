"""Pinned mixed Heretic rows for the independent graph; never imports Toklet.

Original affine4/group64/BF16 metadata and the BF16 PLE projection stay explicit.
The caller owns the authenticated descriptor, numerical runtime and deadline.
"""
import hashlib
import json
import math
import struct
import affine4_tiles as tiles
import bf16_source
import gemma4_text_math as graph

REPOSITORY = 'deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit'
REVISION = 'cde01efcb11f8a12e88a953b752fc16062aa1486'
FILE_BYTES = 3581100878
FILE_SHA256 = 'd4a2995ce3fdafbbd4a3376bc8524a8d26bcbf0f51197701b4e21ab27faa37e1'
CONFIG_SHA256 = '7ff08145776f892538f06b2e6b574a62080959d3097c37e4f8a45664f4478da1'
HEADER_BYTES = 335744
HEADER_SHA256 = 'e12e790370839d89a88cb1ead5e27525011ed3f39a3879f6c2a7162080f4dc50'
PREFIX = 'language_model.model.'
BF16_PROJECTION = 'per_layer_model_projection.weight'


def unique(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate Heretic metadata key')
        result[key] = value
    return result


def header(config, prefix, raw):
    if (type(config) is not bytes or len(config) != 6035
            or hashlib.sha256(config).hexdigest() != CONFIG_SHA256
            or prefix != struct.pack('<Q', HEADER_BYTES)
            or len(raw) != HEADER_BYTES
            or hashlib.sha256(raw).hexdigest() != HEADER_SHA256):
        raise ValueError('Exact Heretic config/header identity required')
    value = json.loads(raw, object_pairs_hook=unique)
    if len([name for name in value if name != '__metadata__']) != 2649:
        raise ValueError('Heretic physical tensor inventory differs')
    return value


class Cursor:
    """Minimal sized-read adapter for the serialized affine row helper.

    The source uses pread and checks permanent descriptor/path identity around
    every read. This cursor owns positioning only; it never owns or closes it.
    """
    def __init__(self, owner, deadline):
        self.owner, self.deadline, self.offset = owner, deadline, 0

    def seek(self, offset):
        bf16_source.integer(offset, 0, self.owner.size - 1)
        self.offset = offset

    def read(self, count):
        raw = self.owner.read(self.offset, count, deadline=self.deadline)
        self.offset += len(raw)
        return raw


def component(owner, inventory, origin, name, dtype, shape):
    bf16_source.integer(origin, 0, owner.size - 1)
    if dtype not in ('U32', 'BF16') or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('Heretic component representation differs')
    item = inventory.get(name)
    if (not isinstance(item, dict) or item.get('dtype') != dtype
            or item.get('shape') != list(shape)):
        raise ValueError('Heretic component dtype/shape differs: ' + name)
    span = item.get('data_offsets')
    if (type(span) is not list or len(span) != 2
            or any(type(v) is not int for v in span)
            or not 0 <= span[0] < span[1] <= owner.size - origin
            or span[1] - span[0] != (4 if dtype == 'U32' else 2) * math.prod(shape)):
        raise ValueError('Heretic component extent differs: ' + name)
    return origin + span[0]


def matrix(owner, cursor, inventory, origin, name, shape, deadline, *, bf16):
    rows, columns = shape
    if bf16:
        offset = component(owner, inventory, origin, name, 'BF16', shape)
        return bf16_source.Rows(owner, rows, columns, offset, deadline)
    if columns % 64:
        raise ValueError('Heretic complete group64 matrix required')
    stem = name.removesuffix('.weight')
    offsets = [component(owner, inventory, origin, key, dtype, sizes)
               for key, dtype, sizes in [
                   (name, 'U32', (rows, columns // 8)),
                   (stem + '.scales', 'BF16', (rows, columns // 64)),
                   (stem + '.biases', 'BF16', (rows, columns // 64))]]
    return tiles.MatrixRows(cursor, owner.size, rows, columns, *offsets)


def weights(owner, config, deadline):
    if (type(owner) is not bf16_source.Source or owner.size != FILE_BYTES
            or owner.sha256 != FILE_SHA256):
        raise ValueError('Pinned authenticated Heretic source required')
    inventory = header(config, owner.read(0, 8, deadline=deadline),
                       owner.read(8, HEADER_BYTES, deadline=deadline))
    origin = 8 + HEADER_BYTES
    cursor = Cursor(owner, deadline)
    matrices, vectors = {}, {}
    for name, shape in graph.shapes(graph.E2B).items():
        full = PREFIX + name
        if len(shape) == 2:
            matrices[name] = matrix(owner, cursor, inventory, origin, full, shape, deadline,
                                    bf16=name == BF16_PROJECTION)
        else:
            offset = component(owner, inventory, origin, full, 'BF16', shape)
            vectors[name] = bf16_source.Rows(owner, 1, shape[0], offset, deadline).read_rows(0, 1).reshape(shape)
    if len(matrices) != 278 or len(vectors) != 262:
        raise ValueError('Heretic graph inventory differs')
    return graph.Weights(graph.E2B, matrices, vectors)
