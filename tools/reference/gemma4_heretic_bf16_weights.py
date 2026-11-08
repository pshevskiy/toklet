"""Separate modified-parent BF16 identity for the unchanged independent graph."""
import hashlib
import json
import struct
import bf16_source
import gemma4_heretic_weights as mixed
import gemma4_text_math as graph

REPOSITORY = 'p-e-w/gemma-4-E2B-it-heretic-ara'
REVISION = 'c9a1d4c031981f14d86eeb0c7d87de7fafd34513'
FILE_BYTES = 10246621918
FILE_SHA256 = '15605c16e479a8dc53928dd4b521a4ead9fe66222dc14d1ce392384e999a544b'
CONFIG_BYTES = 4989
CONFIG_SHA256 = '09f31bccdb5185282ee0ac6a41586958b39523c94cee0205dac6d39407169edb'
HEADER_BYTES = 263952
HEADER_SHA256 = '12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e'
PREFIX = 'model.language_model.'


def header(config, prefix, raw):
    if (type(config) is not bytes or len(config) != CONFIG_BYTES
            or hashlib.sha256(config).hexdigest() != CONFIG_SHA256
            or prefix != struct.pack('<Q', HEADER_BYTES)
            or len(raw) != HEADER_BYTES or hashlib.sha256(raw).hexdigest() != HEADER_SHA256):
        raise ValueError('Exact modified BF16 config/header identity required')
    value = json.loads(raw, object_pairs_hook=mixed.unique)
    if value.get('__metadata__') != {'format': 'pt'} or len(value) != 2012:
        raise ValueError('Modified BF16 physical inventory differs')
    return value


def weights(owner, config, deadline):
    if (type(owner) is not bf16_source.Source or owner.size != FILE_BYTES
            or owner.sha256 != FILE_SHA256):
        raise ValueError('Pinned authenticated modified BF16 source required')
    inventory = header(config, owner.read(0, 8, deadline=deadline),
                       owner.read(8, HEADER_BYTES, deadline=deadline))
    matrices, vectors = {}, {}
    for name, shape in graph.shapes(graph.E2B).items():
        offset = mixed.component(owner, inventory, 8 + HEADER_BYTES, PREFIX + name, 'BF16', shape)
        rows, columns = shape if len(shape) == 2 else (1, shape[0])
        view = bf16_source.Rows(owner, rows, columns, offset, deadline)
        if len(shape) == 2:
            matrices[name] = view
        else:
            vectors[name] = view.read_rows(0, 1).reshape(shape)
    if len(matrices) != 278 or len(vectors) != 262:
        raise ValueError('Modified BF16 graph inventory differs')
    return graph.Weights(graph.E2B, matrices, vectors)
