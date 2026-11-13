"""Independent typed, bounded row reader for an externally pinned 2B child.

Authenticates complete bytes/header/EOF and independently reconstructs the full
physical inventory. Only the eighteen A_log owners use F32. No model expansion,
engine import, source tensor ranges, or numerical runtime occurs on import.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import time

import affine4_tiles as tiles
import bf16_source
import qwen35_cpu_graph as graph
import qwen35_real_weights as base
import qwen35_2b_graph_protocol as protocol

need = protocol.need
GEOMETRY = protocol.GEOMETRY
SOURCE = dict(repository='TheCluster/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-MLX-bf16',
    revision='78efacde810bbe7d1b80410fc5ac60db4f24fd8a',
    container_sha256='f04765ef799190d41131132d43d69624e9ed6eac7b7cdd2a65a2eb3e266878ab',
    config_sha256='ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4',
    header_sha256='ed13c70a5f6268600b730dbc480262f640decf50083cddb23a107f201b6c0060')
STORAGE = dict(bits=4, group_size=64, mode='affine', logical_tensors=320,
               physical_tensors=694, body_bytes=1059315904)
QUALIFICATION = dict(diagnostic_only=True,
    model_numerics_quality_fit_performance_adoption_qualified=False)


def unique(items):
    result = {}
    for key, value in items:
        need(key not in result, 'Duplicate JSON metadata field')
        result[key] = value
    return result


def decode(raw):
    need(type(raw) is bytes, 'Exact JSON bytes required')
    def number(value):
        parsed = float(value)
        need(math.isfinite(parsed), 'Finite JSON number required')
        return parsed
    def constant(_):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=unique, parse_float=number, parse_constant=constant)


def captured(path, limit, expected=None):
    path = Path(path)
    need(type(limit) is int and 0 < limit <= 32 * 1024**2, 'Bounded metadata evidence limit required')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, 'Bounded regular evidence required')
    # Source retains and guards one descriptor, including path identity and EOF.
    if expected is None:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            identity = bf16_source.identity(before)
            need(bf16_source.identity(os.fstat(fd)) == identity, 'Evidence descriptor identity differs')
            raw = os.pread(fd, limit + 1, 0)
            need(0 < len(raw) <= limit and len(raw) == before.st_size
                 and bf16_source.identity(os.fstat(fd)) == identity
                 and bf16_source.identity(path.lstat()) == identity, 'Evidence extent/identity changed')
        finally:
            os.close(fd)
    else:
        with bf16_source.Source(path, before.st_size, protocol.sha(expected),
                                deadline=time.monotonic() + 30) as owner:
            chunks = bytearray()
            for offset in range(0, owner.size, bf16_source.HASH_BLOCK):
                chunks.extend(owner.read(offset, min(bf16_source.HASH_BLOCK, owner.size - offset),
                    deadline=time.monotonic() + 30))
            raw = bytes(chunks)
    return raw


def artifact_manifest(path, expected_sha256):
    raw = captured(path, 64 * 1024, expected_sha256)
    value = decode(raw)
    expected_keys = {'format', 'passed', 'artifact_pins', 'source', 'lineage_certificate_sha256',
                     'conversion_manifest_sha256', 'storage', 'qualification'}
    need(type(value) is dict and set(value) == expected_keys
         and value['format'] == 'toklet-qwen35-2b-affine-child-v1' and value['passed'] is True,
         'Exact successful externally pinned 2B child manifest required')
    for name, expected in [('source', SOURCE), ('storage', STORAGE), ('qualification', QUALIFICATION)]:
        need(protocol.exact(value[name], expected), 'Selected source/diagnostic manifest differs: ' + name)
    protocol.source_pins(value['artifact_pins'])
    for key in ['lineage_certificate_sha256', 'conversion_manifest_sha256']:
        need(protocol.sha(value[key]) != '0' * 64, 'Actual nonzero provenance SHA256 required')
    return value


def logical_specs(geometry=None):
    result = graph.specs(GEOMETRY if geometry is None else geometry)
    return {name: ('f32' if name.endswith('linear_attn.A_log') else kind, shape)
            for name, (kind, shape) in result.items()}


def physical_specs(geometry=None):
    result = {}
    for name, (kind, shape) in logical_specs(geometry).items():
        if kind == 'matrix':
            rows, columns = shape
            stem = name.removesuffix('.weight')
            result[name] = ('U32', [rows, columns // 8])
            for suffix in ('.scales', '.biases'):
                result[stem + suffix] = ('BF16', [rows, columns // 64])
        else:
            result[name] = ('F32' if kind == 'f32' else 'BF16', shape)
    return result


def validate_table(table, origin, size, geometry=None):
    need(type(table) is dict, 'Complete safetensors header table required')
    table = dict(table)
    metadata = table.pop('__metadata__', None)
    need(protocol.exact(metadata, {'format': 'mlx'}), 'Exact selected child metadata required')
    wanted = physical_specs(geometry)
    need(set(table) == set(wanted), 'Complete independent physical inventory differs')
    spans = []
    for name, (dtype, shape) in wanted.items():
        item = table[name]
        need(type(item) is dict and set(item) == {'dtype', 'shape', 'data_offsets'}
             and type(item['dtype']) is str and item['dtype'] == dtype
             and protocol.exact(item['shape'], shape), 'Typed physical component differs: ' + name)
        extent = item['data_offsets']
        need(type(extent) is list and len(extent) == 2 and all(type(n) is int for n in extent)
             and 0 <= extent[0] < extent[1] <= size - origin
             and extent[1] - extent[0] == math.prod(shape) * (2 if dtype == 'BF16' else 4),
             'Exact physical component extent differs: ' + name)
        spans.append(extent)
    spans.sort()
    need(spans and spans[0][0] == 0 and spans[-1][1] == size - origin
         and all(a[1] == b[0] for a, b in zip(spans, spans[1:])), 'Complete nonaliased body coverage required')
    return table


def bind(owner, table, origin, deadline, geometry=None):
    """Typed tiny-test or authenticated full-source binding; eligibility is weights()."""
    g = GEOMETRY if geometry is None else geometry
    table = validate_table(table, origin, owner.size, g)
    cursor = base.Cursor(owner, deadline)
    matrices, vectors = {}, {}
    for name, (kind, shape) in logical_specs(g).items():
        if kind == 'matrix':
            rows, columns = shape
            stem = name.removesuffix('.weight')
            offsets = [origin + table[key]['data_offsets'][0]
                       for key in (name, stem + '.scales', stem + '.biases')]
            matrices[name] = base.Matrix(tiles.MatrixRows(cursor, owner.size, rows, columns, *offsets))
        else:
            item = table[name]
            count = math.prod(shape)
            need(count <= 32768, 'Bounded small typed owner required')
            offset = origin + item['data_offsets'][0]
            raw = owner.read(offset, count * (4 if kind == 'f32' else 2), deadline=deadline)
            if kind == 'f32':
                values = [v[0] for v in struct.iter_unpack('<f', raw)]
            else:
                values = [struct.unpack('<f', struct.pack('<I', word[0] << 16))[0]
                          for word in struct.iter_unpack('<H', raw)]
            need(len(values) == count and all(math.isfinite(v) for v in values), 'Finite complete typed values required')
            vectors[name] = values
    return matrices, vectors


def weights(owner, config, pins, deadline):
    protocol.source_pins(pins)
    need(type(owner) is bf16_source.Source and owner.size == pins['container_bytes']
         and owner.sha256 == pins['container_sha256'], 'Authenticated actual complete 2B child required')
    need(type(config) is bytes and len(config) == pins['config_bytes']
         and hashlib.sha256(config).hexdigest() == pins['config_sha256'], 'Actual pinned child config differs')
    fixture = Path(__file__).resolve().parents[2] / 'tests/fixtures/qwen35-2b-source-v1/source-config.json'
    parent_config = decode(captured(fixture, 2908, SOURCE['config_sha256']))
    need(type(parent_config) is dict and 'quantization' not in parent_config
         and 'quantization_config' not in parent_config, 'Pinned unquantized source wrapper required')
    parent_config['quantization'] = dict(group_size=64, bits=4, mode='affine')
    expected_config = (json.dumps(parent_config, indent=2, allow_nan=False) + '\n').encode()
    need(config == expected_config, 'Complete preserved source wrapper and only selected quantization required')
    header_bytes = pins['header_bytes']
    origin = 8 + header_bytes
    need(owner.read(0, 8, deadline=deadline) == struct.pack('<Q', header_bytes), 'Actual header prefix differs')
    raw = owner.read(8, header_bytes, deadline=deadline)
    need(hashlib.sha256(raw).hexdigest() == pins['header_sha256'], 'Actual complete header SHA256 differs')
    need(owner.size - origin == STORAGE['body_bytes'], 'Selected full 2B body extent differs')
    table = decode(raw)
    need(type(table) is dict and len(table) == 695 and len(logical_specs()) == 320,
         'Complete 2B inventory including metadata differs')
    # bind validates the original metadata and all694 physical descriptors once.
    matrices, vectors = bind(owner, table, origin, deadline)
    need(len(matrices) == 187 and len(vectors) == 133, 'Complete 2B logical owners required')
    return matrices, vectors


Math = base.Math
