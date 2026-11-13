"""Cold selected Qwen3.5-2B source and all-affine4 child storage contract.

The selected BF16 MLX source has an odd payload origin. Every access is byte
based. F32 A_log remains F32; no normalization offset or dtype change is guessed.
"""
import hashlib
import json
import math
import re
import struct

import ministral3_convert_audit as saved
import granite_convert_layout as writes

need = saved.need
PREFIX = 'language_model.model.'
SOURCE_BYTES = 4426559145
SOURCE_SHA256 = 'f04765ef799190d41131132d43d69624e9ed6eac7b7cdd2a65a2eb3e266878ab'
SOURCE_CONFIG_BYTES = 2908
SOURCE_CONFIG_SHA256 = 'ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4'
SOURCE_HEADER_BYTES = 75233
SOURCE_HEADER_SHA256 = 'ed13c70a5f6268600b730dbc480262f640decf50083cddb23a107f201b6c0060'
BODY_BYTES = 1059315904
ROWS = 256
MAX_READ = 4 * 1024**2
REPOSITORY = 'TheCluster/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-MLX-bf16'
REVISION = '78efacde810bbe7d1b80410fc5ac60db4f24fd8a'
SOURCE = dict(repository=REPOSITORY, revision=REVISION, container_sha256=SOURCE_SHA256,
              config_sha256=SOURCE_CONFIG_SHA256, header_sha256=SOURCE_HEADER_SHA256)
STORAGE = dict(bits=4, group_size=64, mode='affine', logical_tensors=320,
               physical_tensors=694, body_bytes=BODY_BYTES)
QUALIFICATION = dict(diagnostic_only=True,
                     model_numerics_quality_fit_performance_adoption_qualified=False)
CHILD_FORMAT = 'toklet-qwen35-2b-affine-child-v1'


def selected_recipe():
    matrices = [dict(module=PREFIX + 'embed_tokens', shape=[248320, 2048], bits=4, group_size=64, mode='affine')]
    small = [dict(name=PREFIX + 'norm.weight', shape=[2048], dtype='BF16', bytes=4096)]
    for layer in range(24):
        p = PREFIX + f'layers.{layer}.'
        for role in ('input_layernorm', 'post_attention_layernorm'):
            small.append(dict(name=p + role + '.weight', shape=[2048], dtype='BF16', bytes=4096))
        roles = [('mlp.gate_proj', [6144, 2048]), ('mlp.up_proj', [6144, 2048]), ('mlp.down_proj', [2048, 6144])]
        if layer % 4 == 3:
            roles += [('self_attn.q_proj', [4096, 2048]), ('self_attn.k_proj', [512, 2048]),
                      ('self_attn.v_proj', [512, 2048]), ('self_attn.o_proj', [2048, 2048])]
            for role in ('q_norm', 'k_norm'):
                small.append(dict(name=p + 'self_attn.' + role + '.weight', shape=[256], dtype='BF16', bytes=512))
        else:
            roles += [('linear_attn.in_proj_a', [16, 2048]), ('linear_attn.in_proj_b', [16, 2048]),
                      ('linear_attn.in_proj_qkv', [6144, 2048]), ('linear_attn.in_proj_z', [2048, 2048]),
                      ('linear_attn.out_proj', [2048, 2048])]
            for role, dtype, shape in [('A_log', 'F32', [16]), ('dt_bias', 'BF16', [16]),
                                       ('norm.weight', 'BF16', [128]), ('conv1d.weight', 'BF16', [6144, 4, 1])]:
                small.append(dict(name=p + 'linear_attn.' + role, shape=shape, dtype=dtype,
                                  bytes=math.prod(shape) * (4 if dtype == 'F32' else 2)))
        matrices.extend(dict(module=p + role, shape=shape, bits=4, group_size=64, mode='affine') for role, shape in roles)
    return sorted(matrices, key=lambda v: v['module']), sorted(small, key=lambda v: v['name'])


def converted_config(raw):
    need(type(raw) is bytes and len(raw) == SOURCE_CONFIG_BYTES and hashlib.sha256(raw).hexdigest() == SOURCE_CONFIG_SHA256,
         'Exact pinned unquantized source config required')
    document = saved.loads(raw)
    need(type(document) is dict and 'quantization' not in document and 'quantization_config' not in document,
         'Unquantized source wrapper required')
    document['quantization'] = dict(group_size=64, bits=4, mode='affine')
    return (json.dumps(document, indent=2, allow_nan=False) + '\n').encode()


def output_layout(matrices, small):
    need(type(matrices) is list and matrices and type(small) is list, 'Complete explicit typed recipe required')
    declarations, logical, plans = {}, {}, {}
    def add(name, dtype, shape, count):
        need(type(name) is str and name.startswith(PREFIX) and name not in declarations
             and re.fullmatch(r'[A-Za-z0-9_.]+', name) is not None, 'Unique text tensor name required')
        need(type(shape) is list and 1 <= len(shape) <= 3 and all(type(v) is int and 1 <= v <= 248320 for v in shape),
             'Bounded typed tensor shape required')
        declarations[name] = dict(dtype=dtype, shape=shape, bytes=count)
    for matrix in matrices:
        need(type(matrix) is dict and set(matrix) == {'module', 'shape', 'bits', 'group_size', 'mode'}, 'Exact affine recipe required')
        shape, module = matrix['shape'], matrix['module']
        need(type(module) is str, 'Explicit matrix module name required')
        need(type(shape) is list and len(shape) == 2 and all(type(v) is int for v in shape)
             and 1 <= shape[0] <= 248320 and 64 <= shape[1] <= 6144 and shape[1] % 64 == 0
             and type(matrix['bits']) is int and matrix['bits'] == 4
             and type(matrix['group_size']) is int and matrix['group_size'] == 64 and matrix['mode'] == 'affine',
             'Bounded affine4/G64 complete matrix required')
        rows, columns = shape; name = module + '.weight'; fields = {}
        need(name not in logical, 'Duplicate logical matrix')
        logical[name] = dict(dtype='BF16', shape=shape)
        for field, dtype, width, stride in [('weight', 'U32', columns // 8, columns // 2),
                                          ('scales', 'BF16', columns // 64, columns // 32), ('biases', 'BF16', columns // 64, columns // 32)]:
            key = module + '.' + field
            add(key, dtype, [rows, width], rows * stride)
            fields[field] = dict(name=key, row_stride=stride)
        plans[name] = dict(shape=shape, bits=4, fields=fields)
    for item in small:
        need(type(item) is dict and set(item) == {'name', 'shape', 'dtype', 'bytes'} and item['dtype'] in ('BF16', 'F32'),
             'Exact original typed small tensor required')
        need(item['name'] not in logical and type(item['bytes']) is int
             and item['bytes'] == math.prod(item['shape']) * (4 if item['dtype'] == 'F32' else 2), 'Exact small tensor extent required')
        logical[item['name']] = dict(dtype=item['dtype'], shape=item['shape'])
        add(item['name'], item['dtype'], item['shape'], item['bytes'])
    header, cursor = {'__metadata__': {'format': 'mlx'}}, 0
    for name in sorted(declarations):
        item = declarations[name]; end = cursor + item['bytes']
        need(end <= 6 * 1024**3, 'Child payload exceeds 6 GiB')
        header[name] = dict(dtype=item['dtype'], shape=item['shape'], data_offsets=[cursor, end]); cursor = end
    raw = json.dumps(header, separators=(',', ':'), sort_keys=True).encode(); raw += b' ' * (-len(raw) % 8)
    need(0 < len(raw) <= 256 * 1024, 'Bounded child header required')
    prefix = struct.pack('<Q', len(raw)) + raw
    return dict(header=header, prefix=prefix, origin=len(prefix), body_bytes=cursor, file_bytes=cursor + len(prefix),
                logical=logical, matrices=plans)


def selected_layout():
    matrices, small = selected_recipe(); result = output_layout(matrices, small)
    need(len(matrices) == 187 and len(small) == 133 and len(result['logical']) == 320
         and len(result['header']) - 1 == 694 and result['body_bytes'] == BODY_BYTES, 'Complete selected child inventory required')
    return result


def source_spans(prefix, raw, size):
    need(prefix == struct.pack('<Q', SOURCE_HEADER_BYTES) and type(raw) is bytes and len(raw) == SOURCE_HEADER_BYTES
         and hashlib.sha256(raw).hexdigest() == SOURCE_HEADER_SHA256 and type(size) is int and size == SOURCE_BYTES,
         'Exact pinned unaligned source prefix/header/extent required')
    table = saved.loads(raw)
    need(type(table) is dict, 'Source tensor table required')
    result = []; expected = selected_layout()['logical']
    for name, item in table.items():
        if name == '__metadata__': continue
        need(type(item) is dict and set(item) == {'dtype', 'shape', 'data_offsets'} and item['dtype'] in ('BF16', 'F32'),
             'Exact source physical descriptor required')
        shape, offsets = item['shape'], item['data_offsets']
        need(type(shape) is list and 1 <= len(shape) <= 5 and all(type(v) is int and 1 <= v <= 248320 for v in shape)
             and type(offsets) is list and len(offsets) == 2 and all(type(v) is int for v in offsets)
             and 0 <= offsets[0] < offsets[1] <= size - 8 - len(raw)
             and offsets[1] - offsets[0] == math.prod(shape) * (4 if item['dtype'] == 'F32' else 2), 'Typed source span differs')
        if name.startswith(PREFIX):
            need(name in expected and item['dtype'] == expected[name]['dtype'] and shape == expected[name]['shape'], 'Selected text representation differs')
        else:
            need(name.startswith('vision_tower.'), 'Unknown excluded source tensor')
        result.append(dict(name=name, dtype=item['dtype'], shape=shape, start=offsets[0], end=offsets[1], selected=name in expected))
    result.sort(key=lambda x:x['start'])
    need(len(result) == 617 and sum(s['selected'] for s in result) == 320 and result[0]['start'] == 0
         and all(a['end'] == b['start'] for a,b in zip(result,result[1:]))
         and result[-1]['end'] + 8 + len(raw) == size, 'Complete tight selected/vision source coverage required')
    return result


def validate_small(item, raw):
    need(type(raw) is bytes and len(raw) == item['bytes'], 'Exact typed original small bytes required')
    if item['dtype'] == 'F32':
        need(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)), 'Nonfinite F32 small tensor')
    else:
        need(item['dtype'] == 'BF16' and all(word & 0x7f80 != 0x7f80 for (word,) in struct.iter_unpack('<H', raw)), 'Nonfinite BF16 small tensor')


def write_small(fd, planned, item, raw):
    validate_small(item, raw); spec = planned['header'][item['name']]
    need(spec['dtype'] == item['dtype'] and spec['shape'] == item['shape'], 'Typed output small layout differs')
    writes.write_all(fd, raw, planned['origin'] + spec['data_offsets'][0])


def artifact_pins(config, planned, actual_sha):
    saved.sha(actual_sha)
    need(type(config) is bytes and len(config) <= 64 * 1024,
         'Bounded actual child configuration required')
    return dict(config_bytes=len(config), config_sha256=hashlib.sha256(config).hexdigest(),
                header_bytes=planned['origin'] - 8, header_sha256=hashlib.sha256(planned['prefix'][8:]).hexdigest(),
                container_bytes=planned['file_bytes'], container_sha256=actual_sha)


def identifier(value):
    saved.sha(value)
    need(value != '0' * 64, 'Nonzero evidence identifier required')
    return value


def child_manifest(pins, lineage_sha256, conversion_sha256):
    """A consumer identity binding; the producer proves its evidence separately."""
    value = dict(format=CHILD_FORMAT, passed=True, artifact_pins=pins, source=dict(SOURCE),
                 lineage_certificate_sha256=identifier(lineage_sha256),
                 conversion_manifest_sha256=identifier(conversion_sha256), storage=dict(STORAGE),
                 qualification=dict(QUALIFICATION))
    return validate_child_manifest(value)


def validate_child_manifest(value):
    need(type(value) is dict and set(value) == {'format', 'passed', 'artifact_pins', 'source',
         'lineage_certificate_sha256', 'conversion_manifest_sha256', 'storage', 'qualification'},
         'Exact child identity manifest required')
    need(type(value['format']) is str and value['format'] == CHILD_FORMAT and value['passed'] is True,
         'Completed diagnostic child identity manifest required')
    for key, expected in [('source', SOURCE), ('storage', STORAGE), ('qualification', QUALIFICATION)]:
        saved.same(value[key], expected, 'Selected child identity declaration differs')
    for key in ('lineage_certificate_sha256', 'conversion_manifest_sha256'):
        identifier(value[key])
    pins = value['artifact_pins']
    need(type(pins) is dict and set(pins) == {'config_bytes', 'config_sha256', 'header_bytes',
         'header_sha256', 'container_bytes', 'container_sha256'}, 'Exact actual artifact pins required')
    for key, maximum in [('config_bytes', 64 * 1024), ('header_bytes', 256 * 1024),
                         ('container_bytes', BODY_BYTES + 256 * 1024 + 8)]:
        need(type(pins[key]) is int and 0 < pins[key] <= maximum, 'Bounded positive artifact extent required')
    need(pins['container_bytes'] == BODY_BYTES + 8 + pins['header_bytes'],
         'Complete artifact extent must bind its header and fixed body')
    for key in ('config_sha256', 'header_sha256', 'container_sha256'):
        identifier(pins[key])
    return value


def capture_child_manifest(raw, expected_sha256):
    need(type(raw) is bytes and 0 < len(raw) <= 64 * 1024
         and hashlib.sha256(raw).hexdigest() == identifier(expected_sha256),
         'Externally pinned bounded child identity bytes required')
    return validate_child_manifest(saved.loads(raw))
