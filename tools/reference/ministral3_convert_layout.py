"""Cold selected Ministral text layout and complete BF16 parent span checks.

All 458 source tensors are accounted for, including excluded rank-four vision
weights. Only the 183 selected text matrices and 53 exact norms enter the child.
No numerical package is imported by this module.
"""
import json
import re
import struct

import granite_convert_layout as generic
import ministral3_quantize as quant
from minicpm5_parent import unique

need = quant.need
MAX_HEADER = 2 * 1024**2
MAX_OBJECT = 8 * 1024**3
EXPECTED_OUTPUT = dict(body_bytes=2662004736, file_bytes=2662082312,
                       header_sha256='669cf6cf4405148e8c9ad3896be700a3dd765c2de352a1c5d476aa0865300413',
                       physical_tensors=602, text_tensors=236, affine4_matrices=130,
                       affine8_matrices=53, exact_norms=53, matrix_tiles=4252,
                       reconstructed_values=3428843520)


def bad_number(value):
    raise ValueError('Nonfinite JSON number: ' + value)


def leaf(name):
    need(type(name) is str and 0 < len(name) <= 255 and name not in ('.', '..')
         and re.fullmatch(r'[A-Za-z0-9_.-]+', name) is not None,
         'Safe source shard or metadata leaf required')
    return name


def selected_recipe():
    """Explicit fixed text geometry; query width4096 is independent of hidden3072."""
    matrices = [dict(module='language_model.model.embed_tokens', shape=[131072, 3072],
                     bits=8, group_size=64, mode='affine')]
    norms = [dict(name='language_model.model.norm.weight', shape=[3072], dtype='BF16', bytes=6144)]
    for layer in range(26):
        prefix = f'language_model.model.layers.{layer}.'
        for role, shape, bits in [
                ('self_attn.q_proj', [4096, 3072], 4),
                ('self_attn.k_proj', [1024, 3072], 4),
                ('self_attn.v_proj', [1024, 3072], 4),
                ('self_attn.o_proj', [3072, 4096], 8),
                ('mlp.gate_proj', [9216, 3072], 4),
                ('mlp.up_proj', [9216, 3072], 4),
                ('mlp.down_proj', [3072, 9216], 8)]:
            matrices.append(dict(module=prefix + role, shape=shape, bits=bits,
                                 group_size=64, mode='affine'))
        for role in ('input_layernorm', 'post_attention_layernorm'):
            norms.append(dict(name=prefix + role + '.weight', shape=[3072], dtype='BF16', bytes=6144))
    return (sorted(matrices, key=lambda item: item['module']),
            sorted(norms, key=lambda item: item['name']))


def selected_inventory():
    matrices, norms = selected_recipe()
    inventory = {item['module'] + '.weight': dict(dtype='BF16', shape=item['shape'])
                 for item in matrices}
    inventory.update({item['name']: dict(dtype='BF16', shape=item['shape']) for item in norms})
    exclusions = {
        'multi_modal_projector.linear_1.weight': [3072, 1024],
        'multi_modal_projector.linear_2.weight': [3072, 3072],
        'multi_modal_projector.norm.weight': [1024],
        'multi_modal_projector.patch_merger.merging_layer.weight': [1024, 4096],
        'vision_tower.ln_pre.weight': [1024],
        'vision_tower.patch_conv.weight': [1024, 3, 14, 14],
    }
    for layer in range(24):
        prefix = f'vision_tower.transformer.layers.{layer}.'
        for role, shape in [
                ('attention.k_proj', [1024, 1024]), ('attention.o_proj', [1024, 1024]),
                ('attention.q_proj', [1024, 1024]), ('attention.v_proj', [1024, 1024]),
                ('attention_norm', [1024]), ('feed_forward.down_proj', [1024, 4096]),
                ('feed_forward.gate_proj', [4096, 1024]), ('feed_forward.up_proj', [4096, 1024]),
                ('ffn_norm', [1024])]:
            exclusions[prefix + role + '.weight'] = shape
    inventory.update({name: dict(dtype='BF16', shape=shape) for name, shape in exclusions.items()})
    return dict(sorted(inventory.items()))


def output_layout(matrices, norms):
    """Sorted compact MLX header, tight U32 codes and separate BF16 metadata."""
    need(type(matrices) is list and matrices and type(norms) is list,
         'Explicit matrix and norm inventory required')
    declarations, logical, plans = {}, {}, {}

    def add(name, dtype, shape, size):
        need(type(name) is str and 0 < len(name) <= 1024 and name not in declarations
             and re.fullmatch(r'[A-Za-z0-9_.]+', name) is not None,
             'Unique bounded output tensor name required')
        declarations[name] = dict(dtype=dtype, shape=shape, bytes=size)

    for matrix in matrices:
        need(type(matrix) is dict and set(matrix) == {'module', 'shape', 'bits', 'group_size', 'mode'},
             'Complete exact matrix recipe required')
        module, shape, bits = matrix['module'], matrix['shape'], matrix['bits']
        need(type(module) is str and module.startswith('language_model.model.')
             and type(shape) is list and len(shape) == 2 and type(shape[0]) is int
             and 0 < shape[0] <= 131072, 'Bounded explicit text matrix shape required')
        rows, columns = shape
        quant.shape(1, columns, bits)
        need(type(matrix['group_size']) is int and matrix['group_size'] == 64
             and matrix['mode'] == 'affine', 'Explicit affine group-64 recipe required')
        name = module + '.weight'
        need(name not in logical, 'Duplicate logical source name')
        logical[name] = shape
        fields = {}
        for field, dtype, width, stride in [
                ('weight', 'U32', columns * bits // 32, columns * bits // 8),
                ('scales', 'BF16', columns // 64, columns // 64 * 2),
                ('biases', 'BF16', columns // 64, columns // 64 * 2)]:
            key = module + '.' + field
            add(key, dtype, [rows, width], rows * stride)
            fields[field] = dict(name=key, row_stride=stride)
        plans[name] = dict(shape=shape, bits=bits, fields=fields)
    for norm in norms:
        need(type(norm) is dict and set(norm) == {'name', 'shape', 'dtype', 'bytes'},
             'Complete exact norm recipe required')
        name, shape = norm['name'], norm['shape']
        need(type(name) is str and name.startswith('language_model.model.') and name.endswith('.weight')
             and type(shape) is list and len(shape) == 1 and type(shape[0]) is int
             and 0 < shape[0] <= 131072 and norm['dtype'] == 'BF16'
             and type(norm['bytes']) is int and norm['bytes'] == shape[0] * 2,
             'Exact BF16 text norm vector required')
        need(name not in logical, 'Duplicate logical source name')
        logical[name] = shape
        add(name, 'BF16', shape, norm['bytes'])
    header, cursor = {'__metadata__': {'format': 'mlx'}}, 0
    for name in sorted(declarations):
        item = declarations[name]
        end = cursor + item['bytes']
        need(end <= 6 * 1024**3, 'Output payload exceeds the engine ceiling')
        header[name] = dict(dtype=item['dtype'], shape=item['shape'], data_offsets=[cursor, end])
        cursor = end
    raw = json.dumps(header, separators=(',', ':'), sort_keys=True).encode()
    raw += b' ' * (-len(raw) % 8)
    need(0 < len(raw) <= MAX_HEADER, 'Bounded output header required')
    prefix = struct.pack('<Q', len(raw)) + raw
    return dict(header=header, prefix=prefix, origin=len(prefix), body_bytes=cursor,
                file_bytes=len(prefix) + cursor, logical=logical, matrices=plans)


def source_spans(prefix, header, size, inventory):
    """Strict tight BF16 spans, including rank4 excluded vision patch weights."""
    need(type(prefix) is bytes and len(prefix) == 8 and type(header) is bytes
         and 0 < len(header) <= MAX_HEADER and int.from_bytes(prefix, 'little') == len(header)
         and type(size) is int and len(header) + 8 < size <= MAX_OBJECT
         and type(inventory) is dict and inventory, 'Invalid complete source header extent')
    parsed = json.loads(header, object_pairs_hook=unique, parse_constant=bad_number)
    need(type(parsed) is dict, 'Source header must be an object')
    spans, origin = [], 8 + len(header)
    for name, item in parsed.items():
        if name == '__metadata__':
            need(type(item) is dict and all(type(k) is str and type(v) is str for k, v in item.items()),
                 'Source metadata requires string fields')
            continue
        need(type(name) is str and name in inventory and len(name) <= 1024
             and type(item) is dict and set(item) == {'dtype', 'shape', 'data_offsets'}
             and item['dtype'] == 'BF16', 'Source tensor declaration or inventory differs')
        expected = inventory[name]
        need(type(expected) is dict and set(expected) == {'dtype', 'shape'}
             and expected['dtype'] == 'BF16', 'Complete BF16 source inventory required')
        shape, offsets = item['shape'], item['data_offsets']
        need(type(shape) is list and len(shape) in (1, 2, 4)
             and all(type(n) is int and 0 < n <= 131072 for n in shape)
             and shape == expected['shape'] and type(offsets) is list and len(offsets) == 2
             and all(type(n) is int and 0 <= n <= size - origin for n in offsets),
             'Source tensor shape/span differs')
        count = 1
        for dimension in shape:
            need(count <= (2**63 - 1) // 2 // dimension, 'Source tensor size overflow')
            count *= dimension
        need(offsets[1] - offsets[0] == 2 * count, 'Source BF16 byte extent differs')
        spans.append(dict(name=name, dtype='BF16', shape=shape,
                          start=offsets[0], end=offsets[1]))
    spans.sort(key=lambda item: item['start'])
    need(spans and spans[0]['start'] == 0
         and all(a['end'] == b['start'] for a, b in zip(spans, spans[1:]))
         and origin + spans[-1]['end'] == size,
         'Source payload has holes, overlaps or wrong final extent')
    return spans


def validate_index(raw, objects, total_body_bytes, inventory=None):
    json.loads(raw, object_pairs_hook=unique, parse_constant=bad_number)
    need(type(objects) is list and objects and type(total_body_bytes) is int
         and 0 < total_body_bytes <= 2 * MAX_OBJECT, 'Complete bounded source objects required')
    names = set()
    for name, _ in objects:
        leaf(name)
        need(name not in names, 'Duplicate source shard identity')
        names.add(name)
    expected = generic.validate_index(raw, objects, total_body_bytes)
    if inventory is not None:
        need(type(inventory) is dict and set(expected) == set(inventory),
             'Incomplete complete-source inventory coverage')
    return expected


write_all = generic.write_all
write_norm = generic.write_norm


def write_matrix_tile(fd, layout, name, first, rows, buffers):
    need(type(rows) is int and 1 <= rows <= quant.ROWS,
         'Bounded Ministral matrix row tile required')
    need(name in layout['matrices'], 'Declared matrix tile required')
    quant.shape(rows, layout['matrices'][name]['shape'][1], layout['matrices'][name]['bits'])
    return generic.write_matrix_tile(fd, layout, name, first, rows, buffers)
