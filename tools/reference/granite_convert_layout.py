"""Cold affine4/8 output layout and bounded file I/O; no numerical imports."""
import json
import os
import struct

import granite_quantize as quant
from minicpm5_parent import plan as parent_plan, unique

need = quant.need


def output_layout(matrices, norms):
    """Row-major U32 codes, separate BF16 metadata/norms, 8-byte body alignment."""
    need(type(matrices) is list and matrices and type(norms) is list,
         'Explicit matrix and norm inventory required')
    declarations = {}
    logical = {}
    plans = {}

    def add(name, dtype, shape, size):
        need(type(name) is str and 0 < len(name) <= 1024 and name not in declarations
             and name != '__metadata__', 'Unique bounded output tensor name required')
        declarations[name] = dict(dtype=dtype, shape=shape, bytes=size)

    for matrix in matrices:
        need(type(matrix) is dict and {'module', 'shape', 'bits', 'group_size', 'mode'} <= set(matrix),
             'Complete matrix recipe required')
        module, shape, bits = matrix['module'], matrix['shape'], matrix['bits']
        need(type(module) is str and module and type(shape) is list and len(shape) == 2
             and type(shape[0]) is int and 0 < shape[0] <= 131072,
             'Bounded explicit matrix shape required')
        rows, columns = shape
        quant.shape(1, columns, bits)
        need(matrix['group_size'] == 64 and type(matrix['group_size']) is int
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
        need(type(norm) is dict and {'name', 'shape', 'dtype', 'bytes'} <= set(norm),
             'Complete norm recipe required')
        name, shape = norm['name'], norm['shape']
        need(type(name) is str and name.endswith('.weight')
             and type(shape) is list and len(shape) == 1 and type(shape[0]) is int
             and 0 < shape[0] <= 131072 and norm['dtype'] == 'BF16'
             and type(norm['bytes']) is int and norm['bytes'] == shape[0] * 2,
             'Exact BF16 norm vector required')
        need(name not in logical, 'Duplicate logical source name')
        logical[name] = shape
        add(name, 'BF16', shape, norm['bytes'])
    header = {'__metadata__': {'format': 'mlx'}}
    cursor = 0
    for name in sorted(declarations):
        item = declarations[name]
        end = cursor + item['bytes']
        need(end <= 6 * 1024**3, 'Output payload exceeds the engine ceiling')
        header[name] = dict(dtype=item['dtype'], shape=item['shape'], data_offsets=[cursor, end])
        cursor = end
    raw = json.dumps(header, separators=(',', ':'), sort_keys=True).encode()
    raw += b' ' * (-len(raw) % 8)
    need(0 < len(raw) <= 2 * 1024**2, 'Bounded output header required')
    prefix = struct.pack('<Q', len(raw)) + raw
    return dict(header=header, prefix=prefix, origin=len(prefix), body_bytes=cursor,
                file_bytes=len(prefix) + cursor, logical=logical, matrices=plans)


def source_spans(prefix, header, size, logical):
    """Reuse the unchanged complete BF16 span validator, then bind logical roles."""
    spans = parent_plan(prefix, header, size)
    for span in spans:
        need(span['name'] in logical and span['shape'] == logical[span['name']],
             'Source tensor name or shape differs from the frozen recipe')
    return spans


def validate_index(raw, objects, total_body_bytes):
    """Never repair a publisher's inconsistent index total or shard mapping."""
    index = json.loads(raw, object_pairs_hook=unique)
    need(type(index) is dict and set(index) == {'metadata', 'weight_map'}
         and type(index['metadata']) is dict
         and set(index['metadata']) in ({'total_size'}, {'total_size', 'total_parameters'})
         and type(index['metadata']['total_size']) is int
         and index['metadata']['total_size'] == total_body_bytes,
         'Source index physical byte total differs')
    if 'total_parameters' in index['metadata']:
        need(type(index['metadata']['total_parameters']) is int
             and index['metadata']['total_parameters'] * 2 == total_body_bytes,
             'Source index BF16 parameter count differs')
    expected = {}
    for filename, spans in objects:
        need(type(filename) is str and filename and '/' not in filename and '\\' not in filename,
             'Source index requires safe shard leaves')
        for span in spans:
            need(span['name'] not in expected, 'Duplicate source tensor across shards')
            expected[span['name']] = filename
    need(type(index['weight_map']) is dict and index['weight_map'] == expected,
         'Source index tensor-to-shard map differs')
    return expected


def write_all(fd, raw, offset):
    need(type(raw) is bytes and type(offset) is int and offset >= 0
         and len(raw) <= 4 * 1024**2, 'Bounded bytes and output offset required')
    view = memoryview(raw)
    cursor = 0
    while cursor < len(view):
        written = os.pwrite(fd, view[cursor:], offset + cursor)
        need(0 < written <= len(view) - cursor, 'Output write failed to advance')
        cursor += written


def write_matrix_tile(fd, layout, name, first, rows, buffers):
    need(name in layout['matrices'] and type(first) is int and first >= 0
         and type(rows) is int and 1 <= rows <= quant.ROWS,
         'Bounded matrix row tile required')
    plan = layout['matrices'][name]
    need(first + rows <= plan['shape'][0], 'Matrix tile exceeds its row extent')
    need(type(buffers) is dict and set(buffers) == set(plan['fields']),
         'Complete three-buffer affine tile required')
    # Validate every component before the first write.
    for field, item in plan['fields'].items():
        need(type(buffers[field]) is bytes and len(buffers[field]) == rows * item['row_stride'],
             'Matrix tile component extent differs')
    for field, item in plan['fields'].items():
        offset = layout['origin'] + layout['header'][item['name']]['data_offsets'][0]
        offset += first * item['row_stride']
        write_all(fd, buffers[field], offset)
        need(os.pread(fd, len(buffers[field]), offset) == buffers[field],
             'Written matrix component differs on readback')


def write_norm(fd, layout, name, raw):
    need(name in layout['logical'] and len(layout['logical'][name]) == 1,
         'Declared norm vector required')
    item = layout['header'][name]
    start, end = item['data_offsets']
    need(type(raw) is bytes and len(raw) == end - start, 'Exact norm byte extent required')
    offset = layout['origin'] + start
    write_all(fd, raw, offset)
    need(os.pread(fd, len(raw), offset) == raw, 'Written norm differs on readback')
