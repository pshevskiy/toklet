"""Cold saved-byte and ordered-journal audit of a frozen Ministral conversion.

This program rehashes actual source/child bytes. It reconciles captured independent
NumPy/upstream FP32 witness hashes and equalities; it does NOT recompute FP32,
execute the admitted runtime, or qualify model numerics, quality, or adoption.
Only stdlib imports are used. Hash reads are <=1 MiB and tile reads <=4 MiB.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct

HASH_BLOCK = 1024 ** 2
TILE_BLOCK = 4 * HASH_BLOCK
METADATA_LIMIT = 4 * HASH_BLOCK
JOURNAL_LIMIT = 32 * HASH_BLOCK
ROWS = 224
COLUMNS = 9216
VERSIONS = {'numpy': '2.3.2', 'mlx': '0.32.3'}
EXPECTED = dict(body_bytes=2662004736, file_bytes=2662082312,
                header_sha256='669cf6cf4405148e8c9ad3896be700a3dd765c2de352a1c5d476aa0865300413',
                physical_tensors=602, text_tensors=236, affine4_matrices=130,
                affine8_matrices=53, exact_norms=53, matrix_tiles=4252,
                reconstructed_values=3428843520)
PREDICATES = ('source_authentication', 'component_readback',
              'independent_upstream_f32_equal', 'finite_text_input',
              'finite_metadata', 'finite_reconstruction', 'norm_bytes_equal',
              'endpoint_unchanged')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
         'Lowercase external SHA256 required')
    return value


def uint(value, maximum=2 ** 63 - 1):
    need(type(value) is int and 0 <= value <= maximum, 'Bounded integer required')
    return value


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON field')
        result[key] = value
    return result


def bad_number(value):
    raise ValueError('Nonfinite JSON number: ' + value)


def finite_number(value):
    number = float(value)
    need(math.isfinite(number), 'Nonfinite JSON number')
    return number


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique, parse_constant=bad_number,
                      parse_float=finite_number)


def same(actual, expected, message):
    """JSON equality must not silently equate bools, integers and floats."""
    need(type(actual) is type(expected), message)
    if type(expected) is dict:
        need(set(actual) == set(expected), message)
        for key in expected:
            same(actual[key], expected[key], message)
    elif type(expected) is list:
        need(len(actual) == len(expected), message)
        for left, right in zip(actual, expected):
            same(left, right, message)
    else:
        need(actual == expected, message)


def identity(value):
    return [value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns]


def pin_identity(value):
    need(type(value) is list and len(value) == 5, 'Five-field file identity required')
    for item in value:
        uint(item)
    return value


class SavedFile:
    """Hold one no-follow regular descriptor and verify its endpoint identity."""
    def __init__(self, path, size=None):
        self.path = Path(path)
        need(self.path.is_absolute(), 'Absolute saved path required')
        self.fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            self.before = self.check()
            if size is not None:
                need(self.before[2] == uint(size), 'Saved file extent differs')
        except BaseException:
            os.close(self.fd)
            raise

    def check(self):
        owned, named = os.fstat(self.fd), self.path.lstat()
        need(stat.S_ISREG(owned.st_mode) and stat.S_ISREG(named.st_mode)
             and identity(owned) == identity(named), 'Descriptor and named regular file differ')
        result = identity(owned)
        if hasattr(self, 'before'):
            same(result, self.before, 'Saved descriptor or path mutated')
        return result

    def read(self, count, offset=0):
        uint(count, TILE_BLOCK)
        uint(offset)
        need(offset + count <= os.fstat(self.fd).st_size, 'Read exceeds saved extent')
        result = bytearray()
        while len(result) < count:
            raw = os.pread(self.fd, count - len(result), offset + len(result))
            need(type(raw) is bytes and 0 < len(raw) <= count - len(result), 'Short saved-file read')
            result.extend(raw)
        return bytes(result)

    def digest(self, count=None, offset=0):
        count = self.before[2] if count is None else uint(count)
        uint(offset)
        need(offset + count <= self.before[2], 'Hash exceeds saved extent')
        result = hashlib.sha256()
        for first in range(0, count, HASH_BLOCK):
            result.update(self.read(min(HASH_BLOCK, count - first), offset + first))
        self.check()
        return result.hexdigest()

    def close(self):
        try:
            self.check()
        finally:
            os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def capture(path, maximum, expected=None):
    uint(maximum, JOURNAL_LIMIT)
    with SavedFile(path) as saved:
        need(saved.before[2] <= maximum, 'Oversize saved metadata')
        raw = b''.join(saved.read(min(HASH_BLOCK, saved.before[2] - first), first)
                       for first in range(0, saved.before[2], HASH_BLOCK))
        if expected is not None:
            need(hashlib.sha256(raw).hexdigest() == sha(expected), 'Saved metadata SHA256 differs')
        return raw


def metadata_descriptor(receipt, path, maximum, expected_sha256, extra=None, capture_file=capture):
    """Validate the complete producer metadata receipt against held actual bytes.

    Config receipts carry both descriptor identities. Ancillary receipts add
    their source leaf and source path. Exact schema checks reject unknown fields;
    both identities must describe the same current regular file, even when a
    replacement has identical bytes. capture_file also retains audit-wide path
    bindings when the production caller supplies bound_capture.
    """
    path = Path(path)
    fields = {'path', 'bytes', 'sha256', 'before_identity', 'after_identity'}
    if extra is not None:
        need(type(extra) is dict and not (set(extra) & fields),
             'Distinct explicit metadata lineage fields required')
        fields |= set(extra)
    need(type(receipt) is dict and set(receipt) == fields,
         'Complete exact identity-bearing metadata descriptor required')
    with SavedFile(path, receipt['bytes']) as saved:
        expected = dict(path=str(path), bytes=saved.before[2], sha256=sha(expected_sha256),
                        before_identity=saved.before, after_identity=saved.before)
        if extra is not None:
            expected.update(extra)
        same(receipt, expected, 'Identity-bearing metadata descriptor differs')
        raw = capture_file(path, maximum, expected_sha256)
        same(len(raw), expected['bytes'], 'Identity-bearing metadata extent differs')
        saved.check()
        return raw


def leaf(value):
    need(type(value) is str and value not in ('', '.', '..')
         and '/' not in value and '\\' not in value, 'Safe source shard leaf required')
    return value


def dimensions(shape, rank):
    need(type(shape) is list and len(shape) == rank
         and all(type(n) is int and 0 < n <= 131072 for n in shape), 'Explicit positive shape required')
    return shape


def output_layout(matrices, norms):
    """Independent cold construction of the exact sorted packed child prefix."""
    need(type(matrices) is list and type(norms) is list, 'Recipe lists required')
    declarations, plans, logical = {}, {}, {}

    def add(name, dtype, shape, size):
        need(type(name) is str and 0 < len(name) <= 1024
             and name not in declarations and name != '__metadata__', 'Duplicate child tensor')
        declarations[name] = dict(dtype=dtype, shape=shape, bytes=size)

    for row in matrices:
        need(type(row) is dict and set(row) == {'module', 'shape', 'bits', 'group_size', 'mode'},
             'Exact affine matrix recipe required')
        module, shape, bits = row['module'], dimensions(row['shape'], 2), row['bits']
        need(type(module) is str and module.startswith('language_model.model.')
             and type(bits) is int and bits in (4, 8)
             and type(row['group_size']) is int and row['group_size'] == 64
             and row['mode'] == 'affine' and 64 <= shape[1] <= COLUMNS and shape[1] % 64 == 0,
             'Unsupported affine geometry')
        name = module + '.weight'
        need(name not in logical, 'Duplicate matrix recipe')
        logical[name] = dict(dtype='BF16', shape=shape)
        fields = {}
        for key, dtype, width, stride in (
                ('weight', 'U32', shape[1] * bits // 32, shape[1] * bits // 8),
                ('scales', 'BF16', shape[1] // 64, shape[1] // 32),
                ('biases', 'BF16', shape[1] // 64, shape[1] // 32)):
            field = module + '.' + key
            add(field, dtype, [shape[0], width], shape[0] * stride)
            fields[key] = dict(name=field, row_stride=stride)
        plans[name] = dict(shape=shape, bits=bits, fields=fields)
    for row in norms:
        need(type(row) is dict and set(row) == {'name', 'shape', 'dtype', 'bytes'}, 'Exact norm recipe required')
        name, shape = row['name'], dimensions(row['shape'], 1)
        need(type(name) is str and name.startswith('language_model.model.') and name.endswith('.weight')
             and row['dtype'] == 'BF16' and uint(row['bytes']) == shape[0] * 2
             and name not in logical, 'Invalid norm recipe')
        logical[name] = dict(dtype='BF16', shape=shape)
        add(name, 'BF16', shape, row['bytes'])
    header, cursor = {'__metadata__': {'format': 'mlx'}}, 0
    for name in sorted(declarations):
        row = declarations[name]
        header[name] = dict(dtype=row['dtype'], shape=row['shape'],
                            data_offsets=[cursor, cursor + row['bytes']])
        cursor += row['bytes']
    raw = json.dumps(header, separators=(',', ':'), sort_keys=True).encode()
    raw += b' ' * (-len(raw) % 8)
    prefix = struct.pack('<Q', len(raw)) + raw
    need(len(prefix) <= METADATA_LIMIT and cursor <= 6 * 1024 ** 3, 'Bounded child layout required')
    return dict(header=header, prefix=prefix, origin=len(prefix), body_bytes=cursor,
                file_bytes=cursor + len(prefix), matrices=plans, logical=logical)


def source_spans(prefix, raw, size, inventory):
    need(type(prefix) is bytes and len(prefix) == 8 and type(raw) is bytes
         and 0 < len(raw) <= 2 * HASH_BLOCK and struct.unpack('<Q', prefix)[0] == len(raw),
         'Bounded source header required')
    header, spans = loads(raw), []
    need(type(header) is dict, 'Source header object required')
    for name, item in header.items():
        if name == '__metadata__':
            need(type(item) is dict, 'Source metadata object required')
            continue
        need(type(item) is dict and set(item) == {'dtype', 'shape', 'data_offsets'}
             and name in inventory, 'Unknown source tensor')
        same(dict(dtype=item['dtype'], shape=item['shape']), inventory[name], 'Source inventory differs')
        dimensions(item['shape'], len(item['shape']))
        need(1 <= len(item['shape']) <= 4 and item['dtype'] == 'BF16', 'BF16 rank1-4 source required')
        offsets = item['data_offsets']
        need(type(offsets) is list and len(offsets) == 2, 'Exact source offsets required')
        start, end = (uint(value) for value in offsets)
        need(end - start == 2 * math.prod(item['shape']), 'Source byte extent differs')
        spans.append(dict(name=name, shape=item['shape'], start=start, end=end))
    spans.sort(key=lambda row: row['start'])
    need(spans and spans[0]['start'] == 0
         and all(a['end'] == b['start'] for a, b in zip(spans, spans[1:]))
         and spans[-1]['end'] + 8 + len(raw) == uint(size), 'Source gaps, overlaps or extent differ')
    return spans


def validate_index(raw, objects, total):
    index = loads(raw)
    need(type(index) is dict and set(index) == {'metadata', 'weight_map'}
         and type(index['metadata']) is dict
         and set(index['metadata']) in ({'total_size'}, {'total_size', 'total_parameters'}),
         'Exact source index required')
    need(uint(index['metadata']['total_size']) == total, 'Source index byte total differs')
    if 'total_parameters' in index['metadata']:
        need(uint(index['metadata']['total_parameters']) * 2 == total, 'Source index parameter count differs')
    owners = {}
    for name, spans in objects:
        leaf(name)
        for row in spans:
            need(row['name'] not in owners, 'Duplicate tensor shard ownership')
            owners[row['name']] = name
    same(index['weight_map'], owners, 'Source index owner map differs')


def contract_layout(contract):
    need(type(contract) is dict and contract.get('format') == 'toklet-ministral3-heretic-mixed-affine-v1',
         'Frozen Ministral contract required')
    sha(contract['source_authentication_receipt_sha256'])
    same(contract['expected_output'], EXPECTED, 'Production fixed inventory differs')
    recipe = contract['recipe']
    for key, expected in dict(device='gpu', input_dtype='BF16', metadata_dtype='BF16',
                              maximum_rows_per_tile=ROWS, maximum_columns=COLUMNS,
                              group_size=64, quantization_runtime=VERSIONS).items():
        same(recipe[key], expected, 'Frozen mechanism differs')
    planned = output_layout(recipe['matrices'], recipe['norms'])
    need(len(recipe['matrices']) == 183 and len(recipe['norms']) == 53
         and sum(row['bits'] == 4 for row in recipe['matrices']) == 130
         and sum(row['bits'] == 8 for row in recipe['matrices']) == 53,
         'Complete mixed recipe required')
    exclusions = contract['source_exclusions']
    need(type(exclusions) is list and len(exclusions) == 222, 'Complete exclusion inventory required')
    inventory = dict(planned['logical'])
    for row in exclusions:
        need(type(row) is dict and set(row) == {'name', 'shape', 'dtype'}
             and row['name'].startswith(('vision_tower.', 'multi_modal_projector.'))
             and row['name'] not in inventory and row['dtype'] == 'BF16', 'Explicit vision exclusion required')
        dimensions(row['shape'], len(row['shape']))
        need(1 <= len(row['shape']) <= 4, 'Bounded exclusion rank required')
        inventory[row['name']] = dict(dtype=row['dtype'], shape=row['shape'])
    same(inventory, contract['expected_source_inventory'], 'Complete source inventory differs')
    need(len(inventory) == 458 and len(planned['header']) - 1 == 602
         and planned['body_bytes'] == EXPECTED['body_bytes']
         and planned['file_bytes'] == EXPECTED['file_bytes']
         and hashlib.sha256(planned['prefix']).hexdigest() == EXPECTED['header_sha256']
         and sum((row['shape'][0] + ROWS - 1) // ROWS for row in recipe['matrices']) == 4252
         and sum(math.prod(row['shape']) for row in recipe['matrices']) == 3428843520,
         'Frozen child layout differs')
    for key, value in dict(audit_limit_bytes=512 * HASH_BLOCK, helper_reserve_bytes=32 * HASH_BLOCK,
                           engine_limit_bytes=6 * 1024 ** 3, ordinary_Metal_context_cap=512).items():
        same(contract[key], value, 'Frozen memory/context policy differs')
    for key in ('model_full_graph_quality_codec_fit_request_performance_adoption_qualified',
                'whole_goal_complete', 'numerical_outputs_exist'):
        same(contract[key], False, 'Prospective contract scope differs')
    return planned, inventory


def component(saved, declaration, first=0, rows=None):
    start, end = declaration['data_offsets']
    stride = (end - start) // declaration['shape'][0]
    count = end - start if rows is None else rows * stride
    offset = first * stride + start
    return dict(dtype=declaration['dtype'], shape=declaration['shape'] if rows is None else
                [rows] + declaration['shape'][1:], offset=offset, bytes=count,
                sha256=saved.digest(count, offset))


def witness(row, rows, columns):
    need(row.get('reconstruction_equal') is True, 'Unqualified FP32 witness')
    same(row['reconstructed_values'], rows * columns, 'FP32 witness count differs')
    need(sha(row['independent_F32_sha256']) == sha(row['upstream_F32_sha256']),
         'Captured independent/upstream FP32 hashes differ')
    for key in ('source_read_ns', 'completed_quantize_readback_ns',
                'independent_decode_ns', 'completed_upstream_decode_ns'):
        uint(row[key])


class Journal:
    """Bounded one-line lookahead, exact event sequence, immutable endpoint."""
    def __init__(self, saved, receipt):
        self.saved, self.offset, self.records = saved, 0, 0
        self.expected_records = uint(receipt['records'], 5000)
        need(saved.before[2] <= JOURNAL_LIMIT and saved.digest() == sha(receipt['sha256']),
             'Journal bytes differ')

    def take(self, event):
        remaining = self.saved.before[2] - self.offset
        need(remaining > 0, 'Missing journal record')
        raw = self.saved.read(min(remaining, 64 * 1024), self.offset)
        end = raw.find(b'\n')
        need(end >= 0 and end > 0, 'Bounded newline-terminated journal record required')
        self.offset += end + 1
        self.records += 1
        row = loads(raw[:end])
        need(type(row) is dict and row.get('event') == event, 'Journal order/event differs')
        if event != 'parent_authenticated':
            need(row.get('parent_object_authenticated') is False, 'Premature parent qualification')
        return row

    def matches(self, event, report):
        row = self.take(event)
        expected = dict(report, event=event)
        if event != 'parent_authenticated':
            expected['parent_object_authenticated'] = False
        same(row, expected, 'Journal/manifest receipt differs')

    def finish(self):
        need(self.offset == self.saved.before[2] and self.records == self.expected_records,
             'Extra, missing or duplicate journal records')
        self.saved.check()


def reports(rows, names):
    need(type(rows) is list and len(rows) == len(names), 'Complete manifest receipts required')
    result = {}
    for row in rows:
        need(type(row) is dict and row.get('name') in names and row['name'] not in result,
             'Unknown or duplicate manifest tensor receipt')
        result[row['name']] = row
    need(set(result) == set(names), 'Missing manifest tensor receipt')
    return result


def source_descriptor(report, file, span, origin, digest):
    for key, value in dict(name=span['name'], source_file=file, source_offset=origin + span['start'],
                           source_bytes=span['end'] - span['start'], source_sha256=digest).items():
        same(report[key], value, 'Source tensor lineage differs')


def endpoints(report, saved):
    same(pin_identity(report['before_identity']), saved.before, 'Captured initial identity differs')
    same(pin_identity(report['after_identity']), saved.before, 'Captured final identity differs')


def runtime_receipts(manifest, observed=None, capture_file=capture):
    receipt = manifest['runtime']
    raw = capture_file(receipt['path'], METADATA_LIMIT, receipt['sha256'])
    same(len(raw), receipt['bytes'], 'Runtime manifest extent differs')
    declared = loads(raw)
    need(type(declared) is dict and set(declared) == {'format', 'versions', 'files'}
         and declared['format'] == 'toklet-granite-quant-runtime-v1'
         and type(declared['files']) is dict and declared['files'], 'Pinned runtime manifest required')
    same(declared['versions'], VERSIONS, 'Runtime versions differ')
    admitted = manifest['admitted_runtime']
    same(admitted['manifest_sha256'], receipt['sha256'], 'Admitted runtime pin differs')
    same(admitted['versions'], VERSIONS, 'Admitted runtime versions differ')
    same(admitted['files'], len(declared['files']), 'Runtime inventory count differs')
    same(admitted['streaming_hash_chunk_limit_bytes'], HASH_BLOCK, 'Runtime hash bound differs')
    need(type(admitted['pycache_prefix']) is str and Path(admitted['pycache_prefix']).is_absolute(),
         'Captured absolute cache prefix required')
    for path, pin in declared['files'].items():
        need(type(pin) is dict and set(pin) == {'bytes', 'sha256'}, 'Runtime file pin required')
        with SavedFile(path, uint(pin['bytes'], 512 * HASH_BLOCK)) as saved:
            if observed is not None:
                if path in observed:
                    same(saved.before, observed[path], 'Runtime file endpoint identity changed')
                else:
                    observed[path] = saved.before
            need(saved.digest() == sha(pin['sha256']), 'Runtime saved bytes differ')
    initial, final = admitted['initial_origins'], manifest['endpoint_origins']
    for origins in (initial, final):
        need(type(origins) is dict and set(origins) == {
            'modules', 'non_system_native_images', 'system_native_images'}, 'Runtime origins required')
        need(type(origins['modules']) is dict and origins['modules'], 'Captured modules required')
        for name, path in origins['modules'].items():
            need(type(name) is str and path in declared['files'], 'Unadmitted module origin')
        for name, path in admitted['imports'].items():
            same(origins['modules'].get(name), path, 'Selected numerical origin differs')
        need(set(admitted['imports']) == {'numpy', 'mlx.core', 'mlx.nn'}, 'Selected numerical imports differ')
        need(type(origins['non_system_native_images']) is dict, 'Captured native image map required')
        for path, pin in origins['non_system_native_images'].items():
            need(path in declared['files'], 'Unadmitted native image origin')
            same(pin, declared['files'][path], 'Native image pin differs')
        system = origins['system_native_images']
        need(type(system) is list and system == sorted(set(system))
             and all(type(path) is str and path.startswith(('/usr/lib/', '/System/')) for path in system),
             'System native image origin differs')
    for key, value in initial['modules'].items():
        same(final['modules'].get(key), value, 'Initial module disappeared or changed')
    for key, value in initial['non_system_native_images'].items():
        same(final['non_system_native_images'].get(key), value, 'Initial native image changed')
    return len(declared['files'])


def child_config(config, matrices):
    need(type(config) is dict and 'quantization' not in config and 'quantization_config' not in config,
         'Unquantized source configuration required')
    result = dict(config)
    recipe = dict(group_size=64, bits=4, mode='affine')
    for row in matrices:
        if row['bits'] == 8:
            recipe[row['module']] = dict(group_size=64, bits=8, mode='affine')
    result['quantization'] = recipe
    return result


def check_tile(tile, source, destination, span, plan, planned, source_file, origin, first, rows):
    """Audit one complete saved row tile, independently of production inventory size."""
    need(type(tile) is dict and set(tile) == {
        'event', 'parent_object_authenticated', 'name', 'first_row', 'rows', 'columns',
        'bits', 'source_file', 'source_offset', 'source_bytes', 'source_sha256', 'fields',
        'independent_F32_sha256', 'upstream_F32_sha256', 'reconstructed_values',
        'reconstruction_equal', 'source_read_ns', 'completed_quantize_readback_ns',
        'independent_decode_ns', 'completed_upstream_decode_ns', 'component_write_readback_ns',
        'component_readback'}, 'Exact tile witness fields required')
    need(tile['event'] == 'matrix_tile' and tile['parent_object_authenticated'] is False
         and tile['component_readback'] is True, 'Unqualified tile receipt')
    uint(tile['component_write_readback_ns'])
    columns = span['shape'][1]
    expected = dict(name=span['name'], source_file=source_file, first_row=first, rows=rows,
                    columns=columns, bits=plan['bits'],
                    source_offset=origin + span['start'] + first * columns * 2,
                    source_bytes=rows * columns * 2)
    for key, value in expected.items():
        same(tile[key], value, 'Matrix tile coverage/lineage differs')
    need(tile['source_bytes'] <= TILE_BLOCK
         and source.digest(tile['source_bytes'], tile['source_offset']) == sha(tile['source_sha256']),
         'Actual source tile bytes differ')
    fields = {}
    for key, field in plan['fields'].items():
        declaration = dict(planned['header'][field['name']])
        declaration['data_offsets'] = [value + planned['origin'] for value in declaration['data_offsets']]
        fields[key] = dict(name=field['name'], **component(destination, declaration, first, rows))
    same(tile['fields'], fields, 'Actual child tile bytes differ')
    witness(tile, rows, columns)
    return rows * columns


def source_authentication(authentication, contract):
    need(type(authentication) is dict and authentication.get('passed') is True
         and authentication.get('remaining_expected_bytes') == 0
         and type(authentication.get('remaining_expected_bytes')) is int,
         'Positive complete source authentication required')
    sources = authentication['sources']
    need(type(sources) is list and len(sources) == 2, 'Complete two-source authentication required')
    selected = None
    for source in (contract['original_source'], contract['source']):
        matches = [row for row in sources if type(row) is dict
                   and all(row.get(key) == source[key] for key in ('repository', 'revision', 'directory'))]
        need(len(matches) == 1, 'Exact prior authenticated source required')
        row = matches[0]
        need(row.get('tensor_payloads_authenticated') is True, 'Unqualified prior source payloads')
        containers = row['containers']
        need(type(containers) is list and len(containers) == 2, 'Complete two-object source authentication required')
        for receipt, pin in zip(containers, source['containers']):
            for key, value in dict(file=pin['container'], bytes=pin['expected_bytes'],
                                   sha256=pin['expected_sha256'], pinned_header_ranges_match=True,
                                   payload_authenticated=True).items():
                same(receipt[key], value, 'Prior source object pin differs')
        metadata = row['metadata']
        need(type(metadata) is list, 'Source metadata receipts required')
        by_name = {}
        for item in metadata:
            name = leaf(item['file'])
            need(name not in by_name, 'Duplicate source metadata receipt')
            uint(item['bytes'], 2 * HASH_BLOCK)
            sha(item['sha256'])
            by_name[name] = item
        for name, key in (('config.json', 'config_sha256'), ('model.safetensors.index.json', 'index_sha256')):
            same(by_name[name]['sha256'], source[key], 'Prior source metadata pin differs')
        if source is contract['source']:
            selected = row
    return selected


def source_guard(guard):
    need(type(guard) is dict and guard.get('passed') is True
         and 'failure' in guard and guard['failure'] is None,
         'Unqualified source acquisition guard')
    same(guard['exit_code'], 0, 'Source acquisition exit code differs')
    for key in ('maximum_observed_charge_with_reserve', 'terminal_direct_pair_rss_charge_with_reserve'):
        uint(guard[key], 512 * HASH_BLOCK)
    need(uint(guard['minimum_free_bytes']) >= 2 * 1024 ** 3 and uint(guard['sample_count']) > 0,
         'Source acquisition disk floor or monitor samples differ')
    same(guard['owned_live_after_cleanup'], [], 'Source acquisition owned process remains')
    same(guard['cleanup_errors'], [], 'Source acquisition cleanup failed')
    gate = guard['last_runtime_gate']
    need(type(gate) is dict and gate.get('passed') is True, 'Unqualified final source storage gate')
    for key, value in dict(remaining_download_bytes=0, child_bytes_reserved=EXPECTED['file_bytes'],
                           disk_floor_bytes=2 * 1024 ** 3, work_reserve_bytes=256 * HASH_BLOCK).items():
        same(gate[key], value, 'Final source storage reservation differs')
    same(gate['required_remaining_bytes'], EXPECTED['file_bytes'] + 2 * 1024 ** 3 + 256 * HASH_BLOCK,
         'Final source reservation sum differs')
    need(uint(gate['free_bytes']) >= gate['required_remaining_bytes'], 'Final source storage admission failed')


def source_closure(authentication, contract, capture_file=capture):
    """Bind the independently guarded complete-source receipt without executing it."""
    selected = source_authentication(authentication, contract)
    same(authentication['format'], 'toklet-ministral3-complete-source-authentication-v1',
         'Source authentication format differs')
    repository = Path(authentication['repository_root'])
    need(repository.is_absolute(), 'Absolute source repository root required')
    for source in (contract['source'], contract['original_source']):
        directory = Path(source['directory'])
        need(directory.parent == repository / 'models' and directory.name not in ('', '.', '..'),
             'Source directory ancestry differs')
    raw_pin, guard_pin = authentication['raw_receipt'], authentication['download_guard']
    raw = capture_file(raw_pin['path'], METADATA_LIMIT, raw_pin['sha256'])
    same(len(raw), raw_pin['bytes'], 'Raw source authentication extent differs')
    captured = loads(raw)
    for key in ('format', 'passed', 'sources', 'remaining_expected_bytes'):
        same(captured[key], authentication[key], 'Raw and closed source authentication differ')
    need(guard_pin['passed'] is True, 'Unqualified source download guard receipt')
    guard_raw = capture_file(guard_pin['path'], METADATA_LIMIT, guard_pin['sha256'])
    same(len(guard_raw), guard_pin['bytes'], 'Source guard receipt extent differs')
    guard = loads(guard_raw)
    source_guard(guard)
    wrapper_pin = authentication['acquisition_wrapper']
    wrapper_raw = capture_file(wrapper_pin['path'], METADATA_LIMIT, wrapper_pin['sha256'])
    same(len(wrapper_raw), wrapper_pin['bytes'], 'Source acquisition wrapper extent differs')
    wrapper = loads(wrapper_raw)
    need(type(wrapper) is dict and wrapper.get('passed') is True, 'Unqualified acquisition wrapper')
    same(wrapper['guard'], guard, 'Acquisition wrapper guard differs')
    same(wrapper['sources_before'], wrapper['sources_after'], 'Source acquisition code or plan changed')
    need(type(wrapper['sources_before']) is dict and wrapper['sources_before'],
         'Source acquisition code and plan lease required')
    same(wrapper['numerical_runtime_or_model_requested'], False, 'Source acquisition scope differs')
    for path, expected in wrapper['sources_before'].items():
        need(type(path) is str and Path(path).is_absolute(), 'Absolute acquisition code or plan lease required')
        capture_file(path, METADATA_LIMIT, sha(expected))
    for source in authentication['sources']:
        for container in source['containers']:
            relative = container['download_result']
            need(type(relative) is str and relative and not Path(relative).is_absolute()
                 and '..' not in Path(relative).parts and Path(relative).name == 'authenticated.json',
                 'Safe repository-relative download receipt required')
            downloaded = loads(capture_file(repository / relative, METADATA_LIMIT))
            same(downloaded, container, 'Captured download and source object receipts differ')
            pin_identity(container['final_identity'])
    return selected


def audit(contract_path, contract_sha256, manifest_path, manifest_sha256):
    """Production entry point. Fixed inventories are never parameterized for tests."""
    metadata_bindings = {}

    def bound_capture(path, maximum, expected=None):
        path = Path(path)
        before = identity(path.lstat())
        raw = capture(path, maximum, expected)
        same(identity(path.lstat()), before, 'Captured metadata endpoint changed')
        if path in metadata_bindings:
            same(before, metadata_bindings[path], 'Captured metadata identity changed')
        else:
            metadata_bindings[path] = before
        return raw

    contract_raw = bound_capture(contract_path, METADATA_LIMIT, contract_sha256)
    contract = loads(contract_raw)
    planned, inventory = contract_layout(contract)
    manifest_raw = bound_capture(manifest_path, METADATA_LIMIT, manifest_sha256)
    manifest = loads(manifest_raw)
    need(type(manifest) is dict and manifest.get('format') == 'toklet-ministral3-mixed-affine-conversion-v1'
         and manifest.get('passed') is True, 'Complete positive producer manifest required')
    need(set(manifest) == {'format', 'passed', 'contract_sha256', 'source_authentication',
         'source_authentication_lineage', 'runtime', 'source_repository', 'source_revision',
         'source_objects', 'admitted_runtime', 'endpoint_origins', 'journal', 'child', 'matrices',
         'exact_norms', 'exclusions', 'ancillary_metadata', 'matrix_count', 'norm_count', 'excluded_count',
         'tiles', 'independently_reconstructed_values', 'complete_positive_predicates',
         'source_tensor_payloads_consumed', 'model_constructed', 'codec_payloads_qualified',
         'model_numerics_quality_fit_performance_adoption_qualified'}, 'Exact producer v1 manifest fields required')
    same(manifest['contract_sha256'], contract_sha256, 'Producer contract pin differs')
    for key, value in dict(matrix_count=183, norm_count=53, excluded_count=222, tiles=4252,
                           independently_reconstructed_values=3428843520,
                           source_tensor_payloads_consumed=True, model_constructed=False,
                           codec_payloads_qualified=False,
                           model_numerics_quality_fit_performance_adoption_qualified=False).items():
        same(manifest[key], value, 'Producer completion or scope differs')
    same(manifest['complete_positive_predicates'], dict.fromkeys(PREDICATES, True),
         'Missing or unqualified producer predicate')
    source = contract['source']
    for key in ('repository', 'revision'):
        same(manifest['source_' + key], source[key], 'Source provenance differs')
    source_dir = Path(source['directory'])
    need(source_dir.is_dir() and not source_dir.is_symlink(), 'Regular selected source directory required')
    source_directory_identity = identity(source_dir.lstat())
    auth_pin = manifest['source_authentication']
    same(auth_pin['sha256'], contract['source_authentication_receipt_sha256'], 'Source auth pin differs')
    authentication_raw = bound_capture(auth_pin['path'], METADATA_LIMIT, auth_pin['sha256'])
    same(len(authentication_raw), auth_pin['bytes'], 'Source authentication extent differs')
    authentication = loads(authentication_raw)
    selected = source_closure(authentication, contract, bound_capture)
    lineage = manifest['source_authentication_lineage']
    need(type(lineage) is dict and set(lineage) == {'raw_receipt', 'download_guard', 'acquisition_wrapper'},
         'Complete producer source authentication lineage required')
    for name, row in lineage.items():
        same(row, {key: authentication[name][key] for key in ('path', 'bytes', 'sha256')},
             'Producer source authentication lineage differs')
    runtime_identities = {}
    runtime_count = runtime_receipts(manifest, runtime_identities, bound_capture)
    config_path, index_path = source_dir / 'config.json', source_dir / 'model.safetensors.index.json'
    config_raw = bound_capture(config_path, HASH_BLOCK, source['config_sha256'])
    index_raw = bound_capture(index_path, HASH_BLOCK, source['index_sha256'])
    source_metadata = {row['file']: row for row in selected['metadata']}
    same(len(config_raw), source_metadata['config.json']['bytes'], 'Authenticated source config extent differs')
    same(len(index_raw), source_metadata['model.safetensors.index.json']['bytes'], 'Authenticated source index extent differs')
    matrices = reports(manifest['matrices'], planned['matrices'])
    norms = reports(manifest['exact_norms'], set(planned['logical']) - set(matrices))
    exclusions = reports(manifest['exclusions'], set(inventory) - set(planned['logical']))
    pins = source['containers']
    need(type(pins) is list and len(pins) == 2, 'Two pinned source objects required')
    object_reports = manifest['source_objects']
    need(type(object_reports) is list and len(object_reports) == 2, 'Two complete object receipts required')
    child = manifest['child']
    same(child['bytes'], planned['file_bytes'], 'Child byte count differs')
    same(child['header_sha256'], EXPECTED['header_sha256'], 'Child prefix pin differs')
    child_path = Path(child['path'])
    need(child_path.name == 'model.safetensors', 'Published child filename required')
    source_ids, objects, tile_count, values = set(), [], 0, 0
    with ExitStack() as stack:
        destination = stack.enter_context(SavedFile(child_path, planned['file_bytes']))
        endpoints(child, destination)
        need(destination.read(planned['origin']) == planned['prefix'], 'Actual child prefix differs')
        child_id = tuple(destination.before[:2])
        journal_pin = manifest['journal']
        saved_journal = stack.enter_context(SavedFile(journal_pin['path'], journal_pin['bytes']))
        journal = Journal(saved_journal, journal_pin)
        for pin, report in zip(pins, object_reports):
            name = leaf(pin['container'])
            saved = stack.enter_context(SavedFile(source_dir / name, pin['expected_bytes']))
            need(tuple(saved.before[:2]) not in source_ids | {child_id}, 'Source/child file identity collision')
            source_ids.add(tuple(saved.before[:2]))
            endpoints(report, saved)
            prior = [item for item in selected['containers'] if item['file'] == name]
            need(len(prior) == 1, 'Prior authenticated source owner differs')
            same(pin_identity(prior[0]['final_identity']), saved.before, 'Prior source endpoint identity differs')
            same(report['path'], str(source_dir / name), 'Source object path differs')
            prefix = saved.read(8)
            count = struct.unpack('<Q', prefix)[0]
            uint(count, 2 * HASH_BLOCK)
            raw = saved.read(count, 8)
            ranges = pin['files']
            need(type(ranges) is list and len(ranges) == 2, 'Exact two source header ranges required')
            for row, expected_range, expected_bytes in zip(ranges, ([0, 7], [8, count + 7]), (8, count)):
                same(row['range'], expected_range, 'Source header range differs')
                same(row['bytes'], expected_bytes, 'Source header range extent differs')
                same(row['content_range'], f'bytes {expected_range[0]}-{expected_range[1]}/{pin["expected_bytes"]}',
                     'Source header content range differs')
            need(hashlib.sha256(prefix).hexdigest() == sha(pin['files'][0]['sha256'])
                 and hashlib.sha256(raw).hexdigest() == sha(pin['files'][1]['sha256'])
                 and count + 8 == uint(pin['payload_origin']), 'Source header pin differs')
            origin = 8 + count
            spans = source_spans(prefix, raw, pin['expected_bytes'], inventory)
            objects.append((name, spans))
            for span in spans:
                tensor_sha = saved.digest(span['end'] - span['start'], origin + span['start'])
                tensor_name = span['name']
                if tensor_name in matrices:
                    item, plan = matrices[tensor_name], planned['matrices'][tensor_name]
                    source_descriptor(item, name, span, origin, tensor_sha)
                    same(item['shape'], span['shape'], 'Matrix shape differs')
                    same(item['bits'], plan['bits'], 'Matrix bit width differs')
                    same(item['source_tensor_sha256'], tensor_sha, 'Matrix source digest alias differs')
                    need(sha(item['independently_reconstructed_F32_sha256']) == sha(item['upstream_F32_sha256'])
                         and item['reconstruction_equal'] is True, 'Unqualified complete matrix witness')
                    same(item['reconstructed_values'], math.prod(span['shape']), 'Complete matrix witness count differs')
                    same(item['tiles'], (span['shape'][0] + ROWS - 1) // ROWS, 'Complete matrix tile count differs')
                    full_fields = {}
                    for key, field in plan['fields'].items():
                        declaration = dict(planned['header'][field['name']])
                        declaration['data_offsets'] = [v + planned['origin'] for v in declaration['data_offsets']]
                        full_fields[key] = dict(name=field['name'], **component(destination, declaration))
                    same(item['fields'], full_fields, 'Stored full component differs')
                    for first in range(0, span['shape'][0], ROWS):
                        rows, columns = min(ROWS, span['shape'][0] - first), span['shape'][1]
                        tile = journal.take('matrix_tile')
                        check_tile(tile, saved, destination, span, plan, planned, name, origin, first, rows)
                        tile_count += 1
                        values += rows * columns
                    journal.matches('matrix_completed', item)
                elif tensor_name in norms:
                    item = norms[tensor_name]
                    source_descriptor(item, name, span, origin, tensor_sha)
                    same(item['copied_BF16_sha256'], tensor_sha, 'Norm source digest alias differs')
                    need(item['norm_bytes_equal'] is True and item['finite_text_input'] is True,
                         'Unqualified norm predicates')
                    same(item['dtype'], 'BF16', 'Norm dtype differs')
                    same(item['shape'], span['shape'], 'Norm shape differs')
                    uint(item['source_read_ns'])
                    uint(item['component_write_readback_ns'])
                    declaration = planned['header'][tensor_name]
                    start, end = declaration['data_offsets']
                    actual = destination.read(end - start, planned['origin'] + start)
                    need(actual == saved.read(span['end'] - span['start'], origin + span['start']),
                         'Actual copied norm bytes differ')
                    same(item['bytes'], len(actual), 'Norm extent differs')
                    same(item['child'], dict(name=tensor_name, dtype='BF16', shape=span['shape'],
                         offset=planned['origin'] + start, bytes=end - start, sha256=tensor_sha),
                         'Norm child descriptor differs')
                    journal.matches('norm_copied', item)
                else:
                    item = exclusions[tensor_name]
                    source_descriptor(item, name, span, origin, tensor_sha)
                    same(item['shape'], span['shape'], 'Excluded tensor shape differs')
                    same(item['dtype'], 'BF16', 'Excluded dtype differs')
                    same(item['excluded'], True, 'Unqualified exclusion')
                    same(item['reason'], 'vision_or_projector_outside_text_child', 'Exclusion reason differs')
                    uint(item['source_read_ns'])
                    journal.matches('excluded_tensor', item)
            need(saved.digest() == sha(pin['expected_sha256']), 'Complete source object SHA256 differs')
            for key, value in dict(file=name, bytes=pin['expected_bytes'], sha256=pin['expected_sha256'],
                                   tensors=len(spans), authenticated=True).items():
                same(report[key], value, 'Complete parent receipt differs')
            same(report['header_bytes'], count, 'Source header extent differs')
            same(report['header_sha256'], hashlib.sha256(raw).hexdigest(), 'Source header digest differs')
            same(report['prefix_bytes'], origin, 'Source prefix extent differs')
            same(report['prefix_sha256'], hashlib.sha256(prefix + raw).hexdigest(), 'Source prefix digest differs')
            journal.matches('parent_authenticated', report)
        validate_index(index_raw, objects, sum(row['end'] - row['start'] for _, spans in objects for row in spans))
        need({row['name'] for _, spans in objects for row in spans} == set(inventory), 'Complete source ownership differs')
        journal.finish()
        need(tile_count == 4252 and values == 3428843520, 'Complete witness coverage differs')
        need(destination.digest() == sha(child['sha256']), 'Complete saved child SHA256 differs')
        child_config_raw = metadata_descriptor(child['config'], child_path.parent / 'config.json',
                                               HASH_BLOCK, child['config_sha256'], capture_file=bound_capture)
        same(loads(child_config_raw), child_config(loads(config_raw), contract['recipe']['matrices']),
             'Child configuration changed semantics')
        same(bound_capture(child_path.parent / 'toklet-conversion.json', METADATA_LIMIT), manifest_raw,
             'Child and evidence manifests differ')
        metadata = {row['file']: row for row in selected['metadata']}
        copied = manifest['ancillary_metadata']
        need(type(copied) is list, 'Ancillary metadata receipts required')
        expected_copies = {name: metadata[name] for name in (
            'generation_config.json', 'tokenizer_config.json', 'chat_template.jinja') if name in metadata}
        need(len(copied) == len(expected_copies), 'Ancillary coverage differs')
        for item in copied:
            name = leaf(item['file'])
            need(name in expected_copies, 'Unknown or duplicate ancillary copy')
            pin = expected_copies.pop(name)
            copied_raw = metadata_descriptor(item, child_path.parent / name, 2 * HASH_BLOCK,
                                             pin['sha256'], extra=dict(file=name, source_path=str(source_dir / name)),
                                             capture_file=bound_capture)
            parent = bound_capture(source_dir / name, 2 * HASH_BLOCK, pin['sha256'])
            same(len(parent), pin['bytes'], 'Ancillary source extent differs')
            need(copied_raw == parent, 'Ancillary copy bytes differ')
        same(bound_capture(config_path, HASH_BLOCK, source['config_sha256']), config_raw, 'Source config changed')
        same(bound_capture(index_path, HASH_BLOCK, source['index_sha256']), index_raw, 'Source index changed')
    same(runtime_receipts(manifest, runtime_identities, bound_capture), runtime_count,
         'Runtime endpoint inventory changed')
    same(identity(source_dir.lstat()), source_directory_identity, 'Source directory endpoint changed')
    for path, before in metadata_bindings.items():
        same(identity(path.lstat()), before, 'Captured metadata changed before audit completion')
    return dict(format='toklet-ministral3-saved-byte-audit-v1', passed=True,
                contract_sha256=contract_sha256, conversion_manifest_sha256=manifest_sha256,
                matrix_count=183, norm_count=53, excluded_count=222, source_objects=2,
                child_tensors=602, tiles=tile_count, captured_witness_values=values,
                runtime_files_rehashed=runtime_count, hash_read_limit_bytes=HASH_BLOCK,
                tile_read_limit_bytes=TILE_BLOCK, audit_limit_bytes=512 * HASH_BLOCK,
                helper_reserve_bytes=32 * HASH_BLOCK, fp32_recomputed=False,
                saved_bytes_and_captured_witnesses_reconciled=True,
                runtime_execution_numerics_quality_fit_performance_adoption_qualified=False,
                whole_goal_complete=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'contract-sha256', 'conversion-manifest',
                 'conversion-manifest-sha256', 'fresh-output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    output = Path(args.fresh_output)
    need(output.is_absolute() and not output.exists() and not output.is_symlink(), 'Fresh absolute audit directory required')
    output.mkdir()
    result = audit(Path(args.contract), sha(args.contract_sha256),
                   Path(args.conversion_manifest), sha(args.conversion_manifest_sha256))
    with (output / 'audit.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
