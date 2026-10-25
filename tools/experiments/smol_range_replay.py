"""Independent bounded reconstruction/coverage audit; does not import producer."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import sys

PINNED = {
    'config.json': [1097, '5da91905efdfde905f566d76d4031e6ca7c4cb2a01dc4e3812a70b855188fc73'],
    'model.safetensors': [143030401, 'ec569c771e56424636be99625aba9031abc525a68a5f4ab16d678eca96fef12c'],
    'model.safetensors.index.json': [48979, 'd016e7514e8bfed162eb9d0716c73244c1b73792db3143e92e54138c1772db44'],
    'tokenizer.json': [3522656, '7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7'],
    'tokenizer_config.json': [3764, 'a27f638bd2831f5c3dea654a75838930f2b11fbe550c4d4e1d5d7bd07157b2ee'],
    'special_tokens_map.json': [655, '2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52'],
}
BLOCK = 32768  # Different chunk boundary from the producer, including tensor tails.


def check(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    seen = {}
    for k, v in items:
        check(k not in seen, 'Duplicate object member')
        seen[k] = v
    return seen


def parse(data):
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite constant')))


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_exact(file, count):
    check(type(count) is int and 0 <= count <= 2 * 1024 * 1024, 'Read bound')
    result = file.read(count)
    check(len(result) == count, 'Truncated read')
    return result


def open_regular(stack, directory, name, size):
    check('/' not in name and name not in ('', '.', '..'), 'Plain filename required')
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    file = stack.enter_context(os.fdopen(fd, 'rb'))
    info = os.fstat(file.fileno())
    check(stat.S_ISREG(info.st_mode) and info.st_size == size, 'Regular-file size differs')
    return file


def full_hash(file, size):
    file.seek(0)
    digest = hashlib.sha256()
    seen = 0
    while chunk := file.read(BLOCK):
        seen += len(chunk)
        check(seen <= size, 'Whole input too long')
        digest.update(chunk)
    check(seen == size, 'Whole input too short')
    return digest.hexdigest()


def derived_inventory(config):
    check(config['hidden_size'] == 576 and config['num_hidden_layers'] == 30
          and config['intermediate_size'] == 1536 and config['vocab_size'] == 49152
          and config['num_attention_heads'] == 9 and config['num_key_value_heads'] == 3
          and config['tie_word_embeddings'] is True, 'Pinned graph geometry differs')
    check(config['quantization'] == {'bits': 8, 'group_size': 64}, 'Pinned packing differs')
    width = config['hidden_size']
    middle = config['intermediate_size']
    attention = {'q_proj': (width, width),
                 'k_proj': (width // config['num_attention_heads'] * config['num_key_value_heads'], width),
                 'v_proj': (width // config['num_attention_heads'] * config['num_key_value_heads'], width),
                 'o_proj': (width, width)}
    feedforward = {'gate_proj': (middle, width), 'up_proj': (middle, width), 'down_proj': (width, middle)}
    matrices = {'model.embed_tokens': (config['vocab_size'], width)}
    result = {'model.norm.weight': ('F16', [width])}
    for layer in range(config['num_hidden_layers']):
        stem = 'model.layers.' + str(layer) + '.'
        for norm in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
            result[stem + norm] = ('F16', [width])
        for group, shapes in (('self_attn.', attention), ('mlp.', feedforward)):
            matrices.update({stem + group + k: shape for k, shape in shapes.items()})
    for name, (rows, columns) in matrices.items():
        result[name + '.weight'] = ('U32', [rows, columns // 4])
        for suffix in ('.scales', '.biases'):
            result[name + suffix] = ('F16', [rows, columns // 64])
    check(len(matrices) == 211 and len(result) == 694, 'Independent inventory count')
    return result


def partition(header, origin, file_size, expected):
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    check(set(tensors) == set(expected), 'Independent physical inventory mismatch')
    records = []
    for name, item in tensors.items():
        check(type(item) is dict and set(item) == {'dtype', 'shape', 'data_offsets'}, 'Unexpected tensor field')
        dtype, shape = expected[name]
        check(item['dtype'] == dtype and item['shape'] == shape, 'Independent shape/dtype mismatch')
        check(all(type(n) is int and n > 0 for n in item['shape']), 'Invalid dimensions')
        extent = item['data_offsets']
        check(type(extent) is list and len(extent) == 2
              and all(type(n) is int and 0 <= n < (1 << 63) for n in extent), 'Invalid extent')
        first, last = extent
        length = 4 if dtype == 'U32' else 2
        for dimension in shape:
            length *= dimension
        check(last - first == length and last <= file_size - origin, 'Independent byte length mismatch')
        records.append((first, last, name, dtype, shape))
    position = 0
    for first, last, *_ in sorted(records):
        check(first == position, 'Independent coverage gap/overlap')
        position = last
    check(position + origin == file_size, 'Independent coverage tail')
    return sorted(records)


def range_hash(file, start, end):
    file.seek(start)
    digest = hashlib.sha256()
    while start < end:
        chunk = read_exact(file, min(BLOCK, end - start))
        digest.update(chunk)
        start += len(chunk)
    return digest.hexdigest()


def witness_ranges(records):
    check(type(records) is list and len(records) <= 1024, 'Range list bound')
    for record in records:
        check(type(record) is dict and set(record) == {'name', 'dtype', 'shape', 'payload_offsets',
                                                      'file_offsets', 'bytes', 'sha256'}, 'Range witness fields')
        check(type(record['bytes']) is int and record['bytes'] > 0, 'Range witness size')
        for field in ('shape', 'payload_offsets', 'file_offsets'):
            values = record[field]
            check(type(values) is list and (1 <= len(values) <= 2 if field == 'shape' else len(values) == 2), 'Range witness list')
            check(all(type(n) is int and 0 <= n < (1 << 63) for n in values), 'Range witness numeric type')
        digest = record['sha256']
        check(type(digest) is str and len(digest) == 64
              and all(c in '0123456789abcdef' for c in digest), 'Range witness digest')


def audit(source, frozen):
    with ExitStack() as stack:
        directory = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
        stack.callback(os.close, directory)
        files = {name: open_regular(stack, directory, name, size) for name, (size, _) in PINNED.items()}
        for name, (size, wanted) in PINNED.items():
            check(full_hash(files[name], size) == wanted, 'First replay full artifact mismatch')
        frozen = Path(frozen)
        # Witness artifacts are bounded regular files; source hashes above are checked first.
        def saved(name, maximum):
            fd = os.open(frozen/name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as f:
                info = os.fstat(f.fileno())
                check(stat.S_ISREG(info.st_mode) and 0 <= info.st_size <= maximum, 'Frozen artifact bound/type')
                result = read_exact(f, info.st_size)
                check(not f.read(1), 'Frozen artifact grew')
                return result
        witness = parse(saved('witness.json', 2 * 1024 * 1024))
        check(set(witness) == {'format', 'repository', 'revision', 'file_inventory', 'shard',
                              'packed_matrices', 'normalization_vectors', 'tie_policy', 'qualification'}, 'Witness fields')
        check(witness['format'] == 'toklet-smol-range-witness-v1'
              and witness['repository'] == 'mlx-community/SmolLM2-135M-Instruct-8bit'
              and witness['revision'] == '0f0d9b8218915bc34d401e1a340b8c049d300d5e', 'Wrong enrollment')
        check(witness['file_inventory'] == [dict(name=n, bytes=s, sha256=h) for n, (s, h) in sorted(PINNED.items())], 'Pinned file inventory differs')
        config = None
        index = None
        for name, copy in (('config.json', 'config.json'), ('model.safetensors.index.json', 'index.json')):
            f = files[name]
            f.seek(0)
            original = read_exact(f, PINNED[name][0])
            check(hash_bytes(original) == PINNED[name][1] and original == saved(copy, 1024 * 1024), 'Exact frozen metadata differs')
            if name == 'config.json':
                config = parse(original)
            else:
                index = parse(original)
        expected = derived_inventory(config)
        check(index['weight_map'] == dict.fromkeys(expected, 'model.safetensors'), 'Independent index mapping')
        f = files['model.safetensors']
        f.seek(0)
        prefix = read_exact(f, 8)
        count = int.from_bytes(prefix, 'little')
        check(0 < count <= 1024 * 1024, 'Independent header bound')
        header_bytes = read_exact(f, count)
        check(prefix == saved('prefix.bin', 8) and header_bytes == saved('header.bin', 1024 * 1024), 'Frozen raw prefix/header differs')
        header = parse(header_bytes)
        origin = count + 8
        plan = partition(header, origin, PINNED['model.safetensors'][0], expected)
        shard = witness['shard']
        check(set(shard) == {'file', 'bytes', 'sha256', 'length_prefix_hex', 'header_bytes',
                            'header_sha256', 'data_start', 'payload_bytes', 'tensors'}, 'Shard witness fields')
        check({k: v for k, v in shard.items() if k != 'tensors'} ==
              dict(file='model.safetensors', bytes=PINNED['model.safetensors'][0], sha256=PINNED['model.safetensors'][1],
                   length_prefix_hex=prefix.hex(), header_bytes=count, header_sha256=hash_bytes(header_bytes),
                   data_start=origin, payload_bytes=PINNED['model.safetensors'][0] - origin), 'Shard witness metadata differs')
        check(index['metadata']['total_size'] == shard['payload_bytes'], 'Independent index total')
        check(type(shard['tensors']) is list and len(shard['tensors']) == len(plan), 'Range witness count differs')
        witness_ranges(shard['tensors'])
        reconstructed = hashlib.sha256(prefix + header_bytes)
        rebuilt = []
        for first, last, name, dtype, shape in plan:
            digest = hashlib.sha256()
            offset = first
            while offset < last:
                block = read_exact(f, min(BLOCK, last - offset))
                digest.update(block)
                reconstructed.update(block)
                offset += len(block)
            rebuilt.append(dict(name=name, dtype=dtype, shape=shape, payload_offsets=[first, last],
                                file_offsets=[origin + first, origin + last], bytes=last - first,
                                sha256=digest.hexdigest()))
        check(not f.read(1) and reconstructed.hexdigest() == PINNED['model.safetensors'][1], 'Independent consumed-byte reconstruction differs')
        check(rebuilt == shard['tensors'], 'Independent tensor digests or ranges differ')
        for record in sorted(rebuilt, key=lambda r: r['name'], reverse=True):
            check(range_hash(f, *record['file_offsets']) == record['sha256'], 'Random-order replay differs')
        trace = []
        by_name = {r['name']: r for r in rebuilt}
        for role in ['input_embedding', 'tied_output']:
            for suffix in ['weight', 'scales', 'biases']:
                name = 'model.embed_tokens.' + suffix
                item = by_name[name]
                actual = range_hash(f, *item['file_offsets'])
                check(actual == item['sha256'], 'Tied reread differs')
                trace.append(dict(role=role, name=name, bytes=item['bytes'], sha256=actual))
        check(not any(n.startswith('lm_head.') for n in expected), 'Real Smol has no explicit redundant head')
        check(witness['tie_policy'] == dict(tied=True, explicit_lm_head=False, rereads=trace), 'Tied role/reread witness differs')
        check(witness['packed_matrices'] == 211 and witness['normalization_vectors'] == 61, 'Packing inventory count')
        for name, (size, wanted) in PINNED.items():
            check(full_hash(files[name], size) == wanted, 'Final replay artifact mismatch')
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin':
            peak *= 1024
        return dict(status='pass', first_and_last_full_file_hashes=6, reconstructed_container_sha256=reconstructed.hexdigest(),
                    physical_ranges=len(rebuilt), random_order_range_reads=len(rebuilt), tied_range_reads=len(trace),
                    packed_matrices=211, normalization_vectors=61, peak_rss_before_return_bytes=peak,
                    peak_scope='Internal observation before return/printing; not completed child lifetime',
                    scope='Content/range replay only; no execution, numerical parity, phone memory or output qualification')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--frozen', required=True)
    args = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    print(json.dumps(audit(args.source, args.frozen), sort_keys=True))
