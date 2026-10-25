"""Offline Smol raw-range witness; no inference or network."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import struct
import sys

REPOSITORY = 'mlx-community/SmolLM2-135M-Instruct-8bit'
REVISION = '0f0d9b8218915bc34d401e1a340b8c049d300d5e'
FILES = {
    'config.json': (1097, '5da91905efdfde905f566d76d4031e6ca7c4cb2a01dc4e3812a70b855188fc73'),
    'model.safetensors': (143030401, 'ec569c771e56424636be99625aba9031abc525a68a5f4ab16d678eca96fef12c'),
    'model.safetensors.index.json': (48979, 'd016e7514e8bfed162eb9d0716c73244c1b73792db3143e92e54138c1772db44'),
    'tokenizer.json': (3522656, '7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7'),
    'tokenizer_config.json': (3764, 'a27f638bd2831f5c3dea654a75838930f2b11fbe550c4d4e1d5d7bd07157b2ee'),
    'special_tokens_map.json': (655, '2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52'),
}
CHUNK = 65536
HEADER_MAX = 1024 * 1024
WITNESS_MAX = 2 * 1024 * 1024


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def document(data):
    require(len(data) <= HEADER_MAX, 'Oversized metadata')
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def integer(value, maximum=(1 << 63) - 1):
    require(type(value) is int and 0 <= value <= maximum, 'Invalid bounded unsigned integer')
    return value


def exact(stream, size):
    integer(size, HEADER_MAX)
    data = stream.read(size)
    require(len(data) == size, 'Short input read')
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stamp(stream):
    s = os.fstat(stream.fileno())
    require(stat.S_ISREG(s.st_mode), 'Input must be a regular file')
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def authenticate(stream, size, digest):
    """No metadata is parsed by this first, bounded full-file pass."""
    stream.seek(0)
    actual = hashlib.sha256()
    count = 0
    while True:
        data = stream.read(CHUNK)
        if not data:
            break
        count += len(data)
        require(count <= size, 'Input grew beyond pinned size')
        actual.update(data)
    require(count == size and actual.hexdigest() == digest, 'Full artifact identity mismatch')
    return dict(bytes=count, sha256=actual.hexdigest())


def read_metadata(stream, expected):
    size, digest = expected
    require(size <= HEADER_MAX, 'Metadata bound')
    stream.seek(0)
    data = exact(stream, size)
    require(not stream.read(1) and sha(data) == digest, 'Metadata changed after first authentication')
    return data, document(data)


def expected_smol():
    logical = {'model.embed_tokens.weight': [49152, 576], 'model.norm.weight': [576]}
    for layer in range(30):
        p = f'model.layers.{layer}'
        for name in ('input_layernorm', 'post_attention_layernorm'):
            logical[f'{p}.{name}.weight'] = [576]
        for name, rows, columns in [('q', 576, 576), ('k', 192, 576), ('v', 192, 576),
                                    ('o', 576, 576), ('gate', 1536, 576),
                                    ('up', 1536, 576), ('down', 576, 1536)]:
            block = 'self_attn' if name in ('q', 'k', 'v', 'o') else 'mlp'
            logical[f'{p}.{block}.{name}_proj.weight'] = [rows, columns]
    physical = {}
    for name, shape in logical.items():
        if len(shape) == 1:
            physical[name] = ('F16', shape)
        else:
            rows, columns = shape
            module = name.removesuffix('.weight')
            physical[name] = ('U32', [rows, columns // 4])
            physical[module + '.scales'] = ('F16', [rows, columns // 64])
            physical[module + '.biases'] = ('F16', [rows, columns // 64])
    require(len(physical) == 694, 'Authoring inventory error')
    return physical


def extract_shard(stream, expected_file, expected_tensors):
    """Hash precisely the consumed extraction bytes in canonical file order."""
    size, digest = expected_file
    stream.seek(0)
    combined = hashlib.sha256()
    prefix = exact(stream, 8)
    combined.update(prefix)
    header_size = struct.unpack('<Q', prefix)[0]
    require(0 < header_size <= HEADER_MAX and header_size + 8 < size, 'Invalid header size')
    header_bytes = exact(stream, header_size)
    combined.update(header_bytes)
    header = document(header_bytes)
    require(type(header) is dict, 'Header object required')
    auxiliary = header.pop('__metadata__', {})
    require(type(auxiliary) is dict and len(auxiliary) <= 16
            and all(type(k) is str and type(v) is str for k, v in auxiliary.items()), 'Invalid shard metadata')
    require(set(header) == set(expected_tensors), 'Physical tensor inventory differs')
    origin = 8 + header_size
    spans = []
    for name, info in header.items():
        require(type(info) is dict and set(info) == {'dtype', 'shape', 'data_offsets'}, 'Tensor fields differ')
        dtype, shape = expected_tensors[name]
        require(info['dtype'] == dtype and info['shape'] == shape, 'Tensor shape/dtype differs')
        require(type(shape) is list and 1 <= len(shape) <= 2, 'Rank bound')
        for dimension in info['shape']:
            require(integer(dimension, 1 << 32) > 0, 'Dimension must be positive')
        offsets = info['data_offsets']
        require(type(offsets) is list and len(offsets) == 2, 'Range pair required')
        start, end = [integer(n, size) for n in offsets]
        require(dtype in ('U32', 'F16'), 'Selected dtype differs')
        count = math.prod(shape) * (4 if dtype == 'U32' else 2)
        require(end > start and end - start == count and end <= size - origin, 'Range size differs')
        spans.append((start, end, name, dtype, shape))
    cursor = 0
    tensors = []
    for start, end, name, dtype, shape in sorted(spans):
        require(start == cursor, 'Physical ranges overlap or leave a gap')
        tensor_digest = hashlib.sha256()
        remaining = end - start
        while remaining:
            data = exact(stream, min(remaining, CHUNK))
            combined.update(data)
            tensor_digest.update(data)
            remaining -= len(data)
        tensors.append(dict(name=name, dtype=dtype, shape=shape, payload_offsets=[start, end],
                            file_offsets=[origin + start, origin + end], bytes=end - start,
                            sha256=tensor_digest.hexdigest()))
        cursor = end
    require(origin + cursor == size and not stream.read(1), 'Uncovered container tail')
    require(combined.hexdigest() == digest, 'Extraction-consumed container hash differs')
    return dict(file='model.safetensors', bytes=size, sha256=digest,
                length_prefix_hex=prefix.hex(), header_bytes=header_size,
                header_sha256=sha(header_bytes), data_start=origin, payload_bytes=cursor,
                tensors=tensors), prefix, header_bytes


def reread(stream, record):
    """Always hash this use, even if the range was validated previously."""
    stream.seek(record['file_offsets'][0])
    digest = hashlib.sha256()
    remaining = record['bytes']
    while remaining:
        data = exact(stream, min(remaining, CHUNK))
        digest.update(data)
        remaining -= len(data)
    require(digest.hexdigest() == record['sha256'], 'Repeated consumed range differs')
    return dict(name=record['name'], bytes=record['bytes'], sha256=digest.hexdigest())


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def publish(root, name, data):
    require(len(data) <= WITNESS_MAX, 'Output bound')
    with (root / name).open('xb') as output:
        output.write(data)


def run(source, output):
    # Output is exclusive and constrained to the repository's ignored artifacts.
    base = Path(__file__).resolve().parents[2] / 'artifacts'
    output = Path(output).absolute()
    require(output.parent.resolve().is_relative_to(base.resolve()), 'Output must be below repository artifacts')
    output.mkdir(mode=0o700)
    with ExitStack() as stack:
        directory = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
        stack.callback(os.close, directory)
        streams = {}
        before = {}
        first = {}
        for name, (size, digest) in FILES.items():
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            stream = stack.enter_context(os.fdopen(fd, 'rb'))
            before[name] = stamp(stream)
            require(before[name]['bytes'] == size, 'Pinned regular-file size differs')
            first[name] = authenticate(stream, size, digest)
            require(stamp(stream) == before[name], 'Input metadata changed during first authentication')
            streams[name] = stream
        # Every file above is authenticated before the first JSON/header parse.
        config_bytes, config = read_metadata(streams['config.json'], FILES['config.json'])
        index_bytes, index = read_metadata(streams['model.safetensors.index.json'], FILES['model.safetensors.index.json'])
        require(config['tie_word_embeddings'] is True and config['quantization'] == {'group_size': 64, 'bits': 8}, 'Pinned Smol policy differs')
        expected = expected_smol()
        require(type(index) is dict and set(index) == {'metadata', 'weight_map'}, 'Index fields differ')
        require(index['weight_map'] == {name: 'model.safetensors' for name in expected}, 'Index physical mapping differs')
        shard, prefix, header = extract_shard(streams['model.safetensors'], FILES['model.safetensors'], expected)
        require(index['metadata']['total_size'] == shard['payload_bytes'], 'Index payload total differs')
        records = {x['name']: x for x in shard['tensors']}
        tied = []
        for role in ('input_embedding', 'tied_output'):
            for suffix in ('weight', 'scales', 'biases'):
                entry = reread(streams['model.safetensors'], records['model.embed_tokens.' + suffix])
                tied.append(dict(role=role, **entry))
        after = {}
        for name, (size, digest) in FILES.items():
            after[name] = authenticate(streams[name], size, digest)
            require(stamp(streams[name]) == before[name], 'Input metadata changed during extraction')
        witness = dict(format='toklet-smol-range-witness-v1', repository=REPOSITORY, revision=REVISION,
                       file_inventory=[dict(name=n, bytes=s, sha256=h) for n, (s, h) in sorted(FILES.items())],
                       shard=shard, packed_matrices=211, normalization_vectors=61,
                       tie_policy=dict(tied=True, explicit_lm_head=False, rereads=tied),
                       qualification='Content witness only; no new engine, numerical, memory, output or device grant')
        data = (json.dumps(witness, sort_keys=True, indent=2) + '\n').encode()
        prepublication_peak = peak_rss()
        publish(output, 'prefix.bin', prefix)
        publish(output, 'header.bin', header)
        publish(output, 'config.json', config_bytes)
        publish(output, 'index.json', index_bytes)
        publish(output, 'witness.json', data)
        receipt = dict(format='toklet-smol-range-producer-v1', first_authentication=first,
                       final_authentication=after, file_stamps=before,
                       authentication_before_metadata=True, extraction_consumed_hash_matched=True,
                       peak_rss_prepublication_bytes=prepublication_peak,
                       peak_scope='Internal observation before output/receipt publication; not completed child lifetime',
                       outputs={n: dict(bytes=(output/n).stat().st_size, sha256=sha((output/n).read_bytes()))
                                for n in ('prefix.bin', 'header.bin', 'config.json', 'index.json', 'witness.json')})
        publish(output, 'producer-receipt.json', (json.dumps(receipt, sort_keys=True, indent=2)+'\n').encode())
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    result = run(arguments.source, arguments.output)
    print(json.dumps(dict(status='pass', physical_tensors=694,
                         peak_rss_prepublication_bytes=result['peak_rss_prepublication_bytes'])))
