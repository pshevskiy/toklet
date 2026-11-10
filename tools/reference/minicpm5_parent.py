"""Authenticate the pinned MiniCPM5 BF16 parent without retaining its payload.

No numerical model is loaded. Tensor/block digests become authenticated only
after BOTH complete object hashes, transfer receipts and unchanged child identity
pass. Quantized conversion lineage and model quality remain separate gates.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

BLOCK = 4 * 1024 ** 2
METADATA_SHA = 'd59473f7e748d5450f21e4e76db175852ac94305f23668e645178a45c3c8a633'
CHILD_SHA = '6fadca0fcdefe8b77fc144851ee2c2fd3c8949ccc0ec756b8ada27e3156c8910'
CHILD_SIZE = 2674327346
PARENT = 'insraq/MiniCPM5-2B-heretic-abliterated'
REVISION = '8d83eaf42e30f4e4f164b8685be0ecdccd8ef0d8'
OBJECTS = [
    ('model-00001-of-00002.safetensors', 4989507944,
     '2a3ac0fb68423d0e359a0a8a937c8a295ddaea08c364efdeef7631b739d6425b'),
    ('model-00002-of-00002.safetensors', 44049184,
     '9f35f94773989092e6d19a527c1e95a7be0252eae0fdcf50310eff4f6e26e11e'),
]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate metadata field')
        result[key] = value
    return result


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metadata(path, maximum=2 * 1024 ** 2):
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum + 1)
    need(len(raw) <= maximum, 'Oversize metadata')
    return json.loads(raw, object_pairs_hook=unique)


def exact(stream, count):
    need(type(count) is int and 0 <= count <= BLOCK, 'Unbounded stream read')
    parts = bytearray()
    while len(parts) < count:
        raw = stream.read(count - len(parts))
        need(type(raw) is bytes and 0 < len(raw) <= count - len(parts),
             'Truncated or invalid parent stream')
        parts.extend(raw)
    return bytes(parts)


def plan(prefix, header, size):
    need(type(prefix) is bytes and len(prefix) == 8
         and type(header) is bytes and 0 < len(header) <= 2 * 1024 ** 2
         and int.from_bytes(prefix, 'little') == len(header)
         and type(size) is int and len(header) + 8 < size <= 6 * 1024 ** 3,
         'Invalid parent header extent')
    parsed = json.loads(header, object_pairs_hook=unique)
    need(type(parsed) is dict, 'Parent header must be object')
    spans = []
    for name, item in parsed.items():
        if name == '__metadata__':
            continue
        need(type(name) is str and name and type(item) is dict
             and set(item) == {'dtype', 'shape', 'data_offsets'}
             and item['dtype'] == 'BF16', 'Parent tensor declaration differs')
        shape, offsets = item['shape'], item['data_offsets']
        need(type(shape) is list and 0 < len(shape) <= 2
             and all(type(n) is int and 0 < n <= 131072 for n in shape)
             and type(offsets) is list and len(offsets) == 2
             and all(type(n) is int and 0 <= n < 6 * 1024 ** 3 for n in offsets)
             and offsets[1] - offsets[0] == 2 * math.prod(shape),
             'Parent tensor shape/span differs')
        spans.append(dict(name=name, shape=shape, start=offsets[0], end=offsets[1]))
    spans.sort(key=lambda row: row['start'])
    need(spans and spans[0]['start'] == 0
         and all(a['end'] == b['start'] for a, b in zip(spans, spans[1:]))
         and 8 + len(header) + spans[-1]['end'] == size,
         'Parent payload has holes, overlaps or wrong final extent')
    return spans


def authenticate(stream, prefix, header, size, expected_sha, norms, journal):
    """Read exact bounded blocks; no payload survives this call except digests."""
    spans = plan(prefix, header, size)
    need(type(norms) is dict and set(norms) <= {row['name'] for row in spans}
         and all(type(v) is bytes and len(v) <= 65536 for v in norms.values()),
         'Unbounded or unknown norm controls')
    need(type(expected_sha) is str and re.fullmatch('[0-9a-f]{64}', expected_sha),
         'Expected object digest invalid')
    whole = hashlib.sha256()
    for expected in [prefix, header]:
        actual = exact(stream, len(expected))
        need(actual == expected, 'Streamed parent header differs from pinned header')
        whole.update(actual)
    origin = 8 + len(header)
    tensors = []
    norm_count = 0
    for row in spans:
        tensor_hash = hashlib.sha256()
        blocks = []
        cursor = row['start']
        if row['name'] in norms:
            need(len(norms[row['name']]) == row['end'] - cursor,
                 'Norm control extent differs')
        while cursor < row['end']:
            count = min(BLOCK, row['end'] - cursor)
            raw = exact(stream, count)
            whole.update(raw)
            tensor_hash.update(raw)
            blocks.append(dict(offset=origin + cursor, bytes=count,
                               sha256=hashlib.sha256(raw).hexdigest()))
            if row['name'] in norms:
                need(cursor == row['start'] and raw == norms[row['name']],
                     'Unquantized parent/child norm bytes differ')
                norm_count += 1
            cursor += count
        item = dict(name=row['name'], dtype='BF16', shape=row['shape'],
                    offset=origin + row['start'], bytes=row['end'] - row['start'],
                    sha256=tensor_hash.hexdigest(), blocks=blocks,
                    exact_child_norm=row['name'] in norms)
        tensors.append(item)
        journal.write(json.dumps(dict(event='tensor_read', object_authenticated=False,
                                      **item), separators=(',', ':')) + '\n')
        journal.flush()
    need(stream.read(1) == b'', 'Trailing parent payload bytes')
    need(whole.hexdigest() == expected_sha, 'Complete parent object SHA-256 differs')
    return dict(bytes=size, sha256=whole.hexdigest(), tensors=tensors,
                exact_child_norms=norm_count)


def controls(folder, child):
    """Bind cached TLS metadata and the already authenticated MLX8 child."""
    path = folder / 'metadata-verified.json'
    need(digest(path) == METADATA_SHA, 'Parent metadata receipt changed')
    receipt = metadata(path)
    need(receipt['passed'] is True and receipt['parent_revision'] == REVISION,
         'Parent metadata identity differs')
    for name, pin in receipt['metadata_sha256'].items():
        need(digest(folder / name) == pin, 'Pinned metadata input changed')
    observed = [(row['rfilename'], row['size'], row['lfs']['sha256'])
                for row in receipt['parent_files']]
    need(observed == OBJECTS, 'Parent object pins differ')
    index = metadata(folder / 'metadata/parent-index.json')['weight_map']
    need(type(index) is dict and len(index) == 381, 'Parent index inventory differs')
    need(child.stat().st_size == CHILD_SIZE and digest(child) == CHILD_SHA,
         'Complete child object identity differs')
    with child.open('rb') as stream:
        child_prefix = exact(stream, 8)
        length = int.from_bytes(child_prefix, 'little')
        need(0 < length <= 2 * 1024 ** 2, 'Child header too large')
        child_header = json.loads(exact(stream, length), object_pairs_hook=unique)
        child_origin = 8 + length
        logical = {}
        norms = {}
        for name, item in child_header.items():
            if name == '__metadata__' or name.endswith(('.scales', '.biases')):
                continue
            need(item['dtype'] in ['BF16', 'U32'], 'Unexpected child tensor dtype')
            shape = item['shape']
            logical[name] = shape if item['dtype'] == 'BF16' else [shape[0], shape[1] * 4]
            if item['dtype'] == 'BF16':
                start, end = item['data_offsets']
                need(shape == [2048] and end - start == 4096, 'Child norm shape differs')
                stream.seek(child_origin + start)
                norms[name] = exact(stream, end - start)
    need(len(logical) == 381 and len(norms) == 85
         and sum(math.prod(shape) for shape in logical.values()) == 2516756480,
         'Child logical/norm inventory differs')
    shards = []
    seen = set()
    for number, (name, size, pin) in enumerate(OBJECTS, 1):
        prefix = (folder / f'metadata/parent-{number}-prefix.bin').read_bytes()
        header = (folder / f'metadata/parent-{number}-header.bin').read_bytes()
        spans = plan(prefix, header, size)
        for row in spans:
            tensor = row['name']
            need(tensor not in seen and index.get(tensor) == name
                 and logical.get(tensor) == row['shape'], 'Parent/child/index inventory differs')
            seen.add(tensor)
        selected_norms = {row['name']: norms[row['name']] for row in spans
                          if row['name'] in norms}
        shards.append((name, size, pin, prefix, header, selected_norms))
    need(seen == set(logical) == set(index), 'Incomplete parent tensor coverage')
    return shards


def transfer(shard, output):
    name, size, pin, prefix, header, norms = shard
    nonce = uuid.uuid4().hex
    url = f'https://huggingface.co/{PARENT}/resolve/{REVISION}/{name}?download=true&toklet_audit={nonce}'
    command = ['/usr/bin/curl', '-q', '--fail', '--location', '--proto', '=https',
               '--proto-redir', '=https', '--connect-timeout', '20', '--max-time', '1200',
               '--max-filesize', str(size), '--no-progress-meter',
               '-H', 'Accept-Encoding: identity', '--dump-header', str(output / (name + '.headers')),
               '--output', '-', '--write-out', '%{stderr}\nTOKLET_CURL_JSON%{json}\n', url]
    started = time.monotonic_ns()
    result = None
    error = None
    resource = None
    with (output / (name + '.curl.log')).open('x') as log, \
         (output / (name + '.tensors.jsonl')).open('x') as journal:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        try:
            result = authenticate(child.stdout, prefix, header, size, pin, norms, journal)
        except BaseException as exception:
            error = type(exception).__name__ + ': ' + str(exception)
        finally:
            child.stdout.close()
            if error is not None and child.poll() is None:
                child.terminate()
            # wait4 is the authoritative final rusage source, without Popen.poll
            # reaping the child before this call on the successful path.
            deadline = time.monotonic() + 15
            while child.returncode is None:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    resource = usage
                    break
                if time.monotonic() > deadline:
                    child.kill()
                    _, status, resource = os.wait4(child.pid, 0)
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                time.sleep(.05)
    completed = time.monotonic_ns()
    raw_log = (output / (name + '.curl.log')).read_text()
    matches = re.findall(r'^TOKLET_CURL_JSON(.+)$', raw_log, re.M)
    transport = {}
    try:
        need(len(matches) == 1, 'Missing/duplicate curl transfer receipt')
        transport = json.loads(matches[0], object_pairs_hook=unique)
        need(type(transport) is dict, 'Invalid curl transfer receipt')
    except ValueError as exception:
        error = (error + '; ' if error else '') + str(exception)
    passed = (error is None and result is not None and child.returncode == 0
              and type(transport.get('http_code')) is int and transport['http_code'] == 200
              and type(transport.get('size_download')) is int and transport['size_download'] == size
              and type(transport.get('ssl_verify_result')) is int
              and transport['ssl_verify_result'] == 0 and resource is not None)
    record = dict(passed=passed, name=name, pid=child.pid, command=command,
                  started_ns=started, completed_ns=completed, exit_code=child.returncode,
                  error=error, http_code=transport.get('http_code'),
                  size_download=transport.get('size_download'),
                  ssl_verify_result=transport.get('ssl_verify_result'),
                  final_wait4_peak_rss_bytes=resource.ru_maxrss if resource else None,
                  payload_retained=False, result=result)
    (output / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    need(passed, 'Parent transfer/authentication failed; primary logs retained: ' + name)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata_directory', type=Path)
    parser.add_argument('child_safetensors', type=Path)
    parser.add_argument('fresh_output', type=Path)
    args = parser.parse_args()
    args.fresh_output.mkdir()
    shards = controls(args.metadata_directory, args.child_safetensors)
    records = []
    for shard in shards:
        row = transfer(shard, args.fresh_output)
        records.append(row)
        print(json.dumps(dict(parent_object_authenticated=row['name'], bytes=row['result']['bytes'],
                              tensors=len(row['result']['tensors']))), flush=True)
    need(digest(args.child_safetensors) == CHILD_SHA, 'Child changed during parent audit')
    need(sum(row['result']['exact_child_norms'] for row in records) == 85,
         'Unquantized norm comparison incomplete')
    result = dict(format='toklet-minicpm5-parent-auth-v1', passed=True,
                  parent_repository=PARENT, parent_revision=REVISION,
                  authenticated_bytes=sum(row['result']['bytes'] for row in records),
                  parent_objects=records, parent_tensor_count=381, exact_child_norm_tensors=85,
                  stream_block_bytes=BLOCK, retained_payload_bytes=0,
                  child_sha256=CHILD_SHA, parent_full_object_identity_verified=True,
                  quantized_conversion_lineage_qualified=False, model_execution_qualified=False,
                  task_quality_adoption_qualified=False, physical_iphone_qualified=False)
    (args.fresh_output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'parent_objects'}))


if __name__ == '__main__':
    main()
