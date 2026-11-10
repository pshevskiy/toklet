"""Bounded authentication of one pinned LFM3 BF16 port and its modified parent.

The candidate is retained once; the parent is streamed and discarded. Exact
tensor equality does not qualify model execution, quality, fit or adoption.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import uuid

BLOCK = 4 * 1024**2
CONTRACT_SHA = 'd9a5b636fcb3e7eb472f3cd1e58848bbaa4703a6acfca4dc4a398eb963751af2'
CHILD_SHA = 'b6ff17ac993dddf0f6f51bd872b12ae1437f4453a54a0b4878cedc7864fbc07e'
PARENT_SHA = '2e44203266e14abedc1d56a0d2379840abf5456729e30ad878117f015e503e28'
SIZE = 3457635448
PARENT = 'Vortecks/LFM3-1.7B-Heretic-Abliterated'
REVISION = 'dc37c60c02a35603456c85c2aecd7bc22b23b89c'
LAYERS = ['conv', 'conv', 'full_attention', 'conv', 'sliding_attention', 'conv',
    'full_attention', 'conv', 'sliding_attention', 'conv', 'full_attention', 'conv',
    'sliding_attention', 'conv', 'full_attention', 'conv', 'sliding_attention',
    'conv', 'full_attention', 'conv', 'sliding_attention', 'conv', 'full_attention', 'conv']


def need(value, message):
    if not value:
        raise ValueError(message)


def unique(items):
    result = {}
    for name, value in items:
        need(name not in result, 'Duplicate source metadata field')
        result[name] = value
    return result


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def document(path):
    with path.open('rb') as stream:
        raw = stream.read(2*1024**2+1)
    need(len(raw) <= 2*1024**2, 'Bounded source metadata required')
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def exact(stream, count):
    need(type(count) is int and 0 <= count <= BLOCK, 'Bounded stream read required')
    result = bytearray()
    while len(result) < count:
        block = stream.read(count-len(result))
        need(type(block) is bytes and 0 < len(block) <= count-len(result), 'Truncated source stream')
        result.extend(block)
    return bytes(result)


def expected_tensors(parent=False):
    need(type(parent) is bool, 'Explicit source layout required')
    result = {'model.embed_tokens.weight': [128000, 2048], 'model.embedding_norm.weight': [2048]}
    for layer, kind in enumerate(LAYERS):
        prefix = f'model.layers.{layer}.'
        for norm in ['operator_norm', 'post_operator_norm', 'ffn_norm', 'post_ffn_norm']:
            result[prefix+norm+'.weight'] = [2048]
        for name, shape in [('w1', [7680, 2048]), ('w2', [2048, 7680]), ('w3', [7680, 2048])]:
            result[prefix+'feed_forward.'+name+'.weight'] = shape
        if kind == 'conv':
            result[prefix+'conv.in_proj.weight'] = [6144, 2048]
            result[prefix+'conv.out_proj.weight'] = [2048, 2048]
            result[prefix+'conv.conv.weight'] = [2048, 1, 7] if parent else [2048, 7, 1]
        else:
            for name, rows in [('q_proj', 2048), ('k_proj', 512), ('v_proj', 512),
                               ('o_proj', 2048), ('gate_proj', 16)]:
                result[prefix+'self_attn.'+name+'.weight'] = [rows, 2048]
            for name in ['q_norm', 'k_norm']:
                result[prefix+'self_attn.'+name+'.weight'] = [128]
    need(len(result) == 286 and sum(math.prod(s) for s in result.values()) == 1728801536,
         'Fixed LFM3 tensor schema differs')
    return result


def validate_spans(rows, payload_bytes):
    """Structural stream primitive; callers separately admit the fixed schema."""
    need(type(rows) is list and rows and type(payload_bytes) is int
         and 0 < payload_bytes <= 6*1024**3, 'Bounded complete source spans required')
    cursor = 0
    names = set()
    for row in rows:
        need(type(row) is dict and set(row) == {'name', 'dtype', 'shape', 'start', 'end'}
             and type(row['name']) is str and row['name'] and row['name'] not in names
             and row['dtype'] == 'BF16' and type(row['shape']) is list and 1 <= len(row['shape']) <= 3
             and all(type(n) is int and 0 < n <= 131072 for n in row['shape'])
             and type(row['start']) is int and type(row['end']) is int
             and row['start'] == cursor and row['end'] <= payload_bytes
             and row['end']-row['start'] == 2*math.prod(row['shape']),
             'Source spans contain invalid types, duplicates, gaps or overlaps')
        cursor = row['end']
        names.add(row['name'])
    need(cursor == payload_bytes, 'Source payload final coverage differs')


def plan(prefix, header, size, parent=False):
    need(type(prefix) is bytes and len(prefix) == 8 and type(header) is bytes
         and 0 < len(header) <= 2*1024**2 and int.from_bytes(prefix, 'little') == len(header)
         and type(size) is int and 8+len(header) < size <= 6*1024**3, 'Source header extent differs')
    parsed = json.loads(header, object_pairs_hook=unique,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    need(type(parsed) is dict, 'Source header object required')
    if '__metadata__' in parsed:
        meta = parsed.pop('__metadata__')
        need(type(meta) is dict and all(type(k) is str and type(v) is str for k, v in meta.items()),
             'Source container metadata must be strings')
    expected = expected_tensors(parent)
    need(set(parsed) == set(expected), 'Complete fixed LFM3 tensor names required')
    rows = []
    for name, shape in expected.items():
        value = parsed[name]
        need(type(value) is dict and set(value) == {'dtype', 'shape', 'data_offsets'}
             and value['dtype'] == 'BF16' and same(value['shape'], shape)
             and type(value['data_offsets']) is list and len(value['data_offsets']) == 2,
             'Fixed source dtype/shape/declaration differs')
        rows.append(dict(name=name, dtype='BF16', shape=shape,
                         start=value['data_offsets'][0], end=value['data_offsets'][1]))
    rows.sort(key=lambda row: row['start'])
    validate_spans(rows, size-8-len(header))
    return rows


def compatible(parent, child):
    if parent['name'].endswith('.conv.conv.weight'):
        a, b = parent['shape'], child['shape']
        return (len(a) == len(b) == 3 and a[1] == b[2] == 1
                and a[0] == b[0] and a[2] == b[1])
    return same(parent['shape'], child['shape'])


def compare_stream(stream, prefix, header, size, pin, parent_rows, child_fd,
                   child_origin, child_rows, journal):
    """Generic bounded byte-comparison primitive after fixed-schema admission.

    Moving a singleton convolution axis preserves channel*K+tap byte order.
    Tensor journal entries remain unauthenticated until the whole object and
    transfer close successfully. No discarded parent payload is retained.
    """
    need(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'Pinned parent digest required')
    need(type(prefix) is bytes and len(prefix) == 8 and type(header) is bytes
         and 0 < len(header) <= 2*1024**2 and int.from_bytes(prefix, 'little') == len(header),
         'Pinned parent header required')
    origin = 8+len(header)
    validate_spans(parent_rows, size-origin)
    info = os.fstat(child_fd)
    need(stat.S_ISREG(info.st_mode) and type(child_origin) is int and child_origin > 8,
         'Regular retained child descriptor required')
    validate_spans(child_rows, info.st_size-child_origin)
    children = {row['name']: row for row in child_rows}
    need(set(children) == {row['name'] for row in parent_rows}, 'Complete one-to-one tensor mapping required')
    whole = hashlib.sha256()
    for value in [prefix, header]:
        got = exact(stream, len(value))
        need(got == value, 'Streamed source header differs')
        whole.update(got)
    tensors = []
    for row in parent_rows:
        child = children[row['name']]
        need(compatible(row, child) and row['end']-row['start'] == child['end']-child['start'],
             'Undeclared source tensor transform')
        tensor_hash = hashlib.sha256()
        blocks = []
        for start in range(0, row['end']-row['start'], BLOCK):
            count = min(BLOCK, row['end']-row['start']-start)
            data = exact(stream, count)
            observed = os.pread(child_fd, count, child_origin+child['start']+start)
            need(len(observed) == count and data == observed, 'Actual source/child tensor bytes differ')
            whole.update(data)
            tensor_hash.update(data)
            blocks.append(dict(parent_offset=origin+row['start']+start,
                child_offset=child_origin+child['start']+start, bytes=count,
                sha256=hashlib.sha256(data).hexdigest()))
        item = dict(name=row['name'], dtype='BF16', parent_shape=row['shape'], child_shape=child['shape'],
            parent_offset=origin+row['start'], child_offset=child_origin+child['start'],
            bytes=row['end']-row['start'], sha256=tensor_hash.hexdigest(), blocks=blocks)
        tensors.append(item)
        journal.write(json.dumps(dict(event='tensor_compared', whole_parent_authenticated=False,
                                     **item), separators=(',', ':'))+'\n')
        journal.flush()
    need(stream.read(1) == b'', 'Trailing parent bytes')
    need(whole.hexdigest() == pin, 'Complete parent object SHA256 differs')
    return dict(bytes=size, sha256=whole.hexdigest(), tensors=tensors, exact_tensor_count=len(tensors))


def verify_child(fd, path):
    before = os.fstat(fd)
    need(stat.S_ISREG(before.st_mode) and before.st_size == SIZE, 'Pinned regular child extent required')
    whole = hashlib.sha256()
    for start in range(0, SIZE, BLOCK):
        data = os.pread(fd, min(BLOCK, SIZE-start), start)
        need(len(data) == min(BLOCK, SIZE-start), 'Short retained child read')
        whole.update(data)
    after = path.lstat()
    need(stat.S_ISREG(after.st_mode) and (after.st_dev, after.st_ino, after.st_size)
         == (before.st_dev, before.st_ino, before.st_size) and whole.hexdigest() == CHILD_SHA,
         'Retained child descriptor/path identity differs')


def controls(metadata, model, contract_path):
    need(digest(contract_path) == CONTRACT_SHA, 'Exact prospective LFM3 source contract required')
    contract = document(contract_path)
    need(contract['before_complete_payload_outcomes'] is True
         and contract['candidate']['sha256'] == CHILD_SHA and contract['parent']['sha256'] == PARENT_SHA
         and contract['candidate']['bytes'] == contract['parent']['bytes'] == SIZE,
         'Fixed source object identities differ')
    receipt = metadata.parent/'metadata-verified-1.json'
    need(digest(receipt) == contract['metadata_verified_sha256']
         and document(receipt)['passed'] is True, 'Frozen metadata verification differs')
    for name, pin in contract['metadata_sha256'].items():
        need(digest(metadata/name) == pin, 'Frozen source metadata changed')
    child_config, parent_config = document(metadata/'child-config.json'), document(metadata/'parent-config.json')
    expected = dict(parent_config)
    need(type(expected.pop('auto_map', None)) is dict and 'model_file' not in expected,
         'Exact declared source config transform required')
    expected['model_file'] = 'lfm3.py'
    need(same(expected, child_config), 'Undeclared converted config change')
    geometry = dict(model_type='lfm3', dtype='bfloat16', architectures=['Lfm3ForCausalLM'],
        hidden_size=2048, intermediate_size=7680, num_hidden_layers=24,
        vocab_size=128000, num_attention_heads=16, num_key_value_heads=4, head_dim=128,
        conv_L_cache=7, conv_bias=False, norm_eps=1e-5, sliding_window=256,
        layer_types=LAYERS, tie_word_embeddings=True, use_sandwich_norm=True,
        max_position_embeddings=262144, rope_parameters={
            'full_attention': dict(partial_rotary_factor=.5, rope_theta=1000000., rope_type='default'),
            'sliding_attention': dict(rope_theta=10000, rope_type='default')})
    need(all(same(child_config.get(k), v) for k, v in geometry.items()), 'Fixed source geometry differs')
    for name in ['config.json', 'generation_config.json', 'tokenizer_config.json',
                 'chat_template.jinja', 'tokenizer.json', 'lfm3.py', 'README.md']:
        need(digest(model/name) == digest(metadata/('child-'+name)), 'Retained candidate asset differs')
    for name in ['generation_config.json', 'tokenizer_config.json', 'chat_template.jinja', 'tokenizer.json']:
        need(digest(metadata/('child-'+name)) == digest(metadata/('parent-'+name)),
             'Source/candidate tokenizer or generation asset differs')
    plans = {}
    for label, parent in [('parent', True), ('child', False)]:
        prefix = (metadata/(label+'-prefix.bin')).read_bytes()
        header = (metadata/(label+'-header.bin')).read_bytes()
        plans[label] = dict(prefix=prefix, header=header, origin=8+len(header),
                            rows=plan(prefix, header, SIZE, parent))
    return contract, plans


def transfer(plans, fd, output):
    url = f'https://huggingface.co/{PARENT}/resolve/{REVISION}/model.safetensors?download=true&toklet_source={uuid.uuid4().hex}'
    command = ['/usr/bin/curl', '-q', '--fail', '--location', '--proto', '=https',
        '--proto-redir', '=https', '--max-redirs', '5', '--connect-timeout', '20',
        '--max-time', '1800', '--max-filesize', str(SIZE), '--no-progress-meter',
        '-H', 'Accept-Encoding: identity', '--output', '-', '--write-out',
        '%{stderr}\nTOKLET_TRANSPORT {"http_code":%{http_code},"size_download":%{size_download},"ssl_verify_result":%{ssl_verify_result}}\n', url]
    started = time.monotonic_ns()
    result = resource = None
    error = None
    with (output/'parent.curl.log').open('x') as log, (output/'tensors.jsonl').open('x') as journal:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        try:
            parent = plans['parent']
            result = compare_stream(child.stdout, parent['prefix'], parent['header'], SIZE,
                PARENT_SHA, parent['rows'], fd, plans['child']['origin'], plans['child']['rows'], journal)
        except BaseException as exception:
            error = type(exception).__name__+': '+str(exception)
        finally:
            child.stdout.close()
            if error is not None:
                try:
                    child.terminate()
                except ProcessLookupError:
                    pass
            deadline = time.monotonic()+15
            while child.returncode is None:
                done, status, resource = os.wait4(child.pid, os.WNOHANG)
                if done:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                resource = None
                if time.monotonic() > deadline:
                    try:
                        child.kill()
                    except ProcessLookupError:
                        pass
                    _, status, resource = os.wait4(child.pid, 0)
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                time.sleep(.05)
    text = (output/'parent.curl.log').read_text()
    matches = re.findall(r'^TOKLET_TRANSPORT (.+)$', text, re.M)
    transport = json.loads(matches[0], object_pairs_hook=unique) if len(matches) == 1 else {}
    passed = (error is None and result is not None and child.returncode == 0 and resource is not None
              and all(type(transport.get(k)) is int and transport[k] == v
                      for k, v in [('http_code', 200), ('size_download', SIZE), ('ssl_verify_result', 0)]))
    record = dict(passed=passed, pid=child.pid, command=command, started_ns=started,
        completed_ns=time.monotonic_ns(), exit_code=child.returncode, error=error, transport=transport,
        final_transfer_wait4_peak_rss_bytes=resource.ru_maxrss if resource else None,
        parent_payload_retained_bytes=0, result=result)
    (output/'parent.json').write_text(json.dumps(record, indent=2)+'\n')
    need(passed, 'Parent transfer/byte comparison failed; primary evidence retained')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['metadata', 'model', 'contract', 'fresh_output']:
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    args.fresh_output.mkdir()
    contract, plans = controls(args.metadata, args.model, args.contract)
    path = args.model/'model.safetensors'
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        verify_child(fd, path)
        need(os.pread(fd, plans['child']['origin'], 0)
             == plans['child']['prefix']+plans['child']['header'], 'Actual retained child header differs')
        parent = transfer(plans, fd, args.fresh_output)
        verify_child(fd, path)
        controls(args.metadata, args.model, args.contract)
    finally:
        os.close(fd)
    need(parent['result']['exact_tensor_count'] == 286, 'Whole source tensor coverage differs')
    result = dict(format='toklet-lfm3-bf16-source-parity-v1', passed=True,
        contract_sha256=CONTRACT_SHA, child_sha256=CHILD_SHA, parent_sha256=PARENT_SHA,
        child_bytes=SIZE, parent_bytes=SIZE, exact_tensor_count=286, exact_convolution_transposes=13,
        exact_norm_tensors=119, logical_parameters=1728801536, parent=parent,
        candidate_payload_retained_bytes=SIZE, parent_payload_retained_bytes=0,
        source_tokenizer_assets_equal=True, source_config_transform_equal=True,
        complete_object_identity_verified=True, source_tensor_bytes_equal=True,
        remote_code_executed=False, model_execution_qualified=False, numerical_qualified=False,
        quality_qualified=False, memory_qualified=False, performance_qualified=False,
        managed_adoption_qualified=False, physical_iphone_qualified=False,
        scope='Bit-exact BF16 repackaging of the specified modified parent under13singleton-axis transforms. No parent payload retained; offline witnesses cannot independently rehash discarded bytes. Original-model quality and every runtime gate remain open.')
    (args.fresh_output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'parent'}))


if __name__ == '__main__':
    main()
