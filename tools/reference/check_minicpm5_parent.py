"""Validate and seal bounded parent-stream witnesses, without claiming rehash.

Discarded payload cannot be rehashed offline. This validates producer/transport
and owned-execution evidence, all tensor/block coverage and exact norm digests;
the resulting seal binds the retained witness bytes for later revalidation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import minicpm5_parent as source

need = source.need


def integer(value, minimum=0):
    return type(value) is int and minimum <= value < 2 ** 63


def sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def bounded_bytes(path):
    with path.open('rb') as stream:
        raw = stream.read(2 * 1024 ** 2 + 1)
    need(len(raw) <= 2 * 1024 ** 2, 'Oversize retained witness')
    return raw


def validate_result(result, shard):
    name, size, pin, prefix, header, norms = shard
    need(type(result) is dict and set(result) == {'bytes', 'sha256', 'tensors', 'exact_child_norms'}
         and integer(result['bytes'], 1) and result['bytes'] == size
         and result['sha256'] == pin, 'Whole-object witness identity differs')
    spans = source.plan(prefix, header, size)
    need(type(result['tensors']) is list and len(result['tensors']) == len(spans),
         'Tensor witness coverage differs')
    origin = len(prefix + header)
    for item, expected in zip(result['tensors'], spans):
        need(type(item) is dict and set(item) == {'name', 'dtype', 'shape', 'offset', 'bytes',
             'sha256', 'blocks', 'exact_child_norm'}
             and item['name'] == expected['name'] and item['dtype'] == 'BF16'
             and same(item['shape'], expected['shape']) and sha(item['sha256'])
             and integer(item['offset']) and item['offset'] == origin + expected['start']
             and integer(item['bytes'], 1) and item['bytes'] == expected['end'] - expected['start']
             and item['exact_child_norm'] is (item['name'] in norms),
             'Tensor witness identity/layout differs')
        if item['name'] in norms:
            need(item['sha256'] == hashlib.sha256(norms[item['name']]).hexdigest(),
                 'Exact copied norm digest differs')
        need(type(item['blocks']) is list
             and len(item['blocks']) == (item['bytes'] + source.BLOCK - 1) // source.BLOCK,
             'Block witness count differs')
        cursor = item['offset']
        end = cursor + item['bytes']
        for block in item['blocks']:
            need(type(block) is dict and set(block) == {'offset', 'bytes', 'sha256'}
                 and integer(block['offset']) and block['offset'] == cursor
                 and integer(block['bytes'], 1) and block['bytes'] == min(source.BLOCK, end - cursor)
                 and sha(block['sha256']), 'Block witness holes/overlap/extent/digest differs')
            if len(item['blocks']) == 1:
                need(block['sha256'] == item['sha256'], 'Single-block/tensor digests differ')
            cursor += block['bytes']
        need(cursor == end, 'Incomplete block witness')
    need(integer(result['exact_child_norms']) and result['exact_child_norms'] == len(norms),
         'Norm witness count differs')


def validate_transfer(record, shard, owned):
    fields = {'passed', 'name', 'pid', 'command', 'started_ns', 'completed_ns', 'exit_code',
              'error', 'http_code', 'size_download', 'ssl_verify_result',
              'final_wait4_peak_rss_bytes', 'payload_retained', 'result'}
    name, size, _, _, _, _ = shard
    need(type(record) is dict and set(record) == fields and record['passed'] is True
         and record['name'] == name and integer(record['pid'], 1)
         and integer(record['exit_code']) and record['exit_code'] == 0
         and record['error'] is None and record['payload_retained'] is False
         and type(record['http_code']) is int and record['http_code'] == 200
         and integer(record['size_download'], 1) and record['size_download'] == size
         and type(record['ssl_verify_result']) is int and record['ssl_verify_result'] == 0,
         'Unsuccessful or partial transfer witness')
    need(integer(record['started_ns'], 1) and integer(record['completed_ns'], 1)
         and owned['started_ns'] <= record['started_ns'] < record['completed_ns'] <= owned['completed_ns']
         and integer(record['final_wait4_peak_rss_bytes'], 1)
         and record['final_wait4_peak_rss_bytes'] <= 256 * 1024 ** 2,
         'Transfer clock/resource enclosure differs')
    command = record['command']
    need(type(command) is list and all(type(x) is str for x in command)
         and len(command) == 24, 'Transfer command shape differs')
    expected = ['/usr/bin/curl', '-q', '--fail', '--location', '--proto', '=https',
                '--proto-redir', '=https', '--connect-timeout', '20', '--max-time', '1200',
                '--max-filesize', str(size), '--no-progress-meter', '-H', 'Accept-Encoding: identity',
                '--dump-header']
    need(command[:18] == expected and Path(command[18]).name == name + '.headers'
         and command[19:23] == ['--output', '-', '--write-out', '%{stderr}\nTOKLET_CURL_JSON%{json}\n']
         and re.fullmatch(re.escape(f'https://huggingface.co/{source.PARENT}/resolve/{source.REVISION}/{name}')
                          + r'\?download=true&toklet_audit=[0-9a-f]{32}', command[-1]),
         'Bounded pinned TLS command differs')
    validate_result(record['result'], shard)


def check(folder, metadata_directory, child, owned_path, freeze_path):
    owned = source.metadata(owned_path)
    need(owned.get('passed') is True and owned.get('owned_group_absent') is True
         and type(owned.get('exit_code')) is int and owned['exit_code'] == 0
         and owned.get('failure') is None and owned.get('source_inputs_unchanged') is True
         and integer(owned.get('started_ns'), 1) and integer(owned.get('completed_ns'), 1)
         and owned['started_ns'] < owned['completed_ns']
         and integer(owned.get('minimum_observed_free_bytes'))
         and owned['minimum_observed_free_bytes'] >= 2 * 1024 ** 3
         and owned.get('final_physical_peak_available') is False,
         'Owned audit was not qualified and closed')
    for key in ['final_auditor_wait4_peak_rss_bytes', 'max_sampled_aggregate_rss_bytes',
                'max_sampled_aggregate_physical_bytes', 'max_sampled_aggregate_lifetime_physical_bytes']:
        need(integer(owned.get(key), 1) and owned[key] <= 256 * 1024 ** 2,
             'Owned source-audit resource witness exceeds bound')
    freeze = source.metadata(freeze_path)
    need(type(freeze.get('source_sha256')) is dict and freeze['source_sha256']
         and all(sha(pin) and source.digest(Path(p)) == pin for p, pin in freeze['source_sha256'].items()),
         'Producer/test/runner/contract inputs changed')
    need(type(freeze.get('command')) is list and len(freeze['command']) == 5
         and all(type(part) is str for part in freeze['command'])
         and Path(freeze['command'][1]).name == 'minicpm5_parent.py'
         and Path(freeze['command'][2]).resolve() == metadata_directory.resolve()
         and Path(freeze['command'][3]).resolve() == child.resolve(),
         'Frozen producer command/source association differs')
    shards = source.controls(metadata_directory, child)
    manifest = source.metadata(folder / 'manifest.json')
    fields = {'format', 'passed', 'parent_repository', 'parent_revision', 'authenticated_bytes',
              'parent_objects', 'parent_tensor_count', 'exact_child_norm_tensors', 'stream_block_bytes',
              'retained_payload_bytes', 'child_sha256', 'parent_full_object_identity_verified',
              'quantized_conversion_lineage_qualified', 'model_execution_qualified',
              'task_quality_adoption_qualified', 'physical_iphone_qualified'}
    need(type(manifest) is dict and set(manifest) == fields
         and manifest['format'] == 'toklet-minicpm5-parent-auth-v1'
         and manifest['passed'] is True and manifest['parent_full_object_identity_verified'] is True
         and manifest['parent_repository'] == source.PARENT and manifest['parent_revision'] == source.REVISION
         and manifest['child_sha256'] == source.CHILD_SHA
         and all(manifest[k] is False for k in ['quantized_conversion_lineage_qualified',
                  'model_execution_qualified', 'task_quality_adoption_qualified', 'physical_iphone_qualified'])
         and all(type(manifest[k]) is int for k in ['authenticated_bytes', 'parent_tensor_count',
                  'exact_child_norm_tensors', 'stream_block_bytes', 'retained_payload_bytes'])
         and manifest['authenticated_bytes'] == sum(row[1] for row in shards)
         and manifest['parent_tensor_count'] == 381 and manifest['exact_child_norm_tensors'] == 85
         and manifest['stream_block_bytes'] == source.BLOCK and manifest['retained_payload_bytes'] == 0
         and type(manifest['parent_objects']) is list and len(manifest['parent_objects']) == len(shards),
         'Manifest identity/coverage/qualification differs')
    previous = owned['started_ns']
    pids = set()
    evidence = [owned_path, freeze_path, folder / 'manifest.json']
    for record, shard in zip(manifest['parent_objects'], shards):
        validate_transfer(record, shard, owned)
        need(previous <= record['started_ns'] and record['pid'] not in pids,
             'Sequential transfer windows overlap or PID duplicates')
        previous = record['completed_ns'];pids.add(record['pid'])
        name = record['name']
        need(same(record, source.metadata(folder / (name + '.json'))), 'Per-object witness copy differs')
        # A retained witness may be copied for publication/offline controls. Bind
        # its command to the original owner's output, without relabelling paths.
        need(Path(record['command'][18]).parent.resolve() == Path(freeze['command'][4]).resolve(),
             'Transfer header destination differs from owned output')
        journal = bounded_bytes(folder / (name + '.tensors.jsonl'))
        need(0 < len(journal) <= 2 * 1024 ** 2 and journal.endswith(b'\n'), 'Invalid tensor journal extent')
        rows = [json.loads(line, object_pairs_hook=source.unique) for line in journal.splitlines()]
        expected = [dict(event='tensor_read', object_authenticated=False, **row)
                    for row in record['result']['tensors']]
        need(same(rows, expected), 'Tensor journal differs from completed whole-object witness')
        raw_log = bounded_bytes(folder / (name + '.curl.log')).decode()
        receipt = re.findall(r'^TOKLET_CURL_JSON(.+)$', raw_log, re.M)
        need(len(receipt) == 1, 'Missing/duplicate raw transport receipt')
        transport = json.loads(receipt[0], object_pairs_hook=source.unique)
        need(type(transport) is dict and type(transport.get('exitcode')) is int
             and transport['exitcode'] == 0
             and all(same(transport.get(key), record[key])
                     for key in ['http_code', 'size_download', 'ssl_verify_result']),
             'Raw curl receipt and completed transport witness differ')
        header_text = bounded_bytes(folder / (name + '.headers')).decode('latin1')
        responses = re.split(r'(?m)^HTTP/\S+ ', header_text)
        need(len(responses) > 1 and re.match(r'200(?:\r?\n| )', responses[-1])
             and re.search(r'(?im)^content-range:', responses[-1]) is None,
             'Final HTTP headers do not describe a complete object')
        lengths = re.findall(r'(?im)^content-length:\s*(\d+)\s*$', responses[-1])
        need(not lengths or (len(lengths) == 1 and int(lengths[0]) == shard[1]),
             'Final HTTP Content-Length differs')
        evidence.extend(folder / (name + suffix) for suffix in ['.json', '.headers', '.curl.log', '.tensors.jsonl'])
    curl_rss = [row['final_wait4_peak_rss_bytes'] for row in manifest['parent_objects']]
    need(same(owned['final_curl_rss_bytes'], curl_rss)
         and owned['final_auditor_wait4_peak_rss_bytes'] + sum(curl_rss) <= 256 * 1024 ** 2,
         'Conservative sum of final process RSS peaks exceeds source-audit budget')
    return dict(format='toklet-minicpm5-parent-witness-seal-v1', passed=True,
                parent_full_object_identity_verified=True, authenticated_bytes=manifest['authenticated_bytes'],
                parent_tensors=381, exact_child_norm_tensors=85,
                block_digests=sum(len(t['blocks']) for row in manifest['parent_objects'] for t in row['result']['tensors']),
                evidence_sha256={str(p): source.digest(p) for p in evidence},
                source_sha256={str(Path(__file__)): source.digest(Path(__file__)),
                               str(Path(source.__file__)): source.digest(Path(source.__file__))},
                payload_rehashed_offline=False, quantized_conversion_lineage_qualified=False,
                quality_adoption_qualified=False,
                scope='Structural/transport/owner validation and retained-witness seal. Discarded payloads cannot be rehashed offline; block hashes rely on the checked full-stream producer, not Merkle membership proofs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['capture', 'metadata_directory', 'child', 'owned', 'freeze', 'fresh_seal']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = check(args.capture, args.metadata_directory, args.child, args.owned, args.freeze)
    with args.fresh_seal.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ['evidence_sha256', 'source_sha256']}))
