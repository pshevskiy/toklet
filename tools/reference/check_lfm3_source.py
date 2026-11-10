"""Validate retained LFM3 stream witnesses and rehash the complete child offline.

The discarded parent cannot be independently rehashed offline. Its whole-object
identity relies on the checked full-stream producer and transport/owner evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import lfm3_source as source

need = source.need
GUARD = 512*1024**2
FALSE_FLAGS = ['remote_code_executed', 'model_execution_qualified', 'numerical_qualified',
    'quality_qualified', 'memory_qualified', 'performance_qualified',
    'managed_adoption_qualified', 'physical_iphone_qualified']


def integer(value, minimum=0):
    return type(value) is int and minimum <= value < 2**63


def bounded(path, limit=2*1024**2):
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    need(len(raw) <= limit, 'Oversized retained witness')
    return raw


def parsed(raw):
    return json.loads(raw, object_pairs_hook=source.unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite witness')))


def validate_tensors(result, plans):
    need(type(result) is dict and set(result) == {'bytes', 'sha256', 'tensors', 'exact_tensor_count'}
         and integer(result['bytes'], 1) and result['bytes'] == source.SIZE
         and result['sha256'] == source.PARENT_SHA and type(result['tensors']) is list
         and integer(result['exact_tensor_count'], 1)
         and result['exact_tensor_count'] == len(result['tensors']) == 286,
         'Whole parent identity or complete tensor count differs')
    children = {row['name']: row for row in plans['child']['rows']}
    need(len(plans['parent']['rows']) == 286, 'Fixed parent tensor plan required')
    fields = {'name', 'dtype', 'parent_shape', 'child_shape', 'parent_offset', 'child_offset',
              'bytes', 'sha256', 'blocks'}
    for item, row in zip(result['tensors'], plans['parent']['rows']):
        child = children[row['name']]
        need(type(item) is dict and set(item) == fields and item['name'] == row['name']
             and item['dtype'] == 'BF16' and source.same(item['parent_shape'], row['shape'])
             and source.same(item['child_shape'], child['shape']) and source.compatible(row, child)
             and integer(item['bytes'], 1) and item['bytes'] == row['end']-row['start']
             and integer(item['parent_offset']) and item['parent_offset'] == plans['parent']['origin']+row['start']
             and integer(item['child_offset']) and item['child_offset'] == plans['child']['origin']+child['start']
             and type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}', item['sha256'])
             and type(item['blocks']) is list
             and len(item['blocks']) == (item['bytes']+source.BLOCK-1)//source.BLOCK,
             'Tensor identity, layout, extent or block count differs')
        cursor = 0
        for block in item['blocks']:
            need(type(block) is dict and set(block) == {'parent_offset', 'child_offset', 'bytes', 'sha256'}
                 and integer(block['parent_offset']) and block['parent_offset'] == item['parent_offset']+cursor
                 and integer(block['child_offset']) and block['child_offset'] == item['child_offset']+cursor
                 and integer(block['bytes'], 1) and block['bytes'] == min(source.BLOCK, item['bytes']-cursor)
                 and type(block['sha256']) is str and re.fullmatch('[0-9a-f]{64}', block['sha256']),
                 'Block witness coverage or digest differs')
            cursor += block['bytes']
        need(cursor == item['bytes'], 'Incomplete tensor block coverage')


def validate_resources(owned, owner, freeze, samples, parent):
    need(owned.get('passed') is True and owned.get('owned_group_absent') is True
         and type(owned.get('exit_code')) is int and owned['exit_code'] == 0
         and owned.get('failure') is None and owned.get('sample_errors') == []
         and owned.get('source_inputs_unchanged') is True
         and owned.get('complete_parent_authenticated') is True
         and owned.get('final_physical_peak_available') is False
         and all(owned.get(k) is False for k in ['model_execution_qualified', 'quality_qualified', 'model_adoption_qualified'])
         and integer(owned.get('parent_payload_retained_bytes')) and owned['parent_payload_retained_bytes'] == 0
         and integer(owned.get('guard_bytes'), 1) and owned['guard_bytes'] == GUARD
         and integer(owned.get('minimum_observed_free_bytes')) and owned['minimum_observed_free_bytes'] >= 2*1024**3
         and integer(owned.get('started_ns'), 1) and integer(owned.get('completed_ns'), 1)
         and owned['started_ns'] < owned['completed_ns']
         and integer(parent.get('started_ns'), 1) and integer(parent.get('completed_ns'), 1)
         and owned['started_ns'] <= parent['started_ns'] < parent['completed_ns'] <= owned['completed_ns'],
         'Owned campaign outcome, clock, budget or qualification differs')
    metrics = ['resident_size', 'phys_footprint', 'lifetime_max_phys_footprint']
    peaks = dict.fromkeys(metrics, 0)
    need(type(samples) is list and samples and integer(owned.get('sampled_observations'), 1)
         and len(samples) == owned['sampled_observations'], 'Complete resource ledger required')
    need(integer(owner.get('pid'), 1) and integer(owner.get('controller_pid'), 1)
         and integer(parent.get('pid'), 1) and len({owner['pid'], owner['controller_pid'], parent['pid']}) == 3
         and source.same(owner.get('command'), freeze.get('command'))
         and integer(owner.get('started_ns'), 1) and owner['started_ns'] == owned['started_ns'],
         'Owner process or command association differs')
    allowed = {owner['pid'], owner['controller_pid'], parent['pid']}
    seen = set()
    sampled_rss = dict.fromkeys(allowed, 0)
    last = owned['started_ns']
    for sample in samples:
        need(type(sample) is dict and set(sample) == {'observed_ns', 'processes'}
             and integer(sample['observed_ns'], 1) and last < sample['observed_ns'] < owned['completed_ns']
             and type(sample['processes']) is list and sample['processes'], 'Invalid resource observation window')
        last = sample['observed_ns']
        pids = set()
        sums = dict.fromkeys(metrics, 0)
        for row in sample['processes']:
            need(type(row) is dict and set(row) == {'pid', 'role', *metrics}
                 and integer(row['pid'], 1) and row['pid'] in allowed and row['pid'] not in pids
                 and row['role'] == ('controller' if row['pid'] == owner['controller_pid'] else 'owned_producer_or_curl')
                 and all(integer(row[key], 1) for key in metrics), 'Resource observation process identity differs')
            pids.add(row['pid'])
            seen.add(row['pid'])
            sampled_rss[row['pid']] = max(sampled_rss[row['pid']], row['resident_size'])
            if row['pid'] == parent['pid']:
                need(parent['started_ns'] <= sample['observed_ns'] <= parent['completed_ns'],
                     'Curl resource observation outside transfer window')
            for key in metrics:
                sums[key] += row[key]
        need(owner['controller_pid'] in pids and owner['pid'] in pids,
             'Controller/producer missing from resource observation')
        for key in metrics:
            need(0 < sums[key] <= GUARD, 'Aggregate source audit resource bound exceeded')
            peaks[key] = max(peaks[key], sums[key])
    need(seen == allowed, 'Parent curl absent from resource ledger')
    for key, metric in [('max_sampled_aggregate_rss_bytes', 'resident_size'),
                        ('max_sampled_aggregate_physical_bytes', 'phys_footprint'),
                        ('max_sampled_aggregate_lifetime_physical_bytes', 'lifetime_max_phys_footprint')]:
        need(integer(owned.get(key), 1) and owned[key] == peaks[metric], 'Resource ledger and summary differ')
    finals = ['final_producer_wait4_peak_rss_bytes', 'final_curl_wait4_peak_rss_bytes',
              'final_controller_peak_rss_bytes']
    final_pids = [owner['pid'], parent['pid'], owner['controller_pid']]
    need(all(integer(owned.get(key), 1) for key in finals)
         and integer(parent.get('final_transfer_wait4_peak_rss_bytes'), 1)
         and owned['final_curl_wait4_peak_rss_bytes'] == parent['final_transfer_wait4_peak_rss_bytes']
         and integer(owned.get('conservative_final_peak_rss_sum_bytes'), 1)
         and sum(owned[key] for key in finals) == owned['conservative_final_peak_rss_sum_bytes'] <= GUARD
         and all(owned[key] >= sampled_rss[pid] for key, pid in zip(finals, final_pids)),
         'Final process RSS peaks differ or exceed source audit bound')


def check(folder, metadata, model, contract, owned_path, freeze_path, owner_path, samples_path):
    _, plans = source.controls(metadata, model, contract)
    owned, freeze, owner = [source.document(p) for p in [owned_path, freeze_path, owner_path]]
    manifest = source.document(folder/'manifest.json')
    manifest_fields = {'format', 'passed', 'contract_sha256', 'child_sha256', 'parent_sha256',
        'child_bytes', 'parent_bytes', 'exact_tensor_count', 'exact_convolution_transposes',
        'exact_norm_tensors', 'logical_parameters', 'parent', 'candidate_payload_retained_bytes',
        'parent_payload_retained_bytes', 'source_tokenizer_assets_equal', 'source_config_transform_equal',
        'complete_object_identity_verified', 'source_tensor_bytes_equal', 'scope', *FALSE_FLAGS}
    need(type(manifest) is dict and set(manifest) == manifest_fields
         and manifest.get('format') == 'toklet-lfm3-bf16-source-parity-v1'
         and all(manifest.get(k) is True for k in ['passed', 'source_tokenizer_assets_equal',
             'source_config_transform_equal', 'complete_object_identity_verified', 'source_tensor_bytes_equal'])
         and all(manifest.get(k) is False for k in FALSE_FLAGS)
         and manifest.get('contract_sha256') == source.CONTRACT_SHA
         and manifest.get('child_sha256') == source.CHILD_SHA and manifest.get('parent_sha256') == source.PARENT_SHA
         and all(integer(manifest.get(k)) and manifest[k] == v for k, v in {
             'child_bytes': source.SIZE, 'parent_bytes': source.SIZE, 'exact_tensor_count': 286,
             'exact_convolution_transposes': 13, 'exact_norm_tensors': 119, 'logical_parameters': 1728801536,
             'candidate_payload_retained_bytes': source.SIZE, 'parent_payload_retained_bytes': 0}.items()),
         'Manifest identity, coverage or qualification differs')
    parent = manifest['parent']
    need(type(parent) is dict and set(parent) == {'passed', 'pid', 'command', 'started_ns', 'completed_ns',
             'exit_code', 'error', 'transport', 'final_transfer_wait4_peak_rss_bytes',
             'parent_payload_retained_bytes', 'result'}
         and parent.get('passed') is True and parent.get('error') is None
         and type(parent.get('exit_code')) is int and parent['exit_code'] == 0
         and integer(parent.get('parent_payload_retained_bytes')) and parent['parent_payload_retained_bytes'] == 0,
         'Unsuccessful or partial parent transfer')
    validate_tensors(parent['result'], plans)
    need(source.same(parent, source.document(folder/'parent.json')), 'Per-object/manifest transfer copies differ')
    need(type(freeze.get('source_sha256')) is dict and freeze['source_sha256']
         and integer(owned.get('source_files'), 1) and owned['source_files'] == len(freeze['source_sha256'])
         and all(integer(freeze.get(k)) and freeze[k] == v for k, v in {
             'guard_bytes': GUARD, 'disk_floor_bytes': 2*1024**3,
             'parent_payload_retained_bytes': 0, 'deadline_seconds': 1900}.items())
         and freeze.get('remote_code_execution') is False and freeze.get('model_execution') is False,
         'Frozen source campaign policy differs')
    command = freeze.get('command')
    need(type(command) is list and len(command) == 11 and all(type(x) is str for x in command)
         and command[1:3] == ['-B', str(Path(source.__file__))]
         and command[3:9] == ['--metadata', str(metadata), '--model', str(model), '--contract', str(contract)]
         and command[9] == '--fresh-output' and Path(command[10]).is_absolute(),
         'Frozen producer command association differs')
    need(str(Path(source.__file__)) in freeze['source_sha256'], 'Producer source pin missing')
    child_path = model/'model.safetensors'
    need(freeze['source_sha256'].get(str(child_path)) == source.CHILD_SHA, 'Frozen child object pin differs')
    for name, pin in freeze['source_sha256'].items():
        need(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'Invalid source input digest')
        if Path(name) != child_path:
            need(source.digest(Path(name)) == pin, 'Frozen source campaign input changed')
    curl = parent.get('command')
    expected = ['/usr/bin/curl', '-q', '--fail', '--location', '--proto', '=https', '--proto-redir', '=https',
        '--max-redirs', '5', '--connect-timeout', '20', '--max-time', '1800', '--max-filesize', str(source.SIZE),
        '--no-progress-meter', '-H', 'Accept-Encoding: identity', '--output', '-', '--write-out',
        '%{stderr}\nTOKLET_TRANSPORT {"http_code":%{http_code},"size_download":%{size_download},"ssl_verify_result":%{ssl_verify_result}}\n']
    need(type(curl) is list and curl[:-1] == expected and type(curl[-1]) is str
         and re.fullmatch(re.escape(f'https://huggingface.co/{source.PARENT}/resolve/{source.REVISION}/model.safetensors')
             + r'\?download=true&toklet_source=[0-9a-f]{32}', curl[-1]), 'Pinned bounded HTTPS command differs')
    raw = bounded(folder/'parent.curl.log').decode()
    receipts = re.findall(r'^TOKLET_TRANSPORT (.+)$', raw, re.M)
    need(len(receipts) == 1 and source.same(parsed(receipts[0]), parent['transport'])
         and source.same(parent['transport'], dict(http_code=200, size_download=source.SIZE, ssl_verify_result=0)),
         'Raw complete-object transport receipt differs')
    journal = bounded(folder/'tensors.jsonl')
    need(journal.endswith(b'\n') and source.same([parsed(line) for line in journal.splitlines()],
         [dict(event='tensor_compared', whole_parent_authenticated=False, **item) for item in parent['result']['tensors']]),
         'Tensor journal differs from authenticated whole-object witness')
    samples = parsed(bounded(samples_path, 16*1024**2))
    validate_resources(owned, owner, freeze, samples, parent)
    # Independent sequential child rehash, including every recorded block and tensor.
    by_name = {item['name']: item for item in parent['result']['tensors']}
    whole = hashlib.sha256()
    with child_path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        head = stream.read(plans['child']['origin'])
        need(head == plans['child']['prefix']+plans['child']['header'], 'Actual child header differs')
        whole.update(head)
        for row in plans['child']['rows']:
            item = by_name[row['name']]
            tensor = hashlib.sha256()
            for block in item['blocks']:
                need(stream.tell() == block['child_offset'], 'Independent child block offset differs')
                data = stream.read(block['bytes'])
                need(len(data) == block['bytes'] and hashlib.sha256(data).hexdigest() == block['sha256'],
                     'Independent retained child block digest differs')
                whole.update(data)
                tensor.update(data)
            need(tensor.hexdigest() == item['sha256'], 'Independent retained child tensor digest differs')
        need(stream.tell() == source.SIZE and stream.read(1) == b'' and whole.hexdigest() == source.CHILD_SHA,
             'Independent complete child object identity differs')
        after = child_path.stat()
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Retained child changed during validation')
    evidence = [folder/name for name in ['manifest.json', 'parent.json', 'parent.curl.log', 'tensors.jsonl']]
    evidence += [owned_path, freeze_path, owner_path, samples_path]
    return dict(format='toklet-lfm3-source-witness-seal-v1', passed=True,
        exact_tensor_count=286, exact_convolution_transposes=13, exact_norm_tensors=119,
        block_digests=sum(len(item['blocks']) for item in by_name.values()),
        child_payload_rehashed_offline=True, parent_payload_rehashed_offline=False,
        evidence_sha256={str(p): source.digest(p) for p in evidence},
        source_sha256={str(Path(__file__)): source.digest(Path(__file__)),
                       str(Path(source.__file__)): source.digest(Path(source.__file__))},
        numerical_qualified=False, model_execution_qualified=False, quality_adoption_qualified=False,
        scope='Every child byte independently rehashed; retained parent stream witnesses/transport/owner validated. Discarded parent identity relies on checked full-stream producer, not offline rehash or Merkle membership proofs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['capture', 'metadata', 'model', 'contract', 'owned', 'freeze', 'owner', 'samples', 'fresh_seal']:
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    need(not args.fresh_seal.exists() and args.fresh_seal.parent.is_dir(), 'Fresh seal path required')
    result = check(args.capture, args.metadata, args.model, args.contract, args.owned, args.freeze, args.owner, args.samples)
    with args.fresh_seal.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['evidence_sha256', 'source_sha256']}))


if __name__ == '__main__':
    main()
