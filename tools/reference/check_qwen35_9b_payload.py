#!/usr/bin/env python3
"""Independent pinned9B packed components/small-value/row proof; no graph or GPU.

The unchanged4B checker supplies only pure BF16/affine arithmetic primitives.
No4B source/admission constants are reused. Entire source and endpoints are
hashed, every retained component is matched, and750rows are checked bitwise.
"""
import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from check_qwen35_payload import affine_row, bf16, digest, hash_file, index_components, need, records

FILE_BYTES = 5038160953
FILE_SHA = '74855e1b61dc1955bdb835c2a3c9af8d6ce22aa9e46b1b99644fbda4afb540d2'
CONFIG_SHA = 'a765dc71b9fd04c2104920f0ce8c500c53a435250106fb833eba3ad6c8258a82'
HEADER_SHA = 'af0f407b4caf8bcd824272e8bf20136a6bfd66b923619e6509c6c72b62cbc615'
MAX_ROWS_BYTES = 64 * 1024**2


def file_sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


CONSUMER_PATHS = frozenset({'tools/reference/check_qwen35_9b_payload.py',
                            'tools/reference/test_check_qwen35_9b_payload.py'})


def consumer_update(path, expected_sha256):
    need(type(expected_sha256) is str and re.fullmatch('[0-9a-f]{64}',expected_sha256)
         and file_sha(path)==expected_sha256,'Consumer update digest differs')
    update=json.loads(path.read_text())
    need(update.get('format')=='toklet-qwen35-reader-consumer-update-v1','Named consumer update required')
    entries=update.get('updates',[])
    need(type(entries) is list and len(entries)==2 and {row.get('path') for row in entries}==CONSUMER_PATHS,
         'Only the two exact reader/test paths may change')
    for row in entries:
        need(set(row)=={'path','before_sha256','after_sha256','snapshot'},'Consumer update fields differ')
        need(all(type(row[k]) is str and re.fullmatch('[0-9a-f]{64}',row[k]) for k in ['before_sha256','after_sha256']),
             'Consumer version digests required')
        need(file_sha(Path(row['snapshot']))==row['before_sha256'],'Original reviewed consumer snapshot differs')
    return update


def timing_enclosure(report, elapsed_seconds):
    load=report['load_ns'];capture=report['load_and_capture_ns']
    need(type(load) is int and type(capture) is int and 0<load<=capture,'Load/capture timing order differs')
    need(type(elapsed_seconds) in (int,float) and math.isfinite(elapsed_seconds) and elapsed_seconds>0,
         'Finite owned elapsed time required')
    limit_ns=elapsed_seconds*1_000_000_000
    need(math.isfinite(limit_ns) and capture<=math.floor(limit_ns),'Capture timing exceeds owned process elapsed')


def closed_capture(native, update=None):
    receipt=json.loads((native/'checks.json').read_text())
    need(receipt.get('format')=='toklet-qwen35-storage-native-v1','Named native receipt required')
    need(receipt.get('passed') is True and receipt.get('group_absent') is True and receipt.get('error') is None,
         'Closed successful native process group required')
    need(type(receipt.get('exit_code')) is int and receipt['exit_code']==0,'Strict zero exit required')
    need(receipt.get('changed_inputs')==[],'Unchanged source endpoints required')
    peak=receipt.get('lifetime_peak_physical_bytes')
    need(type(peak) is int and 0<peak<=6*1024**3,'Strict bounded process peak required')
    elapsed=receipt.get('elapsed_seconds')
    need(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0,'Finite positive process elapsed required')
    pid=receipt.get('owned_pid')
    need(type(pid) is int and pid>0,'Owned process identity required')

    for name,key in [('inputs.json','inputs_sha256'),('console.log','console_sha256'),('time-and-errors.log','time_log_sha256')]:
        need(file_sha(native/name)==receipt.get(key),'Native evidence digest differs: '+name)
    peaks=re.findall(r'^\s*(\d+)\s+peak memory footprint\s*$',(native/'time-and-errors.log').read_text(),re.M)
    need(len(peaks)==1 and int(peaks[0])==peak,'Instrument/receipt process peaks differ')
    need('test result: ok. 1 passed; 0 failed;' in (native/'console.log').read_text(),'One completed explicit native test required')
    inputs=json.loads((native/'inputs.json').read_text())
    binary=Path(inputs['binary'])
    need(file_sha(binary)==inputs['binary_sha256']==receipt['binary_sha256'],'Native binary identity differs')
    need(inputs['test']=='mlx::qwen35::nine_b::payload::tests::actual_payload_probe','Pinned9B positive capture required')
    need(inputs['command']==['/usr/bin/time','-l',str(binary),inputs['test'],'--exact','--ignored','--nocapture','--test-threads=1'],'Exact source capture command required')
    need(inputs['child_allocator']=={'MallocLargeCache':'0','MallocAggressiveMadvise':None,'MallocNanoZone':None},
         'Named controlled child allocator required')
    need(type(inputs['source_pin_count']) is int and inputs['source_pin_count']>0,'Source pin coverage required')
    source=Path(inputs['source_inputs'])
    need(file_sha(source)==inputs['source_inputs_sha256'],'Capture source inputs changed')
    source_record=json.loads(source.read_text());pins=source_record['source_pins'];root=Path(inputs['source_worktree'])
    need(len(pins)==inputs['source_pin_count'],'Capture source pin coverage differs')
    versions={}
    if update is not None:
        need(update['source_inputs_sha256']==inputs['source_inputs_sha256'],'Consumer update source manifest differs')
        versions={row['path']:row for row in update['updates']}
        need(set(versions)==CONSUMER_PATHS,'Exact consumer update paths required')
        for name,row in versions.items():
            need(pins.get(name)==row['before_sha256'] and file_sha(Path(row['snapshot']))==row['before_sha256']
                 and file_sha(root/name)==row['after_sha256'],'Consumer update version differs')
    for name,expected in pins.items():
        actual=file_sha(root/name)
        need(actual==expected or (name in versions and actual==versions[name]['after_sha256']),
             'Capture source endpoint differs')
    need(receipt.get('numerical_execution_qualified') is False and receipt.get('managed_adoption_qualified') is False,
         'Storage-only scope required')
    return receipt


def check(model, native, update=None):
    receipt = closed_capture(native, update)
    need(receipt['passed'] is True and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
         and receipt['changed_inputs'] == [], 'Completed owned native capture required')
    peak = receipt['lifetime_peak_physical_bytes']
    need(type(peak) is int and 0 < peak <= 6*1024**3, 'Measured physical high-water required')
    log = (native/'console.log').read_text()
    reports = records(log, 'QWEN35_9B_PAYLOAD_REPORT')
    need(len(reports) == 1, 'Single native report required')
    report = reports[0]
    need(set(report)=={'container_sha256','container_bytes','config_sha256','header_sha256','source_payload_bytes',
         'retained_tensor_capacity_bytes','metadata_reservation_bytes','admitted_load_bytes','load_ns','load_and_capture_ns',
         'components','rows','rows_bytes','graph_executed','runtime_qualified'},'Exact storage report fields required')

    need(report['container_sha256'] == FILE_SHA and report['container_bytes'] == FILE_BYTES
         and report['config_sha256'] == CONFIG_SHA and report['header_sha256'] == HEADER_SHA
         and report['source_payload_bytes'] == 5038040064
         and report['retained_tensor_capacity_bytes'] == 5040162816
         and report['admitted_load_bytes'] == 5107402752
         and report['metadata_reservation_bytes'] == 64*1024**2
         and report['graph_executed'] is False and report['runtime_qualified'] is False,
         'Native source/accounting/scope differs')
    for name in ['container_bytes', 'source_payload_bytes', 'retained_tensor_capacity_bytes',
                 'admitted_load_bytes', 'metadata_reservation_bytes', 'load_ns',
                 'load_and_capture_ns', 'components', 'rows', 'rows_bytes']:
        need(type(report[name]) is int and report[name] > 0, 'Strict integer report required')
    timing_enclosure(report,receipt['elapsed_seconds'])
    need((model/'config.json').stat().st_size == 2431, 'Config size bound differs')
    config = (model/'config.json').read_bytes()
    need(len(config) == 2431 and digest(config) == CONFIG_SHA, 'Pinned config required')
    with (model/'model.safetensors').open('rb') as file:
        before = file.fileno()
        import os
        stat = os.fstat(before)
        need(stat.st_size == FILE_BYTES and hash_file(file) == FILE_SHA, 'Whole source digest differs')
        file.seek(0)
        need(file.read(8) == struct.pack('<Q', 120881), 'Header prefix differs')
        raw_header = file.read(120881)
        need(digest(raw_header) == HEADER_SHA, 'Header digest differs')
        header = json.loads(raw_header)
        header.pop('__metadata__')
        need(len(header) == 927, 'Source tensor count differs')
        components = index_components(records(log, 'QWEN35_9B_PAYLOAD_COMPONENT'), header)
        need(report['components'] == len(components), 'Reported component coverage differs')
        for name, spec in header.items():
            start, end = spec['data_offsets']
            file.seek(120889+start)
            record = components[name]
            if name.endswith(('.scales', '.biases')) or spec['dtype'] == 'U32':
                h = hashlib.sha256();left = end-start
                while left:
                    raw = file.read(min(left,65536));need(raw, 'Truncated component')
                    left -= len(raw);h.update(raw)
                want = dict(name=name, stored='original', bytes=end-start, sha256=h.hexdigest())
            else:
                need(spec['dtype'] == 'BF16' and end-start <= 65536, 'Small BF16 tensor bound differs')
                raw = file.read(end-start);need(len(raw) == end-start, 'Truncated BF16 component')
                decoded = bf16(raw)
                want = dict(name=name, stored='fp32', bytes=len(decoded), sha256=digest(decoded))
            need(record == want, 'Retained component differs: '+name)
        row_records = records(log, 'QWEN35_9B_PAYLOAD_ROW')
        expected = [(name, row) for name, spec in sorted(header.items()) if spec['dtype'] == 'U32'
                    for row in [0, spec['shape'][0]//2, spec['shape'][0]-1]]
        need(len(row_records) == len(expected) == report['rows'] == 750, 'Sampled row coverage differs')
        body_path = native/'capture/rows.f32le'
        need(body_path.stat().st_size == report['rows_bytes'] <= MAX_ROWS_BYTES, 'Row body bound differs')
        body = body_path.read_bytes();offset = 0;values = 0
        for record, (name, row) in zip(row_records, expected, strict=True):
            spec = header[name];columns = spec['shape'][1]*8
            raw_parts = []
            base = name.removesuffix('.weight')
            for part, stride in [(name,columns//2),(base+'.scales',columns//64*2),(base+'.biases',columns//64*2)]:
                file.seek(120889+header[part]['data_offsets'][0]+row*stride)
                raw = file.read(stride);need(len(raw) == stride, 'Truncated sampled row');raw_parts.append(raw)
            want = affine_row(*raw_parts, columns)
            need(set(record) == {'name','row','columns','offset','bytes','sha256'}
                 and record['name'] == name and record['sha256'] == digest(want), 'Row identity/digest differs')
            for key, value in [('row',row),('columns',columns),('offset',offset),('bytes',len(want))]:
                need(type(record[key]) is int and record[key] == value, 'Row geometry differs')
            need(body[offset:offset+len(want)] == want, 'Native row bits differ')
            offset += len(want);values += columns
        need(offset == len(body), 'Trailing row body')
        need(hash_file(file) == FILE_SHA, 'Source endpoint digest differs')
        after = os.fstat(before)
        need((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns) ==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'Source metadata changed')
    return dict(passed=True,cpu_storage_qualified=True,components=927,packed_matrices=250,fp32_vectors=153,fp32_convolutions=24,
                rows=750,row_values=values,row_bytes=offset,row_body_sha256=digest(body),
                source_sha256=FILE_SHA,native_binary_sha256=receipt['binary_sha256'],
                lifetime_peak_physical_bytes=peak,load_ns=report['load_ns'],
                numerical_graph_qualified=False,runtime_memory_fit_qualified=False,
                quality_qualified=False,managed_adoption_qualified=False,
                scope='Exact source/retained component bytes and selected scalar affine rows only; concurrent workload, no engine latency or graph/device qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('native', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--consumer-update', type=Path)
    parser.add_argument('--consumer-update-sha256')
    args = parser.parse_args()
    need((args.consumer_update is None)==(args.consumer_update_sha256 is None),'Both explicit consumer-update arguments required')
    update=consumer_update(args.consumer_update,args.consumer_update_sha256) if args.consumer_update else None
    result = check(args.model, args.native, update)
    with args.output.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False);file.write('\n')
    print(json.dumps(result))
