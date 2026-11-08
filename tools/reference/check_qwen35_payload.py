#!/usr/bin/env python3
"""Independent source-byte and sampled affine-row proof for Q1.B CPU storage.

This executes no model, tokenizer or GPU. Row reconstruction uses little-endian
U32 nibble extraction and Python FP64 multiply/add followed by FP32 rounding.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

FILE_BYTES = 2367237149
FILE_SHA = '0dfd6217f56cadcc373c90283d58d5e673833650fed1d6c299313fb536999043'
CONFIG_SHA = 'ac6f8df4f4c1fd9d2583996bdd7070ea801b703e8a2b36f6d4b673a94ea419b5'
HEADER_SHA = '7284f68df36a7761569b48321066e92d616306e33b4a9487029673a0ff344cb2'
MAX_ROWS_BYTES = 64 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bf16(raw):
    need(len(raw) % 2 == 0, 'BF16 extent must be even')
    result = bytearray()
    for (bits,) in struct.iter_unpack('<H', raw):
        need(bits & 0x7f80 != 0x7f80, 'BF16 must be finite')
        result.extend(struct.pack('<I', bits << 16))
    return bytes(result)


def affine_row(codes, scales, biases, columns):
    need(type(columns) is int and columns > 0 and columns % 64 == 0, 'Row geometry differs')
    need(len(codes) == columns // 2 and len(scales) == len(biases) == columns // 64 * 2,
         'Row component lengths differ')
    scale = struct.unpack('<' + 'f' * (columns // 64), bf16(scales))
    bias = struct.unpack('<' + 'f' * (columns // 64), bf16(biases))
    result = bytearray()
    for word_index, (word,) in enumerate(struct.iter_unpack('<I', codes)):
        for lane in range(8):
            column = word_index * 8 + lane
            value = scale[column // 64] * ((word >> (4 * lane)) & 15) + bias[column // 64]
            raw = struct.pack('<f', value)
            need(math.isfinite(struct.unpack('<f', raw)[0]), 'Nonfinite reconstructed row')
            result.extend(raw)
    return bytes(result)


def records(text, label):
    return [json.loads(line[len(label)+1:]) for line in text.splitlines() if line.startswith(label+' ')]


def index_components(rows, expected):
    need(len(rows) == len(expected), 'Component count differs')
    result = {}
    for row in rows:
        need(set(row) == {'name', 'stored', 'bytes', 'sha256'}, 'Component fields differ')
        name = row['name']
        need(name in expected and name not in result, 'Component identity/uniqueness differs')
        need(type(row['bytes']) is int and row['bytes'] > 0, 'Component byte count differs')
        result[name] = row
    return result


def hash_file(file):
    file.seek(0)
    h = hashlib.sha256()
    while raw := file.read(1024**2):
        h.update(raw)
    return h.hexdigest()


def check(model, native):
    receipt = json.loads((native/'checks.json').read_text())
    need(receipt['passed'] is True and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
         and receipt['changed_inputs'] == [], 'Completed owned native capture required')
    peak = receipt['lifetime_peak_physical_bytes']
    need(type(peak) is int and 0 < peak <= 6*1024**3, 'Measured physical high-water required')
    log = (native/'console.log').read_text()
    reports = records(log, 'QWEN35_PAYLOAD_REPORT')
    need(len(reports) == 1, 'Single native report required')
    report = reports[0]
    need(report['container_sha256'] == FILE_SHA and report['container_bytes'] == FILE_BYTES
         and report['config_sha256'] == CONFIG_SHA and report['header_sha256'] == HEADER_SHA
         and report['source_payload_bytes'] == 2367117312
         and report['retained_tensor_capacity_bytes'] == 2369040384
         and report['admitted_load_bytes'] == 2436280320
         and report['metadata_reservation_bytes'] == 64*1024**2
         and report['graph_executed'] is False and report['runtime_qualified'] is False,
         'Native source/accounting/scope differs')
    for name in ['container_bytes', 'source_payload_bytes', 'retained_tensor_capacity_bytes',
                 'admitted_load_bytes', 'metadata_reservation_bytes', 'load_ns',
                 'load_and_capture_ns', 'components', 'rows', 'rows_bytes']:
        need(type(report[name]) is int and report[name] > 0, 'Strict integer report required')
    need((model/'config.json').stat().st_size == 3024, 'Config size bound differs')
    config = (model/'config.json').read_bytes()
    need(len(config) == 3024 and digest(config) == CONFIG_SHA, 'Pinned config required')
    with (model/'model.safetensors').open('rb') as file:
        before = file.fileno()
        import os
        stat = os.fstat(before)
        need(stat.st_size == FILE_BYTES and hash_file(file) == FILE_SHA, 'Whole source digest differs')
        file.seek(0)
        need(file.read(8) == struct.pack('<Q', 119829), 'Header prefix differs')
        raw_header = file.read(119829)
        need(digest(raw_header) == HEADER_SHA, 'Header digest differs')
        header = json.loads(raw_header)
        header.pop('__metadata__')
        need(len(header) == 924, 'Source tensor count differs')
        components = index_components(records(log, 'QWEN35_PAYLOAD_COMPONENT'), header)
        need(report['components'] == len(components), 'Reported component coverage differs')
        for name, spec in header.items():
            start, end = spec['data_offsets']
            file.seek(119837+start)
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
        row_records = records(log, 'QWEN35_PAYLOAD_ROW')
        expected = [(name, row) for name, spec in sorted(header.items()) if spec['dtype'] == 'U32'
                    for row in [0, spec['shape'][0]//2, spec['shape'][0]-1]]
        need(len(row_records) == len(expected) == report['rows'] == 747, 'Sampled row coverage differs')
        body_path = native/'capture/rows.f32le'
        need(body_path.stat().st_size == report['rows_bytes'] <= MAX_ROWS_BYTES, 'Row body bound differs')
        body = body_path.read_bytes();offset = 0;values = 0
        for record, (name, row) in zip(row_records, expected, strict=True):
            spec = header[name];columns = spec['shape'][1]*8
            raw_parts = []
            base = name.removesuffix('.weight')
            for part, stride in [(name,columns//2),(base+'.scales',columns//64*2),(base+'.biases',columns//64*2)]:
                file.seek(119837+header[part]['data_offsets'][0]+row*stride)
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
    return dict(passed=True,components=924,packed_matrices=249,fp32_vectors=153,fp32_convolutions=24,
                rows=747,row_values=values,row_bytes=offset,row_body_sha256=digest(body),
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
    args = parser.parse_args()
    result = check(args.model, args.native)
    with args.output.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False);file.write('\n')
    print(json.dumps(result))
