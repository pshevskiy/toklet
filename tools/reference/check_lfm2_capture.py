"""Fail-closed reader of the preregistered LFM2 saved capture.

This executes independent local operator checks on captured operands, never a
forward graph. Source/native trajectory arrays and original weights are inputs,
not executable model code. The external owner separately establishes process,
module/runtime identity and closed-group/resource evidence. An outer adoption
or application qualification is never minted here.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import struct
from pathlib import Path

import numpy as np
from lfm2_capture_contract import (
    ATTENTION, CONVOLUTION, Dispatcher, File, GATES_SHA, HALF_PAD, MAX_NUMERIC,
    PROTOCOL_SHA, SOURCE_MANIFEST_SHA, SOURCE_SHA, STAGES, WORD_PAD, Source,
    Weights, Totals, integer, parse_json, require, strict_equal,
)
from lfm2_capture_bytes import (
    Capture, PhysicalState, RAW_PLANNED, digest, expected_owner_bytes,
    hash_words, numeric_head_hash, same_words,
)
from lfm2_capture_workload import (
    CASES, FAULTS, ORDINARY, PLAIN, PRIMARY, TEACHER, FreeRepeat,
    compare_free_repeats, validate_begin, validate_terminal, verify_protocol,
)

INVENTORY_SHA = 'ecf0ca26c478045d4354dd74f47baddb9d54b4821fa9a3231a7e36c981b9b547'
TOKENIZER_SHA = '98cff83b4f6d7e9d8929bebc62b07e92cf1b3f99c80d16bafe8b84a75448f40b'
CODEC_SHA = 'c0d521d4cf09507e667e68a36ccfd8698be5dd30b72adcf5befb5b18c7cd283a'
EXTERNAL = 512 * 1024**2
PROCESS_CAP = 6 * 1024**3
MEMORY_KEYS = ('packed_weights', 'direct_norms', 'convolution_coefficients', 'zero_bias',
               'cache', 'convolution_history', 'scratch', 'logits', 'tokens', 'diagnostic_gpu',
               'diagnostic_readback_reserved', 'metadata', 'shared_framework_allowance',
               'external_resident', 'admitted', 'cold_peak')
FAULT_DEBUG = {'OmitHistory9': 'OmitHistory(9)', 'Head': 'Head', 'FirstCopy': 'FirstCopy'}
SHA_PATTERN = re.compile('[0-9a-f]{64}')


def sha_string(value):
    require(type(value) is str and SHA_PATTERN.fullmatch(value) is not None, 'SHA256 string')
    return value


def fields(value, keys, name):
    require(type(value) is dict and set(value) == set(keys), name + ' exact keys')
    return value


def error_string(value):
    require(type(value) is str and 0 < len(value) < 4096, 'actual nonempty bounded error')
    return value


def summary_from_stdout(file):
    """Libtest may prefix the first record; find the marker within each line."""
    marker = b'LFM2_REAL_CAPTURE_SUMMARY '
    hits = []; harness = False; consumed = 0
    file.file.seek(0)
    while line := file.file.readline(1024 * 1024 + 1):
        consumed += len(line)
        require(consumed <= file.bytes and len(line) <= 1024 * 1024, 'bounded stdout line/extent')
        if marker in line:
            hits.append(line.split(marker, 1)[1])
        harness |= b'test result: ok. 1 passed; 0 failed;' in line
    require(consumed == file.bytes and len(hits) == 1, 'one complete writer summary')
    require(harness, 'positive literal one-test harness')
    return parse_json(hits[0])


def footer(capture):
    """Authenticate end.prefix against bytes BEFORE end, not complete summary."""
    record = capture.records[-1]
    require(record['kind'] == 'end', 'final end event required')
    prefix = record['body']['prefix']
    f = capture.journal_file
    f.file.seek(0)
    h = hashlib.sha256(); size = 0
    arrays = boundaries = other = 0
    for expected in capture.records[:-1]:
        line = f.file.readline(262145)
        require(parse_json(line) == expected, 'consumed prefix changed')
        h.update(line); size += len(line)
        kind = expected['kind']
        if kind == 'array':
            arrays += 1
        elif kind in ('boundary', 'session_owners', 'session_begin', 'source_drop'):
            boundaries += 1
        elif kind != 'start':
            other += 1
    expected = dict(raw_bytes=capture.raw_file.bytes, raw_sha256=capture.raw_file.sha256,
                    journal_bytes=size, journal_sha256=h.hexdigest(),
                    records=len(capture.records) - 1, arrays=arrays, boundaries=boundaries, other=other)
    require(strict_equal(prefix, expected), 'writer prefix-versus-complete footer identity')
    return record['body']


def memory_ledger(value, model=None, diagnostics=False, all_logits=False):
    fields(value, MEMORY_KEYS, 'native accounted memory')
    for v in value.values():
        integer(v, 0, PROCESS_CAP, 'native memory byte count')
    require(value['admitted'] == sum(value[k] for k in MEMORY_KEYS[:-2]), 'memory component sum')
    require(value['packed_weights'] == 199360512 and value['direct_norms'] == 139500 and
            value['convolution_coefficients'] == 123160 and value['zero_bias'] == 131086,
            'original model physical categories')
    require(value['shared_framework_allowance'] == 64 * 1024**2 and value['metadata'] > 0,
            'accounted framework/metadata premise')
    if model is None:
        require(all(value[k] == 0 for k in ('cache', 'convolution_history', 'scratch', 'logits',
                'tokens', 'diagnostic_gpu', 'diagnostic_readback_reserved', 'external_resident')),
                'model-only ledger')
        return dict(value)
    sizes = expected_owner_bytes(diagnostics, all_logits)
    require(value['convolution_history'] == sum(sizes[i] for i in range(24, 44)) and
            value['scratch'] == sum(sizes[i] for i in range(44, 60)) and
            value['logits'] == sizes[61] + sizes[62] and value['tokens'] == sizes[60] + 512 * 4 and
            value['diagnostic_gpu'] == sum(sizes[i] for i in sizes if i >= 64),
            'concrete physical numerical accounting')
    # Set/Frame metadata is included in cache, excluded from the other metadata
    # field. It is observed separately, not guessed from Python object sizes.
    physical_cache = sum(sizes[i] for i in range(24)) + sizes[63]
    require(physical_cache <= value['cache'] < physical_cache + 1024**2,
            'bounded cache transaction metadata overhead')
    require(value['metadata'] >= model['metadata'] and value['metadata'] < 16 * 1024**2 and
            value['external_resident'] == EXTERNAL and
            value['diagnostic_readback_reserved'] == (88080384 if diagnostics else 0) and
            value['cold_peak'] == max(model['cold_peak'], value['admitted']),
            'Session accounted metadata/readback/external/cold maximum')
    return dict(value)


class Reader:
    def __init__(self, capture, source, weights, protocol, control):
        self.capture, self.source, self.weights = capture, source, weights
        self.protocol, self.control = protocol, control
        self.dispatch = Dispatcher(weights, source)
        self.cursor = 0
        self.attempts = self.rows = self.selected_groups = self.producer_rows = 0
        self.primary_chunks = self.primary_rows = self.guard_records = self.guard_bytes = 0
        self.history_stacks = self.row_hashes = self.measured = 0
        self.run = 0; self.current_attempt = 0
        self.state = None; self.boundary_seq = None
        self.model_table = []; self.model_memory = None
        self.session_memories = {}; self.session_identities = set()
        self.head_baseline = {}; self.state_baseline = {}
        self.row_records = {}; self.free = []
        self.guard_only_snapshot_owners = 0
        self.source_state_main, self.source_state_wide = Totals(), Totals()
        self.source_state_identities = set()

    def peek(self):
        require(self.cursor < len(self.capture.records), 'truncated event stream')
        return self.capture.records[self.cursor]

    def event(self, kind):
        r = self.peek()
        require(r['kind'] == kind, f'expected {kind} at journal sequence {self.cursor}')
        self.cursor += 1
        return r['body']

    def exact_event(self, kind, expected):
        body = self.event(kind)
        require(strict_equal(body, expected), 'literal ' + kind + ' event differs')
        return body

    def array(self, kind, index, *, attempt=None, shape=None, dtype='f16le', query=(0, 0), keys=(0, 0)):
        r = self.peek()
        sid = self.state.session
        aid = self.current_attempt if attempt is None else attempt
        require(r['kind'] == 'array' and r['id'] == [sid, aid, kind, index] and
                r['dtype'] == dtype and r['query'] == list(query) and r['keys'] == list(keys),
                'missing/reordered/foreign array descriptor')
        if shape is not None:
            require(r['rank'] == len(shape) and r['shape'] == list(shape) + [1] * (3 - len(shape)),
                    'frozen logical array shape')
        self.cursor += 1
        return r

    def start(self):
        b = self.event('start')
        fields(b, ('format', 'protocol_sha256', 'inventory_sha256', 'control_sha256',
                   'source_manifest_sha256', 'source_arrays_sha256', 'gates_sha256',
                   'owner_manifest', 'binary', 'authorization', 'stages', 'limits', 'profile', 'scope'), 'start')
        require(b['format'] == 'toklet-lfm2-real-capture-v1' and b['protocol_sha256'] == PROTOCOL_SHA and
                b['inventory_sha256'] == INVENTORY_SHA and b['source_manifest_sha256'] == SOURCE_MANIFEST_SHA and
                b['source_arrays_sha256'] == SOURCE_SHA and b['gates_sha256'] == GATES_SHA,
                'preregistered source/profile identities')
        for key in ('owner_manifest', 'binary', 'authorization'):
            require(b[key] == self.control['capture_control'][key], 'executed capture authority ' + key)
        require(b['control_sha256'] == self.control['capture_control_sha256'] and
                b['stages'] == self.protocol['cases'][0]['stages'] and
                b['profile'] == 'native-half-direct-gain-exact-original-F32-affine-center',
                'exact captured stage/profile contract')
        require(b['limits'] == dict(raw=256*1024**2, raw_inventory=RAW_PLANNED,
                journal=16*1024**2, numeric_capture_reserve=64*1024**2, external_resident=EXTERNAL,
                process=PROCESS_CAP, attempts=104, attempted_rows=367, arrays=14672), 'capture limits')
        for capacity in (0, 513):
            v = self.event('capacity_reject')
            fields(v, ('capacity', 'error', 'scope'), 'capacity rejection')
            require(v['capacity'] == capacity and v['scope'] == 'cold_data_preparation', 'capacity rejection scope')
            error_string(v['error'])
        drop = self.event('source_drop')
        fields(drop, ('load', 'prepared', 'model', 'source_payload_retained', 'whole_container_bytes'), 'source release')
        require(drop['source_payload_retained'] is False and drop['whole_container_bytes'] == 199527820,
                'original CPU store release declaration')
        load = drop['load']; prep = drop['prepared']
        fields(load, ('container_bytes','header_bytes','header_capacity_bytes','payload_bytes',
                      'payload_capacity_bytes','metadata_reservation_bytes','external_resident_bytes',
                      'admitted_load_bytes'), 'CPU load ledger')
        require(load['container_bytes'] == 199527820 and load['header_bytes'] == 36740 and
                load['payload_bytes'] == 199491072 and load['header_capacity_bytes'] >= 36740 and
                load['payload_capacity_bytes'] >= 199491072 and load['external_resident_bytes'] == EXTERNAL,
                'original CPU load extents')
        require(load['admitted_load_bytes'] == 65536 + sum(load[k] for k in
                ('header_capacity_bytes','payload_capacity_bytes','metadata_reservation_bytes','external_resident_bytes')),
                'CPU load admission sum')
        fields(prep, ('source_resident_bytes','descriptor_capacities','descriptor_element_bytes',
                      'typed_BF16_capacities','typed_BF16_lengths','direct_words'), 'prepared CPU owners')
        require(prep['direct_words'] == 65280 and len(prep['typed_BF16_capacities']) == 55 and
                len(prep['descriptor_capacities']) == len(prep['descriptor_element_bytes']) == 3,
                'prepared typed and descriptor inventory')
        for got, minimum in zip(prep['descriptor_capacities'], (16,6,10), strict=True):
            integer(got, minimum, 1024, 'actual descriptor capacity')
        for v in prep['descriptor_element_bytes']:
            integer(v, 1, 4096, 'descriptor element extent')
        minima = [1024] + [1024] * 32 + [64] * 12 + [3072] * 10
        for v, n in zip(prep['typed_BF16_capacities'], minima, strict=True):
            integer(v, n, 8192, 'actual typed BF16 capacity')
        require(prep['typed_BF16_lengths'] == minima and sum(minima) == prep['direct_words'],
                'actual direct-gain/tap lengths')
        cpu = load['payload_capacity_bytes'] + load['metadata_reservation_bytes'] + EXTERNAL
        # The caller passes an actual File stream, not a retained whole-file Vec.
        # Its bounded external reservation is counted once, alongside the owned
        # payload, metadata reserve, typed copies and descriptors.
        cpu += sum(a*b for a,b in zip(prep['descriptor_capacities'],prep['descriptor_element_bytes'],strict=True))
        cpu += 2 * sum(prep['typed_BF16_capacities'])
        require(prep['source_resident_bytes'] == cpu, 'full external/payload/typed preparation overlap')
        self.model_memory = memory_ledger(drop['model'])
        require(self.model_memory['cold_peak'] == self.model_memory['admitted'] + cpu,
                'accounted upload cold peak')
        expected = self.weights.model_owners()
        for e in expected:
            actual = self.event('model_before')
            self.validate_model_owner(actual, e)
            self.model_table.append(actual)
        spans = sorted((r['address'], r['address'] + r['bytes']) for r in self.model_table)
        require(all(a[1] <= b[0] for a,b in zip(spans,spans[1:])), 'model physical overlap')
        require(len({x['device'] for x in self.model_table}) == 1, 'model one-device ownership')
        self.device = self.model_table[0]['device']

    @staticmethod
    def validate_model_owner(actual, expected):
        tail = expected['role'] in ('direct_gain','convolution_tap','zero_bias')
        fields(actual, tuple(expected) + ('address','device') + (('tail',) if tail else ()), 'model owner')
        require(all(actual[k] == v for k,v in expected.items()), 'original model bytes/layout/name')
        integer(actual['address'], 1, 2**64-1, 'model address')
        integer(actual['device'], 1, 2**64-1, 'model device')
        if tail:
            require(actual['tail'] == [0 if expected['role'] == 'zero_bias' else WORD_PAD]*7,
                    'original model guard bytes')

    def history_stack(self):
        d = self.array(11, self.history_stacks, shape=(10,2,2055))
        self.history_stacks += 1
        self.state.initial_or_reset_histories(d)

    def boundary(self, label, *, initial=False, success=None, reset=False):
        before = self.state.last
        d = self.array(9, self.guard_records, shape=(5824 if self.state.diagnostics else 980,), dtype='u8')
        self.guard_records += 1; self.guard_bytes += d['bytes']
        require(self.guard_bytes <= 1024**2, 'cumulative actual guard reserve')
        if initial:
            owners = self.event('session_owners')
            fields(owners, ('session','owners','status_id'), 'Session owner registry')
            require(owners['session'] == self.state.session and owners['status_id'] == 63, 'status registry')
            self.state.owner_table(owners['owners'])
            require(self.state.device == self.device, 'model/Session device mismatch')
            model_ranges = [(r['address'], r['address']+r['bytes']) for r in self.model_table]
            require(all(a+s <= lo or hi <= a for a,s,_ in self.state.table.values()
                        for lo,hi in model_ranges), 'model/Session physical range overlap')
        seq = self.cursor
        b = self.event('boundary')
        fields(b, ('session','attempt','label','cache','owners','guard_array','published_rows',
                   'diagnostic_rows','tokens','model_table'), 'boundary')
        require(b['attempt'] == self.current_attempt and b['label'] == label and
                b['guard_array'] == d['seq'] and b['model_table'] == 'model_before', 'boundary binding')
        meta = b['cache']
        fields(meta, ('offset','generation','epoch','identity','poisoned','status','command'), 'cache metadata')
        for key in ('offset','generation','epoch','identity'):
            integer(meta[key], 0, 2**64-1, 'cache scalar')
        require(type(meta['poisoned']) is bool, 'cache poison flag')
        observed = self.state.boundary(b)
        if initial:
            require(meta['identity'] not in self.session_identities and meta['identity'] > 0 and
                    meta['offset'] == meta['generation'] == meta['epoch'] == 0 and
                    not meta['poisoned'] and meta['command'] is None and
                    meta['status']['words'][0] == 0 and b['published_rows'] == b['diagnostic_rows'] == 0,
                    'fresh Session authority/state')
            self.session_identities.add(meta['identity'])
        elif reset:
            require(meta['identity'] == before['cache']['identity'] and
                    meta['generation'] == before['cache']['generation'] + 1 and
                    meta['epoch'] == before['cache']['epoch'] + 1 and meta['command'] is None and
                    not meta['poisoned'] and meta['status']['words'][0] == 0 and
                    b['published_rows'] == b['diagnostic_rows'] == 0, 'atomic reset authority/logit invalidation')
            old = {r[0]: r for r in before['owners']}
            for r in b['owners']:
                require(r[1] == old[r[0]][1], 'reset changed physical roles')
                if not (24 <= r[0] < 44 or r[0] == 63):
                    require(r == old[r[0]], 'reset mutated KV/scratch/logits/snapshots')
        elif success is not None:
            fault = success.fault
            require(meta['identity'] == before['cache']['identity'] and
                    meta['epoch'] == before['cache']['epoch'] + 1 and
                    meta['generation'] == before['cache']['generation'] + int(fault is None),
                    'attempt generation/epoch authority')
            require(meta['poisoned'] == (fault == 'FirstCopy') and
                    meta['status']['words'][0] == int(fault in ('Head','OmitHistory9')), 'failure status/poison')
            command = meta['command']
            if fault in ('Head','OmitHistory9'):
                require(command is None, 'failed Phase A command discarded after terminal')
            else:
                require(command == dict(status='MTLCommandBufferStatus(4)',error=None,
                                        terminal=True,completed=True), 'completed actual Phase B')
            require(b['diagnostic_rows'] == (len(success.tokens) if fault is None else 0),
                    'diagnostic publication validity')
            require(b['published_rows'] == (len(success.tokens) if self.state.diagnostics else 1)
                    if fault is None else b['published_rows'] == before['published_rows'], 'published row count')
            old = {r[0]: r for r in before['owners']}
            for r in b['owners']:
                id_ = r[0]
                if 24 <= id_ < 44:
                    previous = id_ ^ 1 if fault is None else id_
                elif id_ in (61,62):
                    previous = 123-id_ if fault is None else id_
                else:
                    previous = id_
                require(r[1] == old[previous][1], 'incorrect post-copy bank/logit role publication')
            if fault:
                require(observed[62] == (old[62][1],old[62][2]), 'failed published complete logit owner changed')
        self.boundary_seq = seq
        self.guard_only_snapshot_owners += 293 if self.state.diagnostics else 0
        return b

    def new_session(self, id_, diagnostics):
        self.current_attempt = 0
        b = self.event('session_begin')
        fields(b, ('session','options','memory','capacity','chunk','query_heads','kv_heads','dim',
                   'histories','cache_owners','diagnostic_slots'), 'forward Session begin')
        require(b['session'] == id_ and b['options'] == dict(diagnostics=diagnostics,all_logits=diagnostics) and
                [b[k] for k in ('capacity','chunk','query_heads','kv_heads','dim','histories','cache_owners','diagnostic_slots')] ==
                [512,32,16,8,64,10,6,293 if diagnostics else 0], 'selected-real Session geometry')
        m = memory_ledger(b['memory'], self.model_memory, diagnostics, diagnostics)
        if diagnostics and 1 in self.session_memories:
            require(m == self.session_memories[1], 'repeat Session memory admission changed')
        self.session_memories[id_] = m
        self.state = PhysicalState(self.capture,id_,diagnostics,diagnostics,
                                   (r['address'] for r in self.model_table))
        for i in range(24):
            self.array(6,i,attempt=0,shape=(self.state.sizes[i]//2,))
        self.state.initial_kv()
        self.history_stack()
        self.boundary('initial',initial=True)

    def reset(self):
        before = self.boundary_seq
        self.exact_event('reset_result',dict(session=self.state.session,attempt=self.current_attempt,
                                             run=self.run,result='Ok(())'))
        self.state.reset(); self.history_stack(); self.boundary('reset',reset=True)
        b = self.event('reset')
        fields(b, ('run','session','before_boundary','after_boundary','logits_error','old_KV_unchanged'), 'reset result')
        require(b['run'] == self.run and b['session'] == self.state.session and b['before_boundary'] == before and
                b['after_boundary'] == self.boundary_seq and b['old_KV_unchanged'] is True, 'reset evidence references')
        error_string(b['logits_error'])

    def reject(self, name):
        before = self.state.last; seq = self.boundary_seq
        a = self.event('api_reject_result')
        fields(a,('session','attempt','kind','error'),'API rejection result')
        require(a['session'] == self.state.session and a['attempt'] == self.current_attempt and a['kind'] == name,
                'fixed API rejection')
        error_string(a['error'])
        b = self.boundary('preencode_reject')
        require(all(b[k] == before[k] for k in ('cache','owners','published_rows','diagnostic_rows','tokens')),
                'preencode rejection changed state/command/physical bytes')
        self.exact_event('api_reject',dict(session=self.state.session,attempt=self.current_attempt,kind=name,
                         error=a['error'],before_boundary=seq,after_boundary=self.boundary_seq,command_unchanged=True))

    def histories(self, kind):
        for i in range(20):
            self.array(kind,i,shape=(2055,))

    def stage_arrays(self, chunk):
        """Exact slot-major producer then selected-row order, no omissions."""
        for slot, stage in enumerate(STAGES):
            member = stage.name.rsplit('.',1)[-1]
            kind = {'u':1,'k_rotary':2,'v_projection':3}.get(member)
            if kind is not None:
                width = 1024 if kind == 1 else 512
                producer = self.array(kind,stage.layer,shape=(len(chunk.tokens),width),query=(chunk.start,len(chunk.tokens)))
                require(np.isfinite(self.capture.words(producer).view('<f2')).all(),
                        'all fixed faults occur after finite full actual producers')
            if chunk.fault is None:
                for row in chunk.selected_queries:
                    attention = member in ('attention_scores','attention_probabilities')
                    self.array(0,slot,shape=(16,1,chunk.end) if attention else (stage.width,),dtype=stage.dtype,
                               query=(row,1),keys=(0,chunk.end) if attention else (0,0))
                    self.selected_groups += 1
        self.producer_rows += len(chunk.tokens)

    def compare_state_prefix(self):
        prefix = tuple(int(v) for v in self.state.tokens[:self.state.offset])
        kv = [hash_words(self.state.kv[4*o+r][:-7].reshape(8,512,64)[:,:self.state.offset].copy())
              for o in range(6) for r in range(2)]
        history = [hash_words(self.state.history[2*o]) for o in range(10)]
        current = (kv,history)
        if prefix in self.state_baseline:
            require(current == self.state_baseline[prefix], 'same-native published history/active KV differs by partition/replay')
        else:
            self.state_baseline[prefix] = current

    def row_hash(self, chunk, row):
        b = self.event('row_hash')
        fields(b,('session','attempt','run','row','raw_word_sha256','numeric_sha256'),'native head row hash')
        require(b['session'] == chunk.session and b['attempt'] == self.current_attempt and
                b['run'] == chunk.run and b['row'] == row, 'head row identity')
        sha_string(b['raw_word_sha256']); sha_string(b['numeric_sha256'])
        self.row_records[(chunk.session,self.current_attempt,row)] = b
        prefix = tuple(int(v) for v in self.state.tokens[:row+1])
        if prefix in self.head_baseline:
            require(b['numeric_sha256'] == self.head_baseline[prefix], 'same-native causal/head replay hash')
        else:
            self.head_baseline[prefix] = b['numeric_sha256']
        self.row_hashes += 1
        if chunk.run < 8:
            self.primary_rows += 1
        if row in chunk.selected_queries:
            d = self.capture.get(chunk.session,self.current_attempt,0,292,(row,1))
            self.check_row_words(self.capture.words(d), b)

    @staticmethod
    def check_row_words(words, b):
        require(hash_words(words) == b['raw_word_sha256'] and numeric_head_hash(words) == b['numeric_sha256'],
                'complete captured distribution differs from raw/numeric row hash')

    def attempt(self, chunk, measured=False):
        self.attempts += 1; self.current_attempt = self.attempts
        self.rows += len(chunk.tokens)
        require(self.attempts <= 104 and self.rows <= 367, 'bounded actual attempt workload')
        validate_begin(self.event('attempt_begin'),chunk,self.attempts,self.boundary_seq)
        self.histories(4)
        if measured:
            self.measured += 1
            self.exact_event('allocation',dict(session=chunk.session,attempt=self.current_attempt,
                call=self.measured,count=0,scope='Rust forward only; capture and framework allocations excluded'))
        validate_terminal(self.event('attempt_terminal'),chunk,self.current_attempt)
        if self.state.diagnostics:
            self.stage_arrays(chunk)
        self.histories(5)
        end = chunk.end if chunk.fault is None else chunk.start
        for i in range(12):
            self.array(7,i,shape=(8,end,64),keys=(0,end))
        if self.state.diagnostics:
            self.state.reconstruct(self.current_attempt,chunk.start,chunk.tokens,chunk.fault)
        else:
            self.state.ordinary_commit(self.current_attempt,chunk.start,chunk.tokens)
        self.state.active(self.current_attempt)
        self.boundary('committed' if chunk.fault is None else 'failed',success=chunk)
        if chunk.fault is None:
            for row in (range(chunk.start,chunk.end) if self.state.diagnostics else (chunk.end-1,)):
                self.row_hash(chunk,row)
            self.compare_state_prefix()
        if chunk.run < 8:
            self.primary_chunks += 1
            self.source_state_diagnostics(chunk)
            for row in chunk.selected_queries:
                self.selected(chunk,row)

    def source_state_diagnostics(self, chunk):
        """Frozen source state drift, separate from exact native publication.

        Histories compare only actual observed chunk-boundary old banks to the
        corresponding source per-token before/after row. Candidate/staging banks
        are reconstructed from actual writes, never equated across schedules.
        """
        sid, aid = chunk.session, self.current_attempt
        def compare(name, actual, selection):
            descriptor = self.source.groups[name]
            require(descriptor['route'] == 'state_producer_sideband' and
                    descriptor['gate_usage'] == 'diagnostic sideband', 'source sideband policy')
            for profile, totals in (('main',self.source_state_main),('wide',self.source_state_wide)):
                center = selection(self.source.values(name,profile))
                totals.add(actual,center,descriptor['gate'],f'{name}@run{chunk.run}:{chunk.start}:{chunk.end}')
            self.source_state_identities.add(name)
        for owner, layer in enumerate(CONVOLUTION):
            base=f'cases.{chunk.case}.layers.{layer}.'
            actual=self.capture.values(self.capture.get(sid,aid,1,layer,(chunk.start,len(chunk.tokens))))
            compare(base+'u_producer',actual,lambda x:x[chunk.start:chunk.end])
            for kind,name,row in ((4,'history_before',chunk.start),(5,'history_after',chunk.end-1)):
                words=self.capture.words(self.capture.get(sid,aid,kind,2*owner))[:2048]
                actual=words.view('<f2').astype('<f4').reshape(2,1024)
                compare(base+name,actual,lambda x,row=row:x[row])
        for owner, layer in enumerate(ATTENTION):
            for role,kind,name in ((0,2,'kv_keys_producer'),(1,3,'kv_values_producer')):
                group=f'cases.{chunk.case}.layers.{layer}.{name}'
                actual=self.capture.values(self.capture.get(sid,aid,kind,layer,
                                            (chunk.start,len(chunk.tokens)))).reshape(-1,8,64).transpose(1,0,2)
                compare(group,actual,lambda x:x[:,chunk.start:chunk.end,:])
                active=self.capture.values(self.capture.get(sid,aid,7,2*owner+role))
                compare(group,active,lambda x:x[:,:chunk.end,:])

    def selected(self, chunk, row):
        sid, aid = chunk.session,self.current_attempt
        def captured(slot):
            d = self.capture.get(sid,aid,0,slot,(row,1))
            return self.capture.values(d)
        def u(layer):
            return self.capture.values(self.capture.get(sid,aid,1,layer,(chunk.start,len(chunk.tokens))))
        def old(layer):
            owner = CONVOLUTION.index(layer)
            d = self.capture.get(sid,aid,4,2*owner)
            words = self.capture.words(d)[:2048]
            value = words.view('<f2').astype('<f4').reshape(2,1024)
            require(np.isfinite(value).all(), 'active pre-history finite')
            return value
        # Full actual producers and selected logical slots must agree before
        # either can be used as an independent operator input.
        for layer in CONVOLUTION:
            slot = next(i for i,s in enumerate(STAGES) if s.name == f'layers.{layer}.u')
            same_words(self.capture.words(self.capture.get(sid,aid,0,slot,(row,1))),
                       self.capture.words(self.capture.get(sid,aid,1,layer,(chunk.start,len(chunk.tokens))))[row-chunk.start],
                       'selected u versus full producer')
        for layer in ATTENTION:
            for kind,name in ((2,'k_rotary'),(3,'v_projection')):
                slot = next(i for i,s in enumerate(STAGES) if s.name == f'layers.{layer}.{name}')
                same_words(self.capture.words(self.capture.get(sid,aid,0,slot,(row,1))),
                           self.capture.words(self.capture.get(sid,aid,kind,layer,(chunk.start,len(chunk.tokens))))[row-chunk.start],
                           'selected KV versus full producer')
        self.dispatch.selected(chunk.case,row,chunk.tokens[row-chunk.start],chunk.start,chunk.end,captured,u,old)

    def head(self, index):
        row = self.state.offset-1
        d = self.array(8,index,shape=(65536,),dtype='f32le',query=(row,1))
        words = self.capture.words(d)
        self.check_row_words(words,self.row_records[(self.state.session,self.current_attempt,row)])
        return d,words

    def finish_session(self):
        self.exact_event('session_end',dict(session=self.state.session,offset=self.state.offset,poisoned=False))
        self.state = None

    def run_all(self):
        self.start(); self.new_session(1,True)
        for name in ('empty','token65536','chunk33'):
            self.reject(name)
        for run, chunks in enumerate(PRIMARY):
            self.run = run
            if run:
                self.reset()
            case_index = next(i for i,c in enumerate(CASES) if c[0] == chunks[0].case)
            self.exact_event('run_begin',dict(run=run,session=1,case_id=case_index,mode='primary',
                partition=[len(c.tokens) for c in chunks],selected=list(chunks[0].selected)))
            for chunk in chunks:
                self.attempt(chunk)
        self.run=8; self.reset()
        self.exact_event('run_begin',dict(run=8,session=1,mode='teacher',partition=[8,1,1,1],selected=[]))
        for i,chunk in enumerate(TEACHER):
            self.attempt(chunk); d,_ = self.head(i)
            self.dispatch.teacher(i,self.capture.values(d))
        for repeat in range(2):
            self.run=9+repeat; self.reset()
            self.exact_event('run_begin',dict(run=self.run,session=1,mode='free',repeat=repeat,
                                             partition=[8,1,1,1],selected=[]))
            free = FreeRepeat(repeat)
            for sample in range(4):
                self.attempt(free.next_chunk())
                d,words = self.head(8+4*repeat+sample)
                free.sample(words,self.event('sample'),d['seq'])
                if free.stopped or free.codec_failure:
                    break
            free.finish(); self.free.append(free)
        self.exact_event('assertions',dict(name='free_repeat_ids',passed=True))
        compare_free_repeats(*self.free)
        self.finish_session()
        for chunks in FAULTS:
            self.run=chunks[0].run; self.new_session(chunks[0].session,True)
            fault=chunks[1].fault; debug=FAULT_DEBUG[fault]
            self.exact_event('run_begin',dict(run=self.run,session=chunks[0].session,mode='fault',fault=debug,
                                             partition=[3,2,3,5],selected=[]))
            self.attempt(chunks[0])
            before=self.state.last
            d=self.array(10,0,shape=(32*65536+7,),dtype='f32le')
            old_hash=next(r[2] for r in before['owners'] if r[0]==62)
            require(d['sha256']==old_hash,'full actual published-before-fault owner')
            require(np.all(self.capture.words(d)[-7:] == WORD_PAD), 'actual prior-published guard tail')
            self.guard_bytes += 28
            require(self.guard_bytes <= 1024**2, 'cumulative actual guard reserve')
            self.attempt(chunks[1])
            if fault=='Head':
                b=self.event('fault_word')
                expected=dict(session=self.state.session,attempt=self.current_attempt,owner=61,index=0,
                              bits=0xffffffff,sha256=next(r[2] for r in self.state.last['owners'] if r[0]==61))
                require(b==expected,'actual injected candidate word/hash')
            self.exact_event('assertions',dict(name='fault_publication',fault=debug,failed=True,preserved=True,
                                              poisoned=fault=='FirstCopy',firstKchanged=fault=='FirstCopy'))
            if fault=='FirstCopy':
                self.exact_event('assertions',dict(name='poison_getters',refused=True))
                self.reject('poisoned')
            self.reset()
            for chunk in chunks[2:]:
                self.attempt(chunk)
            self.finish_session()
        self.run=14; self.new_session(5,False)
        self.exact_event('run_begin',dict(run=14,session=5,mode='ordinary',partition=[32,32],selected=[]))
        for i,chunk in enumerate(ORDINARY):
            self.attempt(chunk,True); self.head(16+i)
        self.finish_session()
        self.budgets()
        self.exact_event('assertions',dict(name='full_capture_inventory',attempts=self.attempts,rows=self.rows,
            primary_chunks=78,primary_rows=231,selected_groups=6739,producer_rows=self.producer_rows,
            history_stacks=18,measured=2,same_native_failures=0,codec_failures=0))
        for expected in self.model_table:
            self.exact_event('model_after',expected)
        self.exact_event('assertions',dict(name='immutable_model_before_after',passed=True,sampled_at_each_boundary=False))
        b=self.event('end')
        require(self.cursor==len(self.capture.records),'extra events after completed end')
        require({k:v for k,v in b.items() if k!='prefix'}==dict(attempts=self.attempts,attempted_rows=self.rows,
                primary_chunks=self.primary_chunks,primary_rows=self.primary_rows,selected_groups=self.selected_groups,
                producer_rows=self.producer_rows,guard_bytes=self.guard_bytes),'actual final inventory')
        require(self.primary_chunks==78 and self.primary_rows==231 and self.selected_groups==6739 and
                self.history_stacks==18 and self.measured==2 and self.guard_records<=126 and
                self.producer_rows<=303,'complete workload totals')
        footer(self.capture)
        expected_sidebands={name for name,d in self.source.groups.items()
                            if d['route']=='state_producer_sideband'}
        require(self.source_state_identities==expected_sidebands and len(expected_sidebands)==168,
                'all frozen source sideband identities covered')
        for totals in (self.source_state_main,self.source_state_wide):
            require((totals.groups,totals.values)==(4212,21657600), 'full source-state diagnostic coverage')
        return self.dispatch.finish()

    def budgets(self):
        for i,(diagnostics,all_logits,name) in enumerate(((False,False,'ordinary_last_row'),
                (False,True,'all_logits'),(True,True,'diagnostic_all_logits'))):
            b=self.event('session_begin')
            fields(b,('session','options','memory','budget_only'),'budget-only Session')
            require(b['session']==6+i and b['budget_only'] is True and
                    b['options']==dict(diagnostics=diagnostics,all_logits=all_logits),'budget Session mode')
            m=memory_ledger(b['memory'],self.model_memory,diagnostics,all_logits)
            expected=self.session_memories[1 if diagnostics else 5]['admitted']
            if all_logits and not diagnostics:
                expected+=2*31*65536*4
            require(m['admitted']==expected,'same-owner exact budget admission')
            self.exact_event('budget',dict(session=6+i,mode=name,limit=expected,required=expected,accepted=True,error=None))
            under=self.event('budget')
            require(under==dict(session=6+i,mode=name,limit=expected-1,required=expected,accepted=False,
                    error=f'Budget {{ required: {expected}, limit: {expected-1} }}'),'one-byte-under exact Budget error')

    def report(self):
        return dict(attempts=self.attempts,attempted_rows=self.rows,primary_chunks=self.primary_chunks,
                    primary_rows=self.primary_rows,selected_groups=self.selected_groups,producer_rows=self.producer_rows,
                    history_stacks=self.history_stacks,guard_records=self.guard_records,guard_bytes=self.guard_bytes,
                    row_hashes=self.row_hashes,model_original_owners=len(self.model_table),
                    session_memory=self.session_memories,zero_allocation_calls=self.measured,
                    free=[x.finish() for x in self.free],same_native_prefixes=len(self.head_baseline),
                    physical_prefixes=len(self.state_baseline),
                    source_state_trajectory_diagnostic={'main':self.source_state_main.report(),
                        'wide':self.source_state_wide.report(),'source_identities':len(self.source_state_identities),
                        'fatal':False,'history_scope':'actual before/after chunk old banks only; no invented per-token snapshots'},
                    scope={'model_hashes':'all335 before/after; boundary references immutable table',
                           'ordinary':'two last-row heads and published state equality; no ordinary diagnostic producer reconstruction',
                           'snapshot_hashes':'actual complete hashes/guards with selected logical readback; unselected snapshot words not saved',
                           'row_hashes':'all published rows; full raw heads only selected/teacher/free/ordinary records',
                           'real_trajectory_positions':[0,63],'allocated_capacity':512,
                           'original_unquantized_quality':False,'app':False,'phone':False,'performance':False})


def codec(tokenizer_file, fixture_file, repeats):
    """Pinned raw ByteLevel codec only; no chat renderer or model import."""
    require(tokenizer_file.sha256 == TOKENIZER_SHA and fixture_file.sha256 == CODEC_SHA,
            'qualified original tokenizer/codec fixture identities')
    require(importlib.metadata.version('tokenizers') == '0.22.2', 'qualified codec runtime version')
    from tokenizers import Tokenizer
    # Consume the already-held, authenticated descriptor, not a second pathname.
    raw = tokenizer_file.read(0,tokenizer_file.bytes)
    require(digest(raw)==TOKENIZER_SHA,'consumed tokenizer identity')
    tokenizer = Tokenizer.from_str(raw.decode('utf-8'))
    require(tokenizer.get_vocab_size(with_added_tokens=True)==64400,'exact mapped vocabulary')
    fixture = fixture_file.json(4*1024**2)
    require(len(fixture['cases'])==10,'qualified codec case count')
    for case in fixture['cases']:
        ids=case['ids']
        require(all(type(x) is int and 0<=x<64400 for x in ids),'mapped frozen codec IDs')
        require(tokenizer.encode(case['rendered'],add_special_tokens=False).ids==ids and
                tokenizer.encode(case['rendered'],add_special_tokens=True).ids==case['postprocessor_ids'] and
                tokenizer.decode(ids,skip_special_tokens=False)==case['decoded'],
                'qualified manual bytes/IDs/raw decode fixture changed')
    result=[]
    for repeat in repeats:
        require(repeat.codec_failure is None and all(0<=x<64400 for x in repeat.sampled),
                'unmapped sampled token cannot be decoded or hidden')
        sampled=repeat.sampled
        full=list(PLAIN)+sampled
        result.append(dict(sampled_ids=list(sampled),raw_sampled_decode=tokenizer.decode(sampled,skip_special_tokens=False),
                           raw_prompt_and_sampled_decode=tokenizer.decode(full,skip_special_tokens=False),
                           stop_before_feed=repeat.stopped,pending=None if repeat.stopped else sampled[-1]))
    require(result[0]==result[1],'raw decoded repeat difference')
    return dict(version='0.22.2',cleanup=False,skip_special_tokens=False,bos=1,stop_ids=[7],
                mapped_ids=[0,64399],unmapped_ids=[64400,65535],fixture_cases=10,repeats=result,
                chat_quality_qualified=False)


INPUT_CAPS = dict(capture_control=65536,owner_manifest=16*1024**2,binary=256*1024**2,
                  capture_receipt=16*1024**2,stdout=16*1024**2,stderr=16*1024**2,
                  raw=256*1024**2,journal=16*1024**2,model=199527820,
                  source_manifest=4*1024**2,source_arrays=48049152,gates=2*1024**2,
                  protocol=256*1024,tokenizer=8*1024**2,codec_fixture=4*1024**2)
NATIVE_FILTER='metal::lfm2::real_gpu_tests::selected_real_lfm2_capture_preregistered_workload'


def open_inputs(control):
    fields(control,('format','files'),'reader invocation control')
    require(control['format']=='toklet-lfm2-capture-reader-control-v1' and
            set(control['files'])==set(INPUT_CAPS),'complete immutable reader inputs')
    opened={}
    try:
        for name,cap in INPUT_CAPS.items():
            pin=control['files'][name]
            fields(pin,('path','bytes','sha256'),'input identity')
            require(type(pin['path']) is str and os.path.isabs(pin['path']),'absolute pinned input path')
            integer(pin['bytes'],0 if name=='stderr' else 1,cap,'input cap');sha_string(pin['sha256'])
            opened[name]=File(pin['path'],cap,{k:pin[k] for k in ('bytes','sha256')},allow_empty=name=='stderr')
        require(len({f.identity[:2] for f in opened.values()})==len(opened),'input file aliases')
        return opened
    except BaseException:
        for f in opened.values():
            f.close()
        raise


def authenticate_execution(inputs):
    receipt=inputs['capture_receipt'].json(16*1024**2)
    require(receipt.get('passed') is True and receipt.get('error') is None and
            receipt.get('endpoint_errors')==[] and len(receipt.get('commands',[]))==1,
            'one completed successful native owner receipt')
    command=receipt['commands'][0]
    require(command.get('command')==[inputs['binary'].path,'--exact',NATIVE_FILTER,'--ignored',
                                    '--nocapture','--test-threads=1'] and
            command.get('exit_code')==0 and command.get('primary_error') is None and
            command.get('cleanup_errors')==[] and command.get('process_group_closed') is True,
            'literal native command/terminal/group closure')
    integer(command['pid'],1,2**31-1,'owned native PID')
    integer(command['live_samples'],1,100000,'observed direct resource samples')
    observed=command['observed_maxima']
    for key in ('resident_size','phys_footprint','lifetime_max_phys_footprint'):
        integer(observed[key],1,PROCESS_CAP,'bounded direct native memory observation')
    binary=receipt['binary']
    require(all(binary.get(k)==v for k,v in dict(path=inputs['binary'].path,bytes=inputs['binary'].bytes,
                                                sha256=inputs['binary'].sha256).items()),'executed native binary identity')
    require(binary.get('profile',{}).get('opt_level')=='3' and
            binary['profile'].get('debug_assertions') is False and binary['profile'].get('test') is True,
            'optimized release test artifact proof')
    for name in ('raw','journal','stdout','stderr'):
        f=inputs[name]
        require(receipt['outputs'].get(f.path)==dict(bytes=f.bytes,sha256=f.sha256),
                'complete actual native output receipt binding')
    native=inputs['capture_control'].json(65536)
    fields(native,('format','artifact_directory','output_directory','binary','owner_manifest',
                   'protocol_sha256','inventory_sha256','gates_sha256','source_manifest_sha256',
                   'source_arrays_sha256','authorization'),'native consumed control')
    require(native['format']=='toklet-lfm2-real-capture-control-v1' and
            native['protocol_sha256']==PROTOCOL_SHA and native['inventory_sha256']==INVENTORY_SHA and
            native['gates_sha256']==GATES_SHA and native['source_manifest_sha256']==SOURCE_MANIFEST_SHA and
            native['source_arrays_sha256']==SOURCE_SHA,'native fixed source authority')
    for name in ('binary','owner_manifest'):
        f=inputs[name]
        require(native[name]==dict(path=f.path,bytes=f.bytes,sha256=f.sha256),'native authority pin '+name)
    require(type(native['authorization']) is dict and native['authorization'], 'explicit native launch authorization')
    require(os.path.abspath(native['artifact_directory'])==os.path.dirname(inputs['model'].path) and
            os.path.abspath(native['output_directory'])==os.path.dirname(inputs['raw'].path) and
            os.path.dirname(inputs['raw'].path)==os.path.dirname(inputs['journal'].path), 'native artifact/output locations')
    # The actual owner must have guarded the consumed control as well as binary.
    f=inputs['capture_control']
    require(receipt['bindings'].get(f.path)==dict(bytes=f.bytes,sha256=f.sha256), 'owner consumed control guard')
    return dict(capture_control=native,capture_control_sha256=f.sha256),receipt


def finalize(report, errors):
    """Eligibility is derived once, only after every input endpoint guard."""
    report['endpoint_errors']=list(errors)
    report['passed']=bool(report.get('structural_complete') is True and report.get('codec_passed') is True and
                          report.get('numerical',{}).get('fatal_numerical_passed') is True and
                          report.get('error') is None and not errors)
    report['eligible_for_bounded_native_conformance']=report['passed']
    report['numerical_qualification']=False  # Independent outer task decision.
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--control',required=True)
    parser.add_argument('--control-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=dict(format='toklet-lfm2-capture-reader-v1',passed=False,structural_complete=False,
                codec_passed=False,eligible_for_bounded_native_conformance=False,numerical_qualification=False,
                error=None,endpoint_errors=[],numeric_owned_reserve=MAX_NUMERIC,
                memory_scope={'native_readonly_map_cap':155483328,'source_array_bytes':48049152,
                              'source_arrays_mapped':False,'original_model_bytes':199527820,
                              'original_model_mapped':False,'metadata_runtime_separate':True},
                premises=['ordinary correctly rounded IEEE binary64 backend for exact-affine certificates',
                          'pinned Metal precise published intrinsic bounds, selected RNE stores',
                          'softmax result is a conservative per-value correlated-mass enclosure, not one hidden-vector proof'],
                scope={'original_unquantized_adoption_quality':False,'app':False,'phone':False,'performance':False})
    opened={}; native=source=weights=reader=control_file=None
    endpoint_errors=[]
    # An exclusive result descriptor exists even when admission/reading fails.
    with open(args.output,'x',encoding='utf-8') as output:
        try:
            require(np.__version__ == '2.3.2', 'qualified NumPy version')
            sha_string(args.control_sha256)
            control_file=File(args.control,65536)
            require(control_file.sha256==args.control_sha256,'reader consumed invocation identity')
            control=control_file.json(65536)
            opened=open_inputs(control)
            authority,receipt=authenticate_execution(opened)
            report['inputs']=control['files']
            report['native_observations']={'samples':receipt['commands'][0]['live_samples'],
                'direct_observed_maxima':receipt['commands'][0]['observed_maxima'],
                'scope':'sparse direct process observations; not complete system/framework peaks'}
            summary=summary_from_stdout(opened['stdout'])
            native=Capture(opened['raw'].path,opened['journal'].path,summary)
            protocol=verify_protocol(opened['protocol'].path)
            source=Source(opened['source_manifest'].path,opened['source_arrays'].path,opened['gates'].path)
            weights=Weights(opened['model'].path)
            reader=Reader(native,source,weights,protocol,authority)
            report['numerical']=reader.run_all()
            report['structural_complete']=True
            report['codec']=codec(opened['tokenizer'],opened['codec_fixture'],reader.free)
            report['codec_passed']=True
        except Exception as error:
            report['error']=f'{type(error).__name__}: {error}'
        finally:
            if reader is not None:
                report['capture']=reader.report()
                if 'numerical' not in report:
                    report['partial_numerical']={'strict_coupled':reader.dispatch.coupled.report(),
                        'actual_input':reader.dispatch.local.report(),
                        'trajectory_main_diagnostic':reader.dispatch.main.report(),
                        'trajectory_wide_diagnostic':reader.dispatch.wide.report()}
            guards=list(opened.items())
            if control_file is not None:
                guards.append(('reader_control',control_file))
            if native is not None:
                guards += [('held_native_raw',native.raw_file),('held_native_journal',native.journal_file)]
            if source is not None:
                guards += [('held_source_manifest',source.manifest_file),('held_source_arrays',source.input),
                           ('held_source_gates',source.gate_file)]
            if weights is not None:
                guards.append(('held_model',weights.input))
            for name,file in guards:
                try:
                    file.endpoint()
                except Exception as error:
                    endpoint_errors.append(f'{name}: {type(error).__name__}: {error}')
            finalize(report,endpoint_errors)
            raw=json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n'
            require(len(raw.encode())<=16*1024**2,'bounded reader result')
            output.write(raw);output.flush()
            # Local variables retain no numeric native views at this point.
            if native is not None:
                native.close()
            if source is not None:
                source.close()
            if weights is not None:
                weights.close()
            for file in opened.values():
                file.close()
            if control_file is not None:
                control_file.close()
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
