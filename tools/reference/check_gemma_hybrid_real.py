"""Replay original E2B hybrid arrays/ownership against immutable pre-target data.

Staged FP32 and mathematical results are separate. Historical mathematical
failures are counted, never waived by a passing staged comparison.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import struct
import sys
import check_gemma_hybrid_tiny as checked

require, same = checked.require, checked.same
TEST = 'model::gemma_packed::tests::hybrid::real::original_e2b_hybrid_graph_preserves_frozen_states_and_resident_owners'
PINS = {
    'observation.json': 'ca86bd9836c88d5084f770b9ae84ecdb6e1e5a91d775e7cda315702450e918e9',
    'arrays.bin': '352385699238b422b430b6f535c2eac99b6e90543becb4c46caa0c59f930ed93',
}
INVENTORY = 'a3dff3b9d825fcf4fa6f391262732c5f429345682d683d6a39bc551ddacd68e2'
SCHEDULES = [[4], [1, 1, 1, 1], [1, 3]]


def frozen(directory, inventory_path):
    raw = {name: checked.read(directory/name, 128*1024**2) for name in PINS}
    for name, pin in PINS.items():
        require(hashlib.sha256(raw[name]).hexdigest() == pin, 'frozen reference identity: '+name)
    data = checked.document(raw['observation.json'])
    same(data['tokens'], [2, 100, 257, 100])
    same(data['gate'], dict(atol=1e-5, rtol=1e-4))
    specs = {}
    for spec in data['arrays']:
        key = (spec['contract'], spec['name'])
        require(key not in specs and spec['completed'] is True, 'complete unique reference arrays')
        specs[key] = spec
    require(len(specs) == 1426, 'complete original graph reference')
    body = checked.read(inventory_path, 1024**2)
    require(hashlib.sha256(body).hexdigest() == INVENTORY, 'frozen storage inventory identity')
    return specs, raw['arrays.bin'], checked.document(body)


def records(raw):
    require(len(raw) <= 8*1024**2, 'bounded real capture transcript')
    events, wrapper = [], []
    for line in raw.decode().splitlines():
        require(len(line.encode()) <= 4096, 'bounded capture record')
        if 'GEMMA_HYBRID_' in line:
            before, rest = line.split('GEMMA_HYBRID_', 1)
            require(before in ('', 'test '+TEST+' ... '), 'exact test prefix')
            label, body = rest.split(' ', 1)
            events.append((label, checked.document(body)))
        elif line.strip():
            wrapper.append(line)
    require(len(wrapper) == 3 and wrapper[:2] == ['running 1 test', 'ok'] and
            re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s', wrapper[2]),
            'complete exact successful real wrapper')
    return events


def signatures(schedule):
    offset = 0
    for count in schedule:
        for layer in range(35):
            for stage in ['output', 'self_attn.o_proj', 'post_per_layer_input_norm',
                          'per_layer_projection', 'per_layer_input_gate', 'mlp.up_proj',
                          'self_attn.q_norm', 'q_rotary', 'context']:
                yield f'layer.{layer}.{stage}', offset, count
        for name in ['logits', 'final_norm', 'ple_lookup', 'ple_projection']:
            yield name, offset, count
        for owner in range(15):
            for stage in ['k_rotary', 'self_attn.v_norm']:
                yield f'layer.{owner}.{stage}', offset, count
            for slot in range(offset+count):
                for stage in ['k_rotary', 'self_attn.v_norm']:
                    yield f'layer.{owner}.{stage}', slot, 1
        offset += count
    require(offset == 4, 'four-token schedule')
    yield 'logits', 0, 4


def storage_specs(inventory):
    for index, matrix in enumerate(inventory['matrices']):
        for chunk, value in enumerate(matrix['chunks']):
            for role in ['codes', 'scales', 'biases']:
                spec = value['buffers'][role]
                yield dict(kind='matrix', index=index, chunk=chunk, role=role), spec
    for index, value in enumerate(inventory['vectors']):
        yield dict(kind='vector', index=index), value


def selected(spec, binary, before, count):
    shape = spec['shape']
    axis = 2 if spec['name'].endswith(('rotary', 'context')) else 1
    require(0 <= before < before+count <= shape[axis], 'observation position bound')
    stride = math.prod(shape[axis+1:])
    block = memoryview(binary)[spec['offset']:spec['offset']+spec['bytes']]
    code, size = ('d', 8) if spec['dtype'] == '<f8' else ('f', 4)
    require(len(block) == math.prod(shape)*size and hashlib.sha256(block).hexdigest() == spec['sha256'], 'complete frozen array')
    return (v for index, (v,) in enumerate(struct.iter_unpack('<'+code, block))
            if before <= index//stride % shape[axis] < before+count), math.prod(shape)//shape[axis]*count


def replay(events, raw, specs, binary, inventory):
    iterator = iter(events)
    def take(label):
        actual, value = next(iterator)
        require(actual == label, 'record order: expected '+label+', got '+actual)
        return value
    owners = []
    expected_storage = list(storage_specs(inventory))
    require(len(expected_storage) == 1186, 'complete original GPU inventory')
    def buffers(phase, baseline=None):
        actual = []
        for key, spec in expected_storage:
            record = take('REAL_BUFFER')
            require(type(record) is dict and set(record) == {'phase', 'value'}, 'buffer record fields')
            same(record['phase'], phase)
            value = record['value']; require(type(value) is dict and set(value) == set(key)|{'buffer'}, 'buffer role fields')
            same({k: v for k, v in value.items() if k != 'buffer'}, key)
            b = value['buffer']; require(type(b) is dict and set(b) == {'bytes', 'sha256', 'allocation'}, 'buffer fields')
            pointer = b['allocation']
            require(type(pointer) is str and re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}', pointer), 'live allocation identity')
            same({k: v for k, v in b.items() if k != 'allocation'}, dict(bytes=spec['guarded_bytes'], sha256=spec['guarded_sha256']))
            actual.append(value)
        require(len({v['buffer']['allocation'] for v in actual}) == 1186, 'unique live original buffers')
        if baseline is not None: same(actual, baseline)
        return actual
    owners = buffers('loaded')
    copied = 0
    for index, spec in enumerate(inventory['vectors']):
        value = take('REAL_CPU_VECTOR')
        require(type(value) is dict and set(value) == {'index', 'bytes', 'capacity_bytes', 'sha256'}, 'CPU vector fields')
        same(value['index'], index); same(value['bytes'], spec['fp32_bytes']); same(value['sha256'], spec['fp32_sha256'])
        require(type(value['capacity_bytes']) is int and value['capacity_bytes'] >= value['bytes'], 'actual CPU vector capacity')
        copied += value['capacity_bytes']
    buffers('copied', owners)
    admission = take('REAL_ADMISSION')
    same(set(admission), {'model_bytes', 'norm_copy_bytes', 'raw_admitted_bytes', 'diagnostic_policy_bytes', 'process_policy_bytes', 'numerical_budget',
                          'context_dropped', 'same_device_fresh_context', 'exact_admission', 'minus_one_rejected', 'positions', 'input_tokens'})
    for key, expected in [('diagnostic_policy_bytes', 128*1024**2), ('process_policy_bytes', 6*1024**3), ('numerical_budget', 6*1024**3-128*1024**2),
                          ('positions', 32), ('input_tokens', [2, 100, 257, 100]), ('context_dropped', True), ('same_device_fresh_context', True),
                          ('exact_admission', True), ('minus_one_rejected', True)]: same(admission[key], expected)
    require(type(admission['norm_copy_bytes']) is int and admission['norm_copy_bytes'] >= copied == 1143948, 'complete actual CPU vector ownership')
    require(type(admission['model_bytes']) is int and type(admission['raw_admitted_bytes']) is int and
            2604665212+admission['norm_copy_bytes'] <= admission['model_bytes'] < admission['raw_admitted_bytes'] <= admission['numerical_budget'], 'combined real admission')
    offset = number = comparisons = 0
    summaries = []
    for schedule in SCHEDULES:
        same(take('REAL_SCHEDULE'), schedule)
        stats = {c: dict(compared_values=0, failed_values=0, maximum_gate_ratio=0., failed_records=[]) for c in ['math_fp64', 'staged_fp32']}
        for name, before, count in signatures(schedule):
            wanted, length = selected(specs['staged_fp32', name], binary, before, count)
            size = length*4
            header = dict(record=number, name=name, offset=offset, bytes=size, before=before, count=count, values=length)
            same(take('ARRAY_ATTEMPT'), header)
            value = take('ARRAY'); require(type(value) is dict and set(value) == set(header)|{'sha256'}, 'completed descriptor')
            same({k: v for k, v in value.items() if k != 'sha256'}, header)
            block = memoryview(raw)[offset:offset+size]
            require(len(block) == size and hashlib.sha256(block).hexdigest() == value['sha256'], 'complete native array')
            for contract, stat in stats.items():
                reference = wanted if contract == 'staged_fp32' else selected(specs[contract, name], binary, before, count)[0]
                failed = 0; worst = 0.; examples = []
                for index, ((a,), b) in enumerate(zip(struct.iter_unpack('<f', block), reference, strict=True)):
                    require(math.isfinite(a) and math.isfinite(b), 'finite complete arrays')
                    ratio = abs(a-b)/(1e-5+1e-4*abs(b)); worst = max(worst, ratio)
                    if ratio > 1.:
                        failed += 1
                        if len(examples) < 4: examples.append(dict(index=index, actual=a, reference=b, gate_ratio=ratio))
                stat['compared_values'] += length; stat['failed_values'] += failed
                stat['maximum_gate_ratio'] = max(stat['maximum_gate_ratio'], worst)
                if failed: stat['failed_records'].append(dict(record=number, name=name, before=before, count=count, failed_values=failed, maximum_gate_ratio=worst, examples=examples))
            comparisons += length*2; offset += size; number += 1
        summaries.append(dict(schedule=schedule, contracts=stats))
    buffers('executed', owners)
    same(take('CAPTURE'), dict(records=number, bytes=offset, sha256=hashlib.sha256(raw).hexdigest(), whole_e2b_qualified=False))
    same(take('REAL_COMPLETE'), dict(captured=True, positions=32, tested_input_positions=4, source_path_and_descriptor_unchanged=True,
        resident_buffers_unchanged=True, full_model_window_qualified=False, performance_qualified=False, phone_qualified=False, original_precision_failures_waived=False))
    require(next(iterator, None) is None and len(raw) == offset, 'no extra events or bytes')
    staged = all(row['contracts']['staged_fp32']['failed_values'] == 0 for row in summaries)
    ideal = all(row['contracts']['math_fp64']['failed_values'] == 0 for row in summaries)
    return dict(passed=staged, staged_fp32_passed=staged, ideal_fp64_passed=ideal, records=number, binary_bytes=offset,
        compared_values=comparisons, schedules=summaries, original_unchanged_guarded_buffers=1186, copied_vector_payload_bytes=copied,
        admission=admission, whole_e2b_qualified=False, full_model_window_qualified=False, performance_qualified=False, phone_qualified=False,
        mathematical_failures_waived=False, state_and_memory_scope='Frozen native assertions and transcript relationships, not independent full KV/allocation replay.')


if __name__ == '__main__':
    transcript, arrays, reference, inventory = map(Path, sys.argv[1:])
    specs, binary, source = frozen(reference, inventory)
    result = replay(records(checked.read(transcript, 8*1024**2)), checked.read(arrays, 128*1024**2), specs, binary, source)
    print(json.dumps(result))
    sys.exit(0 if result['passed'] else 1)
