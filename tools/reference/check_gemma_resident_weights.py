"""Independent replay of immutable GPU storage against pre-target byte witnesses.

No model loading, native code or target fixture generation is imported. Complete
capture is mandatory; failed candidate GPU buffers remain explicitly uncaptured.
"""
from pathlib import Path
import hashlib
import json
import re
import struct
import sys
import check_gemma_partitioned as frozen

same = frozen.reference.same
S = 0x5a170031
CONTEXT = 64 * 1024**2
COLD = 64 * 1024**2
STACK = 128 * 1024
INVENTORY_SHA = 'a3dff3b9d825fcf4fa6f391262732c5f429345682d683d6a39bc551ddacd68e2'
TINY_TEST = 'metal::gemma_affine::resident::tests::gpu::immutable_bundle_loading_preserves_sources_failures_and_context_independence'
REAL_TEST = 'metal::gemma_affine::resident::tests::real::complete_original_e2b_storage_matches_independent_inventory_after_context_teardown'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def records(raw, kind):
    require(len(raw) <= 8 * 1024**2, 'bounded capture required')
    prefix = 'GEMMA_RESIDENT_' + kind.upper() + '_'
    test = TINY_TEST if kind == 'tiny' else REAL_TEST
    events, wrapper = [], []
    for line in raw.decode().splitlines():
        require(len(line.encode()) <= 1024**2, 'record bound')
        if prefix in line:
            before, rest = line.split(prefix, 1)
            require(before in ('', 'test ' + test + ' ... '), 'exact test prefix required')
            label, body = rest.split(' ', 1)
            value = frozen.reference.document(body)
            require(type(value) is dict, 'record object required')
            events.append((label, value))
        elif line.strip():
            wrapper.append(line)
    require(len(wrapper) == 3 and wrapper[:2] == ['running 1 test', 'ok'] and
            re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s', wrapper[2]),
            'complete exact successful wrapper required')
    return events


class Events:
    def __init__(self, events):
        self.events = iter(events)

    def take(self, label):
        name, value = next(self.events)
        require(name == label, 'record order differs: ' + label)
        return value

    def end(self):
        require(next(self.events, None) is None, 'extra event')


def pointer(value):
    require(type(value) is str and re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}', value), 'allocation identity required')
    return value


def inventory(directory):
    raw = frozen.reference.read(directory / 'inventory.json', 1024**2)
    require(hashlib.sha256(raw).hexdigest() == INVENTORY_SHA, 'pre-target inventory identity')
    return frozen.reference.document(raw)


def real(events, reference):
    e = Events(events)
    budget = e.take('BUDGET')
    same(set(budget), {'rejected', 'required', 'limit', 'source_bytes_read', 'error'})
    require(budget['rejected'] is True and budget['limit'] == budget['required'] - 1 and
            budget['source_bytes_read'] == 318228, 'budget rejection must precede payload IO')
    same(budget['error'], f"Metal admission requires {budget['required']} bytes, budget {budget['limit']}")
    same(e.take('LOAD'), dict(ok=True, error=None))
    report = e.take('REPORT')
    same(set(report), {'context_dropped', 'matrices', 'vectors', 'identity_components', 'resident_bytes',
        'policy_bytes', 'source_bytes', 'matrix_slots', 'vector_slots', 'bundle_abi', 'vector_abi', 'weights_abi',
        'process_policy_bytes', 'diagnostic_policy_bytes', 'available_bytes', 'config_digest', 'index_digest',
        'header_digest', 'container_bytes', 'container_identity'})
    require(report['context_dropped'] is True, 'construction Context teardown required')
    for name, expected in [('matrices', 278), ('vectors', 262), ('identity_components', 1096),
                           ('bundle_abi', 6728), ('vector_abi', 24), ('weights_abi', 112),
                           ('process_policy_bytes', 6 * 1024**3), ('diagnostic_policy_bytes', 32 * 1024**2)]:
        same(report[name], expected)
    same(report['container_identity'], reference['model']['sha256'])
    same(report['container_bytes'], reference['model']['bytes'])
    for key, name in [('config_digest', 'config.json'), ('index_digest', 'index.json'), ('header_digest', 'header.json')]:
        same(report[key], list(bytes.fromhex(reference['metadata'][name]['sha256'])))
    allocations = set()
    devices = set()
    resident = report['bundle_abi'] + report['matrix_slots'] * 112 + report['vector_slots'] * 24
    require(report['matrix_slots'] >= 278 and report['vector_slots'] >= 262, 'outer capacity')

    def buffer(actual, expected):
        allocation = pointer(actual['allocation'])
        require(allocation not in allocations, 'aliased live storage allocation')
        allocations.add(allocation)
        same({k: v for k, v in actual.items() if k != 'allocation'},
             dict(bytes=expected['guarded_bytes'], sha256=expected['guarded_sha256']))
        return actual['bytes']

    for number, expected in enumerate(reference['matrices']):
        actual = e.take('MATRIX')
        same(set(actual), {'index', 'rows', 'columns', 'owned_bytes', 'chunk_slots', 'chunk_abi', 'device', 'chunks'})
        same(actual['index'], number)
        same(actual['rows'], expected['rows'])
        same(actual['columns'], expected['columns'])
        require(actual['chunk_slots'] >= len(expected['chunks']), 'chunk slots')
        same(actual['chunk_abi'], 112)
        devices.add(pointer(actual['device']))
        require(len(actual['chunks']) == len(expected['chunks']), 'chunk count')
        owned = 112 + actual['chunk_slots'] * 112
        for chunk, wanted in zip(actual['chunks'], expected['chunks'], strict=True):
            same(chunk['first'], wanted['first'])
            same(chunk['rows'], wanted['rows'])
            require(set(chunk) == {'first', 'rows', 'codes', 'scales', 'biases'}, 'chunk fields')
            for role in ('codes', 'scales', 'biases'):
                owned += buffer(chunk[role], wanted['buffers'][role])
        same(actual['owned_bytes'], owned)
        resident += owned - 112
    for number, expected in enumerate(reference['vectors']):
        actual = e.take('VECTOR')
        same(set(actual), {'index', 'values', 'buffer'})
        same(actual['index'], number)
        same(actual['values'], expected['count'])
        resident += buffer(actual['buffer'], expected)
    require(len(allocations) == 1186 and len(devices) == 1, 'complete storage/device inventory')
    same(report['resident_bytes'], resident)
    same(report['source_bytes'], reference['accounting']['maximum_three_source_with_descriptors'])
    same(report['policy_bytes'], resident + report['source_bytes'] + CONTEXT + COLD + STACK)
    same(budget['required'], 2821589444)
    same(report['available_bytes'], report['process_policy_bytes'] - report['diagnostic_policy_bytes'])
    require(report['policy_bytes'] <= report['available_bytes'], 'aggregate admission')
    same(e.take('COMPLETE'), dict(passed=True, context_dropped=True, model_matrices=278,
        model_vectors=262, source_path_and_descriptor_unchanged=True, whole_graph_qualified=False,
        performance_qualified=False, phone_qualified=False, source_or_precision_failures_waived=False))
    e.end()
    return dict(passed=True, matrices=278, vectors=262, guarded_allocations=len(allocations),
                resident_requested_bytes=resident, admitted_load_policy_bytes=report['policy_bytes'],
                maximum_source_capacity_bytes=report['source_bytes'], context_dropped=True,
                whole_graph_qualified=False, performance_qualified=False, phone_qualified=False)


def tiny_expected(cases):
    matrices = []
    for case in cases:
        child = {'rows': case['rows']}
        for role in ('codes', 'scales', 'biases'):
            raw = case[role]
            child[role] = frozen.words(raw + bytes((-len(raw)) % 4)) + [S] * 7
        metadata = [[struct.unpack('<f', struct.pack('<I', code << 16))[0]
                     for (code,) in struct.iter_unpack('<H', case[role])] for role in ('scales', 'biases')]
        maximum = max(abs(s * code + b) for s, b in zip(*metadata, strict=True) for code in range(16))
        owned = 224 + sum(len(child[role]) * 4 for role in ('codes', 'scales', 'biases'))
        matrices.append(dict(rows=case['rows'], columns=case['columns'], owned_bytes=owned,
                             maximum_weight=float(maximum), chunks=[child]))
    vectors = [[0x80000000, 0, 0x3f800000, 0xbf000000], [0x3f800000]]
    resident = 6728 + 2 * 112 + 2 * 24 + sum(m['owned_bytes'] - 112 for m in matrices)
    resident += sum((len(v) + 7) * 4 for v in vectors)
    return matrices, vectors, resident


def tiny(events, cases):
    e = Events(events)
    matrices, vectors, resident = tiny_expected(cases)
    plan = e.take('PLAN')
    minimum = resident + CONTEXT + COLD + STACK + 1440
    for name, expected in [('bundle_struct_bytes', 6728), ('vector_struct_bytes', 24),
            ('weights_struct_bytes', 112), ('minimum_policy_bytes', minimum),
            ('base', 6728 + 224 + 48), ('payload', resident - 6728 - 224 - 48),
            ('source', 1440), ('cold_policy', COLD), ('context_policy', CONTEXT),
            ('hash_stack_policy', STACK), ('real_e2b_model', False), ('whole_graph_qualified', False),
            ('process_allowance_bytes', 512 * 1024**2), ('diagnostic_policy_bytes', 32 * 1024**2)]:
        same(plan[name], expected)
    same(plan['initial_available_bytes'], plan['process_allowance_bytes'] - plan['diagnostic_policy_bytes'] - plan['reader_backing_bytes'])
    require(minimum <= plan['initial_available_bytes'], 'initial admission')
    initial = e.take('INITIAL')
    same({k: v for k, v in initial.items() if k != 'storage'}, dict(source_context_dropped=True,
        identity_components=8, resident_bytes=resident, policy_bytes=minimum, source_bytes=1440))

    def storage(actual, previous=None):
        ids = set()
        def item(value, wanted):
            key = pointer(value['allocation'])
            require(key not in ids, 'tiny live alias')
            ids.add(key)
            same({k: v for k, v in value.items() if k != 'allocation'}, dict(words=len(wanted), data=wanted))
        same(actual['resident_bytes'], resident)
        same(actual['matrix_slots'], 2)
        same(actual['vector_slots'], 2)
        require(len(actual['matrices']) == 2 and len(actual['vectors']) == 2, 'complete tiny owners')
        for got, wanted in zip(actual['matrices'], matrices, strict=True):
            same({k: v for k, v in got.items() if k != 'chunks'}, {k: v for k, v in wanted.items() if k != 'chunks'})
            require(len(got['chunks']) == 1, 'tiny chunks')
            same(got['chunks'][0]['rows'], wanted['rows'])
            for role in ('codes', 'scales', 'biases'):
                item(got['chunks'][0][role], wanted['chunks'][0][role])
        for got, wanted in zip(actual['vectors'], vectors, strict=True):
            same(got['values'], len(wanted))
            item(got['buffer'], wanted + [S] * 7)
        if previous is not None:
            require(not ids.intersection(previous), 'replacement aliases previous owner')
        require(len(ids) == 8, 'tiny allocations')
        return ids
    old_ids = storage(initial['storage'])
    for number, case in enumerate(cases):
        same(e.take('PROJECTION'), dict(matrix=number, ok=True, bits=case['expected']))
    errors = {'changed': 'Gemma source matrix changed/incomplete between passes',
              'late-vector': 'Gemma resident vector nonfinite', 'late-codebook': 'Gemma affine metadata domain'}
    for mode in ['budget', 'changed', 'late-read', 'late-vector', 'late-codebook', 'valid-replacement']:
        actual = e.take('OUTCOME')
        same(actual['previous'], initial['storage'])
        outcome = actual['outcome']
        same(outcome['key'], mode)
        same(outcome['ok'], mode == 'valid-replacement')
        reader = e.take('READER')
        same(reader['key'], mode)
        state = reader['state']
        if mode == 'budget':
            same(state, dict(position=10, bytes_read=0, seeks=0, historical_eof_observed=False))
            same(outcome['error'], f'Metal admission requires {minimum} bytes, budget {minimum-1}')
        else:
            require(state['historical_eof_observed'] is True and state['bytes_read'] > 0 and state['seeks'] > 0, 'two-pass IO required')
        if mode == 'valid-replacement':
            same(outcome['resident_bytes'], resident)
            same(outcome['policy_bytes'], minimum)
            same(outcome['source_bytes'], 1440)
            same(outcome['matrix_slots'], 2)
            same(outcome['vector_slots'], 2)
            storage(outcome['candidate'], old_ids)
        else:
            same(outcome['failed_candidate_buffers_captured'], False)
            if mode in errors:
                same(outcome['error'], errors[mode])
            elif mode == 'late-read':
                require(outcome['error'].endswith('resident original late read failure'), 'original late IO error')
    same(e.take('COMPLETE'), dict(passed=True, steps=6, context_dropped=True, projection_values=90,
        failed_candidate_buffers_captured=False, real_e2b_model=False, whole_graph_qualified=False,
        source_or_precision_failures_waived=False, performance_qualified=False))
    e.end()
    return dict(passed=True, matrices=2, vectors=2, failures=5, replacements=1, projection_values=90,
                resident_requested_bytes=resident, failed_candidate_buffers_captured=False)


if __name__ == '__main__':
    kind, capture, directory = sys.argv[1:]
    events = records(frozen.reference.read(Path(capture), 8 * 1024**2), kind)
    result = tiny(events, frozen.fixture(Path(directory))) if kind == 'tiny' else real(events, inventory(Path(directory)))
    print(json.dumps(result))
