"""Independent replay of bounded resident GPU lookups over original packed bytes.

Reconstruct every device word, allocation epoch and CPU scratch transition. This
qualifies the tiny standalone primitive, never a full graph or large-table fit.
"""
from pathlib import Path
import copy
import re
import sys
import check_gemma_partitioned as base

reference = base.reference
same = reference.same
SENTINEL = base.SENTINEL
CALLER = base.CALLER
PREFIX = 'GEMMA_GATHER_'
TEST = 'metal::gemma_affine::partitioned::gather::tests::frozen_resident_table_gathers_share_weights_and_preserve_transactions'


def records(raw):
    if len(raw) > 32 * 1024**2:
        raise ValueError('bounded native capture required')
    result, wrapper = [], []
    for line in raw.decode('utf-8').splitlines():
        if len(line.encode()) > 4 * 1024**2:
            raise ValueError('bounded record required')
        if PREFIX in line:
            before, rest = line.split(PREFIX, 1)
            if before and before != 'test ' + TEST + ' ... ':
                raise ValueError('unexpected native record prefix')
            label, text = rest.split(' ', 1)
            value = reference.document(text)
            if type(value) is not dict:
                raise ValueError('record object required')
            result.append((label, value))
        elif line.strip():
            wrapper.append(line)
    if len(wrapper) != 3 or wrapper[0] != 'running 1 test' or wrapper[1] != 'ok' or not re.fullmatch(
            r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s', wrapper[2]):
        raise ValueError('complete successful exact-test wrapper required')
    return result


def matrix(case, maximum):
    n, k = case['rows'], case['columns']
    children, owned = [], 112 + 4 * (n + min(maximum, n))
    for first in range(0, n, maximum):
        rows = min(maximum, n - first)
        device = {}
        for name, stride in [('codes', k // 2), ('scales', k // 64 * 2), ('biases', k // 64 * 2)]:
            raw = case[name][first * stride:(first + rows) * stride]
            device[name] = base.words(raw + bytes((-len(raw)) % 4)) + [SENTINEL] * 7
        device.update(input=[0] * (k + 7), output=[0] * (rows + 7), status=[0] * 8)
        owned += 240 + sum(len(v) * 4 for v in device.values())
        children.append(dict(rows=rows, capacity=1, poisoned=False, pending=False, shared_read_safe=True, device=device))
    return dict(rows=n, columns=k, capacity=1, poisoned=False, admitted_bytes=owned, slots=len(children),
                full_scratch_capacity=n, chunk_scratch_capacity=min(maximum, n), staged_bits=[0] * n,
                chunk_bits=[0] * min(maximum, n), children=children)


def workspace(k, owner):
    values = 32 * k
    admitted = (32 + values + 1 + 21) * 4 + 2 * values * 4 + owner + 16 * 1024**2
    return dict(admitted_bytes=admitted, columns=k, capacity=32, poisoned=False, pending=False, shared_read_safe=True,
                full_scratch_capacity=values, chunk_scratch_capacity=values, staged_bits=[0] * values,
                chunk_bits=[0] * values, caller_bits=[], workspace=dict(ids=[0] * 39, output=[0] * (values + 7), status=[0] * 8))


class Replay:
    def __init__(self, captured):
        self.captured = iter(captured)
        self.steps = self.values = self.affines = 0
        self.live = set()
        self.max_owner = self.max_matrix = self.extra_matrices = 0

    def take(self, label, expected=None):
        try:
            got, body = next(self.captured)
        except StopIteration as error:
            raise ValueError('missing ' + label) from error
        if got != label:
            raise ValueError('record order differs: ' + got + '/' + label)
        if expected is not None:
            same(body, expected)
        return body

    def allocations(self, expected, actual):
        if type(actual) is not dict or actual.keys() != expected.keys():
            raise ValueError('complete allocation inventory required')
        for name, words in list(expected.items()):
            entry = actual[name]
            pointer = entry.get('allocation') if type(entry) is dict else None
            if type(pointer) is not str or not re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}', pointer) or pointer in self.live:
                raise ValueError('distinct live non-null allocation required')
            self.live.add(pointer)
            expected[name] = dict(allocation=pointer, words=words)

    def matrix_ids(self, expected, actual):
        if type(actual) is not dict or type(actual.get('children')) is not list or len(actual['children']) != len(expected['children']):
            raise ValueError('complete matrix inventory required')
        for child, observed in zip(expected['children'], actual['children'], strict=True):
            self.allocations(child['device'], observed.get('device'))
        same(actual, expected)
        self.max_matrix = max(self.max_matrix, expected['admitted_bytes'])

    def observe(self, state, table, caller):
        result = copy.deepcopy(state)
        result['matrix'] = copy.deepcopy(table)
        result['caller_bits'] = caller
        return result

    def affine(self, table, case, key, index):
        n, k = case['rows'], case['columns']
        inputs = case['input'][k:2*k]
        outputs = list(base.projection(tuple(case['weights']), k, tuple(inputs)))
        same(outputs, case['expected'][n:2*n])
        self.take('AFFINE_ATTEMPT', dict(key=key, index=index, input_bits=inputs, caller_len=n))
        first = 0
        for child in table['children']:
            count = child['rows']
            device = child['device']
            for name, active in [('input', inputs), ('output', outputs[first:first+count]), ('status', [0])]:
                if index == 0:
                    # Before first observation, only independently derived words
                    # exist. Opaque identities are learned after this evolution.
                    device[name] = active + [SENTINEL] * (len(device[name]) - len(active))
                else:
                    device[name]['words'] = active + [SENTINEL] * (len(device[name]['words']) - len(active))
            table['chunk_bits'][:count] = outputs[first:first+count]
            first += count
        table['staged_bits'][:] = outputs
        self.take('AFFINE_OUTCOME', dict(key=key, index=index, completed=True, error=None))
        body = self.take('AFFINE')
        if index == 0:
            # Learn only opaque pointers from actual bytes; expected values and
            # all owner sizes are derived before accepting the observed record.
            self.matrix_ids(table, body.get('matrix'))
        same(body, dict(key=key, index=index, input_bits=inputs, output_bits=outputs, matrix=table))
        self.affines += 1

    def construct(self, state, table, key, recreate=False):
        k, admitted = state['columns'], state['admitted_bytes']
        if not recreate:
            self.take('BUDGET', dict(key=key, required=admitted, limit=admitted-1, rejected=True,
                                    error=f'Metal admission requires {admitted} bytes, budget {admitted-1}'))
            self.take('CONSTRUCTOR_ATTEMPT', dict(key=key, columns=k, capacity=32, required=admitted))
        self.take('RECREATE_OUTCOME' if recreate else 'CONSTRUCTOR_OUTCOME', dict(key=key, completed=True, error=None))
        body = self.take('RECREATED' if recreate else 'CONSTRUCTOR')
        actual = body.get('observation')
        if type(actual) is not dict:
            raise ValueError('complete workspace required')
        self.allocations(state['workspace'], actual.get('workspace'))
        wanted = dict(key=key, observation=self.observe(state, table, []))
        if not recreate:
            width = body.get('simd_width')
            if type(width) is not int or not 1 <= width <= 65535:
                raise ValueError('actual execution width required')
            wanted['simd_width'] = width
        same(body, wanted)
        self.max_owner = max(self.max_owner, admitted)

    def step(self, state, table, case, key, phase, index, ids, length, error=None, completed_chunks=None, unknown_matrix=False):
        self.take('STEP_ATTEMPT', dict(key=key, phase=phase, index=index, ids=ids, caller_len=length))
        caller = [CALLER] * length
        if error is None or completed_chunks is not None:
            first = completed = 0
            for child in table['children']:
                rows = child['rows']
                slots = [i for i, row in enumerate(ids) if first <= row < first + rows]
                if slots:
                    local = [ids[i] - first for i in slots]
                    output = [v for i in slots for v in case['weights'][ids[i]*case['columns']:(ids[i]+1)*case['columns']]]
                    for name, active in [('ids', local), ('output', output), ('status', [0])]:
                        device = state['workspace'][name]
                        device['words'] = active + [SENTINEL] * (len(device['words']) - len(active))
                    state['chunk_bits'][:len(output)] = output
                    k = state['columns']
                    for offset, slot in enumerate(slots):
                        state['staged_bits'][slot*k:(slot+1)*k] = output[offset*k:(offset+1)*k]
                    completed += 1
                    if completed_chunks is not None and completed == completed_chunks:
                        break
                first += rows
            if error is None:
                caller = state['staged_bits'][:length]
            else:
                state['poisoned'] = True
        self.take('STEP_OUTCOME', dict(key=key, phase=phase, index=index, completed=error is None, error=error, poisoned=state['poisoned']))
        body = self.take('STEP')
        if unknown_matrix:
            self.matrix_ids(table, body.get('observation', {}).get('matrix'))
        view = self.observe(state, table, caller)
        if phase == 'matrix-reject' and index == 4:
            view['matrix']['children'][0]['device'] = None
        same(body, dict(key=key, phase=phase, index=index, completed=error is None, error=error, observation=view))
        self.steps += 1
        if error is None:
            self.values += length

    def run(self, cases):
        meta = self.take('META')
        device, owner = meta.get('device'), meta.get('owner_struct_bytes')
        if type(device) is not str or not 1 <= len(device) <= 256 or type(owner) is not int or owner != 160:
            raise ValueError('named hardware and bounded concrete owner required')
        same(meta, dict(device=device, context_allowance_bytes=64*1024**2, pipeline_policy_bytes=16*1024**2,
                        owner_struct_bytes=owner, routing_stack_bytes=384, whole_graph_qualified=False, actual_large_table_qualified=False))
        count = 0
        for case in cases:
            n, k = case['rows'], case['columns']
            canonical = [0, 6, 7, n-1, 7]
            for maximum in (7, 8):
                self.live = set()  # all prior case resources have been dropped
                key = f"{case['name']}/max{maximum}"
                self.take('MATRIX_ATTEMPT', dict(key=key, rows=n, columns=k, maximum=maximum, capacity=1, budget=1024*1024))
                self.take('MATRIX_OUTCOME', dict(key=key, completed=True, error=None))
                table = matrix(case, maximum)
                self.affine(table, case, key, 0)
                state = workspace(k, owner)
                self.construct(state, table, key)
                for repeat in range(2):
                    self.step(state, table, case, key, 'canonical', repeat, canonical, 5*k)
                for size in (1, 32):
                    ids = [canonical[(i*3+4) % 5] for i in range(size)]
                    self.step(state, table, case, key, 'boundary', size, ids, size*k)
                invalid = [([], 0), ([0]*33, 33*k), ([0, n], 2*k), ([0, 2**64-1], 2*k), ([0, 1], 2*k-1), ([0, 1], 2*k+1)]
                for index, (ids, length) in enumerate(invalid):
                    self.step(state, table, case, key, 'request-reject', index, ids, length, 'Gemma table gather request shape or indices')
                if count == 0:
                    foreign, wrong = matrix(case, maximum), matrix(cases[1], maximum)
                    self.extra_matrices = foreign['admitted_bytes'] + wrong['admitted_bytes']
                    mismatch = 'Gemma table gather matrix Context/shape/state mismatch'
                    for index, other in enumerate((foreign, wrong)):
                        self.step(state, other, case, key, 'matrix-reject', index, [0], k, mismatch, unknown_matrix=True)
                    table['poisoned'] = True
                    self.step(state, table, case, key, 'matrix-reject', 2, [0], k, mismatch)
                    table['poisoned'] = False
                    table['children'][0]['poisoned'] = True
                    self.step(state, table, case, key, 'matrix-reject', 3, [0], k, mismatch)
                    table['children'][0]['poisoned'] = False
                    table['children'][0]['pending'] = True
                    table['children'][0]['shared_read_safe'] = False
                    self.step(state, table, case, key, 'matrix-reject', 4, [0], k, mismatch)
                    table['children'][0]['pending'] = False
                    table['children'][0]['shared_read_safe'] = True
                    before = state['staged_bits'][:]
                    self.step(state, table, case, key, 'composition-failure', 0, [18, 7, 6, 0, 7], 5*k,
                              'Gemma table gather injected composition failure', completed_chunks=2)
                    if state['staged_bits'] == before:
                        raise ValueError('failure must retain changed completed stage')
                    self.step(state, table, case, key, 'poison-retry', 0, canonical, 5*k, 'Gemma table gather poisoned; recreate workspace')
                    # Old and new pipelines/workspaces are simultaneously live
                    # during construction: preserve all identities in this epoch.
                    state = workspace(k, owner)
                    self.construct(state, table, key, recreate=True)
                    self.step(state, table, case, key, 'recover', 0, canonical, 5*k)
                self.affine(table, case, key, 1)
                self.step(state, table, case, key, 'resume', 0, canonical, 5*k)
                self.take('CASE_END', dict(key=key, observation=self.observe(state, table, [])))
                count += 1
        self.take('FINAL', dict(cases=4, composition_failures=1, recreated_workspaces=1, affine_gather_alternations=4,
                               matrix_weights_shared=True, matrix_affine_capacity=1, gather_capacity=32,
                               driver_or_inflight_failure_injected=False, actual_large_table_qualified=False,
                               whole_graph_qualified=False, performance_qualified=False))
        try:
            next(self.captured)
        except StopIteration:
            pass
        else:
            raise ValueError('trailing native record')
        # Conservative simultaneous policy ledger: two contexts, three matrices,
        # two Gather owners/pipelines, largest caller and routing, and separate
        # fixed fixture/JSON/serialization policies. Policies are not residency.
        ledger = dict(contexts=2*64*1024**2, matrix=self.max_matrix, extra_matrices=self.extra_matrices,
                      overlapping_gather_owners=2*self.max_owner, caller=33*128*4, routing_stack=384,
                      fixture_and_host_policy=32*1024**2, capture_and_serialization_policy=128*1024**2)
        total = sum(ledger.values())
        if total > 1024**3:
            raise ValueError('aggregate tiny diagnostic policy exceeds native limit')
        same([count, self.steps, self.affines], [4, 52, 8])
        return dict(passed=True, device=device, cases=4, steps=self.steps, affine_calls=self.affines,
                    successful_output_values=self.values, complete_buffers=True, caller_transactional=True,
                    sticky_poison=True, original_weights_shared=True, aggregate_policy_bytes=total, workload_ledger=ledger,
                    driver_or_inflight_failure_injected=False, actual_large_table_qualified=False,
                    whole_graph_qualified=False, performance_qualified=False)


def replay(capture, directory):
    return Replay(records(reference.read(capture, 32*1024**2))).run(base.fixture(directory))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('native stdout and frozen table fixture directory')
    import json
    print(json.dumps(replay(Path(sys.argv[1]), Path(sys.argv[2])), allow_nan=False))
