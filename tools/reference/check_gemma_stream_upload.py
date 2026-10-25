"""Replay bounded source callbacks and complete uploaded primitive observations.

No engine or producer import. Original packed bytes and exact projection math
come from the independently frozen tiny witness, not a native prediction model.
"""
from pathlib import Path
import copy
import re
import sys
import check_gemma_partitioned as base

same = base.same
TEST = 'metal::gemma_affine::partitioned::upload::tests::gpu::frozen_stream_upload_reuses_source_storage_and_recovers_after_failed_construction'


def records(raw):
    if len(raw) > 32*1024**2:
        raise ValueError('bounded native capture required')
    result, wrapper = [], []
    for line in raw.decode('utf-8').splitlines():
        if len(line.encode()) > 4*1024**2:
            raise ValueError('bounded record required')
        found = False
        for prefix in ('GEMMA_STREAM_', 'GEMMA_PARTITION_'):
            if prefix not in line:
                continue
            before, rest = line.split(prefix, 1)
            if before and before != 'test ' + TEST + ' ... ':
                raise ValueError('unexpected native record prefix')
            label, text = rest.split(' ', 1)
            if prefix == 'GEMMA_PARTITION_' and label not in ('STEP_ATTEMPT', 'STEP_OUTCOME', 'STEP'):
                raise ValueError('only unchanged projection-step protocol allowed')
            body = base.reference.document(text)
            if type(body) is not dict:
                raise ValueError('record object required')
            result.append((label, body))
            found = True
            break
        if not found and line.strip():
            wrapper.append(line)
    if len(wrapper) != 3 or wrapper[0] != 'running 1 test' or wrapper[1] != 'ok' or not re.fullmatch(
            r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s', wrapper[2]):
        raise ValueError('complete successful exact-test wrapper required')
    return result


def cold(case, maximum):
    n, k = case['rows'], case['columns']
    children, owned = [], 112 + 4*(3*n + 3*min(n, maximum))
    for first in range(0, n, maximum):
        rows = min(maximum, n-first)
        device = {}
        for name, stride in [('codes', k//2), ('scales', k//64*2), ('biases', k//64*2)]:
            raw = case[name][first*stride:(first+rows)*stride]
            device[name] = base.words(raw + bytes((-len(raw)) % 4)) + [base.SENTINEL]*7
        device.update(input=[0]*(3*k+7), output=[0]*(3*rows+7), status=[0]*8)
        child_owned = 240 + 4*sum(len(v) for v in device.values())
        children.append(dict(rows=rows, capacity=3, owned_bytes=child_owned, poisoned=False,
                             pending=False, shared_read_safe=True, device=device))
        owned += child_owned
    return dict(owned_bytes=owned, poisoned=False, rows=n, columns=k, capacity=3,
                owner_slots=len(children), full_scratch_capacity=3*n, chunk_scratch_capacity=3*min(n, maximum),
                caller_bits=[], staged_bits=[0]*(3*n), chunk_scratch_bits=[0]*(3*min(n, maximum)), children=children)


def view(state):
    result = copy.deepcopy(state)
    result['caller_bits'] = []
    return result


def pointer(value):
    if type(value) is not str or not re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}', value):
        raise ValueError('non-null opaque allocation required')
    return value


def disjoint_sources(addresses, capacities):
    if type(addresses) is not list or len(addresses) != 3:
        raise ValueError('three source arrays required')
    ranges = []
    for address, capacity in zip(addresses, capacities, strict=True):
        first = int(pointer(address), 16)
        end = first + capacity
        if end >= 1 << 64:
            raise ValueError('source address interval wraps')
        ranges.append((first, end))
    ranges.sort()
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError('source allocation intervals overlap')


class Replay(base.Replay):
    def __init__(self, captured):
        super().__init__(captured)
        self.live = set()
        self.fills = self.published_chunks = self.maximum_owner = self.maximum_source = 0

    def construct_stream(self, case, maximum, key, phase):
        n, k = case['rows'], case['columns']
        state = cold(case, maximum)
        lengths = [min(n, maximum)*k//2, min(n, maximum)*(k//64)*2, min(n, maximum)*(k//64)*2]
        required = state['owned_bytes'] + sum(lengths) + 72
        self.maximum_owner = max(self.maximum_owner, state['owned_bytes'])
        self.maximum_source = max(self.maximum_source, sum(lengths)+72)
        limit = required - int(phase == 'budget')
        self.take('ATTEMPT', dict(key=key, phase=phase, rows=n, columns=k, maximum=maximum, capacity=3,
                                  minimum_working=required, limit=limit, source_capacities=lengths))
        source = {name: [0]*length for name, length in zip(('codes','scales','biases'), lengths, strict=True)}
        addresses = None
        calls = 0
        if phase != 'budget':
            for index, first in enumerate(range(0, n, maximum)):
                rows = min(maximum, n-first)
                active = dict(codes=rows*k//2, scales=rows*(k//64)*2, biases=rows*(k//64)*2)
                body = self.take('FILL_ATTEMPT')
                got = body.get('addresses')
                disjoint_sources(got, lengths)
                if addresses is None:
                    addresses = got
                same(got, addresses)
                same(body, dict(key=key, phase=phase, index=index, first=first, rows=rows, columns=k,
                                addresses=addresses, before={name: source[name][:length] for name, length in active.items()}))
                after = {}
                for name, stride in [('codes', k//2), ('scales', k//64*2), ('biases', k//64*2)]:
                    after[name] = list(case[name][first*stride:(first+rows)*stride])
                action = 'success'
                if phase == 'first-source' and index == 0 or phase == 'late-source' and index == 1:
                    action = 'source-error'
                elif phase == 'late-domain' and index == 1:
                    action = 'invalid-domain'
                    after['scales'][:2] = [0x7f, 0x7f]
                    after['biases'][:2] = [0, 0]
                elif phase == 'panic' and index == 1:
                    action = 'panic'
                self.take('FILL_RESULT', dict(key=key, phase=phase, index=index, action=action, after=after))
                calls += 1
                for name, data in after.items():
                    source[name][:len(data)] = data
                if action != 'success':
                    break
        self.fills += calls
        error = {'budget': f'Metal admission requires {required} bytes, budget {limit}',
                 'first-source': 'stream source error at 0', 'late-source': 'stream source error at 1',
                 'late-domain': 'Gemma affine weight domain'}.get(phase)
        panic = 'original stream callback panic' if phase == 'panic' else None
        success = phase in ('main', 'recover')
        self.take('OUTCOME', dict(key=key, phase=phase, completed=success, error=error, panic=panic, callback_calls=calls))
        if not success:
            return None
        actual = self.take('OWNER')
        observation = actual.get('observation')
        if type(observation) is not dict or type(observation.get('children')) is not list or len(observation['children']) != len(state['children']):
            raise ValueError('complete owner required')
        for expected, observed in zip(state['children'], observation['children'], strict=True):
            if type(observed) is not dict or type(observed.get('device')) is not dict or observed['device'].keys() != expected['device'].keys():
                raise ValueError('six complete uploaded buffers required')
            for name, words in list(expected['device'].items()):
                allocation = pointer(observed['device'][name].get('allocation'))
                if allocation in self.live:
                    raise ValueError('simultaneous resident buffers must not alias')
                self.live.add(allocation)
                expected['device'][name] = dict(allocation=allocation, words=words)
        same(actual, dict(key=key, phase=phase, observation=state))
        self.constructor_count += 1
        self.published_chunks += len(state['children'])
        return state

    def run(self, cases):
        meta = self.take('META')
        device, width = meta.get('device'), meta.get('simd_width')
        if type(device) is not str or not 1 <= len(device) <= 256 or type(width) is not int or not 1 <= width <= 65535:
            raise ValueError('named hardware/width required')
        same(meta, dict(device=device, simd_width=width, context_allowance_bytes=64*1024**2,
                        source_descriptors_bytes=72, parent_bytes=112, child_bytes=240,
                        actual_large_table_qualified=False, whole_graph_qualified=False))
        count = 0
        for case in cases:
            n, k = case['rows'], case['columns']
            for maximum in (7, 8):
                self.live = set()  # prior case owners and source arrays released
                key = f"{case['name']}/max{maximum}"
                state = self.construct_stream(case, maximum, key, 'main')
                for repeat in range(2):
                    self.step(state, case, key, 'repeat', repeat, case['input'], 3, 3*n)
                    same(state['caller_bits'], case['expected'])
                for token in range(3):
                    self.step(state, case, key, 'part', token, case['input'][token*k:(token+1)*k], 1, n)
                self.step(state, case, key, 'resume', 0, case['input'], 3, 3*n)
                if count == 0:
                    for phase in ('budget', 'first-source', 'late-source', 'late-domain', 'panic'):
                        self.construct_stream(case, maximum, key, phase)
                        self.take('UNCHANGED_OWNER', dict(key=key, phase=phase, observation=view(state)))
                    recovered = self.construct_stream(case, maximum, key, 'recover')
                    self.take('UNCHANGED_OWNER', dict(key=key, phase='recover-construct', observation=view(state)))
                    self.step(recovered, case, key+'/recover', 'recover', 0, case['input'], 3, 3*n)
                    self.take('UNCHANGED_OWNER', dict(key=key, phase='recover', observation=view(state)))
                self.take('CASE_END', dict(key=key, observation=view(state)))
                count += 1
        self.take('FINAL', dict(cases=4, successful_owners=5, projection_steps=25, source_errors=2, late_domain_errors=1,
                               callback_panics=1, budget_rejections=1, callback_side_effects_rolled_back=False,
                               gpu_commands_submitted_during_upload=False, actual_large_table_qualified=False,
                               whole_graph_qualified=False, performance_qualified=False))
        try:
            next(self.captured)
        except StopIteration:
            pass
        else:
            raise ValueError('trailing native record')
        same([count, self.constructor_count, self.steps, self.values, self.fills, self.published_chunks], [4,5,25,777,20,13])
        # Conservative diagnostic overlap policy counts even failed uploads that
        # an implementation/driver pool might retain until the outer pool drains.
        ledger = dict(context=64*1024**2, resident_owners_policy=6*self.maximum_owner,
                      loading=self.maximum_source, caller=3*19*4, fixture_host_policy=32*1024**2,
                      observation_serialization_policy=64*1024**2)
        total = sum(ledger.values())
        if total > 1024**3:
            raise ValueError('aggregate diagnostic policy exceeds limit')
        return dict(passed=True, device=device, simd_width=width, cases=4, successful_owners=5,
                    projection_steps=self.steps, successful_projection_values=self.values, source_callback_calls=self.fills,
                    published_chunks=self.published_chunks, complete_uploaded_buffers=True, stable_source_addresses=True,
                    original_callback_errors_preserved=True, callback_panic_recovered=True, warmed_owner_unchanged=True,
                    constructor_atomic=True, external_callback_side_effects_rolled_back=False,
                    aggregate_policy_bytes=total, workload_ledger=ledger, actual_large_table_qualified=False,
                    whole_graph_qualified=False, performance_qualified=False)


def replay(capture, directory):
    return Replay(records(base.reference.read(capture,32*1024**2))).run(base.fixture(directory))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('native stdout and frozen tiny table fixture')
    import json
    print(json.dumps(replay(Path(sys.argv[1]),Path(sys.argv[2])),allow_nan=False))
