"""Replay complete bounded partitioned-affine captures against frozen source bytes.

No engine, Metal, producer import, downloaded model, or inferred partial hashes.
The complete six buffers and both CPU staging arrays are replayed at each step.
"""
from pathlib import Path
from fractions import Fraction
import copy
import functools
import hashlib
import re
import struct
import sys
import check_gemma_table_math as reference

SENTINEL = 0x5A170031
CALLER = 0x42280000
PREFIX = 'GEMMA_PARTITION_'
TEST = 'metal::gemma_affine::partitioned::tests::frozen_partitioned_projection_reuse_transactions'
PINS = {'math.json': '9083f619b0a12202955156631e8b65e7824cb7120641edcad6b906d551aa721f',
        'math.bin': 'cc34796740e66853ecccb950028ff994de37b2bc5a7e12a1cc9877b0ce3a549e',
        'manifest.json': '36aeea18b3143e0024d33e097091ebf47d6fdde83beaf69784d1e2f498cc6683'}

def same(actual, expected):
    reference.same(actual, expected)

def words(raw):
    return list(struct.unpack('<' + 'I' * (len(raw) // 4), raw))

def f32(bits):
    return reference.rational(bits, 23, 8)

def rounded(value):
    return reference.rounded(value, 23, 8)

@functools.lru_cache(maxsize=64)
def projection(weights, columns, input_bits):
    """Bounded cache of independently computed tiny mathematical outputs."""
    rows = len(weights) // columns
    decoded_weights = [f32(v) for v in weights]
    decoded_input = [f32(v) for v in input_bits]
    output = []
    for token in range(len(input_bits) // columns):
        for row in range(rows):
            ordered = Fraction()
            exact = Fraction()
            for column in range(columns):
                product = decoded_weights[row * columns + column] * decoded_input[token * columns + column]
                exact += product
                ordered = reference.rational(reference.rounded(ordered + product, 52, 11), 52, 11)
            bits = rounded(ordered)
            if bits != rounded(exact):
                raise ValueError('tiny exact and ordered contracts disagree')
            output.append(bits)
    return tuple(output)

def fixture(directory):
    for name, wanted in PINS.items():
        raw = reference.read(directory / name, 512 * 1024)
        if hashlib.sha256(raw).hexdigest() != wanted:
            raise ValueError('frozen source identity differs: ' + name)
    reference.replay(directory)
    data = reference.document(reference.read(directory / 'math.json', 512 * 1024))
    binary = reference.read(directory / 'math.bin', 65536)
    cases = []
    for entry in data['cases']:
        def array(role):
            spec = entry['arrays'][role]
            return binary[spec['offset']:spec['offset'] + spec['bytes']]
        cases.append(dict(name=entry['name'], rows=entry['rows'], columns=entry['columns'],
                          codes=array('codes'), scales=array('scales'), biases=array('biases'),
                          weights=words(array('reconstructed')), input=words(array('input')),
                          expected=words(array('output_f32')),
                          ordered=list(struct.unpack('<' + 'd' * (3 * entry['rows']), array('ordered_fp64')))))
    return cases

def records(raw):
    if len(raw) > 32 * 1024**2:
        raise ValueError('bounded native capture required')
    lines = raw.decode('utf-8').splitlines()
    result = []
    wrapper = []
    for line in lines:
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

class Replay:
    def __init__(self, captured):
        self.captured = iter(captured)
        self.steps = 0
        self.values = 0
        self.constructor_count = 0

    def take(self, label, expected=None):
        try:
            got_label, body = next(self.captured)
        except StopIteration as error:
            raise ValueError('missing ' + label) from error
        if got_label != label:
            raise ValueError('record order differs: ' + got_label + '/' + label)
        if expected is not None:
            same(body, expected)
        return body

    def construct(self, case, maximum, capacity, key, budget=False):
        n, k = case['rows'], case['columns']
        count = (n + maximum - 1) // maximum
        children = []
        for first in range(0, n, maximum):
            rows = min(maximum, n - first)
            device = {}
            for name, stride in [('codes', k // 2), ('scales', k // 64 * 2), ('biases', k // 64 * 2)]:
                raw = case[name][first * stride:(first + rows) * stride]
                device[name] = words(raw + bytes((-len(raw)) % 4)) + [SENTINEL] * 7
            device.update(input=[0] * (capacity * k + 7), output=[0] * (capacity * rows + 7), status=[0] * 8)
            owned = sum(len(v) * 4 for v in device.values()) + 240
            children.append(dict(rows=rows, capacity=capacity, owned_bytes=owned, poisoned=False,
                                 pending=False, shared_read_safe=True, device=device))
        full = capacity * n
        scratch = capacity * min(maximum, n)
        required = sum(v['owned_bytes'] for v in children) + (full + scratch) * 4 + 112
        if budget:
            self.take('BUDGET', dict(key=key, required=required, limit=required - 1, rejected=True,
                                   error=f'Metal admission requires {required} bytes, budget {required - 1}'))
        self.take('CONSTRUCTOR_ATTEMPT', dict(key=key, rows=n, columns=k, maximum_rows=maximum, capacity=capacity, required=required))
        self.take('CONSTRUCTOR_OUTCOME', dict(key=key, completed=True, error=None))
        record = self.take('CONSTRUCTOR')
        # Raw allocation identities are opaque; require non-null, unique live
        # allocations and freeze all of them for this owner's complete lifetime.
        ids = set()
        if type(record.get('observation')) is not dict:
            raise ValueError('constructor observation required')
        actual_children = record['observation'].get('children')
        if type(actual_children) is not list or len(actual_children) != count:
            raise ValueError('complete child inventory required')
        for child, actual in zip(children, actual_children, strict=True):
            device = actual.get('device') if type(actual) is dict else None
            if type(device) is not dict or device.keys() != child['device'].keys():
                raise ValueError('six complete device allocations required')
            for name, values in list(child['device'].items()):
                entry = device[name]
                allocation = entry.get('allocation') if type(entry) is dict else None
                if type(allocation) is not str or not re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}', allocation) or allocation in ids:
                    raise ValueError('distinct live non-null allocation identity required')
                ids.add(allocation)
                child['device'][name] = dict(allocation=allocation, words=values)
        state = dict(owned_bytes=required, poisoned=False, rows=n, columns=k, capacity=capacity,
                     owner_slots=count, full_scratch_capacity=full, chunk_scratch_capacity=scratch,
                     caller_bits=[], staged_bits=[0] * full, chunk_scratch_bits=[0] * scratch, children=children)
        same(record, dict(key=key, observation=state))
        self.constructor_count += 1
        return state

    def step(self, state, case, key, phase, index, input_bits, tokens, caller_len, error=None, completed_children=None):
        self.take('STEP_ATTEMPT', dict(key=key, phase=phase, index=index, tokens=tokens, input_bits=input_bits, caller_len=caller_len))
        state['caller_bits'] = [CALLER] * caller_len
        if error is None or completed_children is not None:
            full_output = projection(tuple(case['weights']), case['columns'], tuple(input_bits))
            first = 0
            for child_index, child in enumerate(state['children']):
                if completed_children is not None and child_index >= completed_children:
                    break
                n = child['rows']
                output = []
                for token in range(tokens):
                    start = token * case['rows'] + first
                    output.extend(full_output[start:start + n])
                device = child['device']
                for name, active in [('input', input_bits), ('output', output), ('status', [0])]:
                    length = len(device[name]['words'])
                    device[name]['words'] = active + [SENTINEL] * (length - len(active))
                state['chunk_scratch_bits'][:len(output)] = output
                for token in range(tokens):
                    start = token * case['rows'] + first
                    state['staged_bits'][start:start + n] = output[token * n:(token + 1) * n]
                first += n
            if error is None:
                state['caller_bits'] = state['staged_bits'][:tokens * case['rows']]
            else:
                state['poisoned'] = True
        self.take('STEP_OUTCOME', dict(key=key, phase=phase, index=index, completed=error is None,
                                       error=error, poisoned=state['poisoned']))
        self.take('STEP', dict(key=key, phase=phase, index=index, tokens=tokens, completed=error is None,
                              error=error, observation=state))
        self.steps += 1
        if error is None:
            self.values += caller_len

    def run(self, cases):
        meta = self.take('META')
        device = meta.get('device')
        width = meta.get('simd_width')
        if type(device) is not str or not 1 <= len(device) <= 256 or type(width) is not int or not 1 <= width <= 65535:
            raise ValueError('named hardware and execution width required')
        same(meta, dict(device=device, simd_width=width, context_allowance_bytes=64 * 1024**2,
                        owner_struct_bytes=112, child_struct_bytes=240, real_head_qualified=False, whole_graph_qualified=False))
        first_case = True
        for case in cases:
            n, k = case['rows'], case['columns']
            for maximum in (1, 7, 8, 19):
                key = f"{case['name']}/max{maximum}"
                state = self.construct(case, maximum, 3, key, True)
                for repeat in range(2):
                    self.step(state, case, key, 'repeat', repeat, case['input'], 3, 3 * n)
                    same(state['caller_bits'], case['expected'])
                    for bits, ordered in zip(state['caller_bits'], case['ordered'], strict=True):
                        if abs(float(f32(bits)) - ordered) > 1e-5 + 1e-4 * abs(ordered):
                            raise ValueError('ordered FP64 gate failed')
                for token in range(3):
                    self.step(state, case, key, 'part', token, case['input'][token * k:(token + 1) * k], 1, n)
                self.step(state, case, key, 'resume', 0, case['input'], 3, 3 * n)
                if first_case:
                    before = copy.deepcopy(state['staged_bits'])
                    changed = [rounded(f32(v) + 1) for v in case['input']]
                    self.step(state, case, key, 'composition-failure', 0, changed, 3, 3 * n,
                              'Gemma partitioned affine injected composition failure', 2)
                    if state['staged_bits'] == before:
                        raise ValueError('composition failure did not follow changed completed chunks')
                    self.step(state, case, key, 'poison-retry', 0, case['input'], 3, 3 * n,
                              'Gemma partitioned affine owner poisoned; recreate')
                    # Explicit new owner: same geometry and source; a released
                    # allocation's address can be reused, so begin a new epoch.
                    replacement_key = key + '/recreated'
                    replacement = self.construct(case, maximum, 3, replacement_key)
                    self.step(replacement, case, replacement_key, 'recreate', 0, case['input'], 3, 3 * n)
                    first_case = False
        late = dict(rows=2, columns=64, codes=bytes(64), scales=bytes(4), biases=bytes.fromhex('0000803f'), weights=[0] * 64 + [0x3f800000] * 64)
        key = 'later-admission'
        state = self.construct(late, 1, 1, key)
        self.step(state, late, key, 'warm', 0, [0] * 64, 1, 2)
        conflict = [rounded(Fraction(2)**100), rounded(Fraction(1)), rounded(-Fraction(2)**100)] + [0] * 61
        self.step(state, late, key, 'reject', 0, conflict, 1, 2,
                  'Gemma affine input ordered-sum accuracy bound')
        invalids = [([0x7fc00000] + [0] * 63, 1, 2), ([0] * 63, 1, 2), ([0] * 64, 1, 3),
                    ([], 0, 0), ([0] * 128, 2, 4), ([], 2**64 - 1, 0)]
        for index, (bits, tokens, size) in enumerate(invalids):
            self.step(state, late, key, 'preflight', index, bits, tokens, size,
                      'Gemma partitioned affine input shape/nonfinite value')
        self.step(state, late, key, 'recover', 0, [0] * 64, 1, 2)
        self.take('DOMAIN', dict(rejected=True, error='Gemma affine weight domain'))
        self.take('OLD_HEAD', dict(rejected=True, error='Gemma affine fixed geometry/format outside bounds', large_spans_read=False))
        self.take('FINAL', dict(cases=8, repeats=2, partitions=24, resumes=8, budget_rejections=8, composition_failures=1,
                                same_geometry_recreations=1, late_preflight_rejections=1, recoveries=1, domain_rejections=1,
                                preflight_rejections=6, old_head_geometry_rejected=True, driver_or_inflight_failure_injected=False,
                                real_head_qualified=False, whole_graph_qualified=False))
        try:
            next(self.captured)
        except StopIteration:
            pass
        else:
            raise ValueError('trailing native record')
        same([self.steps, self.constructor_count], [60, 10])
        return dict(passed=True, device=device, simd_width=width, cases=8, owners=10, steps=self.steps,
                    successful_output_values=self.values, complete_buffers=True, composition_failure_after_completed_chunks=2,
                    caller_transactional=True, sticky_poison=True, same_geometry_recreated=True,
                    driver_or_inflight_failure_injected=False, real_head_qualified=False, whole_graph_qualified=False)

def replay(capture, directory):
    return Replay(records(reference.read(capture, 32 * 1024**2))).run(fixture(directory))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('native stdout and frozen table fixture directory')
    import json
    print(json.dumps(replay(Path(sys.argv[1]), Path(sys.argv[2])), allow_nan=False))
