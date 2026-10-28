"""Exact experimental-schedule bridges to authenticated native F2/Q4 journals.

The owned caller authenticates complete candidate/config/input/model/binary and
historical receipts plus measured memory/endpoints. This reader executes no
model or generated program. Unknown fields remain comparison inputs.
"""
import copy
import json
import math
import re

import qwen3_native_continuity as quality

TIMING_FIELDS = frozenset(('command_creation_seconds', 'clock_calibration_seconds',
    'commit_seconds', 'host_encode_seconds', 'wait_seconds', 'gpu_start_seconds',
    'gpu_end_seconds', 'readback_seconds'))
MEMORY_FIELDS = frozenset(('weights_bytes', 'kv_bytes', 'scratch_bytes',
    'history_bytes', 'logits_bytes', 'metadata_bytes', 'metadata_allowance_bytes',
    'admitted_bytes'))
PHASES = ('anchor', 'a32', 'a128', 'b128', 'timing32_full', 'timing128_full',
          'timing32_greedy', 'sampled_replay', 'quality')
LIMIT = 6 * 1024**3
require = quality.require


def nonnegative(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'finite nonnegative timing observation required')


def memory(value):
    require(isinstance(value, dict) and set(value) == MEMORY_FIELDS,
            'complete known session resource schema required')
    require(all(type(v) is int and v > 0 for v in value.values()),
            'positive integer resource accounting required')
    require(value['admitted_bytes'] == sum(v for k, v in value.items() if k != 'admitted_bytes')
            and value['admitted_bytes'] <= LIMIT, 'bounded summed resource accounting required')


def normalize_f2(row, candidate, config_sha256=None):
    """Remove only positively validated identity/timing/resource observations."""
    value = copy.deepcopy(row)
    event = value['event']
    if event == 'start':
        require(re.fullmatch('[0-9a-f]{64}', value.get('config_sha256', '')) is not None,
                'configuration digest required')
        if candidate:
            require(value.get('reference_schedule') == 'parallel_outputs_experimental',
                    'explicit parallel start schedule required')
            require(value['config_sha256'] == config_sha256,
                    'externally authenticated candidate configuration required')
            del value['reference_schedule']
        else:
            require('reference_schedule' not in value, 'original F2 schedule schema required')
        del value['config_sha256']
    elif event == 'ready':
        if candidate:
            require(value.get('reference_schedule') == 'ParallelOutputsExperimental',
                    'actual experimental session schedule required')
            del value['reference_schedule']
        else:
            require('reference_schedule' not in value, 'original F2 readiness required')
        memory(value['memory'])
        require(type(value['application_allowance_bytes']) is int
                and value['application_allowance_bytes'] >= 0
                and value['admitted_bytes'] == value['memory']['admitted_bytes'] + value['application_allowance_bytes']
                and value['admitted_bytes'] <= LIMIT, 'complete process admission required')
        for field in ('load_compile_seconds', 'session_creation_seconds'):
            nonnegative(value[field])
            del value[field]
        del value['memory'], value['admitted_bytes']
    elif event == 'call':
        nonnegative(value['outer_completed_call_seconds'])
        del value['outer_completed_call_seconds']
        timing = value['timing']
        if value.get('rows') == 0:
            require(type(value['rows']) is int and value.get('error') is None and timing is None,
                    'zero-work success must have no stale GPU timing')
            return value
        require(type(value.get('rows')) is int and value['rows'] > 0,
                'positive committed call rows required')
        require(isinstance(timing, dict) and set(timing) == TIMING_FIELDS | {'command_count'},
                'closed complete call timing schema required')
        require(type(timing['command_count']) is int and timing['command_count'] >= 0,
                'integer command count required')
        for field in TIMING_FIELDS:
            nonnegative(timing[field])
        require(timing['gpu_end_seconds'] >= timing['gpu_start_seconds'],
                'completed GPU interval required')
        value['timing'] = {'command_count': timing['command_count']}
    elif event == 'final_state':
        memory(value['memory'])
        del value['memory']
    return value


def complete_f2(rows, phase):
    require(phase in PHASES and len(rows) >= 3, 'explicit complete F2 phase required')
    start, ready, done = rows[0], rows[1], rows[-1]
    require(start.get('event') == 'start' and start.get('format') == 'toklet-qwen3-context4096-native-events-v1'
            and start.get('phase') == phase and start.get('profile') == 'MlxAffine/Compensated/Reference/FP16',
            'frozen F2 arithmetic/phase required')
    capacity = 512 if phase == 'anchor' else 4096
    chunk = 128 if phase in ('a128', 'b128', 'timing128_full') else 32
    diagnostic = phase in ('anchor', 'a32', 'a128', 'b128', 'quality')
    require(ready.get('event') == 'ready' and ready.get('debug_assertions') is False
            and ready.get('capacity') == capacity and ready.get('chunk') == chunk
            and ready.get('diagnostics') is diagnostic
            and ready.get('readback') == ('GreedyToken' if phase == 'timing32_greedy' else 'FullLogits'),
            'frozen F2 execution profile required')
    require(done.get('event') == 'done' and done.get('phase') == phase and done.get('passed') is True
            and done.get('input_endpoints_checked') is True
            and done.get('committed_tokens') == (8192 if phase == 'a32' else capacity),
            'complete F2 work and endpoint proof required')
    expected_rows = capacity if diagnostic else 0
    observed = [row for row in rows if row.get('event') == 'row']
    require(len(observed) == expected_rows
            and [row['position'] for row in observed] == list(range(expected_rows))
            and all(row['finite'] is True and row['matches'] is True for row in observed),
            'all finite matching logical rows required')
    require(sum(row.get('event') == 'start' for row in rows) == 1
            and sum(row.get('event') == 'ready' for row in rows) == 1
            and sum(row.get('event') == 'done' for row in rows) == 1,
            'one complete phase envelope required')


def compare_f2(current_raw, historical_raw, phase, config_sha256):
    """Compare every event, including all4096 A32/Q row and state observations."""
    current, historical = quality.parse(current_raw), quality.parse(historical_raw)
    complete_f2(current, phase)
    complete_f2(historical, phase)
    actual = [json.dumps(normalize_f2(row, True, config_sha256), sort_keys=True,
                         separators=(',', ':'), allow_nan=False) for row in current]
    expected = [json.dumps(normalize_f2(row, False), sort_keys=True,
                           separators=(',', ':'), allow_nan=False) for row in historical]
    differences = [i for i in range(max(len(actual), len(expected)))
                   if (actual[i] if i < len(actual) else None) != (expected[i] if i < len(expected) else None)]
    return dict(phase=phase, complete=True, identical=not differences,
                current_events=len(actual), historical_events=len(expected),
                different_events=len(differences), first_differences=differences[:16],
                logical_rows=512 if phase == 'anchor' else 4096 if phase in ('a32', 'a128', 'b128', 'quality') else 0,
                timing_fields=sorted(TIMING_FIELDS), resource_fields=['ready.memory', 'ready.admitted_bytes', 'final_state.memory'])


def compare_quality(current_raw, historical_raw, mode):
    """Reuse the complete existing bridge after validating its sole new tag."""
    current, historical = quality.parse(current_raw), quality.parse(historical_raw)
    require(current[0].get('event') == 'ready'
            and current[0].get('reference_schedule') == 'parallel_outputs_experimental',
            'explicit full quality experimental schedule required')
    require('reference_schedule' not in historical[0], 'original full quality readiness required')
    del current[0]['reference_schedule']
    return quality.compare(current, historical, mode)
