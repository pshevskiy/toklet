"""Cold corruption controls; no model, GPU, MLX or generated program."""
import copy
import json
import unittest

import qwen3_reference_schedule_continuity as subject


def records(candidate=False):
    resources = {name: 1 for name in subject.MEMORY_FIELDS if name != 'admitted_bytes'}
    resources['admitted_bytes'] = sum(resources.values())
    start = dict(event='start', stage='main', phase='anchor',
                 format='toklet-qwen3-context4096-native-events-v1',
                 profile='MlxAffine/Compensated/Reference/FP16',
                 config_sha256=('b' if candidate else 'a') * 64)
    ready = dict(event='ready', stage='main', debug_assertions=False,
                 capacity=512, chunk=32, diagnostics=True, readback='FullLogits',
                 memory=resources, admitted_bytes=resources['admitted_bytes'] + 64,
                 application_allowance_bytes=64, load_compile_seconds=1,
                 session_creation_seconds=1)
    if candidate:
        start['reference_schedule'] = 'parallel_outputs_experimental'
        ready['reference_schedule'] = 'ParallelOutputsExperimental'
    rows = [dict(event='row', stage='main', position=i, finite=True,
                 matches=True, sha256='c' * 64) for i in range(512)]
    done = dict(event='done', stage='main', phase='anchor', passed=True,
                input_endpoints_checked=True, committed_tokens=512)
    return [start, ready, *rows, done]


def raw(value):
    return b''.join((json.dumps(row, allow_nan=False) + '\n').encode() for row in value)


class Continuity(unittest.TestCase):
    def compare(self, candidate):
        return subject.compare_f2(raw(candidate), raw(records()), 'anchor', 'b' * 64)

    def test_only_validated_metadata_is_excluded(self):
        candidate = records(True)
        candidate[1]['load_compile_seconds'] = 8
        candidate[1]['memory']['metadata_bytes'] += 1
        candidate[1]['memory']['admitted_bytes'] += 1
        candidate[1]['admitted_bytes'] += 1
        self.assertTrue(self.compare(candidate)['identical'])
        candidate[1]['unknown_profile_field'] = 1
        self.assertFalse(self.compare(candidate)['identical'])

    def test_wrong_schedule_configuration_and_resource_sum_reject(self):
        for field, value in (('reference_schedule', 'serial'), ('config_sha256', 'd' * 64)):
            candidate = records(True)
            candidate[0][field] = value
            with self.assertRaises(ValueError):
                self.compare(candidate)
        candidate = records(True)
        candidate[1]['reference_schedule'] = 'Serial'
        with self.assertRaises(ValueError):
            self.compare(candidate)
        candidate = records(True)
        candidate[1]['memory']['admitted_bytes'] += 1
        with self.assertRaises(ValueError):
            self.compare(candidate)

    def test_every_row_and_completion_remain_gate_inputs(self):
        candidate = records(True)
        candidate[510]['sha256'] = 'd' * 64
        self.assertFalse(self.compare(candidate)['identical'])
        for mutate in ('missing_row', 'failed', 'wrong_work'):
            candidate = records(True)
            if mutate == 'missing_row':
                del candidate[510]
            elif mutate == 'failed':
                candidate[-1]['passed'] = False
            else:
                candidate[-1]['committed_tokens'] = 511
            with self.assertRaises(ValueError):
                self.compare(candidate)

    def test_deterministic_types_and_signed_zero_are_exact(self):
        for old, new in ((False, 0), (1, 1.0), (0.0, -0.0)):
            historical, candidate = records(), records(True)
            historical[3]['typed_witness'] = old
            candidate[3]['typed_witness'] = new
            result = subject.compare_f2(raw(candidate), raw(historical), 'anchor', 'b' * 64)
            self.assertFalse(result['identical'])

    def test_call_timing_only_excludes_named_durations(self):
        timing = {name: 1.0 for name in subject.TIMING_FIELDS}
        timing['command_count'] = 1
        row = dict(event='call', label='ordinary_decode', rows=1,
                   outer_completed_call_seconds=2, timing=timing)
        expected = subject.normalize_f2(row, False)
        changed = copy.deepcopy(row)
        changed['timing']['wait_seconds'] = 9
        self.assertEqual(subject.normalize_f2(changed, True), expected)
        changed['timing']['command_count'] = 2
        self.assertNotEqual(subject.normalize_f2(changed, True), expected)
        changed['timing']['unknown'] = 0
        with self.assertRaises(ValueError):
            subject.normalize_f2(changed, True)

    def test_negative_and_incomplete_timing_or_memory_reject(self):
        timing = {name: 1.0 for name in subject.TIMING_FIELDS}
        timing['command_count'] = 1
        row = dict(event='call', rows=1, outer_completed_call_seconds=1, timing=timing)
        for key in subject.TIMING_FIELDS:
            bad = copy.deepcopy(row)
            bad['timing'][key] = -1
            with self.assertRaises(ValueError):
                subject.normalize_f2(bad, True)
        del row['timing']['commit_seconds']
        with self.assertRaises(ValueError):
            subject.normalize_f2(row, True)
        resources = records()[1]['memory']
        resources['extra'] = 1
        with self.assertRaises(ValueError):
            subject.memory(resources)

    def test_fully_cached_call_has_no_gpu_timing(self):
        row = dict(event='call', label='fully_cached', rows=0, error=None,
                   timing=None, outer_completed_call_seconds=0.000001)
        expected = dict(row)
        del expected['outer_completed_call_seconds']
        self.assertEqual(subject.normalize_f2(row, True), expected)
        bad = dict(row, rows=1)
        with self.assertRaises(ValueError):
            subject.normalize_f2(bad, True)
        bad = dict(row, timing={})
        with self.assertRaises(ValueError):
            subject.normalize_f2(bad, True)
        bad = dict(row, error='failure')
        with self.assertRaises(ValueError):
            subject.normalize_f2(bad, True)

    def test_full_quality_adapter_only_removes_positively_checked_tag(self):
        from unittest.mock import patch
        current = [dict(event='ready', reference_schedule='parallel_outputs_experimental', extra=4)]
        historical = [dict(event='ready', extra=4)]
        with patch.object(subject.quality, 'compare', return_value={'identical': True}) as compare:
            result = subject.compare_quality(raw(current), raw(historical), 'tasks')
            self.assertTrue(result['identical'])
            compare.assert_called_once_with(historical, historical, 'tasks')
        current[0]['reference_schedule'] = 'serial'
        with self.assertRaises(ValueError):
            subject.compare_quality(raw(current), raw(historical), 'tasks')


if __name__ == '__main__':
    unittest.main()
