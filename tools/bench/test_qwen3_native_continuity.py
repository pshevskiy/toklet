"""Portable adversarial checks for the exact full-workload continuity bridge."""
import copy
import json
import unittest

import qwen3_native_continuity as bridge


def fixture(mode):
    targets, tasks = (10000, 0) if mode == 'loss' else (0, 100)
    ready = dict(event='ready', format='toklet-qwen3-quality-diagnostic-events-v1',
        mode=mode, full_quality_workload=True, debug_assertions=False,
        projection='mlx_affine', normalization='compensated', attention='reference',
        activation_dtype='FP16', kv_dtype='FP16', context_capacity=1025,
        prefill_chunk_size=128, targets=targets, tasks=tasks, admitted_bytes=1024, memory={})
    if mode == 'loss':
        events = [dict(event='loss', target_index=i, negative_log_likelihood=1.0,
                       target_logit_f32_bits=0x3f800000, logits_sha256='a'*64)
                  for i in range(targets)]
    else:
        report = dict(sampled_tokens=2, emitted_tokens=1, history_tokens=4,
                      evaluated_tokens=3, pending_tokens=1,
                      reason=dict(kind='stop_token', token_id=151645),
                      **{key: 1.0 for key in bridge.TIMINGS})
        events = [dict(event='task_done', id=str(i), visible_token_ids=[17],
                       report=copy.deepcopy(report)) for i in range(tasks)]
    return [ready, *events, dict(event='done', targets=targets, tasks=tasks)]


class Continuity(unittest.TestCase):
    def test_complete_workloads_match(self):
        for mode in ('loss', 'tasks'):
            rows = fixture(mode)
            self.assertTrue(bridge.compare(rows, copy.deepcopy(rows), mode)['identical'])

    def test_only_named_timing_and_resource_observations_are_excluded(self):
        old = fixture('tasks');new = copy.deepcopy(old)
        new[0]['admitted_bytes'] = 2048;new[0]['memory'] = {'scratch_bytes': 123}
        for row in new[1:-1]:
            for key in bridge.TIMINGS:row['report'][key] = 2.0
        self.assertTrue(bridge.compare(new, old, 'tasks')['identical'])
        new[1]['report']['unknown_seconds'] = 2.0
        self.assertFalse(bridge.compare(new, old, 'tasks')['identical'])

    def test_loss_distributions_target_bits_and_loss_are_exact(self):
        old = fixture('loss')
        for key, value in [('logits_sha256', 'b'*64), ('target_logit_f32_bits', 0x3f800001),
                           ('negative_log_likelihood', 1.000000000000001), ('target_index', 7)]:
            new = copy.deepcopy(old);new[1][key] = value
            self.assertFalse(bridge.compare(new, old, 'loss')['identical'], key)

    def test_tokens_stops_pending_and_identity_cannot_transfer(self):
        old = fixture('tasks')
        for key, value in [('sampled_tokens', 1), ('pending_tokens', 0),
                           ('reason', dict(kind='output_limit'))]:
            new = copy.deepcopy(old);new[1]['report'][key] = value
            self.assertFalse(bridge.compare(new, old, 'tasks')['identical'], key)
        for key, value in [('visible_token_ids', [18]), ('id', 'different')]:
            new = copy.deepcopy(old);new[1][key] = value
            self.assertFalse(bridge.compare(new, old, 'tasks')['identical'], key)

    def test_clipping_missing_terminal_and_wrong_profile_reject(self):
        for mode in ('loss', 'tasks'):
            old = fixture(mode)
            for new in (old[:-1], [old[0], *old[2:]], [*old, old[-1]]):
                with self.assertRaises(ValueError):bridge.compare(new, old, mode)
            for key, value in [('attention', 'compensated_reference'), ('targets', True),
                               ('context_capacity', 512), ('full_quality_workload', False),
                               ('admitted_bytes', 6*1024**3+1)]:
                new = copy.deepcopy(old);new[0][key] = value
                with self.assertRaises(ValueError):bridge.compare(new, old, mode)

    def test_parser_rejects_duplicates_nonfinite_and_partial_journals(self):
        for raw in (b'{"event":"x","event":"y"}\n', b'{"x":NaN}\n',
                    b'{"x":1e999}\n', b'{}', b'[]\n'):
            with self.assertRaises(ValueError):bridge.parse(raw)
        rows = fixture('tasks')
        raw = b''.join((json.dumps(row)+'\n').encode() for row in rows)
        self.assertEqual(bridge.parse(raw), rows)
        rows[1]['report']['total_seconds'] = float('inf')
        with self.assertRaises(ValueError):bridge.compare(rows, fixture('tasks'), 'tasks')


if __name__ == '__main__':
    unittest.main()
