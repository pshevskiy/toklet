"""Producer-type restoration, precision and immutable-gate regressions; no GPU."""
import copy
import unittest
from unittest.mock import patch

import m6_fp32_context_audit as typed
from test_m6_long_quality_audit import records
from test_m6_context_scope_audit import records as context_records


def real_reversal():
    return dict(event='comparison_row', target_index=127, target_token=2686, prefix_tokens=4095,
        reference_nll=6.670187660935725, online_nll=6.667561824827409,
        reference_greedy=14856, online_greedy=1075, reference_margin=.0042743682861328125,
        reference_winner_logits=[17.63076, 17.625118], online_winner_logits=[17.626486, 17.628891],
        max_absolute_logit_delta=.027275800704956055, rms_logit_delta=.005032867459099283)


class TypedFP32(unittest.TestCase):
    def test_real_short_f32_json_roundtrip_recovers_exact_f64_margin(self):
        row = real_reversal()
        with self.assertRaisesRegex(ValueError, 'Top-two margin'):
            typed.quality.winner_accounting(row)
        values = records(); values[0]['worker']['phases'][0]['events'][-2] = row
        original = copy.deepcopy(values)
        restored, proof = typed.restore(values)
        changed = restored[0]['worker']['phases'][0]['events'][-2]
        reversal = typed.quality.winner_accounting(changed)
        self.assertEqual(reversal['reference_candidate_gap'], row['reference_margin'])
        self.assertTrue(reversal['numerical_reversal_accounted'])
        self.assertFalse(reversal['investigation_complete'])
        self.assertEqual(values, original)
        self.assertEqual(proof['restored_winner_values'], 4096)
        self.assertEqual(len(proof['changed_prediction_values']), 1)
        self.assertEqual(proof['changed_prediction_values'][0]['original_json_pairs']['reference_winner_logits'], row['reference_winner_logits'])
        self.assertFalse(typed.quality.summarize(restored)['passed'])

    def test_fp32_bit_goldens_and_invalid_finite_range(self):
        for value, expected, bits in ((1.0000000596046448, 1., '3f800000'),
                                      (2.0**-149, 2.0**-149, '00000001'),
                                      (-0.0, -0.0, '80000000')):
            actual, encoded = typed.fp32(value)
            self.assertEqual(actual, expected)
            self.assertEqual(encoded, bits)
        for value in (True, float('nan'), float('inf'), 1e300):
            with self.assertRaises(ValueError): typed.fp32(value)

    def test_f64_quality_metrics_and_all_frozen_gates_remain_unchanged(self):
        values = records()
        for row in values[0]['worker']['phases'][0]['events'][1:-1]: row['online_nll'] = 2.1
        restored, _ = typed.restore(values)
        for original, changed in zip(values[0]['worker']['phases'][0]['events'][1:-1],
                                     restored[0]['worker']['phases'][0]['events'][1:-1], strict=True):
            for field in ('reference_margin', 'max_absolute_logit_delta', 'rms_logit_delta', 'reference_nll', 'online_nll'):
                self.assertEqual(original[field], changed[field])
        result = typed.quality.summarize(restored)
        self.assertFalse(result['passed'])
        self.assertFalse(result['models']['small']['each_depth_gate_passed'])
        self.assertTrue(result['models']['small']['pooled_gate_passed'])
        with self.assertRaises(ValueError): typed.quality.summarize(restored[:-1])

    def test_true_margin_winner_and_error_inconsistencies_still_reject(self):
        for field, value in (('reference_margin', .1), ('max_absolute_logit_delta', .00001),
                             ('reference_winner_logits', [17., 17.625118])):
            values = records(); row = real_reversal(); row[field] = value
            values[0]['worker']['phases'][0]['events'][-2] = row
            restored, _ = typed.restore(values)
            with self.subTest(field=field), self.assertRaises(ValueError): typed.quality.summarize(restored)

    def test_invalid_saved_pair_shapes_are_not_normalized(self):
        for value in ([], [1.], [1., 2., 3.], (1., 2.), None):
            values = records(); values[0]['worker']['phases'][0]['events'][1]['reference_winner_logits'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): typed.restore(values)

    def test_context32_success_and_changed_prediction_stays_unresolved(self):
        paths = [typed.audit.ROOT/f'artifacts/typed-context-{i}.json' for i in range(32)]
        for flip in (False, True):
            values = context_records()
            for value in values: value['format'] = 'toklet-m6-context-collection-v1'
            if flip:
                values[0]['worker']['phases'][0]['events'][-2] = real_reversal()
            proof = dict(passed=True, records=[], bindings={str(p.relative_to(typed.audit.ROOT)): 'saved' for p in paths})
            by_path = dict(zip(paths, values, strict=True))
            with self.subTest(flip=flip), \
                 patch.object(typed.audit, 'read_json', side_effect=lambda p: (by_path[p], 'saved')), \
                 patch.object(typed.audit, 'audit', return_value=proof), \
                 patch.object(typed.audit, 'verify_bindings') as verify:
                result = typed.run(paths, 'context')
            self.assertEqual(result['passed'], not flip)
            self.assertTrue(result['derived']['complete_frozen_context_coverage'])
            self.assertFalse(result['m6_complete'])
            self.assertEqual(len(result['fp32_reconstruction']['changed_prediction_values']), int(flip))
            verify.assert_called_once()

    def test_replay_byte_identity_and_exact_role_count_are_required(self):
        paths = [typed.audit.ROOT/f'artifacts/typed-synthetic-{i}.json' for i in range(8)]
        proof = dict(passed=True, records=[], bindings={str(p.relative_to(typed.audit.ROOT)): 'new' for p in paths})
        with patch.object(typed.audit, 'read_json', return_value=({'format':'toklet-m6-context-collection-v1'}, 'old')), \
             patch.object(typed.audit, 'audit', return_value=proof), \
             self.assertRaisesRegex(ValueError, 'differs from replayed bytes'):
            typed.run(paths, 'quality')
        for kind, rows in (('quality', paths[:-1]), ('context', paths), ('other', paths)):
            with patch.object(typed.audit, 'audit') as replay, self.assertRaises(ValueError): typed.run(rows, kind)
            replay.assert_not_called()


if __name__ == '__main__':
    unittest.main()
