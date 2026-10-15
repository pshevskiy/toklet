"""Portable coverage, pooled-gate and saved-winner consistency checks."""
import unittest
from unittest.mock import patch

import m6_long_quality_audit as quality


def row(index=0, depth=4096):
    return dict(event='comparison_row', target_index=index, target_token=7,
                prefix_tokens=depth-128+index, reference_nll=2., online_nll=2.01,
                reference_greedy=1, online_greedy=1, reference_margin=.1,
                reference_winner_logits=[3., 3.01], online_winner_logits=[3., 3.01],
                max_absolute_logit_delta=.1, rms_logit_delta=.01)


def records():
    return [dict(model=model, depth=depth, plan='quality', worker=dict(phases=[
        dict(request=dict(action='compare'), events=[dict(event='comparison_begin'),
             *(row(i, depth) for i in range(128)), dict(event='comparison_done')])]))
        for model in quality.MODELS for depth in quality.DEPTHS]


class LongQuality(unittest.TestCase):
    def test_all_eight_depths_pool_exactly_512_per_model(self):
        value = quality.summarize(records())
        self.assertTrue(value['passed'])
        self.assertFalse(value['m6_complete'])
        for model in value['models'].values():
            self.assertEqual(model['scored_targets'], 512)
            self.assertAlmostEqual(model['pooled_mean_nll_increase'], .01)

    def test_missing_duplicate_unknown_nonquality_and_target_coverage_reject(self):
        original = records()
        for values in (original[:-1], original+[original[0]], [original[0]]*8):
            with self.assertRaises(ValueError): quality.summarize(values)
        for field, value in (('model', 'other'), ('depth', True), ('depth', 2048), ('plan', 'profile')):
            values = records(); values[0][field] = value
            with self.assertRaises(ValueError): quality.summarize(values)
        for mutate in (lambda e: e.pop(1), lambda e: e[1].update(target_index=True),
                       lambda e: e[2].update(target_index=0), lambda e: e[1].update(prefix_tokens=0)):
            values = records(); mutate(values[0]['worker']['phases'][0]['events'])
            with self.assertRaises(ValueError): quality.summarize(values)

    def test_failing_depth_cannot_hide_behind_passing_pooled_mean(self):
        values = records()
        for event in values[0]['worker']['phases'][0]['events'][1:-1]:
            event['online_nll'] = 2.1
        result = quality.summarize(values)
        self.assertTrue(result['models']['small']['pooled_gate_passed'])
        self.assertFalse(result['models']['small']['each_depth_gate_passed'])
        self.assertFalse(result['passed'])
        for record in values:
            for event in record['worker']['phases'][0]['events'][1:-1]:
                event['online_nll'] = 2.1
        self.assertFalse(quality.summarize(values)['models']['small']['pooled_gate_passed'])

    def test_reversal_reports_both_shifts_and_requires_review(self):
        values = records(); event = values[0]['worker']['phases'][0]['events'][1]
        event.update(online_greedy=2, reference_margin=.0078125,
                     reference_winner_logits=[3., 2.984375], online_winner_logits=[2.9921875, 3.015625])
        result = quality.summarize(values)
        change = result['models']['small']['depths'][0]['argmax_changes'][0]
        self.assertEqual(change['reference_candidate_gap'], .0078125)
        self.assertEqual(change['online_reversed_gap'], .03125)
        self.assertEqual(change['reference_winner_logit_shift'], -.015625)
        self.assertEqual(change['online_winner_logit_shift'], .0234375)
        self.assertEqual(change['differential_shift'], .0390625)
        self.assertTrue(change['numerical_reversal_accounted'])
        self.assertFalse(change['investigation_complete'])
        self.assertTrue(result['frozen_nll_gates_passed'])
        self.assertFalse(result['argmax_investigations_complete'])
        self.assertFalse(result['passed'])

    def test_equal_id_pair_margin_and_lowest_id_tie_inconsistencies_reject(self):
        event = row(); event['online_winner_logits'] = [2.99, 3.01]
        with self.assertRaisesRegex(ValueError, 'Matching winner'): quality.winner_accounting(event)
        event = row(); event.update(online_greedy=2, reference_margin=.1,
                                    online_winner_logits=[2.99, 3.02])
        with self.assertRaisesRegex(ValueError, 'Top-two margin'): quality.winner_accounting(event)
        for a, b, ar, br in ((2, 1, [3., 2.99], [3., 3.01]),
                             (1, 2, [3., 3.01], [2.99, 3.01])):
            event = row(); event.update(reference_greedy=a, online_greedy=b,
                                        reference_margin=0., reference_winner_logits=ar,
                                        online_winner_logits=br)
            with self.assertRaisesRegex(ValueError, 'lowest-ID'): quality.winner_accounting(event)
        # A tie is legal when its selected winner is the lower ID.
        event = row(); event.update(online_greedy=2, reference_margin=0.,
                                    reference_winner_logits=[3., 2.99], online_winner_logits=[3., 3.01])
        self.assertIsNotNone(quality.winner_accounting(event))

    def test_nonfinite_invalid_errors_and_winner_shifts_reject(self):
        for key, value in (('reference_margin', float('nan')), ('rms_logit_delta', .2),
                           ('max_absolute_logit_delta', -.1), ('reference_greedy', True),
                           ('reference_winner_logits', [3., 3.5]), ('online_nll', float('inf'))):
            values = records(); values[0]['worker']['phases'][0]['events'][1][key] = value
            with self.assertRaises(ValueError): quality.summarize(values)

    def test_parsed_collection_must_match_replayed_bytes(self):
        paths = [quality.audit.ROOT/f'artifacts/synthetic-{i}.json' for i in range(8)]
        proof = dict(bindings={str(path.relative_to(quality.audit.ROOT)): 'new' for path in paths})
        with patch.object(quality.audit, 'read_json', return_value=({}, 'old')), \
             patch.object(quality.audit, 'audit', return_value=proof), \
             self.assertRaisesRegex(ValueError, 'Parsed collection'):
            quality.run(paths)
        with patch.object(quality.audit, 'audit') as replay, self.assertRaises(ValueError):
            quality.run(paths[:-1])
        replay.assert_not_called()


if __name__ == '__main__': unittest.main()
