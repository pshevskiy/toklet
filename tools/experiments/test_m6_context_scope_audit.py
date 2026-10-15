"""Portable complete-scope checks; synthetic rows cannot qualify a real model."""
import copy
import unittest
from unittest.mock import patch

import m6_context_scope_audit as scope
from test_m6_long_quality_audit import records as quality_records


def records():
    rows = quality_records()
    for model in scope.quality.MODELS:
        for depth in scope.quality.DEPTHS:
            fractions = {'reference': .9, 'online': .15 if model == 'small' else .21}
            rows.append(dict(model=model, plan='profile', depth=depth,
                profiles=[dict(mode=mode, index=i, attention_fraction=fractions[mode])
                          for mode in ('reference', 'online') for i in range(3)], profile_medians=fractions))
            for plan, warmups, measured in (('long', 1, 2), ('long-confirmation', 0, 1)):
                rows.append(dict(model=model, plan=plan, depth=depth, workloads=[dict(name='boundary',
                    warmups=warmups, measured_pairs=measured, summary={'all_raw_samples_replayed': True})]))
    return rows


class ContextScope(unittest.TestCase):
    def test_all32_roles_preserve_descriptive_scope_and_conditional_rule(self):
        value = scope.summarize(records())
        self.assertTrue(value['passed'])
        self.assertTrue(value['complete_frozen_context_coverage'])
        self.assertFalse(value['m6_complete'])
        self.assertFalse(value['kv_retention_qualified'])
        self.assertFalse(value['models']['small']['conditional_kv_investigation_required'])
        self.assertTrue(value['models']['larger']['conditional_kv_investigation_required'])
        for model in value['models'].values():
            for row in model['depths']:
                self.assertFalse(row['performance_ranking_qualified'])
                self.assertEqual(set(row['descriptive_timings']), {'long', 'long-confirmation'})

    def test_missing_duplicate_unknown_plan_or_depth_rejected(self):
        original = records()
        for rows in (original[:-1], original+[original[0]], [original[0]]*32):
            with self.assertRaises(ValueError): scope.summarize(rows)
        for field, value in (('model', 'other'), ('plan', 'matrix'), ('depth', True), ('depth', 2048)):
            rows = records(); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): scope.summarize(rows)

    def test_profile_coverage_median_nonfinite_and_threshold_boundary(self):
        rows = records()
        index = next(i for i, r in enumerate(rows) if r['plan'] == 'profile')
        for mutate in (lambda r: r['profiles'].pop(),
                       lambda r: r['profiles'][0].update(index=1),
                       lambda r: r['profiles'][0].update(attention_fraction=float('nan')),
                       lambda r: r['profile_medians'].update(online=.01)):
            altered = copy.deepcopy(rows); mutate(altered[index])
            with self.assertRaises(ValueError): scope.summarize(altered)
        for row in rows:
            if row['plan'] == 'profile' and row['model'] == 'small' and row['depth'] == 32768:
                row['profile_medians']['online'] = .20
                for p in row['profiles']:
                    if p['mode'] == 'online': p['attention_fraction'] = .20
        self.assertFalse(scope.summarize(rows)['models']['small']['conditional_kv_investigation_required'])

    def test_single_long_pair_cannot_replace_frozen_two_pair_collection(self):
        for plan in ('long', 'long-confirmation'):
            rows = records(); row = next(r for r in rows if r['plan'] == plan)
            row['workloads'][0]['measured_pairs'] += 1
            with self.assertRaises(ValueError): scope.summarize(rows)
        rows = records(); row = next(r for r in rows if r['plan'] == 'long')
        row['workloads'][0]['warmups'] = True
        with self.assertRaises(ValueError): scope.summarize(rows)

    def test_failed_depth_and_winner_reversal_keep_coverage_but_fail_automatic_quality(self):
        for flip in (False, True):
            rows = records(); event = rows[0]['worker']['phases'][0]['events'][1]
            if flip:
                event.update(online_greedy=2, reference_margin=.0078125,
                    reference_winner_logits=[3., 2.984375], online_winner_logits=[2.9921875, 3.015625])
            else:
                for event in rows[0]['worker']['phases'][0]['events'][1:-1]: event['online_nll'] = 2.1
            value = scope.summarize(rows)
            self.assertTrue(value['complete_frozen_context_coverage'])
            self.assertFalse(value['passed'])
            self.assertFalse(value['m6_complete'])

    def test_different_parsed_and_replayed_record_bytes_rejected(self):
        paths = [scope.audit.ROOT/f'artifacts/synthetic-scope-{i}.json' for i in range(32)]
        proof = dict(bindings={str(p.relative_to(scope.audit.ROOT)): 'new' for p in paths})
        with patch.object(scope.audit, 'read_json', return_value=({}, 'old')), \
             patch.object(scope.audit, 'audit', return_value=proof), \
             self.assertRaisesRegex(ValueError, 'differs from replayed bytes'):
            scope.run(paths)
        with patch.object(scope.audit, 'audit') as replay, self.assertRaises(ValueError): scope.run(paths[:-1])
        replay.assert_not_called()


if __name__ == '__main__':
    unittest.main()
