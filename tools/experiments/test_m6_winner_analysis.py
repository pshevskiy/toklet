"""Independent full-vector verification and evidence corruption regressions."""
import copy
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest

import m6_winner_analysis as probe


class FullDistribution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = [Path(self.tmp.name)/name for name in ('ref', 'online')]
        self.values = ([3., 2.9921875, -2., 1.], [2.984375, 3.015625, -2., 1.])
        hashes = []
        for path, values in zip(self.paths, self.values, strict=True):
            data = struct.pack('<4f', *values); path.write_bytes(data)
            hashes.append(hashlib.sha256(data).hexdigest())
        # Analytical four-score fixture; independent test arithmetic.
        self.row = dict(target_index=0, target_token=2, prefix_tokens=3968,
            reference_greedy=0, online_greedy=1, reference_margin=1/128,
            reference_winner_logits=[3., 2.984375], online_winner_logits=[2.9921875, 3.015625],
            max_absolute_logit_delta=3/128, rms_logit_delta=math.sqrt(((2/128)**2+(3/128)**2)/4),
            reference_nll=5+math.log(1+math.exp(-1/128)+math.exp(-5)+math.exp(-2)),
            online_nll=5.015625+math.log(1+math.exp(-4/128)+math.exp(-5.015625)+math.exp(-2.015625)),
            reference_logits_sha256=hashes[0], online_logits_sha256=hashes[1])

    def run_analysis(self, row=None):
        return probe.analyze(*self.paths, self.row if row is None else row, vocabulary=4)

    def test_full_vector_reversal_keeps_manual_investigation_open(self):
        result = self.run_analysis()
        self.assertTrue(result['original_distributions_reproduced_exactly'])
        self.assertEqual(result['full_vocabulary_values_verified'], 4)
        self.assertEqual(result['reference_top']['winner'], 0)
        self.assertEqual(result['online_top']['winner'], 1)
        self.assertEqual(result['reference_top']['runner_up'], 1)
        self.assertEqual(result['reference_top']['tie_count'], 1)
        self.assertFalse(result['investigation_complete'])
        self.assertFalse(result['numerical_reversal']['investigation_complete'])

    def test_original_hash_and_exact_length_required(self):
        for data in (b'', b'1234', struct.pack('<5f', 3, 2.9921875, -2, 1, 0),
                     struct.pack('<4f', 3, 2.9921875, -2, 0)):
            self.paths[0].write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'original full-distribution'): self.run_analysis()

    def test_independent_ids_margin_candidates_and_reductions_reject_corruption(self):
        for key, value in (('reference_greedy', 1), ('online_greedy', 0),
                           ('reference_margin', .01), ('reference_winner_logits', [3., 2.98]),
                           ('max_absolute_logit_delta', .03), ('rms_logit_delta', .03),
                           ('reference_nll', 5.), ('online_nll', 5.), ('target_token', True)):
            row = copy.deepcopy(self.row); row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.run_analysis(row)

    def test_nonfinite_correctly_hashed_capture_is_rejected(self):
        data = struct.pack('<4f', 3, float('nan'), -2, 1); self.paths[0].write_bytes(data)
        self.row['reference_logits_sha256'] = hashlib.sha256(data).hexdigest()
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): self.run_analysis()

    def test_tie_order_and_all_ties_counted_without_unbounded_report(self):
        result = probe.top([1.] * 32)
        self.assertEqual(result['winner'], 0); self.assertEqual(result['runner_up'], 1)
        self.assertEqual(result['margin'], 0); self.assertEqual(result['tie_count'], 32)
        self.assertEqual(len(result['first_tied_ids']), 16)
        self.assertEqual(result['top_five_ids'], [0, 1, 2, 3, 4])

    def test_signed_zero_candidate_bits_must_match_captured_payload(self):
        values = ([0., -1/256, -2., -1.], [-1/256, 0., -2., -1.])
        row = copy.deepcopy(self.row)
        for path, distribution, mode in zip(self.paths, values, ('reference', 'online'), strict=True):
            data = struct.pack('<4f', *distribution); path.write_bytes(data)
            row[f'{mode}_logits_sha256'] = hashlib.sha256(data).hexdigest()
            row[f'{mode}_nll'] = 2+math.log(1+math.exp(-1/256)+math.exp(-2)+math.exp(-1))
        row.update(reference_margin=1/256, reference_winner_logits=[0., -1/256],
                   online_winner_logits=[-1/256, 0.], max_absolute_logit_delta=1/256,
                   rms_logit_delta=math.sqrt(2*(1/256)**2/4))
        result = self.run_analysis(row)
        self.assertEqual(result['winner_bits_hex']['reference_winner_logits'][0], '00000000')
        self.assertEqual(result['winner_bits_hex']['online_winner_logits'][1], '00000000')
        for key, index in (('reference_winner_logits', 0), ('online_winner_logits', 1)):
            changed = copy.deepcopy(row); changed[key][index] = -0.
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'pair bits'):
                self.run_analysis(changed)

    def test_invalid_vocabulary_and_malformed_pairs_rejected(self):
        for size in (True, 0, 1, 151937):
            with self.assertRaises(ValueError): probe.distribution(self.paths[0], '', size)
        for pair in (None, [], [3.], [3., 2.984375, 0.]):
            row = copy.deepcopy(self.row); row['reference_winner_logits'] = pair
            with self.assertRaises(ValueError): self.run_analysis(row)


if __name__ == '__main__':
    unittest.main()
