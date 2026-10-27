"""Small independent probability, shift, tie and invalid-input controls."""
import math
import unittest

import qwen3_source_comparison as subject


class SourceComparison(unittest.TestCase):
    def arrays(self, a, b=None, c=None):
        import numpy as np
        return {name: np.array(values, dtype="<f4") for name, values in
                (("source", a), ("fp32", a if b is None else b), ("fp64", a if c is None else c))}

    def test_identical_distributions_and_known_nll(self):
        result = subject.observe(self.arrays([-1, 0, 1]), 1)
        for row in result["profiles"].values():
            self.assertEqual((row["greedy"], row["second"], row["margin"]), (2, 1, 1))
            self.assertAlmostEqual(row["negative_log_likelihood"], math.log(math.exp(-1) + 1 + math.e), places=14)
        for row in result["comparisons"].values():
            self.assertEqual((row["kl_left_right"], row["total_variation"]), (0, 0))

    def test_global_logit_shift_preserves_distribution(self):
        result = subject.observe(self.arrays([99, 100, 101], [-1, 0, 1], [9, 10, 11]), 0)
        pair = result["comparisons"]["source.fp32"]
        self.assertEqual(pair["maximum_absolute"], 100)
        self.assertEqual((pair["kl_left_right"], pair["total_variation"], pair["target_nll_difference"]), (0, 0, 0))

    def test_equal_greedy_does_not_imply_distribution_equivalence(self):
        result = subject.observe(self.arrays([0, 0, -1], [0, -1, 0]), 1)
        pair = result["comparisons"]["source.fp32"]
        self.assertTrue(pair["argmax_equal"])
        self.assertEqual(result["profiles"]["source"]["second"], 1)
        self.assertGreater(pair["kl_left_right"], 0.2)
        self.assertGreater(pair["total_variation"], 0.2)
        self.assertAlmostEqual(pair["target_nll_difference"], -1, places=14)

    def test_invalid_profiles_targets_dtype_and_nonfinite_reject(self):
        import numpy as np
        for rows, target in ((self.arrays([1]), 0), (self.arrays([1, 2]), True),
                             (self.arrays([1, 2]), -1), (self.arrays([1, np.nan]), 0),
                             ({"source": np.array([1, 2], dtype="<f4")}, 0)):
            with self.assertRaises(ValueError):
                subject.observe(rows, target)
        rows = self.arrays([1, 2]);rows["source"] = rows["source"].astype("<f8")
        with self.assertRaises(ValueError):
            subject.observe(rows, 0)


if __name__ == "__main__":
    unittest.main()
