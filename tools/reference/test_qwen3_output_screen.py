"""Independent analytic, shift, tail, greedy and admission controls."""
import copy
import math
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import qwen3_output_screen as subject


class OutputScreen(unittest.TestCase):
    def observe(self, actual, expected):
        import numpy as np
        return subject.observe(np.array(actual, dtype="<f4"), np.array(expected, dtype="<f4"))

    def test_identity_and_exact_lowest_id_ties(self):
        value = self.observe([2, 2, -100], [2, 2, -100])
        self.assertTrue(value["passed"])
        self.assertEqual(value["native_greedy"], 0)
        self.assertEqual(value["maximum_absolute_log_probability_error"], 0)
        self.assertAlmostEqual(value["native_log_normalizer"], math.log(2))

    def test_constant_shift_is_removed(self):
        value = self.observe([129, 130, 131], [1, 2, 3])
        self.assertTrue(value["passed"])
        self.assertEqual(value["maximum_absolute_log_probability_error"], 0)

    def test_analytic_two_entry_distribution(self):
        value = self.observe([1, 0], [0, 0])
        expected = 1 + math.log1p(math.exp(-1)) - math.log(2)
        self.assertAlmostEqual(value["maximum_absolute_log_probability_error"], expected, places=14)
        self.assertEqual(value["maximum_vocabulary_id"], 1)
        self.assertEqual(value["failed_vocabulary_entries"], 2)

    def test_low_probability_tail_cannot_hide_failure(self):
        value = self.observe([0, -1000, -1000.125], [0, -1000, -1000])
        self.assertTrue(value["greedy_equal"])
        self.assertFalse(value["passed"])
        self.assertEqual(value["failed_vocabulary_entries"], 1)
        self.assertEqual(value["maximum_absolute_log_probability_error"], 0.125)

    def test_greedy_mismatch_rejects_even_when_error_passes(self):
        value = self.observe([0, 0.0625], [0, 0])
        self.assertLess(value["maximum_absolute_log_probability_error"], 0.05)
        self.assertEqual(value["failed_vocabulary_entries"], 0)
        self.assertFalse(value["passed"])

    def test_extreme_finite_logits_remain_finite(self):
        import numpy as np
        maximum = np.finfo(np.float32).max
        value = self.observe([maximum, -maximum], [maximum, -maximum])
        self.assertTrue(value["passed"])

    def test_invalid_saved_rows_rejected(self):
        import numpy as np
        good = np.array([0, 1], dtype="<f4")
        for bad in ([0], [float("nan"), 0], [float("inf"), 0], [[0, 1]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                subject.observe(np.array(bad, dtype="<f4"), good)
        with self.assertRaises(ValueError):
            subject.observe(good.astype(np.float64), good)

    def test_contract_geometry_tokens_and_typed_criterion(self):
        value = dict(format=subject.FORMAT, positions=512, vocabulary=151936,
                     criterion=dict(maximum_absolute_log_probability_error=0.05,
                         inclusive=True, exact_lowest_id_greedy=True, nonfinite_rejected=True),
                     native_recipe=dict(name="chunk128", chunk=128, segments=[512]),
                     cases=[dict(id="original", tokens=[9707, 11, 1879, 0] * 128),
                            dict(id="future_tail", tokens=([9707, 11, 1879, 0] * 128)[:-1] + [17])])
        self.assertIs(subject.validate_contract(value), value)
        changes = [("positions", 1025), ("vocabulary", 100), ("format", "unversioned")]
        for key, replacement in changes:
            bad = copy.deepcopy(value);bad[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_contract(bad)
        for key, replacement in (("inclusive", 1), ("maximum_absolute_log_probability_error", 0.06)):
            bad = copy.deepcopy(value);bad["criterion"][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_contract(bad)
        bad = copy.deepcopy(value);bad["cases"][1]["tokens"][-1] = 0
        with self.assertRaises(ValueError):
            subject.validate_contract(bad)

    def test_interruption_retains_partial_report_and_restores_handlers(self):
        import json
        for number in (signal.SIGTERM, signal.SIGINT):
            before = {item: signal.getsignal(item) for item in (signal.SIGTERM, signal.SIGINT)}
            with self.subTest(number=number), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "result.json"
                def interrupted(root, contract_path, contract_pin, journal, report):
                    report.update(complete=False, journal=dict(rows=3))
                    signal.raise_signal(number)
                arguments = ["screen", "--contract", "unused", "--contract-sha", "0" * 64,
                             "--contract-bytes", "1", "--journal", "unused", "--output", str(output)]
                with patch.object(subject.sys, "argv", arguments), patch.object(subject, "screen", interrupted):
                    with self.assertRaises(InterruptedError):
                        subject.main()
                value = json.loads(output.read_bytes())
                self.assertFalse(value["passed"])
                self.assertEqual(value["report"]["journal"]["rows"], 3)
                self.assertIn("InterruptedError", value["primary_error"])
                self.assertEqual({item: signal.getsignal(item) for item in before}, before)

    def test_unadmitted_actual_numerical_import_rejects(self):
        import numpy
        with self.assertRaisesRegex(ValueError, "Unadmitted numerical module origin"):
            subject.runtime_origins(Path.cwd(), dict(screen_runtime_and_sources={}, required_import_origins={}))

    def test_exact_empty_runtime_markers_and_replacement_rejects(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "marker";path.touch()
            pin = dict(bytes=0, sha256=hashlib.sha256(b"").hexdigest());audits = []
            identity = subject.runtime_file(path, pin, audits)
            subject.runtime_endpoint(path, pin, identity, audits)
            self.assertTrue(all(audit["endpoint_checked"] for audit in audits))
            for bad in (dict(bytes=False, sha256=pin["sha256"]), dict(bytes=0, sha256="0" * 64)):
                with self.assertRaises(ValueError):
                    subject.runtime_file(path, bad, [])
            alias = Path(folder) / "alias";alias.symlink_to(path)
            with self.assertRaises(OSError):
                subject.runtime_file(alias, pin, [])
            path.write_bytes(b"x")
            with self.assertRaises(ValueError):
                subject.runtime_endpoint(path, pin, identity, [])


if __name__ == "__main__":
    unittest.main()
