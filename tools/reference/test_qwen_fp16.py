"""Offline independent Qwen precision math; no weights, Torch or Apple required."""
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import numpy as np
import fp16
import qwen_fp16 as oracle


class QwenFp16Checks(unittest.TestCase):
    def test_manual_linear_bias_stripes_and_half_ranges(self):
        x = np.asarray([[2, -1, 4], [-1, 0, 2]], dtype=np.float16)
        w = np.asarray([[1, 2, -1], [0.5, -2, 3]], dtype=np.float16)
        bias = np.asarray([0.25, -0.5], dtype=np.float16)
        for precision in (np.float32, np.float64):
            np.testing.assert_array_equal(oracle.linear(x, w, bias, precision), [[-3.75, 14.5], [-2.75, 5]])
            striped = np.tile(w[:1], (oracle.ROW_STRIPE + 1, 1))
            np.testing.assert_array_equal(oracle.linear(x, striped, np.full(len(striped), .25, dtype=np.float16), precision),
                                          np.tile(np.asarray([[-3.75], [-2.75]], dtype=np.float16), (1, len(striped))))
            np.testing.assert_array_equal(oracle.rmsnorm(np.asarray([[65504, -65504]], dtype=np.float16),
                                                        np.ones(2, dtype=np.float16), 1e-6, precision), [[1, -1]])
            np.testing.assert_array_equal(oracle.rmsnorm(np.zeros((1, 2), dtype=np.float16),
                                                        np.ones(2, dtype=np.float16), 1e-6, precision), [[0, 0]])
        # Rounding a dot before bias changes this result. The oracle adds bias first.
        tiny = np.asarray([[1, 1]], dtype=np.float16)
        weight = np.asarray([[1, 2**-11]], dtype=np.float16)
        bias = np.asarray([2**-11], dtype=np.float16)
        self.assertEqual(float(oracle.linear(tiny, weight, bias, np.float32)[0, 0]), 1 + 2**-10)

    def test_manual_causal_gqa_and_offset_rotary(self):
        q = np.zeros((2, 4, 2), dtype=np.float16)
        k = np.zeros((2, 2, 2), dtype=np.float16)
        v = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16)
        for precision in (np.float32, np.float64):
            scores, probabilities, context = oracle.attention(q, k, v, precision)
            self.assertEqual(scores.dtype, np.float32)
            self.assertEqual(probabilities.dtype, np.float32)
            self.assertEqual(context.dtype, np.float16)
            np.testing.assert_array_equal(probabilities[:, 0], [[1, 0]] * 4)
            np.testing.assert_array_equal(probabilities[:, 1], [[.5, .5]] * 4)
            np.testing.assert_array_equal(context, [[1, 2, 1, 2, 3, 4, 3, 4], [3, 4, 3, 4, 5, 6, 5, 6]])
            x = np.asarray([[[1, 2, 3, 4]]], dtype=np.float16)
            np.testing.assert_array_equal(oracle.rotary(x, 0, 1e6, precision), x)
            expected = np.asarray([[[np.cos(1) - 3*np.sin(1), 2*np.cos(.001) - 4*np.sin(.001),
                                    3*np.cos(1) + np.sin(1), 4*np.cos(.001) + 2*np.sin(.001)]]], dtype=np.float16)
            np.testing.assert_array_equal(oracle.rotary(x, 1, 1e6, precision), expected)

    def test_complete_tiny_against_independent_scalar_and_source_gates(self):
        source = json.loads((fp16.SOURCE / "manifest.json").read_text())
        fixture = json.loads((fp16.FIXTURE / "manifest.json").read_text())
        weights = {name.removeprefix("weights."): fp16.read(fp16.FIXTURE, fixture, name)
                   for name in fixture["arrays"] if name.startswith("weights.")}
        for case, spec in source["cases"].items():
            for precision in (np.float32, np.float64):
                actual = oracle.forward(spec["tokens"], spec["position_offset"], weights, source["model"], precision, [0, 1])
                scalar = fp16.forward(spec["tokens"], spec["position_offset"], weights, source["model"], precision)
                self.assertEqual(set(actual), set(scalar))
                for name, value in actual.items():
                    # Projection storage is [tokens,features] in the real-model source.
                    reference = scalar[name].reshape(value.shape)
                    source_key = f"cases.{case}.{name}"
                    expected = fp16.read(fp16.SOURCE, source, source_key).reshape(value.shape)
                    gate = fixture["arrays"][source_key]["gate_against_source_fp32"]
                    np.testing.assert_allclose(value, reference, atol=gate["atol"], rtol=gate["rtol"], err_msg=source_key)
                    self.assertTrue(np.all(np.abs(value.astype(np.float64)-expected) <= gate["atol"]+gate["rtol"]*np.abs(expected)), source_key)
        prompt = oracle.forward(source["cases"]["prompt"]["tokens"], 0, weights, source["model"], np.float32, [0, 1])
        changed = oracle.forward(source["cases"]["future_changed"]["tokens"], 0, weights, source["model"], np.float32, [0, 1])
        np.testing.assert_array_equal(prompt["logits"][:3], changed["logits"][:3])
        self.assertEqual(np.argmax(prompt["logits"][-1]), 47)

    def test_gates_pool_all_cases_and_only_exact_position_prefixes(self):
        cases = {"a": {"tokens": [1, 2], "offset": 0}, "b": {"tokens": [1, 3], "offset": 0},
                 "c": {"tokens": [1], "offset": 9}}
        reconstructed, source = {}, {}
        for case, value in (("a", 1), ("b", 2), ("c", 20)):
            n = len(cases[case]["tokens"])
            for name, shape in (("logits", (n, 2)), ("layers.0.attention_scores", (2, n, n))):
                key = f"cases.{case}.{name}"
                reconstructed[key] = (np.full(shape, value, np.float32), np.full(shape, value+.25, np.float32))
                source[key] = np.zeros(shape, np.float32)
        gates = oracle.gates(reconstructed, source, cases)
        for gate in gates.values():
            self.assertEqual(gate["pooled_fp16_vs_source_max_abs_error"], 20.25)
            self.assertEqual(gate["equivalent_prefix_max_abs_error"], 1)
            self.assertEqual(gate["atol"], 40.5)
            self.assertEqual(gate["rtol"], 1e-4)

    def test_overlapping_or_existing_exports_fail_before_source_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for full, probes in ((root, root / "probes"), (root / "same", root / "same"),
                                 (root / "full", root / "full" / "probes"), (root / "probes" / "full", root / "probes")):
                with self.assertRaisesRegex(ValueError, "existing or overlapping"):
                    oracle.export(full, probes)

    def test_frozen_real_probes_and_corruption_without_source_weights(self):
        manifest = oracle.verify()
        self.assertEqual(len(manifest["arrays"]), 228)
        self.assertEqual(manifest["logical_fp16_weights_bytes"], 988065536)
        self.assertEqual(manifest["maximum_weight_conversion_stripe_bytes"], 34865152)
        for argmax in manifest["argmax"].values():
            self.assertEqual(argmax["source_fp32"], argmax["fp16_fp32_arithmetic"])
            self.assertEqual(argmax["source_fp32"], argmax["fp16_fp64_arithmetic"])
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "probes"
            shutil.copytree(oracle.FIXTURE, candidate)
            path = candidate / "manifest.json"
            edited = json.loads(path.read_text())
            edited["arrays"]["cases.short.logits"]["gate_against_source_fp32"]["atol"] *= 2
            path.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError, "source/gate differs"):
                oracle.verify(candidate)
            path.write_text(json.dumps(manifest))
            payload = candidate / manifest["arrays"]["cases.short.logits"]["file"]
            payload.write_bytes(payload.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "payload differs"):
                oracle.verify(candidate)


if __name__ == "__main__":
    unittest.main()
