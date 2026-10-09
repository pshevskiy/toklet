"""Offline FP16 oracle/identity checks, without Rust or Apple frameworks."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import fp16


class Fp16Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((fp16.FIXTURE / "manifest.json").read_text())
        cls.source = json.loads((fp16.SOURCE / "manifest.json").read_text())

    def test_exact_regeneration_preserves_pre_target_gates(self):
        self.assertEqual(fp16.verify(), self.manifest)
        for name, item in self.manifest["arrays"].items():
            gate = item["gate_against_source_fp32"]
            if gate:
                actual = fp16.read(fp16.FIXTURE, self.manifest, name).astype(np.float64)
                expected = fp16.read(fp16.SOURCE, self.source, name).astype(np.float64)
                self.assertTrue(np.all(np.abs(actual - expected) <= gate["atol"] + gate["rtol"] * np.abs(expected)), name)
                self.assertEqual(gate["atol"], 2 * gate["independent_fp16_vs_source_max_abs_error"] + 1e-5)
                self.assertEqual(gate["rtol"], 1e-4)

    def test_manual_dots_accumulation_and_large_half_normalization(self):
        x = np.asarray([[2, -1, 4], [-1, 0, 2]], dtype=np.float16)
        w = np.asarray([[1, 2, -1], [0.5, -2, 3]], dtype=np.float16)
        bias = np.asarray([0.25, -0.5], dtype=np.float16)
        for precision in (np.float32, np.float64):
            np.testing.assert_array_equal(fp16.linear(x, w, bias, precision), [[-3.75, 14.5], [-2.75, 5]])
            normalized = fp16.rmsnorm(np.asarray([[65504, -65504]], dtype=np.float16),
                                     np.ones(2, dtype=np.float16), 1e-6, precision)
            np.testing.assert_array_equal(normalized, [[1, -1]])
            np.testing.assert_array_equal(fp16.rmsnorm(np.zeros((1, 2), dtype=np.float16),
                                                     np.ones(2, dtype=np.float16), 1e-6, precision), [[0, 0]])
        terms = np.asarray([10000, 0.001, -10000], dtype=np.float16)
        ones = np.ones(3, dtype=np.float16)
        self.assertEqual(fp16.dot(ones, terms, np.float32), np.float32(0.0009765625))
        self.assertEqual(fp16.dot(ones, terms, np.float64), float(terms[1]))

    def test_manual_causal_gqa_and_split_half_rotary(self):
        q = np.zeros((2, 4, 2), dtype=np.float16)
        k = np.zeros((2, 2, 2), dtype=np.float16)
        v = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16)
        for precision in (np.float32, np.float64):
            _, probabilities, context = fp16.attention(q, k, v, precision)
            np.testing.assert_array_equal(probabilities[:, 0], [[1, 0]] * 4)
            np.testing.assert_array_equal(probabilities[:, 1], [[0.5, 0.5]] * 4)
            np.testing.assert_array_equal(context, [[1, 2, 1, 2, 3, 4, 3, 4], [3, 4, 3, 4, 5, 6, 5, 6]])
            x = np.asarray([[[1, 2, 3, 4]]], dtype=np.float16)
            np.testing.assert_array_equal(fp16.rotary(x, 0, 1e6, precision), x)
            rotated = fp16.rotary(x, 1, 1e6, precision)
            expected = np.asarray([[[np.cos(1) - 3 * np.sin(1), 2 * np.cos(0.001) - 4 * np.sin(0.001),
                                     3 * np.cos(1) + np.sin(1), 4 * np.cos(0.001) + 2 * np.sin(0.001)]]], dtype=np.float16)
            np.testing.assert_array_equal(rotated, expected)

    def test_half_storage_cache_layout_causality_and_greedy_history(self):
        m = self.manifest
        for name in m["arrays"]:
            if name.startswith("weights."):
                original = fp16.read(fp16.SOURCE, self.source, name).astype(np.float16)
                np.testing.assert_array_equal(fp16.read(fp16.FIXTURE, m, name).view(np.uint16), original.view(np.uint16))
        for case, item in m["cases"].items():
            n = len(item["tokens"])
            for layer in range(2):
                for name, value in (("k", "k_rotary"), ("v", "v_projection")):
                    cache = fp16.read(fp16.FIXTURE, m, f"cases.{case}.layers.{layer}.cache_{name}")
                    source = fp16.read(fp16.FIXTURE, m, f"cases.{case}.layers.{layer}.{value}")
                    self.assertEqual(cache.shape, (2, 32, 8))
                    self.assertEqual(cache.dtype, np.float16)
                    np.testing.assert_array_equal(cache[:, :n], source.transpose(1, 0, 2))
                    self.assertEqual(np.count_nonzero(cache[:, n:]), 0)
        prompt = fp16.read(fp16.FIXTURE, m, "cases.prompt.logits")
        changed = fp16.read(fp16.FIXTURE, m, "cases.future_changed.logits")
        np.testing.assert_array_equal(prompt[:3].view(np.uint32), changed[:3].view(np.uint32))
        self.assertEqual(m["greedy"]["source_generated_tokens"], [47] * 4)
        self.assertEqual(m["greedy"]["reconstructed_generated_tokens"], [47] * 4)

    def test_corrupted_payload_enlarged_gate_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "fixture"
            shutil.copytree(fp16.FIXTURE, candidate)
            manifest_path = candidate / "manifest.json"
            m = json.loads(manifest_path.read_text())
            m["arrays"]["cases.prompt.logits"]["gate_against_source_fp32"]["atol"] *= 2
            manifest_path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "oracle/gate differs"):
                fp16.verify(candidate)
            shutil.copyfile(fp16.FIXTURE / "manifest.json", manifest_path)
            payload = candidate / m["arrays"]["weights.embed_tokens"]["file"]
            data = bytearray(payload.read_bytes()); data[0] ^= 1; payload.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "payload differs"):
                fp16.verify(candidate)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                fp16.export(candidate)


if __name__ == "__main__":
    unittest.main()
