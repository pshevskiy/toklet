"""Offline admission/integrity checks; Core ML is never imported."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import coreml_probe as probe


class ProbeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((probe.oracle.FIXTURE / "manifest.json").read_text())

    def test_frozen_gate_and_output_shape_finiteness(self):
        gate = self.manifest["cases"]["single"]["attention_fp16_gate"]
        expected = probe.oracle.read(probe.oracle.FIXTURE, self.manifest, "single.context_expected")
        self.assertEqual(probe.check(expected, expected, gate)["maximum_tolerance_fraction"], 0)
        for actual in (expected.reshape(32), np.full_like(expected, np.nan), expected + 1):
            with self.assertRaises(ValueError):
                probe.check(actual, expected, gate)

    def test_prepared_positions_and_head_major_updates_are_bounded(self):
        items = probe.inputs(self.manifest, "capacity", "stateful_attention")
        self.assertEqual(len(items), 32)
        for position, item in enumerate(items):
            self.assertEqual(item["position"].tolist(), [position])
            self.assertEqual(item["position"].dtype, np.int32)
            self.assertEqual(item["q"].shape, (2, 2, 8))
            self.assertEqual(item["k"].shape, (2, 1, 8))
            self.assertEqual(item["v"].shape, (2, 1, 8))
            self.assertEqual(item["q"].dtype, np.float16)
        bad = json.loads(json.dumps(self.manifest))
        bad["cases"]["capacity"]["tokens"] = 33
        with self.assertRaisesRegex(ValueError, "admission"):
            probe.inputs(bad, "capacity", "stateful_attention")

    def test_conversion_identity_changes_and_interrupted_retry_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "weights.bin"
            file.write_bytes(b"original")
            before = probe.directory_hash(root)
            file.write_bytes(b"changed")
            self.assertNotEqual(before, probe.directory_hash(root))
            output = root / "run.json"
            output.with_suffix(".json.samples.jsonl").write_text("incomplete\n")
            with patch.object(probe, "load_coreml") as load:
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    probe.run(root, output, "stateful_attention", "cpu_only")
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    probe.prepare(root)
                load.assert_not_called()

    def test_promoted_state_readback_requires_exact_values_without_rounding(self):
        values = {f"cache_{name}": probe.oracle.read(probe.oracle.FIXTURE, self.manifest,
                   f"single.cache_{name}_expected").astype(np.float32) for name in ("k", "v")}

        class State:
            def read_state(self, name):
                return values[name]

        self.assertEqual(probe.check_cache(State(), self.manifest, "single"),
                         {"k": "float32", "v": "float32"})
        # A one-ULP FP32 change would disappear after rounding to FP16. It must
        # still fail; the readback adapter does not loosen the cache gate.
        values["cache_k"][0, 0, 0] = np.nextafter(values["cache_k"][0, 0, 0], np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, "cache differs"):
            probe.check_cache(State(), self.manifest, "single")

    def test_poison_installation_is_verified_before_prediction(self):
        class State:
            def __init__(self, ignore_writes=False):
                self.ignore_writes = ignore_writes
                self.values = {f"cache_{name}": np.zeros((2, 32, 8), dtype=np.float32)
                               for name in ("k", "v")}

            def write_state(self, name, value):
                self.assert_dtype(value)
                if not self.ignore_writes:
                    self.values[name] = value.copy()

            @staticmethod
            def assert_dtype(value):
                if value.dtype != np.float32:
                    raise ValueError("Public write API requires FP32")

            def read_state(self, name):
                return self.values[name]

        probe.poison_caches(State())
        with self.assertRaisesRegex(ValueError, "installation differs"):
            probe.poison_caches(State(ignore_writes=True))


if __name__ == "__main__":
    unittest.main()
