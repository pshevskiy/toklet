"""Independent slice-oracle integrity checks; never import Core ML."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import coreml_slice


class FrozenSlices(unittest.TestCase):
    def test_independent_export_exactly_reproduces_frozen_gates_and_arrays(self):
        manifest = coreml_slice.verify()
        self.assertEqual(len(manifest["arrays"]), 42)
        self.assertEqual(manifest["cases"]["capacity"]["tokens"], 32)
        self.assertEqual(manifest["cases"]["offset"]["absolute_rotary_offset"], 5)
        for case in manifest["cases"].values():
            self.assertEqual(case["projection_fp32_gate"]["atol"], 1e-5)
            for name in ("projection_fp16_gate", "attention_fp16_gate"):
                self.assertEqual(case[name]["atol"], 2 * case[name]["independent_max_abs_error"] + 1e-5)

    def test_corrupt_payload_and_enlarged_gate_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            shutil.copytree(coreml_slice.FIXTURE, root)
            path = root / "manifest.json"
            original = path.read_text()
            manifest = json.loads(original)
            manifest["cases"]["prompt"]["attention_fp16_gate"]["atol"] += 0.1
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "oracle/gate"):
                coreml_slice.verify(root)
            path.write_text(original)
            payload = root / manifest["arrays"]["prompt.query"]["file"]
            value = bytearray(payload.read_bytes())
            value[0] ^= 1
            payload.write_bytes(value)
            with self.assertRaisesRegex(ValueError, "identity"):
                coreml_slice.verify(root)

    def test_export_never_overwrites_frozen_evidence(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            coreml_slice.export(coreml_slice.FIXTURE)


if __name__ == "__main__":
    unittest.main()
