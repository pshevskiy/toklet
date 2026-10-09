"""Generated manifests cannot silently alter the frozen numerical gate."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import llama_check
import prepare


class Anchors(unittest.TestCase):
    def test_shape_size_and_gate_are_anchored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "logits.f32le"
            np.array([[1, 2], [3, 4]], dtype="<f4").tofile(path)
            item = {"file": path.name, "sha256": prepare.digest(path), "shape": [2, 2],
                    "bytes": 16, "reference_envelope": {"atol": 0.001}}
            anchor = {"full_array_sha256": item["sha256"], "shape": [2, 2],
                      "full_array_bytes": 16, "reference_envelope": {"atol": 0.001}}
            np.testing.assert_array_equal(llama_check.reference_logits(root, item, anchor), [3, 4])
            for field, changed in (("shape", [1, 4]), ("bytes", 8),
                                   ("reference_envelope", {"atol": 1000})):
                altered = copy.deepcopy(item)
                altered[field] = changed
                with self.assertRaisesRegex(ValueError, "identity"):
                    llama_check.reference_logits(root, altered, anchor)
            path.write_bytes(path.read_bytes() + b"extra")
            with self.assertRaisesRegex(ValueError, "identity"):
                llama_check.reference_logits(root, item, anchor)


if __name__ == "__main__":
    unittest.main()
