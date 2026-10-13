import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import qwen_storage_oracle as oracle


class StorageOracleTests(unittest.TestCase):
    def test_manifest_publication_follows_complete_native_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, failure in enumerate((ValueError("over budget"), OSError("native unavailable"))):
                output = root / str(index)
                output.mkdir()
                if isinstance(failure, ValueError):
                    side_effect = None
                    value = {"lifetime_max_phys_footprint":7*1024**3}
                else:
                    side_effect, value = failure, None
                with mock.patch.object(oracle.native_usage,"usage",return_value=value,side_effect=side_effect):
                    with self.assertRaises(type(failure)):
                        oracle.publish_manifest(output,{"test":True},{"native_samples":[]})
                self.assertTrue((output/".manifest.pending").exists())
                self.assertFalse((output/"manifest.json").exists())
            output = root / "success"
            output.mkdir()
            receipt = {"native_samples":[]}
            with mock.patch.object(oracle.native_usage,"usage",return_value={"lifetime_max_phys_footprint":1234}):
                identity = oracle.publish_manifest(output,{"test":True},receipt)
            self.assertEqual(identity,oracle.storage.sha(output/"manifest.json"))
            self.assertFalse((output/".manifest.pending").exists())
            self.assertEqual(receipt["native_samples"][-1]["phase"],"completed")

    def test_output_and_receipt_overlap_rejected_before_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            for receipt in (output, output / "manifest.json", output / "cases.logits.f32le", output.parent):
                with mock.patch.object(oracle.storage, "bounded_bytes") as read:
                    with self.assertRaisesRegex(ValueError, "overlapping"):
                        oracle.export(Path("Qwen2.5-1.5B-Instruct"), Path("unused"), None, output, receipt)
                    read.assert_not_called()

    def test_altered_source_manifest_rejected_before_forward_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text('{"cases":{},"layers":[]}')
            with mock.patch.object(oracle.qwen_fp16, "forward") as forward:
                with self.assertRaisesRegex(ValueError, "identity"):
                    oracle.export(Path("Qwen2.5-1.5B-Instruct"), root, None, root / "out", root / "receipt.json")
                forward.assert_not_called()
            self.assertFalse((root / "out").exists())
            self.assertFalse((root / "receipt.json").exists())

    def test_q4_pooled_precision_and_equivalent_prefix_bounds(self):
        cases = {"a": {"tokens": [1,2], "offset": 0}, "b": {"tokens": [1,3], "offset": 0}}
        reconstructed = {
            "cases.a.final_norm": (np.array([[1],[2]],dtype="f2"), np.array([[1.125],[2]],dtype="f2")),
            "cases.b.final_norm": (np.array([[1.25],[4]],dtype="f2"), np.array([[1.25],[4]],dtype="f2")),
        }
        limits = oracle.q4_gates(reconstructed, cases)
        for value in limits.values():
            self.assertEqual(value["pooled_fp32_fp64_max_abs"], .125)
            self.assertEqual(value["equivalent_prefix_max_abs"], .25)
            self.assertEqual(value["atol"], .5)
            self.assertEqual(value["rtol"], .001)
        # Different absolute positions cannot contribute equivalent-prefix error.
        cases["b"]["offset"] = 5
        for value in oracle.q4_gates(reconstructed,cases).values():
            self.assertEqual(value["atol"], .25)
            self.assertEqual(value["equivalent_prefix_max_abs"], 0)


if __name__ == "__main__":
    unittest.main()
