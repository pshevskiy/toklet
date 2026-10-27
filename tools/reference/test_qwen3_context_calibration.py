"""Small capture/retention controls; no downloads, Metal or full model graph."""
import hashlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import qwen3_context_calibration as subject

FIXTURE = Path(__file__).resolve().parents[2] / "benchmarks/qwen3-context-candidate-v2"


class CalibrationCapture(unittest.TestCase):
    def test_calibration_only_reads_and_exact_demand(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name.startswith("val-"):
                raise AssertionError("Validation input opened before gate freeze")
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "open", guarded):
            manifest, selected = subject.selected_inputs(FIXTURE)
            plan = subject.plan(FIXTURE)
        self.assertEqual(len(selected), 6)
        self.assertEqual(sum(len(row["tokens"]) for row in selected), 1344)
        self.assertEqual(len(manifest["stage_pooling"]), 57)
        self.assertEqual(plan["total_capture_bytes"], 4 * sum(plan["per_graph_bytes"].values()))
        self.assertFalse(plan["validation_evaluated"] or plan["gates_derived"] or plan["adopted"])
        self.assertLessEqual(plan["memory"]["admitted"], plan["limit"])

    def arrays(self):
        import numpy as np
        return dict(a=np.array([[1, -0.0], [2, 3]], dtype="<f2"),
                    b=np.array([[4, 5], [6, 7]], dtype="<f4"))

    def test_complete_typed_rows_hashes_and_no_replacement(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(subject.reference, "geometry", return_value={"a": ([2, 2], "<f2"), "b": ([2, 2], "<f4")}):
            folder = Path(temporary);record = {}
            subject.save_graph(folder, "fixture", self.arrays(), 2, record)
            body = (folder / "fixture.bin").read_bytes()
            self.assertTrue(record["complete"])
            self.assertEqual(record["bytes"], 24)
            self.assertEqual(record["sha256"], hashlib.sha256(body).hexdigest())
            self.assertEqual(body[2:4], b"\x00\x80")
            for item in record["arrays"].values():
                span = body[item["offset"]:item["offset"] + item["bytes"]]
                self.assertEqual(hashlib.sha256(span).hexdigest(), item["sha256"])
            with self.assertRaises(FileExistsError):
                subject.save_graph(folder, "fixture", self.arrays(), 2, {})

    def test_nonfinite_array_retained_before_numerical_rejection(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(subject.reference, "geometry", return_value={"a": ([2, 2], "<f2"), "b": ([2, 2], "<f4")}):
            arrays = self.arrays();arrays["b"][1, 0] = np.nan
            record = {};folder = Path(temporary)
            with self.assertRaisesRegex(ValueError, "finite"):
                subject.save_graph(folder, "fixture", arrays, 2, record)
            self.assertFalse(record["complete"])
            self.assertEqual(set(record["arrays"]), {"a", "b"})
            self.assertEqual(record["bytes"], 24)
            self.assertEqual(record["arrays"]["b"]["nonfinite_values"], 1)
            self.assertTrue(record["arrays"]["b"]["complete"])
            self.assertEqual((folder / "fixture.bin").read_bytes(), arrays["a"].tobytes() + arrays["b"].tobytes())

    def test_partial_write_tracks_actual_prefix_and_current_array(self):
        original = Path.open
        class Partial:
            def __init__(self, path):
                self.file = original(path, "xb")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.file.close()
            def write(self, body):
                self.file.write(body[:2]);return 2
        def opened(path, *args, **kwargs):
            return Partial(path) if path.suffix == ".bin" and args == ("xb",) else original(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(subject.reference, "geometry", return_value={"a": ([2, 2], "<f2"), "b": ([2, 2], "<f4")}), mock.patch.object(Path, "open", opened):
            record = {};folder = Path(temporary)
            with self.assertRaisesRegex(OSError, "Incomplete"):
                subject.save_graph(folder, "fixture", self.arrays(), 2, record)
            self.assertEqual(record["bytes"], 2)
            self.assertEqual(record["arrays"]["a"]["bytes"], 2)
            self.assertEqual(record["arrays"]["a"]["intended_bytes"], 8)
            self.assertFalse(record["arrays"]["a"]["complete"])
            self.assertEqual(record["sha256"], hashlib.sha256((folder / "fixture.bin").read_bytes()).hexdigest())

    def test_export_failure_preserves_prior_capture_attempt_and_closes_owner(self):
        import mlx_qwen3_real as source
        import qwen3_fp16
        owner = SimpleNamespace(closed=False)
        class Weights:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                owner.closed = True
        prepared = dict(memory=dict(admitted=1), cases=[dict(id="mini", rows=2, tokens=[1, 2])], total_capture_bytes=48)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(subject, "plan", return_value=prepared), mock.patch.object(source, "Weights", return_value=Weights()), mock.patch.object(qwen3_fp16, "forward", side_effect=[self.arrays(), RuntimeError("later forward failure")]), mock.patch.object(subject.reference, "geometry", return_value={"a": ([2, 2], "<f2"), "b": ([2, 2], "<f4")}), mock.patch.dict(sys.modules, native_usage=SimpleNamespace(usage=lambda: dict(lifetime_max_phys_footprint=1))):
            folder = Path(temporary) / "capture";report = {}
            with self.assertRaisesRegex(RuntimeError, "later forward"):
                subject.export(folder, None, report)
            saved = json.loads((folder / "manifest.json").read_bytes())
            self.assertTrue(owner.closed)
            self.assertEqual(saved, report)
            self.assertIn("later forward", report["primary_error"])
            self.assertTrue(report["records"]["mini.fp32"]["complete"])
            self.assertFalse(report["attempts"][1]["completed"])
            self.assertTrue((folder / "mini.fp32.bin").is_file())

    def test_shape_dtype_and_inventory_reject(self):
        import numpy as np
        changes = [lambda v: v.pop("b"), lambda v: v.update(a=np.ones((1, 4), dtype="<f2")),
                   lambda v: v.update(a=v["a"].astype("<f4"))]
        for change in changes:
            with tempfile.TemporaryDirectory() as temporary, mock.patch.object(subject.reference, "geometry", return_value={"a": ([2, 2], "<f2"), "b": ([2, 2], "<f4")}):
                arrays = self.arrays();change(arrays)
                with self.assertRaises(ValueError):
                    subject.save_graph(Path(temporary), "fixture", arrays, 2, {})

    def miniature(self, folder):
        plan = dict(cases=[dict(id="mini", rows=2)], total_capture_bytes=48)
        manifest = dict(plan=plan, complete=True, primary_error=None, records={}, attempts=[])
        for contract in ("fp32", "fp64"):
            label = "mini." + contract
            record = dict(case="mini", contract=contract, rows=2)
            subject.save_graph(folder, label, self.arrays(), 2, record)
            manifest["records"][label] = record
            manifest["attempts"].append(dict(label=label, completed=True))
        return plan, manifest

    def test_complete_streaming_replay_and_coherent_schema_mutations(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(subject.reference, "geometry", return_value={"a": ([2, 2], "<f2"), "b": ([2, 2], "<f4")}):
            folder = Path(temporary);plan, original = self.miniature(folder)
            def replay(manifest, report):
                body = json.dumps(manifest).encode();(folder / "manifest.json").write_bytes(body)
                return subject.verify_capture(folder, hashlib.sha256(body).hexdigest(), plan, report)
            report = {};self.assertEqual(replay(original, report), original["records"])
            self.assertTrue(report["complete"])
            self.assertEqual((report["bytes"], report["values"], len(report["arrays"])), (48, 16, 4))
            self.assertTrue(all(row["endpoint_checked"] for row in report["files"]))
            for change in (lambda v: v["records"]["mini.fp32"].update(rows=True),
                    lambda v: v["records"]["mini.fp32"]["arrays"]["a"].update(offset=False),
                    lambda v: v["records"]["mini.fp32"]["arrays"]["b"].update(sha256="0" * 64),
                    lambda v: v["attempts"].pop()):
                manifest = copy.deepcopy(original);change(manifest);partial = {}
                with self.assertRaises(ValueError):
                    replay(manifest, partial)
                self.assertFalse(partial["complete"])


if __name__ == "__main__":
    unittest.main()
