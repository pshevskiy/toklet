"""Portable layout/reader/comparison checks; no checkpoint or GPU execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import ExitStack

import numpy as np
import qwen3_long_context as reference


def manifest():
    arrays = {}
    cursor = 0
    for name, (shape, dtype) in sorted(reference.geometry(512).items()):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        arrays[name] = dict(shape=shape, dtype=dtype, offset=cursor, bytes=size, sha256="0" * 64)
        cursor += size
    records = {case + "." + contract: dict(file=case + "." + contract + ".bin",
               bytes=cursor, sha256="0" * 64, arrays=copy.deepcopy(arrays))
               for case in reference.CASES for contract in reference.CONTRACTS}
    return dict(plan=reference.plan(), records=records)


class LongReference(unittest.TestCase):
    def test_literal_complete_geometry_and_budget(self):
        shapes = reference.geometry(512)
        self.assertEqual(len(shapes), 63)
        self.assertEqual(shapes["layers.27.q_rotary"], ([512, 16, 128], "<f2"))
        self.assertEqual(shapes["layers.27.attention_probabilities"], ([16, 512, 512], "<f4"))
        self.assertEqual(shapes["logits"], ([512, 151936], "<f4"))
        budget = reference.ledger()
        self.assertEqual(budget["capture_bytes"], 508297216)
        self.assertEqual(budget["four_capture_bytes"], 2033188864)
        self.assertLess(budget["admitted"], budget["limit"])

    def test_literal_prefix_and_single_future_change(self):
        a, b = reference.CASES.values()
        self.assertEqual(len(a), 512)
        self.assertEqual(a[:17], ([9707, 11, 1879, 0] * 5)[:17])
        self.assertEqual(a[:511], b[:511])
        self.assertEqual((a[-1], b[-1]), (0, 17))

    def test_existing_gates_are_copied_without_recalibration(self):
        old = reference.load(reference.OLD, reference.OLD_SHA)
        gates = reference.gates()
        for name, gate in gates.items():
            self.assertEqual(gate, old["arrays"]["cases.tile17." + name]["gate"])
        self.assertEqual(gates["logits"]["atol"], 0.07747364044189453)

    def test_schema_and_coherent_layout_mutations_reject(self):
        initial = manifest()
        reference.validate(initial)
        label = "original.fp32"
        mutations = [lambda m: m["plan"].update(context=1024),
                     lambda m: m["records"].pop("future_tail.fp64"),
                     lambda m: m["records"][label].update(file="../original.fp32.bin"),
                     lambda m: m["records"][label].update(bytes=1),
                     lambda m: m["records"][label].update(sha256="g" * 64),
                     lambda m: m["records"][label]["arrays"].pop("logits"),
                     lambda m: m["records"][label]["arrays"]["logits"].update(shape=[512, 64]),
                     lambda m: m["records"][label]["arrays"]["logits"].update(dtype="<f2"),
                     lambda m: m["records"][label]["arrays"]["logits"].update(offset=True),
                     lambda m: m["records"][label]["arrays"]["logits"].update(bytes=1),
                     lambda m: m["plan"]["gates"]["logits"].update(atol=1),
                     lambda m: m["plan"].update(target_executed=True)]
        for mutation in mutations:
            value = copy.deepcopy(initial)
            mutation(value)
            with self.assertRaises(ValueError):
                reference.validate(value)

    def test_streamed_capture_and_readback_authenticate_exact_bytes(self):
        values = {"a": np.array([[0, -0.0], [0.5, -0.5]], dtype="<f2"),
                  "b": np.array([[0.125, 0.25]], dtype="<f4")}
        shapes = {name: (list(value.shape), value.dtype.str) for name, value in values.items()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(reference, "geometry", return_value=shapes):
                record = reference.save_graph(root, "tiny", values)
            expected = values["a"].tobytes() + values["b"].tobytes()
            self.assertEqual((root / "tiny.bin").read_bytes(), expected)
            self.assertEqual(record["sha256"], hashlib.sha256(expected).hexdigest())
            capture = reference.Capture(root, record)
            for name, value in values.items():
                self.assertEqual(capture.array(name).tobytes(), value.tobytes())
            changed = copy.deepcopy(record)
            changed["arrays"]["a"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                reference.Capture(root, changed).array("a")
            (root / "tiny.bin").write_bytes(expected[:-1])
            with self.assertRaises(ValueError):
                reference.Capture(root, record)

    def test_storage_and_nonfinite_rejected_before_publication(self):
        shapes = {"a": ([2, 2], "<f2")}
        for value in [np.ones((2, 2), dtype="<f4"), np.ones((2, 3), dtype="<f2"),
                      np.array([[1, np.nan], [2, 3]], dtype="<f2"),
                      np.arange(8, dtype="<f2").reshape(2, 4)[:, ::2]]:
            with tempfile.TemporaryDirectory() as temporary, mock.patch.object(reference, "geometry", return_value=shapes):
                with self.assertRaises(ValueError):
                    reference.save_graph(Path(temporary), "bad", {"a": value})

    def test_comparison_uses_existing_absolute_relative_gate(self):
        a = np.array([[1, -2], [0, 4]], dtype="<f4")
        b = np.array([[1, -2], [0, 4.5]], dtype="<f4")
        result = reference.compare_arrays(a, b, dict(atol=0.1, rtol=0.01))
        self.assertEqual(result["values"], 4)
        self.assertEqual(result["failed_values"], 1)
        self.assertEqual(result["maximum_absolute"], 0.5)
        self.assertAlmostEqual(result["maximum_gate_fraction"], 0.5 / 0.145)
        with self.assertRaises(ValueError):
            reference.compare_arrays(a, b[:1], dict(atol=0.1, rtol=0.01))

    def test_document_duplicate_and_pin_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "document.json"
            path.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                reference.load(path)
            path.write_text(json.dumps({"a": 1}))
            with self.assertRaises(ValueError):
                reference.load(path, "0" * 64)

    def miniature_comparison(self, change=None, report=None):
        """Coherent saved records exercise the actual comparison branches."""
        values = {"embedding": np.zeros((4, 2), dtype="<f2"),
                  "logits": np.tile(np.array([1, 1.001, 0, 0], dtype="<f4"), (4, 1)),
                  "layers.0.attention_probabilities": np.tri(4, dtype="<f4")[None]}
        shapes = {name: (list(value.shape), value.dtype.str) for name, value in values.items()}
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as patches:
            root = Path(temporary)
            patches.enter_context(mock.patch.object(reference, "CONTEXT", 4))
            patches.enter_context(mock.patch.object(reference, "geometry", return_value=shapes))
            patches.enter_context(mock.patch.object(reference, "validate"))
            old_arrays = {}; data = b""
            for name, value in values.items():
                body = value.tobytes()
                old_arrays["cases.tile17." + name] = dict(shape=list(value.shape), dtype=value.dtype.str,
                    offset=len(data), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
                data += body
            old_path = root / "old.json"; old_path.write_text(json.dumps(dict(arrays=old_arrays)))
            old_data = root / "old.bin"; old_data.write_bytes(data)
            for name, value in [("OLD", old_path), ("OLD_SHA", reference.digest(old_path)),
                                ("OLD_ARRAY", old_data), ("OLD_ARRAY_SHA", reference.digest(old_data))]:
                patches.enter_context(mock.patch.object(reference, name, value))
            records = {}
            for case in reference.CASES:
                for contract in reference.CONTRACTS:
                    label = case + "." + contract
                    current = {key: value.copy() for key, value in values.items()}
                    if change:
                        change(label, current)
                    # Nonfinite controls bypass save_graph's earlier finite
                    # admission, but rebuild whole/per-array hashes coherently.
                    if any(not np.isfinite(value).all() for value in current.values()):
                        with mock.patch.object(np, "isfinite", side_effect=lambda value: np.ones_like(value, dtype=bool)):
                            records[label] = reference.save_graph(root, label, current)
                    else:
                        records[label] = reference.save_graph(root, label, current)
            document = dict(records=records, plan=dict(gates={name: dict(atol=0.1, rtol=0.001) for name in values}))
            return reference.compare(root, document, report=report)

    def test_saved_mask_causal_and_signed_zero_controls(self):
        clean = self.miniature_comparison()
        self.assertTrue(clean["mask_passed"] and clean["causal_passed"] and clean["numerical_gates_passed"])
        self.assertEqual(clean["masked_probability_elements"], 24)
        def mask(label, values):
            if label == "future_tail.fp64":
                values["layers.0.attention_probabilities"][0, 0, 2] = 0.5
        self.assertFalse(self.miniature_comparison(mask)["mask_passed"])
        def sign(label, values):
            if label == "future_tail.fp64":
                values["embedding"][1, 0] = -0.0
        changed = self.miniature_comparison(sign)
        self.assertTrue(changed["numerical_gates_passed"])
        self.assertFalse(changed["causal_passed"])

    def test_saved_argmax_disagreement_never_selects_a_contract(self):
        def changed(label, values):
            if label.endswith("fp64"):
                values["logits"][0, 0] = 1.002
        result = self.miniature_comparison(changed)
        self.assertTrue(result["numerical_gates_passed"])
        self.assertFalse(result["arithmetic_argmax_agreement"])
        self.assertFalse(result["common_greedy_oracle_ready"])
        self.assertEqual(result["arithmetic_argmax_disagreements"], dict(original=[0], future_tail=[0]))

    def test_nonfinite_saved_comparison_retains_partial_report(self):
        def changed(label, values):
            if label == "original.fp64":
                values["logits"][0, 0] = np.nan
        report = {}
        with self.assertRaisesRegex(ValueError, "Nonfinite saved comparison"):
            self.miniature_comparison(changed, report)
        self.assertIn("original.contracts.embedding", report["comparisons"])

    def test_later_forward_failure_preserves_capture_and_primary_error(self):
        import mlx_qwen3_real as source
        import qwen3_fp16
        import sys
        sys.path.insert(0, str(reference.ROOT / "tools/experiments"))
        import native_usage
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(source, "package_pins", return_value={}), \
             mock.patch.object(source, "Weights"), \
             mock.patch.object(qwen3_fp16, "forward", side_effect=[{}, RuntimeError("second forward failure")]), \
             mock.patch.object(reference, "save_graph", return_value=dict(bytes=0)), \
             mock.patch.object(native_usage, "usage", return_value=dict(lifetime_max_phys_footprint=1)):
            root = Path(temporary) / "capture"
            with self.assertRaisesRegex(RuntimeError, "second forward failure"):
                reference.export(root)
            saved = json.loads((root / "manifest.json").read_text())
            self.assertEqual(saved["capture_error"], "RuntimeError: second forward failure")
            self.assertEqual(list(saved["records"]), ["original.fp32"])
            self.assertTrue(saved["capture_attempts"][0]["completed"])
            self.assertFalse(saved["capture_attempts"][1]["completed"])
            self.assertTrue((root / "original.fp32.json").is_file())
            self.assertFalse(json.loads((root / "comparison.json").read_text())["comparison_completed"])


if __name__ == "__main__":
    unittest.main()
