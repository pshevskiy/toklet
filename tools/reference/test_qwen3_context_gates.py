"""Independent small exact errors, prefix selection and authentication controls."""
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import qwen3_context_gates as subject


def miniature(folder):
    cases, records = [], {}
    for source in ("a", "b"):
        for rows in (1, 2, 3):
            label = f"cal-{source}-{rows}"
            cases.append(dict(id=label, rows=rows))
            for contract in ("fp32", "fp64"):
                body = b"";arrays = {}
                for name, kind, dtype in (("embedding", "e", "<f2"), ("logits", "f", "<f4")):
                    # A row-count contribution makes nested-prefix error exactly
                    # 1/4; cross-precision error is exactly 1/16.
                    value = 1 + rows / 8 + (1 / 16 if contract == "fp64" else 0)
                    raw = struct.pack("<" + kind * rows * 2, *([value, value + 1] * rows))
                    arrays[name] = dict(shape=[rows, 2], dtype=dtype, offset=len(body), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                    body += raw
                filename = label + "." + contract + ".bin";(folder / filename).write_bytes(body)
                records[label + "." + contract] = dict(file=filename, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), arrays=arrays)
    pairs = [dict(short=f"cal-{s}-{lo}", long=f"cal-{s}-{hi}", positions=lo) for s in ("a", "b") for lo, hi in ((1, 2), (1, 3), (2, 3))]
    plan = dict(cases=cases, calibration_prefix_pairs=pairs, stage_pooling=dict(embedding="embedding", logits="logits"),
        input_manifest_sha256="toy", recipe=dict(proposal=dict(experimental=True)), profile=dict(toy=True),
        validation_evaluated=False, gates_derived=False)
    return dict(plan=plan, complete=True, primary_error=None, records=records)


class ContextGates(unittest.TestCase):
    def test_fixed_formula_floor_and_no_nonfinite_or_overflow(self):
        self.assertEqual(subject.gate(0, 0)["atol"], 1e-4)
        self.assertEqual(subject.gate(0.125, 0.0625), dict(atol=0.25, rtol=0.001,
            calibration_fp32_fp64_max_abs=0.125, equivalent_prefix_max_abs=0.0625))
        for value in (True, -1, math.nan, math.inf, 1e308):
            with self.assertRaises(ValueError):
                subject.gate(value, 0)

    def test_complete_derivation_uses_exact_prefix_not_long_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary);manifest = miniature(folder);report = {}
            with mock.patch.object(subject.calibration.reference, "load", return_value=manifest):
                result = subject.derive(folder, folder, report)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["comparisons"]), 36)
            self.assertEqual(report["comparisons"][0]["left_label"], "cal-a-1.fp32")
            self.assertEqual(report["comparisons"][0]["right_label"], "cal-a-1.fp64")
            self.assertTrue(report["comparisons"][0]["left"]["complete"])
            for comparison in report["comparisons"]:
                self.assertIn(comparison["left_label"], manifest["records"])
                self.assertIn(comparison["right_label"], manifest["records"])
                self.assertIsInstance(comparison["left"], dict)
                self.assertIsInstance(comparison["right"], dict)
            for value in result["gates"].values():
                self.assertEqual(value, dict(atol=0.5, rtol=0.001,
                    calibration_fp32_fp64_max_abs=0.0625, equivalent_prefix_max_abs=0.25))
            self.assertTrue(all(v["endpoint_checked"] for v in report["parents"]))
            self.assertTrue(any(not v["right"]["full_array"] for v in report["comparisons"]))
            self.assertFalse(result["adopted"] or result["validation_evaluated"] or result["quality_qualified"])
            self.assertTrue(result["gates_derived"])
            self.assertEqual(result["review_status"], "pending")

    def test_large_excluded_tail_cannot_change_prefix_maximum(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary);owners = []
            for label, values, rows in (("short", [1, 2], 1), ("long", [1, 2, 1e20, -1e20], 2)):
                body = struct.pack("<" + "f" * len(values), *values);path = folder / label;path.write_bytes(body)
                record = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
                item = dict(**record, offset=0, shape=[rows, 2], dtype="<f4")
                owners.append((path, record, item))
            aa, bb = owners;audit = {}
            with subject.Owner(aa[0], aa[1]) as left, subject.Owner(bb[0], bb[1]) as right:
                self.assertEqual(subject.maximum(left, aa[2], right, bb[2], audit), 0)
            self.assertEqual(audit["values"], 2)
            self.assertFalse(audit["right"]["full_array"])
            self.assertEqual(audit["maximum_example"]["coordinates"], [0, 0])

    def test_consumed_array_hash_failure_retains_partial_findings(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary);manifest = miniature(folder)
            manifest["records"]["cal-a-1.fp64"]["arrays"]["logits"]["sha256"] = "0" * 64
            report = {}
            with mock.patch.object(subject.calibration.reference, "load", return_value=manifest), self.assertRaisesRegex(ValueError, "array hash"):
                subject.derive(folder, folder, report)
            self.assertFalse(report["complete"])
            self.assertTrue(report["comparisons"][0]["complete"])
            self.assertFalse(report["comparisons"][-1]["complete"])
            self.assertEqual(report["comparisons"][-1]["left_label"], "cal-a-1.fp32")
            self.assertEqual(report["comparisons"][-1]["right_label"], "cal-a-1.fp64")
            self.assertIsInstance(report["comparisons"][-1]["left"], dict)
            self.assertEqual(report["gates"], {})
            self.assertTrue(all(v["endpoint_checked"] for v in report["parents"]))

    def test_nonfinite_and_bad_shapes_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary);manifest = miniature(folder);left = manifest["records"]["cal-a-1.fp32"];right = manifest["records"]["cal-a-1.fp64"]
            for mutation in (lambda v: v.update(shape=[True, 2]), lambda v: v.update(bytes=1)):
                a = dict(left["arrays"]["embedding"]);mutation(a)
                with subject.Owner(folder / left["file"], left) as aa, subject.Owner(folder / right["file"], right) as bb, self.assertRaises(ValueError):
                    subject.maximum(aa, a, bb, right["arrays"]["embedding"], {})
            path = folder / left["file"];body = bytearray(path.read_bytes());body[:2] = struct.pack("<e", math.nan)
            path.write_bytes(body);left["sha256"] = hashlib.sha256(body).hexdigest()
            item = left["arrays"]["embedding"];item["sha256"] = hashlib.sha256(body[:item["bytes"]]).hexdigest()
            with subject.Owner(path, left) as aa, subject.Owner(folder / right["file"], right) as bb, self.assertRaisesRegex(ValueError, "Nonfinite"):
                subject.maximum(aa, item, bb, right["arrays"]["embedding"], {})


if __name__ == "__main__":
    unittest.main()
