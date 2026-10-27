"""Literal packed-row, exact rounding, decomposition and retained-error controls."""
import copy
from fractions import Fraction
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import qwen3_projection_localization as subject


def miniature(root):
    body = bytearray()
    def add(data, dtype=None):
        result = dict(file="parent.bin", offset=len(body), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if dtype:
            result["dtype"] = dtype
        body.extend(data)
        return result
    selected = dict(stage="layers.27.q_projection", position=0, feature=0, category="passing_control",
        weights=dict(weight=add(bytes([0, 1, 2, 3]) * 256), scales=add(struct.pack("<H", 0x3f80) * 16),
                     biases=add(bytes(32))), native_input=add(struct.pack("<f", 1) * 1024),
        native_output=add(struct.pack("<f", 1536)), reference_inputs={}, reference_outputs={},
        canonical_gate=dict(atol=1, rtol=0, pooled_fp32_fp64_max_abs=0, equivalent_prefix_max_abs=0))
    for contract, value in (("fp32", 0.5), ("fp64", 2)):
        selected["reference_inputs"][contract] = add(struct.pack("<e", value) * 1024, "<f2")
        selected["reference_outputs"][contract] = add(struct.pack("<e", value * 1536), "<f2")
    selected["category"] = "native_canonical_failure"
    path = root / "parent.bin"; path.write_bytes(body)
    return selected, dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


class ProjectionDiagnosis(unittest.TestCase):
    def test_nearest_even_exact_midpoints_and_double_rounding(self):
        for kind, epsilon in (("e", Fraction(1, 1024)), ("f", Fraction(1, 2**23))):
            mid = Fraction(1) + epsilon / 2
            tiny = Fraction(1, 2**100)
            self.assertEqual(subject.rounded(mid, kind), 1)
            self.assertEqual(subject.rounded(mid + tiny, kind), float(1 + epsilon))
            self.assertEqual(subject.rounded(mid - tiny, kind), 1)
            self.assertEqual(subject.rounded(mid + epsilon, kind), float(1 + 2 * epsilon))
            self.assertEqual(subject.rounded(-mid, kind), -1)
            self.assertEqual(subject.rounded(-mid - tiny, kind), -float(1 + epsilon))

    def test_subnormal_signed_zero_and_finite_range(self):
        for kind, bit in (("e", 24), ("f", 149)):
            self.assertEqual(subject.rounded(Fraction(1, 2**bit), kind), 2.0**-bit)
            self.assertEqual(struct.pack("<" + kind, subject.rounded(-Fraction(1, 2**(bit + 2)), kind)),
                             struct.pack("<" + kind, -0.0))
            self.assertEqual(subject.rounded(Fraction(0), kind), 0)
        with self.assertRaises(ValueError):
            subject.rounded(Fraction(65505), "e")
        with self.assertRaises(ValueError):
            subject.rounded(Fraction(1), "d")

    def test_literal_little_endian_codes_signed_bf16_and_group_boundary(self):
        codes = struct.pack("<I", 0xff7f0100) * 256
        scales = struct.pack("<H", 0x3f00) + struct.pack("<H", 0xbf80) * 15
        biases = struct.pack("<H", 0xbf80) + struct.pack("<H", 0x4000) * 15
        row = subject.reconstruct(codes, scales, biases)
        self.assertEqual(row[:4], [-1, -0.5, 62.5, 126.5])
        self.assertEqual((row[63], row[64], row[65], row[67]), (126.5, 2, 1, -253))

    def test_nonfinite_lengths_and_non_half_expansion_reject(self):
        for body, kind, count in ((b"", "f", 1), (struct.pack("<e", float("inf")), "e", 1)):
            with self.assertRaises(ValueError):
                subject.finite_values(body, kind, count)
        with self.assertRaises(ValueError):
            subject.half_expanded(struct.pack("<f", 1.0001), 1)
        self.assertEqual(struct.pack("<f", subject.half_expanded(struct.pack("<f", -0.0), 1)[0]), struct.pack("<f", -0.0))
        with self.assertRaises(ValueError):
            subject.reconstruct(bytes(1024), struct.pack("<H", 0x7f80) * 16, bytes(32))

    def test_exact_dot_and_inherited_error_decomposition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); selected, record = miniature(root)
            with subject.Owner(root / "parent.bin", record) as owner:
                value = subject.point({"parent.bin": owner}, selected)
            self.assertTrue(owner.audit["endpoint_checked"])
            self.assertEqual(value["exact_dots"], dict(native="1536", fp32="768", fp64="3072"))
            self.assertEqual(value["decomposition"], dict(native_local="0", inherited_input="768", reference_local="0"))
            self.assertEqual(value["local_extra_error"], dict(native="0", fp32="0", fp64="0"))
            self.assertEqual(value["exact_input_l1_bound"], "768")
            self.assertTrue(value["canonical_failed"] and value["native_own_input_correctly_rounded"])

    def test_span_digest_extent_dtype_and_classification_reject(self):
        mutations = [lambda v: v["native_input"].update(sha256="0" * 64),
            lambda v: v["native_output"].update(offset=True), lambda v: v["native_output"].update(offset=999999),
            lambda v: v["reference_inputs"]["fp32"].update(dtype="<f4"),
            lambda v: v.update(category="passing_control"), lambda v: v["canonical_gate"].update(atol=float("nan"))]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); original, record = miniature(root)
            with subject.Owner(root / "parent.bin", record) as owner:
                for mutation in mutations:
                    selected = copy.deepcopy(original); mutation(selected)
                    with self.assertRaises(ValueError):
                        subject.point({"parent.bin": owner}, selected)

    def test_partial_results_and_endpoint_errors_survive_later_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); selected, record = miniature(root)
            broken = copy.deepcopy(selected); broken["native_output"]["sha256"] = "0" * 64
            report = {}
            with mock.patch.object(subject, "validate"), self.assertRaises(ValueError):
                subject.analyze(root, dict(parents={"parent.bin": record}, points=[selected, broken]), report)
            self.assertFalse(report["complete"])
            self.assertEqual(len(report["points"]), 1)
            self.assertTrue(report["parents"][0]["endpoint_checked"])
            self.assertFalse(report["whole_context_numerically_qualified"])

    def test_parent_truncation_and_symlink_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); selected, record = miniature(root)
            (root / "parent.bin").write_bytes(b"x")
            with self.assertRaises(ValueError):
                with subject.Owner(root / "parent.bin", record):
                    pass
            (root / "alias").symlink_to("parent.bin")
            with mock.patch.object(subject, "validate"), self.assertRaises(ValueError):
                subject.analyze(root, dict(parents={"alias": record}, points=[selected]), {})

    def test_typed_plan_coverage_and_duplicate_controls(self):
        points = []
        for stage in ("layers.27.q_projection", "layers.27.k_projection", "layers.27.v_projection"):
            for category, position in (("native_canonical_failure", 25), ("passing_control", 0)):
                points.extend(dict(stage=stage, position=position, feature=f, category=category) for f in (0, 1))
        points.extend(dict(stage="layers.27.v_projection", position=65, feature=f, category="cpu_envelope_disjoint") for f in (25, 98))
        plan = dict(format="toklet-qwen3-layer27-same-input-projection-plan-v1", point_count=14, columns=1024,
            group_size=64, bits=8, source_linear_bias=0, native_gates_changed=False,
            whole_context_numerically_qualified=False, points=points, parents={str(i): {} for i in range(4)})
        subject.validate(plan)
        for mutation in (lambda v: v.update(bits=True), lambda v: v.update(native_gates_changed=0),
                lambda v: v["points"][1].update(feature=0), lambda v: v["points"][0].update(position=True),
                lambda v: v["points"][0].update(category="passing_control"),
                lambda v: v["points"][0].update(stage="layers.26.q_projection")):
            changed = copy.deepcopy(plan); mutation(changed)
            with self.assertRaises(ValueError):
                subject.validate(changed)

        first = copy.deepcopy(plan)
        first.update(format="toklet-qwen3-first-projection-divergence-plan-v1", point_count=12)
        first["points"] = first["points"][:12]
        for point in first["points"]:
            point["stage"] = point["stage"].replace("layers.27", "layers.0")
            point["position"] = 0
            if point["category"] == "native_canonical_failure":
                point["category"] = "first_divergence"
                point["feature"] += 2
        subject.validate(first)
        first["points"][0]["position"] = 25
        with self.assertRaises(ValueError):
            subject.validate(first)

    def test_exact_half_midpoints_positive_negative_and_subnormal(self):
        for value, low, high in ((Fraction(1), Fraction(4095, 4096), Fraction(2049, 2048)),
                (Fraction(-1), Fraction(-2049, 2048), Fraction(-4095, 4096)),
                (Fraction(0), -Fraction(1, 2**25), Fraction(1, 2**25))):
            result = subject.midpoint_distances(value)
            self.assertEqual((Fraction(result["lower_midpoint"]), Fraction(result["upper_midpoint"])), (low, high))
            self.assertEqual(Fraction(result["above_lower"]), value - low)
            self.assertEqual(Fraction(result["below_upper"]), high - value)
        with self.assertRaises(ValueError):
            subject.midpoint_distances(Fraction(65504))

    def test_identical_inputs_detect_reference_store_boundary_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); selected, _ = miniature(root)
            selected.update(stage="layers.0.q_projection", category="first_divergence")
            data = bytearray((root / "parent.bin").read_bytes())
            for contract in ("fp32", "fp64"):
                for family, body in (("reference_inputs", struct.pack("<e", 1) * 1024),
                                     ("reference_outputs", struct.pack("<e", 1537 if contract == "fp32" else 1536))):
                    item = selected[family][contract];data[item["offset"]:item["offset"] + item["bytes"]] = body
                    item["sha256"] = hashlib.sha256(body).hexdigest()
            (root / "parent.bin").write_bytes(data)
            record = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            with subject.Owner(root / "parent.bin", record) as owner:
                result = subject.point({"parent.bin": owner}, selected)
                self.assertEqual(result["input_rows_identical_to_native"], dict(fp32=True, fp64=True))
                self.assertEqual(result["local_extra_error"], dict(native="0", fp32="1", fp64="0"))
                self.assertEqual(result["decomposition"]["inherited_input"], "0")
                self.assertEqual(result["cpu_and_fused_reconstruction_identical_weights"], 1024)
                broken = copy.deepcopy(selected);broken["category"] = "passing_control"
                with self.assertRaises(ValueError):
                    subject.point({"parent.bin": owner}, broken)
            row = dict(native=selected["native_input"], **selected["reference_inputs"])
            plan = dict(format="toklet-qwen3-first-projection-divergence-plan-v1",
                parents={"parent.bin": record}, points=[selected],
                initial_rows={"embedding": row, "layers.0.input_norm": copy.deepcopy(row)})
            with mock.patch.object(subject, "validate"):
                report = {};subject.analyze(root, plan, report)
                self.assertTrue(report["complete"])
                self.assertEqual(report["initial_rows"]["embedding"], dict(columns=1024, fp32_exact=True, fp64_exact=True))
                plan["initial_rows"]["embedding"]["fp32"] = copy.deepcopy(row["fp32"])
                plan["initial_rows"]["embedding"]["fp32"]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    subject.analyze(root, plan, {})


if __name__ == "__main__":
    unittest.main()
