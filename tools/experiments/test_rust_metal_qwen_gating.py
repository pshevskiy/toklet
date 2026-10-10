"""Offline Qwen product provenance, exact rounding, bounds and driver checks."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_qwen_gating as qg
from test_rust_metal_gating import fixture


def synthetic(root):
    hroot, sroot = root / "half", root / "single"
    hroot.mkdir()
    sroot.mkdir()
    half, single = {"arrays": {}}, {"arrays": {}}

    def save(directory, manifest, key, rows, bits, dtype):
        data = struct.pack(f"<{len(bits)}" + ("H" if dtype == "f16le" else "f"), *bits)
        path = directory / f"{key}.{dtype}"
        path.write_bytes(data)
        item = {"file": path.name, "shape": [rows, qg.WIDTH], "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}
        if dtype == "f16le":
            item["dtype"] = dtype
        manifest["arrays"][key] = item
        return item

    # Literal independent expected bits include signs, normal/subnormal ties,
    # and finite maximum. Vary row, layer and column to detect incorrect slices.
    edges = [(0, 0x3c00, 0), (0x8000, 0x3c00, 0x8000),
             (0x3e00, 0x3c01, 0x3e02), (0xbe00, 0x3c03, 0xbe04),
             (1, 0x3800, 0), (3, 0x3800, 2),
             (0x8003, 0x3800, 0x8002), (0x7bff, 0x3800, 0x77ff)]
    for name, rows in qg.qwen.CASE_ROWS.items():
        for layer in qg.LAYERS:
            prefix = f"cases.{name}.layers.{layer}"
            values = [edges[(row + column + layer) % len(edges)]
                      for row in range(rows) for column in range(qg.WIDTH)]
            gate, up, output = [[value[i] for value in values] for i in range(3)]
            save(hroot, half, f"{prefix}.gate_silu", rows, gate, "f16le")
            save(hroot, half, f"{prefix}.up_projection", rows, up, "f16le")
            key = f"{prefix}.gated"
            oracle = save(hroot, half, key, rows, output, "f16le")
            reference = list(struct.unpack(f"<{len(output)}e", struct.pack(f"<{len(output)}H", *output)))
            ref = save(sroot, single, key, rows, [value + .00001 for value in reference], "f32le")
            oracle.update(source_fp32_full_sha256=ref["sha256"],
                          gate_against_source_fp32={"atol": .001, "rtol": .001})
    return half, single, hroot, sroot


class QwenMetalGating(unittest.TestCase):
    def test_complete_operands_oracle_and_source_rows_reconstruct_frozen_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            source = synthetic(Path(directory))
            half, single, hroot, sroot = source
            inputs, anchors = qg.assemble(*source)
            self.assertEqual(len(inputs), 36)
            self.assertEqual(sum(item["rows"] * item["width"] for item in inputs), 175104)
            stages = {anchor["source_stage"] for anchor in anchors.values()}
            self.assertEqual(len(stages), 12)
            for stage in stages:
                group = [item for item in inputs if anchors[item["id"]]["source_stage"] == stage]
                self.assertEqual([anchors[item["id"]]["row_index"] for item in group], list(range(len(group))))
                prefix = stage.removesuffix(".gated")
                for field, suffix in (("gate_bits", ".gate_silu"), ("up_bits", ".up_projection")):
                    joined = b"".join(struct.pack(f"<{qg.WIDTH}H", *item[field]) for item in group)
                    self.assertEqual(joined, (hroot / half["arrays"][prefix + suffix]["file"]).read_bytes())
                    for item in group:
                        label = "gate_silu" if field == "gate_bits" else "up_projection"
                        row = struct.pack(f"<{qg.WIDTH}H", *item[field])
                        self.assertEqual(anchors[item["id"]][label + "_fp16_sha256"], hashlib.sha256(row).hexdigest())
                joined = b"".join(struct.pack(f"<{qg.WIDTH}H", *anchors[item["id"]]["bits"]) for item in group)
                self.assertEqual(joined, (hroot / half["arrays"][stage]["file"]).read_bytes())
                joined = b"".join(struct.pack(f"<{qg.WIDTH}f", *anchors[item["id"]]["reference"]) for item in group)
                self.assertEqual(joined, (sroot / single["arrays"][stage]["file"]).read_bytes())
                for item in group:
                    self.assertLessEqual(item["rows"] * item["width"], 8192)
                    anchor = anchors[item["id"]]
                    self.assertEqual(anchor["gate"], half["arrays"][stage]["gate_against_source_fp32"])
                    native = fixture([item], anchors)
                    qg.gating.validate(native, [item], anchors)
                    self.assertTrue(all(row["qualification"]["exact_independent_fp16_bits"]
                                        for row in native["result"]["cases"][0]["rows"]))

    def test_exact_half_cannot_bypass_frozen_fp32_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, anchors = qg.assemble(*synthetic(Path(directory)))
            group = inputs[:1]
            anchors[group[0]["id"]]["reference"][0] += 10
            with self.assertRaisesRegex(ValueError, "FP32 source gate"):
                qg.gating.validate(fixture(group, anchors), group, anchors)

    def test_all_complete_payloads_require_shape_dtype_size_hash_and_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            source = synthetic(Path(directory))
            half, single, hroot, sroot = source
            prefix = "cases.single.layers.0"
            for role in ("gate_silu", "up_projection", "gated"):
                item = half["arrays"][f"{prefix}.{role}"]
                for field, bad in (("shape", [qg.WIDTH]), ("dtype", "f32le"), ("bytes", 1)):
                    original = item[field]
                    item[field] = bad
                    with self.subTest(role=role, field=field), self.assertRaises(ValueError):
                        qg.assemble(*source)
                    item[field] = original
                path = hroot / item["file"]
                data = path.read_bytes()
                path.write_bytes(b"\x00\x7c" + data[2:])
                with self.assertRaisesRegex(ValueError, "payload identity"):
                    qg.assemble(*source)
                item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    qg.assemble(*source)
                path.write_bytes(data)
                item["sha256"] = hashlib.sha256(data).hexdigest()
            item = single["arrays"][f"{prefix}.gated"]
            path = sroot / item["file"]
            data = path.read_bytes()
            path.write_bytes(struct.pack("<f", float("inf")) + data[4:])
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            half["arrays"][f"{prefix}.gated"]["source_fp32_full_sha256"] = item["sha256"]
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                qg.assemble(*source)

    def test_product_and_fp32_source_anchors_are_independently_required(self):
        with tempfile.TemporaryDirectory() as directory:
            source = synthetic(Path(directory))
            half, _, hroot, _ = source
            item = half["arrays"]["cases.single.layers.0.gated"]
            original = item["source_fp32_full_sha256"]
            item["source_fp32_full_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "source anchor"):
                qg.assemble(*source)
            item["source_fp32_full_sha256"] = original
            path = hroot / item["file"]
            data = path.read_bytes()
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "independent Half product"):
                qg.assemble(*source)

    def test_driver_retains_same_binary_tiny_and_overflow_checks_without_checkpoint_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = synthetic(root)
            half, single, hroot, sroot = source
            _, anchors = qg.assemble(*source)
            tiny = [{"id": "tiny", "rows": 1, "width": 64, "gate_bits": [0] * 64, "up_bits": [0] * 64}]
            tiny_anchors = {"tiny": {"bits": [0] * 64, "reference": [0.0] * 64, "gate": None}}
            overflow = {"id": "overflow", "rows": 1, "width": 64, "gate_bits": [0x7bff] * 64, "up_bits": [0x4000] * 64}
            contexts = []

            def native(output, build, extra_runs, extra_context):
                contexts.append(extra_context)
                self.assertEqual(len(extra_runs), 38)
                for label, options, validator in extra_runs:
                    self.assertEqual(options[0], "--gating-input")
                    group = json.loads(Path(options[1]).read_text())["cases"]
                    if label == "gating_overflow_regression":
                        event = {"event": "idle_result", "passed": False, "error": {
                            "stage": "validation", "message": "completed gating contains nonfinite Half; result discarded"}}
                        qg.gating.rms.probe.validate_run(label, {"exit_code": 1, "event": event}, validator)
                    else:
                        expected = tiny_anchors if label == "tiny_gating_regression" else anchors
                        validator(fixture(group, expected))

            with patch.object(qg.qwen, "HALF", hroot), patch.object(qg.qwen, "SINGLE", sroot), \
                    patch.object(qg, "checked_references", return_value=(half, single)) as read, \
                    patch.object(qg.gating, "cases", return_value=(tiny, tiny_anchors, overflow)), \
                    patch.object(qg.qwen, "norm_weights", side_effect=AssertionError("checkpoint read")) as weights, \
                    patch.object(qg.qwen, "stream_digest", side_effect=AssertionError("checkpoint hash")) as stream, \
                    patch.object(qg.gating.rms.probe, "run", side_effect=native):
                qg.run(root / "result.json", root / "build", root / "inputs")
                self.assertEqual(read.call_count, 2)
                weights.assert_not_called()
                stream.assert_not_called()
            self.assertFalse(contexts[0]["checkpoint_reads"])
            self.assertEqual(len(contexts[0]["input_sha256"]), 38)
            self.assertEqual(len(contexts[0]["qualification_driver_sha256"]), 8)

    def test_final_reference_recheck_detects_post_load_payload_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = synthetic(root)
            half, single, hroot, sroot = source
            _, anchors = qg.assemble(*source)

            def native(output, build, extra_runs, extra_context):
                label, options, validator = extra_runs[-1]
                self.assertTrue(label.startswith("qwen_gating_row_"))
                group = json.loads(Path(options[1]).read_text())["cases"]
                item = half["arrays"]["cases.single.layers.0.up_projection"]
                path = hroot / item["file"]
                path.write_bytes(path.read_bytes() + b"\0")
                with self.assertRaisesRegex(ValueError, "payload identity"):
                    validator(fixture(group, anchors))

            with patch.object(qg.qwen, "HALF", hroot), patch.object(qg.qwen, "SINGLE", sroot), \
                    patch.object(qg, "checked_references", return_value=(half, single)), \
                    patch.object(qg.gating.rms.probe, "run", side_effect=native):
                qg.run(root / "result.json", root / "build", root / "inputs")

    def test_aliases_existing_inputs_and_protected_roots_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, build = root / "result.json", root / "build"
            with patch.object(qg, "checked_references") as read:
                for inputs in (output, build, build / "inputs", root):
                    with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                        qg.run(output, build, inputs)
                inputs = root / "inputs"
                inputs.mkdir()
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    qg.run(output, build, inputs)
                with self.assertRaisesRegex(ValueError, "source fixtures"):
                    qg.run(qg.qwen.HALF / "new.json", build, root / "new")
                for protected in (qg.gating.rms.FP16, qg.gating.rms.FP32):
                    with self.assertRaisesRegex(ValueError, "tiny source fixtures"):
                        qg.run(protected / "new.json", build, root / "new")
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
