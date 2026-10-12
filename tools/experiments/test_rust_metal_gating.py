"""Offline stored-Half product, ownership and failure-policy checks."""
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_gating as gating


def fixture(inputs, anchors):
    cases = []
    for item in inputs:
        count = item["rows"] * item["width"]
        guards = math.ceil(count / 32) * 32 - count + 7
        size = (count + guards) * 2
        cases.append({
            "id": item["id"], "rows_count": item["rows"], "width": item["width"], "guards": guards, "inactive_lanes": guards - 7, "sentinel_guards": 7,
            "output_guard_bits": gating.GUARD, "distinct_owned_output": True,
            "allocations_reused_across_repetitions": True, "threadgroup_width": 32,
            "threadgroups": math.ceil(count / 32), "shared_buffer_bytes": size * 3,
            "host_input_and_readback_bytes": size * 5, "admitted_bytes": size * 8 + gating.rms.probe.ALLOWANCE,
            "rows": [{"phase": "warmup" if i < 3 else "sample", "repetition": i + 1 if i < 3 else i - 2,
                      "command_completed": True, "immutable_inputs_and_guards": True,
                      "encode_commit_wait_seconds": .001,
                      "output_bits": anchors[item["id"]]["bits"] + [gating.GUARD] * guards} for i in range(13)],
        })
    return {"event": "idle_result", "passed": True, "result": {
        "operation": "gating_fp16_storage_fp32_product", "cases": cases,
        "metadata": {"unified_memory": True, "language_version": "2.3", "math_mode": "safe",
                     "math_functions": "precise", "minimum_macos_major": 15,
                     "maximum_admitted_bytes": max(case["admitted_bytes"] for case in cases),
                     "execution_width": 32, "pipeline_max_threads": 1024, "device_max_threads": [1024, 1024, 1024]},
    }}


class MetalGating(unittest.TestCase):
    def test_frozen_products_and_manual_ties_require_exact_half_bits(self):
        inputs, anchors, overflow = gating.cases()
        self.assertEqual(len(inputs), 11)
        self.assertEqual(sum(item["rows"] * item["width"] for item in inputs), 6720)
        self.assertEqual(anchors[inputs[-1]["id"]]["bits"][:18], [
            0, 0x8000, 0, 0x8000, 0x3e02, 0x3e04, 0xbe02, 0xbe04,
            0, 2, 0x8000, 0x8002, 2, 0x0400, 0x0200, 0x7bff, 0x77ff, 0xfbff,
        ])
        self.assertEqual(overflow["gate_bits"], [0x7bff] * 64)
        self.assertEqual(overflow["up_bits"], [0x4000] * 64)
        with self.assertRaises(OverflowError):
            gating.half_product(0x7bff, 0x4000)
        native = fixture(inputs, anchors)
        gating.validate(native, inputs, anchors)
        for case in native["result"]["cases"]:
            bits = anchors[case["id"]]["bits"]
            for row in case["rows"]:
                self.assertNotIn("output_bits", row)
                q = row["qualification"]
                self.assertTrue(q["exact_independent_fp16_bits"])
                self.assertLessEqual(q["against_source_fp32_max_gate_fraction"], 1)
                self.assertEqual(q["output_sha256"], hashlib.sha256(struct.pack(f"<{len(bits)}H", *bits)).hexdigest())

    def test_data_guards_immutable_operands_and_owned_output_corruption_fail(self):
        inputs, anchors, _ = gating.cases()
        inputs = inputs[-1:]
        for kind in ("negative_zero", "tie", "subnormal", "guard", "integer", "nonfinite", "immutable",
                     "completed", "owned_output", "reuse", "shared_memory", "host_memory", "budget", "launch", "repetition"):
            native = fixture(inputs, anchors)
            case = native["result"]["cases"][0]
            row = case["rows"][-1]
            if kind == "negative_zero": row["output_bits"][1] = 0
            elif kind == "tie": row["output_bits"][4] -= 1
            elif kind == "subnormal": row["output_bits"][9] = 0
            elif kind == "guard": row["output_bits"][-1] ^= 1
            elif kind == "integer": row["output_bits"][0] = 0.0
            elif kind == "nonfinite": row["output_bits"][0] = 0x7c00
            elif kind == "immutable": row["immutable_inputs_and_guards"] = False
            elif kind == "completed": row["command_completed"] = False
            elif kind == "owned_output": case["distinct_owned_output"] = False
            elif kind == "reuse": case["allocations_reused_across_repetitions"] = False
            elif kind == "shared_memory": case["shared_buffer_bytes"] -= 1
            elif kind == "host_memory": case["host_input_and_readback_bytes"] -= 1
            elif kind == "budget": case["admitted_bytes"] -= 1
            elif kind == "launch": case["threadgroups"] += 1
            elif kind == "repetition": row["repetition"] = 9
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                gating.validate(native, inputs, anchors)

    def test_inactive_lanes_are_guarded_and_counted_using_queried_width(self):
        inputs, anchors = gating.tail_cases()
        self.assertEqual(tuple(item["width"] for item in inputs), gating.TAIL_COUNTS)
        native = fixture(inputs, anchors)
        gating.validate_tails(native, inputs, anchors)
        self.assertEqual(native["result"]["cases"][0]["guards"], 38)
        for field in ("inactive_lanes", "sentinel_guards", "guards"):
            native = fixture(inputs, anchors)
            native["result"]["cases"][0][field] -= 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                gating.validate_tails(native, inputs, anchors)
        native = fixture(inputs, anchors)
        native["result"]["cases"][0]["rows"][0]["output_bits"][1] ^= 1
        with self.assertRaises(ValueError):
            gating.validate_tails(native, inputs, anchors)
        native = fixture(inputs, anchors)
        for case in native["result"]["cases"]:
            case["inactive_lanes"] = 0
        with self.assertRaisesRegex(ValueError, "no inactive tail coverage"):
            gating.validate_tails(native, inputs, anchors)

    def test_original_fp32_gate_and_independent_product_are_both_required(self):
        inputs, anchors, _ = gating.cases()
        native = fixture(inputs, anchors)
        anchors[inputs[0]["id"]]["reference"][0] += 10
        with self.assertRaisesRegex(ValueError, "FP32 source gate"):
            gating.validate(native, inputs, anchors)
        payload = gating.rms.payload

        def changed(root, manifest, name, dtype):
            data, item = payload(root, manifest, name, dtype)
            if name == "cases.capacity.layers.0.gated" and dtype == "f16le":
                data = bytes([data[0] ^ 1]) + data[1:]
            return data, item

        with patch.object(gating.rms, "payload", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "independent Half product"):
                gating.cases()

    def test_completed_overflow_requires_structured_rejection_and_discard(self):
        message = "completed gating contains nonfinite Half; result discarded"
        good = {"event": "idle_result", "passed": False, "error": {"stage": "validation", "message": message}}
        self.assertIs(gating.overflow_rejected(good), False)
        for bad in ({**good, "passed": True}, {**good, "error": {"stage": "fixture", "message": message}},
                    {**good, "error": {"stage": "validation", "message": "rolled back"}}):
            with self.assertRaisesRegex(ValueError, "overflow rejection"):
                gating.overflow_rejected(bad)

    def test_aliases_and_existing_inputs_fail_before_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            build = root / "build"
            with patch.object(gating, "cases") as read:
                for inputs in (output, build, build / "inputs", root):
                    with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                        gating.run(output, build, inputs)
                inputs = root / "inputs"
                inputs.mkdir()
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    gating.run(output, build, inputs)
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
