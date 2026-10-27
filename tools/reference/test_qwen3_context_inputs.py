"""Portable split, boundary, provenance and partial-publication controls."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import qwen3_context_inputs as subject

RECIPE = Path(__file__).resolve().parents[2] / "benchmarks/qwen3-context-candidate-v2/recipe.json"


class ContextInputs(unittest.TestCase):
    def recipe(self):
        return json.loads(RECIPE.read_bytes())

    @staticmethod
    def encode(text):
        return list(text.encode())

    def test_explicit_lengths_disjoint_splits_and_prefix_map(self):
        recipe = self.recipe()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "inputs"
            report = subject.write_inputs(root, recipe, self.encode, dict(test_only=True))
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["rows"]), 14)
            self.assertEqual(len(report["stage_pooling"]), 57)
            self.assertEqual(len(report["calibration_prefix_pairs"]), 6)
            for row in report["rows"]:
                body = (root / row["file"]).read_bytes()
                self.assertEqual(subject.identity(body), row["token_array"])
                self.assertEqual(len(body), 4 * row["tokens"])
                self.assertEqual(list(struct.unpack("<" + "I" * row["tokens"], body)), self.encode((recipe["sources"][row["source"]] + "\n\n") * 16)[:row["tokens"]])
            self.assertFalse(report["numerical_gates_derived"] or report["adopted"] or report["quality_qualified"])
            with self.assertRaises(ValueError):
                subject.write_inputs(root, recipe, self.encode, {})

    def test_typed_cases_and_split_content_alias_reject(self):
        changes = [lambda v: v["cases"][0].update(tokens=True),
            lambda v: v["cases"][0].update(id="../escape"), lambda v: v["cases"][0].update(tokens=33),
            lambda v: v["cases"][1].update(id=v["cases"][0]["id"]),
            lambda v: v["proposal"].update(adopted=True), lambda v: v["proposal"]["quality"].update(clipping_allowed=True),
            lambda v: v["sources"].update(validation_prose="  " + v["sources"]["calibration_prose"] + "  ")]
        for change in changes:
            recipe = self.recipe();change(recipe)
            with self.assertRaises(ValueError):
                subject.validate(recipe)

    def test_duplicate_or_cross_split_prefix_reuse_reject(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            subject.build(self.recipe(), lambda _: [7] * 512)
        recipe = self.recipe()
        recipe["sources"]["validation_prose"] = recipe["sources"]["calibration_prose"] + " A separate suffix."
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            subject.build(recipe, self.encode)

    def test_tokenizer_bounds_and_insufficient_text_reject_before_publication(self):
        for encoder in (lambda _: [True] * 512, lambda _: [-1] * 512,
                        lambda _: [151936] * 512, lambda _: [0] * 131073,
                        lambda _: [1], lambda _: (1, 2)):
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "inputs"
                with self.assertRaises(ValueError):
                    subject.write_inputs(destination, self.recipe(), encoder, {})
                self.assertFalse(destination.exists())

    def test_coherently_changed_policy_and_profiles_reject(self):
        for family, key, value in (("proposal", "atol_floor", 0.1), ("proposal", "envelope_multiplier", 3),
                ("proposal", "rtol", 0.01), ("quality", "loss_increase_limit_nats_per_token", 0.5),
                ("quality", "task_drop_limit_percentage_points", 20),
                ("reference_profile", "arithmetic", ["FP64"]), ("reference_profile", "layers", [False, 1, 27]),
                ("native_profile", "projection", "Original"), ("proposal", "rejection_rules", []),
                ("proposal", "cpu_cross_precision_validation", "ignored"), ("proposal", "freeze_order", [])):
            recipe = self.recipe()
            target = recipe["proposal"]["quality"] if family == "quality" else recipe[family]
            target[key] = value
            with self.assertRaises(ValueError):
                subject.validate(recipe)

    def test_partial_success_survives_second_array_write_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "inputs"
            original = Path.open
            def failing(path, *args, **kwargs):
                if path.name == "cal-prose-128.u32le":
                    raise OSError("injected write failure")
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, "open", failing), self.assertRaisesRegex(OSError, "injected"):
                subject.write_inputs(destination, self.recipe(), self.encode, {})
            result = json.loads((destination / "manifest.json").read_bytes())
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["rows"]), 1)
            self.assertIn("injected", result["primary_error"])
            self.assertTrue((destination / "cal-prose-32.u32le").is_file())


if __name__ == "__main__":
    unittest.main()
