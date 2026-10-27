"""Literal source-bit widening, write-prefix and artifact-inventory controls."""
import hashlib
import copy
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import qwen3_quantized_source as subject


class QuantizedSource(unittest.TestCase):
    def test_literal_bf16_sign_subnormal_and_nonfinite_capture(self):
        words = [0x3f80, 0xbf80, 0x8000, 0x0001, 0x7f80, 0x7fc1]
        raw = struct.pack("<6H", *words);wide = struct.pack("<6I", *(v << 16 for v in words))
        self.assertEqual(subject.verify_widening(raw, wide, "bfloat16"),
                         dict(values=6, nonfinite_values=2, finite_widening_mismatches=0))
        changed = bytearray(wide);changed[8:12] = struct.pack("<f", 0.0)
        self.assertEqual(subject.verify_widening(raw, changed, "bfloat16")["finite_widening_mismatches"], 1)

    def test_literal_half_and_fp32_bit_exactness(self):
        values = [0.0, -0.0, 1, -2, 2**-24, 65504]
        raw = struct.pack("<6e", *values);wide = struct.pack("<6f", *values)
        self.assertEqual(subject.verify_widening(raw, wide, "float16"),
                         dict(values=6, nonfinite_values=0, finite_widening_mismatches=0))
        self.assertEqual(subject.verify_widening(wide, wide, "float32")["finite_widening_mismatches"], 0)
        damaged = bytearray(wide);damaged[4] = 1
        self.assertEqual(subject.verify_widening(wide, damaged, "float32")["finite_widening_mismatches"], 1)

    def test_widening_extent_and_dtype_reject(self):
        for raw, wide, kind in ((b"", b"", "float16"), (b"x", bytes(4), "bfloat16"),
                                (bytes(2), bytes(8), "float16"), (bytes(2), bytes(4), "unknown")):
            with self.assertRaises(ValueError):
                subject.verify_widening(raw, wide, kind)

    def test_complete_and_short_writes_retain_accepted_prefix(self):
        body = bytes(range(256)) * 300
        file = io.BytesIO();record = dict(bytes=0)
        subject.write_piece(file, body, record, hashlib.sha256())
        self.assertEqual(file.getvalue(), body)
        self.assertEqual(record, dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
        class Short(io.BytesIO):
            def write(self, value):
                return super().write(value[:7])
        file = Short();record = dict(bytes=0)
        with self.assertRaisesRegex(OSError, "Incomplete"):
            subject.write_piece(file, body, record, hashlib.sha256())
        self.assertEqual(record, dict(bytes=7, sha256=hashlib.sha256(body[:7]).hexdigest()))

    def test_added_shard_and_changed_original_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary);path = folder / "model.safetensors";path.write_bytes(b"original")
            pins = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
            subject.model_inventory(folder, pins)
            extra = folder / "model-extra.safetensors";extra.write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "single-shard"):
                subject.model_inventory(folder, pins)
            extra.unlink();path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs"):
                subject.model_inventory(folder, pins)

    def geometry(self):
        return dict(hidden_size=1024, vocab_size=151936, num_hidden_layers=28, intermediate_size=3072,
                    num_attention_heads=16, num_key_value_heads=8, head_dim=128)

    def test_source_phase_memory_admission_and_overflow_rejection(self):
        result = subject.memory_plan(self.geometry(), 633442994)
        self.assertEqual(result["load"]["fp32_model_initialization"], 2384199680)
        self.assertEqual(result["inference"]["full_output_max_fp32"], 311164928)
        self.assertLessEqual(result["admitted"], 6 * 1024**3)
        self.assertTrue(result["phases_are_separate"])
        with self.assertRaises(ValueError):
            subject.memory_plan(self.geometry(), 2 * 1024**3)
        bad = self.geometry();bad["num_hidden_layers"] = True
        with self.assertRaises(ValueError):
            subject.memory_plan(bad, 633442994)

    def test_local_transitive_sources_are_admitted_without_numerical_imports(self):
        paths = {p.resolve() for p in subject.source_paths()}
        for path in (Path(subject.__file__), Path(subject.calibration.__file__),
                     Path(subject.calibration.inputs.__file__), Path(subject.calibration.reference.__file__),
                     subject.calibration.reference.ROOT / "tools/experiments/native_usage.py"):
            self.assertIn(path.resolve(), paths)

    def test_cleanup_failure_preserves_explicit_failed_manifest(self):
        # Keep the numerical package alive across sys.modules mock restoration.
        import numpy
        self.assertEqual(numpy.__version__, "2.3.2")
        config = dict(**self.geometry(), model_type="qwen3", quantization=dict(bits=8, group_size=64), tie_word_embeddings=True)
        class Model:
            args = SimpleNamespace(tie_word_embeddings=True)
            def eval(self):
                pass
            def parameters(self):
                return {}
            def named_modules(self):
                return []
        def module(name, **attrs):
            result = ModuleType(name);result.__dict__.update(attrs);return result
        mx = module("mlx.core", gpu="test", set_default_device=lambda _: None, device_info=lambda: {},
            set_cache_limit=lambda _: 0, random=SimpleNamespace(seed=lambda _: None),
            get_active_memory=lambda: 0, get_peak_memory=lambda: 0, get_cache_memory=lambda: 0,
            clear_cache=mock.Mock(side_effect=RuntimeError("cleanup failure")))
        modules = {"mlx": module("mlx", core=mx), "mlx.core": mx,
            "mlx.utils": module("mlx.utils", tree_flatten=lambda _: []),
            "mlx_lm": module("mlx_lm"), "mlx_lm.models": module("mlx_lm.models"),
            "mlx_lm.models.qwen3": module("mlx_lm.models.qwen3", Model=Model, ModelArgs=object),
            "mlx_lm.utils": module("mlx_lm.utils", load_model=lambda *args, **kwargs: (Model(), config)),
            "native_usage": module("native_usage", usage=lambda: dict(lifetime_max_phys_footprint=1))}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, subject.ENVIRONMENT), mock.patch.object(subject, "isolation_profile", return_value=dict(test_only=True)), mock.patch.object(subject, "version", side_effect=lambda name: subject.VERSIONS[name]), mock.patch.object(subject, "model_inventory"), mock.patch.object(subject.calibration, "selected_inputs", return_value=(dict(profile={}), [])), mock.patch.object(subject.calibration.reference, "load", return_value=config):
            root = Path(temporary);model_folder = root / "model";model_folder.mkdir();(model_folder / "model.safetensors").write_bytes(b"model")
            report = {}
            with self.assertRaisesRegex(ValueError, "cleanup"):
                subject.export(root / "capture", root, model_folder, {"config.json": "test"}, report, lambda: {})
            saved = json.loads((root / "capture/manifest.json").read_bytes())
            self.assertFalse(saved["complete"])
            self.assertIn("cleanup failure", saved["primary_error"])
            self.assertEqual(len(saved["cleanup_errors"]), 1)

    def test_complete_source_readback_and_coherent_metadata_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary);raw = struct.pack("<4H", 0x3f80, 0x8000, 0x4000, 0xc040)
            wide = struct.pack("<4f", 1, -0.0, 2, -3);storage = {}
            for key, suffix, dtype, body in (("raw", ".source-bits.bin", "<u2", raw), ("widened", ".f32le.bin", "<f4", wide)):
                filename = "mini" + suffix;(folder / filename).write_bytes(body)
                storage[key] = dict(file=filename, dtype=dtype, layout="row-major [tokens,vocabulary]", bytes=len(body),
                                    intended_bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            plan = dict(cases=[dict(id="mini", rows=2)], outputs=dict(columns=2), model_pins=dict(test="identity"), memory=dict(test=1))
            tile = dict(start=0, rows=2, raw_offset=0, widened_offset=0, complete=True, values=4, nonfinite_values=0, finite_widening_mismatches=0)
            record = dict(**storage, rows=2, columns=2, source_dtype="bfloat16", complete=True,
                          nonfinite_values=0, finite_widening_mismatches=0, tiles=[tile])
            manifest = dict(complete=True, primary_error=None, cleanup_errors=[], model_pins=plan["model_pins"],
                memory_admission=plan["memory"], records=dict(mini=record), attempts=[dict(label="mini", completed=True,
                    evaluated=True, observed_shape=[1, 2, 2], observed_dtype="mlx.core.bfloat16")])
            def replay(value, report):
                body = json.dumps(value).encode();(folder / "manifest.json").write_bytes(body)
                return subject.verify_capture(folder, hashlib.sha256(body).hexdigest(), plan, report)
            report = {};self.assertEqual(replay(manifest, report), manifest["records"])
            self.assertTrue(report["complete"])
            self.assertEqual(report["values"], 4)
            self.assertTrue(all(row["endpoint_checked"] for row in report["parents"]))
            for change in (lambda v: v["attempts"][0].update(observed_shape=[True, 2, 2]),
                    lambda v: v["records"]["mini"]["tiles"][0].update(complete=1),
                    lambda v: v["records"]["mini"].update(nonfinite_values=False),
                    lambda v: v.update(cleanup_errors=["failure"])):
                changed = copy.deepcopy(manifest);change(changed)
                with self.assertRaises(ValueError):
                    replay(changed, {})
            changed = bytearray(raw);changed[:2] = struct.pack("<H", 0x3f81)
            (folder / storage["raw"]["file"]).write_bytes(changed)
            manifest["records"]["mini"]["raw"]["sha256"] = hashlib.sha256(changed).hexdigest()
            partial = {}
            with self.assertRaisesRegex(ValueError, "bit-derived"):
                replay(manifest, partial)
            self.assertFalse(partial["complete"])
            self.assertTrue(all(row["endpoint_checked"] for row in partial["parents"]))

    def test_unexpected_interpreter_prefix_rejects_before_imports(self):
        with mock.patch.object(sys, "prefix", "/not-the-reference-prefix"), self.assertRaisesRegex(ValueError, "interpreter"):
            subject.isolation_profile()


if __name__ == "__main__":
    unittest.main()
