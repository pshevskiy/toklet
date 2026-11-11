"""Portable negative controls for the external-engine metadata boundary."""
import hashlib
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import lfm3_external_config as view


class ExternalConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        config = dict(model_type="lfm3", architectures=["Lfm3ForCausalLM"],
                      dtype="bfloat16", model_file="lfm3.py", hidden_size=2048,
                      rope_parameters={"full_attention": {"rope_theta": 1000000}})
        self.pins = {}
        for name, key in view.ASSETS.items():
            data = json.dumps(config).encode() if name == "config.json" else (name + " fixture").encode()
            (self.model / name).write_bytes(data)
            self.pins[key] = hashlib.sha256(data).hexdigest()
        self.configuration = self.root / "configuration.py"
        self.configuration.write_text("# pinned local config fixture\n")
        self.pins["parent-configuration_lfm3.py"] = view.digest(self.configuration)
        (self.model / "model.safetensors").write_bytes(b"synthetic-weight-bytes")
        self.contract = self.root / "contract.json"
        self.contract.write_text(json.dumps(dict(metadata_sha256=self.pins, candidate=dict(
            bytes=(self.model / "model.safetensors").stat().st_size,
            sha256=view.digest(self.model / "model.safetensors")))))
        self.pin = patch.object(view, "CONTRACT_SHA256", view.digest(self.contract))
        self.pin.start()
        self.addCleanup(self.pin.stop)
        self.output = self.root / "view"

    def run_view(self, provider="mlx-vlm"):
        return view.prepare(self.model, self.configuration, self.contract, self.output, provider)

    def test_no_weights_and_only_declared_dispatch_changes(self):
        original = {path.name: path.read_bytes() for path in self.model.iterdir()}
        receipt = self.run_view()
        config = json.loads((self.output / "config.json").read_bytes())
        source = json.loads(original["config.json"])
        del config["auto_map"]
        source.pop("model_file")
        self.assertEqual(config, source)
        self.assertFalse((self.output / "model.safetensors").exists())
        self.assertFalse((self.output / "lfm3.py").exists())
        self.assertFalse(receipt["model_load_ready"])
        self.assertEqual(original, {path.name: path.read_bytes() for path in self.model.iterdir()})
        for name in receipt["files_sha256"]:
            path = self.output / name
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.stat().st_nlink, 1)

    def test_custom_module_is_preserved_without_execution(self):
        receipt = self.run_view("mlx-lm")
        self.assertEqual((self.output / "lfm3.py").read_bytes(), (self.model / "lfm3.py").read_bytes())
        self.assertEqual(list(receipt["config_changes"]), ["auto_map"])

    def test_changed_weight_fails_before_writes(self):
        weights = self.model / "model.safetensors"
        weights.write_bytes(weights.read_bytes()[::-1])
        with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
            self.run_view()
        self.assertFalse(self.output.exists())

    def test_source_change_during_preparation_cannot_publish_receipt(self):
        weights = self.model / "model.safetensors"
        original_digest = view.digest
        reads = 0

        def changing_digest(path):
            nonlocal reads
            if path == weights:
                reads += 1
                if reads == 2:
                    weights.write_bytes(weights.read_bytes()[::-1])
            return original_digest(path)

        with patch.object(view, "digest", changing_digest):
            with self.assertRaisesRegex(ValueError, "Source changed during preparation"):
                self.run_view()
        self.assertFalse((self.output / "receipt.json").exists())

    def test_changed_metadata_fails_before_writes(self):
        (self.model / "tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Metadata changed"):
            self.run_view()
        self.assertFalse(self.output.exists())

    def test_changed_config_code_and_contract_reject(self):
        self.configuration.write_text("raise RuntimeError('must not execute')")
        with self.assertRaisesRegex(ValueError, "Metadata changed"):
            self.run_view()
        self.contract.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Metadata changed"):
            self.run_view()
        self.assertFalse(self.output.exists())

    def test_existing_output_and_checkpoint_subdirectory_reject(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "new directory outside"):
            self.run_view()
        self.output = self.model / "derived"
        with self.assertRaisesRegex(ValueError, "new directory outside"):
            self.run_view()
        self.assertFalse(self.output.exists())

    def test_metadata_growth_cannot_exceed_read_bound(self):
        path = self.root / "growing-metadata"
        path.write_bytes(b"x" * 64)
        # Simulate a file growing after the admission stat but before its read.
        old_stat = SimpleNamespace(st_mode=stat.S_IFREG, st_size=32)
        with patch.object(view, "MAX_METADATA_BYTES", 32), patch.object(Path, "stat", return_value=old_stat):
            with self.assertRaisesRegex(ValueError, "grew beyond its read bound"):
                view.bound_bytes(path, view.digest(path))


if __name__ == "__main__":
    unittest.main()
