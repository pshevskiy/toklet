import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import model
import q4


def source(directory, dtype="F32", split=False):
    directory.mkdir()
    config = {"model_type": "qwen2", "tie_word_embeddings": True, "hidden_size": 4,
              "intermediate_size": 5, "num_attention_heads": 1, "num_key_value_heads": 1, "num_hidden_layers": 1}
    (directory / "config.json").write_text(json.dumps(config))
    shapes = model.dimensions(config)
    names = sorted(shapes)
    groups = [names[:3], names[3:]] if split else [names]
    weight_map = {}
    for index, group in enumerate(groups):
        filename = f"part{index}.safetensors" if split else "model.safetensors"
        header = {}; payload = bytearray()
        for name in group:
            rows, columns = shapes[name]
            values = [float(i % 8) - 2 for i in range(rows * columns)]
            if dtype == "F32": data = struct.pack(f"<{len(values)}f", *values)
            elif dtype == "F16": data = struct.pack(f"<{len(values)}e", *values)
            else: data = b"".join(struct.pack("<f", v)[2:] for v in values)
            start = len(payload); payload.extend(data)
            header[name] = {"dtype": dtype, "shape": [rows, columns], "data_offsets": [start, len(payload)]}
            weight_map[name] = filename
        metadata = json.dumps(header, separators=(",", ":")).encode(); metadata += b" " * (-len(metadata) % 8)
        (directory / filename).write_bytes(struct.pack("<Q", len(metadata)) + metadata + payload)
    if split: (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return shapes


class ModelArtifact(unittest.TestCase):
    def test_streamed_f32_f16_bf16_and_shard_identity(self):
        for dtype, split in [("F32", False), ("F16", True), ("BF16", True)]:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "source"; dst = Path(tmp) / "packed"
                shapes = source(src, dtype, split)
                with patch("sys.stderr"):
                    manifest = model.pack_model(src, dst)
                self.assertEqual(manifest, json.loads((dst / "manifest.json").read_text()))
                self.assertEqual(manifest["config_sha256"], model.digest(src / "config.json"))
                self.assertEqual(len(manifest["matrices"]), 7)
                for entry in manifest["matrices"]:
                    rows, columns = shapes[entry["name"]]
                    values = [float(i % 8) - 2 for i in range(rows * columns)]
                    expected = q4.pack_matrix([values[i:i+columns] for i in range(0, len(values), columns)], columns)
                    self.assertEqual((dst / entry["file"]).read_bytes(), expected)
                    self.assertEqual(entry["packed_sha256"], hashlib.sha256(expected).hexdigest())
                    self.assertEqual(entry["source_f32_sha256"], hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest())
                self.assertEqual(manifest["packed_matrix_bytes"], sum(e["packed_bytes"] for e in manifest["matrices"]))
                for entry in manifest["sources"]: self.assertEqual(entry["sha256"], model.digest(src / entry["file"]))
                with self.assertRaises(FileExistsError): model.pack_model(src, dst)
                self.assertEqual(json.loads((dst / "manifest.json").read_text()), manifest)

    def test_bad_shapes_names_offsets_duplicate_json_and_missing_matrices(self):
        for name in ["", "../part.safetensors", "x/y", "x\\y", ".", ".."]:
            with self.assertRaises(ValueError): model.plain(name)
        with self.assertRaises(ValueError): model.unique([("a", 1), ("a", 2)])
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"; source(src)
            file = src / "model.safetensors"; raw = file.read_bytes()
            count = int.from_bytes(raw[:8], "little"); header = json.loads(raw[8:8+count])
            first = next(iter(header)); header[first]["shape"][0] += 1
            metadata = json.dumps(header).encode()
            file.write_bytes(struct.pack("<Q", len(metadata)) + metadata + raw[8+count:])
            with self.assertRaisesRegex(ValueError, "shape"): model.pack_model(src, Path(tmp) / "packed")
            self.assertFalse((Path(tmp) / "packed").exists())
            header[first]["data_offsets"][0] = 1
            metadata = json.dumps(header).encode(); file.write_bytes(struct.pack("<Q", len(metadata)) + metadata + raw[8+count:])
            with self.assertRaisesRegex(ValueError, "gaps"): model.shards(src)

    def test_changed_source_has_no_usable_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"; source(src); dst = Path(tmp) / "packed"
            real_pack = q4.pack_row; changed = False
            def changed_pack(values):
                nonlocal changed
                if not changed:
                    with (src / "config.json").open("a") as file: file.write(" ")
                    changed = True
                return real_pack(values)
            with patch.object(q4, "pack_row", changed_pack), patch("sys.stderr"):
                with self.assertRaisesRegex(ValueError, "identities changed"): model.pack_model(src, dst)
            self.assertFalse((dst / "manifest.json").exists())

    def test_nonfinite_payload_never_publishes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"; source(src); dst = Path(tmp) / "packed"
            file = src / "model.safetensors"; raw = bytearray(file.read_bytes())
            start = 8 + int.from_bytes(raw[:8], "little"); raw[start:start+4] = struct.pack("<f", float("nan")); file.write_bytes(raw)
            with patch("sys.stderr"), self.assertRaisesRegex(ValueError, "finite"): model.pack_model(src, dst)
            self.assertFalse((dst / "manifest.json").exists())
            self.assertFalse((dst / "model.layers.0.mlp.down_proj.weight.q4").exists())
            # Earlier complete matrices are retained for failure investigation.
            self.assertEqual(len(list(dst.iterdir())), 6)

    def test_config_mutation_between_parse_and_original_first_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"; source(src); dst = Path(tmp) / "packed"
            real_dimensions = model.dimensions
            def mutate(config):
                result = real_dimensions(config)
                with (src / "config.json").open("a") as file: file.write(" ")
                return result
            with patch.object(model, "dimensions", mutate), patch("sys.stderr"):
                with self.assertRaisesRegex(ValueError, "identities changed"): model.pack_model(src, dst)
            self.assertFalse((dst / "manifest.json").exists())

    def test_header_mutation_between_parse_and_original_first_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"; source(src); dst = Path(tmp) / "packed"
            real_loads = json.loads; changed = False
            def mutate(data, *args, **kwargs):
                nonlocal changed
                parsed = real_loads(data, *args, **kwargs)
                if not changed and isinstance(parsed, dict) and any(name.startswith("model.layers.") for name in parsed):
                    file = src / "model.safetensors"
                    raw = file.read_bytes(); file.write_bytes(raw.replace(b'"F32"', b'"F16"', 1))
                    changed = True
                return parsed
            with patch.object(json, "loads", mutate), patch("sys.stderr"):
                with self.assertRaisesRegex(ValueError, "identities changed"): model.pack_model(src, dst)
            self.assertTrue(changed)
            self.assertFalse((dst / "manifest.json").exists())

    def test_json_growth_after_stat_is_bounded_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "config.json"; file.write_bytes(b"{}")
            real_stat = os.fstat
            def mutate(fd):
                result = real_stat(fd)
                with file.open("ab") as output: output.write(b" " * (1024 * 1024 + 32))
                return result
            with patch.object(os, "fstat", mutate):
                with self.assertRaisesRegex(ValueError, "bounded read"): model.read_json(file)

    def test_final_manifest_flush_failure_preserves_only_incomplete_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"; source(src); dst = Path(tmp) / "packed"
            real_open = Path.open
            class FailingExit:
                def __init__(self, file): self.file = file
                def __enter__(self): return self.file
                def __exit__(self, *args):
                    self.file.close(); raise OSError("manifest flush failed")
            def opened(path, *args, **kwargs):
                file = real_open(path, *args, **kwargs)
                return FailingExit(file) if path == dst / "manifest.json" else file
            with patch.object(Path, "open", opened), patch("sys.stderr"):
                with self.assertRaisesRegex(OSError, "manifest flush"): model.pack_model(src, dst)
            self.assertFalse((dst / "manifest.json").exists())
            self.assertEqual(len(list(dst.iterdir())), 7)


if __name__ == "__main__": unittest.main()
