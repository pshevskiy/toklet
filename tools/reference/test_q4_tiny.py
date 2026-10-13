"""Exact independent Q4 oracle regeneration; CPU-only small frozen fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import q4_tiny
import fp16
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"quantization"))
import q4


class Oracle(unittest.TestCase):
    def test_independent_unpack_matches_manual_all_codes_and_tail(self):
        directory=q4_tiny.ROOT/"tests/fixtures/q4"
        manifest=json.loads((directory/"manifest.json").read_text())
        actual=q4_tiny.reconstruct((directory/"rows.q4").read_bytes(),2,65)
        self.assertEqual(actual.tolist(),manifest["reconstructed_rows"])

    def test_exact_frozen_reference_regeneration_without_local_models(self):
        root=q4_tiny.ROOT
        record=json.loads((root/"benchmarks/results/2026-10-13-q4-artifact.json").read_text())
        packed_manifest=record["packed_artifacts"]["tiny"]["manifest"]
        source_path=root/"tests/fixtures/tiny-v1"
        source=json.loads((source_path/"manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            packed=Path(directory)/"packed";packed.mkdir();output=Path(directory)/"oracle"
            (packed/"manifest.json").write_text(json.dumps(packed_manifest,indent=2)+"\n")
            for entry in packed_manifest["matrices"]:
                parts=entry["name"].split(".");key=f"weights.layers.{parts[2]}.{parts[4]}.weight"
                values=fp16.read(source_path,source,key)
                data=q4.pack_matrix(values.tolist(),values.shape[1])
                self.assertEqual(q4_tiny.digest(packed/"manifest.json"),record["packed_artifacts"]["tiny"]["manifest_sha256"])
                (packed/entry["file"]).write_bytes(data)
            q4_tiny.export(packed,output)
            frozen=root/"tests/fixtures/q4-tiny-v1"
            self.assertEqual(sorted(p.name for p in output.iterdir()),sorted(p.name for p in frozen.iterdir()))
            for file in output.iterdir():self.assertEqual(file.read_bytes(),(frozen/file.name).read_bytes(),file.name)


if __name__=="__main__":unittest.main()
