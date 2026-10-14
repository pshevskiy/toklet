import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import qwen_storage_v2_probes as probes


class StorageProbesTests(unittest.TestCase):
    def artifact(self, directory):
        full = directory / "full"
        full.mkdir()
        data = np.arange(8,dtype="<f4").tobytes()
        arrays = {}
        for name in [f"cases.a.stage{i}" for i in range(341)] + ["cases.a.logits"]:
            file = name + ".f32le"
            current = np.arange(200,dtype="<f4").tobytes() if name.endswith(".logits") else data
            (full / file).write_bytes(current)
            arrays[name] = {"file":file,"shape":[2,100] if name.endswith(".logits") else [2,4],"dtype":"f32le","bytes":len(current),
                            "sha256":hashlib.sha256(current).hexdigest(),"gate":{"atol":.0001,"rtol":.001}}
        manifest = {"format":"toklet-streamed-qwen-storage-oracle-v2","q4_version":2,"source_fp32_cases_available":["a"],"arrays":arrays,
                    "argmax":{"a":{"stored_fp32":[2,2],"stored_fp64":[4,4],"source_fp32":[5,5]}}}
        path = full / "manifest.json"
        path.write_text(json.dumps(manifest))
        return full,manifest,path

    def test_full_identity_payload_shape_and_argmax_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full,_,manifest = self.artifact(root)
            identity = probes.storage.sha(manifest)
            output = root / "probes"
            probes.export(full,output,identity)
            frozen = json.loads((output / "manifest.json").read_text())
            self.assertEqual(frozen["full_manifest_sha256"],identity)
            self.assertEqual(len(frozen["arrays"]),342)
            for name,item in frozen["arrays"].items():
                self.assertEqual(probes.storage.sha(output/item["file"]),item["sha256"])
                if name.endswith(".logits"):
                    coordinates = {2,102,4,104,5,105}
                    base = set(np.linspace(0,199,64,dtype=np.int64).tolist())
                    self.assertTrue(coordinates-base)
                    self.assertEqual(set(item["indices"]),base|coordinates)
                    self.assertEqual(item["full_array_bytes"],800)
                else:
                    self.assertEqual(item["indices"],list(range(8)))
                    self.assertEqual(item["full_array_bytes"],32)
            with self.assertRaisesRegex(ValueError,"existing"):
                probes.export(full,output,identity)

    def test_corrupted_payload_rejected_without_completed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full,manifest,path = self.artifact(root)
            identity = probes.storage.sha(path)
            (full/next(iter(manifest["arrays"].values()))["file"]).write_bytes(b"wrong")
            output = root / "out"
            with self.assertRaisesRegex(ValueError,"identity"):
                probes.export(full,output,identity)
            self.assertFalse((output / "manifest.json").exists())

    def test_overlapping_destination_and_invalid_shape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full,manifest,path = self.artifact(root)
            with self.assertRaisesRegex(ValueError,"overlapping"):
                probes.export(full,full/"probes",probes.storage.sha(path))
            next(iter(manifest["arrays"].values()))["shape"] = [-2,-4]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"shape"):
                probes.export(full,root/"out",probes.storage.sha(path))

    def test_missing_argmax_family_rejected_without_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full,manifest,path = self.artifact(root)
            del manifest["argmax"]["a"]["source_fp32"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"argmax inventory"):
                probes.export(full,root/"out",probes.storage.sha(path))
            self.assertFalse((root/"out/manifest.json").exists())

    def test_new_shape_cases_do_not_invent_unavailable_source_argmax(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);full,manifest,path=self.artifact(root)
            manifest['source_fp32_cases_available']=[];del manifest['argmax']['a']['source_fp32']
            path.write_text(json.dumps(manifest));probes.export(full,root/'out',probes.storage.sha(path))
            result=json.loads((root/'out/manifest.json').read_text())
            self.assertEqual(set(result['argmax']['a']),{'stored_fp32','stored_fp64'})
            self.assertFalse(result['source_fp32_cases_available'])


if __name__ == "__main__":
    unittest.main()
