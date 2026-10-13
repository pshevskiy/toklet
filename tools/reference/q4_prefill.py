"""Independent scalar Q4 tiny decoder oracle; freeze before running Metal."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import fp16
import q4_tiny

ROOT = Path(__file__).resolve().parents[2]
POLICY = q4_tiny.POLICY

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


reconstruct = q4_tiny.reconstruct

def export(packed, output):
    if output.exists(): raise ValueError("refusing to overwrite a frozen Q4 reference")
    if np.__version__ != "2.3.2": raise ValueError("use frozen NumPy 2.3.2")
    source_path = ROOT / "tests/fixtures/tiny-v1"
    source = json.loads((source_path/"manifest.json").read_text())
    manifest = json.loads((packed/"manifest.json").read_text())
    if manifest["quantizer"] != "toklet-q4-affine-f32-source-half-metadata-rne-v1" or len(manifest["matrices"]) != 14:
        raise ValueError("Q4 source representation mismatch")
    weights = {name.removeprefix("weights."): fp16.read(source_path, source, name).astype(np.float16)
               for name in source["arrays"] if name.startswith("weights.")}
    identities = {}
    for entry in manifest["matrices"]:
        path = packed / entry["file"]
        if digest(path) != entry["packed_sha256"]: raise ValueError("Q4 matrix identity mismatch")
        parts = entry["name"].split(".")
        key = f"layers.{parts[2]}.{parts[4]}.weight"
        original = fp16.read(source_path, source, f"weights.{key}")
        if list(original.shape) != [entry["rows"], entry["columns"]] or hashlib.sha256(original.astype('<f4').tobytes()).hexdigest() != entry["source_f32_sha256"]:
            raise ValueError("packed tensor does not bind frozen source")
        weights[key] = reconstruct(path.read_bytes(), *original.shape)
        identities[entry["name"]] = entry["packed_sha256"]
    output.mkdir(parents=True)
    arrays = {}; cases = {}
    pattern = [1,7,13,3,47,11,2,5] * 3
    inputs = {f"rows{n}":(np.asarray(pattern[:n],dtype=np.uint32),0) for n in (7,8,9,17,24)}
    inputs["offset17"] = (np.asarray(pattern[:17],dtype=np.uint32),5)
    for case,(tokens,offset) in inputs.items():
        computed = fp16.forward(tokens, offset, weights, source["model"], np.float32)
        wide = fp16.forward(tokens, offset, weights, source["model"], np.float64)
        cases[case] = {"tokens": tokens.tolist(), "position_offset": offset,
                       "greedy": np.argmax(computed["logits"], axis=1).tolist()}
        for stage, value in computed.items():
            if stage.endswith(("attention_scores", "attention_probabilities")): continue
            name = f"cases.{case}.{stage}"
            dtype = "f16le" if value.dtype == np.float16 else "f32le"
            filename = f"{name}.{dtype}"
            data = np.asarray(value, dtype='<f2' if dtype == "f16le" else '<f4').tobytes()
            (output/filename).write_bytes(data)
            envelope = float(np.max(np.abs(value.astype(np.float64)-wide[stage].astype(np.float64))))
            arrays[name] = {"file":filename,"shape":list(value.shape),"dtype":dtype,"bytes":len(data),
                           "sha256":hashlib.sha256(data).hexdigest(),"gate":{"atol":max(1e-4,2*envelope),"rtol":1e-3},
                           "independent_precision_envelope":envelope}
    record = {"format":"toklet-q4-tiny-prefill-numerical-v1","policy":POLICY,"numpy":np.__version__,
              "source_manifest_sha256":digest(source_path/"manifest.json"),
              "packed_manifest_sha256":digest(packed/"manifest.json"),"packed_matrix_hashes":identities,
              "oracle_sources":{p.name:digest(p) for p in (Path(__file__),Path(q4_tiny.__file__),Path(fp16.__file__))},
              "model":source["model"],"cases":cases,"arrays":arrays}
    (output/"manifest.json").write_text(json.dumps(record,indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packed",type=Path); parser.add_argument("output",type=Path)
    args = parser.parse_args(); export(args.packed,args.output)


if __name__ == "__main__": main()
