"""Select bounded probes from an already frozen, hash-admitted storage oracle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import qwen_storage_v2 as storage


def export(full, output, expected_sha):
    if np.__version__ != "2.3.2":
        raise ValueError("Frozen probe NumPy version differs")
    full, output = full.resolve(), output.resolve()
    if output.exists() or full == output or full in output.parents or output in full.parents:
        raise ValueError("Refusing existing or overlapping probe output")
    data, identity = storage.bounded_bytes(full / "manifest.json", 1024**2, expected_sha)
    manifest = json.loads(data)
    if manifest["format"] != "toklet-streamed-qwen-storage-oracle-v2" or manifest.get("q4_version") != 2 or len(manifest["arrays"]) != 342:
        raise ValueError("Frozen storage oracle inventory differs")
    if sum(item["bytes"] for item in manifest["arrays"].values()) > 128*1024**2:
        raise ValueError("Frozen storage payload exceeds bound")
    result = {key: value for key, value in manifest.items() if key != "arrays"}
    helper_sha = storage.sha(Path(__file__))
    result.update(full_manifest_sha256=identity, probe_exporter_sha256=helper_sha, arrays={})
    output.mkdir(parents=True)
    for name, item in manifest["arrays"].items():
        dtype = {"f16le": "<f2", "f32le": "<f4"}[item["dtype"]]
        if item["file"] != f"{name}.{item['dtype']}" or "/" in name or "\\" in name:
            raise ValueError("Frozen storage array filename differs")
        if not 0 < item["bytes"] <= 64*1024**2:
            raise ValueError("Frozen storage array exceeds bound")
        if not 1 <= len(item["shape"]) <= 3 or any(type(v) is not int or v<=0 for v in item["shape"]):
            raise ValueError("Frozen storage array shape differs")
        data, _ = storage.bounded_bytes(full / item["file"], item["bytes"], item["sha256"])
        elements = int(np.prod(item["shape"], dtype=object))
        if len(data) != item["bytes"] or elements*np.dtype(dtype).itemsize != len(data):
            raise ValueError("Frozen storage array shape differs")
        values = np.frombuffer(data, dtype=dtype)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite frozen storage array")
        indices = set(np.linspace(0, elements-1, min(64,elements), dtype=np.int64).tolist())
        if name.endswith(".logits"):
            case = name.split(".")[1]
            families = manifest["argmax"][case]
            expected_families={"stored_fp32","stored_fp64"}
            if case in manifest["source_fp32_cases_available"]: expected_families.add("source_fp32")
            if set(families) != expected_families:
                raise ValueError("Frozen storage argmax inventory differs")
            for ids in families.values():
                if len(item["shape"]) != 2 or len(ids) != item["shape"][0] or any(type(token) is not int or not 0<=token<item["shape"][1] for token in ids):
                    raise ValueError("Frozen storage argmax shape differs")
                for position, token in enumerate(ids):
                    indices.add(position*item["shape"][-1]+token)
        indices = sorted(indices)
        selected = values[indices].tobytes()
        (output / item["file"]).write_bytes(selected)
        result["arrays"][name] = {**item, "indices": indices,
            "full_array_sha256": item["sha256"], "full_array_bytes": item["bytes"],
            "sha256": hashlib.sha256(selected).hexdigest(), "bytes": len(selected)}
    if storage.sha(full / "manifest.json") != identity:
        raise ValueError("Frozen oracle endpoint differs")
    if storage.sha(Path(__file__)) != helper_sha:
        raise ValueError("Probe helper endpoint differs")
    (output / "manifest.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("expected_sha256")
    args = parser.parse_args()
    export(args.full,args.output,args.expected_sha256)
