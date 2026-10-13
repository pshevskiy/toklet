"""Streaming independent Half/Q4 full-array references, frozen before GPU.

Uses the already qualified functional storage-rounding math. Half gates remain
against independent source FP32. Q4 gates compare reconstructed Q4 math; actual
lossy quality is tested separately against the matched unquantized checkpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/reference"))
sys.path.insert(0, str(ROOT / "tools/experiments"))
import numpy as np
import native_usage
import qwen_fp16
import q4_tiny
import qwen_storage as storage


def publish_manifest(output, manifest, receipt):
    # Complete serialization and its temporary UTF8/file buffers before observing
    # lifetime charge. Rejected runs never publish the ordinary artifact marker.
    pending = output / ".manifest.pending"
    with pending.open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    usage = native_usage.usage()
    receipt["native_samples"].append({"phase": "completed", "usage": usage})
    if not 0 < usage["lifetime_max_phys_footprint"] <= 6 * 1024**3:
        raise ValueError("Completed native reference exceeds budget")
    pending.rename(output / "manifest.json")
    return storage.sha(output / "manifest.json")


def q4_gates(reconstructed, cases):
    pooled, equivalent = {}, {}
    for key, (main, wide) in reconstructed.items():
        name = key.split(".", 2)[2]
        delta = float(np.max(np.abs(main.astype(np.float64) - wide)))
        pooled[name] = max(pooled.get(name, 0), delta)
    for first, a in cases.items():
        for second, b in cases.items():
            if first >= second or a["offset"] != b["offset"]:
                continue
            common = 0
            for x, y in zip(a["tokens"], b["tokens"]):
                if x != y:
                    break
                common += 1
            if not common:
                continue
            for name in pooled:
                for precision in (0, 1):
                    left = reconstructed[f"cases.{first}.{name}"][precision]
                    right = reconstructed[f"cases.{second}.{name}"][precision]
                    if name.endswith((".attention_scores", ".attention_probabilities")):
                        left, right = left[:, :common, :common], right[:, :common, :common]
                    else:
                        left, right = left[:common], right[:common]
                    delta = float(np.max(np.abs(left.astype(np.float64) - right)))
                    equivalent[name] = max(equivalent.get(name, 0), delta)
    return {key: {"atol": max(1e-4, 2 * pooled[key.split('.', 2)[2]], 2 * equivalent.get(key.split('.', 2)[2], 0)),
                  "rtol": 1e-3, "pooled_fp32_fp64_max_abs": pooled[key.split('.', 2)[2]],
                  "equivalent_prefix_max_abs": equivalent.get(key.split('.', 2)[2], 0)}
            for key in reconstructed}


def export(model, baseline, packed, output, record):
    output, record = output.resolve(), record.resolve()
    if output.exists() or record.exists() or output == record or output in record.parents or record in output.parents:
        raise ValueError("Refusing existing or overlapping storage oracle evidence")
    if np.__version__ != "2.3.2":
        raise ValueError("Frozen NumPy version differs")
    full_pins = {
        "Qwen2.5-0.5B-Instruct": "bdd368d21dfae574197e3b64423b75d4f1b2b37c083c6172c429e36bb4dd69c7",
        "Qwen2.5-1.5B-Instruct": "676cf5392c0dcb8ceff8d098fae21378b0dfdca92b69f9de7aac48bc23e234fe",
    }
    if model.name not in full_pins: raise ValueError("Unsupported frozen reference checkpoint")
    source_data, source_manifest_sha = storage.bounded_bytes(baseline / "manifest.json", 1024**2, full_pins[model.name])
    source = json.loads(source_data)
    source_identity = source["model"]["weights"]
    pins = {
        "Qwen2.5-0.5B-Instruct": ("fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe", "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"),
        "Qwen2.5-1.5B-Instruct": ("dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee", "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"),
    }
    if model.name not in pins or source_identity["sha256"] != pins[model.name][0]:
        raise ValueError("Only frozen Qwen checkpoint family admitted")
    config_data, config_sha = storage.bounded_bytes(model / "config.json", 1024**2)
    if config_sha != pins[model.name][1]:
        raise ValueError("Pinned source config differs")
    config = json.loads(config_data)
    if not config["tie_word_embeddings"] or config["use_sliding_window"] or config["hidden_act"] != "silu":
        raise ValueError("Unsupported independent reconstruction configuration")
    numerical = {"num_layers": config["num_hidden_layers"], "num_query_heads": config["num_attention_heads"],
                 "num_kv_heads": config["num_key_value_heads"], "head_dim": config["hidden_size"] // config["num_attention_heads"],
                 "rms_norm_eps": config["rms_norm_eps"], "rope_theta": config["rope_theta"]}
    cases, layers = source["cases"], source["layers"]
    expected_cases = {
        "single": {"tokens": [9707], "offset": 0},
        "short": {"tokens": [9707, 11, 358, 2776], "offset": 0},
        "future_changed": {"tokens": [9707, 11, 358, 498], "offset": 0},
        "offset": {"tokens": [11, 358, 2776], "offset": 5},
    }
    expected_layers = [0, 1, 27 if model.name == "Qwen2.5-1.5B-Instruct" else 23]
    stages = ("input_norm", "q_projection", "k_projection", "v_projection", "q_rotary", "k_rotary",
              "attention_scores", "attention_probabilities", "attention_context", "attention_projection",
              "attention_residual", "post_attention_norm", "gate_projection", "up_projection", "gate_silu",
              "gated", "down_projection", "output")
    expected_arrays = {f"cases.{case}.{name}" for case in expected_cases for name in
                       ["embedding", "final_norm", "logits"] + [f"layers.{layer}.{stage}" for layer in expected_layers for stage in stages]}
    if cases != expected_cases or layers != expected_layers or set(source["arrays"]) != expected_arrays:
        raise ValueError("Frozen bounded reference inventory differs")
    output.mkdir(parents=True)
    try:
        receipt_stream = record.open("x")
    except BaseException:
        output.rmdir()
        raise
    started = time.monotonic()
    receipt = {"format": "toklet-streamed-storage-oracle-export-v1", "completed": False,
               "scope": "Independent short full-array reference export only; no device, lossy quality or performance qualification",
               "command": sys.argv, "native_samples": []}
    try:
        with storage.Weights(model / "model.safetensors", source_identity["sha256"], packed) as weights:
            shapes = [item["shape"] for key, item in weights.header.items() if key != "__metadata__"]
            embedding = weights.header["model.embed_tokens.weight"]["shape"]
            max_elements = max(np.prod(s, dtype=object) for s in shapes)
            largest_other = max(np.prod(item["shape"], dtype=object) for key, item in weights.header.items()
                                if key not in ("__metadata__", "model.embed_tokens.weight"))
            maximum_columns = max(s[-1] for s in shapes if len(s) == 2)
            # One tied Half table, one current matrix, bounded source/BLAS/decode
            # stripes, packed bytes, and fixed framework/source/trace allowance.
            admission = {"tied_half_bytes": int(np.prod(embedding, dtype=object)) * 2,
                         "largest_matrix_bytes": int(largest_other) * (4 if packed else 2),
                         "source_stripes_bytes": storage.STRIPE * maximum_columns * 6,
                         "fp64_conversion_stripe_bytes": qwen_fp16.ROW_STRIPE * maximum_columns * 8,
                         "q4_decode_stripes_bytes": storage.DECODE_ROWS * ((maximum_columns + 63) // 64) * 64 * 9 if packed else 0,
                         "maximum_packed_bytes": max((v["packed_bytes"] for v in weights.entries.values()), default=0),
                         "application_reference_allowance_bytes": 512 * 1024**2}
            admission["admitted_bytes"] = sum(admission.values())
            if admission["admitted_bytes"] > 6 * 1024**3 or max_elements > 151936 * 1536:
                raise ValueError("Independent reference exceeds bounded admission")
            receipt["admission"] = admission
            identities = {"source_sha256": source_identity["sha256"], "config_sha256": config_sha,
                          "baseline_manifest_sha256": source_manifest_sha,
                          "oracle_sources": {p.name: storage.sha(p) for p in (Path(__file__), Path(storage.__file__), Path(qwen_fp16.__file__), Path(q4_tiny.__file__), Path(native_usage.__file__))}}
            if packed:
                identities.update(packed_manifest_sha256=weights.manifest_sha,
                                  packed_matrix_hashes={k: v["packed_sha256"] for k, v in weights.entries.items()})
            reconstructed = {}
            for case, spec in cases.items():
                main = qwen_fp16.forward(spec["tokens"], spec["offset"], weights, numerical, np.float32, layers)
                wide = qwen_fp16.forward(spec["tokens"], spec["offset"], weights, numerical, np.float64, layers)
                reconstructed.update({f"cases.{case}.{name}": (value, wide[name]) for name, value in main.items()})
                usage = native_usage.usage()
                receipt["native_samples"].append({"phase": case, "usage": usage})
                if not 0 < usage["lifetime_max_phys_footprint"] <= 6 * 1024**3:
                    raise ValueError("Independent native reference exceeds budget")
                print(f"Reconstructed {case}: {len(main)} arrays", flush=True)
            if set(reconstructed) != set(source["arrays"]):
                raise ValueError("Independent source/stored reconstruction inventory differs")
            reference = {}
            for name, item in source["arrays"].items():
                if item.get("dtype", "f32le") != "f32le" or item["bytes"] > 64 * 1024**2:
                    raise ValueError("Independent FP32 source shape exceeds bound")
                data, _ = storage.bounded_bytes(baseline / item["file"], item["bytes"], item["sha256"])
                if len(data) != item["bytes"] or len(data) != int(np.prod(item["shape"], dtype=object))*4:
                    raise ValueError("Independent FP32 source array size differs")
                reference[name] = np.frombuffer(data, dtype="<f4").reshape(item["shape"])
                if not np.isfinite(reference[name]).all() or reconstructed[name][0].shape != reference[name].shape or reconstructed[name][1].shape != reference[name].shape:
                    raise ValueError("Independent reconstruction/source array differs")
            limits = q4_gates(reconstructed, cases) if packed else qwen_fp16.gates(reconstructed, reference, cases)
            manifest = {"format": "toklet-streamed-qwen-storage-oracle-v1", "dtype": "Q4" if packed else "Half",
                        "policy": {"reference": "Independent NumPy BLAS/reduction FP32 and FP64 with reconstructed Q4 FP32 matrices and Half exclusions/activations",
                                   "atol_floor": 1e-4, "envelope_multiplier": 2.0, "rtol": 1e-3,
                                   "gate": "max(atol_floor, 2*pooled_FP32_FP64_max_abs, 2*equivalent_prefix_max_abs) + rtol*abs(reference)",
                                   "case_scope": "same operation/layer across frozen source cases plus equivalent prefixes"} if packed else qwen_fp16.POLICY,
                        "model": source["model"], "numerical_config": numerical, "cases": cases, "layers": layers,
                        "identities": identities, "memory": admission, "numpy": np.__version__, "arrays": {}, "argmax": {}}
            for name, (main, wide) in sorted(reconstructed.items()):
                dtype = "f16le" if main.dtype == np.float16 else "f32le"
                data = main.astype("<f2" if dtype == "f16le" else "<f4", copy=False).tobytes()
                file = f"{name}.{dtype}"
                (output / file).write_bytes(data)
                manifest["arrays"][name] = {"file": file, "shape": list(main.shape), "dtype": dtype, "bytes": len(data),
                                             "sha256": hashlib.sha256(data).hexdigest(), "gate": limits[name],
                                             "source_fp32_sha256": source["arrays"][name]["sha256"]}
                if name.endswith(".logits"):
                    manifest["argmax"][name.split('.')[1]] = {"stored_fp32": np.argmax(main, axis=1).tolist(),
                        "stored_fp64": np.argmax(wide, axis=1).tolist(), "source_fp32": np.argmax(reference[name], axis=1).tolist()}
            weights.verify()
            if storage.sha(model / "config.json") != config_sha or storage.sha(baseline / "manifest.json") != source_manifest_sha:
                raise ValueError("Independent oracle endpoint differs")
            for p in (Path(__file__), Path(storage.__file__), Path(qwen_fp16.__file__), Path(q4_tiny.__file__), Path(native_usage.__file__)):
                identity = identities["oracle_sources"][p.name]
                if storage.sha(p) != identity: raise ValueError("Independent helper endpoint differs")
            manifest_sha = publish_manifest(output, manifest, receipt)
            receipt.update(completed=True, manifest_sha256=manifest_sha, identities=identities)
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["functional_export_wall_seconds"] = time.monotonic() - started
        try:
            json.dump(receipt, receipt_stream, indent=2, allow_nan=False)
            receipt_stream.write("\n")
        finally:
            receipt_stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("record", type=Path)
    parser.add_argument("--packed", type=Path)
    args = parser.parse_args()
    export(args.model, args.baseline, args.packed, args.output, args.record)
