"""Independent Qwen FP16 storage references; never import or execute Metal/Rust."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import time

# Bound BLAS parallelism before NumPy is imported by an explicit export process.
if __name__ == "__main__":
    for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bench"))
import prepare
import qwen

SOURCE = ROOT / "artifacts/qwen-forward-v2"
SOURCE_PROBES = ROOT / "tests/fixtures/qwen-v2"
FIXTURE = ROOT / "tests/fixtures/qwen-fp16-v1"
ROW_STRIPE = 1024
VERSIONS = {"numpy": "2.3.2", "torch": "2.9.1", "safetensors": "0.8.0"}
POLICY = {"atol_floor": 1e-4, "envelope_multiplier": 2.0, "rtol": 1e-4,
          "case_scope": "same named operation and layer across the four frozen source cases",
          "absolute_limit": "max(atol_floor, 2 * pooled_FP16_FP32_or_FP64_vs_source_max_absolute_error, 2 * equivalent_prefix_FP16_max_absolute_error)",
          "reference": "independent NumPy CPU FP16 storage with FP32 and separate FP64 arithmetic; gates against unchanged qwen-v2 FP32 source"}


def linear(x, weight, bias, precision, output_dtype=np.float16):
    """[tokens,in] x [out,in]; bias after complete dot, before one storage cast.

    Convert at most 1,024 weight rows at a time; no full FP32/FP64 model copy.
    BLAS reduction order is not claimed to match the scalar tiny oracle.
    """
    output = np.empty((len(x), len(weight)), dtype=output_dtype)
    wide_x = x.astype(precision)
    for start in range(0, len(weight), ROW_STRIPE):
        end = min(start + ROW_STRIPE, len(weight))
        values = wide_x @ weight[start:end].astype(precision).T
        if bias is not None:
            values += bias[start:end].astype(precision)
        output[:, start:end] = values
    return output


def rmsnorm(x, weight, epsilon, precision):
    wide = x.astype(precision)
    mean = np.mean(wide * wide, axis=-1, keepdims=True, dtype=precision)
    inverse = precision(1) / np.sqrt(mean + precision(epsilon))
    return (wide * inverse * weight.astype(precision)).astype(np.float16)


def rotary(x, offset, theta, precision):
    dimension = x.shape[-1]
    half = dimension // 2
    frequencies = precision(theta) ** (-np.arange(half, dtype=precision) * precision(2 / dimension))
    angles = np.arange(offset, offset + len(x), dtype=precision)[:, None, None] * frequencies[None, None]
    first, second = x[..., :half].astype(precision), x[..., half:].astype(precision)
    cosine, sine = np.cos(angles), np.sin(angles)
    return np.concatenate([first * cosine - second * sine, second * cosine + first * sine], axis=-1).astype(np.float16)


def attention(q, k, v, precision):
    n, heads, dimension = q.shape
    group = heads // k.shape[1]
    scores = np.empty((heads, n, n), dtype=np.float32)
    probabilities = np.empty_like(scores)
    context = np.empty((n, heads, dimension), dtype=np.float16)
    causal = np.tri(n, dtype=bool)
    for head in range(heads):
        kv_head = head // group
        scores[head] = q[:, head].astype(precision) @ k[:, kv_head].astype(precision).T / np.sqrt(precision(dimension))
        valid = np.where(causal, scores[head].astype(precision), -np.inf)
        exp = np.exp(valid - np.max(valid, axis=-1, keepdims=True))
        probabilities[head] = exp / np.sum(exp, axis=-1, keepdims=True, dtype=precision)
        context[:, head] = probabilities[head].astype(precision) @ v[:, kv_head].astype(precision)
    return scores, probabilities, context.reshape(n, heads * dimension)


def forward(tokens, offset, weights, config, precision, layers):
    """Round every stored named activation; retain FP32 scores/probabilities/logits."""
    n, dimension = len(tokens), config["head_dim"]
    x = weights["embed_tokens"][tokens].copy()
    values = {"embedding": x}
    for index in range(config["num_layers"]):
        prefix = f"layers.{index}"

        def save(name, tensor):
            if not np.isfinite(tensor).all():
                raise ValueError(f"Nonfinite FP16 reconstruction: {prefix}.{name}")
            if index in layers:
                values[f"{prefix}.{name}"] = tensor
            return tensor

        normalized = save("input_norm", rmsnorm(x, weights[f"{prefix}.input_layernorm"], config["rms_norm_eps"], precision))
        projected = {}
        for name, heads in (("q", config["num_query_heads"]), ("k", config["num_kv_heads"]), ("v", config["num_kv_heads"])):
            projected[name] = save(f"{name}_projection", linear(normalized, weights[f"{prefix}.{name}_proj.weight"],
                                     weights[f"{prefix}.{name}_proj.bias"], precision)).reshape(n, heads, dimension)
        query = save("q_rotary", rotary(projected["q"], offset, config["rope_theta"], precision))
        key = save("k_rotary", rotary(projected["k"], offset, config["rope_theta"], precision))
        scores, probabilities, context = attention(query, key, projected["v"], precision)
        save("attention_scores", scores)
        save("attention_probabilities", probabilities)
        save("attention_context", context)
        projected_attention = save("attention_projection", linear(context, weights[f"{prefix}.o_proj.weight"], None, precision))
        x = save("attention_residual", (x.astype(precision) + projected_attention).astype(np.float16))
        normalized = save("post_attention_norm", rmsnorm(x, weights[f"{prefix}.post_attention_layernorm"], config["rms_norm_eps"], precision))
        gate = save("gate_projection", linear(normalized, weights[f"{prefix}.gate_proj.weight"], None, precision))
        up = save("up_projection", linear(normalized, weights[f"{prefix}.up_proj.weight"], None, precision))
        wide = gate.astype(precision)
        exp = np.exp(-np.abs(wide))
        sigmoid = np.where(wide >= 0, precision(1) / (precision(1) + exp), exp / (precision(1) + exp))
        activated = save("gate_silu", (wide * sigmoid).astype(np.float16))
        gated = save("gated", (activated.astype(precision) * up).astype(np.float16))
        down = save("down_projection", linear(gated, weights[f"{prefix}.down_proj.weight"], None, precision))
        x = save("output", (x.astype(precision) + down).astype(np.float16))
    values["final_norm"] = rmsnorm(x, weights["norm"], config["rms_norm_eps"], precision)
    values["logits"] = linear(values["final_norm"], weights["embed_tokens"], None, precision, np.float32)
    if not all(np.isfinite(array).all() for array in values.values()):
        raise ValueError("Nonfinite final FP16 reconstruction")
    return values


def gates(reconstructed, source, cases):
    """Derive limits only from independent CPU arithmetic and frozen source data."""
    pooled, equivalent, errors = {}, {}, {}
    for key, (main, wide) in reconstructed.items():
        expected = source[key].astype(np.float64)
        error = max(float(np.max(np.abs(value.astype(np.float64) - expected))) for value in (main, wide))
        name = key.split(".", 2)[2]
        pooled[name] = max(pooled.get(name, 0.0), error)
        errors[key] = {"independent_fp16_vs_source_max_abs_error": error,
                       "fp32_vs_fp64_arithmetic_max_abs_error": float(np.max(np.abs(main.astype(np.float64) - wide)))}
    for first_name, first in cases.items():
        for second_name, second in cases.items():
            if first_name >= second_name or first["offset"] != second["offset"]:
                continue
            common = 0
            for a, b in zip(first["tokens"], second["tokens"]):
                if a != b:
                    break
                common += 1
            if not common:
                continue
            for name in pooled:
                for precision in (0, 1):
                    a = reconstructed[f"cases.{first_name}.{name}"][precision]
                    b = reconstructed[f"cases.{second_name}.{name}"][precision]
                    if name.endswith((".attention_scores", ".attention_probabilities")):
                        a, b = a[:, :common, :common], b[:, :common, :common]
                    else:
                        a, b = a[:common], b[:common]
                    error = float(np.max(np.abs(a.astype(np.float64) - b)))
                    equivalent[name] = max(equivalent.get(name, 0.0), error)
    for key, error in errors.items():
        name = key.split(".", 2)[2]
        error.update(pooled_fp16_vs_source_max_abs_error=pooled[name], equivalent_prefix_max_abs_error=equivalent.get(name, 0.0),
                     atol=max(POLICY["atol_floor"], 2 * pooled[name], 2 * equivalent.get(name, 0.0)), rtol=POLICY["rtol"])
    return errors


def read(root, item):
    path = root / item["file"]
    data = path.read_bytes()
    dtype = {"f16le": "<f2", "f32le": "<f4"}[item.get("dtype", "f32le")]
    shape = [len(item["indices"])] if "indices" in item else item["shape"]
    if (len(data) != item["bytes"] or prepare.digest(path) != item["sha256"]
            or len(data) != int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize):
        raise ValueError(f"Reference payload differs: {path}")
    values = np.frombuffer(data, dtype=dtype).reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite reference payload: {path}")
    return values


def load_weights(sources):
    # The artifact is BF16. Copy each tensor directly to FP16 before closing the
    # source mapping; no full FP32/FP64 model is ever constructed or retained.
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    weights = {}
    with safe_open(str(ROOT / sources["model"]["weights"]["file"]), framework="pt", device="cpu") as archive:
        for name in archive.keys():
            key = name.removeprefix("model.").replace(".self_attn.", ".").replace(".mlp.", ".")
            key = key.removesuffix(".weight") if key.endswith(("embed_tokens.weight", "layernorm.weight", "norm.weight")) else key
            value = archive.get_tensor(name).to(torch.float16).numpy()
            if not np.isfinite(value).all():
                raise ValueError(f"Source FP16 range overflow: {name}")
            weights[key] = value
    return weights


def export(full, probes):
    if full.exists() or probes.exists() or full.resolve() == probes.resolve() or full.resolve() in probes.resolve().parents or probes.resolve() in full.resolve().parents:
        raise ValueError("Refusing existing or overlapping reference directories")
    sources = prepare.load_json(prepare.SOURCES)
    for package, expected in VERSIONS.items():
        if version(package) != expected:
            raise ValueError(f"Pinned reference package differs: {package}")
    if any(sources["baselines"][package] != VERSIONS[package] for package in ("numpy", "torch")):
        raise ValueError("Reference/source package versions differ")
    prepare.validate_source(sources["model"]["weights"])
    config_asset = next(item for item in sources["assets"] if item["id"] == "qwen.config.json")
    prepare.validate_source(config_asset)
    original, selected = prepare.load_json(SOURCE / "manifest.json"), prepare.load_json(SOURCE_PROBES / "manifest.json")
    for key in ("model", "cases", "layers", "policy", "argmax", "versions", "exporter_sha256"):
        if original[key] != selected[key]:
            raise ValueError(f"Full/source identity differs: {key}")
    if original["model"] != sources["model"] or original["cases"] != qwen.CASES or original["layers"] != qwen.LAYERS:
        raise ValueError("Frozen Qwen source identity differs")
    baseline = {}
    for name, item in original["arrays"].items():
        if (item["sha256"] != selected["arrays"][name]["full_array_sha256"]
                or item["bytes"] != selected["arrays"][name]["full_array_bytes"]):
            raise ValueError(f"Full/source array identity differs: {name}")
        baseline[name] = read(SOURCE, item)
    config = prepare.load_json(ROOT / config_asset["path"])
    model = {"num_layers": config["num_hidden_layers"], "num_query_heads": config["num_attention_heads"],
             "num_kv_heads": config["num_key_value_heads"], "head_dim": config["hidden_size"] // config["num_attention_heads"],
             "rms_norm_eps": config["rms_norm_eps"], "rope_theta": config["rope_theta"]}
    if not config["tie_word_embeddings"] or config["use_sliding_window"] or config["hidden_act"] != "silu":
        raise ValueError("This explicit reconstruction requires the pinned tied Qwen configuration")
    weights = load_weights(sources)
    reconstructed = {}
    for case, spec in qwen.CASES.items():
        main = forward(spec["tokens"], spec["offset"], weights, model, np.float32, qwen.LAYERS)
        wide = forward(spec["tokens"], spec["offset"], weights, model, np.float64, qwen.LAYERS)
        for name, value in main.items():
            key = f"cases.{case}.{name}"
            if value.shape != baseline[key].shape or wide[name].shape != value.shape:
                raise ValueError(f"Source reconstruction shape differs: {key}")
            reconstructed[key] = (value, wide[name])
        print(f"Reconstructed {case}: {len(main)} independent FP16 storage arrays", flush=True)
    if set(reconstructed) != set(baseline):
        raise ValueError("Source reconstruction inventory differs")
    limits = gates(reconstructed, baseline, qwen.CASES)
    identity = {"format": "toklet-qwen-fp16-v1", "generator_sha256": prepare.digest(Path(__file__)),
                "source_probes_manifest_sha256": prepare.digest(SOURCE_PROBES / "manifest.json"),
                "source_full_manifest_sha256": prepare.digest(SOURCE / "manifest.json"), "model": sources["model"],
                "config_sha256": config_asset["sha256"], "cases": qwen.CASES, "layers": qwen.LAYERS, "policy": POLICY,
                "versions": VERSIONS,
                "logical_fp16_weights_bytes": sum(value.nbytes for value in weights.values()),
                "maximum_weight_conversion_stripe_bytes": max(min(ROW_STRIPE, len(value)) * value.shape[1] * 8 for value in weights.values() if value.ndim == 2),
                "precision": {"weights_activations": "FP16", "scores_probabilities_logits": "FP32",
                              "arithmetic": "NumPy CPU FP32 BLAS/reductions; separate FP64 reconstruction with identical storage-rounding boundaries",
                              "projection_bias": "after dot, before output cast", "weight_conversion_rows": ROW_STRIPE},
                "scope": "Short independent numerical references, frozen before Metal evaluation. No cached/long-context execution, held-out quality or device qualification.",
                "argmax": {}, "arrays": {}}
    manifests = [json.loads(json.dumps(identity)), json.loads(json.dumps(identity))]
    for directory in (full, probes):
        directory.mkdir(parents=True)
    for name, (main, wide) in sorted(reconstructed.items()):
        dtype = "f16le" if main.dtype == np.float16 else "f32le"
        flattened = main.reshape(-1)
        indices = set(np.linspace(0, flattened.size - 1, min(64, flattened.size), dtype=np.int64).tolist())
        if name.endswith(".logits"):
            case = name.split(".")[1]
            argmax = {"source_fp32": np.argmax(baseline[name], axis=-1).tolist(),
                      "fp16_fp32_arithmetic": np.argmax(main, axis=-1).tolist(), "fp16_fp64_arithmetic": np.argmax(wide, axis=-1).tolist()}
            for manifest in manifests:
                manifest["argmax"][case] = argmax
            for position in range(len(main)):
                for token in (argmax[k][position] for k in argmax):
                    indices.add(position * main.shape[-1] + token)
        indices = sorted(indices)
        full_data = main.astype({"f16le": "<f2", "f32le": "<f4"}[dtype], copy=False).tobytes()
        for directory, manifest, value in ((full, manifests[0], main), (probes, manifests[1], flattened[indices])):
            filename = f"{name}.{dtype}"
            path = directory / filename
            path.write_bytes(value.astype({"f16le": "<f2", "f32le": "<f4"}[dtype], copy=False).tobytes())
            item = {"file": filename, "shape": list(main.shape), "dtype": dtype, "bytes": path.stat().st_size,
                    "sha256": prepare.digest(path), "gate_against_source_fp32": limits[name],
                    "source_fp32_full_sha256": original["arrays"][name]["sha256"]}
            if directory == probes:
                item.update(indices=indices, full_array_sha256=prepare.digest(full / filename), full_array_bytes=len(full_data))
            manifest["arrays"][name] = item
    for directory, manifest in zip((full, probes), manifests):
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        verify(directory)
    return manifests


def verify(root=FIXTURE):
    manifest = prepare.load_json(root / "manifest.json")
    original = prepare.load_json(SOURCE_PROBES / "manifest.json")
    if (manifest["format"] != "toklet-qwen-fp16-v1" or manifest["policy"] != POLICY
            or manifest["versions"] != VERSIONS
            or manifest["generator_sha256"] != prepare.digest(Path(__file__))
            or manifest["source_probes_manifest_sha256"] != prepare.digest(SOURCE_PROBES / "manifest.json")
            or manifest["model"] != original["model"] or manifest["cases"] != qwen.CASES
            or manifest["layers"] != qwen.LAYERS or set(manifest["arrays"]) != set(original["arrays"])):
        raise ValueError("Frozen Qwen FP16 identity/inventory differs")
    for name, item in manifest["arrays"].items():
        read(root, item)
        gate = item["gate_against_source_fp32"]
        dtype = "f32le" if name.endswith((".attention_scores", ".attention_probabilities", ".logits")) else "f16le"
        elements = int(np.prod(item["shape"], dtype=object))
        if item["dtype"] != dtype or item["file"] != f"{name}.{dtype}" or not all(np.isfinite(value) and value >= 0 for value in gate.values()):
            raise ValueError(f"Frozen Qwen FP16 dtype/gate differs: {name}")
        if "indices" in item:
            indices = item["indices"]
            expected = set(np.linspace(0, elements - 1, min(64, elements), dtype=np.int64).tolist())
            if name.endswith(".logits"):
                case = name.split(".")[1]
                argmax = manifest["argmax"][case]
                if argmax["source_fp32"] != original["argmax"][case]:
                    raise ValueError("Frozen Qwen FP16 source argmax differs")
                for tokens in argmax.values():
                    if len(tokens) != item["shape"][0] or any(not isinstance(token, int) or token < 0 or token >= item["shape"][-1] for token in tokens):
                        raise ValueError("Frozen Qwen FP16 argmax shape differs")
                    expected.update(position * item["shape"][-1] + token for position, token in enumerate(tokens))
            if indices != sorted(expected) or item["full_array_bytes"] != elements * (2 if dtype == "f16le" else 4):
                raise ValueError(f"Frozen Qwen FP16 selected coverage differs: {name}")
        if (item["source_fp32_full_sha256"] != original["arrays"][name]["full_array_sha256"]
                or item["shape"] != original["arrays"][name]["shape"]
                or gate["pooled_fp16_vs_source_max_abs_error"] < gate["independent_fp16_vs_source_max_abs_error"]
                or gate["atol"] != max(POLICY["atol_floor"], 2 * gate["pooled_fp16_vs_source_max_abs_error"], 2 * gate["equivalent_prefix_max_abs_error"])
                or gate["rtol"] != POLICY["rtol"]):
            raise ValueError(f"Frozen Qwen FP16 source/gate differs: {name}")
    return manifest


def regenerate_check(full, probes):
    """Explicit real-model check; exact regeneration also detects altered envelopes."""
    verify(full); verify(probes)
    with tempfile.TemporaryDirectory(prefix="toklet-qwen-fp16-") as directory:
        candidates = (Path(directory) / "full", Path(directory) / "probes")
        export(*candidates)
        for original, candidate in zip((full, probes), candidates):
            inventory = {path.name for path in original.iterdir()}
            if inventory != {path.name for path in candidate.iterdir()}:
                raise ValueError("Regenerated Qwen FP16 inventory differs")
            for name in inventory:
                if prepare.digest(original / name) != prepare.digest(candidate / name):
                    raise ValueError(f"Regenerated Qwen FP16 payload/gate differs: {name}")


def recorded_export(full, probes, path):
    if (path.exists() or full.exists() or probes.exists()
            or any(directory.resolve() == path.resolve() or directory.resolve() in path.resolve().parents for directory in (full, probes))):
        raise ValueError("Refusing existing or overlapping export evidence")
    import mlx_baseline as common

    def command(args):
        result = subprocess.run(args, text=True, capture_output=True, timeout=20)
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    def power():
        return {name: command(["/usr/bin/pmset", "-g", option]) for name, option in
                (("source", "batt"), ("configuration", "custom"), ("thermal", "therm"))}

    record = {"format": "toklet-qwen-fp16-export-v1", "completed": False,
              "generator_sha256": prepare.digest(Path(__file__)), "engine_revision": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
              "working_tree_status": command(["git", "-C", str(ROOT), "status", "--porcelain"]), "power_start": power(),
              "blas_thread_environment": {name: os.environ.get(name) for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
              "scope": "Independent functional reference export only; elapsed time is not an inference benchmark. Physical charge peak may exclude clean mapped source pages; total system memory/energy unavailable."}
    start = time.perf_counter()
    try:
        manifests = export(full, probes)
        record.update(completed=True, logical_fp16_weights_bytes=manifests[0]["logical_fp16_weights_bytes"],
                      maximum_weight_conversion_stripe_bytes=manifests[0]["maximum_weight_conversion_stripe_bytes"],
                      full_manifest_sha256=prepare.digest(full / "manifest.json"), probes_manifest_sha256=prepare.digest(probes / "manifest.json"),
                      argmax=manifests[0]["argmax"], versions=manifests[0]["versions"])
    except Exception as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["functional_export_wall_seconds"] = time.perf_counter() - start
        record["physical_process_charge_peak_bytes"] = common.footprint_peak()
        record["power_end"] = power()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as file:
            file.write(json.dumps(record, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probes-output", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--regenerate-check", action="store_true")
    parser.add_argument("--record", type=Path, help="optional macOS functional export/memory provenance")
    args = parser.parse_args()
    if args.check:
        verify(args.check)
    elif args.output and args.probes_output:
        watchdog = threading.Timer(600, lambda: os._exit(124))  # Only this explicit export process.
        watchdog.start()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if args.regenerate_check:
                    if args.record:
                        parser.error("--record belongs to a fresh export")
                    regenerate_check(args.output, args.probes_output)
                elif args.record:
                    recorded_export(args.output, args.probes_output, args.record)
                else:
                    export(args.output, args.probes_output)
        finally:
            watchdog.cancel()
    else:
        parser.error("provide --check or both --output and --probes-output")
    print("Verified independent Qwen FP16 reference identities; no target execution", flush=True)


if __name__ == "__main__":
    main()
