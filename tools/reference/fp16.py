"""Freeze full tiny FP16-storage/FP32-accumulation gates before Metal execution."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/fixtures/tiny-v1"
FIXTURE = ROOT / "tests/fixtures/fp16-v1"
DTYPES = {"f16le": "<f2", "f32le": "<f4", "u32le": "<u4"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root, manifest, name):
    item = manifest["arrays"][name]
    path = root / item["file"]
    data = path.read_bytes()
    dtype = DTYPES[item["dtype"]]
    if (len(data) != item["bytes"] or digest(path) != item["sha256"]
            or len(data) != int(np.prod(item["shape"], dtype=object)) * np.dtype(dtype).itemsize):
        raise ValueError(f"Fixture payload differs: {name}")
    values = np.frombuffer(data, dtype=dtype).reshape(item["shape"])
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite fixture: {name}")
    return values


def dot(a, b, precision):
    total = precision(0)
    for index in range(len(a)):
        total = precision(total + precision(precision(a[index]) * precision(b[index])))
    return total


def linear(x, weight, bias, precision, output_dtype=np.float16):
    output = np.empty((len(x), len(weight)), dtype=output_dtype)
    for row in range(len(x)):
        for column in range(len(weight)):
            value = dot(x[row], weight[column], precision)
            if bias is not None:
                value = precision(value + precision(bias[column]))
            output[row, column] = value
    return output


def rmsnorm(x, weight, eps, precision):
    result = np.empty_like(x)
    for row in range(len(x)):
        mean = precision(dot(x[row], x[row], precision) / precision(x.shape[1]))
        inverse = precision(1) / np.sqrt(precision(mean + precision(eps)))
        for feature in range(x.shape[1]):
            result[row, feature] = precision(precision(x[row, feature]) * inverse) * precision(weight[feature])
    return result


def rotary(x, offset, theta, precision):
    output = np.empty_like(x)
    half = x.shape[-1] // 2
    for token in range(len(x)):
        for feature in range(half):
            inverse = precision(theta) ** precision(-2 * feature / x.shape[-1])
            angle = precision(precision(offset + token) * inverse)
            cosine, sine = np.cos(angle), np.sin(angle)
            for head in range(x.shape[1]):
                a, b = precision(x[token, head, feature]), precision(x[token, head, feature + half])
                output[token, head, feature] = precision(a * cosine) - precision(b * sine)
                output[token, head, feature + half] = precision(b * cosine) + precision(a * sine)
    return output


def attention(q, k, v, precision):
    n, heads, dimension = q.shape
    group = heads // k.shape[1]
    scores = np.empty((heads, n, n), dtype=np.float32)
    probabilities = np.zeros_like(scores)
    context = np.empty((n, heads * dimension), dtype=np.float16)
    for head in range(heads):
        for token in range(n):
            for position in range(n):
                scores[head, token, position] = precision(dot(q[token, head], k[position, head // group], precision)
                                                         / np.sqrt(precision(dimension)))
            valid = scores[head, token, :token + 1].astype(precision)
            maximum = max(valid)
            exp = np.exp(valid - maximum)
            denominator = precision(0)
            for value in exp:
                denominator = precision(denominator + value)
            probabilities[head, token, :token + 1] = exp / denominator
            for feature in range(dimension):
                context[token, head * dimension + feature] = dot(
                    probabilities[head, token, :token + 1], v[:token + 1, head // group, feature], precision)
    return scores, probabilities, context


def forward(tokens, offset, weights, model, precision):
    """Scalar independent math; FP16 stored intermediates, FP32 scores/logits."""
    x = weights["embed_tokens"][tokens].copy()
    values = {"embedding": x}
    for layer in range(model["num_layers"]):
        prefix = f"layers.{layer}"

        def save(name, value):
            if not np.isfinite(value).all():
                raise ValueError(f"FP16 reconstruction overflow: {prefix}.{name}")
            values[f"{prefix}.{name}"] = value
            return value

        normalized = save("input_norm", rmsnorm(x, weights[f"{prefix}.input_layernorm"], model["rms_norm_eps"], precision))
        projected = {}
        for name, heads in (("q", model["num_query_heads"]), ("k", model["num_kv_heads"]), ("v", model["num_kv_heads"])):
            projected[name] = save(f"{name}_projection", linear(normalized, weights[f"{prefix}.{name}_proj.weight"],
                                   weights[f"{prefix}.{name}_proj.bias"], precision).reshape(-1, heads, model["head_dim"]))
        q = save("q_rotary", rotary(projected["q"], offset, model["rope_theta"], precision))
        k = save("k_rotary", rotary(projected["k"], offset, model["rope_theta"], precision))
        scores, probabilities, context = attention(q, k, projected["v"], precision)
        save("attention_scores", scores)
        save("attention_probabilities", probabilities)
        save("attention_context", context)
        projected_attention = save("attention_projection", linear(context, weights[f"{prefix}.o_proj.weight"], None, precision))
        x = save("attention_residual", (x.astype(precision) + projected_attention).astype(np.float16))
        normalized = save("post_attention_norm", rmsnorm(x, weights[f"{prefix}.post_attention_layernorm"], model["rms_norm_eps"], precision))
        gate = save("gate_projection", linear(normalized, weights[f"{prefix}.gate_proj.weight"], None, precision))
        up = save("up_projection", linear(normalized, weights[f"{prefix}.up_proj.weight"], None, precision))
        wide = gate.astype(precision)
        exp = np.exp(-np.abs(wide))
        sigmoid = np.where(wide >= 0, precision(1) / (precision(1) + exp), exp / (precision(1) + exp))
        activated = save("gate_silu", (wide * sigmoid).astype(np.float16))
        gated = save("gated", (activated.astype(precision) * up).astype(np.float16))
        down = save("down_projection", linear(gated, weights[f"{prefix}.down_proj.weight"], None, precision))
        x = save("output", (x.astype(precision) + down).astype(np.float16))
    normalized = rmsnorm(x, weights["norm"], model["rms_norm_eps"], precision)
    values["final_norm"] = normalized
    values["logits"] = linear(normalized, weights["embed_tokens"], None, precision, np.float32)
    if not all(np.isfinite(array).all() for array in values.values()):
        raise ValueError("Nonfinite FP16 decoder reconstruction")
    return values


def export(output):
    if output.exists():
        raise ValueError("Refusing to overwrite frozen FP16 fixture")
    if np.__version__ != "2.3.2":
        raise ValueError("Use frozen NumPy 2.3.2")
    source = json.loads((SOURCE / "manifest.json").read_text())
    weights = {name.removeprefix("weights."): read(SOURCE, source, name).astype(np.float16)
               for name in source["arrays"] if name.startswith("weights.")}
    output.mkdir(parents=True)
    arrays = {}

    def save(name, value, gate=None):
        dtype = "u32le" if value.dtype == np.uint32 else "f16le" if value.dtype == np.float16 else "f32le"
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite export: {name}")
        path = output / f"{name}.{dtype}"
        path.write_bytes(np.asarray(value, dtype=DTYPES[dtype]).tobytes(order="C"))
        arrays[name] = {"file": path.name, "shape": list(value.shape), "dtype": dtype,
                        "bytes": path.stat().st_size, "sha256": digest(path), "gate_against_source_fp32": gate}

    for name, value in weights.items():
        save(f"weights.{name}", value)

    def reconstruct(prefix, tokens, offset, source_prefix):
        computed = forward(tokens, offset, weights, source["model"], np.float32)
        wide = forward(tokens, offset, weights, source["model"], np.float64)
        for name, value in computed.items():
            expected = read(SOURCE, source, f"{source_prefix}.{name}")
            envelope = max(float(np.max(np.abs(candidate.astype(np.float64) - expected)))
                           for candidate in (value, wide[name]))
            gate = {"atol": 2 * envelope + 1e-5, "rtol": 1e-4,
                    "independent_fp16_vs_source_max_abs_error": envelope,
                    "fp32_vs_fp64_accumulation_max_abs_error": float(np.max(np.abs(value.astype(np.float64) - wide[name])))}
            save(f"{prefix}.{name}", value, gate)
        return computed

    for case, item in source["cases"].items():
        prefix = f"cases.{case}"
        computed = reconstruct(prefix, item["tokens"], item["position_offset"], prefix)
        save(f"{prefix}.tokens", np.asarray(item["tokens"], dtype=np.uint32))
        for layer in range(source["model"]["num_layers"]):
            for name, value in (("k", computed[f"layers.{layer}.k_rotary"]),
                                ("v", computed[f"layers.{layer}.v_projection"])):
                cache = np.zeros((source["model"]["num_kv_heads"], source["model"]["capacity"], source["model"]["head_dim"]), dtype=np.float16)
                cache[:, :len(value), :] = value.transpose(1, 0, 2)
                save(f"{prefix}.layers.{layer}.cache_{name}", cache)
    history = list(source["greedy"]["prompt_tokens"])
    greedy = []
    for step, token in enumerate(source["greedy"]["generated_tokens"]):
        computed = forward(history, 0, weights, source["model"], np.float32)["logits"][-1]
        wide = forward(history, 0, weights, source["model"], np.float64)["logits"][-1]
        expected = read(SOURCE, source, f"greedy.step_{step}.logits")
        envelope = max(float(np.max(np.abs(value.astype(np.float64) - expected))) for value in (computed, wide))
        save(f"greedy.step_{step}.logits", computed, {"atol": 2 * envelope + 1e-5, "rtol": 1e-4,
             "independent_fp16_vs_source_max_abs_error": envelope,
             "fp32_vs_fp64_accumulation_max_abs_error": float(np.max(np.abs(computed.astype(np.float64) - wide)))})
        greedy.append(int(np.argmax(computed)))
        history.append(token)  # Compare the immutable source history, never target-generated tokens.
    save("greedy.tokens", np.asarray(greedy, dtype=np.uint32))
    manifest = {
        "format": "toklet-tiny-fp16-v1", "generator_sha256": digest(Path(__file__)),
        "source_manifest_sha256": digest(SOURCE / "manifest.json"), "numpy_version": np.__version__,
        "model": source["model"], "cases": source["cases"], "arrays": arrays,
        "greedy": {"history": "source FP32 frozen teacher-forced history", "source_generated_tokens": source["greedy"]["generated_tokens"],
                   "reconstructed_generated_tokens": greedy, "tie_breaking": "lowest_token_id", "stop_policy": "fixed_4_steps_no_eos"},
        "precision": {"weights_and_named_activations": "FP16", "scores_probabilities_logits": "FP32",
                      "accumulation": "ordered scalar FP32; separate FP64 reconstruction with the same storage-rounding boundaries",
                      "rotary_coefficients": "FP32 in main reconstruction; FP64 in independent envelope",
                      "projection_bias": "added after dot, before output storage rounding"},
        "gate_policy": "Per-array 2 * max(abs(independent FP16-storage FP32/FP64 reconstruction - frozen source FP32)) + 1e-5; rtol=1e-4. Frozen before target evaluation.",
        "layout": "Row-major linear weights [out,in]; activations [tokens,hidden]; Q/K/V [tokens,heads,dimension]; separate K/V [kv_heads,32,dimension] with zero invalid suffix. Headerless little-endian, no alignment guarantee.",
        "scope": "Independent tiny numerical fixtures only, before Metal execution. Cache values are reconstruction references, not a requirement for bit-exact cross-device arithmetic; invalid suffix must remain zero. No real-model/quality/GPU qualification.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(root=FIXTURE):
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest["source_manifest_sha256"] != digest(SOURCE / "manifest.json")
            or manifest["generator_sha256"] != digest(Path(__file__))
            or manifest["numpy_version"] != np.__version__):
        raise ValueError("Frozen FP16 oracle identity differs")
    for name in manifest["arrays"]:
        read(root, manifest, name)
    with tempfile.TemporaryDirectory(prefix="toklet-fp16-oracle-") as directory:
        candidate = Path(directory) / "candidate"
        export(candidate)
        names = {path.name for path in candidate.iterdir()}
        if names != {path.name for path in root.iterdir()}:
            raise ValueError("Frozen FP16 inventory differs")
        for name in names:
            if (candidate / name).read_bytes() != (root / name).read_bytes():
                raise ValueError(f"Frozen FP16 oracle/gate differs: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--build", type=Path)
    group.add_argument("--check", type=Path, default=FIXTURE)
    args = parser.parse_args()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        manifest = export(args.build) if args.build else verify(args.check)
    print(f"Verified {manifest['format']}: {len(manifest['arrays'])} arrays with independent pre-target gates")


if __name__ == "__main__":
    main()
