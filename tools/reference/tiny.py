"""Independent NumPy oracle and immutable tiny-model fixture exporter."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "tests/fixtures/tiny-v1"
MODEL = {
    "vocab_size": 64,
    "num_layers": 2,
    "hidden_size": 32,
    "num_query_heads": 4,
    "num_kv_heads": 2,
    "head_dim": 8,
    "intermediate_size": 64,
    "capacity": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "rope_pairing": "split_half",
    "tie_word_embeddings": True,
    "projection_biases": ["q", "k", "v"],
}
CASES = {
    "prompt": {"tokens": [1, 7, 13, 3], "position_offset": 0},
    "future_changed": {"tokens": [1, 7, 13, 17], "position_offset": 0},
    "offset": {"tokens": [7, 13, 3], "position_offset": 5},
    "single": {"tokens": [11], "position_offset": 0},
    "capacity": {
        "tokens": [(7 * i + 1) % 64 for i in range(32)],
        "position_offset": 0,
    },
}
OPERATIONS = (
    "linear", "rmsnorm", "rotary", "attention_scores", "softmax",
    "attention_output", "silu", "gated_mlp", "residual", "logits",
)
TOLERANCES = {op: {"atol": 1e-5, "rtol": 1e-4} for op in OPERATIONS}


def weights():
    # The RNG is only used here. Rust consumes the exported bytes, never this seed.
    rng = np.random.Generator(np.random.PCG64(20261009))
    result = {}

    def normal(name, shape, scale):
        result[name] = rng.standard_normal(shape).astype(np.float32) * np.float32(scale)

    def norm(name):
        result[name] = rng.uniform(0.8, 1.2, 32).astype(np.float32)

    normal("embed_tokens", (64, 32), 0.2)
    for layer in range(2):
        prefix = f"layers.{layer}"
        norm(f"{prefix}.input_layernorm")
        norm(f"{prefix}.post_attention_layernorm")
        for name, out_features in (("q", 32), ("k", 16), ("v", 16)):
            normal(f"{prefix}.{name}_proj.weight", (out_features, 32), 0.12)
            normal(f"{prefix}.{name}_proj.bias", (out_features,), 0.03)
        normal(f"{prefix}.o_proj.weight", (32, 32), 0.12)
        normal(f"{prefix}.gate_proj.weight", (64, 32), 0.1)
        normal(f"{prefix}.up_proj.weight", (64, 32), 0.1)
        normal(f"{prefix}.down_proj.weight", (32, 64), 0.1)
    norm("norm")
    return result


def linear(x, weight, bias=None):
    result = x @ weight.T
    return result if bias is None else result + bias


def rmsnorm(x, weight, eps):
    # Scaling before squaring also defines behavior for extreme finite inputs.
    scale = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(scale == 0, x.dtype.type(1), scale)
    scaled = x / scale
    rms = scale * np.sqrt(np.mean(scaled * scaled, axis=-1, keepdims=True))
    denominator = np.hypot(rms, np.sqrt(x.dtype.type(eps)))
    return x / denominator * weight


def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def silu(x):
    exp = np.exp(-np.abs(x))
    sigmoid = np.where(x >= 0, 1 / (1 + exp), exp / (1 + exp))
    return x * sigmoid


def rotary(x, offset, theta):
    dimension = x.shape[-1]
    half = dimension // 2
    dtype = x.dtype
    powers = np.arange(0, dimension, 2, dtype=dtype) / dtype.type(dimension)
    inv_frequency = dtype.type(theta) ** (-powers)
    positions = np.arange(offset, offset + x.shape[0], dtype=dtype)
    angles = positions[:, None, None] * inv_frequency[None, None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    first, second = x[..., :half], x[..., half:]
    return np.concatenate((first * cos - second * sin, second * cos + first * sin), axis=-1)


def forward(tokens, offset, model_weights, dtype):
    w = {name: array.astype(dtype, copy=False) for name, array in model_weights.items()}
    x = w["embed_tokens"][tokens]
    values = {"embedding": (x, "linear")}
    for layer in range(MODEL["num_layers"]):
        prefix = f"layers.{layer}"

        def save(name, value, operation):
            values[f"{prefix}.{name}"] = (value, operation)
            return value

        normalized = save("input_norm", rmsnorm(x, w[f"{prefix}.input_layernorm"], MODEL["rms_norm_eps"]), "rmsnorm")
        projections = {}
        for name, heads in (("q", 4), ("k", 2), ("v", 2)):
            projected = linear(normalized, w[f"{prefix}.{name}_proj.weight"], w[f"{prefix}.{name}_proj.bias"])
            projections[name] = save(f"{name}_projection", projected.reshape(-1, heads, 8), "linear")
        q = save("q_rotary", rotary(projections["q"], offset, MODEL["rope_theta"]), "rotary")
        k = save("k_rotary", rotary(projections["k"], offset, MODEL["rope_theta"]), "rotary")
        # Four query heads share two KV heads: [0, 0, 1, 1].
        mapping = np.arange(4) // 2
        mapped_k, mapped_v = k[:, mapping, :], projections["v"][:, mapping, :]
        scores = save("attention_scores", np.einsum("thd,shd->hts", q, mapped_k) / np.sqrt(dtype(8)), "attention_scores")
        causal = np.tril(np.ones((len(tokens), len(tokens)), dtype=bool))
        probabilities = save("attention_probabilities", softmax(np.where(causal[None, :, :], scores, -np.inf)), "softmax")
        context = save("attention_context", np.einsum("hts,shd->thd", probabilities, mapped_v).reshape(-1, 32), "attention_output")
        attention = save("attention_projection", linear(context, w[f"{prefix}.o_proj.weight"]), "linear")
        x = save("attention_residual", x + attention, "residual")
        normalized = save("post_attention_norm", rmsnorm(x, w[f"{prefix}.post_attention_layernorm"], MODEL["rms_norm_eps"]), "rmsnorm")
        gate = save("gate_projection", linear(normalized, w[f"{prefix}.gate_proj.weight"]), "linear")
        up = save("up_projection", linear(normalized, w[f"{prefix}.up_proj.weight"]), "linear")
        activation = save("gate_silu", silu(gate), "silu")
        gated = save("gated", activation * up, "gated_mlp")
        down = save("down_projection", linear(gated, w[f"{prefix}.down_proj.weight"]), "linear")
        x = save("output", x + down, "residual")
    final = rmsnorm(x, w["norm"], MODEL["rms_norm_eps"])
    values["final_norm"] = (final, "rmsnorm")
    # Tied output projection uses the embedding matrix; no second weight copy.
    values["logits"] = (linear(final, w["embed_tokens"]), "logits")
    return values


def probes(dtype):
    def array(values):
        # FP64 envelopes start from exactly the same stored FP32 input values.
        return np.asarray(values, dtype=np.float32).astype(dtype)

    matrix = array([[1, 2, -1], [0.5, -2, 3]])
    inputs = array([[2, -1, 4], [-1, 0, 2]])
    bias = array([0.25, -0.5])
    extreme = array([-1000, -2, 0, 2, 1000])
    large = array([[1e20, -1e20, 0, 1e20]])
    return {
        "manual.weight": (matrix, "linear"),
        "manual.inputs": (inputs, "linear"),
        "manual.bias": (bias, "linear"),
        "manual.output": (linear(inputs, matrix, bias), "linear"),
        "edge.silu.input": (extreme, "silu"),
        "edge.silu.output": (silu(extreme), "silu"),
        "edge.softmax.input": (array([[1000, 1001, 999]]), "softmax"),
        "edge.softmax.output": (softmax(array([[1000, 1001, 999]])), "softmax"),
        "edge.rmsnorm.large_input": (large, "rmsnorm"),
        "edge.rmsnorm.large_output": (rmsnorm(large, np.ones(4, dtype=dtype), 1e-6), "rmsnorm"),
        "edge.rmsnorm.zero_output": (rmsnorm(np.zeros((1, 32), dtype=dtype), np.ones(32, dtype=dtype), 1e-6), "rmsnorm"),
    }


def expected_arrays(model_weights, dtype, cases, greedy):
    result = probes(dtype)
    for name, case in cases.items():
        for intermediate, value in forward(case["tokens"], case["position_offset"], model_weights, dtype).items():
            result[f"cases.{name}.{intermediate}"] = value
    history = list(greedy["prompt_tokens"])
    for step, token in enumerate(greedy["generated_tokens"]):
        logits = forward(history, 0, model_weights, dtype)["logits"][0][-1]
        result[f"greedy.step_{step}.logits"] = (logits, "logits")
        # FP64 comparisons use the frozen history, including any near-tie choice.
        history.append(token)
    return result


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def compare(actual, expected, operation):
    tolerance = TOLERANCES[operation]
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    allowed = tolerance["atol"] + tolerance["rtol"] * np.abs(expected)
    fraction = float(np.max(difference / allowed))
    if fraction > 1:
        raise ValueError(f"{operation}: frozen tolerance exceeded ({fraction:.6g})")
    return {
        "max_absolute_error": float(np.max(difference)),
        "max_relative_error_floor_1e-6": float(np.max(difference / np.maximum(np.abs(expected), 1e-6))),
        "max_fraction_of_tolerance": fraction,
    }


def generate(destination):
    if np.__version__ != "2.3.2":
        raise ValueError("Generate with the pinned NumPy 2.3.2 environment")
    if destination.exists():
        raise ValueError(f"Refusing to overwrite frozen fixture directory: {destination}")
    model_weights = weights()
    history = list(CASES["prompt"]["tokens"])
    generated = []
    for _ in range(4):
        logits = forward(history, 0, model_weights, np.float32)["logits"][0][-1]
        token = int(np.argmax(logits))  # NumPy selects the lowest index on a tie.
        generated.append(token)
        history.append(token)
    greedy = {"prompt_tokens": CASES["prompt"]["tokens"], "generated_tokens": generated, "tie_breaking": "lowest_token_id", "stop_policy": "fixed_4_steps_no_eos"}
    fp32 = expected_arrays(model_weights, np.float32, CASES, greedy)
    fp64 = expected_arrays(model_weights, np.float64, CASES, greedy)
    envelope = {}
    for name, (array, operation) in fp32.items():
        # Compare calculations using the same frozen FP32 inputs and weights.
        envelope[name] = compare(array, fp64[name][0], operation)
    arrays = {f"weights.{name}": (array, "weight", None) for name, array in model_weights.items()}
    arrays.update({name: (array, "reference", operation) for name, (array, operation) in fp32.items()})
    for name, case in CASES.items():
        arrays[f"cases.{name}.tokens"] = (np.asarray(case["tokens"], dtype=np.uint32), "tokens", None)
    arrays["greedy.tokens"] = (np.asarray(generated, dtype=np.uint32), "tokens", None)
    destination.mkdir(parents=True)
    metadata = {}
    for name, (array, kind, operation) in sorted(arrays.items()):
        dtype = "u32le" if array.dtype == np.uint32 else "f32le"
        data = np.ascontiguousarray(array, dtype="<u4" if dtype == "u32le" else "<f4").tobytes()
        filename = f"{name}.{dtype}"
        (destination / filename).write_bytes(data)
        metadata[name] = {"file": filename, "shape": list(array.shape), "dtype": dtype, "bytes": len(data), "sha256": sha256(data), "kind": kind, "operation": operation}
    manifest = {
        "schema_version": 1,
        "fixture_id": "toklet-tiny-fp32-v1",
        "generator": "tools/reference/tiny.py",
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "tool_versions": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.system(), "machine": platform.machine()},
        "weight_generation": {"rng": "NumPy PCG64", "seed": 20261009, "rust_uses_exported_bytes": True},
        "model": MODEL,
        "layouts": {"linear_weight": "[out_features, in_features], C row-major", "activation": "[tokens, hidden], C row-major", "qkv": "[tokens, heads, head_dim], C row-major", "attention_scores_and_probabilities": "[query_heads, query_tokens, key_tokens], C row-major", "array_encoding": "headerless little-endian, contiguous, no padding; alignment is not guaranteed"},
        "cases": CASES,
        "greedy": greedy,
        "numerical_policy": {"storage": "FP32", "oracle_computation": "NumPy FP32", "envelope_reference": "FP64 computations on the same frozen FP32 weights and inputs", "tolerances": TOLERANCES, "non_finite_inputs": "reject", "empty_or_entirely_masked_softmax": "reject", "measured_fp32_vs_fp64": envelope},
        "arrays": metadata,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify(destination)


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["model"] != MODEL:
        raise ValueError("Unsupported fixture schema or topology")
    arrays = {}
    for name, metadata in manifest["arrays"].items():
        data = (directory / metadata["file"]).read_bytes()
        if len(data) != metadata["bytes"] or sha256(data) != metadata["sha256"]:
            raise ValueError(f"Corrupt fixture array: {name}")
        dtype = "<u4" if metadata["dtype"] == "u32le" else "<f4"
        arrays[name] = np.frombuffer(data, dtype=dtype).reshape(metadata["shape"])
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"Non-finite fixture array: {name}")
    model_weights = {name.removeprefix("weights."): array for name, array in arrays.items() if name.startswith("weights.")}
    expected = expected_arrays(model_weights, np.float32, manifest["cases"], manifest["greedy"])
    for name, (array, operation) in expected.items():
        if manifest["numerical_policy"]["tolerances"][operation] != TOLERANCES[operation]:
            raise ValueError(f"Changed frozen tolerance: {operation}")
        compare(array, arrays[name], operation)
    for name, case in manifest["cases"].items():
        np.testing.assert_array_equal(arrays[f"cases.{name}.tokens"], case["tokens"])
    np.testing.assert_array_equal(arrays["greedy.tokens"], manifest["greedy"]["generated_tokens"])
    for step, token in enumerate(manifest["greedy"]["generated_tokens"]):
        if int(np.argmax(arrays[f"greedy.step_{step}.logits"])) != token:
            raise ValueError(f"Greedy trace mismatch at step {step}")
    np.testing.assert_array_equal(arrays["manual.output"], [[-3.75, 14.5], [-2.75, 5]])
    np.testing.assert_array_equal(arrays["cases.prompt.logits"][:3], arrays["cases.future_changed.logits"][:3])
    largest = max(item["max_fraction_of_tolerance"] for item in manifest["numerical_policy"]["measured_fp32_vs_fp64"].values())
    print(f"Verified {len(arrays)} frozen arrays; FP32/FP64 envelope uses at most {largest:.4%} of the frozen tolerance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", type=Path, metavar="NEW_DIRECTORY", help="export a new fixture; refuses an existing directory")
    mode.add_argument("--check", type=Path, metavar="DIRECTORY", help="verify hashes and recompute the frozen reference")
    args = parser.parse_args()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if args.generate is not None:
            generate(args.generate)
        else:
            verify(args.check or DEFAULT_FIXTURE)


if __name__ == "__main__":
    main()
