"""Freeze independent tiny projection/stateful-attention slice gates before Core ML."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "tests/fixtures/tiny-v1"
FIXTURE = ROOT / "tests/fixtures/coreml-slice-v1"
CASES = ("single", "prompt", "future_changed", "offset", "capacity")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root, manifest, name):
    item = manifest["arrays"][name]
    path = root / item["file"]
    payload = path.read_bytes()
    dtype = {"f32le": "<f4", "f16le": "<f2"}[item["dtype"]]
    if (len(payload) != item["bytes"] or digest(path) != item["sha256"]
            or len(payload) != int(np.prod(item["shape"], dtype=object)) * np.dtype(dtype).itemsize):
        raise ValueError(f"Frozen slice array identity differs: {name}")
    array = np.frombuffer(payload, dtype=dtype).reshape(item["shape"])
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite slice fixture: {name}")
    return array


def projection(input, weight, bias, half):
    """Independent row-wise ordered FP64 dots, with optional FP16 storage."""
    dtype = np.float16 if half else np.float32
    x, w, b = (array.astype(dtype).astype(np.float64) for array in (input, weight, bias))
    output = np.empty((len(x), len(w)), dtype=dtype)
    for row in range(len(x)):
        for column in range(len(w)):
            total = 0.0
            for feature in range(x.shape[1]):
                total += float(x[row, feature]) * float(w[column, feature])
            output[row, column] = total + float(b[column])
    return output.astype(np.float32)


def attention(query, key, value, half, rounded_scores):
    """Causal GQA scalar oracle; no Core ML, Torch, or shared runtime kernels."""
    dtype = np.float16 if half else np.float32
    q, k, v = (array.astype(dtype).astype(np.float64) for array in (query, key, value))
    output = np.empty((len(q), 32), dtype=dtype)
    for token in range(len(q)):
        for head in range(4):
            scores = []
            for position in range(token + 1):
                total = 0.0
                for feature in range(8):
                    total += float(q[token, head, feature]) * float(k[position, head // 2, feature])
                if rounded_scores:
                    total = float(dtype(dtype(total) * dtype(1.0 / np.sqrt(8.0))))
                else:
                    total /= np.sqrt(8.0)
                scores.append(total)
            scores = np.asarray(scores, dtype=np.float64)
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            probabilities = probabilities.astype(dtype).astype(np.float64)
            for feature in range(8):
                total = 0.0
                for position in range(token + 1):
                    total += float(probabilities[position]) * float(v[position, head // 2, feature])
                output[token, head * 8 + feature] = total
    return output.astype(np.float32)


def gate(expected, reconstructed, fp16):
    error = max(float(np.max(np.abs(array.astype(np.float64) - expected)))
                for array in reconstructed)
    # Frozen before target evaluation. FP32 retains the tiny-v1 policy; FP16
    # uses twice the independent storage/operation-rounding envelope plus floor.
    atol = 2.0 * error + 1e-5 if fp16 else 1e-5
    return {"atol": atol, "rtol": 1e-4, "independent_max_abs_error": error,
            "policy": "2 * independent_max_abs_error + 1e-5" if fp16 else "tiny-v1 unchanged"}


def export(output):
    if output.exists():
        raise ValueError("Refusing to overwrite frozen slice fixtures")
    output.mkdir(parents=True)
    tiny = json.loads((TINY / "manifest.json").read_text())
    arrays = {}

    def save(name, values, dtype="f32le"):
        array = np.asarray(values, dtype="<f2" if dtype == "f16le" else "<f4")
        path = output / f"{name}.{dtype}"
        path.write_bytes(array.tobytes(order="C"))
        arrays[name] = {"file": path.name, "shape": list(array.shape), "dtype": dtype,
                        "bytes": path.stat().st_size, "sha256": digest(path)}

    weight = read(TINY, tiny, "weights.layers.0.q_proj.weight")
    bias = read(TINY, tiny, "weights.layers.0.q_proj.bias")
    save("projection.weight", weight)
    save("projection.bias", bias)
    cases = {}
    for case in CASES:
        prefix = f"cases.{case}.layers.0"
        input = read(TINY, tiny, f"{prefix}.input_norm")
        expected_projection = read(TINY, tiny, f"{prefix}.q_projection").reshape(len(input), 32)
        q = read(TINY, tiny, f"{prefix}.q_rotary")
        k = read(TINY, tiny, f"{prefix}.k_rotary")
        v = read(TINY, tiny, f"{prefix}.v_projection")
        context = read(TINY, tiny, f"{prefix}.attention_context")
        projection_f32 = projection(input, weight, bias, False)
        if not np.all(np.abs(projection_f32 - expected_projection) <= 1e-5 + 1e-4 * np.abs(expected_projection)):
            raise ValueError("Independent projection exceeds original tiny policy")
        attention_f32 = attention(q, k, v, False, False)
        if not np.all(np.abs(attention_f32 - context) <= 1e-5 + 1e-4 * np.abs(context)):
            raise ValueError("Independent attention exceeds original tiny policy")
        save(f"{case}.projection_input", input)
        save(f"{case}.projection_expected", expected_projection)
        save(f"{case}.query", q, "f16le")
        save(f"{case}.key", k, "f16le")
        save(f"{case}.value", v, "f16le")
        save(f"{case}.context_expected", context)
        cache_k = np.zeros((2, 32, 8), dtype=np.float16)
        cache_v = cache_k.copy()
        cache_k[:, :len(k), :] = k.astype(np.float16).transpose(1, 0, 2)
        cache_v[:, :len(v), :] = v.astype(np.float16).transpose(1, 0, 2)
        save(f"{case}.cache_k_expected", cache_k, "f16le")
        save(f"{case}.cache_v_expected", cache_v, "f16le")
        cases[case] = {
            "tokens": len(input), "absolute_rotary_offset": tiny["cases"][case]["position_offset"],
            "projection_fp32_gate": gate(expected_projection, [projection_f32], False),
            "projection_fp16_gate": gate(expected_projection, [projection(input, weight, bias, True)], True),
            "attention_fp16_gate": gate(context, [attention(q, k, v, True, mode) for mode in (False, True)], True),
        }
    manifest = {
        "format": "toklet-coreml-slice-v1", "generator_sha256": digest(Path(__file__)),
        "source_fixture_id": tiny["fixture_id"], "source_manifest_sha256": digest(TINY / "manifest.json"),
        "numpy_version": np.__version__, "arrays": arrays, "cases": cases,
        "layout": "Projection [tokens,32] and [out=32,in=32]; query [tokens,4,8]; key/value [tokens,2,8]; persistent K/V [2,32,8]",
        "precision": "Projection FP32 or FP16 storage; attention FP16 inputs/state/outputs. FP64 independent dots, stable softmax; envelope also rounds dot/scale to FP16.",
        "scope": "First-layer projection and pre-rotated causal GQA attention only. No normalization, rotary implementation, complete decoder, logits, generation, quality or engine qualification.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(root=FIXTURE):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source_manifest_sha256"] != digest(TINY / "manifest.json"):
        raise ValueError("Independent source fixture changed")
    if manifest["generator_sha256"] != digest(Path(__file__)) or manifest["numpy_version"] != np.__version__:
        raise ValueError("Frozen oracle identity/version differs")
    for name in manifest["arrays"]:
        read(root, manifest, name)
    # The gate cannot be enlarged through a modified manifest. Reconstruct it
    # independently from the immutable source, then compare every exact byte.
    with tempfile.TemporaryDirectory(prefix="toklet-coreml-oracle-") as directory:
        candidate = Path(directory) / "candidate"
        export(candidate)
        expected_files = {path.name for path in candidate.iterdir()}
        if {path.name for path in root.iterdir()} != expected_files:
            raise ValueError("Unexpected frozen fixture inventory")
        for name in expected_files:
            if (root / name).read_bytes() != (candidate / name).read_bytes():
                raise ValueError(f"Frozen oracle/gate differs: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--build", type=Path)
    group.add_argument("--check", type=Path, default=FIXTURE)
    args = parser.parse_args()
    manifest = export(args.build) if args.build else verify(args.check)
    print(f"Verified {manifest['format']}: {len(manifest['arrays'])} arrays and five independent pre-target gates")


if __name__ == "__main__":
    main()
