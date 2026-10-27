"""Independent C512 Qwen3 captures using the unchanged batched CPU graph.

Half activations/KV, original FP32 affine weights and FP32 logits remain the
storage contract. FP32/FP64 arithmetic are separate witnesses, not upstream BF16.
No Rust/Metal execution or new gate calibration takes place here.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CONTEXT = 512
LAYERS = [0, 1, 27]
TOKENS = [9707, 11, 1879, 0] * 128
CASES = {"original": TOKENS, "future_tail": TOKENS[:-1] + [17]}
CONTRACTS = ["fp32", "fp64"]
OLD = ROOT / "tests/fixtures/mlx-qwen3-real-v1/manifest.json"
OLD_SHA = "4594118378d2c583c80c8f0987e06d3bbca0fa77188d4869e670252070b23fc7"
OLD_ARRAY = ROOT / "artifacts/m8-qwen3-real-reference-v1/arrays.bin"
OLD_ARRAY_SHA = "812c8302de38ca6bf2c073a23940cfef27a64a950c1bfa8e841444606d3dd010"
LIMIT = 6 * 1024**3
STRIPE = 65536


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path, expected=None):
    with Path(path).open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2 or (expected and hashlib.sha256(raw).hexdigest() != expected):
        raise ValueError("Bounded reference document differs")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate reference field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def geometry(rows):
    """Literal independent boundary inventory, not derived from captured output."""
    result = {"embedding": ([rows, 1024], "<f2")}
    for layer in LAYERS:
        for stage in ["input_norm", "q_projection", "k_projection", "v_projection",
                      "q_norm", "k_norm", "q_rotary", "k_rotary", "attention_scores",
                      "attention_probabilities", "attention_context", "attention_projection",
                      "attention_residual", "post_attention_norm", "gate_projection",
                      "up_projection", "gate_silu", "gated", "down_projection", "output"]:
            shape, dtype = [rows, 1024], "<f2"
            if stage == "q_projection":
                shape = [rows, 2048]
            elif stage in ["q_norm", "q_rotary"]:
                shape = [rows, 16, 128]
            elif stage in ["k_norm", "k_rotary"]:
                shape = [rows, 8, 128]
            elif stage == "attention_context":
                shape = [rows, 2048]
            elif stage in ["attention_scores", "attention_probabilities"]:
                shape, dtype = [16, rows, rows], "<f4"
            elif stage in ["gate_projection", "up_projection", "gate_silu", "gated"]:
                shape = [rows, 3072]
            result[f"layers.{layer}.{stage}"] = (shape, dtype)
    result["final_norm"] = ([rows, 1024], "<f2")
    result["logits"] = ([rows, 151936], "<f4")
    return result


def gates():
    old = load(OLD, OLD_SHA)
    return {name: old["arrays"]["cases.tile17." + name]["gate"] for name in geometry(CONTEXT)}


def ledger():
    capture = sum(math.prod(shape) * (2 if dtype == "<f2" else 4)
                  for shape, dtype in geometry(CONTEXT).values())
    assert capture == 508297216
    # Production owns only one returned graph; comparison starts after weights
    # close. Conservatively count all four resident capture mappings as well.
    parts = dict(tied_fp32=622329856, source_mapping=633442994,
                 returned_graph=capture, current_matrix_and_blas=384 * 1024**2,
                 live_operator_temporaries=384 * 1024**2,
                 comparison_mapped_pages=4 * capture, short_reference_mapped_pages=41115648,
                 comparison_stripes=32 * 1024**2,
                 python_numpy_frameworks=512 * 1024**2)
    return dict(parts=parts, admitted=sum(parts.values()), limit=LIMIT,
                capture_bytes=capture, four_capture_bytes=4 * capture)


def plan():
    return dict(format="toklet-qwen3-context512-reference-v1", context=CONTEXT,
                cases=CASES, contracts=CONTRACTS, layers=LAYERS,
                old_manifest_sha256=OLD_SHA, old_arrays_sha256=OLD_ARRAY_SHA,
                storage="Half activations/KV; FP32 affine weights/logits; separate FP32/FP64 arithmetic",
                gates=gates(), memory=ledger(), target_executed=False,
                upstream_bf16_qualified=False, held_out_quality_qualified=False,
                phone_qualified=False, model_capacity_raised=False)


def validate(manifest):
    if manifest["plan"] != plan() or set(manifest["records"]) != {
            case + "." + contract for case in CASES for contract in CONTRACTS}:
        raise ValueError("Reference plan or capture coverage differs")
    expected = geometry(CONTEXT)
    for label, record in manifest["records"].items():
        if record["file"] != label + ".bin" or set(record["arrays"]) != set(expected):
            raise ValueError("Reference array coverage/path differs")
        cursor = 0
        for name in sorted(expected):
            item = record["arrays"][name]
            shape, dtype = expected[name]
            size = math.prod(shape) * (2 if dtype == "<f2" else 4)
            if (item["shape"] != shape or item["dtype"] != dtype
                    or type(item["offset"]) is not int or item["offset"] != cursor
                    or type(item["bytes"]) is not int or item["bytes"] != size):
                raise ValueError("Reference layout differs: " + name)
            for value in [item["sha256"], record["sha256"]]:
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("Reference hash differs")
            cursor += size
        if record["bytes"] != cursor or cursor != ledger()["capture_bytes"]:
            raise ValueError("Reference byte count differs")


def save_graph(folder, label, values):
    import numpy as np
    expected = geometry(CONTEXT)
    if set(values) != set(expected):
        raise ValueError("Graph boundary inventory differs")
    arrays, whole, cursor = {}, hashlib.sha256(), 0
    with (folder / (label + ".bin")).open("xb") as stream:
        for name in sorted(expected):
            value = values[name]
            shape, dtype = expected[name]
            if (list(value.shape) != shape or value.dtype != np.dtype(dtype)
                    or not value.flags.c_contiguous or not np.isfinite(value).all()):
                raise ValueError("Graph shape/storage/finite contract differs: " + name)
            raw = memoryview(value).cast("B")
            sha = hashlib.sha256()
            for start in range(0, len(raw), STRIPE):
                part = raw[start:start + STRIPE]
                stream.write(part); sha.update(part); whole.update(part)
            arrays[name] = dict(shape=shape, dtype=dtype, offset=cursor,
                                bytes=len(raw), sha256=sha.hexdigest())
            cursor += len(raw)
    return dict(file=label + ".bin", bytes=cursor, sha256=whole.hexdigest(), arrays=arrays)


class Capture:
    """Read-only mapped captures, sliced before converting to comparison FP64."""
    def __init__(self, folder, record):
        import numpy as np
        path = folder / record["file"]
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Frozen capture differs")
        self.record = record
        self.mapping = np.memmap(path, dtype="u1", mode="r")

    def array(self, name):
        import numpy as np
        item = self.record["arrays"][name]
        raw = self.mapping[item["offset"]:item["offset"] + item["bytes"]]
        sha = hashlib.sha256()
        for start in range(0, len(raw), STRIPE):
            sha.update(raw[start:start + STRIPE])
        if sha.hexdigest() != item["sha256"]:
            raise ValueError("Frozen array differs")
        return np.ndarray(item["shape"], dtype=item["dtype"], buffer=raw)


def compare_arrays(left, right, gate):
    import numpy as np
    if left.shape != right.shape:
        raise ValueError("Comparison shapes differ")
    # Rows are at most 151936 elements; each stripe stays under 4MiB at FP64.
    maximum, fraction, failures, count = 0.0, 0.0, 0, 0
    for index in range(left.shape[0]):
        a = left[index].astype(np.float64)
        b = right[index].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite saved comparison")
        delta = np.abs(a - b)
        bound = gate["atol"] + gate["rtol"] * np.abs(b)
        maximum = max(maximum, float(np.max(delta)))
        fraction = max(fraction, float(np.max(delta / bound)))
        failures += int(np.count_nonzero(delta > bound)); count += a.size
    return dict(values=count, maximum_absolute=maximum, maximum_gate_fraction=fraction,
                failed_values=failures)


def compare(folder, manifest, report=None):
    import numpy as np
    validate(manifest)
    records, comparisons, causal, selections = manifest["records"], {}, {}, {}
    if report is None:
        report = {}
    # These dictionaries remain visible to the caller if a saved comparison
    # fails partway through. The caller persists them with the primary error.
    report.update(comparisons=comparisons, causal_prefix=causal, selections=selections)
    # Read the previous independent17-position oracle, never a native capture.
    if digest(OLD_ARRAY) != OLD_ARRAY_SHA:
        raise ValueError("Frozen short reference differs")
    old = load(OLD, OLD_SHA)
    old_map = np.memmap(OLD_ARRAY, dtype="u1", mode="r")
    def old_array(name):
        item = old["arrays"]["cases.tile17." + name]
        raw = old_map[item["offset"]:item["offset"] + item["bytes"]]
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Frozen short array differs")
        return np.ndarray(item["shape"], dtype=item["dtype"], buffer=raw)
    masked_zeros = 0
    nonzero_masked = 0
    for case in CASES:
        main, wide = [Capture(folder, records[case + "." + contract]) for contract in CONTRACTS]
        for name, gate in manifest["plan"]["gates"].items():
            a, b = main.array(name), wide.array(name)
            comparisons[case + ".contracts." + name] = compare_arrays(a, b, gate)
            for contract, value in [("fp32", a), ("fp64", b)]:
                prefix = value[:, :17, :17] if name.endswith(("attention_scores", "attention_probabilities")) else value[:17]
                comparisons[case + "." + contract + ".old17." + name] = compare_arrays(prefix, old_array(name), gate)
                if name.endswith("attention_probabilities"):
                    for head in value:
                        for row in range(CONTEXT):
                            nonzero_masked += int(np.count_nonzero(head[row, row + 1:]))
                            masked_zeros += CONTEXT - row - 1
            if name == "logits":
                for contract, value in [("fp32", a), ("fp64", b)]:
                    ids, margins = [], []
                    for row in value:
                        ids.append(int(np.argmax(row)))
                        top = np.partition(row, -2)[-2:]
                        margins.append(float(max(top) - min(top)))
                    selections[case + "." + contract] = dict(ids=ids, margins=margins)
        del a, b, main, wide
        gc.collect()
    for contract in CONTRACTS:
        first, second = [Capture(folder, records[case + "." + contract]) for case in CASES]
        for name in geometry(CONTEXT):
            a, b = first.array(name), second.array(name)
            if name.endswith(("attention_scores", "attention_probabilities")):
                a, b = a[:, :CONTEXT - 1, :CONTEXT - 1], b[:, :CONTEXT - 1, :CONTEXT - 1]
            else:
                a, b = a[:CONTEXT - 1], b[:CONTEXT - 1]
            # Exact typed equality, including signed zero, in bounded rows.
            different = sum(int(a[i].tobytes() != b[i].tobytes()) for i in range(a.shape[0]))
            causal[contract + "." + name] = dict(rows=a.shape[0], differing_rows=different)
        del a, b, first, second
        gc.collect()
    disagreements = {case: [i for i, (a, b) in enumerate(zip(selections[case + ".fp32"]["ids"], selections[case + ".fp64"]["ids"], strict=True)) if a != b] for case in CASES}
    return dict(comparisons=comparisons, causal_prefix=causal, selections=selections,
                arithmetic_argmax_disagreements=disagreements, masked_probability_elements=masked_zeros, nonzero_masked_probabilities=nonzero_masked,
                arithmetic_argmax_agreement=not any(disagreements.values()),
                common_greedy_oracle_ready=not any(disagreements.values()),
                mask_passed=nonzero_masked == 0,
                numerical_gates_passed=all(v["failed_values"] == 0 for v in comparisons.values()),
                causal_passed=all(v["differing_rows"] == 0 for v in causal.values()))


def export(folder):
    for variable in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
        os.environ[variable] = "1"
    import numpy as np
    import mlx_qwen3_real as source
    import qwen3_fp16
    sys.path.insert(0, str(ROOT / "tools/experiments"))
    import native_usage
    if np.__version__ != "2.3.2" or ledger()["admitted"] > LIMIT:
        raise ValueError("Reference version/memory admission differs")
    producers = [Path(__file__).resolve(), OLD, ROOT / "tools/reference/mlx_qwen3_real.py",
                 ROOT / "tools/reference/mlx_qwen3_checkpoint.py", ROOT / "tools/reference/qwen3_fp16.py",
                 ROOT / "tools/reference/qwen_fp16.py", ROOT / "tools/reference/qwen.py",
                 ROOT / "tools/bench/prepare.py", ROOT / "tools/experiments/native_usage.py"]
    bindings = {str(path.relative_to(ROOT)): digest(path) for path in producers}
    packages = source.package_pins()
    folder.mkdir()  # Never replace an earlier or partial capture.
    manifest = dict(plan=plan(), producer_pins=bindings, package_pins=packages,
                    model_pins=source.PINS, records={}, usage=[], capture_attempts=[], capture_error=None)
    (folder / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    numerical = dict(num_layers=28, num_query_heads=16, num_kv_heads=8,
                     head_dim=128, rms_norm_eps=1e-6, rope_theta=1e6)
    capture_error = None
    try:
        with source.Weights(source.MODEL) as weights:
            for case, tokens in CASES.items():
                for contract, precision in [("fp32", np.float32), ("fp64", np.float64)]:
                    label = case + "." + contract
                    attempt = dict(label=label, completed=False)
                    manifest["capture_attempts"].append(attempt)
                    values = qwen3_fp16.forward(tokens, 0, weights, numerical, precision, LAYERS)
                    manifest["records"][label] = save_graph(folder, label, values)
                    del values
                    gc.collect()
                    manifest["usage"].append(dict(phase=label, native=native_usage.usage()))
                    attempt["completed"] = True
                    # Immutable per-attempt journal survives later failures.
                    (folder / (label + ".json")).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
                    print(json.dumps(dict(captured=label, bytes=manifest["records"][label]["bytes"])), flush=True)
            weights.verify()
        gc.collect()
        validate(manifest)
    except BaseException as error:
        capture_error = error
        manifest["capture_error"] = type(error).__name__ + ": " + str(error)
    finally:
        # Preserve all accumulated descriptors even on source/forward failure.
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if capture_error is not None:
        (folder / "comparison.json").write_text(json.dumps(dict(comparison_completed=False, capture_error=manifest["capture_error"]), indent=2) + "\n")
        raise capture_error
    result = dict(comparison_completed=False, primary_error=None, endpoint_errors=[])
    comparison_error = None
    try:
        result.update(compare(folder, manifest, report=result))
        result["comparison_completed"] = True
    except BaseException as error:
        comparison_error = error
        result["primary_error"] = type(error).__name__ + ": " + str(error)
    finally:
        for name, check in [
                ("completed_native_usage", native_usage.usage),
                ("source_endpoints_unchanged", lambda: all(digest(ROOT / name) == pin for name, pin in {**bindings, **packages}.items()) and source.package_pins() == packages),
                ("model_endpoints_unchanged", lambda: all(digest(source.MODEL / name) == pin for name, pin in source.PINS.items()))]:
            try:
                result[name] = check()
            except BaseException as error:
                result["endpoint_errors"].append(name + ": " + type(error).__name__ + ": " + str(error))
        result["memory_passed"] = 0 < result.get("completed_native_usage", {}).get("lifetime_max_phys_footprint", 0) <= LIMIT
        (folder / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if comparison_error is not None:
        raise comparison_error
    passed = not result["endpoint_errors"] and all(result.get(key, False) for key in ["source_endpoints_unchanged", "model_endpoints_unchanged", "memory_passed", "numerical_gates_passed", "causal_passed", "mask_passed"])
    print(json.dumps(dict(passed=passed, manifest_sha256=digest(folder / "manifest.json"), comparison_sha256=digest(folder / "comparison.json"), argmax_disagreements=result["arithmetic_argmax_disagreements"], common_greedy_oracle_ready=result["common_greedy_oracle_ready"])), flush=True)
    if not passed:
        raise ValueError("Frozen long reference has retained numerical/causal/memory discrepancies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True, type=Path)
    export(parser.parse_args().export)
