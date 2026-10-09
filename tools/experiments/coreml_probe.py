"""Explicit public Core ML projection/persistent-KV slice experiment; no engine path."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/reference"))
sys.path.insert(0, str(ROOT / "tools/bench"))
import coreml_slice as oracle
import mlx_baseline as common

STATE_KINDS = ("stateful_attention", "stateful_attention_fp32_accum")
KINDS = ("projection_fp32", "projection_fp16", *STATE_KINDS)
POLICIES = ("cpu_only", "cpu_and_ne")
EXTRAS = {"coremltools": "9.0", "cattrs": "25.3.0", "pyaml": "25.7.0"}


def directory_hash(root):
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(bytes.fromhex(oracle.digest(path)))
    return digest.hexdigest()


def identity():
    return {
        "engine_revision": common.command(["git", "rev-parse", "HEAD"]),
        "working_tree_dirty": bool(common.command(["git", "status", "--porcelain"])),
        "driver_sha256": oracle.digest(Path(__file__)),
        "oracle_sha256": oracle.digest(Path(oracle.__file__)),
        "fixture_manifest_sha256": oracle.digest(oracle.FIXTURE / "manifest.json"),
        "extras_lock_sha256": oracle.digest(Path(__file__).with_name("coreml-extras.lock")),
        "dependency_versions": {name: importlib.metadata.version(name) for name in (
            *EXTRAS, "numpy", "protobuf", "sympy", "tqdm", "packaging", "attrs", "PyYAML", "typing_extensions")},
        "python": platform.python_version(),
        "device": common.command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
        "hardware_model": common.command(["/usr/sbin/sysctl", "-n", "hw.model"]),
        "physical_memory_bytes": int(common.command(["/usr/sbin/sysctl", "-n", "hw.memsize"])),
        "os": common.command(["sw_vers", "-productVersion"]),
        "os_build": common.command(["sw_vers", "-buildVersion"]),
    }


def power():
    return {"source": common.command(["/usr/bin/pmset", "-g", "batt"]),
            "configuration": common.command(["/usr/bin/pmset", "-g", "custom"]),
            "thermal": common.command(["/usr/bin/pmset", "-g", "therm"])}


def load_coreml():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("This explicit Core ML probe requires Apple Silicon macOS")
    for name, expected in EXTRAS.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Isolated experiment version differs: {name}")
    start = time.perf_counter()
    import coremltools as ct
    if not Path(ct.__file__).resolve().is_relative_to(ROOT / "artifacts/coreml-tools"):
        raise ValueError("Use the isolated artifacts/coreml-tools installation")
    return ct, time.perf_counter() - start


def program(ct, kind, manifest):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    if kind.startswith("projection"):
        half = kind == "projection_fp16"
        dtype = np.float16 if half else np.float32
        storage = types.fp16 if half else types.fp32
        weights = oracle.read(oracle.FIXTURE, manifest, "projection.weight").astype(dtype)
        bias = oracle.read(oracle.FIXTURE, manifest, "projection.bias").astype(dtype)

        @mb.program(input_specs=[mb.TensorSpec((1, 32), dtype=storage)], opset_version=ct.target.macOS15)
        def projection(x):
            return mb.linear(x=x, weight=weights, bias=bias, name="output")
        return projection

    @mb.program(input_specs=[
        mb.TensorSpec((2, 2, 8), dtype=types.fp16),
        mb.TensorSpec((2, 1, 8), dtype=types.fp16),
        mb.TensorSpec((2, 1, 8), dtype=types.fp16),
        mb.TensorSpec((1,), dtype=types.int32),
        mb.StateTensorSpec((2, 32, 8), dtype=types.fp16),
        mb.StateTensorSpec((2, 32, 8), dtype=types.fp16),
    ], opset_version=ct.target.macOS15)
    def attention(q, k, v, position, cache_k, cache_v):
        old_k = mb.read_state(input=cache_k)
        new_k = mb.scatter(data=old_k, indices=position, updates=k, axis=1, mode="update")
        keys = mb.coreml_update_state(state=cache_k, value=new_k)
        old_v = mb.read_state(input=cache_v)
        new_v = mb.scatter(data=old_v, indices=position, updates=v, axis=1, mode="update")
        values = mb.coreml_update_state(state=cache_v, value=new_v)
        precise = kind == "stateful_attention_fp32_accum"
        if precise:
            q = mb.cast(x=q, dtype="fp32")
            keys = mb.cast(x=keys, dtype="fp32")
            values = mb.cast(x=values, dtype="fp32")
        scores = mb.matmul(x=q, y=keys, transpose_y=True)
        dtype = np.float32 if precise else np.float16
        scores = mb.mul(x=scores, y=dtype(1 / np.sqrt(8)))
        valid = mb.less_equal(x=np.arange(32, dtype=np.int32).reshape(1, 1, 32), y=position)
        masked = mb.select(cond=valid, a=scores, b=dtype(-65504))
        probabilities = mb.softmax(x=masked, axis=-1)
        context = mb.matmul(x=probabilities, y=values)
        if precise:
            context = mb.cast(x=context, dtype="fp16")
        return mb.reshape(x=context, shape=[1, 32], name="output")
    return attention


def prepare(output):
    if output.exists():
        raise ValueError("Refusing to overwrite conversion artifacts")
    manifest = oracle.verify()
    ct, import_seconds = load_coreml()
    record = identity()
    record.update({"format": "toklet-coreml-slice-build-v1", "import_seconds": import_seconds,
                   "conversion_source": "public MIL graph with immutable independent fixture arrays; no PyTorch conversion",
                   "deployment_target": "macOS15/iOS18", "models": {}})
    output.mkdir(parents=True)
    for kind in KINDS:
        graph = program(ct, kind, manifest)
        start = time.perf_counter()
        model = ct.convert(graph, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                           compute_precision=ct.precision.FLOAT32 if kind in ("projection_fp32", "stateful_attention_fp32_accum") else ct.precision.FLOAT16,
                           compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        conversion_seconds = time.perf_counter() - start
        path = output / f"{kind}.mlpackage"
        start = time.perf_counter()
        model.save(str(path))
        save_seconds = time.perf_counter() - start
        spec = model.get_spec()
        record["models"][kind] = {
            "package": path.name, "package_sha256": directory_hash(path),
            "conversion_seconds": conversion_seconds, "save_seconds": save_seconds,
            "state_names": [state.name for state in spec.description.state],
            "logical_state_bytes": 2048 if kind in STATE_KINDS else 0,
            "logical_weights_bytes": 4224 if kind == "projection_fp32" else 2112 if kind == "projection_fp16" else 0,
            "computation": "FP32 graph arithmetic" if kind in ("projection_fp32", "stateful_attention_fp32_accum") else "FP16 graph arithmetic",
            "state_types": {feature.name: {"shape": list(feature.type.stateType.arrayType.shape),
                                           "data_type_enum": feature.type.stateType.arrayType.dataType}
                            for feature in spec.description.state},
            "input_types": {feature.name: {"shape": list(feature.type.multiArrayType.shape),
                                           "data_type_enum": feature.type.multiArrayType.dataType}
                            for feature in spec.description.input},
            "output_types": {feature.name: {"shape": list(feature.type.multiArrayType.shape),
                                            "data_type_enum": feature.type.multiArrayType.dataType}
                             for feature in spec.description.output},
        }
        print(f"Converted {kind}: {conversion_seconds:.3f}s", flush=True)
    record["process_peak_footprint_bytes"] = common.footprint_peak()
    (output / "manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def check(actual, expected, gate):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Target output shape/finiteness differs")
    error = np.abs(actual.astype(np.float64) - expected)
    tolerance = gate["atol"] + gate["rtol"] * np.abs(expected.astype(np.float64))
    fraction = float(np.max(error / tolerance))
    if fraction > 1:
        raise ValueError(f"Target exceeds frozen gate: tolerance fraction {fraction}; maximum absolute error {float(error.max())}")
    return {"maximum_abs_error": float(error.max()), "maximum_tolerance_fraction": fraction}


def inputs(manifest, case, kind):
    count = manifest["cases"][case]["tokens"]
    if not 0 < count <= 32:
        raise ValueError("Slice capacity admission failed")
    if kind.startswith("projection"):
        dtype = np.float16 if kind == "projection_fp16" else np.float32
        x = oracle.read(oracle.FIXTURE, manifest, f"{case}.projection_input").astype(dtype)
        return [{"x": x[row:row + 1]} for row in range(count)]
    q, k, v = (oracle.read(oracle.FIXTURE, manifest, f"{case}.{name}") for name in ("query", "key", "value"))
    return [{"q": q[row].reshape(2, 2, 8), "k": k[row].reshape(2, 1, 8),
             "v": v[row].reshape(2, 1, 8), "position": np.asarray([row], dtype=np.int32)}
            for row in range(count)]


def check_cache(state, manifest, case):
    dtypes = {}
    for name in ("k", "v"):
        actual = state.read_state(f"cache_{name}")
        expected = oracle.read(oracle.FIXTURE, manifest, f"{case}.cache_{name}_expected")
        # Public read_state returns lossless FP32 values on this runtime even
        # for an FP16 state spec. Compare every value's exact promoted bits;
        # accept no numerical tolerance or FP32-to-FP16 rounding of actual data.
        if (actual.shape != expected.shape or actual.dtype not in (np.float16, np.float32)
                or not np.array_equal(actual.astype(np.float32).view(np.uint32),
                                      expected.astype(np.float32).view(np.uint32))):
            raise ValueError(f"Persistent {name} cache differs from exact frozen FP16 prefix")
        dtypes[name] = str(actual.dtype)
    return dtypes


def sequence(model, manifest, case, kind):
    prepared = inputs(manifest, case, kind)
    state_start = time.perf_counter()
    state = model.make_state() if kind in STATE_KINDS else None
    state_seconds = time.perf_counter() - state_start if state else 0.0
    elapsed = [0.0] * len(prepared)
    outputs = np.empty((len(prepared), 32), dtype=np.float32)
    start = time.perf_counter()
    for row, request in enumerate(prepared):
        prediction_start = time.perf_counter()
        response = model.predict(request, state=state)
        elapsed[row] = time.perf_counter() - prediction_start
        value = np.asarray(response["output"])
        if value.shape != (1, 32):
            raise ValueError("Target output shape differs")
        outputs[row] = value[0]
    wall_seconds = time.perf_counter() - start
    expected_name = "context_expected" if state else "projection_expected"
    gate_name = "attention_fp16_gate" if state else f"{kind}_gate"
    expected = oracle.read(oracle.FIXTURE, manifest, f"{case}.{expected_name}")
    metrics = check(outputs, expected, manifest["cases"][case][gate_name])
    readback_dtypes = check_cache(state, manifest, case) if state else None
    return {"prediction_seconds": elapsed, "completed_prediction_sum_seconds": sum(elapsed),
            "sequence_wall_seconds": wall_seconds, "state_creation_seconds": state_seconds,
            "returned_output_dtype": str(value.dtype),
            "state_readback_dtypes": readback_dtypes,
            "tokens": len(prepared), "cache_exact": True if state else None, **metrics}


def poison_caches(state):
    # The public Python write API accepts FP32 arrays for this FP16 state.
    # 0.5 is exactly representable in both types. Verify the complete installed
    # state before prediction, so an ignored/adapted write cannot pass masking.
    expected = np.full((2, 32, 8), 0.5, dtype=np.float32)
    for name in ("k", "v"):
        state.write_state(f"cache_{name}", expected)
    for name in ("k", "v"):
        actual = state.read_state(f"cache_{name}")
        if (actual.shape != expected.shape or actual.dtype not in (np.float16, np.float32)
                or not np.array_equal(actual.astype(np.float32).view(np.uint32),
                                      expected.view(np.uint32))):
            raise ValueError(f"Poisoned {name} cache installation differs")


def state_isolation(model, manifest):
    a, b = model.make_state(), model.make_state()
    first = inputs(manifest, "prompt", "stateful_attention")
    second = inputs(manifest, "single", "stateful_attention")
    outputs = []
    for request in first[:2]:
        outputs.append(model.predict(request, state=a)["output"][0])
    other = model.predict(second[0], state=b)["output"]
    for request in first[2:]:
        outputs.append(model.predict(request, state=a)["output"][0])
    check(np.asarray(outputs), oracle.read(oracle.FIXTURE, manifest, "prompt.context_expected"),
          manifest["cases"]["prompt"]["attention_fp16_gate"])
    check(np.asarray(other), oracle.read(oracle.FIXTURE, manifest, "single.context_expected"),
          manifest["cases"]["single"]["attention_fp16_gate"])
    check_cache(a, manifest, "prompt")
    check_cache(b, manifest, "single")
    fresh = model.make_state()
    if any(np.count_nonzero(fresh.read_state(f"cache_{name}")) for name in ("k", "v")):
        raise ValueError("New state is not empty")
    # A diagnostic poisoned future suffix must remain masked from the first row.
    poison_caches(fresh)
    actual = model.predict(second[0], state=fresh)["output"]
    check(np.asarray(actual), oracle.read(oracle.FIXTURE, manifest, "single.context_expected"),
          manifest["cases"]["single"]["attention_fp16_gate"])
    return {"interleaved_independent_states": True, "fresh_state_zero": True,
            "poison_installation_verified": True, "poisoned_future_masked": True}


def compute_plan(ct, model, policy):
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=policy)
        rows = []

        def visit(block):
            for operation in block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
                rows.append({"operation": operation.operator_name,
                             "outputs": [value.name for value in operation.outputs],
                             "anticipated_preferred_device": type(usage.preferred_compute_device).__name__ if usage else None,
                             "anticipated_supported_devices": [type(device).__name__ for device in usage.supported_compute_devices] if usage else None,
                             "estimated_relative_cost": cost.weight if cost else None})
                for nested in operation.blocks:
                    visit(nested)
        for function in plan.model_structure.program.functions.values():
            visit(function.block)
        return {"available": True, "operations": rows,
                "scope": "Anticipated device assignment/cost only; no actual execution trace."}
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def run(packages, output, kind, policy_name):
    raw = output.with_suffix(output.suffix + ".samples.jsonl")
    if output.exists() or raw.exists():
        raise ValueError("Refusing to overwrite runtime evidence")
    manifest = oracle.verify()
    build = json.loads((packages / "manifest.json").read_text())
    if (build["driver_sha256"] != oracle.digest(Path(__file__))
            or build["fixture_manifest_sha256"] != oracle.digest(oracle.FIXTURE / "manifest.json")):
        raise ValueError("Conversion code or frozen oracle identity differs")
    item = build["models"][kind]
    package = packages / item["package"]
    if directory_hash(package) != item["package_sha256"]:
        raise ValueError("Converted package identity differs")
    ct, import_seconds = load_coreml()
    policy = ct.ComputeUnit.CPU_ONLY if policy_name == "cpu_only" else ct.ComputeUnit.CPU_AND_NE
    record = identity()
    if record["dependency_versions"] != build["dependency_versions"]:
        raise ValueError("Conversion/runtime dependency identities differ")
    record.update({"format": "toklet-coreml-slice-run-v1", "kind": kind, "policy": policy_name,
                   "conversion_manifest_sha256": oracle.digest(packages / "manifest.json"),
                   "package_sha256": item["package_sha256"], "import_seconds": import_seconds,
                   "power_start": power(), "cases": {}, "completed": False,
                   "actual_runtime_placement": "CPU by explicit policy" if policy_name == "cpu_only" else None,
                   "limitations": ["Slice only; no full decoder/generation/quality or engine adoption.",
                                    "Predict is synchronous; individual prediction timers exclude prepared input dictionaries, fixture reads, state creation, correctness/cache readback and transport. Native predict output allocation is included.",
                                    "Core ML's Python predict wrapper promotes FP16 input multiarrays to FP32 before the native call; these API/conversion costs and model-boundary conversions are included. Graph/state precision remains explicitly typed.",
                                    "Sequence wall time additionally includes Python loop/output-copy overhead. Component times overlap; do not add prediction sum to wall time.",
                                    "Own-process footprint includes Python/import/conversion/load support, but does not establish total Core ML service/device memory or energy.",
                                    "Compute plan reports anticipated placement, not actual ANE execution. Full Xcode/Instruments execution trace is unavailable in this command-line-tools environment.",
                                    "Core ML import warns about installed Torch 2.9.1; graphs use public MIL directly, not the untested Torch conversion frontend."]})
    output.parent.mkdir(parents=True, exist_ok=True)
    failure = None
    with raw.open("x") as checkpoint:
        try:
            stage = "model_compile_load"
            start = time.perf_counter()
            model = ct.models.MLModel(str(package), compute_units=policy)
            record["model_compile_load_seconds"] = time.perf_counter() - start
            record["compute_plan"] = compute_plan(ct, model, policy)
            for case in oracle.CASES:
                samples = []
                for repetition in range(13):
                    stage = f"{case}/{'warmup' if repetition < 3 else 'measured'}/{repetition + 1}"
                    sample = sequence(model, manifest, case, kind)
                    if repetition >= 3:
                        samples.append(sample)
                        checkpoint.write(json.dumps({"case": case, "sample": sample}, allow_nan=False) + "\n")
                        checkpoint.flush()
                values = [sample["completed_prediction_sum_seconds"] for sample in samples]
                record["cases"][case] = {"warmups": 3, "repetitions": 10, "samples": samples,
                                          "prediction_sum_median_seconds": statistics.median(values),
                                          "prediction_sum_min_seconds": min(values), "prediction_sum_max_seconds": max(values)}
                print(f"{kind}/{policy_name}/{case}: {statistics.median(values):.6f}s", flush=True)
            if kind in STATE_KINDS:
                stage = "state_isolation_and_future_mask"
                record["state_checks"] = state_isolation(model, manifest)
            record["completed"] = True
        except Exception as error:
            failure = error
            record["failure"] = {"stage": stage, "type": type(error).__name__, "message": str(error)}
        finally:
            record["power_end"] = power()
            record["process_peak_footprint_bytes"] = common.footprint_peak()
            if (record["process_peak_footprint_bytes"] is None
                    or record["process_peak_footprint_bytes"] > 6 * 1024**3):
                record["completed"] = False
                record["memory_qualification"] = "unavailable or over budget"
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if failure:
        raise failure
    if not record["completed"]:
        raise ValueError("Memory qualification did not pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--run", type=Path)
    parser.add_argument("--packages", type=Path)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--policy", choices=POLICIES)
    args = parser.parse_args()
    if args.prepare:
        if args.packages or args.kind or args.policy:
            parser.error("prepare accepts no runtime options")
        prepare(args.prepare)
    else:
        if not args.packages or not args.kind or not args.policy:
            parser.error("run requires packages, kind, and policy")
        run(args.packages, args.run, args.kind, args.policy)


if __name__ == "__main__":
    # Bound only this explicit experiment process; completed raw rows survive a
    # native-call hang, but no final success record is emitted after a timeout.
    watchdog = threading.Timer(600, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        main()
    finally:
        watchdog.cancel()
