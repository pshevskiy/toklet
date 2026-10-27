"""Original local quantized Qwen3 source profile; calibration diagnostics only.

MLX-LM is a reference implementation here, never a Toklet execution backend.
No tokenizer, remote code, adapter or model-configuration override is invoked.
"""
import gc
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import site
import sys
import sysconfig

import bf16_source_reference as runtime
import qwen3_context_calibration as calibration

VERSIONS = {"mlx": "0.32.3", "mlx-lm": "0.32.0", "transformers": "5.7.0", "numpy": "2.3.2"}
PACKAGES = ("mlx_lm", "transformers", "huggingface_hub", "sentencepiece", "google", "yaml", "packaging",
    "regex", "tqdm", "typer", "click", "filelock", "fsspec", "hf_xet", "httpx", "httpcore", "anyio",
    "certifi", "idna", "h11", "typing_extensions.py", "sniffio", "rich", "shellingham", "annotated_doc",
    "safetensors", "requests", "urllib3", "charset_normalizer", "pygments", "markdown_it", "mdurl")
ENVIRONMENT = {"USE_TORCH": "0", "USE_TF": "0", "USE_FLAX": "0", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "MLXLM_USE_MODELSCOPE": "False"}
LIMIT = 6 * 1024**3


def isolation_profile():
    """Verify package availability rather than trusting legacy backend flags."""
    prefix = calibration.reference.ROOT / ".venv/mlx-source-only-v1"
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    if (Path(sys.prefix).resolve() != prefix or Path(sys.exec_prefix).resolve() != prefix
            or purelib != prefix / "lib/python3.12/site-packages"
            or not sys.flags.no_user_site or site.ENABLE_USER_SITE is not False):
        raise ValueError("Isolated MLX-only reference interpreter required")
    config = (prefix / "pyvenv.cfg").read_bytes()
    if (len(config) > 4096 or b"include-system-site-packages = false" not in config
            or list(purelib.glob("*.pth")) or (purelib / "torch").exists()
            or list(purelib.glob("torch*.dist-info")) or importlib.util.find_spec("torch") is not None
            or any(name == "torch" or name.startswith("torch.") for name in sys.modules)):
        raise ValueError("Reference environment exposes optional Torch or site hooks")
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    allowed = {purelib, stdlib, stdlib / "lib-dynload", stdlib.parent / "python312.zip",
               calibration.reference.ROOT / "tools/reference", calibration.reference.ROOT / "tools/experiments",
               calibration.reference.ROOT / "tools/bench",
               Path(sys.argv[0]).resolve().parent}
    paths = [str(Path(path or ".").resolve()) for path in sys.path]
    if any(Path(path) not in allowed for path in paths):
        raise ValueError("Reference sys.path escaped admitted interpreter/source roots")
    return dict(profile="isolated-mlx-only-v1", prefix=str(prefix), purelib=str(purelib),
        system_site_packages=False, user_site_packages=False, torch_discoverable=False,
        pth_files=[], sys_path=paths, pyvenv_config_sha256=hashlib.sha256(config).hexdigest(),
        python_target=str(Path(sys.executable).resolve()),
        legacy_environment_flags_are_not_torch_isolation=True)


def source_paths(extra=()):
    """Extend the unchanged cold runtime contract with source dependencies."""
    site = Path(sysconfig.get_path("purelib")).resolve()
    selected = [Path(__file__).resolve(), Path(calibration.__file__).resolve(),
        Path(calibration.inputs.__file__).resolve(), Path(calibration.reference.__file__).resolve(),
        calibration.reference.ROOT / "tools/experiments/native_usage.py", *extra]
    for name in PACKAGES:
        path = site / name
        if path.is_file():
            selected.append(path)
        elif path.is_dir():
            selected.extend(q for q in path.rglob("*") if q.is_file() and "__pycache__" not in q.parts)
        elif name in ("mlx_lm", "transformers", "huggingface_hub"):
            raise ValueError("Required source package absent: " + name)
    # Version/availability probes inspect installed metadata even for disabled
    # optional backends. Their code and native libraries are not thereby admitted.
    for folder in site.glob("*.dist-info"):
        selected.extend(q for q in folder.rglob("*") if q.is_file() and "__pycache__" not in q.parts)
    return runtime.sources(selected)


def memory_plan(config, packed_bytes):
    """Conservative phase reserves, not claims about distinct live allocations.

    Conditional copy reserves are charged only as worst-case admission headroom;
    actual shared allocations are represented once in allocator/physical measures.
    """
    if (type(packed_bytes) is not int or packed_bytes <= 0
            or any(type(config.get(k)) is not int or config[k] != v for k, v in
                   (("hidden_size", 1024), ("vocab_size", 151936), ("num_hidden_layers", 28),
                    ("intermediate_size", 3072), ("num_attention_heads", 16),
                    ("num_key_value_heads", 8), ("head_dim", 128)))):
        raise ValueError("Frozen source geometry required for memory admission")
    hidden, vocab, layers = 1024, 151936, 28
    linear_per_layer = 2048 * hidden + 2 * hidden * hidden + hidden * 2048 + 3 * 3072 * hidden
    norms = layers * (2 * hidden + 2 * 128) + hidden
    fp32_initialization = 4 * (vocab * hidden + layers * linear_per_layer + norms)
    largest_fp32_matrix = 4 * vocab * hidden
    load = dict(original_packed_storage=packed_bytes, fp32_model_initialization=fp32_initialization,
        quantized_initialization_reserve=packed_bytes, conditional_packed_copy_reserve=packed_bytes,
        quantization_workspace_reserve=2 * largest_fp32_matrix, python_source_frameworks=512 * 1024**2)
    inference = dict(original_packed_storage=packed_bytes, graph_workspace_reserve=3 * 1024**3,
        full_output_max_fp32=512 * vocab * 4, conversion_and_verification_tiles=160 * 1024**2,
        python_source_frameworks=512 * 1024**2)
    admitted = max(sum(load.values()), sum(inference.values()))
    if admitted > LIMIT:
        raise ValueError("Source phase memory admission exceeds limit")
    return dict(load=load, inference=inference, admitted=admitted, limit=LIMIT,
        phases_are_separate=True, conditional_reserves_are_not_measured_duplicates=True)


def model_inventory(folder, pins):
    folder = Path(folder)
    if (folder.is_symlink() or sorted(folder.glob("model*.safetensors")) != [folder / "model.safetensors"]):
        raise ValueError("Exact original single-shard source inventory required")
    for name, expected in pins.items():
        path = folder / name
        if path.is_symlink() or not path.is_file() or calibration.reference.digest(path) != expected:
            raise ValueError("Original source artifact differs: " + name)


def verify_widening(raw, wide, dtype):
    """Exact finite source-bit widening, retaining nonfinites for rejection."""
    import numpy as np
    width = {"bfloat16": 2, "float16": 2, "float32": 4}.get(dtype)
    if width is None or not raw or len(raw) > 64 * 1024**2 or len(raw) % width or len(wide) != len(raw) // width * 4:
        raise ValueError("Source/widened tile extent differs")
    if dtype == "bfloat16":
        expected = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
    elif dtype == "float16":
        expected = np.frombuffer(raw, dtype="<f2").astype("<f4").view("<u4")
    else:
        expected = np.frombuffer(raw, dtype="<u4")
    actual = np.frombuffer(wide, dtype="<u4")
    finite = expected & 0x7f800000 != 0x7f800000
    return dict(values=len(raw) // width, nonfinite_values=int(np.count_nonzero(~finite)),
                finite_widening_mismatches=int(np.count_nonzero(finite & (expected != actual))))


def write_piece(file, body, descriptor, digest):
    """Track accepted bytes even when a writer returns a short count."""
    for start in range(0, len(body), 65536):
        part = body[start:start + 65536];written = file.write(part)
        if type(written) is not int or not 0 <= written <= len(part):
            raise OSError("Invalid source capture write count")
        descriptor["bytes"] += written;digest.update(part[:written]);descriptor["sha256"] = digest.hexdigest()
        if written != len(part):
            raise OSError("Incomplete source capture write")


def memory(mx):
    return dict(active_bytes=mx.get_active_memory(), peak_bytes=mx.get_peak_memory(), cache_bytes=mx.get_cache_memory())


def admit_observed_memory(observation):
    if (not 0 < observation["native"]["lifetime_max_phys_footprint"] <= LIMIT
            or not 0 <= observation["mlx"]["peak_bytes"] <= LIMIT):
        raise ValueError("Original source measured memory limit exceeded")


def verify_capture(folder, manifest_sha, expected_plan, report):
    """Authenticate original bits and widened values without importing MLX-LM."""
    from qwen3_context_precision import Owner
    folder = Path(folder)
    manifest = calibration.reference.load(folder / "manifest.json", manifest_sha)
    report.update(complete=False, parents=[], cases=[], values=0)
    cases = expected_plan["cases"];columns = expected_plan["outputs"]["columns"]
    if (manifest.get("complete") is not True or manifest.get("primary_error") is not None
            or manifest.get("cleanup_errors") != [] or manifest.get("model_pins") != expected_plan["model_pins"]
            or manifest.get("memory_admission") != expected_plan["memory"]
            or set(manifest["records"]) != {case["id"] for case in cases}
            or len(manifest["attempts"]) != len(cases)):
        raise ValueError("Complete original-source manifest/profile required")
    for case, attempt in zip(cases, manifest["attempts"], strict=True):
        record = manifest["records"][case["id"]]
        dtype = record["source_dtype"];width = {"bfloat16": 2, "float16": 2, "float32": 4}.get(dtype)
        if (width is None or attempt.get("label") != case["id"] or attempt.get("completed") is not True
                or attempt.get("evaluated") is not True or attempt.get("observed_shape") != [1, case["rows"], columns]
                or any(type(dim) is not int for dim in attempt["observed_shape"])
                or attempt.get("observed_dtype", "").removeprefix("mlx.core.") != dtype
                or record.get("complete") is not True or type(record.get("rows")) is not int
                or record["rows"] != case["rows"] or type(record.get("columns")) is not int
                or record["columns"] != columns or type(record["nonfinite_values"]) is not int or record["nonfinite_values"] != 0
                or type(record["finite_widening_mismatches"]) is not int or record["finite_widening_mismatches"] != 0):
            raise ValueError("Original source case/dtype/coverage differs")
        expected_tiles = [(start, min(32, case["rows"] - start)) for start in range(0, case["rows"], 32)]
        if len(record["tiles"]) != len(expected_tiles):
            raise ValueError("Original source tile count differs")
        for tile, (start, rows) in zip(record["tiles"], expected_tiles, strict=True):
            expected = dict(start=start, rows=rows, raw_offset=start * columns * width,
                widened_offset=start * columns * 4, complete=True, values=rows * columns,
                nonfinite_values=0, finite_widening_mismatches=0)
            if tile != expected or tile.get("complete") is not True or any(type(tile[k]) is not int for k in expected if k != "complete"):
                raise ValueError("Original source tile schema/extent differs")
        owners = []
        from contextlib import ExitStack
        with ExitStack() as stack:
            for key, suffix, stride, storage in (("raw", ".source-bits.bin", width, "<u2" if width == 2 else "<u4"),
                                                ("widened", ".f32le.bin", 4, "<f4")):
                item = record[key];size = case["rows"] * columns * stride
                if (item.get("file") != case["id"] + suffix or item.get("dtype") != storage
                        or type(item.get("bytes")) is not int or item["bytes"] != size or item["intended_bytes"] != size
                        or item.get("layout") != "row-major [tokens,vocabulary]"):
                    raise ValueError("Original source raw storage descriptor differs")
                owner = Owner(folder / item["file"], dict(bytes=size, sha256=item["sha256"]))
                report["parents"].append(owner.audit);owners.append(stack.enter_context(owner))
            raw, wide = owners
            raw.file.seek(0);wide.file.seek(0)
            remaining = case["rows"] * columns * width
            observation = dict(case=case["id"], values=0, complete=False)
            report["cases"].append(observation)
            while remaining:
                size = min(65536, remaining);body = raw.file.read(size);widened = wide.file.read(size // width * 4)
                if len(body) != size or len(widened) != size // width * 4:
                    raise ValueError("Source bit/widened stream truncated")
                result = verify_widening(body, widened, dtype)
                observation["values"] += result["values"];report["values"] += result["values"]
                if result["nonfinite_values"] or result["finite_widening_mismatches"]:
                    raise ValueError("Source bit-derived widening differs or is nonfinite")
                remaining -= size
            observation["complete"] = True
    report["complete"] = True
    return manifest["records"]


def export(folder, input_folder, model_folder, pins, report, endpoint):
    """Caller admits runtime before import and verifies endpoints afterward."""
    import os
    if any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise ValueError("Frozen offline source environment required")
    availability = isolation_profile()
    if {name: version(name) for name in VERSIONS} != VERSIONS:
        raise ValueError("Pinned original source runtime required")
    # These imports must occur only after external cold runtime admission.
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.utils import load_model
    import native_usage
    endpoint()  # Import-only origin/native audit before constructing a model.
    inputs, cases = calibration.selected_inputs(input_folder)
    model_inventory(model_folder, pins)
    config = calibration.reference.load(Path(model_folder) / "config.json", pins["config.json"])
    if (config.get("model_type") != "qwen3" or "model_file" in config
            or config.get("quantization") != {"bits": 8, "group_size": 64}
            or config.get("tie_word_embeddings") is not True):
        raise ValueError("Original static tied Qwen3 affine profile required")
    admission = memory_plan(config, (Path(model_folder) / "model.safetensors").stat().st_size)
    folder = Path(folder);folder.mkdir()
    report.update(complete=False, primary_error=None, records={}, attempts=[], memory=[],
        input_manifest_sha256=calibration.INPUT_SHA, model_pins=pins, environment=ENVIRONMENT,
        availability_profile=availability,
        memory_admission=admission, input_profile=inputs["profile"],
        versions=VERSIONS, config=config, dtype_override=False, tokenizer_invoked=False,
        batch=1, offset=0, cache=None, model_config=None, trust_remote_code=False,
        native_executed=False, validation_evaluated=False, source_parity_qualified=False,
        quality_qualified=False, adopted=False)
    primary, model = None, None
    output = tile = bits = wide = original = widened = raw_body = wide_body = None
    try:
        mx.set_default_device(mx.gpu)
        report["device"] = mx.device_info()
        report["previous_cache_limit"] = mx.set_cache_limit(0)
        mx.random.seed(0)
        model, loaded = load_model(Path(model_folder), lazy=False, strict=True, model_config=None,
            get_model_classes=lambda config: (Model, ModelArgs), trust_remote_code=False)
        if loaded != config or not isinstance(model, Model) or model.args.tie_word_embeddings is not True:
            raise ValueError("Loaded original source config/class differs")
        model.eval()
        report["class"] = dict(module=type(model).__module__, name=type(model).__name__)
        report["parameters"] = [dict(name=name, shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes)
                                for name, value in tree_flatten(model.parameters())]
        report["quantized_modules"] = [dict(name=name, module=type(value).__module__, kind=type(value).__name__,
            bits=value.bits, group_size=value.group_size, mode=value.mode)
            for name, value in model.named_modules() if hasattr(value, "bits") and hasattr(value, "group_size")]
        report["tied_output"] = "original Model.__call__ uses embedding.as_linear; no independent lm_head"
        report["memory"].append(dict(phase="loaded", mlx=memory(mx), native=native_usage.usage()))
        admit_observed_memory(report["memory"][-1])
        endpoint();model_inventory(model_folder, pins)
        for case in cases:
            label = case["id"];attempt = dict(label=label, completed=False);report["attempts"].append(attempt)
            output = model(mx.array([case["tokens"]], dtype=mx.uint32), cache=None)
            attempt.update(observed_shape=list(output.shape), observed_dtype=str(output.dtype), evaluated=False)
            dtype = str(output.dtype).removeprefix("mlx.core.")
            width = {"bfloat16": 2, "float16": 2, "float32": 4}.get(dtype)
            if width is None or tuple(output.shape) != (1, case["rows"], 151936):
                raise ValueError("Original source output shape/dtype differs")
            mx.eval(output)  # Complete the admitted original graph before capture.
            attempt["evaluated"] = True
            record = dict(rows=case["rows"], columns=151936, source_dtype=dtype, complete=False,
                raw=dict(file=label + ".source-bits.bin", dtype="<u2" if width == 2 else "<u4",
                         layout="row-major [tokens,vocabulary]", bytes=0, intended_bytes=case["rows"] * 151936 * width),
                widened=dict(file=label + ".f32le.bin", dtype="<f4", layout="row-major [tokens,vocabulary]",
                             bytes=0, intended_bytes=case["rows"] * 151936 * 4),
                nonfinite_values=0, finite_widening_mismatches=0, tiles=[])
            report["records"][label] = record;raw_hash=hashlib.sha256();wide_hash=hashlib.sha256()
            with (folder / record["raw"]["file"]).open("xb") as raw_file, (folder / record["widened"]["file"]).open("xb") as wide_file:
                for start in range(0, case["rows"], 32):
                    end = min(start + 32, case["rows"]);tile = output[0, start:end]
                    bits = tile.view(mx.uint16 if width == 2 else mx.uint32);wide = tile.astype(mx.float32)
                    mx.eval(bits, wide)
                    original = np.array(bits, copy=True).astype("<u2" if width == 2 else "<u4", copy=False)
                    widened = np.array(wide, copy=True).astype("<f4", copy=False)
                    raw_body = memoryview(original).cast("B");wide_body = memoryview(widened).cast("B")
                    progress = dict(start=start, rows=end-start, raw_offset=record["raw"]["bytes"], widened_offset=record["widened"]["bytes"], complete=False)
                    record["tiles"].append(progress)
                    write_piece(raw_file, raw_body, record["raw"], raw_hash)
                    write_piece(wide_file, wide_body, record["widened"], wide_hash)
                    observed = verify_widening(raw_body, wide_body, dtype)
                    record["nonfinite_values"] += observed["nonfinite_values"]
                    record["finite_widening_mismatches"] += observed["finite_widening_mismatches"]
                    progress.update(complete=True, **observed)
                    del raw_body, wide_body, original, widened, bits, wide, tile
            del output;gc.collect();mx.clear_cache()
            record["memory"] = dict(mlx=memory(mx), native=native_usage.usage())
            if record["nonfinite_values"] or record["finite_widening_mismatches"]:
                raise ValueError("Original source nonfinite or inexact widening rejected after capture")
            if any(value["bytes"] != value["intended_bytes"] for value in (record["raw"], record["widened"])):
                raise ValueError("Original source capture coverage differs")
            admit_observed_memory(record["memory"])
            record["complete"] = True;attempt["completed"] = True
            with (folder / (label + ".json")).open("x") as file:
                json.dump(report, file, indent=2, allow_nan=False);file.write("\n")
            print(json.dumps(dict(captured=label, dtype=dtype, rows=case["rows"])), flush=True)
        model_inventory(model_folder, pins);endpoint();report["complete"] = True
    except BaseException as error:
        primary = error;report["primary_error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        output = tile = bits = wide = original = widened = raw_body = wide_body = model = None
        gc.collect()
        report["cleanup_errors"] = []
        try:
            mx.clear_cache()
            report["memory"].append(dict(phase="closed", mlx=memory(mx), native=native_usage.usage()))
        except BaseException as error:
            report["cleanup_errors"].append(type(error).__name__ + ": " + str(error))
            report["complete"] = False
            if primary is not None:
                primary.add_note("Source cleanup: " + type(error).__name__ + ": " + str(error))
            else:
                report["primary_error"] = "Source cleanup: " + type(error).__name__ + ": " + str(error)
        try:
            with (folder / "manifest.json").open("x") as file:
                json.dump(report, file, indent=2, allow_nan=False);file.write("\n")
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("Source partial manifest: " + type(error).__name__ + ": " + str(error))
    if report["cleanup_errors"]:
        raise ValueError("Original source cleanup failed")
    return report
