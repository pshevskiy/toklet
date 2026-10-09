"""Explicit native BF16 MLX-LM baseline on the frozen token workload."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

import prepare


def command(args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


def footprint_peak(pid=None):
    """Read our own kernel-reported physical peak, outside timed work."""
    pid = os.getpid() if pid is None else pid
    with tempfile.TemporaryDirectory(prefix="toklet-footprint-") as directory:
        path = Path(directory) / "footprint.json"
        try:
            subprocess.run([
                "/usr/bin/footprint", "-p", str(pid), "-f", "bytes",
                "--noCategories", "-j", str(path),
            ], check=True, capture_output=True, timeout=20)
            result = prepare.load_json(path)
            if result.get("unit") != "byte" or result.get("bytes per unit") != 1:
                return None
            for process in result.get("processes", []):
                if process.get("pid") == pid:
                    return process.get("auxiliary", {}).get("phys_footprint_peak")
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    return None


def summarize(samples, prompt_count):
    if not samples:
        raise ValueError("No measured samples")
    rates, prefills = [], []
    for sample in samples:
        events = sample["token_event_seconds"]
        if len(events) != sample["emitted_tokens"] or not events:
            raise ValueError("Token event count mismatch")
        if any(b < a for a, b in zip(events, events[1:])) or events[0] < 0:
            raise ValueError("Token timestamps must be monotonic")
        if len(events) > 1 and events[-1] > events[0]:
            rates.append((len(events) - 1) / (events[-1] - events[0]))
        prefill_count = sample.get("prefill_tokens", prompt_count)
        if prefill_count > 0 and sample["prefill_seconds"] > 0:
            prefills.append(prefill_count / sample["prefill_seconds"])
    ttft = [sample["ttft_seconds"] for sample in samples]
    return {
        "ttft_median_seconds": statistics.median(ttft),
        "ttft_min_seconds": min(ttft), "ttft_max_seconds": max(ttft),
        "decode_tokens_per_second_median": statistics.median(rates) if rates else None,
        "request_median_seconds": statistics.median(sample["total_seconds"] for sample in samples),
        "prefill_tokens_per_second_median": statistics.median(prefills) if prefills else None,
    }


def measure(model, mx, make_cache, prompt, continuation):
    start = time.perf_counter()
    cache = make_cache(model)
    inputs = mx.array([prompt], dtype=mx.uint32)
    prefill_start = time.perf_counter()
    # Match pinned generate_step's default 2048-token prefill scheduling.
    # Our admitted prompts fit one chunk: evaluate only prefix KV, allowing
    # MLX's lazy graph to omit unused prefix vocabulary projections.
    if len(prompt) > 1:
        model(inputs[:, :-1], cache=cache)
        mx.eval([entry.state for entry in cache])
        mx.clear_cache()
    logits = model(inputs[:, -1:], cache=cache)[:, -1, :]
    mx.eval(logits, [entry.state for entry in cache])
    first = time.perf_counter()
    events = [first - start]
    # The first distribution comes from prefill. Each later distribution commits
    # the preceding fixed token; the final emitted token remains pending.
    for token in continuation[:-1]:
        logits = model(mx.array([[token]], dtype=mx.uint32), cache=cache)[:, -1, :]
        mx.eval(logits, [entry.state for entry in cache])
        events.append(time.perf_counter() - start)
    sample = {
        "prefill_seconds": first - prefill_start,
        "ttft_seconds": events[0], "decode_seconds": events[-1] - events[0],
        "total_seconds": events[-1], "first_visible_text_seconds": None,
        "token_event_seconds": events, "emitted_tokens": len(continuation),
        "process_peak_footprint_bytes": footprint_peak(),
        "gpu_peak_allocation_bytes": mx.get_peak_memory(),
    }
    metadata = {
        "kv_dtype": str(cache[0].keys.dtype),
        "allocated_kv_bytes": sum(entry.nbytes for entry in cache),
        "committed_tokens": cache[0].offset,
        "allocated_capacity": cache[0].keys.shape[2],
    }
    return sample, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-length", type=int, choices=(128, 512, 1024, 1920), default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a baseline record")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("This explicit baseline requires Apple Silicon macOS")
    prepare.verify(prepare.CONTRACT)
    sources = prepare.load_json(prepare.SOURCES)
    for package in ("mlx", "mlx-lm", "transformers", "torch", "tokenizers", "numpy"):
        if version(package) != sources["baselines"][package]:
            raise ValueError(f"Baseline version mismatch: {package}")
    prepare.validate_source(sources["model"]["weights"])
    for asset in sources["assets"]:
        if asset["role"] == "model_metadata_or_tokenizer":
            prepare.validate_source(asset)
    # Importing MLX can require GPU access; ordinary checks never enter here.
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from jsonschema import Draft202012Validator

    if mx.default_device().type != mx.gpu:
        raise ValueError("Baseline requires an accessible Metal GPU")
    manifest = prepare.load_json(prepare.CONTRACT / "manifest.json")
    prompt_array = manifest["arrays"][f"prompt_{args.prompt_length}"]
    continuation_array = manifest["arrays"]["continuation_128"]
    prompt = prepare.read_tokens(prepare.CONTRACT / prompt_array["file"])
    continuation = prepare.read_tokens(prepare.CONTRACT / continuation_array["file"])
    if len(prompt) + len(continuation) > 2048:
        raise ValueError("Request exceeds admitted capacity")
    initial_power = command(["/usr/bin/pmset", "-g", "batt"])
    initial_thermal = command(["/usr/bin/pmset", "-g", "therm"])
    load_start = time.perf_counter()
    model, _ = load(str(prepare.ROOT / "models/Qwen2.5-0.5B-Instruct"), lazy=False, trust_remote_code=False)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - load_start
    parameters = tree_flatten(model.parameters())
    if {str(array.dtype) for _, array in parameters} != {str(mx.bfloat16)}:
        raise ValueError("Native source baseline must retain BF16 weights")
    weights_bytes = sum(array.nbytes for _, array in parameters)
    for _ in range(3):
        measure(model, mx, make_prompt_cache, prompt, continuation)
    samples = []
    mx.reset_peak_memory()
    for index in range(10):
        sample, cache_info = measure(model, mx, make_prompt_cache, prompt, continuation)
        samples.append(sample)
        print(f"Measured request {index + 1}/10: {sample['total_seconds']:.3f}s", flush=True)
    if cache_info["kv_dtype"] != str(mx.bfloat16):
        raise ValueError("Native baseline KV dtype differs from BF16")
    peaks = [sample["process_peak_footprint_bytes"] for sample in samples if sample["process_peak_footprint_bytes"] is not None]
    power = command(["/usr/bin/pmset", "-g", "batt"])
    settings = command(["/usr/bin/pmset", "-g", "custom"])
    thermal = command(["/usr/bin/pmset", "-g", "therm"])
    result = {
        "schema_version": 1, "contract_id": manifest["contract_id"],
        "comparison_track": "application", "runtime": "MLX-LM",
        "identity": {
            "engine_commit": command(["git", "rev-parse", "HEAD"]),
            "working_tree_dirty": bool(command(["git", "status", "--porcelain"])),
            "baseline_versions": {name: version(name) for name in ("mlx", "mlx-lm", "transformers", "torch", "tokenizers", "numpy")},
            "compiler": f"Python {platform.python_version()}; prebuilt MLX Metal wheels",
            "build_flags": [], "model_revision": sources["model"]["revision"],
            "tokenizer_revision": sources["model"]["tokenizer_revision"],
            "model_sha256": sources["model"]["weights"]["sha256"],
            "tokenizer_sha256": next(asset["sha256"] for asset in sources["assets"] if asset["id"] == "qwen.tokenizer.json"),
            "device": {
                "name": mx.device_info()["device_name"],
                "chip": command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
                "physical_memory_bytes": int(command(["/usr/sbin/sysctl", "-n", "hw.memsize"])),
                "os_version": command(["/usr/bin/sw_vers", "-productVersion"]),
                "os_build": command(["/usr/bin/sw_vers", "-buildVersion"]),
                "power_mode": power + "\n" + settings,
                "thermal_conditions": thermal,
            },
        },
        "representation": {"weight_dtype": "BF16", "kv_dtype": "BF16", "packed_format_version": None, "group_size": None},
        "workload": {"name": f"prompt_{len(prompt)}_teacher_forced_128", "prompt_tokens": len(prompt), "output_steps": len(continuation), "context_capacity": 2048, "used_context_before_request": 0, "prompt_sha256": prompt_array["sha256"], "continuation_sha256": continuation_array["sha256"], "stop_policy": "fixed_length_teacher_forced_ignore_eos", "cache_state": "empty"},
        "timing_scope": {"includes_tokenization": False, "includes_sampling": False, "includes_detokenization": False, "completion_boundary": "mx.eval(prefix_cache_states), then mx.eval(last_prompt_token_logits, all_layer_cache_states); mx.eval(logits, all_layer_cache_states) after each decode token", "first_visible_text_policy": "not measured; teacher-forced token events only"},
        "setup": {"model_load_seconds": load_seconds, "conversion_seconds": None, "compilation_seconds": None, "tokenization_seconds": None, "warmup_repetitions": 3},
        "samples": samples, "summary": summarize(samples, len(prompt)),
        "memory": {"engine_working_set_limit_bytes": 6 * 1024**3, "logical_weights_bytes": weights_bytes, "allocated_kv_bytes": cache_info["allocated_kv_bytes"], "scratch_bytes": None, "peak_process_bytes": max(peaks) if peaks else None, "system_memory_pressure": None, "measurement_instrument": "own-process footprint auxiliary.phys_footprint_peak; MLX get_peak_memory allocations; logical persistent tensor sizes"},
        "energy": {"joules": None, "instrument": None, "scope": None},
        "notes": [
            f"Adapter SHA256 {prepare.digest(Path(__file__))}; Python dependency lock SHA256 {prepare.digest(prepare.ROOT / 'tools/bench/requirements.lock')}",
            f"Power at start: {initial_power}; thermal report at start: {initial_thermal}",
            "Native unquantized BF16 source checkpoint; quality scores are not measured in this baseline, so this record is not a qualified comparison ranking.",
            "Fresh MLX KV cache per warm request; capacity grows in native 256-token increments. Admitted limit is 2048; actual allocated capacity and bytes are reported separately.",
            f"Final committed tokens {cache_info['committed_tokens']}; actual allocated cache capacity {cache_info['allocated_capacity']}; final emitted token remains pending.",
            "Prefill timer excludes cache-object and input-array preparation; TTFT/total include both. No host wait occurs between layers.",
            "Prefill follows pinned MLX-LM generate_step's default 2048-token chunk scheduling: all-but-last prompt tokens update KV without evaluating vocabulary logits, followed by cache cleanup and the last prompt token's distribution. Sampling/log-softmax are excluded.",
            "Model load excludes Python/module startup. First-use kernel compilation can occur during warmup and is not isolated. Physical process peak covers loading and all requests; GPU allocation peak covers measured requests after reset.",
            "Pre-load source hash verification reads the weight file and can warm the filesystem cache. Reported model load is not a cold-storage measurement.",
            "System memory-pressure state, energy, and first-visible-text latency are unavailable. Process/GPU/logical memory figures are not added together because they can describe shared physical memory.",
        ],
    }
    schema = prepare.load_json(prepare.BENCH / "result.schema.json")
    Draft202012Validator(schema).validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
