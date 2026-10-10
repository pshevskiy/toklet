"""Frozen CPU/Metal comparison runner; three warmups and ten completed samples."""
import argparse
import json
import platform
import subprocess
import threading
import tempfile
from pathlib import Path

import mlx_baseline as common
import prepare


WORKLOADS = {"short": 128, "medium": 512, "long": 1024, "boundary": 1920,
             "append": 512, "cached": 512, "pending": 512, "changed": 512}


def requests(name, arrays, contract):
    base = arrays[f"prompt_{WORKLOADS[name]}"]
    fixed = arrays["continuation_128"]
    setup = []
    request = {"action": "reset", "tokens": base, "continuation": fixed}
    cache = "empty"
    if name in ("append", "cached", "pending", "changed"):
        setup = [{"action": "prepare", "tokens": base}]
    if name == "append":
        request.update(action="append", tokens=arrays["append_64"])
        cache = "exact_prefix_append"
    elif name == "cached":
        request["action"] = "prompt"
        cache = "fully_cached"
    elif name == "pending":
        setup.append({"action": "stage", "tokens": [contract["continuation"]["pending_sample"]["pending_token"]]})
        request.update(action="continue", tokens=[])
        cache = "pending_sample"
    elif name == "changed":
        request.update(action="prompt", tokens=arrays["changed_prompt_512"])
        cache = "changed_prefix_replay"
    return setup, request, cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, help="optional macOS sample report during first warmup")
    parser.add_argument("--backend", choices=("cpu", "llama"), default="cpu")
    args = parser.parse_args()
    raw = args.output.with_suffix(args.output.suffix+".samples.jsonl")
    stderr_path = args.output.with_suffix(args.output.suffix+".stderr.txt")
    if (args.output.exists() or raw.exists() or stderr_path.exists() or args.profile and args.profile.exists()
            or args.profile and args.profile.resolve() in (args.output.resolve(), raw.resolve(), stderr_path.resolve())):
        raise ValueError("Refusing to overwrite evidence")
    prepare.verify(prepare.CONTRACT)
    sources = prepare.load_json(prepare.SOURCES)
    prepare.validate_source(sources["model"]["weights"])
    for asset in sources["assets"]:
        if asset["role"] == "model_metadata_or_tokenizer":
            prepare.validate_source(asset)
    manifest = prepare.load_json(prepare.CONTRACT/"manifest.json")
    arrays = {name: prepare.read_tokens(prepare.CONTRACT/item["file"])
              for name, item in manifest["arrays"].items() if name != "held_out_loss"}
    setup, request, cache = requests(args.workload, arrays, manifest)
    binary = prepare.ROOT/"target/release/examples/cpu_bench"
    model_path = prepare.ROOT/"models/Qwen2.5-0.5B-Instruct"
    backend_versions, backend_notes = {}, []
    compiler = common.command(["rustc", "--version"])
    build_flags = ["cargo build --locked --offline --release --example cpu_bench"]
    if args.backend == "llama":
        import llama_prepare
        build = llama_prepare.checked_build()
        converted = prepare.load_json(llama_prepare.MODEL/"manifest.json")
        binary = prepare.ROOT/build["binary"]
        model_path = llama_prepare.MODEL/"model.gguf"
        expected = sources["baselines"]["llama.cpp_commit"]
        if (build["llama_revision"] != expected or converted["llama_revision"] != expected
                or converted["source_sha256"] != sources["model"]["weights"]["sha256"]
                or prepare.digest(model_path) != converted["gguf_sha256"]
                or prepare.digest(binary) != build["binary_sha256"]):
            raise ValueError("Pinned converted model or comparison build identity differs")
        backend_versions = {"llama.cpp": expected}
        compiler = build["compiler"]
        build_flags = build["configure_command"] + ["cmake --build --target toklet-llama-bench -j 2"]
        backend_notes = [f"GGUF SHA256 {converted['gguf_sha256']}; exact FP32 payload verified independently; offline conversion {converted['conversion_seconds']} seconds",
                         f"Worker SHA256 {build['worker_sha256']}; CMake {build['cmake']}; Ninja {build['ninja']}",
                         "Metal offload requested for all layers; FP32 KV, Flash Attention disabled, one host thread; vendor diagnostics identify actual placement.",
                         "Conservative admission reserves two complete GGUF copies, FP32 KV and a 1 GiB workspace allowance; scratch is unavailable.",
                         "Native phys_footprint_peak is a kernel charge counter that omits clean mapped weight pages and may not fully represent shared GPU memory. Total mapped/shared physical peak is unavailable; do not interpret this small counter as the full working set or a memory advantage.",
                         "On this pinned M1 implementation, kernel_mul_mm_f32_f32 uses half matrix operands with float accumulation; FP32 storage does not imply equal arithmetic. Small-batch numerical probes do not qualify the 128-token GEMM path or held-out quality.",
                         "Model load starts after backend/device initialization. Initial embedded shader compilation and later lazy pipeline compilation are not separately isolated; three warmups precede samples."]
    engine_commit = common.command(["git", "rev-parse", "HEAD"])
    dirty = bool(common.command(["git", "status", "--porcelain"]))
    runner_sha, binary_sha = prepare.digest(Path(__file__)), prepare.digest(binary)
    common_sha = prepare.digest(Path(common.__file__))
    cmd = [str(binary), str(model_path)]
    power = common.command(["/usr/bin/pmset", "-g", "batt"])
    thermal = common.command(["/usr/bin/pmset", "-g", "therm"])
    # Vendor shader/model logs can exceed a pipe buffer before ready. Spool them
    # outside memory instead of blocking the child on an unread stderr pipe.
    errors = tempfile.TemporaryFile(mode="w+t")
    try:
        worker = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=errors, text=True)
    except BaseException:
        errors.close()
        raise
    # Large scalar matrix entries can take hours. Bound a malfunction and clean
    # up only our child; raw rows are checkpointed after each measured request.
    watchdog = threading.Timer(24*60*60, worker.kill)
    watchdog.start()
    profile = None
    try:
        def read():
            line = worker.stdout.readline()
            if not line:
                errors.seek(0)
                raise RuntimeError(errors.read())
            return json.loads(line)

        def send(value):
            worker.stdin.write(json.dumps(value)+"\n")
            worker.stdin.flush()
            return read()

        ready = read()
        if ready["event"] != "ready":
            raise ValueError("Worker did not admit model/session")
        # This checks full-output admission without dispatch and keeps the same
        # live empty session for subsequent warmups.
        rejection = send({"action": "reset", "tokens": arrays["prompt_1921"], "continuation": arrays["continuation_128"]})
        if rejection["event"] != "error":
            raise ValueError("Over-capacity request was accepted")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        samples = []
        for index in range(13):
            for prep in setup:
                if send(prep)["event"] != "prepared":
                    raise ValueError("Cache preparation failed")
            if index == 0 and args.profile:
                args.profile.parent.mkdir(parents=True, exist_ok=True)
                profile = subprocess.Popen(["/usr/bin/sample", str(worker.pid), "5", "1", "-file", str(args.profile)],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result = send(request)
            if result["event"] != "sample":
                raise ValueError(result)
            full = result["sample"]
            expected_prefill = {"append": 64, "cached": 0, "pending": 1}.get(args.workload, WORKLOADS[args.workload])
            if full["prefill_tokens"] != expected_prefill or full["pending_tokens"] != 1 or full["emitted_tokens"] != 128:
                raise ValueError("Benchmark state differs from contract")
            print(f"{args.workload}: {'warmup' if index < 3 else 'sample'} {index+1 if index < 3 else index-2}: {full['total_seconds']:.3f}s", flush=True)
            if index >= 3:
                sample = {key: full[key] for key in ("prefill_seconds", "ttft_seconds", "decode_seconds", "total_seconds", "token_event_seconds", "emitted_tokens")}
                sample.update(first_visible_text_seconds=None, process_peak_footprint_bytes=common.footprint_peak(worker.pid), gpu_peak_allocation_bytes=None)
                samples.append(sample)
                with raw.open("a") as stream:
                    stream.write(json.dumps({"sample": sample, "state": full})+"\n")
        if profile and profile.wait(timeout=20) != 0:
            raise ValueError("Native sampling failed")
        if args.workload == "boundary":
            finished = send({"action":"prepare", "tokens": request["tokens"]+request["continuation"]})
            if finished.get("evaluated_tokens") != 2048 or finished.get("pending_tokens") != 0:
                raise ValueError("Final admitted position did not commit")
            if send({"action":"stage", "tokens":[1]})["event"] != "error":
                raise ValueError("Token beyond completed capacity was accepted")
            backend_notes.append("After measured boundary requests, the final pending token was evaluated: all 2048 positions committed; a further token was rejected before mutation.")
        prompt_name = "append_64" if args.workload == "append" else "changed_prompt_512" if args.workload == "changed" else f"prompt_{WORKLOADS[args.workload]}"
        prompt_item = manifest["arrays"][prompt_name]
        peaks = [s["process_peak_footprint_bytes"] for s in samples if s["process_peak_footprint_bytes"] is not None]
        after_peak = common.footprint_peak(worker.pid)
        if after_peak is not None:
            peaks.append(after_peak)
        if ready["memory_admitted_bytes"] > 6*1024**3 or peaks and max(peaks) > 6*1024**3:
            raise ValueError("Working set exceeds admitted limit")
        prefills = {"append": 64, "cached": 0, "pending": 1}.get(args.workload, WORKLOADS[args.workload])
        result = {
            "schema_version": 1, "contract_id": manifest["contract_id"], "comparison_track": "controlled_engine", "runtime": "Toklet scalar CPU" if args.backend == "cpu" else "llama.cpp Metal",
            "identity": {"engine_commit": engine_commit, "working_tree_dirty": dirty,
                "baseline_versions": backend_versions, "compiler": compiler, "build_flags": build_flags,
                "model_revision": sources["model"]["revision"], "tokenizer_revision": sources["model"]["tokenizer_revision"], "model_sha256": sources["model"]["weights"]["sha256"],
                "tokenizer_sha256": next(a["sha256"] for a in sources["assets"] if a["id"] == "qwen.tokenizer.json"),
                "device": {"name": common.command(["/usr/sbin/sysctl", "-n", "hw.model"]), "chip": common.command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
                    "physical_memory_bytes": int(common.command(["/usr/sbin/sysctl", "-n", "hw.memsize"])), "os_version": common.command(["/usr/bin/sw_vers", "-productVersion"]),
                    "os_build": common.command(["/usr/bin/sw_vers", "-buildVersion"]), "power_mode": power+"\n"+common.command(["/usr/bin/pmset", "-g", "custom"]), "thermal_conditions": thermal}},
            "representation": {"weight_dtype": "FP32", "kv_dtype": "FP32", "packed_format_version": None, "group_size": None},
            "workload": {"name": args.workload+"_teacher_forced_128", "prompt_tokens": prompt_item["count"], "output_steps": 128, "context_capacity": 2048,
                "used_context_before_request": 0 if cache == "empty" else 513 if cache == "pending_sample" else 512,
                "prompt_sha256": prompt_item["sha256"], "continuation_sha256": manifest["arrays"]["continuation_128"]["sha256"],
                "stop_policy": "fixed_length_teacher_forced_ignore_eos", "cache_state": cache},
            "timing_scope": {"includes_tokenization": False, "includes_sampling": False, "includes_detokenization": False,
                "completion_boundary": "synchronous session evaluation through final projection; token timestamps before staging next fixed ID" if args.backend == "cpu" else "llama_decode then llama_synchronize and completed host logits; timestamp before staging next fixed ID", "first_visible_text_policy": "not measured; fixed token events only"},
            "setup": {"model_load_seconds": ready["load_seconds"], "conversion_seconds": ready["conversion_seconds"], "compilation_seconds": None, "tokenization_seconds": None, "warmup_repetitions": 3},
            "samples": samples, "summary": common.summarize(samples, prefills),
            "memory": {"engine_working_set_limit_bytes": 6*1024**3, "logical_weights_bytes": ready["logical_weights_bytes"], "allocated_kv_bytes": ready["allocated_kv_bytes"],
                "scratch_bytes": ready["scratch_bytes"], "peak_process_bytes": max(peaks) if peaks else None, "system_memory_pressure": None, "measurement_instrument": "own-child footprint auxiliary.phys_footprint_peak; logical session allocations"},
            "energy": {"joules": None, "instrument": None, "scope": None},
            "notes": [f"Runner SHA256 {runner_sha}; binary SHA256 {binary_sha}; common helper SHA256 {common_sha}; driver Python {platform.python_version()}",
                f"Admitted process estimate {ready['memory_admitted_bytes']} bytes; chunk {ready['chunk_size']}; fixed capacity 2048; capacity rejection {rejection['message']}",
                "Model load includes any nested payload conversion. Toklet conversion includes decoding/finite checks/storage but excludes reads, allocation and final model validation; llama.cpp GGUF conversion is a separate offline phase. Source hashing warms file cache.",
                "Each cached-state preparation is completed outside request timing. Repeated input edits/reset replay from zero in this implementation.",
                "Request timer excludes admission, bounded event allocation, JSON transport, output, sampling and tokenizer. Total includes staging final pending fixed ID; first distribution comes from prefill.",
                "Unavailable system memory pressure, energy, and visible text remain null. This record does not qualify held-out quality or imply a speed ranking.",
                "Thermal report at end: "+common.command(["/usr/bin/pmset", "-g", "therm"])]+backend_notes}
        from jsonschema import Draft202012Validator
        Draft202012Validator(prepare.load_json(prepare.BENCH/"result.schema.json")).validate(result)
        worker.stdin.close()
        if worker.wait(timeout=20) != 0:
            errors.seek(0)
            raise ValueError(errors.read())
        if args.backend == "llama":
            errors.seek(0)
            stderr_path.write_text(errors.read())
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
        print(json.dumps(result["summary"], sort_keys=True), flush=True)
    finally:
        watchdog.cancel()
        try:
            if worker.poll() is None:
                worker.kill()
            try:
                worker.wait(timeout=20)
            finally:
                for stream in (worker.stdin, worker.stdout, errors):
                    stream.close()
        finally:
            if profile and profile.poll() is None:
                profile.terminate()
                profile.wait(timeout=20)


if __name__ == "__main__":
    main()
