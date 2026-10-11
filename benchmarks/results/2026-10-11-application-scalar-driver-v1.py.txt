"""Bounded greedy CLI application trace with separately attributed text timings."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import threading
import time

import mlx_baseline as common
import prepare
import cpu_cli


def trace_requests(manifest, max_tokens):
    path = prepare.CONTRACT / "application_trace.json"
    if prepare.digest(path) != manifest["authored_text_sha256"][path.name]:
        raise ValueError("Frozen application transcript changed")
    messages = prepare.load_json(path)
    if [m["role"] for m in messages] != ["system", "user", "assistant", "user", "assistant", "user"]:
        raise ValueError("Unexpected authored transcript roles")
    return [{"action": "generate", "messages": messages[:end], "max_new_tokens": max_tokens}
            for end in (2, 4, 6)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true", help="one two-token-per-turn functional check, not a benchmark")
    parser.add_argument("--cpu-projection", choices=("scalar", "tiled"), default="scalar")
    parser.add_argument("--cpu-qualification", type=Path,
                        help="completed functional qualification for an isolated CLI")
    args = parser.parse_args()
    if args.cpu_projection == "tiled" and not args.cpu_qualification:
        raise ValueError("Tiled application measurements require completed CLI qualification")
    raw = args.output.with_suffix(args.output.suffix + ".samples.jsonl")
    if args.output.exists() or raw.exists():
        raise ValueError("Refusing to overwrite application evidence")
    binary = prepare.ROOT / "target/release/toklet"
    qualification = None
    qualification_sha = None
    build_flags = ["cargo build --locked --offline --release"]
    if args.cpu_qualification:
        qualification_sha = prepare.digest(args.cpu_qualification)
        binary, qualification = cpu_cli.qualified_cli(args.cpu_qualification)
        if prepare.digest(args.cpu_qualification) != qualification_sha:
            raise ValueError("CLI qualification changed during admission")
        build_flags = qualification["build_commands"]
    prepare.verify(prepare.CONTRACT)
    sources = prepare.load_json(prepare.SOURCES)
    prepare.validate_source(sources["model"]["weights"])
    for asset in sources["assets"]:
        if asset["role"] == "model_metadata_or_tokenizer":
            prepare.validate_source(asset)
    manifest = prepare.load_json(prepare.CONTRACT / "manifest.json")
    max_tokens = 2 if args.probe else 128
    requests = trace_requests(manifest, max_tokens)
    identity = {"engine_commit": common.command(["git", "rev-parse", "HEAD"]),
                "working_tree_dirty": bool(common.command(["git", "status", "--porcelain"])),
                "binary_sha256": prepare.digest(binary), "driver_sha256": prepare.digest(Path(__file__)),
                "common_helper_sha256": prepare.digest(Path(common.__file__)),
                "preparation_helper_sha256": prepare.digest(Path(prepare.__file__)),
                "python": platform.python_version(), "compiler": common.command(["rustc", "--version"]),
                "build_flags": build_flags, "cpu_projection": args.cpu_projection,
                "cli_qualification_sha256": qualification_sha,
                "cli_qualification_helper_sha256": prepare.digest(Path(cpu_cli.__file__)),
                "cpu_worker_qualification_helper_sha256": prepare.digest(Path(cpu_cli.cpu.__file__)),
                "model": sources["model"],
                "tokenizer_sha256": next(a["sha256"] for a in sources["assets"] if a["id"] == "qwen.tokenizer.json"),
                "transcript_sha256": manifest["authored_text_sha256"]["application_trace.json"]}
    environment = {"hardware": common.command(["/usr/sbin/sysctl", "-n", "hw.model"]),
                   "chip": common.command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
                   "physical_memory_bytes": int(common.command(["/usr/sbin/sysctl", "-n", "hw.memsize"])),
                   "os_version": common.command(["/usr/bin/sw_vers", "-productVersion"]),
                   "os_build": common.command(["/usr/bin/sw_vers", "-buildVersion"]),
                   "power": common.command(["/usr/bin/pmset", "-g", "batt"]) + "\n" + common.command(["/usr/bin/pmset", "-g", "custom"]),
                   "thermal_start": common.command(["/usr/bin/pmset", "-g", "therm"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(mode="w+t") as errors:
        started = time.perf_counter()
        command = [str(binary), "--model", str(prepare.ROOT/"models/Qwen2.5-0.5B-Instruct"),
                   "--interactive", "--json", "--context-capacity", "2048", "--prefill-chunk", "128"]
        if qualification:
            command += ["--cpu-projection", args.cpu_projection]
        worker = subprocess.Popen(command,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True)
        watchdog = threading.Timer(24*60*60, worker.kill)
        watchdog.start()
        try:
            def read():
                line = worker.stdout.readline()
                received = time.perf_counter()
                if not line:
                    errors.seek(0)
                    raise ValueError(errors.read())
                return json.loads(line), received

            def send(value):
                worker.stdin.write(json.dumps(value) + "\n")
                worker.stdin.flush()

            ready, ready_at = read()
            if ready.get("event") != "ready":
                raise ValueError(ready)
            cpu_cli.check_ready(ready, args.cpu_projection, qualification is not None)
            setup_seconds = ready_at - started
            samples = []
            native_setup = None
            warmups, repetitions = (0, 1) if args.probe else (3, 10)
            for iteration in range(warmups + repetitions):
                turns = []
                scenario_start = time.perf_counter()
                for request in requests:
                    submitted = time.perf_counter()
                    send(request)
                    tokens, text = [], []
                    first_token = first_text = None
                    events = []
                    while True:
                        event, received = read()
                        if event["event"] in ("token", "text"):
                            if event["event"] == "token":
                                if event["index"] != len(tokens):
                                    raise ValueError("Token indices are not contiguous")
                                tokens.append(event["token_id"])
                                events.append(received - submitted)
                                if first_token is None:
                                    first_token = received - submitted
                            text.append(event["text"])
                            if event["text"] and first_text is None:
                                first_text = received - submitted
                        elif event["event"] == "done":
                            done = received - submitted
                            stats = event["statistics"]
                            cpu_cli.check_statistics(stats, args.cpu_projection, qualification is not None)
                            if native_setup is None:
                                native_setup = {key: stats[key] for key in ("load_seconds", "conversion_seconds", "tokenizer_load_seconds")}
                            break
                        else:
                            raise ValueError(event)
                    generation = stats["generation"]
                    if generation["emitted_tokens"] != len(tokens) or generation["reason"]["kind"] not in ("output_limit", "stop_token"):
                        raise ValueError("Unexpected generation state or stop reason")
                    for field in ("tokenization_seconds", "input_formatting_seconds", "detokenization_seconds", "publication_seconds"):
                        if not 0 <= stats[field] < float("inf"):
                            raise ValueError("Invalid attributed timing")
                    # A limited interactive response may retain an incomplete
                    # Unicode suffix for continue. Explicit reset finalizes it
                    # while it still belongs to this authored response. The
                    # next full transcript has scripted assistant content, so
                    # each turn intentionally uses an empty numerical cache.
                    send({"action": "reset"})
                    while True:
                        event, received = read()
                        if event["event"] == "text":
                            text.append(event["text"])
                            if event["text"] and first_text is None:
                                first_text = received - submitted
                        elif event["event"] == "reset":
                            finalized = received - submitted
                            break
                        else:
                            raise ValueError(event)
                    visible = "".join(text)
                    turns.append({"submission_to_first_token_seconds": first_token,
                                  "submission_to_first_visible_text_seconds": first_text,
                                  "submission_to_done_seconds": done,
                                  "submission_to_finalized_seconds": finalized,
                                  "received_token_event_seconds": events, "token_ids": tokens,
                                  "visible_text": visible, "visible_text_sha256": hashlib.sha256(visible.encode()).hexdigest(),
                                  "statistics": stats})
                sample = {"turns": turns, "scenario_seconds": time.perf_counter() - scenario_start}
                print(f"application: {'warmup' if iteration < warmups else 'sample'} {iteration+1 if iteration < warmups else iteration-warmups+1}: {sample['scenario_seconds']:.3f}s", flush=True)
                if iteration >= warmups:
                    sample["process_peak_footprint_bytes"] = common.footprint_peak(worker.pid)
                    samples.append(sample)
                    with raw.open("a") as stream:
                        stream.write(json.dumps(sample) + "\n")
            peaks = [s["process_peak_footprint_bytes"] for s in samples if s["process_peak_footprint_bytes"] is not None]
            if not peaks or max(peaks) > 6*1024**3 or ready["memory_admitted_bytes"] > 6*1024**3:
                raise ValueError("Unavailable or excessive process footprint/admission")
            worker.stdin.close()
            if worker.wait(timeout=20) != 0 or worker.stdout.read().strip():
                raise ValueError("Unexpected worker exit or final records")
            if qualification:
                if prepare.digest(args.cpu_qualification) != qualification_sha:
                    raise ValueError("CLI qualification changed during measurement")
                cpu_cli.qualified_cli(args.cpu_qualification)
            for path, key in ((binary, "binary_sha256"), (Path(__file__), "driver_sha256"),
                              (Path(common.__file__), "common_helper_sha256"),
                              (Path(prepare.__file__), "preparation_helper_sha256"),
                              (Path(cpu_cli.__file__), "cli_qualification_helper_sha256"),
                              (Path(cpu_cli.cpu.__file__), "cpu_worker_qualification_helper_sha256")):
                if prepare.digest(path) != identity[key]:
                    raise ValueError("Application binary/driver/helper changed during measurement")
            result = {"record_version": 1, "contract_id": manifest["contract_id"], "comparison_track": "application",
                      "qualification": "functional_probe" if args.probe else "three_warmups_ten_samples",
                      "runtime": f"Toklet {args.cpu_projection} CPU CLI", "identity": identity, "environment": environment,
                      "representation": {"weights": "FP32", "kv": "FP32", "context_capacity": 2048, "prefill_chunk": 128},
                      "policy": {"turns": 3, "max_output_tokens_per_turn": max_tokens, "stop": "pinned EOS or output limit",
                                 "greedy": True, "scripted_assistant_history": True, "cache": "empty per authored turn; resident model and reusable buffers"},
                      "setup": {"submission_to_ready_seconds": setup_seconds, **native_setup,
                                "warmup_repetitions": warmups, "storage_cache": "source hashing warmed files; no cold-storage claim"},
                      "samples": samples, "summary": {"scenario_median_seconds": statistics.median(s["scenario_seconds"] for s in samples),
                                                       "scenario_min_seconds": min(s["scenario_seconds"] for s in samples),
                                                       "scenario_max_seconds": max(s["scenario_seconds"] for s in samples)},
                      "memory": {"engine_working_set_limit_bytes": 6*1024**3, "admitted_bytes": ready["memory_admitted_bytes"],
                                 "logical_allocations": ready["logical_memory"],
                                 "peak_process_bytes": max(peaks), "instrument": "own-child auxiliary.phys_footprint_peak", "system_pressure": None},
                      "energy": None, "thermal_end": common.command(["/usr/bin/pmset", "-g", "therm"]),
                      "timing_scope": "Driver submission through complete JSON-line receipt (includes transport/dispatch); visible text is the first nonempty text payload. Numerical/sampling/text/publication components are native per-turn statistics. Finalization includes explicit reset/Unicode tail. Native totals overlap component times; do not add them.",
                      "notes": ["Tokenization excludes chat formatting; publication counts token/text serialization, writes and flushes, excluding the done statistics record.",
                                "Native model conversion is nested inside loading; load/tokenizer setup appears only on the first request, outside warm request timers.",
                                "Final reset text timing belongs to response finalization and is outside native generation's output counters.",
                                "The frozen assistant messages replace generated replies in later authored turns. This trace is latency evidence, not task-quality scoring or live tool execution."]}
            prepare.write_json(args.output, result)
        finally:
            watchdog.cancel()
            if worker.poll() is None:
                worker.kill()
            try:
                worker.wait(timeout=20)
            finally:
                worker.stdin.close()
                worker.stdout.close()


if __name__ == "__main__":
    main()
