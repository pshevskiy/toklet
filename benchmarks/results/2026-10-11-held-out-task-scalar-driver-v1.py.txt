"""Persistent collection of all frozen tasks; never executes generated code.

Each case starts with empty KV via a verified reset. An immediate hidden EOS is
valid. Watchdogs bound readiness, each case and shutdown independently; timer
cancellation is joined before the next phase, preventing stale group signals.
This collects quality responses, not comparative latency or a pass-rate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import cpu_cli
import mlx_baseline as common
import task_cases as tasks
import linux_code_isolation_probe as boundary
from linux_task_evidence import unique_object
from task_cases_probe import finite_float, invalid_constant, signal_owned_group

ROOT = tasks.prepare.ROOT
LIMIT = 6 * 1024**3


def child(binary, cases, projection, raw, errors):
    if projection not in ("scalar", "tiled") or not 0 < len(cases) <= 100:
        raise ValueError("Held-out task mode or count differs")
    for case in cases:
        tasks.validate_case(case)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Held-out task identities repeat")
    command = [str(binary), "--model", str(ROOT / "models/Qwen2.5-0.5B-Instruct"),
               "--interactive", "--json", "--context-capacity", "2048", "--prefill-chunk", "128",
               "--cpu-projection", projection]
    worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=errors, text=True, encoding="utf-8", start_new_session=True)
    killed = []
    watchdog_fired = []
    cleanup_errors = []

    def kill():
        if signal_owned_group(worker, cleanup_errors):
            killed.append(True)

    def timeout():
        watchdog_fired.append(True)
        kill()

    watchdog = None

    def arm(seconds):
        nonlocal watchdog
        if watchdog is not None:
            watchdog.cancel()
            watchdog.join()
        if watchdog_fired:
            raise ValueError("Held-out task watchdog already expired")
        watchdog = threading.Timer(seconds, timeout)
        watchdog.start()

    record = {"projection": projection, "command": command, "passed": False,
              "worker_pid": worker.pid, "cases": [],
              "watchdog_policy": {"ready_seconds": 1200, "case_seconds": 1800, "shutdown_seconds": 20}}
    stream_sha = hashlib.sha256()
    try:
        def read(case_id=None):
            line = worker.stdout.readline(65537)
            if not line:
                raise ValueError("Held-out task worker exited before response completion")
            raw.write(json.dumps({"projection": projection, "case_id": case_id, "line": line}) + "\n")
            raw.flush()
            stream_sha.update(line.encode())
            if len(line.encode("utf-8")) > 65536 or not line.endswith("\n"):
                raise ValueError("Held-out task transport line exceeds bound")
            event = json.loads(line, object_pairs_hook=unique_object, parse_float=finite_float, parse_constant=invalid_constant)
            if not isinstance(event, dict):
                raise ValueError("Held-out task event must be an object")
            return event

        def send(value):
            worker.stdin.write(json.dumps(value) + "\n")
            worker.stdin.flush()

        arm(1200)
        ready = read()
        if ready.get("event") != "ready":
            raise ValueError("Held-out task worker did not become ready")
        cpu_cli.check_ready(ready, projection, True)
        if type(ready.get("memory_admitted_bytes")) is not int or not 0 < ready["memory_admitted_bytes"] <= LIMIT:
            raise ValueError("Held-out task CLI admission exceeds the memory limit")
        record["ready"] = ready
        for case in cases:
            arm(1800)
            send(tasks.request(case))
            events = []
            for _ in range(case["max_output_tokens"] + 2):
                event = read(case["id"])
                events.append(event)
                if event.get("event") not in ("token", "text", "done"):
                    raise ValueError("Held-out task generation transport failed")
                if event.get("event") == "done":
                    break
            else:
                raise ValueError("Held-out task generation exceeds event bound")
            send({"action": "reset"})
            for _ in range(2):
                event = read(case["id"])
                events.append(event)
                if event.get("event") not in ("text", "reset"):
                    raise ValueError("Held-out task reset transport failed")
                if event.get("event") == "reset":
                    break
            else:
                raise ValueError("Held-out task reset exceeds event bound")
            response = tasks.collect_response(case, events, projection)
            cli_peak = common.footprint_peak(worker.pid)
            driver_peak = common.footprint_peak()
            if (type(cli_peak) is not int or type(driver_peak) is not int
                    or cli_peak <= 0 or driver_peak <= 0 or cli_peak + driver_peak > LIMIT):
                raise ValueError("Held-out task conservative joint physical charge exceeds the memory limit")
            record["cases"].append({"request": tasks.request(case), "semantic_events": events, "response": response,
                                    "cli_physical_peak_bytes": cli_peak, "driver_physical_peak_bytes": driver_peak,
                                    "conservative_joint_physical_peak_bytes": cli_peak + driver_peak})
        arm(20)
        worker.stdin.close()
        if worker.wait(timeout=20) != 0 or watchdog_fired:
            raise ValueError("Held-out task worker failed after response completion")
        trailing = worker.stdout.read(65537)
        if trailing:
            record["trailing_stdout"] = trailing
            stream_sha.update(trailing.encode())
            raw.write(json.dumps({"projection": projection, "case_id": None,
                                  "trailing": True, "line": trailing}) + "\n")
            raw.flush()
            raise ValueError("Held-out task worker emitted trailing output")
        record["passed"] = True
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if watchdog is not None:
            watchdog.cancel()
            watchdog.join()
        if watchdog_fired:
            record["passed"] = False
            record.setdefault("failure", {"type": "WatchdogExpired", "message": "Held-out task phase deadline expired"})
        # Failed leaders can leave descendants holding our pipes. The entire
        # group belongs to this invocation, even when its leader has exited.
        if not record["passed"] or worker.poll() is None:
            kill()
        try:
            worker.wait(timeout=20)
        except BaseException as error:
            cleanup_errors.append({"type": type(error).__name__, "message": str(error)})
            kill()
            try:
                worker.wait(timeout=20)
            except BaseException as error:
                cleanup_errors.append({"type": type(error).__name__, "message": str(error)})
        for stream in (worker.stdout, worker.stdin):
            try:
                if not stream.closed:
                    stream.close()
            except BaseException as error:
                cleanup_errors.append({"type": type(error).__name__, "message": str(error)})
        try:
            survivors = boundary.live_group_members(worker.pid)
            record["initial_surviving_group_members"] = survivors
            if survivors:
                record["passed"] = False
                record.setdefault("failure", {"type": "UnexpectedDescendants",
                                              "message": "Owned descendants survived normal leader completion"})
                kill()
                for attempt in range(20):
                    survivors = boundary.live_group_members(worker.pid)
                    if not survivors:
                        break
                    if attempt < 19:
                        time.sleep(0.1)
                if survivors:
                    cleanup_errors.append({"type": "SurvivingGroup", "message": str(survivors)})
            record["surviving_group_members"] = survivors
        except BaseException as error:
            cleanup_errors.append({"type": type(error).__name__, "message": str(error)})
        if cleanup_errors:
            record["passed"] = False
            record.setdefault("failure", {"type": "CleanupError", "message": "Held-out task process cleanup failed"})
        record.update(exit_code=worker.returncode, watchdog_killed=bool(watchdog_fired),
                      process_group_killed=bool(killed), cleanup_errors=cleanup_errors,
                      stdout_sha256=stream_sha.hexdigest())
    return record


def idle(path):
    proof = tasks.prepare.load_json(path)
    accepted = ("toklet-full-scalar-loss-natural-exit-idle-v1",
                "toklet-full-tiled-loss-natural-exit-idle-v1",
                "toklet-held-out-task-natural-exit-idle-v1")
    pids = proof.get("exited_owned_pids")
    if (proof.get("format") not in accepted or proof.get("all_owned_processes_absent") is not True
            or not isinstance(pids, list) or not pids or len(pids) > 8
            or any(type(pid) is not int or pid <= 1 for pid in pids) or len(set(pids)) != len(pids)):
        raise ValueError("Completed owned workload idle proof is required")
    query = subprocess.run(["ps", "-p", ",".join(map(str, pids)), "-o", "pid="],
                           capture_output=True, text=True)
    if query.returncode != 1 or query.stdout or query.stderr:
        raise ValueError("Previous owned workload has not exited")
    groups = proof.get("owned_group_members")
    if (not isinstance(groups, dict) or not groups
            or any(not isinstance(key, str) or not key.isdecimal() or int(key) not in pids or value != []
                   for key, value in groups.items())):
        raise ValueError("Completed owned group proof differs")
    for key in groups:
        if boundary.live_group_members(int(key)):
            raise ValueError("Previous owned workload group is not idle")
    original = tasks.prepare.load_json(ROOT / "benchmarks/results/2026-10-11-interrupted-workload-idle-v1.json")
    snapshot = boundary.idle(original)
    for section in proof["end_verified"].values():
        for name, expected in section.items():
            if tasks.prepare.digest(ROOT / name) != expected:
                raise ValueError("Protected workload identity changed: " + name)
    machines = json.loads(subprocess.check_output(
        ["/opt/podman/bin/podman", "machine", "list", "--format=json"], text=True))
    selected = [item for item in machines if item.get("Name") == "podman-machine-default"]
    if len(selected) != 1 or selected[0].get("Running") is not False or selected[0].get("Starting") is not False:
        raise ValueError("Task collection requires the qualification VM stopped")
    return {"idle_receipt_sha256": tasks.prepare.digest(path),
            "owned_absence_query": {"command": query.args, "exit_code": query.returncode,
                                    "stdout": query.stdout, "stderr": query.stderr},
            "held_original_snapshot": snapshot, "stopped_machine": selected[0]}


def run(args):
    output = args.output.resolve()
    evidence_paths = [output, Path(str(output) + ".native.jsonl"), Path(str(output) + ".stderr.txt")]
    if (not output.is_relative_to((ROOT / "benchmarks/raw").resolve())
            or any(path.exists() for path in evidence_paths)):
        raise ValueError("Fresh raw task collection paths are required")
    if common.command(["git", "status", "--porcelain"]):
        raise ValueError("Checked committed source is required before task collection")
    admission = idle(args.idle_proof)
    binary, qualification = cpu_cli.qualified_cli(args.cli_qualification)
    cases = tasks.load_cases()
    sources = tasks.prepare.load_json(tasks.prepare.SOURCES)
    assets = [sources["model"]["weights"]] + [item for item in sources["assets"]
                                               if item["role"] == "model_metadata_or_tokenizer"]
    for asset in assets:
        tasks.prepare.validate_source(asset)
    generation = tasks.prepare.load_json(ROOT / "models/Qwen2.5-0.5B-Instruct/generation_config.json")
    if generation.get("eos_token_id") != list(tasks.EOS_IDS):
        raise ValueError("Frozen model stop policy differs")
    import task_cases_probe
    import linux_task_evidence
    modules = (tasks, tasks.prepare, cpu_cli, cpu_cli.cpu, common,
               task_cases_probe, task_cases_probe.quality, boundary,
               linux_task_evidence, linux_task_evidence.child,
               linux_task_evidence.child.isolation,
               linux_task_evidence.child.isolation.authored)
    helpers = {str(Path(module.__file__).resolve().relative_to(ROOT)): tasks.prepare.digest(Path(module.__file__))
               for module in modules}
    helpers[str(Path(__file__).resolve().relative_to(ROOT))] = tasks.prepare.digest(Path(__file__))
    identities = {"cli_qualification_sha256": tasks.prepare.digest(args.cli_qualification),
                  "binary_sha256": tasks.prepare.digest(binary),
                  "contract_manifest_sha256": tasks.prepare.digest(tasks.prepare.CONTRACT / "manifest.json"),
                  "sources_manifest_sha256": tasks.prepare.digest(tasks.prepare.SOURCES),
                  "idle_receipt_sha256": tasks.prepare.digest(args.idle_proof)}
    record = {"format": "toklet-held-out-task-collection-v1", "completed": False,
              "scope": "All 100 frozen finalized quality responses only; no generated-code execution, task pass-rate, timing acceptance or milestone closure.",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "engine_revision": common.command(["git", "rev-parse", "HEAD"]),
              "working_tree_dirty": False, "projection": args.cpu_projection,
              "admission": admission, "binary_path": str(binary),
              "compiler": qualification["compiler"], "compiled_source_sha256": qualification["source_sha256"],
              "helper_sha256": helpers, "model": sources["model"],
              "validated_local_model_assets": {asset.get("path", asset.get("file")): asset["sha256"] for asset in assets},
              "task_payload_sha256": tasks.TASK_SHA,
              "case_inventory": [{"id": case["id"], "category": case["category"],
                                  "prompt_token_count": len(case["prompt_tokens"])} for case in cases], **identities}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with evidence_paths[1].open("x", encoding="utf-8") as raw, evidence_paths[2].open("x", encoding="utf-8") as errors:
            result = child(binary, cases, args.cpu_projection, raw, errors)
            record["run"] = result
            if not result["passed"]:
                raise ValueError("Held-out task collection failed: " + str(result.get("failure")))
        if len(result["cases"]) != 100:
            raise ValueError("Held-out finalized response coverage differs")
        for name, expected in helpers.items():
            if tasks.prepare.digest(ROOT / name) != expected:
                raise ValueError("Collection helper changed: " + name)
        current = {"cli_qualification_sha256": tasks.prepare.digest(args.cli_qualification),
                   "binary_sha256": tasks.prepare.digest(binary),
                   "contract_manifest_sha256": tasks.prepare.digest(tasks.prepare.CONTRACT / "manifest.json"),
                   "sources_manifest_sha256": tasks.prepare.digest(tasks.prepare.SOURCES),
                   "idle_receipt_sha256": tasks.prepare.digest(args.idle_proof)}
        if current != identities or tasks.load_cases() != cases:
            raise ValueError("Frozen collection admission identities changed")
        cpu_cli.qualified_cli(args.cli_qualification)
        for asset in assets:
            tasks.prepare.validate_source(asset)
        idle(args.idle_proof)
        record["completed"] = True
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)[:4096]}
        raise
    finally:
        record["completed_utc"] = datetime.now(timezone.utc).isoformat()
        record["native_raw_sha256"] = tasks.prepare.digest(evidence_paths[1]) if evidence_paths[1].exists() else None
        record["stderr_sha256"] = tasks.prepare.digest(evidence_paths[2]) if evidence_paths[2].exists() else None
        with output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-proof", type=Path, required=True)
    parser.add_argument("--cli-qualification", type=Path, required=True)
    parser.add_argument("--cpu-projection", choices=("scalar", "tiled"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
