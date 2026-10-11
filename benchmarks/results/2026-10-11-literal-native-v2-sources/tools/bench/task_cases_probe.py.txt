"""Qualify task response handling with authored short Qwen requests, not scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading

import cpu_cli
import mlx_baseline as common
import prepare
import quality
import task_cases as tasks

ROOT = prepare.ROOT
LIMIT = 6 * 1024**3
FIXTURE_SHA = cpu_cli.FIXTURE_SHA


def idle(proof):
    if proof.get("completed_workload") != "cached" or proof.get("measured_requests") != 10:
        raise ValueError("Task checks require the completed cached-workload idle proof")
    pids = [proof["held_orchestration_pid"], proof["defunct_transport_pid"], proof["exited_numerical_worker_pid"]]
    snapshot = subprocess.check_output(["ps", "-o", "pid,ppid,stat,time,command", "-p", ",".join(map(str,pids))], text=True)
    rows = {int(row[0]): row for line in snapshot.splitlines()[1:] if len(row := line.split(None,4)) == 5}
    if (pids[2] in rows or pids[0] not in rows or pids[1] not in rows
            or "T" not in rows[pids[0]][2] or not rows[pids[1]][2].startswith("Z")
            or int(rows[pids[1]][1]) != pids[0]):
        raise ValueError("Cached worker has not exited beneath the held orchestration")
    return snapshot


def probe_cases():
    path = ROOT / "tests/fixtures/qwen-v2/manifest.json"
    if prepare.digest(path) != FIXTURE_SHA:
        raise ValueError("Task probe frozen source identity differs")
    fixture = prepare.load_json(path)
    result = []
    for name, category in (("future_changed", "prose"), ("short", "continuation")):
        source = fixture["cases"][name]
        if source["offset"] != 0 or len(source["tokens"]) != 4:
            raise ValueError("Task probe source token/position differs")
        cap = tasks.CAPS[category]
        case = {"id": f"probe/{name}", "category": category, "prompt_text": "raw frozen numerical fixture IDs",
                "prompt_tokens": source["tokens"], "max_output_tokens": cap,
                "stop_policy": f"eos_or_{cap}", "scoring": tasks.SCORERS[category],
                "expected": "A" if category == "prose" else "authored_probe_only",
                "expected_first_token": fixture["argmax"][name][-1]}
        tasks.validate_case(case)
        result.append(case)
    return result


def finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Task probe JSON contains a nonfinite number")
    return value


def invalid_constant(text):
    raise ValueError(f"Task probe JSON contains invalid constant {text}")


def compare_modes(runs):
    if [run["projection"] for run in runs] != ["scalar", "tiled"] or any(not run["passed"] for run in runs):
        raise ValueError("Task probe mode coverage differs")
    left, right = [run["cases"] for run in runs]
    if len(left) != 2 or len(right) != 2:
        raise ValueError("Task probe case coverage differs")
    for a, b in zip(left, right):
        if (a["request"] != b["request"] or a["expected_first_token"] != b["expected_first_token"]
                or any(a["response"][key] != b["response"][key] for key in
                       ("id", "category", "visible_token_ids", "visible_text", "visible_text_sha256", "stop_reason"))):
            raise ValueError("Task probe scalar/tiled finalized response differs")


def signal_owned_group(worker, errors):
    """Signal our session group; distinguish Darwin's empty-group EPERM."""
    try:
        os.killpg(worker.pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as error:
        # Darwin can report EPERM for a group containing only dead members.
        # Do not suppress it for a live leader or any surviving descendant.
        if worker.poll() is not None:
            try:
                snapshot = subprocess.check_output(["ps", "-axo", "pgid=,stat="], text=True)
                members = [row.split() for row in snapshot.splitlines()]
                if not any(len(row) == 2 and int(row[0]) == worker.pid and not row[1].startswith("Z") for row in members):
                    return False
            except BaseException as probe_error:
                errors.append({"type": type(probe_error).__name__, "message": str(probe_error)})
        errors.append({"type": type(error).__name__, "message": str(error)})
        return False
    except BaseException as error:
        errors.append({"type": type(error).__name__, "message": str(error)})
        return False


def child(binary, cases, projection, raw, errors):
    command = [str(binary), "--model", str(ROOT / "models/Qwen2.5-0.5B-Instruct"),
               "--interactive", "--json", "--context-capacity", "2048", "--prefill-chunk", "128",
               "--cpu-projection", projection]
    worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=errors, text=True, start_new_session=True)
    killed = []
    watchdog_fired = []
    cleanup_errors = []

    def kill():
        if signal_owned_group(worker, cleanup_errors):
            killed.append(True)

    def timeout():
        watchdog_fired.append(True)
        kill()

    watchdog = threading.Timer(1200, timeout)
    watchdog.start()
    record = {"projection": projection, "command": command, "passed": False, "cases": []}
    stream_sha = hashlib.sha256()
    try:
        def read(case_id=None):
            line = worker.stdout.readline(65537)
            if not line:
                raise ValueError("Task probe worker exited before response completion")
            raw.write(json.dumps({"projection": projection, "case_id": case_id, "line": line}) + "\n")
            raw.flush()
            stream_sha.update(line.encode())
            if len(line.encode("utf-8")) > 65536 or not line.endswith("\n"):
                raise ValueError("Task probe transport line exceeds bound")
            event = json.loads(line, parse_float=finite_float, parse_constant=invalid_constant)
            if not isinstance(event, dict):
                raise ValueError("Task probe event must be an object")
            return event

        def send(value):
            worker.stdin.write(json.dumps(value) + "\n")
            worker.stdin.flush()

        ready = read()
        if ready.get("event") != "ready":
            raise ValueError("Task probe worker did not become ready")
        cpu_cli.check_ready(ready, projection, True)
        if type(ready.get("memory_admitted_bytes")) is not int or not 0 < ready["memory_admitted_bytes"] <= LIMIT:
            raise ValueError("Task probe CLI admission exceeds the memory limit")
        record["ready"] = ready
        for case in cases:
            send(tasks.request(case))
            events = []
            for _ in range(case["max_output_tokens"] + 2):
                event = read(case["id"])
                events.append(event)
                if event.get("event") not in ("token", "text", "done"):
                    raise ValueError("Task probe generation transport failed")
                if event.get("event") == "done":
                    break
            else:
                raise ValueError("Task probe generation exceeds event bound")
            send({"action": "reset"})
            for _ in range(2):
                event = read(case["id"])
                events.append(event)
                if event.get("event") not in ("text", "reset"):
                    raise ValueError("Task probe reset transport failed")
                if event.get("event") == "reset":
                    break
            else:
                raise ValueError("Task probe reset exceeds event bound")
            response = tasks.collect_response(case, events, projection)
            if not response["visible_token_ids"] or response["visible_token_ids"][0] != case["expected_first_token"]:
                raise ValueError("Task probe first token differs from independent numerical reference")
            cli_peak = common.footprint_peak(worker.pid)
            driver_peak = common.footprint_peak()
            if (type(cli_peak) is not int or type(driver_peak) is not int
                    or cli_peak <= 0 or driver_peak <= 0 or cli_peak + driver_peak > LIMIT):
                raise ValueError("Task probe conservative joint physical charge exceeds the memory limit")
            record["cases"].append({"request": tasks.request(case), "expected_first_token": case["expected_first_token"],
                                    "semantic_events": events, "response": response,
                                    "cli_physical_peak_bytes": cli_peak, "driver_physical_peak_bytes": driver_peak,
                                    "conservative_joint_physical_peak_bytes": cli_peak + driver_peak})
        worker.stdin.close()
        if worker.wait(timeout=20) != 0 or watchdog_fired:
            raise ValueError("Task probe worker failed after response completion")
        trailing = worker.stdout.read(65537)
        if trailing:
            record["trailing_stdout"] = trailing
            stream_sha.update(trailing.encode())
            raw.write(json.dumps({"projection": projection, "case_id": None,
                                  "trailing": True, "line": trailing}) + "\n")
            raw.flush()
            raise ValueError("Task probe worker emitted trailing output")
        record["passed"] = True
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        watchdog.cancel()
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
        if cleanup_errors:
            record["passed"] = False
            record.setdefault("failure", {"type": "CleanupError", "message": "Task probe process cleanup failed"})
        record.update(exit_code=worker.returncode, watchdog_killed=bool(watchdog_fired),
                      process_group_killed=bool(killed), cleanup_errors=cleanup_errors,
                      stdout_sha256=stream_sha.hexdigest())
    return record


def run(args):
    output = args.output.resolve()
    evidence = [output, Path(str(output)+".native.jsonl"), Path(str(output)+".stderr.txt")]
    if not output.is_relative_to((ROOT / "benchmarks/raw").resolve()) or any(path.exists() for path in evidence):
        raise ValueError("Task probe requires fresh raw evidence paths")
    proof = prepare.load_json(args.idle_proof)
    snapshot = idle(proof)
    record = {"format": "toklet-task-response-probe-v1", "completed": False, "checks": [], "runs": [],
              "started_utc": datetime.now(timezone.utc).isoformat(), "idle_snapshot": snapshot,
              "idle_proof_sha256": prepare.digest(args.idle_proof),
              "scope": "Frozen task admission and authored short CLI response handling only. No held-out model outputs, isolated HumanEval execution, quality scores, timing acceptance or milestone completion."}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification_sha = prepare.digest(args.cli_qualification)
        binary, qualification = cpu_cli.qualified_cli(args.cli_qualification)
        record.update(cli_qualification_sha256=qualification_sha, binary_sha256=prepare.digest(binary),
                      binary_path=str(binary), compiler=qualification["compiler"],
                      compiled_source_sha256=qualification["source_sha256"],
                      contract_manifest_sha256=prepare.digest(prepare.CONTRACT / "manifest.json"),
                      sources_manifest_sha256=prepare.digest(prepare.SOURCES),
                      engine_revision=common.command(["git", "rev-parse", "HEAD"]),
                      working_tree_dirty=bool(common.command(["git", "status", "--porcelain"])))
        prepared = tasks.load_cases()
        record["prepared_task_inventory"] = {"cases": len(prepared), "categories": tasks.CATEGORIES,
                                             "payload_sha256": tasks.TASK_SHA, "held_out_outputs_executed": False}
        sources = prepare.load_json(prepare.SOURCES)
        assets = [sources["model"]["weights"]] + [item for item in sources["assets"] if item["role"] == "model_metadata_or_tokenizer"]
        for asset in assets:
            prepare.validate_source(asset)
        record["validated_local_model_assets"] = {asset.get("path", asset.get("file")): asset["sha256"] for asset in assets}
        generation_config = prepare.load_json(ROOT / "models/Qwen2.5-0.5B-Instruct/generation_config.json")
        if generation_config.get("eos_token_id") != list(tasks.EOS_IDS):
            raise ValueError("Task probe model EOS policy differs")
        modules = (tasks, cpu_cli, cpu_cli.cpu, common, prepare, quality)
        helpers = {str(Path(module.__file__).relative_to(ROOT)): prepare.digest(Path(module.__file__)) for module in modules}
        helpers[str(Path(__file__).relative_to(ROOT))] = prepare.digest(Path(__file__))
        record.update(helper_sha256=helpers, model=sources["model"], fixture_manifest_sha256=FIXTURE_SHA)
        record["environment"] = {
            "hardware": common.command(["/usr/sbin/sysctl", "-n", "hw.model"]),
            "chip": common.command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
            "physical_memory_bytes": int(common.command(["/usr/sbin/sysctl", "-n", "hw.memsize"])),
            "os_version": common.command(["/usr/bin/sw_vers", "-productVersion"]),
            "os_build": common.command(["/usr/bin/sw_vers", "-buildVersion"]),
            "power_start": common.command(["/usr/bin/pmset", "-g", "batt"]) + "\n" + common.command(["/usr/bin/pmset", "-g", "custom"]),
            "thermal_start": common.command(["/usr/bin/pmset", "-g", "therm"]),
            "energy": None, "system_memory_pressure": None,
        }
        for command in (["cargo", "fmt", "--check"],
                        ["cargo", "clippy", "--locked", "--offline", "--all-targets", "--", "-D", "warnings"],
                        ["cargo", "test", "--locked", "--offline"],
                        [str(ROOT / ".venv/bin/python"), "-m", "unittest", "discover", "-s", "tools/bench", "-p", "test_*.py"]):
            quality.checked(command, record)
        cases = probe_cases()
        record["probe_cases"] = cases
        with evidence[1].open("x") as raw, evidence[2].open("x") as errors:
            for projection in ("scalar", "tiled"):
                result = child(binary, cases, projection, raw, errors)
                record["runs"].append(result)
                if not result["passed"]:
                    raise ValueError(f"Task response probe failed: {result.get('failure')}")
        compare_modes(record["runs"])
        record["scalar_tiled_semantics_match"] = True
        for name, sha in helpers.items():
            if prepare.digest(ROOT / name) != sha:
                raise ValueError("Task probe helper changed during execution")
        if prepare.digest(binary) != record["binary_sha256"] or prepare.digest(args.cli_qualification) != qualification_sha:
            raise ValueError("Task probe qualified executable/evidence changed")
        cpu_cli.qualified_cli(args.cli_qualification)
        if tasks.load_cases() != prepared:
            raise ValueError("Frozen task cases changed during the probe")
        if (prepare.digest(prepare.CONTRACT / "manifest.json") != record["contract_manifest_sha256"]
                or prepare.digest(prepare.SOURCES) != record["sources_manifest_sha256"]):
            raise ValueError("Task probe contract/source manifests changed")
        probe_cases()
        for asset in assets:
            prepare.validate_source(asset)
        record["completed"] = True
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["native_raw_sha256"] = prepare.digest(evidence[1]) if evidence[1].exists() else None
        record["stderr_sha256"] = prepare.digest(evidence[2]) if evidence[2].exists() else None
        with output.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-proof", type=Path, required=True)
    parser.add_argument("--cli-qualification", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
