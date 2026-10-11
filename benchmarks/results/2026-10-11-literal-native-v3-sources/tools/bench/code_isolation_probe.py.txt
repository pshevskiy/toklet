"""Qualify fixed authored Python isolation probes in a verified idle window."""
import argparse
import base64
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time

import code_isolation_child as child
import mlx_baseline as common
import prepare
import quality
from task_cases_probe import signal_owned_group

ROOT = prepare.ROOT
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"}
SDK_HEADER = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sandbox.h")
SDK_EXPORTS = SDK_HEADER.parent.parent / "lib/system/libsystem_sandbox.tbd"


def idle(proof):
    if proof.get("completed_workload") != "pending" or proof.get("measured_requests") != 10:
        raise ValueError("Isolation checks require a complete pending-workload idle proof")
    pids = [proof["held_orchestration_pid"], proof["defunct_transport_pid"], proof["exited_numerical_worker_pid"]]
    snapshot = subprocess.check_output(["ps", "-o", "pid,ppid,stat,time,command", "-p", ",".join(map(str, pids))], text=True)
    rows = {int(row[0]): row for line in snapshot.splitlines()[1:] if len(row := line.split(None, 4)) == 5}
    if (pids[2] in rows or pids[0] not in rows or pids[1] not in rows
            or "T" not in rows[pids[0]][2] or not rows[pids[1]][2].startswith("Z")
            or int(rows[pids[1]][1]) != pids[0]):
        raise ValueError("Pending worker has not exited beneath held orchestration")
    return snapshot


def live_group_members(pid):
    snapshot = subprocess.check_output(["ps", "-axo", "pid=,pgid=,stat="], text=True)
    return [int(row[0]) for line in snapshot.splitlines() if len(row := line.split()) == 3
            and int(row[1]) == pid and not row[2].startswith("Z")]


def execute(mode, read_path, write_path, directory):
    """Bound two pipes and wall time; reap only our new session group."""
    if mode not in child.MODES:
        raise ValueError("Only fixed authored isolation probes are admitted")
    command = [sys.executable, "-I", "-S", "-B", str(ROOT / "tools/bench/code_isolation_child.py"),
               mode, str(read_path), str(write_path)]
    started = time.monotonic()
    selector = selectors.DefaultSelector()
    try:
        worker = subprocess.Popen(command, cwd=directory, env=ENVIRONMENT, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                                  close_fds=True)
    except BaseException:
        selector.close()
        raise
    deadline = started + child.POLICY["wall_seconds"]
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    cleanup_errors = []
    record = {"mode": mode, "command": command, "wall_timed_out": False, "output_limit_exceeded": False}
    killed = False
    try:
        for stream, name in ((worker.stdout, "stdout"), (worker.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                record["wall_timed_out"] = True
                break
            for key, _ in sorted(selector.select(min(0.05, remaining)), key=lambda item: item[0].data != "stdout"):
                payload = os.read(key.fileobj.fileno(), 4096)
                if not payload:
                    selector.unregister(key.fileobj)
                    continue
                available = child.POLICY["output_bytes"] - sum(len(value) for value in captured.values())
                captured[key.data].extend(payload[:available])
                if len(payload) > available:
                    record["output_limit_exceeded"] = True
                    break
            if record["output_limit_exceeded"]:
                break
        if record["wall_timed_out"] or record["output_limit_exceeded"]:
            killed = signal_owned_group(worker, cleanup_errors)
        try:
            worker.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            record["wall_timed_out"] = True
            killed = signal_owned_group(worker, cleanup_errors) or killed
            worker.wait(timeout=20)
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        selector.close()
        # Even an exited leader may have left descendants. No process outside
        # this newly owned session group is signaled.
        killed = signal_owned_group(worker, cleanup_errors) or killed
        try:
            worker.wait(timeout=20)
        except BaseException as error:
            cleanup_errors.append({"type": type(error).__name__, "message": str(error)})
        for stream in (worker.stdout, worker.stderr):
            try:
                stream.close()
            except BaseException as error:
                cleanup_errors.append({"type": type(error).__name__, "message": str(error)})
        try:
            surviving = live_group_members(worker.pid)
        except BaseException as error:
            surviving = None
            cleanup_errors.append({"type": type(error).__name__, "message": str(error)})
        record.update(exit_code=worker.returncode, process_group_killed=killed,
                      cleanup_errors=cleanup_errors, surviving_group_members=surviving,
                      elapsed_seconds=time.monotonic() - started)
        for name, payload in captured.items():
            record[name + "_base64"] = base64.b64encode(payload).decode("ascii")
            record[name + "_captured_bytes"] = len(payload)
            record[name + "_captured_sha256"] = hashlib.sha256(payload).hexdigest()
    return record


def check_record(record):
    """A qualified boundary is distinct from a passing source-test result."""
    if (record.get("failure") or record["cleanup_errors"] or record["surviving_group_members"] != []
            or type(record["exit_code"]) is not int
            or type(record["elapsed_seconds"]) not in (int, float)
            or not math.isfinite(record["elapsed_seconds"]) or record["elapsed_seconds"] < 0
            or any(type(record[key]) is not bool for key in
                   ("wall_timed_out", "output_limit_exceeded", "process_group_killed"))):
        raise ValueError("Authored isolation transport/cleanup failed")
    output = base64.b64decode(record["stdout_base64"], validate=True)
    errors = base64.b64decode(record["stderr_base64"], validate=True)
    for name, payload in (("stdout", output), ("stderr", errors)):
        if (type(record[name + "_captured_bytes"]) is not int
                or len(payload) != record[name + "_captured_bytes"]
                or hashlib.sha256(payload).hexdigest() != record[name + "_captured_sha256"]):
            raise ValueError("Authored isolation raw capture identity differs")
    if len(output) + len(errors) > child.POLICY["output_bytes"]:
        raise ValueError("Authored isolation capture exceeds its bound")
    lines = output.splitlines()
    if not lines:
        raise ValueError("Authored isolation lacks active-boundary evidence")
    ready = json.loads(lines[0])
    if not isinstance(ready, dict):
        raise ValueError("Authored isolation admission must be an object")
    mode = record["mode"]
    expected_limits = {name: [value, value] for name, (_, value) in child.limits().items()}
    observed_limits = ready.get("limits")
    exact_limits = (isinstance(observed_limits, dict) and set(observed_limits) == set(expected_limits)
                    and all(isinstance(pair, list) and len(pair) == 2
                            and all(type(value) is int for value in pair)
                            and pair == expected_limits[name] for name, pair in observed_limits.items()))
    if (ready.get("event") != "isolated" or ready.get("mode") != mode
            or json.dumps(ready.get("policy"), sort_keys=True, allow_nan=False) != json.dumps(child.POLICY, sort_keys=True, allow_nan=False)
            or not exact_limits
            or not isinstance(ready.get("profile"), str) or not ready["profile"]
            or type(ready.get("isolated_flag")) is not int or ready["isolated_flag"] != 1
            or type(ready.get("no_site_flag")) is not int or ready["no_site_flag"] != 1
            or type(ready.get("optimize_flag")) is not int or ready["optimize_flag"] != 0
            or ready.get("dont_write_bytecode") is not True
            or ready.get("environment_keys") != sorted(ENVIRONMENT)):
        raise ValueError("Authored isolation admission differs")
    if mode in ("stdout_flood", "stderr_flood"):
        if (not record["output_limit_exceeded"] or record["wall_timed_out"]
                or not record["process_group_killed"] or record["exit_code"] != -signal.SIGKILL
                or len(output) + len(errors) != child.POLICY["output_bytes"]):
            raise ValueError("Authored output bound was not enforced")
    elif mode == "wall_timeout":
        if (not record["wall_timed_out"] or record["output_limit_exceeded"]
                or not record["process_group_killed"] or record["exit_code"] != -signal.SIGKILL):
            raise ValueError("Authored wall bound was not enforced")
    elif mode == "cpu_timeout":
        if (record["wall_timed_out"] or record["output_limit_exceeded"]
                or record["process_group_killed"]
                or record["exit_code"] not in (-signal.SIGKILL, -signal.SIGXCPU)
                or len(lines) != 1 or errors):
            raise ValueError("Authored CPU bound was not enforced before wall deadline")
    elif mode == "early_exit":
        if record["exit_code"] != 0 or len(lines) != 1 or errors:
            raise ValueError("Authored early-exit evidence differs")
    else:
        if (record["exit_code"] != 0 or record["wall_timed_out"] or record["output_limit_exceeded"]
                or len(lines) != 2 or errors):
            raise ValueError("Authored isolation lacks complete result evidence")
        final = json.loads(lines[1])
        if not isinstance(final, dict):
            raise ValueError("Authored isolation result event must be an object")
        if final.get("event") != "probe_result" or final.get("mode") != mode:
            raise ValueError("Authored isolation result event differs")
        result = final.get("result", {})
        if not isinstance(result, dict):
            raise ValueError("Authored isolation result must be an object")
        if mode == "benign":
            expected = result == {"check_returned": True} and result["check_returned"] is True
        elif mode == "assertion_failure":
            expected = (result == {"check_returned": False, "exception": "AssertionError"}
                        and result["check_returned"] is False)
        elif mode in ("file_read", "file_write", "network", "fork", "exec"):
            expected = (result.get("denied") is True and type(result.get("errno")) is int
                        and result["errno"] in (errno.EPERM, errno.EACCES)
                        and result.get("exception") == "PermissionError")
        elif mode == "data_limit":
            expected = (result == {"bounded": True, "exception": "MemoryError", "requested_bytes": 512 * 1024**2}
                        and result["bounded"] is True and type(result["requested_bytes"]) is int)
        elif mode == "address_space_limit":
            expected = (result == {"bounded": True, "exception": "OSError", "errno": errno.ENOMEM,
                                  "requested_bytes": 2 * 1024**3}
                        and result["bounded"] is True and type(result["requested_bytes"]) is int
                        and type(result["errno"]) is int)
        else:
            expected = False
        if not expected:
            raise ValueError("Authored isolation capability/resource probe failed")
    return ready


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "benchmarks/raw").resolve()) or output.exists():
        raise ValueError("Isolation qualification requires a fresh raw output")
    proof = prepare.load_json(args.idle_proof)
    snapshot = idle(proof)
    record = {"format": "toklet-authored-code-isolation-probe-v1", "completed": False,
              "scope": "Authored native capability probes only. No dataset/model outputs or generated code; no quality score or executor adoption.",
              "policy": child.POLICY, "modes": list(child.MODES), "runs": [], "checks": [],
              "started_utc": datetime.now(timezone.utc).isoformat(), "idle_snapshot": snapshot,
              "idle_proof_sha256": prepare.digest(args.idle_proof)}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        sources = [Path(child.__file__), Path(__file__), ROOT / "tools/bench/task_cases_probe.py",
                   ROOT / "tools/bench/quality.py", ROOT / "tools/bench/prepare.py", Path(common.__file__),
                   ROOT / "tools/bench/test_code_isolation_probe.py"]
        identities = {str(path.relative_to(ROOT)): prepare.digest(path) for path in sources}
        python = Path(sys.executable).resolve()
        record.update(helper_sha256=identities, python_executable=str(python), python_sha256=prepare.digest(python),
                      sdk_header_path=str(SDK_HEADER), sdk_header_sha256=prepare.digest(SDK_HEADER),
                      sdk_exports_path=str(SDK_EXPORTS), sdk_exports_sha256=prepare.digest(SDK_EXPORTS),
                      python_version=sys.version,
                      engine_revision=common.command(["git", "rev-parse", "HEAD"]),
                      working_tree_dirty=bool(common.command(["git", "status", "--porcelain"])),
                      hardware=common.command(["/usr/sbin/sysctl", "-n", "hw.model"]),
                      chip=common.command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
                      physical_memory_bytes=int(common.command(["/usr/sbin/sysctl", "-n", "hw.memsize"])),
                      os_version=common.command(["/usr/bin/sw_vers", "-productVersion"]),
                      os_build=common.command(["/usr/bin/sw_vers", "-buildVersion"]))
        for command in (["cargo", "fmt", "--check"],
                        ["cargo", "clippy", "--locked", "--offline", "--all-targets", "--", "-D", "warnings"],
                        ["cargo", "test", "--locked", "--offline"],
                        [str(ROOT / ".venv/bin/python"), "-m", "unittest", "discover", "-s", "tools/bench", "-p", "test_*.py"]):
            quality.checked(command, record)
        with tempfile.TemporaryDirectory(prefix="authored-isolation-", dir=ROOT / "artifacts") as temporary:
            directory = Path(temporary)
            read_path = directory / "owned-outside-read.txt"
            write_path = directory / "owned-outside-write.txt"
            sentinel = b"OWNED ISOLATION SENTINEL; NO PRIVATE DATA"
            read_path.write_bytes(sentinel)
            working = directory / "case"
            working.mkdir()
            for mode in child.MODES:
                item = execute(mode, read_path, write_path, working)
                record["runs"].append(item)
                active = check_record(item)
                if active["python"] != sys.version:
                    raise ValueError("Authored child interpreter version differs")
                if read_path.read_bytes() != sentinel or write_path.exists():
                    raise ValueError("Authored outside-file sentinel changed")
                item["expectation_passed"] = True
        for name, sha in identities.items():
            if prepare.digest(ROOT / name) != sha:
                raise ValueError("Authored isolation helper changed")
        if (prepare.digest(python) != record["python_sha256"]
                or prepare.digest(SDK_HEADER) != record["sdk_header_sha256"]
                or prepare.digest(SDK_EXPORTS) != record["sdk_exports_sha256"]):
            raise ValueError("Authored isolation runtime/SDK identity changed")
        record["completed"] = True
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with output.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-proof", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
