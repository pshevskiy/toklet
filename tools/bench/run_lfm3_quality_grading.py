"""Retain ownership of the LFM3 grading window through verified guest shutdown."""
import argparse
import json
import os
from pathlib import Path
import sys

import grading_support as support
import lfm3_grade_tasks as adapter
from lfm3_quality_inputs import digest, read_bound
import linux_code_isolation_probe as boundary
import paired_task_grading as runner
from run_llama3_quality_grading import acceptable, group_alive, read_inner_result

ROOT = adapter.ROOT


def observe():
    capture = boundary.capture([runner.PODMAN, "machine", "list", "--format=json"], 20, 262144)
    support.require(not capture.get("failure") and capture["exit_code"] == 0
                    and not capture["wall_timed_out"] and not capture["output_limit_exceeded"]
                    and capture["cleanup_errors"] == [] and capture["surviving_group_members"] == [],
                    "Outer machine query failed")
    return json.loads(boundary.output(capture))


def admit_pair(args):
    pair = tuple(args.roles)
    support.require(pair in adapter.PAIRS, "Unsupported LFM3 quality comparison")
    results = {}
    for profile, role, folder, expected in (
        ("baseline", pair[0], args.baseline, args.baseline_sha256),
        ("candidate", pair[1], args.candidate, args.candidate_sha256),
    ):
        receipt_path = folder / "result.json"
        receipt = json.loads(read_bound(receipt_path, expected))
        support.require(receipt["passed"] is True and receipt["role"] == role
                        and receipt["first"] == 0 and receipt["end"] == receipt["tasks"] == 100
                        and receipt["input_reconciliation_sha256"] == adapter.INPUTS_SHA256,
                        "Complete independently decoded role required before starting the guest")
        results[profile] = dict(receipt_sha256=expected, collection_sha256=receipt["collection_sha256"],
                                decoded_tasks_sha256=receipt["decoded_sha256"])
    return results


def run(args):
    identities = admit_pair(args)
    output = args.output.resolve()
    support.require(output.is_relative_to(ROOT / "artifacts"), "Fresh outer artifact folder required")
    output.mkdir()
    command = [sys.executable, "-B", str(ROOT / "tools/bench/lfm3_grade_tasks.py"),
               "--inputs", str(args.inputs), "--baseline", str(args.baseline),
               "--baseline-sha256", args.baseline_sha256, "--candidate", str(args.candidate),
               "--candidate-sha256", args.candidate_sha256, "--output", str(output / "inner"),
               "--roles", *args.roles]
    pins = {str(path): digest(path) for path in [
        Path(__file__), Path(adapter.__file__), Path(runner.__file__), Path(support.__file__),
        args.baseline / "result.json", args.candidate / "result.json",
        args.inputs / "reconciliation.json",
    ]}
    prior = runner.checked_json(runner.PRIOR, runner.PRIOR_SHA, 16 * 1024**2)
    client = Path(runner.PODMAN)
    client_pin = dict(bytes=client.stat().st_size, sha256=prior["podman_sha256"])
    if args.qualified_cpu_witness:
        import linux_process_cpu_retry as witness

        command.append("--qualified-cpu-witness")
        pins[str(witness.PROOF)] = witness.PROOF_SHA
    environment = dict(os.environ, TZ="Europe/Moscow")
    plan = dict(command=command, pins=pins, client_pin=client_pin, timezone=environment["TZ"],
                decoded_inputs=identities, pair=args.roles)
    support.write_json(output / "execution-plan.json", plan)
    record = dict(completed=False, pair=args.roles, generated_code_runs_on_host=False)

    def guard():
        support.verify_file(client, client_pin)
        support.require(all(digest(Path(path)) == pin for path, pin in pins.items()), "Grading window input changed")

    try:
        guard()
        window = support.supervise_window(command, output / "child.stdout", output / "child.stderr",
                                          environment, observe, group_alive)
        record["window"] = window
        guard()
        pid = window.get("pid")
        support.require(type(pid) is int and pid > 1 and not group_alive(pid), "Grading group remains live")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("Grading process remains live")
        inner, pin = read_inner_result(output / "inner/result.json")
        support.require(acceptable(window, inner) and inner["pair"] == args.roles
                        and inner["decoded_inputs"] == identities, "Grading decision or restoration differs")
        if args.qualified_cpu_witness:
            policy = inner["cpu_budget_policy"]
            support.require(policy["qualification_sha256"] == witness.PROOF_SHA
                            and policy["threshold_ns"] == 4_000_000_000
                            and policy["original_attempts_preserved"] is True,
                            "Qualified CPU witness policy differs")
        record.update(completed=True, decision=inner["decision"], inner_result_sha256=pin["sha256"],
                      independent_group_absence_verified=True)
    except (OSError, KeyError, ValueError, TypeError) as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error)[:4096])
    support.write_json(output / "result.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    for name in ("baseline-sha256", "candidate-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--roles", nargs=2, required=True)
    parser.add_argument("--qualified-cpu-witness", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({key: result.get(key) for key in ("completed", "pair", "decision", "failure")}))
    raise SystemExit(0 if result["completed"] else 1)
