"""Saved-evidence checks for a literal evaluator; no runtime adoption.

This checks an owned execution receipt, not adversarial grader integrity.
Generated Python shares the interpreter with its source tests and can tamper
with that interpreter. The host lifecycle must establish ownership, runtime
identity and cleanup independently before providing this record.
"""
import base64
import hashlib
import json
import math
import signal

import linux_task_child as child
import linux_code_isolation_probe as boundary

INPUT_BYTES = child.INPUT_BYTES


def serialized_payload(payload):
    child.admit_payload(payload)
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(data) > INPUT_BYTES:
        raise ValueError("Serialized literal program exceeds its input bound")
    # This is an input-pipe payload, never an assumed unbounded OCI argument.
    return data


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Evaluator JSON repeats a field")
        value[key] = item
    return value


def reject_constant(value):
    raise ValueError("Evaluator JSON contains a nonfinite constant")


def finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Evaluator JSON contains a nonfinite number")
    return parsed


def decode_event(line):
    value = json.loads(line, object_pairs_hook=unique_object,
                       parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("Evaluator event must be an object")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def validate_ready(ready, binding, runtime, python_version):
    keys = {"event", "profile", "binding", "policy", "limits", "runtime", "preloaded_modules",
            "environment_keys", "removed_environment_keys", "python", "isolated_flag",
            "no_site_flag", "optimize_flag", "dont_write_bytecode"}
    limits = {name: [value, value] for name, (_, value) in child.isolation.limits().items()}
    if (set(ready) != keys or ready["event"] != "isolated" or ready["profile"] != child.PROFILE
            or canonical(ready["binding"]) != canonical(binding)
            or canonical(ready["policy"]) != canonical(child.isolation.POLICY)
            or canonical(ready["limits"]) != canonical(limits)
            or canonical(ready["runtime"]) != canonical(runtime)
            or ready["python"] != python_version or ready["preloaded_modules"] != list(child.PRELOADS)
            or ready["environment_keys"] != sorted(boundary.ENVIRONMENT)
            or ready["removed_environment_keys"] not in ([], ["HOME"], ["HOSTNAME"], ["HOME", "HOSTNAME"])
            or any(type(ready[key]) is not int or ready[key] != expected for key, expected in
                   (("isolated_flag", 1), ("no_site_flag", 1), ("optimize_flag", 0)))
            or ready["dont_write_bytecode"] is not True):
        raise ValueError("Literal evaluator admission differs")


def verified_capture(record):
    captured = {}
    for channel in ("stdout", "stderr"):
        data = base64.b64decode(record[channel + "_base64"], validate=True)
        if (type(record[channel + "_captured_bytes"]) is not int
                or len(data) != record[channel + "_captured_bytes"]
                or hashlib.sha256(data).hexdigest() != record[channel + "_captured_sha256"]):
            raise ValueError("Literal evaluator raw capture identity differs")
        captured[channel] = data
    if sum(map(len, captured.values())) > child.isolation.POLICY["output_bytes"]:
        raise ValueError("Literal evaluator raw capture exceeds its bound")
    return captured


def classify(record, payload, runtime, python_version):
    """Infrastructure errors raise; a resolved model/resource outcome is binary.

    Exit137 alone remains ambiguous. Only recorded independent wall/output
    watchdogs or actual OOM state resolve resource termination. Exit zero
    without the complete source-test result is premature exit, not pass.
    A nonzero exit without a terminal result remains conservatively unresolved:
    this receipt cannot distinguish candidate exit from interpreter failure.
    """
    binding = child.admit_payload(payload)
    data = serialized_payload(payload)
    if (record.get("input_completed") is not True
            or type(record.get("input_bytes")) is not int or record["input_bytes"] != len(data)
            or type(record.get("input_sent_bytes")) is not int or record["input_sent_bytes"] != len(data)
            or record.get("input_sha256") != hashlib.sha256(data).hexdigest()):
        raise ValueError("Literal evaluator input pipe identity or completion differs")
    state = record.get("container_state")
    flags = ("wall_timed_out", "output_limit_exceeded", "process_group_killed", "container_forced_kill")
    if (record.get("failure") or record.get("cleanup_errors") != []
            or record.get("surviving_group_members") != []
            or record.get("container_cleanup_errors") != [] or record.get("container_removed") is not True
            or any(type(record.get(key)) is not bool for key in flags)
            or not isinstance(state, dict) or state.get("Running") is not False
            or type(state.get("OOMKilled")) is not bool or state.get("Error") not in (None, "")
            or type(state.get("ExitCode")) is not int or not 0 <= state["ExitCode"] <= 255
            or type(record.get("container_exit_code")) is not int
            or state["ExitCode"] != record["container_exit_code"]
            or type(record.get("transport_exit_code")) is not int):
        raise ValueError("Literal evaluator termination or cleanup is unresolved")
    code = record["container_exit_code"]
    captured = verified_capture(record)
    if record["transport_exit_code"] != code:
        owned_kill = record["transport_exit_code"] == -signal.SIGKILL and record["process_group_killed"]
        killed_container = code == 137 and (record["wall_timed_out"] or record["output_limit_exceeded"])
        # The remote child may finish before the bounded attach client is
        # killed. Resolve this race only as saturated output resource failure;
        # admission and successful owned cleanup remain mandatory below.
        finished_output = (code == 0 and record["output_limit_exceeded"]
                           and not record["wall_timed_out"] and not state["OOMKilled"]
                           and not record["container_forced_kill"]
                           and sum(map(len, captured.values())) == child.isolation.POLICY["output_bytes"])
        if not (owned_kill and (killed_container or finished_output)):
            raise ValueError("Literal evaluator transport and container exits differ")
    first, separator, rest = captured["stdout"].partition(b"\n")
    if not separator:
        raise ValueError("Literal evaluator lacks completed admission evidence")
    validate_ready(decode_event(first), binding, runtime, python_version)
    known_resource = record["wall_timed_out"] or record["output_limit_exceeded"] or state["OOMKilled"]
    if known_resource:
        # Watchdogs may truncate a frame. Neither partial JSON nor a candidate
        # output string can turn this admitted resource stop into a pass.
        if record["output_limit_exceeded"] and sum(map(len, captured.values())) != child.isolation.POLICY["output_bytes"]:
            raise ValueError("Literal evaluator output watchdog lacks a saturated capture")
        if code == 0 and not record["output_limit_exceeded"]:
            raise ValueError("Literal evaluator resource stop contradicts clean exit")
        return {"outcome": "resource_failure", "passed": False, "binding": binding,
                "reason": "output_limit" if record["output_limit_exceeded"] else "wall_limit" if record["wall_timed_out"] else "container_oom"}
    if record["process_group_killed"] or record["container_forced_kill"] or code == 137:
        raise ValueError("Literal evaluator termination is ambiguous")
    if captured["stderr"]:
        raise ValueError("Literal evaluator emitted unframed controller stderr")
    terminal = None
    output_frames = 0
    for line in rest.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError("Literal evaluator event is incomplete")
        event = decode_event(line)
        if terminal is not None:
            raise ValueError("Literal evaluator emitted events after completion")
        if event.get("event") == "program_output":
            if (set(event) != {"event", "channel", "text"}
                    or event["channel"] not in ("stdout", "stderr")
                    or not isinstance(event["text"], str) or not 0 < len(event["text"]) <= 1024):
                raise ValueError("Literal evaluator output frame differs")
            output_frames += 1
        elif event.get("event") == "program_result":
            if (set(event) != {"event", "binding", "outcome", "normal_completion", "exception"}
                    or canonical(event["binding"]) != canonical(binding)
                    or type(event["normal_completion"]) is not bool
                    or event["outcome"] not in ("passed", "model_failure", "resource_failure", "infrastructure_failure")):
                raise ValueError("Literal evaluator terminal identity differs")
            passed = event["outcome"] == "passed"
            if (event["normal_completion"] != passed
                    or passed and event["exception"] is not None
                    or not passed and (not isinstance(event["exception"], str) or not 0 < len(event["exception"]) <= 128)):
                raise ValueError("Literal evaluator terminal outcome contradicts completion")
            terminal = event
        else:
            raise ValueError("Literal evaluator protocol contains an unexpected event")
    if terminal is None:
        if code != 0:
            raise ValueError("Literal evaluator exited without a resolved completion")
        return {"outcome": "model_failure", "passed": False, "binding": binding,
                "reason": "premature_exit", "output_frames": output_frames}
    if code != 0:
        raise ValueError("Literal evaluator terminal result contradicts container exit")
    if terminal["outcome"] == "infrastructure_failure":
        raise ValueError("Literal evaluator compatibility failure leaves case unresolved")
    return {"outcome": terminal["outcome"], "passed": terminal["outcome"] == "passed",
            "binding": binding, "reason": terminal["exception"], "output_frames": output_frames}
