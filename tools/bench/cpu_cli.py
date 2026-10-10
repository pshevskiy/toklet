"""Bind opt-in application measurements to reviewed CLI functional evidence."""
from pathlib import Path

import cpu
import prepare

FORMAT = "toklet-cpu-tiled-cli-functional-v1"
FIXTURE_SHA = "691429f672aa89875e06475f48b5b691343f03c08e47fd204fa60fec2eef20d1"
CASE_NAMES = ["future_changed", "short", "single"]
MODES = {"frozen_scalar": "scalar", "default": "scalar", "scalar": "scalar", "tiled": "tiled"}


def compiled_sources():
    return {str(path.relative_to(prepare.ROOT)) for path in (prepare.ROOT / "src").rglob("*.rs")} | {
        "Cargo.toml", "Cargo.lock", "tests/cli.rs"}


def check_ready(ready, projection, qualified):
    actual = ready.get("cpu_projection", None if qualified else "scalar")
    if (actual != projection or ready.get("backend") != "cpu"
            or ready.get("context_capacity") != 2048):
        raise ValueError("CLI projection/backend/capacity differs from request")


def check_statistics(stats, projection, qualified):
    actual = stats.get("cpu_projection", None if qualified else "scalar")
    if actual != projection:
        raise ValueError("CLI statistics projection differs from request")


def qualified_cli(path):
    qualification = prepare.load_json(path)
    if (qualification.get("format") != FORMAT or qualification.get("completed") is not True):
        raise ValueError("CLI qualification is incomplete or unsupported")
    sources = qualification["source_sha256"]
    if set(sources) != compiled_sources():
        raise ValueError("Qualified CLI compiled source inventory differs")
    for name, sha in sources.items():
        if prepare.digest(prepare.ROOT / name) != sha:
            raise ValueError(f"Qualified CLI source differs: {name}")
    numerical = qualification["numerical_qualification"]
    numerical_path = prepare.ROOT / numerical["path"]
    if prepare.digest(numerical_path) != numerical["sha256"]:
        raise ValueError("CLI numerical qualification identity differs")
    _, compiler, numerical_record = cpu.qualified_cpu_worker(numerical_path)
    if qualification["compiler"] != compiler or qualification["rustflags"] != numerical_record["rustflags"]:
        raise ValueError("CLI/numerical compiler or flags identity differs")
    checks = qualification["checks"]
    required = [
        ["cargo", "fmt", "--check"],
        ["cargo", "clippy", "--locked", "--offline", "--all-targets", "--", "-D", "warnings"],
        ["cargo", "test", "--locked", "--offline"],
        qualification["build_commands"][1],
    ]
    if (not all(any(check["command"] == command for check in checks) for command in required)
            or any(check["exit_code"] != 0 or check.get("timed_out") is not False for check in checks)):
        raise ValueError("CLI build/check qualification differs")
    fixture_path = prepare.ROOT / "tests/fixtures/qwen-v2/manifest.json"
    if (qualification["fixture_manifest_sha256"] != FIXTURE_SHA
            or prepare.digest(fixture_path) != FIXTURE_SHA):
        raise ValueError("CLI frozen reference identity differs")
    fixture = prepare.load_json(fixture_path)
    runs = qualification["runs"]
    if set(runs) != set(MODES):
        raise ValueError("CLI functional mode inventory differs")
    for role, projection in MODES.items():
        run = runs[role]
        if run["passed"] is not True or run["exit_code"] != 0 or run["cpu_projection"] != projection:
            raise ValueError("CLI functional mode failed")
        check_ready(run["ready"], projection, role != "frozen_scalar")
        if (not 0 < run["physical_process_charge_peak_bytes"] <= 6 * 1024**3
                or not 0 < run["ready"]["memory_admitted_bytes"] <= 6 * 1024**3):
            raise ValueError("CLI functional memory qualification differs")
        if [case["name"] for case in run["cases"]] != CASE_NAMES:
            raise ValueError("CLI functional case inventory differs")
        for case in run["cases"]:
            expected = fixture["argmax"][case["name"]][-1]
            if (case["tokens"] != fixture["cases"][case["name"]]["tokens"]
                    or case["expected_first_token"] != expected
                    or len(case["token_ids"]) != 2 or case["token_ids"][0] != expected
                    or any(type(token) is not int or not 0 <= token < 151936 for token in case["token_ids"])):
                raise ValueError("CLI functional token/reference qualification differs")
            events = case["semantic_events"]
            if [event["event"] for event in events if event["event"] != "text"] != [
                    "token", "done", "token", "done", "reset"]:
                raise ValueError("CLI functional event sequence differs")
            tokens = [event for event in events if event["event"] == "token"]
            if [event["token_id"] for event in tokens] != case["token_ids"] or any(event["index"] != 0 for event in tokens):
                raise ValueError("CLI functional token events differ")
            generations = [event["generation"] for event in events if event["event"] == "done"]
            count = len(case["tokens"])
            for index, generation in enumerate(generations):
                expected_state = {"reason": {"kind": "output_limit"}, "prefill_tokens": count if index == 0 else 1,
                                  "reused_tokens": 0 if index == 0 else count, "sampled_tokens": 1, "emitted_tokens": 1,
                                  "history_tokens": count + index + 1, "evaluated_tokens": count + index, "pending_tokens": 1}
                if any(generation.get(name) != value for name, value in expected_state.items()):
                    raise ValueError("CLI functional pending/session state differs")
        if run["cases"] != runs["frozen_scalar"]["cases"]:
            raise ValueError("CLI functional session semantics differ from frozen scalar")
        if role in ("scalar", "tiled"):
            without_mode = lambda ready: {name: value for name, value in ready.items() if name != "cpu_projection"}
            if without_mode(run["ready"]) != without_mode(runs["default"]["ready"]):
                raise ValueError("CLI functional ownership/admission differs")
    binary = prepare.ROOT / qualification["binary_path"]
    if not binary.resolve().is_relative_to((prepare.ROOT / "artifacts").resolve()):
        raise ValueError("Qualified CLI binary must use isolated artifact storage")
    if prepare.digest(binary) != qualification["binary_sha256"]:
        raise ValueError("Qualified CLI binary identity differs")
    return binary, qualification
