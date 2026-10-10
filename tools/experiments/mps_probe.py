"""Bounded separate-process MPS dtype/shape evaluation with raw checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import metal_probe as common
import mps_oracle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_suffix(".swift")
SHADER = Path(__file__).with_name("mps_projection.metal")
TINY_HALF = ROOT / "tests/fixtures/fp16-v1"
TINY_SOURCE = ROOT / "tests/fixtures/tiny-v1"
sys.path.insert(0, str(ROOT / "tools/bench"))
import mlx_baseline as memory


def run_child(args, raw, stderr, timeout=180):
    """The child waits after completion, so its own footprint is read while idle."""
    events, killed = [], []
    with subprocess.Popen(args, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=stderr, start_new_session=True) as child:
        def kill():
            try:
                os.killpg(child.pid, signal.SIGKILL)
                killed.append(True)
            except ProcessLookupError:
                pass
        watchdog = threading.Timer(timeout, kill)
        watchdog.start()
        peak = None
        try:
            for line in child.stdout:
                raw.write(line); raw.flush()
                event = json.loads(line)
                events.append(event)
                if event.get("event") in ("complete", "failure"):
                    peak = memory.footprint_peak(child.pid)
                    child.stdin.write("quit\n"); child.stdin.flush()
                    break
            code = child.wait(timeout=20)
        except BaseException:
            kill(); child.wait()
            raise
        finally:
            watchdog.cancel()
    return {"exit_code": code, "watchdog_killed": bool(killed), "physical_process_charge_peak_bytes": peak,
            "events": events, "completed": code == 0 and bool(events) and events[-1].get("event") == "complete"}


def validate_result(result, mode):
    shapes = {case["name"]: case["shape"] for case in mps_oracle.verify()["cases"]}
    shapes["single_cast_bias_edge"] = [1, 2, 1]
    cases = result["cases"]
    if (len(cases) != len(shapes) or {case["name"] for case in cases} != set(shapes)
            or result["mode"] != mode or result["warmups"] != 3 or result["repetitions"] != 10):
        raise ValueError("Native case inventory/protocol differs")
    for case in cases:
        if "skipped" in case:
            if mode not in ("f16", "f16-stored") or case["name"] != "float_output_range":
                raise ValueError("Unexpected native skipped case")
            continue
        samples = case["samples"]
        if (case["shape"] != shapes[case["name"]] or len(samples) != 10
                or [row["sample_index"] for row in samples] != list(range(10))
                or not case["exact_outputs"] or not case["guards_untouched"]
                or not 0 < case["logical_buffer_bytes"] <= 6*1024**3):
            raise ValueError("Native numerical/state qualification failed")
        for row in samples:
            if not math.isfinite(row["completed_host_seconds"]) or row["completed_host_seconds"] <= 0:
                raise ValueError("Invalid completed-work sample")
        if mode == "f16-stored" and (not case["resident_fp16_storage"] or not case["stored_row_padding_untouched"] or case["cpu_bias_reset_required"]
                or case["stored_row_bytes"] < shapes[case["name"]][2]*2
                or any(row["excluded_cpu_bias_reset_seconds"] != 0 for row in samples)):
            raise ValueError("Resident FP16 projection protocol differs")
    expected = set(json.loads((TINY_HALF/"manifest.json").read_text())["cases"])
    tiny = result["tiny_projection_checks"]
    if (len(tiny) != len(expected) or {row["case"] for row in tiny} != expected
            or any(not math.isfinite(row["maximum_tolerance_fraction"]) or not 0 <= row["maximum_tolerance_fraction"] <= 1 or not row["guards_untouched"] for row in tiny)):
        raise ValueError("Tiny projection qualification failed")


def run(output, build, modes=("f32", "f16", "f16-to-f32")):
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= {"f32", "f16", "f16-to-f32", "f16-stored"}:
        raise ValueError("Invalid/duplicate MPS storage modes")
    paths = [output, *[output.with_suffix(output.suffix+f".{mode}.{suffix}") for mode in modes for suffix in ("jsonl", "stderr.txt")]]
    compile_stderr = output.with_suffix(output.suffix+".compile.stderr.txt")
    if build.exists() or compile_stderr.exists() or any(path.exists() for path in paths):
        raise ValueError("Refusing existing MPS evidence/build artifacts")
    mps_oracle.verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True)
    binary = build / "mps-probe"
    source_hash, fixture_hash, shader_hash = common.digest(SOURCE), common.digest(mps_oracle.FIXTURE), common.digest(SHADER)
    args = ["xcrun", "swiftc", "-O", "-warnings-as-errors", "-target", "arm64-apple-macosx15.0", str(SOURCE), "-o", str(binary)]
    record = {"format": "toklet-mps-primitive-evaluation-v1", "all_attempts_finished": False,
              "driver_sha256": common.digest(Path(__file__)), "helper_sha256": common.digest(Path(common.__file__)),
              "oracle_generator_sha256": common.digest(Path(mps_oracle.__file__)), "source_sha256": source_hash, "fixture_sha256": fixture_hash, "shader_sha256": shader_hash,
              "tiny_fp16_manifest_sha256": common.digest(TINY_HALF/"manifest.json"), "tiny_source_manifest_sha256": common.digest(TINY_SOURCE/"manifest.json"),
              "independent_bias_edge_reference_sha256": common.digest(ROOT/"tools/reference/test_qwen_fp16.py"),
              "compile_command": args, "engine_revision": common.command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
              "working_tree_status": common.command(["git", "-C", str(ROOT), "status", "--porcelain"]),
              "compiler": common.command(["xcrun", "swiftc", "--version"]), "sdk_version": common.command(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
              "power_start": common.power(), "modes": {}, "total_gpu_system_peak_bytes": None, "energy": None,
              "scope": "Exact synthetic/manual public matrix checks and isolated completed-work timings. Optional f16-stored includes resident GPU bias initialization and final cast in one command. Unsupported/failed attempts preserved; no trained-model, accumulation dtype, resident decoder or performance-release qualification."}
    try:
        with compile_stderr.open("x") as errors:
            common.bounded(args, errors, 300)
        binary_hash = common.digest(binary)
        record["binary_sha256"] = binary_hash
        for mode in modes:
            if common.digest(SOURCE) != source_hash or common.digest(SHADER) != shader_hash or common.digest(mps_oracle.FIXTURE) != fixture_hash or common.digest(binary) != binary_hash:
                raise ValueError("Compiled source/fixture/binary changed")
            raw_path = output.with_suffix(output.suffix+f".{mode}.jsonl")
            stderr_path = output.with_suffix(output.suffix+f".{mode}.stderr.txt")
            with raw_path.open("x") as raw, stderr_path.open("x") as errors:
                attempt = run_child([str(binary.resolve()), mode, str(mps_oracle.FIXTURE.resolve()), str(TINY_HALF), str(TINY_SOURCE), str(SHADER)], raw, errors)
            if attempt["completed"]:
                result = attempt["events"][-1]
                if (result["source_sha256"] != source_hash or result["shader_sha256"] != shader_hash or result["fixture_sha256"] != fixture_hash or result["binary_sha256"] != binary_hash
                        or result["mode"] != mode or result["warmups"] != 3 or result["repetitions"] != 10):
                    raise ValueError("Native result provenance/protocol differs")
                for key in ("tiny_fp16_manifest_sha256", "tiny_source_manifest_sha256"):
                    if result[key] != record[key]:
                        raise ValueError("Native tiny reference identity differs")
                validate_result(result, mode)
            peak = attempt["physical_process_charge_peak_bytes"]
            if peak is not None and peak > 6*1024**3:
                raise ValueError("Probe physical charge exceeds working-set ceiling")
            attempt.update(raw_sha256=common.digest(raw_path), stderr_sha256=common.digest(stderr_path))
            if attempt["completed"]:
                # The final native result retains every measured sample; raw
                # checkpoints preserve the duplicate progress events separately.
                attempt["result"] = attempt.pop("events")[-1]
            record["modes"][mode] = attempt
            print(f"MPS {mode}: {'complete' if attempt['completed'] else 'failed/unqualified'} (exit {attempt['exit_code']})", flush=True)
        record["all_attempts_finished"] = True
    except Exception as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["power_end"] = common.power()
        with output.open("x") as file:
            file.write(json.dumps(record, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("f32", "f16", "f16-to-f32", "f16-stored"), action="append",
                        help="select storage mode(s); default preserves the original three-mode experiment")
    args = parser.parse_args()
    run(args.output, args.build_dir, tuple(args.mode) if args.mode else ("f32", "f16", "f16-to-f32"))


if __name__ == "__main__":
    main()
