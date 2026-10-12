"""Qualify pinned Rust Metal bindings in a fresh isolated package/process."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time

import metal_probe as common

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_suffix(".rs")
SHADER = Path(__file__).with_suffix(".metal")
MANIFEST = Path(__file__).with_suffix(".Cargo.toml")
LOCK = Path(__file__).with_suffix(".lock")
RMS_SOURCE = Path(__file__).with_name("rust_metal_rms.rs")
RMS_SHADER = Path(__file__).with_name("rust_metal_rms.metal")
EMBEDDING_SOURCE = Path(__file__).with_name("rust_metal_embedding.rs")
EMBEDDING_SHADER = Path(__file__).with_name("rust_metal_embedding.metal")
RESIDUAL_SOURCE = Path(__file__).with_name("rust_metal_residual.rs")
RESIDUAL_SHADER = Path(__file__).with_name("rust_metal_residual.metal")
SILU_SOURCE = Path(__file__).with_name("rust_metal_silu.rs")
SILU_SHADER = Path(__file__).with_name("rust_metal_silu.metal")
GATING_SOURCE = Path(__file__).with_name("rust_metal_gating.rs")
GATING_SHADER = Path(__file__).with_name("rust_metal_gating.metal")
PROJECTION_SOURCE = Path(__file__).with_name("rust_metal_projection.rs")
PROJECTION_SHADER = Path(__file__).with_name("rust_metal_projection.metal")
MLP_SOURCE = Path(__file__).with_name("rust_metal_mlp.rs")
ROTARY_SOURCE = Path(__file__).with_name("rust_metal_rotary.rs")
ROTARY_SHADER = Path(__file__).with_name("rust_metal_rotary.metal")
CACHE_SOURCE = Path(__file__).with_name("rust_metal_cache.rs")
CACHE_SHADER = Path(__file__).with_name("rust_metal_cache.metal")
ATTENTION_SOURCE = Path(__file__).with_name("rust_metal_attention.rs")
ATTENTION_SHADER = Path(__file__).with_name("rust_metal_attention.metal")
ATTENTION_GRAPH_SOURCE = Path(__file__).with_name("rust_metal_attention_graph.rs")
LIMIT = 6 * 1024**3
ALLOWANCE = 64 * 1024**2
INPUT_GUARD, OUTPUT_GUARD = 0x4642FEDC, 0x46A35CE1
SHAPES = (1, 17, 257)
sys.path.insert(0, str(ROOT/"tools/bench"))
import mlx_baseline as memory


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def validate_result(event):
    """Independent dyadic FP32 add oracle; exact values and all guard bits."""
    if event.get("event") != "idle_result" or event.get("passed") is not True:
        raise ValueError("Native Rust Metal check failed")
    result = event["result"]
    metadata = result["metadata"]
    if (metadata["unified_memory"] is not True or metadata["language_version"] != "2.3"
            or metadata["math_mode"] != "safe" or metadata["math_functions"] != "precise"
            or metadata["minimum_macos_major"] != 15
            or not 0 < metadata["maximum_admitted_bytes"] <= LIMIT
            or metadata["execution_width"] <= 0 or metadata["pipeline_max_threads"] <= 0):
        raise ValueError("Unqualified device/compiler/admission policy")
    device_threads = metadata["device_max_threads"]
    if len(device_threads) != 3 or any(width <= 0 for width in device_threads):
        raise ValueError("Unqualified device dispatch limits")
    cases = result["cases"]
    if [case["count"] for case in cases] != list(SHAPES):
        raise ValueError("Native case inventory differs")
    for case in cases:
        count, width = case["count"], case["threadgroup_width"]
        if (case["guards"] != 7 or not 0 < width <= min(metadata["execution_width"], metadata["pipeline_max_threads"],device_threads[0])
                or case["threadgroups"] != (count+width-1)//width
                or case["shared_buffer_bytes"] != (count+7)*4*2
                or case["host_input_and_readback_bytes"] != (count+7)*4*3
                or case["admitted_bytes"] != (count+7)*4*5+ALLOWANCE
                or case["allocations_reused_across_repetitions"] is not True):
            raise ValueError("Native buffer/launch/accounting differs")
        expected_input = [float_bits((i % 29)*0.25-3.5) for i in range(count)] + [INPUT_GUARD]*7
        expected_output = [float_bits((i % 29)*0.25-2.5) for i in range(count)] + [OUTPUT_GUARD]*7
        rows = case["rows"]
        if len(rows) != 13:
            raise ValueError("Native warmup/sample count differs")
        for index, row in enumerate(rows):
            if (row["phase"] != ("warmup" if index < 3 else "sample")
                    or row["repetition"] != (index+1 if index < 3 else index-2)
                    or row["command_completed"] is not True
                    or row["input_bits"] != expected_input or row["output_bits"] != expected_output
                    or not math.isfinite(row["encode_commit_wait_seconds"])
                    or row["encode_commit_wait_seconds"] <= 0):
                raise ValueError("Native completed-work/value/guard check differs")
    if metadata["maximum_admitted_bytes"] != max(case["admitted_bytes"] for case in cases):
        raise ValueError("Native maximum admission differs")


def run_child(args, raw, stderr, timeout=60):
    """Read own-child memory only after its explicit completed-work handshake."""
    with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=stderr, text=True, start_new_session=True) as child:
        killed = []
        def kill():
            try:
                os.killpg(child.pid, signal.SIGKILL)
                killed.append(True)
            except ProcessLookupError:
                pass
        watchdog = threading.Timer(timeout, kill)
        watchdog.start()
        try:
            line = child.stdout.readline(1024*1024+1)
            raw.write(line); raw.flush()
            if not line or len(line) > 1024*1024 or not line.endswith("\n"):
                raise ValueError("Missing or overlong native completion record")
            event = json.loads(line)
            if event.get("event") != "idle_result":
                raise ValueError("Native process did not enter completed-work idle state")
            peak = memory.footprint_peak(child.pid)
            child.stdin.write("done\n"); child.stdin.flush()
            child.stdin.close()
            trailing = child.stdout.read(1024*1024+1)
            if trailing:
                raw.write(trailing); raw.flush()
                raise ValueError("Unexpected native trailing protocol output")
            code = child.wait(timeout=20)
            return {"exit_code":code,"watchdog_killed":bool(killed),
                    "physical_process_charge_peak_bytes":peak,"event":event}
        except BaseException:
            kill(); child.wait(timeout=20)
            raise
        finally:
            watchdog.cancel()


def build_commands(build):
    """Keep every Cargo output inside this fresh qualification package."""
    env_command = ["env", "MACOSX_DEPLOYMENT_TARGET=15.0",
                   f"CARGO_TARGET_DIR={(build/'target').resolve()}"]
    manifest = str((build/"Cargo.toml").resolve())
    return [env_command+["cargo",subcommand,"--manifest-path",manifest]+options
            for subcommand,options in
            (("fmt",["--check"]),
             ("clippy",["--locked","--offline","--all-targets","--","-D","warnings"]),
             ("test",["--locked","--offline"]),
             ("build",["--locked","--offline","--release","--target","aarch64-apple-darwin"]))]


def check_paths(output,build):
    stderr_path = output.with_suffix(output.suffix+".stderr.txt")
    native_raw = output.with_suffix(output.suffix+".native.jsonl")
    if any(build.resolve() == path.resolve() or build.resolve() in path.resolve().parents
           for path in (output,stderr_path,native_raw)):
        raise ValueError("Evidence must be outside the build directory")
    if any(path.exists() for path in (output, stderr_path, native_raw, build)):
        raise ValueError("Refusing to overwrite qualification evidence or build artifacts")
    return stderr_path,native_raw


def validate_run(mode,result,validator):
    """A custom validator returns False only for a qualified negative fixture."""
    event=result["event"]
    if validator is not None:
        passed=validator(event) is not False # Compaction happens after validation.
        if result["exit_code"]!=(0 if passed else 1) or event.get("passed") is not passed:
            raise ValueError("Primitive validation/exit policy differs")
    elif mode=="normal":
        validate_result(event)
        if result["exit_code"]!=0:raise ValueError("Passing native probe did not exit cleanly")
    elif (result["exit_code"]!=1 or event.get("passed") is not False
            or event["error"]["stage"]!=("shader_compilation" if mode=="invalid_shader" else "admission")):
        raise ValueError("Expected native failure was not preserved")


def run(output, build, extra_runs=(), extra_context=None):
    stderr_path,native_raw = check_paths(output,build)
    output.parent.mkdir(parents=True, exist_ok=True)
    (build/"src").mkdir(parents=True)
    copies = {SOURCE:build/"src/main.rs", SHADER:build/"src/rust_metal_probe.metal",
              MANIFEST:build/"Cargo.toml", LOCK:build/"Cargo.lock",
              RMS_SOURCE:build/"src/rust_metal_rms.rs", RMS_SHADER:build/"src/rust_metal_rms.metal",
              EMBEDDING_SOURCE:build/"src/rust_metal_embedding.rs", EMBEDDING_SHADER:build/"src/rust_metal_embedding.metal",
              RESIDUAL_SOURCE:build/"src/rust_metal_residual.rs", RESIDUAL_SHADER:build/"src/rust_metal_residual.metal",
              SILU_SOURCE:build/"src/rust_metal_silu.rs", SILU_SHADER:build/"src/rust_metal_silu.metal",
              GATING_SOURCE:build/"src/rust_metal_gating.rs", GATING_SHADER:build/"src/rust_metal_gating.metal",
              PROJECTION_SOURCE:build/"src/rust_metal_projection.rs", PROJECTION_SHADER:build/"src/rust_metal_projection.metal", MLP_SOURCE:build/"src/rust_metal_mlp.rs", ROTARY_SOURCE:build/"src/rust_metal_rotary.rs", ROTARY_SHADER:build/"src/rust_metal_rotary.metal", CACHE_SOURCE:build/"src/rust_metal_cache.rs", CACHE_SHADER:build/"src/rust_metal_cache.metal", ATTENTION_SOURCE:build/"src/rust_metal_attention.rs", ATTENTION_SHADER:build/"src/rust_metal_attention.metal", ATTENTION_GRAPH_SOURCE:build/"src/rust_metal_attention_graph.rs"}
    source_hashes = {str(path.relative_to(ROOT)):common.digest(path) for path in copies}
    for source, target in copies.items():
        shutil.copyfile(source,target)
    checks = build_commands(build)
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_rms.rs").resolve())])
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_embedding.rs").resolve())])
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_residual.rs").resolve())])
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_silu.rs").resolve())])
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_gating.rs").resolve())])
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_projection.rs").resolve())])
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_mlp.rs").resolve())])
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_rotary.rs").resolve())])
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_cache.rs").resolve())])
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_attention.rs").resolve())])
    checks.insert(0,["rustfmt","--edition","2024","--check",str((build/"src/rust_metal_attention_graph.rs").resolve())])
    record = {"format":"toklet-rust-metal-binding-probe-v1","completed":False,
              "engine_revision":common.command(["git","-C",str(ROOT),"rev-parse","HEAD"]),
              "working_tree_status":common.command(["git","-C",str(ROOT),"status","--porcelain"]),
              "driver_sha256":common.digest(Path(__file__)),"source_sha256":source_hashes,
              "common_helper_sha256":common.digest(Path(common.__file__)),
              "memory_helper_sha256":common.digest(Path(memory.__file__)),
              "rustc":common.command(["rustc","--version"]),"cargo":common.command(["cargo","--version"]),
              "build_commands":checks,"rustflags":os.environ.get("RUSTFLAGS"),
              "sdk_path":common.command(["xcrun","--sdk","macosx","--show-sdk-path"]),
              "sdk_version":common.command(["xcrun","--sdk","macosx","--show-sdk-version"]),
              "os":common.command(["sw_vers"]),
              "hardware":common.command(["/usr/sbin/sysctl","-n","machdep.cpu.brand_string","hw.model","hw.memsize"]),
              "power_start":common.power(),"checks":[],"native_runs":[],
              "actual_gpu_trace":None,"energy":None,"total_external_device_memory":None,
              "extra_context":extra_context,
              "scope":"Standalone Rust public-API/ownership and optional FP16 RMS/embedding/residual/SiLU/gating/MPS projection checks. No full model backend, device-performance acceptance, M4 completion or mobile qualification."}
    stage = "build_checks"
    try:
        with stderr_path.open("x") as stderr:
            for command in checks:
                started = time.perf_counter()
                stdout = common.bounded(command,stderr,300)
                record["checks"].append({"command":command,"stdout":stdout,"wall_seconds":time.perf_counter()-started})
            for source,target in copies.items():
                expected = source_hashes[str(source.relative_to(ROOT))]
                if common.digest(source) != expected or common.digest(target) != expected:
                    raise ValueError("Qualified source or copied build input changed")
            binary = build/"target/aarch64-apple-darwin/release/toklet-rust-metal-probe"
            binary_hash = common.digest(binary)
            record["binary_sha256"] = binary_hash
            stage = "native_qualification"
            with native_raw.open("x") as raw:
                jobs = [("normal",[],None),("invalid_shader",["--invalid-shader"],None),
                        ("insufficient_budget",["--budget-bytes","1"],None)]+list(extra_runs)
                for mode,options,validator in jobs:
                    command = [str(binary.resolve())]+options
                    result = run_child(command,raw,stderr)
                    record["native_runs"].append({"mode":mode,"command":command,**result})
                    validate_run(mode,result,validator)
                    peak = result["physical_process_charge_peak_bytes"]
                    if result["watchdog_killed"] or peak is not None and peak > LIMIT:
                        raise ValueError("Native watchdog or memory qualification failed")
            if common.digest(binary) != binary_hash:
                raise ValueError("Binary changed during native qualification")
            for source,target in copies.items():
                expected = source_hashes[str(source.relative_to(ROOT))]
                if common.digest(source) != expected or common.digest(target) != expected:
                    raise ValueError("Source changed during native qualification")
            record["completed"] = True
    except Exception as error:
        record["failure"] = {"stage":stage,"type":type(error).__name__,"message":str(error)}
        raise
    finally:
        record["power_end"] = common.power()
        record["stderr_sha256"] = common.digest(stderr_path) if stderr_path.exists() else None
        record["native_raw_sha256"] = common.digest(native_raw) if native_raw.exists() else None
        with output.open("x") as file:
            file.write(json.dumps(record,indent=2,allow_nan=False)+"\n")
    print(f"Completed standalone Rust Metal binding qualification: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--build-dir",type=Path,required=True)
    args = parser.parse_args()
    run(args.output,args.build_dir)


if __name__ == "__main__":
    main()
