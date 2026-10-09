"""Synthetic projection pairs on Qwen shapes; not an end-to-end speed claim."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import threading

import mlx_baseline as common
import prepare


SHAPES = [(1, 896, 896), (128, 896, 896), (1, 896, 128), (128, 896, 128),
          (1, 896, 4864), (128, 896, 4864), (1, 4864, 896), (128, 4864, 896),
          (1, 896, 151936)]


def validate(row, shape):
    if tuple(row[key] for key in ("tokens", "in_features", "out_features")) != shape:
        raise ValueError("Unexpected projection shape")
    if (row["warmup_pairs"] != 3 or row["measured_pairs"] != 10
            or len(row["samples"]) != 10 or row["bitwise_equal_every_pair"] is not True):
        raise ValueError("Missing complete correctness/measurement pairs")
    for i, sample in enumerate(row["samples"]):
        if sample["candidate_first"] != ((i + 3) % 2 == 1):
            raise ValueError("Pair order was not alternated")
        for key in ("scalar_seconds", "tiled_seconds"):
            value = sample[key]
            if type(value) not in (int, float) or not 0 < value < 3600:
                raise ValueError("Invalid completed operation time")
    tokens, inputs, outputs = shape
    if (row["input_bytes"] != tokens * inputs * 4
            or row["weights_bytes"] != inputs * outputs * 4
            or row["output_buffers_bytes"] != tokens * outputs * 8
            or row["candidate_accumulator_stack_bytes"] != 128):
        raise ValueError("Unexpected allocation accounting")
    for key in ("input_sha256", "weights_sha256", "output_sha256"):
        value = row[key]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid synthetic data identity")
    scalar = [sample["scalar_seconds"] for sample in row["samples"]]
    tiled = [sample["tiled_seconds"] for sample in row["samples"]]
    row["summary"] = {
        "scalar_median_seconds": statistics.median(scalar),
        "scalar_min_seconds": min(scalar), "scalar_max_seconds": max(scalar),
        "tiled_median_seconds": statistics.median(tiled),
        "tiled_min_seconds": min(tiled), "tiled_max_seconds": max(tiled),
        "median_pair_speedup": statistics.median(a / b for a, b in zip(scalar, tiled)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.output.with_suffix(args.output.suffix + ".rows.jsonl")
    if args.output.exists() or raw.exists():
        raise ValueError("Refusing to overwrite projection evidence")
    binary = prepare.ROOT / "target/release/examples/cpu_projection_bench"
    source_paths = [prepare.ROOT / path for path in (
        "src/cpu/mod.rs", "src/cpu/tiled.rs", "examples/cpu_projection_bench.rs",
        "tools/bench/projection.py", "tools/bench/mlx_baseline.py", "Cargo.lock")]
    record = {
        "format": "toklet-projection-pairs-v1", "synthetic_weights": True,
        "scope": "Complete validated FP32-storage/ordered-FP64 projection; bias absent. Allocation, data hashing, comparisons, transport, memory inspection excluded.",
        "engine_revision": common.command(["git", "rev-parse", "HEAD"]),
        "working_tree_dirty": bool(common.command(["git", "status", "--porcelain"])),
        "source_sha256": {str(path.relative_to(prepare.ROOT)): prepare.digest(path) for path in source_paths},
        "binary_sha256": prepare.digest(binary),
        "compiler": common.command(["rustc", "--version"]),
        "build_command": "cargo build --locked --offline --release --example cpu_projection_bench",
        "rustflags": os.environ.get("RUSTFLAGS"),
        "device": common.command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
        "hardware_model": common.command(["/usr/sbin/sysctl", "-n", "hw.model"]),
        "physical_memory_bytes": int(common.command(["/usr/sbin/sysctl", "-n", "hw.memsize"])),
        "os": common.command(["sw_vers", "-productVersion"]),
        "os_build": common.command(["sw_vers", "-buildVersion"]),
        "architecture": platform.machine(),
        "power_start": common.command(["/usr/bin/pmset", "-g", "batt"]),
        "power_configuration_start": common.command(["/usr/bin/pmset", "-g", "custom"]),
        "thermal_start": common.command(["/usr/bin/pmset", "-g", "therm"]),
        "rows": [],
        "limitations": ["Representative projection shapes with deterministic synthetic data; not trained weights, decoder quality, KV state, or full workload qualification.",
                         "Full input/weight finite validation is included in both operation timers. Median pair speedup is descriptive, not a retain decision.",
                         "Engine integration and a second confirming run are separate gates."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("x") as checkpoint:
        worker = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True)
        watchdog = threading.Timer(3600, worker.kill)
        watchdog.start()
        try:
            for shape in SHAPES:
                line = worker.stdout.readline(65537)
                if not line.endswith("\n") or len(line.encode()) > 65536:
                    raise ValueError("Incomplete/oversized projection record")
                row = json.loads(line)
                validate(row, shape)
                row["process_peak_footprint_bytes"] = common.footprint_peak(worker.pid)
                if (row["process_peak_footprint_bytes"] is None
                        or row["process_peak_footprint_bytes"] > 6 * 1024**3):
                    raise ValueError("Projection memory instrument unavailable or over budget")
                checkpoint.write(json.dumps(row, allow_nan=False) + "\n")
                checkpoint.flush()
                record["rows"].append(row)
                print(f"Shape {shape}: paired speedup {row['summary']['median_pair_speedup']:.3f}", flush=True)
                worker.stdin.write("next\n")
                worker.stdin.flush()
            worker.stdin.close()
            if worker.stdout.read(1) or worker.wait(timeout=30) != 0:
                raise ValueError("Projection worker did not exit cleanly")
        finally:
            watchdog.cancel()
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            worker.stdout.close()
            if not worker.stdin.closed:
                worker.stdin.close()
    record["power_end"] = common.command(["/usr/bin/pmset", "-g", "batt"])
    record["power_configuration_end"] = common.command(["/usr/bin/pmset", "-g", "custom"])
    record["thermal_end"] = common.command(["/usr/bin/pmset", "-g", "therm"])
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
