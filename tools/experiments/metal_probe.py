"""Build a fresh standalone Swift/Metal probe and retain bounded native evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_suffix(".swift")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=30)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def power():
    return {name: command(["/usr/bin/pmset", "-g", option]) for name, option in
            (("source", "batt"), ("configuration", "custom"), ("thermal", "therm"))}


def bounded(args, stderr, timeout):
    # Each newly created process group belongs only to this explicit probe.
    # Reap the child and its compiler subprocesses if a native call hangs.
    with subprocess.Popen(args, text=True, stdout=subprocess.PIPE, stderr=stderr,
                          start_new_session=True) as child:
        try:
            stdout, _ = child.communicate(timeout=timeout)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            raise
        if child.returncode:
            raise RuntimeError(f"Probe stage exited {child.returncode}: {args[0]}")
        return stdout


def run(output, build):
    stderr_path = output.with_suffix(output.suffix + ".stderr.txt")
    if output.exists() or stderr_path.exists() or build.exists():
        raise ValueError("Refusing to overwrite probe evidence/build artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True)
    source_hash = digest(SOURCE)
    binary = build / "metal-probe"
    args = ["xcrun", "swiftc", "-O", "-warnings-as-errors", "-target", "arm64-apple-macosx15.0",
            str(SOURCE), "-o", str(binary)]
    record = {"format": "toklet-metal-toolchain-probe-v1", "driver_sha256": digest(Path(__file__)), "compile_command": args,
              "engine_revision": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
              "working_tree_status": command(["git", "-C", str(ROOT), "status", "--porcelain"]),
              "compiler": command(["xcrun", "swiftc", "--version"]),
              "active_developer_directory": command(["xcode-select", "-p"]),
              "sdk_path": command(["xcrun", "--sdk", "macosx", "--show-sdk-path"]),
              "sdk_version": command(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
              "physical_process_peak_bytes": None, "actual_gpu_trace": None, "energy": None,
              "power_start": power(), "completed": False}
    stage = "open_stderr"
    try:
        with stderr_path.open("x") as stderr:
            stage = "swift_compile"
            start = time.perf_counter()
            bounded(args, stderr, 300)
            record["swift_compile_seconds"] = time.perf_counter() - start
            binary_hash = digest(binary)
            if digest(SOURCE) != source_hash:
                raise ValueError("Source changed during compilation")
            stage = "native_probe"
            start = time.perf_counter()
            result = json.loads(bounded([str(binary.resolve())], stderr, 60))
            record["cold_process_wall_seconds"] = time.perf_counter() - start
            if (result["source_sha256"] != source_hash or result["binary_sha256"] != binary_hash
                    or digest(SOURCE) != source_hash or digest(binary) != binary_hash):
                raise ValueError("Compiled source/binary identity differs")
            record.update(result)
            record["completed"] = True
    except Exception as error:
        record["failure"] = {"stage": stage, "type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["power_end"] = power()
        for tool in ("metal", "xctrace"):
            record[tool + "_tool"] = command(["xcrun", "--find", tool])
        with output.open("x") as file:
            file.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"Completed functional Metal check: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.build_dir)


if __name__ == "__main__":
    main()
