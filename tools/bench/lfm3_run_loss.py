"""Collect one or more complete frozen Toklet loss blocks under the engine guard."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time

from lfm3_loss_report import CHECKPOINTS, TARGETS, score_rows
from lfm3_quality_inputs import INPUTS_SHA256, digest, read_bound, require

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from native_usage import usage

CAP = 6 * 1024**3
FLOOR = 2 * 1024**3
STALL_SECONDS = 1800


def write(path, result):
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def free(folder):
    disk = os.statvfs(folder)
    return disk.f_bavail * disk.f_frsize


def check_ledger(stdout, blocks, first, end):
    records = [json.loads(line) for line in stdout.read_bytes().splitlines()]
    require(len(records) == end - first + 1, "Native block ledger is incomplete")
    ready = records[0]
    require(ready["event"] == "ready"
            and bytes(ready["source_sha256"]).hex() == CHECKPOINTS["modified_parent"]
            and ready["first_block"] == first and ready["end_block"] == end
            and ready["context_capacity"] == 2048 and ready["chunk"] == 128
            and sum(ready[k] for k in ("model_bytes", "load_allowance_bytes", "diagnostic_bytes")) <= CAP,
            "Native checkpoint or admission ledger differs")
    for actual, expected in zip(records[1:], blocks[first:end], strict=True):
        require(actual == {"event": "block", "block": expected["block"],
                           "targets": expected["targets"], "end": expected["end"]},
                "Native block completion differs from the frozen plan")
    return ready


def run(args):
    spec = args.inputs / "reconciliation.json"
    reconciliation = json.loads(read_bound(spec, INPUTS_SHA256))
    blocks = reconciliation["loss_blocks"]
    require(len(blocks) == 10 and 0 <= args.first_block < args.end_block <= len(blocks),
            "Invalid frozen loss block selection")
    source = args.inputs / "stock/held_out_loss.u32le"
    raw = read_bound(source, reconciliation["loss_ids_sha256"])
    ids = tuple(row[0] for row in struct.iter_unpack("<I", raw))
    require(len(ids) == TARGETS + 1, "Held-out input length differs")
    first, end = args.first_block, args.end_block
    begin, stop = blocks[first]["start"], blocks[end - 1]["end"]
    require(digest(args.binary) == args.binary_sha256, "Native loss binary differs")
    own = Path(__file__).resolve()
    inputs = [args.binary, spec, source, own, own.with_name("lfm3_loss_report.py"),
              own.with_name("lfm3_quality_inputs.py"), own.parents[1] / "experiments/native_usage.py"]
    pins = {str(path): digest(path) for path in inputs}
    args.output.mkdir()
    out = args.output / "scores.jsonl"
    stdout = args.output / "stdout.log"
    command = [str(args.binary), str(args.model), str(source), str(out), str(first), str(end)]
    minimum = free(args.output)
    require(minimum >= FLOOR, "Disk floor exceeded before launch")
    write(args.output / "freeze.json", dict(command=command, pins=pins, rss_ceiling=CAP,
          disk_floor=FLOOR, input_reconciliation_sha256=INPUTS_SHA256, target_start=begin, target_end=stop))
    maxima = [0, 0]
    max_physical = samples = size = 0
    started = progress = time.monotonic()
    failure = None
    with stdout.open("xb") as log, (args.output / "stderr.log").open("xb") as errors, \
            (args.output / "memory.jsonl").open("x") as memory:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=errors,
                                 start_new_session=True)
        try:
            while True:
                pid, status, stats = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                try:
                    measured = [usage(), usage(child.pid)]
                except OSError:
                    pid, status, stats = os.wait4(child.pid, os.WNOHANG)
                    if pid:
                        child.returncode = os.waitstatus_to_exitcode(status)
                        break
                    raise
                rss = [p["resident_size"] for p in measured]
                physical = sum(p["phys_footprint"] for p in measured)
                maxima = [max(previous, current) for previous, current in zip(maxima, rss, strict=True)]
                max_physical = max(max_physical, physical)
                now = time.monotonic()
                memory.write(json.dumps(dict(elapsed_seconds=now - started, processes=measured)) + "\n")
                memory.flush()
                samples += 1
                minimum = min(minimum, free(args.output))
                require(sum(rss) <= CAP and physical <= CAP and minimum >= FLOOR, "Native memory or disk guard failed")
                current_size = out.stat().st_size if out.is_file() else 0
                if current_size > size:
                    size, progress = current_size, now
                require(now - progress < STALL_SECONDS, "Native loss worker stopped producing rows")
                time.sleep(0.2)
        except BaseException as error:
            failure = repr(error)
        finally:
            if child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, stats = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
    peaks = [resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, stats.ru_maxrss]
    consistent = all(a <= b for a, b in zip(maxima, peaks, strict=True))
    unchanged = all(Path(p).is_file() and digest(Path(p)) == h for p, h in pins.items())
    actual = digest(out) if out.is_file() else None
    ready = None
    if failure is None:
        try:
            require(child.returncode == 0 and samples > 0 and sum(peaks) <= CAP and consistent and unchanged,
                    "Native worker exit, resource accounting or input pins did not qualify")
            ready = check_ledger(stdout, blocks, first, end)
            score_rows(out, actual, ids, begin, stop)
        except (OSError, ValueError, TypeError, KeyError) as error:
            failure = repr(error)
    report = dict(format="toklet-lfm3-native-loss-window-v1", passed=failure is None,
                  failure=failure, command=command, pid=child.pid, exit_code=child.returncode,
                  target_start=begin, target_end=stop, scores_sha256=actual, ready=ready,
                  freeze_sha256=digest(args.output / "freeze.json"), sample_count=samples,
                  final_rss_bytes=peaks, conservative_peak_sum_bytes=sum(peaks),
                  max_sampled_physical_bytes=max_physical, minimum_free_bytes=minimum,
                  sample_peaks_consistent=consistent, inputs_unchanged=unchanged,
                  elapsed_seconds=time.monotonic() - started)
    write(args.output / "guard.json", report)
    print(json.dumps(report))
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "binary", "model", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--first-block", type=int, required=True)
    parser.add_argument("--end-block", type=int, required=True)
    raise SystemExit(0 if run(parser.parse_args()) else 1)
