"""Collect the frozen LFM3 task outputs through Toklet's existing console route."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import select
import signal
import subprocess
import sys
import time

from lfm3_quality_inputs import EOS, INPUTS_SHA256, VOCAB, digest, require, task_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from native_usage import usage

LIMIT = 6 * 1024**3
FLOOR = 2 * 1024**3
CHECKPOINT = "b6ff17ac993dddf0f6f51bd872b12ae1437f4453a54a0b4878cedc7864fbc07e"
CASE_TIMEOUT = 1800


def write_row(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


class Console:
    def __init__(self, command, error, transcript, memory, folder):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=error, start_new_session=True)
        self.transcript = transcript
        self.memory = memory
        self.folder = folder
        self.buffer = bytearray()
        self.started = time.monotonic()
        self.sample_after = self.started
        self.min_free = 0
        self.max_rss = [0, 0]
        self.max_physical = 0
        self.samples = 0
        self.terminal = None

    def reap(self, flags):
        if self.terminal is not None:
            return True
        pid, status, stats = os.wait4(self.process.pid, flags)
        if not pid:
            return False
        self.process.returncode = os.waitstatus_to_exitcode(status)
        self.terminal = stats
        return True

    def sample(self, now):
        if now < self.sample_after:
            return
        try:
            processes = [usage(), usage(self.process.pid)]
        except OSError:
            if self.reap(os.WNOHANG):
                return
            raise
        write_row(self.memory, {"elapsed_seconds": now - self.started, "processes": processes})
        self.sample_after = now + 0.2
        self.samples += 1
        rss = [p["resident_size"] for p in processes]
        physical = sum(p["phys_footprint"] for p in processes)
        self.max_rss = [max(previous, current) for previous, current in zip(self.max_rss, rss, strict=True)]
        self.max_physical = max(self.max_physical, physical)
        disk = os.statvfs(self.folder)
        free = disk.f_bavail * disk.f_frsize
        self.min_free = min(self.min_free, free) if self.min_free else free
        require(sum(rss) <= LIMIT and physical <= LIMIT and free >= FLOOR, "Memory or disk guard failed")

    def receive(self, deadline):
        while b"\n" not in self.buffer:
            now = time.monotonic()
            require(now < deadline, "Timed out waiting for an owned console event")
            self.sample(now)
            ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
            if ready:
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    require(not self.buffer, "Console stopped with a partial event")
                    return None
                self.buffer.extend(chunk)
                require(len(self.buffer) <= 1024 * 1024, "Console event exceeded transport bound")
        end = self.buffer.index(b"\n")
        event = json.loads(self.buffer[:end])
        del self.buffer[:end + 1]
        require(type(event) is dict and event.get("event") != "error", "Console rejected the request")
        write_row(self.transcript, {"received": event})
        return event

    def send(self, request):
        write_row(self.transcript, {"sent": request})
        self.process.stdin.write((json.dumps(request, separators=(",", ":")) + "\n").encode())
        self.process.stdin.flush()

    def finish(self):
        self.process.stdin.close()
        require(self.receive(time.monotonic() + 15) is None, "Unexpected event after final reset")
        self.reap(0)
        require(self.process.returncode == 0, "Console exited unsuccessfully")

    def abort(self):
        if not self.reap(os.WNOHANG):
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.reap(0)


def collect(console, task, ordinal):
    prompt = task["prompt_tokens"]
    cap = task["max_output_tokens"]
    console.send({"action": "generate", "tokens": prompt, "max_new_tokens": cap})
    started = time.monotonic_ns()
    deadline = time.monotonic() + CASE_TIMEOUT
    tokens, text = [], []
    previous = 0.0
    finalized = False
    while True:
        event = console.receive(deadline)
        require(event is not None, "Console exited before task completion")
        kind = event["event"]
        if kind == "done":
            break
        if kind == "token":
            token = event["token_id"]
            elapsed = event["elapsed_seconds"]
            require(not finalized and event["index"] == len(tokens) < cap and type(token) is int
                    and 0 <= token < VOCAB and token != EOS and type(elapsed) in (int, float)
                    and math.isfinite(elapsed) and elapsed >= previous,
                    "Invalid visible token, order or timestamp")
            tokens.append(token)
            previous = elapsed
        else:
            require(kind == "text" and not finalized, "Invalid text finalization order")
            finalized = True
        require(type(event["text"]) is str, "Invalid console text")
        text.append(event["text"])

    statistics = event["statistics"]
    generation = statistics["generation"]
    reason = generation["reason"]
    if reason == {"kind": "stop_token", "token_id": EOS}:
        sampled = len(tokens) + 1
        require(sampled <= cap, "Stop consumed an unadmitted sample")
    else:
        require(reason == {"kind": "output_limit"} and len(tokens) == cap and not finalized,
                "Task did not finish at an original stopping boundary")
        sampled = cap
    expected = dict(reused_tokens=0, prefill_tokens=len(prompt), sampled_tokens=sampled,
                    emitted_tokens=len(tokens), history_tokens=len(prompt) + sampled,
                    evaluated_tokens=len(prompt) + sampled - 1, pending_tokens=1)
    require(all(generation[k] == v for k, v in expected.items()), "Task token state differs")
    require(statistics["model_profile"] == "lfm3-bf16" and statistics["decode_evaluations"] == sampled - 1
            and statistics["tokenization_seconds"] == statistics["input_formatting_seconds"] == 0
            and sum(statistics["logical_memory"].values()) == statistics["memory_admitted_bytes"] <= LIMIT,
            "Task route, prepared IDs or logical admission differ")
    console.send({"action": "reset"})
    reset = console.receive(time.monotonic() + 15)
    if reset and reset.get("event") == "text":
        require(not finalized and reason["kind"] == "output_limit" and type(reset["text"]) is str,
                "Duplicate or invalid reset finalization")
        text.append(reset["text"])
        reset = console.receive(time.monotonic() + 15)
    require(reset == {"event": "reset"}, "Independent task reset was not acknowledged")
    visible = "".join(text)
    encoded = visible.encode("utf-8")
    require(len(encoded) <= 65536, "Task output exceeds the literal grader's text bound")
    return dict(index=ordinal, id=task["id"], category=task["category"], visible_token_ids=tokens,
                visible_text=visible, visible_text_sha256=hashlib.sha256(encoded).hexdigest(),
                prompt_tokens=len(prompt), output_reserve=cap, generation=generation,
                observed_seconds=(time.monotonic_ns() - started) / 1e9)


def run(args):
    require(0 <= args.first < args.end <= 100, "Invalid case range")
    tasks, _ = task_inputs(args.inputs, "native")
    require(digest(args.binary) == args.binary_sha256, "Release binary differs")
    paths = [Path(__file__).resolve(), Path(__file__).with_name("lfm3_quality_inputs.py").resolve(),
             args.inputs / "reconciliation.json", args.inputs / "cases.jsonl",
             args.inputs / "native/tasks.jsonl", args.binary]
    pins = {str(p): digest(p) for p in paths}
    command = [str(args.binary), "--model-profile", "lfm3-bf16", "--model", str(args.model),
               "--json", "--interactive", "--context-capacity", "2048", "--prefill-chunk", "128"]
    args.output.mkdir()
    report = dict(format="toklet-lfm3-native-task-collection-v1", command=command, pins=pins,
                  input_reconciliation_sha256=INPUTS_SHA256, first=args.first, end=args.end,
                  completed=0, passed=False, failure=None)
    with (args.output / "freeze.json").open("x") as receipt:
        write_row(receipt, report)
    with (args.output / "stderr.log").open("x") as errors, (args.output / "events.jsonl").open("x") as transcript, \
         (args.output / "memory.jsonl").open("x") as memory, (args.output / "tasks.jsonl").open("x") as results:
        console = Console(command, errors, transcript, memory, args.output)
        report["pid"] = console.process.pid
        try:
            ready = console.receive(time.monotonic() + 240)
            require(ready and ready["event"] == "ready" and ready["model_profile"] == "lfm3-bf16"
                    and ready["checkpoint_load"]["sha256"] == CHECKPOINT
                    and ready["execution"]["prefill_chunk_size"] == 128 and ready["context_capacity"] == 2048
                    and ready["execution"]["weight_dtype"] == ready["execution"]["kv_dtype"] == "BF16"
                    and sum(ready["logical_memory"].values()) == ready["memory_admitted_bytes"] <= LIMIT,
                    "Console ready identity or admission differs")
            for index in range(args.first, args.end):
                result = collect(console, tasks[index], index)
                write_row(results, result)
                report["completed"] += 1
                print(json.dumps({"completed": report["completed"], "last_case": result["id"]}), flush=True)
            console.finish()
        except BaseException as error:
            report["failure"] = repr(error)
            console.abort()
        peaks = [resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, console.terminal.ru_maxrss]
        report.update(exit_code=console.process.returncode, peak_rss_bytes=peaks,
                      sample_count=console.samples, max_sampled_physical_bytes=console.max_physical,
                      minimum_free_bytes=console.min_free, tasks_sha256=digest(args.output / "tasks.jsonl"),
                      elapsed_seconds=time.monotonic() - console.started)
        report["passed"] = (report["failure"] is None and report["completed"] == args.end - args.first
                            and console.samples > 0 and console.process.returncode == 0
                            and sum(peaks) <= LIMIT and all(a <= b for a, b in zip(console.max_rss, peaks, strict=True))
                            and all(digest(Path(p)) == v for p, v in pins.items()))
        with (args.output / "result.json").open("x") as receipt:
            write_row(receipt, report)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["binary", "model", "inputs", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    raise SystemExit(0 if run(parser.parse_args()) else 1)
