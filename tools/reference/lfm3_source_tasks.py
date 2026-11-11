"""Own a restartable, guarded LFM3 reference generation campaign."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time

TOOLS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TOOLS / "bench"), str(TOOLS / "experiments")]
from lfm3_quality_inputs import EOS, INPUTS_SHA256, VOCAB, digest, read_bound, require, task_inputs
from native_usage import usage

RUNTIME_SHA = "1ca684c819f499a2ede6074c364b64c64c55b3fbd7fc29ccd23ff25a937202ed"
SOURCE_SHA = "d9a5b636fcb3e7eb472f3cd1e58848bbaa4703a6acfca4dc4a398eb963751af2"
LIMIT = 512 * 1024**2
FLOOR = 2 * 1024**3


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def write(path, value):
    with path.open("x") as stream:
        stream.write(dump(value))
        stream.flush()
        os.fsync(stream.fileno())


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def require_disk(path):
    info = os.statvfs(path)
    free = info.f_bavail * info.f_frsize
    require(free >= FLOOR, "Disk floor exceeded")
    return free


def identity(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "Checkpoint must be a regular file")
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def verify_runtime(path):
    require(digest(path) == RUNTIME_SHA, "Runtime inventory changed")
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            if line == b'  "files": {\n':
                break
        else:
            raise ValueError("Runtime file map missing")
        for line in stream:
            if line == b"  },\n":
                break
            [(name, expected)] = json.loads(b"{" + line.strip().rstrip(b",") + b"}").items()
            require(digest(Path(name)) == expected, f"Runtime input changed: {name}")
            count += 1
        else:
            raise ValueError("Runtime file map incomplete")
    require(count == 28982 and digest(path) == RUNTIME_SHA, "Runtime inventory differs")
    return count


def guard_phase(command, label, evidence, fixture):
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    minimum_free = require_disk(evidence)
    maxima = [0, 0]
    maximum_physical = 0
    samples = 0
    failure = None
    with (evidence / f"{label}.log").open("x") as log, (evidence / f"{label}-memory.jsonl").open("x") as memory:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
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
                now = time.monotonic()
                rss = [p["resident_size"] for p in measured]
                physical = sum(p["phys_footprint"] for p in measured)
                maxima = [max(a, b) for a, b in zip(maxima, rss, strict=True)]
                maximum_physical = max(maximum_physical, physical)
                memory.write(dump({"elapsed_seconds": now - start, "processes": measured}))
                samples += 1
                minimum_free = min(minimum_free, require_disk(evidence))
                require(sum(rss) <= LIMIT and physical <= LIMIT, "512 MiB source guard exceeded")
                require(now - start < 1800, "Source phase timed out")
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
    ok = (failure is None and child.returncode == 0 and samples > 0 and sum(peaks) <= LIMIT
          and all(a <= b for a, b in zip(maxima, peaks, strict=True)) and fixture.is_file())
    write(evidence / f"{label}-guard.json",
          dict(passed=ok, command=command, exit_code=child.returncode, failure=failure,
               final_peak_rss_bytes=peaks, max_sampled_physical_bytes=maximum_physical,
               minimum_free_bytes=minimum_free, samples=samples,
               elapsed_seconds=time.monotonic() - start,
               fixture_sha256=digest(fixture) if fixture.is_file() else None))
    require(ok, f"Guarded source phase failed: {label}")
    return read(fixture)


def record_samples(task_map, sampled, active, head, want_digest):
    data = head / "samples.jsonl"
    require(digest(data) == want_digest, "Head samples differ from their receipt")
    records = rows(data)
    require([r["index"] for r in records] == active, "Missing or misordered task samples")
    for row in records:
        index, token, value = row["index"], row["token_id"], row["max_logit"]
        require(type(token) is int and 0 <= token < VOCAB and math.isfinite(value)
                and len(bytes.fromhex(row["logit_sha256"])) == 32, "Invalid tied-head decision")
        require(len(sampled[index]) < task_map[index]["max_output_tokens"], "Task sampled beyond its cap")
        sampled[index].append(row)
    return [i for i in active if sampled[i][-1]["token_id"] != EOS
            and len(sampled[i]) < task_map[i]["max_output_tokens"]]


def read_prior(output, task_map, sampled, active, freeze_sha):
    prior = None
    commits = sorted(output.glob("round-*-*/commit.json"))
    for index, path in enumerate(commits):
        receipt = read(path)
        require(receipt["round"] == index and receipt["freeze_sha256"] == freeze_sha
                and receipt["previous_commit_sha256"] == (digest(prior / "commit.json") if prior else None)
                and receipt["active"] == active, "Incomplete or changed round chain")
        active = record_samples(task_map, sampled, active, path.parent / "head", receipt["samples_sha256"])
        prior = path.parent
    return prior, active, len(commits)


def checkpoint_inputs(args):
    source = json.loads(read_bound(args.source / "prospective-contract-1.json", SOURCE_SHA))
    key = "stock_metadata_reference" if args.role == "stock" else "candidate"
    item = source[key]
    current = identity(args.model)
    require(current[2] == item["bytes"] and digest(args.model) == item["sha256"]
            and current == identity(args.model), "Source checkpoint differs")
    return item, current


def run_round(args, number, task_map, sampled, active, prior, freeze, checkpoint, pins):
    attempt = 0
    while (folder := args.output / f"round-{number:03d}-{attempt:03d}").exists():
        attempt += 1
    folder.mkdir()
    state_dir = folder / "state"
    state_dir.mkdir()
    plan = []
    for index in active:
        previous = sampled[index]
        prompt = task_map[index]["prompt_tokens"]
        plan.append(dict(index=index, start=len(prompt) + len(previous) - 1 if previous else 0,
                         tokens=[previous[-1]["token_id"]] if previous else prompt))
    manifest = folder / "plan.json"
    write(manifest, dict(round=number, cases=plan, file_identity=checkpoint))
    plan_sha = digest(manifest)
    executable = [str(args.python), "-B", str(args.driver), str(Path(__file__).with_name("lfm3_source_task_worker.py")),
                  "--source", str(args.source), "--model", str(args.model), "--role", args.role,
                  "--plan", str(manifest), "--state", str(state_dir)]
    previous = None
    states = {}
    for layer in range(0, 24, args.layer_block):
        end = min(layer + args.layer_block, 24)
        label = f"layers-{layer:02d}-{end:02d}"
        path = folder / label
        command = executable + ["--first", str(layer), "--end", str(end), "--output", str(path)]
        if previous:
            command += ["--previous", str(previous)]
        if prior:
            command += ["--prior", str(prior)]
        result = guard_phase(command, label, folder, path / "fixture.json")
        require(result["first_layer"] == layer and result["end_layer"] == end
                and result["plan_sha256"] == plan_sha, "Unexpected layer phase")
        expected = {f"{n:02d}-{case:03d}.bf16" for n in range(layer, end) for case in active}
        require(set(result["state"]) == expected and not states.keys() & expected, "Incomplete task cache output")
        states.update(result["state"])
        previous = path
    head = folder / "head"
    result = guard_phase(executable + ["--output", str(head), "--previous", str(previous)],
                         "head", folder, head / "fixture.json")
    require(result["plan_sha256"] == plan_sha and result["sample_count"] == len(active),
            "Incomplete output decisions")
    next_active = record_samples(task_map, sampled, active, head, result["samples_sha256"])
    require(identity(args.model) == checkpoint and all(digest(Path(p)) == v for p, v in pins.items()),
            "Source campaign inputs changed during a round")
    pending = folder / "commit.tmp"
    write(pending, dict(round=number, active=active, freeze_sha256=freeze,
                        plan_sha256=plan_sha, state=states, samples_sha256=result["samples_sha256"],
                        previous_commit_sha256=digest(prior / "commit.json") if prior else None))
    pending.rename(folder / "commit.json")
    for path in folder.glob("layers-*/hidden"):
        shutil.rmtree(path)
    if prior:
        previous_state = prior / "state"
        if previous_state.is_dir():
            shutil.rmtree(previous_state)
    return folder, next_active


def run(args):
    require(0 <= args.first < args.end <= 100 and 1 <= args.layer_block <= 8, "Invalid case or layer scope")
    tasks, _ = task_inputs(args.inputs, args.role)
    task_map = {index: tasks[index] for index in range(args.first, args.end)}
    item, checkpoint = checkpoint_inputs(args)
    source = args.source / "metadata-1"
    profile = "stock" if args.role == "stock" else "native"
    inputs = [args.inputs / p for p in ["reconciliation.json", "cases.jsonl", profile + "/tasks.jsonl"]]
    inputs += [args.source / "prospective-contract-1.json", args.driver,
               source / "parent-modeling_lfm3.py", source / "parent-config.json",
               source / "stock_mlx-config.json", TOOLS / "bench/lfm3_quality_inputs.py",
               TOOLS / "experiments/native_usage.py"]
    inputs += [Path(__file__).with_name(n) for n in (
        "lfm3_source_tasks.py", "lfm3_source_task_worker.py", "lfm3_task_layers.py", "lfm3_task_checkpoint.py",
        "lfm3_context1025_fixture.py", "lfm3_context513_fixture.py", "lfm3_graph_bf16_fixture.py",
        "lfm3_bf16_fixture.py", "lfm3_attention_bf16_fixture.py")]
    pins = {str(path): digest(path) for path in inputs}
    if not args.resume:
        args.output.mkdir()
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_disk(args.output)
        count = verify_runtime(args.runtime)
        freeze = args.output / "freeze.json"
        planned = dict(role=args.role, model=str(args.model), checkpoint=item, file_identity=checkpoint,
                       pins=pins, runtime=str(args.runtime), runtime_sha256=RUNTIME_SHA, runtime_files=count,
                       first=args.first, end=args.end, layer_block=args.layer_block,
                       input_reconciliation_sha256=INPUTS_SHA256, reference_bytes=LIMIT, disk_floor_bytes=FLOOR)
        if args.resume:
            require(read(freeze) == planned and not (args.output / "result.json").exists(), "Cannot resume changed or completed campaign")
        else:
            write(freeze, planned)
        freeze_sha = digest(freeze)
        sampled = {index: [] for index in task_map}
        prior, active, number = read_prior(args.output, task_map, sampled, list(task_map), freeze_sha)
        while active:
            prior, active = run_round(args, number, task_map, sampled, active, prior, freeze_sha, checkpoint, pins)
            number += 1
            print(dump({"rounds": number, "remaining_cases": len(active), "role": args.role}).strip(), flush=True)
        complete = []
        for index, sample in sampled.items():
            task = task_map[index]
            tokens = [r["token_id"] for r in sample]
            stop = tokens[-1] == EOS
            cap = task["max_output_tokens"]
            require(len(tokens) <= cap and (stop or len(tokens) == cap)
                    and EOS not in tokens[:-1], "Invalid terminal task sequence")
            reason = {"kind": "stop_token", "token_id": EOS} if stop else {"kind": "output_limit"}
            prompt_len = len(task["prompt_tokens"])
            complete.append(dict(index=index, id=task["id"], category=task["category"], prompt_tokens=prompt_len,
                                 output_reserve=cap, samples=sample, visible_token_ids=tokens[:-1] if stop else tokens,
                                 reason=reason, history_tokens=prompt_len + len(tokens),
                                 committed_tokens=prompt_len + len(tokens) - 1, pending_tokens=1))
        require(checkpoint_inputs(args) == (item, checkpoint)
                and all(digest(Path(p)) == v for p, v in pins.items())
                and verify_runtime(args.runtime) == count, "Source campaign boundary changed")
        pending = args.output / "tasks.tmp"
        with pending.open("w") as stream:
            for task in complete:
                stream.write(dump(task))
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(args.output / "tasks.jsonl")
        write(args.output / "result.json", dict(passed=True, role=args.role, rounds=number, cases=len(complete),
              first=args.first, end=args.end, source_checkpoint_sha256=item["sha256"], freeze_sha256=freeze_sha,
              tasks_sha256=digest(args.output / "tasks.jsonl"), runtime_verified_files=count))
        if prior and (prior / "state").is_dir():
            shutil.rmtree(prior / "state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "model", "inputs", "output", "driver", "runtime"]:
        parser.add_argument("--" + name, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--python", type=lambda p: Path(p).absolute(), required=True)
    parser.add_argument("--role", choices=["stock", "modified_parent"], required=True)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--layer-block", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
