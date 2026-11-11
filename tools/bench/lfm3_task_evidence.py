"""Reconcile task journals with the guarded source rounds and CLI events."""
import json
import math

from lfm3_quality_inputs import EOS, INPUTS_SHA256, VOCAB, digest, require, task_inputs
from lfm3_quality_tasks import CHECKPOINT, LIMIT, collect

SOURCE_CAP = 512 * 1024**2
DISK_FLOOR = 2 * 1024**3


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open("rb") as stream:
        for line in stream:
            require(line.endswith(b"\n"), "Partial evidence row")
            yield json.loads(line)


class Transcript:
    def __init__(self, stream):
        self.stream = stream

    def send(self, request):
        require(next(self.stream, None) == {"sent": request}, "Recorded CLI request differs")

    def receive(self, _deadline):
        row = next(self.stream, None)
        require(row is not None and set(row) == {"received"}, "Expected a recorded CLI event")
        return row["received"]


def audit_native(folder, tasks, first, records, receipt=None):
    transcript = rows(folder / "events.jsonl")
    wire = Transcript(transcript)
    ready = wire.receive(0)
    require(ready["event"] == "ready" and ready["model_profile"] == "lfm3-bf16"
            and ready["checkpoint_load"]["sha256"] == CHECKPOINT
            and ready["execution"]["prefill_chunk_size"] == 128 and ready["context_capacity"] == 2048
            and ready["execution"]["weight_dtype"] == ready["execution"]["kv_dtype"] == "BF16"
            and sum(ready["logical_memory"].values()) == ready["memory_admitted_bytes"] <= LIMIT,
            "CLI transcript identity or admission differs")
    for index, raw in enumerate(records, first):
        replayed = collect(wire, tasks[index], index)
        require(raw.keys() == replayed.keys()
                and all(raw[key] == value for key, value in replayed.items() if key != "observed_seconds")
                and type(raw["observed_seconds"]) in (int, float)
                and math.isfinite(raw["observed_seconds"]) and raw["observed_seconds"] >= 0,
                "CLI task record differs from its request, events or reset")
    result = dict(replayed_cases=len(records))
    if receipt is not None:
        require(next(transcript, None) is None, "Unexpected CLI events after the final reset")
        memory = folder / "memory.jsonl"
        count = peak_physical = 0
        maxima = [0, 0]
        for sample in rows(memory):
            owner, child = sample["processes"]
            rss = [owner["resident_size"], child["resident_size"]]
            physical = owner["phys_footprint"] + child["phys_footprint"]
            require(sum(rss) <= LIMIT and physical <= LIMIT, "CLI memory sample exceeds its guard")
            maxima = [max(a, b) for a, b in zip(maxima, rss, strict=True)]
            peak_physical = max(peak_physical, physical)
            count += 1
        require(count == receipt["sample_count"] > 0 and peak_physical == receipt["max_sampled_physical_bytes"]
                and receipt["minimum_free_bytes"] >= DISK_FLOOR
                and all(a <= b for a, b in zip(maxima, receipt["peak_rss_bytes"], strict=True)),
                "CLI memory samples differ from the receipt")
        result.update(memory_sha256=digest(memory), events_sha256=digest(folder / "events.jsonl"))
    return result


def check_phase(folder, label, role, number, plan_sha, checkpoint):
    guard = read(folder / f"{label}-guard.json")
    path = folder / label / "fixture.json"
    require(guard["passed"] and guard["exit_code"] == 0 and guard["failure"] is None
            and guard["fixture_sha256"] == digest(path)
            and sum(guard["final_peak_rss_bytes"]) <= SOURCE_CAP
            and guard["max_sampled_physical_bytes"] <= SOURCE_CAP
            and guard["minimum_free_bytes"] >= DISK_FLOOR and guard["samples"] > 0,
            "A source phase lacks successful resource evidence")
    count = max_physical = 0
    maxima = [0, 0]
    for sample in rows(folder / f"{label}-memory.jsonl"):
        owner, child = sample["processes"]
        rss = [owner["resident_size"], child["resident_size"]]
        physical = owner["phys_footprint"] + child["phys_footprint"]
        require(sum(rss) <= SOURCE_CAP and physical <= SOURCE_CAP, "Source phase exceeded its live guard")
        maxima = [max(a, b) for a, b in zip(maxima, rss, strict=True)]
        max_physical = max(max_physical, physical)
        count += 1
    require(count == guard["samples"] and max_physical == guard["max_sampled_physical_bytes"]
            and all(a <= b for a, b in zip(maxima, guard["final_peak_rss_bytes"], strict=True)),
            "Source phase samples differ from their receipt")
    fixture = read(path)
    require(fixture["role"] == role and fixture["round"] == number
            and fixture["checkpoint_sha256"] == checkpoint and fixture["plan_sha256"] == plan_sha,
            "Source phase identity or plan differs")
    return fixture, sum(guard["final_peak_rss_bytes"])


def audit_source(folder, tasks, records, receipt):
    freeze = read(folder / "freeze.json")
    role, first, end = receipt["role"], receipt["first"], receipt["end"]
    checkpoint = receipt["source_checkpoint_sha256"]
    freeze_sha = digest(folder / "freeze.json")
    step = freeze["layer_block"]
    require(freeze_sha == receipt["freeze_sha256"]
            and freeze["role"] == role and freeze["first"] == first and freeze["end"] == end
            and freeze["checkpoint"]["sha256"] == checkpoint and 1 <= step <= 8
            and freeze["input_reconciliation_sha256"] == INPUTS_SHA256,
            "Source collection inputs or layer plan differ")
    samples = {i: [] for i in range(first, end)}
    active = list(samples)
    prior_sha = None
    phase_count = max_peak = 0
    paths = sorted(folder.glob("round-*-*/commit.json"))
    require(len(paths) == receipt["rounds"] > 0, "Source round coverage differs")
    for number, path in enumerate(paths):
        commit = read(path)
        round_dir = path.parent
        plan_path = round_dir / "plan.json"
        plan_sha = digest(plan_path)
        require(active and commit["round"] == number and commit["freeze_sha256"] == freeze_sha
                and commit["previous_commit_sha256"] == prior_sha and commit["plan_sha256"] == plan_sha
                and commit["active"] == active, "Source round chain or active cases differ")
        expected_plan = []
        for i in active:
            original = tasks[i]["prompt_tokens"]
            seen = samples[i]
            expected_plan.append(dict(index=i, start=len(original) + len(seen) - 1 if seen else 0,
                                      tokens=[seen[-1]["token_id"]] if seen else original))
        require(read(plan_path) == dict(round=number, cases=expected_plan, file_identity=freeze["file_identity"]),
                "Source round replaced a prompt or sampled token")
        states = {}
        for first_layer in range(0, 24, step):
            last_layer = min(24, first_layer + step)
            label = f"layers-{first_layer:02d}-{last_layer:02d}"
            fixture, peak = check_phase(round_dir, label, role, number, plan_sha, checkpoint)
            require(fixture["first_layer"] == first_layer and fixture["end_layer"] == last_layer
                    and set(fixture["hidden"]) == {f"{i:03d}.bf16" for i in active}
                    and set(fixture["state"]) == {f"{j:02d}-{i:03d}.bf16" for j in range(first_layer, last_layer)
                                                  for i in active}, "Source layer/case coverage differs")
            states.update(fixture["state"])
            max_peak, phase_count = max(max_peak, peak), phase_count + 1
        head, peak = check_phase(round_dir, "head", role, number, plan_sha, checkpoint)
        max_peak, phase_count = max(max_peak, peak), phase_count + 1
        sample_file = round_dir / "head/samples.jsonl"
        require(states == commit["state"] and head["sample_count"] == len(active)
                and digest(sample_file) == head["samples_sha256"] == commit["samples_sha256"],
                "Source cache or tied-head evidence differs")
        values = list(rows(sample_file))
        require([v["index"] for v in values] == active, "Missing or reordered source decisions")
        for v in values:
            i, token = v["index"], v["token_id"]
            require(type(token) is int and 0 <= token < VOCAB
                    and len(bytes.fromhex(v["logit_sha256"])) == 32 and math.isfinite(v["max_logit"])
                    and len(samples[i]) < tasks[i]["max_output_tokens"], "Invalid source decision")
            samples[i].append(v)
        active = [i for i in active if samples[i][-1]["token_id"] != EOS
                  and len(samples[i]) < tasks[i]["max_output_tokens"]]
        prior_sha = digest(path)
    require(not active and len(records) == end - first, "Source task collection is incomplete")
    for i, record in enumerate(records, first):
        require(record["index"] == i and record["samples"] == samples[i],
                "Source final task differs from the guarded rounds")
    return dict(replayed_cases=len(records), replayed_rounds=len(paths), guarded_phases=phase_count,
                maximum_source_rss_bytes=max_peak, final_round_sha256=prior_sha)


def audit_source_collection(folder, inputs, receipt):
    tasks, _ = task_inputs(inputs, receipt["role"])
    path = folder / "tasks.jsonl"
    require(digest(path) == receipt["tasks_sha256"], "Source task file differs")
    return audit_source(folder, tasks, list(rows(path)), receipt)
