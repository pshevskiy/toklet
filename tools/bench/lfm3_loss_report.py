"""Reconcile complete or in-progress LFM3 loss campaigns without changing the gate."""
import argparse
import json
import math
from pathlib import Path
import struct

from lfm3_quality_inputs import INPUTS_SHA256, VOCAB, digest, read_bound, require

LOSS_CONTRACT_SHA = "614fdb8d1988f959304346328ca075b05dea60ab0b3d22c4af271eca14624723"
CHECKPOINTS = {
    "stock": "91983df3e2f1ae164cb714daff3c5094cc17c8997a876f3e97b52711149bda70",
    "modified_parent": "b6ff17ac993dddf0f6f51bd872b12ae1437f4453a54a0b4878cedc7864fbc07e",
}
TARGETS = 10000
LIMIT = 0.05


def read(path):
    return json.loads(path.read_bytes())


def score_rows(path, expected, ids, first, end):
    records = [json.loads(line) for line in read_bound(path, expected).splitlines()]
    require(len(records) == end - first, "Missing or extra held-out predictions")
    for offset, row in enumerate(records, first):
        nll = row["nll"]
        require(row["target_index"] == offset and row["input_id"] == ids[offset]
                and row["target_id"] == ids[offset + 1] and type(nll) in (int, float)
                and math.isfinite(nll) and nll >= 0
                and type(row["greedy_id"]) is int and 0 <= row["greedy_id"] < VOCAB,
                "Held-out prediction identity or value differs")
    return records


def reference_blocks(folder, blocks, role, ids):
    found = {}
    for block in blocks:
        index = block["block"]
        base = folder / "source-loss" / role / f"{index:02d}"
        stages = ["layers-0", "layers-1", "layers-2", "head"]
        receipts = [folder / f"attested-loss-{role}-{index:02d}-{phase}-checks.json" for phase in stages]
        if any(not path.is_file() for path in receipts):
            continue
        previous = None
        for phase, receipt in zip(stages, receipts, strict=True):
            guard = read(receipt)
            fixture_path = base / phase / "fixture.json"
            require(guard["passed"] and guard["inputs_unchanged"] and guard["exit_code"] == 0
                    and guard["fixture_sha256"] == digest(fixture_path)
                    and guard["runtime_before_files"] == guard["runtime_after_files"] == 28982
                    and guard["conservative_final_peak_rss_sum_bytes"] <= 512 * 1024**2
                    and guard["max_sampled_physical_bytes"] <= 512 * 1024**2
                    and guard["minimum_free_bytes"] >= 2 * 1024**3,
                    "Source phase guard did not qualify")
            evidence = read(fixture_path)
            require(evidence["profile"] == role and evidence["block"] == block
                    and evidence["phase"] == phase and evidence["checkpoint_sha256"] == CHECKPOINTS[role]
                    and evidence["source_loss_contract_sha256"] == LOSS_CONTRACT_SHA
                    and evidence.get("previous_hidden_sha256") == previous, "Source phase continuity differs")
            previous = evidence.get("hidden_sha256")
            if previous:
                require(digest(fixture_path.parent / "hidden.bf16le") == previous, "Source hidden payload differs")
        path = base / "head/scores.jsonl"
        data = score_rows(path, evidence["scores_sha256"], ids, block["start"], block["end"])
        require(math.fsum(row["nll"] for row in data) == evidence["nll_sum"], "Source NLL sum differs")
        found.update((row["target_index"], row) for row in data)
    return found


def native_windows(windows, ids):
    found = {}
    for window in windows:
        receipt = read(window / "guard.json")
        require(receipt["passed"] and receipt["exit_code"] == 0 and receipt["inputs_unchanged"]
                and receipt["sample_peaks_consistent"] and receipt["conservative_peak_sum_bytes"] <= 6 * 1024**3
                and receipt["minimum_free_bytes"] >= 2 * 1024**3, "Native window did not qualify")
        if (window / "comparison.json").is_file():
            report = read(window / "comparison.json")
            first, end = 0, 1024
            expected = report["native_scores_sha256"]
        else:
            first, end = receipt["target_start"], receipt["target_end"]
            expected = receipt["scores_sha256"]
        require(0 <= first < end <= TARGETS and first % 1024 == 0 and (end % 1024 == 0 or end == TARGETS),
                "Native window cuts a frozen loss block")
        data = score_rows(window / "scores.jsonl", expected, ids, first, end)
        require(not set(range(first, end)) & found.keys(), "Overlapping native loss windows")
        found.update((row["target_index"], row) for row in data)
    return found


def comparison(baseline, candidate):
    overlap = sorted(baseline.keys() & candidate.keys())
    if not overlap:
        return dict(paired_targets=0, complete=False, gate_passed=None)
    delta = [candidate[i]["nll"] - baseline[i]["nll"] for i in overlap]
    mean_delta = math.fsum(delta) / len(delta)
    complete = overlap == list(range(TARGETS))
    return dict(paired_targets=len(overlap), complete=complete,
                baseline_mean_nll=math.fsum(baseline[i]["nll"] for i in overlap) / len(overlap),
                candidate_mean_nll=math.fsum(candidate[i]["nll"] for i in overlap) / len(overlap),
                mean_nll_increase=mean_delta, maximum_absolute_nll_delta=max(map(abs, delta)),
                greedy_differences=sum(baseline[i]["greedy_id"] != candidate[i]["greedy_id"] for i in overlap),
                gate_passed=(mean_delta <= LIMIT) if complete else None)


def run(args):
    folder = args.inputs
    reconciliation = json.loads(read_bound(folder / "reconciliation.json", INPUTS_SHA256))
    policy = json.loads(read_bound(folder / "source-loss-contract.json", LOSS_CONTRACT_SHA))
    require(policy["target_count"] == TARGETS and policy["maximum_mean_nll_increase"] == LIMIT
            and policy["input_reconciliation_sha256"] == INPUTS_SHA256, "Original loss gate differs")
    raw = read_bound(folder / "stock/held_out_loss.u32le", reconciliation["loss_ids_sha256"])
    ids = tuple(row[0] for row in struct.iter_unpack("<I", raw))
    require(len(ids) == TARGETS + 1 and all(v < VOCAB for v in ids), "Invalid held-out token extent")
    blocks = reconciliation["loss_blocks"]
    require(len(blocks) == 10 and blocks[-1]["end"] == TARGETS, "Original blocks differ")
    stock = reference_blocks(folder, blocks, "stock", ids)
    parent = reference_blocks(folder, blocks, "modified_parent", ids)
    native = native_windows(args.native_window, ids)
    result = dict(format="toklet-lfm3-loss-report-v1", threshold_nats_per_token=LIMIT,
                  counts=dict(stock=len(stock), modified_parent=len(parent), native=len(native)),
                  stock_to_modified=comparison(stock, parent), modified_to_native=comparison(parent, native),
                  full_task_quality_scored=False)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--native-window", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
