"""Calibration-only distributions: original quantized source versus CPU graphs.

These descriptive results neither derive tolerances nor replace held-out quality.
All logits are authenticated saved FP32 values, including exact source widening.
"""
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path

import qwen3_context_calibration as calibration
from qwen3_context_precision import Owner

CPU_SHA = "e7e65c2b6f578bd31478085dcf9221aa0c8776ae8c115cb9574782249c59af81"
PAIRS = (("source", "fp32"), ("source", "fp64"), ("fp32", "fp64"))


def observe(rows, target):
    import numpy as np
    if set(rows) != {"source", "fp32", "fp64"}:
        raise ValueError("Exact three-profile comparison required")
    shape = rows["source"].shape
    if (len(shape) != 1 or not 2 <= shape[0] <= 151936
            or target is not None and (type(target) is not int or not 0 <= target < shape[0])):
        raise ValueError("Finite distribution/target shape required")
    profiles, probabilities, logs, wide = {}, {}, {}, {}
    for name, row in rows.items():
        if row.shape != shape or row.dtype != np.dtype("<f4") or not np.isfinite(row).all():
            raise ValueError("Saved finite FP32 distributions required")
        x = row.astype(np.float64);first = int(np.argmax(x));other = x.copy();other[first] = -np.inf
        second = int(np.argmax(other));shifted = x - x[first]
        logp = shifted - math.log(float(np.sum(np.exp(shifted), dtype=np.float64)))
        profiles[name] = dict(greedy=first, second=second, margin=float(x[first] - x[second]),
            target_logit=None if target is None else float(x[target]),
            negative_log_likelihood=None if target is None else float(-logp[target]))
        probabilities[name], logs[name], wide[name] = np.exp(logp), logp, x
    pairs = {}
    for left, right in PAIRS:
        a, b = wide[left], wide[right];delta = np.abs(a - b);index = int(np.argmax(delta))
        p, q = probabilities[left], probabilities[right]
        pairs[left + "." + right] = dict(maximum_absolute=float(delta[index]), maximum_vocabulary_id=index,
            left_logit=float(a[index]), right_logit=float(b[index]),
            kl_left_right=float(np.sum(p * (logs[left] - logs[right]), dtype=np.float64)),
            kl_right_left=float(np.sum(q * (logs[right] - logs[left]), dtype=np.float64)),
            total_variation=float(0.5 * np.sum(np.abs(p - q), dtype=np.float64)),
            argmax_equal=profiles[left]["greedy"] == profiles[right]["greedy"],
            target_nll_difference=None if target is None else profiles[left]["negative_log_likelihood"] - profiles[right]["negative_log_likelihood"])
    return dict(profiles=profiles, comparisons=pairs)


def compare(source_folder, source_sha, cpu_folder, cpu_fixture, input_folder, journal, report):
    import numpy as np
    source = calibration.reference.load(Path(source_folder) / "manifest.json", source_sha)
    cpu = calibration.reference.load(Path(cpu_fixture) / "manifest.json", CPU_SHA)
    _, cases = calibration.selected_inputs(input_folder)
    if (source.get("complete") is not True or source.get("primary_error") is not None
            or source.get("cleanup_errors") != [] or source.get("input_manifest_sha256") != calibration.INPUT_SHA
            or source["model_pins"] != cpu["plan"]["profile"]["required_model_files"]
            or set(source["records"]) != {case["id"] for case in cases}):
        raise ValueError("Matched complete original-source calibration required")
    report.update(complete=False, cases=[], parents=[], journal=dict(bytes=0, complete_rows=0),
        calibration_only=True, held_out_quality=False, numerical_qualified=False, adopted=False,
        accumulation="NumPy FP64 stable log-sum-exp/reductions; descriptive metrics, no gates",
        overlapping_prefix_cases=True, source_dtypes=sorted({v["source_dtype"] for v in source["records"].values()}))
    digest = hashlib.sha256()
    try:
        with Path(journal).open("xb") as output:
            for case in cases:
                label, n = case["id"], case["rows"];columns = 151936
                summary = dict(case=label, rows=0, targets=0, complete=False, profiles={}, comparisons={})
                report["cases"].append(summary)
                losses = {name: [] for name in ("source", "fp32", "fp64")}
                pair_rows = {a + "." + b: [] for a, b in PAIRS}
                with ExitStack() as stack:
                    owners, items, hashes = {}, {}, {}
                    for name in losses:
                        if name == "source":
                            item = source["records"][label]["widened"];path = Path(source_folder) / item["file"]
                            record = dict(bytes=item["bytes"], sha256=item["sha256"])
                            if item["bytes"] != n * columns * 4 or item["dtype"] != "<f4":
                                raise ValueError("Source widened logit extent differs")
                            items[name] = dict(offset=0, sha256=item["sha256"])
                        else:
                            record = cpu["records"][label + "." + name];item = record["arrays"]["logits"]
                            path = Path(cpu_folder) / record["file"]
                            if item["shape"] != [n, columns] or item["dtype"] != "<f4":
                                raise ValueError("CPU logit shape/storage differs")
                            items[name] = item
                        owner = Owner(path, dict(bytes=record["bytes"], sha256=record["sha256"]))
                        report["parents"].append(owner.audit);owners[name] = stack.enter_context(owner)
                        owner.file.seek(items[name]["offset"]);hashes[name] = hashlib.sha256()
                    for position in range(n):
                        rows = {}
                        for name, owner in owners.items():
                            body = owner.file.read(columns * 4)
                            if len(body) != columns * 4:
                                raise ValueError("Calibration logit row truncated")
                            hashes[name].update(body);rows[name] = np.frombuffer(body, dtype="<f4")
                        target = case["tokens"][position + 1] if position + 1 < n else None
                        value = observe(rows, target)
                        encoded = (json.dumps(dict(case=label, position=position, target=target, **value), separators=(",", ":"), allow_nan=False) + "\n").encode()
                        written = output.write(encoded)
                        if type(written) is int and 0 <= written <= len(encoded):
                            report["journal"]["bytes"] += written;digest.update(encoded[:written])
                        if written != len(encoded):
                            raise OSError("Incomplete comparison journal write")
                        report["journal"]["complete_rows"] += 1;summary["rows"] += 1
                        if target is not None:
                            summary["targets"] += 1
                            for name in losses:losses[name].append(value["profiles"][name]["negative_log_likelihood"])
                        for name in pair_rows:pair_rows[name].append(dict(position=position, **value["comparisons"][name]))
                    for name, h in hashes.items():
                        if h.hexdigest() != items[name]["sha256"]:
                            raise ValueError("Consumed full logit-array hash differs")
                summary["profiles"] = {name: dict(mean_diagnostic_nll=math.fsum(values) / len(values)) for name, values in losses.items()}
                for name, values in pair_rows.items():
                    worst = max(values, key=lambda v: v["maximum_absolute"])
                    summary["comparisons"][name] = dict(argmax_disagreements=sum(not v["argmax_equal"] for v in values),
                        maximum_absolute=worst, mean_kl_left_right=math.fsum(v["kl_left_right"] for v in values) / n,
                        mean_kl_right_left=math.fsum(v["kl_right_left"] for v in values) / n,
                        mean_total_variation=math.fsum(v["total_variation"] for v in values) / n)
                summary["complete"] = True
                print(json.dumps(dict(compared=label, rows=n, targets=summary["targets"])), flush=True)
        report["complete"] = True
    finally:
        report["journal"]["sha256"] = digest.hexdigest()
    return report
