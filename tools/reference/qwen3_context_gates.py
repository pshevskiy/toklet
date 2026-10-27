"""Derive experimental gates solely from frozen, CPU-only calibration bytes.

Native output, historical C512 failures and validation arrays are not inputs.
The resulting envelope remains experimental and requires review before use.
"""
from contextlib import ExitStack
import hashlib
import math
from pathlib import Path

import qwen3_context_calibration as calibration
from qwen3_context_precision import Owner, coordinates

CALIBRATION_SHA = "e7e65c2b6f578bd31478085dcf9221aa0c8776ae8c115cb9574782249c59af81"


def gate(cross_precision, prefix):
    for value in (cross_precision, prefix):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Finite nonnegative calibration errors required")
    atol = max(1e-4, 2 * cross_precision, 2 * prefix)
    if not math.isfinite(atol):
        raise ValueError("Calibration envelope overflow")
    return dict(atol=atol, rtol=1e-3,
                calibration_fp32_fp64_max_abs=cross_precision, equivalent_prefix_max_abs=prefix)


def stream(owner, item, size, audit):
    """A prefix is authenticated by its immutable full parent and consumed hash.

    Full-array reads additionally match the frozen array hash. A shorter prefix
    deliberately does not pretend to match a full-array hash.
    """
    width = {"<f2": 2, "<f4": 4}.get(item.get("dtype"))
    if (width is None or type(size) is not int or size <= 0 or size % width
            or type(item.get("offset")) is not int or item["offset"] < 0
            or type(item.get("bytes")) is not int or not size <= item["bytes"]
            or item["offset"] + item["bytes"] > owner.record["bytes"]):
        raise ValueError("Calibration consumed extent differs")
    owner.file.seek(item["offset"])
    remaining = size;digest = hashlib.sha256()
    audit.update(bytes=0, complete=False, full_array=size == item["bytes"])
    try:
        while remaining:
            body = owner.file.read(min(65536, remaining))
            if not body or len(body) % width:
                raise ValueError("Calibration span truncated")
            remaining -= len(body);audit["bytes"] += len(body);digest.update(body)
            yield body
        if size == item["bytes"] and digest.hexdigest() != item["sha256"]:
            raise ValueError("Consumed calibration array hash differs")
        audit["complete"] = True
    finally:
        audit["sha256"] = digest.hexdigest()


def maximum(left, a, right, b, audit):
    import numpy as np
    shape = a["shape"]
    if (a["dtype"] != b["dtype"] or shape[1:] != b["shape"][1:] or shape[0] > b["shape"][0]
            or any(type(dim) is not int or dim <= 0 for dim in [*shape, *b["shape"]])):
        raise ValueError("Calibration prefix shape/storage differs")
    width = 2 if a["dtype"] == "<f2" else 4
    size = math.prod(shape) * width
    if a["bytes"] != size or b["bytes"] != math.prod(b["shape"]) * width:
        raise ValueError("Calibration shape/byte extent differs")
    audit.update(maximum_absolute=0.0, maximum_example=None, values=0, complete=False, left={}, right={})
    for aa, bb in zip(stream(left, a, size, audit["left"]), stream(right, b, size, audit["right"]), strict=True):
        x, y = (np.frombuffer(body, dtype=a["dtype"]).astype(np.float64) for body in (aa, bb))
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite calibration comparison")
        delta = np.abs(x - y);index = int(np.argmax(delta));peak = float(delta[index])
        if audit["maximum_example"] is None or peak > audit["maximum_absolute"]:
            audit["maximum_absolute"] = peak
            audit["maximum_example"] = dict(index=audit["values"] + index,
                coordinates=coordinates(audit["values"] + index, shape), left_value=float(x[index]),
                right_value=float(y[index]), absolute_difference=peak)
        audit["values"] += x.size
    audit["complete"] = True
    return audit["maximum_absolute"]


def derive(folder, fixture, report):
    """Read only the frozen calibration capture; retain partial findings."""
    manifest = calibration.reference.load(Path(fixture) / "manifest.json", CALIBRATION_SHA)
    plan = manifest["plan"]
    if (manifest.get("complete") is not True or manifest.get("primary_error") is not None
            or plan.get("validation_evaluated") is not False or plan.get("gates_derived") is not False):
        raise ValueError("Complete calibration-only capture required")
    names = list(plan["stage_pooling"])
    if any(plan["stage_pooling"][name] != name for name in names):
        raise ValueError("Exact state/layer pooling required")
    report.update(complete=False, parents=[], comparisons=[], gates={}, calibration_manifest_sha256=CALIBRATION_SHA,
        native_executed=False, validation_evaluated=False, adopted=False, numerical_qualified=False,
        quality_qualified=False, gates_derived=False, envelope_ceiling="pending independent review")
    pooled = {name: dict(cross_precision=0.0, prefix=0.0) for name in names}
    witnesses = {name: {} for name in names}
    records = manifest["records"]

    def compare(left_label, right_label, kind):
        with ExitStack() as stack:
            owners = []
            for label in (left_label, right_label):
                record = records[label]
                if record["file"] != label + ".bin" or Path(record["file"]).name != record["file"]:
                    raise ValueError("Calibration record path differs")
                owner = Owner(Path(folder) / record["file"], dict(bytes=record["bytes"], sha256=record["sha256"]))
                report["parents"].append(owner.audit);owners.append(stack.enter_context(owner))
            for name in names:
                observation = dict(left_label=left_label, right_label=right_label, kind=kind, name=name)
                report["comparisons"].append(observation)
                value = maximum(owners[0], records[left_label]["arrays"][name], owners[1],
                                records[right_label]["arrays"][name], observation)
                if kind not in witnesses[name] or value > pooled[name][kind]:
                    pooled[name][kind] = value
                    witnesses[name][kind] = dict(left=left_label, right=right_label, **observation["maximum_example"])

    for case in plan["cases"]:
        compare(case["id"] + ".fp32", case["id"] + ".fp64", "cross_precision")
    for pair in plan["calibration_prefix_pairs"]:
        for contract in ("fp32", "fp64"):
            compare(pair["short"] + "." + contract, pair["long"] + "." + contract, "prefix")
    report["gates"] = {name: gate(values["cross_precision"], values["prefix"]) for name, values in pooled.items()}
    report.update(complete=True, gates_derived=True, maximum_witnesses=witnesses)
    return dict(format="toklet-qwen3-context-gates-experimental-v2", calibration_manifest_sha256=CALIBRATION_SHA,
        input_manifest_sha256=plan["input_manifest_sha256"], input_proposal=plan["recipe"]["proposal"],
        profile=plan["profile"], gates=report["gates"], complete=True, adopted=False,
        gates_derived=True, review_status="pending", maximum_witnesses=witnesses,
        envelope_ceiling="pending independent review", numerical_qualified=False, source_parity_qualified=False,
        quality_qualified=False, validation_evaluated=False, precision_or_capacity_changed=False)
