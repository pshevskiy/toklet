"""Experimental CPU-only calibration capture; validation and native stay unopened.

The unchanged independent graph produces one complete result at a time. This
module captures references only: gate derivation/review is a later operation.
"""
import gc
import hashlib
import json
import math
from pathlib import Path
import struct

import qwen3_context_inputs as inputs
import qwen3_long_context as reference

INPUT_SHA = "959fb8bf1de469c2d8036bf79698664ef829a72d26a81759e4a426ddd2f4bd09"
LIMIT = 6 * 1024**3


def selected_inputs(folder):
    manifest = reference.load(Path(folder) / "manifest.json", INPUT_SHA)
    inputs.validate(manifest["recipe"])
    if (manifest.get("complete") is not True or manifest.get("adopted") is not False
            or manifest.get("numerical_gates_derived") is not False):
        raise ValueError("Frozen experimental inputs required")
    selected = []
    for row in manifest["rows"]:
        if row["split"] != "calibration":
            continue  # Never read a validation token array in this task.
        if row["file"] != row["id"] + ".u32le" or Path(row["file"]).name != row["file"]:
            raise ValueError("Calibration filename differs")
        path = Path(folder) / row["file"]
        with path.open("rb") as file:
            body = file.read(512 * 4 + 1)
        if len(body) != row["tokens"] * 4 or inputs.identity(body) != row["token_array"]:
            raise ValueError("Calibration input bytes differ")
        selected.append(dict(id=row["id"], rows=row["tokens"], tokens=list(struct.unpack("<" + "I" * row["tokens"], body)),
                             source=row["source"], token_array=row["token_array"]))
    if len(selected) != 6 or sorted(row["rows"] for row in selected) != inputs.COUNTS["calibration"]:
        raise ValueError("Complete calibration split required")
    return manifest, selected


def plan(folder):
    manifest, rows = selected_inputs(folder)
    shapes = {str(count): {name: dict(shape=shape, dtype=dtype) for name, (shape, dtype) in reference.geometry(count).items()}
              for count in (32, 128, 512)}
    per_case = {count: sum(math.prod(shape) * (2 if dtype == "<f2" else 4)
                          for shape, dtype in reference.geometry(count).values()) for count in (32, 128, 512)}
    return dict(format="toklet-qwen3-context-calibration-capture-v2", input_manifest_sha256=INPUT_SHA,
        cases=rows, contracts=["fp32", "fp64"], shapes=shapes,
        profile=manifest["profile"], recipe=manifest["recipe"], stage_pooling=manifest["stage_pooling"],
        calibration_prefix_pairs=manifest["calibration_prefix_pairs"],
        per_graph_bytes={str(count): size for count, size in per_case.items()},
        total_capture_bytes=2 * sum(per_case[row["rows"]] for row in rows),
        # The established conservative C512 ledger counts one returned graph,
        # source/table ownership, temporary BLAS matrices and four mappings.
        memory=reference.ledger(), limit=LIMIT, native_executed=False, validation_evaluated=False,
        gates_derived=False, adopted=False, quality_qualified=False, phone_qualified=False)


def save_graph(folder, label, values, rows, record):
    import numpy as np
    geometry = reference.geometry(rows)
    record.update(file=label + ".bin", arrays={}, bytes=0, complete=False)
    if set(values) != set(geometry):
        raise ValueError("Calibration graph boundary inventory differs")
    whole = hashlib.sha256()
    try:
        with (Path(folder) / record["file"]).open("xb") as file:
            for name in sorted(geometry):
                value = values[name];shape, dtype = geometry[name]
                if (list(value.shape) != shape or value.dtype != np.dtype(dtype)
                        or not value.flags.c_contiguous):
                    raise ValueError("Calibration shape/storage contract differs: " + name)
                raw = memoryview(value).cast("B");digest = hashlib.sha256()
                item = dict(shape=shape, dtype=dtype, offset=record["bytes"], intended_bytes=len(raw),
                            bytes=0, complete=False, nonfinite_values=0, sha256=digest.hexdigest())
                record["arrays"][name] = item
                try:
                    for start in range(0, len(raw), 65536):
                        body = raw[start:start + 65536]
                        written = file.write(body)
                        if type(written) is not int or not 0 <= written <= len(body):
                            raise OSError("Invalid reference write count")
                        item["bytes"] += written;record["bytes"] += written
                        digest.update(body[:written]);whole.update(body[:written])
                        if written != len(body):
                            raise OSError("Incomplete reference write")
                        item["nonfinite_values"] += int(np.count_nonzero(~np.isfinite(np.frombuffer(body, dtype=dtype))))
                    item["complete"] = True
                finally:
                    item["sha256"] = digest.hexdigest()
                # Retain the complete returned failing array before rejection.
                if item["nonfinite_values"]:
                    raise ValueError("Calibration nonfinite saved values: " + name)
        record["complete"] = True
    finally:
        record["sha256"] = whole.hexdigest()


def verify_capture(folder, manifest_sha, expected_plan, report):
    """Stream every saved byte/value without a model owner or resident graphs."""
    import numpy as np
    from qwen3_context_precision import Owner
    folder = Path(folder)
    manifest = reference.load(folder / "manifest.json", manifest_sha)
    report.update(complete=False, files=[], arrays=[], values=0, bytes=0)
    expected = {case["id"] + "." + contract: (case, contract)
                for case in expected_plan["cases"] for contract in ("fp32", "fp64")}
    if (manifest.get("plan") != expected_plan or manifest.get("complete") is not True
            or manifest.get("primary_error") is not None or set(manifest["records"]) != set(expected)
            or manifest["attempts"] != [dict(label=label, completed=True) for label in expected]):
        raise ValueError("Complete calibration manifest/attempts differ")
    for label, (case, contract) in expected.items():
        record = manifest["records"][label]
        geometry = reference.geometry(case["rows"])
        if (record.get("case") != case["id"] or record.get("contract") != contract
                or type(record.get("rows")) is not int or record["rows"] != case["rows"]
                or record.get("file") != label + ".bin" or record.get("complete") is not True
                or set(record["arrays"]) != set(geometry)):
            raise ValueError("Calibration record identity/coverage differs")
        owner = Owner(folder / record["file"], dict(bytes=record["bytes"], sha256=record["sha256"]))
        report["files"].append(owner.audit)
        cursor = 0
        with owner:
            for name in sorted(geometry):
                item = record["arrays"][name];shape, dtype = geometry[name]
                size = math.prod(shape) * (2 if dtype == "<f2" else 4)
                if (set(item) != {"shape", "dtype", "offset", "intended_bytes", "bytes", "complete", "nonfinite_values", "sha256"}
                        or item["shape"] != shape or item["dtype"] != dtype
                        or any(type(item[key]) is not int for key in ("offset", "intended_bytes", "bytes", "nonfinite_values"))
                        or item["offset"] != cursor or item["bytes"] != size or item["intended_bytes"] != size
                        or item["complete"] is not True or item["nonfinite_values"] != 0):
                    raise ValueError("Calibration array schema/extent differs")
                observation = dict(label=label, name=name, values=0, hash_checked=False)
                report["arrays"].append(observation)
                for _, body in owner.blocks(item):
                    values = np.frombuffer(body, dtype=dtype)
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite captured calibration value")
                    observation["values"] += len(values);report["values"] += len(values)
                observation["hash_checked"] = True
                cursor += size
            if cursor != record["bytes"]:
                raise ValueError("Calibration full-file extent differs")
            report["bytes"] += cursor
    if report["bytes"] != expected_plan["total_capture_bytes"]:
        raise ValueError("Complete calibration readback byte demand differs")
    report["complete"] = True
    return manifest["records"]


def export(folder, input_folder, report):
    """Retain each attempt immediately, including partial descriptors on failure."""
    import numpy as np
    import mlx_qwen3_real as source
    import qwen3_fp16
    import native_usage
    folder = Path(folder)
    prepared = plan(input_folder)
    if np.__version__ != "2.3.2" or prepared["memory"]["admitted"] > LIMIT:
        raise ValueError("Calibration runtime/memory admission differs")
    folder.mkdir()
    report.update(plan=prepared, complete=False, primary_error=None, records={}, attempts=[], usage=[])
    numerical = dict(num_layers=28, num_query_heads=16, num_kv_heads=8, head_dim=128,
                     rms_norm_eps=1e-6, rope_theta=1e6)
    with (folder / "plan.json").open("x") as file:
        json.dump(prepared, file, indent=2);file.write("\n")
    primary = None
    try:
        with source.Weights(source.MODEL) as weights:
            for case in prepared["cases"]:
                for contract, precision in (("fp32", np.float32), ("fp64", np.float64)):
                    label = case["id"] + "." + contract
                    attempt = dict(label=label, completed=False)
                    report["attempts"].append(attempt)
                    record = dict(case=case["id"], contract=contract, rows=case["rows"])
                    report["records"][label] = record
                    values = qwen3_fp16.forward(case["tokens"], 0, weights, numerical, precision, [0, 1, 27])
                    save_graph(folder, label, values, case["rows"], record)
                    del values
                    gc.collect()
                    usage = native_usage.usage()
                    report["usage"].append(dict(label=label, native=usage))
                    if not 0 < usage["lifetime_max_phys_footprint"] <= LIMIT:
                        raise ValueError("Calibration physical memory limit exceeded")
                    attempt["completed"] = True
                    with (folder / (label + ".json")).open("x") as file:
                        json.dump(report, file, indent=2, allow_nan=False);file.write("\n")
                    print(json.dumps(dict(captured=label, bytes=record["bytes"])), flush=True)
            weights.verify()
        if sum(row["bytes"] for row in report["records"].values()) != prepared["total_capture_bytes"]:
            raise ValueError("Complete calibration byte demand differs")
        report["complete"] = True
    except BaseException as error:
        primary = error;report["primary_error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        try:
            with (folder / "manifest.json").open("x") as file:
                json.dump(report, file, indent=2, allow_nan=False);file.write("\n")
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("Calibration partial manifest: " + type(error).__name__ + ": " + str(error))
    return report
