"""Independent raw replay of every native C512 diagnostic row.

Canonical FP32 gates define acceptance. FP64 observations stay diagnostic. No
native computation, reference recalibration or model/tokenizer loading occurs.
"""
import hashlib
import json
import math
import struct

import qwen3_context_precision as precision

FIELDS = {"name", "start", "rows", "columns", "dtype", "offset", "bytes", "sha256"}
POSITIONS, STATES, RAW_BYTES = 512, 57, 504102912


def decode(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate native diagnostic field")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite native diagnostic metadata: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def document(path, limit=8 * 1024**2):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Native diagnostic document exceeds bound")
    return decode(raw)


def chunks(recipe):
    start = 0
    for segment in recipe["segments"]:
        remaining = segment
        while remaining:
            rows = min(recipe["chunk"], remaining)
            yield start, rows
            start += rows; remaining -= rows
    if start != POSITIONS:
        raise ValueError("Native recipe does not cover512 rows")


def frames(path, recipe, manifest):
    with path.open("rb") as file:
        body = file.read(4 * 1024**2 + 1)
    if len(body) > 4 * 1024**2:
        raise ValueError("Native journal exceeds bound")
    records = [decode(line) for line in body.splitlines()]
    names = {name for name in manifest["plan"]["gates"]
             if not name.endswith(("attention_scores", "attention_probabilities"))}
    if len(names) != STATES:
        raise ValueError("Native stage inventory differs")
    cursor, index = 0, 0
    for start, rows in chunks(recipe):
        pending = set(names)
        for _ in range(STATES):
            if index >= len(records):
                raise ValueError("Missing native diagnostic frame")
            record = records[index]; index += 1
            if (set(record) != FIELDS or not isinstance(record["name"], str) or record["name"] not in pending
                    or any(type(record[key]) is not int for key in ("start", "rows", "columns", "offset", "bytes"))):
                raise ValueError("Native frame keyset/duplicate stage differs")
            pending.remove(record["name"])
            shape = manifest["records"]["original.fp32"]["arrays"][record["name"]]["shape"]
            columns = math.prod(shape[1:])
            wanted = dict(start=start, rows=rows, columns=columns, offset=cursor,
                          bytes=rows * columns * 4, dtype="<f4")
            if any(record[key] != value for key, value in wanted.items()):
                raise ValueError("Native frame layout or row coverage differs")
            cursor += record["bytes"]
            yield record
    if index != len(records) or cursor != RAW_BYTES:
        raise ValueError("Extra native frames/bytes")


def statistics():
    return dict(values=0, failed_values=0, nonfinite_values=0, maximum_absolute=0.0,
                maximum_gate_fraction=0.0, first_failures=[])


def observe(result, actual, expected, position, gate):
    import numpy as np
    valid = np.isfinite(actual) & np.isfinite(expected)
    result["values"] += len(actual)
    result["nonfinite_values"] += int(np.count_nonzero(~valid))
    # Match Rust's finite-only metrics without introducing NaN into summaries.
    a = actual[valid]; b = expected[valid]
    if not len(a):
        return
    delta = np.abs(a - b); bound = gate["atol"] + gate["rtol"] * np.abs(b)
    result["maximum_absolute"] = max(result["maximum_absolute"], float(np.max(delta)))
    result["maximum_gate_fraction"] = max(result["maximum_gate_fraction"], float(np.max(delta / bound)))
    failed = np.flatnonzero(delta > bound)
    result["failed_values"] += len(failed)
    indices = np.flatnonzero(valid)
    for index in failed[:max(0, 8 - len(result["first_failures"]))]:
        result["first_failures"].append(dict(position=position, feature=int(indices[index]),
            actual=float(a[index]), expected=float(b[index]), delta=float(delta[index]), bound=float(bound[index])))


def passed(stats):
    return all(v["failed_values"] == 0 and v["nonfinite_values"] == 0 for v in stats.values())


def same_statistic(observed, reported):
    """Examples carry f32 fields; metrics/delta/bound carry binary64 fields.

    Rust serializes f32 examples using shortest round-tripping decimals. Decode
    their declared f32 type before comparing bits; numerical acceptance still
    uses complete raw arrays and unchanged binary64 metric calculations.
    """
    if set(observed) != set(reported):
        return False
    for key in observed:
        if key != "first_failures" and observed[key] != reported[key]:
            return False
    if len(observed["first_failures"]) != len(reported["first_failures"]):
        return False
    for a, b in zip(observed["first_failures"], reported["first_failures"], strict=True):
        if set(a) != set(b):
            return False
        for key in a:
            if key in ("actual", "expected"):
                if any(type(value) not in (int, float) or not math.isfinite(value) for value in (a[key], b[key])):
                    return False
                try:
                    if struct.pack("<f", a[key]) != struct.pack("<f", b[key]):
                        return False
                except (OverflowError, struct.error):
                    return False
            elif a[key] != b[key]:
                return False
    return True


def same_statistics(observed, reported):
    return set(observed) == set(reported) and all(same_statistic(value, reported[name]) for name, value in observed.items())


def integer(value, wanted=None):
    if type(value) is not int or value < 0 or (wanted is not None and value != wanted):
        raise ValueError("Native integer metadata differs")


def boolean(value, wanted=None):
    if type(value) is not bool or (wanted is not None and value is not wanted):
        raise ValueError("Native boolean metadata differs")


def stats_metadata(values):
    keys = set(statistics())
    for value in values.values():
        if set(value) != keys:
            raise ValueError("Native statistics keyset differs")
        for key in ("values", "failed_values", "nonfinite_values"):
            integer(value[key])
        for key in ("maximum_absolute", "maximum_gate_fraction"):
            number = value[key]
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError("Native metric metadata differs")
        if not isinstance(value["first_failures"], list) or len(value["first_failures"]) > 8:
            raise ValueError("Native failure examples exceed bound")
        for point in value["first_failures"]:
            if set(point) != {"position", "feature", "actual", "expected", "delta", "bound"}:
                raise ValueError("Native example keyset differs")
            integer(point["position"]); integer(point["feature"])
            for key in ("actual", "expected", "delta", "bound"):
                if type(point[key]) not in (int, float) or not math.isfinite(point[key]):
                    raise ValueError("Native failure value metadata differs")


def case_metadata(native, recipe, case, budget):
    if native["case"] != case or native["recipe"] != recipe["name"]:
        raise ValueError("Native case/recipe metadata differs")
    integer(native["chunk"], recipe["chunk"])
    if not isinstance(native["segments"], list) or native["segments"] != recipe["segments"]:
        raise ValueError("Native segment metadata differs")
    for segment in native["segments"]:
        integer(segment)
    integer(native["rows_per_state"], POSITIONS); integer(native["native_bytes"], RAW_BYTES)
    integer(native["snapshots"], sum(1 for _ in chunks(recipe)) * STATES)
    integer(native["admitted_bytes"])
    if not 0 < native["admitted_bytes"] <= budget:
        raise ValueError("Native numerical admission differs from frozen budget")
    boolean(native["state_checks_passed"], True)
    boolean(native["canonical_passed"]); boolean(native["fp64_diagnostic_passed"])
    if len(native["argmax_ids"]) != POSITIONS:
        raise ValueError("Native argmax metadata coverage differs")
    for token in native["argmax_ids"]:
        if token is not None:
            integer(token)
            if token >= 151936:
                raise ValueError("Native argmax vocabulary differs")
    for position in native["canonical_argmax_mismatches"]:
        integer(position)
    stats_metadata(native["canonical"]); stats_metadata(native["fp64_diagnostic"])


def aggregate_metadata(summary, plan):
    integer(summary["capacity"], plan["context"])
    integer(summary["stage_snapshots"], plan["total_stage_snapshots"])
    if (summary["manifest_sha256"] != precision.MANIFEST_SHA
            or summary["comparison_sha256"] != precision.COMPARISON_SHA
            or len(summary["recipes"]) != len(plan["recipes"]) * len(plan["cases"])):
        raise ValueError("Native aggregate reference/coverage metadata differs")
    boolean(summary["canonical_fp32_passed"])
    boolean(summary["fp64_is_diagnostic_only"], True); boolean(summary["future_prefix_exact"], True)
    for key in ("source_quality_qualified", "held_out_quality_qualified", "full_model_window_qualified", "phone_qualified", "precision_or_capacity_changed"):
        boolean(summary[key], False)


def contract_metadata(value, plan):
    expected = plan["pending_generation"]
    if value["sequence"] != "original" or value["resumed_ids"] != [expected["first_sample"], expected["continue_one_sample"]]:
        raise ValueError("Native generation sequence metadata differs")
    for token in value["resumed_ids"]:
        integer(token)
    for key, wanted in [("capacity", plan["context"]), ("callback_cancel_prompt", expected["prompt_tokens"]),
                        ("requested_output", expected["requested_output"]), ("emitted_before_cancel", expected["cancel_after_emitted"])]:
        integer(value[key], wanted)
    for key in ("passed", "pending_then_flushed", "rejected_513_state_unchanged", "output_failure_retry", "sampled_history_fresh_replay", "partial_discard_reappend"):
        boolean(value[key], True)
    boolean(value["mid_gpu_cancellation_qualified"], False); boolean(value["canonical_fp32_passed"])
    stats_metadata(dict(canonical=value["canonical"], fp64=value["fp64_diagnostic"]))


def replay_case(folder, label, recipe, case, manifest, frozen, sources, owner, report):
    import numpy as np
    native = document(folder / (label + ".json"))
    names = {name for name in manifest["plan"]["gates"]
             if not name.endswith(("attention_scores", "attention_probabilities"))}
    canonical = {name: statistics() for name in names}
    wide = {name: statistics() for name in names}
    digests = {(contract, name): hashlib.sha256() for contract in ("fp32", "fp64") for name in names}
    prefixes = {name: hashlib.sha256() for name in names}
    actual_digests = {name: hashlib.sha256() for name in names}
    ids, byte_count, snapshots = [], 0, 0
    result = dict(label=label, complete=False, canonical=canonical, fp64_diagnostic=wide)
    report.append(result)
    for frame in frames(folder / (label + ".jsonl"), recipe, manifest):
        name, start, rows, columns = [frame[k] for k in ("name", "start", "rows", "columns")]
        owner.file.seek(frame["offset"])
        items = {contract: manifest["records"][case + "." + contract]["arrays"][name]
                 for contract in ("fp32", "fp64")}
        for contract, item in items.items():
            width = 2 if item["dtype"] == "<f2" else 4
            sources[case + "." + contract].file.seek(item["offset"] + start * columns * width)
        digest = hashlib.sha256()
        for row in range(rows):
            body = owner.file.read(columns * 4)
            if len(body) != columns * 4:
                raise ValueError("Truncated native row")
            digest.update(body); byte_count += len(body)
            actual_digests[name].update(body)
            if start + row < POSITIONS - 1:
                prefixes[name].update(body)
            actual = np.frombuffer(body, dtype="<f4").astype(np.float64)
            for contract, stats in [("fp32", canonical), ("fp64", wide)]:
                item = items[contract]; width = 2 if item["dtype"] == "<f2" else 4
                expected_body = sources[case + "." + contract].file.read(columns * width)
                if len(expected_body) != columns * width:
                    raise ValueError("Truncated oracle row")
                digests[contract, name].update(expected_body)
                expected = np.frombuffer(expected_body, dtype=item["dtype"]).astype(np.float64)
                observe(stats[name], actual, expected, start + row, manifest["plan"]["gates"][name])
            if name == "logits":
                ids.append(int(np.argmax(actual)) if np.isfinite(actual).all() else None)
        if digest.hexdigest() != frame["sha256"]:
            raise ValueError("Consumed native frame bytes differ")
        snapshots += 1
    for (contract, name), digest in digests.items():
        item = manifest["records"][case + "." + contract]["arrays"][name]
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Consumed complete oracle array differs")
        if canonical[name]["values"] != math.prod(item["shape"]) or wide[name]["values"] != math.prod(item["shape"]):
            raise ValueError("Native scalar coverage differs")
    expected_ids = frozen["selections"][case + ".fp32"]["ids"]
    mismatches = [i for i, (a, b) in enumerate(zip(ids, expected_ids, strict=True)) if a != b]
    if (not same_statistics(canonical, native["canonical"]) or not same_statistics(wide, native["fp64_diagnostic"])
            or ids != native["argmax_ids"] or mismatches != native["canonical_argmax_mismatches"]
            or passed(canonical) != native["canonical_passed"] or passed(wide) != native["fp64_diagnostic_passed"]
            or byte_count != native["native_bytes"] or snapshots != native["snapshots"]):
        raise ValueError("Native summary does not replay exactly")
    result.update(complete=True, canonical_passed=passed(canonical), argmax_mismatches=mismatches,
                  bytes=byte_count, snapshots=snapshots, prefix_hashes={name: v.hexdigest() for name, v in prefixes.items()},
                  actual_state_hashes={name: v.hexdigest() for name, v in actual_digests.items()})


def verify(folder, oracle_folder, fixture, plan, final_row, report):
    from contextlib import ExitStack
    import numpy as np
    manifest = precision.reference.load(fixture / "manifest.json", precision.MANIFEST_SHA)
    frozen = precision.reference.load(fixture / "comparison.json", precision.COMPARISON_SHA)
    precision.reference.validate(manifest)
    if final_row["position"] != 511 or final_row["columns"] != 151936 or final_row["dtype"] != "<f4":
        raise ValueError("Pre-target generation source-row scope differs")
    summary = document(folder / "summary.json")
    aggregate_metadata(summary, plan)
    report.update(complete=False, recipes=[], owners=[], canonical_passed=False,
                  fp64_is_diagnostic_only=True, numerical_cpu_envelope_failures_retained=7282)
    with ExitStack() as owners:
        sources = {label: owners.enter_context(precision.Owner(oracle_folder / record["file"], record))
                   for label, record in manifest["records"].items()}
        report["owners"].extend(owner.audit for owner in sources.values())
        for recipe in plan["recipes"]:
            prefix_hashes = []
            for case in plan["cases"]:
                label = recipe["name"] + "-" + case
                native = document(folder / (label + ".json"))
                case_metadata(native, recipe, case, plan["numerical_session_budget"])
                if native["native_bytes"] != RAW_BYTES or not native["state_checks_passed"]:
                    raise ValueError("Native case extent/state evidence differs")
                with precision.Owner(folder / (label + ".bin"), dict(bytes=RAW_BYTES, sha256=native["native_sha256"])) as owner:
                    report["owners"].append(owner.audit)
                    replay_case(folder, label, recipe, case, manifest, frozen, sources, owner, report["recipes"])
                prefix_hashes.append(report["recipes"][-1]["prefix_hashes"])
                if native not in summary["recipes"]:
                    raise ValueError("Native case summary differs from aggregate")
            if prefix_hashes[0] != prefix_hashes[1]:
                raise ValueError("Native future-prefix bytes differ")
        contracts = document(folder / "contracts.json")
        contract_metadata(contracts, plan)
        if not (contracts["passed"] and contracts["sequence"] == "original"
                and contracts["resumed_ids"] == [1879, 0] and contracts["pending_then_flushed"]
                and contracts["output_failure_retry"] and contracts["sampled_history_fresh_replay"]
                and contracts["partial_discard_reappend"] and contracts["rejected_513_state_unchanged"]
                and contracts == summary["contracts"]):
            raise ValueError("Original-only pending/recovery contract differs")
        # This separately frozen source row predates target execution. It binds
        # the one-row generation witness without a self-generated subset hash.
        frames_path = folder / "contracts-final-logits.jsonl"
        with frames_path.open("rb") as file:
            body = file.read(4097)
        entries = body.splitlines()
        if len(body) > 4096 or len(entries) != 1:
            raise ValueError("Generation final-row journal differs")
        frame = decode(entries[0])
        columns = 151936
        if (set(frame) != FIELDS
                or any(type(frame[key]) is not int for key in ("start", "rows", "columns", "offset", "bytes"))
                or any(frame[key] != value for key, value in dict(
                name="logits", start=511, rows=1, columns=columns, dtype="<f4", offset=0, bytes=columns * 4).items())):
            raise ValueError("Generation final-row layout differs")
        with precision.Owner(folder / "contracts-final-logits.bin", dict(bytes=columns * 4, sha256=frame["sha256"])) as owner:
            report["owners"].append(owner.audit)
            owner.file.seek(0); body = owner.file.read(columns * 4)
            actual = np.frombuffer(body, dtype="<f4").astype(np.float64)
            generation = {}
            for contract in ("fp32", "fp64"):
                item = manifest["records"]["original." + contract]["arrays"]["logits"]
                file = sources["original." + contract].file
                file.seek(item["offset"] + 511 * columns * 4)
                expected_body = file.read(columns * 4)
                if len(expected_body) != columns * 4 or hashlib.sha256(expected_body).hexdigest() != final_row[contract]["sha256"]:
                    raise ValueError("Consumed generation oracle row differs")
                expected = np.frombuffer(expected_body, dtype="<f4").astype(np.float64)
                result = statistics(); observe(result, actual, expected, 511, manifest["plan"]["gates"]["logits"])
                key = "canonical" if contract == "fp32" else "fp64_diagnostic"
                if not same_statistic(result, contracts[key]):
                    raise ValueError("Generation final-row statistics differ")
                generation[contract] = result
            report["generation"] = generation
            if passed(dict(logits=generation["fp32"])) != contracts["canonical_fp32_passed"]:
                raise ValueError("Generation canonical acceptance differs")
    if not all(row["endpoint_checked"] and row["checksum_passes"] == 2 and row["endpoint_error"] is None for row in report["owners"]):
        raise ValueError("Native/oracle endpoint authentication incomplete")
    snapshots = sum(row["snapshots"] for row in report["recipes"])
    canonical = all(row["canonical_passed"] and not row["argmax_mismatches"] for row in report["recipes"]) and passed(dict(logits=report["generation"]["fp32"]))
    if (snapshots != 5016 or summary["stage_snapshots"] != snapshots
            or canonical != summary["canonical_fp32_passed"] or not summary["fp64_is_diagnostic_only"]
            or len(summary["recipes"]) != 8 or not summary["future_prefix_exact"]
            or any(summary[key] is not False for key in ("source_quality_qualified", "held_out_quality_qualified", "full_model_window_qualified", "phone_qualified", "precision_or_capacity_changed"))):
        raise ValueError("Native aggregate coverage/acceptance differs")
    report.update(complete=True, canonical_passed=canonical, stage_snapshots=snapshots,
                  all_state_rows=512, numerical_source_quality_qualified=False,
                  full_model_window_qualified=False, phone_qualified=False,
                  cross_schedule_byte_identical={case: len({tuple(sorted(row["actual_state_hashes"].items()))
                      for row in report["recipes"] if row["label"].endswith("-" + case)}) == 1 for case in plan["cases"]})
