"""Stream frozen C512 failures without changing reference math or target gates.

The original asymmetric FP32-vs-FP64 test and the overlap of two target-acceptance
intervals are different questions. Exact rational endpoints with outward guarded
radii prevent floating endpoint rounding from creating a false disjoint claim.
"""
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat

import qwen3_long_context as reference

ELEMENTS = 65536
LIMIT = 512 * 1024**2
MANIFEST_SHA = "21d54e567025f70fdda01d7f1bede1a71f730c6b7c10aec62f5aaffec52a7f08"
COMPARISON_SHA = "1bbd129b5bf1811e6cdc898a433983c95d0547c9aedc05a03ad9c361564ce801"


def intervals(a, b, gate):
    """Inclusive, conservative intervals; overlap is necessary, not sufficient.

    The outward one-ULP guard is diagnostic only, never a changed acceptance
    tolerance. Include the exact algebraic radius as well as the evaluated FP64
    radius. Endpoints/gap use exact saved binary values through Fraction.
    """
    values = [a, b, gate["atol"], gate["rtol"]]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Finite scalar interval inputs required")
    atol, rtol = float(gate["atol"]), float(gate["rtol"])
    if atol <= 0 or rtol < 0:
        raise ValueError("Positive absolute/nonnegative relative gates required")
    def radius(value):
        evaluated = atol + rtol * abs(value)
        outward = math.nextafter(evaluated, math.inf)
        if not math.isfinite(outward):
            raise ValueError("Interval radius overflow")
        exact = Fraction.from_float(atol) + Fraction.from_float(rtol) * abs(Fraction.from_float(value))
        return max(exact, Fraction.from_float(outward)), evaluated
    a, b = float(a), float(b)
    ra, ta = radius(a); rb, tb = radius(b)
    aa, bb = Fraction.from_float(a), Fraction.from_float(b)
    gap = max(aa - ra, bb - rb) - min(aa + ra, bb + rb)
    return dict(disjoint=gap > 0, exact_guarded_gap=str(gap),
                fp32_radius=str(ra), fp64_radius=str(rb),
                fp32_evaluated_tolerance=ta, fp64_evaluated_tolerance=tb)


def coordinates(index, shape):
    if (not isinstance(shape, list) or not shape
            or any(type(width) is not int or width <= 0 for width in shape)
            or type(index) is not int or index < 0 or index >= math.prod(shape)):
        raise ValueError("Coordinate index outside array")
    result = []
    for width in reversed(shape):
        if type(width) is not int or width <= 0:
            raise ValueError("Invalid coordinate shape")
        index, component = divmod(index, width)
        result.append(component)
    return list(reversed(result))


def semantic(name, point):
    if name.endswith(("attention_scores", "attention_probabilities")):
        return dict(query_head=point[0], query_position=point[1], key_position=point[2])
    if name == "logits":
        return dict(token_position=point[0], vocabulary_id=point[1])
    if name.endswith(("q_norm", "q_rotary")):
        return dict(token_position=point[0], query_head=point[1], head_component=point[2])
    if name.endswith(("k_norm", "k_rotary")):
        return dict(token_position=point[0], kv_head=point[1], head_component=point[2])
    return dict(token_position=point[0], feature=point[1])


class Owner:
    """Immutable descriptor/path endpoints and hashes of actual consumed bytes."""
    def __init__(self, path, record):
        self.path, self.record = Path(path), record
        self.file = None
        self.audit = dict(path=str(self.path), checksum_passes=0, endpoint_checked=False, endpoint_error=None)

    @staticmethod
    def identity(value):
        return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def checksum(self):
        self.file.seek(0)
        result = hashlib.sha256()
        remaining = self.record["bytes"]
        while remaining:
            body = self.file.read(min(65536, remaining))
            if not body:
                raise ValueError("Captured file is truncated")
            result.update(body)
            remaining -= len(body)
        if self.file.read(1):
            raise ValueError("Captured file grew beyond admitted extent")
        if result.hexdigest() != self.record["sha256"]:
            raise ValueError("Captured file hash differs")
        self.audit["checksum_passes"] += 1

    def __enter__(self):
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        self.file = os.fdopen(fd, "rb", buffering=0)
        try:
            value = os.fstat(fd)
            self.before = self.identity(value)
            if (not stat.S_ISREG(value.st_mode) or type(self.record["bytes"]) is not int
                    or not 0 < self.record["bytes"] <= 1024**3 or value.st_size != self.record["bytes"]):
                raise ValueError("Captured regular file size differs")
            if self.identity(self.path.lstat()) != self.before:
                raise ValueError("Captured path identity differs")
            self.checksum()
            return self
        except BaseException:
            self.file.close()
            raise

    def blocks(self, item):
        size = item["bytes"]
        if item["dtype"] not in ("<f2", "<f4"):
            raise ValueError("Unsupported captured dtype")
        width = 2 if item["dtype"] == "<f2" else 4
        shape = item["shape"]
        if (not isinstance(shape, list) or not 1 <= len(shape) <= 3
                or any(type(width) is not int or width <= 0 for width in shape)
                or type(size) is not int or size <= 0 or type(item["offset"]) is not int
                or size != math.prod(item["shape"]) * width or item["offset"] < 0
                or item["offset"] + size > self.record["bytes"]):
            raise ValueError("Captured array extent differs")
        self.file.seek(item["offset"])
        digest, remaining, offset = hashlib.sha256(), size, 0
        while remaining:
            body = self.file.read(min(ELEMENTS * width, remaining))
            if not body or len(body) % width:
                raise ValueError("Captured array is truncated")
            digest.update(body); remaining -= len(body)
            yield offset, body
            offset += len(body) // width
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Consumed array bytes differ")

    def __exit__(self, kind, value, traceback):
        try:
            self.checksum()
            if self.identity(os.fstat(self.file.fileno())) != self.before or self.identity(self.path.lstat()) != self.before:
                raise ValueError("Captured file/path endpoint differs")
            self.audit["endpoint_checked"] = True
        except BaseException as error:
            self.audit["endpoint_error"] = type(error).__name__ + ": " + str(error)
            if kind is None:
                raise
            # Preserve the primary read/comparison error as the raised cause.
            value.add_note("Captured endpoint check: " + type(error).__name__ + ": " + str(error))
        finally:
            self.file.close()


def inspect_pair(left, right, name, item, gate, expected, selections, report):
    import numpy as np
    state = dict(name=name, expected_failed_values=expected["failed_values"],
                 failed_values=0, disjoint_values=0, affected_positions=[],
                 authenticated=False, maximum_absolute=0.0, maximum_gate_fraction=0.0, findings=[])
    report.append(state)
    positions = set()
    for (start, a_raw), (other, b_raw) in zip(left.blocks(item[0]), right.blocks(item[1]), strict=True):
        if start != other or len(a_raw) != len(b_raw):
            raise ValueError("Captured pair blocks differ")
        a = np.frombuffer(a_raw, dtype=item[0]["dtype"]).astype(np.float64)
        b = np.frombuffer(b_raw, dtype=item[1]["dtype"]).astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite captured pair")
        delta = np.abs(a - b)
        bound = gate["atol"] + gate["rtol"] * np.abs(b)
        state["maximum_absolute"] = max(state["maximum_absolute"], float(np.max(delta)))
        state["maximum_gate_fraction"] = max(state["maximum_gate_fraction"], float(np.max(delta / bound)))
        for index in np.flatnonzero(delta > bound):
            state["failed_values"] += 1
            if state["failed_values"] > expected["failed_values"]:
                raise ValueError("Failure count exceeds frozen report")
            point = coordinates(start + int(index), item[0]["shape"])
            meaning = semantic(name, point)
            position = meaning.get("token_position", meaning.get("query_position"))
            positions.add(position)
            interval = intervals(float(a[index]), float(b[index]), gate)
            state["disjoint_values"] += int(interval["disjoint"])
            selection = {contract: dict(id=selections[contract]["ids"][position], margin=selections[contract]["margins"][position]) for contract in ("fp32", "fp64")}
            state["findings"].append(dict(coordinates=point, semantic=meaning,
                fp32=float(a[index]), fp64=float(b[index]), absolute_difference=float(delta[index]),
                original_fp64_bound=float(bound[index]), interval=interval, raw_greedy_at_position=selection,
                causally_masked_attention_entry=(meaning["key_position"] > meaning["query_position"]
                    if "key_position" in meaning else None)))
    state["affected_positions"] = sorted(positions)
    for key in ("failed_values", "maximum_absolute", "maximum_gate_fraction"):
        if state[key] != expected[key]:
            raise ValueError("Saved failure summary differs: " + key)
    state["authenticated"] = True
    return state


def analyze(fixture, raw_folder, report):
    manifest = reference.load(fixture / "manifest.json", MANIFEST_SHA)
    comparison = reference.load(fixture / "comparison.json", COMPARISON_SHA)
    reference.validate(manifest)
    failures = {key: value for key, value in comparison["comparisons"].items() if value["failed_values"]}
    if len(failures) != 32 or sum(value["failed_values"] for value in failures.values()) != 7282:
        raise ValueError("Exact frozen discrepancy set required")
    if any(not any(key.startswith(case + ".contracts.") for case in reference.CASES) for key in failures):
        raise ValueError("Unexpected frozen comparison role")
    report.update(arrays=[], cases={}, owners=[], complete=False, numerical_context_qualified=False,
                  interval_interpretation="Conservative inclusive exact-rational intervals with outward one-ULP64 radius guard; overlap is necessary only, without a representability or execution claim.")
    for case in reference.CASES:
        report["active_case"] = case
        with ExitStack() as owners:
            left, right = [owners.enter_context(Owner(raw_folder / manifest["records"][case + "." + contract]["file"], manifest["records"][case + "." + contract])) for contract in reference.CONTRACTS]
            report["owners"].extend([left.audit, right.audit])
            for key, expected in failures.items():
                if not key.startswith(case + ".contracts."):
                    continue
                name = key.split(".contracts.", 1)[1]
                report["active_array"] = key
                items = [manifest["records"][case + "." + contract]["arrays"][name] for contract in reference.CONTRACTS]
                selections = {contract: comparison["selections"][case + "." + contract] for contract in reference.CONTRACTS}
                state = inspect_pair(left, right, name, items, manifest["plan"]["gates"][name], expected, selections, report["arrays"])
                state["case"] = case
    unique_coordinates, unique_pairs = set(), set()
    for case in reference.CASES:
        states = [value for value in report["arrays"] if value["case"] == case]
        report["cases"][case] = dict(failed_values=sum(v["failed_values"] for v in states),
            disjoint_values=sum(v["disjoint_values"] for v in states),
            affected_positions=sorted({p for state in states for p in state["affected_positions"]}))
        for state in states:
            for finding in state["findings"]:
                key = state["name"], tuple(finding["coordinates"])
                unique_coordinates.add(key)
                unique_pairs.add((*key, finding["fp32"].hex(), finding["fp64"].hex()))
    if (len(report["arrays"]) != 32 or sum(v["failed_values"] for v in report["cases"].values()) != 7282
            or not all(v["authenticated"] for v in report["arrays"])
            or not all(v["endpoint_checked"] and v["checksum_passes"] == 2 and v["endpoint_error"] is None for v in report["owners"])):
        raise ValueError("Incomplete authenticated discrepancy report")
    report.update(complete=True, case_specific_failures=sum(v["failed_values"] for v in report["cases"].values()),
                  disjoint_case_values=sum(v["disjoint_values"] for v in report["cases"].values()),
                  unique_stage_coordinates=len(unique_coordinates), unique_coordinate_value_pairs=len(unique_pairs))
