"""Selected, exact same-input Q/K/V diagnosis; never a qualification gate.

Original 8-bit packed rows reconstruct FP32 weights once. Saved Half input
rows then feed an exact rational dot and a single nearest-even Half rounding.
The native/reference discrepancy separates into local arithmetic, inherited
input differences, and reference arithmetic. No graph or model is instantiated.
"""
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import struct

from qwen3_context_precision import Owner


def rational(value):
    if not math.isfinite(value):
        raise ValueError("Nonfinite saved value")
    return Fraction.from_float(value)


def rounded(value, kind):
    """Nearest-even binary16/32, including midpoint double-rounding controls.

    Float64 conversion only seeds a three-candidate search. Candidate distances
    are exact rationals, so conversion cannot decide a midpoint incorrectly.
    This diagnostic deliberately excludes overflow, rather than specifying a
    general converter's infinity behavior. Exact cancellation publishes +0.
    """
    if kind not in ("e", "f") or not isinstance(value, Fraction):
        raise ValueError("Expected rational binary16/32 input")
    width, limit = (2, 0x7bff) if kind == "e" else (4, 0x7f7fffff)
    fmt = "<" + kind
    maximum = struct.unpack(fmt, limit.to_bytes(width, "little"))[0]
    if abs(value) > rational(maximum):
        raise ValueError("Finite rounding range exceeded")
    approximate = struct.pack(fmt, float(abs(value)))
    seed = int.from_bytes(approximate, "little")
    candidates = range(max(0, seed - 1), min(limit, seed + 1) + 1)
    best = min(candidates, key=lambda bits: (
        abs(rational(struct.unpack(fmt, bits.to_bytes(width, "little"))[0]) - abs(value)),
        bits & 1))
    if value < 0:
        best |= 1 << (width * 8 - 1)
    return struct.unpack(fmt, best.to_bytes(width, "little"))[0]


def finite_values(body, kind, count):
    if len(body) != struct.calcsize("<" + kind) * count:
        raise ValueError("Saved row length differs")
    result = list(struct.unpack("<" + kind * count, body))
    for value in result:
        rational(value)
    return result


def half_expanded(body, count):
    values = finite_values(body, "f", count)
    for value in values:
        try:
            restored = struct.unpack("<e", struct.pack("<e", value))[0]
        except OverflowError as error:
            raise ValueError("Native row is not Half-expanded") from error
        if struct.pack("<f", restored) != struct.pack("<f", value):
            raise ValueError("Native row is not Half-expanded")
    return values


def reconstruct(codes, scales, biases):
    """Four LSB-first bytes per LE U32; BF16 metadata for each 64 columns."""
    if len(codes) != 1024 or len(scales) != 32 or len(biases) != 32:
        raise ValueError("Packed row geometry differs")
    def metadata(body):
        values = [struct.unpack("<f", struct.pack("<I", word << 16))[0]
                  for (word,) in struct.iter_unpack("<H", body)]
        return [rational(value) for value in values]
    ss, bb = metadata(scales), metadata(biases)
    return [rounded(ss[index // 64] * code + bb[index // 64], "f")
            for index, code in enumerate(codes)]


def midpoint_distances(value):
    """Exact signed distances to the Half interval around the rounded dot."""
    center = rounded(value, "e")
    bits = int.from_bytes(struct.pack("<e", center), "little")
    magnitude, negative = bits & 0x7fff, bool(bits & 0x8000)
    if magnitude == 0x7bff:
        raise ValueError("Midpoint diagnostic excludes finite endpoints")
    def number(word):
        return rational(struct.unpack("<e", word.to_bytes(2, "little"))[0])
    if magnitude == 0:
        lower, upper = number(0x8001), number(1)
    elif negative:
        lower, upper = number(bits + 1), number(bits - 1)
    else:
        lower, upper = number(bits - 1), number(bits + 1)
    low, high = (lower + rational(center)) / 2, (upper + rational(center)) / 2
    return dict(lower_midpoint=str(low), upper_midpoint=str(high),
                above_lower=str(value - low), below_upper=str(high - value))


def reconstruction_identity(codes, scales, biases, weights):
    """Replay CPU exporter's separate FP64 multiply/add, then FP32 cast."""
    def metadata(body):
        return [rational(struct.unpack("<f", struct.pack("<I", word << 16))[0])
                for (word,) in struct.iter_unpack("<H", body)]
    ss, bb = metadata(scales), metadata(biases)
    sequential = [struct.unpack("<f", struct.pack("<f", float(ss[i // 64]) * code + float(bb[i // 64])))[0]
                  for i, code in enumerate(codes)]
    if any(struct.pack("<f", a) != struct.pack("<f", b) for a, b in zip(sequential, weights)):
        raise ValueError("CPU separate and fused weight reconstruction differ")
    return len(weights)


def span(owners, item, size):
    if (item.get("file") not in owners or type(item.get("offset")) is not int
            or item["offset"] < 0 or type(item.get("bytes")) is not int
            or item["bytes"] != size):
        raise ValueError("Selected span geometry differs")
    owner = owners[item["file"]]
    if item["offset"] + size > owner.record["bytes"]:
        raise ValueError("Selected span exceeds parent")
    owner.file.seek(item["offset"])
    body = owner.file.read(size)
    if len(body) != size or hashlib.sha256(body).hexdigest() != item.get("sha256"):
        raise ValueError("Selected span bytes differ")
    return body


def point(owners, selected):
    packed = [span(owners, selected["weights"][key], size)
              for key, size in (("weight", 1024), ("scales", 32), ("biases", 32))]
    weights = reconstruct(*packed)
    inputs = {"native": half_expanded(span(owners, selected["native_input"], 4096), 1024)}
    outputs = {"native": half_expanded(span(owners, selected["native_output"], 4), 1)[0]}
    for contract in ("fp32", "fp64"):
        for family in ("reference_inputs", "reference_outputs"):
            if selected[family][contract].get("dtype") != "<f2":
                raise ValueError("Expected saved Half reference")
        inputs[contract] = finite_values(span(owners, selected["reference_inputs"][contract], 2048), "e", 1024)
        outputs[contract] = finite_values(span(owners, selected["reference_outputs"][contract], 2), "e", 1)[0]
    ww = [rational(value) for value in weights]
    dots = {key: sum((w * rational(x) for w, x in zip(ww, row)), Fraction(0))
            for key, row in inputs.items()}
    halves = {key: rounded(value, "e") for key, value in dots.items()}
    actual, expected = rational(outputs["native"]), rational(outputs["fp32"])
    terms = dict(native_local=actual - dots["native"], inherited_input=dots["native"] - dots["fp32"],
                 reference_local=dots["fp32"] - expected)
    if sum(terms.values()) != actual - expected:
        raise ValueError("Exact error decomposition differs")
    gate = selected["canonical_gate"]
    if (set(gate) != {"atol", "rtol", "pooled_fp32_fp64_max_abs", "equivalent_prefix_max_abs"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in gate.values())
            or gate["atol"] <= 0):
        raise ValueError("Invalid canonical gate")
    bound = gate["atol"] + gate["rtol"] * abs(outputs["fp32"])
    failed = abs(outputs["native"] - outputs["fp32"]) > bound
    category = selected["category"]
    if ((category == "native_canonical_failure" and not failed)
            or (category == "passing_control" and failed)):
        raise ValueError("Frozen selection classification differs")
    equal_inputs = {key: all(struct.pack("<e", a) == struct.pack("<e", b)
                            for a, b in zip(inputs["native"], inputs[key])) for key in ("fp32", "fp64")}
    if selected["stage"].startswith("layers.0."):
        same_output = struct.pack("<e", outputs["native"]) == struct.pack("<e", outputs["fp32"])
        if not all(equal_inputs.values()) or same_output != (category == "passing_control"):
            raise ValueError("First divergence requires identical complete inputs and different outputs")
    exact_input_l1 = sum((abs(w) * abs(rational(a) - rational(b))
                          for w, a, b in zip(ww, inputs["native"], inputs["fp32"])), Fraction(0))
    return dict(stage=selected["stage"], position=selected["position"], feature=selected["feature"], category=category,
        input_rows_identical_to_native=equal_inputs,
        cpu_and_fused_reconstruction_identical_weights=reconstruction_identity(*packed, weights),
        half_midpoint_distances={key: midpoint_distances(value) for key, value in dots.items()},
        saved=outputs, exact_dots={key: str(value) for key, value in dots.items()}, correctly_rounded_half=halves,
        local_extra_error={key: str(rational(outputs[key]) - rational(halves[key])) for key in dots},
        correct_rounding_error={key: str(rational(halves[key]) - value) for key, value in dots.items()},
        exact_difference=str(actual - expected), decomposition={key: str(value) for key, value in terms.items()},
        exact_input_l1_bound=str(exact_input_l1), canonical_failed=failed, canonical_bound=bound,
        native_own_input_correctly_rounded=struct.pack("<e", outputs["native"]) == struct.pack("<e", halves["native"]))


def validate(plan):
    profiles = {"toklet-qwen3-layer27-same-input-projection-plan-v1": (27, 14),
                "toklet-qwen3-first-projection-divergence-plan-v1": (0, 12)}
    if plan.get("format") not in profiles:
        raise ValueError("Frozen projection plan differs")
    layer, count = profiles[plan["format"]]
    if (any(type(plan.get(key)) is not int or plan[key] != value for key, value in
                   (("point_count", count), ("columns", 1024), ("group_size", 64), ("bits", 8), ("source_linear_bias", 0)))
            or plan.get("native_gates_changed") is not False
            or plan.get("whole_context_numerically_qualified") is not False
            or len(plan.get("points", [])) != count or len(plan.get("parents", {})) != 4):
        raise ValueError("Frozen projection plan differs")
    seen, categories = set(), {}
    for selected in plan["points"]:
        stage = selected.get("stage")
        if stage not in tuple(f"layers.{layer}.{kind}_projection" for kind in ("q", "k", "v")):
            raise ValueError("Unexpected projection")
        position, feature, category = (selected.get(key) for key in ("position", "feature", "category"))
        if (type(position) is not int or not 0 <= position < 512 or type(feature) is not int
                or (layer == 0 and position != 0)
                or not 0 <= feature < (2048 if stage.endswith("q_projection") else 1024)
                or category not in (("first_divergence", "passing_control") if layer == 0 else
                                    ("native_canonical_failure", "passing_control", "cpu_envelope_disjoint"))
                or (stage, position, feature) in seen):
            raise ValueError("Invalid/duplicate selection")
        seen.add((stage, position, feature))
        categories[(stage, category)] = categories.get((stage, category), 0) + 1
    expected = {(f"layers.{layer}.{kind}_projection", category): 2 for kind in ("q", "k", "v")
                for category in (("first_divergence", "passing_control") if layer == 0 else
                                 ("native_canonical_failure", "passing_control"))}
    if layer == 27:
        expected[("layers.27.v_projection", "cpu_envelope_disjoint")] = 2
    if categories != expected:
        raise ValueError("Selection control coverage differs")


def analyze(root, plan, report):
    """Append each completed point before continuing; endpoint failures survive."""
    validate(plan)
    report.update(complete=False, points=[], parents=[], native_gates_changed=False,
                  whole_context_numerically_qualified=False, quality_qualified=False, phone_qualified=False)
    root = Path(root).resolve()
    with ExitStack() as stack:
        owners = {}
        for key, record in plan["parents"].items():
            path = root / key
            if Path(key).is_absolute() or ".." in Path(key).parts or path.resolve() != path:
                raise ValueError("Parent path escaped root or used symlink")
            owner = Owner(path, record)
            report["parents"].append(owner.audit)
            owners[key] = stack.enter_context(owner)
        if plan.get("format") == "toklet-qwen3-first-projection-divergence-plan-v1":
            initial = plan.get("initial_rows", {})
            if set(initial) != {"embedding", "layers.0.input_norm"}:
                raise ValueError("Initial first-row coverage differs")
            report["initial_rows"] = {}
            for name, selected in initial.items():
                native = half_expanded(span(owners, selected["native"], 4096), 1024)
                for contract in ("fp32", "fp64"):
                    if selected[contract].get("dtype") != "<f2":
                        raise ValueError("Expected initial saved Half reference")
                    other = finite_values(span(owners, selected[contract], 2048), "e", 1024)
                    if any(struct.pack("<e", a) != struct.pack("<e", b) for a, b in zip(native, other)):
                        raise ValueError("Initial first rows differ")
                report["initial_rows"][name] = dict(columns=1024, fp32_exact=True, fp64_exact=True)
        for selected in plan["points"]:
            report["points"].append(point(owners, selected))
    report["complete"] = True
