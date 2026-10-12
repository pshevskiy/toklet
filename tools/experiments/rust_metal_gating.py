"""Qualify stored-Half gating against independent products and tiny source gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import rust_metal_silu as silu

residual = silu.residual
rms = silu.rms
ROOT = rms.ROOT
GUARD = 0x55a1


def half_product(a, b):
    """Independent FP64 product -> Half RNE, including signed zero.

    Finite Half operands have at most 11 significant bits; their product needs
    at most 22, with nonzero magnitude between 2**-48 and 2**32. The product is
    exact in both FP32 and FP64, so the reference does not copy GPU arithmetic.
    """
    left, right = struct.unpack("<2e", struct.pack("<2H", a, b))
    return struct.unpack("<H", struct.pack("<e", left * right))[0]


def cases():
    half = rms.manifest(rms.FP16, rms.FP16_MANIFEST_SHA)
    single = rms.manifest(rms.FP32, rms.FP32_MANIFEST_SHA)
    if (half["source_manifest_sha256"] != rms.FP32_MANIFEST_SHA
            or half["model"] != single["model"] or half["model"]["num_layers"] != 2
            or half["model"]["intermediate_size"] != 64):
        raise ValueError("Frozen gating topology differs")
    inputs = []
    anchors = {}
    for name, spec in sorted(half["cases"].items()):
        rows = len(spec["tokens"])
        count = rows * 64
        for layer in range(2):
            prefix = f"cases.{name}.layers.{layer}"
            key = f"{prefix}.gated"
            payloads = [rms.payload(rms.FP16, half, f"{prefix}.{stage}", "f16le")
                        for stage in ("gate_silu", "up_projection", "gated")]
            reference, ref = rms.payload(rms.FP32, single, key, "f32le")
            if ref["shape"] != [rows, 64] or any(item["shape"] != [rows, 64] for _, item in payloads):
                raise ValueError("Frozen gating shape differs")
            (gate, gate_item), (up, up_item), (oracle, item) = payloads
            a = list(struct.unpack(f"<{count}H", gate))
            b = list(struct.unpack(f"<{count}H", up))
            bits = list(struct.unpack(f"<{count}H", oracle))
            values = list(struct.unpack(f"<{count}f", reference))
            if any(value & 0x7c00 == 0x7c00 for value in (*a, *b, *bits)) or not all(map(math.isfinite, values)):
                raise ValueError("Frozen gating payload contains nonfinite values")
            if bits != [half_product(left, right) for left, right in zip(a, b)]:
                raise ValueError("Frozen gating oracle differs from independent Half product")
            inputs.append({"id": key, "rows": rows, "width": 64, "gate_bits": a, "up_bits": b})
            anchors[key] = {
                "bits": bits, "reference": values, "gate": item["gate_against_source_fp32"],
                "gate_silu_fp16_sha256": gate_item["sha256"], "up_projection_fp16_sha256": up_item["sha256"],
                "oracle_fp16_sha256": item["sha256"], "source_fp32_sha256": ref["sha256"],
            }
    if len(inputs) != 10 or sum(item["rows"] * 64 for item in inputs) != 5632:
        raise ValueError("Frozen gating inventory differs")
    # Expected bits were fixed in the preimplementation design, before device
    # evaluation. Recheck independently: zero signs, normal ties in both
    # directions, signed subnormal ties, normal boundary and finite maximum.
    edges = [
        (0, 0x3c00, 0), (0x8000, 0x3c00, 0x8000),
        (0x8000, 0xbc00, 0), (0, 0xbc00, 0x8000),
        (0x3e00, 0x3c01, 0x3e02), (0x3e00, 0x3c03, 0x3e04),
        (0xbe00, 0x3c01, 0xbe02), (0xbe00, 0x3c03, 0xbe04),
        (1, 0x3800, 0), (3, 0x3800, 2),
        (0x8001, 0x3800, 0x8000), (0x8003, 0x3800, 0x8002),
        (1, 0x3e00, 2), (0x0400, 0x3bff, 0x0400),
        (0x0400, 0x3800, 0x0200), (0x7bff, 0x3c00, 0x7bff),
        (0x7bff, 0x3800, 0x77ff), (0x7bff, 0xbc00, 0xfbff),
    ]
    if any(half_product(a, b) != expected for a, b, expected in edges):
        raise ValueError("Manual gating Half RNE oracle differs")
    count = 17 * 64
    repeated = [edges[index % len(edges)] for index in range(count)]
    bits = [value for _, _, value in repeated]
    key = "manual.product_rounding"
    inputs.append({"id": key, "rows": 17, "width": 64,
                   "gate_bits": [a for a, _, _ in repeated], "up_bits": [b for _, b, _ in repeated]})
    anchors[key] = {
        "bits": bits, "reference": list(struct.unpack(f"<{count}e", struct.pack(f"<{count}H", *bits))),
        "gate": None,
        "manual_oracle": "Preselected Half product RNE normal/subnormal ties, zero signs, finite maximum; independently rechecked with exact FP64 products and IEEE Half RNE",
        "oracle_fp16_sha256": hashlib.sha256(struct.pack(f"<{count}H", *bits)).hexdigest(),
    }
    overflow = {"id": "manual.overflow", "rows": 1, "width": 64,
                "gate_bits": [0x7bff] * 64, "up_bits": [0x4000] * 64}
    return inputs, anchors, overflow


TAIL_COUNTS = (1, 31, 32, 33, 63, 65, 127, 129, 4863)


def tail_cases():
    """Bounded irregular inventories from preselected independent RNE edges.

    These lengths target the previously observed width32, but launch padding
    and qualification always use queried capabilities, never assumed widths.
    """
    original, source, _ = cases()
    manual = original[-1]
    anchor = source[manual["id"]]
    inputs, anchors = [], {}
    for count in TAIL_COUNTS:
        key = f"manual.tail.{count}"
        a = [manual["gate_bits"][i % 18] for i in range(count)]
        b = [manual["up_bits"][i % 18] for i in range(count)]
        expected = [anchor["bits"][i % 18] for i in range(count)]
        if expected != [half_product(left, right) for left, right in zip(a, b)]:
            raise ValueError("Independent tail product oracle differs")
        inputs.append({"id": key, "rows": 1, "width": count, "gate_bits": a, "up_bits": b})
        anchors[key] = {"bits": expected,
                       "reference": list(struct.unpack(f"<{count}e", struct.pack(f"<{count}H", *expected))),
                       "gate": None, "manual_oracle": anchor["manual_oracle"],
                       "oracle_fp16_sha256": hashlib.sha256(struct.pack(f"<{count}H", *expected)).hexdigest()}
    if sum(TAIL_COUNTS) > 8192:
        raise ValueError("Tail inventory exceeds native admission")
    return inputs, anchors


def validate_tails(event, inputs, anchors):
    # Require actual inactive-lane coverage; an unusual device width must not
    # silently turn this into an aligned-only qualification.
    if event.get("passed") is not True:
        raise ValueError("Native tail check failed")
    native = event["result"]["cases"]
    if not any(case.get("inactive_lanes", 0) > 0 for case in native):
        raise ValueError("Queried dispatch provides no inactive tail coverage")
    validate(event, inputs, anchors)


def validate(event, inputs, anchors):
    if event.get("event") != "idle_result" or event.get("passed") is not True:
        raise ValueError("Native gating check failed")
    result = event["result"]
    metadata = result["metadata"]
    if (result["operation"] != "gating_fp16_storage_fp32_product" or metadata["unified_memory"] is not True
            or metadata["language_version"] != "2.3" or metadata["math_mode"] != "safe"
            or metadata["math_functions"] != "precise" or metadata["minimum_macos_major"] != 15
            or not 0 < metadata["maximum_admitted_bytes"] <= rms.probe.LIMIT
            or len(metadata["device_max_threads"]) != 3 or min(metadata["device_max_threads"]) <= 0
            or [case["id"] for case in result["cases"]] != [item["id"] for item in inputs]):
        raise ValueError("Native gating policy/inventory differs")
    for case, item in zip(result["cases"], inputs):
        count = item["rows"] * item["width"]
        width = case["threadgroup_width"]
        inactive = ((count + width - 1) // width) * width - count if width > 0 else 0
        guards = inactive + 7
        size = (count + guards) * 2
        if (case["rows_count"] != item["rows"] or case["width"] != item["width"] or case["guards"] != guards
                or case.get("inactive_lanes") != inactive or case.get("sentinel_guards") != 7
                or case["output_guard_bits"] != GUARD or case["distinct_owned_output"] is not True
                or case["allocations_reused_across_repetitions"] is not True or len(case["rows"]) != 13
                or not 0 < width <= min(metadata["execution_width"], metadata["pipeline_max_threads"], metadata["device_max_threads"][0])
                or case["threadgroups"] != (count + width - 1) // width or case["shared_buffer_bytes"] != size * 3
                or case["host_input_and_readback_bytes"] != size * 5
                or case["admitted_bytes"] != size * 8 + rms.probe.ALLOWANCE):
            raise ValueError("Native gating shape/ownership/launch/admission differs")
        anchor = anchors[item["id"]]
        for index, row in enumerate(case["rows"]):
            bits = row["output_bits"]
            if (row["phase"] != ("warmup" if index < 3 else "sample")
                    or row["repetition"] != (index + 1 if index < 3 else index - 2)
                    or row["command_completed"] is not True or row["immutable_inputs_and_guards"] is not True
                    or not math.isfinite(row["encode_commit_wait_seconds"]) or row["encode_commit_wait_seconds"] <= 0
                    or len(bits) != count + guards or any(type(value) is not int or not 0 <= value <= 65535 for value in bits)
                    or bits[count:] != [GUARD] * guards or bits[:count] != anchor["bits"]):
                raise ValueError("Gating completed-work/exact bits/guards differ")
            payload = struct.pack(f"<{count}H", *bits[:count])
            actual = struct.unpack(f"<{count}e", payload)
            errors = [abs(value - ref) for value, ref in zip(actual, anchor["reference"])]
            gate = anchor["gate"]
            fraction = max(error / (gate["atol"] + gate["rtol"] * abs(ref))
                           for error, ref in zip(errors, anchor["reference"])) if gate else 0.0
            if fraction > 1 or gate is None and max(errors) > 0:
                raise ValueError("Frozen gating FP32 source gate failed")
            row["qualification"] = {
                "exact_independent_fp16_bits": True, "source_fp32_max_abs_error": max(errors),
                "against_source_fp32_max_gate_fraction": fraction,
                "output_sha256": hashlib.sha256(payload).hexdigest(), "output_values": count, "output_guard_values": guards,
            }
            del row["output_bits"]
        case["anchors"] = {key: value for key, value in anchor.items() if key not in ("bits", "reference")}
    if metadata["maximum_admitted_bytes"] != max(case["admitted_bytes"] for case in result["cases"]):
        raise ValueError("Native gating maximum admission differs")


def overflow_rejected(event):
    if (event.get("event") != "idle_result" or event.get("passed") is not False
            or event.get("error", {}).get("stage") != "validation"
            or event["error"]["message"] != "completed gating contains nonfinite Half; result discarded"):
        raise ValueError("Completed gating overflow rejection was not preserved")
    return False  # Base driver additionally requires exit1 for this negative fixture.


def run(output, build, directory):
    rms.probe.check_paths(output, build)
    evidence = (output, output.with_suffix(output.suffix + ".stderr.txt"), output.with_suffix(output.suffix + ".native.jsonl"))
    if directory.exists():
        raise ValueError("Refusing to overwrite gating inputs")
    for path in (build, *evidence):
        if directory.resolve() == path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:
            raise ValueError("Gating inputs/build/evidence must be disjoint")
    if any(root.resolve() == path.resolve() or root.resolve() in path.resolve().parents
           for root in (rms.FP16, rms.FP32) for path in (build, directory, *evidence)):
        raise ValueError("Gating outputs must be outside source fixtures")
    inputs, anchors, overflow = cases()
    tails, tail_anchors = tail_cases()
    rms_inputs, rms_anchors = rms.cases()
    embedding = residual.embedding
    embedding_inputs, embedding_anchors = embedding.cases()
    residual_inputs, residual_anchors, residual_overflow = residual.cases()
    silu_inputs, silu_anchors = silu.cases()
    directory.mkdir(parents=True)
    jobs = []
    identities = {}
    payloads = [("rms_regression", {"format": "toklet-metal-rms-input-v1", "cases": rms_inputs}, "--rms-input",
                 lambda event: rms.validate(event, rms_inputs, rms_anchors))]
    for label, source in zip(("embedding_frozen_regression", "embedding_manual_regression"), embedding_inputs):
        payloads.append((label, source, "--embedding-input", lambda event, source=source: embedding.validate(event, source, embedding_anchors)))
    payloads.extend([
        ("residual_regression", {"format": "toklet-metal-residual-input-v1", "cases": residual_inputs}, "--residual-input",
         lambda event: residual.validate(event, residual_inputs, residual_anchors)),
        ("residual_overflow_regression", {"format": "toklet-metal-residual-input-v1", "cases": [residual_overflow]}, "--residual-input", residual.overflow_rejected),
        ("silu_regression", {"format": "toklet-metal-silu-input-v1", "cases": silu_inputs}, "--silu-input",
         lambda event: silu.validate(event, silu_inputs, silu_anchors)),
        ("gating_frozen_manual", {"format": "toklet-metal-gating-input-v1", "cases": inputs}, "--gating-input",
         lambda event: validate(event, inputs, anchors)),
        ("gating_inactive_tails", {"format": "toklet-metal-gating-input-v1", "cases": tails}, "--gating-input",
         lambda event: validate_tails(event, tails, tail_anchors)),
        ("gating_overflow", {"format": "toklet-metal-gating-input-v1", "cases": [overflow]}, "--gating-input", overflow_rejected),
    ])
    for label, payload, option, validator in payloads:
        path = directory / f"{label}.json"
        with path.open("x") as file:
            file.write(json.dumps(payload, separators=(",", ":")) + "\n")
        identities[str(path.resolve())] = rms.probe.common.digest(path)
        jobs.append((label, [option, str(path.resolve())], validator))
    drivers = {str(Path(module.__file__).resolve()): rms.probe.common.digest(Path(module.__file__))
               for module in (rms, embedding, residual, silu)}
    drivers[str(Path(__file__).resolve())] = rms.probe.common.digest(Path(__file__))
    previous = jobs[-1][2]

    def final(event):
        outcome = previous(event)
        rms.manifest(rms.FP16, rms.FP16_MANIFEST_SHA)
        rms.manifest(rms.FP32, rms.FP32_MANIFEST_SHA)
        for path, sha in {**identities, **drivers}.items():
            if rms.probe.common.digest(Path(path)) != sha:
                raise ValueError("Gating input/driver identity changed")
        return outcome

    jobs[-1] = (jobs[-1][0], jobs[-1][1], final)
    context = {
        "qualification_driver_sha256": drivers, "input_sha256": identities,
        "fp16_manifest_sha256": rms.FP16_MANIFEST_SHA, "source_fp32_manifest_sha256": rms.FP32_MANIFEST_SHA,
        "scope": "10 frozen tiny stored-Half gating stages, exact manual product RNE edges and separate bounded inactive-tail inventories; padding uses queried width plus7 sentinels. Exact independent product bits and unchanged source gates required. Completed overflow discarded. No Qwen, SiLU fusion, MLP/cache composition, performance or M4 completion.",
    }
    rms.probe.run(output, build, extra_runs=jobs, extra_context=context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.build_dir, args.input_dir)


if __name__ == "__main__":
    main()
