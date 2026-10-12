"""Qualify saved Qwen stored-Half products with unchanged source gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import rust_metal_gating as gating
import rust_metal_qwen_silu as saved

qwen = saved.qwen
ROOT = qwen.ROOT
WIDTH = 4864
LAYERS = (0, 1, 23)


def checked_references():
    # Four pinned full/selected manifests, topology, gates and config are
    # validated without opening or hashing the source checkpoint.
    return saved.checked_references()


def assemble(half, single, half_root, single_root):
    inputs = []
    anchors = {}
    for name, rows in qwen.CASE_ROWS.items():
        for layer in LAYERS:
            prefix = f"cases.{name}.layers.{layer}"
            stage = f"{prefix}.gated"
            gate_item = half["arrays"][f"{prefix}.gate_silu"]
            up_item = half["arrays"][f"{prefix}.up_projection"]
            oracle_item = half["arrays"][stage]
            # The original source declares FP32 globally; this read view adds
            # that declared dtype without editing an artifact or gate policy.
            ref_item = {"dtype": "f32le", **single["arrays"][stage]}
            payloads = [saved.payload(half_root, item, "f16le", rows)
                        for item in (gate_item, up_item, oracle_item)]
            reference = saved.payload(single_root, ref_item, "f32le", rows)
            if oracle_item["source_fp32_full_sha256"] != ref_item["sha256"]:
                raise ValueError("Qwen gating FP32 source anchor differs")
            count = rows * WIDTH
            gate, up, oracle = payloads
            gate_bits, up_bits, oracle_bits = [list(struct.unpack(f"<{count}H", data))
                                                for data in payloads]
            ref_values = list(struct.unpack(f"<{count}f", reference))
            if (any(value & 0x7c00 == 0x7c00 for values in (gate_bits, up_bits, oracle_bits) for value in values)
                    or not all(map(math.isfinite, ref_values))):
                raise ValueError("Qwen gating source/oracle contains nonfinite values")
            # Both finite Half operands multiply exactly in FP64. Verify the
            # entire frozen oracle before any row slicing or native execution.
            try:
                expected = [gating.half_product(a, b) for a, b in zip(gate_bits, up_bits)]
            except OverflowError as error:
                raise ValueError("Qwen gating independent product exceeds finite Half") from error
            if oracle_bits != expected:
                raise ValueError("Qwen gating oracle differs from independent Half product")
            for row in range(rows):
                key = f"{stage}.row.{row}"
                start, end = row * WIDTH, (row + 1) * WIDTH
                inputs.append({"id": key, "rows": 1, "width": WIDTH,
                               "gate_bits": gate_bits[start:end], "up_bits": up_bits[start:end]})
                anchors[key] = {
                    "bits": oracle_bits[start:end], "reference": ref_values[start:end],
                    "gate": oracle_item["gate_against_source_fp32"], "source_stage": stage, "row_index": row,
                    "gate_silu_full_fp16_sha256": gate_item["sha256"],
                    "up_projection_full_fp16_sha256": up_item["sha256"],
                    "oracle_full_fp16_sha256": oracle_item["sha256"],
                    "source_full_fp32_sha256": ref_item["sha256"],
                    "gate_silu_fp16_sha256": hashlib.sha256(gate[start * 2:end * 2]).hexdigest(),
                    "up_projection_fp16_sha256": hashlib.sha256(up[start * 2:end * 2]).hexdigest(),
                    "oracle_fp16_sha256": hashlib.sha256(oracle[start * 2:end * 2]).hexdigest(),
                    "source_fp32_sha256": hashlib.sha256(reference[start * 4:end * 4]).hexdigest(),
                }
    if len(inputs) != 36 or sum(item["rows"] * item["width"] for item in inputs) != 175104:
        raise ValueError("Qwen gating selected row inventory differs")
    return inputs, anchors


def run(output, build, directory):
    qwen.check_paths(output, build, directory)
    evidence = (output, output.with_suffix(output.suffix + ".stderr.txt"), output.with_suffix(output.suffix + ".native.jsonl"))
    if any(root.resolve() == path.resolve() or root.resolve() in path.resolve().parents
           for root in (gating.rms.FP16, gating.rms.FP32) for path in (build, directory, *evidence)):
        raise ValueError("Qwen gating outputs must be outside tiny source fixtures")
    half, single = checked_references()
    inputs, anchors = assemble(half, single, qwen.HALF, qwen.SINGLE)
    tiny_inputs, tiny_anchors, overflow = gating.cases()
    tail_inputs, tail_anchors = gating.tail_cases()
    directory.mkdir(parents=True)
    identities = {}
    jobs = []
    groups = [
        ("tiny_gating_regression", tiny_inputs, lambda event: gating.validate(event, tiny_inputs, tiny_anchors)),
        ("gating_inactive_tails_regression", tail_inputs, lambda event: gating.validate_tails(event, tail_inputs, tail_anchors)),
        ("gating_overflow_regression", [overflow], gating.overflow_rejected),
    ] + [(f"qwen_gating_row_{index + 1}", [item],
          lambda event, item=item: gating.validate(event, [item], anchors))
         for index, item in enumerate(inputs)]
    for label, group, validator in groups:
        path = directory / f"{label}.json"
        with path.open("x") as file:
            file.write(json.dumps({"format": "toklet-metal-gating-input-v1", "cases": group}, separators=(",", ":")) + "\n")
        sha = gating.rms.probe.common.digest(path)
        identities[str(path.resolve())] = sha

        def qualified(event, validate=validator, fixture=path, expected_sha=sha):
            result = validate(event)
            if gating.rms.probe.common.digest(fixture) != expected_sha:
                raise ValueError("Qwen gating batch changed during execution")
            return result  # Preserve False for the completed-overflow fixture.

        jobs.append((label, ["--gating-input", str(path.resolve())], qualified))
    modules = (gating, saved, qwen, gating.rms, gating.silu, gating.residual, gating.residual.embedding)
    drivers = {str(Path(module.__file__).resolve()): gating.rms.probe.common.digest(Path(module.__file__))
               for module in modules}
    drivers[str(Path(__file__).resolve())] = gating.rms.probe.common.digest(Path(__file__))
    previous = jobs[-1][2]

    def final(event):
        previous(event)
        current_half, current_single = checked_references()
        if assemble(current_half, current_single, qwen.HALF, qwen.SINGLE) != (inputs, anchors):
            raise ValueError("Qwen gating saved references changed during execution")
        gating.rms.manifest(gating.rms.FP16, gating.rms.FP16_MANIFEST_SHA)
        gating.rms.manifest(gating.rms.FP32, gating.rms.FP32_MANIFEST_SHA)
        for path, sha in {**identities, **drivers}.items():
            if gating.rms.probe.common.digest(Path(path)) != sha:
                raise ValueError("Qwen gating driver/input changed")

    jobs[-1] = (jobs[-1][0], jobs[-1][1], final)
    context = {
        "qualification_driver_sha256": drivers, "input_sha256": identities,
        "manifest_sha256": {str(root.relative_to(ROOT)): sha for root, sha in qwen.IDENTITIES.items()},
        "tiny_manifest_sha256": {"fp16": gating.rms.FP16_MANIFEST_SHA, "source_fp32": gating.rms.FP32_MANIFEST_SHA},
        "config_sha256": qwen.CONFIG_SHA, "frozen_source_checkpoint_sha256": qwen.WEIGHT_SHA,
        "frozen_source_checkpoint_bytes": qwen.WEIGHT_BYTES, "checkpoint_reads": False,
        "scope": "12 selected width4864 gated stages split into36 complete rows; both stored-Half operands and frozen products independently checked before native work. Unchanged pre-GPU FP32 source gates. Same-binary tiny, inactive-tail and completed-overflow regression. No checkpoint reads, fusion, MLP/decoder/cache composition, sessions, quality, performance, traces or M4 completion.",
    }
    gating.rms.probe.run(output, build, extra_runs=jobs, extra_context=context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.build_dir, args.input_dir)


if __name__ == "__main__":
    main()
