"""Freeze exact matrix-primitive inputs/oracles before any target execution."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/mps-matrix-v1/manifest.json"
SHAPES = [(1, 896, 896), (128, 896, 896), (1, 896, 128), (128, 896, 128),
          (1, 896, 4864), (128, 896, 4864), (1, 4864, 896), (128, 4864, 896),
          (1, 896, 151936), (5, 7, 3)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference():
    cases, tables = [], {}
    for tokens, inputs, outputs in SHAPES:
        if inputs not in tables:
            # Integer arithmetic is independent of CPU/GPU reduction association.
            # Products are exactly represented by FP16 operands and FP32 products.
            tables[inputs] = [[sum(((t*7+i) % 13-6) * ((o+3*i) % 17-8) for i in range(inputs))
                               for o in range(17)] for t in range(13)]
        cases.append({"name": f"{tokens}x{inputs}x{outputs}", "kind": "synthetic", "shape": [tokens, inputs, outputs],
                      "integer_table": tables[inputs], "fp16_output_admissible": True,
                      "maximum_absolute_partial_sum_numerator": 48*inputs+24})
    cases += [
        {"name": "manual_bias", "kind": "manual", "shape": [2, 3, 2],
         "left": [[2, -1, 4], [-1, 0, 2]], "right": [[1, 2, -1], [.5, -2, 3]], "bias": [.25, -.5],
         "expected": [[-3.75, 14.5], [-2.75, 5]], "fp16_output_admissible": True},
        {"name": "half_product_range_cancellation", "kind": "manual", "shape": [1, 2, 1],
         "left": [[65504, 65504]], "right": [[2, -1]], "bias": [0], "expected": [[65504]], "fp16_output_admissible": True},
        {"name": "float_output_range", "kind": "manual", "shape": [1, 1, 1],
         "left": [[65504]], "right": [[2]], "bias": [0], "expected": [[131008]], "fp16_output_admissible": False},
    ]
    return {"format": "toklet-mps-matrix-v1", "generator_sha256": digest(Path(__file__)), "cases": cases,
            "layout": "left [tokens,in], right [out,in] transposed by MPS, result [tokens,out]; actual MPS row padding is separate",
            "synthetic": {"left": "((7*token+feature)%13-6)/32", "right": "((output+3*feature)%17-8)/64",
                          "bias": "(output%7-3)/256", "expected": "integer_table[token%13][output%17]/2048+bias"},
            "gate": "Exact FP32 bits for FP32 result; one nearest-even FP16 rounding of exact result for FP16 result. No target-dependent tolerance.",
            "scope": "Manual/dyadic matrix primitive checks at representative shapes, before GPU evaluation; no trained-model quality or engine qualification."}


def export(path):
    if path.exists():
        raise ValueError("Refusing to overwrite frozen MPS oracle")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference(), indent=2, sort_keys=True) + "\n")


def verify(path=FIXTURE):
    manifest = json.loads(path.read_text())
    if manifest != reference():
        raise ValueError("Frozen MPS inputs/oracle differ")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--export", type=Path)
    group.add_argument("--check", type=Path, default=FIXTURE)
    args = parser.parse_args()
    export(args.export) if args.export else verify(args.check)
    print("Verified independent exact MPS matrix oracle")


if __name__ == "__main__":
    main()
