"""Capture the pinned LFM3 eager CPU stages independently of Toklet."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import resource
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

TORCH_REVISION = "5811a8d7da873dd699ff6687092c225caffcf1bb"
PARENT_SHA256 = "e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab"
HEADS, KV_HEADS, WIDTH = 16, 4, 128
SCALE = WIDTH ** -0.5
# prefix, queries, local window, input profile
CASES = (
    (0, 1, False, "edges"), (1, 1, False, "mixed"),
    (2, 1, False, "mixed"), (3, 1, False, "mixed"),
    (0, 2, False, "edges"), (0, 3, False, "mixed"),
    (0, 4, False, "mixed"), (4, 1, False, "mixed"),
    (6, 1, False, "mixed"), (7, 1, False, "mixed"),
    (8, 1, False, "mixed"), (30, 1, False, "cancellation"),
    (31, 1, False, "mixed"), (32, 1, False, "mixed"),
    (0, 32, False, "mixed"), (0, 128, False, "mixed"),
    (254, 4, True, "mixed"), (255, 1, True, "mixed"),
    (256, 1, True, "mixed"), (511, 32, True, "mixed"),
    (1023, 128, True, "mixed"), (511, 1, False, "mixed"),
    (1023, 32, False, "mixed"), (1023, 128, False, "mixed"),
    (2047, 1, False, "mixed"),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_functions(path):
    data = path.read_bytes()
    if sha(data) != PARENT_SHA256:
        raise ValueError("modified parent source hash differs")
    names = {"repeat_kv", "eager_attention_forward"}
    functions = [
        node for node in ast.parse(data).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise ValueError("pinned eager functions missing")
    tree = ast.Module(
        body=ast.parse("from __future__ import annotations").body + functions,
        type_ignores=[],
    )
    scope = {"torch": torch, "nn": nn}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
    return scope["repeat_kv"], scope["eager_attention_forward"]


def inputs(shape, seed, profile):
    x = np.arange(np.prod(shape), dtype=np.uint32) + np.uint32(seed)
    x ^= x >> 16
    x *= np.uint32(0x7FEB352D)
    x ^= x >> 15
    x *= np.uint32(0x846CA68B)
    x ^= x >> 16
    sign = (x >> 16) & 0x8000
    if profile == "edges":
        words = np.array([0, 1, 0x7F, 0x80, 0x3700, 0x3B80, 0x3F80, 0x4000])
        values = words[x % len(words)] | sign
    elif profile == "cancellation":
        values = (0x3600 + (x & 0x7FF)) | sign
    else:
        values = (0x3D00 + (x & 0x3FF)) | sign
    return torch.from_numpy(values.astype("<u2").reshape(shape)).view(torch.bfloat16)


def bf16_bytes(tensor):
    return tensor.contiguous().view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()


def diagnostic(actual, expected):
    error = (actual.double() - expected).abs()
    return {
        "words": error.numel(),
        "maximum_absolute": error.max().item(),
        "failures": (error > 1e-5 + 1e-4 * expected.abs()).count_nonzero().item(),
    }


@torch.inference_mode()
def collect(output, parent, contract):
    if torch.version.git_version != TORCH_REVISION:
        raise ValueError("wrong Torch source revision")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    repeat_kv, eager_attention = source_functions(parent)
    output.mkdir()
    reports = []
    for i, (prefix, tokens, local, profile) in enumerate(CASES):
        origin = max(0, prefix - 255) if local else 0
        keys = prefix + tokens - origin
        seed = (i + 1) * 0xF13
        q = inputs((tokens, HEADS, WIDTH), seed, profile)
        k = inputs((KV_HEADS, keys, WIDTH), seed + 0x397, profile)
        v = inputs((KV_HEADS, keys, WIDTH), seed + 0xB21, profile)
        query = q.transpose(0, 1).unsqueeze(0)
        key, value = k.unsqueeze(0), v.unsqueeze(0)
        repeated_k = repeat_kv(key, HEADS // KV_HEADS).transpose(2, 3)
        repeated_v = repeat_kv(value, HEADS // KV_HEADS)
        key_pos = torch.arange(origin, prefix + tokens)[None, :]
        query_pos = torch.arange(prefix, prefix + tokens)[:, None]
        mask = key_pos <= query_pos
        if local:
            mask &= key_pos > query_pos - 256
        mask = torch.where(mask, torch.zeros((), dtype=q.dtype), torch.finfo(q.dtype).min)
        mask = mask[None, None, :, :]

        qk = torch.matmul(query, repeated_k)
        scaled = qk * SCALE
        scores = scaled + mask
        softmax = torch.softmax(scores, dim=-1, dtype=torch.float32)
        probabilities = softmax.to(q.dtype)
        context = torch.matmul(probabilities, repeated_v).transpose(1, 2).contiguous()
        exact_context, exact_probabilities = eager_attention(
            SimpleNamespace(num_key_value_groups=HEADS // KV_HEADS, training=False),
            query, key, value, mask, SCALE,
        )
        assert torch.equal(context.view(torch.uint16), exact_context.view(torch.uint16))
        assert torch.equal(probabilities.view(torch.uint16), exact_probabilities.view(torch.uint16))

        stages = {}
        for name, tensor in [
            ("qk", qk), ("scaled", scaled), ("scores", scores),
            ("probabilities", probabilities), ("context", context),
        ]:
            raw = bf16_bytes(tensor)
            stages[name] = {"words": len(raw) // 2, "sha256": sha(raw)}
            (output / f"{i:02d}-{name}.bf16le").write_bytes(raw)

        # These are mathematical references with the source inputs at the
        # corresponding rounding boundary. No BF16 quantization is charged
        # against the independent FP32 diagnostic.
        diagnostic_softmax = diagnostic(softmax, torch.softmax(scores.double(), dim=-1))
        (output / f"{i:02d}-softmax.f32le").write_bytes(
            softmax.numpy().astype("<f4", copy=False).tobytes()
        )
        reports.append({
            "prefix": prefix, "tokens": tokens, "key_start": origin, "keys": keys,
            "window": 256 if local else None, "profile": profile, "seed": seed,
            "inputs": {name: sha(bf16_bytes(t)) for name, t in [("q", q), ("k", k), ("v", v)]},
            "source_strides": {
                name: list(t.stride()) for name, t in
                [("query", query), ("transposed_key", repeated_k), ("value", repeated_v)]
            },
            "stages": stages, "source_softmax_fp64_diagnostic": diagnostic_softmax,
        })

    result = {
        "torch_revision": torch.version.git_version,
        "parent_source_sha256": PARENT_SHA256,
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "contract_sha256": sha(contract.read_bytes()),
        "fixture_completed": True,
        "cases": reports,
    }
    (output / "fixture.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "cases": len(reports),
        "source_stage_words": sum(s["words"] for c in reports for s in c["stages"].values()),
        "child_peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    collect(args.output, args.parent, args.contract)


if __name__ == "__main__":
    main()
