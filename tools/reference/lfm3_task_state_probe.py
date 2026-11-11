"""Compare independent cached task-layer steps with the committed deep fixture."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import lfm3_graph_bf16_fixture as base
from lfm3_task_checkpoint import Checkpoint, sha
from lfm3_task_layers import forward_layer

REFERENCE_MANIFEST = "13f3199da158c64ab9679e88bbf297aa415e8b8987325f8abded7eef3735fa7a"
REFERENCE_PAYLOAD = "cec9c4fd912930550de8c2ad61d1e0508c2243a80e221cd5d619a7823cc16ae0"


def fixture_raw(source, observation):
    source.seek(observation["offset"])
    raw = bytearray(source.read(observation["words"] * 2))
    assert len(raw) == observation["words"] * 2 and sha(raw) == observation["sha256"]
    return raw


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base.torch = torch
    fixture = (args.fixture / "fixture.json").read_bytes()
    assert sha(fixture) == REFERENCE_MANIFEST
    fixture = json.loads(fixture)
    [schedule] = fixture["schedules"]
    rows = schedule["records"]
    layer = args.layer
    words = steps = 0
    with torch.inference_mode(), args.model.open("rb", buffering=0) as stream, \
            (args.fixture / "source.bf16le").open("rb") as reference:
        assert hashlib.file_digest(reference, "sha256").hexdigest() == REFERENCE_PAYLOAD
        checkpoint = Checkpoint(args.source, args.model, "modified_parent", stream)
        checkpoint.check(full=True)
        kind = checkpoint.layer_kinds[layer]
        weights = checkpoint.layer(layer)
        cache = expected_keys = expected_values = None
        start = 0
        for chunk, count in enumerate(schedule["chunks"]):
            if layer == 0:
                ids = fixture["tokens"][start:start + count]
                hidden = torch.cat([checkpoint.tensor("model.embed_tokens.weight", token, 1) for token in ids])
            else:
                [input_row] = [r for r in rows if r["chunk"] == chunk and
                               r["layer"] == layer - 1 and r["stage"] == "layer_output"]
                hidden = torch.frombuffer(fixture_raw(reference, input_row), dtype=torch.bfloat16).reshape(count, base.HIDDEN)
            output, cache = forward_layer(hidden, weights, kind, start, cache, checkpoint.eager)
            expected = {r["stage"]: r for r in rows if r["chunk"] == chunk and r["layer"] == layer}
            assert base.raw(output) == fixture_raw(reference, expected["layer_output"])
            words += expected["layer_output"]["words"]
            if kind == "conv":
                assert base.raw(cache.tensors[0]) == fixture_raw(reference, expected["history"])
                words += expected["history"]["words"]
            else:
                k, v = [torch.frombuffer(fixture_raw(reference, expected[stage]), dtype=torch.bfloat16)
                        .reshape(base.KV, count, base.WIDTH) for stage in ("key_proposal", "value_proposal")]
                if expected_keys is not None:
                    k, v = torch.cat((expected_keys, k), dim=1), torch.cat((expected_values, v), dim=1)
                if kind == "sliding_attention":
                    k, v = k[:, -255:].contiguous(), v[:, -255:].contiguous()
                assert base.raw(k) == base.raw(cache.tensors[0])
                assert base.raw(v) == base.raw(cache.tensors[1])
                expected_keys, expected_values = k, v
                words += k.numel() + v.numel()
            start += count
            steps += 1
        assert start == 1025
        checkpoint.check(full=True)
    with args.output.open("x") as report:
        json.dump(dict(passed=True, layer=layer, route=kind, steps=steps, words=words,
                       positions=start, fixture_sha256=REFERENCE_MANIFEST), report, indent=2)
        report.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "model", "fixture", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=range(24), required=True)
    run(parser.parse_args())
