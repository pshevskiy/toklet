"""Run one bounded phase of the independent LFM3 task source."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import lfm3_graph_bf16_fixture as base
from lfm3_task_checkpoint import Checkpoint, sha
from lfm3_task_layers import LayerCache, forward_layer


def read_json(path, limit=8 * 1024**2):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    assert len(raw) <= limit
    return json.loads(raw)


def read_tensor(path, shape, expected):
    size = 2
    for axis in shape:
        size *= axis
    with path.open("rb") as stream:
        raw = bytearray(stream.read(size + 1))
    assert len(raw) == size and sha(raw) == expected
    return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape)


def cache_key(layer, case):
    return f"{layer:02d}-{case:03d}.bf16"


def read_cache(folder, prior, layer, case, kind, position):
    name = cache_key(layer, case)
    path = folder / "state" / name
    expected = prior["state"][name]
    if kind == "conv":
        shape = (6, base.HIDDEN)
        cache = read_tensor(path, shape, expected)
        return LayerCache((cache,))
    rows = min(position, 255) if kind == "sliding_attention" else position
    cache = read_tensor(path, (2, base.KV, rows, base.WIDTH), expected)
    return LayerCache(tuple(cache.unbind(0)))


def write_payload(path, raw):
    with path.open("xb") as output:
        assert output.write(raw) == len(raw)
    return sha(raw)


def layer_pass(checkpoint, plan, output, state_dir, prior, previous, first, end):
    ids = plan["cases"]
    hidden = output / "hidden"
    hidden.mkdir()
    produced = {}
    for layer in range(first, end):
        weights = checkpoint.layer(layer)
        kind = checkpoint.layer_kinds[layer]
        # Keep one layer bank. Each task retains independent cache and positions.
        layer_hidden = output / f"hidden-{layer:02d}"
        layer_hidden.mkdir()
        current = {}
        for case in ids:
            index, position = case["index"], case["start"]
            name = f"{index:03d}.bf16"
            inputs = case["tokens"]
            if layer == 0:
                x = torch.cat([checkpoint.tensor("model.embed_tokens.weight", token, 1) for token in inputs])
            elif layer == first:
                source = previous["path"] / "hidden" / name
                x = read_tensor(source, (len(inputs), base.HIDDEN), previous["hidden"][name])
            else:
                x = read_tensor(hidden / name, (len(inputs), base.HIDDEN), prior_hidden[name])
            if position:
                assert len(inputs) == 1 and prior is not None
                cache = read_cache(prior["path"], prior, layer, index, kind, position)
            else:
                cache = None
            states = []
            for chunk in x.split(128):
                y, cache = forward_layer(chunk, weights, kind, position, cache, checkpoint.eager)
                states.append(y)
                position += len(chunk)
            assert position == case["start"] + len(inputs)
            current[name] = write_payload(layer_hidden / name, base.raw(torch.cat(states)))
            key = cache_key(layer, index)
            raw = b"".join(base.raw(tensor) for tensor in cache.tensors)
            produced[key] = write_payload(state_dir / key, raw)
        del weights
        # Intermediate hidden rows are owned by this attempt and only the next
        # layer uses them. The current output is complete before removing them.
        for path in hidden.iterdir():
            path.unlink()
        hidden.rmdir()
        layer_hidden.rename(hidden)
        prior_hidden = current
    return dict(hidden=prior_hidden, state=produced, first_layer=first, end_layer=end)


def tied_head(checkpoint, plan, previous, output):
    cases = plan["cases"]
    normalized = []
    gain = checkpoint.tensor("model.embedding_norm.weight")
    for case in cases:
        name = f"{case['index']:03d}.bf16"
        final = read_tensor(previous["path"] / "hidden" / name,
                            (len(case["tokens"]), base.HIDDEN), previous["hidden"][name])
        normalized.append(base.norm(final[-1:], gain))
    maximum = [-float("inf")] * len(cases)
    greedy = [0] * len(cases)
    hashes = [hashlib.sha256() for _ in cases]
    for start in range(0, base.VOCAB, 1024):
        weights = checkpoint.tensor("model.embed_tokens.weight", start, min(1024, base.VOCAB - start))
        for i, row in enumerate(normalized):
            logits = torch.nn.functional.linear(row, weights)[0]
            assert bool(torch.isfinite(logits).all())
            score, local_id = logits.max(dim=0)
            if float(score) > maximum[i]:
                maximum[i], greedy[i] = float(score), start + int(local_id)
            hashes[i].update(base.raw(logits))
    with (output / "samples.jsonl").open("x") as stream:
        for case, token, digest, peak in zip(cases, greedy, hashes, maximum, strict=True):
            stream.write(json.dumps(dict(index=case["index"], token_id=token,
                                         logit_sha256=digest.hexdigest(), max_logit=peak),
                                    separators=(",", ":"), allow_nan=False) + "\n")
    return dict(samples_sha256=sha((output / "samples.jsonl").read_bytes()), sample_count=len(cases))


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base.torch = torch
    plan = read_json(args.plan)
    assert 0 < len(plan["cases"]) <= 100
    assert [r["index"] for r in plan["cases"]] == sorted({r["index"] for r in plan["cases"]})
    for case in plan["cases"]:
        assert 0 <= case["index"] < 100 and 0 <= case["start"] < 2048
        assert 0 < len(case["tokens"]) <= 2048 - case["start"]
        assert all(type(t) is int and 0 <= t < base.VOCAB for t in case["tokens"])
    previous = None
    if args.previous:
        previous = read_json(args.previous / "fixture.json")
        previous["path"] = args.previous
        assert previous["plan_sha256"] == sha(args.plan.read_bytes())
    prior = None
    if args.prior:
        prior = read_json(args.prior / "commit.json")
        prior["path"] = args.prior
        assert prior["round"] + 1 == plan["round"]
    args.output.mkdir()
    with args.model.open("rb", buffering=0) as stream, torch.inference_mode():
        checkpoint = Checkpoint(args.source, args.model, args.role, stream)
        assert checkpoint.identity == tuple(plan["file_identity"])
        checkpoint.check()
        if args.end is None:
            assert previous is not None and previous["end_layer"] == 24
            result = tied_head(checkpoint, plan, previous, args.output)
        else:
            assert 0 <= args.first < args.end <= 24
            assert (args.first == 0 and previous is None) or previous["end_layer"] == args.first
            assert (plan["round"] == 0) == (prior is None)
            result = layer_pass(checkpoint, plan, args.output, args.state, prior, previous, args.first, args.end)
        checkpoint.check()
    result.update(role=args.role, round=plan["round"], plan_sha256=sha(args.plan.read_bytes()),
                  checkpoint_sha256=checkpoint.checkpoint["sha256"], canonical=checkpoint.canonical)
    with (args.output / "fixture.json").open("x") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "source", "plan", "output", "state"]:
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ["previous", "prior"]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--role", choices=["stock", "modified_parent"], required=True)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--end", type=int)
    run(parser.parse_args())
