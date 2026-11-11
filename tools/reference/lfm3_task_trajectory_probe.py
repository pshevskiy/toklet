"""Recompute every sampled tied-head decision for one guarded source task."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

import lfm3_context513_fixture as rollover
from lfm3_task_checkpoint import Checkpoint, sha

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from lfm3_quality_inputs import digest, read_bound, task_inputs


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = rollover.base
    base.torch = torch
    receipt_raw = read_bound(args.collection / "result.json", args.collection_sha256)
    receipt = json.loads(receipt_raw)
    assert receipt["passed"] and receipt["role"] == args.role and receipt["cases"] == 1
    tasks, _ = task_inputs(args.inputs, args.role)
    [result] = [json.loads(line) for line in read_bound(
        args.collection / "tasks.jsonl", receipt["tasks_sha256"]).splitlines()]
    task = tasks[result["index"]]
    assert result["id"] == task["id"] and len(result["samples"]) > 1
    prompt = task["prompt_tokens"]
    tokens = prompt + [s["token_id"] for s in result["samples"][:-1]]
    chunks = [min(128, len(prompt) - start) for start in range(0, len(prompt), 128)]
    selected_offset = len(chunks) - 1
    chunks += [1] * (len(result["samples"]) - 1)
    assert len(tokens) == sum(chunks) and len(tokens) <= 513
    args.output.mkdir()
    with torch.inference_mode(), args.model.open("rb", buffering=0) as model:
        checkpoint = Checkpoint(args.source, args.model, args.role, model)
        checkpoint.check(full=True)
        assert checkpoint.checkpoint["sha256"] == receipt["source_checkpoint_sha256"]
        provenance = dict(collection_sha256=args.collection_sha256, role=args.role,
                          tokens=len(tokens), case=result["id"], checkpoint_sha256=checkpoint.checkpoint["sha256"])
        if args.previous:
            prior = json.loads((args.previous / "fixture.json").read_bytes())
            raw = bytearray((args.previous / "hidden.bf16le").read_bytes())
            assert all(prior[k] == v for k, v in provenance.items())
            assert prior["end_layer"] == args.first and sha(raw) == prior["hidden_sha256"]
            hidden = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(len(tokens), base.HIDDEN)
        else:
            assert args.first == 0
            hidden = torch.cat([checkpoint.tensor("model.embed_tokens.weight", t, 1) for t in tokens])
        if args.end is not None:
            assert 0 <= args.first < args.end <= 24 and args.end - args.first <= 8
            with open("/dev/null", "wb") as discarded:
                for layer in range(args.first, args.end):
                    weights = checkpoint.layer(layer)
                    hidden, _ = rollover.execute_layer(hidden, weights, chunks, checkpoint.layer_kinds[layer],
                                                       checkpoint.eager, layer, discarded)
                    del weights
            raw = base.raw(hidden)
            with (args.output / "hidden.bf16le").open("xb") as stream:
                stream.write(raw)
            evidence = dict(end_layer=args.end, hidden_sha256=sha(raw))
        else:
            assert args.first == 24 and args.previous
            ends = torch.tensor(chunks).cumsum(0)[selected_offset:] - 1
            norm = base.norm(hidden[ends], checkpoint.tensor("model.embedding_norm.weight"))
            count = len(ends)
            maxima = [float("-inf")] * count
            winners = [0] * count
            hashes = [hashlib.sha256() for _ in range(count)]
            for at in range(0, base.VOCAB, 1024):
                head = checkpoint.tensor("model.embed_tokens.weight", at, min(1024, base.VOCAB - at))
                for i, row in enumerate(norm):
                    logits = torch.nn.functional.linear(row[None], head)[0]
                    assert bool(torch.isfinite(logits).all())
                    peak, selected = logits.max(dim=0)
                    if float(peak) > maxima[i]:
                        maxima[i], winners[i] = float(peak), at + int(selected)
                    hashes[i].update(base.raw(logits))
            actual = [dict(index=result["index"], token_id=token, logit_sha256=d.hexdigest(), max_logit=peak)
                      for token, d, peak in zip(winners, hashes, maxima, strict=True)]
            assert actual == result["samples"]
            evidence = dict(exact=True, decision_count=count, exact_bf16_words=count * base.VOCAB,
                            decision_sha256=sha(json.dumps(actual, sort_keys=True).encode()))
        checkpoint.check(full=True)
    assert digest(args.collection / "result.json") == args.collection_sha256
    with (args.output / "fixture.json").open("x") as stream:
        json.dump(provenance | evidence, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "inputs", "collection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--role", choices=["stock", "modified_parent"], required=True)
    parser.add_argument("--collection-sha256", required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--end", type=int)
    run(parser.parse_args())
