"""Memory-bounded phased source capture at the original LFM3 loss-block depth."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import lfm3_context513_fixture as rollover


PHASES = {"layers-0": (0, 8), "layers-1": (8, 16), "layers-2": (16, 24)}


def grouped(eager, module, query, key, value, mask, scale):
    """Apply the source eager operation independently per KV group."""
    outputs = [
        eager(module, query[:, head * 4:(head + 1) * 4], key[:, head:head + 1],
              value[:, head:head + 1], mask, scale)[0]
        for head in range(key.shape[1])
    ]
    return rollover.base.torch.cat(outputs, dim=2), None


def group_checks(torch, eager):
    generator = torch.Generator().manual_seed(1025)
    reports = []
    for queries, keys, window in [(128, 640, None), (128, 1024, None), (1, 1025, None), (128, 383, 256)]:
        def operands(heads, rows):
            return torch.randn((1, heads, rows, 128), generator=generator).to(torch.bfloat16)
        query, key, value = operands(16, queries), operands(4, keys), operands(4, keys)
        k = torch.arange(keys)[None, :]
        q = torch.arange(keys - queries, keys)[:, None]
        visible = k <= q
        if window:
            visible &= k > q - window
        mask = torch.where(visible, torch.zeros((), dtype=query.dtype),
                           torch.finfo(query.dtype).min)[None, None]
        module = SimpleNamespace(num_key_value_groups=4, training=False)
        full, _ = eager(module, query, key, value, mask, 128 ** -0.5)
        split, _ = grouped(eager, module, query, key, value, mask, 128 ** -0.5)
        raw = rollover.base.raw(full)
        assert raw == rollover.base.raw(split)
        reports.append(dict(queries=queries, keys=keys, window=window, words=len(raw)//2,
                            sha256=rollover.base.sha(raw)))
    return reports


def collect(args):
    import torch
    base = rollover.base
    base.torch = torch
    assert torch.version.git_version == rollover.TORCH_REVISION
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    contract_raw = args.contract.read_bytes()
    contract = json.loads(contract_raw)
    assert contract["format"] == "toklet-lfm3-context1025-contract-v1"
    [chunks] = contract["schedules"]
    assert len(chunks) == 14 and sum(chunks) == 1025 and max(chunks) <= 128
    tokens = [124894] + [(i * 3253 + 41) % 124000 for i in range(1, 1025)]
    config = args.config.read_bytes()
    assert base.sha(config) == base.CONFIG_SHA
    routes = json.loads(config)["layer_types"]
    _, eager = rollover.source_functions(args.parent)
    args.output.mkdir()
    result = dict(phase=args.phase, tokens=tokens, chunks=chunks,
                  source_sha256=rollover.MODEL_SHA, torch_revision=rollover.TORCH_REVISION,
                  contract_sha256=base.sha(contract_raw),
                  producer_sha256=base.sha(Path(__file__).read_bytes()), fixture_completed=True)
    if args.phase == "group-check":
        with torch.inference_mode():
            result["group_checks"] = group_checks(torch, eager)
        (args.output / "fixture.json").write_text(json.dumps(result, indent=2) + "\n")
        return

    def bounded_eager(module, query, key, value, mask, scale):
        if key.shape[2] <= 513:
            return eager(module, query, key, value, mask, scale)
        return grouped(eager, module, query, key, value, mask, scale)

    with args.model.open("rb", buffering=0) as stream, torch.inference_mode(), \
         (args.output / "source.bf16le").open("xb") as output:
        assert args.model.stat().st_size == rollover.MODEL_BYTES
        assert hashlib.file_digest(stream, "sha256").hexdigest() == rollover.MODEL_SHA
        stream.seek(0)
        size = struct.unpack("<Q", stream.read(8))[0]
        header = stream.read(size)
        assert base.sha(header) == rollover.HEADER_SHA
        schema = json.loads(header)
        origin = 8 + size

        def weight(name, start=0, rows=None):
            meta = schema[name]
            assert meta["dtype"] == "BF16"
            low, high = meta["data_offsets"]
            shape = meta["shape"]
            if rows is not None:
                assert len(shape) == 2 and start + rows <= shape[0]
                low += start * shape[1] * 2
                high = low + rows * shape[1] * 2
                shape = [rows, shape[1]]
            buf = bytearray(high - low)
            stream.seek(origin + low)
            assert stream.readinto(buf) == len(buf)
            return torch.frombuffer(buf, dtype=torch.bfloat16).reshape(shape)

        if args.phase == "layers-0":
            assert args.previous is None
            hidden = torch.cat([weight("model.embed_tokens.weight", token, 1) for token in tokens])
        else:
            previous = json.loads((args.previous / "fixture.json").read_bytes())
            buf = bytearray((args.previous / "hidden.bf16le").read_bytes())
            assert len(buf) == 1025 * base.HIDDEN * 2
            assert previous["contract_sha256"] == result["contract_sha256"]
            assert previous["producer_sha256"] == result["producer_sha256"]
            assert base.sha(buf) == previous["hidden_sha256"]
            result["input_hidden_sha256"] = previous["hidden_sha256"]
            hidden = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(1025, base.HIDDEN)

        records = []
        if args.phase in PHASES:
            first, last = PHASES[args.phase]
            for layer in range(first, last):
                prefix = f"model.layers.{layer}."
                weights = {n.removeprefix(prefix): weight(n) for n in schema if n.startswith(prefix)}
                hidden, rows = rollover.execute_layer(hidden, weights, chunks, routes[layer],
                                                      bounded_eager, layer, output)
                records.extend(rows)
                del weights
                print(f"captured layer {layer}", flush=True)
            raw = base.raw(hidden)
            (args.output / "hidden.bf16le").write_bytes(raw)
            result["hidden_sha256"] = base.sha(raw)
        else:
            final = base.norm(hidden, weight("model.embedding_norm.weight"))
            del hidden
            selected = final[torch.tensor(chunks).cumsum(0) - 1]
            logits = torch.empty((len(chunks), base.VOCAB), dtype=torch.bfloat16)
            for start in range(0, base.VOCAB, 1024):
                weights = weight("model.embed_tokens.weight", start, min(1024, base.VOCAB - start))
                logits[:, start:start + weights.shape[0]] = torch.cat(
                    [torch.nn.functional.linear(row[None], weights) for row in selected]
                )
            for i, (a, b) in enumerate(zip(final.split(chunks), logits)):
                records.append(rollover.record(output, None, i, "final_norm", a))
                records.append(rollover.record(output, None, i, "logits", b))
        stream.seek(0)
        assert hashlib.file_digest(stream, "sha256").hexdigest() == rollover.MODEL_SHA
        result.update(records=records, source_bytes=output.tell())
    (args.output / "fixture.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["model", "parent", "config", "contract", "output"]:
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--phase", required=True, choices=["group-check", *PHASES, "head"])
    parser.add_argument("--previous", type=Path)
    collect(parser.parse_args())
