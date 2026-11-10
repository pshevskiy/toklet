"""Layer-streamed eager BF16 LFM3 graph oracle for a fixed short token history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

from lfm3_attention_bf16_fixture import source_functions, TORCH_REVISION
from lfm3_bf16_fixture import MODEL_SHA, HEADER_SHA, MODEL_BYTES

CONFIG_SHA = "720834e9b1f3b321d59ce1adb5714e42d8f880d722d9abf9b796ce008cf087a3"
TOKENS = [124894, 42, 1234, 567]
SCHEDULES = [[4], [2, 1, 1], [1, 3]]
HIDDEN, HEADS, KV, WIDTH, VOCAB = 2048, 16, 4, 128, 128000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw(value):
    return value.contiguous().view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()


def norm(value, gain):
    # The parent normalizes in FP32, casts before multiplying the gain.
    x = value.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)).to(value.dtype) * gain


def rotary(value, offset, local):
    theta, active = (10000, 64) if local else (1000000, 32)
    inv = 1.0 / (theta ** (torch.arange(0, 2 * active, 2, dtype=torch.float32) / WIDTH))
    inv = torch.cat((inv, torch.zeros(WIDTH // 2 - active)))
    a = torch.arange(offset, offset + value.shape[0], dtype=torch.float32)[:, None] * inv
    angle = torch.cat((a, a), dim=-1)
    c, s = angle.cos().to(value.dtype)[:, None], angle.sin().to(value.dtype)[:, None]
    x = value.float()
    rotated = torch.cat((-x[..., WIDTH // 2:], x[..., :WIDTH // 2]), dim=-1)
    return (x * c.float() + rotated * s.float()).to(value.dtype)


def execute_layer(x, w, chunks, kind, eager):
    def linear(v, suffix):
        return torch.nn.functional.linear(v, w[suffix])

    def op(v, suffix):
        return norm(v, w[suffix + ".weight"])

    start = 0
    keys, values, history = None, None, None
    output, observations = [], []
    for count in chunks:
        h = x[start:start + count]
        prepared = op(h, "operator_norm")
        if kind == "conv":
            b, c, v = linear(prepared, "conv.in_proj.weight").chunk(3, dim=-1)
            bx = b * v
            previous = bx.new_zeros((6, HIDDEN)) if history is None else history
            convolution_input = torch.cat((previous, bx), dim=0)
            history = convolution_input[-6:].clone()
            y = torch.nn.functional.conv1d(
                convolution_input.T.unsqueeze(0).contiguous(),
                w["conv.conv.weight"].reshape(HIDDEN, 1, 7), groups=HIDDEN,
            ).squeeze(0).T
            mixed = linear(c * y, "conv.out_proj.weight")
            stages = {"history": sha(raw(history))}
        else:
            local = kind == "sliding_attention"
            q = op(linear(prepared, "self_attn.q_proj.weight").reshape(count, HEADS, WIDTH), "self_attn.q_norm")
            k = op(linear(prepared, "self_attn.k_proj.weight").reshape(count, KV, WIDTH), "self_attn.k_norm")
            q = rotary(q, start, local).transpose(0, 1).unsqueeze(0)
            new_k = rotary(k, start, local).transpose(0, 1).contiguous()
            new_v = linear(prepared, "self_attn.v_proj.weight").reshape(count, KV, WIDTH).transpose(0, 1).contiguous()
            keys = new_k if keys is None else torch.cat((keys, new_k), dim=1)
            values = new_v if values is None else torch.cat((values, new_v), dim=1)
            key_pos = torch.arange(keys.shape[1])[None, :]
            query_pos = torch.arange(start, start + count)[:, None]
            mask = key_pos <= query_pos
            if local:
                mask &= key_pos > query_pos - 256
            mask = torch.where(mask, torch.zeros((), dtype=h.dtype), torch.finfo(h.dtype).min)
            result, _ = eager(
                SimpleNamespace(num_key_value_groups=HEADS // KV, training=False), q,
                keys.unsqueeze(0), values.unsqueeze(0), mask[None, None], WIDTH ** -0.5,
            )
            gate = linear(prepared, "self_attn.gate_proj.weight").sigmoid()
            mixed = linear((result.squeeze(0) * gate[..., None]).reshape(count, HIDDEN), "self_attn.o_proj.weight")
            stages = {"key_proposal": sha(raw(new_k)), "value_proposal": sha(raw(new_v))}
        residual = h + op(mixed, "post_operator_norm")
        ffn = op(residual, "ffn_norm")
        ff = linear(torch.nn.functional.silu(linear(ffn, "feed_forward.w1.weight")) *
                    linear(ffn, "feed_forward.w3.weight"), "feed_forward.w2.weight")
        h = residual + op(ff, "post_ffn_norm")
        output.append(h)
        stages["layer_output"] = sha(raw(h))
        observations.append(stages)
        start += count
    return torch.cat(output), observations


def collect(model, parent, config, contract, output):
    global torch, SimpleNamespace
    import torch
    from types import SimpleNamespace

    assert torch.version.git_version == TORCH_REVISION
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert sha(config.read_bytes()) == CONFIG_SHA
    routes = json.loads(config.read_bytes())["layer_types"]
    assert len(routes) == 24
    _, eager = source_functions(parent)
    output.mkdir()
    with model.open("rb", buffering=0) as stream, torch.inference_mode():
        assert model.stat().st_size == MODEL_BYTES
        assert hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA
        stream.seek(0)
        count = struct.unpack("<Q", stream.read(8))[0]
        header = stream.read(count)
        assert sha(header) == HEADER_SHA
        schema = json.loads(header)
        origin = 8 + count

        def weight(name, start=0, rows=None):
            meta = schema[name]
            assert meta["dtype"] == "BF16"
            lo, hi = meta["data_offsets"]
            shape = meta["shape"]
            if rows is not None:
                assert len(shape) == 2 and start + rows <= shape[0]
                lo += start * shape[1] * 2
                hi = lo + rows * shape[1] * 2
                shape = [rows, shape[1]]
            buf = bytearray(hi - lo)
            stream.seek(origin + lo)
            assert stream.readinto(buf) == len(buf)
            return torch.frombuffer(buf, dtype=torch.bfloat16).reshape(shape)

        embeddings = torch.cat([weight("model.embed_tokens.weight", t, 1) for t in TOKENS])
        hidden = [embeddings] * len(SCHEDULES)
        reports = [{"chunks": chunk_sizes, "layers": []} for chunk_sizes in SCHEDULES]
        for i, kind in enumerate(routes):
            prefix = f"model.layers.{i}."
            w = {n.removeprefix(prefix): weight(n) for n in schema if n.startswith(prefix)}
            for j, schedule in enumerate(SCHEDULES):
                hidden[j], observed = execute_layer(hidden[j], w, schedule, kind, eager)
                reports[j]["layers"].append(observed)
            del w
            print(f"captured layer {i}", flush=True)
        gain = weight("model.embedding_norm.weight")
        final = [norm(h, gain) for h in hidden]
        logits = []
        for i, sizes in enumerate(SCHEDULES):
            logits.append(torch.empty((len(sizes), VOCAB), dtype=torch.bfloat16))
            reports[i]["final_norm"] = [sha(raw(row)) for row in final[i].split(sizes)]
        for start in range(0, VOCAB, 1024):
            weights = weight("model.embed_tokens.weight", start, min(1024, VOCAB - start))
            for i, sizes in enumerate(SCHEDULES):
                selected = torch.tensor(sizes).cumsum(0) - 1
                rows = final[i][selected]
                # Each source chunk asks its tied head for its last row.
                logits[i][:, start:start + weights.shape[0]] = torch.cat(
                    [torch.nn.functional.linear(row[None], weights) for row in rows]
                )
        for i, rows in enumerate(logits):
            reports[i]["logits"] = [sha(raw(row)) for row in rows]
            (output / f"logits-{i}.bf16le").write_bytes(raw(rows))
        stream.seek(0)
        assert hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA
    result = dict(tokens=TOKENS, schedules=reports, torch_revision=TORCH_REVISION,
                  model_sha256=MODEL_SHA, contract_sha256=sha(contract.read_bytes()),
                  producer_sha256=sha(Path(__file__).read_bytes()), fixture_completed=True)
    (output / "fixture.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "parent", "config", "contract", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    collect(args.model, args.parent, args.config, args.contract, args.output)


if __name__ == "__main__":
    main()
