"""Bounded layer-streamed LFM3 eager reference through two local-window wraps."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import lfm3_graph_bf16_fixture as base
from lfm3_attention_bf16_fixture import source_functions, TORCH_REVISION
from lfm3_bf16_fixture import MODEL_SHA, HEADER_SHA, MODEL_BYTES


def record(file, layer, chunk, stage, tensor):
    raw = base.raw(tensor)
    result = dict(layer=layer, chunk=chunk, stage=stage, offset=file.tell(),
                  words=len(raw)//2, sha256=base.sha(raw))
    file.write(raw)
    return result


def execute_layer(x, w, chunks, kind, eager, layer, output):
    torch = base.torch
    def linear(v, key):
        return torch.nn.functional.linear(v, w[key + ".weight"])
    def norm(v, key):
        return base.norm(v, w[key + ".weight"])

    start = 0
    keys = values = history = None
    rows, records = [], []
    for i, count in enumerate(chunks):
        end = start + count
        h = x[start:end]
        z = norm(h, "operator_norm")
        if kind == "conv":
            b, c, v = linear(z, "conv.in_proj").chunk(3, dim=-1)
            bx = b * v
            old = bx.new_zeros((6, base.HIDDEN)) if history is None else history
            u = torch.cat((old, bx))
            history = u[-6:].clone()
            v = torch.nn.functional.conv1d(
                u.T.unsqueeze(0).contiguous(), w["conv.conv.weight"].reshape(base.HIDDEN, 1, 7),
                groups=base.HIDDEN,
            ).squeeze(0).T
            result = linear(c * v, "conv.out_proj")
            records.append(record(output, layer, i, "history", history))
        else:
            local = kind == "sliding_attention"
            q = norm(linear(z, "self_attn.q_proj").reshape(count, base.HEADS, base.WIDTH), "self_attn.q_norm")
            k = norm(linear(z, "self_attn.k_proj").reshape(count, base.KV, base.WIDTH), "self_attn.k_norm")
            q = base.rotary(q, start, local).transpose(0, 1).unsqueeze(0)
            k = base.rotary(k, start, local).transpose(0, 1).contiguous()
            v = linear(z, "self_attn.v_proj").reshape(count, base.KV, base.WIDTH).transpose(0, 1).contiguous()
            origin = max(0, start - 255) if local else 0
            assert keys is None or keys.shape[1] == start - origin
            context_k = k if keys is None else torch.cat((keys, k), dim=1)
            context_v = v if values is None else torch.cat((values, v), dim=1)
            keys = context_k[:, -255:].contiguous() if local else context_k
            values = context_v[:, -255:].contiguous() if local else context_v
            key_pos = torch.arange(origin, end)[None, :]
            query_pos = torch.arange(start, end)[:, None]
            mask = key_pos <= query_pos
            if local:
                mask &= key_pos > query_pos - 256
            mask = torch.where(mask, torch.zeros((), dtype=z.dtype), torch.finfo(z.dtype).min)
            a, _ = eager(SimpleNamespace(num_key_value_groups=4, training=False), q,
                         context_k.unsqueeze(0), context_v.unsqueeze(0), mask[None, None], 128 ** -0.5)
            gate = linear(z, "self_attn.gate_proj").sigmoid()
            result = linear((a.squeeze(0)*gate[...,None]).reshape(count, base.HIDDEN), "self_attn.o_proj")
            records.append(record(output, layer, i, "key_proposal", k) |
                           dict(key_start=origin, logical_keys=end-origin, window=256 if local else None))
            records.append(record(output, layer, i, "value_proposal", v))
        residual = h + norm(result, "post_operator_norm")
        f = norm(residual, "ffn_norm")
        ff = linear(torch.nn.functional.silu(linear(f, "feed_forward.w1")) *
                    linear(f, "feed_forward.w3"), "feed_forward.w2")
        result = residual + norm(ff, "post_ffn_norm")
        rows.append(result)
        records.append(record(output, layer, i, "layer_output", result))
        start = end
    return torch.cat(rows), records


def collect(args):
    import torch
    base.torch = torch
    assert torch.version.git_version == TORCH_REVISION
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    contract_raw = args.contract.read_bytes()
    contract = json.loads(contract_raw)
    assert contract["format"] == "toklet-lfm3-context513-contract-v1"
    schedules = contract["schedules"]
    tokens = [124894]+[(i*3253+41)%124000 for i in range(1,513)]
    assert all(sum(s)==len(tokens) and max(s)<=128 for s in schedules)
    config = args.config.read_bytes()
    assert base.sha(config) == base.CONFIG_SHA
    routes = json.loads(config)["layer_types"]
    _, eager = source_functions(args.parent)
    args.output.mkdir()
    with args.model.open("rb", buffering=0) as stream, torch.inference_mode(), \
         (args.output/"source.bf16le").open("xb") as output:
        assert args.model.stat().st_size == MODEL_BYTES
        assert hashlib.file_digest(stream,"sha256").hexdigest()==MODEL_SHA
        stream.seek(0)
        n=struct.unpack("<Q",stream.read(8))[0]
        header=stream.read(n)
        assert base.sha(header)==HEADER_SHA
        schema=json.loads(header)
        origin=8+n

        def weight(name, start=0, rows=None):
            meta=schema[name]
            assert meta["dtype"]=="BF16"
            lo,hi=meta["data_offsets"]
            shape=meta["shape"]
            if rows is not None:
                assert len(shape)==2 and start+rows<=shape[0]
                lo+=start*shape[1]*2
                hi=lo+rows*shape[1]*2
                shape=[rows,shape[1]]
            buf=bytearray(hi-lo)
            stream.seek(origin+lo)
            assert stream.readinto(buf)==len(buf)
            return torch.frombuffer(buf,dtype=torch.bfloat16).reshape(shape)

        embeddings=torch.cat([weight("model.embed_tokens.weight", t, 1) for t in tokens])
        hidden=[embeddings]*len(schedules)
        reports=[dict(chunks=chunks,records=[]) for chunks in schedules]
        for layer,kind in enumerate(routes):
            prefix=f"model.layers.{layer}."
            weights={n.removeprefix(prefix):weight(n) for n in schema if n.startswith(prefix)}
            for i,chunks in enumerate(schedules):
                hidden[i],observations=execute_layer(hidden[i],weights,chunks,kind,eager,layer,output)
                reports[i]["records"].extend(observations)
            del weights
            print(f"captured layer {layer}",flush=True)

        final=[base.norm(x,weight("model.embedding_norm.weight")) for x in hidden]
        selected=[x[torch.tensor(s).cumsum(0)-1] for x,s in zip(final,schedules)]
        logits=[torch.empty((len(s),base.VOCAB),dtype=torch.bfloat16) for s in schedules]
        for start in range(0,base.VOCAB,1024):
            weights=weight("model.embed_tokens.weight",start,min(1024,base.VOCAB-start))
            for x,y in zip(selected,logits):
                y[:,start:start+weights.shape[0]]=torch.cat(
                    [torch.nn.functional.linear(row[None],weights) for row in x]
                )
        for report,chunks,x,y in zip(reports,schedules,final,logits):
            for i,(a,b) in enumerate(zip(x.split(chunks),y)):
                report["records"].append(record(output,None,i,"final_norm",a))
                report["records"].append(record(output,None,i,"logits",b))

        stream.seek(0)
        assert hashlib.file_digest(stream,"sha256").hexdigest()==MODEL_SHA
        total=output.tell()
    result=dict(format="toklet-lfm3-context513-source-v1",tokens=tokens,schedules=reports,
                source_bytes=total,source_sha256=MODEL_SHA,torch_revision=TORCH_REVISION,
                contract_sha256=base.sha(contract_raw),producer_sha256=base.sha(Path(__file__).read_bytes()),
                fixture_completed=True)
    (args.output/"fixture.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ["model","parent","config","contract","output"]:
        parser.add_argument("--"+option,type=Path,required=True)
    collect(parser.parse_args())
