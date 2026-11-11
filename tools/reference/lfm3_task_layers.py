"""Pinned eager LFM3 reference arithmetic with explicit per-layer task state."""
from dataclasses import dataclass
from types import SimpleNamespace

import torch

import lfm3_graph_bf16_fixture as base
from lfm3_context1025_fixture import grouped


@dataclass
class LayerCache:
    """Six convolution rows or logical K/V [heads, rows, width] in BF16."""

    tensors: tuple


def forward_layer(hidden, weights, kind, start, cache, eager):
    count = len(hidden)
    end = start + count
    assert 0 < count <= 128 and 0 <= start < end <= 2048
    assert hidden.shape == (count, base.HIDDEN) and hidden.dtype == torch.bfloat16
    assert (cache is None) == (start == 0)

    def linear(value, name):
        return torch.nn.functional.linear(value, weights[name + ".weight"])

    def norm(value, name):
        return base.norm(value, weights[name + ".weight"])

    prepared = norm(hidden, "operator_norm")
    if kind == "conv":
        b, c, v = linear(prepared, "conv.in_proj").chunk(3, dim=-1)
        history = b.new_zeros((6, base.HIDDEN)) if cache is None else cache.tensors[0]
        assert history.shape == (6, base.HIDDEN) and history.dtype == torch.bfloat16
        convolution = torch.cat((history, b * v))
        mixed = torch.nn.functional.conv1d(
            convolution.T.unsqueeze(0).contiguous(),
            weights["conv.conv.weight"].reshape(base.HIDDEN, 1, 7),
            groups=base.HIDDEN,
        ).squeeze(0).T
        mixed = linear(c * mixed, "conv.out_proj")
        new_cache = LayerCache((convolution[-6:].clone(),))
    else:
        assert kind in ("full_attention", "sliding_attention")
        local = kind == "sliding_attention"
        q = norm(linear(prepared, "self_attn.q_proj").reshape(count, base.HEADS, base.WIDTH), "self_attn.q_norm")
        k = norm(linear(prepared, "self_attn.k_proj").reshape(count, base.KV, base.WIDTH), "self_attn.k_norm")
        q = base.rotary(q, start, local).transpose(0, 1).unsqueeze(0)
        k = base.rotary(k, start, local).transpose(0, 1).contiguous()
        v = linear(prepared, "self_attn.v_proj").reshape(count, base.KV, base.WIDTH).transpose(0, 1).contiguous()
        origin = max(0, start - 255) if local else 0
        if cache is None:
            keys, values = k, v
        else:
            old_k, old_v = cache.tensors
            assert old_k.shape == old_v.shape == (base.KV, start - origin, base.WIDTH)
            assert old_k.dtype == old_v.dtype == torch.bfloat16
            keys, values = torch.cat((old_k, k), dim=1), torch.cat((old_v, v), dim=1)
        key_pos = torch.arange(origin, end)[None, :]
        query_pos = torch.arange(start, end)[:, None]
        visible = key_pos <= query_pos
        if local:
            visible &= key_pos > query_pos - 256
        mask = torch.where(visible, torch.zeros((), dtype=hidden.dtype),
                           torch.finfo(hidden.dtype).min)[None, None]
        module = SimpleNamespace(num_key_value_groups=4, training=False)
        if keys.shape[1] > 513:
            attention, _ = grouped(eager, module, q, keys.unsqueeze(0), values.unsqueeze(0), mask, 128 ** -0.5)
        else:
            attention, _ = eager(module, q, keys.unsqueeze(0), values.unsqueeze(0), mask, 128 ** -0.5)
        gate = linear(prepared, "self_attn.gate_proj").sigmoid()
        mixed = linear((attention.squeeze(0) * gate[..., None]).reshape(count, base.HIDDEN), "self_attn.o_proj")
        new_cache = LayerCache((keys[:, -255:].contiguous(), values[:, -255:].contiguous()) if local else (keys, values))

    residual = hidden + norm(mixed, "post_operator_norm")
    ffn = norm(residual, "ffn_norm")
    ffn = linear(torch.nn.functional.silu(linear(ffn, "feed_forward.w1")) *
                 linear(ffn, "feed_forward.w3"), "feed_forward.w2")
    return residual + norm(ffn, "post_ffn_norm"), new_cache
