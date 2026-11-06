"""Independent original-BF16-weight, staged-FP32 Gemma4 assistant reference.

The binary inputs contain one scaled 1536-wide target embedding, one target
post-normalization hidden row, then chronological local K/V and full K/V.
A single round attends through the last committed position inclusively.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import struct

import numpy as np

ASSISTANT_SHA = '93682eb1c97639d18f007704dc880bd74cbe530adaf7b1bb561213863fdad2a6'


class Weights:
    def __init__(self, source):
        with source.open('rb') as file:
            if os.fstat(file.fileno()).st_size != 157_565_344:
                raise ValueError('Assistant extent differs')
            if hashlib.file_digest(file, 'sha256').hexdigest() != ASSISTANT_SHA:
                raise ValueError('Assistant SHA-256 differs')
            file.seek(0)
            length, = struct.unpack('<Q', file.read(8))
            if length != 5520:
                raise ValueError('Assistant header length differs')
            self.header = json.loads(file.read(length))
            self.payload = np.memmap(file, dtype=np.uint8, offset=8 + length, mode='r')

    def bf16(self, name, rows=None):
        entry = self.header[name]
        first, end = entry['data_offsets']
        if entry['dtype'] != 'BF16':
            raise ValueError('BF16 weights required')
        data = self.payload[first:end].view('<u2').reshape(entry['shape'])
        if rows is not None:
            data = data[rows]
        return (data.astype('<u4') << 16).view('<f4').astype(np.float64)

    def ordering(self):
        first, end = self.header['masked_embedding.token_ordering']['data_offsets']
        return self.payload[first:end].view('<i8').reshape(2048, 128)


def norm(source, gain):
    x = source.astype(np.float64)
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + float(np.float32(1e-6))) * gain).astype(np.float32)


def rotate(query, position, full):
    width = query.shape[-1]
    count = (width // 4 if full else width) // 2
    theta = 1_000_000 if full else 10_000
    angles = position * np.power(float(theta), -2 * np.arange(count, dtype=np.float64) / width)
    s = np.sin(angles).astype(np.float32)
    c = np.cos(angles).astype(np.float32)
    output = query.copy()
    first = query[:, :count]
    second = query[:, width // 2:width // 2 + count]
    output[:, :count] = first * c - second * s
    output[:, width // 2:width // 2 + count] = second * c + first * s
    return output


def evaluate(weights, concatenated, local, full, position):
    def linear(name, x, rows=None):
        return (weights.bf16(name, rows) @ x.astype(np.float64)).astype(np.float32)

    hidden = linear('pre_projection.weight', concatenated)
    outputs = {'pre_projection': hidden}
    for i in range(4):
        prefix = f'model.layers.{i}.'
        keys, values = full if i == 3 else local
        d = keys.shape[-1]
        x = norm(hidden, weights.bf16(prefix + 'input_layernorm.weight'))
        q = linear(prefix + 'self_attn.q_proj.weight', x).reshape(4, d)
        q = norm(q, weights.bf16(prefix + 'self_attn.q_norm.weight'))
        q = rotate(q, position, i == 3)
        scores = (q.astype(np.float64) @ keys.astype(np.float64).T).astype(np.float32)
        exps = np.exp(scores.astype(np.float64) - np.max(scores, axis=-1, keepdims=True))
        probs = (exps / np.sum(exps, axis=-1, keepdims=True)).astype(np.float32)
        ctx = (probs.astype(np.float64) @ values.astype(np.float64)).astype(np.float32).reshape(-1)
        x = linear(prefix + 'self_attn.o_proj.weight', ctx)
        hidden = hidden + norm(x, weights.bf16(prefix + 'post_attention_layernorm.weight'))
        x = norm(hidden, weights.bf16(prefix + 'pre_feedforward_layernorm.weight'))
        gate = linear(prefix + 'mlp.gate_proj.weight', x).astype(np.float64)
        up = linear(prefix + 'mlp.up_proj.weight', x)
        cubic = gate + 0.044715 * gate * gate * gate
        gelu = (0.5 * gate * (1 + np.tanh(math.sqrt(2 / math.pi) * cubic))).astype(np.float32)
        x = linear(prefix + 'mlp.down_proj.weight', gelu * up)
        hidden = hidden + norm(x, weights.bf16(prefix + 'post_feedforward_layernorm.weight'))
        hidden = hidden * weights.bf16(prefix + 'layer_scalar').astype(np.float32)
        outputs[f'layer_{i}'] = hidden
    final = norm(hidden, weights.bf16('model.norm.weight'))
    outputs['final_norm'] = final
    outputs['post_projection'] = linear('post_projection.weight', final)
    scores = linear('masked_embedding.centroids.weight', final)
    centroid_ids = np.lexsort((np.arange(2048), -scores))[:32]
    ids = weights.ordering()[centroid_ids].reshape(-1)
    logits = linear('model.embed_tokens.weight', final, ids)
    outputs.update(centroid_scores=scores, ids=ids, logits=logits)
    order = np.lexsort((ids, -logits))[:2]
    return outputs, int(ids[order[0]]), float(logits[order[0]]) - float(logits[order[1]])


def inputs(path, position):
    local_keys = min(position + 1, 512)
    full_keys = position + 1
    expected = 3072 + 2 * (256 * local_keys + 512 * full_keys)
    with path.open('rb') as file:
        if os.fstat(file.fileno()).st_size != expected * 4:
            raise ValueError('Snapshot extent differs')
        x = np.fromfile(file, dtype='<f4', count=expected)
    if x.size != expected or not np.isfinite(x).all():
        raise ValueError('Finite snapshot of expected shape required')
    offset = 3072
    states = []
    for rows, width in ((local_keys, 256), (full_keys, 512)):
        count = rows * width
        key = x[offset:offset + count].reshape(rows, width)
        value = x[offset + count:offset + 2 * count].reshape(rows, width)
        states.append((key, value))
        offset += 2 * count
    return x[:3072], *states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assistant', type=Path)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('position', type=int, choices=range(1025), metavar='[0..1024]')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    arrays, token, margin = evaluate(Weights(args.assistant), *inputs(args.snapshot, args.position), args.position)
    args.output.mkdir()
    records = []
    with (args.output / 'reference.bin').open('xb') as f:
        for name, value in arrays.items():
            dtype = '<u4' if name == 'ids' else '<f4'
            data = value.astype(dtype).tobytes()
            records.append(dict(name=name, dtype=dtype, elements=value.size, offset=f.tell(), sha256=hashlib.sha256(data).hexdigest()))
            f.write(data)
    receipt = dict(assistant_sha256=ASSISTANT_SHA, position=args.position,
                   snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                   winner=token, margin=margin, arrays=records)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
