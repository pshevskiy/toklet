"""Replay the native Gemma4 layer-12 attention and projection at position 433."""
import hashlib
import json
import math
from pathlib import Path
import sys

import check_gemma_hybrid1025 as prior
import gemma4_e2b_context_math as arithmetic

METADATA = dict(layer=12, producer=12, position=433, chunk_start=416,
                chunk_rows=32, head_dim=256, heads=8, kv_heads=1,
                window=512, start=0, end=448)
ARRAYS = [('attention.q', 433, 1, 8*256),
          ('attention.context', 433, 1, 8*256),
          ('attention.output', 433, 1, 1536),
          ('attention.0.k', 0, 416, 416*256),
          ('attention.0.v', 0, 416, 416*256),
          ('attention.2.k', 416, 32, 32*256),
          ('attention.2.v', 416, 32, 32*256)]


def native_operands(log, body):
    events = []
    for line in log.decode().splitlines():
        if 'GEMMA_HYBRID_' in line:
            _, event = line.split('GEMMA_HYBRID_', 1)
            name, data = event.split(' ', 1)
            events.append((name, json.loads(data)))
    if (not events or events[0] != ('REAL_ATTENTION_BOUNDARY', METADATA)
            or len(events) != 2 * len(ARRAYS) + 3):
        raise ValueError('Attention producer, geometry, or framing differs')
    offset = 0
    arrays = {}
    for index, (name, start, rows, count) in enumerate(ARRAYS):
        block = body[offset:offset + 4*count]
        attempt = dict(record=index, name=name, before=start, count=rows,
                       offset=offset, bytes=4*count, values=count)
        saved = dict(attempt, sha256=hashlib.sha256(block).hexdigest())
        if len(block) != 4*count or events[2*index+1:2*index+3] != [
                ('ARRAY_ATTEMPT', attempt), ('ARRAY', saved)]:
            raise ValueError('Captured attention span or array hash differs')
        arrays[name] = block
        offset += len(block)
    if (events[-2:] != [
            ('REAL_ATTENTION_BOUNDARY_COMPLETE',
             dict(committed=448, fault_preserved_prefix=True, committed_operands_equal=True)),
            ('CAPTURE', dict(records=len(ARRAYS), bytes=offset,
                             sha256=hashlib.sha256(body).hexdigest(), whole_e2b_qualified=False))]
            or offset != len(body)):
        raise ValueError('Complete attention capture required')
    return arrays


def compare(arrays, weights):
    import numpy as np

    def read(name, shape):
        row = np.frombuffer(arrays['attention.'+name], dtype='<f4').reshape(shape)
        if not np.isfinite(row).all():
            raise ValueError('Nonfinite actual attention value')
        return row

    q = read('q', (8, 256))
    context = read('context', (1, 2048))
    output = read('output', (1536,))
    k, v = (np.concatenate([read(f'0.{name}', (416, 256)), read(f'2.{name}', (32, 256))])
            for name in ('k', 'v'))
    # First local window starts at 0 and the causal tail ends at query position 433.
    k, v = k[:434], v[:434]
    result = {}
    for contract, dtype in [('math_fp64', np.float64), ('staged_fp32', np.float32)]:
        expected = []
        for head in q:
            scores = np.asarray([arithmetic.product_sum(head, key) for key in k], dtype=dtype)
            maximum = float(max(scores))
            exponential = [math.exp(float(value) - maximum) for value in scores]
            mass = math.fsum(exponential)
            probabilities = np.asarray([value / mass for value in exponential], dtype=dtype)
            expected.extend(arithmetic.product_sum(probabilities, column) for column in v.T)
        projected = weights.linear('layers.12.self_attn.o_proj.weight', context, dtype).reshape(-1)
        result[contract] = dict(
            context=prior.prior.fractions(context.reshape(-1), np.asarray(expected, dtype=dtype)),
            projection=prior.prior.fractions(output, projected))
    return result


def run(directory):
    proof = json.loads((prior.ROOT/'benchmarks/results/2026-11-04-gemma-hybrid1025-v1.json').read_text())
    if not proof['state_passed'] or proof['numerical_passed']:
        raise ValueError('Prior state pass and numerical rejection required')
    archived = prior.ROOT/'artifacts/gemma-hybrid1025-2026-11-04'
    read = prior.prior.bounded.read
    previous_log = read(archived/'native.stdout', 2*1024**2)
    previous_body = read(archived/'arrays.bin', 128*1024**2)
    for name, data in [('native.stdout', previous_log), ('arrays.bin', previous_body)]:
        path = str((archived/name).relative_to(prior.ROOT))
        if hashlib.sha256(data).hexdigest() != proof['receipts_sha256'][path]:
            raise ValueError('Historical native identity differs')
    _, retained, _ = prior.native_capture(previous_log, previous_body)
    arrays = native_operands(read(directory/'gpu.stdout', 65536),
                             read(directory/'arrays.bin', 2*1024**2))
    for name in ('k','v'):
        combined = arrays[f'attention.0.{name}'] + arrays[f'attention.2.{name}']
        if retained[512, 12, name][:len(combined)] != combined:
            raise ValueError('Consumed K/V differs from archived producer-12 rows')
    source = prior.oracle.source_graph

    def guard():
        for name, expected in source.source.PINS.items():
            if source.source.sha(source.source.MODEL/name) != expected:
                raise ValueError('Pinned original packed source differs')

    guard()
    with (source.source.MODEL/'model.safetensors').open('rb') as stream:
        metrics = compare(arrays, source.weights(stream))
    guard()
    return dict(metrics=metrics, exact_archived_operands=True,
                local_passed=all(x['failed_values']==0 for values in metrics.values() for x in values.values()),
                earlier_full_gate_remains_open=True)


if __name__ == '__main__':
    result = run(Path(sys.argv[1]))
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(not result['local_passed'])
