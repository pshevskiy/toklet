"""Compare the layer-12 post-attention tail from frozen native operands."""
import hashlib
import json
import math
from pathlib import Path
import sys

import check_gemma_attention_boundary as attention
import check_gemma_norm_witness as norm_witness

METADATA = dict(layer=12, position=433, chunk_start=416, chunk_rows=32,
                context=1025, hidden=1536, ple=256, intermediate=6144)
FIELDS = [('input', 1536), ('attention', 1536), ('ple', 256), ('ff_up', 6144),
          ('ple_gate', 256), ('ple_projection', 1536), ('ple_norm', 1536), ('output', 1536)]
ARCHIVES = {
    'gemma-attention-boundary-2026-11-06': (
        'e7f77885d05ad9e0fff9cd8fba07efb993f4f4a149c8f6699cc62f7c73463ffa',
        '60538909a0921e1a5dba81ee6069b96bd7290799aace3da41e22cfcdeb9e825a',
    ),
    'gemma-norm-witness-2026-11-05': (
        '1c932281503a4c292d7f74167c01b1b59fe5b8c8d4e9a8d4f750efa1c7f82584',
        'aeb321604df730beeb950d91e6ef8b063e78cb96a1d7fba50ebcd6cb04ed7fd1',
    ),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def capture(log, body):
    events = []
    for line in log.decode().splitlines():
        if 'GEMMA_HYBRID_' in line:
            name, value = line.split('GEMMA_HYBRID_', 1)[1].split(' ', 1)
            events.append((name, json.loads(value)))
    expected = [('REAL_LAYER_TAIL', METADATA)]
    rows, offset = {}, 0
    for index, (name, count) in enumerate(FIELDS):
        raw = body[offset:offset + 4 * count]
        if len(raw) != 4 * count:
            raise ValueError('Truncated native tail array')
        row = dict(record=index, name='tail.' + name, offset=offset,
                   bytes=len(raw), before=433, count=1, values=count)
        expected.extend([('ARRAY_ATTEMPT', row), ('ARRAY', dict(row, sha256=sha(raw)))])
        rows[name] = raw
        offset += len(raw)
    expected.extend([
        ('REAL_LAYER_TAIL_COMPLETE', dict(committed=448, both_faults_preserved_prefix=True)),
        ('CAPTURE', dict(records=len(FIELDS), bytes=offset, sha256=sha(body),
                         whole_e2b_qualified=False)),
    ])
    if events != expected or len(body) != offset:
        raise ValueError('Native tail framing or array identity differs')
    return rows


def compare(rows, weights):
    import numpy as np

    actual = {name: np.frombuffer(raw, dtype='<f4') for name, raw in rows.items()}
    if not all(np.isfinite(values).all() for values in actual.values()):
        raise ValueError('Nonfinite native tail operand')

    base = 'layers.12.'
    eps = float(np.float32(1e-6))
    result = {}
    for contract, dtype in [('math_fp64', np.float64), ('staged_fp32', np.float32)]:
        def rounded(x):
            return np.asarray(x, dtype=dtype)

        def add(a, b):
            return rounded(a.astype(np.float64) + b.astype(np.float64))

        def multiply(a, b):
            return rounded(a.astype(np.float64) * b.astype(np.float64))

        def norm(x, suffix):
            scale = weights.vectors[base + suffix + '.weight']
            denominator = math.sqrt(math.fsum(float(v) * float(v) for v in x) / len(x) + eps)
            return rounded([float(v) / denominator * float(s) for v, s in zip(x, scale, strict=True)])

        def linear(x, suffix):
            return weights.linear(base + suffix + '.weight', x.reshape(1, -1), dtype).reshape(-1)

        def gelu(x):
            return rounded([.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) *
                            (v + .044715 * v**3))) for v in map(float, x)])

        after_attention = add(actual['input'], norm(actual['attention'], 'post_attention_layernorm'))
        ff_input = norm(after_attention, 'pre_feedforward_layernorm')
        ff_gate = linear(ff_input, 'mlp.gate_proj')
        ff_up = linear(ff_input, 'mlp.up_proj')
        ff_output = linear(multiply(gelu(ff_gate), ff_up), 'mlp.down_proj')
        after_ff = add(after_attention, norm(ff_output, 'post_feedforward_layernorm'))
        ple_gate = linear(after_ff, 'per_layer_input_gate')
        ple_projection = linear(multiply(gelu(ple_gate), actual['ple']), 'per_layer_projection')
        ple_norm = norm(ple_projection, 'post_per_layer_input_norm')
        output = multiply(add(after_ff, ple_norm), weights.vectors[base + 'layer_scalar'])
        result[contract] = {
            name: attention.prior.prior.fractions(actual[name], reference)
            for name, reference in [('ff_up', ff_up), ('ple_gate', ple_gate),
                                    ('ple_projection', ple_projection), ('ple_norm', ple_norm),
                                    ('output', output)]
        }
    return result


def run(directory):
    root = attention.prior.ROOT
    read = attention.prior.prior.bounded.read
    rows = capture(read(directory / 'gpu.stdout', 65536), read(directory / 'arrays.bin', 65536))
    archived = {}
    for name, pins in ARCHIVES.items():
        folder = root / 'artifacts' / name
        files = read(folder / 'gpu.stdout', 65536), read(folder / 'arrays.bin', 2 * 1024**2)
        if tuple(map(sha, files)) != pins:
            raise ValueError('Historical capture identity differs')
        archived[name] = files
    attention_rows = attention.native_operands(*archived['gemma-attention-boundary-2026-11-06'])
    norm_rows = norm_witness.read_rows(
        *archived['gemma-norm-witness-2026-11-05'], norm_witness.head.original_row())
    if rows['attention'] != attention_rows['attention.output'] or rows['output'] != norm_rows['norm.13.input']:
        raise ValueError('Layer tail does not join the archived attention and norm boundaries')

    source = attention.prior.oracle.source_graph

    def guard():
        for name, expected in source.source.PINS.items():
            if source.source.sha(source.source.MODEL / name) != expected:
                raise ValueError('Pinned packed source differs')

    guard()
    with (source.source.MODEL / 'model.safetensors').open('rb') as stream:
        metrics = compare(rows, source.weights(stream))
    guard()
    return dict(metrics=metrics, archived_boundary_operands_exact=True,
                local_passed=all(report['output']['failed_values'] == 0 for report in metrics.values()),
                whole_graph_qualified=False)


if __name__ == '__main__':
    report = run(Path(sys.argv[1]))
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(not report['local_passed'])
