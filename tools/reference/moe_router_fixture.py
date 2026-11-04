"""Independent selected-router oracle: sorting plus explicit IEEE FP32 stores."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)

def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]

def case(name, rows, selected):
    rows = [[f32(v) for v in row] for row in rows]
    ids, gates, wide = [], [], []
    for row in rows:
        chosen = sorted(range(len(row)), key=lambda i: (-row[i], i))[:selected]
        ordered = [f32(math.exp(f32(row[i] - row[chosen[0]]))) for i in chosen]
        total = 0.
        for value in ordered:
            total = f32(total + value)
        ids.extend(chosen)
        gates.extend(f32(value / total) for value in ordered)
        exact = [math.exp(float(row[i]) - float(row[chosen[0]])) for i in chosen]
        mass = math.fsum(exact)
        wide.extend(value / mass for value in exact)
    return dict(name=name, tokens=len(rows), experts=len(rows[0]), selected=selected,
                score_bits=[bits(v) for row in rows for v in row], ids=ids,
                gate_bits=[bits(v) for v in gates], wide_gates=wide)

def export():
    cases = [case('all-equal', [[0.] * 32], 8),
             case('boundary-ties', [[float(i // 4) for i in range(32)]], 8),
             case('negative', [[-float(i + 1) for i in range(32)]], 8),
             case('signed-zero', [[-0., 0., -0., 0.]], 2),
             case('extreme-finite', [[3.4e38, -3.4e38, 0., 3.4e38]], 3),
             case('single-expert', [[-4.]], 1),
             case('all-selected', [[-2., 2., 1., -1.]], 4)]
    for count in [7, 8, 9]:
        cases.append(case('batch-' + str(count),
                          [[((expert * 13 + token * 7) % 32 - 16) / 4 for expert in range(32)]
                           for token in range(count)], 8))
    scores = [float(i - 16) / 8 for i in range(32)]
    permutation = [(i * 13) % 32 for i in range(32)]
    cases.append(case('expert-permutation', [[scores[i] for i in permutation]], 8))
    path = Path(__file__).resolve()
    return dict(format='toklet-moe-router-v1', producer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                policy='descending score, ascending ID ties; selected-only FP32 softmax in ranked slot order',
                gate_atol=2e-7, gate_rtol=2e-6,
                scope='Routing leaf only; no expert projections, weighted reduction, Metal or model execution', cases=cases)

if __name__ == '__main__':
    with Path(sys.argv[1]).open('x') as file:
        json.dump(export(), file, indent=2, allow_nan=False)
        file.write('\n')
