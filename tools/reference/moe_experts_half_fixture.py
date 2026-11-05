"""Independent Half boundaries over the scalar expert fixture inputs."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import moe_experts_fixture as scalar


def half(value):
    return struct.unpack('<e', struct.pack('<e', value))[0]


def words(values):
    return [struct.unpack('<H', struct.pack('<e', value))[0] for value in values]


def rows(values, width):
    return [values[i:i + width] for i in range(0, len(values), width)]


def decode(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def dense(case):
    data = case['arrays']
    hidden, intermediate, selected = (case[k] for k in ('hidden', 'intermediate', 'selected'))
    inputs, fused, down = ([half(decode(x)) for x in data[k]] for k in ('input', 'gate_up', 'down'))
    fused_rows = rows(rows(fused, hidden), 2 * intermediate)
    down_rows = rows(rows(down, intermediate), hidden)
    gate, up, activated, values = [], [], [], []
    for token, ids in zip(rows(inputs, hidden), rows(case['ids'], selected), strict=True):
        for expert in ids:
            projections = [half(scalar.dot(token, weight)) for weight in fused_rows[expert]]
            g, u = projections[:intermediate], projections[intermediate:]
            glu = [half(scalar.f32(half(scalar.silu(a)) * b)) for a, b in zip(g, u, strict=True)]
            gate.extend(g)
            up.extend(u)
            activated.extend(glu)
            values.extend(half(scalar.dot(glu, weight)) for weight in down_rows[expert])
    return encode(case, inputs, fused, down, gate, up, activated, values)


def encode(case, inputs, fused, down, gate, up, activated, values):
    selected, hidden = case['selected'], case['hidden']
    gate_bits = case['arrays']['gates']
    output = []
    for block, weights in zip(rows(rows(values, hidden), selected), rows(list(map(decode, gate_bits)), selected), strict=True):
        output.extend(half(scalar.dot(column, weights)) for column in zip(*block, strict=True))
    result = {k: v for k, v in case.items() if k != 'arrays'}
    result.update(input=words(inputs), gate_bits=gate_bits, gate=words(gate), up=words(up),
                  activated=words(activated), values=words(values), output=words(output))
    if fused is not None:
        result.update(fused=words(fused), down=words(down))
    return result


def sparse():
    # Each expert writes one distinct output column. The weight coordinates are
    # frozen as triples (expert, output row, input column), not packed offsets.
    ids = [31, 1, 16, 3, 8, 5, 6, 0]
    inputs = [0.] * 1024
    inputs[1023] = 1.
    inputs[511] = -0.5
    fused_coordinates, down_coordinates = [], []
    gate, up, activated, values = [], [], [], []
    for expert in ids:
        g, u = [0.] * 512, [0.] * 512
        g[511] = (expert + 1) / 32
        u[511] = -0.5
        fused_coordinates.extend([[expert, 511, 1023, words([g[511]])[0]], [expert, 1023, 511, words([1.])[0]]])
        down_coordinates.append([expert, 1023 - expert, 511, words([0.5])[0]])
        glu = [0.] * 512
        glu[511] = half(scalar.f32(half(scalar.silu(g[511])) * u[511]))
        out = [0.] * 1024
        out[1023 - expert] = half(glu[511] * 0.5)
        gate.extend(g)
        up.extend(u)
        activated.extend(glu)
        values.extend(out)
    case = dict(name='granite-geometry', tokens=1, experts=32, selected=8, hidden=1024,
                intermediate=512, ids=ids, arrays=dict(gates=[scalar.bits(0.125)] * 8))
    result = encode(case, inputs, None, None, gate, up, activated, values)
    result.update(fused_coordinates=fused_coordinates, down_coordinates=down_coordinates)
    return result


def rounding_boundaries():
    inputs = [1.003, -0.813, 0.329]
    weights = [[0.31, 0.31, 0.31, 0.31, -0.122, 0.674],
               [0.31, 0.31, 1.014, 0.31, 0.31, 0.31]]
    case = dict(name='gate-and-up-rounding', tokens=1, experts=2, selected=2,
                hidden=3, intermediate=1, ids=[0, 1],
                arrays=dict(input=list(map(scalar.bits, inputs)),
                            gate_up=list(map(scalar.bits, weights[0] + weights[1])),
                            down=[scalar.bits(1.)] * 6, gates=[scalar.bits(0.5)] * 2))
    result = dense(case)
    # Removing either initial Half store changes an observable GLU result.
    for expert in [0, 1]:
        gate, up = [scalar.dot(list(map(half, inputs)), list(map(half, weight)))
                    for weight in rows(weights[expert], 3)]
        bypass = half(scalar.f32(half(scalar.silu(half(gate) if expert == 0 else gate))
                                 * (up if expert == 0 else half(up))))
        assert result['activated'][expert] != words([bypass])[0]
    return result


def export():
    source = scalar.export()
    controls = [dict(name='ranked-cancellation', values=words([4096., 1., -4096.]),
                     gates=list(map(scalar.bits, [4096., 1., 4096.])), expected=0),
                dict(name='reordered', values=words([4096., -4096., 1.]),
                     gates=list(map(scalar.bits, [4096., 4096., 1.])), expected=0x3c00),
                dict(name='separate-product', values=words([-4096., 4100.]),
                     gates=list(map(scalar.bits, [1., 1. - 2**-10 + 2**-20])), expected=0)]
    for control in controls:
        values = [struct.unpack('<e', struct.pack('<H', x))[0] for x in control['values']]
        assert words([half(scalar.dot(values, list(map(decode, control['gates']))))])[0] == control['expected']
    return dict(format='toklet-moe-experts-half-v1',
                producers=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__, scalar.__file__]],
                cases=[dense(c) for c in source['cases']] + [rounding_boundaries(), sparse()], reduction=controls)


if __name__ == '__main__':
    with Path(sys.argv[1]).open('x') as file:
        json.dump(export(), file, separators=(',', ':'), allow_nan=False)
        file.write('\n')
