"""Independent ratio-one Qwen3.5 fixture with exact F32 A_log parameters.

Reuses the unchanged independently qualified scalar arithmetic, with explicit
geometry and typed input bytes. No target/model/native output is consumed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import qwen35_cpu_graph as arithmetic

GEOMETRY = dict(hidden=64, intermediate=128, vocab=32, layers=4,
                query_heads=4, kv_heads=2, head_dim=16, rotated_dims=8,
                key_heads=2, value_heads=2, key_dim=32, value_dim=32,
                max_positions=4096, rope_theta=10000000.0, rms_epsilon=1e-6)
GATE = dict(atol=1e-5, rtol=1e-4)
FORMAT = 'toklet-qwen35-cpu-graph-v1'
F32_A_LOG = (-0.9876543283462524, -0.3333333432674408)
ROOT = Path(__file__).resolve().parents[2]


def specs():
    result = arithmetic.specs(GEOMETRY)
    for name, (kind, shape) in list(result.items()):
        if name.endswith('linear_attn.A_log'):
            result[name] = ('f32', shape)
    return result


def artifact():
    raw, table, matrices, vectors, backing = bytearray(), {}, {}, {}, 0
    def save(data):
        offset = len(raw)
        raw.extend(data)
        return dict(offset=offset, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    for role, (name, (kind, shape)) in enumerate(sorted(specs().items())):
        if kind == 'matrix':
            rows, columns = shape
            codes, scales, biases, values = bytearray(), bytearray(), bytearray(), []
            for row in range(rows):
                output = []
                for group in range(columns // 64):
                    scale = (0.015625, -0.0078125, 0.00390625)[(row + group + role) % 3]
                    bias = -7 * scale + ((row + role) % 3 - 1) * 0.00390625
                    sw, scale = arithmetic.bf16(scale)
                    bw, bias = arithmetic.bf16(bias)
                    scales.extend(struct.pack('<H', sw)); biases.extend(struct.pack('<H', bw))
                    nibbles = [(row * 11 + column * 7 + (row ^ (column >> 2)) * 3 + role * 5) % 16
                               for column in range(group * 64, (group + 1) * 64)]
                    codes.extend(bytes(nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, 64, 2)))
                    output.extend(arithmetic.f32(scale * value + bias) for value in nibbles)
                values.append(output)
            table[name] = dict(kind=kind, shape=shape, codes=save(codes), scales=save(scales), biases=save(biases))
            matrices[name] = values
            backing += len(codes) + len(scales) + len(biases)
        else:
            count = math.prod(shape)
            if kind == 'f32':
                values = list(F32_A_LOG)
                data = struct.pack('<2f', *values)
            else:
                if name.endswith('dt_bias'):
                    values = [0.125, -0.25]
                elif 'conv1d' in name:
                    values = [(-0.125, 0.25, -0.5, 1.0)[i % 4] + ((i // 4 + role) % 3 - 1) * 0.015625
                              for i in range(count)]
                else:
                    values = [1 + ((i + role) % 5 - 2) * 0.03125 for i in range(count)]
                encoded = [arithmetic.bf16(value) for value in values]
                data = b''.join(struct.pack('<H', word) for word, _ in encoded)
                values = [value for _, value in encoded]
            if len(values) != count or not all(math.isfinite(v) for v in values):
                raise ValueError('Exact finite typed fixture extent required')
            table[name] = dict(kind=kind, shape=shape, data=save(data))
            vectors[name] = values
            backing += count * 4
    return bytes(raw), table, matrices, vectors, backing


def sources():
    return (Path(__file__).resolve(), Path(arithmetic.__file__).resolve())


def generate(destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Fresh typed fixture directory required')
    weights, table, matrices, vectors, backing = artifact()
    body, cases = bytearray(), []
    for name, tokens, offset in arithmetic.CASES:
        profiles = {}
        for profile, staged in (('ideal_fp64', False), ('staged_fp32', True)):
            arrays = []
            for observation in arithmetic.forward(matrices, vectors, tokens, offset, staged, geometry=GEOMETRY):
                row = dict(observation)
                values = row.pop('values')
                raw = struct.pack('<' + ('f' if staged else 'd') * len(values), *values)
                arrays.append(dict(**row, offset=len(body), bytes=len(raw), count=len(values), sha256=hashlib.sha256(raw).hexdigest()))
                body.extend(raw)
                if len(body) > 16 * 1024**2:
                    raise ValueError('Typed tiny fixture exceeds bound')
            profiles[profile] = arrays
        cases.append(dict(name=name, tokens=list(tokens), position_offset=offset, profiles=profiles))
    manifest = dict(format=FORMAT, geometry=GEOMETRY, gate=GATE, weights=table, backing_bytes=backing,
                    weights_bytes=len(weights), weights_sha256=hashlib.sha256(weights).hexdigest(),
                    reference_bytes=len(body), reference_sha256=hashlib.sha256(body).hexdigest(), cases=cases,
                    fixture_only=True, real_model_qualified=False, typed_F32_A_log=True,
                    producers={str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources()})
    destination.mkdir()
    (destination / 'weights.bin').write_bytes(weights)
    (destination / 'reference.bin').write_bytes(body)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return dict(weights_bytes=len(weights), reference_bytes=len(body), arrays=sum(len(p) for c in cases for p in c['profiles'].values()),
                manifest_sha256=hashlib.sha256((destination / 'manifest.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(json.dumps(generate(parser.parse_args().output)))
