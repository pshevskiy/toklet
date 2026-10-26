"""Freeze deterministic KV ownership data before session-capacity changes.

No engine, graph, numerical package or model file is imported. Two KV heads
deliberately expose physical-stride differences between wide and narrow caches.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct

CAPACITIES = (1, 2, 3, 4, 5, 17, 32)


def prepare(output):
    output.mkdir()
    body = bytearray(); producers = []
    for producer in range(15):
        full = producer % 5 == 4; dims = 3 if full else 2
        record = dict(producer=producer, kind='full' if full else 'sliding', heads=2, head_dim=dims)
        for kind in ('k', 'v'):
            raw = bytearray()
            for head in range(2):
                for position in range(32):
                    for feature in range(dims):
                        value = (producer * 1024 + head * 128 + position * 8 + feature + .25 if kind == 'k'
                                 else -(producer * 512 + head * 64 + position * 4 + feature + .5))
                        raw.extend(struct.pack('<f', value))
            record[kind] = dict(shape=[2, 32, dims], dtype='<f4', offset=len(body), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            body.extend(raw)
        producers.append(record)
    cases = [dict(capacity=capacity, chunk_rows=chunk,
                  steps=(capacity + chunk - 1) // chunk,
                  view_records=2 * 35 * ((capacity + chunk - 1) // chunk))
             for capacity in CAPACITIES for chunk in sorted({1, min(3, capacity), min(8, capacity), capacity})]
    manifest = dict(format='toklet-session-capacity-oracle-v1', window=512,
                    original_global_capacity=32, original_max_positions=512,
                    layers=35, producer_count=15,
                    routes=[layer if layer < 15 else 14 if layer % 5 == 4 else 13 for layer in range(35)],
                    producers=producers, cases=cases,
                    binary=dict(file='kv.bin', bytes=len(body), sha256=hashlib.sha256(body).hexdigest()),
                    real_model=False, numerical_graph_executed=False, larger_context_qualified=False,
                    full_model_window_qualified=False, app_eligibility_changed=False)
    if len(body) != 16896 or sum(case['view_records'] for case in cases) != 6860:
        raise ValueError('fixed capacity witness size differs')
    with (output / 'kv.bin').open('xb') as file:
        if file.write(body) != len(body): raise OSError('short KV oracle write')
    encoded = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    with (output / 'manifest.json').open('xb') as file:
        if file.write(encoded) != len(encoded): raise OSError('short capacity manifest write')
    return dict(passed=True, manifest_bytes=len(encoded), manifest_sha256=hashlib.sha256(encoded).hexdigest(), binary=manifest['binary'], cases=len(cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().output)))
