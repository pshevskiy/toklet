"""Pre-target exact integer KV witness; no Toklet implementation imports.

Two heads, local W512 versus full-prefix storage, fifteen producers and 35
routes. Values encode producer/head/absolute-position/feature without rounding.
Expected views concatenate logical old rows and proposed rows independently of
ring iteration; committed slots are assigned by the documented modulo rule.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct

CAPACITIES = (511, 512, 513, 1024, 1025)
CUTS = (511, 512, 513, 1024, 1025)


def value(producer, head, position, feature, v=False):
    x = producer * 100000 + head * 10000 + position * 4 + feature + 1
    return -x if v else x


def digest(producer, positions, dims, v=False):
    body = bytearray()
    for head in range(2):
        for position in positions:
            for feature in range(dims):
                body.extend(struct.pack('<f', 0 if position is None else value(producer, head, position, feature, v)))
    return hashlib.sha256(body).hexdigest()


def schedule(capacity, crossing=False):
    if crossing:
        assert capacity == 1025
        return [32]*15+[30,7]+[32]*15+[21,7]
    steps = []; before = 0; turn = 0
    while before < capacity:
        next_cut = min(x for x in (*CUTS, capacity) if x > before)
        count = min((31, 7, 1, 32, 3)[turn % 5], next_cut - before, capacity - before)
        steps.append(count); before += count; turn += 1
    return steps


def produce():
    kinds = ['full' if layer % 5 == 4 else 'local' for layer in range(35)]
    routes = []; last = {}
    for layer, kind in enumerate(kinds):
        if layer < 15: last[kind] = layer
        routes.append(last[kind])
    cases = []
    for capacity, crossing in [*( (c, False) for c in CAPACITIES), (1025, True)]:
        before = 0; steps = []
        for count in schedule(capacity, crossing):
            after = before + count; owners = []
            for producer in range(15):
                local = kinds[producer] == 'local'; dims = 2 if local else 3
                slots = min(512, capacity) if local else capacity
                start = max(0, before - slots + 1) if local else 0
                old = list(range(start, before)); segments = []
                # Group old positions by physically contiguous modulo slots.
                for position in old:
                    if not segments or position % slots == 0: segments.append([])
                    segments[-1].append(position)
                segments.append(list(range(before, after)))
                spans = [dict(start=ps[0], keys=len(ps), head_stride=(count if ps[0] == before else slots)*dims,
                              k_sha256=digest(producer, ps, dims), v_sha256=digest(producer, ps, dims, True)) for ps in segments if ps]
                ring = [None] * slots
                for position in range(after): ring[position % slots] = position
                visible = [[max(0, q-slots+1) if local else 0, q+1] for q in range(before, after)]
                owners.append(dict(producer=producer, dims=dims, start=start, end=after,
                                   k_sha256=digest(producer, range(start, after), dims),
                                   v_sha256=digest(producer, range(start, after), dims, True),
                                   segments=spans, visible=visible,
                                   committed_k_sha256=digest(producer, ring, dims),
                                   committed_v_sha256=digest(producer, ring, dims, True)))
            steps.append(dict(before=before, count=count, producers=owners)); before=after
        cases.append(dict(capacity=capacity, crossing=crossing, chunks=schedule(capacity, crossing), steps=steps))
    return dict(format='toklet-gemma-long-capacity-integer-kv-v1', heads=2, window=512,
                layers=kinds, shared_layers=20, routes=routes, capacities=list(CAPACITIES),
                recipe='signed(producer*100000+head*10000+position*4+feature+1)-exact-f32',
                native_context_raised=False, real_graph_qualified=False, cases=cases)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    record=produce();args.output.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(record,indent=2)+'\n').encode()
    with args.output.open('xb')as file:file.write(raw)
    print(json.dumps(dict(passed=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),cases=len(record['cases']),steps=sum(len(v['steps'])for v in record['cases']))))


if __name__=='__main__':main()
