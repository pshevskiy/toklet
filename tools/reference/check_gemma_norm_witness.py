"""Replay two weighted Gemma RMSNorm boundaries from actual native inputs."""
import hashlib
import json
import math
from pathlib import Path
import sys

import check_gemma_head_witness as head
import check_gemma_v_boundary as v_boundary

ROOT = head.ROOT
HIDDEN = head.HIDDEN
STAGES = [('norm.13.input', 433), ('norm.13.output', 433),
          ('norm.final.input', 1024), ('norm.final.output', 1024)]


def read_rows(transcript, body, archived):
    events = []
    marker = 'GEMMA_HYBRID_'
    for line in transcript.decode().splitlines():
        if marker in line:
            prefix, payload = line.split(marker, 1)
            assert not prefix or prefix == (
                'test model::gemma_packed::tests::hybrid::real::wrap::'
                'diagnostic1025_same_input_norms ... '
            )
            label, value = payload.split(' ', 1)
            events.append((label, json.loads(value)))
    assert len(body) == len(STAGES)*HIDDEN*4 and len(events) == 10
    rows = {}
    for index, (name, position) in enumerate(STAGES):
        offset = index*HIDDEN*4
        raw = memoryview(body)[offset:offset+HIDDEN*4]
        expected = dict(record=index, name=name, offset=offset, bytes=HIDDEN*4,
                        before=position, count=1, values=HIDDEN)
        assert events[index*2] == ('ARRAY_ATTEMPT', expected)
        assert events[index*2+1] == ('ARRAY', dict(expected, sha256=hashlib.sha256(raw).hexdigest()))
        rows[name] = raw
    assert events[-2] == ('REAL_FINAL_NORM_ROW', dict(position=1024, sha256=hashlib.sha256(archived).hexdigest()))
    assert events[-1] == ('CAPTURE', dict(records=4, bytes=len(body),
                                         sha256=hashlib.sha256(body).hexdigest(),
                                         whole_e2b_qualified=False))
    return rows


def compare(directory):
    import numpy as np

    read = head.prior.prior.bounded.read
    rows = read_rows(read(directory/'gpu.stdout', 65536), read(directory/'arrays.bin', 32768), head.original_row())
    proof = json.loads((ROOT/'benchmarks/results/2026-11-04-gemma-v-boundary-v1.json').read_text())
    prior = ROOT/'artifacts/gemma-v-boundary-2026-11-04'
    logs, body = read(prior/'native.stdout', 65536), read(prior/'arrays.bin', 32768)
    for name, data in [('native.stdout', logs), ('arrays.bin', body)]:
        key = str((prior/name).relative_to(ROOT))
        assert hashlib.sha256(data).hexdigest() == proof['receipts_sha256'][key]
    assert rows['norm.13.output'] == v_boundary.native_rows(logs, body)[13, 'input']
    head_proof = json.loads((ROOT/'benchmarks/results/2026-11-05-gemma-head-witness-v1.json').read_text())
    assert head_proof['passed'] and head_proof['local_head_passed']
    assert hashlib.sha256(rows['norm.final.output']).hexdigest() == head_proof['normalized_row_sha256']

    source = head.prior.oracle.source_graph
    def guard():
        for name, expected in source.source.PINS.items():
            assert source.source.sha(source.source.MODEL/name) == expected
    guard()
    result = {}
    with (source.source.MODEL/'model.safetensors').open('rb') as stream:
        weights = source.weights(stream)
        for stage, gain in [('norm.13', 'layers.13.input_layernorm.weight'),
                            ('norm.final', 'norm.weight')]:
            x = np.frombuffer(rows[stage+'.input'], dtype='<f4')
            y = np.frombuffer(rows[stage+'.output'], dtype='<f4')
            assert np.isfinite(x).all() and np.isfinite(y).all()
            scale = weights.vectors[gain]
            assert scale.shape == (HIDDEN,)
            eps = float(np.float32(1e-6))
            denom = math.sqrt(math.fsum(float(v)*float(v) for v in x)/HIDDEN+eps)
            expected = np.array([float(v)/denom*float(s) for v,s in zip(x,scale,strict=True)])
            result[stage] = {name: head.prior.prior.fractions(y, expected.astype(dtype))
                             for name,dtype in [('math_fp64',np.float64),('staged_fp32',np.float32)]}
    guard()
    return dict(local_norms_passed=all(report['failed_values']==0 for stage in result.values()
                                      for report in stage.values()), comparisons=result,
                archived_head_and_v_inputs_exact=True, whole_graph_qualified=False)


if __name__ == '__main__':
    print(json.dumps(compare(Path(sys.argv[1])), indent=2, allow_nan=False))
