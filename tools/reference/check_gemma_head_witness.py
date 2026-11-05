"""Replay four rejected Gemma head rows from the actual normalized native input."""
import hashlib
import json
from pathlib import Path
import sys

import check_gemma_hybrid1025 as prior
import gemma4_e2b_context_math as contract

ROOT = prior.ROOT
IDS = [1822, 6212, 23958, 74252]
POSITION = 1024
HIDDEN = 1536


def native_input(transcript, body, archived_row):
    events = []
    for line in transcript.decode().splitlines():
        marker = 'GEMMA_HYBRID_'
        if marker in line:
            before, payload = line.split(marker, 1)
            assert not before or before == (
                'test model::gemma_packed::tests::hybrid::real::wrap::'
                'diagnostic1025_same_input_final_head ... '
            )
            label, raw = payload.split(' ', 1)
            events.append((label, json.loads(raw)))
    assert [name for name, _ in events] == ['REAL_FINAL_HEAD_ROW', 'ARRAY_ATTEMPT', 'ARRAY', 'CAPTURE']
    _, row = events[0]
    assert row['position'] == POSITION
    assert row['sha256'] == hashlib.sha256(archived_row).hexdigest()
    assert row['allocated_bytes'] == 271117548
    expected = dict(record=0, name='head.normalized', offset=0, bytes=HIDDEN*4,
                    before=POSITION, count=1, values=HIDDEN)
    digest = hashlib.sha256(body).hexdigest()
    assert len(body) == HIDDEN*4
    assert events[1][1] == expected
    assert events[2][1] == dict(expected, sha256=digest)
    assert events[3][1] == dict(records=1, bytes=HIDDEN*4, sha256=digest, whole_e2b_qualified=False)
    return digest


def original_row():
    proof = json.loads((ROOT/'benchmarks/results/2026-11-04-gemma-hybrid1025-v1.json').read_text())
    assert proof['state_passed'] and not proof['numerical_passed']
    captured = ROOT/'artifacts/gemma-hybrid1025-2026-11-04'
    read = prior.prior.bounded.read
    logs = read(captured/'native.stdout', 2*1024**2)
    raw = read(captured/'arrays.bin', 128*1024**2)
    for name, data in [('native.stdout', logs), ('arrays.bin', raw)]:
        key = str((captured/name).relative_to(ROOT))
        assert hashlib.sha256(data).hexdigest() == proof['receipts_sha256'][key]
    logits, _, _ = prior.native_capture(logs, raw)
    return logits[POSITION]


def compare(directory):
    import numpy as np

    archived = original_row()
    read = prior.prior.bounded.read
    body = read(directory/'arrays.bin', HIDDEN*4)
    digest = native_input(read(directory/'gpu.stdout', 65536), body, archived)
    hidden = np.frombuffer(body, dtype='<f4')
    assert hidden.shape == (HIDDEN,) and np.isfinite(hidden).all()
    source = prior.oracle.source_graph

    def guard():
        for name, expected in source.source.PINS.items():
            assert source.source.sha(source.source.MODEL/name) == expected

    guard()
    with (source.source.MODEL/'model.safetensors').open('rb') as stream:
        weights = source.weights(stream)
        rows = weights.gather('embed_tokens.weight', IDS)
        dots = [contract.product_sum(hidden, row) for row in rows]
    guard()
    actual = np.frombuffer(archived, dtype='<f4')[IDS]
    reports = {}
    for name, staged, dtype in [('math_fp64', False, np.float64), ('staged_fp32', True, np.float32)]:
        reference = contract.softcap(np.array(dots, dtype=dtype), staged)
        reports[name] = prior.prior.fractions(actual, reference)
    return dict(position=POSITION, ids=IDS, normalized_row_sha256=digest,
                archived_native_row_exact=True, local_head=reports,
                local_head_passed=all(row['failed_values'] == 0 for row in reports.values()),
                earlier_full_gate_remains_open=True)


if __name__ == '__main__':
    print(json.dumps(compare(Path(sys.argv[1])), indent=2, allow_nan=False))
