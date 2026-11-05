"""Locate the first observed staged Gemma4 difference at position 48."""
import hashlib
import json
from pathlib import Path
import sys

import check_gemma_hybrid1025 as prior

POSITION = 48
FIELDS = [
    ('layer.2.output', 1536),
    ('layer.3.input_layernorm', 1536), ('layer.3.self_attn.q_norm', 2048),
    ('layer.3.q_rotary', 2048), ('layer.3.context', 2048),
    ('layer.3.self_attn.o_proj', 1536), ('layer.3.mlp.up_proj', 6144),
    ('layer.3.per_layer_input_gate', 256), ('layer.3.per_layer_projection', 1536),
    ('layer.3.post_per_layer_input_norm', 1536), ('layer.3.ple_input', 256),
    ('layer.3.output', 1536), ('layer.4.input_layernorm', 1536),
    ('layer.4.self_attn.v_proj', 512), ('layer.4.self_attn.k_norm', 512),
]
KV = [('k', 'k_rotary'), ('v', 'self_attn.v_norm')]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def capture(log, body):
    fields = FIELDS + [(f'layer.{i}.{name}', 512 if i == 4 else 256)
                       for i in range(5) for _, name in KV]
    return capture_records(log, body, fields, 'FIRST_STAGED')


def capture_records(log, body, fields, event):
    events = []
    for line in log.decode().splitlines():
        if 'GEMMA_HYBRID_' in line:
            name, value = line.split('GEMMA_HYBRID_', 1)[1].split(' ', 1)
            events.append((name, json.loads(value)))
    expected = [('REAL_' + event, dict(position=POSITION, prefix=64, chunk=32, context=1025))]
    rows, offset = {}, 0
    for index, (name, count) in enumerate(fields):
        raw = body[offset:offset + 4 * count]
        if len(raw) != 4 * count:
            raise ValueError('Truncated staged prefix record')
        row = dict(record=index, name=name, offset=offset, bytes=len(raw),
                   before=POSITION, count=1, values=count)
        expected.extend([('ARRAY_ATTEMPT', row), ('ARRAY', dict(row, sha256=sha(raw)))])
        rows[name] = raw
        offset += len(raw)
    expected.extend([
        ('REAL_' + event + '_COMPLETE', dict(committed=64, faults_preserved_prefix=True)),
        ('CAPTURE', dict(records=len(fields), bytes=offset, sha256=sha(body),
                         whole_e2b_qualified=False)),
    ])
    if events != expected or offset != len(body):
        raise ValueError('Staged prefix framing differs')
    return rows


def run(folder):
    import io
    import numpy as np

    root = prior.ROOT
    read = prior.prior.bounded.read
    proof = json.loads((root / 'benchmarks/results/2026-11-04-gemma-hybrid1025-v1.json').read_text())
    if not proof['state_passed'] or proof['numerical_passed']:
        raise ValueError('Historical state pass and numerical rejection required')

    def archived(path, limit):
        raw = read(path, limit)
        if sha(raw) != proof['receipts_sha256'][str(path.relative_to(root))]:
            raise ValueError('Historical 1,025 capture differs')
        return raw

    native = root / 'artifacts/gemma-hybrid1025-2026-11-04'
    _, kv, _ = prior.native_capture(archived(native / 'native.stdout', 2 * 1024**2),
                                    archived(native / 'arrays.bin', 128 * 1024**2))
    reference = root / 'artifacts/gemma-e2b-wrap-reference-2026-11-04/full1025/reference'
    archived(reference / 'result.json', 2 * 1024**2)
    archived(reference / 'rows.bin', 160 * 1024**2)
    retained = prior.references(reference)
    rows = capture(read(folder / 'gpu.stdout', 65536), read(folder / 'arrays.bin', 256 * 1024))
    details, earliest = {}, {}
    reference_report = json.loads(read(folder / 'reference-complete.json', 65536))
    reference_rows = read(folder / 'staged-complete.npz', 2 * 1024**2)
    if (not reference_report['passed'] or reference_report['prefix'] != 64
            or reference_report['position'] != POSITION or sha(reference_rows) != reference_report['rows_sha256']):
        raise ValueError('Independent prefix export differs')
    with np.load(io.BytesIO(reference_rows), allow_pickle=False) as oracle:
        if oracle.files != reference_report['fields']:
            raise ValueError('Independent prefix inventory differs')
        for owner in range(15):
            stride = 512 if owner % 5 == 4 else 256
            for kind, stage in KV:
                name = f'layer.{owner}.{stage}'
                raw = kv[512, owner, kind]
                independent = retained['staged_fp32', kind, owner][:len(raw)]
                span = slice(POSITION * stride * 4, (POSITION + 1) * stride * 4)
                if owner < 5 and (rows[name] != raw[span] or oracle[name].astype('<f4').tobytes() != independent[span]):
                    raise ValueError('Prefix capture does not join both archived K/V endpoints')
                left = np.frombuffer(raw, dtype='<u4').reshape(-1, stride)
                right = np.frombuffer(independent, dtype='<u4').reshape(-1, stride)
                differing = np.flatnonzero(np.any(left != right, axis=1))
                earliest[name] = int(differing[0]) if differing.size else None
        if min(p for p in earliest.values() if p is not None) != POSITION:
            raise ValueError('Archived earliest observed position differs')
        for name, raw in rows.items():
            actual = np.frombuffer(raw, dtype='<f4')
            expected = oracle[name]
            if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError('Nonfinite or mismatched stage')
            details[name] = dict(bit_differences=int(np.count_nonzero(actual.view('<u4') != expected.view('<u4'))),
                                 **prior.prior.fractions(actual.tolist(), expected.tolist()))
    return dict(passed=True, position=POSITION, archived_endpoints_exact=True,
                earliest_archived_kv_differences=earliest, stages=details, whole_graph_qualified=False)


if __name__ == '__main__':
    print(json.dumps(run(Path(sys.argv[1])), indent=2, allow_nan=False))
