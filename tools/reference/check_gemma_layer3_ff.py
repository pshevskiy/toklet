"""Compare the bound layer-3 feed-forward chain at position 48."""
import io
import json
import math
from pathlib import Path
import sys

import check_gemma_first_staged as prior

FIELDS = [
    ('post_attention_residual', 1536), ('pre_feedforward_layernorm', 1536),
    ('mlp.gate_proj', 6144), ('ff_product', 6144), ('mlp.down_proj', 1536),
    ('post_feedforward_layernorm', 1536), ('post_ff_residual', 1536),
    ('per_layer_input_gate', 256),
]


def run(folder):
    import numpy as np

    root = prior.prior.ROOT
    source = root / 'artifacts/gemma-first-staged-2026-11-06'
    read = prior.prior.prior.bounded.read
    proof = json.loads((root / 'benchmarks/results/2026-11-06-gemma-first-staged-v1.json').read_text())
    if not proof['passed'] or not proof['archived_endpoints_exact']:
        raise ValueError('Authenticated position-48 prefix evidence required')

    def archived(name, limit):
        path = source / name
        raw = read(path, limit)
        if prior.sha(raw) != proof['receipts_sha256'][str(path.relative_to(root))]:
            raise ValueError('Position-48 witness digest differs')
        return raw

    earlier = prior.capture(archived('gpu.stdout', 65536), archived('arrays.bin', 256 * 1024))
    actual = prior.capture_records(read(folder / 'gpu.stdout', 65536),
                                   read(folder / 'arrays.bin', 256 * 1024), FIELDS, 'LAYER3_FF')
    if actual['per_layer_input_gate'] != earlier['layer.3.per_layer_input_gate']:
        raise ValueError('Native feed-forward replay did not join the archived gate')
    with np.load(io.BytesIO(archived('staged-complete.npz', 2 * 1024**2)), allow_pickle=False) as oracle:
        def stage(name):
            return oracle['layer.3.' + name]

        residual = stage('input') + stage('post_attention_layernorm')
        factor = math.sqrt(2 / math.pi)
        gelu = np.array([.5 * v * (1 + math.tanh(factor * (v + .044715 * v**3)))
                         for v in map(float, stage('mlp.gate_proj'))], dtype=np.float32)
        reference = dict(post_attention_residual=residual, ff_product=gelu * stage('mlp.up_proj'),
                         post_ff_residual=residual + stage('post_feedforward_layernorm'))
        result = {}
        for name, raw in actual.items():
            seen = np.frombuffer(raw, dtype='<f4')
            expected = reference[name] if name in reference else stage(name)
            if seen.shape != expected.shape:
                raise ValueError('Stage width differs')
            result[name] = dict(bit_differences=int(np.count_nonzero(seen.view('<u4') != expected.view('<u4'))),
                                **prior.prior.prior.fractions(seen.tolist(), expected.tolist()))
    return dict(passed=True, first_bit_difference=next((k for k,v in result.items() if v['bit_differences']), None),
                stages=result, archived_gate_exact=True, whole_graph_qualified=False)


if __name__ == '__main__':
    print(json.dumps(run(Path(sys.argv[1])), indent=2, allow_nan=False))
