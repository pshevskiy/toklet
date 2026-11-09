"""Export frozen CPU-only MiniCPM5 Half-storage profiles before native outcomes."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import sys
import sysconfig
import time

import minicpm5 as model

ROOT = Path(__file__).resolve().parents[2]
STAGES = ['input_norm', 'q_projection', 'k_projection', 'v_projection', 'q_rotary',
          'k_rotary', 'attention_scores', 'attention_probabilities', 'attention_context',
          'attention_projection', 'attention_residual', 'post_attention_norm',
          'gate_projection', 'up_projection', 'gate_silu', 'gated', 'down_projection', 'output']


def admission():
    ledger = dict(pread_and_hash_bytes=16 * 1024 ** 2,
                  affine_decode_tile_bytes=512 * 1024 ** 2,
                  linear_widening_bytes=256 * 1024 ** 2,
                  blas_workspace_bytes=256 * 1024 ** 2,
                  current_graph_bytes=384 * 1024 ** 2,
                  retained_dual_traces_bytes=256 * 1024 ** 2,
                  numpy_and_framework_bytes=512 * 1024 ** 2,
                  metadata_publication_and_margin_bytes=256 * 1024 ** 2)
    ledger['admitted_bytes'] = sum(ledger.values())
    model.need(ledger['admitted_bytes'] <= 6 * 1024 ** 3, 'Reference admission exceeds6GiB')
    return ledger


def expected_arrays():
    expected = {}
    for case, spec in model.CASES.items():
        n = len(spec['tokens'])
        stages = {'embedding': ([n, 2048], '<f2'), 'final_norm': ([n, 2048], '<f2'),
                  'logits': ([n, 130560], '<f4')}
        for layer in model.LAYERS:
            for stage in STAGES:
                shape, dtype = [n, 2048], '<f2'
                if stage in ['k_projection', 'v_projection']:
                    shape = [n, 256]
                elif stage == 'q_rotary':
                    shape = [n, 16, 128]
                elif stage == 'k_rotary':
                    shape = [n, 2, 128]
                elif stage in ['attention_scores', 'attention_probabilities']:
                    shape, dtype = [16, n, n], '<f4'
                elif stage in ['gate_projection', 'up_projection', 'gate_silu', 'gated']:
                    shape = [n, 6144]
                stages[f'layers.{layer}.{stage}'] = (shape, dtype)
        expected.update({f'cases.{case}.{stage}': item for stage, item in stages.items()})
    expected.update({f'greedy.step_{i}.logits': ([130560], '<f4') for i in range(4)})
    expected.update({f'scoring.prefix_{i}.logits': ([130560], '<f4') for i in range(1, 4)})
    return expected


def package_pins():
    model.need(importlib.metadata.version('numpy') == '2.3.2', 'Pinned NumPy2.3.2 required')
    site = Path(sysconfig.get_path('purelib')).resolve()
    paths = [p for p in (site / 'numpy').rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.suffix in ['.py', '.so', '.dylib']]
    for folder in site.glob('numpy*.libs'):
        paths += [p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.so', '.dylib']]
    model.need(paths, 'NumPy package files absent')
    return {str(p): model.sha(p) for p in sorted(paths)}


def producer_pins():
    names = ['tools/reference/minicpm5.py', 'tools/reference/minicpm5_reference.py',
             'tools/reference/test_minicpm5.py', 'tools/reference/qwen_fp16.py',
             'tools/experiments/native_usage.py']
    return {name: model.sha(ROOT / name) for name in names}


def export(source, destination, contract):
    import numpy as np
    destination = Path(destination)
    destination.mkdir()  # Never replace old raw arrays, manifests or failures.
    raw_contract = Path(contract).read_bytes()
    frozen = json.loads(raw_contract)
    model.need(frozen['cases'] == model.CASES and frozen['layers'] == model.LAYERS
               and frozen['model_sha256'] == model.PINS['model.safetensors']
               and frozen['policy']['atol_floor'] == model.POLICY['atol_floor']
               and frozen['policy']['envelope_multiplier'] == model.POLICY['envelope_multiplier']
               and frozen['policy']['rtol'] == model.POLICY['rtol'], 'Prospective reference contract differs')
    ledger, producers, packages = admission(), producer_pins(), package_pins()
    arrays, envelopes, selections = {}, {}, []
    inventory = expected_arrays()
    model.need(sum(math.prod(shape) * np.dtype(dtype).itemsize
                   for shape, dtype in inventory.values()) <= 128 * 1024 ** 2,
               'Independent raw output exceeds bound')
    outputs = {name: (destination / (name + '.bin')).open('xb') for name in ['fp32', 'fp64']}
    started = time.monotonic_ns()
    completed = []
    def save(name, pair, group):
        shape, dtype = inventory[name]
        for profile, value in zip(['fp32', 'fp64'], pair, strict=True):
            value = np.asarray(value)
            model.need(list(value.shape) == shape and value.dtype == np.dtype(dtype)
                       and np.isfinite(value).all(), 'Reference shape/dtype/finiteness differs: ' + name)
            raw = value.astype(dtype, copy=False).tobytes()
            stream = outputs[profile]
            arrays[profile + '.' + name] = dict(file=profile + '.bin', offset=stream.tell(),
                bytes=len(raw), shape=shape, dtype=dtype, sha256=hashlib.sha256(raw).hexdigest(), gate_group=group)
            stream.write(raw)
        difference = float(np.max(np.abs(pair[0].astype(np.float64) - pair[1].astype(np.float64))))
        if name.startswith('cases.'):
            envelopes[group] = max(envelopes.get(group, 0.0), difference)
    try:
        with model.Source(source) as weights:
            for name, spec in model.CASES.items():
                pair = [model.forward(spec['tokens'], spec['position_offset'], weights,
                                      model.GEOMETRY, precision, model.LAYERS)
                        for precision in [np.float32, np.float64]]
                model.need(set(pair[0]) == set(pair[1]), 'Independent profile coverage differs')
                for stage in pair[0]:
                    save('cases.' + name + '.' + stage, [p[stage] for p in pair], stage)
                completed.append(name)
                (destination / 'progress.json').write_text(json.dumps(dict(completed_cases=completed,
                    profiles=['fp32', 'fp64'],native_outcomes_seen=False), indent=2) + '\n')
            tokens = model.CASES['short']['tokens'].copy()
            for step in range(4):
                pair = [model.forward(tokens, 0, weights, model.GEOMETRY, p, [])['logits'][-1]
                        for p in [np.float32, np.float64]]
                save(f'greedy.step_{step}.logits', pair, 'logits')
                selected = int(np.argmax(pair[1]))
                selections.append(dict(prefix=tokens.copy(), fp32_id=int(np.argmax(pair[0])), fp64_id=selected))
                tokens.append(selected)
            for count in range(1, 4):
                pair = [model.forward(model.CASES['short']['tokens'][:count], 0, weights,
                                      model.GEOMETRY, p, [])['logits'][-1]
                        for p in [np.float32, np.float64]]
                save(f'scoring.prefix_{count}.logits', pair, 'logits')
    finally:
        for stream in outputs.values():
            stream.close()
    model.need(set(arrays) == {p + '.' + n for p in ['fp32', 'fp64'] for n in inventory},
               'Incomplete independent array coverage')
    gates = {name: dict(atol=max(1e-4, 2 * value), rtol=1e-3,
                       pooled_cpu_profile_max_abs=value) for name, value in envelopes.items()}
    model.need(producers == producer_pins() and packages == package_pins(), 'Reference producer/package changed')
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory = dict(peak_rss_bytes=int(raw_rss if sys.platform == 'darwin' else raw_rss * 1024))
    if sys.platform == 'darwin':
        import importlib.util
        spec = importlib.util.spec_from_file_location('owned_native_usage', ROOT / 'tools/experiments/native_usage.py')
        usage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(usage)
        memory.update(usage.usage())
    model.need(max(memory.values()) <= 6 * 1024 ** 3, 'Observed reference memory exceeds6GiB')
    result = dict(format='toklet-minicpm5-independent-short-v1',
        model=dict(repository=model.REPOSITORY, revision=model.REVISION, files=model.PINS),
        geometry=model.GEOMETRY, cases=model.CASES, layers=model.LAYERS, policy=model.POLICY,
        contract_sha256=hashlib.sha256(raw_contract).hexdigest(), producer_pins=producers,
        package_pins=packages, python=sys.version, thread_environment={k: os.environ.get(k) for k in
            ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']},
        memory_admission=ledger, observed_memory=memory, arrays=arrays, gates=gates, greedy=selections,
        files={p + '.bin': dict(bytes=(destination / (p + '.bin')).stat().st_size,
                               sha256=model.sha(destination / (p + '.bin'))) for p in ['fp32', 'fp64']},
        elapsed_ns=time.monotonic_ns() - started, native_outcomes_seen=False,
        parent_lineage_authenticated=False, model_quality_qualified=False, model_adoption_qualified=False,
        gate_scope='Six authored cases only; teacher/scoring logits use the same frozen cases-only pooledlogit gate, never enlarge it')
    with (destination / 'manifest.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed=True, arrays=len(arrays), selected=selections,
                         elapsed_ns=result['elapsed_ns'], quality_qualified=False)), flush=True)


if __name__ == '__main__':
    model.need(len(sys.argv) == 4, 'Usage: minicpm5_reference.py MODEL EXCLUSIVE_OUTPUT CONTRACT')
    export(*sys.argv[1:])
