"""Frozen larger Qwen2.5 MLX8 task collection over the ordinary native CLI."""
import argparse
import json
import os
from pathlib import Path

import metal_cli
import metal_tasks as shared

ROOT = shared.ROOT
SOURCE = ROOT / 'models/MLX-Qwen2.5-1.5B-Instruct-8bit-c7e1ff2'
RAW_SOURCE = ROOT/'artifacts/qwen15-quality-tasks-2026-11-06/raw-model'
RAW_ASSETS = {'config.json', 'model.safetensors', 'model.safetensors.index.json'}
CAPACITY = 562
CHUNK = 32
EXTRA = ('--metal-attention', 'reference', '--no-eos', '--stop-token', '151643', '--stop-token', '151645')
IDENTITY = dict(backend='metal', model='qwen2.5-1.5b-instruct-mlx8-c7e1ff2',
                projection_mode='MlxAffine', normalization_mode='Compensated',
                readback_mode='FullLogits', attention_mode='Reference')
EXECUTION = dict(weight_dtype='MLX affine + FP16 exclusions', kv_dtype='FP16',
                 projection_mode='MlxAffine', normalization_mode='Compensated',
                 readback_mode='FullLogits', prefill_chunk_size=CHUNK,
                 quantizer='MLX affine (per-tensor bits/groups)')
MODEL_PINS = {
    'model.safetensors': '8b76d08a5a70261b9ba4da5c06ac28bf09ee848e2322bae901e84764b7bba8e8',
    'config.json': 'a939169f5c833fd4c1d872dc07f521c6e5098075ac53bf007b8595343e980937',
    'tokenizer.json': 'a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf',
    'tokenizer_config.json': 'f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7',
    'model.safetensors.index.json': 'b885e07bdf11ca668b338f4bceed9ac0000dfcab56abe4b92670b748291d3643',
    'special_tokens_map.json': '76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd',
}
BASELINE = ROOT/'benchmarks/results/2026-10-14-larger-cpu-task-reference-full-v1.json'
WINDOW = ROOT/'benchmarks/results/2026-10-14-backend-task-larger-cpu-task-reference-full-v1-window.json'
GRADE = ROOT/'benchmarks/results/2026-10-14-backend-task-larger-cpu-task-reference-full-v1-probe.json'
WINDOW_SHA = '1976c32a4fbe536f0827ecc7717b31b2791478631e5fe8363f6538d92ddfc87b'
GRADE_SHA = 'dc9cbf20e8985a91ecf9091dc79c9835382d3259c3ddb3d620dad803550c2f93'
digest = shared.metal_engine.digest


def ready(event, _model):
    if (event.get('event') != 'ready' or event.get('backend') != 'metal'
            or event.get('cpu_projection') != 'none' or event.get('context_capacity') != CAPACITY
            or event.get('execution') != EXECUTION):
        raise ValueError('Actual larger MLX8 CLI execution differs')
    memory = metal_cli.memory(event)
    if memory['weights_bytes'] != 1640334262 or memory['tokenizer_application_allowance_bytes'] != 128 * 1024**2:
        raise ValueError('Actual larger MLX8 resident weights differ')


def reference(cases):
    if digest(WINDOW) != WINDOW_SHA or digest(GRADE) != GRADE_SHA:
        raise ValueError('Larger original grading identity differs')
    window = json.loads(WINDOW.read_text())
    grade = json.loads(GRADE.read_text())
    baseline = json.loads(BASELINE.read_text())
    if (not window['completed'] or not window['quality_completed'] or not window['restored_stopped']
            or not grade['completed'] or not grade['quality_completed']
            or window['probe_sha256'] != GRADE_SHA or window['collection_sha256'] != digest(BASELINE)
            or grade['collection_sha256'] != digest(BASELINE)
            or grade['native_raw_sha256'] != digest(BASELINE.with_suffix('.raw.jsonl'))
            or grade['task_payload_sha256'] != shared.tasks.TASK_SHA or grade['unresolved_case_ids']
            or grade['aggregate']['cases'] != 100 or grade['aggregate']['passed'] != 11
            or baseline['task_ids'] != [c['id'] for c in cases]):
        raise ValueError('Larger original grade or case binding differs')
    return grade


def command(binary):
    return [str(binary), '--model', str(RAW_SOURCE), '--backend', 'metal',
            '--metal-projection', 'mlx-affine', '--interactive', '--json',
            '--context-capacity', str(CAPACITY), '--prefill-chunk', str(CHUNK), *EXTRA]


def absent(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return
    raise ValueError('Owned CLI process group remains live')


def files(binary):
    manifest = shared.controlled.prepare.CONTRACT/'manifest.json'
    contract = json.loads(manifest.read_text())
    tasks = ROOT/contract['quality']['tasks_file']
    modules = (shared, metal_cli, shared.tasks, shared.reference, shared.metal_engine, shared.native_usage)
    paths = {binary, Path(__file__), ROOT/'Cargo.toml', ROOT/'Cargo.lock', manifest, tasks,
             BASELINE, BASELINE.with_suffix('.raw.jsonl'), WINDOW, GRADE}
    paths.update(Path(module.__file__) for module in modules)
    paths.update((ROOT/'src').rglob('*.rs'))
    paths.update((ROOT/'src').rglob('*.metal'))
    paths.update(SOURCE/name for name in MODEL_PINS)
    paths.update(RAW_SOURCE/name for name in RAW_ASSETS)
    return {str(path.resolve()): digest(path) for path in sorted(paths)}


def collect(binary, output, check):
    cases = shared.tasks.load_cases()
    reference(cases)
    pins = files(binary)
    if {p.name for p in SOURCE.iterdir() if p.name != 'README.md'} != set(MODEL_PINS):
        raise ValueError('Unexpected larger MLX8 asset inventory')
    if {p.name for p in RAW_SOURCE.iterdir()} != RAW_ASSETS:
        raise ValueError('The native task worker must have only raw numerical assets')
    if any(pins[str(RAW_SOURCE/name)] != MODEL_PINS[name] for name in RAW_ASSETS):
        raise ValueError('Raw-only numerical asset clone differs')
    if any(pins[str(SOURCE/name)] != expected for name,expected in MODEL_PINS.items()):
        raise ValueError('Pinned larger MLX8 artifact differs')
    record = dict(format='toklet-qwen15-mlx8-task-collection-v1', passed=False, check_only=check,
                  identity=IDENTITY, identities=pins, task_ids=[c['id'] for c in cases],
                  task_payload_sha256=shared.tasks.TASK_SHA)
    with (output.with_suffix('.raw.jsonl').open('x') as raw,
          output.with_suffix('.stderr').open('xb') as stderr,
          output.open('x') as report):
        try:
            run = shared.child(binary, RAW_SOURCE, IDENTITY['model'], cases, raw, stderr, check,
                               admission_check=ready, projection_argument='mlx-affine', extra_args=EXTRA,
                               capacity=CAPACITY, chunk=CHUNK, external_tokenizer=SOURCE/'tokenizer.json')
            record['run'] = run
            if not run['passed'] or not run['model_exited'] or run['command'] != command(binary):
                raise ValueError('Larger MLX8 native collection failed')
            absent(run['worker_pid'])
            if files(binary) != pins:
                raise ValueError('Larger MLX8 task identities changed')
            record['passed'] = True
        finally:
            json.dump(record, report, indent=2, allow_nan=False)
            report.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    collect(args.binary.resolve(), args.output.resolve(), args.check)
