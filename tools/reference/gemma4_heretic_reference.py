"""Explicit short mixed-source oracle using the unchanged independent graph.

Retain the last prompt position of every stage in both mathematical FP64 and
staged FP32 contracts. This producer never imports or executes the Rust engine,
scores no held-out quality and grants no model, context or app qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import affine4_tiles
import bf16_source
import gemma4_affine_reference as runtime_owner
import gemma4_e2b_context_math as context_math
import gemma4_heretic_weights as adapter
import gemma4_text_math as graph

ROOT = Path(__file__).resolve().parents[2]
TOKENS = [2, 105, 2364, 107, 1567, 506, 5279, 529, 7001, 236761, 106, 107, 105, 4368, 107]
MAX_CAPTURE = 64 * 1024**2
CONTEXT_TOKENS_SHA256 = 'ade0775a55a1248340329388c5ba6fced9c12b42ef9974c6d10953fc5cca7f15'
PHASES = {'compat32': (32, [0, 15, 30, 31]), 'pilot64': (64, [0, 31, 63]),
          'full512': (512, [0, 31, 255, 511])}


def extras():
    return [Path(__file__).resolve(), Path(adapter.__file__).resolve(),
            Path(bf16_source.__file__).resolve(), Path(context_math.__file__).resolve(),
            Path(graph.__file__).resolve()]


def position_axis(name):
    return 2 if name.endswith(('rotary', 'context')) else 1


def inputs(phase):
    if phase == 'public15':
        return TOKENS.copy(), [14]
    if phase not in PHASES:
        raise ValueError('Named Heretic reference phase required')
    raw = runtime_owner.runtime.captured(ROOT / 'benchmarks/gemma-e2b-context-v1/tokens.u32le',
                                         2048, CONTEXT_TOKENS_SHA256)
    corpus = runtime_owner.runtime.captured(ROOT / 'benchmarks/gemma-quality-v1/held_out_loss.u32le',
                                            40004, '877cbe18574b47e4b6a8d1b58fb79621c37e0531bd1f9504e50ce63d186f8c46')
    if raw != corpus[:2048]:
        raise ValueError('Frozen context/quality prefix differs')
    import struct
    count, positions = PHASES[phase]
    return list(struct.unpack('<512I', raw))[:count], positions.copy()


def shapes(tokens=15, selected=1):
    if type(tokens) is not int or not 1 <= tokens <= 512 or type(selected) is not int or not 1 <= selected <= 8:
        raise ValueError('Bounded reference row counts required')
    value = graph.stage_shapes(graph.E2B, 1)
    for name, shape in value.items():
        dimensions = list(shape)
        dimensions[position_axis(name)] = tokens
        value[name] = tuple(dimensions)
    for name in ['head', 'logits']:
        value[name] = (1, selected, graph.E2B.vocab)
    return value


def complete_source_hash(owner, deadline):
    digest = hashlib.sha256()
    for offset in range(0, owner.size, bf16_source.HASH_BLOCK):
        digest.update(owner.read(offset, min(bf16_source.HASH_BLOCK, owner.size - offset), deadline=deadline))
    if digest.hexdigest() != adapter.FILE_SHA256:
        raise ValueError('Heretic source endpoint bytes differ')
    return digest.hexdigest()


def collect(output, directory, admission, endpoint, phase='public15'):
    import numpy as np
    if np.__version__ != '2.3.2':
        raise ValueError('Pinned NumPy2.3.2 required')
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Fresh repository artifact output required')
    output.mkdir()
    tokens, positions = inputs(phase)
    expected = shapes(len(tokens), len(positions))
    retained = set(expected) if phase == 'public15' else {'logits'}
    planned = sum(math.prod(v) // v[position_axis(n)] * len(positions) * 12
                  for n, v in expected.items() if n in retained)
    if planned > MAX_CAPTURE or len(expected) != 713:
        raise ValueError('Short reference capture plan differs')
    record = dict(format='toklet-gemma-heretic-math-v1', completed=False,
                  repository=adapter.REPOSITORY, revision=adapter.REVISION,
                  phase=phase, tokens=tokens, selected_positions=positions,
                  gate=dict(atol=1e-5, rtol=1e-4), planned_capture_bytes=planned,
                  runtime_admission=admission, source=None, arrays=[],
                  contracts={}, primary_error=None, endpoint_errors=[],
                  engine_executed=False, held_out_quality_scored=False,
                  full_model_fidelity_qualified=False, managed_adoption_qualified=False,
                  process_limit_bytes=6 * 1024**3, memory_ledger_mib=context_math.LEDGER_MIB)
    owner = None
    try:
        if hashlib.sha256(Path(graph.__file__).read_bytes()).hexdigest() != context_math.ORIGINAL_SOURCE_SHA256:
            raise ValueError('Frozen mathematical implementation changed')
        if sum(context_math.LEDGER_MIB.values()) * 1024**2 + MAX_CAPTURE > record['process_limit_bytes']:
            raise ValueError('Reference memory ledger exceeds6GiB')
        config = runtime_owner.runtime.captured(directory / 'config.json', 6035, adapter.CONFIG_SHA256)
        deadline = time.monotonic() + (3600 if phase == 'full512' else 1800)
        with bf16_source.Source(directory / 'model.safetensors', adapter.FILE_BYTES,
                                adapter.FILE_SHA256, deadline=deadline) as owner:
            weights = adapter.weights(owner, config, deadline)
            with (output / 'states.bin').open('xb') as binary:
                for contract, staged in [('math_fp64', False), ('staged_fp32', True)]:
                    start = time.monotonic()
                    seen = set()
                    print(json.dumps(dict(event='contract-start', contract=contract)), flush=True)

                    def capture(name, value):
                        if (name in seen or name not in expected or tuple(value.shape) != expected[name]
                                or value.dtype != np.dtype('<f4' if staged else '<f8')
                                or not np.isfinite(value).all()):
                            raise ValueError('Reference stage inventory/values differ: ' + name)
                        seen.add(name)
                        if name.startswith('layer.') and name.endswith('.output'):
                            print(json.dumps(dict(event='layer', contract=contract,
                                                  layer=int(name.split('.')[1]),
                                                  elapsed_seconds=time.monotonic() - start)), flush=True)
                        if name not in retained:
                            return
                        selected_rows = list(range(len(positions))) if name in ['head', 'logits'] else positions
                        chosen = np.take(value, selected_rows, axis=position_axis(name))
                        raw = chosen.tobytes(order='C')
                        if binary.tell() + len(raw) > planned:
                            raise ValueError('Reference stage capture exceeds plan')
                        item = dict(contract=contract, name=name, positions=positions,
                                    shape=list(chosen.shape), dtype=chosen.dtype.str,
                                    offset=binary.tell(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                        if binary.write(raw) != len(raw):
                            raise OSError('Short reference capture write')
                        record['arrays'].append(item)
                        if name == 'logits':
                            extension = 'f32le' if staged else 'f64le'
                            path = output / (contract + '.logits.' + extension)
                            with path.open('xb') as file:
                                if file.write(raw) != len(raw):
                                    raise OSError('Short reference logit write')
                            item['logits_file'] = path.name
                            record['contracts'][contract] = dict(greedy_ids=[int(np.argmax(row)) for row in chosen[0]],
                                                               logits_sha256=item['sha256'])

                    context_math.evaluate(weights, tokens, staged,
                                          selected_positions=positions, sink=capture)
                    if seen != set(expected):
                        raise ValueError('Reference stages incomplete')
                    record['contracts'][contract]['collection_seconds'] = time.monotonic() - start
                if binary.tell() != planned:
                    raise ValueError('Reference body length differs')
            record['source_endpoint_sha256'] = complete_source_hash(owner, deadline)
        with (output / 'states.bin').open('rb') as file:
            record['states_sha256'] = hashlib.file_digest(file, 'sha256').hexdigest()
        record['completed'] = True
    except BaseException as error:
        record['primary_error'] = type(error).__name__ + ': ' + str(error)[:2048]
    finally:
        if owner is not None:
            record['source'] = owner.audit()
        try:
            record['runtime_endpoint'] = endpoint()
        except BaseException as error:
            record['endpoint_errors'].append(type(error).__name__ + ': ' + str(error)[:2048])
        record['completed'] = record['completed'] and record['primary_error'] is None and not record['endpoint_errors']
        (output / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-directory', type=Path, required=True)
    parser.add_argument('--phase', choices=['public15', *PHASES], default='public15')
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--native-manifest', type=Path, required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    args = parser.parse_args()
    admission, guard = runtime_owner.runtime_context(
        args.runtime_manifest, args.expected_runtime_sha256,
        args.native_manifest, args.expected_native_sha256, extra_sources=extras())
    result = collect(args.output, args.model_directory, admission, guard, args.phase)
    print(json.dumps(dict(completed=result['completed'], error=result['primary_error'])))
    raise SystemExit(0 if result['completed'] else 1)
