"""Closed independent coupled Qwen reference at frozen workloads; not a target execution.

The scalar graph predates Rust composition. Streaming and bounded row projections
change storage/access only; fixture regeneration must retain every original byte.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import affine4_tiles
import bf16_source
import check_qwen35_payload as pin
import gemma4_affine_reference as runtime_owner
import qwen35_cpu_graph as graph
import qwen35_real_weights as weights

ROOT = Path(__file__).resolve().parents[2]
MAX_CAPTURE = 256 * 1024**2
ARRAYS = 1051
VALUES = 16067328
LEDGER = dict(python_state_and_temporaries=2*1024**3, interpreter_and_runtime=512*1024**2,
    small_weight_vectors=64*1024**2, tile_and_source_buffers=128*1024**2, capture=MAX_CAPTURE)


def plan(phase):
    if phase == 'one': tokens, offset = [1], 0
    elif phase == 'context17': tokens, offset = list(graph.TOKENS), 37
    else: raise ValueError('Named frozen Qwen reference phase required')
    # Last-token full KV includes every committed input; attention scores and
    # probabilities also grow with context. Other captured stage shapes stay fixed.
    extra_per_token = 8*2*4*256 + 8*2*16
    values = VALUES + (len(tokens)-1)*extra_per_token
    return tokens, offset, values


def extras():
    return [Path(v.__file__).resolve() for v in [weights, graph, bf16_source, pin]] + [Path(__file__).resolve()]


def source_endpoint(owner, deadline):
    digest = hashlib.sha256()
    for offset in range(0, owner.size, bf16_source.HASH_BLOCK):
        digest.update(owner.read(offset, min(bf16_source.HASH_BLOCK, owner.size-offset), deadline=deadline))
    weights.need(digest.hexdigest() == pin.FILE_SHA, 'Whole source endpoint differs')
    return digest.hexdigest()


def collect(output, directory, admission, endpoint, phase='one'):
    import numpy as np
    weights.need(np.__version__ == '2.3.2', 'Pinned NumPy2.3.2 required')
    tokens, offset, planned_values = plan(phase)
    output = output.resolve()
    weights.need(output.is_relative_to(ROOT/'artifacts'), 'Fresh local artifact output required')
    output.mkdir()
    record = dict(format='toklet-qwen35-real-reference-v1', completed=False,
        repository=weights.REPOSITORY, revision=weights.REVISION, source_sha256=pin.FILE_SHA,
        phase=phase, tokens=tokens, position_offset=offset, geometry=weights.GEOMETRY, gate=graph.GATE,
        arrays=[], planned_rows_bytes=planned_values*12, ledger=LEDGER, runtime_admission=admission,
        source=None, primary_error=None, endpoint_errors=[], engine_executed=False,
        contextual_numerical_qualified=False, quality_qualified=False, managed_adoption_qualified=False)
    owner = None
    try:
        weights.need(sum(LEDGER.values()) <= 6*1024**3 and planned_values*12 <= MAX_CAPTURE, 'Resource plan exceeds fixed bounds')
        config = runtime_owner.runtime.captured(directory/'config.json', 3024, pin.CONFIG_SHA)
        deadline = time.monotonic() + (3300 if phase == 'context17' else 1800)
        with bf16_source.Source(directory/'model.safetensors', pin.FILE_BYTES, pin.FILE_SHA, deadline=deadline) as owner:
            matrices, vectors = weights.weights(owner, config, deadline)
            with (output/'rows.bin').open('xb') as binary:
                for profile, staged in [('ideal_fp64', False), ('staged_fp32', True)]:
                    seen = set()
                    count = 0
                    started = time.monotonic()
                    print(json.dumps(dict(event='profile-start', profile=profile)), flush=True)

                    def save(row):
                        nonlocal count
                        bf16_source.check_deadline(deadline, None)
                        if row['stage'] == 'layer_output' and row['layer'] == 31:
                            print(json.dumps(dict(event='token-complete', profile=profile, token=row['token'],
                                elapsed_seconds=time.monotonic()-started)), flush=True)
                        if row['token'] != len(tokens)-1: return
                        values = row.pop('values')
                        key = (row['token'], row['layer'], row['stage'])
                        weights.need(key not in seen, 'Duplicate observation')
                        seen.add(key)
                        data = np.asarray(values, dtype='<f4' if staged else '<f8')
                        weights.need(data.ndim == 1 and np.isfinite(data).all(), 'Finite complete reference arrays required')
                        raw = data.tobytes(order='C')
                        weights.need(binary.tell()+len(raw) <= planned_values*12 <= MAX_CAPTURE, 'Capture bound exceeded')
                        record['arrays'].append(dict(profile=profile, **row, count=len(values), offset=binary.tell(),
                            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
                        weights.need(binary.write(raw) == len(raw), 'Short reference write')
                        count += len(values)
                        if row['stage'] == 'layer_output':
                            print(json.dumps(dict(event='layer', profile=profile, layer=row['layer'],
                                elapsed_seconds=time.monotonic()-started)), flush=True)

                    trace = graph.forward(matrices, vectors, tokens, offset, staged,
                        geometry=weights.GEOMETRY, math_type=weights.Math, sink=save)
                    weights.need(trace == [] and len(seen) == ARRAYS and count == planned_values,
                        'Complete streaming observation inventory differs')
                weights.need(binary.tell() == planned_values*12, 'Final capture size differs')
            record['source_endpoint_sha256'] = source_endpoint(owner, deadline)
        record['source'] = owner.audit()
        record['rows_sha256'] = runtime_owner.sha(output/'rows.bin')
        record['completed'] = True
    except BaseException as error:
        record['primary_error'] = dict(type=type(error).__name__, message=str(error)[:4096])
    finally:
        if owner is not None:
            if not owner.closed:
                try: owner.close()
                except BaseException as error: record['endpoint_errors'].append(str(error)[:4096])
            record['source'] = owner.audit()
        try: record['runtime_endpoint'] = endpoint()
        except BaseException as error: record['endpoint_errors'].append(str(error)[:4096])
        if record['endpoint_errors']: record['completed'] = False
        (output/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--native-manifest', type=Path, required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    parser.add_argument('--phase', choices=['one', 'context17'], default='one')
    args = parser.parse_args()
    admission, endpoint = runtime_owner.runtime_context(args.runtime_manifest,
        args.expected_runtime_sha256, args.native_manifest, args.expected_native_sha256,
        extra_sources=extras())
    result = collect(args.output, args.model_directory, admission, endpoint, args.phase)
    print(json.dumps({k: result[k] for k in ['completed', 'primary_error', 'endpoint_errors']}))
    raise SystemExit(0 if result['completed'] else 1)
