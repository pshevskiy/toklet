"""Closed independent9B one-input/context17 coupled reference; not a target execution.

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
import check_qwen35_9b_payload as pin
import check_qwen35_payload as arithmetic
import gemma4_affine_reference as runtime_owner
import qwen35_cpu_graph as graph
import qwen35_9b_weights as weights

ROOT = Path(__file__).resolve().parents[2]
MAX_CAPTURE = 256 * 1024**2
ARRAYS = 1051
VALUES = 16660224
LEDGER = dict(python_state_and_temporaries=2*1024**3, interpreter_and_runtime=512*1024**2,
    small_weight_vectors=64*1024**2, tile_and_source_buffers=128*1024**2, capture=MAX_CAPTURE)


def plan(phase):
    if phase == 'one': tokens, offset = [1], 0
    elif phase == 'context17': tokens, offset = list(graph.TOKENS), 37
    else: raise ValueError('Frozen9B one-input or context17 phase required')
    values = VALUES + (len(tokens)-1)*(8*2*4*256 + 8*2*16)
    return tokens, offset, values


def extras():
    return [Path(v.__file__).resolve() for v in [weights, weights.base, weights.large,
        graph, bf16_source, pin, arithmetic]] + [Path(__file__).resolve()]


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
    weights.need(output.parent.is_dir(), 'Existing parent for fresh exclusive artifact output required')
    output.mkdir()
    record = dict(format='toklet-qwen35-9b-real-reference-v1', completed=False,
        repository=weights.REPOSITORY, revision=weights.REVISION, source_sha256=pin.FILE_SHA,
        phase=phase, tokens=tokens, position_offset=offset, geometry=weights.GEOMETRY, gate=graph.GATE,
        arrays=[], planned_rows_bytes=planned_values*12, ledger=LEDGER, runtime_admission=admission,
        profile_ns={}, source_open_and_bind_ns=0, source_endpoint_ns=0, collection_ns=0,
        source=None, primary_error=None, endpoint_errors=[], engine_executed=False,
        contextual_numerical_qualified=False, quality_qualified=False, managed_adoption_qualified=False)
    owner = None
    collection_started = time.monotonic_ns()
    try:
        weights.need(sum(LEDGER.values()) <= 6*1024**3 and planned_values*12 <= MAX_CAPTURE, 'Resource plan exceeds fixed bounds')
        config = runtime_owner.runtime.captured(directory/'config.json', weights.CONFIG_BYTES, pin.CONFIG_SHA)
        deadline = time.monotonic() + 3600
        source_started = time.monotonic_ns()
        with bf16_source.Source(directory/'model.safetensors', pin.FILE_BYTES, pin.FILE_SHA, deadline=deadline) as owner:
            matrices, vectors = weights.weights(owner, config, deadline)
            record['source_open_and_bind_ns'] = time.monotonic_ns()-source_started
            with (output/'rows.bin').open('xb') as binary:
                for profile, staged in [('ideal_fp64', False), ('staged_fp32', True)]:
                    seen = set()
                    count = 0
                    started = time.monotonic()
                    profile_started = time.monotonic_ns()
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
                        geometry=weights.GEOMETRY, math_type=weights.math_type(matrices), sink=save)
                    weights.need(trace == [] and len(seen) == ARRAYS and count == planned_values,
                        'Complete streaming observation inventory differs')
                    record['profile_ns'][profile] = time.monotonic_ns()-profile_started
                weights.need(binary.tell() == planned_values*12, 'Final capture size differs')
            endpoint_started = time.monotonic_ns()
            record['source_endpoint_sha256'] = source_endpoint(owner, deadline)
            record['source_endpoint_ns'] = time.monotonic_ns()-endpoint_started
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
        record['collection_ns'] = time.monotonic_ns()-collection_started
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
