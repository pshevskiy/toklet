"""Distinct fixed-source Huihui-derived4B reference at original128/256 checkpoints.

Only identity and bounded capture plumbing are new. Packed-row reconstruction,
projection arithmetic and both original graph profiles remain unchanged. The
ordered recurrence helper has explicit inverse-source and bitwise controls; its
use does not promote the rejected original source or qualify this new model.
Numerical execution starts only after the caller admits the actual runtime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import bf16_source
import build_qwen35_vector_graph as builder
import gemma4_affine_reference as runtime_owner
import qwen35_real_weights as base
import qwen35_recurrent_vector as recurrent
import qwen35_vector_graph as graph

REPOSITORY = 'TheWirelessPhoenix/Qwen3.5-4B-abliterated_4bit'
REVISION = '916d9f7c2e27ed41e2b9ed39f3a2badccb2bbc7c'
FILE_BYTES = 2367237149
FILE_SHA = '244b47c46b759664c223a996046857720826a388bdf3969620a8f12ebb6e14e0'
CONFIG_BYTES = 3024
CONFIG_SHA = '13a514e981196ba683d8d53312789384ff5d9b9e73c7bfa137010397212c4af3'
HEADER_BYTES = 119829
HEADER_SHA = '7284f68df36a7761569b48321066e92d616306e33b4a9487029673a0ff344cb2'
ORIGIN = 8 + HEADER_BYTES
GEOMETRY = dict(base.GEOMETRY)
TOKENS = (1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10, 19, 8, 21, 23, 29)
CHECKPOINTS = (128, 256)
PROFILES = (('ideal_fp64', False), ('staged_fp32', True))
POSITION_OFFSET = 0
ARRAYS = 1051
VALUES = 16067328
ROWS_BYTES = 461893632
DEADLINE_SECONDS = 21600
PROCESS_LIMIT = 6 * 1024**3
HELPER_BYTES = 32 * 1024**2
MAX_CAPTURE = 512 * 1024**2
LEDGER = dict(python_state_and_temporaries=2 * 1024**3,
    interpreter_and_runtime=512 * 1024**2, small_weight_vectors=64 * 1024**2,
    tile_and_source_buffers=128 * 1024**2, capture=MAX_CAPTURE,
    guard_helper=HELPER_BYTES, controller_and_logs=64 * 1024**2)
SOURCE_PINS = dict(container_bytes=FILE_BYTES, container_sha256=FILE_SHA,
    config_bytes=CONFIG_BYTES, config_sha256=CONFIG_SHA, header_bytes=HEADER_BYTES,
    header_sha256=HEADER_SHA, payload_origin=ORIGIN)

# The builder changes only the scalar recurrence block and proves its exact
# inverse. These hashes fix that mathematical transformation and its helper.
GRAPH_PROVENANCE = dict(
    original_sha256='faa580d347df630267e8082377eb5555d89de76c9f7b149fa424239f88c46582',
    vector_sha256='fd7d173a829c2db1ca04032d039094a4546cd237b88f78870f2f641e0c92943c',
    helper_sha256='425d121dd27dd08d3f7baee95a41a446fbfcd8500ae4df192e0286777ca6b792',
    builder_sha256='932e0acf2cd658a17bd70a80a88c9febf7ffa2e9ad0945427607b406ad69c506',
    inverse_patch_exact=True)
ARITHMETIC_EQUIVALENCE = dict(
    source_proof='build_qwen35_vector_graph.validate: exact inverse recurrence-only patch',
    portable_controls='tools/reference/test_qwen35_recurrent_vector.py: real-shape repeated recurrence, signed zero/subnormal/cancellation, complete tied/untied traces in both original profiles',
    historical_checkpoint_proof='artifacts/qwen35-context256-2026-11-09/reference-graph-vector-experiment/full256-parity.json',
    historical_checkpoint_proof_sha256='96153cd566f2c98d7ef62275d0b80632050bca65f6444f5d6e98a09034768386',
    historical_scope='Original-source staged_fp32 checkpoint bytes only; not an ideal_fp64 full-context proof, new4 capture, native fidelity or quality qualification')
need = base.need


def plan():
    """Fresh metadata only: one causal256-input pass per fixed profile."""
    return ([TOKENS[i % len(TOKENS)] for i in range(256)], POSITION_OFFSET,
            {n: VALUES + (n - 1) * 16640 for n in CHECKPOINTS})


def checkpoint(token):
    if type(token) is not int or not 0 <= token < 256:
        raise ValueError('Typed bounded new4 checkpoint token required')
    return token + 1 if token + 1 in CHECKPOINTS else None


def graph_provenance():
    proof = builder.validate()
    need(proof == GRAPH_PROVENANCE, 'Frozen scalar/vector recurrence source proof differs')
    return proof


def extras():
    graph_provenance()
    modules = (base, base.pin, base.graph, base.tiles, graph, recurrent, builder,
               bf16_source)
    return sorted({Path(module.__file__).resolve() for module in modules}
                  | {Path(__file__).resolve()})


def weights_binding(owner, config, deadline):
    """Admit only this authenticated new4 owner, then use neutral row binding.

    The already-sanitized child arrays are consumed directly: norm+1 and conv
    reshaping were conversion operations and must not be repeated here.
    """
    need(type(owner) is bf16_source.Source and owner.size == FILE_BYTES
         and owner.sha256 == FILE_SHA and not owner.closed and not owner.poisoned,
         'Authenticated open fixed new4 source required')
    need(type(config) is bytes and len(config) == CONFIG_BYTES
         and hashlib.sha256(config).hexdigest() == CONFIG_SHA,
         'Pinned new4 config differs')
    need(owner.read(0, 8, deadline=deadline) == struct.pack('<Q', HEADER_BYTES),
         'Pinned new4 header prefix differs')
    raw = owner.read(8, HEADER_BYTES, deadline=deadline)
    need(len(raw) == HEADER_BYTES and hashlib.sha256(raw).hexdigest() == HEADER_SHA,
         'Pinned new4 header differs')
    table = json.loads(raw, object_pairs_hook=base.unique,
                       parse_constant=runtime_owner.runtime.bad_json_number)
    need(type(table) is dict and '__metadata__' in table,
         'Complete new4 safetensors metadata required')
    table.pop('__metadata__')
    need(len(table) == 924, 'Fixed new4 physical inventory differs')
    spans = []
    for item in table.values():
        need(type(item) is dict and set(item) == {'dtype', 'shape', 'data_offsets'},
             'Exact new4 physical component required')
        span = item['data_offsets']
        need(type(span) is list and len(span) == 2
             and all(type(n) is int for n in span)
             and 0 <= span[0] < span[1] <= FILE_BYTES - ORIGIN,
             'Bounded new4 physical component span required')
        spans.append(span)
    spans.sort()
    need(spans[0][0] == 0 and spans[-1][1] == FILE_BYTES - ORIGIN
         and all(a[1] == b[0] for a, b in zip(spans, spans[1:])),
         'Complete contiguous new4 payload coverage differs')
    matrices, vectors = base.bind(owner, table, ORIGIN, GEOMETRY, deadline)
    need(len(matrices) == 249 and len(vectors) == 177,
         'Fixed new4 logical inventory differs')
    return matrices, vectors


def source_endpoint(owner, deadline):
    digest = hashlib.sha256()
    for offset in range(0, owner.size, bf16_source.HASH_BLOCK):
        digest.update(owner.read(offset, min(bf16_source.HASH_BLOCK, owner.size - offset),
                                 deadline=deadline))
    need(digest.hexdigest() == FILE_SHA, 'Whole new4 source endpoint differs')
    return digest.hexdigest()


def collect(output, directory, admission, endpoint):
    """Heavy entry point; only a separately guarded admitted caller executes it."""
    tokens, offset, values = plan()
    output, directory = Path(output).absolute(), Path(directory).absolute()
    need(output.parent.is_dir() and not output.exists(),
         'Existing parent and exclusive fresh reference output required')
    output.mkdir()
    record = dict(format='toklet-qwen35-huihui4b-context256-reference-v1', completed=False,
        reference_variant='ordered_vector_recurrence_context256', graph_provenance=None,
        arithmetic_equivalence=dict(ARITHMETIC_EQUIVALENCE), source_pins=dict(SOURCE_PINS),
        deadline_seconds=DEADLINE_SECONDS, process_limit_bytes=PROCESS_LIMIT,
        repository=REPOSITORY, revision=REVISION, source_sha256=FILE_SHA,
        tokens=tokens, position_offset=offset, checkpoints=list(CHECKPOINTS),
        geometry=GEOMETRY, gate=graph.GATE, arrays=[], planned_rows_bytes=ROWS_BYTES,
        ledger=LEDGER, runtime_admission=admission, source=None,
        source_endpoint_sha256=None, rows_sha256=None, runtime_endpoint=None,
        primary_error=None, endpoint_errors=[], engine_executed=False,
        process=dict(pid=os.getpid(), parent_pid=os.getppid(), group=os.getpgrp()),
        execution_environment={key: os.environ.get(key) for key in [
            'PYTHONNOUSERSITE', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'OMP_NUM_THREADS', 'MallocLargeCache', 'MallocNanoZone', 'MallocAggressiveMadvise']},
        contextual_numerical_qualified=False, request_metrics_qualified=False,
        quality_qualified=False, managed_adoption_qualified=False)
    owner, config = None, None
    try:
        import numpy as np
        need(np.__version__ == '2.3.2', 'Pinned NumPy2.3.2 required')
        record['graph_provenance'] = graph_provenance()
        need(sum(LEDGER.values()) <= PROCESS_LIMIT and sum(values.values()) * 12 == ROWS_BYTES
             and ROWS_BYTES <= MAX_CAPTURE, 'Reference plan exceeds fixed limits')
        config = runtime_owner.runtime.captured(directory / 'config.json', CONFIG_BYTES, CONFIG_SHA)
        need(len(config) == CONFIG_BYTES, 'Exact new4 config length required')
        deadline = time.monotonic() + DEADLINE_SECONDS
        with bf16_source.Source(directory / 'model.safetensors', FILE_BYTES, FILE_SHA,
                                deadline=deadline) as owner:
            matrices, vectors = weights_binding(owner, config, deadline)
            rows_digest = hashlib.sha256()
            with (output / 'rows.bin').open('xb') as binary:
                for profile, staged in PROFILES:
                    seen = {n: set() for n in CHECKPOINTS}
                    counts = {n: 0 for n in CHECKPOINTS}
                    started = time.monotonic()
                    print(json.dumps(dict(event='profile-start', profile=profile)), flush=True)

                    def save(row):
                        bf16_source.check_deadline(deadline, None)
                        if row['stage'] == 'layer_output' and row['layer'] == 31:
                            print(json.dumps(dict(event='token-complete', profile=profile,
                                token=row['token'], elapsed_seconds=time.monotonic() - started)),
                                flush=True)
                        n = checkpoint(row['token'])
                        if n is None:
                            return
                        key = row['layer'], row['stage']
                        data = np.asarray(row.pop('values'), dtype='<f4' if staged else '<f8')
                        need(key not in seen[n] and data.ndim == 1 and len(data) > 0
                             and np.isfinite(data).all(),
                             'Unique complete finite new4 checkpoint arrays required')
                        need(counts[n] + len(data) <= values[n]
                             and len(seen[n]) < ARRAYS, 'Checkpoint inventory bound exceeded')
                        raw = data.tobytes(order='C')
                        need(binary.tell() + len(raw) <= ROWS_BYTES <= MAX_CAPTURE,
                             'New4 capture byte bound exceeded')
                        descriptor = dict(checkpoint=n, profile=profile, token=row['token'],
                            layer=row['layer'], stage=row['stage'], count=len(data),
                            offset=binary.tell(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                        need(binary.write(raw) == len(raw), 'Short new4 reference write')
                        rows_digest.update(raw)
                        record['arrays'].append(descriptor)
                        seen[n].add(key)
                        counts[n] += len(data)

                    trace = graph.forward(matrices, vectors, tokens, offset, staged,
                        geometry=GEOMETRY, math_type=base.Math, sink=save)
                    need(trace == [] and all(len(seen[n]) == ARRAYS and counts[n] == values[n]
                                             for n in CHECKPOINTS),
                         'Complete new4 checkpoint inventory differs')
                need(binary.tell() == ROWS_BYTES and len(record['arrays']) == 4204,
                     'Final new4 reference extent differs')
                binary.flush()
                os.fsync(binary.fileno())
            record['source_endpoint_sha256'] = source_endpoint(owner, deadline)
        record['source'] = owner.audit()
        record['rows_sha256'] = rows_digest.hexdigest()
        record['completed'] = True
    except BaseException as error:
        record['primary_error'] = dict(type=type(error).__name__, message=str(error)[:4096])
    finally:
        if owner is not None:
            if not owner.closed:
                try:
                    owner.close()
                except BaseException as error:
                    record['endpoint_errors'].append(str(error)[:4096])
            record['source'] = owner.audit()
        if config is not None:
            try:
                need(runtime_owner.runtime.captured(directory / 'config.json', CONFIG_BYTES,
                     CONFIG_SHA) == config, 'New4 config endpoint differs')
            except BaseException as error:
                record['endpoint_errors'].append(str(error)[:4096])
        try:
            record['runtime_endpoint'] = endpoint()
        except BaseException as error:
            record['endpoint_errors'].append(str(error)[:4096])
        if record['endpoint_errors']:
            record['completed'] = False
        with (output / 'result.json').open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--native-manifest', type=Path, required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    args = parser.parse_args()
    admission, endpoint = runtime_owner.runtime_context(args.runtime_manifest,
        args.expected_runtime_sha256, args.native_manifest, args.expected_native_sha256,
        extra_sources=extras())
    result = collect(args.output, args.model_directory, admission, endpoint)
    print(json.dumps({key: result[key] for key in ['completed', 'primary_error', 'endpoint_errors']}))
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
