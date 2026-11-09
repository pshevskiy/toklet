"""Independent unchanged scalar graph through128/256inputs, before text requests.

Capture both full checkpoints in one causal pass per arithmetic profile. No
Toklet code or native intermediate values enter this source-backed reference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import qwen35_real_reference as prior

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = (128, 256)
DEADLINE_SECONDS = 21600
MAX_CAPTURE = 512 * 1024**2
LEDGER = {**prior.LEDGER, 'capture': MAX_CAPTURE}


def plan():
    tokens = [prior.graph.TOKENS[i % len(prior.graph.TOKENS)] for i in range(256)]
    values = {n: prior.VALUES + (n-1)*16640 for n in CHECKPOINTS}
    return tokens, 0, values


def checkpoint(token):
    if type(token) is not int or not 0 <= token < 256:
        raise ValueError('Typed bounded Qwen reference token required')
    return token+1 if token+1 in CHECKPOINTS else None


def extras():
    return [*prior.extras(), Path(__file__).resolve()]


def collect(output, directory, admission, endpoint):
    import numpy as np
    prior.weights.need(np.__version__ == '2.3.2', 'Pinned NumPy2.3.2 required')
    tokens, offset, values = plan()
    output = output.resolve()
    prior.weights.need(output.is_relative_to(ROOT/'artifacts'), 'Fresh local artifact output required')
    output.mkdir()
    rows_bytes = sum(values.values())*12
    record = dict(format='toklet-qwen35-context256-reference-v1', completed=False,
        repository=prior.weights.REPOSITORY, revision=prior.weights.REVISION,
        source_sha256=prior.pin.FILE_SHA, tokens=tokens, position_offset=offset,
        checkpoints=list(CHECKPOINTS), geometry=prior.weights.GEOMETRY, gate=prior.graph.GATE,
        arrays=[], planned_rows_bytes=rows_bytes, ledger=LEDGER, runtime_admission=admission,
        source=None, primary_error=None, endpoint_errors=[], engine_executed=False,
        process=dict(pid=os.getpid(),parent_pid=os.getppid(),group=os.getpgrp()),
        execution_environment={key:os.environ.get(key) for key in ['PYTHONNOUSERSITE','OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MallocLargeCache','MallocNanoZone','MallocAggressiveMadvise']},
        contextual_numerical_qualified=False, request_metrics_qualified=False,
        quality_qualified=False, managed_adoption_qualified=False)
    owner = None
    try:
        prior.weights.need(sum(LEDGER.values()) <= 6*1024**3 and rows_bytes <= MAX_CAPTURE,
                           'Reference resource plan exceeds unchanged6GiB bound')
        config = prior.runtime_owner.runtime.captured(directory/'config.json', 3024, prior.pin.CONFIG_SHA)
        deadline = time.monotonic()+DEADLINE_SECONDS
        with prior.bf16_source.Source(directory/'model.safetensors', prior.pin.FILE_BYTES,
                                     prior.pin.FILE_SHA, deadline=deadline) as owner:
            matrices, vectors = prior.weights.weights(owner, config, deadline)
            with (output/'rows.bin').open('xb') as binary:
                for profile, staged in [('ideal_fp64', False), ('staged_fp32', True)]:
                    seen = {n: set() for n in CHECKPOINTS}
                    counts = {n: 0 for n in CHECKPOINTS}
                    started = time.monotonic()
                    print(json.dumps(dict(event='profile-start', profile=profile)), flush=True)

                    def save(row):
                        prior.bf16_source.check_deadline(deadline, None)
                        if row['stage'] == 'layer_output' and row['layer'] == 31:
                            print(json.dumps(dict(event='token-complete', profile=profile, token=row['token'],
                                                  elapsed_seconds=time.monotonic()-started)), flush=True)
                        n = checkpoint(row['token'])
                        if n is None:
                            return
                        data = np.asarray(row.pop('values'), dtype='<f4' if staged else '<f8')
                        key = row['layer'], row['stage']
                        prior.weights.need(key not in seen[n] and data.ndim == 1 and np.isfinite(data).all(),
                                           'Unique complete finite reference arrays required')
                        seen[n].add(key)
                        raw = data.tobytes(order='C')
                        prior.weights.need(binary.tell()+len(raw) <= rows_bytes <= MAX_CAPTURE, 'Reference capture bound exceeded')
                        record['arrays'].append(dict(checkpoint=n, profile=profile, **row, count=len(data),
                            offset=binary.tell(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
                        prior.weights.need(binary.write(raw) == len(raw), 'Short reference write')
                        counts[n] += len(data)

                    trace = prior.graph.forward(matrices, vectors, tokens, offset, staged,
                        geometry=prior.weights.GEOMETRY, math_type=prior.weights.Math, sink=save)
                    prior.weights.need(trace == [] and all(len(seen[n]) == prior.ARRAYS and counts[n] == values[n]
                                                          for n in CHECKPOINTS), 'Complete checkpoint inventories differ')
                prior.weights.need(binary.tell() == rows_bytes, 'Final reference extent differs')
            record['source_endpoint_sha256'] = prior.source_endpoint(owner, deadline)
        record['source'] = owner.audit()
        record['rows_sha256'] = prior.runtime_owner.sha(output/'rows.bin')
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
        try:
            record['runtime_endpoint'] = endpoint()
        except BaseException as error:
            record['endpoint_errors'].append(str(error)[:4096])
        if record['endpoint_errors']:
            record['completed'] = False
        (output/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True);parser.add_argument('--expected-runtime-sha256',required=True)
    parser.add_argument('--native-manifest',type=Path,required=True);parser.add_argument('--expected-native-sha256',required=True)
    args=parser.parse_args()
    admission,endpoint=prior.runtime_owner.runtime_context(args.runtime_manifest,args.expected_runtime_sha256,
        args.native_manifest,args.expected_native_sha256,extra_sources=extras())
    result=collect(args.output,args.model_directory,admission,endpoint)
    print(json.dumps({key:result[key] for key in ['completed','primary_error','endpoint_errors']}))
    raise SystemExit(0 if result['completed'] else 1)
