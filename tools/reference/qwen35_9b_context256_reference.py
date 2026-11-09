"""Independent pinned9B 128/256 checkpoints with the qualified exact vector graph.

One causal pass per original arithmetic profile; no target engine inputs/imports.
The original one/context17 producer and all numerical arithmetic stay unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import qwen35_9b_vector_reference as prior

CHECKPOINTS=(128,256)
POSITION_OFFSET=37
DEADLINE_SECONDS=28800
MAX_CAPTURE=512*1024**2
LEDGER={**prior.LEDGER,'capture':MAX_CAPTURE}


def plan():
    tokens=[prior.graph.TOKENS[i%len(prior.graph.TOKENS)] for i in range(256)]
    values={n:prior.VALUES+(n-1)*16640 for n in CHECKPOINTS}
    return tokens,POSITION_OFFSET,values


def checkpoint(token):
    if type(token) is not int or not 0<=token<256:raise ValueError('Typed bounded9B checkpoint token required')
    return token+1 if token+1 in CHECKPOINTS else None


def extras():
    return [*prior.extras(),Path(__file__).resolve()]


def collect(output,directory,admission,endpoint):
    import numpy as np
    prior.weights.need(np.__version__=='2.3.2','Pinned NumPy2.3.2 required')
    tokens,offset,values=plan();output=output.resolve()
    prior.weights.need(output.parent.is_dir(),'Existing parent for exclusive fresh artifact output required')
    output.mkdir();rows_bytes=sum(values.values())*12
    record=dict(format='toklet-qwen35-9b-context256-reference-v1',completed=False,
        reference_variant='ordered_vector_recurrence_context256',graph_provenance=prior.builder.validate(),
        repository=prior.weights.REPOSITORY,revision=prior.weights.REVISION,source_sha256=prior.pin.FILE_SHA,
        tokens=tokens,position_offset=offset,checkpoints=list(CHECKPOINTS),geometry=prior.weights.GEOMETRY,
        gate=prior.graph.GATE,arrays=[],planned_rows_bytes=rows_bytes,ledger=LEDGER,runtime_admission=admission,
        profile_ns={},source_open_and_bind_ns=0,source_endpoint_ns=0,collection_ns=0,
        source=None,primary_error=None,endpoint_errors=[],engine_executed=False,
        process=dict(pid=os.getpid(),parent_pid=os.getppid(),group=os.getpgrp()),
        execution_environment={k:os.environ.get(k) for k in ['PYTHONNOUSERSITE','PYTHONPYCACHEPREFIX',
            'OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS',
            'MallocLargeCache','MallocNanoZone','MallocAggressiveMadvise']},
        contextual_numerical_qualified=False,request_metrics_qualified=False,
        quality_qualified=False,managed_adoption_qualified=False)
    owner=None;started=time.monotonic_ns()
    try:
        prior.weights.need(sum(LEDGER.values())<=6*1024**3 and rows_bytes<=MAX_CAPTURE,'Reference resource plan exceeds frozen limits')
        config=prior.runtime_owner.runtime.captured(directory/'config.json',prior.weights.CONFIG_BYTES,prior.pin.CONFIG_SHA)
        deadline=time.monotonic()+DEADLINE_SECONDS;source_started=time.monotonic_ns()
        with prior.bf16_source.Source(directory/'model.safetensors',prior.pin.FILE_BYTES,prior.pin.FILE_SHA,deadline=deadline) as owner:
            matrices,vectors=prior.weights.weights(owner,config,deadline)
            record['source_open_and_bind_ns']=time.monotonic_ns()-source_started
            with (output/'rows.bin').open('xb') as binary:
                for profile,staged in [('ideal_fp64',False),('staged_fp32',True)]:
                    seen={n:set() for n in CHECKPOINTS};counts={n:0 for n in CHECKPOINTS}
                    profile_started=time.monotonic_ns()
                    print(json.dumps(dict(event='profile-start',profile=profile)),flush=True)
                    def save(row):
                        prior.bf16_source.check_deadline(deadline,None)
                        if row['stage']=='layer_output' and row['layer']==31:
                            print(json.dumps(dict(event='token-complete',profile=profile,token=row['token'],
                                elapsed_seconds=(time.monotonic_ns()-profile_started)/1e9)),flush=True)
                        n=checkpoint(row['token'])
                        if n is None:return
                        key=(row['layer'],row['stage']);data=np.asarray(row.pop('values'),dtype='<f4' if staged else '<f8')
                        prior.weights.need(key not in seen[n] and data.ndim==1 and np.isfinite(data).all(),'Unique complete finite checkpoint arrays required')
                        seen[n].add(key);raw=data.tobytes(order='C')
                        prior.weights.need(binary.tell()+len(raw)<=rows_bytes<=MAX_CAPTURE,'Checkpoint capture bound exceeded')
                        record['arrays'].append(dict(checkpoint=n,profile=profile,**row,count=len(data),
                            offset=binary.tell(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
                        prior.weights.need(binary.write(raw)==len(raw),'Short reference checkpoint write')
                        counts[n]+=len(data)
                    trace=prior.graph.forward(matrices,vectors,tokens,offset,staged,
                        geometry=prior.weights.GEOMETRY,math_type=prior.weights.math_type(matrices),sink=save)
                    prior.weights.need(trace==[] and all(len(seen[n])==prior.ARRAYS and counts[n]==values[n] for n in CHECKPOINTS),
                        'Complete streaming checkpoint inventory differs')
                    record['profile_ns'][profile]=time.monotonic_ns()-profile_started
                prior.weights.need(binary.tell()==rows_bytes,'Final checkpoint capture size differs')
            endpoint_started=time.monotonic_ns();record['source_endpoint_sha256']=prior.source_endpoint(owner,deadline)
            record['source_endpoint_ns']=time.monotonic_ns()-endpoint_started
        record['source']=owner.audit();record['rows_sha256']=prior.runtime_owner.sha(output/'rows.bin');record['completed']=True
    except BaseException as error:
        record['primary_error']=dict(type=type(error).__name__,message=str(error)[:4096])
    finally:
        if owner is not None:
            if not owner.closed:
                try:owner.close()
                except BaseException as error:record['endpoint_errors'].append(str(error)[:4096])
            record['source']=owner.audit()
        try:record['runtime_endpoint']=endpoint()
        except BaseException as error:record['endpoint_errors'].append(str(error)[:4096])
        if record['endpoint_errors']:record['completed']=False
        record['collection_ns']=time.monotonic_ns()-started
        (output/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime-manifest',type=Path,required=True);p.add_argument('--expected-runtime-sha256',required=True)
    p.add_argument('--native-manifest',type=Path,required=True);p.add_argument('--expected-native-sha256',required=True);a=p.parse_args()
    admission,endpoint=prior.runtime_owner.runtime_context(a.runtime_manifest,a.expected_runtime_sha256,
        a.native_manifest,a.expected_native_sha256,extra_sources=extras())
    result=collect(a.output,a.model_directory,admission,endpoint)
    print(json.dumps({k:result[k] for k in ['completed','primary_error','endpoint_errors']}))
    raise SystemExit(0 if result['completed'] else 1)
