"""Prospective bounded same-artifact source-loss producer; explicit invocation only.

Reuse the frozen independent graph without selected-layer traces. One257-row
FP32 logits matrix lives at a time. Write per-target evidence immediately;
never retain four full blocks or main and wide matrices together.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
import prepare_llama3_loss1024 as inputs
import mlx_llama3_real as source


def row_statistics(row,target):
    import numpy as np
    if row.ndim!=1 or row.dtype!=np.float32 or not len(row) or type(target) is not int or not 0<=target<len(row):
        raise ValueError('Invalid loss row/target')
    if not np.isfinite(row).all():raise ValueError('Nonfinite loss distribution')
    maximum=float(np.max(row));target_logit=float(row[target])
    # Stable across the full FP32 finite domain; do not subtract two enormous
    # nearly equal logZ/target values after adding logmass to the maximum.
    mass=math.fsum(math.exp(float(value)-maximum) for value in row)
    nll=(maximum-target_logit)+math.log(mass)
    if not (math.isfinite(mass) and 1<=mass<=len(row) and math.isfinite(nll) and nll>=0):
        raise ValueError('Invalid FP64 loss statistic')
    raw=row.astype('<f4',copy=False).tobytes()
    return dict(vocab=len(row),row_sha256=hashlib.sha256(raw).hexdigest(),
                target=target,target_fp32_bits=struct.unpack('<I',struct.pack('<f',target_logit))[0],
                maximum=maximum,exp_mass=mass,nll=nll,greedy_lowest_id=int(np.argmax(row)))


def produce(folder,input_folder,input_sha,short_folder,short_sha):
    folder=Path(folder)
    if os.path.lexists(folder):raise FileExistsError('Fresh loss output required')
    folder.mkdir();report=dict(format='toklet-llama3-loss1024-source-v1',completed=False,errors=[],usage=[],blocks=[],
                              input_manifest_sha256=input_sha,short_reference_manifest_sha256=short_sha,
                              model=dict(repository=source.REPO,revision=source.REVISION,file_pins=source.PINS),
                              native_executed=False,full_quality_qualified=False)
    try:
        return execute(folder,input_folder,input_sha,short_folder,short_sha,report)
    except BaseException as error:
        report['errors'].append(dict(type=type(error).__name__,message=str(error),notes=getattr(error,'__notes__',[])))
        try:(folder/'failure.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        except BaseException as cleanup:error.add_note('Failure receipt: '+repr(cleanup))
        raise


def execute(folder,input_folder,input_sha,short_folder,short_sha,report):
    for name in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
    if importlib.metadata.version('numpy')!='2.3.2':raise ValueError('Pinned NumPy differs')
    manifest,ids=inputs.load(input_folder,input_sha)
    source.verify(Path(short_folder),short_sha)
    paths=list(source.PRODUCER_NAMES)+['tools/reference/llama3_loss1024.py','tools/bench/prepare_llama3_loss1024.py']
    producer={name:source.sha(ROOT/name) for name in paths};packages=source.package_pins()
    import numpy as np
    import llama3_fp16 as graph
    sys.path.insert(0,str(ROOT/'tools/experiments'))
    import native_usage
    report.update(producer_pins=producer,package_pins=packages,numpy=np.__version__,admission=source.admission(),
                  policy=manifest['policy'],blocks_contract=manifest['blocks'],profile=source.NATIVE_PROFILE,
                  storage='FP32 logits; main NumPy FP32/Half, wide NumPy FP64/Half; canonical rotary unchanged')
    if np.__version__!='2.3.2':raise ValueError('Imported NumPy differs')
    def usage(phase):
        value=dict(phase=phase,usage=native_usage.usage());report['usage'].append(value)
        with (folder/'progress.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        if not 0<value['usage']['lifetime_max_phys_footprint']<=source.MEMORY_LIMIT:raise ValueError('Source exceeded6GiB')
    totals={'main':[],'wide':[]};counts={'main':0,'wide':0}
    usage('before')
    with (folder/'targets.jsonl').open('x') as journal,source.Weights(source.MODEL) as weights:
        for block in manifest['blocks']:
            tokens=ids[block['input_start']:block['input_end']]
            if len(tokens)!=257:raise ValueError('Fixed block length differs')
            for name,precision in [('main',np.float32),('wide',np.float64)]:
                # Full-block independent causal graph with empty local KV,
                # absolute position0 each block. Final row is unscored.
                outputs=graph.forward(tokens,0,weights,source.NUMERICAL,precision,[],source.SCALING)
                logits=outputs['logits']
                if logits.shape!=(257,128256) or logits.dtype!=np.float32:raise ValueError('Loss graph shape/dtype differs')
                all_rows=hashlib.sha256()
                for row in range(256):
                    record=row_statistics(logits[row],tokens[row+1])
                    all_rows.update(logits[row].astype('<f4',copy=False).tobytes())
                    record.update(profile=name,block=block['block'],local_row=row,
                                  global_target_index=block['input_start']+row+1)
                    journal.write(json.dumps(record,allow_nan=False)+'\n');journal.flush()
                    totals[name].append(record['nll']);counts[name]+=1
                last=logits[256].astype('<f4',copy=False).tobytes();all_rows.update(last)
                report['blocks'].append(dict(block=block['block'],profile=name,rows=257,targets=256,
                                             all_rows_sha256=all_rows.hexdigest(),last_unscored_row_sha256=hashlib.sha256(last).hexdigest()))
                del logits,outputs,last
                gc.collect();usage(str(block['block'])+'.'+name+'.released')
        weights.verify()
    if counts!={'main':1024,'wide':1024}:raise ValueError('Incomplete target coverage')
    means={name:math.fsum(values)/1024 for name,values in totals.items()}
    report.update(targets_per_profile=counts,mean_nll=means,main_minus_wide=means['main']-means['wide'],
                  targets_file=dict(file='targets.jsonl',bytes=(folder/'targets.jsonl').stat().st_size,sha256=source.sha(folder/'targets.jsonl')))
    inputs.load(input_folder,input_sha)
    if source.sha(Path(short_folder)/'manifest.json')!=short_sha:raise ValueError('Short reference identity changed')
    for name,digest in {**producer,**packages}.items():
        if source.sha(ROOT/name)!=digest:raise ValueError('Producer/runtime endpoint differs: '+name)
    if source.package_pins()!=packages:raise ValueError('Runtime closure differs')
    for name,digest in source.PINS.items():
        if source.sha(source.MODEL/name)!=digest:raise ValueError('Model endpoint differs: '+name)
    usage('completed')
    report['completed']=True
    (folder/'source.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['output','inputs','short-reference']:parser.add_argument('--'+name,type=Path,required=True)
    for name in ['inputs-sha256','short-reference-sha256']:parser.add_argument('--'+name,required=True)
    a=parser.parse_args();result=produce(a.output,a.inputs,a.inputs_sha256,a.short_reference,a.short_reference_sha256)
    print(json.dumps(dict(completed=result['completed'],mean_nll=result['mean_nll']),allow_nan=False))
