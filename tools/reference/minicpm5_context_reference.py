"""Bound-only CPU domain extension under unchanged original MiniCPM5 gates."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import minicpm5 as model
import minicpm5_context_math as graph
import minicpm5_context_runtime as runtime

ROOT=Path(__file__).resolve().parents[2]
COUNTS=[128,512,1025]
CONTRACT_SHA='b5464b7d67cee7fa8d0df28afc5fb0e68f0c5d550232e572f93858dc4f61bad3'
QUALITY_SHA='fe954ea5afa82f3fc33981c139727bef4fb3cd10f1b6485b7f0001f3b43875a7'
TOKENS_SHA='297451dd509fd620d91a997dac58ad40f514f4fa2f63efcc92253a314c7476cc'
OLD_SHA='360cf36c6f743fb4af37c07dbd0e0c78d22e0c9922653629f0707131c88657b3'
need=model.need


def parsed(raw):
    return json.loads(raw,object_pairs_hook=runtime.common.json_unique,
                      parse_constant=runtime.common.bad_json_number)


def sources():
    return [ROOT/name for name in ['tools/reference/minicpm5_context_reference.py',
        'tools/reference/minicpm5_context_math.py','tools/reference/minicpm5_context_runtime.py',
        'tools/reference/test_minicpm5_context_reference.py','tools/reference/minicpm5.py',
        'tools/reference/qwen_fp16.py','tools/reference/qwen.py','tools/bench/prepare.py']]


def graph_identity(original, extended):
    def function(raw):
        tree=ast.parse(raw)
        return copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    a,b=function(original),function(extended)
    # Exactly one integer literal changes, inside the original admission call.
    matches=[]
    admissions=[node for node in ast.walk(b) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Name) and node.func.id=='need']
    need(len(admissions)==1,'Original one admission call required')
    for node in ast.walk(admissions[0]):
        if isinstance(node,ast.Constant) and type(node.value) is int and node.value==1025:
            matches.append(node)
    need(len(matches)==1,'One explicit domain-bound literal required')
    matches[0].value=512
    need(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),
         'Original CPU arithmetic/storage/graph body changed')
    return dict(arithmetic_ast_equal_after_restoring_guard=True,original_guard=512,diagnostic_guard=1025,
        original_sha256=hashlib.sha256(original).hexdigest(),extended_sha256=hashlib.sha256(extended).hexdigest())


def admission():
    mib=dict(pread_and_hash=16,affine_decode_tiles=512,linear_widening=256,blas_workspace=256,
        current_graph=512,retained_dual_profiles=3072,numpy_and_framework=512,publication_margin=256)
    total=sum(mib.values())*1024**2;need(total<=6*1024**3,'CPU admission exceeds6GiB')
    return dict(components_mib=mib,admitted_bytes=total,retains_full_model_copy=False)


def plan(contract, old, quality, tokens):
    need(contract['format']=='toklet-minicpm5-context1025-contract-v1'
         and contract['prospective_before_deeper_cpu_native_outcomes'] is True
         and [c['tokens'] for c in contract['cases']]==COUNTS
         and [c['name'] for c in contract['cases']]==[f'prefix{n}' for n in COUNTS]
         and all(type(c['position_offset']) is int and c['position_offset']==0 for c in contract['cases']),
         'Fixed deeper cases differ')
    gates={k:v for k,v in old['gates'].items() if not k.endswith(('attention_scores','attention_probabilities'))}
    need(len(gates)==51 and contract['gates']==gates and set(contract['columns'])==set(gates),
         'Original six-case gates changed or expanded')
    need(contract['model']==dict(repository=model.REPOSITORY,revision=model.REVISION,
              sha256=model.PINS['model.safetensors'])
         and contract['quality_inputs_manifest_sha256']==QUALITY_SHA
         and contract['quality_token_sha256']==TOKENS_SHA
         and contract['original_reference_manifest_sha256']==OLD_SHA,'Pinned source identity differs')
    need(quality['format']=='toklet-minicpm5-complete-quality-inputs-v1'
         and quality['loss_targets']==10000 and quality['task_cases']==100
         and quality['quality_scored'] is False and len(tokens)==40004
         and hashlib.sha256(tokens).hexdigest()==TOKENS_SHA,'Full quality input identity differs')
    need(contract['native']==dict(chunks=[1,32],normalization='Compensated',attention='Reference',
        projection='MlxAffine',rms_store='serial baseline',capacity=1025,capture_limit_bytes=4*1024**3)
         and contract['memory_ceiling_bytes']==6*1024**3
         and contract['disk_floor_bytes']==2*1024**3
         and contract['unchanged_ordinary_capacity']==512,'Frozen limits/modes differ')
    ids=list(struct.unpack('<10001I',tokens))
    need(all(t<130560 for t in ids),'Quality token outside vocabulary')
    arrays={}
    for count in COUNTS:
        for stage,columns in contract['columns'].items():
            need(type(columns) is int and columns>0,'Invalid reference row geometry')
            arrays[f'prefix{count}.{stage}']=dict(shape=[count,columns],dtype='<f4' if stage=='logits' else '<f2',gate=stage)
    total=sum(v['shape'][0]*v['shape'][1]*(4 if v['dtype']=='<f4' else 2) for v in arrays.values())*2
    need(total==2640983040 and total<=3*1024**3,'Complete reference output exceeds fixed bound')
    return ids,arrays,total


def export(directory, output, contract, contract_sha, old, quality, tokens, admitted):
    import numpy as np
    need(np.__version__=='2.3.2','Pinned CPU NumPy required')
    ids,inventory,expected_bytes=plan(contract,old,quality,tokens)
    identity=graph_identity(Path(model.__file__).read_bytes(),Path(graph.__file__).read_bytes())
    ledger=admission();output.mkdir();arrays={};completed=[];started=time.monotonic_ns()
    streams={name:(output/(name+'.bin')).open('xb') for name in ['fp32','fp64']}
    try:
        with model.Source(directory) as weights:
            for count in COUNTS:
                for name,precision in [('fp32',np.float32),('fp64',np.float64)]:
                    clock=time.monotonic_ns()
                    print(json.dumps(dict(event='profile-start',case=f'prefix{count}',profile=name)),flush=True)
                    values=graph.forward(ids[:count],0,weights,model.GEOMETRY,precision,[0,1,41])
                    for stage in contract['columns']:
                        key=f'prefix{count}.{stage}';spec=inventory[key];value=values[stage]
                        need(value.flags.c_contiguous and value.size==count*spec['shape'][1]
                             and value.dtype==np.dtype(spec['dtype']), 'CPU storage/shape differs: '+key)
                        stream=streams[name];offset=stream.tell();digest=hashlib.sha256();flat=value.reshape(-1)
                        for first in range(0,len(flat),65536):
                            tile=flat[first:first+65536]
                            need(np.isfinite(tile).all(),'Nonfinite reference output')
                            raw=tile.tobytes();need(stream.write(raw)==len(raw),'Short CPU reference write');digest.update(raw)
                        arrays[name+'.'+key]=dict(**spec,file=name+'.bin',offset=offset,
                            bytes=stream.tell()-offset,sha256=digest.hexdigest())
                    del values
                    streams[name].flush();completed.append(dict(case=f'prefix{count}',profile=name,elapsed_ns=time.monotonic_ns()-clock))
                    (output/'progress.json').write_text(json.dumps(dict(completed=completed,native_outcomes_seen=False),indent=2)+'\n')
                    print(json.dumps(dict(event='profile-completed',**completed[-1])),flush=True)
    finally:
        for stream in streams.values():stream.close()
    need(len(arrays)==306 and sum(a['bytes'] for a in arrays.values())==expected_bytes,'Incomplete CPU reference coverage')
    return dict(format='toklet-minicpm5-context1025-reference-v1',completed=True,
        model=dict(repository=model.REPOSITORY,revision=model.REVISION,files=model.PINS),
        cases=contract['cases'],quality_manifest_sha256=QUALITY_SHA,contract_sha256=contract_sha,
        original_reference_manifest_sha256=OLD_SHA,gates=contract['gates'],columns=contract['columns'],
        graph_identity=identity,memory_admission=ledger,admitted_runtime=admitted,arrays=arrays,
        files={name+'.bin':dict(bytes=(output/(name+'.bin')).stat().st_size,sha256=model.sha(output/(name+'.bin')))
               for name in streams},profile_records=completed,elapsed_ns=time.monotonic_ns()-started,
        native_outcomes_seen=False,quality_scored=False,managed_adoption_qualified=False,
        gates_recomputed_or_relaxed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['environment_root','model','old_reference','quality_inputs','contract','contract_sha256',
                 'runtime_manifest','runtime_sha256','fresh_output']:
        parser.add_argument('--'+name.replace('_','-'),required=True)
    args=parser.parse_args();runtime.bind_environment(args.environment_root)
    need(args.contract_sha256==CONTRACT_SHA,'Exact prospective contract digest required')
    frozen=parsed(runtime.common.captured(Path(args.contract),2*1024**2,CONTRACT_SHA))
    old=parsed(runtime.common.captured(Path(args.old_reference)/'manifest.json',2*1024**2,OLD_SHA))
    quality=parsed(runtime.common.captured(Path(args.quality_inputs)/'manifest.json',2*1024**2,QUALITY_SHA))
    tokens=runtime.common.captured(Path(args.quality_inputs)/'held_out_loss.u32le',40004,TOKENS_SHA)
    admitted,endpoint=runtime.admit(Path(args.runtime_manifest),args.runtime_sha256,sources())
    result=export(Path(args.model),Path(args.fresh_output),frozen,args.contract_sha256,old,quality,tokens,admitted)
    result['endpoint_origins']=endpoint()
    with (Path(args.fresh_output)/'manifest.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed=True,arrays=len(result['arrays']),bytes=sum(a['bytes'] for a in result['arrays'].values()),quality_scored=False)))


if __name__=='__main__':main()
