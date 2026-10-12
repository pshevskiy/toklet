"""Qualify a resident tiny gated MLP with GPU propagation and Half boundaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import rust_metal_mlp_projections as stages
p=stages.projection
rms=p.rms
HALF_STAGES=('gate_projection','gate_silu','up_projection','gated','down_projection')


def cases():
    groups,projection_anchors=stages.batches()
    sources={c['id']:c for _,items in groups for c in items}
    half=rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA)
    single=rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
    inputs,anchors=[],{}
    for name,spec in sorted(half['cases'].items()):
        rows=len(spec['tokens'])
        for layer in range(2):
            prefix=f'cases.{name}.layers.{layer}'
            gate=sources[f'{prefix}.gate_projection'];up=sources[f'{prefix}.up_projection'];down=sources[f'{prefix}.down_projection']
            if gate['input_bits']!=up['input_bits']:raise ValueError('MLP projection normalized inputs differ')
            case={'format':'toklet-metal-mlp-input-v1','id':f'{prefix}.mlp','rows':rows,'input_bits':gate['input_bits'],
                  'gate_weight_bits':gate['weight_bits'],'up_weight_bits':up['weight_bits'],'down_weight_bits':down['weight_bits']}
            expected=[]
            for stage in HALF_STAGES:
                key=f'{prefix}.{stage}';count=rows*(32 if stage=='down_projection' else 64)
                data,item=rms.payload(rms.FP16,half,key,'f16le');reference,ref=rms.payload(rms.FP32,single,key,'f32le')
                shape=[rows,32 if stage=='down_projection' else 64]
                if item['shape']!=shape or ref['shape']!=shape:raise ValueError('MLP stage shape differs')
                bits=list(struct.unpack(f'<{count}H',data));values=list(struct.unpack(f'<{count}f',reference))
                if any(n&0x7c00==0x7c00 for n in bits) or not all(map(math.isfinite,values)):raise ValueError('MLP stage reference nonfinite')
                expected.append({'bits':bits,'source':values,'gate':item['gate_against_source_fp32'],
                                 'oracle_sha256':item['sha256'],'source_sha256':ref['sha256']})
            anchors[case['id']]={'stages':expected,'input_sha256':projection_anchors[gate['id']]['input_sha256'],
                                'weight_sha256':[projection_anchors[source['id']]['weight_sha256'] for source in (gate,up,down)]}
            inputs.append(case)
    if len(inputs)!=10:raise ValueError('MLP frozen inventory differs')
    overflow={'format':'toklet-metal-mlp-input-v1','id':'manual.overflow','rows':1,'input_bits':[0x4000]+[0]*31,
              'gate_weight_bits':[0x7bff]+[0]*2047,'up_weight_bits':[0x3c00]+[0]*2047,'down_weight_bits':[0x3c00]+[0]*2047}
    return inputs,anchors,overflow


def validate(event,case,anchor):
    if event.get('event')!='idle_result' or event.get('passed') is not True:raise ValueError('Native composed MLP failed')
    r=event['result'];m=r['metadata'];rows=case['rows'];width=r['threadgroup_width']
    if (r['operation']!='resident_tiny_mlp_half_boundaries' or r['id']!=case['id'] or r['rows_count']!=rows
            or r['hidden']!=32 or r['intermediate']!=64 or m['mps_device_supported'] is not True
            or m['unified_memory'] is not True or m['language_version']!='2.3' or m['math_mode']!='safe'
            or m['math_functions']!='precise' or m['minimum_macos_major']!=15 or r['allocations_reused'] is not True):
        raise ValueError('MLP policy/topology differs')
    caps=m['graph_pipeline_caps']
    if [c['name'] for c in caps]!=['toklet_projection_bias','toklet_silu','toklet_gating']:raise ValueError('MLP pipeline inventory differs')
    limits=[m['execution_width'],m['pipeline_max_threads'],m['device_max_threads'][0]]+[n for c in caps for n in (c['execution_width'],c['max_threads'])]
    if type(width) is not int or not 0<width<=min(limits):raise ValueError('MLP launch limits differ')
    cap64=math.ceil(math.ceil(rows*64/width)*width/64);cap32=math.ceil(math.ceil(rows*32/width)*width/32)
    if r['capacity_rows64']!=cap64 or r['capacity_rows32']!=cap32 or r['half_intermediate_stride']!=64:raise ValueError('MLP inactive/contiguous storage differs')
    dims=[('input_stride',32),('gate_weight_stride',32),('down_weight_stride',64),('bias_stride',64),
          ('half_output_stride',32),('float_intermediate_stride',64),('float_output_stride',32)]
    if any(type(r[k]) is not int or r[k]<minimum for k,minimum in dims):raise ValueError('MLP queried strides differ')
    inp=(rows*r['input_stride']+7)*2+2*(64*r['gate_weight_stride']+7)*2+(32*r['down_weight_stride']+7)*2+(r['bias_stride']+7)*2
    out=4*(cap64*64+7)*2+(cap32*r['half_output_stride']+7)*2+2*(cap64*r['float_intermediate_stride']+7)*4+(cap32*r['float_output_stride']+7)*4
    shared=inp+out;total=shared*3+rms.probe.ALLOWANCE
    if (r['shared_buffer_bytes']!=shared or r['host_snapshot_readback_bytes']!=shared*2 or r['admitted_bytes']!=total
            or m['maximum_admitted_bytes']!=total or total>rms.probe.LIMIT):raise ValueError('MLP memory accounting differs')
    half=r['first_half_intermediates'];single=r['first_fp32_projection_bits']
    counts=[rows*64]*4+[rows*32]
    if len(half)!=5 or len(single)!=3 or any(len(a)!=n for a,n in zip(half,counts)) or any(len(a)!=n for a,n in zip(single,(rows*64,rows*64,rows*32))):raise ValueError('MLP intermediate inventory differs')
    if any(type(n) is not int or not 0<=n<=65535 or n&0x7c00==0x7c00 for array in half for n in array):raise ValueError('MLP intermediate Half nonfinite/type differs')
    if any(type(n) is not int or not 0<=n<2**32 for array in single for n in array):raise ValueError('MLP FP32 bit inventory differs')
    for source,index in zip(single,(0,2,4)):
        values=struct.unpack(f'<{len(source)}f',struct.pack(f'<{len(source)}I',*source))
        if not all(map(math.isfinite,values)) or p.half_bits(values)!=half[index]:raise ValueError('MLP projection Half boundary differs')
    if half[3]!=[p.gating.half_product(a,b) for a,b in zip(half[1],half[2])]:raise ValueError('MLP gating must consume stored Half operands')
    qualifications=[]
    for bits,expected,name in zip(half,anchor['stages'],HALF_STAGES):
        values=struct.unpack(f'<{len(bits)}e',struct.pack(f'<{len(bits)}H',*bits))
        errors=[abs(a-b) for a,b in zip(values,expected['source'])];gate=expected['gate']
        fraction=max(error/(gate['atol']+gate['rtol']*abs(ref)) for error,ref in zip(errors,expected['source']))
        if fraction>1:raise ValueError('Frozen composed MLP source gate failed')
        qualifications.append({'stage':name,'source_fp32_max_gate_fraction':fraction,'source_fp32_max_abs_error':max(errors),
                               'equal_frozen_half_values':sum(a==b for a,b in zip(bits,expected['bits'])),
                               'output_values':len(bits),'output_sha256':hashlib.sha256(struct.pack(f'<{len(bits)}H',*bits)).hexdigest()})
    if len(r['rows'])!=13:raise ValueError('MLP repetition inventory differs')
    for i,row in enumerate(r['rows']):
        if (row['phase']!=('warmup' if i<3 else 'sample') or row['repetition']!=(i+1 if i<3 else i-2)
                or row['command_completed'] is not True or row['commit_wait_pairs']!=1
                or row['inputs_guards_and_intermediates_unchanged'] is not True or row['output_bits']!=half[4]
                or not math.isfinite(row['encode_commit_wait_seconds']) or row['encode_commit_wait_seconds']<=0):raise ValueError('MLP completed graph/repetition differs')
        row['output_sha256']=qualifications[-1]['output_sha256'];del row['output_bits']
    r['qualification']=qualifications
    r['anchors']={'input_sha256':anchor['input_sha256'],'weight_sha256':anchor['weight_sha256'],
                  'stages':[{k:v for k,v in stage.items() if k not in ('bits','source')} for stage in anchor['stages']]}
    del r['first_half_intermediates'];del r['first_fp32_projection_bits']


def overflow_rejected(event):
    if (event.get('passed') is not False or event.get('error',{}).get('stage')!='validation'
            or event['error']['message']!='completed MLP nonfinite; result discarded'):raise ValueError('Completed MLP overflow discard differs')
    return False


def run(output,build,directory):
    rms.probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+'.stderr.txt'),output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():raise ValueError('Refusing to overwrite MLP inputs')
    for path in (build,*evidence):
        if directory.resolve()==path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:raise ValueError('MLP inputs/build/evidence must be disjoint')
    for source in (rms.FP16,rms.FP32):
        if any(source.resolve()==path.resolve() or source.resolve() in path.resolve().parents for path in (build,directory,*evidence)):raise ValueError('MLP outputs must be outside frozen fixtures')
    inputs,anchors,overflow=cases();groups,stage_anchors=stages.batches();query,qa,projection_overflow=p.cases();tiny,ta,gating_overflow=p.gating.cases();tails,tla=p.gating.tail_cases()
    directory.mkdir(parents=True);jobs=[];identities={}
    for index,case in enumerate(inputs):
        path=directory/f'mlp_{index+1}.json'
        with path.open('x')as file:file.write(json.dumps(case,separators=(',',':'))+'\n')
        identities[str(path.resolve())]=rms.probe.common.digest(path)
        jobs.append((f'mlp_{index+1}',['--mlp-input',str(path.resolve())],lambda e,case=case:validate(e,case,anchors[case['id']])))
    path=directory/'mlp_overflow.json'
    with path.open('x')as file:file.write(json.dumps(overflow,separators=(',',':'))+'\n')
    identities[str(path.resolve())]=rms.probe.common.digest(path);jobs.append(('mlp_overflow',['--mlp-input',str(path.resolve())],overflow_rejected))
    payloads=[(label,cs,'--projection-input',lambda e,cs=cs:p.validate(e,cs,stage_anchors))for label,cs in groups]
    payloads += [('query_manual_regression',query,'--projection-input',lambda e:p.validate(e,query,qa)),
                 ('projection_overflow_regression',[projection_overflow],'--projection-input',p.overflow_rejected),
                 ('gating_regression',tiny,'--gating-input',lambda e:p.gating.validate(e,tiny,ta)),
                 ('gating_tails_regression',tails,'--gating-input',lambda e:p.gating.validate_tails(e,tails,tla)),
                 ('gating_overflow_regression',[gating_overflow],'--gating-input',p.gating.overflow_rejected)]
    for label,cs,option,validator in payloads:
        path=directory/f'{label}.json'
        with path.open('x')as file:file.write(json.dumps({'format':'toklet-metal-projection-input-v1'if option=='--projection-input'else'toklet-metal-gating-input-v1','cases':cs},separators=(',',':'))+'\n')
        identities[str(path.resolve())]=rms.probe.common.digest(path);jobs.append((label,[option,str(path.resolve())],validator))
    modules=(stages,p,p.gating,rms,p.gating.silu,p.gating.residual,p.gating.residual.embedding)
    drivers={str(Path(m.__file__).resolve()):rms.probe.common.digest(Path(m.__file__))for m in modules};drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        result=previous(event)
        if cases()!=(inputs,anchors,overflow):raise ValueError('MLP frozen inputs changed')
        for name,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(name))!=sha:raise ValueError('MLP input/driver identity changed')
        return result
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context={'input_sha256':identities,'qualification_driver_sha256':drivers,
        'fp16_manifest_sha256':rms.FP16_MANIFEST_SHA,'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,
        'scope':'10 composed tiny MLPs with GPU propagation across gate -> storedHalf -> SiLU -> storedHalf, up -> storedHalf, gating -> storedHalf -> down -> storedHalf, one command commit/wait. All five intermediate original source gates required. First complete intermediate frame retained; native host requires all13 identical. Full stage/input/guard byte readbacks not archived per repetition. Same-binary original projection/gating regressions. No residual, full layer/decoder/KV/Qwen, performance or M4 completion.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--build-dir',type=Path,required=True);parser.add_argument('--input-dir',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.build_dir,args.input_dir)

if __name__=='__main__':main()
