"""Qualify GPU propagation through a tiny full-prefix layer."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import rust_metal_cached_attention as cached
import rust_metal_attention_graph as branch
attention=cached.attention
rms=cached.rms
EXTRA_HALF_STAGES=('input_norm','attention_residual','post_attention_norm','gate_projection','gate_silu','up_projection','gated','down_projection','output')
HALF_STAGES=('q_projection','k_projection','v_projection','q_rotary','k_rotary','cache_k','cache_v','attention_context','attention_projection')+EXTRA_HALF_STAGES


def cases():
    source_cases,source_anchors,_=branch.cases()
    half=rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA);single=rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
    if half['model']['rms_norm_eps']!=1.0e-6:raise ValueError('Layer epsilon differs')
    inputs=[];anchors={}
    for original in source_cases:
        case=copy.deepcopy(original);case['format']='toklet-metal-layer-input-v1';case['id']=original['id'].replace('.attention_branch','.layer')
        prefix=case['id'].removesuffix('.layer');layer=int(prefix.rsplit('.',1)[1]);name=prefix.split('.')[1];rows=case['rows']
        key=f'cases.{name}.embedding' if layer==0 else f'cases.{name}.layers.0.output'
        data,item=rms.payload(rms.FP16,half,key,'f16le')
        if item['shape']!=[rows,32]:raise ValueError('Layer source input shape differs')
        case['input_bits']=list(struct.unpack(f'<{rows*32}H',data));hashes={**source_anchors[original['id']]['input_sha256'],'input_bits':item['sha256']}
        for field,key,shape in [(f'{s}_weight_bits',f'weights.layers.{layer}.{s}_proj.weight',[32,64] if s=='down' else [64,32]) for s in ('gate','up','down')]+[('input_norm_weight_bits',f'weights.layers.{layer}.input_layernorm',[32]),('post_norm_weight_bits',f'weights.layers.{layer}.post_attention_layernorm',[32])]:
            data,item=rms.payload(rms.FP16,half,key,'f16le')
            if item['shape']!=shape:raise ValueError('Layer weight shape differs')
            case[field]=list(struct.unpack(f'<{math.prod(shape)}H',data));hashes[field]=item['sha256']
        stages=copy.deepcopy(source_anchors[original['id']]['stages'][:9])
        for stage in EXTRA_HALF_STAGES:
            width=64 if stage in ('gate_projection','gate_silu','up_projection','gated') else 32
            data,item=rms.payload(rms.FP16,half,f'{prefix}.{stage}','f16le');source,ref=rms.payload(rms.FP32,single,f'{prefix}.{stage}','f32le')
            if item['shape']!=[rows,width] or ref['shape']!=[rows,width]:raise ValueError('Layer stage shape differs')
            stages.append({'bits':list(struct.unpack(f'<{rows*width}H',data)),'source':list(struct.unpack(f'<{rows*width}f',source)),
                           'gate':item['gate_against_source_fp32'],'oracle_sha256':item['sha256'],'source_sha256':ref['sha256']})
        stages+=copy.deepcopy(source_anchors[original['id']]['stages'][9:])
        inputs.append(case);anchors[case['id']]={'stages':stages,'input_sha256':hashes}
    if len(inputs)!=10:raise ValueError('Layer inventory differs')
    # All incoming Half values are finite. A deliberately extreme normalization
    # weight makes the first stored normalization nonfinite, before later GPU
    # stages; publication must discard the completed layer result.
    overflow=copy.deepcopy(inputs[-1]);overflow['id']='manual.overflow';overflow['rows']=1;overflow['position']=0
    overflow['input_bits']=[0x3c00]+[0]*31;overflow['input_norm_weight_bits']=[0x7bff]+[0]*31
    return inputs,anchors,overflow


def validate(event,item,anchor):
    if event.get('event')!='idle_result' or event.get('passed') is not True:raise ValueError('Native layer failed')
    r=event['result'];m=r['metadata'];rows=item['rows'];width=r['threadgroup_width']
    if (r['operation']!='resident_tiny_layer_half_boundaries' or r['id']!=item['id'] or r['rows_count']!=rows or r['position']!=item['position']
            or m['mps_device_supported'] is not True or m['unified_memory'] is not True or m['language_version']!='2.3' or m['math_mode']!='safe'
            or m['math_functions']!='precise' or m['minimum_macos_major']!=15 or r['allocations_reused'] is not True):raise ValueError('Layer policy/topology differs')
    caps=m['graph_pipeline_caps']
    if [c['name'] for c in caps]!=['toklet_projection_bias','toklet_rotary','toklet_cache_write','toklet_attention','toklet_rms_norm','toklet_residual','toklet_silu','toklet_gating']:raise ValueError('Layer pipeline inventory differs')
    if not 0<width<=min([m['execution_width'],m['pipeline_max_threads'],m['device_max_threads'][0]]+[v for c in caps for v in (c['execution_width'],c['max_threads'])]):raise ValueError('Layer pipeline bounds differ')
    il,hl,fl=r['input_layouts'],r['half_layouts'],r['float_layouts'];units=math.ceil(rows*4/width)*width;qcap=max(math.ceil(units*8/32),math.ceil(rows/width)*width);wcap=math.ceil(math.ceil(rows*64/width)*width/64);kcap=math.ceil(math.ceil(rows*8/width)*width*2/16)
    descriptors=[(rows,32,2,rows),(32,32,2,32),(16,32,2,16),(16,32,2,16),(32,32,2,32),(1,32,2,1),(1,16,2,1),(1,16,2,1),(1,64,2,1),(1,32,2,1),(1,32,2,1),(64,32,2,64),(64,32,2,64),(32,64,2,32)]
    half_desc=[(rows,32,2,qcap),(rows,16,2,kcap),(rows,16,2,kcap),(rows,32,2,qcap),(rows,16,2,kcap),(2,256,2,2),(2,256,2,2),(rows,32,2,qcap),(rows,32,2,qcap)]+[(rows,32,2,qcap)]*3+[(rows,64,2,wcap)]*4+[(rows,32,2,qcap)]*2
    float_desc=[(rows,32,4,qcap),(rows,16,4,kcap),(rows,16,4,kcap),(rows*4,rows,4,units),(rows*4,rows,4,units),(rows,32,4,qcap)]+[(rows,64,4,wcap)]*2+[(rows,32,4,qcap)]
    if [len(il),len(hl),len(fl)]!=[14,18,9]:raise ValueError('Layer layout inventory differs')
    for layouts,descs in [(il,descriptors),(hl,half_desc),(fl,float_desc)]:
        for l,(lr,cols,scalar,capacity) in zip(layouts,descs):
            if (l['rows']!=lr or l['columns']!=cols or l['stride']<cols or l['elements']!=capacity*l['stride']+7 or l['bytes']!=l['elements']*scalar):raise ValueError('Layer guarded layout differs')
    if il[0]['stride']!=32 or any(hl[i]['stride']!=n for i,n in [(0,32),(1,16),(2,16),(3,32),(4,16),(7,32),(8,32),(9,32),(10,32),(11,32),(12,64),(13,64),(14,64),(15,64),(16,32),(17,32)]):raise ValueError('Layer contiguous Half ABI differs')
    shared=sum(l['bytes'] for layouts in (il,hl,fl) for l in layouts);total=shared*3+rms.probe.ALLOWANCE
    if r['shared_buffer_bytes']!=shared or r['host_snapshot_readback_bytes']!=shared*2 or r['admitted_bytes']!=total or m['maximum_admitted_bytes']!=total or total>rms.probe.LIMIT:raise ValueError('Layer admission differs')
    halves=r['first_half_intermediates'];floats=r['first_fp32_intermediates'];counts=[rows*32,rows*16,rows*16,rows*32,rows*16,512,512,rows*32,rows*32]+[rows*32]*3+[rows*64]*4+[rows*32]*2
    if len(halves)!=18 or len(floats)!=9 or any(len(a)!=n for a,n in zip(halves,counts)) or any(len(a)!=n for a,n in zip(floats,[rows*32,rows*16,rows*16,4*rows*rows,4*rows*rows,rows*32,rows*64,rows*64,rows*32])):raise ValueError('Layer intermediate inventory differs')
    if any(type(n) is not int or not 0<=n<=65535 for array in halves for n in array) or any(type(n) is not int or not 0<=n<2**32 for array in floats for n in array):raise ValueError('Layer intermediate bit types differ')
    for source,target in zip((floats[0],floats[1],floats[2],floats[5],floats[6],floats[7],floats[8]),(halves[0],halves[1],halves[2],halves[8],halves[12],halves[14],halves[16])):
        values=struct.unpack(f'<{len(source)}f',struct.pack(f'<{len(source)}I',*source))
        if not all(map(math.isfinite,values)) or attention.rotary.composed.p.half_bits(values)!=target:raise ValueError('Layer projection Half boundary differs')
    for source,chunk in [(halves[5],halves[4]),(halves[6],halves[2])]:
        expected=[0]*512
        for head in range(2):
            for token in range(rows):expected[(head*32+token)*8:(head*32+token+1)*8]=chunk[(token*2+head)*8:(token*2+head+1)*8]
        if source!=expected:raise ValueError('Layer cache must consume actual GPU K/V')
    product=attention.rotary.composed.p.gating.half_product
    if halves[15]!=[product(a,b) for a,b in zip(halves[13],halves[14])]:raise ValueError('Layer gating must consume actual stored Half operands')
    def residual(a,b):
        av=struct.unpack(f'<{len(a)}e',struct.pack(f'<{len(a)}H',*a));bv=struct.unpack(f'<{len(b)}e',struct.pack(f'<{len(b)}H',*b))
        return list(struct.unpack(f'<{len(a)}H',struct.pack(f'<{len(a)}e',*(x+y for x,y in zip(av,bv)))))
    if halves[10]!=residual(item['input_bits'],halves[8]) or halves[17]!=residual(halves[10],halves[16]):raise ValueError('Layer residual must consume actual GPU state/delta')
    arrays=halves+[floats[3],floats[4]];qualifications=[]
    for index,(bits,expected) in enumerate(zip(arrays,anchor['stages'])):
        count=len(bits);data=struct.pack(f'<{count}{"H" if index<18 else "I"}',*bits);values=struct.unpack(f'<{count}{"e" if index<18 else "f"}',data)
        if not all(map(math.isfinite,values)):raise ValueError('Layer nonfinite intermediate')
        errors=[abs(a-b) for a,b in zip(values,expected['source'])];gate=expected['gate'];fraction=max(e/(gate['atol']+gate['rtol']*abs(b)) for e,b in zip(errors,expected['source'])) if gate else 0.0
        if fraction>1:raise ValueError('Frozen propagated layer source gate failed')
        qualifications.append({'stage':(HALF_STAGES+('attention_scores','attention_probabilities'))[index],'values':count,'equal_frozen_bits':sum(a==b for a,b in zip(bits,expected['bits'])),
                               'source_fp32_max_gate_fraction':fraction,'reference_max_abs_error':max(errors),'original_source_gate_required':gate is not None,'output_sha256':hashlib.sha256(data).hexdigest()})
    for head in range(4):
        for token in range(rows):
            start=(head*rows+token)*rows
            if any(floats[4][start+position]!=0 for position in range(token+1,rows)):raise ValueError('Layer causal mask differs')
    if len(r['rows'])!=13:raise ValueError('Layer repetition inventory differs')
    for i,row in enumerate(r['rows']):
        if row['phase']!=('warmup' if i<3 else 'sample') or row['repetition']!=(i+1 if i<3 else i-2) or row['command_completed'] is not True or row['commit_wait_pairs']!=1 or row['inputs_guards_and_intermediates_unchanged'] is not True or row['output_bits']!=halves[17] or not math.isfinite(row['encode_commit_wait_seconds']) or row['encode_commit_wait_seconds']<=0:raise ValueError('Layer completed/repeated propagation differs')
        row['output_sha256']=qualifications[17]['output_sha256'];del row['output_bits']
    r['qualification']=qualifications;r['anchors']={**anchor,'stages':[{k:v for k,v in s.items() if k not in ('bits','source')} for s in anchor['stages']]}
    del r['first_half_intermediates'];del r['first_fp32_intermediates']


def overflow_rejected(event):
    if event.get('event')!='idle_result' or event.get('passed') is not False or event.get('error',{}).get('stage')!='validation' or event['error'].get('message')!='completed layer nonfinite; result discarded':raise ValueError('Layer nonfinite failure differs')
    return False


def run(output,build,directory):
    rms.probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+'.stderr.txt'),output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():raise ValueError('Refusing to overwrite cached attention inputs')
    for path in (build,*evidence):
        if directory.resolve()==path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:raise ValueError('Cached attention paths must be disjoint')
    if any(root.resolve()==path.resolve() or root.resolve() in path.resolve().parents for root in (rms.FP16,rms.FP32) for path in (build,directory,*evidence)):raise ValueError('Cached attention outputs must be outside fixtures')
    inputs,anchors,overflow=cases();cache_queries,cq_anchors=cached.cases();full,fa=attention.cases();directory.mkdir(parents=True);jobs=[];identities={}
    payloads=[]
    for i,item in enumerate(inputs):payloads.append((f'layer_{i+1}',item,'--layer-input',lambda e,item=item:validate(e,item,anchors[item['id']])))
    payloads.append(('layer_overflow',overflow,'--layer-input',overflow_rejected))
    for label,items,refs in [('cached',cache_queries,cq_anchors),('full',full,fa)]:
        for i,item in enumerate(items):payloads.append((f'{label}_{i+1}',item,'--attention-input',lambda e,item=item,refs=refs:attention.validate(e,item,refs[item['id']])))
    invalid={**cache_queries[0],'offset':32,'valid':32};payloads.append(('invalid_cached_offset',invalid,'--attention-input',cached.rejected))
    ci,ca=attention.cache.cases();ri,ra,ro=attention.rotary.cases();mi,ma,mo=attention.rotary.composed.cases()
    payloads.extend([('cache_regression',{'format':'toklet-metal-cache-input-v1','cases':ci},'--cache-input',lambda e:attention.cache.validate(e,ci,ca)),
                     ('rotary_regression',{'format':'toklet-metal-rotary-input-v1','cases':ri},'--rotary-input',lambda e:attention.rotary.validate(e,ri,ra)),
                     ('rotary_overflow',{'format':'toklet-metal-rotary-input-v1','cases':[ro]},'--rotary-input',attention.rotary.overflow_rejected)])
    for i,item in enumerate(mi):payloads.append((f'mlp_{i+1}',item,'--mlp-input',lambda e,item=item:attention.rotary.composed.validate(e,item,ma[item['id']])))
    payloads.append(('mlp_overflow',mo,'--mlp-input',attention.rotary.composed.overflow_rejected))
    for label,payload,option,validator in payloads:
        path=directory/f'{label}.json'
        with path.open('x')as stream:stream.write(json.dumps(payload,separators=(',',':'))+'\n')
        identities[str(path.resolve())]=rms.probe.common.digest(path);jobs.append((label,[option,str(path.resolve())],validator))
    p=attention.rotary.composed.p
    modules=(branch,cached,attention,attention.cache,attention.rotary,attention.rotary.attention,attention.rotary.composed,attention.rotary.composed.stages,p,p.gating,rms,p.gating.silu,p.gating.residual,p.gating.residual.embedding)
    drivers={str(Path(m.__file__).resolve()):rms.probe.common.digest(Path(m.__file__)) for m in modules};drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        result=previous(event)
        if cases()!=(inputs,anchors,overflow):raise ValueError('Cached attention frozen inputs changed')
        for name,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(name))!=sha:raise ValueError('Cached attention input/driver identity changed')
        return result
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context={'input_sha256':identities,'qualification_driver_sha256':drivers,'fp16_manifest_sha256':rms.FP16_MANIFEST_SHA,
        'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,'scope':'10 complete tiny GPU layers from saved layer input through normalization/QKV/rotary/cache/GQA/O/residual/postnorm/MLP/finalresidual. One command/wait plus two bounded GPUblits preserving residual diagnostics. Original18sourcegates plus exactactualGPUcache/gating/residual/cast boundaries. First18Half/9F32 arrays/native12laterframesexact/allfinallayeroutputs retained. Layers use saved inputs, not two-layer decoder propagation/session/Qwen/quality/performance/M4 closure.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
    a=p.parse_args();run(a.output,a.build_dir,a.input_dir)

if __name__=='__main__':main()
