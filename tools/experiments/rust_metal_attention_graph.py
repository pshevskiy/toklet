"""Qualify GPU propagation through a tiny full-prefix attention branch."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import rust_metal_cached_attention as cached
attention=cached.attention
rms=cached.rms
HALF_STAGES=('q_projection','k_projection','v_projection','q_rotary','k_rotary','cache_k','cache_v','attention_context','attention_projection')


def cases():
    half=rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA);single=rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA);inputs=[];anchors={}
    for name,spec in sorted(half['cases'].items()):
        rows=len(spec['tokens'])
        for layer in range(2):
            prefix=f'cases.{name}.layers.{layer}';identity=f'{prefix}.attention_branch';case={'format':'toklet-metal-attention-graph-input-v1','id':identity,'rows':rows,'position':spec['position_offset']};hashes={}
            for field,key,shape in [('input_bits',f'{prefix}.input_norm',[rows,32])]+[(f'{stage}_weight_bits',f'weights.layers.{layer}.{stage}_proj.weight',[32 if stage in ('q','o') else 16,32]) for stage in ('q','k','v','o')]+[(f'{stage}_bias_bits',f'weights.layers.{layer}.{stage}_proj.bias',[32 if stage=='q' else 16]) for stage in ('q','k','v')]:
                data,item=rms.payload(rms.FP16,half,key,'f16le')
                if item['shape']!=shape:raise ValueError('Attention branch input shape differs')
                case[field]=list(struct.unpack(f'<{math.prod(shape)}H',data));hashes[field]=item['sha256']
            stages=[]
            shapes=[[rows,4,8],[rows,2,8],[rows,2,8],[rows,4,8],[rows,2,8],[2,32,8],[2,32,8],[rows,32],[rows,32],[4,rows,rows],[4,rows,rows]]
            for index,(stage,shape) in enumerate(zip(HALF_STAGES+('attention_scores','attention_probabilities'),shapes)):
                dtype='f16le' if index<9 else 'f32le';data,item=rms.payload(rms.FP16,half,f'{prefix}.{stage}',dtype)
                if stage in ('cache_k','cache_v'):
                    source=struct.pack('<512f',*struct.unpack('<512e',data));ref={'shape':shape,'sha256':None}
                else:source,ref=rms.payload(rms.FP32,single,f'{prefix}.{stage}','f32le')
                if item['shape']!=shape or ref['shape']!=shape:raise ValueError('Attention branch reference shape differs')
                count=math.prod(shape);stages.append({'bits':list(struct.unpack(f'<{count}{"H" if index<9 else "I"}',data)),
                    'source':list(struct.unpack(f'<{count}f',source)),'gate':item['gate_against_source_fp32'],'oracle_sha256':item['sha256'],'source_sha256':ref['sha256']})
            inputs.append(case);anchors[identity]={'stages':stages,'input_sha256':hashes}
    if len(inputs)!=10:raise ValueError('Attention branch inventory differs')
    overflow=copy.deepcopy(inputs[-1]);overflow['id']='manual.overflow';overflow['rows']=1;overflow['position']=0;overflow['input_bits']=[0x4000]+[0]*31;overflow['q_weight_bits']=[0x7bff]+[0]*1023
    return inputs,anchors,overflow


def validate(event,item,anchor):
    if event.get('event')!='idle_result' or event.get('passed') is not True:raise ValueError('Native attention branch failed')
    r=event['result'];m=r['metadata'];rows=item['rows'];width=r['threadgroup_width']
    if (r['operation']!='resident_tiny_attention_branch_half_boundaries' or r['id']!=item['id'] or r['rows_count']!=rows or r['position']!=item['position']
            or m['mps_device_supported'] is not True or m['unified_memory'] is not True or m['language_version']!='2.3' or m['math_mode']!='safe'
            or m['math_functions']!='precise' or m['minimum_macos_major']!=15 or r['allocations_reused'] is not True):raise ValueError('Attention branch policy/topology differs')
    caps=m['graph_pipeline_caps']
    if [c['name'] for c in caps]!=['toklet_projection_bias','toklet_rotary','toklet_cache_write','toklet_attention']:raise ValueError('Attention branch pipeline inventory differs')
    if not 0<width<=min([m['execution_width'],m['pipeline_max_threads'],m['device_max_threads'][0]]+[v for c in caps for v in (c['execution_width'],c['max_threads'])]):raise ValueError('Attention branch pipeline bounds differ')
    il,hl,fl=r['input_layouts'],r['half_layouts'],r['float_layouts'];units=math.ceil(rows*4/width)*width;qcap=math.ceil(units*8/32);kcap=math.ceil(math.ceil(rows*8/width)*width*2/16)
    descriptors=[(rows,32,2,rows),(32,32,2,32),(16,32,2,16),(16,32,2,16),(32,32,2,32),(1,32,2,1),(1,16,2,1),(1,16,2,1),(1,32,2,1)]
    half_desc=[(rows,32,2,qcap),(rows,16,2,kcap),(rows,16,2,kcap),(rows,32,2,qcap),(rows,16,2,kcap),(2,256,2,2),(2,256,2,2),(rows,32,2,qcap),(rows,32,2,qcap)]
    float_desc=[(rows,32,4,qcap),(rows,16,4,kcap),(rows,16,4,kcap),(rows*4,rows,4,units),(rows*4,rows,4,units),(rows,32,4,qcap)]
    if [len(il),len(hl),len(fl)]!=[9,9,6]:raise ValueError('Attention branch layout inventory differs')
    for layouts,descs in [(il,descriptors),(hl,half_desc),(fl,float_desc)]:
        for l,(lr,cols,scalar,capacity) in zip(layouts,descs):
            if (l['rows']!=lr or l['columns']!=cols or l['stride']<cols or l['elements']!=capacity*l['stride']+7 or l['bytes']!=l['elements']*scalar):raise ValueError('Attention branch guarded layout differs')
    if il[0]['stride']!=32 or any(hl[i]['stride']!=n for i,n in [(0,32),(1,16),(2,16),(3,32),(4,16),(7,32),(8,32)]):raise ValueError('Attention branch contiguous Half ABI differs')
    shared=sum(l['bytes'] for layouts in (il,hl,fl) for l in layouts);total=shared*3+rms.probe.ALLOWANCE
    if r['shared_buffer_bytes']!=shared or r['host_snapshot_readback_bytes']!=shared*2 or r['admitted_bytes']!=total or m['maximum_admitted_bytes']!=total or total>rms.probe.LIMIT:raise ValueError('Attention branch admission differs')
    halves=r['first_half_intermediates'];floats=r['first_fp32_intermediates'];counts=[rows*32,rows*16,rows*16,rows*32,rows*16,512,512,rows*32,rows*32]
    if len(halves)!=9 or len(floats)!=6 or any(len(a)!=n for a,n in zip(halves,counts)) or any(len(a)!=n for a,n in zip(floats,[rows*32,rows*16,rows*16,4*rows*rows,4*rows*rows,rows*32])):raise ValueError('Attention branch intermediate inventory differs')
    if any(type(n) is not int or not 0<=n<=65535 for array in halves for n in array) or any(type(n) is not int or not 0<=n<2**32 for array in floats for n in array):raise ValueError('Attention branch intermediate bit types differ')
    for source,target in zip((floats[0],floats[1],floats[2],floats[5]),(halves[0],halves[1],halves[2],halves[8])):
        values=struct.unpack(f'<{len(source)}f',struct.pack(f'<{len(source)}I',*source))
        if not all(map(math.isfinite,values)) or attention.rotary.composed.p.half_bits(values)!=target:raise ValueError('Attention branch projection Half boundary differs')
    for source,chunk in [(halves[5],halves[4]),(halves[6],halves[2])]:
        expected=[0]*512
        for head in range(2):
            for token in range(rows):expected[(head*32+token)*8:(head*32+token+1)*8]=chunk[(token*2+head)*8:(token*2+head+1)*8]
        if source!=expected:raise ValueError('Attention branch cache must consume actual GPU K/V')
    arrays=halves+[floats[3],floats[4]];qualifications=[]
    for index,(bits,expected) in enumerate(zip(arrays,anchor['stages'])):
        count=len(bits);data=struct.pack(f'<{count}{"H" if index<9 else "I"}',*bits);values=struct.unpack(f'<{count}{"e" if index<9 else "f"}',data)
        if not all(map(math.isfinite,values)):raise ValueError('Attention branch nonfinite intermediate')
        errors=[abs(a-b) for a,b in zip(values,expected['source'])];gate=expected['gate'];fraction=max(e/(gate['atol']+gate['rtol']*abs(b)) for e,b in zip(errors,expected['source'])) if gate else 0.0
        if fraction>1:raise ValueError('Frozen propagated attention branch source gate failed')
        qualifications.append({'stage':(HALF_STAGES+('attention_scores','attention_probabilities'))[index],'values':count,'equal_frozen_bits':sum(a==b for a,b in zip(bits,expected['bits'])),
                               'source_fp32_max_gate_fraction':fraction,'reference_max_abs_error':max(errors),'original_source_gate_required':gate is not None,'output_sha256':hashlib.sha256(data).hexdigest()})
    for head in range(4):
        for token in range(rows):
            start=(head*rows+token)*rows
            if any(floats[4][start+position]!=0 for position in range(token+1,rows)):raise ValueError('Attention branch causal mask differs')
    if len(r['rows'])!=13:raise ValueError('Attention branch repetition inventory differs')
    for i,row in enumerate(r['rows']):
        if row['phase']!=('warmup' if i<3 else 'sample') or row['repetition']!=(i+1 if i<3 else i-2) or row['command_completed'] is not True or row['commit_wait_pairs']!=1 or row['inputs_guards_and_intermediates_unchanged'] is not True or row['output_bits']!=halves[8] or not math.isfinite(row['encode_commit_wait_seconds']) or row['encode_commit_wait_seconds']<=0:raise ValueError('Attention branch completed/repeated propagation differs')
        row['output_sha256']=qualifications[8]['output_sha256'];del row['output_bits']
    r['qualification']=qualifications;r['anchors']={**anchor,'stages':[{k:v for k,v in s.items() if k not in ('bits','source')} for s in anchor['stages']]}
    del r['first_half_intermediates'];del r['first_fp32_intermediates']


def overflow_rejected(event):
    if event.get('event')!='idle_result' or event.get('passed') is not False or event.get('error',{}).get('stage')!='validation' or event['error'].get('message')!='completed attention branch nonfinite; result discarded':raise ValueError('Attention branch nonfinite failure differs')
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
    for i,item in enumerate(inputs):payloads.append((f'branch_{i+1}',item,'--attention-graph-input',lambda e,item=item:validate(e,item,anchors[item['id']])))
    payloads.append(('branch_overflow',overflow,'--attention-graph-input',overflow_rejected))
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
    modules=(cached,attention,attention.cache,attention.rotary,attention.rotary.attention,attention.rotary.composed,attention.rotary.composed.stages,p,p.gating,rms,p.gating.silu,p.gating.residual,p.gating.residual.embedding)
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
        'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,'scope':'10 propagated tiny attention branches: actual Q/K/V MPS projection Half -> Q/K rotary Half -> rawKVwrite -> causalGQA FP32softmax/Halfcontext -> Oprojection Half. One command/wait. Saved normalized input only; all original9sourcegates plus exact actualGPU cachetranspose, first9Half+6F32 arrays retained/native12laterframesexact; allfinalOoutputs retained. No RMS/residual/MLP/fullLayer/session/Qwen/quality/performance/M4 closure.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
    a=p.parse_args();run(a.output,a.build_dir,a.input_dir)

if __name__=='__main__':main()
