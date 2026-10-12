"""Qualify tiny full-prefix causal GQA with original intermediate gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import rust_metal_cache as cache
rotary=cache.rotary
rms=cache.rms
HALF_GUARD,FLOAT_GUARD=0x55a1,0x4642fedc


def cases():
    half=rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA);single=rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
    inputs=[];anchors={}
    for name,spec in sorted(half['cases'].items()):
        rows=len(spec['tokens'])
        for layer in range(2):
            prefix=f'cases.{name}.layers.{layer}';identity=f'{prefix}.attention';payload={};hashes={}
            for key,stage,shape in [('q','q_rotary',[rows,4,8]),('k','cache_k',[2,32,8]),('v','cache_v',[2,32,8])]:
                data,item=rms.payload(rms.FP16,half,f'{prefix}.{stage}','f16le')
                if item['shape']!=shape:raise ValueError('Attention input shape differs')
                payload[f'{key}_bits']=list(struct.unpack(f'<{math.prod(shape)}H',data));hashes[key]=item['sha256']
            stages=[]
            for key,shape,dtype in [('attention_scores',[4,rows,rows],'f32le'),('attention_probabilities',[4,rows,rows],'f32le'),('attention_context',[rows,32],'f16le')]:
                data,item=rms.payload(rms.FP16,half,f'{prefix}.{key}',dtype);source,ref=rms.payload(rms.FP32,single,f'{prefix}.{key}','f32le')
                if item['shape']!=shape or ref['shape']!=shape:raise ValueError('Attention reference shape differs')
                count=math.prod(shape);bits=list(struct.unpack(f'<{count}{"I" if dtype=="f32le" else "H"}',data))
                stages.append({'bits':bits,'reference':list(struct.unpack(f'<{count}f',source)),'gate':item['gate_against_source_fp32'],
                               'oracle_sha256':item['sha256'],'source_sha256':ref['sha256']})
            inputs.append({'format':'toklet-metal-attention-input-v1','id':identity,'rows':rows,'query_heads':4,'kv_heads':2,'dimension':8,'capacity':32,**payload})
            anchors[identity]={'stages':stages,'input_sha256':hashes,'rotary_position_offset':spec['position_offset']}
    if len(inputs)!=10:raise ValueError('Attention complete inventory differs')
    return inputs,anchors


def validate(event,item,anchor):
    if event.get('event')!='idle_result' or event.get('passed') is not True:raise ValueError('Native attention failed')
    r=event['result'];m=r['metadata'];rows=item['rows'];width=r['threadgroup_width'];units=math.ceil(rows*4/width)*width
    h=units*8+7;f=units*rows+7;shared=h*4+519*4+f*8
    if (r['operation']!='tiny_causal_gqa_fp32_half_context' or r['id']!=item['id'] or r['rows_count']!=rows or r['query_heads']!=4
            or r['kv_heads']!=2 or r['dimension']!=8 or r['capacity']!=32 or m['unified_memory'] is not True
            or m['language_version']!='2.3' or m['math_mode']!='safe' or m['math_functions']!='precise' or m['minimum_macos_major']!=15
            or not 0<width<=min(m['execution_width'],m['pipeline_max_threads'],m['device_max_threads'][0])
            or r['threadgroups']!=math.ceil(rows*4/width) or r['half_storage_elements']!=h or r['fp32_storage_elements']!=f
            or r['half_guard_bits']!=HALF_GUARD or r['fp32_guard_bits']!=FLOAT_GUARD or r['allocations_reused'] is not True
            or r['shared_buffer_bytes']!=shared or r['host_baseline_frames_readback_bytes']!=shared*2 or r['admitted_bytes']!=shared*3+rms.probe.ALLOWANCE
            or m['maximum_admitted_bytes']!=r['admitted_bytes'] or r['admitted_bytes']>rms.probe.LIMIT):raise ValueError('Attention policy/topology/memory differs')
    arrays=[r['first_score_bits'],r['first_probability_bits'],r['first_context_bits']];counts=[4*rows*rows]*2+[rows*32]
    qualifications=[]
    for index,(bits,count,size,guard,expected) in enumerate(zip(arrays,counts,(f,f,h),(FLOAT_GUARD,FLOAT_GUARD,HALF_GUARD),anchor['stages'])):
        if len(bits)!=size or bits[count:]!=[guard]*(size-count) or any(type(n) is not int or not 0<=n<(2**32 if index<2 else 2**16) for n in bits):raise ValueError('Attention full intermediate guard/type differs')
        data=struct.pack(f'<{count}{"I" if index<2 else "H"}',*bits[:count]);values=struct.unpack(f'<{count}{"f" if index<2 else "e"}',data)
        if not all(map(math.isfinite,values)):raise ValueError('Attention nonfinite completed result')
        gate=expected['gate'];errors=[abs(a-b) for a,b in zip(values,expected['reference'])]
        fraction=max(e/(gate['atol']+gate['rtol']*abs(b)) for e,b in zip(errors,expected['reference']))
        if fraction>1:raise ValueError('Frozen attention source gate failed')
        qualifications.append({'stage':('scores','probabilities','context')[index],'values':count,'guard_values':size-count,'source_fp32_max_gate_fraction':fraction,
                               'source_fp32_max_abs_error':max(errors),'equal_frozen_bits':sum(a==b for a,b in zip(bits[:count],expected['bits'])),
                               'output_sha256':hashlib.sha256(data).hexdigest()})
    probabilities=struct.unpack(f'<{counts[1]}f',struct.pack(f'<{counts[1]}I',*arrays[1][:counts[1]]))
    for head in range(4):
        for token in range(rows):
            start=(head*rows+token)*rows;valid=probabilities[start:start+token+1]
            if any(value<0 for value in valid) or any(arrays[1][start+position]!=0 for position in range(token+1,rows)):
                raise ValueError('Attention causal probabilities differ')
    if len(r['rows'])!=13:raise ValueError('Attention repetition inventory differs')
    for i,row in enumerate(r['rows']):
        if (row['phase']!=('warmup' if i<3 else 'sample') or row['repetition']!=(i+1 if i<3 else i-2) or row['command_completed'] is not True
                or row['inputs_guards_intermediates_valid'] is not True or row['commit_wait_pairs']!=1 or row['context_bits']!=arrays[2][:counts[2]]
                or not math.isfinite(row['encode_commit_wait_seconds']) or row['encode_commit_wait_seconds']<=0):raise ValueError('Attention completed/repeated frame differs')
        row['context_sha256']=qualifications[2]['output_sha256'];del row['context_bits']
    r['qualification']=qualifications;r['anchors']={**anchor,'stages':[{k:v for k,v in s.items() if k not in ('bits','reference')} for s in anchor['stages']]}
    del r['first_score_bits'];del r['first_probability_bits'];del r['first_context_bits']


def run(output,build,directory):
    rms.probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+'.stderr.txt'),output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():raise ValueError('Refusing to overwrite attention inputs')
    for path in (build,*evidence):
        if directory.resolve()==path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:raise ValueError('Attention paths must be disjoint')
    if any(root.resolve()==path.resolve() or root.resolve() in path.resolve().parents for root in (rms.FP16,rms.FP32) for path in (build,directory,*evidence)):raise ValueError('Attention outputs must be outside fixtures')
    inputs,anchors=cases();directory.mkdir(parents=True);jobs=[];identities={}
    for i,item in enumerate(inputs):
        path=directory/f'attention_{i+1}.json'
        with path.open('x')as stream:stream.write(json.dumps(item,separators=(',',':'))+'\n')
        identities[str(path.resolve())]=rms.probe.common.digest(path);jobs.append((f'attention_{i+1}',['--attention-input',str(path.resolve())],lambda e,item=item:validate(e,item,anchors[item['id']])))
    # Lossless layout, rotary, and full MLP regressions cover the boundaries
    # consumed here; detailed projection tests remain in their accepted tasks.
    ci,ca=cache.cases();ri,ra,ro=rotary.cases();mi,ma,mo=rotary.composed.cases()
    payloads=[('cache_regression',{'format':'toklet-metal-cache-input-v1','cases':ci},'--cache-input',lambda e:cache.validate(e,ci,ca)),
              ('rotary_regression',{'format':'toklet-metal-rotary-input-v1','cases':ri},'--rotary-input',lambda e:rotary.validate(e,ri,ra)),
              ('rotary_overflow',{'format':'toklet-metal-rotary-input-v1','cases':[ro]},'--rotary-input',rotary.overflow_rejected)]
    for i,item in enumerate(mi):payloads.append((f'mlp_{i+1}',item,'--mlp-input',lambda e,item=item:rotary.composed.validate(e,item,ma[item['id']])))
    payloads.append(('mlp_overflow',mo,'--mlp-input',rotary.composed.overflow_rejected))
    for label,payload,option,validator in payloads:
        path=directory/f'{label}.json'
        with path.open('x')as stream:stream.write(json.dumps(payload,separators=(',',':'))+'\n')
        identities[str(path.resolve())]=rms.probe.common.digest(path);jobs.append((label,[option,str(path.resolve())],validator))
    modules=(cache,rotary,rotary.attention,rotary.composed,rotary.composed.stages,rotary.composed.p,rotary.composed.p.gating,rms,rotary.composed.p.gating.silu,rotary.composed.p.gating.residual,rotary.composed.p.gating.residual.embedding)
    drivers={str(Path(m.__file__).resolve()):rms.probe.common.digest(Path(m.__file__)) for m in modules};drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        result=previous(event)
        if cases()!=(inputs,anchors):raise ValueError('Attention frozen inputs changed')
        for name,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(name))!=sha:raise ValueError('Attention input/driver identity changed')
        return result
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context={'input_sha256':identities,'qualification_driver_sha256':drivers,'fp16_manifest_sha256':rms.FP16_MANIFEST_SHA,
        'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,'scope':'10 saved-input tiny full-prefix causal GQA jobs. Stable orderedFP32 scores/softmax/context then Half context, head mapping4->2 with head-major cap32. Original3intermediate gates; firstfullframe+guards retained/native12subsequentframesexact; everyfinalcontext retained. No nonzero cached prefix/propagation/session/Qwen/quality/performance/M4 closure.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
    a=p.parse_args();run(a.output,a.build_dir,a.input_dir)

if __name__=='__main__':main()
