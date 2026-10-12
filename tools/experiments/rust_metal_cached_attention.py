"""Qualify nonzero tiny cached GQA prefixes, irregular chunks and stale tails."""
import argparse
import copy
import json
from pathlib import Path
import rust_metal_attention as attention
rms=attention.rms


def cases():
    full,full_anchors=attention.cases();inputs=[];anchors={}
    segments=[(0,1),(1,3),(4,7),(11,13),(24,8),(31,1)]
    for layer in range(2):
        parent=next(i for i in full if i['id']==f'cases.capacity.layers.{layer}.attention');source=full_anchors[parent['id']]
        for start,rows in segments:
            end=start+rows;case={**copy.deepcopy(parent),'id':f'capacity.layer{layer}.chunk{start}_{end}','rows':rows,'offset':start,'valid':end,
                              'q_bits':parent['q_bits'][start*32:end*32]}
            stages=[]
            for index,original in enumerate(source['stages']):
                if index<2:indices=[(head*32+token)*32+position for head in range(4) for token in range(start,end) for position in range(end)]
                else:indices=list(range(start*32,end*32))
                stages.append({**original,'bits':[original['bits'][i] for i in indices],'reference':[original['reference'][i] for i in indices]})
            anchor={**source,'stages':stages,'parent_id':parent['id'],'query_slice':[start,end],'source_columns':[0,end]}
            for stale in (False,True):
                variant=copy.deepcopy(case)
                if stale:
                    variant['id']+='.stale'
                    for key,base in [('k_bits',0x7bff),('v_bits',0xfbff)]:
                        for head in range(2):
                            for position in range(end,32):variant[key][(head*32+position)*8:(head*32+position+1)*8]=[base]*8
                inputs.append(variant);anchors[variant['id']]={**anchor,'stale_capacity_poisoned':stale}
    if len(inputs)!=24:raise ValueError('Cached attention inventory differs')
    return inputs,anchors


def rejected(event):
    if event.get('event')!='idle_result' or event.get('passed') is not False or event.get('error',{}).get('stage')!='fixture' or event['error'].get('message')!='attention prototype shape outside bounds':
        raise ValueError('Cached attention position rejection differs')
    return False


def run(output,build,directory):
    rms.probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+'.stderr.txt'),output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():raise ValueError('Refusing to overwrite cached attention inputs')
    for path in (build,*evidence):
        if directory.resolve()==path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:raise ValueError('Cached attention paths must be disjoint')
    if any(root.resolve()==path.resolve() or root.resolve() in path.resolve().parents for root in (rms.FP16,rms.FP32) for path in (build,directory,*evidence)):raise ValueError('Cached attention outputs must be outside fixtures')
    inputs,anchors=cases();full,fa=attention.cases();directory.mkdir(parents=True);jobs=[];identities={}
    payloads=[]
    for label,items,refs in [('cached',inputs,anchors),('full',full,fa)]:
        for i,item in enumerate(items):payloads.append((f'{label}_{i+1}',item,'--attention-input',lambda e,item=item,refs=refs:attention.validate(e,item,refs[item['id']])))
    invalid={**inputs[0],'offset':32,'valid':32};payloads.append(('invalid_cached_offset',invalid,'--attention-input',rejected))
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
    modules=(attention,attention.cache,attention.rotary,attention.rotary.attention,attention.rotary.composed,attention.rotary.composed.stages,p,p.gating,rms,p.gating.silu,p.gating.residual,p.gating.residual.embedding)
    drivers={str(Path(m.__file__).resolve()):rms.probe.common.digest(Path(m.__file__)) for m in modules};drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        result=previous(event)
        if cases()!=(inputs,anchors):raise ValueError('Cached attention frozen inputs changed')
        for name,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(name))!=sha:raise ValueError('Cached attention input/driver identity changed')
        return result
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context={'input_sha256':identities,'qualification_driver_sha256':drivers,'fp16_manifest_sha256':rms.FP16_MANIFEST_SHA,
        'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,'scope':'Tiny cached-prefix GQA, bounded offset+rows=valid<=32. Both capacityfixture layers at irregular chunks plus last-token decode, each repeated with unused K/V capacity replaced by finite maxpositive/negative Half. Unchanged parent scores/probabilities/context gates sliced by query/columns. Same-binary fullprefix/cache/rotary/MLPregressions. No cachewritepropagation/session/Qwen/performance/fullbackend/M4 closure.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
    a=p.parse_args();run(a.output,a.build_dir,a.input_dir)

if __name__=='__main__':main()
