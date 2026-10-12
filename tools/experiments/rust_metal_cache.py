"""Qualify lossless tiny token-major to head-major K/V writes with saved state."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import rust_metal_rotary as rotary
rms=rotary.rms
K_GUARD,V_GUARD=0x55a1,0x529b


def expected(item):
    k,v=item['initial_k_bits'].copy(),item['initial_v_bits'].copy()
    for token in range(item['rows']):
        for head in range(2):
            source=(token*2+head)*8;target=(head*32+item['offset']+token)*8
            k[target:target+8]=item['k_bits'][source:source+8];v[target:target+8]=item['v_bits'][source:source+8]
    return k,v


def cases():
    manifest=rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA);inputs=[];anchors={}
    if (manifest['model']['capacity'],manifest['model']['head_dim'],manifest['model']['num_kv_heads'])!=(32,8,2):raise ValueError('Frozen cache topology differs')
    for name,spec in sorted(manifest['cases'].items()):
        rows=len(spec['tokens'])
        for layer in range(2):
            prefix=f'cases.{name}.layers.{layer}';identity=f'{prefix}.cache_write';bits={};hashes={}
            for stage,key,shape in [('k',f'{prefix}.k_rotary',[rows,2,8]),('v',f'{prefix}.v_projection',[rows,2,8]),
                                    ('cache_k',f'{prefix}.cache_k',[2,32,8]),('cache_v',f'{prefix}.cache_v',[2,32,8])]:
                data,item=rms.payload(rms.FP16,manifest,key,'f16le')
                if item['shape']!=shape:raise ValueError('Frozen cache shape differs')
                bits[stage]=list(struct.unpack(f'<{math.prod(shape)}H',data));hashes[stage]=item['sha256']
            # Frozen cache arrays store each local fixture at capacity offset0,
            # including the rotary offset5 fixture. Rotary position and placement
            # are separate; no invented five-token committed prefix.
            case={'id':identity,'rows':rows,'offset':0,'heads':2,'dimension':8,'capacity':32,'k_bits':bits['k'],'v_bits':bits['v'],
                  'initial_k_bits':[0]*512,'initial_v_bits':[0]*512}
            oracle=expected(case)
            if oracle!=(bits['cache_k'],bits['cache_v']):raise ValueError('Independent frozen cache reconstruction differs')
            inputs.append(case);anchors[identity]={'k':oracle[0],'v':oracle[1],'source_sha256':hashes,'rotary_position_offset':spec['position_offset']}
    for label,rows,offset in [('boundary',1,31),('append',3,7),('overwrite',2,0),('full',32,0)]:
        count=rows*16;edge=[0,0x8000,1,0x8001,0x3c00,0xbc00,0x7bff,0xfbff]
        case={'id':f'manual.{label}','rows':rows,'offset':offset,'heads':2,'dimension':8,'capacity':32,
              'k_bits':[0x3000+i%1024 for i in range(count)],'v_bits':[0xb000+i%1024 for i in range(count)],
              'initial_k_bits':[0x4400+i%256 for i in range(512)],'initial_v_bits':[0xc400+i%256 for i in range(512)]}
        case['k_bits'][:8]=edge;case['v_bits'][:8]=edge[::-1];k,v=expected(case)
        inputs.append(case);anchors[case['id']]={'k':k,'v':v,'manual_oracle':'Independent lossless head-major layout reconstruction; signedzero/subnormal/maxfinite and stale prefix/tail patterns'}
    if len(inputs)!=14:raise ValueError('Cache inventory differs')
    return inputs,anchors


def validate(event,inputs,anchors):
    if event.get('event')!='idle_result' or event.get('passed') is not True:raise ValueError('Native cache write failed')
    r=event['result'];m=r['metadata']
    if (r['operation']!='kv_half_token_to_head_major_write' or m['unified_memory'] is not True or m['language_version']!='2.3'
            or m['math_mode']!='safe' or m['math_functions']!='precise' or m['minimum_macos_major']!=15
            or [c['id'] for c in r['cases']]!=[i['id'] for i in inputs]):raise ValueError('Cache policy/inventory differs')
    for case,item in zip(r['cases'],inputs):
        count=item['rows']*16;width=case['threadgroup_width'];rounded=math.ceil(count/width)*width;input_bytes=(rounded+7)*2;cache_bytes=519*2
        shared=2*(input_bytes+cache_bytes);total=shared*3+rms.probe.ALLOWANCE
        if (case['rows_count']!=item['rows'] or case['offset']!=item['offset'] or case['heads']!=2 or case['dimension']!=8 or case['capacity']!=32
                or case['layout']!='heads_capacity_dimension' or case['input_guard_values']!=rounded-count+7 or case['cache_guard_values']!=7
                or case['k_guard_bits']!=K_GUARD or case['v_guard_bits']!=V_GUARD or case['committed_position_advanced'] is not False
                or case['allocations_reused_across_repetitions'] is not True or len(case['rows'])!=13
                or not 0<width<=min(m['execution_width'],m['pipeline_max_threads'],m['device_max_threads'][0]) or case['threadgroups']!=math.ceil(count/width)
                or case['shared_buffer_bytes']!=shared or case['host_baseline_readback_bytes']!=2*shared or case['admitted_bytes']!=total or total>rms.probe.LIMIT):
            raise ValueError('Cache shape/layout/admission differs')
        anchor=anchors[item['id']]
        for i,row in enumerate(case['rows']):
            if (row['phase']!=('warmup' if i<3 else 'sample') or row['repetition']!=(i+1 if i<3 else i-2)
                    or row['command_completed'] is not True or row['inputs_targets_untouched_and_guards_valid'] is not True
                    or not math.isfinite(row['encode_commit_wait_seconds']) or row['encode_commit_wait_seconds']<=0):raise ValueError('Cache completed-work evidence differs')
            for key,guard in [('k',K_GUARD),('v',V_GUARD)]:
                bits=row[f'cache_{key}_bits']
                if any(type(n) is not int or not 0<=n<=65535 for n in bits) or bits!=anchor[key]+[guard]*7:raise ValueError('Exact cache targets/prefix/tail/guards differ')
                row[f'cache_{key}_sha256']=hashlib.sha256(struct.pack('<519H',*bits)).hexdigest();del row[f'cache_{key}_bits']
            row['qualification']={'exact_half_bits':True,'cache_values':1024,'guard_values':14}
        case['anchors']={k:v for k,v in anchor.items() if k not in ('k','v')}
    if m['maximum_admitted_bytes']!=max(c['admitted_bytes'] for c in r['cases']):raise ValueError('Cache maximum admission differs')


def rejected(event):
    if (event.get('event')!='idle_result' or event.get('passed') is not False or event.get('error',{}).get('stage')!='fixture'
            or event['error'].get('message')!='cache prototype shape/position outside bounds'):raise ValueError('Cache boundary rejection differs')
    return False


def run(output,build,directory):
    rms.probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+'.stderr.txt'),output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():raise ValueError('Refusing to overwrite rotary inputs')
    for path in (build,*evidence):
        if directory.resolve()==path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:
            raise ValueError('Rotary inputs/build/evidence must be disjoint')
    if any(root.resolve()==path.resolve() or root.resolve() in path.resolve().parents for root in (rms.FP16,rms.FP32) for path in (build,directory,*evidence)):
        raise ValueError('Rotary outputs must be outside source fixtures')
    cache_inputs,cache_anchors=cases();inputs,anchors,overflow=rotary.cases();mlps,ma,mo=rotary.composed.cases();ag,aa=rotary.attention.batches();sg,sa=rotary.composed.stages.batches()
    p=rotary.composed.p;query,qa,po=p.cases();tiny,ta,go=p.gating.cases();tails,tla=p.gating.tail_cases()
    payloads=[]
    for i,c in enumerate(mlps):
        payloads.append((f'mlp_{i+1}',c,'--mlp-input',lambda e,c=c:rotary.composed.validate(e,c,ma[c['id']])))
    payloads.append(('mlp_overflow',mo,'--mlp-input',rotary.composed.overflow_rejected))
    for label,items in ag+sg:
        anchor=aa if label[0] in ('k','v','o') else sa
        payloads.append((label,{'format':'toklet-metal-projection-input-v1','cases':items},'--projection-input',lambda e,items=items,anchor=anchor:p.validate(e,items,anchor)))
    for label,items,option,validator in (
        ('query_manual_regression',query,'--projection-input',lambda e:p.validate(e,query,qa)),
        ('projection_overflow_regression',[po],'--projection-input',p.overflow_rejected),
        ('gating_regression',tiny,'--gating-input',lambda e:p.gating.validate(e,tiny,ta)),
        ('gating_tails_regression',tails,'--gating-input',lambda e:p.gating.validate_tails(e,tails,tla)),
        ('gating_overflow_regression',[go],'--gating-input',p.gating.overflow_rejected),
        ('rotary_frozen_manual',inputs,'--rotary-input',lambda e:rotary.validate(e,inputs,anchors)),
        ('rotary_overflow',[overflow],'--rotary-input',rotary.overflow_rejected)):
        fmt='toklet-metal-projection-input-v1' if option=='--projection-input' else 'toklet-metal-gating-input-v1' if option=='--gating-input' else 'toklet-metal-rotary-input-v1'
        payloads.append((label,{'format':fmt,'cases':items},option,validator))
    invalid={**cache_inputs[0],'rows':1,'offset':32,'k_bits':cache_inputs[0]['k_bits'][:16],'v_bits':cache_inputs[0]['v_bits'][:16]}
    payloads += [('cache_frozen_manual',{'format':'toklet-metal-cache-input-v1','cases':cache_inputs},'--cache-input',lambda e:validate(e,cache_inputs,cache_anchors)),
                 ('cache_capacity_rejection',{'format':'toklet-metal-cache-input-v1','cases':[invalid]},'--cache-input',rejected)]
    directory.mkdir(parents=True);jobs=[];identities={}
    for label,payload,option,validator in payloads:
        path=directory/f'{label}.json'
        with path.open('x') as stream:stream.write(json.dumps(payload,separators=(',',':'))+'\n')
        identities[str(path.resolve())]=rms.probe.common.digest(path);jobs.append((label,[option,str(path.resolve())],validator))
    modules=(rotary,rotary.attention,rotary.composed,rotary.composed.stages,p,p.gating,rms,p.gating.silu,p.gating.residual,p.gating.residual.embedding)
    drivers={str(Path(m.__file__).resolve()):rms.probe.common.digest(Path(m.__file__)) for m in modules};drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        result=previous(event)
        if rotary.cases()!=(inputs,anchors,overflow) or cases()!=(cache_inputs,cache_anchors):raise ValueError('Rotary frozen inputs changed')
        for name,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(name))!=sha:raise ValueError('Rotary input/driver identity changed')
        return result
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context={'input_sha256':identities,'qualification_driver_sha256':drivers,
        'fp16_manifest_sha256':rms.FP16_MANIFEST_SHA,'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,
        'scope':'Lossless tiny token-major to separate head-major K/V staging writes. Ten frozen local offset0 cases plus four stale-prefix/tail/manual/last-slot/full-capacity cases. SourceK already rotated; no rotary propagation, valid-prefix/session advancement, full-layer/decoder/Qwen/quality/performance/M4 closure. Same-binary rotary/attention/MLP/projection/gating regressions.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
    a=p.parse_args();run(a.output,a.build_dir,a.input_dir)

if __name__=='__main__':main()
