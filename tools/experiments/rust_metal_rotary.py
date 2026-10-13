"""Qualify stable FP32 rotary with Half storage under frozen tiny source gates."""
import argparse
import hashlib
import math
from pathlib import Path
import struct
import json

import rust_metal_residual as residual
import rust_metal_attention_projections as attention
import rust_metal_mlp as composed

rms=residual.rms
ROOT=rms.ROOT
GUARD=0x55a1


def cases():
    half=rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA)
    single=rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
    model=half['model']
    if model!=single['model'] or (model['head_dim'],model['num_query_heads'],model['num_kv_heads'],model['rope_theta'],model['rope_pairing'])!=(8,4,2,1_000_000.0,'split_half'):
        raise ValueError('Frozen rotary topology differs')
    inputs=[];anchors={}
    for name,spec in sorted(half['cases'].items()):
        rows=len(spec['tokens'])
        for layer in range(2):
            for stage,heads in (('q',4),('k',2)):
                count=rows*heads*8;prefix=f'cases.{name}.layers.{layer}';key=f'{prefix}.{stage}_rotary'
                source,source_item=rms.payload(rms.FP16,half,f'{prefix}.{stage}_projection','f16le')
                oracle,item=rms.payload(rms.FP16,half,key,'f16le');reference,ref=rms.payload(rms.FP32,single,key,'f32le')
                if any(record['shape']!=[rows,heads,8] for record in (source_item,item,ref)):
                    raise ValueError('Frozen rotary shape differs')
                inputs.append({'id':key,'rows':rows,'heads':heads,'dimension':8,'theta':1_000_000.0,'offset':spec['position_offset'],
                               'input_bits':list(struct.unpack(f'<{count}H',source))})
                anchors[key]={'bits':list(struct.unpack(f'<{count}H',oracle)),'reference':list(struct.unpack(f'<{count}f',reference)),
                              'gate':item['gate_against_source_fp32'],'input_fp16_sha256':source_item['sha256'],
                              'oracle_fp16_sha256':item['sha256'],'source_fp32_sha256':ref['sha256']}
    if len(inputs)!=20 or sum(i['rows']*i['heads']*8 for i in inputs)!=4224:
        raise ValueError('Frozen rotary inventory differs')
    # Position zero is an exact identity for these nonzero dyadic inputs,
    # exposing split-half indexing independently of trigonometric tolerances.
    bits=list(struct.unpack('<16H',struct.pack('<16e',*range(1,17))))
    key='manual.position_zero'
    inputs.append({'id':key,'rows':1,'heads':2,'dimension':8,'theta':1_000_000.0,'offset':0,'input_bits':bits})
    anchors[key]={'bits':bits,'reference':list(range(1,17)),'gate':None,'manual_oracle':'Nonzero dyadic identity at position0; exact Half bits'}
    overflow={'id':'manual.overflow','rows':1,'heads':2,'dimension':8,'theta':1_000_000.0,'offset':1,
              'input_bits':[0x7bff]*4+[0xfbff]*4+[0]*8}
    return inputs,anchors,overflow


def validate(event,inputs,anchors):
    if event.get("event")!="idle_result" or event.get("passed") is not True:raise ValueError("Native rotary check failed")
    result=event["result"];metadata=result["metadata"]
    if (result["operation"]!="rotary_split_half_fp32_half_store" or metadata["unified_memory"] is not True
            or metadata["language_version"]!="2.3" or metadata["math_mode"]!="safe" or metadata["math_functions"]!="precise"
            or metadata["minimum_macos_major"]!=15 or not 0<metadata["maximum_admitted_bytes"]<=rms.probe.LIMIT
            or len(metadata["device_max_threads"])!=3 or min(metadata["device_max_threads"])<=0
            or [case["id"] for case in result["cases"]]!=[item["id"] for item in inputs]):
        raise ValueError("Native rotary policy/inventory differs")
    for case,item in zip(result["cases"],inputs):
        count=item["rows"]*item["heads"]*item["dimension"];width=case["threadgroup_width"];rounded=math.ceil((count//2)/width)*width*2;guards=rounded-count+7;size=(rounded+7)*2
        if (case["rows_count"]!=item["rows"] or case["heads"]!=item["heads"] or case["dimension"]!=item["dimension"] or case["theta"]!=1_000_000.0 or case["offset"]!=item["offset"] or case["pairing"]!="split_half" or case["guards"]!=guards
                or case["output_guard_bits"]!=GUARD or case["allocations_reused_across_repetitions"] is not True
                or len(case["rows"])!=13 or not 0<width<=min(metadata["execution_width"],metadata["pipeline_max_threads"],metadata["device_max_threads"][0])
                or case["threadgroups"]!=math.ceil((count//2)/width) or case["shared_buffer_bytes"]!=size*2
                or case["host_input_and_readback_bytes"]!=size*3 or case["admitted_bytes"]!=size*5+rms.probe.ALLOWANCE):
            raise ValueError("Native rotary shape/ownership/launch/admission differs")
        anchor=anchors[item["id"]]
        for index,row in enumerate(case["rows"]):
            bits=row["output_bits"]
            if (row["phase"]!=("warmup" if index<3 else "sample") or row["repetition"]!=(index+1 if index<3 else index-2)
                    or row["command_completed"] is not True or row["immutable_input_and_guards"] is not True
                    or not math.isfinite(row["encode_commit_wait_seconds"]) or row["encode_commit_wait_seconds"]<=0
                    or len(bits)!=rounded+7 or any(type(value) is not int or not 0<=value<=65535 for value in bits)
                    or bits[count:]!=[GUARD]*guards or any(value & 0x7c00==0x7c00 for value in bits[:count])):
                raise ValueError("rotary completed-work/finite bits/guards differ")
            payload=struct.pack(f"<{count}H",*bits[:count]);actual=struct.unpack(f"<{count}e",payload)
            errors=[abs(value-ref) for value,ref in zip(actual,anchor["reference"])];gate=anchor["gate"]
            fraction=max(error/(gate["atol"]+gate["rtol"]*abs(ref)) for error,ref in zip(errors,anchor["reference"])) if gate else 0.0
            exact=bits[:count]==anchor["bits"]
            if fraction>1 or gate is None and not exact:raise ValueError("Frozen rotary FP32 source gate or exact manual bits failed")
            row["qualification"]={"exact_independent_fp16_bits":exact,
                "independent_fp16_differing_values":sum(a!=b for a,b in zip(bits[:count],anchor["bits"])),
                "source_fp32_max_abs_error":max(errors),"against_source_fp32_max_gate_fraction":fraction,
                "output_sha256":hashlib.sha256(payload).hexdigest(),"output_values":count,"output_guard_values":guards}
            del row["output_bits"]
        case["anchors"]={key:value for key,value in anchor.items() if key not in ("bits","reference")}
    if metadata["maximum_admitted_bytes"]!=max(case["admitted_bytes"] for case in result["cases"]):
        raise ValueError("Native rotary maximum admission differs")


def overflow_rejected(event):
    if (event.get('event')!='idle_result' or event.get('passed') is not False
            or event.get('error',{}).get('stage')!='validation'
            or event['error'].get('message')!='completed rotary contains nonfinite Half; result discarded'):
        raise ValueError('Rotary overflow failure policy differs')
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
    inputs,anchors,overflow=cases();mlps,ma,mo=composed.cases();ag,aa=attention.batches();sg,sa=composed.stages.batches()
    p=composed.p;query,qa,po=p.cases();tiny,ta,go=p.gating.cases();tails,tla=p.gating.tail_cases()
    payloads=[]
    for i,c in enumerate(mlps):
        payloads.append((f'mlp_{i+1}',c,'--mlp-input',lambda e,c=c:composed.validate(e,c,ma[c['id']])))
    payloads.append(('mlp_overflow',mo,'--mlp-input',composed.overflow_rejected))
    for label,items in ag+sg:
        anchor=aa if label[0] in ('k','v','o') else sa
        payloads.append((label,{'format':'toklet-metal-projection-input-v1','cases':items},'--projection-input',lambda e,items=items,anchor=anchor:p.validate(e,items,anchor)))
    for label,items,option,validator in (
        ('query_manual_regression',query,'--projection-input',lambda e:p.validate(e,query,qa)),
        ('projection_overflow_regression',[po],'--projection-input',p.overflow_rejected),
        ('gating_regression',tiny,'--gating-input',lambda e:p.gating.validate(e,tiny,ta)),
        ('gating_tails_regression',tails,'--gating-input',lambda e:p.gating.validate_tails(e,tails,tla)),
        ('gating_overflow_regression',[go],'--gating-input',p.gating.overflow_rejected),
        ('rotary_frozen_manual',inputs,'--rotary-input',lambda e:validate(e,inputs,anchors)),
        ('rotary_overflow',[overflow],'--rotary-input',overflow_rejected)):
        fmt='toklet-metal-projection-input-v1' if option=='--projection-input' else 'toklet-metal-gating-input-v1' if option=='--gating-input' else 'toklet-metal-rotary-input-v1'
        payloads.append((label,{'format':fmt,'cases':items},option,validator))
    directory.mkdir(parents=True);jobs=[];identities={}
    for label,payload,option,validator in payloads:
        path=directory/f'{label}.json'
        with path.open('x') as stream:stream.write(json.dumps(payload,separators=(',',':'))+'\n')
        identities[str(path.resolve())]=rms.probe.common.digest(path);jobs.append((label,[option,str(path.resolve())],validator))
    modules=(attention,composed,composed.stages,p,p.gating,rms,p.gating.silu,p.gating.residual,p.gating.residual.embedding)
    drivers={str(Path(m.__file__).resolve()):rms.probe.common.digest(Path(m.__file__)) for m in modules};drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        result=previous(event)
        if cases()!=(inputs,anchors,overflow):raise ValueError('Rotary frozen inputs changed')
        for name,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(name))!=sha:raise ValueError('Rotary input/driver identity changed')
        return result
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context={'input_sha256':identities,'qualification_driver_sha256':drivers,
        'fp16_manifest_sha256':rms.FP16_MANIFEST_SHA,'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,
        'scope':'20 saved-input tiny Q/K split-half rotary stages, frozen theta1e6/dimension8 and absolute offsets, exact position0/manual and finite-input overflow. GPU FP32 angle/trig/products/sums then Half store. Original source gates, mapped inactive guards; same-binary attention/MLP/projection/gating regressions. No propagated decoder, KV writes, Qwen, quality, speed or M4 completion.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--input-dir",type=Path,required=True);args=parser.parse_args();run(args.output,args.build_dir,args.input_dir)


if __name__=="__main__":main()
