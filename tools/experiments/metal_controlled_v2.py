"""Frozen larger v2 controlled matrix with truthful representation and admission."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import metal_controlled as shared
import metal_quality_v2 as loss

ROOT=shared.ROOT
MODEL='larger'
MODE='Q4CompensatedTiled'


def ready(event,model):
    if model!=MODEL:raise ValueError('Only fixed larger v2 is admitted')
    required={'event':'ready','format':'toklet-metal-controlled-v1','debug_assertions':False,'backend':'metal',
              'projection_mode':MODE,'normalization_mode':'Compensated','readback_mode':'FullLogits',
              'weight_dtype':'Q4 v2 + FP16 exclusions','kv_dtype':'FP16','hidden_size':1536,'num_layers':28,
              'vocab_size':151936,'context_capacity':2048,'chunk_size':128,'application_bytes':64*1024**2,
              'packed_format_version':2,'group_size':8,'quantizer':'toklet-q4-affine-f32-source-half-metadata-rne-g8-v2'}
    if any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in required.items()):raise ValueError('Actual v2 controlled mode differs')
    widths=event.get('q4_compensated_execution_widths')
    if type(widths) is not list or len(widths)!=2 or any(type(w) is not int or not 0<w<=0xffffffff for w in widths) or widths[0]!=widths[1]:
        raise ValueError('Actual v2 queried widths differ')
    m=event.get('memory');keys=('weights_bytes','kv_bytes','scratch_bytes','history_bytes','logits_bytes','metadata_bytes','metadata_allowance_bytes')
    if (type(m) is not dict or set(m)!=set(keys)|{'admitted_bytes'} or any(type(m[k]) is not int or m[k]<0 for k in m)
            or sum(m[k] for k in keys)!=m['admitted_bytes'] or m['weights_bytes']!=loss.SPECS[MODEL]['weights'] or m['logits_bytes']!=151936*8):
        raise ValueError('Actual v2 controlled memory differs')
    if type(event.get('memory_admitted_bytes')) is not int or event['memory_admitted_bytes']!=m['admitted_bytes']+event['application_bytes'] or not 0<event['memory_admitted_bytes']<=shared.LIMIT:
        raise ValueError('V2 complete admission differs')


def run(binary,output,check=False,*,workloads=None):
    selected=tuple(shared.frozen.WORKLOADS) if workloads is None else tuple(workloads)
    if not selected or len(set(selected))!=len(selected) or any(name not in shared.frozen.WORKLOADS for name in selected):
        raise ValueError('Distinct frozen controlled workloads required')
    paths=[output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):raise ValueError('Exclusive v2 controlled evidence exists')
    manifest,arrays=shared.contract();spec=shared.MODELS[MODEL];source=ROOT/'models'/spec[0];packed=ROOT/'artifacts/m5-larger-q4-v2'
    geometry_path=ROOT/loss.GEOMETRY_REL
    geometry=json.loads(loss.shared.frozen_bytes(geometry_path,1024**2,loss.GEOMETRY_SHA))
    if geometry.get('passed') is not True or geometry.get('review',{}).get('accepted') is not True:raise ValueError('Reviewed v2 numerical geometry required')
    artifact_pins={p:v['sha256'] for p,v in geometry['bindings'].items() if p.startswith('artifacts/m5-larger-q4-v2/')}
    if artifact_pins.get('artifacts/m5-larger-q4-v2/manifest.json')!=loss.SPECS[MODEL]['packed']:raise ValueError('Frozen v2 controlled artifact differs')
    modules=(shared,loss,loss.shared,loss.large,loss.large.quality_rows,shared.frozen,shared.prepare,shared.common,
             shared.metal_engine,shared.native_usage,shared.metal_prefill,shared.metal_continue)
    files=[binary,Path(__file__),ROOT/'examples/metal_bench.rs',ROOT/'Cargo.toml',ROOT/'Cargo.lock',geometry_path,
           shared.prepare.CONTRACT/'manifest.json',ROOT/'benchmarks/result.schema.json',ROOT/'benchmarks/sources.json',ROOT/'tools/bench/requirements.lock',
           source/'model.safetensors',source/'config.json',source/'tokenizer.json']
    files += [Path(m.__file__) for m in modules]+list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))
    files += [shared.prepare.CONTRACT/v['file'] for v in manifest['arrays'].values()]+[ROOT/p for p in artifact_pins]
    identities={str(p):shared.metal_engine.digest(p) for p in files};shared.initial_pins(identities,manifest,source,spec)
    pins={geometry_path:loss.GEOMETRY_SHA,**{ROOT/p:sha for p,sha in artifact_pins.items()}}
    if any(identities.get(str(p))!=sha for p,sha in pins.items()):raise ValueError('Initial v2 controlled inputs differ')
    observation=shared.metal_prefill.observation
    observations={'compiler':observation(['rustc','--version']),'device':{'name':observation(['/usr/sbin/sysctl','-n','hw.model']),
                  'chip':observation(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string']),'physical_memory_bytes':int(observation(['/usr/sbin/sysctl','-n','hw.memsize'])),
                  'os_version':observation(['/usr/bin/sw_vers','-productVersion']),'os_build':observation(['/usr/bin/sw_vers','-buildVersion']),
                  'power_mode':observation(['/usr/bin/pmset','-g','batt'])+'\n'+observation(['/usr/bin/pmset','-g','custom']),
                  'thermal_conditions':observation(['/usr/bin/pmset','-g','therm'])}}
    record={'format':'toklet-q4-v2-controlled-collection-v1','passed':False,'check_only':check,'measurements_completed':False,
            'scope':'Selected frozen larger v2 controlled workloads; release quality, application and gain acceptance separate',
            'selected_workloads':list(selected),'full_matrix_completed':False,
            'model':MODEL,'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'identities':identities,
            'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            'observations':observations,'runs':[],'results':[]}
    with paths[0].open('x') as summary,paths[1].open('x') as raw,paths[2].open('xb') as stderr:
        try:
            for name in selected:
                item=shared.worker(binary,source,MODEL,name,manifest,arrays,raw,stderr,check,admission_check=ready,extra_args=(str(packed),'q4-v2-tiled'))
                record['runs'].append(item)
                if not item['passed']:raise ValueError(f'V2 controlled {name} failed')
                if not check:
                    admission=item['ready']
                    note=f"Actual {MODE}/Compensated/FullLogits, queried widths{admission['q4_compensated_execution_widths']}; complete admission{admission['memory_admitted_bytes']}. Q4 codes use8effectivebits/weight including metadata before padding; actual artifact/residency in parent evidence."
                    record['results'].append(shared.result(item,MODEL,manifest,observations,record['revision'],record['working_tree_dirty'],
                                            representation={'weight_dtype':'Q4','kv_dtype':'FP16','packed_format_version':2,'group_size':8},
                                            runtime='Toklet Q4 v2 CompensatedTiled Metal',execution_note=note))
            record.update(passed=True,measurements_completed=not check,full_matrix_completed=not check and selected==tuple(shared.frozen.WORKLOADS))
        except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
        finally:
            record['endpoint_errors']=[]
            try:record['identities_unchanged']=all(shared.metal_engine.digest(Path(p))==sha for p,sha in identities.items())
            except BaseException as error:record['identities_unchanged']=False;record['endpoint_errors'].append(str(error))
            try:record['power_after']=shared.metal_prefill.endpoints()
            except BaseException as error:record['endpoint_errors'].append(str(error))
            if not record['identities_unchanged'] or record['endpoint_errors']:record.update(passed=False,measurements_completed=False,full_matrix_completed=False)
            json.dump(record,summary,indent=2,allow_nan=False);summary.write('\n')
    if not record['passed']:raise ValueError('V2 controlled collection failed; see retained evidence')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--check',action='store_true');parser.add_argument('--workload',choices=tuple(shared.frozen.WORKLOADS),action='append')
    args=parser.parse_args();run(args.binary.resolve(),args.output.resolve(),args.check,workloads=args.workload)
