"""Frozen larger v2 three-turn application trace; defaults preserved."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import metal_application as shared
import metal_tasks_v2 as producer
import task_grade_admission_v2 as collection

ROOT=shared.ROOT
QUALITY_REL='benchmarks/results/2026-10-14-q4-v2-task-quality-qualification-v1.json'
QUALITY_SHA='ecee8abce37d56171c730a980344dd737764d3fa6aa87c26ac794d79bff7b217'


def quality_prerequisite():
    q=json.loads(producer.loss.shared.frozen_bytes(ROOT/QUALITY_REL,2*1024**2,QUALITY_SHA))
    if (q.get('passed') is not True or q.get('quality_completed') is not True or q.get('model')!='larger'
            or q.get('review',{}).get('accepted') is not True or q.get('cases')!=100
            or q.get('q4_passed')!=11 or q.get('fp32_passed')!=11 or q.get('drop_percentage_points')!=0):
        raise ValueError('Reviewed full larger v2 quality required')
    producer.loss_prerequisite()


def run(output,check=False):
    output=Path(output).resolve();paths=[output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):raise ValueError('Exclusive v2 application evidence exists')
    quality_prerequisite();binary,q=producer.qualified_cli();manifest=shared.admission.frozen_manifest()
    source=ROOT/'models'/producer.loss.SPECS['larger']['name'];packed=ROOT/'artifacts/m5-larger-q4-v2'
    cap=2 if check else 128;requests,prompts=shared.trace_inputs(source,manifest,cap)
    files=collection.inventory(binary,q,source,manifest)|{str(Path(__file__).resolve()),str(Path(shared.__file__).resolve()),
            str(Path(collection.__file__).resolve()),str(Path(shared.admission.__file__).resolve()),str(Path(shared.admission.evidence.__file__).resolve()),str(ROOT/QUALITY_REL),str(shared.controlled.prepare.CONTRACT/'application_trace.json')}
    identities={p:shared.metal_engine.digest(Path(p)) for p in files};shared.controlled.initial_pins(identities,manifest,source,shared.controlled.MODELS['larger'])
    pins=producer.compiled_pins(q);pins.update({ROOT/producer.QUAL_REL:producer.QUAL_SHA,ROOT/producer.LOSS_REL:producer.LOSS_SHA,
         ROOT/QUALITY_REL:QUALITY_SHA,shared.controlled.prepare.CONTRACT/'application_trace.json':manifest['authored_text_sha256']['application_trace.json']})
    pins.update({source/name:sha for name,sha in producer.cli.shared.ASSET_PINS.items()})
    pins.update({ROOT/p:v['sha256'] for p,v in q['bindings'].items() if p.startswith('artifacts/m5-larger-q4-v2/')})
    producer.check_initial_pins(identities,pins)
    observation=shared.controlled.metal_prefill.observation
    environment={'hardware':observation(['/usr/sbin/sysctl','-n','hw.model']),'chip':observation(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string']),
        'physical_memory_bytes':int(observation(['/usr/sbin/sysctl','-n','hw.memsize'])),'os':observation(['/usr/bin/sw_vers','-productVersion']),
        'os_build':observation(['/usr/bin/sw_vers','-buildVersion']),'compiler':observation(['rustc','--version']),
        'power':observation(['/usr/bin/pmset','-g','batt'])+'\n'+observation(['/usr/bin/pmset','-g','custom']),'thermal_start':observation(['/usr/bin/pmset','-g','therm'])}
    record={'format':'toklet-q4-v2-application-v1','passed':False,'check_only':check,'measurements_completed':False,
        'quality_completed':False,'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'identities':identities,'model':'larger','backend':'metal','projection_mode':producer.cli.MODE,'normalization_mode':'Compensated',
        'readback_mode':'FullLogits','weight_dtype':'Q4 v2 + FP16 exclusions','packed_format_version':2,'group_size':8,'environment':environment,
        'policy':{'warmups':0 if check else 3,'repetitions':1 if check else 10,'turns':3,'output_cap':cap,'greedy':True,
                  'cache':'empty per turn; scripted assistant history; resident model'},
        'timing_scope':'Driver submission through receipt of complete JSON token/nonempty text/done/reset events; completed native numerical/sampling/formatting/tokenization/text/publication components overlap and are not summed; scenario includes per-turn driver validation/memory instrumentation. Model setup and independent oracle decode outside warm request timers; no cold storage/energy claim.'}
    with paths[0].open('x') as summary,paths[1].open('x') as raw,paths[2].open('xb') as stderr:
        try:
            record['run']=shared.child(binary,source,'larger',requests,prompts,raw,stderr,check,admission_check=producer.ready,
                  projection_argument=producer.cli.ARG,extra_args=('--q4-weights',str(packed)))
            if not record['run']['passed']:raise ValueError('V2 application worker failed')
            record.update(passed=True,measurements_completed=not check)
            if not check:
                values=[s['scenario_seconds'] for s in record['run']['samples']]
                record['summary']={'scenario_median_seconds':statistics.median(values),'scenario_min_seconds':min(values),'scenario_max_seconds':max(values)}
        except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
        finally:
            record['endpoint_errors']=[];record['identities_unchanged']=False
            try:record['identities_unchanged']=all(shared.metal_engine.digest(Path(p))==sha for p,sha in identities.items())
            except BaseException as error:record['endpoint_errors'].append(f'{type(error).__name__}: {error}')
            try:record['environment']['thermal_end']=observation(['/usr/bin/pmset','-g','therm'])
            except BaseException as error:record['endpoint_errors'].append(f'thermal_end: {type(error).__name__}: {error}')
            if record['endpoint_errors'] or not record['identities_unchanged']:record.update(passed=False,measurements_completed=False)
            json.dump(record,summary,indent=2,allow_nan=False);summary.write('\n')
    if not record['passed']:raise ValueError('V2 application collection failed; evidence retained')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();run(args.output,args.check)
