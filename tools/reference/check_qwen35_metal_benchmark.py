"""Paired coupled forward17 baseline evidence, distinct from request TTFT."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import check_qwen35_metal_context as context

need,exact,document,read=context.need,context.exact,context.document,context.read


def samples(rows,cohort):
    need(type(rows) is list and len(rows)==26,'One complete alternating paired cohort required')
    groups={'scalar_cpu':[],'metal_projection_cpu_state':[]};maximum=0.0
    for index,row in enumerate(rows):
        need(set(row)=={'cohort','iteration','warmup','engine','elapsed_ns','tokens','position_offset','work','maximum_cpu_gpu_logit_gate_ratio'},'Exact paired sample fields required')
        iteration,slot=divmod(index,2);scalar=(slot==0)==((iteration+cohort)%2==0)
        engine='scalar_cpu' if scalar else 'metal_projection_cpu_state'
        for name,value in dict(cohort=cohort,iteration=iteration,warmup=iteration<3,engine=engine,tokens=17,position_offset=37,work=context.work(17)).items():
            need(exact(row[name],value),'Frozen paired work/order differs: '+name)
        need(type(row['elapsed_ns']) is int and row['elapsed_ns']>0,'Completed positive caller walltime required')
        ratio=row['maximum_cpu_gpu_logit_gate_ratio']
        need(type(ratio) in [int,float] and math.isfinite(ratio) and 0<=ratio<=1,'All-logit pairwise numerical gate required')
        maximum=max(maximum,ratio)
        if iteration>=3:groups[engine].append(row['elapsed_ns'])
    result={}
    for engine,values in groups.items():
        need(len(values)==10,'Ten measured forwards per engine required')
        median=statistics.median(values)
        result[engine]=dict(samples_ns=values,median_ns=median,mad_ns=statistics.median([abs(v-median) for v in values]),minimum_ns=min(values),maximum_ns=max(values))
    return dict(cohort=cohort,engines=result,improvement_fraction=1-result['metal_projection_cpu_state']['median_ns']/result['scalar_cpu']['median_ns'],maximum_cpu_gpu_logit_gate_ratio=maximum)


def conditions(environment,elapsed_seconds):
    need(set(environment)=={'start','periodic','sampling','end'},'Both endpoints and external observations required')
    need(type(elapsed_seconds) in [int,float] and math.isfinite(elapsed_seconds) and elapsed_seconds>0,'Finite owned run duration required')
    need(environment['sampling']=='External30s CLI snapshots, not engine instrumentation, energy or continuous thermal telemetry','Declared external sampling policy differs')
    commands=dict(power=['/usr/bin/pmset','-g','batt'],thermal=['/usr/bin/pmset','-g','therm'],memory_pressure=['/usr/bin/memory_pressure','-Q'],
        hardware=['/usr/sbin/sysctl','-n','hw.model','machdep.cpu.brand_string','hw.memsize'],os=['/usr/bin/sw_vers'],configuration=['/usr/bin/pmset','-g','custom'])
    start,end=environment['start'],environment['end']
    need(type(start) is dict and type(end) is dict and set(start)==set(end)==set(commands),'Complete external endpoint inventory required')
    periodic=environment['periodic']
    need(type(periodic) is list and all(type(v) is dict and set(v)=={'elapsed_seconds','observations'} and type(v['observations']) is dict and set(v['observations'])=={'power','thermal','memory_pressure'} for v in periodic),'Complete periodic observation inventory required')
    for observation in [start,*[v['observations'] for v in periodic],end]:
        for name,item in observation.items():
            need(type(item) is dict and set(item)=={'command','exit_code','output'} and exact(item['command'],commands[name]),'Named external command observation required')
    for name in ['hardware','os','configuration']:
        need(type(start[name]['exit_code']) is int and type(end[name]['exit_code']) is int
             and start[name]['exit_code']==end[name]['exit_code']==0 and type(start[name]['output']) is str
             and bool(start[name]['output']) and start[name]['output']==end[name]['output'],'Hardware/OS/power configuration drift')
    observations=[start,*[v['observations'] for v in periodic],end]
    elapsed=[v['elapsed_seconds'] for v in periodic]
    need(all(type(v) in [float,int] and math.isfinite(v) and 0<v<elapsed_seconds for v in elapsed)
         and all(a<b for a,b in zip(elapsed,elapsed[1:])),'Ordered finite external sample times required')
    # A deleted middle observation must not turn a sparsely observed long run
    # into a qualified30s study. Allow one interval of collection/scheduling slack.
    need(all(b-a<=60 for a,b in zip([0,*elapsed],[*elapsed,elapsed_seconds])),'External observations must cover the owned run without gaps over60s')
    thermal=['Note: No thermal warning level has been recorded','Note: No performance warning level has been recorded','Note: No CPU power status has been recorded']
    qualified=True
    for sample in observations:
        for name in ['power','thermal','memory_pressure']:
            need(set(sample[name])=={'command','exit_code','output'},'Complete external observation required')
            if type(sample[name]['exit_code']) is not int or sample[name]['exit_code']!=0 or type(sample[name]['output']) is not str:qualified=False
        power=sample['power'].get('output','').splitlines()
        qualified &= bool(power) and power[0]=="Now drawing from 'AC Power'"
        qualified &= sample['thermal'].get('output','').splitlines()==thermal
        pressure=re.findall(r'^System-wide memory free percentage: (\d+)%\s*$',sample['memory_pressure'].get('output',''),re.M)
        qualified &= len(pressure)==1 and 0<=int(pressure[0])<=100
    return qualified


def run(folder,cohort):
    receipt=context.closed(folder,'benchmark');report=receipt['report']
    log=read(folder/'console.log',2*1024**2).decode()
    labels=lambda tag:[context.prior.decode(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith(tag+' ')]
    footer=labels('QWEN35_METAL_BENCH_REPORT');need(len(footer)==1 and exact(footer[0],report),'Receipt/footer agreement required')
    tokens,offset,_=context.producer.plan('context17')
    for name,value in dict(cohort=cohort,phase='context17',source_sha256=context.producer.pin.FILE_SHA,config_sha256=context.producer.pin.CONFIG_SHA,header_sha256=context.producer.pin.HEADER_SHA,
        tokens=tokens,position_offset=offset,capacity=512,chunk=32,diagnostic_allowance_bytes=128*1024**2,
        all17_greedy_ids_match=True,warmups_per_engine=3,measurements_per_engine=10,
        whole_gpu_graph_qualified=False,request_ttft_qualified=False,sustained_qualified=False,quality_qualified=False,managed_adoption_qualified=False).items():
        need(exact(report[name],value),'Frozen coupled benchmark contract differs: '+name)
    # GPU ledger is already independently derived for the exact same512/32
    # source geometry in both native coupled fidelity receipts.
    source=context.document(folder.parent/'actual-context17-1/checks.json')['report']
    gpu=context.memory(source);need(exact(report['gpu_memory'],gpu),'GPU benchmark owner ledger differs')
    _,load,cpu_load=context.owners();need(exact(report['gpu_load_report'],load),'Consuming source owner ledger differs')
    # CPU layouts are fixed by this qualified arm64 ABI: default executor is
    # zero-sized, Model280/Layer1000/Session1152; all numerical banks match GPU.
    cpu={**gpu,'weights_and_model_bytes':2369040384+280+32*1000,
        'scratch_bytes':504320,'metadata_bytes':1152+24*144+8*48}
    cpu['admitted_bytes']=sum(value for name,value in cpu.items() if name!='admitted_bytes')
    need(exact(report['cpu_memory'],cpu),'Scalar reference owner ledger differs')
    need(exact(report['combined_admitted_bytes'],gpu['admitted_bytes']+cpu['admitted_bytes']+128*1024**2)
         and report['combined_admitted_bytes']<=context.LIMIT,'All simultaneous reference/candidate/diagnostic owners must fit6GiB')
    need(exact(report['cpu_load_overlap_admitted_bytes'],gpu['admitted_bytes']+cpu_load+128*1024**2)
         and report['cpu_load_overlap_admitted_bytes']<=context.LIMIT,'Cold scalar load overlap must fit6GiB')
    rows=labels('QWEN35_METAL_SAMPLE');summary=samples(rows,cohort)
    need(exact(report['maximum_cpu_gpu_logit_gate_ratio'],summary['maximum_cpu_gpu_logit_gate_ratio']),'Aggregate pairwise numerical ratio differs')
    need(sum(r['elapsed_ns'] for r in rows)<receipt['elapsed_seconds']*1e9,'Timers cannot exceed complete owned run')
    need(exact(document(folder/'inputs.json')['external_environment_sample_seconds'],30),'Frozen30s external sampling interval required')
    summary.update(peak_bytes=receipt['lifetime_peak_physical_bytes'],environment_qualified=conditions(document(folder/'environment.json'),receipt['elapsed_seconds']))
    return summary


def check(first,second):
    # Performance never substitutes for independent coupled correctness.
    need(first.parent==second.parent,'One frozen comparison task required')
    inputs=[document(folder/'inputs.json') for folder in [first,second]]
    need(inputs[0]['binary_sha256']==inputs[1]['binary_sha256'] and exact(inputs[0]['source_pins'],inputs[1]['source_pins']), 'Independent runs must use the same build/source inputs')
    for name in ['fidelity-one-final.json','fidelity-context17-final.json']:
        need(document(first.parent/name)['coupled_projection_fidelity_passed'] is True,'Independent coupled fidelity must pass before retention')
        digest=hashlib.sha256(read(first.parent/name,2*1024**2)).hexdigest()
        need(all(v['coupled_fidelity_sha256'][name]==digest for v in inputs),'Pre-benchmark coupled fidelity identity differs')
    environments=[document(folder/'environment.json') for folder in [first,second]]
    need(all(exact(environments[0]['start'][name],environments[1]['start'][name]) for name in ['hardware','os','configuration']), 'Confirmation hardware/OS/power configuration differs')
    runs=[run(first,0),run(second,1)]
    return dict(passed=True,primary='Completed full17input/offset37 caller walltime, CPU scalar vs Metal projection/CPU-state, resident weights and empty numerical prefix',
        runs=runs,retain_coupled_projection_candidate=all(r['environment_qualified'] and r['improvement_fraction']>=.05 for r in runs),
        warmups_total=12,measured_samples_total=40,whole_gpu_graph_qualified=False,request_ttft_qualified=False,sustained_qualified=False,quality_qualified=False,managed_adoption_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('first',type=Path);parser.add_argument('second',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=check(args.first,args.second)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result));raise SystemExit(0 if result['retain_coupled_projection_candidate'] else 1)
