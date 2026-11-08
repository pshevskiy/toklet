"""Closed pinned text codec evidence and phase baselines; no model admission."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import qwen35_text_codec_reference as producer

FIXTURE_SHA='ffc3ca64bf65f337d58bfef0f565d6f5c993eb480d349ebd400bdfb2ce8011db'
LIMIT=512*1024**2
COUNTS=dict(structure=1,chat=6,text=10,control=33,bytes=6,rejection=43,padding=1)


def need(value,message):
    if not value:raise ValueError(message)


def exact(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def read(path,limit=2*1024**2):
    need(not path.is_symlink() and path.is_file(),'Regular bounded evidence required')
    with path.open('rb') as stream:raw=stream.read(limit+1)
    need(len(raw)<=limit,'Evidence exceeds read bound');return raw


def document(path):
    return json.loads(read(path),object_pairs_hook=producer.runtime.common.json_unique,parse_constant=producer.runtime.common.bad_json_number)


def digest(path):return hashlib.sha256(read(path,512*1024**2)).hexdigest()


def environment(value):
    need(set(value)=={'start','end','sampling'} and value['sampling']=='External start/end CLI observations; text-only processes, no continuous telemetry','Exact external endpoint scope required')
    commands=dict(hardware=['/usr/sbin/sysctl','-n','hw.model','machdep.cpu.brand_string','hw.memsize'],os=['/usr/bin/sw_vers'],configuration=['/usr/bin/pmset','-g','custom'],
        power=['/usr/bin/pmset','-g','batt'],thermal=['/usr/bin/pmset','-g','therm'],memory_pressure=['/usr/bin/memory_pressure','-Q'])
    qualified=True
    for endpoint in [value['start'],value['end']]:
        need(set(endpoint)==set(commands),'Complete external observation inventory required')
        for name,command in commands.items():
            item=endpoint[name]
            need(set(item)=={'command','exit_code','output'} and exact(item['command'],command),'Named external command required')
            need(type(item['output']) is str,'External command output must be text')
            qualified &= type(item['exit_code']) is int and item['exit_code']==0
        power=endpoint['power']['output'].splitlines()
        qualified &= bool(power) and power[0]=="Now drawing from 'AC Power'"
        qualified &= endpoint['thermal']['output'].splitlines()==['Note: No thermal warning level has been recorded','Note: No performance warning level has been recorded','Note: No CPU power status has been recorded']
        pressure=re.findall(r'^System-wide memory free percentage: (\d+)%\s*$',endpoint['memory_pressure']['output'],re.M)
        qualified &= len(pressure)==1 and 0<=int(pressure[0])<=100
    for name in ['hardware','os','configuration']:
        need(exact(value['start'][name],value['end'][name]),'Environment endpoint drift')
    return qualified


def closure(receipt,root,inventory,limit):
    need(receipt['passed'] is True and exact(receipt['exit_code'],0) and receipt['error'] is None and receipt['group_absent'] is True,'Closed successful owned process required')
    peak=receipt['lifetime_peak_physical_bytes']
    need(type(peak) is int and 0<peak<=limit,'Measured lifetime physical peak exceeds scope')
    need(set(receipt['files_sha256'])==set(inventory),'Exact sealed evidence inventory required')
    for name in inventory:need(digest(root/name)==receipt['files_sha256'][name],'Sealed evidence differs: '+name)
    need(type(receipt['elapsed_seconds']) in [int,float] and 0<receipt['elapsed_seconds']<301,'Finite owned process duration required')


def reference(root,name):
    need(name in ['reference-1','reference-2'],'Named independent reference required')
    receipt=document(root/(name+'-checks.json'))
    inventory=[name+'-console.log',name+'-time.log',name+'-runtime.json',name+'/observation.json',name+'/records.jsonl']
    closure(receipt,root,inventory,1024**3)
    peak=re.findall(r'^\s*(\d+)\s+peak memory footprint\s*$',read(root/(name+'-time.log')).decode(),re.M)
    need(len(peak)==1 and int(peak[0])==receipt['lifetime_peak_physical_bytes'],'Reference peak/raw output differs')
    observation=document(root/name/'observation.json');need(exact(observation,receipt['observation']),'Reference observation/receipt differs')
    need(observation['passed'] is True and observation['primary_error'] is None and observation['endpoint_errors']==[],'Closed source/runtime endpoints required')
    for field,value in dict(format='toklet-qwen35-text-reference-v1',process_limit_bytes=1024**3,repository=producer.REPO,revision=producer.REVISION,assets=producer.PINS,records=100,
        model_inference_executed=False,codec_native_executed=False,streaming_qualified=False,quality_qualified=False,managed_adoption_qualified=False).items():
        need(exact(observation[field],value),'Reference scope differs: '+field)
    raw=read(root/name/'records.jsonl',1024**2)
    need(observation['bytes']==len(raw) and observation['sha256']==hashlib.sha256(raw).hexdigest()==FIXTURE_SHA,'Frozen original witness identity differs')
    rows=[json.loads(line,object_pairs_hook=producer.runtime.common.json_unique) for line in raw.splitlines()]
    need(exact(dict(Counter(row['kind'] for row in rows)),COUNTS),'Complete original witness inventory required')
    manifest=document(root/(name+'-runtime.json'));admission=observation['runtime_admission']
    need(manifest['format']=='toklet-text-reference-runtime-v1','Named text runtime manifest required')
    need(receipt['runtime_manifest_sha256']==admission['manifest_sha256']==digest(root/(name+'-runtime.json')),'Reference runtime manifest differs')
    need(exact(manifest['versions'],producer.runtime.VERSIONS) and exact(admission['versions'],producer.runtime.VERSIONS)
         and exact(admission['files'],len(manifest['files'])),'Exact runtime versions/inventory required')
    for endpoint in [admission['initial_origins'],observation['runtime_endpoint_origins']]:
        need(set(endpoint)=={'modules','non_system_native_images','system_native_images'},'Complete runtime origin endpoint required')
        need(all(path in manifest['files'] for path in endpoint['modules'].values()),'Imported origin absent from closure')
        need(all(exact(pin,manifest['files'].get(path)) for path,pin in endpoint['non_system_native_images'].items()),'Native runtime image absent from closure')
    return rows,receipt


def allocator_profile(inputs,report,receipt,process,contract_sha256):
    need(inputs['allocator_profile']==receipt['allocator_profile']=='apple_no_large_cache','Only explicit bounded allocator profile qualified')
    need(inputs['allocator_contract_sha256']==contract_sha256,'Frozen allocator contract differs')
    expected=dict(MallocLargeCache='0',MallocAggressiveMadvise=None,MallocNanoZone=None)
    need(exact(inputs['allocator_environment'],expected) and exact(report['allocator_environment'],expected),'Effective child allocator profile differs')
    # Parent settings may differ: the child must independently report the exact
    # declared profile after exec. Never reinterpret an inherited0 as absence.
    inherited=inputs['inherited_allocator_environment']
    need(type(inherited) is dict and set(inherited)==set(expected)
         and all(v is None or type(v) is str for v in inherited.values()),'Inherited allocator settings must be explicit')
    need(exact(process,receipt['process']) and set(process)=={'pid','group','parent_pid'} and all(type(v) is int and v>0 for v in process.values())
         and process['pid']==process['group'] and process['pid']!=process['parent_pid'],'Exact owned native process identity required')
    need(receipt['nonbudget_checks_passed'] is True and receipt['budget_rejected'] is False
         and exact(receipt['component_physical_limit_bytes'],LIMIT) and exact(receipt['engine_process_limit_bytes'],6*1024**3),'Unchanged component/engine caps required')


def native(folder,kind):
    need(kind in ['fidelity','baseline'],'Named native text phase required')
    receipt=document(folder/'checks.json');root=folder.parent
    inputs=document(folder/'inputs.json');version=inputs.get('format','v1')
    need(version in ['v1','toklet-qwen35-text-native-v2'],'Named native evidence format required')
    inventory=['console.log','time-and-errors.log','inputs.json','environment.json']
    if version!='v1':inventory.append('process.json')
    if kind=='baseline':need(version=='toklet-qwen35-text-native-v2','Explicit allocator/process baseline evidence required')
    closure(receipt,folder,inventory,LIMIT)
    need(receipt['kind']==kind and receipt['changed_inputs']==[],'Native scope/input changes')
    raw=read(folder/'time-and-errors.log').decode();peak=re.findall(r'^\s*(\d+)\s+peak memory footprint\s*$',raw,re.M)
    need(len(peak)==1 and int(peak[0])==receipt['lifetime_peak_physical_bytes'],'Native peak/raw output differs')
    build_file=inputs.get('build_file','native-build-1/checks.json')
    need(build_file in ['native-build-1/checks.json','native-build-2/checks.json','native-build-3/checks.json'],'Named native build required')
    build=document(root/build_file)
    need(build['passed'] is True and inputs['binary_sha256']==build['binary_sha256']==receipt['binary_sha256'],'Exact compiled native binary required')
    need(exact(inputs['source_pins'],build['source_pins']) and exact(receipt['source_pin_count'],len(build['source_pins'])),'Compiled source manifest differs')
    need(digest(Path(build['binary']))==build['binary_sha256'],'Retained native executable differs')
    need(inputs['contract_sha256']==digest(root/'acceptance-contract.json') and inputs['fixture_sha256']==FIXTURE_SHA,'Native frozen contract/fixture differs')
    need(exact(inputs['assets'],producer.PINS),'Native source assets differ')
    need(inputs['coupled_model_qualified'] is False,'Codec cannot qualify a model graph')
    test='actual_source_codec_and_streams' if kind=='fidelity' else 'matched_codec_phase_baseline'
    need(exact(inputs['command'],['/usr/bin/time','-l',build['binary'],'app_chat::qwen35::tests::'+test,'--exact','--ignored','--nocapture','--test-threads=1']),'Exact owned native test command required')
    log=read(folder/'console.log').decode();tag='QWEN35_TEXT_REPORT' if kind=='fidelity' else 'QWEN35_TEXT_BASELINE_REPORT'
    reports=[json.loads(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith(tag+' ')]
    need(len(reports)==1 and exact(reports[0],receipt['report']) and '1 passed; 0 failed' in log,'One matching native footer and successful test required')
    report=reports[0]
    common={'fixture_sha256','codec_allowance_bytes','stream_allowance_bytes','model_inference_executed','managed_adoption_qualified'}
    if version!='v1':common.add('allocator_environment')
    specific={'records','stream_schedules','stop_ids','text_vocab','padded_model_vocab','load_ns','all_text_ids_decode_streams_exact','padded_ids_rejected','controller_qualified'} if kind=='fidelity' else {'warmups','measurements','scope'}
    need(set(report)==common|specific,'Exact native report schema required')
    for field,value in dict(fixture_sha256=FIXTURE_SHA,codec_allowance_bytes=LIMIT,stream_allowance_bytes=4*1024**2,model_inference_executed=False,managed_adoption_qualified=False).items():
        need(exact(report[field],value),'Native report scope differs: '+field)
    if version!='v1':
        contract_file='allocator-profile-contract-2.json' if build_file=='native-build-3/checks.json' else 'allocator-profile-contract.json'
        need(inputs.get('allocator_contract_file','allocator-profile-contract.json')==contract_file,'Named allocator contract must match the native build')
        allocator_profile(inputs,report,receipt,document(folder/'process.json'),digest(root/contract_file))
    return receipt,report,log


def fidelity(root,folder):
    need(folder.parent==root,'One frozen text qualification task required')
    a,first=reference(root,'reference-1');b,second=reference(root,'reference-2');need(exact(a,b),'Independent original witnesses differ')
    receipt,report,_=native(folder,'fidelity')
    for field,value in dict(records=100,stream_schedules=165,stop_ids=[248044,248046],text_vocab=248077,padded_model_vocab=248320,
        all_text_ids_decode_streams_exact=True,padded_ids_rejected=True,controller_qualified=False).items():
        need(exact(report[field],value),'Exact complete native text scope required: '+field)
    need(type(report['load_ns']) is int and report['load_ns']>0,'Observed construction duration required')
    return dict(passed=True,allocator_profile=document(folder/'inputs.json').get('allocator_profile','not_captured_v1'),source_records=100,stream_schedules=165,reference_peaks_bytes=[first['lifetime_peak_physical_bytes'],second['lifetime_peak_physical_bytes']],native_peak_bytes=receipt['lifetime_peak_physical_bytes'],stop_ids=report['stop_ids'],
        text_codec_qualified=True,controller_qualified=False,model_context_qualified=False,request_ttft_qualified=False,instrumentation_overhead_qualified=False,quality_qualified=False,managed_adoption_qualified=False)


def samples(rows,prompt_ids,decode_ids):
    need(type(rows) is list and len(rows)==13,'Three warmups/ten measurements required')
    result={k:[] for k in ['load_ns','encode_ns','decode_ns','stream_ns']}
    for iteration,row in enumerate(rows):
        need(set(row)=={'iteration','warmup','load_ns','encode_ns','decode_ns','stream_ns','prompt_ids','decode_ids'},'Exact phase sample fields required')
        for key,value in dict(iteration=iteration,warmup=iteration<3,prompt_ids=prompt_ids,decode_ids=decode_ids).items():need(exact(row[key],value),'Frozen sample work/order differs: '+key)
        for key in result:
            need(type(row[key]) is int and row[key]>0,'Positive completed phase nanoseconds required')
            if iteration>=3:result[key].append(row[key])
    return {key:dict(samples_ns=v,median_ns=statistics.median(v),mad_ns=statistics.median(abs(x-statistics.median(v)) for x in v),minimum_ns=min(v),maximum_ns=max(v)) for key,v in result.items()}


def baseline(root,first,second):
    need(first.parent==second.parent==root and first!=second,'Two independent owned baseline folders required')
    rows,_=reference(root,'reference-1')
    prompt=next(r for r in rows if r['kind']=='chat' and r['name']=='code');decode=next(r for r in rows if r['kind']=='text' and r['input']=='Привет 世界 🙂')
    runs=[];all_inputs=[]
    for folder in [first,second]:
        receipt,report,log=native(folder,'baseline');need(exact(report['warmups'],3) and exact(report['measurements'],10),'Baseline repetition policy differs')
        need(report['scope']=='Repeated authenticated construction with warm filesystem cache; encode three-turn code chat, decode Unicode, one-ID stream including construction/finalization; no model/TTFT/overhead claim','Frozen phase timing scope differs')
        observed=[json.loads(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith('QWEN35_TEXT_SAMPLE ')]
        phase=samples(observed,len(prompt['ids']),len(decode['ids']))
        need(sum(sum(r[k] for k in phase) for r in observed)<receipt['elapsed_seconds']*1e9,'Phases exceed owned process duration')
        conditions=document(folder/'environment.json')
        runs.append(dict(phases=phase,peak_bytes=receipt['lifetime_peak_physical_bytes'],environment_qualified=environment(conditions)))
        all_inputs.append(document(folder/'inputs.json'))
    need(exact(all_inputs[0]['source_pins'],all_inputs[1]['source_pins']) and all_inputs[0]['binary_sha256']==all_inputs[1]['binary_sha256'],'Independent baseline build/source inputs differ')
    need(document(first/'process.json')['pid']!=document(second/'process.json')['pid'],'Confirmation must be a distinct owned process')
    need(all_inputs[0]['build_file']==all_inputs[1]['build_file'],'Confirmation build metadata differs')
    fidelity_path=root/('fidelity-hardened-final.json' if all_inputs[0]['build_file']=='native-build-3/checks.json' else 'fidelity-profile-final.json')
    need(document(fidelity_path)['passed'] is True and document(fidelity_path)['allocator_profile']=='apple_no_large_cache','Profile exact text fidelity prerequisite required')
    need(all(v['text_fidelity_sha256']==digest(fidelity_path) for v in all_inputs),'Pre-baseline profile fidelity binding differs')
    environments=[document(p/'environment.json') for p in [first,second]]
    need(all(exact(environments[0]['start'][name],environments[1]['start'][name]) for name in ['hardware','os','configuration']),'Confirmation device/OS/configuration differs')
    need(all(run['environment_qualified'] for run in runs),'Controlled AC/thermal/pressure observations required for phase baseline qualification')
    return dict(passed=True,allocator_profile='apple_no_large_cache',runs=runs,warmups_total=6,measured_samples_total=20,scope='Explicit MallocLargeCache=0 process profile: text construction/encode/decode/stream only; warm filesystem cache, no numerical model, request or speedup comparison',
        request_ttft_qualified=False,instrumentation_overhead_qualified=False,sustained_qualified=False,model_adoption_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('native',type=Path);parser.add_argument('--baseline-second',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=baseline(args.root,args.native,args.baseline_second) if args.baseline_second else fidelity(args.root,args.native)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result))
