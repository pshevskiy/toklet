"""Closed128/256 full-array fidelity and native bit/state recovery, no request route."""
import argparse
import hashlib
from pathlib import Path
import re
import check_qwen35_metal_context as gpu
import check_qwen35_metal_benchmark as environment_check
import qwen35_context256_reference as producer

prior=gpu.prior
need,exact,document,read=prior.need,prior.exact,prior.document,prior.read
LIMIT=6*1024**3
CONTRACT_SHA='e1fc2ee35043eb54ce0d604ad7cc7b676f43b6e55e8f7821cae74748c5b478d4'
FREEZE_SHA='4279693b3ac9de2c5531b1498dd35cad949e8566de47e344b28e043e04496775'
PROFILE_SHA='f34e38bbc1f818b38456bd7f9de58b7f508fa49ddeace8ce035e8f7747eba806'
TEST='model::qwen35::tests::metal::real::context256::actual_source_context256'


def digest(path):
    need(not path.is_symlink() and path.is_file(),'Regular sealed evidence required')
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def frozen(root):
    need(digest(root/'acceptance-contract.json')==CONTRACT_SHA and digest(root/'adapter-freeze.json')==FREEZE_SHA,
         'Frozen pre-implementation/collection contract differs')
    contract=document(root/'acceptance-contract.json');tokens,offset,values=producer.plan()
    need(exact(contract['tokens'],tokens) and exact(contract['checkpoints'],[128,256]) and exact(contract['position_offset'],offset)
         and exact(contract['gate'],producer.prior.graph.GATE) and exact(contract['reference_rows_bytes'],sum(values.values())*12),
         'Frozen context plan differs')
    return tokens,values


def closure(folder,success,inventory,deadline):
    receipt=document(folder/'checks.json')
    need(receipt[success] is True and exact(receipt['exit_code'],0) and receipt['error'] is None
         and receipt['group_absent'] is True,'Closed successful owned process required')
    peak=receipt['lifetime_peak_physical_bytes']
    need(type(peak) is int and 0<peak<=LIMIT,'Measured lifetime physical peak must fit6GiB')
    duration=receipt['elapsed_seconds']
    need(type(duration) in [int,float] and 0<duration<deadline+120,'Bounded owned duration required')
    need(set(receipt['files_sha256'])==set(inventory),'Complete sealed evidence inventory required')
    for name in inventory:need(digest(folder/name)==receipt['files_sha256'][name],'Evidence seal differs: '+name)
    peaks=re.findall(r'^\s*(\d+)\s+peak memory footprint\s*$',read(folder/'time-and-errors.log',2*1024**2).decode(),re.M)
    need(len(peaks)==1 and int(peaks[0])==peak,'Raw instrument peak differs')
    process=document(folder/'process.json')
    need(set(process)=={'pid','group','parent_pid'} and all(type(v) is int and v>0 for v in process.values())
         and process['pid']==process['group'] and process['pid']!=process['parent_pid'],'Exact owned process identity required')
    return receipt,process


def reference_rows(result,raw):
    import numpy as np
    need(np.__version__=='2.3.2','Qualified NumPy required')
    _,_,counts=producer.plan();offset=0;rows={};descriptors=result['arrays']
    need(type(descriptors) is list and len(descriptors)==4204,'Both complete independent profiles/checkpoints required')
    cursor=0
    for profile,width,dtype in [('ideal_fp64',8,'<f8'),('staged_fp32',4,'<f4')]:
        for n in [128,256]:
            wanted=prior.expected(n);arrays={};values=0
            for item in descriptors[cursor:cursor+1051]:
                need(set(item)=={'checkpoint','profile','token','layer','stage','count','offset','bytes','sha256'},
                     'Exact independent descriptor schema required')
                need(exact(item['checkpoint'],n) and item['profile']==profile,'Independent checkpoint/profile differs')
                key=prior.key(item,n-1)
                need(key in wanted and key not in arrays,'Unknown/duplicate independent stage')
                need(all(type(item[field]) is int for field in ['count','offset','bytes'])
                     and item['count']==wanted[key] and item['offset']==offset and item['bytes']==wanted[key]*width,
                     'Independent complete geometry differs')
                block=memoryview(raw)[offset:offset+item['bytes']]
                need(len(block)==item['bytes'] and hashlib.sha256(block).hexdigest()==item['sha256'],'Independent array seal differs')
                data=np.frombuffer(block,dtype=dtype);need(np.isfinite(data).all(),'Finite independent values required')
                arrays[key]=data;offset+=len(block);values+=len(data)
            need(set(arrays)==set(wanted) and values==counts[n],'Complete independent checkpoint inventory required')
            rows[profile,n]=arrays;cursor+=1051
    need(offset==len(raw),'No trailing independent bytes permitted')
    return rows


def runtime_origins(endpoint,imports,runtime,native):
    need(type(endpoint) is dict and set(endpoint)=={'numpy_modules','non_system_native_images','system_native_images'},
         'Complete runtime origin endpoint required')
    modules=endpoint['numpy_modules'];images=endpoint['non_system_native_images'];system=endpoint['system_native_images']
    need(type(modules) is dict and type(images) is dict and bool(images) and type(system) is list
         and all(type(path) is str and path.startswith(('/usr/lib/','/System/')) for path in system),
         'Typed runtime module/image inventory required')
    need(modules.get('numpy')==imports['numpy'] and 'numpy._core._multiarray_umath' in modules
         and all(type(name) is str and type(path) is str and path in runtime['files'] for name,path in modules.items()),
         'Actual NumPy root/core imports must match admission')
    allowed={**runtime['files'],**native['files']}
    need(all(type(path) is str and path in allowed and exact(pin,allowed[path]) for path,pin in images.items()),
         'Native runtime image outside admitted closure')


def owned_reference_process(actual,wrapper):
    need(type(actual) is dict and set(actual)=={'pid','group','parent_pid'}
         and all(type(value) is int and value>0 for value in actual.values())
         and actual['group']==wrapper['group'] and actual['parent_pid']==wrapper['pid']
         and actual['pid'] not in [wrapper['pid'],wrapper['parent_pid']],
         'Reference must be the owned instrument child in its admitted process group')


def reference(root,folder):
    tokens,values=frozen(root)
    inventory={'console.log','time-and-errors.log','inputs.json','runtime.json','native.json','process.json',
               'capture/result.json','capture/rows.bin'}
    receipt,process=closure(folder,'collection_passed',inventory,21600)
    need(receipt['changed_sources']==[],'Independent sources changed during collection')
    result=document(folder/'capture/result.json')
    need(result['completed'] is True and result['primary_error'] is None and result['endpoint_errors']==[],
         'Complete independent source/runtime endpoints required')
    fields=dict(format='toklet-qwen35-context256-reference-v1',repository=producer.prior.weights.REPOSITORY,
        revision=producer.prior.weights.REVISION,source_sha256=producer.prior.pin.FILE_SHA,tokens=tokens,
        position_offset=0,checkpoints=[128,256],geometry=producer.prior.weights.GEOMETRY,gate=producer.prior.graph.GATE,
        planned_rows_bytes=sum(values.values())*12,ledger=producer.LEDGER,engine_executed=False,
        contextual_numerical_qualified=False,request_metrics_qualified=False,quality_qualified=False,managed_adoption_qualified=False)
    need(set(result)==set(fields)|{'completed','arrays','runtime_admission','source','primary_error','endpoint_errors',
        'process','execution_environment','source_endpoint_sha256','rows_sha256','runtime_endpoint'},'Exact independent result schema required')
    for field,value in fields.items():
        need(exact(result[field],value),'Frozen independent scope differs: '+field)
    source=result['source']
    need(source['closed'] is True and source['poisoned'] is False and source['poison_reason'] is None
         and exact(source['bytes'],producer.prior.pin.FILE_BYTES)
         and source['sha256']==result['source_endpoint_sha256']==producer.prior.pin.FILE_SHA,
         'Closed authenticated source endpoint required')
    need(type(source['reads']) is int and source['reads']>0 and type(source['read_bytes']) is int
         and source['read_bytes']>=producer.prior.pin.FILE_BYTES,'Complete source audit required')
    inputs=document(folder/'inputs.json');freeze=document(root/'adapter-freeze.json')
    need(inputs['freeze_sha256']==FREEZE_SHA and inputs['contract_sha256']==CONTRACT_SHA
         and exact(inputs['deadline_seconds'],21600),'Pre-execution contract binding differs')
    need(inputs['driver_file']=='run-reference-2.py' and inputs['driver_sha256']==digest(root/'run-reference-2.py'),
         'Named independent driver differs')
    runtime=document(folder/'runtime.json');native=document(folder/'native.json')
    need(runtime['format']=='toklet-mlx-reference-runtime-v1' and exact(runtime['versions'],producer.prior.runtime_owner.runtime.VERSIONS),
         'Qualified reference package closure required')
    need(len(freeze['files'])==len(inputs['source_pins'])==8,'Exact frozen producer source inventory required')
    for relative,pin in freeze['files'].items():
        candidates=[p for p in inputs['source_pins'] if Path(p).parts[-len(Path(relative).parts):]==Path(relative).parts]
        need(len(candidates)==1 and inputs['source_pins'][candidates[0]]==pin and runtime['files'][candidates[0]]['sha256']==pin,
             'Frozen producer/runtime source differs')
    for name in ['runtime','native']:
        need(digest(folder/(name+'.json'))==inputs[name+'_manifest_sha256'],'Externally admitted runtime manifest differs')
    sources=[Path(path) for path in inputs['source_pins'] if Path(path).name=='qwen35_context256_reference.py']
    need(len(sources)==1,'Exact independent collector source required')
    work=sources[0].parents[2];canonical=root/'reference-2'
    command=['/usr/bin/time','-l',str(work/'.venv/bin/python'),'-s','-X','pycache_prefix='+str(canonical/'child-pycache'),
        str(sources[0]),'--model-directory',str(root.parent/'qwen35-4b-payload-2026-11-08/model'),
        '--output',str(work/'artifacts/qwen35-context256-reference-2-2026-11-09'),
        '--runtime-manifest',str(canonical/'runtime.json'),'--expected-runtime-sha256',inputs['runtime_manifest_sha256'],
        '--native-manifest',str(canonical/'native.json'),'--expected-native-sha256',inputs['native_manifest_sha256']]
    need(exact(inputs['command'],command),'Exact cold independent command required')
    admission=result['runtime_admission']
    need(admission['admitted']['manifest_sha256']==inputs['runtime_manifest_sha256']
         and exact(admission['admitted']['files'],len(runtime['files']))
         and admission['native_manifest_sha256']==inputs['native_manifest_sha256'],'Runtime admission differs')
    need(admission['admitted']['pycache_prefix']==str(canonical/'child-pycache')
         and set(admission['admitted']['imports'])=={'numpy','mlx.core','mlx.nn'}
         and all(path in runtime['files'] for path in admission['admitted']['imports'].values()),'Cold import/cache admission differs')
    for endpoint in [admission['initial_origins'],result['runtime_endpoint']]:
        runtime_origins(endpoint,admission['admitted']['imports'],runtime,native)
    expected_environment=dict(PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1',
                              MallocLargeCache='0',MallocNanoZone=None,MallocAggressiveMadvise=None)
    need(exact(result['execution_environment'],expected_environment)
         and exact(inputs['environment'],{key:expected_environment[key] for key in ['PYTHONNOUSERSITE','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']})
         and exact(inputs['allocator_environment'],{key:expected_environment[key] for key in ['MallocLargeCache','MallocNanoZone','MallocAggressiveMadvise']}),
         'Actual/declared owned reference environment differs')
    owned_reference_process(result['process'],process)
    inherited=inputs['inherited_allocator_environment']
    need(type(inherited) is dict and set(inherited)=={'MallocLargeCache','MallocNanoZone','MallocAggressiveMadvise'}
         and all(v is None or type(v) is str for v in inherited.values()),'Inherited reference environment must remain explicit')
    raw=read(folder/'capture/rows.bin',512*1024**2)
    need(len(raw)==sum(values.values())*12 and hashlib.sha256(raw).hexdigest()==result['rows_sha256'],
         'Complete independent body differs')
    return reference_rows(result,raw),receipt


def partition_calls(chunk):
    need(type(chunk) is int and chunk in [32,17,1],'Frozen replay partition required')
    result=[];token=0
    for end in [128,256]:
        while token<end:
            count=min(chunk,end-token);result.append(count);token+=count
    return result


def validate_partitions(partitions):
    need(type(partitions) is list and len(partitions)==3,'Three complete partition proofs required')
    for item,chunk in zip(partitions,[32,17,1],strict=True):
        need(set(item)=={'chunk','calls','full_logits_bits_exact','checkpoint_state_bits_exact','warmed_rust_allocations','forward_ns'}
             and exact(item['chunk'],chunk) and exact(item['calls'],partition_calls(chunk)) and item['full_logits_bits_exact'] is True
             and item['checkpoint_state_bits_exact'] is True and exact(item['warmed_rust_allocations'],0)
             and type(item['forward_ns']) is int and item['forward_ns']>0,'Complete checkpoint-aligned partition proof required')


def native(root,folder):
    tokens,values=frozen(root)
    inventory={'console.log','time-and-errors.log','inputs.json','environment.json','process.json','capture/rows.f32le','capture/logits.f32le'}
    receipt,process=closure(folder,'passed',inventory,1200)
    need(receipt['changed_inputs']==[] and exact(receipt['process'],process),'Native source/process binding differs')
    inputs=document(folder/'inputs.json')
    need(inputs['format']=='toklet-qwen35-context256-native-v1' and inputs['build_file']=='native-build-1/checks.json'
         and inputs['contract_sha256']==CONTRACT_SHA and inputs['producer_freeze_sha256']==FREEZE_SHA,'Frozen native scope differs')
    need(digest(root/'compiled-profile-contract.json')==PROFILE_SHA and inputs['profile_sha256']==PROFILE_SHA,'Frozen compiled profile differs')
    profile=document(root/'compiled-profile-contract.json');build=document(root/inputs['build_file'])
    need(build['passed'] is True and build['changed_inputs']==[]
         and digest(root/inputs['build_file'])==inputs['build_sha256']==profile['build_sha256']
         and digest(root/'native-build-1/compiler.jsonl')==profile['compiler_sha256'],'Exact native build required')
    need(profile['binary_sha256']==build['binary_sha256']==inputs['binary_sha256']==receipt['binary_sha256']
         and digest(Path(build['binary']))==build['binary_sha256'] and exact(build['source_pins'],inputs['source_pins'])
         and exact(receipt['source_pin_count'],len(build['source_pins'])),'Retained native binary/source pins differ')
    need(exact(profile['profile'],dict(opt_level='3',debuginfo=0,debug_assertions=False,overflow_checks=False,test=True))
         and exact(profile['features'],['default','metal']) and profile['rustflags']==[],'Qualified release profile required')
    need(exact(inputs['command'],['/usr/bin/time','-l',build['binary'],TEST,'--exact','--ignored','--nocapture','--test-threads=1'])
         and inputs['driver_file']=='run-native-2.py' and inputs['driver_sha256']==digest(root/'run-native-2.py')
         and exact(inputs['deadline_seconds'],1200) and exact(inputs['process_limit_bytes'],LIMIT)
         and exact(inputs['external_environment_sample_seconds'],30),'Exact native command/driver/limits required')
    log=read(folder/'console.log',4*1024**2).decode()
    reports=[prior.decode(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith('QWEN35_CONTEXT256_REPORT ')]
    need(len(reports)==1 and exact(reports[0],receipt['report']) and '1 passed; 0 failed' in log,'One matching successful native footer required')
    report=reports[0]
    fields=dict(source_sha256=producer.prior.pin.FILE_SHA,tokens=tokens,position_offset=0,capacity=512,chunk=32,
        diagnostic_allowance_bytes=256*1024**2,capture_bytes=sum(values.values())*4,capture_arrays=[1051,1051],
        capture_values=[values[128],values[256]],all_logit_bytes=256*248320*4,
        failed_work=dict(outcome='Error',tokens_completed=2,tokens_committed=0,projections_completed=502,gathers_completed=3,
                         convolution_steps_completed=49,recurrence_steps_completed=49,attention_heads_completed=256),
        cancelled_work={**gpu.work(0,'Cancelled'),'projections_completed':1,'gathers_completed':1},
        gpu_failed_work={**gpu.work(0,'Error'),'projections_completed':248,'gathers_completed':1,
                         'convolution_steps_completed':24,'recurrence_steps_completed':24,'attention_heads_completed':128},
        boundary_work=gpu.work(0,'Error'),final_work=gpu.work(0,'Idle'),failed_prefix_preserved=True,retry_bits_exact=True,
        repair_prefix_preserved=True,stale_future_nan_masked=True,reset_exact=True,admission_minus_one_rejected=True,
        request128plus128_required_capacity=256,request512plus128_rejected=True,request1024plus128_rejected=True,
        warmed_rust_allocations=0,independent_fidelity_qualified=False,request_metrics_qualified=False,
        instrumentation_overhead_qualified=False,quality_qualified=False,managed_adoption_qualified=False)
    extra={'memory','wrapper_memory','load_report','layout','model_owned_bytes','model_learned_bytes','execution_owned_bytes',
           'checkpoints','partitions','diagnostic_capture_ns','allocator_environment'}
    need(set(report)==set(fields)|extra,'Exact native footer scope/schema required')
    for field,value in fields.items():need(exact(report[field],value),'Native state/work contract differs: '+field)
    memory=gpu.memory(report)
    wrapper=dict(numerical=memory,pending_history_bytes=2048,wrapper_metadata_bytes=208,admitted_bytes=memory['admitted_bytes']+2256)
    need(exact(report['wrapper_memory'],wrapper) and wrapper['admitted_bytes']+256*1024**2<=LIMIT,'Numerical/wrapper/diagnostic owners must fit6GiB')
    allocator=dict(MallocLargeCache='0',MallocNanoZone=None,MallocAggressiveMadvise=None)
    need(exact(report['allocator_environment'],allocator) and exact(inputs['allocator_environment'],allocator),'Actual/declared allocator settings differ')
    inherited=inputs['inherited_allocator_environment']
    need(type(inherited) is dict and set(inherited)==set(allocator) and all(v is None or type(v) is str for v in inherited.values()),'Inherited allocator settings must remain explicit')
    partitions=report['partitions'];validate_partitions(partitions)
    need(type(report['diagnostic_capture_ns']) is int and report['diagnostic_capture_ns']>0
         and report['diagnostic_capture_ns']+sum(v['forward_ns'] for v in partitions)<receipt['elapsed_seconds']*1e9,'Diagnostic timing exceeds owned duration')
    descriptors=[prior.decode(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith('QWEN35_CONTEXT256_ARRAY ')]
    need(len(descriptors)==2102,'Both native checkpoint descriptors required')
    raw=read(folder/'capture/rows.f32le',256*1024**2);need(len(raw)==sum(values.values())*4,'Complete native checkpoint body required')
    rows={};origin=0
    for slot,n in enumerate([128,256]):
        selected=descriptors[slot*1051:(slot+1)*1051];adjusted=[]
        for item in selected:
            need(set(item)=={'checkpoint','token','position','layer','stage','head','count','offset','bytes','sha256'}
                 and exact(item['checkpoint'],n) and type(item['offset']) is int,'Exact native descriptor required')
            adjusted.append({**{k:v for k,v in item.items() if k!='checkpoint'},'offset':item['offset']-origin})
        size=values[n]*4;rows[n]=gpu.load_arrays(adjusted,memoryview(raw)[origin:origin+size],tokens[:n],0,prior.expected(n));origin+=size
    need(origin==len(raw),'No trailing native bytes permitted')
    checkpoints=report['checkpoints'];need(type(checkpoints) is list and len(checkpoints)==2,'Both committed state checkpoints required')
    logfile=folder/'capture/logits.f32le';need(logfile.stat().st_size==256*248320*4,'Complete bounded all-logit replay file required')
    with logfile.open('rb') as stream:
        for item,n in zip(checkpoints,[128,256],strict=True):
            need(set(item)=={'inputs','snapshot','greedy_lowest_id'} and exact(item['inputs'],n)
                 and type(item['snapshot']) is str and re.fullmatch('[a-f0-9]{64}',item['snapshot'])
                 and exact(item['greedy_lowest_id'],int(rows[n][None,'logits',None].argmax())),'Committed checkpoint state/greedy differs')
            stream.seek((n-1)*248320*4);need(stream.read(248320*4)==rows[n][None,'logits',None].tobytes(),'Replay file/captured full logits differ')
    conditions=environment_check.conditions(document(folder/'environment.json'),receipt['elapsed_seconds'])
    return rows,receipt,conditions


def check(root,native_folder,reference_folder):
    actual,native_receipt,conditions=native(root,native_folder);expected,reference_receipt=reference(root,reference_folder)
    rows=[];greedy=[]
    for n in [128,256]:
        for key,data in actual[n].items():
            for profile in ['ideal_fp64','staged_fp32']:
                want=expected[profile,n][key]
                rows.append(dict(checkpoint=n,profile=profile,layer=key[0],stage=key[1],head=key[2],**prior.statistics(data,want)))
                if key==(None,'logits',None):greedy.append(dict(checkpoint=n,profile=profile,native_id=int(data.argmax()),reference_id=int(want.argmax())))
    mismatches=[v for v in greedy if v['native_id']!=v['reference_id']];passed=all(v['failed_values']==0 for v in rows) and not mismatches
    return dict(context256_coupled_fidelity_passed=passed,checkpoints=[128,256],arrays=len(rows),compared_values=sum(v['values'] for v in rows),
        failed_values=sum(v['failed_values'] for v in rows),maximum_gate_ratio=max(v['maximum_gate_ratio'] for v in rows),rows=rows,greedy=greedy,
        unresolved_greedy_mismatches=mismatches,gate=producer.prior.graph.GATE,native_peak_bytes=native_receipt['lifetime_peak_physical_bytes'],
        reference_peak_bytes=reference_receipt['lifetime_peak_physical_bytes'],native_environment_qualified=conditions,
        request_metrics_qualified=False,instrumentation_overhead_qualified=False,performance_qualified=False,quality_qualified=False,
        managed_adoption_qualified=False,scope='Actual128/256 raw inputs at offset0; full checkpoint state/stage/logit fidelity and native all-logit bit replay/recovery only')


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('native',type=Path)
    parser.add_argument('reference',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=check(args.root,args.native,args.reference)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}));raise SystemExit(0 if result['context256_coupled_fidelity_passed'] else 1)
