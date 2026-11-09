"""Closed pinned9B128/256 all-value gates and native replay/recovery evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import check_qwen35_9b_reference as oracle
import check_qwen35_9b_context17 as shallow
import qwen35_9b_context256_reference as producer

need,exact,document,read=oracle.need,oracle.exact,oracle.document,oracle.read
LIMIT=6*1024**3
DIAGNOSTIC=320*1024**2
NATIVE_DEADLINE_SECONDS=3600
TEST='model::qwen35::tests::metal::nine_b_context256::actual_source_context256'
SOURCES=oracle.VECTOR_SOURCES|{'tools/reference/qwen35_9b_context256_reference.py'}
# These hashes bind this fixed scientific evaluation, not a general model loader.
CONTRACT_SHA='965bede1d068dff7aef1995045acc15b31c1f28745ca95a382b1631a3726352f'
FREEZE_SHA='0a70e5f01c44f00cd6c880510aab1f4b4e5b81cc369d6e91ccdf5160c9025b6a'
PROFILE_SHA='fbfc2afa26b620baea9c8161b3140e93ef93aaa7a250269990a117f12fbdcac4'
TIMES=['context_ns','cpu_load_ns','upload_ns','bind_ns','session_ns','observed_forward_ns','validation_ns','source_endpoint_ns']


def sha(path):return oracle.file_sha(path)


def frozen(root):
    need(sha(root/'acceptance-contract.json')==CONTRACT_SHA,
        'Pre-execution deeper contract/freeze differs')
    contract=document(root/'acceptance-contract.json');tokens,offset,values=producer.plan()
    fixed=dict(tokens=tokens,position_offset=offset,checkpoints=[128,256],values={str(n):v for n,v in values.items()},
        native_rows_bytes=sum(values.values())*4,reference_rows_bytes=sum(values.values())*12,
        reference_variant='ordered_vector_recurrence_context256',
        gate=dict(oracle.graph.GATE,profiles=['ideal_fp64','staged_fp32']))
    need(all(exact(contract.get(k),v) for k,v in fixed.items()),'Frozen deeper workload/gate differs')
    need(sha(Path(contract['approved_helper_publication']))==contract['approved_helper_publication_sha256'],
        'Approved exact-helper prerequisite changed')
    limits=contract['limits']
    need(exact(limits['mac_bytes'],LIMIT) and exact(limits['capacity'],512) and exact(limits['chunk'],32)
        and exact(limits['reference_capture_bytes'],producer.MAX_CAPTURE)
        and exact(limits['native_diagnostic_reserve_bytes'],DIAGNOSTIC)
        and exact(limits['reference_deadline_seconds'],producer.DEADLINE_SECONDS)
        and exact(limits['native_deadline_seconds'],NATIVE_DEADLINE_SECONDS),'Frozen deeper resource bounds differ')
    return tokens,offset,values


def runtime_origins(endpoint,admission,runtime,native):
    need(type(endpoint) is dict and set(endpoint)=={'numpy_modules','non_system_native_images','system_native_images'},
        'Complete runtime origin endpoint required')
    modules=endpoint['numpy_modules'];images=endpoint['non_system_native_images'];system=endpoint['system_native_images']
    need(type(modules) is dict and modules.get('numpy')==admission['imports']['numpy']
        and 'numpy._core._multiarray_umath' in modules
        and all(type(k) is str and type(v) is str and v in runtime['files'] for k,v in modules.items()),
        'Admitted NumPy root/core module origins required')
    allowed={**runtime['files'],**native['files']}
    need(type(images) is dict and images and all(p in allowed and exact(v,allowed[p]) for p,v in images.items()),
        'Admitted non-system native image origins required')
    need(type(system) is list and all(type(p) is str and p.startswith(('/usr/lib/','/System/')) for p in system),
        'System native image inventory differs')


def reference_rows(result,raw):
    import numpy as np
    _,_,counts=producer.plan();cursor=0;offset=0;rows={};items=result['arrays']
    need(type(items) is list and len(items)==4204,'Both complete independent checkpoint/profile inventories required')
    for profile,width,dtype in [('ideal_fp64',8,'<f8'),('staged_fp32',4,'<f4')]:
        for n in producer.CHECKPOINTS:
            wanted=oracle.adapter.expected_native(n);arrays={}
            for item in items[cursor:cursor+1051]:
                need(set(item)=={'checkpoint','profile','token','layer','stage','count','offset','bytes','sha256'},
                    'Exact independent checkpoint descriptor fields required')
                need(exact(item['checkpoint'],n) and item['profile']==profile,'Independent checkpoint/profile differs')
                key=oracle.common.key(item,n-1)
                need(key in wanted and key not in arrays,'Independent checkpoint owner differs')
                need(all(type(item[f]) is int for f in ['count','offset','bytes']) and item['count']==wanted[key]
                    and item['offset']==offset and item['bytes']==wanted[key]*width,'Complete independent checkpoint extent required')
                block=memoryview(raw)[offset:offset+item['bytes']]
                need(len(block)==item['bytes'] and hashlib.sha256(block).hexdigest()==item['sha256'],'Independent array digest differs')
                data=np.frombuffer(block,dtype=dtype);need(np.isfinite(data).all(),'Finite independent checkpoint required')
                arrays[key]=data;offset+=len(block)
            need(set(arrays)==set(wanted) and sum(len(v) for v in arrays.values())==counts[n],'Complete independent checkpoint values required')
            rows[profile,n]=arrays;cursor+=1051
    need(offset==len(raw),'Trailing/missing independent checkpoint bytes')
    return rows



def successful_reference_owned(folder,files):
    # The older one/context17 reader retains its256MiB cap. This separately
    # named workload admits only its bounded512MiB body and64MiB metadata files.
    receipt=document(folder/'checks.json')
    need(receipt.get('format')=='toklet-qwen35-9b-context256-reference-owned-v1'
        and receipt.get('passed') is True and receipt.get('group_absent') is True
        and receipt.get('error') is None and receipt.get('cleanup_errors')==[] and receipt.get('changed_inputs')==[],
        'Closed successful deeper reference owner required')
    need(type(receipt.get('exit_code')) is int and receipt['exit_code']==0,'Strict deeper reference zero exit required')
    elapsed=receipt.get('elapsed_seconds')
    need(type(elapsed) in [int,float] and oracle.math.isfinite(elapsed) and elapsed>0,'Finite deeper reference owned lifetime required')
    for name in ['lifetime_peak_physical_bytes','observed_conservative_charge_sum_peak_bytes']:
        need(type(receipt.get(name)) is int and 0<receipt[name]<=LIMIT,'Bounded deeper reference physical charge required')
    need(type(receipt.get('owned_pid')) is int and receipt['owned_pid']>0
        and type(receipt.get('native_snapshots')) is int and receipt['native_snapshots']>0,'Observed deeper reference process identity required')
    need(set(receipt.get('files_sha256',{}))==files,'Exact deeper reference sealed inventory required')
    for name,digest in receipt['files_sha256'].items():
        bound=producer.MAX_CAPTURE if name=='capture/rows.bin' else 64*1024**2
        need(hashlib.sha256(read(folder/name,bound)).hexdigest()==digest,'Sealed deeper reference evidence changed: '+name)
    peaks=re.findall(r'^\s*(\d+)\s+peak memory footprint\s*$',(folder/'time-and-errors.log').read_text(),re.M)
    need(len(peaks)==1 and int(peaks[0])==receipt['lifetime_peak_physical_bytes'],'Deeper reference instrument/receipt peaks differ')
    process=document(folder/'process.json')
    need(type(process.get('owned_pid')) is int and process['owned_pid']==process.get('group')==receipt['owned_pid'],
        'Deeper reference owner/process-group binding differs')
    oracle.live_memory(read(folder/'native-memory.jsonl',64*1024**2),receipt)
    return receipt


def reference(root,folder):
    tokens,offset,values=frozen(root)
    files={'console.log','time-and-errors.log','native-memory.jsonl','process.json','inputs.json','runtime.json','native.json','capture/result.json','capture/rows.bin'}
    receipt=successful_reference_owned(folder,files)
    need(receipt['elapsed_seconds']<producer.DEADLINE_SECONDS+720,'Bounded deeper reference owned lifetime required')
    inputs=document(folder/'inputs.json');freeze=document(root/'adapter-freeze.json');work=Path(inputs['source_worktree'])
    need(sha(root/'adapter-freeze.json')==inputs['freeze_sha256']==FREEZE_SHA and freeze['frozen_before_real_reference'] is True
        and freeze['reference_variant']=='vector_context256' and set(freeze['files'])==SOURCES
        and exact(inputs['source_pins'],freeze['files']) and exact(freeze['gate'],oracle.graph.GATE)
        and sha(Path(freeze['contract']))==freeze['contract_sha256']==CONTRACT_SHA,'Pre-execution deeper source freeze differs')
    need(all(sha(work/p)==h for p,h in freeze['files'].items()),'Deeper reference producer endpoints changed')
    base_freeze=dict(freeze,reference_variant='vector',files={p:h for p,h in freeze['files'].items() if p in oracle.VECTOR_SOURCES})
    _,fields=oracle.reference_variant('vector',base_freeze,work)
    need(sha(root/'independent-inputs-before-1.json')==receipt['prelaunch_inputs_sha256']==sha(folder/'inputs.json'),
        'Deeper reference prelaunch inputs differ')
    collectors={str(root/p) for p in ['run-independent-1.py','owned-oracle.py','prepare-reference-runtime.py']}|{str(work/'tools/experiments/native_usage.py')}
    need(set(inputs['collector_pins'])==collectors and all(sha(Path(p))==h for p,h in inputs['collector_pins'].items()),
        'Deeper reference collector endpoints differ')
    need(receipt['native_usage_sha256']==sha(work/'tools/experiments/native_usage.py'),'Reference native observer changed')
    for name in ['runtime','native']:
        need(sha(folder/(name+'.json'))==inputs[name+'_manifest_sha256']==sha(Path(inputs[name+'_manifest'])),
            'Admitted deeper runtime manifest differs')
    expected_command=['/usr/bin/time','-l',inputs['python'],'-s',str(work/'tools/reference/qwen35_9b_context256_reference.py'),
        '--model-directory',inputs['model_directory'],'--output',str(root/'independent-1/capture'),
        '--runtime-manifest',inputs['runtime_manifest'],'--expected-runtime-sha256',inputs['runtime_manifest_sha256'],
        '--native-manifest',inputs['native_manifest'],'--expected-native-sha256',inputs['native_manifest_sha256']]
    need(exact(inputs['command'],expected_command) and exact(inputs['deadline_seconds'],producer.DEADLINE_SECONDS),
        'Exact deeper reference command/deadline required')
    result=document(folder/'capture/result.json')
    fixed=dict(format='toklet-qwen35-9b-context256-reference-v1',reference_variant='ordered_vector_recurrence_context256',
        graph_provenance=fields['graph_provenance'],repository=oracle.adapter.REPOSITORY,revision=oracle.adapter.REVISION,
        source_sha256=oracle.adapter.pin.FILE_SHA,tokens=tokens,position_offset=offset,checkpoints=[128,256],geometry=oracle.adapter.GEOMETRY,
        gate=oracle.graph.GATE,planned_rows_bytes=sum(values.values())*12,ledger=producer.LEDGER,
        engine_executed=False,contextual_numerical_qualified=False,request_metrics_qualified=False,quality_qualified=False,managed_adoption_qualified=False)
    extra={'completed','arrays','runtime_admission','source','primary_error','endpoint_errors','process','execution_environment',
        'runtime_endpoint','profile_ns','source_open_and_bind_ns','source_endpoint_ns','collection_ns','source_endpoint_sha256','rows_sha256'}
    need(set(result)==set(fixed)|extra and result['completed'] is True and result['primary_error'] is None and result['endpoint_errors']==[],
        'Exact successful deeper reference result required')
    for k,v in fixed.items():need(exact(result[k],v),'Frozen deeper reference field differs: '+k)
    need(set(result['profile_ns'])=={'ideal_fp64','staged_fp32'},'Both deeper profile timings required')
    phases={**result['profile_ns'],**{k:result[k] for k in ['source_open_and_bind_ns','source_endpoint_ns']}}
    oracle.timing(phases,list(phases),receipt['elapsed_seconds'])
    need(type(result['collection_ns']) is int and sum(phases.values())<=result['collection_ns']<=receipt['elapsed_seconds']*1e9,
        'Deeper reference phase/collection/owned lifetime enclosure differs')
    audit=result['source']
    need(audit['closed'] is True and audit['poisoned'] is False and audit['poison_reason'] is None
        and exact(audit['bytes'],oracle.adapter.pin.FILE_BYTES) and audit['sha256']==result['source_endpoint_sha256']==oracle.adapter.pin.FILE_SHA
        and type(audit['reads']) is int and audit['reads']>0 and type(audit['read_bytes']) is int and audit['read_bytes']>=oracle.adapter.pin.FILE_BYTES,
        'Authenticated deeper source endpoint required')
    runtime=document(folder/'runtime.json');native=document(folder/'native.json');admission=result['runtime_admission']
    need(runtime['format']=='toklet-mlx-reference-runtime-v1' and exact(runtime['versions'],producer.prior.runtime_owner.runtime.VERSIONS)
        and admission['admitted']['manifest_sha256']==inputs['runtime_manifest_sha256']
        and exact(admission['admitted']['files'],len(runtime['files'])) and admission['native_manifest_sha256']==inputs['native_manifest_sha256'],
        'Admitted deeper Python/native runtime differs')
    need(set(admission['admitted']['imports'])=={'numpy','mlx.core','mlx.nn'}
        and all(p in runtime['files'] for p in admission['admitted']['imports'].values())
        and admission['admitted']['pycache_prefix']==str(root/'independent-pycache-1'),'Admitted imports/fresh bytecode cache differ')
    need(all(str(work/p) in runtime['files'] and runtime['files'][str(work/p)]['sha256']==h for p,h in freeze['files'].items()),
        'Frozen deeper sources must be in admitted runtime')
    for endpoint in [admission['initial_origins'],result['runtime_endpoint']]:runtime_origins(endpoint,admission['admitted'],runtime,native)
    child=inputs['child_environment'];expected_env=dict(MallocLargeCache='0',PYTHONNOUSERSITE='1',PYTHONPYCACHEPREFIX=str(root/'independent-pycache-1'),
        OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1')
    need(exact(child,expected_env) and exact(result['execution_environment'],dict(expected_env,MallocNanoZone=None,MallocAggressiveMadvise=None)),
        'Declared/actual deeper reference child environment differs')
    process=result['process'];wrapper=document(folder/'process.json')
    need(type(process) is dict and set(process)=={'pid','parent_pid','group'} and all(type(v) is int and v>0 for v in process.values())
        and process['group']==wrapper['group']==receipt['owned_pid'] and process['parent_pid']==wrapper['owned_pid']
        and process['pid'] not in [wrapper['owned_pid'],wrapper['parent_pid']],'Owned instrument child/process binding differs')
    raw=read(folder/'capture/rows.bin',producer.MAX_CAPTURE)
    need(len(raw)==sum(values.values())*12 and hashlib.sha256(raw).hexdigest()==result['rows_sha256'],'Complete deeper reference body differs')
    return reference_rows(result,raw),receipt


def partition_calls(chunk):
    need(type(chunk) is int and chunk in [32,17,1],'Frozen deeper replay partition required')
    calls=[];token=0
    for end in producer.CHECKPOINTS:
        while token<end:
            count=min(chunk,end-token);calls.append(count);token+=count
    return calls


def report_facts(report,abi,elapsed):
    tokens,offset,values=producer.plan();memory=shallow.MEMORY;execution=shallow.EXECUTION
    fixed=dict(phase='context256',source_sha256=oracle.adapter.pin.FILE_SHA,config_sha256=oracle.adapter.pin.CONFIG_SHA,
        header_sha256=oracle.adapter.pin.HEADER_SHA,source_bytes=oracle.adapter.pin.FILE_BYTES,source_endpoint_sha256=oracle.adapter.pin.FILE_SHA,
        tokens=tokens,position_offset=offset,capacity=512,chunk=32,config=oracle.adapter.GEOMETRY,load_report=shallow.UPLOAD,
        memory=memory,wrapper_memory=dict(numerical=memory,pending_history_bytes=2048,wrapper_metadata_bytes=208,admitted_bytes=memory['admitted_bytes']+2256),
        layout=abi['nine_b'],model_owned_bytes=7624,model_learned_bytes=5174465592,execution_owned_bytes=execution,distinct_head=True,
        diagnostic_allowance_bytes=DIAGNOSTIC,diagnostic_requested_bytes=254345296,all_logit_bytes=254279680,
        capture_bytes=sum(values.values())*4,capture_arrays=[1051,1051],capture_values=[values[128],values[256]],
        repair_peak_admitted_bytes=memory['admitted_bytes']+execution,
        failed_work=dict(outcome='Error',tokens_completed=2,tokens_committed=0,projections_completed=502,gathers_completed=3,
            convolution_steps_completed=49,recurrence_steps_completed=49,attention_heads_completed=256),
        cancelled_work=dict(shallow.work(0,'Cancelled'),projections_completed=1,gathers_completed=1),
        poisoned_work=dict(shallow.work(0,'Error'),projections_completed=248,gathers_completed=1,convolution_steps_completed=24,
            recurrence_steps_completed=24,attention_heads_completed=128),blocked_work=shallow.work(0,'Error'),rejected_work=shallow.work(0,'Error'),
        final_work=shallow.work(0,'Idle'),prefix_tokens=128,prefix_preserved=True,stale_future_nan_masked=True,
        repair_overlap_minus_one_rejected=True,retry_all_logits_bits_exact=True,retry_warmed_rust_allocations=0,
        reset_exact=True,admission_minus_one_rejected=True,request128plus128_required_capacity=256,request512plus128_rejected=True,
        request1024plus128_rejected=True,warmed_rust_allocations=0,whole_gpu_graph_qualified=False,independent_fidelity_qualified=False,
        request_metrics_qualified=False,instrumentation_overhead_qualified=False,quality_qualified=False,managed_adoption_qualified=False,
        allocator_environment=dict(MallocLargeCache='0',MallocNanoZone=None,MallocAggressiveMadvise=None))
    need(set(report)==set(fixed)|set(TIMES)|{'repair_ns','checkpoints','partitions','all_logits_sha256'},'Exact deeper native report fields required')
    for k,v in fixed.items():need(exact(report[k],v),'Frozen deeper native field differs: '+k)
    need(type(report['all_logits_sha256']) is str and re.fullmatch('[0-9a-f]{64}',report['all_logits_sha256']),
        'Full256-logit diagnostic digest required')
    need(sum(v for k,v in memory.items() if k!='admitted_bytes')==memory['admitted_bytes']
        and memory['admitted_bytes']+2256+DIAGNOSTIC<=LIMIT and report['repair_peak_admitted_bytes']+DIAGNOSTIC<=LIMIT,
        'Deeper numerical/wrapper/repair/diagnostic ownership exceeds6GiB')
    oracle.timing(report,TIMES,elapsed);oracle.timing(report,['repair_ns'],elapsed)
    parts=report['partitions'];need(type(parts) is list and len(parts)==3,'Three deeper replay partitions required')
    for row,chunk in zip(parts,[32,17,1],strict=True):
        need(set(row)=={'chunk','calls','full_logits_bits_exact','checkpoint_state_bits_exact','warmed_rust_allocations','forward_ns'}
            and exact(row['chunk'],chunk) and exact(row['calls'],partition_calls(chunk)) and row['full_logits_bits_exact'] is True
            and row['checkpoint_state_bits_exact'] is True and exact(row['warmed_rust_allocations'],0),'Complete deeper partition proof required')
        oracle.timing(row,['forward_ns'],elapsed)
    need(sum(p['forward_ns'] for p in parts)+report['repair_ns']<=report['validation_ns'],'Nested validation phase enclosure differs')
    checkpoints=report['checkpoints'];need(type(checkpoints) is list and len(checkpoints)==2,'Both deeper state checkpoints required')
    for row,n in zip(checkpoints,[128,256],strict=True):
        need(set(row)=={'inputs','snapshot','greedy_lowest_id'} and exact(row['inputs'],n)
            and type(row['snapshot']) is str and re.fullmatch('[0-9a-f]{64}',row['snapshot'])
            and type(row['greedy_lowest_id']) is int and 0<=row['greedy_lowest_id']<248320,'Typed deeper state/greedy checkpoint required')


def successful_native_owned(folder):
    files={'console.log','time-and-errors.log','native-memory.jsonl','process.json','inputs.json','capture/rows.f32le','owned-checks.json'}
    receipt=oracle.successful_owned(folder,'toklet-qwen35-9b-context256-native-v1',files)
    need(receipt['elapsed_seconds']<NATIVE_DEADLINE_SECONDS+120,'Bounded deeper native owned lifetime required')
    need(type(receipt['observed_native_lifetime_peak_bytes']) is int and 0<receipt['observed_native_lifetime_peak_bytes']<=LIMIT,
        'Strict bounded deeper native observed lifetime peak required')
    primary=document(folder/'owned-checks.json')
    core={'passed','exit_code','error','cleanup_errors','group_absent','owned_pid','elapsed_seconds',
        'lifetime_peak_physical_bytes','observed_native_lifetime_peak_bytes','observed_conservative_charge_sum_peak_bytes',
        'native_snapshots','native_usage_sha256','console_sha256','time_log_sha256','native_memory_log_sha256',
        'group_probe_fallbacks','live_guard_scope'}
    extra={'format','cohort','binary_sha256','changed_inputs','prelaunch_inputs_sha256','report',
        'owned_checks_sha256','postprocess_errors','files_sha256'}
    need(type(primary) is dict and set(primary)==core and set(receipt)==core|extra,
        'Exact primary/final deeper native receipt schema required')
    need(receipt['postprocess_errors']==[] and sha(folder/'owned-checks.json')==receipt['owned_checks_sha256'],
        'Successful linked deeper native postprocessing required')
    for key in core:need(exact(receipt[key],primary[key]),'Primary/final deeper native receipt differs: '+key)
    for name,key in [('console.log','console_sha256'),('time-and-errors.log','time_log_sha256'),('native-memory.jsonl','native_memory_log_sha256')]:
        need(primary[key]==receipt['files_sha256'][name],'Primary/raw deeper native digest differs: '+name)
    need(primary['live_guard_scope']=='Conservative sum of native physical charge counters for the owned group plus collector; not a deduplicated whole-machine footprint.',
        'Deeper native physical-charge scope differs')
    fallback=primary['group_probe_fallbacks']
    need(type(fallback) is list and all(type(row) is dict and set(row)=={'group','process_table_present'}
        and type(row['group']) is int and row['group']==primary['owned_pid'] and type(row['process_table_present']) is bool
        for row in fallback),'Typed deeper native group-probe observations required')
    return receipt


def native(root,folder,abi_path):
    import numpy as np
    _,offset,values=frozen(root)
    receipt=successful_native_owned(folder)
    need(type(receipt['cohort']) is int and receipt['cohort'] in [1,2],'Named deeper native cohort required')
    inputs=document(folder/'inputs.json');binary=Path(inputs['binary']);work=Path(inputs['source_worktree']);build_path=Path(inputs['source_inputs'])
    need(inputs['test']==TEST and exact(inputs['command'],['/usr/bin/time','-l',str(binary),TEST,'--exact','--ignored','--nocapture','--test-threads=1']),
        'Exact deeper native capture command required')
    need(exact(inputs['child_allocator'],dict(MallocLargeCache='0',MallocAggressiveMadvise=None,MallocNanoZone=None)),'Controlled deeper allocator differs')
    need(sha(binary)==inputs['binary_sha256']==receipt['binary_sha256'] and sha(build_path)==inputs['source_inputs_sha256'],
        'Deeper native binary/build manifest changed')
    build=document(build_path);pins=build['source_pins']
    need(build['passed'] is True and build['binary_sha256']==inputs['binary_sha256'] and type(inputs['source_pin_count']) is int
        and len(pins)==inputs['source_pin_count'] and all(sha(work/p)==h for p,h in pins.items()),'Compiled deeper source endpoints differ')
    profile_path=root/'compiled-profile-contract.json'
    need(sha(profile_path)==PROFILE_SHA,'Pre-capture compiled profile changed')
    profile=document(profile_path)
    need(profile['frozen_before_source_captures'] is True and profile['build_sha256']==sha(build_path)
        and profile['binary_sha256']==inputs['binary_sha256'] and exact(profile['source_inputs'],3967)
        and len(pins)==3967 and sha(build_path.parent/'compiler.jsonl')==profile['compiler_sha256']
        and sha(build_path.parent/'inputs.json')==profile['inputs_sha256'],'Frozen optimized build/profile binding differs')
    prelaunch=root/f"native-{receipt['cohort']}-inputs.json"
    need(sha(prelaunch)==receipt['prelaunch_inputs_sha256']==sha(folder/'inputs.json'),'Deeper native prelaunch inputs differ')
    collectors={str(root/p) for p in ['run-context256-capture-2.py','owned-native.py']}|{str(work/'tools/experiments/native_usage.py')}
    need(set(inputs['collector_pins'])==collectors and all(sha(Path(p))==h for p,h in inputs['collector_pins'].items())
        and receipt['native_usage_sha256']==sha(work/'tools/experiments/native_usage.py'),'Deeper native collector endpoints differ')
    abi=document(abi_path)
    need(abi['frozen_before_source_captures'] is True and abi['native_build_sha256']==sha(build_path)
        and all(sha(abi_path.parent/p)==h for p,h in abi['layout_receipts_sha256'].items()),'Pre-capture deeper ABI observation required')
    lines=(folder/'console.log').read_text().splitlines()
    reports=[oracle.common.decode(l.split(' ',1)[1].encode()) for l in lines if l.startswith('QWEN35_9B_CONTEXT256_REPORT ')]
    need(len(reports)==1 and exact(reports[0],receipt['report']) and 'test result: ok. 1 passed; 0 failed;' in '\n'.join(lines),
        'One successful matching deeper native report required')
    report=reports[0];report_facts(report,abi,receipt['elapsed_seconds'])
    items=[oracle.common.decode(l.split(' ',1)[1].encode()) for l in lines if l.startswith('QWEN35_9B_CONTEXT256_ARRAY ')]
    need(len(items)==2102,'Both full native checkpoint descriptors required')
    raw=read(folder/'capture/rows.f32le',256*1024**2);need(len(raw)==sum(values.values())*4,'Full deeper native body required')
    rows={};cursor=0;position=0
    for n,checkpoint in zip([128,256],report['checkpoints'],strict=True):
        wanted=oracle.adapter.expected_native(n);arrays={}
        for item in items[cursor:cursor+1051]:
            need(set(item)=={'checkpoint','token','position','layer','stage','head','count','offset','bytes','sha256'},
                'Exact deeper native descriptor required')
            need(all(type(item[k]) is int for k in ['checkpoint','token','position','count','offset','bytes'])
                and item['checkpoint']==n and item['token']==n-1 and item['position']==offset+n-1,'Frozen deeper native token/position differs')
            need((item['layer'] is None or type(item['layer']) is int) and (item['head'] is None or type(item['head']) is int),
                'Strict deeper native owner required')
            key=(item['layer'],item['stage'],item['head'])
            need(key in wanted and key not in arrays and item['count']==wanted[key] and item['offset']==position
                and item['bytes']==wanted[key]*4,'Complete deeper native extent required')
            block=memoryview(raw)[position:position+item['bytes']]
            need(len(block)==item['bytes'] and hashlib.sha256(block).hexdigest()==item['sha256'],'Deeper native array digest differs')
            data=np.frombuffer(block,dtype='<f4');need(np.isfinite(data).all(),'Finite deeper native values required')
            arrays[key]=data;position+=len(block)
        need(set(arrays)==set(wanted) and sum(len(v) for v in arrays.values())==values[n],'Complete deeper native checkpoint values required')
        need(int(arrays[None,'logits',None].argmax())==checkpoint['greedy_lowest_id'],'Deeper checkpoint greedy/tie report differs')
        rows[n]=arrays;cursor+=1051
    need(position==len(raw),'Trailing/missing native checkpoint bytes')
    return rows,receipt


def check(root,native_folder,reference_folder,abi_path):
    import numpy as np
    need(np.__version__=='2.3.2','Pinned NumPy2.3.2 required')
    actual,a=native(root,native_folder,abi_path);expected,b=reference(reference_folder.parent,reference_folder);rows=[];greedy=[]
    for n in [128,256]:
        for key,value in actual[n].items():
            for profile in ['ideal_fp64','staged_fp32']:
                want=expected[profile,n][key]
                rows.append(dict(checkpoint=n,profile=profile,layer=key[0],stage=key[1],head=key[2],**oracle.common.statistics(value,want)))
                if key==(None,'logits',None):greedy.append(dict(checkpoint=n,profile=profile,native_id=int(value.argmax()),reference_id=int(want.argmax())))
    mismatches=[r for r in greedy if r['native_id']!=r['reference_id']];failed=sum(r['failed_values'] for r in rows)
    return dict(format='toklet-qwen35-9b-context256-fidelity-v1',consumer_sha256=sha(Path(__file__)),
        contract_sha256=CONTRACT_SHA,reference_freeze_sha256=FREEZE_SHA,source_sha256=oracle.adapter.pin.FILE_SHA,
        native_checks_sha256=sha(native_folder/'checks.json'),reference_checks_sha256=sha(reference_folder/'checks.json'),
        context256_coupled_fidelity_passed=failed==0 and not mismatches,checkpoints=[128,256],arrays=len(rows),
        compared_values=sum(r['values'] for r in rows),failed_values=failed,maximum_gate_ratio=max(r['maximum_gate_ratio'] for r in rows),
        rows=rows,greedy=greedy,unresolved_greedy_mismatches=mismatches,gate=oracle.graph.GATE,
        native_capture_sha256=sha(native_folder/'capture/rows.f32le'),independent_result_sha256=sha(reference_folder/'capture/result.json'),
        native_peak_bytes=a['lifetime_peak_physical_bytes'],reference_peak_bytes=b['lifetime_peak_physical_bytes'],
        whole_gpu_graph_qualified=False,request_metrics_qualified=False,instrumentation_overhead_qualified=False,
        quality_qualified=False,managed_adoption_qualified=False,
        scope='Pinned9B128/256 raw inputs offset37; complete final-checkpoint stages/recurrent+KV state/logits, native all256-logit partitions and prefix128 recovery only; no request/quality/adoption scope')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('native',type=Path);p.add_argument('reference',type=Path)
    p.add_argument('--abi',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=check(a.root,a.native,a.reference,a.abi)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','greedy']}))
    raise SystemExit(0 if result['context256_coupled_fidelity_passed'] else 1)
