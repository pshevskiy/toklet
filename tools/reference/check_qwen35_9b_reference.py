"""Strict closed9B one-input all-stage/state/logit comparison; no adoption gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import check_qwen35_real_reference as common
import qwen35_9b_reference as producer
import qwen35_9b_weights as adapter
import qwen35_cpu_graph as graph

need, exact, document, read = adapter.need, common.exact, common.document, common.read
LIMIT = 6*1024**3
NATIVE_TEST = 'model::qwen35::tests::nine_b_real::actual_source_one'
MEMORY = dict(weights_and_model_bytes=5040195208, recurrent_bank_bytes=150994944,
    convolution_bank_bytes=7077888, kv_bytes=33554432, scratch_bytes=578048,
    logits_bytes=63569920, history_bytes=2048, metadata_bytes=4992,
    metadata_allowance_bytes=67108864, admitted_bytes=5363086344)
WORK = dict(outcome='Success', tokens_completed=1, tokens_committed=1,
    projections_completed=249, gathers_completed=1, convolution_steps_completed=24,
    recurrence_steps_completed=24, attention_heads_completed=128)


def file_sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def live_memory(raw, receipt):
    rows=[common.decode(line) for line in raw.splitlines()]
    need(len(rows)==receipt['native_snapshots'],'Live sample count differs')
    previous=-1;maximum=0;observed=0;owned_seen=False
    for row in rows:
        elapsed=row['elapsed_seconds']
        need(type(elapsed) in (int,float) and math.isfinite(elapsed)
             and 0<=elapsed<=receipt['elapsed_seconds'] and elapsed>=previous,
             'Live sample outside owned lifetime')
        previous=elapsed;members=row['members'];need(type(members) is list,'Live member list required')
        usages=[row['collector'],*[m['usage'] for m in members]]
        for usage in usages:
            need(set(usage)=={'resident_size','phys_footprint','lifetime_max_phys_footprint'}
                 and all(type(v) is int and v>=0 for v in usage.values()),'Strict native memory counter required')
        for member in members:
            need(type(member['pid']) is int and member['pid']>0 and type(member['ppid']) is int,'Strict live process identity required')
            owned_seen |= member['pid']==receipt['owned_pid']
            observed=max(observed,member['usage']['lifetime_max_phys_footprint'])
        charge=sum(usage['phys_footprint'] for usage in usages)
        need(type(row['conservative_charge_sum_bytes']) is int and row['conservative_charge_sum_bytes']==charge<=LIMIT,
             'Live conservative charge sum differs')
        maximum=max(maximum,charge)
    need(owned_seen and maximum==receipt['observed_conservative_charge_sum_peak_bytes']
         and observed==receipt['observed_native_lifetime_peak_bytes'],'Live observation/receipt counters differ')


def successful_owned(folder, format_name, files):
    receipt = document(folder/'checks.json')
    need(receipt.get('format') == format_name and receipt.get('passed') is True
         and receipt.get('group_absent') is True and receipt.get('error') is None
         and receipt.get('changed_inputs') == [] and receipt.get('cleanup_errors') == [],
         'Closed successful owned collection required')
    need(type(receipt.get('exit_code')) is int and receipt['exit_code'] == 0, 'Strict zero exit required')
    elapsed = receipt.get('elapsed_seconds')
    need(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0, 'Finite owned elapsed required')
    for name in ['lifetime_peak_physical_bytes','observed_conservative_charge_sum_peak_bytes']:
        need(type(receipt.get(name)) is int and 0<receipt[name]<=LIMIT, 'Bounded native physical charge required')
    need(type(receipt.get('owned_pid')) is int and receipt['owned_pid']>0, 'Owned process identity required')
    need(type(receipt.get('native_snapshots')) is int and receipt['native_snapshots']>0, 'Live native observations required')
    need(set(receipt.get('files_sha256',{})) == files, 'Exact sealed file inventory required')
    for name,digest in receipt['files_sha256'].items():
        need(hashlib.sha256(read(folder/name,producer.MAX_CAPTURE)).hexdigest()==digest,'Sealed evidence changed: '+name)
    peaks=re.findall(r'^\s*(\d+)\s+peak memory footprint\s*$',(folder/'time-and-errors.log').read_text(),re.M)
    need(len(peaks)==1 and int(peaks[0])==receipt['lifetime_peak_physical_bytes'],'Instrument/receipt peaks differ')
    process=document(folder/'process.json')
    need(type(process.get('owned_pid')) is int and process['owned_pid']==process.get('group')==receipt['owned_pid'],
         'Owned process/group binding differs')
    live_memory(read(folder/'native-memory.jsonl',64*1024**2),receipt)
    return receipt


def timing(report, fields, elapsed):
    need(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0,'Finite owned lifetime required')
    for name in fields:need(type(report.get(name)) is int and 0<report[name]<=elapsed*1e9,'Phase outside owned lifetime: '+name)
    need(sum(report[name] for name in fields)<=elapsed*1e9,'Phase sum exceeds owned lifetime')


def native_report(report, elapsed):
    fixed=dict(source_sha256=adapter.pin.FILE_SHA, source_endpoint_sha256=adapter.pin.FILE_SHA,
        container_bytes=adapter.pin.FILE_BYTES, config_sha256=adapter.pin.CONFIG_SHA,
        header_sha256=adapter.pin.HEADER_SHA, tokens=[1], position_offset=0, capacity=512, chunk=32,
        memory=MEMORY, work=WORK, capture_bytes=producer.VALUES*4, arrays=producer.ARRAYS,
        distinct_head=True, warmed_rust_allocations=0, late_head_failure_prefix_preserved=True,
        cancelled_prefix_preserved=True, retry_bits_equal=True, retry_warmed_rust_allocations=0,
        first_token_executed=True, independent_real_source_fidelity_qualified=False,
        quality_qualified=False, managed_adoption_qualified=False)
    phases=['load_ns','bind_ns','observed_forward_ns','plain_forward_ns','source_endpoint_ns']
    need(set(report)==set(fixed)|set(phases)|{'late_head_failure_work','greedy_lowest_id','timing_scope'},
         'Exact native report fields required')
    for name,value in fixed.items():need(exact(report.get(name),value),'Frozen native report differs: '+name)
    need(sum(v for k,v in MEMORY.items() if k!='admitted_bytes')==MEMORY['admitted_bytes']<=LIMIT-128*1024**2,
         'Native diagnostic reserve differs')
    failed=report['late_head_failure_work']
    need(set(failed)==set(WORK) and all(type(v) is int for k,v in failed.items() if k!='outcome')
         and failed.get('outcome')=='Error' and type(failed.get('projections_completed')) is int
         and failed['projections_completed']==249 and type(failed.get('tokens_committed')) is int
         and failed['tokens_committed']==0,'Completed late-head failure/prefix witness required')
    need(type(report.get('greedy_lowest_id')) is int and 0<=report['greedy_lowest_id']<248320,'Strict greedy ID required')
    need(report['timing_scope']=='CPU raw-input capture/replay and late-head failure/cancel/reset; no controlled performance or independent fidelity claim',
         'Native timing/qualification scope differs')
    timing(report,phases,elapsed)


def native(folder):
    import numpy as np
    files={'console.log','time-and-errors.log','native-memory.jsonl','process.json','inputs.json','capture/rows.f32le'}
    receipt=successful_owned(folder,'toklet-qwen35-9b-graph-native-v1',files)
    inputs=document(folder/'inputs.json');binary=Path(inputs['binary'])
    need(inputs['test']==NATIVE_TEST and inputs['command']==['/usr/bin/time','-l',str(binary),NATIVE_TEST,
         '--exact','--ignored','--nocapture','--test-threads=1'],'Exact native capture command required')
    need(inputs['child_allocator']==dict(MallocLargeCache='0',MallocAggressiveMadvise=None,MallocNanoZone=None),
         'Controlled child allocator differs')
    need(file_sha(binary)==inputs['binary_sha256']==receipt['binary_sha256'],'Native binary changed')
    source=Path(inputs['source_inputs']);need(file_sha(source)==inputs['source_inputs_sha256'],'Compiled input manifest changed')
    need(type(receipt.get('cohort')) is int and receipt['cohort'] in [1,2],'Named native cohort required')
    prelaunch=source.parent.parent/f"native-{receipt['cohort']}-inputs.json"
    need(file_sha(prelaunch)==receipt.get('prelaunch_inputs_sha256')==file_sha(folder/'inputs.json'),
         'Prelaunch native inputs changed')
    build=document(source);need(build['passed'] is True and build['binary_sha256']==inputs['binary_sha256'],'Successful matching release build required')
    pins=build['source_pins'];root=Path(inputs['source_worktree'])
    need(type(inputs['source_pin_count']) is int and len(pins)==inputs['source_pin_count']
         and all(file_sha(root/name)==digest for name,digest in pins.items()),'Native source endpoints changed')
    need(receipt['native_usage_sha256']==file_sha(root/'tools/experiments/native_usage.py'),'Native observation helper changed')
    collectors=inputs['collector_pins']
    need(set(collectors)=={str(source.parent.parent/'run-nine-b-capture.py'),str(source.parent.parent/'owned-oracle.py'),
         str(root/'tools/experiments/native_usage.py')} and all(file_sha(Path(p))==h for p,h in collectors.items()),
         'Native collector source endpoints changed')
    lines=(folder/'console.log').read_text().splitlines()
    need('test result: ok. 1 passed; 0 failed;' in '\n'.join(lines),'One completed native test required')
    reports=[common.decode(l.split(' ',1)[1].encode()) for l in lines if l.startswith('QWEN35_9B_GRAPH_REPORT ')]
    need(len(reports)==1 and exact(reports[0],receipt['report']),'Sealed native report differs')
    report=reports[0];native_report(report,receipt['elapsed_seconds'])
    descriptors=[common.decode(l.split(' ',1)[1].encode()) for l in lines if l.startswith('QWEN35_9B_GRAPH_ARRAY ')]
    wanted=adapter.expected_native();need(len(descriptors)==len(wanted)==1051,'Complete native observation inventory required')
    raw=read(folder/'capture/rows.f32le',128*1024**2);need(len(raw)==producer.VALUES*4,'Native capture size differs')
    offset=0;rows={}
    for item in descriptors:
        need(set(item)=={'token','position','layer','stage','head','count','offset','bytes','sha256'},'Exact native descriptor fields required')
        for k in ['token','position','count','offset','bytes']:need(type(item[k]) is int,'Strict native array integer required')
        need(item['token']==item['position']==0,'Frozen native position differs')
        need(item['layer'] is None or type(item['layer']) is int,'Strict native layer required')
        need(item['head'] is None or type(item['head']) is int,'Strict native head required')
        key=(item['layer'],item['stage'],item['head'])
        need(key in wanted and key not in rows and item['count']==wanted[key]
             and item['offset']==offset and item['bytes']==wanted[key]*4,'Native complete array extent/owner differs')
        block=memoryview(raw)[offset:offset+item['bytes']]
        need(len(block)==item['bytes'] and hashlib.sha256(block).hexdigest()==item['sha256'],'Native array digest differs')
        data=np.frombuffer(block,dtype='<f4');need(np.isfinite(data).all(),'Nonfinite native array')
        rows[key]=data;offset+=len(block)
    need(offset==len(raw) and set(rows)==set(wanted),'Native trailing/missing data')
    need(int(rows[None,'logits',None].argmax())==report['greedy_lowest_id'],'Native greedy/tie report differs')
    return rows,receipt


def reference(folder, phase='one'):
    tokens, position, planned_values = producer.plan(phase)
    import numpy as np
    files={'console.log','time-and-errors.log','native-memory.jsonl','process.json','inputs.json',
           'runtime.json','native.json','capture/result.json','capture/rows.bin'}
    receipt=successful_owned(folder,'toklet-qwen35-9b-reference-owned-v1',files)
    inputs=document(folder/'inputs.json');freeze_path=Path(inputs['freeze']);freeze=document(freeze_path)
    need(file_sha(freeze_path)==inputs['freeze_sha256'] and freeze['frozen_before_real_reference'] is True
         and exact(freeze['gate'],graph.GATE),'Pre-execution producer freeze differs')
    prelaunch=freeze_path.parent/'independent-inputs-before-1.json'
    need(file_sha(prelaunch)==receipt.get('prelaunch_inputs_sha256')==file_sha(folder/'inputs.json'),
         'Prelaunch independent inputs changed')
    need(file_sha(Path(freeze['contract']))==freeze['contract_sha256'],'Acceptance contract changed')
    need(exact(inputs['source_pins'],freeze['files']),'Reference producer inventory differs')
    root=Path(inputs['source_worktree'])
    need(all(file_sha(root/name)==digest for name,digest in freeze['files'].items()),'Reference producer endpoint changed')
    collectors=inputs['collector_pins']
    need(set(collectors)=={str(freeze_path.parent/'run-independent-1.py'),str(freeze_path.parent/'owned-oracle.py'),
         str(freeze_path.parent/'prepare-reference-runtime.py'),str(root/'tools/experiments/native_usage.py')}
         and all(file_sha(Path(p))==h for p,h in collectors.items()),'Independent collector endpoints changed')
    need(receipt['native_usage_sha256']==file_sha(root/'tools/experiments/native_usage.py'),'Independent native observer changed')
    for name in ['runtime','native']:
        need(file_sha(folder/(name+'.json'))==inputs[name+'_manifest_sha256']==file_sha(Path(inputs[name+'_manifest'])),
             'Runtime manifest changed')
    result=document(folder/'capture/result.json')
    need(result['completed'] is True and result['primary_error'] is None and result['endpoint_errors']==[],
         'Closed successful independent result required')
    fixed=dict(format='toklet-qwen35-9b-real-reference-v1',repository=adapter.REPOSITORY,revision=adapter.REVISION,
        source_sha256=adapter.pin.FILE_SHA,phase=phase,tokens=tokens,position_offset=position,geometry=adapter.GEOMETRY,
        gate=graph.GATE,planned_rows_bytes=planned_values*12,ledger=producer.LEDGER,engine_executed=False,
        contextual_numerical_qualified=False,quality_qualified=False,managed_adoption_qualified=False)
    need(set(result)==set(fixed)|{'completed','arrays','runtime_admission','source','primary_error',
         'endpoint_errors','source_endpoint_sha256','rows_sha256','runtime_endpoint','profile_ns',
         'source_open_and_bind_ns','source_endpoint_ns','collection_ns'},'Exact reference result fields required')
    for name,value in fixed.items():need(exact(result.get(name),value),'Frozen reference contract differs: '+name)
    need(set(result['profile_ns'])=={'ideal_fp64','staged_fp32'},'Both completed profile times required')
    phases={**result['profile_ns'],**{k:result[k] for k in ['source_open_and_bind_ns','source_endpoint_ns']}}
    timing(phases,list(phases),receipt['elapsed_seconds'])
    need(type(result['collection_ns']) is int and sum(phases.values())<=result['collection_ns']<=receipt['elapsed_seconds']*1e9,
         'Independent phase/collection/owned timing enclosure differs')
    need(inputs['command']==['/usr/bin/time','-l',inputs['python'],'-s',str(root/'tools/reference/qwen35_9b_reference.py'),
         '--model-directory',inputs['model_directory'],'--output',str(freeze_path.parent/'independent-1/capture'),
         '--runtime-manifest',inputs['runtime_manifest'],'--expected-runtime-sha256',inputs['runtime_manifest_sha256'],
         '--native-manifest',inputs['native_manifest'],'--expected-native-sha256',inputs['native_manifest_sha256'],'--phase',phase],
         'Exact independent command required')
    audit=result['source']
    need(audit['closed'] is True and audit['poisoned'] is False and audit['poison_reason'] is None
         and type(audit['bytes']) is int and audit['bytes']==adapter.pin.FILE_BYTES
         and audit['sha256']==result['source_endpoint_sha256']==adapter.pin.FILE_SHA,'Authenticated source closure differs')
    need(type(audit['reads']) is int and audit['reads']>0 and type(audit['read_bytes']) is int
         and audit['read_bytes']>=adapter.pin.FILE_BYTES,'Source read audit differs')
    runtime=document(folder/'runtime.json');native_manifest=document(folder/'native.json');allowed={**runtime['files'],**native_manifest['files']}
    need(result['runtime_admission']['admitted']['manifest_sha256']==inputs['runtime_manifest_sha256']
         and result['runtime_admission']['native_manifest_sha256']==inputs['native_manifest_sha256'],'Runtime admission binding differs')
    source_entries={str(root/name):digest for name,digest in freeze['files'].items()}
    need(all(path in runtime['files'] and runtime['files'][path]['sha256']==digest for path,digest in source_entries.items()),
         'Frozen producer outside admitted runtime')
    endpoint=result['runtime_endpoint'];need(endpoint['numpy_modules'] and endpoint['non_system_native_images'],'Runtime endpoint origins missing')
    need(all(path in allowed and exact(value,allowed[path]) for path,value in endpoint['non_system_native_images'].items())
         and all(path in runtime['files'] for path in endpoint['numpy_modules'].values()),'Runtime endpoint origin differs')
    raw=read(folder/'capture/rows.bin',producer.MAX_CAPTURE)
    need(len(raw)==planned_values*12 and hashlib.sha256(raw).hexdigest()==result['rows_sha256'],'Independent body differs')
    need(type(result['arrays']) is list and len(result['arrays'])==2*1051,'Full independent observation inventory required')
    wanted=adapter.expected_native(len(tokens));rows={};offset=0
    for index,(profile,width,dtype) in enumerate([('ideal_fp64',8,'<f8'),('staged_fp32',4,'<f4')]):
        arrays={}
        for item in result['arrays'][index*1051:(index+1)*1051]:
            need(set(item)=={'profile','token','layer','stage','count','offset','bytes','sha256'},'Exact independent descriptor required')
            key=common.key(item,token=len(tokens)-1)
            need(item['profile']==profile and key in wanted and key not in arrays,'Independent owner/profile differs')
            for field in ['count','offset','bytes']:need(type(item[field]) is int,'Strict independent extent required')
            need(item['count']==wanted[key] and item['offset']==offset and item['bytes']==wanted[key]*width,'Independent extent differs')
            block=memoryview(raw)[offset:offset+item['bytes']]
            need(len(block)==item['bytes'] and hashlib.sha256(block).hexdigest()==item['sha256'],'Independent array digest differs')
            data=np.frombuffer(block,dtype=dtype);need(np.isfinite(data).all(),'Nonfinite reference array')
            arrays[key]=data;offset+=len(block)
        need(set(arrays)==set(wanted),'Missing independent stages');rows[profile]=arrays
    need(offset==len(raw),'Independent trailing bytes')
    return rows,receipt


def check(native_folder, reference_folder):
    import numpy as np
    need(np.__version__=='2.3.2','Pinned NumPy2.3.2 required')
    actual,native_receipt=native(native_folder);expected,reference_receipt=reference(reference_folder)
    rows=[];greedy=[]
    for key,values in actual.items():
        for profile,arrays in expected.items():
            want=arrays[key];rows.append(dict(profile=profile,layer=key[0],stage=key[1],head=key[2],**common.statistics(values,want)))
            if key==(None,'logits',None):greedy.append(dict(profile=profile,native_id=int(values.argmax()),reference_id=int(want.argmax())))
    mismatches=[row for row in greedy if row['native_id']!=row['reference_id']]
    failed=sum(row['failed_values'] for row in rows)
    return dict(one_input_coupled_fidelity_passed=failed==0 and not mismatches,compared_values=sum(row['values'] for row in rows),
        failed_values=failed,maximum_gate_ratio=max(row['maximum_gate_ratio'] for row in rows),rows=rows,greedy=greedy,
        unresolved_greedy_mismatches=mismatches,gate=graph.GATE,native_capture_sha256=file_sha(native_folder/'capture/rows.f32le'),
        independent_result_sha256=file_sha(reference_folder/'capture/result.json'),native_peak_bytes=native_receipt['lifetime_peak_physical_bytes'],
        independent_peak_bytes=reference_receipt['lifetime_peak_physical_bytes'],contextual_numerical_qualified=False,
        quality_qualified=False,managed_adoption_qualified=False,
        scope='Pinned9B raw input1 offset0 all final stages/recurrent+KV state/full logits under both unchanged profiles only; no context/GPU/codec/request/quality or adoption qualification')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('native',type=Path);p.add_argument('reference',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=check(a.native,a.reference)
    with a.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','greedy']}))
    raise SystemExit(0 if result['one_input_coupled_fidelity_passed'] else 1)
