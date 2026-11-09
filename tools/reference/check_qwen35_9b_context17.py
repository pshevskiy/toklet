"""Closed9B context17 hybrid inventory/state comparison under original profiles."""
import argparse
import hashlib
from pathlib import Path
import re
import check_qwen35_9b_reference as oracle

need,exact,document,read=oracle.need,oracle.exact,oracle.document,oracle.read
LIMIT=6*1024**3
TEST='model::qwen35::tests::metal::nine_b_context17::actual_source_context17'
TOKENS=list(oracle.graph.TOKENS)
VALUES=16926464
EXECUTION=41425068
UPLOAD=dict(matrices=250,packed_gpu_bytes=5036000192,small_cpu_capacity_bytes=4245504,
    retained_requested_bytes=5040247864,retained_policy_bytes=5174465592,admitted_load_policy_bytes=5784265128,
    maximum_upload_source_capacity_bytes=37748808,retained_cpu_matrix_bytes=0)
MEMORY=dict(weights_and_model_bytes=5174473216,recurrent_bank_bytes=150994944,convolution_bank_bytes=7077888,
    kv_bytes=33554432,scratch_bytes=42003116,logits_bytes=63569920,history_bytes=2048,
    metadata_bytes=5184,metadata_allowance_bytes=67108864,admitted_bytes=5538789612)
TIMES=['context_ns','cpu_load_ns','upload_ns','bind_ns','session_ns','observed_forward_ns','source_endpoint_ns']


def work(tokens,outcome='Success'):
    return dict(outcome=outcome,tokens_completed=tokens,tokens_committed=tokens,
        projections_completed=tokens*249,gathers_completed=tokens,convolution_steps_completed=tokens*24,
        recurrence_steps_completed=tokens*24,attention_heads_completed=tokens*128)


def report_facts(report,abi,elapsed):
    fixed=dict(phase='context17',source_sha256=oracle.adapter.pin.FILE_SHA,
        config_sha256=oracle.adapter.pin.CONFIG_SHA,header_sha256=oracle.adapter.pin.HEADER_SHA,
        source_bytes=oracle.adapter.pin.FILE_BYTES,source_endpoint_sha256=oracle.adapter.pin.FILE_SHA,
        distinct_head=True,tokens=TOKENS,position_offset=37,capacity=512,chunk=32,config=oracle.adapter.GEOMETRY,
        load_report=UPLOAD,model_owned_bytes=7624,model_learned_bytes=5174465592,memory=MEMORY,
        execution_owned_bytes=EXECUTION,layout=abi['nine_b'],observed_work=work(17),capture_bytes=VALUES*4,
        arrays=1051,reset_allocations=[0,0,0],all_logits_bits_exact=True,warmed_rust_allocations=0,
        admission_minus_one_rejected=True,whole_gpu_graph_qualified=False,independent_fidelity_qualified=False,
        quality_qualified=False,managed_adoption_qualified=False)
    need(set(report)==set(fixed)|set(TIMES)|{'partitions','recovery','greedy_lowest_id'},'Exact context17 hybrid report fields required')
    for name,value in fixed.items():need(exact(report[name],value),'Frozen context17 hybrid contract differs: '+name)
    need(type(report['greedy_lowest_id']) is int and 0<=report['greedy_lowest_id']<248320,'Strict contextual greedy ID required')
    need(sum(v for k,v in MEMORY.items() if k!='admitted_bytes')==MEMORY['admitted_bytes']<=LIMIT-128*1024**2,
         'Fixed context17 owner/diagnostic sum differs')
    need(UPLOAD['retained_requested_bytes']==UPLOAD['packed_gpu_bytes']+UPLOAD['small_cpu_capacity_bytes']+abi['nine_b']['metal_store']
         and MEMORY['weights_and_model_bytes']==UPLOAD['retained_policy_bytes']+abi['nine_b']['model']+32*abi['nine_b']['layer'],
         'Concrete store/model metadata differs')
    oracle.timing(report,TIMES,elapsed)
    parts=report['partitions'];need(type(parts) is list and len(parts)==3,'All contextual partitions required')
    snapshots=[];timings=[report[name] for name in TIMES]
    for chunk,row in zip([17,1,3],parts,strict=True):
        need(set(row)=={'chunk','snapshot','all_logits_bits_exact','warmed_rust_allocations','forward_ns'},'Exact partition report required')
        need(exact(row['chunk'],chunk) and row['all_logits_bits_exact'] is True and exact(row['warmed_rust_allocations'],0),
             'Bit-exact zero-allocation partitions required')
        need(type(row['snapshot']) is str and re.fullmatch('[0-9a-f]{64}',row['snapshot']) is not None,'State snapshot digest required')
        oracle.timing(row,['forward_ns'],elapsed);snapshots.append(row['snapshot']);timings.append(row['forward_ns'])
    need(len(set(snapshots))==1,'Partition state snapshots differ')
    recovery=report['recovery']
    expected=dict(prefix_tokens=5,prefix_preserved=True,stale_nonfinite_kv_masked=True,
        failed_work=dict(outcome='Error',tokens_completed=2,tokens_committed=0,projections_completed=502,
            gathers_completed=3,convolution_steps_completed=49,recurrence_steps_completed=49,attention_heads_completed=256),
        cancelled_work=dict(work(0,'Cancelled'),projections_completed=1,gathers_completed=1),
        poisoned_work=dict(work(1,'Error'),tokens_completed=0,tokens_committed=0,projections_completed=248),
        blocked_work=work(0,'Error'),retry_work=work(12),rejected_work=work(0,'Error'),
        repair_overlap_minus_one_rejected=True,repair_peak_admitted_bytes=MEMORY['admitted_bytes']+EXECUTION,
        retry_warmed_rust_allocations=0,retry_all_logits_bits_exact=True)
    need(type(recovery) is dict and set(recovery)==set(expected)|{'repair_ns'},'Complete nonzero-prefix recovery report required')
    for name,value in expected.items():need(exact(recovery[name],value),'Contextual recovery contract differs: '+name)
    oracle.timing(recovery,['repair_ns'],elapsed);timings.append(recovery['repair_ns'])
    need(sum(timings)<=elapsed*1e9,'All disjoint contextual phase times must fit owned lifetime')
    need(recovery['repair_peak_admitted_bytes']<=LIMIT-128*1024**2,'Contextual repair overlap exceeds reserve')


def native(folder,abi_path):
    import numpy as np
    files={'console.log','time-and-errors.log','native-memory.jsonl','process.json','inputs.json','capture/rows.f32le'}
    receipt=oracle.successful_owned(folder,'toklet-qwen35-9b-context17-native-v1',files)
    need(type(receipt['cohort']) is int and receipt['cohort'] in [1,2],'Named context17 native cohort required')
    inputs=document(folder/'inputs.json');binary=Path(inputs['binary']);root=Path(inputs['source_worktree']);build_path=Path(inputs['source_inputs'])
    need(inputs['test']==TEST and inputs['command']==['/usr/bin/time','-l',str(binary),TEST,'--exact','--ignored','--nocapture','--test-threads=1'],
         'Exact contextual native command required')
    need(exact(inputs['child_allocator'],dict(MallocLargeCache='0',MallocAggressiveMadvise=None,MallocNanoZone=None)),
         'Controlled contextual child allocator differs')
    need(oracle.file_sha(binary)==inputs['binary_sha256']==receipt['binary_sha256'],'Contextual native binary changed')
    need(oracle.file_sha(build_path)==inputs['source_inputs_sha256'],'Contextual build manifest changed')
    build=document(build_path);need(build['passed'] is True and build['binary_sha256']==inputs['binary_sha256'],'Successful contextual build required')
    pins=build['source_pins'];need(type(inputs['source_pin_count']) is int and inputs['source_pin_count']==len(pins)
        and all(oracle.file_sha(root/p)==h for p,h in pins.items()),'Contextual compiled inputs changed')
    prelaunch=build_path.parent.parent/f"native-{receipt['cohort']}-inputs.json"
    need(oracle.file_sha(prelaunch)==receipt['prelaunch_inputs_sha256']==oracle.file_sha(folder/'inputs.json'),'Prelaunch context17 inputs changed')
    collectors=inputs['collector_pins'];need(set(collectors)=={str(build_path.parent.parent/'run-context17-capture.py'),
        str(build_path.parent.parent/'owned-oracle.py'),str(root/'tools/experiments/native_usage.py')}
        and all(oracle.file_sha(Path(p))==h for p,h in collectors.items()),'Contextual collector endpoints changed')
    need(receipt['native_usage_sha256']==oracle.file_sha(root/'tools/experiments/native_usage.py'),'Contextual native observer changed')
    abi=document(abi_path);need(abi['frozen_before_source_captures'] is True and abi['native_build_sha256']==oracle.file_sha(build_path),
        'Pre-capture context17 ABI observation required')
    need(all(oracle.file_sha(abi_path.parent/p)==h for p,h in abi['layout_receipts_sha256'].items()),'Contextual ABI receipt changed')
    lines=(folder/'console.log').read_text().splitlines();need('test result: ok. 1 passed; 0 failed;' in '\n'.join(lines),'One completed contextual native test required')
    reports=[oracle.common.decode(l.split(' ',1)[1].encode()) for l in lines if l.startswith('QWEN35_9B_CONTEXT_REPORT ')]
    need(len(reports)==1 and exact(reports[0],receipt['report']),'Sealed contextual report differs');report=reports[0]
    report_facts(report,abi,receipt['elapsed_seconds'])
    descriptors=[oracle.common.decode(l.split(' ',1)[1].encode()) for l in lines if l.startswith('QWEN35_9B_CONTEXT_ARRAY ')]
    wanted=oracle.adapter.expected_native(17);need(len(descriptors)==len(wanted)==1051,'Complete contextual inventory required')
    raw=read(folder/'capture/rows.f32le',128*1024**2);need(len(raw)==VALUES*4,'Contextual body extent differs')
    rows={};offset=0
    for item in descriptors:
        need(set(item)=={'token','position','layer','stage','head','count','offset','bytes','sha256'},'Exact contextual descriptor required')
        for k in ['token','position','count','offset','bytes']:need(type(item[k]) is int,'Strict contextual descriptor integer required')
        need(item['token']==16 and item['position']==53,'Frozen contextual final token/position differs')
        need(item['layer'] is None or type(item['layer']) is int,'Strict contextual layer required')
        need(item['head'] is None or type(item['head']) is int,'Strict contextual head required')
        key=(item['layer'],item['stage'],item['head'])
        need(key in wanted and key not in rows and item['count']==wanted[key] and item['offset']==offset
             and item['bytes']==wanted[key]*4,'Complete contextual owner/extent differs')
        block=memoryview(raw)[offset:offset+item['bytes']];need(len(block)==item['bytes'] and hashlib.sha256(block).hexdigest()==item['sha256'],
             'Contextual array digest differs')
        data=np.frombuffer(block,dtype='<f4');need(np.isfinite(data).all(),'Nonfinite contextual native values')
        rows[key]=data;offset+=len(block)
    need(offset==len(raw) and set(rows)==set(wanted),'Missing/trailing contextual body')
    need(int(rows[None,'logits',None].argmax())==report['greedy_lowest_id'],'Contextual greedy/tie report differs')
    return rows,receipt


def check(folder,independent,abi_path):
    import numpy as np
    need(np.__version__=='2.3.2','Pinned NumPy2.3.2 required')
    actual,receipt=native(folder,abi_path);expected,reference=oracle.reference(independent,phase='context17')
    rows=[];greedy=[]
    for key,values in actual.items():
        for profile,arrays in expected.items():
            want=arrays[key];rows.append(dict(profile=profile,layer=key[0],stage=key[1],head=key[2],**oracle.common.statistics(values,want)))
            if key==(None,'logits',None):greedy.append(dict(profile=profile,native_id=int(values.argmax()),reference_id=int(want.argmax())))
    mismatches=[r for r in greedy if r['native_id']!=r['reference_id']];failed=sum(r['failed_values'] for r in rows)
    return dict(context17_hybrid_fidelity_passed=failed==0 and not mismatches,compared_values=sum(r['values'] for r in rows),
        failed_values=failed,maximum_gate_ratio=max(r['maximum_gate_ratio'] for r in rows),rows=rows,greedy=greedy,
        unresolved_greedy_mismatches=mismatches,gate=oracle.graph.GATE,native_capture_sha256=oracle.file_sha(folder/'capture/rows.f32le'),
        independent_result_sha256=oracle.file_sha(independent/'capture/result.json'),abi_contract_sha256=oracle.file_sha(abi_path),
        native_peak_bytes=receipt['lifetime_peak_physical_bytes'],independent_peak_bytes=reference['lifetime_peak_physical_bytes'],
        whole_gpu_graph_qualified=False,deeper_context_qualified=False,quality_qualified=False,managed_adoption_qualified=False,
        scope='Pinned9B hybrid context17 offset37 only; complete final stages/recurrent+KV states/logits under both original profiles. No deeper contexts/defaultallocator/codec/appfit/request/quality/adoption or4B128/256 rehabilitation.')


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('native',type=Path);p.add_argument('independent',type=Path)
    p.add_argument('--abi',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=check(a.native,a.independent,a.abi)
    with a.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','greedy']}));raise SystemExit(0 if result['context17_hybrid_fidelity_passed'] else 1)
