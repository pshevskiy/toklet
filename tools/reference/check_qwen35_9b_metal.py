"""Closed9B hybrid projection/gather + CPU-state one-input oracle comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import check_qwen35_9b_reference as oracle

need, exact, document, read = oracle.need, oracle.exact, oracle.document, oracle.read
LIMIT=6*1024**3
TEST='model::qwen35::tests::metal::nine_b_real::actual_source_one'
CPU_TENSOR_BYTES=5040162816
PACKED_BYTES=5035917312
SMALL_BYTES=4245504
POLICY=128*1024**2
TIMES=['context_ns','load_ns','upload_ns','bind_ns','observed_forward_ns','plain_forward_ns','repair_ns','source_endpoint_ns']
SCOPE='Hybrid GPU projection/gather with CPU state; diagnostic capture/replay/failure/repair only, no controlled performance or independent fidelity claim'


def report_facts(report, abi, elapsed):
    fixed=dict(source_sha256=oracle.adapter.pin.FILE_SHA,source_endpoint_sha256=oracle.adapter.pin.FILE_SHA,
        container_bytes=oracle.adapter.pin.FILE_BYTES,config_sha256=oracle.adapter.pin.CONFIG_SHA,header_sha256=oracle.adapter.pin.HEADER_SHA,
        tokens=[1],position_offset=0,capacity=512,chunk=32,phase='one',cpu_tensor_capacity_bytes=CPU_TENSOR_BYTES,
        layout=abi['nine_b'],distinct_head=True,arrays=1051,capture_bytes=66640896,work=oracle.WORK,
        warmed_rust_allocations=0,retry_warmed_rust_allocations=0,late_head_failure_prefix_preserved=True,
        cancelled_prefix_preserved=True,retry_bits_equal=True,poisoned_prefix_preserved=True,
        repair_overlap_minus_one_rejected=True,first_token_executed=True,independent_real_source_fidelity_qualified=False,
        whole_gpu_graph_qualified=False,contextual_numerical_qualified=False,quality_qualified=False,managed_adoption_qualified=False,timing_scope=SCOPE)
    extra={'memory','upload_report','embedding_slot','head_slot','execution_owned_bytes','late_head_failure_work',
           'poisoned_work','repair_peak_admitted_bytes','greedy_lowest_id'}
    need(set(report)==set(fixed)|set(TIMES)|extra,'Exact9B hybrid report fields required')
    for name,value in fixed.items():need(exact(report.get(name),value),'Frozen9B hybrid contract differs: '+name)
    for name in ['embedding_slot','head_slot','execution_owned_bytes','repair_peak_admitted_bytes','greedy_lowest_id']:
        need(type(report[name]) is int and report[name]>=0,'Strict9B hybrid owner/count required')
    need(report['embedding_slot']!=report['head_slot'] and {report['embedding_slot'],report['head_slot']}=={0,1},
         'Separate pinned9B head/embedding slots required')
    need(0<=report['greedy_lowest_id']<248320 and report['execution_owned_bytes']>0,'Completed hybrid output/workspace required')
    upload=report['upload_report']
    keys={'matrices','packed_gpu_bytes','small_cpu_capacity_bytes','retained_requested_bytes','retained_policy_bytes',
          'admitted_load_policy_bytes','maximum_upload_source_capacity_bytes','retained_cpu_matrix_bytes'}
    need(set(upload)==keys and all(type(v) is int and v>=0 for v in upload.values()),'Strict complete upload report required')
    need(upload['matrices']==250 and upload['retained_cpu_matrix_bytes']==0 and upload['small_cpu_capacity_bytes']==SMALL_BYTES,
         'Consumed250-matrix9B ownership differs')
    need(PACKED_BYTES<upload['packed_gpu_bytes']<PACKED_BYTES+2*1024**2,'Packed GPU data/metadata extent differs')
    need(upload['retained_requested_bytes']==upload['packed_gpu_bytes']+SMALL_BYTES+abi['nine_b']['metal_store'],
         'Retained original owners/metadata sum differs')
    need(upload['retained_policy_bytes']==upload['retained_requested_bytes']+POLICY
         and upload['retained_policy_bytes']<=upload['admitted_load_policy_bytes']<=LIMIT-128*1024**2,
         'GPU load/retained policy differs')
    need(upload['maximum_upload_source_capacity_bytes']==37748808,'Bounded upload source capacities differ')
    memory=report['memory'];need(set(memory)==set(oracle.MEMORY) and all(type(v) is int and v>0 for v in memory.values()),
         'Strict complete numerical memory report required')
    fixed_memory=dict(recurrent_bank_bytes=150994944,convolution_bank_bytes=7077888,kv_bytes=33554432,
        logits_bytes=63569920,history_bytes=2048,metadata_bytes=5184,metadata_allowance_bytes=64*1024**2)
    for name,value in fixed_memory.items():need(memory[name]==value,'Numerical owner geometry differs: '+name)
    model_meta=abi['nine_b']['model']+32*abi['nine_b']['layer']
    need(memory['weights_and_model_bytes']==upload['retained_policy_bytes']+model_meta,'Model metadata/backing count differs')
    need(memory['scratch_bytes']==578048+report['execution_owned_bytes'],'CPU plus GPU scratch sum differs')
    need(sum(v for k,v in memory.items() if k!='admitted_bytes')==memory['admitted_bytes']<=LIMIT-128*1024**2,
         'Numerical admission/diagnostic reserve differs')
    peak=memory['admitted_bytes']+report['execution_owned_bytes']
    need(report['repair_peak_admitted_bytes']==peak<=LIMIT-128*1024**2,'Repair overlap admission differs')
    failed=dict(oracle.WORK,outcome='Error',tokens_completed=0,tokens_committed=0)
    need(exact(report['late_head_failure_work'],failed),'Late observed-head failure prefix/work differs')
    need(exact(report['poisoned_work'],dict(failed,projections_completed=248)),'Late GPU-head poisoning prefix/work differs')
    oracle.timing(report,TIMES,elapsed)


def native(folder, abi_path):
    import numpy as np
    files={'console.log','time-and-errors.log','native-memory.jsonl','process.json','inputs.json','capture/rows.f32le'}
    receipt=oracle.successful_owned(folder,'toklet-qwen35-9b-metal-native-v1',files)
    need(receipt['case'] in ['source9b-1','source9b-2'],'Named9B source cohort required')
    inputs=document(folder/'inputs.json');binary=Path(inputs['binary']);root=Path(inputs['source_worktree']);build_path=Path(inputs['source_inputs'])
    need(inputs['case']==receipt['case'] and inputs['test']==TEST and inputs['command']==['/usr/bin/time','-l',str(binary),TEST,
         '--exact','--ignored','--nocapture','--test-threads=1'],'Exact9B hybrid command required')
    need(exact(inputs['child_allocator'],dict(MallocLargeCache='0',MallocAggressiveMadvise=None,MallocNanoZone=None)),
         'Controlled child allocator differs')
    need(oracle.file_sha(binary)==inputs['binary_sha256']==receipt['binary_sha256'],'Native binary changed')
    need(oracle.file_sha(build_path)==inputs['source_inputs_sha256'],'Compiled input manifest changed')
    build=document(build_path);need(build['passed'] is True and build['binary_sha256']==inputs['binary_sha256'],'Matching successful release build required')
    pins=build['source_pins'];need(type(inputs['source_pin_count']) is int and len(pins)==inputs['source_pin_count']
         and all(oracle.file_sha(root/p)==h for p,h in pins.items()),'Compiled source endpoints changed')
    prelaunch=build_path.parent.parent/(receipt['case']+'-inputs-before-1.json')
    need(oracle.file_sha(prelaunch)==receipt['prelaunch_inputs_sha256']==oracle.file_sha(folder/'inputs.json'),
         'Prelaunch9B hybrid inputs changed')
    collectors=inputs['collector_pins'];expected={str(build_path.parent.parent/'run-explicit-1.py'),str(build_path.parent.parent/'owned-metal.py'),str(root/'tools/experiments/native_usage.py')}
    need(set(collectors)==expected and all(oracle.file_sha(Path(p))==h for p,h in collectors.items()),'Collector endpoints changed')
    need(receipt['native_usage_sha256']==oracle.file_sha(root/'tools/experiments/native_usage.py'),'Native observation helper changed')
    abi=document(abi_path);need(abi['frozen_before_source_captures'] is True and abi['native_build_sha256']==oracle.file_sha(build_path),
         'Pre-capture ABI/build observation required')
    for path,digest in abi['layout_receipts_sha256'].items():need(oracle.file_sha(abi_path.parent/path)==digest,'ABI receipt changed')
    lines=(folder/'console.log').read_text().splitlines();need('test result: ok. 1 passed; 0 failed;' in '\n'.join(lines),'One completed hybrid test required')
    reports=[oracle.common.decode(l.split(' ',1)[1].encode()) for l in lines if l.startswith('QWEN35_9B_METAL_REPORT ')]
    need(len(reports)==1 and exact(reports[0],receipt['reports'].get('QWEN35_9B_METAL_REPORT')),'Sealed hybrid report differs')
    report=reports[0];report_facts(report,abi,receipt['elapsed_seconds'])
    descriptors=[oracle.common.decode(l.split(' ',1)[1].encode()) for l in lines if l.startswith('QWEN35_9B_METAL_ARRAY ')]
    wanted=oracle.adapter.expected_native();need(len(descriptors)==1051,'Complete hybrid stage inventory required')
    raw=read(folder/'capture/rows.f32le',128*1024**2);need(len(raw)==66640896,'Hybrid body length differs')
    rows={};offset=0
    for item in descriptors:
        need(set(item)=={'token','position','layer','stage','head','count','offset','bytes','sha256'},'Exact hybrid array descriptor required')
        for k in ['token','position','count','offset','bytes']:need(type(item[k]) is int,'Strict hybrid array count required')
        need(item['token']==item['position']==0 and type(item['stage']) is str,'Frozen hybrid position/stage differs')
        need(item['layer'] is None or type(item['layer']) is int,'Strict hybrid layer required')
        need(item['head'] is None or type(item['head']) is int,'Strict hybrid head required')
        key=(item['layer'],item['stage'],item['head'])
        need(key in wanted and key not in rows and item['count']==wanted[key] and item['offset']==offset
             and item['bytes']==wanted[key]*4,'Hybrid complete array owner/extent differs')
        block=memoryview(raw)[offset:offset+item['bytes']]
        need(len(block)==item['bytes'] and hashlib.sha256(block).hexdigest()==item['sha256'],'Hybrid array digest differs')
        values=np.frombuffer(block,dtype='<f4');need(np.isfinite(values).all(),'Nonfinite hybrid array')
        rows[key]=values;offset+=len(block)
    need(offset==len(raw) and set(rows)==set(wanted),'Trailing/missing hybrid data')
    need(int(rows[None,'logits',None].argmax())==report['greedy_lowest_id'],'Hybrid greedy/tie report differs')
    return rows,receipt


def check(folder, independent, abi_path):
    import numpy as np
    need(np.__version__=='2.3.2','Pinned NumPy2.3.2 required')
    actual,receipt=native(folder,abi_path);expected,ref_receipt=oracle.reference(independent)
    rows=[];greedy=[]
    for key,values in actual.items():
        for profile,arrays in expected.items():
            want=arrays[key];rows.append(dict(profile=profile,layer=key[0],stage=key[1],head=key[2],**oracle.common.statistics(values,want)))
            if key==(None,'logits',None):greedy.append(dict(profile=profile,native_id=int(values.argmax()),reference_id=int(want.argmax())))
    mismatch=[r for r in greedy if r['native_id']!=r['reference_id']];failed=sum(r['failed_values'] for r in rows)
    return dict(one_input_hybrid_fidelity_passed=failed==0 and not mismatch,compared_values=sum(r['values'] for r in rows),
        failed_values=failed,maximum_gate_ratio=max(r['maximum_gate_ratio'] for r in rows),rows=rows,greedy=greedy,
        unresolved_greedy_mismatches=mismatch,gate=oracle.graph.GATE,native_capture_sha256=oracle.file_sha(folder/'capture/rows.f32le'),
        independent_result_sha256=oracle.file_sha(independent/'capture/result.json'),abi_contract_sha256=oracle.file_sha(abi_path),
        native_peak_bytes=receipt['lifetime_peak_physical_bytes'],independent_peak_bytes=ref_receipt['lifetime_peak_physical_bytes'],
        whole_gpu_graph_qualified=False,contextual_numerical_qualified=False,quality_qualified=False,managed_adoption_qualified=False,
        scope='Pinned9B raw input1 offset0 only: Metal FP32 projections/gather plus CPU recurrence/attention/state, all final stages/states/logits under both unchanged profiles. No whole-GPU graph/context/codec/request/quality/adoption qualification.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('native',type=Path);p.add_argument('independent',type=Path);p.add_argument('--abi',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=check(a.native,a.independent,a.abi)
    with a.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','greedy']}));raise SystemExit(0 if result['one_input_hybrid_fidelity_passed'] else 1)
