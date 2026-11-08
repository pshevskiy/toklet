"""Strict isolated Metal-projection/CPU-state coupled evidence; no managed route."""
import argparse
import hashlib
import math
from pathlib import Path
import re
import check_qwen35_real_reference as prior
import qwen35_real_reference as producer
import qwen35_real_weights as weights

need,exact,document,read=prior.need,prior.exact,prior.document,prior.read
LIMIT=6*1024**3
ABI=dict(model=184,layer=232,session=1344,delta_state=144,kv=48,projection=240,gather=160,metal_store=2168,gpu_weights=112)


def work(tokens,outcome='Success',committed=None):
    return dict(outcome=outcome,tokens_completed=tokens,tokens_committed=tokens if committed is None else committed,
        projections_completed=tokens*249,gathers_completed=tokens,convolution_steps_completed=tokens*24,
        recurrence_steps_completed=tokens*24,attention_heads_completed=tokens*128)


def closed(folder,kind='capture'):
    receipt=document(folder/'checks.json')
    need(receipt['passed'] is True and exact(receipt['exit_code'],0) and receipt['error'] is None
         and receipt['group_absent'] is True and receipt['changed_inputs']==[] and receipt['kind']==kind,
         'Closed successful owned native run required')
    peak=receipt['lifetime_peak_physical_bytes']
    need(type(peak) is int and 0<peak<=LIMIT,'Measured physical peak must fit6GiB')
    inventory={'console.log','time-and-errors.log','inputs.json','environment.json'}
    if kind=='capture':inventory.add('capture/rows.f32le')
    need(set(receipt['files_sha256'])==inventory,'Exact sealed native inventory required')
    for name,digest in receipt['files_sha256'].items():
        need(hashlib.sha256(read(folder/name,256*1024**2)).hexdigest()==digest,'Native evidence seal differs')
    peaks=re.findall(r'^\s*(\d+)\s+peak memory footprint\s*$',read(folder/'time-and-errors.log',2*1024**2).decode(),re.M)
    need(len(peaks)==1 and int(peaks[0])==peak,'Peak must equal raw instrument output')
    inputs=document(folder/'inputs.json');root=folder.parent
    need(inputs['binary_sha256']==receipt['binary_sha256'] and exact(receipt['source_pin_count'],len(inputs['source_pins'])), 'Native binary/input binding differs')
    abi_file=inputs.get('abi_layout_file','abi-layout.json')
    profile_file=inputs.get('build_profile_file','build-profile-2.json')
    build_file=inputs.get('build_file','native-build-2/checks.json')
    need(re.fullmatch(r'abi-layout(?:-3)?\.json',abi_file)
         and re.fullmatch(r'build-profile-[23]\.json',profile_file)
         and re.fullmatch(r'native-build-[23]/checks\.json',build_file),'Named qualified build metadata required')
    for field,file in [('contract_sha256','acceptance-contract.json'),('abi_layout_sha256',abi_file),('build_profile_sha256',profile_file)]:
        need(inputs[field]==hashlib.sha256(read(root/file,2*1024**2)).hexdigest(),'Frozen native metadata differs: '+field)
    abi=document(root/abi_file);profile=document(root/profile_file)
    build=document(root/build_file)
    need(build['passed'] is True and exact(build['source_pins'],inputs['source_pins'])
         and build['binary_sha256']==receipt['binary_sha256'],'Compiled native source manifest differs')
    need(abi['frozen_before_actual_source_numerical_collection'] is True and exact(abi['layout'],ABI)
         and abi['binary_sha256']==profile['binary_sha256']==receipt['binary_sha256'], 'Qualified arm64 ABI/binary differs')
    need(exact(profile['features'],['default','metal']) and exact(profile['rustflags'],[])
         and exact(profile['profile'],dict(opt_level='3',debuginfo=0,debug_assertions=False,overflow_checks=False,test=True)), 'Qualified release build profile differs')
    return receipt


def owners():
    matrices=[];small=0;conversion=0
    for name,(kind,shape) in sorted(producer.weights.graph.specs(weights.GEOMETRY).items()):
        count=math.prod(shape)
        if kind=='matrix':
            rows,columns=shape;chunks=math.ceil(rows/16384)
            packed=count//2+2*(count//64)*2
            # Qualified arm64 Chunk is Layout64 +3 Buffer(pointer,len)16.
            gpu=packed+chunks*(112+3*7*4)+ABI['gpu_weights']
            scratch=4*(columns+2*min(rows,16384)+rows+22)+ABI['projection']
            source=min(rows,16384)*columns*9//16+3*24
            matrices.append(dict(name=name,rows=rows,columns=columns,packed=packed,gpu=gpu,scratch=scratch,source=source))
        else:small+=count*4;conversion=max(conversion,count*2)
    need(len(matrices)==249,'Complete source matrix inventory required')
    packed=sum(v['packed'] for v in matrices);gpu=sum(v['gpu'] for v in matrices)
    retained=gpu+small+ABI['metal_store'];policy=128*1024**2
    remaining=packed+small;resident=0;peak=remaining+policy
    for matrix in matrices:
        peak=max(peak,remaining+resident+policy+matrix['gpu']+matrix['source'])
        remaining-=matrix['packed'];resident+=matrix['gpu']
    need(remaining==small,'Consumed CPU ownership must leave only small tensors')
    load=dict(matrices=249,packed_gpu_bytes=gpu,small_cpu_capacity_bytes=small,
        retained_requested_bytes=retained,retained_policy_bytes=retained+policy,
        admitted_load_policy_bytes=max(peak,retained+policy),
        maximum_upload_source_capacity_bytes=max(v['source'] for v in matrices),retained_cpu_matrix_bytes=0)
    return matrices,load,packed+small+64*1024**2+64*1024+conversion


def memory(report):
    geometry=weights.GEOMETRY;matrices,load,_=owners()
    need(exact(report['layout'],ABI) and exact(report['load_report'],load),'Native ABI/consuming-load ledger must reconcile')
    model_owned=ABI['model']+32*ABI['layer'];learned=load['retained_policy_bytes']
    need(exact(report['model_owned_bytes'],model_owned) and exact(report['model_learned_bytes'],learned),'Model views/backing ledger differs')
    h,i,v=geometry['hidden'],geometry['intermediate'],geometry['vocab']
    qw=geometry['query_heads']*geometry['head_dim'];kw=geometry['kv_heads']*geometry['head_dim']
    dk=geometry['key_heads']*geometry['key_dim'];dv=geometry['value_heads']*geometry['value_dim'];cv=2*dk+dv
    cpu_scratch=4*(6*h+3*i+3*cv+2*dk+4*dv+4*geometry['value_heads']+8*qw+4*kw+2*512)
    # Gather pipeline16MiB,3 guarded GPU buffers and2 host vectors. The embedded
    # Gather body belongs to Session metadata and is not counted twice here.
    gather=16*1024**2+4*(1+h+1+21+2*h)
    execution=sum(m['scratch'] for m in matrices)+gather
    wanted=dict(weights_and_model_bytes=learned+model_owned,
        recurrent_bank_bytes=24*dv*geometry['key_dim']*3*4,
        convolution_bank_bytes=24*cv*3*3*4,kv_bytes=8*kw*512*2*4,
        scratch_bytes=cpu_scratch+execution,logits_bytes=32*v*2*4,history_bytes=512*4,
        metadata_bytes=ABI['session']+24*ABI['delta_state']+8*ABI['kv'],metadata_allowance_bytes=64*1024**2)
    wanted['admitted_bytes']=sum(wanted.values())
    need(exact(report['memory'],wanted) and exact(report['execution_owned_bytes'],execution)
         and wanted['admitted_bytes']<=LIMIT-128*1024**2,'Every session owner must reconcile within6GiB')
    return wanted


def load_arrays(descriptors,body,tokens,offset,expected):
    arrays={};cursor=0
    import numpy as np
    need(np.__version__=='2.3.2','Qualified NumPy2.3.2 reader required')
    need(type(descriptors) is list and len(descriptors)==len(expected),'All native arrays required')
    for row in descriptors:
        need(set(row)=={'token','position','layer','stage','head','count','offset','bytes','sha256'},'Exact native array descriptor required')
        need(exact(row['token'],len(tokens)-1) and exact(row['position'],offset+len(tokens)-1),'Native observation coordinate differs')
        need((row['layer'] is None or type(row['layer']) is int and 0<=row['layer']<32)
             and (row['head'] is None or type(row['head']) is int and 0<=row['head']<16)
             and type(row['stage']) is str,'Strict native layer/head/stage types required')
        key=row['layer'],row['stage'],row['head']
        need(key in expected and key not in arrays,'Duplicate/unknown native array')
        for name,value in dict(count=expected[key],offset=cursor,bytes=expected[key]*4).items():need(exact(row[name],value),'Native array coverage differs')
        raw=memoryview(body)[cursor:cursor+row['bytes']];need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Native array seal differs')
        data=np.frombuffer(raw,dtype='<f4');need(np.isfinite(data).all(),'Finite complete native array required')
        arrays[key]=data;cursor+=len(raw)
    need(cursor==len(body) and set(arrays)==set(expected),'No missing/trailing native outputs allowed')
    return arrays


def native(folder,phase):
    receipt=closed(folder);report=receipt['report'];inputs=document(folder/'inputs.json')
    tokens,offset,values=producer.plan(phase)
    need(inputs['phase']==receipt['phase']==report['phase']==phase,'Frozen source workload phase differs')
    log=read(folder/'console.log',2*1024**2).decode()
    labels=lambda tag:[prior.decode(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith(tag+' ')]
    footers=labels('QWEN35_METAL_REPORT')
    need(len(footers)==1 and exact(footers[0],report),'Receipt/footer agreement required')
    for name,value in dict(source_sha256=producer.pin.FILE_SHA,config_sha256=producer.pin.CONFIG_SHA,header_sha256=producer.pin.HEADER_SHA,source_bytes=producer.pin.FILE_BYTES,
        tokens=tokens,position_offset=offset,capacity=512,chunk=32,config=weights.GEOMETRY,observed_work=work(len(tokens)),
        capture_bytes=values*4,arrays=1051,reset_allocations=[0,0,0],all_logits_bits_exact=True,warmed_rust_allocations=0,
        admission_minus_one_rejected=True,whole_gpu_graph_qualified=False,independent_fidelity_qualified=False,
        quality_qualified=False,managed_adoption_qualified=False).items():
        need(exact(report[name],value),'Frozen native field differs: '+name)
    memory(report)
    for name in ['context_ns','cpu_load_ns','upload_ns','bind_ns','session_ns','observed_forward_ns']:
        need(type(report[name]) is int and report[name]>0,'Completed native setup/diagnostic timing required')
    need(type(report['partitions']) is list and len(report['partitions'])==3,'Three partition proofs required')
    snapshots=[]
    for row,chunk in zip(report['partitions'],[17,1,3],strict=True):
        need(set(row)=={'chunk','snapshot','all_logits_bits_exact','warmed_rust_allocations','forward_ns'}
             and exact(row['chunk'],chunk) and row['all_logits_bits_exact'] is True
             and exact(row['warmed_rust_allocations'],0) and type(row['forward_ns']) is int and row['forward_ns']>0
             and type(row['snapshot']) is str and re.fullmatch('[a-f0-9]{64}',row['snapshot']), 'Partition proof differs')
        snapshots.append(row['snapshot'])
    need(len(set(snapshots))==1,'Committed state bits differ across partitions')
    if phase=='one':need(report['recovery'] is None,'One-input scope must not claim contextual recovery')
    else:
        recovery=report['recovery']
        failed=work(0,'Error',0);failed.update(tokens_completed=2,projections_completed=502,gathers_completed=3,convolution_steps_completed=49,recurrence_steps_completed=49,attention_heads_completed=256)
        cancelled=work(0,'Cancelled',0);cancelled.update(projections_completed=1,gathers_completed=1)
        poisoned=work(0,'Error',0);poisoned.update(projections_completed=248,gathers_completed=1,convolution_steps_completed=24,recurrence_steps_completed=24,attention_heads_completed=128)
        for name,value in dict(prefix_tokens=5,prefix_preserved=True,stale_nonfinite_kv_masked=True,failed_work=failed,cancelled_work=cancelled,
            poisoned_work=poisoned,blocked_work=work(0,'Error',0),retry_work=work(12),rejected_work=work(0,'Error',0),
            repair_overlap_minus_one_rejected=True,repair_peak_admitted_bytes=report['memory']['admitted_bytes']+report['execution_owned_bytes'],
            retry_warmed_rust_allocations=0,retry_all_logits_bits_exact=True).items():need(exact(recovery[name],value),'Recovery/work ledger differs: '+name)
        need(type(recovery['repair_ns']) is int and recovery['repair_ns']>0,'Completed cold repair timing required')
    body=read(folder/'capture/rows.f32le',128*1024**2);need(len(body)==values*4,'Complete native body required')
    arrays=load_arrays(labels('QWEN35_METAL_ARRAY'),body,tokens,offset,prior.expected(len(tokens)))
    need(exact(report['greedy_lowest_id'],int(arrays[(None,'logits',None)].argmax())),'Native lowest-ID greedy marker differs')
    return arrays,receipt


def check(folder,reference_folder,phase):
    actual,receipt=native(folder,phase);expected,peak=prior.reference(reference_folder,phase)
    rows=[];greedy=[]
    for profile,arrays in expected.items():
        for key,want in arrays.items():
            rows.append(dict(profile=profile,layer=key[0],stage=key[1],head=key[2],**prior.statistics(actual[key],want)))
        greedy.append(dict(profile=profile,native_id=int(actual[(None,'logits',None)].argmax()),reference_id=int(arrays[(None,'logits',None)].argmax())))
    mismatches=[v for v in greedy if v['native_id']!=v['reference_id']]
    return dict(coupled_projection_fidelity_passed=all(v['failed_values']==0 for v in rows) and not mismatches,
        phase=phase,compared_values=sum(v['values'] for v in rows),failed_values=sum(v['failed_values'] for v in rows),maximum_gate_ratio=max(v['maximum_gate_ratio'] for v in rows),
        rows=rows,greedy=greedy,unresolved_greedy_mismatches=mismatches,native_peak_bytes=receipt['lifetime_peak_physical_bytes'],independent_peak_bytes=peak,
        native_capture_sha256=receipt['files_sha256']['capture/rows.f32le'],
        independent_result_sha256=hashlib.sha256(read(reference_folder/'capture/result.json',2*1024**2)).hexdigest(),
        gate=producer.graph.GATE,whole_gpu_graph_qualified=False,request_ttft_qualified=False,sustained_qualified=False,quality_qualified=False,managed_adoption_qualified=False,
        scope='Full final stages/state/logits at frozen raw inputs; Metal projections/gather with CPU recurrent/attention state. No whole-GPUgraph,codec,quality or adoption qualification.')


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('native',type=Path);parser.add_argument('reference',type=Path);parser.add_argument('--phase',choices=['one','context17'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=check(args.native,args.reference,args.phase)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','greedy']}));raise SystemExit(0 if result['coupled_projection_fidelity_passed'] else 1)
