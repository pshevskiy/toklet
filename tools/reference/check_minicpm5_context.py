"""Bounded original-gate comparisons; completed rejection is not qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import minicpm5_context_reference as producer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import minicpm5_profile as phases
import minicpm5_performance as phase_common

need=producer.need
COUNTS=producer.COUNTS


def integer(value,minimum=0):return type(value) is int and value>=minimum
def sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def same(a,b):return type(a) is type(b) and json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def sources():
    return [*producer.sources(),Path(__file__).resolve(),ROOT/'test_check_minicpm5_context.py',
            Path(phases.__file__).resolve(),Path(phase_common.__file__).resolve()]


class Raw:
    def __init__(self,path,pin,maximum):
        need(type(pin) is dict and set(pin)=={'bytes','sha256'} and integer(pin['bytes'],1)
             and pin['bytes']<=maximum and sha(pin['sha256']),'Bounded pinned binary required')
        self.path,self.pin=path,pin
        self.fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
        try:self.verify()
        except BaseException:os.close(self.fd);raise

    def verify(self):
        info=os.fstat(self.fd)
        need(stat.S_ISREG(info.st_mode) and info.st_size==self.pin['bytes'],'Consumed binary extent/type differs')
        digest=hashlib.sha256()
        for start in range(0,info.st_size,4*1024**2):digest.update(self.read(start,min(4*1024**2,info.st_size-start)))
        current=self.path.lstat()
        need(digest.hexdigest()==self.pin['sha256'] and stat.S_ISREG(current.st_mode)
             and (current.st_dev,current.st_ino,current.st_size)==(info.st_dev,info.st_ino,info.st_size),
             'Consumed binary descriptor/path digest differs')

    def read(self,offset,size):
        need(integer(offset) and integer(size) and offset+size<=self.pin['bytes'],'Binary slice outside admitted extent')
        raw=os.pread(self.fd,size,offset);need(len(raw)==size,'Short binary slice');return raw

    def close(self):os.close(self.fd)


def lines(path,maximum,pin=None):
    consumed=Raw(path,pin,128*1024**2) if pin is not None else None
    try:
        with (os.fdopen(os.dup(consumed.fd),'rb') if consumed else path.open('rb')) as stream:
            for index in range(maximum+1):
                raw=stream.readline(65537)
                if not raw:break
                need(index<maximum and len(raw)<=65536 and raw.endswith(b'\n'),'Bounded complete journal lines required')
                yield producer.parsed(raw)
        if consumed:consumed.verify()
    finally:
        if consumed:consumed.close()


def evidence(path,external_sha):
    seal=producer.parsed(producer.runtime.common.captured(path,2*1024**2,external_sha))
    roles={'native_receipt','native_owner','native_samples','native_freeze','native_capture',
           'reference_receipt','reference_owner','reference_samples','reference_freeze'}
    need(type(seal) is dict and set(seal)=={'format','files'}
         and seal['format']=='toklet-minicpm5-context1025-evidence-v1'
         and type(seal['files']) is dict and set(seal['files'])==roles,'Exact externally sealed capture evidence required')
    values={}
    for role,pin in seal['files'].items():
        need(type(pin) is dict and set(pin)=={'path','bytes','sha256'} and type(pin['path']) is str
             and Path(pin['path']).is_absolute() and integer(pin['bytes'],1)
             and pin['bytes']<=(16 if role.endswith('_samples') else 2)*1024**2 and sha(pin['sha256']),
             'Bounded absolute evidence identity required')
        raw=producer.runtime.common.captured(Path(pin['path']),pin['bytes'],pin['sha256'])
        need(len(raw)==pin['bytes'],'Evidence extent differs')
        values[role]=producer.parsed(raw)
    return seal,values


def load_reference(folder,external_sha,contract,old,quality,tokens):
    raw=producer.runtime.common.captured(folder/'manifest.json',2*1024**2,external_sha)
    value=producer.parsed(raw)
    _,inventory,_=producer.plan(contract,old,quality,tokens)
    need(value['format']=='toklet-minicpm5-context1025-reference-v1' and value['completed'] is True
         and value['native_outcomes_seen'] is False and value['quality_scored'] is False
         and value['managed_adoption_qualified'] is False and value['gates_recomputed_or_relaxed'] is False
         and same(value['gates'],contract['gates']) and same(value['columns'],contract['columns'])
         and same(value['cases'],contract['cases']) and value['contract_sha256']==producer.CONTRACT_SHA
         and value['quality_manifest_sha256']==producer.QUALITY_SHA
         and value['original_reference_manifest_sha256']==producer.OLD_SHA
         and same(value['model'],dict(repository=producer.model.REPOSITORY,revision=producer.model.REVISION,files=producer.model.PINS))
         and same(value['memory_admission'],producer.admission())
         and same(value['graph_identity'],producer.graph_identity(Path(producer.model.__file__).read_bytes(),Path(producer.graph.__file__).read_bytes())),
         'Complete frozen CPU reference identity/flags differ')
    need(integer(value['elapsed_ns'],1) and type(value['profile_records']) is list and len(value['profile_records'])==6,
         'Complete CPU profile clocks required')
    for row,(count,profile) in zip(value['profile_records'],[(n,p) for n in COUNTS for p in ['fp32','fp64']]):
        need(type(row) is dict and set(row)=={'case','profile','elapsed_ns'} and row['case']==f'prefix{count}'
             and row['profile']==profile and integer(row['elapsed_ns'],1),'Ordered CPU profile identity/clock differs')
    need(sum(r['elapsed_ns'] for r in value['profile_records'])<=value['elapsed_ns'],'CPU profile clocks exceed run')
    need(type(value['arrays']) is dict and set(value['arrays'])=={p+'.'+n for p in ['fp32','fp64'] for n in inventory},
         'Complete CPU array inventory differs')
    need(type(value['files']) is dict and set(value['files'])=={'fp32.bin','fp64.bin'},'CPU file inventory differs')
    for profile in ['fp32','fp64']:
        spans=[]
        for name,spec in inventory.items():
            item=value['arrays'][profile+'.'+name]
            size=spec['shape'][0]*spec['shape'][1]*(4 if spec['dtype']=='<f4' else 2)
            need(type(item) is dict and set(item)=={'shape','dtype','gate','file','offset','bytes','sha256'}
                 and all(same(item[k],v) for k,v in spec.items()) and item['file']==profile+'.bin'
                 and integer(item['offset']) and type(item['bytes']) is int and item['bytes']==size
                 and sha(item['sha256']),'CPU array metadata differs')
            spans.append((item['offset'],item['offset']+size))
        spans.sort();cursor=0
        for first,last in spans:need(first==cursor,'CPU array overlap/hole');cursor=last
        need(value['files'][profile+'.bin']['bytes']==cursor,'CPU file coverage differs')
    return value


def validate_frame(row,kind,count,chunk,stage,start,rows,columns,offset):
    need(type(row) is dict and set(row)=={'event','kind','case','chunk','stage','row_start','shape','dtype','offset','bytes','sha256'}
         and row['event']=='completed' and row['kind']==kind and row['case']==f'prefix{count}'
         and type(row['chunk']) is int and row['chunk']==chunk and row['stage']==stage
         and type(row['row_start']) is int and row['row_start']==start
         and same(row['shape'],[rows,columns]) and row['dtype']=='<f4'
         and type(row['offset']) is int and row['offset']==offset
         and type(row['bytes']) is int and row['bytes']==rows*columns*4 and sha(row['sha256']),
         'Exact native frame identity/coverage/storage differs')


def validate_owner(owner,freeze,native=True):
    need(type(owner) is dict and owner.get('passed') is True and owner.get('owned_group_absent') is True
         and (owner.get('native_capture_completed') is True if native else owner.get('native_outcomes_seen') is False)
         and owner.get('source_inputs_unchanged') is True and type(owner.get('exit_code')) is int
         and owner['exit_code']==0 and owner.get('failure') is None and owner.get('sample_errors')==[]
         and owner.get('quality_scored') is False and owner.get('model_adoption_qualified') is False
         and owner.get('final_physical_peak_available') is False
         and integer(owner.get('started_ns'),1) and integer(owner.get('completed_ns'),1)
         and owner['started_ns']<owner['completed_ns'] and integer(owner.get('minimum_observed_free_bytes'))
         and owner['minimum_observed_free_bytes']>=2*1024**3,'Closed bounded owner required')
    need(type(owner.get('guard_bytes')) is int and owner['guard_bytes']==6*1024**3
         and type(freeze.get('guard_bytes')) is int and freeze['guard_bytes']==6*1024**3
         and type(freeze.get('disk_floor_bytes')) is int and freeze['disk_floor_bytes']==2*1024**3,
         'Original resource policy differs')
    for field in ['final_wait4_peak_rss_bytes','max_sampled_aggregate_rss_bytes',
                  'max_sampled_aggregate_physical_bytes','max_sampled_aggregate_lifetime_physical_bytes']:
        need(integer(owner.get(field),1) and owner[field]<=6*1024**3,'Owned6GiB resource bound differs')
    need(type(freeze.get('source_sha256')) is dict and freeze['source_sha256']
         and all(sha(h) and producer.model.sha(Path(p))==h for p,h in freeze['source_sha256'].items()),
         'Frozen source/binary/reference inputs changed')
    if native:
        need(type(owner.get('source_files')) is int and owner['source_files']==len(freeze['source_sha256']),
             'Native frozen input coverage differs')


def validate_samples(owner,record,samples,freeze):
    need(type(record) is dict and set(record)=={'pid','started_ns','command'} and integer(record['pid'],1)
         and type(record['started_ns']) is int and record['started_ns']==owner['started_ns']
         and type(record['command']) is list and record['command']
         and all(type(x) is str for x in record['command']) and same(record['command'],freeze.get('command')),
         'Owned PID/clock/command association differs')
    need(type(samples) is list and 2<=len(samples)<=9000 and type(owner.get('sampled_observations')) is int
         and len(samples)==owner['sampled_observations'],'Complete bounded owned samples required')
    peaks={k:0 for k in ['resident_size','phys_footprint','lifetime_max_phys_footprint']}
    previous=owner['started_ns']-1;lifetime=0
    for sample in samples:
        need(type(sample) is dict and set(sample)=={'observed_ns','processes'} and integer(sample['observed_ns'],1)
             and previous<sample['observed_ns']<owner['completed_ns'] and sample['observed_ns']>=owner['started_ns']
             and type(sample['processes']) is list and len(sample['processes'])==1,
             'Ordered enclosed single-process numerical owner samples required')
        previous=sample['observed_ns'];row=sample['processes'][0]
        need(type(row) is dict and set(row)=={'pid',*peaks} and type(row['pid']) is int and row['pid']==record['pid']
             and all(integer(row[k]) for k in peaks) and row['lifetime_max_phys_footprint']>=row['phys_footprint']
             and row['lifetime_max_phys_footprint']>=lifetime,'Owned sample identity/counters differ')
        lifetime=row['lifetime_max_phys_footprint']
        for key in peaks:
            need(row[key]<=6*1024**3,'Observed owner exceeds6GiB')
            peaks[key]=max(peaks[key],row[key])
    for key,field in [('resident_size','max_sampled_aggregate_rss_bytes'),('phys_footprint','max_sampled_aggregate_physical_bytes'),
                      ('lifetime_max_phys_footprint','max_sampled_aggregate_lifetime_physical_bytes')]:
        need(peaks[key]==owner[field],'Owner receipt does not reconcile with actual samples')
    need(owner['final_wait4_peak_rss_bytes']>=peaks['resident_size'],'Final RSS peak omits sampled peak')


def reference_command(command,args,reference,native_command):
    need(type(command) is list and len(command)==24 and command[1:4]==['-s','-B','-X']
         and command[4].startswith('pycache_prefix=') and Path(command[5]).resolve()==Path(producer.__file__).resolve(),
         'Original independent CPU command differs')
    options=dict(zip(command[6::2],command[7::2]))
    expected={'--environment-root':args.environment_root,'--model':native_command[2],
              '--old-reference':args.old_reference,'--quality-inputs':args.quality_inputs,
              '--contract':args.contract,'--contract-sha256':producer.CONTRACT_SHA,'--fresh-output':args.reference}
    need(set(options)=={*expected,'--runtime-manifest','--runtime-sha256'}
         and all(options[k]==v for k,v in expected.items())
         and options['--runtime-sha256']==reference['admitted_runtime']['manifest_sha256']
         and command[4]=='pycache_prefix='+reference['admitted_runtime']['pycache_prefix'],
         'Independent CPU source/input/runtime association differs')
    runtime=producer.parsed(producer.runtime.common.captured(Path(options['--runtime-manifest']),2*1024**2,options['--runtime-sha256']))
    need(runtime['format']==producer.runtime.FORMAT and same(runtime['versions'],producer.runtime.VERSIONS)
         and len(runtime['files'])==reference['admitted_runtime']['files'],'Original CPU runtime admission differs')


def compare(native,reference,gate,np):
    need(phase_common.finite(gate['atol']) and phase_common.finite(gate['rtol'])
         and gate['atol']>0 and gate['rtol']>=0,'Finite positive absolute/error policy required')
    need(native.shape==reference.shape and np.isfinite(native).all() and np.isfinite(reference).all(),
         'Finite equal-shaped comparison required')
    delta=np.abs(native-reference);limit=gate['atol']+gate['rtol']*np.abs(reference)
    return delta,limit,delta>limit


def profile_consistency(reference,raw,contract,np):
    groups={}
    for count in COUNTS:
        for stage,columns in contract['columns'].items():
            key=f'prefix{count}.{stage}';items=[reference['arrays'][p+'.'+key] for p in ['fp32','fp64']]
            group=dict(compared=0,disjoint_coordinates=0,maximum_cpu_delta=0.,witnesses=[])
            digests=[hashlib.sha256(),hashlib.sha256()]
            for first in range(0,count*columns,65536):
                n=min(65536,count*columns-first);pair=[]
                for profile,item,digest in zip(['fp32','fp64'],items,digests):
                    width=4 if item['dtype']=='<f4' else 2
                    consumed=raw[profile].read(item['offset']+first*width,n*width);digest.update(consumed)
                    values=np.frombuffer(consumed,dtype=item['dtype']).astype(np.float64)
                    need(np.isfinite(values).all(),'Nonfinite independent source profile');pair.append(values)
                a,b=pair;gate=contract['gates'][stage]
                ta=gate['atol']+gate['rtol']*np.abs(a);tb=gate['atol']+gate['rtol']*np.abs(b)
                bad=np.maximum(a-ta,b-tb)>np.minimum(a+ta,b+tb)
                group['compared']+=n;group['disjoint_coordinates']+=int(np.count_nonzero(bad))
                group['maximum_cpu_delta']=max(group['maximum_cpu_delta'],float(np.max(np.abs(a-b))))
                if len(group['witnesses'])<4:
                    for i in np.flatnonzero(bad)[:4-len(group['witnesses'])]:
                        i=int(i);index=first+i
                        group['witnesses'].append(dict(row=index//columns,column=index%columns,fp32=float(a[i]),fp64=float(b[i]),
                            fp32_interval=[float(a[i]-ta[i]),float(a[i]+ta[i])],fp64_interval=[float(b[i]-tb[i]),float(b[i]+tb[i])]))
            need(all(d.hexdigest()==item['sha256'] for d,item in zip(digests,items)),'Consumed CPU array digest differs')
            groups[key]=group
    return dict(compared_coordinates=sum(v['compared'] for v in groups.values()),
                disjoint_coordinates=sum(v['disjoint_coordinates'] for v in groups.values()),groups=groups,
                scope='Original unchanged bounds around both independent CPU profiles; positive gaps cannot contain a common native value. No recalibration.')


def validate_capture(capture,reference_sha):
    need(capture['format']=='toklet-minicpm5-context1025-native-v1' and capture['completed'] is True
         and capture['repository']==producer.model.REPOSITORY and capture['revision']==producer.model.REVISION
         and capture['contract_sha256']==producer.CONTRACT_SHA and capture['quality_manifest_sha256']==producer.QUALITY_SHA
         and capture['reference_sha256']==reference_sha
         and same(capture['counts'],COUNTS) and same(capture['chunks'],[1,32])
         and capture['rms_store']=='serial baseline' and capture['numerical_comparison_pending'] is True
         and all(capture[k] is False for k in ['quality_scored','managed_adoption_qualified','performance_or_overhead_qualified','physical_iphone_qualified']),
         'Native capture identity/qualification differs')
    for key,want in [('capacity',1025),('ordinary_capacity_unchanged',512),('arrays',87624),
                     ('array_bytes',3546040320),('completed_chunks',1718),('memory_limit_bytes',6*1024**3),
                     ('application_allowance_bytes',32*1024**2)]:
        need(type(capture[key]) is int and capture[key]==want,'Complete native coverage/policy differs')
    need(integer(capture['maximum_combined_admitted_bytes'],1) and capture['maximum_combined_admitted_bytes']<=6*1024**3,
         'Native requested-owner bound differs')
    need(set(capture['files'])=={'arrays.bin','arrays.jsonl','chunks.jsonl'},'Native file inventory differs')
    need(integer(capture.get('elapsed_ns'),1) and integer(capture.get('load_wall_ns'),1)
         and capture['load_wall_ns']<=capture['elapsed_ns'],'Positive enclosed native load clock required')


def expected_memory(chunk,columns):
    """Fixed arm64 Mac layout: guards, snapshot/readback owners, Rust metadata.

    These captured shapes have contiguous MPS rows. Buffer/Cache/Views/Snapshot
    sizes are16/32/120/72 bytes for the frozen arm64 build; no cache-line or
    device execution-width assumption is involved. A changed ABI needs review.
    """
    need(type(chunk) is int and chunk in [1,32] and len(columns)==51,'Fixed diagnostic plan required')
    h,q,k,m,widest=2048,2048,256,6144,130560
    guarded=lambda width,scalar:(chunk*width+7)*scalar
    scratch=(3*guarded(h,2)+3*guarded(q,2)+3*guarded(k,2)+4*guarded(m,2)
             +guarded(widest,4)+8*(16*chunk*1025+7)+4*(chunk+7)+32+2*(widest+7))
    #50Half snapshots plus oneFloat logits snapshot, then CPU float readback.
    snapshot=sum((chunk*width+7)*(4 if stage=='logits' else 2) for stage,width in columns.items())
    scratch+=snapshot+4*chunk*sum(columns.values())+51*(2+72)
    result=dict(weights_bytes=2674218150,kv_bytes=(2*1025*128+7)*2*2*42,
        scratch_bytes=scratch,history_bytes=1025*4,logits_bytes=130560*8,
        metadata_bytes=chunk*120+42*32,metadata_allowance_bytes=64*1024**2)
    result['admitted_bytes']=sum(result.values())
    return result


def metric(timing,count,chunk,start,capture,columns,previous,gpu_end):
    rows=min(chunk,count-start)
    need(type(timing) is dict and timing.get('case')==f'prefix{count}'
         and all(type(timing.get(k)) is int and timing[k]==v for k,v in
            [('chunk',chunk),('row_start',start),('rows',rows),('committed_after',start+rows),('pending_after',count-start-rows)])
         and integer(timing.get('started_since_run_ns')) and integer(timing.get('completed_since_run_ns'),1)
         and previous<=timing['started_since_run_ns']<timing['completed_since_run_ns']<=capture['elapsed_ns']
         and integer(timing.get('wall_ns'),1) and timing['wall_ns']<=timing['completed_since_run_ns']-timing['started_since_run_ns']
         and type(timing.get('completed_timing')) is dict,'Completed native clock/history enclosure differs')
    phases.timing(dict(wall_nanoseconds=timing['wall_ns'],completed_timing=timing['completed_timing']))
    completed=timing['completed_timing']
    need(gpu_end<=completed['gpu_start_seconds'],'Serial completed GPU windows overlap or reorder')
    need(same(timing.get('session_memory'),expected_memory(chunk,columns)),
         'Exact KV/scratch/history/logit/metadata/allowance ownership differs')
    need(timing['session_memory']['admitted_bytes']+32*1024**2<=6*1024**3,'Combined owner exceeds6GiB')
    return timing['completed_since_run_ns'],completed['gpu_end_seconds']


def serial_spans(capture,first_host,last_host,wall_sum,first_gpu,last_gpu):
    load=capture['load_wall_ns'];elapsed=capture['elapsed_ns']
    need(load<=first_host and load+last_host-first_host<=elapsed and load+wall_sum<=elapsed,
         'Disjoint load plus serial host work exceeds run')
    need(load+phases.duration_ns(last_gpu-first_gpu)<=elapsed+phases.timestamp_grain_ns(first_gpu,last_gpu),
         'Disjoint load plus serial completed GPU span exceeds run')


def check(folder,reference,reference_raw,contract,np,reference_sha,capture):
    validate_capture(capture,reference_sha)
    for name,pin in capture['files'].items():
        need(integer(pin['bytes'],1) and sha(pin['sha256']) and (folder/name).stat().st_size==pin['bytes']
             and producer.model.sha(folder/name)==pin['sha256'],'Native file hash/extent differs')
    native=Raw(folder/'arrays.bin',capture['files']['arrays.bin'],4*1024**3)
    events=iter(lines(folder/'arrays.jsonl',90000,capture['files']['arrays.jsonl']))
    metrics=iter(lines(folder/'chunks.jsonl',1800,capture['files']['chunks.jsonl']))
    groups={};cursor=0;frames=0;chunk_count=0;previous=capture['load_wall_ns'];maximum_admitted=0
    first_host=first_gpu=None;previous_gpu=0.;wall_sum=0
    fields=['input_norm','q_projection','k_projection','v_projection','q_rotary','k_rotary',
        'attention_context','attention_projection','attention_residual','post_attention_norm','gate_projection',
        'gate_silu','up_projection','gated','down_projection','output']
    stages=['embedding','final_norm','logits']+[f'layers.{layer}.{field}' for layer in [0,1,41] for field in fields]
    def frame(kind,count,chunk,stage,start,rows):
        nonlocal cursor,frames
        row=next(events,None);columns=contract['columns'][stage]
        validate_frame(row,kind,count,chunk,stage,start,rows,columns,cursor)
        digest=hashlib.sha256();size=rows*columns
        for first in range(0,size,65536):
            n=min(65536,size-first);raw=native.read(cursor+first*4,n*4);digest.update(raw)
            got=np.frombuffer(raw,dtype='<f4').astype(np.float64)
            for profile in ['fp32','fp64']:
                item=reference['arrays'][f'{profile}.prefix{count}.{stage}'];width=4 if item['dtype']=='<f4' else 2
                raw_ref=reference_raw[profile].read(item['offset']+(start*columns+first)*width,n*width)
                wanted=np.frombuffer(raw_ref,dtype=item['dtype']).astype(np.float64)
                delta,limit,failed=compare(got,wanted,contract['gates'][stage],np)
                key=f'{profile}.chunk{chunk}.prefix{count}.{kind}.{stage}'
                aggregate=groups.setdefault(key,dict(compared=0,failed=0,max_absolute_error=0.,max_tolerance_fraction=0.,witnesses=[]))
                aggregate['compared']+=n;aggregate['failed']+=int(np.count_nonzero(failed))
                aggregate['max_absolute_error']=max(aggregate['max_absolute_error'],float(np.max(delta)))
                aggregate['max_tolerance_fraction']=max(aggregate['max_tolerance_fraction'],float(np.max(delta/limit)))
                if len(aggregate['witnesses'])<4:
                    for index in np.flatnonzero(failed)[:4-len(aggregate['witnesses'])]:
                        index=int(index);flat=start*columns+first+index
                        aggregate['witnesses'].append(dict(row=flat//columns,column=flat%columns,native=float(got[index]),
                            reference=float(wanted[index]),absolute_error=float(delta[index]),limit=float(limit[index])))
        need(digest.hexdigest()==row['sha256'],'Consumed native frame hash differs')
        cursor+=row['bytes'];frames+=1
    try:
        for count in COUNTS:
            for chunk in [1,32]:
                for start in range(0,count,chunk):
                    rows=min(chunk,count-start);timing=next(metrics,None)
                    previous,previous_gpu=metric(timing,count,chunk,start,capture,contract['columns'],previous,previous_gpu)
                    if first_host is None:first_host=timing['started_since_run_ns'];first_gpu=timing['completed_timing']['gpu_start_seconds']
                    chunk_count+=1;wall_sum+=timing['wall_ns']
                    maximum_admitted=max(maximum_admitted,timing['session_memory']['admitted_bytes']+32*1024**2)
                    for stage in stages:frame('prefill',count,chunk,stage,start,rows)
                frame('corrected_last',count,chunk,'logits',count-1,1)
        need(next(events,None) is None and next(metrics,None) is None and cursor==3546040320
             and frames==87624 and chunk_count==1718 and len(groups)==624,'Whole native/reference comparison coverage differs')
        need(maximum_admitted==capture['maximum_combined_admitted_bytes'],'Maximum requested owner ledger differs')
        serial_spans(capture,first_host,previous,wall_sum,first_gpu,previous_gpu)
        native.verify()
    finally:
        native.close();events.close();metrics.close()
    expected_keys={'case','chunk','partial_discard_and_exact_prefix','fully_cached','pending_or_boundary_append',
        'rejections_preserve_state','changed_prefix_discard_and_restore','corrected_logits_captured_for_original_gate','reset_replay_bitwise'}
    need(type(capture['functional']) is list and len(capture['functional'])==6,'Complete functional controls required')
    for row,(count,chunk) in zip(capture['functional'],[(n,c) for n in COUNTS for c in [1,32]]):
        need(set(row)==expected_keys and row['case']==f'prefix{count}' and type(row['chunk']) is int and row['chunk']==chunk
             and all(row[k] is True for k in expected_keys-{'case','chunk'}),'Functional prefix/recovery/capacity control differs')
    return dict(completed=True,numerical_gates_passed=all(row['failed']==0 for row in groups.values()),
        compared_values=sum(row['compared'] for row in groups.values()),failed_values=sum(row['failed'] for row in groups.values()),
        groups=groups,arrays=frames,completed_chunks=chunk_count,functional_controls_passed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['environment_root','contract','reference','reference_sha256','old_reference','quality_inputs',
                 'runtime_manifest','runtime_sha256','capture','evidence','evidence_sha256','fresh_report']:
        parser.add_argument('--'+name.replace('_','-'),required=True)
    args=parser.parse_args();producer.runtime.bind_environment(args.environment_root)
    contract=producer.parsed(producer.runtime.common.captured(Path(args.contract),2*1024**2,producer.CONTRACT_SHA))
    old=producer.parsed(producer.runtime.common.captured(Path(args.old_reference)/'manifest.json',2*1024**2,producer.OLD_SHA))
    quality=producer.parsed(producer.runtime.common.captured(Path(args.quality_inputs)/'manifest.json',2*1024**2,producer.QUALITY_SHA))
    tokens=producer.runtime.common.captured(Path(args.quality_inputs)/'held_out_loss.u32le',40004,producer.TOKENS_SHA)
    seal,records=evidence(Path(args.evidence),args.evidence_sha256)
    owner=records['native_receipt'];freeze=records['native_freeze']
    ref_owner=records['reference_receipt'];ref_freeze=records['reference_freeze']
    validate_owner(owner,freeze);validate_owner(ref_owner,ref_freeze,native=False)
    for kind,receipt,frozen in [('native',owner,freeze),('reference',ref_owner,ref_freeze)]:
        validate_samples(receipt,records[kind+'_owner'],records[kind+'_samples'],frozen)
    need(ref_owner['completed_ns']<=owner['started_ns'],'Independent CPU reference did not close before native capture')
    selected=sources()
    admitted,endpoint=producer.runtime.admit(Path(args.runtime_manifest),args.runtime_sha256,selected)
    import numpy as np
    literal=np.array([0.,1.,-1.,100.],dtype=np.float64)
    _,_,mask=compare(literal+np.array([.05,.12,-.1,1.2]),literal,dict(atol=.1,rtol=.01),np)
    need(mask.tolist()==[False,True,False,True],'Independent literal original-bound predicate differs')
    reference=load_reference(Path(args.reference),args.reference_sha256,contract,old,quality,tokens)
    command=freeze.get('command')
    need(type(command) is list and len(command)==9 and command[1]=='context'
         and Path(command[3]).resolve()==Path(args.reference).resolve() and command[4]==args.reference_sha256
         and Path(command[5]).resolve()==Path(args.contract).resolve() and command[6]==producer.CONTRACT_SHA
         and Path(command[7]).resolve()==Path(args.quality_inputs).resolve()
         and Path(command[8]).resolve()==Path(args.capture).resolve()
         and command[0] in freeze['source_sha256'],'Native command/input association differs')
    reference_command(ref_freeze['command'],args,reference,command)
    need(freeze.get('completed_independent_reference_sha256')==args.reference_sha256
         and freeze['source_sha256'].get(str(Path(args.reference)/'manifest.json'))==args.reference_sha256
         and freeze['source_sha256'].get(seal['files']['reference_receipt']['path'])==seal['files']['reference_receipt']['sha256']
         and all(str(p) in ref_freeze['source_sha256'] for p in producer.sources())
         and Path(seal['files']['native_capture']['path']).resolve()==(Path(args.capture)/'capture.json').resolve(),
         'CPU/native artifact lineage differs')
    capture=records['native_capture']
    validate_capture(capture,args.reference_sha256)
    need(integer(capture.get('elapsed_ns'),1) and capture['elapsed_ns']<=owner['completed_ns']-owner['started_ns']
         and reference['elapsed_ns']<=ref_owner['completed_ns']-ref_owner['started_ns'],
         'Capture clocks exceed owned process lifetimes')
    raw={}
    try:
        for p in ['fp32','fp64']:raw[p]=Raw(Path(args.reference)/(p+'.bin'),reference['files'][p+'.bin'],3*1024**3)
        consistency=profile_consistency(reference,raw,contract,np)
        result=check(Path(args.capture),reference,raw,contract,np,args.reference_sha256,capture)
        result['source_profile_consistency']=consistency
        need(not result['numerical_gates_passed'] or consistency['disjoint_coordinates']==0,
             'Native pass contradicts disjoint original CPU profile intervals')
        for value in raw.values():value.verify()
    finally:
        for value in raw.values():value.close()
    need(sources()==selected,'Comparison runtime/source inventory changed')
    result.update(format='toklet-minicpm5-context1025-comparison-v1',admitted_runtime=admitted,endpoint_origins=endpoint(),
        reference_sha256=args.reference_sha256,contract_sha256=producer.CONTRACT_SHA,
        evidence_sha256=args.evidence_sha256,owned_sample_receipts_reconciled=True,
        quality_scored=False,managed_adoption_qualified=False,performance_or_overhead_qualified=False,
        scope='All captured coordinates against both unchanged original profiles/gates. Completed numerical rejection remains a rejection; no quality/default adoption.')
    with Path(args.fresh_report).open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','admitted_runtime','endpoint_origins','source_profile_consistency']}))


ROOT=Path(__file__).resolve().parent
if __name__=='__main__':main()
