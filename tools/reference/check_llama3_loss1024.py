"""Fixed paired loss screen from authenticated statistics, not saved logits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
import prepare_llama3_loss1024 as inputs
import mlx_llama3_real as source

LIMIT=.05
VOCAB=128256


def digest(value):
    return isinstance(value,str)and len(value)==64 and all(c in '0123456789abcdef'for c in value)


def close(actual,expected):
    return type(actual)in(int,float)and math.isfinite(actual)and abs(actual-expected)<=8*math.ulp(max(1.,abs(expected)))


def statistics(record,target):
    integer_fields=['target','target_fp32_bits','vocab','greedy_lowest_id','block','local_row','global_target_index']
    if any(type(record.get(k))is not int for k in integer_fields):raise ValueError('Integer target fields required')
    if (record['target']!=target or record['vocab']!=VOCAB or not 0<=record['greedy_lowest_id']<VOCAB
            or not 0<=record['target_fp32_bits']<2**32 or not digest(record['row_sha256'])):
        raise ValueError('Target statistics identity differs')
    maximum,mass,nll=(record[k]for k in ['maximum','exp_mass','nll'])
    if any(type(v)not in(int,float)or not math.isfinite(v)for v in [maximum,mass,nll]):raise ValueError('Finite FP64 statistics required')
    target_value=struct.unpack('<f',struct.pack('<I',record['target_fp32_bits']))[0]
    if (not math.isfinite(target_value)or maximum<target_value or not 1<=mass<=VOCAB or nll<0
            or struct.unpack('<f',struct.pack('<f',maximum))[0]!=maximum):raise ValueError('Invalid distribution statistics')
    expected=(maximum-target_value)+math.log(mass)
    if not close(nll,expected):raise ValueError('Stable NLL formula differs')
    if target_value==maximum and record['greedy_lowest_id']>target:raise ValueError('Lowest-ID maximum contradicts target')
    return nll


def paired_delta(native,wide):
    if len(native)!=1024 or len(wide)!=1024 or any(not math.isfinite(v)for v in [*native,*wide]):raise ValueError('Exactly1024 finite paired targets required')
    return math.fsum(a-b for a,b in zip(native,wide,strict=True))/1024


def bounded_json(path,limit=4*1024**2):
    with path.open('rb')as f:raw=f.read(limit+1)
    if len(raw)>limit:raise ValueError('Oversized report')
    return json.loads(raw,object_pairs_hook=inputs.unique),raw


def journal(path):
    with path.open('rb')as f:raw=f.read(4*1024**2+1)
    if len(raw)>4*1024**2:raise ValueError('Oversized target journal')
    return [json.loads(line,object_pairs_hook=inputs.unique)for line in raw.splitlines()],raw


def collect(records,ids,native=False):
    labels=[32,128]if native else['main','wide']
    values={k:{}for k in labels};hashes={k:{}for k in labels};unscored={k:{}for k in labels};details={k:{}for k in labels}
    for record in records:
        label=record.get('chunk')if native else record.get('profile')
        if label not in labels or (native and type(label)is not int):raise ValueError('Unknown loss profile/schedule')
        if native and record.get('kind')=='unscored':
            block=record.get('block')
            if type(block)is not int or block not in range(4)or record.get('local_row')!=256 or not digest(record.get('row_sha256'))or not digest(record.get('numeric_row_sha256'))or block in unscored[label]:raise ValueError('Unscored native inventory differs')
            unscored[label][block]=record['numeric_row_sha256'];continue
        if native and record.get('kind')!='target':raise ValueError('Unknown native event')
        block,row,index=(record.get(k)for k in ['block','local_row','global_target_index'])
        if any(type(v)is not int for v in (block,row,index))or block not in range(4)or not 0<=row<256 or index!=block*256+row+1 or index in values[label]:raise ValueError('Target coverage/alignment differs')
        values[label][index]=statistics(record,ids[index])
        if native:
            if not digest(record.get('numeric_row_sha256')):raise ValueError('Numeric row hash invalid')
            hashes[label][index]=record['numeric_row_sha256'];details[label][index]=record
    for label in labels:
        if set(values[label])!=set(range(1,1025)):raise ValueError('Incomplete target coverage')
    if native:
        if any(set(unscored[k])!=set(range(4))for k in labels)or hashes[32]!=hashes[128]or unscored[32]!=unscored[128]:raise ValueError('Exact native numeric row schedules differ')
        for index in range(1,1025):
            a,b=details[32][index],details[128][index]
            for key in ['maximum','exp_mass','nll','greedy_lowest_id']:
                if a[key]!=b[key]:raise ValueError('Native row statistics schedules differ')
            av,bv=(struct.unpack('<f',struct.pack('<I',r['target_fp32_bits']))[0]for r in (a,b))
            if av!=bv:raise ValueError('Native target values differ')
    return {label:[values[label][i]for i in range(1,1025)]for label in labels}


def check(input_folder,input_sha,source_folder,source_sha,native_folder):
    manifest,ids=inputs.load(input_folder,input_sha)
    report,raw=bounded_json(source_folder/'source.json')
    if hashlib.sha256(raw).hexdigest()!=source_sha:raise ValueError('Frozen source report identity differs')
    if (report['format']!='toklet-llama3-loss1024-source-v1'or report['completed']is not True or report['errors']
            or report['input_manifest_sha256']!=input_sha or report['short_reference_manifest_sha256']!='da088a718df353857f2dd5f4946a38b1233377a2d604595870623862152912f6'
            or report['model']!=dict(repository=source.REPO,revision=source.REVISION,file_pins=source.PINS)
            or report['policy']!=inputs.POLICY or report['blocks_contract']!=inputs.BLOCKS or report['profile']!=source.NATIVE_PROFILE
            or report['numpy']!='2.3.2'or report['admission']!=source.admission()
            or report['native_executed']is not False or report['full_quality_qualified']is not False):raise ValueError('Source identity/profile differs')
    expected={(b,p)for b in range(4)for p in ['main','wide']};seen=set()
    for block in report['blocks']:
        key=(block['block'],block['profile'])
        if key not in expected or key in seen or block['rows']!=257 or block['targets']!=256 or not digest(block['all_rows_sha256'])or not digest(block['last_unscored_row_sha256']):raise ValueError('Source block inventory differs')
        seen.add(key)
    if seen!=expected:raise ValueError('Source block coverage differs')
    records,target_raw=journal(source_folder/'targets.jsonl');pin=report['targets_file']
    if pin!=dict(file='targets.jsonl',bytes=len(target_raw),sha256=hashlib.sha256(target_raw).hexdigest()):raise ValueError('Source target journal differs')
    if len(records)!=2048 or report['targets_per_profile']!={'main':1024,'wide':1024}:raise ValueError('Source target count differs')
    baseline=collect(records,ids)
    for key in ['main','wide']:
        if not close(report['mean_nll'][key],math.fsum(baseline[key])/1024):raise ValueError('Source mean differs')
    if not close(report['main_minus_wide'],report['mean_nll']['main']-report['mean_nll']['wide']):raise ValueError('Source diagnostic delta differs')
    native,_=bounded_json(native_folder/'native.json')
    if (native['format']!='toklet-llama3-loss1024-native-v1'or native['repository']!=source.REPO or native['revision']!=source.REVISION
            or native['input_manifest_sha256']!=input_sha or native['policy']!=inputs.POLICY or native['blocks']!=inputs.BLOCKS
            or native['profile']!='MlxAffine/Compensated/Reference/Half'or native['readback']!='FullLogits'
            or native['source_comparison_pending']is not True or any(native[k]is not False for k in ['unquantized_quality_qualified','full_task_quality_qualified','app_qualified','iphone_qualified'])
            or type(native['maximum_admitted_bytes'])is not int or not 0<native['maximum_admitted_bytes']<=6*1024**3
            or native['application_allowance_bytes']!=64*1024**2):raise ValueError('Native input/profile differs')
    records,_=journal(native_folder/'targets.jsonl')
    if len(records)!=2056:raise ValueError('Native event count differs')
    actual=collect(records,ids,True);results=[];summaries={r['chunk']:r for r in native['summaries']}
    if set(summaries)!={32,128}or len(native['summaries'])!=2:raise ValueError('Native summary inventory differs')
    for chunk in [32,128]:
        mean=math.fsum(actual[chunk])/1024
        if summaries[chunk]['targets']!=1024 or not close(summaries[chunk]['mean_nll'],mean):raise ValueError('Native mean differs')
        delta=paired_delta(actual[chunk],baseline['wide'])
        results.append(dict(chunk=chunk,mean_nll=mean,wide_mean_nll=math.fsum(baseline['wide'])/1024,native_minus_wide=delta,passed=delta<=LIMIT))
    return dict(format='toklet-llama3-loss1024-comparison-v1',passed=all(r['passed']for r in results),targets=1024,limit=LIMIT,
                input_manifest_sha256=input_sha,source_report_sha256=source_sha,results=results,
                source_main_minus_wide=paired_delta(baseline['main'],baseline['wide']),exact_native_numeric_row_schedules=True,
                statistics_coherence_tolerance='8 ulps of max(1,abs(recomputed)); not added to primary0.05 gate',
                evidence_scope='Authenticated per-target statistics and row hashes; full logits not retained or independently rescored',
                per_target=[dict(global_target_index=i+1,target=ids[i+1],wide_nll=baseline['wide'][i],main_nll=baseline['main'][i],native_nll=actual[32][i],native_minus_wide=actual[32][i]-baseline['wide'][i])for i in range(1024)],
                original_unquantized_quality_qualified=False,full_task_quality_qualified=False,app_qualified=False,iphone_qualified=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','source','native','output']:parser.add_argument('--'+name,type=Path,required=True)
    for name in ['inputs-sha256','source-sha256']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();result=check(args.inputs,args.inputs_sha256,args.source,args.source_sha256,args.native)
    with args.output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k]for k in ['passed','targets','limit','results']}));sys.exit(0 if result['passed']else 1)
