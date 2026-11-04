"""Replay Gemma32 capture; preserve independent state and numerical decisions."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct
import gemma_hybrid32 as contract
import check_gemma_hybrid_tiny as bounded

TEST='model::gemma_packed::tests::hybrid::real::original_e2b_hybrid32_boundary_schedules_and_reset'


def fractions(actual,reference):
    if len(actual)!=len(reference):raise ValueError('Full row length differs')
    failed=0;worst=0.;examples=[]
    for index,(a,b)in enumerate(zip(actual,reference,strict=True)):
        if not math.isfinite(a)or not math.isfinite(b):raise ValueError('Nonfinite row')
        ratio=abs(a-b)/(1e-5+1e-4*abs(b));worst=max(worst,ratio)
        if ratio>1:
            failed+=1
            if len(examples)<8:examples.append(dict(index=index,actual=a,reference=b,gate_ratio=ratio))
    return dict(values=len(actual),failed_values=failed,maximum_gate_ratio=worst,examples=examples)


def replay(transcript,body,reference_folder):
    reference=bounded.document(bounded.read(reference_folder/'result.json',2*1024**2))
    if reference.get('source_pins')!=contract.reference.source.PINS or not reference['completed']or reference['tokens']!=contract.inputs()or reference['selected_positions']!=contract.POSITIONS or reference['gate']!=dict(atol=1e-5,rtol=1e-4):raise ValueError('Reference contract differs')
    raw=bounded.read(reference_folder/'rows.bin',16*1024**2)
    if hashlib.sha256(raw).hexdigest()!=reference['rows_sha256']:raise ValueError('Reference bytes differ')
    rows={};offset=0
    for spec in reference['arrays']:
        key=(spec['contract'],spec['position']);size=262144*(4 if key[0]=='staged_fp32'else 8)
        if key in rows or key[0]not in ['math_fp64','staged_fp32']or key[1]not in contract.POSITIONS or spec['offset']!=offset or spec['bytes']!=size:raise ValueError('Reference row inventory differs')
        block=raw[offset:offset+size]
        if len(block)!=size or hashlib.sha256(block).hexdigest()!=spec['sha256']:raise ValueError('Reference row digest differs')
        rows[key]=struct.unpack('<'+str(262144)+('f'if key[0]=='staged_fp32'else'd'),block);offset+=size
    if len(rows)!=8 or offset!=len(raw):raise ValueError('Incomplete reference rows')
    events=[];wrapper=[]
    for line in transcript.decode().splitlines():
        if 'GEMMA_HYBRID_'in line:
            prefix,rest=line.split('GEMMA_HYBRID_',1)
            if prefix not in ['',f'test {TEST} ... ']:raise ValueError('Unexpected test prefix')
            name,value=rest.split(' ',1);events.append((name,bounded.document(value.encode())))
        elif line.strip():wrapper.append(line)
    import re
    if len(wrapper)!=3 or wrapper[:2]!=['running 1 test','ok']or not re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s',wrapper[2]):raise ValueError('Successful complete test wrapper required')
    cursor=0;byte_offset=0;number=0;reports=[];state=[];row_pins={}
    def take(name):
        nonlocal cursor
        if cursor>=len(events)or events[cursor][0]!=name:raise ValueError('Capture event inventory differs: '+name)
        value=events[cursor][1];cursor+=1;return value
    for schedule in [[32],[31,1],[1,15,16]]:
        if take('REAL_BOUNDARY_SCHEDULE')!=schedule:raise ValueError('Schedule differs')
        for name in ['boundary.logits','boundary.reset_logits']:
            for position in contract.POSITIONS:
                attempt=take('ARRAY_ATTEMPT');spec=take('ARRAY')
                expected=dict(record=number,name=name,offset=byte_offset,bytes=1048576,before=position,count=1,values=262144)
                if attempt!=expected or {k:v for k,v in spec.items()if k!='sha256'}!=expected:raise ValueError('Native selected row identity differs')
                block=body[byte_offset:byte_offset+1048576];sha=hashlib.sha256(block).hexdigest()
                if len(block)!=1048576 or sha!=spec['sha256']:raise ValueError('Native row digest differs')
                if position in row_pins and row_pins[position]!=sha:raise ValueError('Selected schedule/reset row differs')
                row_pins[position]=sha;values=struct.unpack('<262144f',block)
                reports.append(dict(record=number,position=position,contracts={c:fractions(values,rows[c,position])for c in ['math_fp64','staged_fp32']}))
                byte_offset+=1048576;number+=1
        value=take('REAL_BOUNDARY_STATE')
        if value['schedule']!=schedule or value['positions']!=32 or value['overflow_preserved']is not True or value['reset_reuse']is not True or type(value['allocated_bytes'])is not int or value['allocated_bytes']<=0:raise ValueError('Boundary state completion differs')
        state.append(value)
    capture=take('CAPTURE')
    if capture!=dict(records=24,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),whole_e2b_qualified=False)or byte_offset!=len(body):raise ValueError('Full capture identity differs')
    complete=take('REAL_BOUNDARY_COMPLETE')
    if complete!=dict(state_passed=True,positions=32,selected_rows=contract.POSITIONS,numerical_decision='separate reference replay',prior_precision_failures_waived=False,phone_qualified=False,app_qualified=False)or cursor!=len(events):raise ValueError('Full boundary completion differs')
    numerical={c:dict(compared_values=sum(v['contracts'][c]['values']for v in reports),failed_values=sum(v['contracts'][c]['failed_values']for v in reports),maximum_gate_ratio=max(v['contracts'][c]['maximum_gate_ratio']for v in reports))for c in ['math_fp64','staged_fp32']}
    return dict(state_passed=True,unique_selected_rows=4,unique_values_per_contract=1048576,numerical=numerical,numerical_passed=all(v['failed_values']==0 for v in numerical.values()),rows=reports,state=state,prior_precision_failures_waived=False,app_qualified=False,phone_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('transcript',type=Path);parser.add_argument('arrays',type=Path);parser.add_argument('reference',type=Path);args=parser.parse_args()
    result=replay(bounded.read(args.transcript,1024**2),bounded.read(args.arrays,32*1024**2),args.reference)
    print(json.dumps(result,indent=2,allow_nan=False))
