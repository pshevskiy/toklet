"""Replay frozen Gemma512 capture with separate state and dual numerical verdicts."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import struct
import gemma4_e2b_context_reference as oracle
import check_gemma_hybrid32 as prior

ROOT=Path(__file__).resolve().parents[2]
TEST='model::gemma_packed::tests::hybrid::real::diagnostic512_boundary_schedules_and_reset'
POSITIONS=[0,31,255,511]
PARTITIONS=[[32]*16,[1]+[31]*16+[15]]


def references(folder,phase):
    record=prior.bounded.document(prior.bounded.read(folder/'result.json',2*1024**2))
    ids,positions=oracle.inputs(phase)
    if not record['completed']or record['phase']!=phase or record['tokens']!=ids or record['selected_positions']!=positions or record['source_pins']!=oracle.source_graph.source.PINS:
        raise ValueError('Qualified reference phase differs')
    raw=prior.bounded.read(folder/'rows.bin',16*1024**2)
    if hashlib.sha256(raw).hexdigest()!=record['rows_sha256']:raise ValueError('Reference body digest differs')
    rows={};offset=0
    expected=[(kind,pos)for kind in ['math_fp64','staged_fp32']for pos in positions]
    if [(v['contract'],v['position'])for v in record['arrays']]!=expected:raise ValueError('Reference row inventory differs')
    for spec in record['arrays']:
        size=262144*(8 if spec['contract']=='math_fp64'else 4);block=memoryview(raw)[offset:offset+size]
        if spec['offset']!=offset or spec['bytes']!=size or len(block)!=size or hashlib.sha256(block).hexdigest()!=spec['sha256']:raise ValueError('Reference row bounds/digest differ')
        rows[spec['contract'],spec['position']]=block;offset+=size
    if offset!=len(raw):raise ValueError('Reference trailing bytes')
    return rows


def replay(transcript,body,prefix_folder,full_folder):
    spec=prior.bounded.document(prior.bounded.read(ROOT/'benchmarks/gemma-hybrid512-v1/manifest.json',65536))
    if spec['partitions']!=PARTITIONS or spec['selected_positions']!=POSITIONS or spec['reset_prefix_tokens']!=32:raise ValueError('Frozen native contract differs')
    parent=ROOT/spec['oracle_contract']['path']
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=spec['oracle_contract']['sha256']:raise ValueError('Oracle contract changed')
    events=[];wrapper=[]
    for line in transcript.decode().splitlines():
        if 'GEMMA_HYBRID_'in line:
            prefix,rest=line.split('GEMMA_HYBRID_',1)
            if prefix not in ['',f'test {TEST} ... ']:raise ValueError('Unexpected test wrapper prefix')
            name,raw=rest.split(' ',1);events.append((name,prior.bounded.document(raw.encode())))
        elif line.strip():wrapper.append(line)
    if len(wrapper)!=3 or wrapper[:2]!=['running 1 test','ok']or not re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s',wrapper[2]):raise ValueError('Complete successful test wrapper required')
    cursor=0;offset=0;number=0;initial=None;main={};state=[]
    def take(name):
        nonlocal cursor
        if cursor>=len(events)or events[cursor][0]!=name:raise ValueError('Native event inventory differs: '+name)
        value=events[cursor][1];cursor+=1;return value
    def array(name,before,count):
        nonlocal offset,number
        expected=dict(record=number,name=name,offset=offset,bytes=count*1048576,before=before,count=count,values=count*262144)
        if take('ARRAY_ATTEMPT')!=expected:raise ValueError('Native attempt identity differs')
        value=take('ARRAY');block=memoryview(body)[offset:offset+expected['bytes']]
        if {k:v for k,v in value.items()if k!='sha256'}!=expected or len(block)!=expected['bytes']or hashlib.sha256(block).hexdigest()!=value['sha256']:raise ValueError('Native row body identity differs')
        offset+=len(block);number+=1;return block
    for index,partition in enumerate(PARTITIONS):
        if take('REAL_CONTEXT512_SCHEDULE')!=partition:raise ValueError('Partition differs')
        if index==0:initial=array('context512.initial_logits',0,32)
        for position in POSITIONS:
            block=array('context512.logits',position,1)
            if index==0:main[position]=block
            elif block!=main[position]:raise ValueError('Partition row bytes differ')
        reset=array('context512.reset_logits',0,32)
        if reset!=initial:raise ValueError('Reset prefix bytes differ')
        value=take('REAL_CONTEXT512_STATE')
        expected=dict(schedule=partition,committed_before_rejection=512,rejected_absolute_position=512,overflow_preserved=True,reset_prefix_tokens=32,reset_all32_rows_exact=True,all15_valid_kv_prefixes_exact=True,allocated_bytes=263761128)
        if value!=expected:raise ValueError('Boundary state completion differs')
        state.append(value)
    expected=dict(records=11,bytes=109051904,sha256=hashlib.sha256(body).hexdigest(),whole_e2b_qualified=False)
    if take('CAPTURE')!=expected or offset!=len(body):raise ValueError('Full native capture differs')
    if take('REAL_CONTEXT512_COMPLETE')!=dict(state_passed=True,positions=512,selected_rows=POSITIONS,prior_precision_failures_waived=False,app_qualified=False,phone_qualified=False)or cursor!=len(events):raise ValueError('Full native completion differs')
    for position in [0,31]:
        if main[position]!=initial[position*1048576:(position+1)*1048576]:raise ValueError('Native causal prefix rows differ')
    if any(not math.isfinite(v)for(v,)in struct.iter_unpack('<f',initial)):raise ValueError('Nonfinite initial/reset prefix')
    prefix=references(prefix_folder,'compat32');full=references(full_folder,'full512')
    for kind in ['math_fp64','staged_fp32']:
        for position in [0,31]:
            if prefix[kind,position]!=full[kind,position]:raise ValueError('Reference causal prefix differs')
    reports=[]
    for group,blocks,reference in [('prefix',{p:initial[p*1048576:(p+1)*1048576]for p in [0,15,30,31]},prefix),('main',main,full)]:
        for position,block in blocks.items():
            actual=struct.unpack('<262144f',block);contracts={}
            for kind in ['math_fp64','staged_fp32']:
                fmt='<262144d'if kind=='math_fp64'else'<262144f'
                contracts[kind]=prior.fractions(actual,struct.unpack(fmt,reference[kind,position]))
            reports.append(dict(group=group,position=position,contracts=contracts))
    numerical={kind:dict(compared_values=sum(v['contracts'][kind]['values']for v in reports),failed_values=sum(v['contracts'][kind]['failed_values']for v in reports),maximum_gate_ratio=max(v['contracts'][kind]['maximum_gate_ratio']for v in reports))for kind in ['math_fp64','staged_fp32']}
    return dict(state_passed=True,numerical_passed=all(v['failed_values']==0 for v in numerical.values()),numerical=numerical,rows=reports,state=state,main_native_records=8,main_reference_positions=4,prefix_reference_positions=4,all32_reset_rows_exact=True,prior_precision_failures_waived=False,app_qualified=False,phone_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transcript',type=Path);parser.add_argument('arrays',type=Path);parser.add_argument('prefix_reference',type=Path);parser.add_argument('full_reference',type=Path);args=parser.parse_args()
    result=replay(prior.bounded.read(args.transcript,1024**2),prior.bounded.read(args.arrays,128*1024**2),args.prefix_reference,args.full_reference)
    print(json.dumps(result,indent=2,allow_nan=False))
