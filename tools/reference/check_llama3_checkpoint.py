"""Check bounded real-Llama captures; fixed source policy, no inference or retries."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import mlx_llama3_real as reference

VOCAB=128256
CHUNKS=[32,128,3]
MAXIMUM=256*1024**2


def inventory():
    expected={}
    for case,spec in reference.CASES.items():
        prefix='cases.'+case+'.'
        stages={name[len(prefix):]:shape for name,(shape,_) in reference.expected_arrays().items()
                if name.startswith(prefix)and not name.endswith(('attention_scores','attention_probabilities'))}
        if len(stages)!=51:raise ValueError('Fixed capture stages differ')
        for requested in CHUNKS:
            chunk=min(requested,512-spec['offset'])
            for start in range(0,len(spec['tokens']),chunk):
                rows=min(chunk,len(spec['tokens'])-start)
                for stage,shape in stages.items():
                    expected[(f'chunk{requested}.cases.{case}.{stage}',start)]=[rows,math.prod(shape[1:])]
    for step in range(4):
        for group in ['greedy','native_generated']:expected[(f'{group}.step_{step}.logits',0)]=[1,VOCAB]
    for count in range(1,4):expected[(f'scoring.prefix_{count}.logits',0)]=[1,VOCAB]
    for chunk in [32,128]:expected[(f'boundary.chunk{chunk}.logits',511)]=[1,VOCAB]
    return expected


def validate_inventory(items):
    if not isinstance(items,list)or len(items)!=len(inventory()):raise ValueError('Capture inventory size differs')
    pending=inventory();offset=0
    for item in items:
        key=(item['name'],item['row_start']);shape=pending.pop(key,None)
        if (type(item['row_start'])is not int or not isinstance(item['shape'],list)
                or any(type(n)is not int for n in item['shape'])or shape is None or item['shape']!=shape or item['event']!='completed' or item['dtype']!='<f4'
                or type(item['offset'])is not int or item['offset']!=offset
                or type(item['bytes'])is not int or item['bytes']!=math.prod(shape)*4
                or not isinstance(item['sha256'],str)or len(item['sha256'])!=64
                or any(c not in '0123456789abcdef'for c in item['sha256'])):
            raise ValueError('Capture row/stage/layout differs: '+str(key))
        offset+=item['bytes']
    if pending or offset>MAXIMUM:raise ValueError('Incomplete or oversized capture')
    return offset


def qualify(actual, expected, gate):
    count=failed=0;maximum=fraction=0.
    for value,(target,)in zip(actual,expected,strict=True):
        if not math.isfinite(value)or not math.isfinite(target):raise ValueError('Nonfinite comparison')
        error=abs(value-target);limit=gate['atol']+gate['rtol']*abs(target)
        count+=1;failed+=error>limit;maximum=max(maximum,error);fraction=max(fraction,error/limit)
    return dict(values=count,failed=failed,max_absolute_error=maximum,max_gate_fraction=fraction)


def exact_logits_controls(logits,boundary,initial,selections,capture):
    for case,spec in reference.CASES.items():
        full=logits[(case,'chunk32')]
        if len(full)!=len(spec['tokens'])*VOCAB or any(logits[(case,'chunk'+str(chunk))]!=full for chunk in [128,3]):raise ValueError('Exact native schedules differ: '+case)
    for chunk in CHUNKS:
        if logits[('short','chunk'+str(chunk))][:3*VOCAB]!=logits[('future_changed','chunk'+str(chunk))][:3*VOCAB]:raise ValueError('Native causal prefix differs')
    if boundary['boundary.chunk32.logits']!=boundary['boundary.chunk128.logits']:raise ValueError('Full512 boundary schedules differ')
    if initial['greedy.step_0.logits']!=initial['native_generated.step_0.logits']:raise ValueError('Initial native journey distribution differs')
    for group,field in [('greedy','native_selected_on_teacher_forced_prefixes'),('native_generated','native_generated_ids')]:
        if [selections[f'{group}.step_{step}.logits']for step in range(4)]!=capture[field]:raise ValueError('Recorded native lowest-ID maximum differs')


def bounded_json(path,limit):
    with path.open('rb')as f:raw=f.read(limit+1)
    if len(raw)>limit:raise ValueError('Oversized capture metadata')
    return json.loads(raw,object_pairs_hook=reference.unique)


def check(source,source_sha,folder):
    oracle=reference.verify(source,source_sha)
    capture=bounded_json(folder/'capture.json',4*1024**2)
    if (capture['format']!='toklet-llama3-checkpoint-capture-v1' or capture['repository']!=reference.REPO
            or capture['revision']!=reference.REVISION or capture['source_manifest_sha256']!=source_sha
            or capture['profile']!='MlxAffine/Compensated/Reference/Half' or capture['implemented_positions']!=512
            or capture['declared_positions']!=131072 or capture['requested_chunks']!=CHUNKS
            or capture['effective_chunk']!='min(requested,512-position_offset)'
            or capture['numerical_comparison_pending']is not True
            or any(capture[key]is not False for key in ['held_out_quality_qualified','app_qualified','iphone_qualified'])):
        raise ValueError('Fixed capture profile/qualification state differs')
    admitted=capture['maximum_admitted_bytes'];app=capture['application_allowance_bytes']
    if type(admitted)is not int or not 0<admitted<=6*1024**3 or type(app)is not int or app!=32*1024**2:
        raise ValueError('Reported diagnostic memory admission differs')
    size=validate_inventory(capture['arrays'])
    if size!=capture['array_bytes']or(folder/'arrays.bin').stat().st_size!=size:raise ValueError('Native capture size differs')
    with(folder/'arrays.jsonl').open('rb')as f:raw=f.read(8*1024**2+1)
    if len(raw)>8*1024**2:raise ValueError('Oversized descriptor journal')
    ledger=[json.loads(line,object_pairs_hook=reference.unique)for line in raw.splitlines()]
    if len(ledger)!=2*len(capture['arrays']):raise ValueError('Incomplete descriptor journal')
    for index,item in enumerate(capture['arrays']):
        started=ledger[2*index]
        if started!=dict(event='started',name=item['name'],row_start=item['row_start'],shape=item['shape'],offset=item['offset'],expected_bytes=item['bytes'])or ledger[2*index+1]!=item:
            raise ValueError('Descriptor journal does not match final capture')
    metrics={};logits={};selections={};boundary={};observed=0;initial={}
    with(folder/'arrays.bin').open('rb')as native,(source/'arrays.bin').open('rb')as expected:
        for item in capture['arrays']:
            data=native.read(item['bytes'])
            if len(data)!=item['bytes']or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Native array hash differs')
            values=array('f');values.frombytes(data)
            if sys.byteorder!='little':values.byteswap()
            if any(not math.isfinite(v)for v in values):raise ValueError('Nonfinite native array')
            name=item['name'];key=name
            if name.startswith('chunk'):
                schedule,key=name.split('.',1)
                if key.endswith('.logits'):
                    case=key.split('.')[1];logits.setdefault((case,schedule),array('f')).extend(values)
            if key.startswith(('cases.','greedy.','scoring.')):
                spec=oracle['arrays'][key];columns=math.prod(spec['shape'][1:])if len(spec['shape'])>1 else spec['shape'][0]
                width=2 if spec['dtype']=='<f2'else 4
                expected.seek(spec['offset']+item['row_start']*columns*width)
                raw=expected.read(len(values)*width)
                if len(raw)!=len(values)*width:raise ValueError('Reference slice truncated')
                metrics[(name,item['row_start'])]=qualify(values,struct.iter_unpack('<e'if width==2 else'<f',raw),spec['gate'])
            else:observed+=len(values)
            if name.startswith(('greedy.','native_generated.')):
                selections[name]=max(range(VOCAB),key=values.__getitem__)
            if name.startswith('boundary.'):boundary[name]=values
            if name in ('greedy.step_0.logits','native_generated.step_0.logits'):initial[name]=values
        if native.read(1):raise ValueError('Trailing native bytes')
    exact_logits_controls(logits,boundary,initial,selections,capture)
    failed=sum(v['failed']for v in metrics.values())
    return dict(format='toklet-llama3-checkpoint-comparison-v1',passed=failed==0,source_manifest_sha256=source_sha,
                arrays=len(capture['arrays']),compared_values=sum(v['values']for v in metrics.values()),failed_values=failed,
                max_absolute_error=max(v['max_absolute_error']for v in metrics.values()),max_gate_fraction=max(v['max_gate_fraction']for v in metrics.values()),
                source_compared_arrays=len(metrics),raw_state_observed_values=observed,
                exact_numeric_logits_schedules=True,exact_numeric_logits_causal_prefix=True,exact_numeric_logits_boundary_schedules=True,
                native_generated_ids=capture['native_generated_ids'],metrics=[dict(name=k[0],row_start=k[1],**v)for k,v in metrics.items()],
                full_quality_qualified=False,app_qualified=False,iphone_qualified=False,
                scope='Fixed short source comparisons;512 boundary state and raw-native journey are separate finite/state controls, not held-out quality')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('source_sha');parser.add_argument('capture',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=check(args.source,args.source_sha,args.capture)
    with args.output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k]for k in ['passed','compared_values','failed_values','max_gate_fraction']}))
    sys.exit(0 if result['passed']else 1)
