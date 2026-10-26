"""Independent complete original packed projection/session replay.

Numerical expectations derive from the frozen pre-target tiny tables, exact
power-of-two metadata transforms, and independent rational/FP64 projection math.
Target input, output and cached bounds never define the expected schedule.
"""
from pathlib import Path
import copy,re,struct,sys
import check_gemma_immutable_weights as weightsref
base=weightsref.old.base
mathref=base.reference
same=base.same
S,CALLER=base.SENTINEL,base.CALLER
TEST='metal::gemma_affine::projection::tests::gpu::immutable_projection_reuses_weights_and_preserves_complete_transactions'
ROLES=('original','double','hot','oversized','rows','width')
OWNER_BYTES=240
HOT=0x4b800000
MISMATCH='Gemma projection weights device/shape/chunk mismatch'

def records(raw):
    if len(raw)>32*1024**2:raise ValueError('bounded transcript required')
    events=[];wrapper=[]
    for line in raw.decode().splitlines():
        if len(line.encode())>4*1024**2:raise ValueError('bounded record required')
        if 'GEMMA_PROJECTION_' in line:
            prefix,body=line.split('GEMMA_PROJECTION_',1)
            if prefix not in ('','test '+TEST+' ... '):raise ValueError('unexpected test prefix')
            label,text=body.split(' ',1);value=mathref.document(text)
            if type(value)is not dict:raise ValueError('object record required')
            events.append((label,value))
        elif line.strip():wrapper.append(line)
    if len(wrapper)!=3 or wrapper[:2]!=['running 1 test','ok'] or not re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s',wrapper[2]):raise ValueError('exact successful wrapper required')
    return events

def policy():
    c=dict(two_contexts=128*1024**2,immutable_weights=8*1024**2,source_loading=4*1024**2,
           two_projection_workspaces=8*1024**2,frozen_source_fixture=32*1024**2,
           serialization=64*1024**2,diagnostic_state=64*1024**2,caller=1024**2)
    return dict(categories=c,total_policy_bytes=sum(c.values()),native_process_allowance_bytes=512*1024**2,exact_physical_residency=False)

def doubled(bits,mantissa):
    if bits&((1<<(mantissa+8))-1)==0:return bits
    return mathref.rounded(mathref.rational(bits,mantissa,8)*2,mantissa,8)

def variant(case,role):
    item=copy.deepcopy(case);n,k=case['rows'],case['columns']
    if role=='double':
        for name in ('scales','biases'):item[name]=b''.join(struct.pack('<H',doubled(v,7)) for v, in struct.iter_unpack('<H',case[name]))
        item['weights']=[doubled(v,23) for v in case['weights']]
    if role in ('hot','rows','width'):
        if role=='rows':n+=1
        if role=='width':k=128 if k==64 else 64
        item.update(rows=n,columns=k,codes=bytes(n*k//2),scales=bytes(n*k//32),
                    biases=struct.pack('<H',0x4b80 if role=='hot' else 0x3f80)*(n*k//64),
                    weights=[HOT if role=='hot' else 0x3f800000]*(n*k))
    return item

def partition(role,maximum):return 15-maximum if role=='double' else 9 if role=='oversized' else 8 if role in ('rows','width') else maximum

def request(case,index):
    n,k=case['rows'],case['columns']
    owner=1 if index in (1,3,17,19) else 2 if index in (4,5,20) else 4 if index==7 else 5 if index==8 else 3 if index==9 else 0
    count=3 if index in (0,1,6,16,17,18,19,20,21) else 2 if index==3 else 4 if index==10 else 0 if index==15 else 1
    inputs=[case['input'][i%len(case['input'])] for i in range(count*k)]
    if index in (4,20):inputs=[0x3f800000 if i%k==0 else 0 for i in range(count*k)]
    if index==5:inputs=[0x3f800000]*k
    if index==11:inputs[0]=0x7fc00000
    if index==14:inputs.pop()
    length=count*n+ (1 if index==12 else -1 if index==13 else 0)
    error=('Gemma affine input ordered-sum accuracy bound' if index==5 else MISMATCH if index in (7,8,9)
           else 'Gemma projection input shape/nonfinite value' if 10<=index<=15 else 'Gemma projection injected composition failure' if index==17
           else 'Gemma projection workspace poisoned; recreate' if index==18 else None)
    return dict(index=index,role=ROLES[owner],tokens=count,input_bits=inputs,caller_length=length),error

def cold(n,k):
    full,chunk=3*n,24
    owned=(3*k+24+1+21+full+chunk)*4+OWNER_BYTES
    return dict(rows=n,columns=k,maximum_rows=8,capacity=3,owned_bytes=owned,poisoned=False,pending=False,shared_read_safe=True,
                staged_capacity=full,chunk_capacity=chunk,staged_bits=[0]*full,chunk_bits=[0]*chunk,
                **{name:dict(length=count,words=[0]*count) for name,count in [('input',3*k+7),('output',31),('status',8)]})

def evolve(state,case,maximum,attempt,error):
    caller=[CALLER]*attempt['caller_length'];tokens=attempt['tokens'];n,k=case['rows'],case['columns']
    if error is None or attempt['index']==17:
        output=base.projection(tuple(case['weights']),k,tuple(attempt['input_bits']))
        for index,first in enumerate(range(0,n,maximum)):
            rows=min(maximum,n-first)
            values=[v for token in range(tokens) for v in output[token*n+first:token*n+first+rows]]
            for name,active in [('input',attempt['input_bits']),('output',values),('status',[0])]:
                state[name]['words']=active+[S]*(state[name]['length']-len(active))
            state['chunk_bits'][:len(values)]=values
            for token in range(tokens):
                start=token*n+first;state['staged_bits'][start:start+rows]=values[token*rows:(token+1)*rows]
            if attempt['index']==17 and index==1:break
        state['poisoned']=error is not None
        if error is None:caller=state['staged_bits'][:attempt['caller_length']]
    return caller

class Replay:
    def __init__(self,events):self.events=iter(events);self.good=0;self.callbacks=0;self.chunks=0;self.devices=0
    def take(self,label,wanted=None):
        try:actual,value=next(self.events)
        except StopIteration as error:raise ValueError('missing '+label) from error
        if actual!=label:raise ValueError('wrong record '+actual+'/'+label)
        if wanted is not None:same(value,wanted)
        return value
    @staticmethod
    def pointer(value):
        if type(value)is not str or not re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}',value):raise ValueError('native identity required')
        return value
    def enroll(self,wanted,actual):
        pointer=self.pointer(actual.get('allocation'))
        if pointer in self.live:raise ValueError('live GPU allocation alias')
        self.live.add(pointer);self.devices+=1;wanted['allocation']=pointer;same(actual,wanted)
    def owner(self,case,maximum,device,actual):
        expected=weightsref.model(case,maximum,device)
        chunks=[]
        for child,got in zip(expected['children'],actual['chunks'],strict=True):
            entry=dict(rows=child['rows'],columns=case['columns'])
            for role in ('codes','scales','biases'):
                entry[role]=dict(length=len(child[role]),words=child[role]);self.enroll(entry[role],got[role])
            chunks.append(entry)
        view=dict(rows=case['rows'],columns=case['columns'],owned_bytes=expected['owned_bytes'],owner_slots=expected['slots'],chunk_struct_bytes=112,
                  device=device,maximum_bits=struct.unpack('<Q',struct.pack('<d',expected['max_abs_weight']))[0],chunks=chunks)
        same(actual,view);self.chunks+=len(chunks);return view
    def workspace(self,n,k,actual):
        state=cold(n,k)
        for name in ('input','output','status'):self.enroll(state[name],actual[name])
        same(actual,state);return state
    def run(self,cases):
        self.take('PLAN',policy())
        for case in cases:
            n,k=case['rows'],case['columns']
            for maximum in (7,8):
                self.live=set();key=f"{case['name']}/{maximum}"
                self.take('CASE',dict(key=key,rows=n,columns=k,partition=maximum,projection_struct_bytes=OWNER_BYTES,weights_struct_bytes=112))
                context=self.take('CONSTRUCTION_CONTEXT');name=context.get('device');device=self.pointer(context.get('identity'))
                if type(name)is not str or not 1<=len(name)<=256:raise ValueError('named device required')
                same(context,dict(device=name,identity=device));owners=[];variants=[]
                for role in ROLES:
                    source=variant(case,role);variants.append(source);part=partition(role,maximum);v=1 if role=='double' else 2 if role=='hot' else 3 if role in ('rows','width') else 0
                    self.take('UPLOAD_ATTEMPT',dict(role=role,variant=v,rows=source['rows'],columns=source['columns'],maximum_rows=part,working_budget=1024**2))
                    for first in range(0,source['rows'],part):
                        self.take('FILL',dict(role=role,first_row=first,rows=min(part,source['rows']-first)));self.callbacks+=1
                    self.take('UPLOAD_OUTCOME',dict(role=role,completed=True,error=None))
                    actual=self.take('WEIGHTS');owners.append(self.owner(source,part,device,actual['observation']));same(actual,dict(role=role,observation=owners[-1]))
                self.take('CONSTRUCTION_CONTEXT_DROPPED',dict(owners=6,devices=[device]*6))
                self.take('LOOKUP_CONTEXT',dict(device=name,identity=device,same_devices=[True]*6))
                required=cold(n,k)['owned_bytes']
                self.take('BUDGET_OUTCOME',dict(required=required,limit=required-1,completed=False,error=f'Metal admission requires {required} bytes, budget {required-1}'))
                self.take('WORKSPACE_ATTEMPT',dict(phase='initial',budget=required));self.take('WORKSPACE_OUTCOME',dict(phase='initial',completed=True,error=None))
                state=self.workspace(n,k,self.take('COLD'));retired=None
                for index in range(22):
                    if index==19:
                        self.take('WORKSPACE_ATTEMPT',dict(phase='fresh',budget=required));self.take('WORKSPACE_OUTCOME',dict(phase='fresh',completed=True,error=None))
                        body=self.take('RECOVERY');same(body['retired'],state);retired=copy.deepcopy(state);state=self.workspace(n,k,body['fresh']);same(body,dict(retired=retired,fresh=state))
                    attempt,error=request(case,index);self.take('STEP_ATTEMPT',attempt)
                    source=variants[ROLES.index(attempt['role'])];part=partition(attempt['role'],maximum)
                    caller=evolve(state,source,part,attempt,error)
                    self.take('STEP_OUTCOME',dict(index=index,completed=error is None,error=error,l1_calls=1 if index in (4,5,20) else 0))
                    self.take('STEP',dict(index=index,caller_bits=caller,workspace=state,retired=retired,weights=owners))
                    if error is None:self.good+=len(caller)
                for role,owner in zip(ROLES,owners,strict=True):self.take('FINAL_WEIGHTS',dict(role=role,observation=owner))
                self.take('FINAL',dict(key=key,workspace=state,retired=retired))
        result=dict(cases=4,steps=88,construction_contexts_dropped=4,same_geometry_weight_swaps=True,foreign_device_qualified=False,
                    driver_or_inflight_failure_injected=False,whole_graph_qualified=False,performance_qualified=False)
        self.take('RESULT',result)
        try:next(self.events)
        except StopIteration:pass
        else:raise ValueError('unexpected trailing record')
        same([self.good,self.callbacks,self.chunks,self.devices],[1500,60,60,204])
        return dict(passed=True,**result,successful_output_values=self.good,source_callbacks=self.callbacks,published_chunks=self.chunks,
                    gpu_allocations_across_disjoint_epochs=self.devices,aggregate_workload_policy_bytes=policy()['total_policy_bytes'])

def replay(log,fixture):return Replay(records(mathref.read(log,32*1024**2))).run(base.fixture(fixture))
if __name__=='__main__':
    import json
    if len(sys.argv)!=3:raise SystemExit('native stdout and frozen table fixture directory')
    print(json.dumps(replay(Path(sys.argv[1]),Path(sys.argv[2])),allow_nan=False))
