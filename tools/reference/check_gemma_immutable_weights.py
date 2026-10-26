"""Independent byte/state replay for immutable GPU weights and shared lookup.

Uses only pre-existing packed-table mathematics and legacy buffer-layout helpers;
no native result is used to construct expected numerical values or state.
"""
from pathlib import Path
import copy,re,struct,sys
import check_gemma_gather as old

same=old.same
S=old.SENTINEL
CALLER=old.CALLER
TEST='metal::gemma_affine::partitioned::gather::tests::weights::immutable_weights_outlive_context_and_reuse_fresh_workspaces'

def records(raw):
    if len(raw)>32*1024**2:raise ValueError('bounded transcript required')
    result=[];wrapper=[]
    for line in raw.decode().splitlines():
        if len(line.encode())>4*1024**2:raise ValueError('bounded record required')
        if 'GEMMA_WEIGHTS_' in line:
            prefix,tail=line.split('GEMMA_WEIGHTS_',1)
            if prefix not in ('','test '+TEST+' ... '):raise ValueError('unexpected test prefix')
            label,body=tail.split(' ',1);result.append((label,old.reference.document(body)))
        elif line.strip():wrapper.append(line)
    if len(wrapper)!=3 or wrapper[:2]!=['running 1 test','ok'] or not re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s',wrapper[2]):raise ValueError('complete exact successful wrapper required')
    return result

def policy():
    categories=dict(two_context_policies=2*64*1024**2,two_workspace_policies=2*32*1024**2,live_weight_owner_allowance=1024**2,
                    legacy_owner_allowance=1024**2,failed_upload_overlap_allowance=4*1024**2,source_staging_allowance=1024**2,
                    fixture_host_policy=32*1024**2,serialization_policy=64*1024**2,caller_policy=1024**2)
    total=sum(categories.values())
    if total>512*1024**2:raise ValueError('workload exceeds process policy')
    return dict(categories=categories,total_policy_bytes=total,native_process_allowance_bytes=512*1024**2,exact_physical_residency=False)

def parts(case,first,rows):
    k=case['columns']
    return {role:list(case[role][first*stride:(first+rows)*stride]) for role,stride in [('codes',k//2),('scales',k//32),('biases',k//32)]}

def model(case,maximum,device):
    chunks=[];owned=112;maximum_value=0.
    for first in range(0,case['rows'],maximum):
        n=min(maximum,case['rows']-first);source=parts(case,first,n)
        child=dict(rows=n,shared_read_safe=True,allocations=[])
        for role,raw in source.items():
            raw=bytes(raw);values=old.base.words(raw+bytes((-len(raw))%4))+[S]*7
            child[role]=values;owned+=len(values)*4
        chunks.append(child);owned+=112
    for (sb,),(bb,) in zip(struct.iter_unpack('<H',case['scales']),struct.iter_unpack('<H',case['biases']),strict=True):
        a=old.reference.rational(sb,7,8);b=old.reference.rational(bb,7,8)
        for code in range(16):maximum_value=max(maximum_value,abs(old.base.f32(old.reference.rounded(a*code+b,23,8))))
    return dict(rows=case['rows'],columns=case['columns'],owned_bytes=owned,max_abs_weight=float(maximum_value),slots=len(chunks),chunk_struct_bytes=112,device=device,children=chunks)

def budget(case,maximum):
    owner=model(case,maximum,'')['owned_bytes'];k=case['columns'];rows=min(maximum,case['rows'])
    return owner+rows*k//2+2*rows*k//32+72

class Replay(old.Replay):
    def __init__(self,events):
        super().__init__(events);self.good_values=0;self.callbacks=0;self.published_chunks=0;self.failures=0;self.legacy=0
    @staticmethod
    def pointer(value):
        if type(value)is not str or not re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}',value):raise ValueError('native object identity required')
        return value
    def enroll_model(self,wanted,actual):
        for expected,got in zip(wanted['children'],actual['children'],strict=True):
            buffers={name:expected[name] for name in ('codes','scales','biases')}
            self.allocations(buffers,{name:got[name] for name in buffers})
            expected.update(buffers);expected['allocations']=[buffers[name]['allocation'] for name in ('codes','scales','biases')]
        same(wanted,actual);self.published_chunks+=len(wanted['children'])
    def callbacks_for(self,case,maximum,key,label=''):
        pointers=None
        for first in range(0,case['rows'],maximum):
            n=min(maximum,case['rows']-first)
            if not label:
                attempt=self.take('FILL_ATTEMPT');addresses=attempt.get('addresses')
                if pointers is None:
                    if type(addresses)is not list or len(addresses)!=3:raise ValueError('three source addresses required')
                    intervals=[];k=case['columns'];lengths=[min(maximum,case['rows'])*k//2,min(maximum,case['rows'])*k//32,min(maximum,case['rows'])*k//32]
                    for value,length in zip(addresses,lengths,strict=True):
                        start=int(self.pointer(value),16);end=start+length
                        if end>2**64 or any(start<right and left<end for left,right in intervals):raise ValueError('source intervals overlap or overflow')
                        intervals.append((start,end))
                    pointers=addresses
                self.take_value(attempt,dict(key=key,first_row=first,rows=n,addresses=pointers))
            body=dict(key=key,first_row=first,**parts(case,first,n))
            if label:body['rows']=n
            self.take(label+'FILL',body);self.callbacks+=1
    @staticmethod
    def take_value(actual,expected):same(actual,expected)
    def fault(self,kind,case,maximum,key,weights):
        required=budget(case,maximum);limit=required-1 if kind=='budget' else required
        self.take('FAULT_ATTEMPT',dict(key=key,kind=kind,budget=limit,minimum=required))
        count=0
        if kind!='budget':
            scratch={name:[0]*len(raw) for name,raw in parts(case,0,maximum).items()}
            for first in ([0] if kind=='first-source' else [0,maximum]):
                body=dict(key=key,kind=kind,first_row=first,rows=maximum,**{r+'_before':v for r,v in scratch.items()})
                self.take('FAULT_CALLBACK_ATTEMPT',body);count+=1;self.callbacks+=1
                if kind=='first-source':continue
                scratch=parts(case,first,maximum)
                if first and kind=='domain':scratch['scales'][:2]=[127,127];scratch['biases'][:2]=[127,127]
                self.take('FAULT_CALLBACK',dict(key=key,kind=kind,first_row=first,**scratch))
        errors={'budget':f'Metal admission requires {required} bytes, budget {limit}','first-source':'immutable original first source',
                'late-source':'immutable original late source','domain':'Gemma affine weight domain','panic':None}
        self.take('FAULT_OUTCOME',dict(key=key,kind=kind,callbacks=count,unwound=kind=='panic',published=False,error=errors[kind],panic_payload='immutable original panic' if kind=='panic' else None))
        self.take('FAULT_BASELINE',dict(key=key,kind=kind,weights=weights));self.failures+=1
    def cold_workspace(self,key,k,label):
        wanted=old.workspace(k,160);wanted['device']=wanted.pop('workspace')
        actual=self.take(label)
        self.allocations(wanted['device'],actual['workspace']['device'])
        same(actual,dict(key=key,workspace=wanted));return wanted
    @staticmethod
    def evolve(state,case,maximum,ids,length,error=None,completed_chunks=None):
        caller=[CALLER]*length
        if error is None or completed_chunks is not None:
            completed=0;k=case['columns']
            for first in range(0,case['rows'],maximum):
                n=min(maximum,case['rows']-first);slots=[i for i,row in enumerate(ids) if first<=row<first+n]
                if not slots:continue
                local=[ids[i]-first for i in slots]
                output=[v for i in slots for v in case['weights'][ids[i]*k:(ids[i]+1)*k]]
                for name,values in [('ids',local),('output',output),('status',[0])]:
                    state['device'][name]['words']=values+[S]*(len(state['device'][name]['words'])-len(values))
                state['chunk_bits'][:len(output)]=output
                for offset,slot in enumerate(slots):state['staged_bits'][slot*k:(slot+1)*k]=output[offset*k:(offset+1)*k]
                completed+=1
                if completed_chunks is not None and completed==completed_chunks:break
            if error is None:caller=state['staged_bits'][:length]
            else:state['poisoned']=True
        view=copy.deepcopy(state);view['caller_bits']=caller;return view
    def check_step(self,state,case,maximum,key,index,ids,length,weights,error=None,completed=None):
        self.take('STEP_ATTEMPT',dict(key=key,index=index,ids=ids,caller_len=length))
        view=self.evolve(state,case,maximum,ids,length,error,completed)
        self.take('STEP_OUTCOME',dict(key=key,index=index,completed=error is None,error=error))
        self.take('STEP',dict(key=key,index=index,workspace=view,weights=weights))
        self.steps+=1
        if error is None:self.good_values+=length
    def run(self,cases):
        self.take('PLAN',policy())
        self.take('META',dict(context_policy_bytes=67108864,owner_struct_bytes=112,session_struct_bytes=160,whole_graph_qualified=False,performance_qualified=False))
        for case in cases:
            n,k=case['rows'],case['columns'];canonical=[0,6,7,n-1,7]
            for maximum in (7,8):
                self.live=set();key=f"{case['name']}/max{maximum}"
                context=self.take('CONTEXT');device=context.get('device');identity=self.pointer(context.get('identity'))
                if type(device)is not str or not 1<=len(device)<=256:raise ValueError('named device required')
                same(context,dict(key=key,phase='construction',device=device,identity=identity))
                self.take('CONSTRUCT_ATTEMPT',dict(key=key,rows=n,columns=k,maximum=maximum,budget=budget(case,maximum)))
                self.callbacks_for(case,maximum,key)
                self.take('CONSTRUCT_OUTCOME',dict(key=key,completed=True,error=None,callbacks=(n+maximum-1)//maximum))
                wanted=model(case,maximum,identity);body=self.take('WEIGHTS');self.enroll_model(wanted,body['observation']);same(body,dict(key=key,observation=wanted))
                if n==19 and maximum==7:
                    for kind in ('budget','first-source','late-source','domain','panic'):self.fault(kind,case,maximum,key,wanted)
                self.take('CONSTRUCTION_CONTEXT_DROPPED',dict(key=key,weights=wanted))
                self.take('CONTEXT',dict(key=key,phase='reuse',device=device,identity=identity,same_device=True))
                state=self.cold_workspace(key,k,'WORKSPACE')
                for index in (0,1):self.check_step(state,case,maximum,key,index,canonical,len(canonical)*k,wanted)
                self.check_step(state,case,maximum,key,2,list(reversed(canonical)),len(canonical)*k,wanted)
                for index,count in ((3,1),(4,32)):self.check_step(state,case,maximum,key,index,[n-1]*count,count*k,wanted)
                self.check_step(state,case,maximum,key,5,[n],k,wanted,'Gemma table gather request shape or indices')
                self.check_step(state,case,maximum,key,6,canonical,len(canonical)*k,wanted,'Gemma table gather injected composition failure',2)
                self.check_step(state,case,maximum,key,7,canonical,len(canonical)*k,wanted,'Gemma table gather poisoned; recreate workspace')
                recovered=self.cold_workspace(key,k,'RECOVERED_WORKSPACE')
                self.check_step(recovered,case,maximum,key,8,canonical,len(canonical)*k,wanted)
                other=next(c for c in cases if c['columns']!=k);other_wanted=model(other,maximum,identity)
                self.take('MISMATCH_OWNER_ATTEMPT',dict(key=key,rows=other['rows'],columns=other['columns'],maximum=maximum,budget=budget(other,maximum)))
                self.callbacks_for(other,maximum,key,'MISMATCH_')
                self.take('MISMATCH_OWNER_OUTCOME',dict(key=key,completed=True,error=None,callbacks=len(other_wanted['children'])))
                body=self.take('MISMATCH_OWNER');self.enroll_model(other_wanted,body['weights']);same(body,dict(key=key,weights=other_wanted))
                rejected=copy.deepcopy(recovered);rejected['caller_bits']=[CALLER]*k
                view=dict(workspace=rejected,weights=wanted,other_weights=other_wanted)
                self.take('MISMATCH_BEFORE',dict(key=key,observation=view))
                self.take('MISMATCH_OUTCOME',dict(key=key,completed=False,error='Gemma table gather weights device/shape mismatch'))
                self.take('MISMATCH_AFTER',dict(key=key,observation=view))
                self.check_step(recovered,case,maximum,key,9,canonical,len(canonical)*k,wanted)
                legacy=old.matrix(case,maximum)
                body=self.take('LEGACY');self.matrix_ids(legacy,body['observation']['matrix'])
                view=self.evolve(recovered,case,maximum,canonical,len(canonical)*k)
                view['workspace']=view.pop('device');view['matrix']=legacy
                same(body,dict(key=key,completed=True,error=None,observation=view));self.legacy+=1
                self.take('FINAL',dict(key=key,weights=wanted))
        self.take('RESULT',dict(cases=4,steps=40,source_contexts_dropped=4,foreign_device_qualified=False,whole_graph_qualified=False,driver_or_inflight_failure_injected=False,performance_qualified=False))
        try:next(self.captured)
        except StopIteration:pass
        else:raise ValueError('unexpected trailing event')
        return dict(passed=True,cases=4,steps=self.steps,successful_values=self.good_values,published_weights=8,published_chunks=self.published_chunks,
                    callbacks=self.callbacks,construction_failures=self.failures,legacy_paths=self.legacy,source_contexts_dropped=4,
                    aggregate_workload_policy_bytes=policy()['total_policy_bytes'],whole_graph_qualified=False,foreign_device_qualified=False,performance_qualified=False)

def replay(path,directory):return Replay(records(old.reference.read(path,32*1024**2))).run(old.base.fixture(directory))
if __name__=='__main__':
    import json
    if len(sys.argv)!=3:raise SystemExit('native stdout and immutable table fixture directory')
    print(json.dumps(replay(Path(sys.argv[1]),Path(sys.argv[2]))))
