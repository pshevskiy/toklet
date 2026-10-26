"""Independent admission/reader controls and complete captured-state mutations."""
from pathlib import Path
import copy,json,sys,unittest
import check_gemma_immutable_weights as c

def fixture_directory():
    return Path(ARGS[1]) if len(ARGS)==2 else Path.cwd()/'tests/fixtures/gemma-table-math-v1'

def controller_fixture(cases):
    """Synthetic reader/control-flow witness only; never a GPU qualification.

    Numerical rows come from the already frozen independent tiny fixture.
    Routing groups global requested rows by chunk, independently of Replay's
    traversal. Opaque device/buffer addresses are deliberately fabricated.
    """
    events=[];address=0x10000;device='0x1000'
    def emit(label,**fields):events.append((label,copy.deepcopy(fields)))
    def pointer():
        nonlocal address
        address+=0x1000;return hex(address)
    def allocate(values):return dict(allocation=pointer(),words=list(values))
    def owner(case,maximum):
        prior=c.old.matrix(case,maximum);children=[];size=112
        for oldchild in prior['children']:
            data={role:allocate(oldchild['device'][role]) for role in ('codes','scales','biases')}
            children.append(dict(rows=oldchild['rows'],shared_read_safe=True,allocations=[data[r]['allocation'] for r in data],**data))
            size+=112+sum(len(v['words'])*4 for v in data.values())
        scale=[c.old.base.f32(v<<16) for v, in __import__('struct').iter_unpack('<H',case['scales'])]
        bias=[c.old.base.f32(v<<16) for v, in __import__('struct').iter_unpack('<H',case['biases'])]
        maximum_value=max(max(abs(b),abs(15*s+b)) for s,b in zip(scale,bias,strict=True))
        return dict(rows=case['rows'],columns=case['columns'],owned_bytes=size,max_abs_weight=float(maximum_value),slots=len(children),chunk_struct_bytes=112,device=device,children=children)
    def source(case,first,count):
        k=case['columns'];return {r:list(case[r][first*stride:(first+count)*stride]) for r,stride in [('codes',k//2),('scales',k//32),('biases',k//32)]}
    def minimum(case,maximum,weights):
        count=min(case['rows'],maximum);k=case['columns']
        return weights['owned_bytes']+count*k//2+count*k//16+72
    def callbacks(case,maximum,key,prefix=''):
        locations=[pointer() for _ in range(3)]
        for first in range(0,case['rows'],maximum):
            count=min(maximum,case['rows']-first)
            if not prefix:emit('FILL_ATTEMPT',key=key,first_row=first,rows=count,addresses=locations)
            fields=dict(key=key,first_row=first,**source(case,first,count))
            if prefix:fields['rows']=count
            emit(prefix+'FILL',**fields)
    def cold(k):
        result=c.old.workspace(k,160);result['device']={r:allocate(v) for r,v in result.pop('workspace').items()};return result
    def route(state,case,maximum,ids,limit=None):
        k=case['columns'];groups=sorted({row//maximum for row in ids})
        for number,group in enumerate(groups,1):
            slots=[(i,row) for i,row in enumerate(ids) if row//maximum==group]
            values=[word for _,row in slots for word in case['weights'][row*k:(row+1)*k]]
            state['device']['ids']['words']=[row-group*maximum for _,row in slots]+[c.S]*(39-len(slots))
            state['device']['output']['words']=values+[c.S]*(32*k+7-len(values))
            state['device']['status']['words']=[0]+[c.S]*7
            state['chunk_bits'][:len(values)]=values
            for slot,row in slots:state['staged_bits'][slot*k:(slot+1)*k]=case['weights'][row*k:(row+1)*k]
            if number==limit:state['poisoned']=True;break
    def view(state,caller):
        result=copy.deepcopy(state);result['caller_bits']=caller;return result
    def step(state,case,maximum,key,index,ids,weights,error=None,limit=None):
        length=len(ids)*case['columns'];emit('STEP_ATTEMPT',key=key,index=index,ids=ids,caller_len=length)
        if error is None or limit:route(state,case,maximum,ids,limit)
        caller=[c.CALLER]*length if error else [v for row in ids for v in case['weights'][row*case['columns']:(row+1)*case['columns']]]
        emit('STEP_OUTCOME',key=key,index=index,completed=error is None,error=error)
        emit('STEP',key=key,index=index,workspace=view(state,caller),weights=weights)
    emit('PLAN',**c.policy())
    emit('META',context_policy_bytes=67108864,owner_struct_bytes=112,session_struct_bytes=160,whole_graph_qualified=False,performance_qualified=False)
    for case in cases:
        n,k=case['rows'],case['columns'];ids=[0,6,7,n-1,7]
        for maximum in (7,8):
            key=f"{case['name']}/max{maximum}";weights=owner(case,maximum);working=minimum(case,maximum,weights)
            emit('CONTEXT',key=key,phase='construction',device='Synthetic controller device',identity=device)
            emit('CONSTRUCT_ATTEMPT',key=key,rows=n,columns=k,maximum=maximum,budget=working)
            callbacks(case,maximum,key)
            emit('CONSTRUCT_OUTCOME',key=key,completed=True,error=None,callbacks=len(weights['children']))
            emit('WEIGHTS',key=key,observation=weights)
            if n==19 and maximum==7:
                for kind in ('budget','first-source','late-source','domain','panic'):
                    limit=working-1 if kind=='budget' else working
                    emit('FAULT_ATTEMPT',key=key,kind=kind,budget=limit,minimum=working);calls=0
                    scratch={r:[0]*len(v) for r,v in source(case,0,maximum).items()}
                    if kind!='budget':
                        for first in ([0] if kind=='first-source' else [0,maximum]):
                            calls+=1;emit('FAULT_CALLBACK_ATTEMPT',key=key,kind=kind,first_row=first,rows=maximum,**{r+'_before':v for r,v in scratch.items()})
                            if kind=='first-source':continue
                            scratch=source(case,first,maximum)
                            if first and kind=='domain':scratch['scales'][:2]=[127,127];scratch['biases'][:2]=[127,127]
                            emit('FAULT_CALLBACK',key=key,kind=kind,first_row=first,**scratch)
                    errors={'budget':f'Metal admission requires {working} bytes, budget {limit}','first-source':'immutable original first source','late-source':'immutable original late source','domain':'Gemma affine weight domain','panic':None}
                    emit('FAULT_OUTCOME',key=key,kind=kind,callbacks=calls,unwound=kind=='panic',published=False,error=errors[kind],panic_payload='immutable original panic' if kind=='panic' else None)
                    emit('FAULT_BASELINE',key=key,kind=kind,weights=weights)
            emit('CONSTRUCTION_CONTEXT_DROPPED',key=key,weights=weights)
            emit('CONTEXT',key=key,phase='reuse',device='Synthetic controller device',identity=device,same_device=True)
            state=cold(k);emit('WORKSPACE',key=key,workspace=state)
            for index,request in [(0,ids),(1,ids),(2,list(reversed(ids))),(3,[n-1]),(4,[n-1]*32)]:step(state,case,maximum,key,index,request,weights)
            step(state,case,maximum,key,5,[n],weights,'Gemma table gather request shape or indices')
            step(state,case,maximum,key,6,ids,weights,'Gemma table gather injected composition failure',2)
            step(state,case,maximum,key,7,ids,weights,'Gemma table gather poisoned; recreate workspace')
            recovered=cold(k);emit('RECOVERED_WORKSPACE',key=key,workspace=recovered)
            step(recovered,case,maximum,key,8,ids,weights)
            other=next(v for v in cases if v['columns']!=k);other_weights=owner(other,maximum)
            emit('MISMATCH_OWNER_ATTEMPT',key=key,rows=other['rows'],columns=other['columns'],maximum=maximum,budget=minimum(other,maximum,other_weights))
            callbacks(other,maximum,key,'MISMATCH_');emit('MISMATCH_OWNER_OUTCOME',key=key,completed=True,error=None,callbacks=len(other_weights['children']))
            emit('MISMATCH_OWNER',key=key,weights=other_weights)
            mismatch=dict(workspace=view(recovered,[c.CALLER]*k),weights=weights,other_weights=other_weights)
            emit('MISMATCH_BEFORE',key=key,observation=mismatch)
            emit('MISMATCH_OUTCOME',key=key,completed=False,error='Gemma table gather weights device/shape mismatch')
            emit('MISMATCH_AFTER',key=key,observation=mismatch)
            step(recovered,case,maximum,key,9,ids,weights)
            legacy=c.old.matrix(case,maximum)
            for child in legacy['children']:child['device']={r:allocate(v) for r,v in child['device'].items()}
            route(recovered,case,maximum,ids);legacy_view=view(recovered,[v for row in ids for v in case['weights'][row*k:(row+1)*k]])
            legacy_view['workspace']=legacy_view.pop('device');legacy_view['matrix']=legacy
            emit('LEGACY',key=key,completed=True,error=None,observation=legacy_view)
            emit('FINAL',key=key,weights=weights)
    emit('RESULT',cases=4,steps=40,source_contexts_dropped=4,foreign_device_qualified=False,whole_graph_qualified=False,driver_or_inflight_failure_injected=False,performance_qualified=False)
    return events

class Portable(unittest.TestCase):
    def test_ownership_and_staging_budgets_include_odd_metadata_padding(self):
        cases=c.old.base.fixture(fixture_directory())
        expected={(19,7):(2068,2644),(19,8):(2068,2716),(11,7):(904,1228),(11,8):(904,1264)}
        for case in cases:
            for maximum in (7,8):
                key=(case['rows'],maximum)
                self.assertEqual((c.model(case,maximum,'')['owned_bytes'],c.budget(case,maximum)),expected[key])
    def test_complete_workload_policy_keeps_context_and_scratch_separate(self):
        p=c.policy();self.assertEqual(p['total_policy_bytes'],310378496);self.assertLess(p['total_policy_bytes'],512*1024**2)
        self.assertEqual(p['categories']['two_context_policies'],134217728)
    def test_complete_synthetic_controller_protocol_and_changed_value_rejection(self):
        cases=c.old.base.fixture(fixture_directory())
        events=controller_fixture(cases)
        transcript='running 1 test\n'+'\n'.join('GEMMA_WEIGHTS_'+label+' '+json.dumps(item) for label,item in events)+'\nok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n'
        events=c.records(transcript.encode())
        # Explicitly synthetic: checks reader control flow, not native execution.
        self.assertEqual(c.Replay(events).run(cases)['successful_values'],22272)
        changed=copy.deepcopy(events)
        next(v for k,v in changed if k=='STEP')['workspace']['caller_bits'][0]=0
        with self.assertRaises(ValueError):c.Replay(changed).run(cases)
    def test_strict_record_wrapper_and_duplicate_fields(self):
        for raw in [b'GEMMA_WEIGHTS_PLAN {"a":1,"a":2}\n',b'GEMMA_WEIGHTS_PLAN {"a":NaN}\n',b'running 2 tests\n',b'x'*(32*1024**2+1)]:
            with self.assertRaises(ValueError):c.records(raw)

class Actual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if len(ARGS)!=2 or ARGS[0]=='portable':raise unittest.SkipTest('actual native capture supplied explicitly')
        capture,fixture=map(Path,ARGS)
        cls.events=c.records(c.old.reference.read(capture,32*1024**2));cls.cases=c.old.base.fixture(fixture)
    @staticmethod
    def item(events,label):return next(v for k,v in events if k==label)
    def rejected(self,change):
        events=copy.deepcopy(self.events);change(events)
        with self.assertRaises((ValueError,KeyError,IndexError,TypeError,StopIteration)):
            c.Replay(events).run(self.cases)
    def test_complete_capture(self):
        r=c.Replay(self.events).run(self.cases)
        self.assertEqual(r['steps'],40);self.assertEqual(r['source_contexts_dropped'],4)
        self.assertEqual(r['successful_values'],22272);self.assertEqual(r['published_chunks'],20)
        self.assertEqual(r['callbacks'],27);self.assertEqual(r['construction_failures'],5)
    def test_budget_record(self):self.rejected(lambda e:self.item(e,'CONSTRUCT_ATTEMPT').update(budget=1))
    def test_owner_bytes(self):self.rejected(lambda e:self.item(e,'WEIGHTS')['observation'].update(owned_bytes=1))
    def test_retained_device_identity(self):self.rejected(lambda e:self.item(e,'WEIGHTS')['observation'].update(device='0x12345'))
    def test_dropped_context_owner_contents(self):self.rejected(lambda e:self.item(e,'CONSTRUCTION_CONTEXT_DROPPED')['weights']['children'][0]['codes']['words'].__setitem__(0,0))
    def test_new_context_identity(self):
        def change(e):next(v for k,v in e if k=='CONTEXT' and v['phase']=='reuse')['identity']='0x1234'
        self.rejected(change)
    def test_source_bytes(self):self.rejected(lambda e:self.item(e,'FILL')['codes'].__setitem__(0,255))
    def test_source_offset(self):self.rejected(lambda e:self.item(e,'FILL').update(first_row=1))
    def test_source_reused_pointer(self):
        def change(e):[v for k,v in e if k=='FILL_ATTEMPT'][1]['addresses'][0]='0x123456'
        self.rejected(change)
    def test_coherent_source_alias(self):
        def change(e):
            for k,v in e:
                if k=='FILL_ATTEMPT':v['addresses']=['0x1000']*3
        self.rejected(change)
    def test_coherent_source_wrap(self):
        def change(e):
            for k,v in e:
                if k=='FILL_ATTEMPT':v['addresses'][0]='0xfffffffffffffff0'
        self.rejected(change)
    def test_immutable_live_alias(self):
        def change(e):
            d=self.item(e,'WEIGHTS')['observation']['children'][0];d['scales']['allocation']=d['codes']['allocation'];d['allocations'][1]=d['allocations'][0]
        self.rejected(change)
    def test_original_error(self):self.rejected(lambda e:self.item(e,'FAULT_OUTCOME').update(error='replaced original'))
    def test_original_panic_payload(self):
        self.rejected(lambda e:next(v for k,v in e if k=='FAULT_OUTCOME' and v['kind']=='panic').update(panic_payload='replaced panic'))
    def test_zero_callback_budget_reject(self):self.rejected(lambda e:self.item(e,'FAULT_OUTCOME').update(callbacks=1))
    def test_fault_baseline_changed(self):self.rejected(lambda e:self.item(e,'FAULT_BASELINE')['weights']['children'][0]['biases']['words'].__setitem__(0,0))
    def test_step_result(self):self.rejected(lambda e:self.item(e,'STEP_OUTCOME').update(completed=False,error='failed'))
    def test_safe_observation_contract(self):self.rejected(lambda e:self.item(e,'STEP')['weights']['children'][0].update(shared_read_safe=False))
    def test_allocation_replacement(self):self.rejected(lambda e:self.item(e,'STEP')['workspace']['device']['ids'].update(allocation='0x9876'))
    def test_caller_values(self):self.rejected(lambda e:self.item(e,'STEP')['workspace']['caller_bits'].__setitem__(0,0))
    def test_staged_values(self):self.rejected(lambda e:self.item(e,'STEP')['workspace']['staged_bits'].__setitem__(0,0))
    def test_chunk_tail(self):self.rejected(lambda e:self.item(e,'STEP')['workspace']['chunk_bits'].__setitem__(-1,42))
    def test_code_buffer_guard(self):self.rejected(lambda e:self.item(e,'WEIGHTS')['observation']['children'][0]['codes']['words'].__setitem__(-1,0))
    def test_workspace_status(self):self.rejected(lambda e:self.item(e,'STEP')['workspace']['device']['status']['words'].__setitem__(0,1))
    def test_coherent_all_target_numerical_views(self):
        def change(e):
            for k,v in e:
                if k=='STEP':
                    w=v['workspace']
                    for name in ('caller_bits','staged_bits','chunk_bits'):w[name]=[0]*len(w[name])
                    w['device']['output']['words']=[0]*len(w['device']['output']['words'])
        self.rejected(change)
    def test_sticky_poison(self):
        self.rejected(lambda e:next(v for k,v in e if k=='STEP' and v['index']==7)['workspace'].update(poisoned=False))
    def test_recovery_alias(self):
        def change(e):
            old=self.item(e,'WORKSPACE')['workspace'];new=self.item(e,'RECOVERED_WORKSPACE')['workspace'];new['device']['ids']['allocation']=old['device']['ids']['allocation']
        self.rejected(change)
    def test_new_shape_branch_error(self):self.rejected(lambda e:self.item(e,'MISMATCH_OUTCOME').update(error='Gemma table gather request shape or indices'))
    def test_new_shape_branch_state_change(self):self.rejected(lambda e:self.item(e,'MISMATCH_AFTER')['observation']['workspace']['staged_bits'].__setitem__(0,0))
    def test_other_owner_live_alias(self):
        def change(e):
            first=self.item(e,'WEIGHTS')['observation']['children'][0]['codes']['allocation'];other=self.item(e,'MISMATCH_OWNER')['weights']['children'][0];other['codes']['allocation']=first;other['allocations'][0]=first
        self.rejected(change)
    def test_legacy_matrix_scratch_untouched(self):self.rejected(lambda e:self.item(e,'LEGACY')['observation']['matrix']['staged_bits'].__setitem__(0,1))
    def test_legacy_workspace_values(self):self.rejected(lambda e:self.item(e,'LEGACY')['observation']['caller_bits'].__setitem__(0,0))
    def test_missing_drop_witness(self):self.rejected(lambda e:e.pop(next(i for i,(k,v) in enumerate(e) if k=='CONSTRUCTION_CONTEXT_DROPPED')))
    def test_foreign_device_grant(self):self.rejected(lambda e:self.item(e,'RESULT').update(foreign_device_qualified=True))

ARGS=sys.argv[1:]
if __name__=='__main__':
    actual=len(ARGS)==2 and ARGS[0]!='portable'
    unittest.main(argv=[sys.argv[0],'Actual' if actual else 'Portable'])
