"""Portable reader controls and explicit captured projection mutation checks."""
from pathlib import Path
from fractions import Fraction
import copy,json,struct,sys,unittest
import check_gemma_projection_session as c

ARGS=sys.argv[1:]
def fixture_directory():return Path(ARGS[1]) if len(ARGS)==2 else Path.cwd()/'tests/fixtures/gemma-table-math-v1'

def controller_fixture(cases):
    """Complete synthetic reader-flow control only; never native/GPU evidence.

    Uses the pre-target numerical source/math, fabricated object IDs and an
    independently arranged global-output-to-child scatter for mutable state.
    """
    events=[];address=0x10000;device='0x1000'
    def emit(label,**value):events.append((label,copy.deepcopy(value)))
    def buf(values):
        nonlocal address
        address+=0x1000
        return dict(allocation=hex(address),length=len(values),words=list(values))
    def owner(source,maximum):
        prior=c.weightsref.model(source,maximum,device);chunks=[]
        for item in prior['children']:
            chunks.append(dict(rows=item['rows'],columns=source['columns'],**{role:buf(item[role]) for role in ('codes','scales','biases')}))
        return dict(rows=source['rows'],columns=source['columns'],owned_bytes=prior['owned_bytes'],owner_slots=len(chunks),
                    chunk_struct_bytes=112,device=device,maximum_bits=struct.unpack('<Q',struct.pack('<d',prior['max_abs_weight']))[0],chunks=chunks)
    def workspace(n,k):
        full=3*n;chunk=24
        return dict(rows=n,columns=k,maximum_rows=8,capacity=3,owned_bytes=(3*k+24+1+21+full+chunk)*4+240,
                    poisoned=False,pending=False,shared_read_safe=True,staged_capacity=full,chunk_capacity=chunk,
                    staged_bits=[0]*full,chunk_bits=[0]*chunk,input=buf([0]*(3*k+7)),output=buf([0]*31),status=buf([0]*8))
    emit('PLAN',**c.policy())
    for case in cases:
        n,k=case['rows'],case['columns']
        for maximum in (7,8):
            key=f"{case['name']}/{maximum}"
            emit('CASE',key=key,rows=n,columns=k,partition=maximum,projection_struct_bytes=240,weights_struct_bytes=112)
            emit('CONSTRUCTION_CONTEXT',device='Synthetic controller only',identity=device)
            owners=[];sources=[]
            for role in c.ROLES:
                source=c.variant(case,role);sources.append(source);part=c.partition(role,maximum)
                variant=1 if role=='double' else 2 if role=='hot' else 3 if role in ('rows','width') else 0
                emit('UPLOAD_ATTEMPT',role=role,variant=variant,rows=source['rows'],columns=source['columns'],maximum_rows=part,working_budget=1024**2)
                for start in range(0,source['rows'],part):emit('FILL',role=role,first_row=start,rows=min(part,source['rows']-start))
                emit('UPLOAD_OUTCOME',role=role,completed=True,error=None);owners.append(owner(source,part));emit('WEIGHTS',role=role,observation=owners[-1])
            emit('CONSTRUCTION_CONTEXT_DROPPED',owners=6,devices=[device]*6)
            emit('LOOKUP_CONTEXT',device='Synthetic controller only',identity=device,same_devices=[True]*6)
            state=workspace(n,k);required=state['owned_bytes'];retired=None
            emit('BUDGET_OUTCOME',required=required,limit=required-1,completed=False,error=f'Metal admission requires {required} bytes, budget {required-1}')
            emit('WORKSPACE_ATTEMPT',phase='initial',budget=required)
            emit('WORKSPACE_OUTCOME',phase='initial',completed=True,error=None)
            emit('COLD',**state)
            for index in range(22):
                if index==19:
                    emit('WORKSPACE_ATTEMPT',phase='fresh',budget=required)
                    emit('WORKSPACE_OUTCOME',phase='fresh',completed=True,error=None)
                    retired=state;state=workspace(n,k);emit('RECOVERY',retired=retired,fresh=state)
                attempt,error=c.request(case,index);source=sources[c.ROLES.index(attempt['role'])];part=c.partition(attempt['role'],maximum)
                emit('STEP_ATTEMPT',**attempt)
                result=[c.CALLER]*attempt['caller_length']
                if error is None or index==17:
                    count=attempt['tokens'];global_values=c.base.projection(tuple(source['weights']),k,tuple(attempt['input_bits']))
                    blocks=[list(range(first,min(first+part,n))) for first in range(0,n,part)]
                    if index==17:blocks=blocks[:2]
                    for block in blocks:
                        result_words=[global_values[token*n+row] for token in range(count) for row in block]
                        for role,active in [('input',attempt['input_bits']),('output',result_words),('status',[0])]:
                            state[role]['words']=list(active)+[c.S]*(state[role]['length']-len(active))
                        for i,value in enumerate(result_words):state['chunk_bits'][i]=value
                        for token in range(count):
                            for row in block:state['staged_bits'][token*n+row]=global_values[token*n+row]
                    state['poisoned']=error is not None
                    if error is None:result=list(global_values)
                emit('STEP_OUTCOME',index=index,completed=error is None,error=error,l1_calls=1 if index in (4,5,20) else 0)
                emit('STEP',index=index,caller_bits=result,workspace=state,retired=retired,weights=owners)
            for role,value in zip(c.ROLES,owners,strict=True):emit('FINAL_WEIGHTS',role=role,observation=value)
            emit('FINAL',key=key,workspace=state,retired=retired)
    emit('RESULT',cases=4,steps=88,construction_contexts_dropped=4,same_geometry_weight_swaps=True,foreign_device_qualified=False,
         driver_or_inflight_failure_injected=False,whole_graph_qualified=False,performance_qualified=False)
    return events

def wrap(events):return ('running 1 test\n'+'\n'.join('GEMMA_PROJECTION_'+label+' '+json.dumps(value) for label,value in events)+'\nok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n').encode()

class Portable(unittest.TestCase):
    def test_all_workspace_bytes_and_separate_aggregate_policy(self):
        self.assertEqual(c.cold(19,128)['owned_bytes'],2284);self.assertEqual(c.cold(11,64)['owned_bytes'],1420)
        self.assertEqual(c.policy()['total_policy_bytes'],324009984)
    def test_metadata_doubling_preserves_original_zero_signs_and_power_of_two_reconstruction(self):
        for case in c.base.fixture(fixture_directory()):
            other=c.variant(case,'double')
            for before,after in zip(case['weights'],other['weights'],strict=True):
                if before&0x7fffffff:self.assertEqual(c.base.f32(after),2*c.base.f32(before))
                else:self.assertEqual(after,before)
            self.assertEqual(other['scales'][:2],b'\x00\x80');self.assertEqual(other['biases'][:2],b'\x00\x80')
    def test_sparse_hot_input_admits_only_l1_proof_and_dense_input_rejects(self):
        for k in (64,128):
            maximum=Fraction(2)**24;gamma=Fraction(2*k,2**53);limit=Fraction(1,1600000)
            self.assertGreater(gamma*k*maximum,limit);self.assertLess(gamma*maximum,limit)
    def test_strict_record_limits_json_and_complete_success_wrapper(self):
        for raw in [b'GEMMA_PROJECTION_PLAN {"x":1,"x":2}\n',b'GEMMA_PROJECTION_PLAN {"x":NaN}\n',b'running 1 test\n',b'x'*(32*1024**2+1)]:
            with self.assertRaises(ValueError):c.records(raw)
    def test_complete_synthetic_reader_control_is_not_gpu_evidence(self):
        cases=c.base.fixture(fixture_directory());events=controller_fixture(cases)
        result=c.Replay(c.records(wrap(events))).run(cases);self.assertEqual(result['successful_output_values'],1500)
        changed=copy.deepcopy(events);next(v for k,v in changed if k=='STEP')['caller_bits'][0]^=1
        with self.assertRaises(ValueError):c.Replay(changed).run(cases)

class Actual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if len(ARGS)!=2 or ARGS[0]=='portable':raise unittest.SkipTest('explicit actual capture required')
        cls.cases=c.base.fixture(Path(ARGS[1]));cls.events=c.records(c.mathref.read(Path(ARGS[0]),32*1024**2))
    @staticmethod
    def item(events,label,index=0):return [value for key,value in events if key==label][index]
    def reject(self,change):
        events=copy.deepcopy(self.events);change(events)
        with self.assertRaises(ValueError):c.Replay(events).run(self.cases)
    def test_complete_actual_capture(self):self.assertEqual(c.Replay(self.events).run(self.cases)['steps'],88)
    def test_owner_struct_bytes(self):self.reject(lambda e:self.item(e,'CASE').update(projection_struct_bytes=1))
    def test_constructor_context_identity(self):self.reject(lambda e:self.item(e,'CONSTRUCTION_CONTEXT').update(identity='0x1234'))
    def test_missing_context_drop(self):self.reject(lambda e:e.pop(next(i for i,(k,v) in enumerate(e) if k=='CONSTRUCTION_CONTEXT_DROPPED')))
    def test_drop_device_identity(self):self.reject(lambda e:self.item(e,'CONSTRUCTION_CONTEXT_DROPPED')['devices'].__setitem__(0,'0x1234'))
    def test_reuse_context_device(self):self.reject(lambda e:self.item(e,'LOOKUP_CONTEXT').update(identity='0x1234'))
    def test_reuse_device_mismatch(self):self.reject(lambda e:self.item(e,'LOOKUP_CONTEXT')['same_devices'].__setitem__(0,False))
    def test_source_partition(self):self.reject(lambda e:self.item(e,'FILL').update(rows=1))
    def test_cached_codebook_maximum(self):self.reject(lambda e:self.item(e,'WEIGHTS')['observation'].update(maximum_bits=0))
    def test_weight_owner_slots(self):self.reject(lambda e:self.item(e,'WEIGHTS')['observation'].update(owner_slots=1))
    def test_original_packed_bytes(self):self.reject(lambda e:self.item(e,'WEIGHTS')['observation']['chunks'][0]['codes']['words'].__setitem__(0,0))
    def test_metadata_guard(self):self.reject(lambda e:self.item(e,'WEIGHTS')['observation']['chunks'][-1]['scales']['words'].__setitem__(-1,0))
    def test_metadata_padding(self):
        def change(events):
            owner=next(value['observation'] for label,value in events if label=='WEIGHTS' and value['role']=='original' and value['observation']['columns']==64)
            child=next(value for value in owner['chunks'] if value['rows']%2)
            values=child['scales']['words'];values[-8]^=0x10000
        self.reject(change)
    def test_weight_buffer_live_alias(self):
        def change(e):
            body=self.item(e,'WEIGHTS')['observation']['chunks'][0];body['biases']['allocation']=body['scales']['allocation']
        self.reject(change)
    def test_budget_adjacent_boundary(self):self.reject(lambda e:self.item(e,'BUDGET_OUTCOME').update(limit=1))
    def test_actual_workspace_capacity(self):self.reject(lambda e:self.item(e,'COLD').update(staged_capacity=1))
    def test_output_buffer_guard(self):self.reject(lambda e:self.item(e,'STEP')['workspace']['output']['words'].__setitem__(-1,0))
    def test_input_inactive_tail(self):self.reject(lambda e:self.item(e,'STEP',2)['workspace']['input']['words'].__setitem__(-8,0))
    def test_status_word(self):self.reject(lambda e:self.item(e,'STEP')['workspace']['status']['words'].__setitem__(0,1))
    def test_status_guard(self):self.reject(lambda e:self.item(e,'STEP')['workspace']['status']['words'].__setitem__(-1,0))
    def test_cpu_chunk_stale_tail(self):self.reject(lambda e:self.item(e,'STEP',2)['workspace']['chunk_bits'].__setitem__(-1,42))
    def test_cpu_staged_inactive_tail(self):self.reject(lambda e:self.item(e,'STEP',2)['workspace']['staged_bits'].__setitem__(-1,42))
    def test_original_input_schedule(self):self.reject(lambda e:self.item(e,'STEP_ATTEMPT')['input_bits'].__setitem__(0,1))
    def test_caller_result(self):self.reject(lambda e:self.item(e,'STEP')['caller_bits'].__setitem__(0,1))
    def test_l1_only_branch_count(self):self.reject(lambda e:self.item(e,'STEP_OUTCOME',4).update(l1_calls=0))
    def test_bound_rejection_original_error(self):self.reject(lambda e:self.item(e,'STEP_OUTCOME',5).update(error='different error'))
    def test_width_rejection_original_error(self):self.reject(lambda e:self.item(e,'STEP_OUTCOME',8).update(error='different error'))
    def test_oversized_chunk_rejection_preserves_scratch(self):self.reject(lambda e:self.item(e,'STEP',9)['workspace']['input']['words'].__setitem__(0,1))
    def test_capacity_rejection_preserves_caller(self):self.reject(lambda e:self.item(e,'STEP',10)['caller_bits'].__setitem__(0,0))
    def test_composition_failure_poisons_parent(self):self.reject(lambda e:self.item(e,'STEP',17)['workspace'].update(poisoned=False))
    def test_composition_failure_preserves_caller(self):self.reject(lambda e:self.item(e,'STEP',17)['caller_bits'].__setitem__(0,0))
    def test_poison_retry_state(self):self.reject(lambda e:self.item(e,'STEP',18)['workspace']['staged_bits'].__setitem__(0,42))
    def test_recovery_live_alias(self):
        def change(e):
            value=self.item(e,'RECOVERY');value['fresh']['input']['allocation']=value['retired']['input']['allocation']
        self.reject(change)
    def test_retired_workspace_immutable(self):self.reject(lambda e:self.item(e,'STEP',19)['retired']['chunk_bits'].__setitem__(0,42))
    def test_final_weight_mutation(self):self.reject(lambda e:self.item(e,'FINAL_WEIGHTS')['observation']['chunks'][0]['biases']['words'].__setitem__(0,0))
    def test_coherent_all_numerical_views(self):
        def change(e):
            value=self.item(e,'STEP');value['caller_bits'][0]=1;value['workspace']['staged_bits'][0]=1;value['workspace']['output']['words'][0]=1
        self.reject(change)
    def test_missing_outcome(self):self.reject(lambda e:e.pop(next(i for i,(k,v) in enumerate(e) if k=='STEP_OUTCOME')))
    def test_result_cannot_grant_whole_graph(self):self.reject(lambda e:self.item(e,'RESULT').update(whole_graph_qualified=True))
    def test_result_cannot_grant_foreign_device(self):self.reject(lambda e:self.item(e,'RESULT').update(foreign_device_qualified=True))

if __name__=='__main__':
    suite=unittest.TestLoader().loadTestsFromTestCase(Portable if len(ARGS)==2 and ARGS[0]=='portable' else Actual)
    result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful() else 1)
