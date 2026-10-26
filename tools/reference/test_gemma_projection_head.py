"""Complete no-GPU reader controls and explicit real projection capture mutations."""
from pathlib import Path
import copy,json,struct,sys,unittest
import check_gemma_projection_head as c
ARGS=sys.argv[1:]

def controller_fixture(assets):
    """Synthetic reader-flow control, never a native observation.

    Original source/expected bytes precede the target implementation. This
    generator updates final whole-row state directly; the reader traverses every
    child. Device identities are fabricated and confer no hardware qualification.
    """
    events=[];binary=bytearray();address=0x10000;device='0x1000'
    def emit(label,**body):events.append((label,copy.deepcopy(body)))
    def pointer():
        nonlocal address
        address+=0x1000;return hex(address)
    def array(key,role,dtype,raw):
        item=dict(key=key,role=role,dtype=dtype,words=len(raw)//4,bytes=len(raw),offset=len(binary),sha256=c.sha(raw))
        emit('ARRAY_ATTEMPT',**item);binary.extend(raw);emit('ARRAY',**item);return item
    def capture(key,state):return [array(key,r,'U32' if r in ('input','output','status') else 'F32',state[r]) for r in ('input','output','status','staged','chunk')]
    def lookup(index):
        emit('LOOKUP_ATTEMPT',index=index,ids=[2,100,257,100]);emit('LOOKUP_OUTCOME',index=index,completed=True,error=None,caller_sha256=c.sha(assets.lookup))
        item=array(f'lookup/{index}','caller','F32',assets.lookup);emit('LOOKUP',index=index,array=item,private_lookup_scratch_captured=False)
    emit('PLAN',**c.policy());emit('CONSTRUCTION_CONTEXT',device='Synthetic controller only',identity=device);emit('SPANS',entries=assets.spans)
    emit('UPLOAD_ATTEMPT',working_budget=240651512,maximum_rows=c.TILE)
    for i in range(16):
        emit('FILL_ATTEMPT',first_row=i*c.TILE,rows=c.TILE,addresses=['0x10000000','0x20000000','0x30000000'],lengths=[12582912,786432,786432])
        emit('FILL',first_row=i*c.TILE,consumed=assets.fills[i])
    emit('UPLOAD_OUTCOME',completed=True,error=None);chunks=[]
    for old in assets.original.buffers:
        chunk=dict(rows=c.TILE)
        for role,value in old.items():chunk[role]=dict(allocation=pointer(),words=value['words'],bytes=value['bytes'],contents=dict(sha256=value['sha256'],guards=value['guards']))
        chunks.append(chunk)
    weights=dict(rows=c.N,columns=c.K,owned_bytes=c.WEIGHTS,owner_slots=16,chunk_struct_bytes=112,device=device,chunks=chunks)
    emit('WEIGHTS',**weights);emit('CONSTRUCTION_CONTEXT_DROPPED',owners=1,device=device)
    emit('EXECUTION_CONTEXT',device='Synthetic controller only',identity=device,same_device=True)
    emit('WORKSPACE_ATTEMPT',projection_budget=c.WORKSPACE,lookup_budget=c.LOOKUP)
    emit('PROJECTION_OUTCOME',completed=True,error=None);emit('GATHER_OUTCOME',completed=True,error=None)
    state=c.initial();ids={role:pointer() for role in ('input','output','status')}
    emit('COLD_OBSERVATION',**c.workspace(state,ids));items=capture('cold',state);emit('COLD',arrays=items);lookup(0)
    for phase,index,tokens in [('repeat',0,4),('repeat',1,4),('part',0,1),('part',1,1),('part',2,1),('part',3,1),('resume',0,4)]:
        key=f'{phase}/{index}';inputs=assets.input if tokens==4 else assets.input[index*c.K*4:(index+1)*c.K*4]
        result=assets.expected if tokens==4 else assets.expected[index*c.N*4:(index+1)*c.N*4]
        # Direct final-row assignment differs from Replay's per-child scatter.
        if tokens==4:state['staged'][:]=result
        else:state['staged'][:c.N*4]=result
        final=b''.join(result[((token+1)*c.N-c.TILE)*4:(token+1)*c.N*4] for token in range(tokens))
        state['input']=inputs+c.S*(6151-len(inputs)//4);state['output']=final+c.S*(65543-len(final)//4)
        state['status']=bytes(4)+c.G;state['chunk'][:len(final)]=final
        emit('STEP_ATTEMPT',key=key,tokens=tokens,input_sha256=c.sha(inputs))
        emit('STEP_OUTCOME',key=key,completed=True,error=None,caller_sha256=c.sha(result),workspace=c.workspace(state,ids),weights=weights)
        items=capture(key,state);returned=array(key,'caller','F32',result)
        emit('STEP',key=key,arrays=items,caller=returned,overwritten_child_device_outputs_separately_captured=False,captured_scratch_is_failed_request_result=False)
    lookup(1);emit('FINAL_WEIGHTS',**weights)
    emit('RESULT',steps=7,lookups=2,arrays=49,binary_bytes=len(binary),chunks=16,construction_context_dropped=True,weights_shared_with_lookup=True,
         overwritten_child_device_outputs_separately_captured=False,private_lookup_scratch_captured=False,source_cpu_table_retained=False,
         foreign_device_qualified=False,whole_graph_qualified=False,original_bf16_source_qualified=False,performance_qualified=False)
    return events,bytes(binary)

def wrap(events):return ('running 1 test\n'+'\n'.join('GEMMA_PROJECTION_HEAD_'+label+' '+json.dumps(value) for label,value in events)+'\nok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n').encode()

class Portable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if len(ARGS)!=4 or ARGS[0]!='portable':raise unittest.SkipTest('explicit original local source inputs required; no GPU')
        cls.assets=c.Assets(*map(Path,ARGS[1:]))
    @classmethod
    def tearDownClass(cls):cls.assets.close()
    def test_accounting_keeps_weights_context_and_both_workspaces_separate(self):
        self.assertEqual(c.policy()['total_policy_bytes'],439229020);self.assertEqual(c.WORKSPACE,4743496);self.assertEqual(c.LOOKUP,16851208)
        self.assertEqual(c.WEIGHTS,226495664)
    def test_complete_raw_inventory_and_returned_values(self):
        self.assertEqual(4743256+3*(4743256+4194304)+4*(4743256+1048576)+2*24576,c.RAW_BYTES)
        self.assertEqual((3*4+4)*c.N,4194304)
    def test_strict_bounded_json_records_and_success_wrapper(self):
        for raw in [b'GEMMA_PROJECTION_HEAD_PLAN {"x":1,"x":2}\n',b'GEMMA_PROJECTION_HEAD_PLAN {"x":NaN}\n',b'running 1 test\n',b'x'*(8*1024**2+1)]:
            with self.assertRaises(ValueError):c.parse(raw)
    def test_complete_synthetic_reader_control_is_not_native_evidence(self):
        events,raw=controller_fixture(self.assets);result=c.verify(c.parse(wrap(events)),raw,self.assets)
        self.assertEqual(result['observed_gpu_allocations'],51);self.assertEqual(result['returned_projection_values'],4194304)
        changed=bytearray(raw);changed[0]^=1
        with self.assertRaises(ValueError):c.verify(events,changed,self.assets)

class Actual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if len(ARGS)!=5:raise unittest.SkipTest('explicit actual capture, original model and frozen references required')
        log,binary,model,head,lookup=map(Path,ARGS);cls.assets=c.Assets(model,head,lookup)
        try:
            with log.open('rb') as stream:cls.events=c.parse(stream.read(8*1024**2+1))
            with binary.open('rb') as stream:cls.raw=stream.read(64*1024**2+1)
        except BaseException:cls.assets.close();raise
    @classmethod
    def tearDownClass(cls):cls.assets.close()
    @staticmethod
    def item(events,label,index=0):return [value for key,value in events if key==label][index]
    def reject(self,change):
        events=copy.deepcopy(self.events);raw=bytearray(self.raw);change(events,raw)
        with self.assertRaises(ValueError):c.verify(events,raw,self.assets)
    def test_complete_actual_capture(self):self.assertTrue(c.verify(self.events,self.raw,self.assets)['passed'])
    def test_context_device(self):self.reject(lambda e,b:self.item(e,'CONSTRUCTION_CONTEXT').update(identity='0x1234'))
    def test_missing_context_drop(self):self.reject(lambda e,b:e.pop(next(i for i,(k,v) in enumerate(e) if k=='CONSTRUCTION_CONTEXT_DROPPED')))
    def test_drop_identity(self):self.reject(lambda e,b:self.item(e,'CONSTRUCTION_CONTEXT_DROPPED').update(device='0x1234'))
    def test_new_context_mismatch(self):self.reject(lambda e,b:self.item(e,'EXECUTION_CONTEXT').update(same_device=False))
    def test_span_offset(self):self.reject(lambda e,b:self.item(e,'SPANS')['entries'][0].update(offset=1))
    def test_source_chunk_hash(self):self.reject(lambda e,b:self.item(e,'FILL')['consumed'][0].update(sha256='0'*64))
    def test_source_pointer_reuse(self):self.reject(lambda e,b:self.item(e,'FILL_ATTEMPT',1)['addresses'].__setitem__(0,'0x1234'))
    def test_source_alias(self):
        def change(e,b):
            for label,value in e:
                if label=='FILL_ATTEMPT':value['addresses'][1]=value['addresses'][0]
        self.reject(change)
    def test_source_interval_overflow(self):
        def change(e,b):
            for label,value in e:
                if label=='FILL_ATTEMPT':value['addresses'][0]='0xfffffffffffffff0'
        self.reject(change)
    def test_owner_size(self):self.reject(lambda e,b:self.item(e,'WEIGHTS').update(owned_bytes=1))
    def test_owner_slots(self):self.reject(lambda e,b:self.item(e,'WEIGHTS').update(owner_slots=1))
    def test_owner_payload_hash(self):self.reject(lambda e,b:self.item(e,'WEIGHTS')['chunks'][0]['codes']['contents'].update(sha256='0'*64))
    def test_owner_guard(self):self.reject(lambda e,b:self.item(e,'WEIGHTS')['chunks'][0]['scales']['contents']['guards'].__setitem__(0,0))
    def test_owner_live_alias(self):
        def change(e,b):
            value=self.item(e,'WEIGHTS')['chunks'][0];value['biases']['allocation']=value['codes']['allocation']
        self.reject(change)
    def test_session_live_alias(self):
        def change(e,b):
            value=self.item(e,'COLD_OBSERVATION');value['output']['allocation']=value['input']['allocation']
        self.reject(change)
    def test_source_session_alias(self):self.reject(lambda e,b:self.item(e,'COLD_OBSERVATION')['input'].update(allocation=self.item(e,'WEIGHTS')['chunks'][0]['codes']['allocation']))
    def test_session_actual_capacity(self):self.reject(lambda e,b:self.item(e,'COLD_OBSERVATION').update(staged_capacity=1))
    def test_session_state_poison(self):self.reject(lambda e,b:self.item(e,'STEP_OUTCOME')['workspace'].update(poisoned=True))
    def test_session_state_pending(self):self.reject(lambda e,b:self.item(e,'STEP_OUTCOME')['workspace'].update(pending=True))
    def test_unsafe_shared_read(self):self.reject(lambda e,b:self.item(e,'STEP_OUTCOME')['workspace'].update(shared_read_safe=False))
    def test_original_outcome(self):self.reject(lambda e,b:self.item(e,'STEP_OUTCOME').update(error='different error',completed=False))
    def test_caller_raw_bytes(self):
        def change(e,b):value=next(v for k,v in e if k=='ARRAY' and v['role']=='caller' and v['key']=='repeat/0');b[value['offset']]^=1
        self.reject(change)
    def test_coherent_final_gpu_cpu_and_caller_values(self):
        def change(e,b):
            body=self.item(e,'STEP_OUTCOME');hashes={};target=c.N-c.TILE+123
            for label,value in e:
                if label=='ARRAY' and value['key']=='repeat/0' and value['role'] in ('output','staged','chunk','caller'):
                    word=target if value['role'] in ('staged','caller') else 123
                    at=value['offset']+word*4;b[at]^=1;raw=memoryview(b)[value['offset']:value['offset']+value['bytes']];hashes[value['offset']]=c.sha(raw)
            for label,value in e:
                if label in ('ARRAY','ARRAY_ATTEMPT') and value['offset'] in hashes:value['sha256']=hashes[value['offset']]
                if label=='STEP' and value['key']=='repeat/0':
                    for item in [*value['arrays'],value['caller']]:
                        if item['offset'] in hashes:item['sha256']=hashes[item['offset']]
            step=self.item(e,'STEP');body['caller_sha256']=step['caller']['sha256']
            for item in step['arrays']:
                if item['role']=='output':body['workspace']['output']['contents']['sha256']=item['sha256']
                if item['role']=='staged':body['workspace']['staged_sha256']=item['sha256']
                if item['role']=='chunk':body['workspace']['chunk_sha256']=item['sha256']
        self.reject(change)
    def test_inactive_output_or_guard_bytes(self):
        def change(e,b):value=next(v for k,v in e if k=='ARRAY' and v['role']=='output' and v['key']=='part/0');b[value['offset']+value['bytes']-1]^=1
        self.reject(change)
    def test_stale_cpu_chunk_tail(self):
        def change(e,b):value=next(v for k,v in e if k=='ARRAY' and v['role']=='chunk' and v['key']=='part/0');b[value['offset']+value['bytes']-1]^=1
        self.reject(change)
    def test_lookup_value(self):self.reject(lambda e,b:self.item(e,'LOOKUP_OUTCOME').update(caller_sha256='0'*64))
    def test_lookup_source_ids(self):self.reject(lambda e,b:self.item(e,'LOOKUP_ATTEMPT')['ids'].__setitem__(0,0))
    def test_array_dtype(self):self.reject(lambda e,b:self.item(e,'ARRAY_ATTEMPT').update(dtype='F16'))
    def test_array_offset(self):self.reject(lambda e,b:self.item(e,'ARRAY_ATTEMPT').update(offset=4))
    def test_raw_truncation(self):self.reject(lambda e,b:b.pop())
    def test_raw_extra_byte(self):self.reject(lambda e,b:b.append(0))
    def test_final_weights(self):self.reject(lambda e,b:self.item(e,'FINAL_WEIGHTS')['chunks'][0]['biases']['contents'].update(sha256='0'*64))
    def test_cannot_claim_intermediate_capture(self):self.reject(lambda e,b:self.item(e,'RESULT').update(overwritten_child_device_outputs_separately_captured=True))
    def test_cannot_claim_private_lookup_capture(self):self.reject(lambda e,b:self.item(e,'RESULT').update(private_lookup_scratch_captured=True))
    def test_cannot_waive_original_source_failures(self):self.reject(lambda e,b:self.item(e,'RESULT').update(original_bf16_source_qualified=True))
    def test_cannot_grant_whole_graph(self):self.reject(lambda e,b:self.item(e,'RESULT').update(whole_graph_qualified=True))
    def test_cannot_grant_performance(self):self.reject(lambda e,b:self.item(e,'RESULT').update(performance_qualified=True))

if __name__=='__main__':
    suite=unittest.TestLoader().loadTestsFromTestCase(Portable if len(ARGS)==4 and ARGS[0]=='portable' else Actual)
    result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful() else 1)
