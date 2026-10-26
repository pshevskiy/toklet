"""Portable ownership ledgers and adversarial complete real-table replay."""
from pathlib import Path
import copy,json,sys,unittest
import check_gemma_immutable_real_tables as c

class Portable(unittest.TestCase):
    def test_requested_weight_storage_excludes_all_per_matrix_projection_scratch(self):
        self.assertEqual(c.geometry(1536),dict(resident=226495664,gather=17367416,source=14155848,working=240651512))
        self.assertEqual(c.geometry(8960),dict(resident=1321209008,gather=20218232,source=82575432,working=1403784440))
        old=sum(c.original.geometry(k)['resident'] for _,_,k in c.TABLES)
        new=sum(c.geometry(k)['resident'] for _,_,k in c.TABLES)
        self.assertEqual(old-new,5004032)
    def test_new_policy_counts_both_contexts_and_does_not_adopt_historical_owner_policy(self):
        p=c.policy();self.assertEqual(p['total_policy_bytes'],1829144492)
        self.assertEqual(p['categories']['two_context_policies'],134217728)
        self.assertEqual(p['categories']['resident_owners'],1547704672)
        self.assertNotIn('context_policy',p['categories'])
        self.assertEqual(c.original.policy()['categories']['resident_owners'],1552708704)
    def test_complete_transformed_prior_capture_is_reader_control_only(self):
        if len(ARGS)!=4 or ARGS[0]!='portable':self.skipTest('explicit prior captured reader-control inputs required')
        log,binary,folder=map(Path,ARGS[1:])
        raw=c.mathref.read(log,8*1024**2);body=c.mathref.read(binary,64*1024**2)
        self.assertEqual(c.sha(raw),'156006b62863d09ec5e8964e990b28267ad7350977e152b0414613c54533160e')
        self.assertEqual(c.sha(body),'0e0f2f27ccd5089971a9a1d4820d2e908112eab025fc09c0c59a16d31c6aadc6')
        for name,wanted in [('oracle.json','3f4b8bb0d14c8f914301161c0dc76ab474e707da4978478d4b942a4012a14978'),('rows.bin','462e1375dfc3f761cc2a40c57fb831daa5f0de968f83a0a5b6be25dea348580e'),('manifest.json','4c0b76433b671140db1dbb8fcb5001719bd13eca0faf1d4d2606f525aeee915b')]:
            self.assertEqual(c.sha(c.mathref.read(folder/name,2*1024**2)),wanted)
        oracle=c.base.load(folder/'oracle.json',1024**2);rows=c.mathref.read(folder/'rows.bin',2*1024**2)
        c.original.compare_oracle(folder,oracle,rows)
        events=[];cold=0;device_identity='0x1000';device=None
        for label,item in c.original.parse(raw):
            item=copy.deepcopy(item)
            if label=='PLAN':item=c.policy()
            if label=='META':
                device=item['device'];item.pop('child_struct_bytes');item['chunk_struct_bytes']=112;item['device_identity']=device_identity
            if label=='UPLOAD_ATTEMPT':
                item.pop('capacity');item['working_budget']=c.geometry(item['columns'])['working']
            if label in ('MATRIX','MATRIX_FINAL'):
                old=item['observation'];k=old['columns']
                item['observation']=dict(rows=old['rows'],columns=k,owned_bytes=c.geometry(k)['resident'],owner_slots=16,
                                         chunk_struct_bytes=112,device=device_identity,
                                         children=[dict(rows=ch['rows'],**{name:ch['device'][name] for name in ('codes','scales','biases')}) for ch in old['children']])
                label='IMMUTABLE' if label=='MATRIX' else 'IMMUTABLE_FINAL'
            if label=='RESULT':item.update(construction_context_dropped=True,foreign_device_qualified=False)
            events.append((label,item))
            if label=='IMMUTABLE':
                cold+=1
                if cold==2:
                    events.append(('CONSTRUCTION_CONTEXT_DROPPED',dict(weights=2,devices=[device_identity,device_identity])))
                    events.append(('LOOKUP_CONTEXT',dict(device=device,device_identity=device_identity,same_devices=[True,True])))
        transcript='running 1 test\n'+'\n'.join('GEMMA_TABLES_'+label+' '+json.dumps(item) for label,item in events)+'\nok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n'
        parsed=c.parse(transcript.encode())
        # Fabricated ownership/context records exercise only new reader flow.
        # Actual qualification must use the separately executed native command.
        result=c.verify(parsed,body,oracle,rows)
        self.assertEqual(result['compared_values'],1102080);self.assertEqual(result['live_device_allocations'],102)
        changed=bytearray(body);changed[0]^=1
        with self.assertRaises(ValueError):c.verify(parsed,changed,oracle,rows)
    def test_strict_bounded_records_and_success_wrapper(self):
        for raw in [b'GEMMA_TABLES_PLAN {"a":1,"a":2}\n',b'GEMMA_TABLES_PLAN {"a":NaN}\n',b'running 2 tests\n',b'x'*(8*1024**2+1)]:
            with self.assertRaises(ValueError):c.parse(raw)

class Actual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if len(ARGS)!=5:raise unittest.SkipTest('complete actual inputs supplied explicitly')
        log,binary,model,frozen,oracle=map(Path,ARGS)
        cls.oracle,cls.rows=c.original.build_oracle(model,frozen);c.original.compare_oracle(oracle,cls.oracle,cls.rows)
        cls.events=c.parse(c.mathref.read(log,8*1024**2));cls.binary=c.mathref.read(binary,64*1024**2)
    @staticmethod
    def item(events,label):return next(v for key,v in events if key==label)
    def rejected(self,change):
        events=copy.deepcopy(self.events);binary=bytearray(self.binary);change(events,binary)
        with self.assertRaises((ValueError,KeyError,IndexError,TypeError,StopIteration)):
            c.verify(events,binary,self.oracle,self.rows)
    def test_complete_real_capture(self):
        r=c.verify(self.events,self.binary,self.oracle,self.rows)
        self.assertEqual(r['compared_values'],1102080);self.assertEqual(r['live_device_allocations'],102)
        self.assertTrue(r['construction_context_dropped']);self.assertEqual(r['arrays'],84)
    def test_device_identity(self):self.rejected(lambda e,b:self.item(e,'IMMUTABLE')['observation'].update(device='0x9876'))
    def test_missing_context_drop(self):self.rejected(lambda e,b:e.pop(next(i for i,(k,v) in enumerate(e) if k=='CONSTRUCTION_CONTEXT_DROPPED')))
    def test_context_drop_devices(self):self.rejected(lambda e,b:self.item(e,'CONSTRUCTION_CONTEXT_DROPPED')['devices'].__setitem__(0,'0x1234'))
    def test_lookup_context_identity(self):self.rejected(lambda e,b:self.item(e,'LOOKUP_CONTEXT').update(device_identity='0x9876'))
    def test_lookup_context_mismatch(self):self.rejected(lambda e,b:self.item(e,'LOOKUP_CONTEXT').update(same_devices=[False,True]))
    def test_owner_slots(self):self.rejected(lambda e,b:self.item(e,'IMMUTABLE')['observation'].update(owner_slots=15))
    def test_owner_bytes(self):self.rejected(lambda e,b:self.item(e,'IMMUTABLE')['observation'].update(owned_bytes=1))
    def test_source_offset(self):self.rejected(lambda e,b:self.item(e,'FILL')['consumed'][0].update(offset=1))
    def test_source_digest(self):self.rejected(lambda e,b:self.item(e,'FILL')['consumed'][1].update(sha256='0'*64))
    def test_source_address_instability(self):
        def change(e,b):[v for k,v in e if k=='FILL_ATTEMPT'][1]['addresses'][0]='0x123456'
        self.rejected(change)
    def test_coherent_source_alias(self):
        def change(e,b):
            for k,v in e:
                if k=='FILL_ATTEMPT':v['addresses']=['0x1000']*3
        self.rejected(change)
    def test_coherent_source_interval_wrap(self):
        def change(e,b):
            for k,v in e:
                if k=='FILL_ATTEMPT':v['addresses'][0]='0xfffffffffffffff0'
        self.rejected(change)
    def test_guarded_weight_digest(self):self.rejected(lambda e,b:self.item(e,'IMMUTABLE')['observation']['children'][0]['codes'].update(sha256='f'*64))
    def test_weight_guard(self):self.rejected(lambda e,b:self.item(e,'IMMUTABLE')['observation']['children'][0]['scales']['guards'].__setitem__(0,0))
    def test_live_weight_alias(self):
        def change(e,b):
            ch=self.item(e,'IMMUTABLE')['observation']['children'][0];ch['biases']['allocation']=ch['codes']['allocation']
        self.rejected(change)
    def test_final_weight_replacement(self):self.rejected(lambda e,b:self.item(e,'IMMUTABLE_FINAL')['observation']['children'][0]['codes'].update(allocation='0x1234'))
    def test_aggregate_underaccount(self):self.rejected(lambda e,b:self.item(e,'PLAN')['categories'].update(two_context_policies=0))
    def test_loading_budget(self):self.rejected(lambda e,b:self.item(e,'UPLOAD_ATTEMPT').update(working_budget=1))
    def test_workspace_live_alias(self):
        def change(e,b):
            w=self.item(e,'WORKSPACES');w['ple_lookup']['device'][0]['allocation']=w['embedding']['device'][0]['allocation']
        self.rejected(change)
    def test_original_outcome_failure(self):self.rejected(lambda e,b:self.item(e,'STEP_OUTCOME').update(completed=False,error='original failure'))
    def test_completed_shared_access(self):self.rejected(lambda e,b:self.item(e,'SNAPSHOT').update(shared_read_safe=False,pending=True))
    def test_request_order(self):
        def change(e,b):next(v for k,v in e if k=='STEP_ATTEMPT' and v['index']==1)['ids'].reverse()
        self.rejected(change)
    def changed_array(self,role):
        def change(e,b):
            a=next(v for k,v in e if k=='ARRAY' and v['key']=='ple_lookup/2' and v['role']==role)
            start=a['offset'];b[start]^=1;digest=c.sha(b[start:start+a['bytes']])
            for key,v in e:
                if key in ('ARRAY','ARRAY_ATTEMPT') and v['offset']==start:v['sha256']=digest
                if key=='STEP':
                    for item in v['observation']['arrays']+[d['array'] for d in v['observation']['device']]:
                        if item['offset']==start:item['sha256']=digest
                if key=='SNAPSHOT' and v['key']==a['key']:
                    for item in v['host']:
                        if item['role']==role:item['sha256']=digest
                    for item in v['device']:
                        if item['role']==role:item['identity']['sha256']=digest
        self.rejected(change)
    def test_caller_value(self):self.changed_array('caller')
    def test_staged_value(self):self.changed_array('staged')
    def test_chunk_value(self):self.changed_array('chunk')
    def test_last_child_output(self):self.changed_array('output')
    def test_local_ids(self):self.changed_array('ids')
    def test_status(self):self.changed_array('status')
    def test_coherent_all_target_numerical_views(self):
        def change(e,b):
            digests={};roles={}
            for key,a in e:
                if key=='ARRAY' and a['role'] in ('caller','staged','chunk','output') and not a['key'].endswith('/cold'):
                    start=a['offset'];size=a['bytes'];b[start:start+size]=bytes(size);digests[start]=c.sha(b[start:start+size]);roles[(a['key'],a['role'])]=digests[start]
            for key,v in e:
                if key in ('ARRAY','ARRAY_ATTEMPT') and v['offset'] in digests:v['sha256']=digests[v['offset']]
                if key=='STEP':
                    for a in v['observation']['arrays']+[d['array'] for d in v['observation']['device']]:
                        if a['offset'] in digests:a['sha256']=digests[a['offset']]
                if key=='SNAPSHOT':
                    for a in v['host']:
                        if (v['key'],a['role']) in roles:a['sha256']=roles[(v['key'],a['role'])]
                    for a in v['device']:
                        if (v['key'],a['role']) in roles:a['identity']['sha256']=roles[(v['key'],a['role'])]
        self.rejected(change)
    def test_missing_array_completion(self):self.rejected(lambda e,b:e.pop(next(i for i,(k,v) in enumerate(e) if k=='ARRAY')))
    def test_truncated_raw(self):self.rejected(lambda e,b:b.pop())
    def test_whole_graph_grant(self):self.rejected(lambda e,b:self.item(e,'RESULT').update(whole_graph_qualified=True))
    def test_foreign_device_grant(self):self.rejected(lambda e,b:self.item(e,'RESULT').update(foreign_device_qualified=True))

ARGS=sys.argv[1:]
if __name__=='__main__':unittest.main(argv=[sys.argv[0],'Actual' if len(ARGS)==5 else 'Portable'])
