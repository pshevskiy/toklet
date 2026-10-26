"""Portable controls and adversarial replay over complete captured real outputs."""
from pathlib import Path
import contextlib,copy,io,json,struct,sys,tempfile,unittest
from unittest.mock import patch
import check_gemma_real_tables as c

class Portable(unittest.TestCase):
    def test_independent_geometry_and_aggregate_loading_admission(self):
        self.assertEqual(c.geometry(1536),dict(child=14227868,resident=228760112,gather=17367416,source=14155848,working=242915960))
        self.assertEqual(c.geometry(8960),dict(child=82677148,resident=1323948592,gather=20218232,source=82575432,working=1406524024))
        p=c.policy();self.assertEqual(p['total_policy_bytes'],sum(p['categories'].values()));self.assertLess(p['total_policy_bytes'],3*1024**3)
    def test_all_chunk_boundaries_order_duplicates_and_limits(self):
        self.assertEqual(len(c.SELECTED),35);self.assertEqual(len(c.BOUNDARY),32)
        self.assertEqual(c.BOUNDARY[:4],[0,16383,16384,32767]);self.assertEqual(c.BOUNDARY[-1],262143)
        self.assertEqual(c.schedules()[2],list(reversed(c.BOUNDARY)));self.assertEqual(c.schedules()[4],[16384]*32)
    def test_exact_bfloat16_oracle_zero_signs_and_rounding(self):
        self.assertEqual(c.mathref.rounded(c.mathref.rational(0x3e00,7,8)*15+c.mathref.rational(0xbe80,7,8),23,8),0x3fd00000)
        self.assertEqual(c.mathref.rounded(0,23,8,True),0x80000000)
    @contextlib.contextmanager
    def small_preparation(self,late=False):
        class Source(io.BytesIO):
            def read(self,n=-1):
                if late and self.tell()>=176:raise OSError('injected late original read')
                return super().read(min(n,32))
        class Model:
            def open(self,mode):return Source(bytes(288))
        def spans(header,stem,k):
            start=0 if stem=='embed_tokens' else 144
            return [dict(role=r,name=stem+'.'+r,dtype=d,shape=[4,8 if r=='codes' else 1],offset=start+o,bytes=n,stride=s)
                    for r,d,o,n,s in [('codes','U32',0,128,32),('scales','BF16',128,8,2),('biases','BF16',136,8,2)]]
        with contextlib.ExitStack() as stack:
            for name,value in [('N',4),('TILE',2),('TABLES',[('embedding','embed_tokens',64),('ple_lookup','embed_tokens_per_layer',64)]),
                               ('ORIGINAL',[0,1,2,1]),('BOUNDARY',[0,1,2,3]),('SELECTED',[0,1,2,3])]:stack.enter_context(patch.object(c,name,value))
            stack.enter_context(patch.object(c,'authenticate_inputs',return_value=bytes(2048)))
            stack.enter_context(patch.object(c,'read_header',return_value={}))
            stack.enter_context(patch.object(c,'spans',side_effect=spans))
            yield Model()
    def test_prepared_oracle_round_trips_through_replay_entry_comparison(self):
        with tempfile.TemporaryDirectory() as temporary,self.small_preparation() as model:
            folder=Path(temporary)/'oracle'
            expected,raw=c.build_oracle(model,None)
            c.prepare(model,None,folder)
            c.compare_oracle(folder,expected,raw)
            with self.assertRaises(ValueError):c.compare_oracle(folder,dict(expected,binary_bytes=8),raw)
    def test_preparation_late_original_read_retains_partial_rows_progress_and_cause(self):
        with tempfile.TemporaryDirectory() as temporary,self.small_preparation(late=True) as model:
            folder=Path(temporary)/'oracle'
            with self.assertRaisesRegex(OSError,'injected late original read'):c.prepare(model,None,folder)
            report=json.loads((folder/'oracle.json').read_bytes());manifest=json.loads((folder/'manifest.json').read_bytes())
            self.assertFalse(report['passed']);self.assertEqual(report['attempt']['phase'],'source-chunk')
            self.assertEqual(report['attempt']['consumed_bytes'],32);self.assertEqual(len((folder/'rows.bin').read_bytes()),1024)
            self.assertFalse(manifest['passed']);self.assertIn('injected late original read',manifest['primary_error'])
    def test_preparation_publication_failure_retains_first_files_and_manifest(self):
        original=Path.open
        def fail(path,*args,**kwargs):
            if path.name=='oracle.json':raise OSError('injected oracle publication')
            return original(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temporary,self.small_preparation() as model:
            folder=Path(temporary)/'oracle'
            with patch.object(Path,'open',fail),self.assertRaisesRegex(OSError,'injected oracle publication'):c.prepare(model,None,folder)
            manifest=json.loads((folder/'manifest.json').read_bytes())
            self.assertFalse(manifest['passed']);self.assertEqual(manifest['files']['rows.bin']['bytes'],2048)
            self.assertIn('injected oracle publication',manifest['preparation_state']['primary_error'])
    def test_preparation_secondary_publication_failure_preserves_original_read(self):
        original=Path.open
        def fail(path,*args,**kwargs):
            if path.name=='oracle.json':raise OSError('secondary publication failure')
            return original(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temporary,self.small_preparation(late=True) as model:
            folder=Path(temporary)/'oracle'
            with patch.object(Path,'open',fail),self.assertRaisesRegex(OSError,'injected late original read') as caught:c.prepare(model,None,folder)
            self.assertIn('secondary publication failure',str(caught.exception.__notes__))
            manifest=json.loads((folder/'manifest.json').read_bytes())
            self.assertIn('injected late original read',manifest['primary_error']);self.assertFalse(manifest['passed'])
    def test_strict_parser_rejects_extra_wrapper_and_duplicate_fields(self):
        for raw in [b'GEMMA_TABLES_PLAN {"a":1,"a":2}\n',b'running 2 tests\n',b'GEMMA_TABLES_PLAN {"a":NaN}\n',b'x'*(8*1024**2+1)]:
            with self.assertRaises(ValueError):c.parse(raw)

class Actual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if len(ARGS)!=5:raise unittest.SkipTest('actual captured inputs supplied explicitly')
        log,binary,model,frozen,oracle=map(Path,ARGS)
        cls.oracle,cls.expected=c.build_oracle(model,frozen)
        c.same(c.base.load(oracle/'oracle.json',1024**2),cls.oracle)
        c.same(c.mathref.read(oracle/'rows.bin',2*1024**2),cls.expected)
        cls.events=c.parse(c.mathref.read(log,8*1024**2));cls.binary=c.mathref.read(binary,64*1024**2)
    def rejected(self,change):
        events=copy.deepcopy(self.events);body=bytearray(self.binary);change(events,body)
        with self.assertRaises((ValueError,KeyError,IndexError,TypeError,StopIteration)):
            c.verify(events,body,self.oracle,self.expected)
    @staticmethod
    def item(events,label):return next(v for key,v in events if key==label)
    def test_complete_real_capture(self):
        result=c.verify(self.events,self.binary,self.oracle,self.expected)
        self.assertEqual(result['steps'],12);self.assertEqual(result['live_device_allocations'],198)
    def test_source_consumed_bytes(self):self.rejected(lambda e,b:self.item(e,'FILL')['consumed'][0].update(sha256='0'*64))
    def test_source_row_offset(self):self.rejected(lambda e,b:self.item(e,'FILL')['consumed'][1].update(offset=1))
    def test_source_stride(self):self.rejected(lambda e,b:self.item(e,'UPLOAD_ATTEMPT')['spans'][0].update(stride=1))
    def test_source_pointer_instability(self):
        def change(e,b):[v for k,v in e if k=='FILL_ATTEMPT'][1]['addresses'][0]='0x12345678'
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
    def test_resident_byte_digest(self):
        self.rejected(lambda e,b:self.item(e,'MATRIX')['observation']['children'][15]['device']['codes'].update(sha256='f'*64))
    def test_resident_guard(self):
        self.rejected(lambda e,b:self.item(e,'MATRIX')['observation']['children'][15]['device']['scales']['guards'].__setitem__(0,0))
    def test_resident_allocation_alias(self):
        def change(e,b):
            d=self.item(e,'MATRIX')['observation']['children'][0]['device'];d['biases']['allocation']=d['codes']['allocation']
        self.rejected(change)
    def test_resident_final_replacement(self):self.rejected(lambda e,b:self.item(e,'MATRIX_FINAL')['observation'].update(owned_bytes=1))
    def test_aggregate_underaccount(self):self.rejected(lambda e,b:self.item(e,'PLAN')['categories'].update(source_staging=0))
    def test_construction_budget(self):self.rejected(lambda e,b:self.item(e,'UPLOAD_ATTEMPT').update(working_budget=1))
    def test_original_outcome_failure(self):self.rejected(lambda e,b:self.item(e,'STEP_OUTCOME').update(completed=False,error='original failure'))
    def test_uncompleted_shared_access(self):self.rejected(lambda e,b:self.item(e,'STEP')['observation'].update(shared_read_safe=False,pending=True))
    def test_boundary_request_order(self):
        def change(e,b):
            v=[v for k,v in e if k=='STEP_ATTEMPT' and v['index']==1][0];v['ids'].reverse()
        self.rejected(change)
    def test_workspace_live_alias(self):
        def change(e,b):
            s=self.item(e,'WORKSPACES');s['ple_lookup']['device'][0]['allocation']=s['embedding']['device'][0]['allocation']
        self.rejected(change)
    def test_workspace_replacement(self):self.rejected(lambda e,b:self.item(e,'STEP')['observation']['device'][0].update(allocation='0x12345'))
    def mutate_array(self,role):
        def change(e,b):
            wanted=next(v for k,v in e if k=='ARRAY' and v['key']=='ple_lookup/2' and v['role']==role)
            offset=wanted['offset'];b[offset]^=1
            digest=c.sha(b[offset:offset+wanted['bytes']])
            for key,v in e:
                if key in ('ARRAY','ARRAY_ATTEMPT') and v['offset']==offset:v['sha256']=digest
                if key=='STEP':
                    for a in v['observation']['arrays']+[d['array'] for d in v['observation']['device']]:
                        if a['offset']==offset:a['sha256']=digest
        self.rejected(change)
    def test_caller_value(self):self.mutate_array('caller')
    def test_staged_value(self):self.mutate_array('staged')
    def test_chunk_value(self):self.mutate_array('chunk')
    def test_output_value(self):self.mutate_array('output')
    def test_local_index(self):self.mutate_array('ids')
    def test_status(self):self.mutate_array('status')
    def test_coherent_all_target_numerical_views(self):
        def change(e,b):
            manifests=[v for k,v in e if k=='ARRAY' and v['role'] in ('caller','staged','chunk','output') and not v['key'].endswith('/cold')]
            digests={}
            for a in manifests:
                start=a['offset'];size=a['bytes'];b[start:start+size]=bytes(size);digests[start]=c.sha(b[start:start+size])
            for key,v in e:
                if key in ('ARRAY','ARRAY_ATTEMPT') and v['offset'] in digests:v['sha256']=digests[v['offset']]
                if key=='STEP':
                    for a in v['observation']['arrays']+[d['array'] for d in v['observation']['device']]:
                        if a['offset'] in digests:a['sha256']=digests[a['offset']]
        self.rejected(change)
    def test_guard_inactive_output_tail(self):
        def change(e,b):
            a=next(v for k,v in e if k=='ARRAY' and v['key']=='embedding/3' and v['role']=='output');b[a['offset']+a['bytes']-4]^=1
        self.rejected(change)
    def test_missing_original_outcome(self):self.rejected(lambda e,b:e.pop(next(i for i,(k,v) in enumerate(e) if k=='STEP_OUTCOME')))
    def test_truncated_binary(self):self.rejected(lambda e,b:b.pop())
    def test_unqualified_graph_grant(self):self.rejected(lambda e,b:self.item(e,'RESULT').update(whole_graph_qualified=True))

ARGS=sys.argv[1:]
if __name__=='__main__':unittest.main(argv=[sys.argv[0]])
