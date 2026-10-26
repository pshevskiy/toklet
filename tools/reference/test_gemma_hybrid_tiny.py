"""Portable verifier controls; synthetic packets never qualify a GPU graph."""
from pathlib import Path
import copy,hashlib,json,struct,sys,unittest
import check_gemma_hybrid_tiny as check

class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.binary=check.fixture(Path(sys.argv[-1]));cls.events=[];raw=bytearray()
        for number,(name,before,count) in enumerate(check.signatures(cls.data)):
            values=check.values(cls.data,cls.binary,'staged_fp32',name,before,count)
            block=b''.join(struct.pack('<f',v) for v in values)
            head=dict(record=number,name=name,offset=len(raw),bytes=len(block),before=before,count=count,values=len(values))
            cls.events.extend([('ARRAY_ATTEMPT',head),('ARRAY',dict(head,sha256=hashlib.sha256(block).hexdigest()))]);raw.extend(block)
        cls.raw=bytes(raw)
        cls.events.extend([('CAPTURE',dict(records=len(cls.events)//2,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),whole_e2b_qualified=False)),('TINY_COMPLETE',None)])
    def rejects(self,events=None,raw=None):
        with self.assertRaises((ValueError,KeyError,TypeError,IndexError)):
            check.replay(events if events is not None else self.events,raw if raw is not None else self.raw,self.data,self.binary)
    def test_complete_synthetic_packet(self):
        result=check.replay(self.events,self.raw,self.data,self.binary)
        self.assertEqual(result['records'],3805);self.assertEqual(result['binary_bytes'],3048704)
        self.assertFalse(result['whole_e2b_qualified'])
    def test_missing_and_extra_events(self):
        self.rejects(self.events[:-1]);self.rejects(self.events+[self.events[-1]])
    def test_record_order_and_false_boolean_integer(self):
        events=copy.deepcopy(self.events);events[0][1]['record']=False;self.rejects(events)
        events=copy.deepcopy(self.events);events[0],events[1]=events[1],events[0];self.rejects(events)
    def test_name_and_position(self):
        for key,value in [('name','logits'),('before',33),('count',0)]:
            events=copy.deepcopy(self.events);events[0][1][key]=value;self.rejects(events)
    def test_hash_and_partial_binary(self):
        events=copy.deepcopy(self.events);events[1][1]['sha256']='0'*64;self.rejects(events)
        self.rejects(raw=self.raw[:-1]);self.rejects(raw=self.raw+b'x')
    def test_coherent_changed_numerical_value_is_rejected(self):
        events=copy.deepcopy(self.events);raw=bytearray(self.raw);raw[:4]=struct.pack('<f',100.)
        spec=events[1][1];spec['sha256']=hashlib.sha256(raw[:spec['bytes']]).hexdigest()
        events[-2][1]['sha256']=hashlib.sha256(raw).hexdigest();self.rejects(events,bytes(raw))
    def test_nonfinite_value_is_rejected(self):
        events=copy.deepcopy(self.events);raw=bytearray(self.raw);raw[:4]=bytes.fromhex('0000807f')
        events[1][1]['sha256']=hashlib.sha256(raw[:events[1][1]['bytes']]).hexdigest();self.rejects(events,bytes(raw))
    def test_unknown_field_and_scope(self):
        events=copy.deepcopy(self.events);events[1][1]['extra']=True;self.rejects(events)
        events=copy.deepcopy(self.events);events[-2][1]['whole_e2b_qualified']=True;self.rejects(events)
    def test_partial_wrapper_and_duplicate_json(self):
        for raw in [b'',b'running 1 test\n',b'GEMMA_HYBRID_ARRAY {"record":0,"record":1}\n',b'test wrong ... GEMMA_HYBRID_ARRAY {}\n']:
            with self.assertRaises(ValueError):check.records(raw)
    def test_complete_wrapper_parses_exact_events(self):
        raw='\nrunning 1 test\n'
        for i,(name,body) in enumerate(self.events):
            text='frozen_graph=true scalar_order_unchanged=true full_e2b_qualified=false' if name=='TINY_COMPLETE' else json.dumps(body)
            raw+=('test '+check.TEST+' ... ' if i==0 else '')+'GEMMA_HYBRID_'+name+' '+text+'\n'
        raw+='ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 1.00s\n'
        self.assertEqual(check.records(raw.encode()),self.events)

if __name__=='__main__':unittest.main(argv=[sys.argv[0]])
