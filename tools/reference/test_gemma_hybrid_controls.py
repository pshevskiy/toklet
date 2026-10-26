"""Synthetic failure controls for the ownership reader, never GPU evidence."""
from pathlib import Path
import copy,json,sys,unittest
import check_gemma_hybrid_controls as control
import check_gemma_hybrid_tiny as check

class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.binary=check.fixture(Path(sys.argv[-1]));cls.events=[]
        weights=control.expected_weights(cls.data,cls.binary);identity=1
        for i,matrix in enumerate(weights):
            for chunk in matrix['chunks']:
                for label in ['codes','scales','biases']:
                    chunk[label]['identity']=hex(identity);identity+=1
            cls.events.append(('WEIGHT_BEFORE',dict(index=i,matrix=matrix)))
        before=copy.deepcopy(cls.events)
        m=dict(weights_bytes=100000,kv_bytes=200000,scratch_bytes=300000,history_bytes=256,logits_bytes=4352,metadata_bytes=50000,metadata_allowance_bytes=96*1024**2)
        m['admitted_bytes']=sum(m.values())
        memory=dict(raw_admitted_bytes=m['admitted_bytes']-512,model_bytes=120000,generation_wrapper_bytes=512,generation=m,raw_minus_one_rust_allocations=0,generation_minus_one_rust_allocations=0,smaller_request_capacity=5,physical_capacity=32)
        cls.events.append(('MEMORY',memory))
        extra=memory['raw_admitted_bytes']-memory['model_bytes']-64*1024**2
        for gather,calls in [(False,12),(True,1)]:cls.events.append(('FAILURE',dict(gather=gather,successful_calls_before_injection=calls,committed_prefix=3,prefix_unchanged=True,reset_rejected=True,reuse_rejected=True,fresh_replay_tokens=5,extra_live_session_bytes=extra,in_flight_failure_qualified=False)))
        cls.events.extend(('WEIGHT_AFTER',record) for _,record in before)
        cls.events.append(('COMPLETE',dict(generation_stop_cancel_output_failure_continue=True,foreign_source_rejected=True,weight_buffers_unchanged=True,different_device_tested=False,whole_e2b_qualified=False,performance_qualified=False,phone_qualified=False)))
    def rejects(self,events):
        with self.assertRaises((ValueError,KeyError,TypeError,IndexError)):control.replay(events,self.data,self.binary)
    def test_complete(self):
        result=control.replay(self.events,self.data,self.binary)
        self.assertEqual(result['matrices'],35);self.assertEqual(result['unique_unchanged_buffers'],105)
    def test_missing_extra_reordered(self):
        self.rejects(self.events[:-1]);self.rejects(self.events+[self.events[-1]])
        events=copy.deepcopy(self.events);events[0],events[1]=events[1],events[0];self.rejects(events)
    def test_changed_weight_and_guard_hash(self):
        for key,value in [('words',1),('sha256','0'*64)]:
            events=copy.deepcopy(self.events);events[0][1]['matrix']['chunks'][0]['codes'][key]=value;self.rejects(events)
    def test_identity_alias_or_changed_owner(self):
        events=copy.deepcopy(self.events);events[0][1]['matrix']['chunks'][0]['scales']['identity']=events[0][1]['matrix']['chunks'][0]['codes']['identity'];self.rejects(events)
        events=copy.deepcopy(self.events);events[38][1]['matrix']['chunks'][0]['codes']['identity']='0xabcdef';self.rejects(events)
    def test_memory_categories_and_boolean_counts(self):
        for key,value in [('metadata_allowance_bytes',1),('history_bytes',True),('admitted_bytes',1)]:
            events=copy.deepcopy(self.events);events[35][1]['generation'][key]=value;self.rejects(events)
    def test_failures_and_unqualified_scope(self):
        events=copy.deepcopy(self.events);events[36][1]['in_flight_failure_qualified']=True;self.rejects(events)
        events=copy.deepcopy(self.events);events[-1][1]['whole_e2b_qualified']=True;self.rejects(events)
    def test_unknown_field(self):
        events=copy.deepcopy(self.events);events[35][1]['extra']=True;self.rejects(events)
    def test_wrapper_and_duplicate_json(self):
        with self.assertRaises(ValueError):control.records(b'GEMMA_HYBRID_CONTROL_COMPLETE {"x":1,"x":1}\n')
        raw='\nrunning 1 test\n'
        for i,(label,record) in enumerate(self.events):raw+=('test '+control.TEST+' ... ' if i==0 else '')+control.PREFIX+label+' '+json.dumps(record)+'\n'
        raw+='ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 1.00s\n'
        self.assertEqual(control.records(raw.encode()),self.events)
        with self.assertRaises(ValueError):control.records(raw.replace('ok\n','FAILED\n').encode())

if __name__=='__main__':unittest.main(argv=[sys.argv[0]])
