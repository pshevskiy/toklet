"""Synthetic real-reference reader controls; never native qualification."""
from pathlib import Path
import copy
import hashlib
import json
import struct
import sys
import unittest
import check_gemma_hybrid_real as check

class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference,inventory=map(Path,sys.argv[-2:])
        cls.specs,cls.binary,cls.inventory=check.frozen(reference,inventory)
        cls.events=[]
        owners=[]
        for number,(key,spec) in enumerate(check.storage_specs(cls.inventory)):
            owners.append(dict(key,buffer=dict(allocation=hex(number+1),bytes=spec['guarded_bytes'],sha256=spec['guarded_sha256'])))
        def buffers(phase):
            cls.events.extend(('REAL_BUFFER',dict(phase=phase,value=copy.deepcopy(value))) for value in owners)
        buffers('loaded')
        for index,spec in enumerate(cls.inventory['vectors']):
            cls.events.append(('REAL_CPU_VECTOR',dict(index=index,bytes=spec['fp32_bytes'],capacity_bytes=spec['fp32_bytes'],sha256=spec['fp32_sha256'])))
        buffers('copied')
        cls.events.append(('REAL_ADMISSION',dict(model_bytes=2606000000,norm_copy_bytes=1160000,raw_admitted_bytes=2800000000,
            diagnostic_policy_bytes=128*1024**2,process_policy_bytes=6*1024**3,numerical_budget=6*1024**3-128*1024**2,
            context_dropped=True,same_device_fresh_context=True,exact_admission=True,minus_one_rejected=True,positions=32,input_tokens=[2,100,257,100])))
        raw=bytearray();number=0
        for schedule in check.SCHEDULES:
            cls.events.append(('REAL_SCHEDULE',schedule))
            for name,before,count in check.signatures(schedule):
                values,length=check.selected(cls.specs['staged_fp32',name],cls.binary,before,count)
                block=b''.join(struct.pack('<f',v) for v in values)
                header=dict(record=number,name=name,offset=len(raw),bytes=len(block),before=before,count=count,values=length)
                cls.events.extend([('ARRAY_ATTEMPT',header),('ARRAY',dict(header,sha256=hashlib.sha256(block).hexdigest()))]);raw.extend(block);number+=1
        cls.raw=bytes(raw);buffers('executed')
        cls.events.append(('CAPTURE',dict(records=number,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),whole_e2b_qualified=False)))
        cls.events.append(('REAL_COMPLETE',dict(captured=True,positions=32,tested_input_positions=4,source_path_and_descriptor_unchanged=True,
            resident_buffers_unchanged=True,full_model_window_qualified=False,performance_qualified=False,phone_qualified=False,original_precision_failures_waived=False)))
    def replay(self,events=None,raw=None):
        return check.replay(self.events if events is None else events,self.raw if raw is None else raw,self.specs,self.binary,self.inventory)
    def rejects(self,events=None,raw=None):
        with self.assertRaises((ValueError,KeyError,TypeError,IndexError,StopIteration)):self.replay(events,raw)
    def test_complete_preserves_mathematical_failures(self):
        result=self.replay();self.assertTrue(result['staged_fp32_passed']);self.assertFalse(result['ideal_fp64_passed'])
        self.assertEqual(result['records'],3016);self.assertEqual(result['binary_bytes'],66600960)
        self.assertFalse(result['whole_e2b_qualified'])
    def test_missing_extra_and_reordered(self):
        self.rejects(self.events[:-1]);self.rejects(self.events+[self.events[-1]])
        events=copy.deepcopy(self.events);events[0],events[1]=events[1],events[0];self.rejects(events)
    def test_source_hash_or_aliased_buffer(self):
        for key,value in [('sha256','0'*64),('allocation','0x2')]:
            events=copy.deepcopy(self.events);events[0][1]['value']['buffer'][key]=value;self.rejects(events)
    def test_norm_copy_bytes_and_source_identity(self):
        events=copy.deepcopy(self.events);events[1186][1]['sha256']='0'*64;self.rejects(events)
        events=copy.deepcopy(self.events);events[1186][1]['capacity_bytes']=True;self.rejects(events)
    def test_admission_scope_and_typed_counts(self):
        index=2*1186+262
        for key,value in [('positions',131072),('exact_admission',1),('model_bytes',True)]:
            events=copy.deepcopy(self.events);events[index][1][key]=value;self.rejects(events)
    def test_partial_binary_and_descriptor(self):
        self.rejects(raw=self.raw[:-1])
        events=copy.deepcopy(self.events);index=next(i for i,(label,_) in enumerate(events) if label=='ARRAY');events[index][1]['sha256']='0'*64;self.rejects(events)
    def test_changed_value_retained_as_staged_failure(self):
        events=copy.deepcopy(self.events);raw=bytearray(self.raw);raw[:4]=struct.pack('<f',100.)
        index=next(i for i,(label,_) in enumerate(events) if label=='ARRAY');spec=events[index][1]
        spec['sha256']=hashlib.sha256(raw[:spec['bytes']]).hexdigest();events[-2][1]['sha256']=hashlib.sha256(raw).hexdigest()
        result=self.replay(events,bytes(raw));self.assertFalse(result['passed']);self.assertGreater(result['schedules'][0]['contracts']['staged_fp32']['failed_values'],0)
    def test_terminal_and_wrapper(self):
        events=copy.deepcopy(self.events);events[-1][1]['original_precision_failures_waived']=True;self.rejects(events)
        with self.assertRaises(ValueError):check.records(b'GEMMA_HYBRID_REAL_COMPLETE {"x":1,"x":1}\n')
        raw='\nrunning 1 test\n'+''.join(('test '+check.TEST+' ... ' if i==0 else '')+'GEMMA_HYBRID_'+label+' '+json.dumps(value)+'\n' for i,(label,value) in enumerate(self.events))
        raw+='ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 1.00s\n'
        self.assertEqual(check.records(raw.encode()),self.events)

if __name__=='__main__':unittest.main(argv=[sys.argv[0]])
