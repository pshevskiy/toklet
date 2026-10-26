"""Frozen prior scalar bytes shaped as synthetic hybrid packets: reader only."""
from pathlib import Path
import copy
import hashlib
import struct
import sys
import unittest
import check_gemma_hybrid_generation as check

class Generation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(sys.argv[-1]);cls.cases={}
        for profile,assets in check.ASSETS.items():
            state,witness,expected=check.references(root,profile)
            old=root/assets[0]/'observation/arrays.bin'
            raw=check.checked.read(old,64*1024**2)
            # The native scalar bytes are frozen in accepted publications. This
            # synthetic wrapper never supplies new GPU or model evidence.
            pins={'raw':'4bbf0269f4616c08c3acdb0c55fb6eaa130f376979ab160584b8504c225c366b','basic-hello':'306e25123325ca2e3166479c701837ff5aaa72542f816066ad67153efa097c12','basic-hello-stream':'81e7f0c5ed93caf9e776de8855239de6b62398ab22e57b7d58a82c7dc248ed5c'}
            check.require(hashlib.sha256(raw).hexdigest()==pins[profile],'frozen scalar bytes')
            observation=copy.deepcopy(state);m=dict(weights_bytes=2605000000,kv_bytes=6000000,scratch_bytes=150000000,history_bytes=256,logits_bytes=67108864,metadata_bytes=200000,metadata_allowance_bytes=100663296)
            m['admitted_bytes']=sum(m.values());observation['memory']=m
            observation.update(execution_backend='hybrid',construction_context_dropped=True,model_bytes_including_store=2605100000,hybrid_norm_copy_bytes=1160000)
            reserve=16*1024**2+(1024**3 if profile!='raw' else 0)+(4*1024**2 if profile=='basic-hello-stream' else 0)
            observation.update(numerical_budget_bytes=6*1024**3-reserve,diagnostic_allowance_bytes=16*1024**2,combined_admitted_bytes=m['admitted_bytes']+reserve)
            observation['generation_admission_probe']=dict(required=m['admitted_bytes'],limit=m['admitted_bytes']-1,minus_one_expected_budget_error=True,error=f"MemoryBudget {{ required: {m['admitted_bytes']}, limit: {m['admitted_bytes']-1} }}",exact_memory_unchanged=True)
            cls.cases[profile]=(observation,raw,state,witness,expected)
    def replay(self,profile='raw',changed=None,raw=None):
        observation,binary,state,witness,expected=self.cases[profile]
        return check.replay(profile,observation if changed is None else changed,binary if raw is None else raw,state,witness,expected)
    def rejects(self,profile='raw',changed=None,raw=None):
        with self.assertRaises((ValueError,KeyError,TypeError,IndexError)):self.replay(profile,changed,raw)
    def test_three_complete_synthetic_profiles(self):
        for profile,rows in [('raw',16),('basic-hello',22),('basic-hello-stream',24)]:
            result=self.replay(profile);self.assertTrue(result['passed']);self.assertEqual(result['rows'],rows);self.assertFalse(result['whole_e2b_qualified'])
    def test_wrong_pending_stop_or_text(self):
        for key,value in [('pending',0),('history',[])]:
            observation=copy.deepcopy(self.cases['raw'][0]);observation['scenarios'][0][key]=value;self.rejects(changed=observation)
        observation=copy.deepcopy(self.cases['basic-hello-stream'][0]);observation['scenarios'][0]['stream']['chunks'][-1]['text']='unexpected';self.rejects('basic-hello-stream',observation)
    def test_budget_probe_and_boolean_byte_count(self):
        observation=copy.deepcopy(self.cases['raw'][0]);observation['generation_admission_probe']['minus_one_expected_budget_error']=False;self.rejects(changed=observation)
        observation=copy.deepcopy(self.cases['raw'][0]);observation['memory']['history_bytes']=True;self.rejects(changed=observation)
    def test_missing_extra_and_truncated_capture(self):
        observation=copy.deepcopy(self.cases['raw'][0]);observation['arrays'].pop();self.rejects(changed=observation)
        self.rejects(raw=self.cases['raw'][1][:-1]);self.rejects(raw=self.cases['raw'][1]+b'x')
    def test_nonfinite_and_changed_hash(self):
        observation=copy.deepcopy(self.cases['raw'][0]);raw=bytearray(self.cases['raw'][1]);raw[:4]=bytes.fromhex('0000807f')
        observation['arrays'][0]['sha256']=hashlib.sha256(raw[:1048576]).hexdigest();self.rejects(changed=observation,raw=bytes(raw))
    def test_unchanged_argmax_does_not_hide_numerical_failure(self):
        observation=copy.deepcopy(self.cases['raw'][0]);raw=bytearray(self.cases['raw'][1]);first=struct.unpack('<f',raw[:4])[0];raw[:4]=struct.pack('<f',first+0.1)
        # All same-prefix rows must receive the same perturbation, retaining the
        # schedule equivalence gate while breaking only the numerical contract.
        for item in observation['arrays']:
            if item['step']==0:
                offset=item['offset'];raw[offset:offset+4]=raw[:4];item['sha256']=hashlib.sha256(raw[offset:offset+1048576]).hexdigest()
        result=self.replay(changed=observation,raw=bytes(raw));self.assertFalse(result['passed']);self.assertTrue(result['schedules_bitwise_identical'])
    def test_profile_scope_and_rendering(self):
        observation=copy.deepcopy(self.cases['basic-hello'][0]);observation['rendered_prompt']='Hi';self.rejects('basic-hello',observation)
        observation=copy.deepcopy(self.cases['raw'][0]);observation['metal_qualified']=True;self.rejects(changed=observation)
    def test_timing_and_prefix_identity(self):
        observation=copy.deepcopy(self.cases['raw'][0]);observation['scenarios'][0]['reports'][0]['total_seconds']=True;self.rejects(changed=observation)
        observation=copy.deepcopy(self.cases['raw'][0]);observation['arrays'][0]['prefix']=[2];self.rejects(changed=observation)

if __name__=='__main__':unittest.main(argv=[sys.argv[0]])
