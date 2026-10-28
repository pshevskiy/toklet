"""CPU-only malformed capture and unchanged numerical-gate controls."""
import copy
import struct
import unittest
from unittest import mock
import check_llama3_checkpoint as check

class Checker(unittest.TestCase):
    def test_nonoverlapping_complete_rows_required(self):
        items=[];cursor=0
        for(name,start),shape in check.inventory().items():
            size=shape[0]*shape[1]*4
            items.append(dict(name=name,row_start=start,shape=shape,event='completed',dtype='<f4',offset=cursor,bytes=size,sha256='0'*64));cursor+=size
        self.assertEqual(len(items),1288)
        self.assertEqual(check.validate_inventory(items),cursor)
        bad=copy.deepcopy(items);bad[-1]=bad[-2].copy()
        with self.assertRaises(ValueError):check.validate_inventory(bad)
        bad=copy.deepcopy(items);bad[0]['row_start']=1
        with self.assertRaises(ValueError):check.validate_inventory(bad)
        bad=copy.deepcopy(items);bad[0]['shape'][1]+=1
        with self.assertRaises(ValueError):check.validate_inventory(bad)
        for key,value in [('row_start',False),('row_start',0.0),('shape',[1.0,2048])]:
            bad=copy.deepcopy(items);bad[0][key]=value
            with self.assertRaises(ValueError):check.validate_inventory(bad)
        with self.assertRaises(ValueError):check.validate_inventory(items[:-1])
    def test_gate_requires_finite_values_and_keeps_exact_failure_count(self):
        target=struct.pack('<3f',1,2,3);gate=dict(atol=.01,rtol=0.)
        result=check.qualify([1,2.005,3.02],struct.iter_unpack('<f',target),gate)
        self.assertEqual((result['values'],result['failed']),(3,1));self.assertGreater(result['max_gate_fraction'],1)
        with self.assertRaises(ValueError):check.qualify([float('nan')],[(0.,)],gate)
        with self.assertRaises(ValueError):check.qualify([0.],[(float('inf'),)],gate)
        with self.assertRaises(ValueError):check.qualify([0.,1.],[(0.,)],gate)
    def test_schedule_causal_boundary_and_selection_disagreements_reject(self):
        with mock.patch.object(check,'VOCAB',2):
            logits={(case,'chunk'+str(chunk)):[1.,2.]*len(spec['tokens'])
                    for case,spec in check.reference.CASES.items()for chunk in check.CHUNKS}
            boundary={f'boundary.chunk{chunk}.logits':[1.,2.]for chunk in [32,128]}
            initial={f'{group}.step_0.logits':[1.,2.]for group in ['greedy','native_generated']}
            selected={f'{group}.step_{step}.logits':1 for group in ['greedy','native_generated']for step in range(4)}
            report=dict(native_selected_on_teacher_forced_prefixes=[1]*4,native_generated_ids=[1]*4)
            check.exact_logits_controls(logits,boundary,initial,selected,report)
            bad=copy.deepcopy(logits);bad[('short','chunk3')][-1]=3.
            with self.assertRaisesRegex(ValueError,'schedules'):check.exact_logits_controls(bad,boundary,initial,selected,report)
            bad=copy.deepcopy(logits)
            for chunk in check.CHUNKS:bad[('future_changed','chunk'+str(chunk))][0]=3.
            with self.assertRaisesRegex(ValueError,'causal'):check.exact_logits_controls(bad,boundary,initial,selected,report)
            bad=copy.deepcopy(boundary);bad['boundary.chunk128.logits'][0]=3.
            with self.assertRaisesRegex(ValueError,'boundary'):check.exact_logits_controls(logits,bad,initial,selected,report)
            bad=copy.deepcopy(initial);bad['native_generated.step_0.logits'][0]=3.
            with self.assertRaisesRegex(ValueError,'Initial'):check.exact_logits_controls(logits,boundary,bad,selected,report)
            bad=copy.deepcopy(report);bad['native_generated_ids'][0]=0
            with self.assertRaisesRegex(ValueError,'maximum'):check.exact_logits_controls(logits,boundary,initial,selected,bad)
if __name__=='__main__':unittest.main()
