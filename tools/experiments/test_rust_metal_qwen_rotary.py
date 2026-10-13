import copy
import unittest
import rust_metal_qwen_rotary as r
from test_rust_metal_rotary import fixture

class QwenRotary(unittest.TestCase):
    def test_eight_frozen_gates_and_exact_identity(self):
        inputs,a=r.cases();self.assertEqual(len(inputs),9)
        for c in inputs:r.validate(fixture([c],a),c,a[c['id']])
        c=inputs[0];bad=copy.deepcopy(a[c['id']]);bad['reference'][0]+=1000
        with self.assertRaises(ValueError):r.validate(fixture([c],a),c,bad)
    def test_dimension_and_repetition_fail_closed(self):
        inputs,a=r.cases();c=inputs[0]
        for kind in('dimension','repeat'):
            e=fixture([c],a)
            if kind=='dimension':e['result']['cases'][0]['dimension']=8
            else:e['result']['cases'][0]['rows'][-1]['output_bits'][0]^=1
            with self.assertRaises(ValueError):r.validate(e,c,a[c['id']])
if __name__=='__main__':unittest.main()
