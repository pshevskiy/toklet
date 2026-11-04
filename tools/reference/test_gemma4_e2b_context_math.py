"""CPU-only qualification of exact ordered products and extended row scheduling."""
from pathlib import Path
import hashlib
import math
import struct
import unittest
import json
import tempfile
from unittest.mock import patch
import gemma4_e2b_context_reference as producer
import numpy as np
import gemma4_e2b_context_math as candidate
import gemma4_long_witness as previous
import test_gemma4_text_math as fixture


class ContextMathControls(unittest.TestCase):
    def test_products_preserve_scalar_float64_rounding_and_fsum_order(self):
        rng=np.random.default_rng(197)
        for dtype in [np.float32,np.float64]:
            for count in [1,4,16,31,32,64,128,256,512]:
                for scale in [1.,1e-20,1e20]:
                    a=(rng.standard_normal(count)*scale).astype(dtype)
                    b=(rng.standard_normal(count)/scale).astype(dtype)
                    expected=math.fsum(float(x)*float(y)for x,y in zip(a,b,strict=True))
                    self.assertEqual(struct.pack('<d',expected),struct.pack('<d',candidate.product_sum(a,b)))
        for a,b in [(np.array([1e16,1.,-1e16]),np.ones(3)),(np.array([np.nextafter(0.,1.),-0.]),np.ones(2))]:
            self.assertEqual(struct.pack('<d',math.fsum(float(x)*float(y)for x,y in zip(a,b))),struct.pack('<d',candidate.product_sum(a,b)))
        with self.assertRaises(ValueError):candidate.product_sum(np.ones(1),np.ones(2))
        with self.assertRaises(ValueError):candidate.product_sum(np.ones((1,2)),np.ones((1,2)))

    def test_selected_head_and_chunked_long_math_preserve_previous_oracle(self):
        self.assertEqual(hashlib.sha256(Path(candidate.frozen.__file__).read_bytes()).hexdigest(),candidate.ORIGINAL_SOURCE_SHA256)
        weights,_,_=fixture.fixture()
        for count,selected in [(4,[0,2,3]),(32,[0,15,30,31]),(64,[0,31,63]),
                               (1025,[0,31,511,512,1023,1024])]:
            ids=([2,7,13,7]*((count+3)//4))[:count]
            for staged in [False,True]:
                expected={};actual={}
                previous.evaluate(weights,ids,staged,sink=lambda name,value:expected.update({name:value.copy()}))
                candidate._evaluate(weights,ids,staged,selected_positions=selected,sink=lambda name,value:actual.update({name:value.copy()}))
                self.assertEqual(set(expected),set(actual))
                for name in expected:
                    wanted=expected[name][:,selected,:]if name in ['head','logits']else expected[name]
                    if staged or name not in ['head','logits']:np.testing.assert_array_equal(actual[name],wanted,err_msg=name)
                    else:np.testing.assert_allclose(actual[name],wanted,rtol=0,atol=1e-10,err_msg=name)

    def test_frozen_input_contract_rejects_size_and_identity_mutations(self):
        for phase,count in [('compat32',32),('pilot64',64),('full512',512)]:
            ids,positions=producer.inputs(phase)
            self.assertEqual(len(ids),count)
            self.assertTrue(all(0<=v<count for v in positions))
        with tempfile.TemporaryDirectory()as name:
            folder=Path(name)
            manifest=(producer.CONTRACT/'manifest.json').read_bytes()
            raw=(producer.CONTRACT/'tokens.u32le').read_bytes()
            (folder/'manifest.json').write_bytes(manifest)
            for changed in [raw[:-4],bytes([raw[0]^1])+raw[1:]]:
                (folder/'tokens.u32le').write_bytes(changed)
                with patch.object(producer,'CONTRACT',folder):
                    with self.assertRaises(ValueError):producer.inputs('full512')

    def test_public_real_admission_and_selection_bounds(self):
        weights,_,_=fixture.fixture()
        with self.assertRaises(ValueError):candidate.evaluate(weights,[2],False,selected_positions=[0],sink=lambda *_:None)
        for ids,selected in [([2]*1026,[0]),([2],[True]),([2,3],[1,0]),([2,3],[0,0]),([2],[1]),([2],[])]:
            with self.assertRaises(ValueError):candidate._evaluate(weights,ids,False,selected_positions=selected,sink=lambda *_:None)
        with self.assertRaises(ValueError):candidate._evaluate(weights,[2],False,selected_positions=[0],sink=None)


if __name__=='__main__':unittest.main()
