"""Portable softcap scaling, saturation, precision and evidence regressions."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import gemma_softcap as oracle


def sha(raw):return hashlib.sha256(raw).hexdigest()


class Softcap(unittest.TestCase):
    def test_hand_division_final_scale_identity_and_clip_counterexamples(self):
        x=oracle.inputs('hand',15,30.);y=oracle.mathematics(x,15,30.,False)
        self.assertAlmostEqual(y[13],30*math.tanh(1),places=13)
        self.assertGreater(abs(y[13]-x[13]),7);self.assertGreater(abs(y[13]-math.tanh(1)),20)
        self.assertGreater(abs(y[10]-30*math.tanh(float(x[10]))),20)
        self.assertGreater(np.max(abs(y-np.clip(x,-30,30))),7)

    def test_odd_monotone_bounds_and_signed_zero(self):
        for cap in (30.,.5,1/1024):
            x=oracle.inputs('grid',129,cap)
            for staged in (False,True):
                y=oracle.mathematics(x,129,cap,staged);self.assertTrue(np.all(np.diff(y)>=0));self.assertTrue(np.all(abs(y)<=cap));np.testing.assert_array_equal(y,-y[::-1])
                z=oracle.mathematics(np.array([-0.,0.],np.float32),2,cap,staged);self.assertTrue(np.signbit(z[0]));self.assertFalse(np.signbit(z[1]))

    def test_staged_division_overflow_has_finite_saturated_output(self):
        for name,cap in (('alternate_half',.5),('alternate_small',1/1024)):
            x=oracle.inputs(name,17,cap)
            with np.errstate(over='ignore'):divided=(x.astype(np.float64)/cap).astype(np.float32)
            self.assertEqual(np.count_nonzero(np.isinf(divided)),2)
            expected=(np.tanh(divided.astype(np.float64)).astype(np.float32).astype(np.float64)*cap).astype(np.float32)
            y=oracle.mathematics(x,17,cap,True);np.testing.assert_array_equal(y,expected);self.assertTrue(np.isfinite(y).all());self.assertEqual(y[0],-cap);self.assertEqual(y[-1],cap)
            np.testing.assert_allclose(y,oracle.mathematics(x,17,cap,False),rtol=1e-6,atol=1e-8)

    def test_empty_chunks_immutable_and_invalid_inputs(self):
        x=oracle.inputs('grid',129,30.);before=x.copy();y=oracle.mathematics(x,129,30.,True)
        for start,stop in ((0,1),(1,65),(65,129)):np.testing.assert_array_equal(oracle.mathematics(x[start:stop],stop-start,30.,True),y[start:stop])
        np.testing.assert_array_equal(x,before);self.assertEqual(oracle.mathematics(np.array([],np.float32),0,30.,True).shape,(0,))
        for cap in (0.,-1.,float('nan'),float('inf'),True,None,1+1j,np.array(30.),.1,31.,10**400):
            with self.assertRaises(ValueError):oracle.mathematics(x,129,cap,False)
        for width in (-1,130,True):
            with self.assertRaises(ValueError):oracle.geometry(width,30.)
        for value in (x.astype(np.float64),np.full_like(x,np.nan),x.reshape(3,43)):
            with self.assertRaises(ValueError):oracle.mathematics(value,129,30.,False)
        with self.assertRaises(ValueError):oracle.inputs('other',17,30.)
        with self.assertRaises(ValueError):oracle.mathematics(x,129,30.,1)

    def test_external_identity_and_strict_metadata(self):
        raw=oracle.export()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'math.json';path.write_bytes(raw);self.assertEqual(len(oracle.verify(path,sha(raw))['cases']),6)
            with self.assertRaises(ValueError):oracle.verify(path,'0'*64)
            base=json.loads(raw);changes=[]
            for key,value in [('full_vocab_or_model_qualified',True),('non30_caps_are_synthetic_operator_cases',False),('extra',1)]:
                item=copy.deepcopy(base);item[key]=value;changes.append(item)
            for key,value in [('width',True),('cap',1.),('staged_contract_division_overflow_count',True)]:
                item=copy.deepcopy(base);item['cases'][1][key]=value;changes.append(item)
            for group,value in [('input',True),('input',.1),('math',True),('staged_fp32',.1)]:
                item=copy.deepcopy(base);item['cases'][1][group]['values'][0]=value;changes.append(item)
            for item in changes:
                raw=oracle.encoded(item);path.write_bytes(raw)
                with self.assertRaises(ValueError):oracle.verify(path,sha(raw))

    def test_unique_selected_source_preserves_compile_decorator(self):
        raw=b'@partial(mx.compile,shapeless=True)\ndef logit_softcap(cap,x): return mx.tanh(x/cap)*cap'
        nodes=oracle.source_nodes({'gemma4_text.py':raw});self.assertEqual(set(nodes),{'logit_softcap'});self.assertEqual(ast.unparse(nodes['logit_softcap'].decorator_list[0]),'partial(mx.compile, shapeless=True)')
        for bad in (b'pass',raw+b'\ndef logit_softcap(cap,x): return x'):
            with self.assertRaises(ValueError):oracle.source_nodes({'gemma4_text.py':bad})

    def test_later_failure_retains_previous_raw_output(self):
        data=json.loads(oracle.export());data['cases']=data['cases'][1:3]
        core=types.ModuleType('mlx.core');core.__file__='mock-core';core.cpu,core.gpu='cpu','gpu';core.array,core.eval=np.array,lambda x:None;core.default_device,core.set_default_device=lambda:'cpu',lambda x:None;core.calls=0;core.tanh=np.tanh
        nn=types.ModuleType('mlx.nn');nn.__file__='mock-nn';mlx=types.ModuleType('mlx');mlx.core,mlx.nn=core,nn
        sources={'gemma4_text.py':b'def logit_softcap(cap,x):\n mx.calls+=1\n if mx.calls==2: raise RuntimeError("synthetic later softcap failure")\n return mx.tanh(x/cap)*cap'}
        admitted=dict(imports={'numpy':np.__file__,'mlx.core':core.__file__,'mlx.nn':nn.__file__})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dest=root/'out';dest.mkdir();reference=root/'math.json';reference.write_bytes(oracle.encoded(data))
            with mock.patch.object(oracle,'SOURCES',{k:sha(v) for k,v in sources.items()}),mock.patch.object(oracle,'verify',return_value=data),mock.patch.object(oracle.runtime,'admit_runtime',return_value=admitted),mock.patch.dict(sys.modules,{'mlx':mlx,'mlx.core':core,'mlx.nn':nn}):
                self.assertFalse(oracle.compare(reference,sha(reference.read_bytes()),sources,'cpu',dest,root/'runtime','0'*64))
            report=json.loads((dest/'observation.json').read_bytes());self.assertIn('synthetic later softcap failure',report['primary_error']);self.assertTrue(report['cases'][0]['completed']);self.assertFalse(report['cases'][1]['completed'])
            item=report['cases'][0]['arrays']['source_output'];raw=(dest/'observed.bin').read_bytes()[item['offset']:item['offset']+item['bytes']];self.assertEqual(sha(raw),item['sha256']);self.assertEqual(len(report['comparisons']),2)


if __name__=='__main__':unittest.main()
