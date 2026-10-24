"""Portable unlearned per-head RMS geometry and evidence regressions."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import gemma_v_norm as oracle


def sha(raw):return hashlib.sha256(raw).hexdigest()


class VNorm(unittest.TestCase):
    def test_hand_uses_mean_per_head_without_learned_scale(self):
        x=oracle.inputs('hand',1,1,2,4);y=oracle.mathematics(x,1,1,2,4,False)
        expected=np.array([[[[2/np.sqrt(1+oracle.EPSILON),0,0,0],[0,0,-3/np.sqrt(2.25+oracle.EPSILON),0]]]])
        np.testing.assert_allclose(y,expected,rtol=1e-14,atol=1e-14)
        l2=x.astype(np.float64)/np.sqrt(np.sum(x.astype(np.float64)**2,-1,keepdims=True)+oracle.EPSILON)
        self.assertGreater(np.max(abs(y-l2)),.9);self.assertGreater(np.max(abs(y-x)),.9)
        combined=x.astype(np.float64)/np.sqrt(np.mean(x.astype(np.float64)**2,(-2,-1),keepdims=True)+oracle.EPSILON)
        self.assertGreater(np.max(abs(y-combined)),.2)

    def test_zero_width_one_and_epsilon_dominated_nonzero(self):
        x=oracle.inputs('zeros',1,2,2,7);np.testing.assert_array_equal(oracle.mathematics(x,1,2,2,7,False),x)
        x=np.array([[[[2.],[-3.]]]],np.float32);y=oracle.mathematics(x,1,1,2,1,False)
        np.testing.assert_allclose(y.reshape(-1),[2/np.sqrt(4+oracle.EPSILON),-3/np.sqrt(9+oracle.EPSILON)],rtol=1e-14)
        x=oracle.inputs('epsilon_dominated',1,2,2,4);y=oracle.mathematics(x,1,2,2,4,False)
        self.assertGreater(np.max(abs(y)),0);self.assertLess(np.max(abs(y)),.002)
        no_eps=x/np.sqrt(np.mean(x*x,-1,keepdims=True));self.assertGreater(np.max(abs(y-no_eps)),1)
        x=np.array([[[[-0.,0.]]]],np.float32);y=oracle.mathematics(x,1,1,1,2,True);self.assertTrue(np.signbit(y.reshape(-1)[0]));self.assertFalse(np.signbit(y.reshape(-1)[1]))

    def test_staged_cast_independent_rows_chunks_and_input_immutability(self):
        x=oracle.inputs('batch',2,3,3,5);before=x.copy();full=oracle.mathematics(x,2,3,3,5,False)
        np.testing.assert_array_equal(oracle.mathematics(x,2,3,3,5,True),full.astype(np.float32))
        for start,stop in ((0,1),(1,3)):
            part=oracle.mathematics(x[:,start:stop],2,stop-start,3,5,False);np.testing.assert_array_equal(part,full[:,start:stop])
        for head in range(3):np.testing.assert_array_equal(oracle.mathematics(x[:,:,head:head+1],2,3,1,5,False),full[:,:,head:head+1])
        np.testing.assert_array_equal(x,before);self.assertEqual(oracle.mathematics(oracle.inputs('empty',1,0,2,4),1,0,2,4,True).shape,(1,0,2,4))

    def test_geometry_dtype_finite_and_pattern_validation(self):
        good=(1,2,2,4)
        for index,values in {0:[0,3,True],1:[-1,4,False],2:[0,4,True],3:[0,513,False]}.items():
            for value in values:
                args=list(good);args[index]=value
                with self.assertRaises(ValueError):oracle.inputs('batch',*args)
        with self.assertRaises(ValueError):oracle.inputs('unknown',*good)
        with self.assertRaises(ValueError):oracle.inputs('hand',*good)
        x=oracle.inputs('batch',*good)
        for value in (x.astype(np.float64),np.full_like(x,np.nan),x.reshape(-1)):
            with self.assertRaises(ValueError):oracle.mathematics(value,*good,False)
        with self.assertRaises(ValueError):oracle.mathematics(x,*good,1)

    def test_external_identity_and_strict_metadata(self):
        raw=oracle.export()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'math.json';path.write_bytes(raw);self.assertEqual(len(oracle.verify(path,sha(raw))['cases']),8)
            with self.assertRaises(ValueError):oracle.verify(path,'0'*64)
            base=json.loads(raw);changes=[]
            for key,value in [('attention_or_kv_qualified',True),('constructor_qualified',True),('extra',1)]:
                item=copy.deepcopy(base);item[key]=value;changes.append(item)
            for key,value in [('heads',True),('epsilon',.001),('width',5)]:
                item=copy.deepcopy(base);item['cases'][1][key]=value;changes.append(item)
            for group,value in [('input',True),('input',.1),('math',True),('staged_fp32',.1)]:
                item=copy.deepcopy(base);item['cases'][1][group]['values'][0]=value;changes.append(item)
            for item in changes:
                raw=oracle.encoded(item);path.write_bytes(raw)
                with self.assertRaises(ValueError):oracle.verify(path,sha(raw))

    def test_unique_selected_source(self):
        raw=b'class RMSNormNoScale:\n def __call__(self,x): return mx.fast.rms_norm(x,None,self.eps)'
        nodes=oracle.source_nodes({'gemma4_text.py':raw});self.assertEqual(set(nodes),{'RMSNormNoScale'})
        for bad in (b'pass',raw+b'\nclass RMSNormNoScale: pass',raw+b'\n def __call__(self,x): return x'):
            with self.assertRaises(ValueError):oracle.source_nodes({'gemma4_text.py':bad})

    def test_later_failure_retains_previous_completed_raw_output(self):
        data=json.loads(oracle.export());data['cases']=data['cases'][1:3]
        core=types.ModuleType('mlx.core');core.__file__='mock-core';core.cpu,core.gpu='cpu','gpu';core.array,core.eval=np.array,lambda x:None;core.default_device,core.set_default_device=lambda:'cpu',lambda x:None;core.calls=0
        core.fast=types.SimpleNamespace(rms_norm=lambda x,w,eps:(x.astype(np.float64)/np.sqrt(np.mean(x.astype(np.float64)**2,-1,keepdims=True)+eps)).astype(np.float32))
        nn=types.ModuleType('mlx.nn');nn.__file__='mock-nn';mlx=types.ModuleType('mlx');mlx.core,mlx.nn=core,nn
        sources={'gemma4_text.py':b'class RMSNormNoScale:\n def __call__(self,x):\n  mx.calls+=1\n  if mx.calls==2: raise RuntimeError("synthetic later V failure")\n  return mx.fast.rms_norm(x,None,self.eps)'}
        admitted=dict(imports={'numpy':np.__file__,'mlx.core':core.__file__,'mlx.nn':nn.__file__})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dest=root/'out';dest.mkdir();reference=root/'math.json';reference.write_bytes(oracle.encoded(data))
            with mock.patch.object(oracle,'SOURCES',{k:sha(v) for k,v in sources.items()}),mock.patch.object(oracle,'verify',return_value=data),mock.patch.object(oracle.runtime,'admit_runtime',return_value=admitted),mock.patch.dict(sys.modules,{'mlx':mlx,'mlx.core':core,'mlx.nn':nn}):
                self.assertFalse(oracle.compare(reference,sha(reference.read_bytes()),sources,'cpu',dest,root/'runtime','0'*64))
            report=json.loads((dest/'observation.json').read_bytes());self.assertIn('synthetic later V failure',report['primary_error']);self.assertTrue(report['cases'][0]['completed']);self.assertFalse(report['cases'][1]['completed'])
            item=report['cases'][0]['arrays']['source_output'];raw=(dest/'observed.bin').read_bytes()[item['offset']:item['offset']+item['bytes']];self.assertEqual(sha(raw),item['sha256']);self.assertEqual(len(report['comparisons']),2)


if __name__=='__main__':unittest.main()
