"""Portable dense decoder ordering, branch and retained-evidence regressions."""
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
import gemma_decoder_order as oracle


def sha(raw):return hashlib.sha256(raw).hexdigest()


def norm(x,w):
    x=np.asarray(x,np.float64)
    return x/np.sqrt(np.mean(x*x,-1,keepdims=True)+oracle.EPSILON)*w


class DecoderOrder(unittest.TestCase):
    def test_hand_norm_residual_scalar_and_gate_order(self):
        args=(1,1,4,6,2,'full',True);x=oracle.inputs(*args);r=oracle.mathematics(x,*args,False)
        a=x['hidden']+norm(x['attention_output'],x['post_attention_norm'])
        np.testing.assert_allclose(r['attention_residual'],a,rtol=1e-14,atol=1e-14)
        self.assertGreater(np.max(abs(a-norm(x['hidden']+x['attention_output'],x['post_attention_norm']))),.1)
        np.testing.assert_allclose(r['ff_input'],norm(a,x['pre_ff_norm']),rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(r['ff_residual'],a+norm(r['ff_down'],x['post_ff_norm']),rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(r['ple_gate'],r['ff_residual']@x['ple_gate_weight'].T,rtol=1e-14,atol=1e-14)
        self.assertGreater(np.max(abs(r['ple_gate']-x['hidden']@x['ple_gate_weight'].T)),.1)
        np.testing.assert_allclose(r['ple_activated'],.5*r['ple_gate']*(1+np.tanh(math.sqrt(2/math.pi)*(r['ple_gate']+.044715*r['ple_gate']**3))),rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(r['ple_product'],r['ple_activated']*x['per_layer_input'],rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(r['ple_residual'],r['ff_residual']+norm(r['ple_projected'],x['post_ple_norm']),rtol=1e-14,atol=1e-14)
        np.testing.assert_array_equal(r['output'],r['ple_residual']*-1.25)
        self.assertGreater(np.max(abs(r['output']-(r['ff_residual']*-1.25+r['ple_norm']))),.1)
        self.assertLess(np.min(r['attention_input']),0)
        self.assertGreater(np.max(abs(r['attention_input']-norm(x['hidden'],1+x['input_norm']))),.1)

    def test_each_ple_guard_term_and_synthetic_absent_scalar(self):
        args=(1,2,5,7,3);x=oracle.inputs(*args,'full',True)
        enabled=oracle.mathematics(x,*args,'full',True,False)
        for mode in sorted(oracle.MODES-{'full'}):
            r=oracle.mathematics(x,*args,mode,True,False)
            self.assertNotIn('ple_gate',r);np.testing.assert_array_equal(r['output'],r['ff_residual']*-1.25)
            self.assertGreater(np.max(abs(r['output']-enabled['output'])),.1)
        r=oracle.mathematics(x,*args,'full',False,False)
        np.testing.assert_array_equal(r['output'],r['ple_residual']);np.testing.assert_array_equal(r['output']*-1.25,enabled['output'])

    def test_staged_storage_boundaries_and_exact_chunk_reuse(self):
        args=(2,3,5,7,3,'full',True);x=oracle.inputs(*args);before={k:v.copy() for k,v in x.items()};r=oracle.mathematics(x,*args,True)
        a=(x['hidden'].astype(np.float64)+r['attention_norm'].astype(np.float64)).astype(np.float32)
        np.testing.assert_array_equal(r['attention_residual'],a)
        np.testing.assert_array_equal(r['ff_gate'],(r['ff_input'].astype(np.float64)@x['gate_weight'].astype(np.float64).T).astype(np.float32))
        np.testing.assert_array_equal(r['ff_product'],(r['ff_activated'].astype(np.float64)*r['ff_up'].astype(np.float64)).astype(np.float32))
        for start,stop in ((0,1),(1,3)):
            sliced=dict(x,**{key:x[key][:,start:stop] for key in ('hidden','attention_output','per_layer_input')})
            chunk=oracle.mathematics(sliced,2,stop-start,5,7,3,'full',True,True)
            for key in r:np.testing.assert_array_equal(chunk[key],r[key][:,start:stop])
        for key in x:np.testing.assert_array_equal(before[key],x[key])
        self.assertTrue(all(v.dtype==np.dtype('float32') for v in r.values()))

    def test_empty_and_width_one_shapes(self):
        for args in ((1,0,4,6,2,'full',True),(1,1,1,1,1,'full',True)):
            x=oracle.inputs(*args);r=oracle.mathematics(x,*args,False)
            self.assertEqual(r['output'].shape,(args[0],args[1],args[2]));self.assertTrue(np.isfinite(r['output']).all())

    def test_geometry_dtype_finite_and_inventory_reject(self):
        good=(1,2,4,6,2,'full',True)
        for i,values in {0:[0,3,True],1:[-1,4,False],2:[0,1537],3:[0,13],4:[0,257],5:[True,'moe'],6:[1,None]}.items():
            for value in values:
                args=list(good);args[i]=value
                with self.assertRaises(ValueError):oracle.inputs(*args)
        with self.assertRaises(ValueError):oracle.inputs(1,1,1536,12,256,'full',True)
        x=oracle.inputs(*good)
        for key in x:
            for replacement in (x[key].astype(np.float64),np.full_like(x[key],np.nan),np.zeros((1,),np.float32)):
                changed=dict(x);changed[key]=replacement
                # Scalar already has shape (1,), so test a different wrong shape.
                if key=='layer_scalar' and replacement.shape==(1,) and np.isfinite(replacement).all() and replacement.dtype==np.dtype('float32'):changed[key]=np.zeros((2,),np.float32)
                with self.assertRaises(ValueError):oracle.mathematics(changed,*good,False)
        del x['input_norm']
        with self.assertRaises(ValueError):oracle.mathematics(x,*good,False)

    def test_external_identity_and_strict_descriptor_metadata(self):
        raw=oracle.export()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'math.json';path.write_bytes(raw);self.assertEqual(len(oracle.verify(path,sha(raw))['cases']),12)
            with self.assertRaises(ValueError):oracle.verify(path,'0'*64)
            base=json.loads(raw);changes=[]
            for key,value in [('attention_kv_rotary_qualified',True),('scalar_absent_is_synthetic',False),('extra',1)]:
                item=copy.deepcopy(base);item[key]=value;changes.append(item)
            for key,value in [('with_scalar',1),('epsilon',.001),('hidden',True),('ple_mode','other')]:
                item=copy.deepcopy(base);item['cases'][1][key]=value;changes.append(item)
            for group,key,value in [('inputs','hidden',True),('inputs','input_norm',.1),('staged_fp32','output',.1),('math','output',True)]:
                item=copy.deepcopy(base);item['cases'][1][group][key]['values'][0]=value;changes.append(item)
            for item in changes:
                raw=oracle.encoded(item);path.write_bytes(raw)
                with self.assertRaises(ValueError):oracle.verify(path,sha(raw))

    def test_unique_selected_source_and_decorator_is_preserved(self):
        raw=b'@partial(mx.compile, shapeless=True)\ndef geglu(gate,x): return nn.gelu_approx(gate)*x\nclass MLP:\n def __call__(self,x): return x\nclass DecoderLayer:\n def __call__(self,x): return x'
        nodes=oracle.source_nodes({'gemma4_text.py':raw});self.assertEqual(set(nodes),{'geglu','MLP','DecoderLayer'});self.assertEqual(len(nodes['geglu'].decorator_list),1)
        self.assertEqual(ast.unparse(nodes['geglu'].decorator_list[0]),'partial(mx.compile, shapeless=True)')
        for bad in (b'pass',raw+b'\nclass MLP: pass',raw+b'\n def __call__(self,x): return x',raw+b'\ndef geglu(x): return x'):
            with self.assertRaises(ValueError):oracle.source_nodes({'gemma4_text.py':bad})

    def test_failed_source_retains_completed_separate_stage_probes(self):
        data=json.loads(oracle.export());data['cases']=[data['cases'][1]]
        core=types.ModuleType('mlx.core');core.__file__='mock-core';core.cpu,core.gpu='cpu','gpu';core.array,core.eval=np.array,lambda x:None;core.default_device,core.set_default_device=lambda:'cpu',lambda x:None;core.multiply=np.multiply
        class Linear:
            def __init__(self,*args,**kwargs):pass
            def __call__(self,x):return x@self.weight.T
        class Norm:
            def __init__(self,dims,eps):self.eps=eps
            def __call__(self,x):return norm(x,self.weight).astype(np.float32)
        nn=types.ModuleType('mlx.nn');nn.__file__='mock-nn';nn.Linear,nn.RMSNorm=Linear,Norm;nn.gelu_approx=lambda x:(.5*x*(1+np.tanh(math.sqrt(2/math.pi)*(x+.044715*x**3)))).astype(np.float32)
        mlx=types.ModuleType('mlx');mlx.core,mlx.nn=core,nn
        sources={'gemma4_text.py':b'def geglu(gate,x): return nn.gelu_approx(gate)*x\nclass MLP:\n def __call__(self,x): return self.down_proj(geglu(self.gate_proj(x),self.up_proj(x)))\nclass DecoderLayer:\n def __call__(self,*args,**kwargs): raise RuntimeError("synthetic decoder failure")'}
        admitted=dict(imports={'numpy':np.__file__,'mlx.core':core.__file__,'mlx.nn':nn.__file__})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dest=root/'out';dest.mkdir();reference=root/'math.json';reference.write_bytes(oracle.encoded(data))
            with mock.patch.object(oracle,'SOURCES',{k:sha(v) for k,v in sources.items()}),mock.patch.object(oracle,'verify',return_value=data),mock.patch.object(oracle.runtime,'admit_runtime',return_value=admitted),mock.patch.dict(sys.modules,{'mlx':mlx,'mlx.core':core,'mlx.nn':nn}):
                self.assertFalse(oracle.compare(reference,sha(reference.read_bytes()),sources,'cpu',dest,root/'runtime','0'*64))
            observed=json.loads((dest/'observation.json').read_bytes());self.assertIn('synthetic decoder failure',observed['primary_error'])
            row=observed['cases'][0];self.assertFalse(row['completed']);self.assertEqual(set(row['arrays']),set(data['cases'][0]['math'])|{'source_mlp_output'})
            item=row['arrays']['source_mlp_output'];raw=(dest/'observed.bin').read_bytes()[item['offset']:item['offset']+item['bytes']];self.assertEqual(sha(raw),item['sha256'])


if __name__=='__main__':unittest.main()
