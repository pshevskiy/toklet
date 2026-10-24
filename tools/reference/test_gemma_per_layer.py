"""Portable per-layer scale, axis and rounding witnesses; no Apple runtime."""
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
import gemma_per_layer as oracle


def sha(raw):return hashlib.sha256(raw).hexdigest()


class PerLayer(unittest.TestCase):
    def test_hand_lookup_scale_axis_and_direct_learned_norm(self):
        x=dict(ids=np.array([[1]],np.int32),raw_embedding=np.array([[[1.,0.,0.,0.]]],np.float32),table=np.zeros((4,4),np.float32),projection_weight=np.array([[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[2.,0.,0.,0.]],np.float32),norm_weight=np.array([.75,-.5],np.float32));x['table'][1]=[1.,2.,3.,4.]
        result=oracle.mathematics(x,1,1,4,2,2,True,False)
        np.testing.assert_array_equal(result['scaled_embedding'],[[[2.,0.,0.,0.]]]);np.testing.assert_array_equal(result['projection_scaled'],[[[1.,0.,0.,2.]]])
        expected=np.array([[[[.75/np.sqrt(.5+oracle.EPSILON),0.],[0.,-1/np.sqrt(2+oracle.EPSILON)]]]])
        np.testing.assert_allclose(result['projection_norm'],expected,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(result['lookup'].reshape(-1),np.array([1.,2.,3.,4.])*np.sqrt(2),rtol=1e-14)
        np.testing.assert_allclose(result['output'],(expected+result['lookup'])*np.sqrt(.5),rtol=1e-14,atol=1e-14)
        self.assertGreater(abs(result['projection_norm'][0,0,0,0]-.75/np.sqrt(1.25+oracle.EPSILON)),.3)
        self.assertLess(result['projection_norm'][0,0,1,1],0.)

    def test_staged_fp32_retains_scale_boundaries_without_algebraic_cancellation(self):
        x=oracle.inputs(2,3,5,3,3,True);r=oracle.mathematics(x,2,3,5,3,3,True,True)
        np.testing.assert_array_equal(r['scaled_embedding'],(x['raw_embedding'].astype(np.float64)*np.float32(np.sqrt(5))).astype(np.float32))
        projected=(r['scaled_embedding'].astype(np.float64)@x['projection_weight'].astype(np.float64).T).astype(np.float32)
        np.testing.assert_array_equal(r['projection'],projected)
        np.testing.assert_array_equal(r['projection_scaled'],(projected.astype(np.float64)*np.float32(5**-.5)).astype(np.float32))
        cancelled=(x['raw_embedding'].astype(np.float64)@x['projection_weight'].astype(np.float64).T).astype(np.float32)
        self.assertTrue(np.any(r['projection_scaled'].view(np.uint32)!=cancelled.view(np.uint32)))
        for value in r.values():self.assertEqual(value.dtype,np.dtype('float32'))

    def test_chunking_immutable_operands_projected_only_and_empty(self):
        x=oracle.inputs(2,3,5,3,3,True);before={k:v.copy() for k,v in x.items()}
        full=oracle.mathematics(x,2,3,5,3,3,True,False)
        for start,stop in [(0,1),(1,3)]:
            sliced=dict(x,ids=x['ids'][:,start:stop],raw_embedding=x['raw_embedding'][:,start:stop])
            chunk=oracle.mathematics(sliced,2,stop-start,5,3,3,True,False)
            for key in full:np.testing.assert_allclose(chunk[key],full[key][:,start:stop],rtol=1e-13,atol=1e-13)
        only=oracle.mathematics(x,2,3,5,3,3,False,False);np.testing.assert_array_equal(only['output'],full['projection_norm']);self.assertNotIn('combined',only)
        for key,value in before.items():np.testing.assert_array_equal(x[key],value)
        x=oracle.inputs(1,0,4,3,2,True);self.assertEqual(oracle.mathematics(x,1,0,4,3,2,True,False)['output'].shape,(1,0,3,2))

    def test_invalid_geometry_operand_dtype_id_and_finiteness_reject(self):
        good=(1,2,4,2,2,True)
        for i,values in {0:[0,3,True],1:[-1,4,False],2:[0,1537],3:[0,36],4:[0,257],5:[1,None]}.items():
            for value in values:
                args=list(good);args[i]=value
                with self.assertRaises(ValueError):oracle.inputs(*args)
        with self.assertRaises(ValueError):oracle.inputs(1,1,1536,35,256,True)
        x=oracle.inputs(*good)
        for key in x:
            changed=dict(x);changed[key]=changed[key].astype(np.float64)
            with self.assertRaises(ValueError):oracle.mathematics(changed,*good,False)
        for value in [-1,4]:
            changed=copy.deepcopy(x);changed['ids'][0,0]=value
            with self.assertRaises(ValueError):oracle.mathematics(changed,*good,False)
        for key in ('raw_embedding','table','projection_weight','norm_weight'):
            changed=copy.deepcopy(x);changed[key].reshape(-1)[-1]=np.nan
            with self.assertRaises(ValueError):oracle.mathematics(changed,*good,False)

    def test_bounded_external_identity_and_exact_metadata_dtypes(self):
        raw=oracle.export()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'math.json';path.write_bytes(raw);self.assertEqual(len(oracle.verify(path,sha(raw))['cases']),7)
            with self.assertRaises(ValueError):oracle.verify(path,'0'*64)
            base=json.loads(raw);changes=[]
            for key,value in [('bf16_checkpoint_qualified',True),('extra',1)]:
                item=copy.deepcopy(base);item[key]=value;changes.append(item)
            for key,value in [('with_lookup',1),('epsilon',.001),('layers',True)]:
                item=copy.deepcopy(base);item['cases'][1][key]=value;changes.append(item)
            for group,key,value in [('inputs','ids',True),('inputs','raw_embedding',.1),('staged_fp32','output',.1),('math','output',True)]:
                item=copy.deepcopy(base);item['cases'][1][group][key]['values'][0]=value;changes.append(item)
            for item in changes:
                raw=oracle.encoded(item);path.write_bytes(raw)
                with self.assertRaises(ValueError):oracle.verify(path,sha(raw))

    def test_unique_selected_helpers(self):
        raw=b'class Gemma4TextModel:\n def _get_per_layer_inputs(self): pass\n def _project_per_layer_inputs(self): pass'
        self.assertEqual(set(oracle.source_nodes({'gemma4_text.py':raw})),{'_get_per_layer_inputs','_project_per_layer_inputs'})
        for bad in [b'pass',raw+b'\nclass Gemma4TextModel: pass',raw+b'\n def _get_per_layer_inputs(self): pass']:
            with self.assertRaises(ValueError):oracle.source_nodes({'gemma4_text.py':bad})

    def test_failed_helper_retains_completed_separate_stage_probes(self):
        data=json.loads(oracle.export());data['cases']=[data['cases'][1]]
        core=types.ModuleType('mlx.core');core.__file__='mock-core';core.cpu,core.gpu='cpu','gpu';core.array,core.eval=np.array,lambda x:None;core.default_device,core.set_default_device=lambda:'cpu',lambda x:None
        core.unflatten=lambda x,axis,shape:x.reshape((*x.shape[:axis],*shape))
        case=data['cases'][0];core.lookup=np.array(case['staged_fp32']['lookup']['values'],np.float32).reshape(case['staged_fp32']['lookup']['shape'])
        class Embedding:
            def __init__(self,*args):pass
            def __call__(self,ids):return self.weight[ids]
        class Linear:
            def __init__(self,*args,**kwargs):pass
            def __call__(self,x):return x@self.weight.T
        class Norm:
            def __init__(self,dims,eps):self.eps=eps
            def __call__(self,x):return (x.astype(np.float64)/np.sqrt(np.mean(x.astype(np.float64)**2,-1,keepdims=True)+self.eps)*self.weight).astype(np.float32)
        nn=types.ModuleType('mlx.nn');nn.__file__='mock-nn';nn.Embedding,nn.Linear,nn.RMSNorm=Embedding,Linear,Norm
        mlx=types.ModuleType('mlx');mlx.core,mlx.nn=core,nn
        sources={'gemma4_text.py':b'class Gemma4TextModel:\n def _get_per_layer_inputs(self,*args):return mx.lookup\n def _project_per_layer_inputs(self,*args):raise RuntimeError("synthetic per-layer helper failure")'}
        admitted=dict(imports={'numpy':np.__file__,'mlx.core':core.__file__,'mlx.nn':nn.__file__})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dest=root/'out';dest.mkdir();reference=root/'math.json';reference.write_bytes(oracle.encoded(data))
            with mock.patch.object(oracle,'SOURCES',{k:sha(v) for k,v in sources.items()}),mock.patch.object(oracle,'verify',return_value=data),mock.patch.object(oracle.runtime,'admit_runtime',return_value=admitted),mock.patch.dict(sys.modules,{'mlx':mlx,'mlx.core':core,'mlx.nn':nn}):
                self.assertFalse(oracle.compare(reference,sha(reference.read_bytes()),sources,'cpu',dest,root/'runtime','0'*64))
            observed=json.loads((dest/'observation.json').read_bytes());self.assertIn('synthetic per-layer helper failure',observed['primary_error'])
            item=observed['cases'][0]['arrays']['source_lookup'];raw=(dest/'observed.bin').read_bytes()[item['offset']:item['offset']+item['bytes']];self.assertEqual(sha(raw),item['sha256']);self.assertEqual(len(observed['cases'][0]['arrays']),11)


if __name__=='__main__':unittest.main()
