"""Portable causal/sliding GQA witnesses; no Apple runtime imported."""
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

import gemma_local_attention as oracle


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class LocalAttention(unittest.TestCase):
    def test_hand_scale_one_and_grouped_head_mapping(self):
        q=np.array([[[1.,1.]]],np.float32)
        k=np.array([[[0.,0.],[1.,1.]]],np.float32)
        v=np.array([[[1.,0.],[0.,1.]]],np.float32)
        before=[x.copy() for x in (q,k,v)]
        r=oracle.math_values(q,k,v,1,2,1,1,2,2,0,1.)
        np.testing.assert_allclose(r['output'],[[[0.11920292202211755,0.8807970779778823]]],rtol=1e-14,atol=1e-14)
        self.assertGreater(abs(r['output'][0,0,0]-.19557031749304313),.07)
        for a,b in zip((q,k,v),before):np.testing.assert_array_equal(a,b)
        q=np.zeros((4,1,1),np.float32);k=np.zeros((2,3,1),np.float32)
        v=np.array([[[1.],[3.],[5.]],[[11.],[13.],[15.]]],np.float32)
        r=oracle.math_values(q,k,v,1,3,4,2,1,2,0,1.)
        np.testing.assert_array_equal(r['mask'],[[False,True,True]])
        np.testing.assert_array_equal(r['output'][:,0,0],[4.,4.,14.,14.])

    def test_future_and_expired_keys_cannot_affect_context(self):
        q=np.ones((2,5,4),np.float32)
        k=np.arange(20,dtype=np.float32).reshape(1,5,4)*np.float32(.125)
        v=k.copy();r=oracle.math_values(q,k,v,5,5,2,1,4,3,0,1.)
        altered_k,altered_v=k.copy(),v.copy();altered_k[:,4]+=20;altered_v[:,4]-=30
        changed=oracle.math_values(q,altered_k,altered_v,5,5,2,1,4,3,0,1.)
        np.testing.assert_array_equal(changed['output'][:,:4],r['output'][:,:4])
        altered_k,altered_v=k.copy(),v.copy();altered_k[:,0]+=20;altered_v[:,0]-=30
        changed=oracle.math_values(q,altered_k,altered_v,5,5,2,1,4,3,0,1.)
        np.testing.assert_array_equal(changed['output'][:,3:],r['output'][:,3:])
        self.assertTrue((r['probabilities'][:,~r['mask']]==0).all())
        np.testing.assert_allclose(r['probabilities'].sum(-1),1.,rtol=1e-14,atol=1e-14)

    def test_chunk_offsets_origin_translation_and_every_fixture_mask(self):
        q=np.arange(4*5*3,dtype=np.float32).reshape(4,5,3)*np.float32(.125)
        k=np.arange(2*5*3,dtype=np.float32).reshape(2,5,3)*np.float32(.125)
        v=(k-2).copy();full=oracle.math_values(q,k,v,5,5,4,2,3,3,0,1.)
        for chunk in (1,2,3):
            pieces=[]
            for first in range(0,5,chunk):
                end=min(5,first+chunk)
                pieces.append(oracle.math_values(q[:,first:end],k[:,:end],v[:,:end],end-first,end,4,2,3,3,0,1.)['output'])
            np.testing.assert_array_equal(np.concatenate(pieces,axis=1),full['output'])
        shifted=oracle.math_values(q,k,v,5,5,4,2,3,3,17,1.)
        np.testing.assert_array_equal(shifted['mask'],full['mask']);np.testing.assert_array_equal(shifted['output'],full['output'])
        for row in json.loads(oracle.export())['cases']:
            n,k=row['queries'],row['keys'];mask=np.array(row['arrays']['mask']['values'],bool).reshape(n,k)
            for i in range(n):
                for j in range(k):
                    expected=j<=k-n+i and (row['window'] is None or j>k-n+i-row['window'])
                    self.assertEqual(bool(mask[i,j]),expected)

    def test_invalid_geometry_shape_dtype_and_finiteness_reject(self):
        params=[1,2,2,1,4,3,0,1.]
        q=np.ones((2,1,4),np.float32);k=np.ones((1,2,4),np.float32);v=k.copy()
        for index,values in {0:[-1,3,True],1:[0,9],2:[0,3],3:[0,3],4:[0,513],5:[0,513,True],6:[-1,33,True],7:[0.,float('nan'),float('inf'),.1,True,10**400]}.items():
            for bad in values:
                args=list(params);args[index]=bad
                with self.assertRaises(ValueError):oracle.math_values(q,k,v,*args)
        for tensor in range(3):
            for bad in (np.nan,np.inf,-np.inf):
                arrays=[q.copy(),k.copy(),v.copy()];arrays[tensor].reshape(-1)[-1]=bad
                with self.assertRaises(ValueError):oracle.math_values(*arrays,*params)
            for bad in ([q,k,v][tensor].astype(np.float16),np.ones((1,1,1),np.float32)):
                arrays=[q,k,v];arrays[tensor]=bad
                with self.assertRaises(ValueError):oracle.math_values(*arrays,*params)

    def test_bounded_external_format_and_boolean_masks(self):
        raw=oracle.export()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'math.json';path.write_bytes(raw)
            self.assertEqual(len(oracle.verify(path,sha(raw))['cases']),11)
            with self.assertRaises(ValueError):oracle.verify(path,'0'*64)
            base=json.loads(raw);mutations=[]
            for key,value in [('device_qualified',True),('extra',1),('runtime',dict(numpy='other'))]:
                changed=copy.deepcopy(base);changed[key]=value;mutations.append(changed)
            for name,value in [('queries',True),('window',0),('origin',-1)]:
                changed=copy.deepcopy(base);changed['cases'][1][name]=value;mutations.append(changed)
            for key,value in [('mask',1),('q',.1),('output',True),('q',10**400)]:
                changed=copy.deepcopy(base);changed['cases'][1]['arrays'][key]['values'][0]=value;mutations.append(changed)
            for changed in mutations:
                raw=oracle.encoded(changed);path.write_bytes(raw)
                with self.assertRaises(ValueError):oracle.verify(path,sha(raw))
            alias=path.parent/'alias';alias.symlink_to(path)
            with self.assertRaises(ValueError):oracle.verify(alias,sha(raw))

    def test_selected_source_definitions_are_unique(self):
        good={'base.py':b'def create_causal_mask(): pass\ndef scaled_dot_product_attention(): pass'}
        self.assertEqual(set(oracle.source_nodes(good)),{'create_causal_mask','scaled_dot_product_attention'})
        for raw in (b'pass',good['base.py']+b'\ndef create_causal_mask():pass'):
            with self.assertRaises(ValueError):oracle.source_nodes({'base.py':raw})

    def test_context_failure_retains_completed_mask(self):
        sources={'base.py':b'def create_causal_mask(*args,**kwargs): return mx.mask\ndef scaled_dot_product_attention(*args,**kwargs): raise RuntimeError("synthetic attention failure")'}
        identities={name:sha(raw) for name,raw in sources.items()};data=json.loads(oracle.export())
        core=types.ModuleType('mlx.core');core.__file__='mock-core';core.cpu,core.gpu='cpu','gpu'
        core.array,core.eval=np.array,lambda value:None;core.default_device,core.set_default_device=lambda:'cpu',lambda device:None
        core.mask=np.ones((1,1),np.bool_)
        nn=types.ModuleType('mlx.nn');nn.__file__='mock-nn'
        mlx=types.ModuleType('mlx');mlx.core,mlx.nn=core,nn
        admitted=dict(imports={'numpy':np.__file__,'mlx.core':core.__file__,'mlx.nn':nn.__file__})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);destination=root/'out';destination.mkdir();reference=root/'math.json';reference.write_bytes(oracle.encoded(data))
            with mock.patch.object(oracle,'SOURCES',identities),mock.patch.object(oracle,'verify',return_value=data),mock.patch.object(oracle.runtime,'admit_runtime',return_value=admitted),mock.patch.dict(sys.modules,{'mlx':mlx,'mlx.core':core,'mlx.nn':nn}):
                self.assertFalse(oracle.compare(reference,sha(reference.read_bytes()),sources,'cpu',destination,root/'runtime','0'*64))
            result=json.loads((destination/'observation.json').read_bytes());self.assertIn('synthetic attention failure',result['primary_error'])
            item=result['cases'][1]['arrays']['mask'];raw=(destination/'observed.bin').read_bytes()[item['offset']:item['offset']+item['bytes']]
            self.assertEqual(sha(raw),item['sha256']);self.assertTrue(result['comparisons'][0]['passed'])


if __name__=='__main__':unittest.main()
