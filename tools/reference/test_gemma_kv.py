"""Portable KV ownership/layout witnesses; no Apple runtime imported."""
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
import gemma_kv as oracle


def sha(raw):return hashlib.sha256(raw).hexdigest()


class KvWitness(unittest.TestCase):
    def test_ring_spans_attention_union_and_wrap_boundaries(self):
        data=json.loads(oracle.export())
        self.assertEqual(sum(len(c['steps']) for c in data['cases']),30)
        for case in data['cases']:
            for step in case['steps']:
                offset,after=step['before'],step['after'];window=case['window']
                old_positions=[step['positions_before'][slot] for first,count in step['old_spans'] for slot in range(first,first+count)]
                first=0 if window is None else max(0,offset-window+1)
                self.assertEqual(old_positions,list(range(first,offset)))
                self.assertEqual(old_positions+list(range(offset,after)),step['required_positions'])
                for q,(begin,end) in zip(range(offset,after),step['visible_ranges']):
                    self.assertEqual(list(range(begin,end)),[p for p in range(q+1) if window is None or q-p<window])
                expected=[None]*case['capacity']
                for p in range(after):
                    if window is None or p>=after-window:expected[p%case['capacity']]=p
                self.assertEqual(step['positions_after'],expected)
        wrapped=oracle.layout(5,3,3,3)
        self.assertEqual(wrapped['old_spans'],[[0,2]])
        self.assertEqual(wrapped['required_positions'],[3,4,5,6,7])
        self.assertEqual(oracle.layout(4,1,3,3)['old_spans'],[[2,1],[0,1]])
        self.assertEqual(oracle.layout(512,2,512,512)['positions_after'][:3],[512,513,2])

    def test_final_ring_is_insufficient_for_early_chunk_queries(self):
        step=oracle.layout(2,2,3,3)
        self.assertEqual(step['required_positions'],[0,1,2,3])
        self.assertEqual(step['positions_after'],[3,1,2])
        # Uniform attention at query 2 needs key 0, lost by premature overwrite.
        q=np.zeros((1,2,1),np.float32);k=np.zeros((1,4,1),np.float32)
        v=np.array([[[3.],[0.],[0.],[9.]]],np.float32)
        context=oracle.context(q,k,v,2,3)
        self.assertEqual(context[0,0,0],1.)
        self.assertEqual(context[0,1,0],3.)
        self.assertNotEqual(context[0,0,0],float(v[0,1:3].mean()))

    def test_abort_discards_candidate_and_preserves_exact_committed_state(self):
        # Required native policy witness; this does not execute upstream rollback.
        for case in json.loads(oracle.export())['cases']:
            for step in case['steps']:
                if not step['abort_expected']:continue
                before={key:copy.deepcopy(step[key]) for key in ('before','generation_before','positions_before')}
                for key in ('ring_before_k','ring_before_v'):before[key]=copy.deepcopy(step['arrays'][key])
                committed=copy.deepcopy(before)
                candidate=copy.deepcopy(step['arrays']['ring_after_k'])
                self.assertNotEqual(candidate,before['ring_before_k'])
                del candidate
                self.assertEqual(committed,before)
                # Retry is identical to the originally proposed state, not a second append.
                self.assertEqual(oracle.layout(committed['before'],step['count'],case['window'],case['capacity'])['positions_after'],step['positions_after'])

    def test_logical_chunk_equivalence_gqa_and_immutable_inputs(self):
        q=np.zeros((4,7,2),np.float32);k=np.zeros((2,7,2),np.float32)
        v=np.array([[[p,p+1] for p in range(7)],[[p+10,p+11] for p in range(7)]],np.float32)
        originals=[a.copy() for a in (q,k,v)]
        full=oracle.context(q,k,v,0,3)
        for chunks in ((1,1,1,1,1,1,1),(2,2,3),(5,2)):
            pieces=[];offset=0
            for n in chunks:
                pieces.append(oracle.context(q[:,offset:offset+n],k[:,:offset+n],v[:,:offset+n],offset,3));offset+=n
            np.testing.assert_array_equal(np.concatenate(pieces,axis=1),full)
        np.testing.assert_array_equal(full[0],full[1]);np.testing.assert_array_equal(full[2],full[3])
        np.testing.assert_allclose(full[2]-full[0],10.,rtol=0.,atol=2e-15)
        for a,b in zip((q,k,v),originals):np.testing.assert_array_equal(a,b)

    def test_invalid_geometry_dtype_values_and_external_identity(self):
        for args in ((-1,1,3,3),(0,0,3,3),(0,True,3,3),(0,1,True,1),(0,1,3,4),(516,1,None,516),(2,2,None,3),(0,512,512,512)):
            with self.assertRaises(ValueError):oracle.layout(*args)
        q=np.ones((2,1,2),np.float32);k=np.ones((1,2,2),np.float32);v=k.copy()
        for index in range(3):
            for invalid in (np.nan,np.inf,-np.inf):
                args=[q.copy(),k.copy(),v.copy()];args[index].reshape(-1)[-1]=invalid
                with self.assertRaises(ValueError):oracle.context(*args,1,3)
            args=[q,k,v];args[index]=args[index].astype(np.float16)
            with self.assertRaises(ValueError):oracle.context(*args,1,3)
        with self.assertRaises(ValueError):oracle.context(q,k,v,True,3)
        raw=oracle.export()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'math.json';path.write_bytes(raw)
            self.assertEqual(len(oracle.verify(path,sha(raw))['cases']),7)
            with self.assertRaises(ValueError):oracle.verify(path,'0'*64)
            base=json.loads(raw);mutations=[]
            for key,value in [('device_qualified',True),('extra',1),('runtime',dict(numpy='other'))]:
                changed=copy.deepcopy(base);changed[key]=value;mutations.append(changed)
            for key,value in [('capacity',True),('window',False),('abort_step',False)]:
                changed=copy.deepcopy(base);changed['cases'][1][key]=value;mutations.append(changed)
            changed=copy.deepcopy(base);changed['cases'][1]['steps'][0]['positions_after'][0]=False;mutations.append(changed)
            for key,value in [('new_k',.1),('context',True),('q',10**400)]:
                changed=copy.deepcopy(base);changed['cases'][1]['steps'][0]['arrays'][key]['values'][0]=value;mutations.append(changed)
            for changed in mutations:
                raw=oracle.encoded(changed);path.write_bytes(raw)
                with self.assertRaises(ValueError):oracle.verify(path,sha(raw))
            alias=path.parent/'alias';alias.symlink_to(path)
            with self.assertRaises(ValueError):oracle.verify(alias,sha(raw))

    def test_unique_selected_source_bodies_and_deferred_annotations(self):
        sources={'base.py':b'def create_causal_mask(): pass\ndef scaled_dot_product_attention(): pass',
            'cache.py':b'def create_attention_mask(): pass\nclass _BaseCache: pass\nclass KVCache: pass\nclass RotatingKVCache: pass'}
        self.assertEqual(len(oracle.source_nodes(sources)),6)
        for key in sources:
            changed=dict(sources);changed[key]+=b'\n'+sources[key]
            with self.assertRaises(ValueError):oracle.source_nodes(changed)
            changed=dict(sources);changed[key]=b'pass'
            with self.assertRaises(ValueError):oracle.source_nodes(changed)

    def test_failed_update_retains_completed_mask_and_restores_device(self):
        sources={'base.py':b'def create_causal_mask(): pass\ndef scaled_dot_product_attention(): pass',
            'cache.py':b'def create_attention_mask(): pass\nclass _BaseCache: pass\nclass KVCache: pass\nclass RotatingKVCache:\n def __init__(self,**kw): self.offset=0; self.keys=None; self.values=None\n def size(self): return 0\n def make_mask(self,*args,**kw)->MissingAnnotation: return mx.mask\n def update_and_fetch(self,*args): raise RuntimeError("synthetic KV update failure")'}
        identities={name:sha(raw) for name,raw in sources.items()};data=json.loads(oracle.export());data['cases']=[data['cases'][2]]
        core=types.ModuleType('mlx.core');core.__file__='mock-core';core.cpu,core.gpu='cpu','gpu'
        core.array,core.eval=np.array,lambda value:None;restored=[]
        core.default_device,core.set_default_device=lambda:'previous',restored.append
        core.mask=np.tril(np.ones((5,5),np.bool_))
        nn=types.ModuleType('mlx.nn');nn.__file__='mock-nn';mlx=types.ModuleType('mlx');mlx.core,mlx.nn=core,nn
        admitted=dict(imports={'numpy':np.__file__,'mlx.core':core.__file__,'mlx.nn':nn.__file__})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dest=root/'out';dest.mkdir();reference=root/'math.json';reference.write_bytes(oracle.encoded(data))
            with mock.patch.object(oracle,'SOURCES',identities),mock.patch.object(oracle,'verify',return_value=data),mock.patch.object(oracle.runtime,'admit_runtime',return_value=admitted),mock.patch.dict(sys.modules,{'mlx':mlx,'mlx.core':core,'mlx.nn':nn}):
                self.assertFalse(oracle.compare(reference,sha(reference.read_bytes()),sources,'cpu',dest,root/'runtime','0'*64))
            result=json.loads((dest/'observation.json').read_bytes());self.assertIn('synthetic KV update failure',result['primary_error'])
            item=result['cases'][0]['steps'][0]['arrays']['mask'];raw=(dest/'observed.bin').read_bytes()[item['offset']:item['offset']+item['bytes']]
            self.assertEqual(sha(raw),item['sha256']);self.assertEqual(raw,core.mask.tobytes())
            self.assertFalse(result['rollback_qualified']);self.assertEqual(restored,['cpu','previous'])


if __name__=='__main__':unittest.main()
