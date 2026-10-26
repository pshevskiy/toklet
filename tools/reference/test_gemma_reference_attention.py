"""Frozen numerical fixtures and meaningful boundary/failure checks for FP64 BLAS."""
from pathlib import Path
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import gemma_reference_attention as candidate
import gemma_reference_attention_oracle as source
import gemma4_reference_graph as graph
import test_gemma4_text_math as old

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    directory=ROOT/'tests/fixtures/gemma-reference-attention-v1'
    meta=(directory/'manifest.json').read_bytes();raw=(directory/'arrays.bin').read_bytes()
    if hashlib.sha256(meta).hexdigest()!='36365f416c5f9572bfad7d6ff42c51760e43085025857648905350f86924edd8':
        raise ValueError('frozen attention fixture manifest differs')
    if hashlib.sha256(raw).hexdigest()!='f7a13f39536c816d566a5caf167b6f439f11b59b34ae206561b93d5ac28a1b7f':
        raise ValueError('frozen attention fixture binary differs')
    return json.loads(meta),raw


def inputs(meta,case,contract):
    result=[]
    for name in ('q','k','v'):
        item=next(v for v in meta['inputs']if(v['case'],v['contract'],v['name'])==(case['name'],contract,name))
        origin=case['query_start']if name=='q'else case['key_start']
        heads=case['heads']if name=='q'else 1;n=case['queries']if name=='q'else case['keys']
        values=np.array([source.recipe(case['name'],name,head,origin+position,d)
                         for head in range(heads)for position in range(n)for d in range(case['head_dim'])],dtype=item['dtype']).reshape(item['shape'])
        if hashlib.sha256(values.tobytes(order='C')).hexdigest()!=item['sha256']:
            raise ValueError('reconstructed attention input differs')
        result.append(values)
    return result


def expected(meta,raw,case,contract,name):
    item=next(v for v in meta['arrays']if(v['case'],v['contract'],v['name'])==(case['name'],contract,name))
    body=raw[item['offset']:item['offset']+item['bytes']]
    if hashlib.sha256(body).hexdigest()!=item['sha256']:raise ValueError('frozen attention output differs')
    return np.frombuffer(body,dtype=item['dtype']).reshape(item['shape'])


class ReferenceAttention(unittest.TestCase):
    def test_every_frozen_score_probability_context_and_context_only_path(self):
        meta,raw=fixture()
        for case in meta['cases']:
            for contract in ('math','staged_fp32'):
                q,k,v=inputs(meta,case,contract);before=[x.tobytes()for x in(q,k,v)]
                options=dict(staged=contract=='staged_fp32',window=case['window'],query_start=case['query_start'],key_start=case['key_start'])
                context,scores,probabilities=candidate.attention(q,k,v,diagnostics=True,**options)
                for name,value in [('context',context),('scores',scores),('probabilities',probabilities)]:
                    target=expected(meta,raw,case,contract,name)
                    self.assertEqual(value.dtype,target.dtype);old.gate(value,target)
                old.gate(candidate.attention(q,k,v,**options),expected(meta,raw,case,contract,'context'))
                self.assertEqual([x.tobytes()for x in(q,k,v)],before)

    def test_invalid_absolute_coverage_window_dtype_and_shapes_reject(self):
        q=np.ones((1,2,3,2));k=np.ones((1,1,7,2));v=k.copy()
        base=dict(staged=False,window=4,query_start=4,key_start=0)
        for changed in [dict(staged=1),dict(diagnostics=1),dict(window=0),dict(window=513),
                        dict(window=True),dict(query_start=True),dict(key_start=-1),dict(query_start=5),
                        dict(query_start=21,key_start=17,window=512),dict(query_start=5,key_start=1,window=None),
                        dict(query_start=1024,key_start=1020)]:
            with self.assertRaises(ValueError):candidate.attention(q,k,v,**(base|changed))
        for arrays in [(q[0],k,v),(q.astype('float32'),k,v),(q,k,v.astype('int32')),
                       (q,np.ones((1,2,7,2)),np.ones((1,2,7,2))),
                       (np.ones((2,2,3,2)),k,v),(np.ones((1,9,3,2)),k,v),
                       (q,np.ones((1,1,0,2)),np.ones((1,1,0,2))),
                       (np.ones((1,2,0,2)),k,v),(q,k,np.ones((1,1,7,3)))]:
            before=[x.tobytes()for x in arrays]
            with self.assertRaises(ValueError):candidate.attention(*arrays,**base)
            self.assertEqual([x.tobytes()for x in arrays],before)

    def test_nonfinite_dot_stage_cast_shift_and_output_are_explicit_failures(self):
        for staged,magnitude in [(False,1e308),(True,np.finfo('float32').max)]:
            dtype=np.float32 if staged else np.float64
            q=np.full((1,1,1,1),magnitude,dtype=dtype);k=q.copy();v=np.ones_like(q)
            with self.assertRaisesRegex(ValueError,'Nonfinite reference attention scores'):
                candidate.attention(q,k,v,staged=staged)
        q=np.ones((1,1,1,1));k=np.array([1e308,-1e308]).reshape(1,1,2,1);v=np.ones_like(k)
        with self.assertRaisesRegex(ValueError,'Nonfinite reference attention softmax shift'):
            candidate.attention(q,k,v,staged=False,query_start=1)
        # Computed masked columns are checked too. Visible-only scalar scores
        # would be finite for these two queries, while this tile overflows.
        q=np.array([1e308,0.]).reshape(1,1,2,1)
        k=np.array([1.,1e308]).reshape(1,1,2,1);v=np.ones_like(k)
        with self.assertRaisesRegex(ValueError,'Nonfinite reference attention scores'):
            candidate.attention(q,k,v,staged=False)
        q=np.zeros((1,1,1,1),dtype='float32');k=np.zeros((1,1,7,1),dtype='float32')
        v=np.full_like(k,np.finfo('float32').max)
        with self.assertRaisesRegex(ValueError,'Nonfinite reference attention context'):
            candidate.attention(q,k,v,staged=True,query_start=6)
        for bad in (float('nan'),float('inf')):
            q=np.full((1,1,1,1),bad)
            with self.assertRaisesRegex(ValueError,'Finite reference attention inputs'):
                candidate.attention(q,np.ones_like(q),np.ones_like(q),staged=False)

    def test_window_one_exact_key_selection_and_long_query_tiles(self):
        for staged,dtype in [(False,np.float64),(True,np.float32)]:
            q=np.ones((1,2,4,3),dtype=dtype);k=np.ones((1,1,7,3),dtype=dtype)
            v=np.arange(21,dtype=dtype).reshape(1,1,7,3)
            actual=candidate.attention(q,k,v,staged=staged,window=1,query_start=3)
            np.testing.assert_array_equal(actual,np.broadcast_to(v[0,0,3:7],(1,2,4,3)))
            q=np.zeros((1,1,1025,1),dtype=dtype);k=q.copy();v=np.ones_like(k)
            old.gate(candidate.attention(q,k,v,staged=staged),np.ones_like(q))
            with self.assertRaisesRegex(ValueError,'diagnostic query bound'):
                candidate.attention(q,k,v,staged=staged,diagnostics=True)

    def test_complete_short_graph_keeps_all_frozen_operator_states(self):
        weights,manifest,raw=old.fixture()
        for staged,contract in [(False,'math'),(True,'staged_fp32')]:
            states=graph.evaluate(weights,manifest['token_ids'],staged)
            self.assertEqual(set(states),set(manifest[contract]))
            for name,item in manifest[contract].items():
                target=np.frombuffer(raw[item['offset']:item['offset']+item['bytes']],dtype=item['dtype']).reshape(item['shape'])
                old.gate(states[name],target)

    def test_graph_bounds_and_sink_failure_remain_separate_from_reductions(self):
        weights,manifest,_=old.fixture()
        with mock.patch.object(weights,'gather',side_effect=AssertionError('unexpected numerical call')):
            for ids in ([],[0]*1026,[True],[-1],[17]):
                with self.assertRaises(ValueError):graph.evaluate(weights,ids,True,sink=lambda *_:None)
            with self.assertRaises(ValueError):graph.evaluate(weights,[0]*33,True)
        with self.assertRaises(ValueError):graph.evaluate(SimpleNamespace(profile=graph.WITNESS),[0],True)
        with mock.patch.object(weights,'gather',side_effect=AssertionError('unexpected real numerical call')):
            weights.profile=graph.E2B
            try:
                with self.assertRaises(ValueError):graph.evaluate(weights,[0]*5,True,sink=lambda *_:None)
            finally:weights.profile=graph.WITNESS
        before=weights.matrices['embed_tokens.weight'].stream.getvalue();seen=[]
        def fail(name,_):
            seen.append(name)
            if name=='layer.2.output':raise OSError('injected reference sink failure')
        with self.assertRaisesRegex(OSError,'injected reference sink failure'):
            graph.evaluate(weights,manifest['token_ids'],True,sink=fail)
        self.assertEqual(seen[-1],'layer.2.output');self.assertNotIn('logits',seen)
        self.assertEqual(weights.matrices['embed_tokens.weight'].stream.getvalue(),before)


if __name__=='__main__':unittest.main()
