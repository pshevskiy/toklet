"""Portable complete-graph math, causality and evidence-boundary checks."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import gemma_text_graph as graph


class TextGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights=graph.operands();cls.ids=graph.token_ids()
        cls.ideal=graph.mathematics(cls.weights,cls.ids,False)
        cls.staged=graph.mathematics(cls.weights,cls.ids,True)

    def test_inventory_finite_and_concrete_geometry(self):
        for result in (self.ideal,self.staged):
            self.assertEqual({k:list(v.shape) for k,v in result.items()},graph.stage_shapes())
            self.assertTrue(all(np.isfinite(v).all() for v in result.values()))
        self.assertTrue(all(v.dtype==np.dtype('float32') for v in self.staged.values()))
        self.assertTrue(all(v.dtype==np.dtype('float64') for v in self.ideal.values()))
        self.assertEqual(graph.weight_shapes(),{k:list(v.shape) for k,v in self.weights.items()})
        for i in (2,3):self.assertFalse(any(f'model.layers.{i}.self_attn.{key}.' in name for name in self.weights for key in ('k_proj','v_proj','k_norm')))
        self.assertEqual(self.weights['model.layers.0.mlp.gate_proj.weight'].shape,(12,8))
        self.assertEqual(self.weights['model.layers.2.mlp.gate_proj.weight'].shape,(24,8))
        for _,chunks in graph.SCHEDULES:self.assertEqual(sum(chunks),32)
        self.assertEqual(len(set(self.ids)),len(set(graph.token_ids())))

    def test_embedding_scaling_ple_and_tied_output_boundary(self):
        weights=self.weights;r=self.ideal
        np.testing.assert_array_equal(r['embedding'],weights['model.embed_tokens.weight'][self.ids][None])
        np.testing.assert_allclose(r['scaled_embedding'],r['embedding']*math.sqrt(8),rtol=1e-14,atol=1e-14)
        projection=r['scaled_embedding']@weights['model.per_layer_model_projection.weight'].astype(np.float64).T
        np.testing.assert_allclose(r['ple_projection'],projection,rtol=1e-14,atol=1e-14)
        projection=(projection/math.sqrt(8)).reshape(1,32,4,2)
        eps=float(np.float32(1e-6))
        norm=projection/np.sqrt(np.mean(projection**2,-1,keepdims=True)+eps)*weights['model.per_layer_projection_norm.weight']
        np.testing.assert_allclose(r['ple_norm'],norm,rtol=1e-13,atol=1e-13)
        ple=(norm+r['ple_lookup'].reshape(1,32,4,2)*math.sqrt(2))/math.sqrt(2)
        for i in range(4):np.testing.assert_allclose(r[f'layer.{i}.ple_input'],ple[:,:,i],rtol=1e-13,atol=1e-13)
        expected=r['final_norm']@weights['model.embed_tokens.weight'].astype(np.float64).T
        np.testing.assert_allclose(r['head'],expected,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(r['logits'],30*np.tanh(expected/30),rtol=1e-13,atol=1e-13)
        self.assertLessEqual(np.max(abs(r['logits'])),30.)
        np.testing.assert_array_equal(self.staged['scaled_embedding'],(self.staged['embedding'].astype(np.float64)*float(np.float32(math.sqrt(8)))).astype(np.float32))
        self.assertGreater(np.max(abs(r['head']-r['logits'])),1e-6)

    def test_direct_norm_and_shared_gqa_local_visibility(self):
        r=self.ideal;eps=float(np.float32(1e-6))
        x=r['layer.0.input'];w=self.weights['model.layers.0.input_layernorm.weight'].astype(np.float64)
        norm=x/np.sqrt(np.mean(x*x,-1,keepdims=True)+eps)
        np.testing.assert_allclose(r['layer.0.input_layernorm'],norm*w,rtol=1e-13,atol=1e-13)
        self.assertGreater(np.max(abs(r['layer.0.input_layernorm']-norm*(1+w))),.1)
        for i in range(4):
            producer=i if i<2 else i-2
            q=r[f'layer.{i}.q_rotary'];k=r[f'layer.{producer}.k_rotary'];v=r[f'layer.{producer}.self_attn.v_norm'].transpose(0,2,1,3)
            for pos in (0,1,3,4,7,15,31):
                start=max(0,pos-3) if i%2==0 else 0
                for head in (0,1):
                    scores=q[0,head,pos]@k[0,0,start:pos+1].T
                    probs=np.exp(scores-scores.max());probs/=probs.sum()
                    expected=probs@v[0,0,start:pos+1]
                    np.testing.assert_allclose(r[f'layer.{i}.context'][0,head,pos],expected,rtol=1e-12,atol=1e-12)
        pos=31;q=r['layer.2.q_rotary'][0,1,pos];k=r['layer.0.k_rotary'][0,0];v=r['layer.0.self_attn.v_norm'][0,:,0]
        scores=q@k.T;probs=np.exp(scores-scores.max());probs/=probs.sum()
        self.assertGreater(np.max(abs(r['layer.2.context'][0,1,pos]-probs@v)),.01)
        # Full attention uses scale 1, not the usual inverse square root.
        q=r['layer.3.q_rotary'][0,0,pos];k=r['layer.1.k_rotary'][0,0];v=r['layer.1.self_attn.v_norm'][0,:,0]
        scores=q@k.T/4;probs=np.exp(scores-scores.max());probs/=probs.sum()
        self.assertGreater(np.max(abs(r['layer.3.context'][0,0,pos]-probs@v)),.01)

    def test_future_tokens_do_not_change_any_past_stage(self):
        changed=self.ids[:20]+[(x+1)%17 for x in self.ids[20:]]
        result=graph.mathematics(self.weights,changed,True)
        for name,expected in self.staged.items():
            axis=2 if name.endswith(('q_rotary','k_rotary','context')) else 1
            selection=[slice(None)]*expected.ndim;selection[axis]=slice(0,20)
            np.testing.assert_array_equal(result[name][tuple(selection)],expected[tuple(selection)],err_msg=name)
        self.assertGreater(np.max(abs(result['logits'][:,20:]-self.staged['logits'][:,20:])),.01)
        for name,value in self.weights.items():np.testing.assert_array_equal(value,graph.operands()[name])

    def test_rotary_half_pair_inactive_tail_and_absolute_positions(self):
        r=self.staged
        for i in range(4):
            q=r[f'layer.{i}.self_attn.q_norm'].transpose(0,2,1,3)
            rotated=r[f'layer.{i}.q_rotary'];dims=q.shape[-1];pairs=dims//2 if i%2==0 else 2
            for pair in range(pairs):
                for pos in (0,1,7,15,31):
                    freq=(1e4 if i%2==0 else 1e6)**(2*pair/dims)
                    c,s=float(np.float32(math.cos(pos/freq))),float(np.float32(math.sin(pos/freq)))
                    a,b=q[:,:,pos,pair].astype(np.float64),q[:,:,pos,pair+dims//2].astype(np.float64)
                    np.testing.assert_array_equal(rotated[:,:,pos,pair],((a*c).astype(np.float32).astype(np.float64)-(b*s).astype(np.float32).astype(np.float64)).astype(np.float32))
                    np.testing.assert_array_equal(rotated[:,:,pos,pair+dims//2],((a*s).astype(np.float32).astype(np.float64)+(b*c).astype(np.float32).astype(np.float64)).astype(np.float32))
            if i%2:
                np.testing.assert_array_equal(rotated[:,:,:,2:8],q[:,:,:,2:8])
                np.testing.assert_array_equal(rotated[:,:,:,10:16],q[:,:,:,10:16])

    def test_exact_inputs_validation(self):
        for ids in (self.ids[:-1],[True]+self.ids[1:],[17]+self.ids[1:],tuple(self.ids)):
            with self.assertRaises(ValueError):graph.mathematics(self.weights,ids,True)
        for bad in (1,None):
            with self.assertRaises(ValueError):graph.mathematics(self.weights,self.ids,bad)
        for replacement in (np.array([1.],np.float32),self.weights['model.norm.weight'].astype(np.float64),np.full((8,),np.nan,np.float32)):
            with self.assertRaises(ValueError):graph.mathematics(dict(self.weights,**{'model.norm.weight':replacement}),self.ids,True)

    def test_external_identity_strict_inventory_and_numbers(self):
        raw=graph.export();base=json.loads(raw)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'math.json';path.write_bytes(raw);self.assertEqual(graph.verify(path,hashlib.sha256(raw).hexdigest())['token_ids'],self.ids)
            with self.assertRaises(ValueError):graph.verify(path,'0'*64)
            changes=[]
            for key,value in [('native_graph_implemented',True),('upstream_long_rotary_gate_closed',True),('extra',1)]:
                item=copy.deepcopy(base);item[key]=value;changes.append(item)
            item=copy.deepcopy(base);item['config']['max_position_embeddings']=131072;changes.append(item)
            item=copy.deepcopy(base);item['token_ids'][0]=True;changes.append(item)
            for group in ('weights','math','staged_fp32'):
                key=next(iter(base[group]));item=copy.deepcopy(base);del item[group][key];changes.append(item)
                for value in (True,10**400):
                    item=copy.deepcopy(base);item[group][key]['values'][0]=value;changes.append(item)
                item=copy.deepcopy(base);item[group][key]['shape'][0]=True;changes.append(item)
            for item in changes:
                raw=graph.encoded(item);path.write_bytes(raw)
                with self.assertRaises(ValueError):graph.verify(path,hashlib.sha256(raw).hexdigest())

    def test_unique_original_source_selection_and_decorators(self):
        sources={}
        names={
            'base.py':['create_causal_mask','create_attention_mask','scaled_dot_product_attention'],
            'cache.py':['create_attention_mask','_BaseCache','KVCache','RotatingKVCache'],
            'rope_utils.py':['ProportionalRoPE','initialize_rope'],
            'gemma4_text.py':['RMSNormNoScale','logit_softcap','geglu','MLP','Attention','DecoderLayer','Gemma4TextModel','Model']}
        for filename,selected in names.items():
            sources[filename]='\n'.join(('class '+name+':\n pass' if name[0].isupper() or name.startswith('_') else ('@partial(mx.compile, shapeless=True)\n' if name in ('geglu','logit_softcap') else '')+'def '+name+'(*args): return args') for name in selected).encode()
        result=graph.source_nodes(sources)
        for node in result['gemma4_text.py']:
            if node.name in ('geglu','logit_softcap'):self.assertEqual(ast.unparse(node.decorator_list[0]),'partial(mx.compile, shapeless=True)')
        for filename in names:
            bad=dict(sources);bad[filename]+=b'\n'+sources[filename]
            with self.assertRaises(ValueError):graph.source_nodes(bad)

    def test_fixed_observation_inventory_fits_deliberate_report_bound(self):
        rows=[];comparisons=[];arrays=0
        for traced in (False,True):
            for schedule,chunks in graph.SCHEDULES:
                before=0
                for index,count in enumerate(chunks):
                    row=dict(schedule=schedule,trace=traced,step=index,before=before,count=count,completed=True,
                        routes=[dict(layer=i,producer=i if i<2 else i-2,offset=before,cache_present=i<2,
                            kv_shape=[1,1,32,16],normal_update_only=True) for i in range(4)] if traced else [],
                        cache_offsets=[before+count]*2,cache_sizes=[min(before+count,4),before+count],arrays={})
                    for name in (graph.stage_shapes() if traced else ['logits']):
                        shape=graph.stage_shapes()[name][:];axis=2 if name.endswith(('q_rotary','k_rotary','context')) else 1;shape[axis]=count
                        row['arrays'][name]=dict(dtype='<f4',shape=shape,offset=graph.MAX_RAW,bytes=math.prod(shape)*4,sha256='f'*64)
                        for contract in ('math','staged_fp32'):
                            comparisons.append(dict(schedule=schedule,trace=traced,step=index,name=name,expected_name=name,
                                contract=contract,axis=axis,passed=True,maximum_absolute_error=1.2345678901234567e-300,
                                maximum_error_to_policy_ratio=1.2345678901234567e-300))
                        arrays+=1
                    rows.append(row);before+=count
        self.assertEqual((len(rows),arrays,len(comparisons)),(82,3731,7462))
        # Reserve 64 KiB beyond the exact fixed inventory for report metadata.
        size=len(graph.encoded(dict(rows=rows,comparisons=comparisons)))+64*1024
        self.assertGreater(size,2*1024*1024)
        self.assertLess(size,graph.MAX_OBSERVATION_JSON)


if __name__=='__main__':unittest.main()
