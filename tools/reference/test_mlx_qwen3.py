"""Portable Qwen3 oracle semantics/identity checks; no MLX import or GPU needed."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import mlx_qwen3 as m

MANIFEST_SHA = "5dc5b7ec0fccd8ac6876814b88a917107993aab5f5e07dcb222f57ec377a7a4a"


class Math(unittest.TestCase):
    def model(self):
        config=dict(num_layers=1,num_query_heads=2,num_kv_heads=1,head_dim=4,rms_norm_eps=1e-6,rope_theta=10000.)
        weights={'embed_tokens':np.linspace(-.3,.4,32,dtype=np.float32).reshape(8,4),'norm':np.ones(4,dtype=np.float32)}
        for name in ['input_layernorm','post_attention_layernorm']:weights['layers.0.'+name]=np.ones(4,dtype=np.float32)
        for name,rows in [('q',8),('k',4),('v',4),('o',4),('gate',4),('up',4),('down',4)]:
            weights['layers.0.'+name+'_proj.weight']=np.linspace(-.15,.2,rows*4,dtype=np.float32).reshape(rows,4)
        weights['layers.0.o_proj.weight']=np.linspace(-.2,.25,32,dtype=np.float32).reshape(4,8)
        weights['layers.0.q_norm']=np.array([.5,1.5,.75,1.25],dtype=np.float32)
        weights['layers.0.k_norm']=np.array([1.25,.75,1.5,.5],dtype=np.float32)
        weights['output']=np.roll(weights['embed_tokens'],2,axis=0)*np.float32(-1.25)
        return weights,config
    def test_bias_free_projection_and_single_position_gqa(self):
        w,c=self.model();r=m.forward([1],0,w,c,np.float64,np.float32)
        expected=r['layers.0.input_norm'].astype(np.float64)@w['layers.0.q_proj.weight'].T
        np.testing.assert_allclose(r['layers.0.q_projection'].reshape(1,8),expected,atol=1e-7,rtol=1e-6)
        np.testing.assert_array_equal(r['layers.0.attention_probabilities'],np.ones((2,1,1)))
        expected_v=np.tile(r['layers.0.v_projection'][:,0,:],(1,2))
        np.testing.assert_array_equal(r['layers.0.attention_context'],expected_v)
    def test_qk_norm_is_per_head_before_rotary_with_distinct_weights(self):
        w,c=self.model();r=m.forward([1,2],0,w,c,np.float64,np.float32)
        for name in ['q','k']:
            x=r['layers.0.'+name+'_projection'].astype(np.float64)
            expected=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+c['rms_norm_eps'])*w['layers.0.'+name+'_norm']
            np.testing.assert_allclose(r['layers.0.'+name+'_norm'],expected,atol=1e-7,rtol=1e-6)
        # Flattening heads into one RMS reduction is a different graph.
        q=r['layers.0.q_projection'].reshape(2,8).astype(np.float64)
        wrong=(q/np.sqrt(np.mean(q*q,axis=-1,keepdims=True)+c['rms_norm_eps'])*np.tile(w['layers.0.q_norm'],2)).reshape(2,2,4)
        self.assertGreater(float(np.max(np.abs(wrong-r['layers.0.q_norm']))),.01)
        self.assertEqual(r['layers.0.attention_context'].shape,(2,8))
    def test_qk_norm_handles_zero_and_epsilon_dominated_heads(self):
        w,c=self.model();w['layers.0.q_proj.weight'].fill(0);w['layers.0.k_proj.weight'].fill(1e-8)
        r=m.forward([1],0,w,c,np.float64,np.float32)
        np.testing.assert_array_equal(r['layers.0.q_norm'],np.zeros((1,2,4)))
        x=r['layers.0.k_projection'].astype(np.float64)
        expected=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+c['rms_norm_eps'])*w['layers.0.k_norm']
        np.testing.assert_allclose(r['layers.0.k_norm'],expected,atol=1e-10,rtol=1e-6)
    def test_explicit_zero_bias_is_not_an_absent_bias(self):
        w,c=self.model();w['layers.0.q_proj.bias']=np.zeros(4,dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'explicit projection biases'):m.forward([1],0,w,c,np.float32,np.float32)
    def test_future_change_cannot_affect_earlier_rows(self):
        w,c=self.model();a=m.forward([1,2],0,w,c,np.float32,np.float16);b=m.forward([1,5],0,w,c,np.float32,np.float16)
        for name in ['embedding','layers.0.q_rotary','layers.0.k_rotary','layers.0.attention_context','final_norm','logits']:
            np.testing.assert_array_equal(a[name][:1],b[name][:1],err_msg=name)
    def test_nonzero_rotary_offset_uses_split_pair_at_actual_position(self):
        w,c=self.model();r=m.forward([1],7,w,c,np.float64,np.float32)
        value=r['layers.0.q_norm'][0,0].astype(np.float64)
        angles=np.array([7.,7./100.]);a=value[:2];b=value[2:]
        expected=np.concatenate([a*np.cos(angles)-b*np.sin(angles),b*np.cos(angles)+a*np.sin(angles)])
        np.testing.assert_allclose(r['layers.0.q_rotary'][0,0],expected,atol=1e-7,rtol=1e-6)
    def test_separate_head_and_storage_boundaries(self):
        w,c=self.model();a=m.forward([1,2],0,w,c,np.float32,np.float16)
        b=m.forward([1,2],0,dict(w,output=w['embed_tokens']),c,np.float32,np.float16)
        self.assertFalse(np.array_equal(a['logits'],b['logits']))
        for name,value in a.items():
            self.assertEqual(value.dtype,np.float32 if name.endswith(('attention_scores','attention_probabilities','logits')) else np.float16,name)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=m.verify(m.DEFAULT, MANIFEST_SHA);cls.raw=(m.DEFAULT/'arrays.bin').read_bytes()
    def array(self,name):
        item=self.manifest['arrays'][name];raw=self.raw[item['offset']:item['offset']+item['bytes']]
        return np.frombuffer(raw,dtype=item['dtype']).reshape(item['shape'])
    def test_frozen_manifest_policy_and_complete_coverage(self):
        self.assertEqual(hashlib.sha256((m.DEFAULT/'manifest.json').read_bytes()).hexdigest(),MANIFEST_SHA)
        self.assertEqual(len(self.manifest['arrays']),891);self.assertEqual(set(self.manifest['arrays']),set(m.array_contract()))
        self.assertEqual(self.manifest['policy'],m.POLICY)
        for variant,tied in [('tied',True),('untied',False)]:
            config=self.manifest['variants'][variant]['config'];self.assertEqual(config,m.configuration(tied))
            self.assertFalse(config['attention_bias']);self.assertFalse(config['mlp_bias']);self.assertEqual(config['head_dim'],16);self.assertNotEqual(config['hidden_size'],config['head_dim']*config['num_attention_heads'])
    def test_upstream_full_and_actual_cached_logits_agree_with_fixed_fp32_gate(self):
        for variant in ['tied','untied']:
            for backend in ['CPU','GPU']:
                for case,item in m.CASES.items():
                    if item['position_offset']:continue
                    np.testing.assert_allclose(self.array(f'{variant}.native.{backend}.{case}.logits'),self.array(f'{variant}.fp32.cases.{case}.logits'),atol=1e-5,rtol=1e-4)
                np.testing.assert_allclose(self.array(f'{variant}.native.{backend}.cached.logits'),self.array(f'{variant}.fp32.cases.prompt.logits'),atol=1e-5,rtol=1e-4)
    def test_untied_head_changes_distribution_and_selected_token(self):
        a=self.array('untied.half.cases.prompt.logits');b=self.array('untied.witness.tied_substitution.logits')
        self.assertNotEqual(int(np.argmax(a[-1])),int(np.argmax(b[-1])));self.assertGreater(float(np.max(np.abs(a-b))),.1)
        witness=self.manifest['variants']['untied']['untied_witness']
        self.assertEqual(int(np.argmax(a[-1])),witness['expected_last_id']);self.assertEqual(int(np.argmax(b[-1])),witness['tied_substitution_last_id'])
    def test_fixture_distinguishes_bypassed_and_swapped_norms(self):
        main=self.array('tied.half.cases.prompt.logits')
        for label in ['norms_bypassed','norms_swapped']:
            self.assertGreater(float(np.max(np.abs(main-self.array('tied.witness.'+label+'.logits')))),.001)
    def test_fixture_causality_and_half_vs_fp32_dtypes_are_explicit(self):
        for variant in ['tied','untied']:
            for precision in ['half','fp32']:
                for stage in ['embedding','layers.0.q_rotary','layers.1.output','logits']:
                    a=self.array(f'{variant}.{precision}.cases.prompt.{stage}')
                    b=self.array(f'{variant}.{precision}.cases.future_changed.{stage}')
                    np.testing.assert_array_equal(a[:3],b[:3])
            self.assertEqual(self.array(f'{variant}.half.cases.prompt.embedding').dtype,np.float16)
            self.assertEqual(self.array(f'{variant}.half.cases.prompt.logits').dtype,np.float32)
    def test_stdlib_only_check_and_exclusive_publication(self):
        result=subprocess.run([sys.executable,'-S',str(Path(m.__file__)),'--check',str(m.DEFAULT),'--manifest-sha256',MANIFEST_SHA],capture_output=True,text=True,timeout=30,check=True)
        self.assertEqual(json.loads(result.stdout),dict(verified=True,arrays=891))
        with self.assertRaises(FileExistsError):m.export(m.DEFAULT)
        with tempfile.TemporaryDirectory() as directory:
            dangling=Path(directory)/'dangling';dangling.symlink_to(Path(directory)/'absent')
            with self.assertRaises(FileExistsError):m.export(dangling)
    def test_external_manifest_identity_is_required(self):
        with self.assertRaises(ValueError):m.verify(m.DEFAULT,'0'*64)
        result=subprocess.run([sys.executable,'-S',str(Path(m.__file__)),'--check',str(m.DEFAULT)],capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0)
    def test_metadata_gate_and_payload_mutations_reject(self):
        for mutation in ['normWitness','normShape','head','typedConfig','gate','coverage','shape','dtype','file','payload','missingGate','missingEnvelope','missingKind','overflow','nan','boolEnvelope','boolGate','versions','runtime','greedy','caseGreedy','witness']:
            with tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);value=copy.deepcopy(self.manifest)
                for name in value['files']:
                    target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((m.DEFAULT/name).read_bytes())
                key='untied.half.cases.prompt.logits'
                if mutation=='normWitness':
                    item=value['arrays']['tied.witness.norms_swapped.logits'];item['shape']=[1,256]
                elif mutation=='normShape':value['arrays']['tied.half.cases.prompt.layers.0.q_norm']['shape']=[4,64]
                elif mutation=='head':value['variants']['untied']['config']['tie_word_embeddings']=True
                elif mutation=='typedConfig':value['variants']['tied']['config']['head_dim']=16.0
                elif mutation=='gate':value['arrays'][key]['gate']['atol']+=.01
                elif mutation=='coverage':value['arrays'].pop(key)
                elif mutation=='shape':value['arrays'][key]['shape']=[1,256]
                elif mutation=='dtype':value['arrays'][key]['dtype']='<f2'
                elif mutation=='file':value['files']['extra']={'sha256':'f'*64,'bytes':1}
                elif mutation=='missingGate':value['arrays'][key].pop('gate')
                elif mutation=='missingEnvelope':value['arrays'][key].pop('precision_envelope')
                elif mutation=='missingKind':value['arrays'][key].pop('gate_kind')
                elif mutation=='overflow':value['arrays'][key].update(precision_envelope=1e308,gate=dict(atol=float('inf'),rtol=.001))
                elif mutation=='nan':value['arrays'][key]['precision_envelope']=float('nan')
                elif mutation=='boolEnvelope':value['arrays'][key]['precision_envelope']=False
                elif mutation=='boolGate':value['arrays'][key]['gate']['rtol']=True
                elif mutation=='versions':value['versions']['mlx']='other'
                elif mutation=='runtime':
                    for name in list(value['producer_pins']):
                        if name.endswith('/mlx/lib/mlx.metallib'):value['producer_pins'].pop(name)
                elif mutation=='greedy':value['variants']['untied']['greedy']['tokens'][0]=64
                elif mutation=='caseGreedy':value['variants']['untied']['cases']['prompt']['greedy'].pop()
                elif mutation=='witness':value['variants']['untied']['untied_witness']['maximum_difference']+=1
                else:(folder/'arrays.bin').write_bytes(b'changed')
                (folder/'manifest.json').write_text(json.dumps(value))
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.verify(folder)


if __name__=='__main__':unittest.main()
