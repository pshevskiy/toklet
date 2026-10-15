"""Portable fail-closed M6 loss admission and guard ordering."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import m6_quality as m6
import test_metal_quality as old


class M6Quality(unittest.TestCase):
    def test_online_readiness_requires_actual_mode_and_selected_representation(self):
        tokens=[1,2,3]
        event,*_,digest=old.fixture(tokens)
        event.update(quantized=False,weight_dtype='FP16',projection_mode='HalfCompensated',
                     normalization_mode='Original',half_compensated_execution_widths=[32,32],
                     hidden_size=896,num_layers=24,vocab_size=151936,attention_mode='Online')
        event['memory']['weights_bytes']=988069596
        event['memory']['admitted_bytes']=sum(v for k,v in event['memory'].items() if k!='admitted_bytes')
        event['memory_admitted_bytes']=event['memory']['admitted_bytes']+64*1024**2
        m6.ready(event,tokens,digest,False,128,model='small')
        for mode in [None,'Reference','online',True]:
            mutated=copy.deepcopy(event);mutated['attention_mode']=mode
            with self.assertRaises(ValueError):m6.ready(mutated,tokens,digest,False,128,model='small')
        with self.assertRaises(ValueError):m6.ready(event,tokens,digest,True,128,model='small')
        with self.assertRaises(ValueError):m6.ready(event,tokens,digest,False,128,model='larger')

    def test_larger_ready_requires_frozen_v2_modes_and_residency(self):
        tokens=[1,2,3]
        event,*_,digest=old.fixture(tokens)
        event.update(weight_dtype='Q4 v2 + FP16 exclusions',projection_mode='Q4CompensatedTiled',
                     normalization_mode='Compensated',q4_compensated_execution_widths=[32,32],
                     q4_format='V2',q4_quantizer='toklet-q4-affine-f32-source-half-metadata-rne-g8-v2',
                     hidden_size=1536,num_layers=28,vocab_size=151936,attention_mode='Online')
        event['memory']['weights_bytes']=1777247428
        event['memory']['admitted_bytes']=sum(v for k,v in event['memory'].items() if k!='admitted_bytes')
        event['memory_admitted_bytes']=event['memory']['admitted_bytes']+64*1024**2
        m6.ready(event,tokens,digest,True,128,model='larger')
        for key,value in [('projection_mode','Original'),('q4_format','V1'),('hidden_size',896)]:
            mutated=copy.deepcopy(event);mutated[key]=value
            with self.assertRaises(ValueError):m6.ready(mutated,tokens,digest,True,128,model='larger')

    def test_compiled_closure_and_prelaunch_reassertion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'src/x.rs';source.parent.mkdir();source.write_text('fn x(){}')
            snapshot=root/'artifacts/snapshot';(snapshot/'src').mkdir(parents=True)
            (snapshot/'src/x.rs').write_bytes(source.read_bytes())
            binary=root/'artifacts/worker';binary.write_bytes(b'compiled')
            manifest=root/'artifacts/build.json'
            def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
            payload={'format':'toklet-m6-quality-build-v1','sources':{'src/x.rs':sha(source)},
                     'snapshot':'artifacts/snapshot','binary':{'path':'artifacts/worker','sha256':sha(binary)}}
            manifest.write_text(json.dumps(payload))
            with patch.object(m6,'ROOT',root),patch.object(m6,'runtime_paths',return_value=[source]):
                value,check=m6.build_admission(manifest,binary);self.assertEqual(value,payload)
                for path in [source,snapshot/'src/x.rs',binary,manifest]:
                    before=path.read_bytes();path.write_bytes(before+b'changed')
                    with self.assertRaises(ValueError):check()
                    path.write_bytes(before);check()
                saved_manifest=manifest.read_bytes()
                manifest.write_bytes(b' '*(1024**2+1))
                with self.assertRaisesRegex(ValueError,'exceeds1MiB'):check()
                with self.assertRaisesRegex(ValueError,'exceeds1MiB'):m6.build_admission(manifest,binary)
                manifest.write_bytes(saved_manifest);check()
                for mutation in ['extra','missing','snapshot']:
                    wrong=copy.deepcopy(payload)
                    if mutation=='extra':wrong['sources']['extra']='0'*64
                    if mutation=='missing':wrong['sources']={}
                    if mutation=='snapshot':wrong['snapshot']='../outside'
                    manifest.write_text(json.dumps(wrong))
                    with self.assertRaises(ValueError):m6.build_admission(manifest,binary)
                    manifest.write_text(json.dumps(payload))

    def test_build_rejection_precedes_worker_launch(self):
        def reject():raise ValueError('compiled source changed')
        with patch.object(m6.shared.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'compiled source changed'):
                m6.shared.child(Path('bin'),Path('source'),None,Path('ids'),[1,2],False,128,
                                io.StringIO(),io.BytesIO(),before_launch=reject)
            launch.assert_not_called()

    def collect(self,full=True,failed_guard=False,divergent=False,mean=None,model="small"):
        probes=[{'name':name,'tokens':[1,2,3,4],'corpus_sha256':name,'source_nll':[1.,1.,1.],
                 'path':'unused','source_logit_path':'unused','source_logit_sha256':'0'*64} for name in ['future_changed','short']]
        expected=[(p['name'],p['tokens'],p['corpus_sha256']) for p in probes]
        calls=[]
        def child(*args,**kwargs):
            calls.append((args,kwargs))
            if failed_guard:return {'passed':False}
            value=1.+(.01 if divergent and args[6]==128 else 0.)
            return {'passed':True,'losses':[{'negative_log_likelihood':value} for _ in range(3)],
                    'mean_negative_log_likelihood':m6.v2.SPECS[model]['mean'] if mean is None else mean}
        result={}
        with patch.object(m6.shared.quality,'probe_inputs',return_value=probes),\
             patch.object(m6.v2,'probe_specs',return_value=expected),\
             patch.object(m6.shared,'child',side_effect=child):
            m6.collect(Path('bin'),Path('source'),None,Path('corpus'),[1]*10001,{'runs':[{'losses':[{'negative_log_likelihood':1.}]*3}]*4},model,full,
                       io.StringIO(),io.BytesIO(),lambda:None,result)
        return result,calls

    def test_guards_precede_full_loss_with_exact_selection(self):
        result,calls=self.collect()
        self.assertEqual(len(calls),5)
        self.assertEqual([args[6] for args,_ in calls],[1,128,1,128,128])
        self.assertEqual(len(calls[-1][0][4]),10001)
        self.assertTrue(all(kwargs['extra_args']==('small','half-compensated','online') for _,kwargs in calls))
        self.assertTrue(all(args[5] is False and callable(kwargs['before_launch']) for args,kwargs in calls))
        self.assertTrue(result['probes_completed'] and result['collection_completed'] and result['quality_completed'])
        self.assertEqual(result['scored_targets'],10000)

    def test_larger_collection_selects_v2_and_preserves_matched_fp32_probes(self):
        result,calls=self.collect(model='larger')
        self.assertEqual(len(calls),5)
        self.assertTrue(all(args[5] is True for args,_ in calls))
        self.assertTrue(all(k['extra_args']==('larger',m6.v2.MODE,'online') for _,k in calls))
        self.assertTrue(result['passed'] and result['quality_completed'])
        self.assertTrue(all(p['matched_fp32_nll']==[1.,1.,1.] and 'source_nll' not in p for p in result['probes']))

    def test_probe_only_does_not_claim_full_collection(self):
        result,calls=self.collect(False)
        self.assertEqual(len(calls),4)
        self.assertTrue(result['passed'] and result['probes_completed'])
        self.assertFalse(result['collection_completed'] or result['quality_completed'])

    def test_failed_guard_prevents_full_run(self):
        result,calls=self.collect(failed_guard=True)
        self.assertEqual(len(calls),1)
        self.assertFalse(result['passed'] or result['probes_completed'] or result['collection_completed'])

    def test_guard_discrepancy_is_not_tolerance_tuned(self):
        with self.assertRaises(ValueError):self.collect(divergent=True)

    def test_unchanged_quality_gate_preserves_completed_rejection(self):
        result,_=self.collect(mean=m6.v2.SPECS['small']['mean']+.051)
        self.assertTrue(result['collection_completed'] and result['quality_completed'])
        self.assertFalse(result['passed'] or result['quality_gate']['passed'])
        self.assertEqual(result['quality_gate']['maximum_increase_nats_per_token'],.05)


if __name__=='__main__':unittest.main()
