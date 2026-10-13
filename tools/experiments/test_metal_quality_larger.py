"""Larger source/mode/reference admission and owned transport checks; no GPU."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

import metal_quality_larger as large
import metal_quality as shared
from test_metal_quality import fixture,struct_bytes


class LargerLoss(unittest.TestCase):
    def test_parsed_contract_and_all_initial_pins_are_reasserted(self):
        source=Path('/source');corpus=Path('/contract/corpus');baseline=Path('/baseline');packed=Path('/packed')
        pins={str(source/'model.safetensors'):large.SOURCE_SHA,str(source/'config.json'):large.CONFIG_SHA,
              str(source/'tokenizer.json'):large.TOKENIZER_SHA,str(corpus):shared.CORPUS_SHA,
              str(corpus.parent/'manifest.json'):'admitted-contract',str(baseline):large.BASELINE_SHA,str(packed/'manifest.json'):large.PACKED_SHA}
        large.check_initial_pins(pins,source,corpus,baseline,packed,True,'admitted-contract')
        # Includes replacement after parsing but before the snapshot: endpoint
        # stability of the replacement must not make it the admitted contract.
        for key in pins:
            altered=dict(pins);altered[key]='replacement'
            with self.subTest(path=key),self.assertRaises(ValueError):large.check_initial_pins(altered,source,corpus,baseline,packed,True,'admitted-contract')

    def test_larger_precision_geometry_and_residency_are_required(self):
        tokens=[1,2,3];event,*_,sha=fixture(tokens)
        event.update(hidden_size=1536,num_layers=28,vocab_size=151936,projection_mode='Q4Compensated',normalization_mode='Compensated')
        event['memory']['weights_bytes']=1223992516
        event['memory']['admitted_bytes']=sum(v for k,v in event['memory'].items() if k!='admitted_bytes')
        event['memory_admitted_bytes']=event['memory']['admitted_bytes']+64*1024**2
        large.ready(event,tokens,sha,True,128)
        for key,value in [('hidden_size',896),('num_layers',True),('vocab_size',151643),('projection_mode','Q4Tiled'),('normalization_mode','Original')]:
            altered=copy.deepcopy(event);altered[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):large.ready(altered,tokens,sha,True,128)
        altered=copy.deepcopy(event);altered['memory']['weights_bytes']-=1
        altered['memory']['admitted_bytes']-=1;altered['memory_admitted_bytes']-=1
        with self.assertRaisesRegex(ValueError,'resident'):large.ready(altered,tokens,sha,True,128)

    def test_reference_requires_full_causal_coverage_and_exact_source(self):
        tokens=[1]*10001;ready,rows,chunks,blocks,done,_=fixture(tokens,8)
        ready.update(backend='cpu',cpu_projection='tiled',weight_dtype='FP32',kv_dtype='FP32',hidden_size=1536,num_layers=28,vocab_size=151936,corpus_sha256=shared.CORPUS_SHA)
        run={'passed':True,'exit_code':0,'cleanup_errors':[],'ready':ready,'losses':rows,'chunks':chunks,'blocks':blocks,'done':done}
        value={'format':'toklet-cpu-row-loss-v1','passed':True,'collection_completed':True,'probes_completed':True,'larger':True,'scored_targets':10000,
               'context_policy':shared.POLICY,'mean_negative_log_likelihood':done['mean_negative_log_likelihood'],'identities_unchanged':True,'endpoint_errors':[],
               'identities':{'/snapshot/models/Qwen2.5-1.5B-Instruct/model.safetensors':large.SOURCE_SHA},'runs':[{'passed':True,'exit_code':0,'cleanup_errors':[]} for _ in range(4)]+[run],'probes':[{},{}]}
        with patch.object(large,'BASELINE_MEAN',done['mean_negative_log_likelihood']):
            self.assertIs(large.reference(value,tokens),value)
            for key,v in [('scored_targets',9999),('larger',False),('collection_completed',False),('identities_unchanged',False)]:
                altered=copy.deepcopy(value);altered[key]=v
                with self.subTest(key=key),self.assertRaises(ValueError):large.reference(altered,tokens)
            altered=copy.deepcopy(value);altered['runs'][-1]['losses'][0]['target_token']=2
            with self.assertRaisesRegex(ValueError,'alignment'):large.reference(altered,tokens)
            altered=copy.deepcopy(value);altered['runs'][-1]['ready']['weight_dtype']='FP16'
            with self.assertRaisesRegex(ValueError,'configuration'):large.reference(altered,tokens)
            altered=copy.deepcopy(value);altered['identities']['/snapshot/models/Qwen2.5-1.5B-Instruct/model.safetensors']='0'*64
            with self.assertRaisesRegex(ValueError,'source'):large.reference(altered,tokens)

    def test_custom_admission_and_argument_preserve_owned_failure_cleanup(self):
        real=subprocess.Popen;children=[];calls=[]
        def launch(command,**kwargs):
            calls.append(command)
            child=real([sys.executable,'-c','print(\'{"event":"unexpected"}\',flush=True);import time;time.sleep(60)'],**kwargs)
            child.wait=Mock(wraps=child.wait);children.append(child);return child
        admission=Mock(side_effect=ValueError('larger admission failed'))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);token=p/'tokens';token.write_bytes(struct_bytes([1,2]))
            with (p/'raw').open('x') as raw,(p/'stderr').open('xb') as stderr,patch.object(shared.subprocess,'Popen',launch):
                result=shared.child(p/'binary',p,p,token,[1,2],True,128,raw,stderr,admission=admission,extra_args=('larger',))
        self.assertEqual(calls[0][-1],'larger');self.assertFalse(result['passed']);self.assertIn('larger admission failed',result['error'])
        self.assertEqual(result['cleanup_errors'],[]);admission.assert_called_once();children[0].wait.assert_called_once_with(timeout=5)
        self.assertIsNotNone(children[0].returncode)

    def test_existing_output_rejects_before_worker_or_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'evidence.json';p.write_text('immutable')
            with patch.object(shared.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'destination exists'):large.run(Path('/missing/binary'),p,True)
                launch.assert_not_called()
            self.assertEqual(p.read_text(),'immutable')


if __name__=='__main__':unittest.main()
