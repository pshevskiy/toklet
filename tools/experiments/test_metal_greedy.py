"""Fixed selection modes, complete generation state and owned cleanup; no GPU."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

import metal_greedy as greedy


def fixture():
    a={k:10 for k in greedy.prefill.continuation.FIELDS}
    a.update(weights_bytes=1223992516,scratch_bytes=1000,logits_bytes=151936*8);a['admitted_bytes']=sum(a.values())
    b=dict(a,scratch_bytes=1032,logits_bytes=8);b['admitted_bytes']=sum(b[k] for k in greedy.prefill.continuation.FIELDS)
    admission={'event':'ready','format':'toklet-metal-greedy-v1','debug_assertions':False,
               'readback_modes':['FullLogits','GreedyToken'],'projection_mode':'Q4CompensatedTiled','normalization_mode':'Compensated',
               'greedy_execution_width':32,'hidden_size':1536,'num_layers':28,'vocab_size':151936,'capacity':1024,'chunk':128,
               'prompt_tokens':512,'output_tokens':128,'warmups':3,'samples':10,'application_bytes':64*1024**2,'full_memory':a,'token_memory':b}
    admission['combined_admitted_bytes']=a['admitted_bytes']+b['admitted_bytes']-a['weights_bytes']+admission['application_bytes']
    events=[]
    for i in range(13):
        report={'reason':{'kind':'output_limit'},'reused_tokens':0,'prefill_tokens':512,'sampled_tokens':128,'emitted_tokens':128,
                'history_tokens':640,'evaluated_tokens':639,'pending_tokens':1,'first_token_seconds':3,'prefill_seconds':2,'decode_seconds':4,'sampling_seconds':0.01}
        timing={k:0 for k in ('command_creation_seconds','clock_calibration_seconds','commit_seconds','host_encode_seconds','wait_seconds','gpu_start_seconds','gpu_end_seconds','readback_seconds')};timing['command_count']=1
        events.append({'event':'sample','run':i,'order':'full_first' if i%2==0 else 'token_first','token_ids':[11]*128,'token_ids_equal':True,
                       'full_report':dict(report,total_seconds=10+i/100),'token_report':dict(report,total_seconds=9+i/100),
                       'full_memory':copy.deepcopy(a),'token_memory':copy.deepcopy(b),'full_last_timing':dict(timing),'token_last_timing':dict(timing)})
    return admission,events


class Greedy(unittest.TestCase):
    def test_complete_sequence_states_order_and_raw_analysis(self):
        admission,events=fixture();greedy.ready(admission)
        for i,event in enumerate(events):greedy.sample(event,i,admission)
        result=greedy.analyze(events)
        self.assertEqual(result['full']['raw_seconds'],[10+i/100 for i in range(3,13)])
        self.assertAlmostEqual(result['improvement_fraction'],1-9.075/10.075)
        with self.assertRaises(ValueError):greedy.analyze(events[:-1])

    def test_admission_modes_storage_geometry_width_and_budget_mutations(self):
        admission,_=fixture()
        for key,value in [('readback_modes',['FullLogits','FullLogits']),('projection_mode','Original'),('normalization_mode','Original'),
                          ('chunk',256),('output_tokens',1),('greedy_execution_width',True),('greedy_execution_width',0),('num_layers',24),('combined_admitted_bytes',greedy.LIMIT+1)]:
            changed=copy.deepcopy(admission);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):greedy.ready(changed)
        for key in ('weights_bytes','logits_bytes','scratch_bytes','kv_bytes'):
            changed=copy.deepcopy(admission);changed['token_memory'][key]+=1;changed['token_memory']['admitted_bytes']+=1
            with self.subTest(key=key),self.assertRaises(ValueError):greedy.ready(changed)

    def test_complete_token_semantic_timing_and_memory_mutations_fail(self):
        admission,events=fixture()
        for key,value in [('run',True),('order','token_first'),('token_ids_equal',1),('token_ids',[11]*127),('token_ids',[11]*127+[True]),('token_ids',[151936]*128)]:
            changed=copy.deepcopy(events[0]);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):greedy.sample(changed,0,admission)
        for key,value in [('prefill_tokens',513),('history_tokens',639),('pending_tokens',0),('decode_seconds',math.nan),('sampling_seconds',math.inf),('total_seconds',1)]:
            changed=copy.deepcopy(events[0]);changed['token_report'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):greedy.sample(changed,0,admission)
        changed=copy.deepcopy(events[0]);changed['token_memory']['scratch_bytes']+=1
        with self.assertRaisesRegex(ValueError,'buffers'):greedy.sample(changed,0,admission)
        changed=copy.deepcopy(events[0]);changed['token_last_timing']['gpu_start_seconds']=1
        with self.assertRaisesRegex(ValueError,'completion order'):greedy.sample(changed,0,admission)

    def test_owned_failure_reaps_and_preserves_primary_and_cleanup_errors(self):
        children=[];real=subprocess.Popen
        def launch(command,**kwargs):
            self.assertTrue(kwargs['start_new_session'])
            child=real([sys.executable,'-c','print(\'{"event":"bad"}\',flush=True);import time;time.sleep(60)'],**kwargs)
            child.wait=Mock(wraps=child.wait);children.append(child);return child
        def digest(path):
            return {'model.safetensors':greedy.prefill.LARGER_SOURCE_SHA,'config.json':greedy.prefill.LARGER_CONFIG_SHA,
                    'manifest.json':greedy.prefill.LARGER_PACKED_SHA,'prompt_512.u32le':greedy.PROMPT_SHA}.get(path.name,'test')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'src').mkdir();output=root/'result.json'
            with patch.object(greedy,'ROOT',root),patch.object(greedy,'digest',digest),patch.object(greedy.prefill,'observation',return_value='test'),patch.object(greedy.prefill,'endpoints',side_effect=[{},OSError('endpoint failure')]),patch.object(greedy.subprocess,'check_output',return_value='test'),patch.object(greedy.subprocess,'Popen',launch):
                with self.assertRaisesRegex(ValueError,'collection failed'):greedy.run(root/'binary',output)
            result=json.loads(output.read_text())
            self.assertFalse(result['passed']);self.assertIn('admission/mode',result['error']);self.assertIn('endpoint failure',result['cleanup_errors'][0])
            children[0].wait.assert_called_once_with(timeout=5);self.assertIsNotNone(children[0].returncode)
            with patch.object(greedy.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'destination exists'):greedy.run(root/'binary',output)
                launch.assert_not_called()

    def test_frozen_pin_mutation_fails_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'src').mkdir()
            pins={'model.safetensors':greedy.prefill.LARGER_SOURCE_SHA,'config.json':greedy.prefill.LARGER_CONFIG_SHA,
                  'manifest.json':greedy.prefill.LARGER_PACKED_SHA,'prompt_512.u32le':greedy.PROMPT_SHA}
            for mutated in pins:
                def digest(path):return 'replacement' if path.name==mutated else pins.get(path.name,'test')
                with self.subTest(pin=mutated),patch.object(greedy,'ROOT',root),patch.object(greedy,'digest',digest),patch.object(greedy.subprocess,'Popen') as launch:
                    with self.assertRaisesRegex(ValueError,'Frozen larger'):greedy.run(root/'binary',root/'new.json')
                    launch.assert_not_called()


if __name__=='__main__':unittest.main()
