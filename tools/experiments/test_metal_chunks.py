"""Fixed chunk modes, causal states, raw timing and owned cleanup; no GPU."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

import metal_chunks as chunks


def fixture(check=False):
    memories=[]
    for i in range(4):
        memory={k:10 for k in chunks.prefill.continuation.FIELDS}
        memory.update(weights_bytes=1223992516,scratch_bytes=1000*(i+1));memory['admitted_bytes']=sum(memory.values());memories.append(memory)
    admission={'event':'ready','format':'toklet-metal-chunks-v1','check_only':check,'debug_assertions':False,'chunks':chunks.CHUNKS,
               'projection_mode':'Q4CompensatedTiled','normalization_mode':'Compensated','q4_execution_widths':[32,32],
               'hidden_size':1536,'num_layers':28,'vocab_size':151936,'capacity':2031 if check else 2048,'position_offset':17 if check else 0,
               'prompt_tokens':1024,'output_tokens':1,'warmups':0 if check else 3,'samples':3 if check else 10,'requires_logit_bits':True,
               'application_bytes':64*1024**2,'memories':memories}
    admission['combined_admitted_bytes']=sum(m['admitted_bytes'] for m in memories)-3*memories[0]['weights_bytes']+admission['application_bytes']
    events=[]
    for i in range(3 if check else 13):
        case='reset' if not check or i==0 else 'append7' if i==1 else 'edit1000'
        report={'reason':{'kind':'output_limit'},'reused_tokens':0 if case=='reset' else 1024 if case=='append7' else 1000,
                'prefill_tokens':1024 if case=='reset' else 7 if case=='append7' else 31,'sampled_tokens':1,'emitted_tokens':1,
                'history_tokens':1025 if case=='reset' else 1032,'evaluated_tokens':1024 if case=='reset' else 1031,'pending_tokens':1,'decode_seconds':0}
        reports=[dict(report,first_token_seconds=10-j+i/100,prefill_seconds=9-j,total_seconds=11-j) for j in range(4)]
        events.append({'event':'sample','run':i,'case':case,'order':[(i+j)%4 for j in range(4)],'reports':reports,'greedy_ids':[11]*4,'logit_bits_equal':True,'memories':copy.deepcopy(memories)})
    return admission,events


class Chunks(unittest.TestCase):
    def test_complete_mode_states_and_rotation(self):
        for check in [False,True]:
            admission,events=fixture(check);chunks.ready(admission,check)
            for i,event in enumerate(events):chunks.sample(event,i,admission,check)
        rows=chunks.analyze(fixture()[1])['rows']
        self.assertEqual([r['chunk'] for r in rows],chunks.CHUNKS)
        self.assertEqual(rows[0]['raw_seconds'],[10+i/100 for i in range(3,13)])
        self.assertAlmostEqual(rows[0]['median_seconds'],10.075)
        self.assertAlmostEqual(rows[3]['improvement_vs_128_fraction'],1-7.075/10.075)
        with self.assertRaises(ValueError):chunks.analyze(events)

    def test_admission_precision_geometry_shared_budget_and_width_mutations(self):
        admission,_=fixture()
        for key,value in [('chunks',[128,256,512,512]),('capacity',2031),('position_offset',17),('check_only',True),
                          ('projection_mode','Q4Tiled'),('normalization_mode','Original'),('requires_logit_bits',1),
                          ('q4_execution_widths',[32,16]),('q4_execution_widths',[True,True]),('num_layers',24),('combined_admitted_bytes',chunks.LIMIT+1)]:
            changed=copy.deepcopy(admission);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):chunks.ready(changed)
        changed=copy.deepcopy(admission);changed['memories'][1]['weights_bytes']+=1;changed['memories'][1]['admitted_bytes']+=1
        with self.assertRaisesRegex(ValueError,'weights'):chunks.ready(changed)

    def test_all_row_refinement_timing_and_cache_mutations_fail(self):
        admission,events=fixture(True)
        for key,value in [('run',True),('case','reset'),('order',[1,True,3,0]),('greedy_ids',[11,12,11,11]),('logit_bits_equal',1)]:
            changed=copy.deepcopy(events[1]);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):chunks.sample(changed,1,admission,True)
        for key,value in [('prefill_tokens',8),('reused_tokens',0),('evaluated_tokens',1030),('pending_tokens',0),('first_token_seconds',math.nan),('decode_seconds',1)]:
            changed=copy.deepcopy(events[1]);changed['reports'][2][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):chunks.sample(changed,1,admission,True)
        changed=copy.deepcopy(events[1]);changed['memories'][3]['scratch_bytes']+=1
        with self.assertRaisesRegex(ValueError,'buffers'):chunks.sample(changed,1,admission,True)

    def test_owned_failure_reaps_and_preserves_primary_and_cleanup_errors(self):
        children=[];commands=[];real=subprocess.Popen
        def launch(command,**kwargs):
            self.assertTrue(kwargs['start_new_session']);commands.append(command)
            child=real([sys.executable,'-c','print(\'{"event":"bad"}\',flush=True);import time;time.sleep(60)'],**kwargs)
            child.wait=Mock(wraps=child.wait);children.append(child);return child
        def digest(path):
            return {'model.safetensors':chunks.prefill.LARGER_SOURCE_SHA,'config.json':chunks.prefill.LARGER_CONFIG_SHA,
                    'manifest.json':chunks.prefill.LARGER_PACKED_SHA,'prompt_1024.u32le':chunks.PROMPT_SHA}.get(path.name,'test')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'src').mkdir();output=root/'result.json'
            with patch.object(chunks,'ROOT',root),patch.object(chunks,'digest',digest),patch.object(chunks.prefill,'observation',return_value='test'),patch.object(chunks.prefill,'endpoints',side_effect=[{},OSError('endpoint failure')]),patch.object(chunks.subprocess,'check_output',return_value='test'),patch.object(chunks.subprocess,'Popen',launch):
                with self.assertRaisesRegex(ValueError,'collection failed'):chunks.run(root/'binary',output,check=True)
            result=json.loads(output.read_text())
            self.assertFalse(result['passed']);self.assertIn('admission/mode',result['error']);self.assertIn('endpoint failure',result['cleanup_errors'][0]);self.assertEqual(commands[0][-1],'check')
            children[0].wait.assert_called_once_with(timeout=5);self.assertIsNotNone(children[0].returncode)
            with patch.object(chunks.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'destination exists'):chunks.run(root/'binary',output)
                launch.assert_not_called()

    def test_frozen_pin_mutation_fails_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'src').mkdir()
            pins={'model.safetensors':chunks.prefill.LARGER_SOURCE_SHA,'config.json':chunks.prefill.LARGER_CONFIG_SHA,
                  'manifest.json':chunks.prefill.LARGER_PACKED_SHA,'prompt_1024.u32le':chunks.PROMPT_SHA}
            for mutated in pins:
                def digest(path):return 'replacement' if path.name==mutated else pins.get(path.name,'test')
                with self.subTest(pin=mutated),patch.object(chunks,'ROOT',root),patch.object(chunks,'digest',digest),patch.object(chunks.subprocess,'Popen') as launch:
                    with self.assertRaisesRegex(ValueError,'Frozen larger'):chunks.run(root/'binary',root/'new.json')
                    launch.assert_not_called()


if __name__=='__main__':unittest.main()
