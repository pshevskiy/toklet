"""Offline frozen application grammar, timing boundaries and owned cleanup."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import metal_application as m
import test_metal_tasks as fixtures
from test_task_cases import inventory,events

class Application(unittest.TestCase):
 def fixture(self,eos=False):
  case=inventory()[0][40];stream=events(case,count=0 if eos else 2,eos=eos,text='A',projection='tiled');ready=fixtures.MetalTasks().admission();s=next(v['statistics'] for v in stream if v['event']=='done');s.update(copy.deepcopy({k:v for k,v in ready.items() if k not in ('event','context_capacity')}))
  for k in ('input_formatting_seconds','tokenization_seconds','detokenization_seconds','publication_seconds','load_seconds','conversion_seconds','tokenizer_load_seconds'):s[k]=0.01
  return case['prompt_tokens'],stream,ready
 def test_actual_mode_empty_prompt_pending_stop_and_reset(self):
  for eos in (False,True):
   prompt,stream,ready=self.fixture(eos);r=m.collect_response(prompt,2,stream,ready);self.assertEqual(r['statistics']['backend'],'metal');self.assertEqual(r['stop_reason']['kind'],'stop_token' if eos else 'output_limit')
  for mutate in (lambda s:s.update(backend='cpu'),lambda s:s.update(cpu_projection='scalar'),lambda s:s['execution'].update(projection_mode='Original'),lambda s:s['generation'].update(reused_tokens=1),lambda s:s.update(tokenization_seconds=float('nan')),lambda s:s['generation'].update(reason={'kind':'cancelled'})):
   prompt,stream,ready=self.fixture();mutate(next(v['statistics'] for v in stream if v['event']=='done'))
   with self.assertRaises(ValueError):m.collect_response(prompt,2,stream,ready)
 def test_error_missing_reset_eos_indices_and_cap_reject(self):
  for mutate in (lambda e:e.insert(0,{'event':'error'}),lambda e:e.pop(),lambda e:e[0].update(token_id=151645),lambda e:e[0].update(index=True)):
   prompt,stream,ready=self.fixture();mutate(stream)
   with self.assertRaises(ValueError):m.collect_response(prompt,2,stream,ready)
  prompt,stream,ready=self.fixture()
  for cap in (True,1,8,128):
   with self.assertRaises(ValueError):m.collect_response(prompt,cap,stream,ready)
 def test_received_text_can_first_appear_after_done_at_reset(self):
  events=[{'event':'token','text':''},{'event':'done'},{'event':'text','text':'tail'},{'event':'reset'}];r=m.received_timing(events,[1,2,3,4]);self.assertEqual(r['submission_to_first_token_seconds'],1);self.assertEqual(r['submission_to_first_visible_text_seconds'],3);self.assertEqual(r['submission_to_done_seconds'],2);self.assertEqual(r['submission_to_finalized_seconds'],4)
  r=m.received_timing([{'event':'done'},{'event':'reset'}],[1,2]);self.assertIsNone(r['submission_to_first_token_seconds']);self.assertIsNone(r['submission_to_first_visible_text_seconds'])
  for times in ([1,2,3],[1,float('nan'),3,4],[1,2,4,3]):
   with self.assertRaises(ValueError):m.received_timing(events,times)
 def test_exact_bounded_trace_bytes_checked_before_parse(self):
  manifest=m.admission.frozen_manifest()
  with patch.object(m.admission,'bounded',return_value=b'[]'),patch.object(m.json,'loads',side_effect=AssertionError('must not parse')):
   with self.assertRaisesRegex(ValueError,'transcript differs'):m.trace_inputs(Path('source'),manifest,2)
 def test_both_caps_preserve_frozen_scripted_turns_and_prompt_ids(self):
  manifest=m.admission.frozen_manifest();source=m.ROOT/'models/Qwen2.5-0.5B-Instruct'
  requests,prompts=m.trace_inputs(source,manifest,2);full,full_prompts=m.trace_inputs(source,manifest,128)
  self.assertEqual(list(map(len,prompts)),[36,76,117]);self.assertEqual(prompts,full_prompts)
  self.assertEqual([len(v['messages']) for v in requests],[2,4,6]);self.assertEqual([v['max_new_tokens'] for v in requests],[2]*3);self.assertEqual([v['max_new_tokens'] for v in full],[128]*3)
  self.assertEqual([v['messages'] for v in requests],[v['messages'] for v in full])
 def test_bad_ready_reaps_owned_worker_before_decoder(self):
  real=subprocess.Popen;owned=[]
  def launch(command,**kw):
   self.assertTrue(kw['start_new_session']);p=real([sys.executable,'-c','print(\'{"event":"wrong"}\',flush=True);import time;time.sleep(60)'],**kw);p.wait=Mock(wraps=p.wait);owned.append(p);return p
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   with (root/'raw').open('x') as raw,(root/'stderr').open('xb') as stderr,patch.object(m.subprocess,'Popen',launch),patch.object(m.tasks,'finalized') as decode:r=m.child(root/'binary',root,'small',[],[],raw,stderr,True)
   decode.assert_not_called()
  self.assertFalse(r['passed']);self.assertFalse(r['model_exited']);self.assertEqual(r['cleanup_errors'],[]);self.assertIsNotNone(owned[0].returncode);owned[0].wait.assert_called_once_with(timeout=5)

 def test_final_thermal_failure_preserves_summary_and_primary_error(self):
  manifest=m.admission.frozen_manifest();binary=m.ROOT/'artifacts/m5-cli-half-native-binary-v1/toklet';original=m.metal_engine.digest
  saved=json.loads((m.ROOT/'benchmarks/results/2026-10-14-small-half-tasks-check-v1.json').read_text())['identities']
  def digest(path):return saved.get(str(path)) or original(path)
  for failed in (False,True):
   thermals=0
   def observe(command):
    nonlocal thermals
    if command[-1]=='therm':
     thermals+=1
     if thermals==2:raise TimeoutError('thermal observation failed')
    return '17179869184' if command[-1]=='hw.memsize' else 'fixed'
   def child(binary,source,model,requests,prompts,raw,stderr,check):
    raw.write('retained native evidence\n');raw.flush()
    return {'passed':not failed,'error':'original worker failure' if failed else None,'samples':[{'scenario_seconds':1.0}],'cleanup_errors':[]}
   with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'output.json'
    with patch.object(m.admission,'qualified_binary',return_value=(binary,saved[str(binary)])),patch.object(m,'trace_inputs',return_value=([],[])),patch.object(m.metal_engine,'digest',side_effect=digest),patch.object(m.controlled.metal_prefill,'observation',side_effect=observe),patch.object(m,'child',side_effect=child):
     with self.assertRaisesRegex(ValueError,'evidence retained'):m.run(path,'small',False)
    result=json.loads(path.read_text());self.assertFalse(result['passed']);self.assertFalse(result['measurements_completed']);self.assertIn('thermal_end: TimeoutError',result['endpoint_errors'][0]);self.assertEqual(result['run']['samples'],[{'scenario_seconds':1.0}]);self.assertEqual(path.with_suffix('.raw.jsonl').read_text(),'retained native evidence\n')
    if failed:self.assertEqual(result['run']['error'],'original worker failure');self.assertIn('owned worker failed',result['error'])

if __name__=='__main__':unittest.main()
