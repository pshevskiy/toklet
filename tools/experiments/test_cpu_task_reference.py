"""Offline raw reference memory/state/phase and owned worker cleanup."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import cpu_task_reference as ref

class Reference(unittest.TestCase):
 def case(self):return {'id':'prose/0','category':'prose','prompt_tokens':[1,7,13],'prompt_text':'prepared','max_output_tokens':32,'stop_policy':'eos_or_32','scoring':ref.tasks.SCORERS['prose'],'expected':'A'}
 def admission(self):
  m={'weights_bytes':ref.WEIGHTS,'kv_bytes':57344*562,'scratch_bytes':147456*8+8*562,'history_bytes':4*562,'logits_bytes':151936*8,'metadata_bytes':9408,'metadata_allowance_bytes':ref.APP};m['admitted_bytes']=sum(m.values())
  return {'event':'ready','format':'toklet-cpu-task-reference-v1','debug_assertions':False,'hidden_size':1536,'num_layers':28,'vocab_size':151936,'backend':'cpu','cpu_projection':'tiled','weight_dtype':'FP32','kv_dtype':'FP32','normalization_mode':'OrderedFP64','readback_mode':'FullLogits','context_capacity':562,'chunk_size':8,'tokenizer_resident':False,'eos_ids':[151643,151645],'application_bytes':ref.APP,'memory':m,'memory_admitted_bytes':m['admitted_bytes']+ref.APP}
 def events(self,cap=32,stop=False):
  ids=[] if stop else [42]*cap;sampled=len(ids)+int(stop);prompt=self.case()['prompt_tokens'];history=prompt+ids+([151645] if stop else [])
  g={'reason':{'kind':'stop_token','token_id':151645} if stop else {'kind':'output_limit'},'reused_tokens':0,'prefill_tokens':3,'sampled_tokens':sampled,'emitted_tokens':len(ids),'history_tokens':len(history),'evaluated_tokens':len(history)-1,'pending_tokens':1,'prefill_seconds':.01,'decode_seconds':.02,'sampling_seconds':.001,'total_seconds':cap+1.,'first_token_seconds':None if stop else .1}
  s={'generation':g,'backend':'cpu','cpu_projection':'tiled','tokenizer_resident':False,'memory':self.admission()['memory'],'memory_admitted_bytes':self.admission()['memory_admitted_bytes'],'state':{'history_tokens':len(history),'evaluated_tokens':len(history)-1,'pending_tokens':1,'history_sha256':ref.history_sha(history)}}
  return [{'event':'token','token_id':v,'index':i,'elapsed_seconds':i+.1} for i,v in enumerate(ids)]+[{'event':'done','statistics':s},{'event':'reset'}]
 def test_exact_raw_admission_and_no_resident_tokenizer(self):
  ref.ready(self.admission());self.assertEqual(self.admission()['memory_admitted_bytes'],6343713560)
  self.assertGreater(self.admission()['memory_admitted_bytes']+ref.TOKENIZER_ALLOWANCE,ref.LIMIT)
  for key,value in [('tokenizer_resident',True),('backend','metal'),('context_capacity',True),('chunk_size',128),('cpu_projection','scalar'),('debug_assertions',True),('memory_admitted_bytes',ref.LIMIT+1)]:
   e=self.admission();e[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):ref.ready(e)
  e=self.admission();e['memory']['logits_bytes']//=2
  with self.assertRaises(ValueError):ref.ready(e)
 def test_full_caps_immediate_hidden_eos_and_probe_are_distinct(self):
  for stop in (False,True):ref.validate_response(self.case(),self.events(stop=stop),self.admission())
  ref.validate_response(self.case(),self.events(cap=8),self.admission(),8)
  with self.assertRaisesRegex(ValueError,'cap exhaustion'):ref.validate_response(self.case(),self.events(cap=8),self.admission())
  with self.assertRaises(ValueError):ref.validate_response(self.case(),self.events(cap=8),self.admission(),9)
 def test_error_cancellation_state_mode_and_reset_mutations_reject(self):
  for mutate in (lambda e:e.insert(0,{'event':'error'}),lambda e:e.pop(),lambda e:e[-2]['statistics']['generation'].update(reason={'kind':'cancelled'}),lambda e:e[-2]['statistics']['state'].update(history_sha256='0'*64),lambda e:e[-2]['statistics'].update(backend='metal'),lambda e:e[0].update(token_id=151645),lambda e:e[0].update(text='unexpected'),lambda e:e[-2]['statistics']['generation'].update(pending_tokens=True),lambda e:e[-2]['statistics']['generation'].update(decode_seconds=float('nan'))):
   events=self.events();mutate(events)
   with self.subTest(events=events[-2:]),self.assertRaises(ValueError):ref.validate_response(self.case(),events,self.admission())
 def test_decoder_requires_owned_model_exit(self):
  with patch('tokenizers.Tokenizer.from_file') as factory:
   with self.assertRaisesRegex(ValueError,'exit/reaping'):ref.finalize([],Path('tokenizer.json'),model_exited=False)
   factory.assert_not_called();factory.return_value.decode.return_value='A'
   item={'visible_token_ids':[42]};ref.finalize([item],Path('tokenizer.json'),model_exited=True);self.assertEqual(item['visible_text'],'A')
 def test_bad_ready_reaps_and_cannot_construct_tokenizer(self):
  real=subprocess.Popen;owned=[]
  def launch(command,**kw):
   self.assertTrue(kw['start_new_session']);p=real([sys.executable,'-c','print(\'{"event":"wrong"}\',flush=True);import time;time.sleep(60)'],**kw);p.wait=Mock(wraps=p.wait);owned.append(p);return p
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   with (root/'raw').open('x') as raw,(root/'stderr').open('xb') as stderr,patch.object(ref.subprocess,'Popen',launch),patch.object(ref,'finalize') as finalize:r=ref.child(root/'binary',root,[self.case()]*100,raw,stderr,True)
   finalize.assert_not_called()
  self.assertFalse(r['passed']);self.assertFalse(r['model_exited']);self.assertEqual(r['cleanup_errors'],[]);self.assertIn('actual mode/geometry',r['error']);self.assertIsNotNone(owned[0].returncode);owned[0].wait.assert_called_once_with(timeout=5)

if __name__=='__main__':unittest.main()
