"""Offline truthful backend/mode task admission, grammar and owned cleanup."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import metal_tasks as m
from test_task_cases import inventory,events

class MetalTasks(unittest.TestCase):
 def admission(self,model='small'):
  mem={k:1 for k in m.cli.FIELDS};mem['weights_bytes']=m.controlled.MODELS[model][5];mem['tokenizer_application_allowance_bytes']=247772624
  return {'event':'ready','backend':'metal','cpu_projection':'none','context_capacity':2048,'execution':m.cli.execution('HalfCompensated',128),'logical_memory':mem,'memory_admitted_bytes':sum(mem.values())}
 def fixture(self,cap=None,eos=False):
  case=inventory()[0][40];e=events(case,count=0 if eos else cap,eos=eos,text='A',projection='tiled');s=next(v['statistics'] for v in e if v['event']=='done');s.update({k:v for k,v in self.admission().items() if k not in ('event','context_capacity')});return case,e
 def test_actual_metal_backend_and_modes_preserve_frozen_grammar(self):
  for model in m.controlled.MODELS:m.ready(self.admission(model),model)
  for eos in (False,True):
   case,e=self.fixture(eos=eos);r=m.collect_response(case,e,self.admission());self.assertEqual(r['statistics']['cpu_projection'],'none');self.assertEqual(r['statistics']['backend'],'metal')
  for mutate in (lambda s:s.update(backend='cpu'),lambda s:s.update(cpu_projection='scalar'),lambda s:s['execution'].update(projection_mode='Original'),lambda s:s['logical_memory'].update(weights_bytes=1)):
   case,e=self.fixture();mutate(next(v['statistics'] for v in e if v['event']=='done'))
   with self.assertRaises(ValueError):m.collect_response(case,e,self.admission())
 def test_probe_cap_and_full_case_caps_are_distinct(self):
  case,e=self.fixture(cap=8);m.collect_response(case,e,self.admission(),8)
  with self.assertRaises(ValueError):m.collect_response(case,e,self.admission())
  with self.assertRaises(ValueError):m.collect_response(case,e,self.admission(),True)
 def test_error_reset_state_timers_and_visible_eos_reject(self):
  for mutate in (lambda e:e.insert(0,{'event':'error'}),lambda e:e.pop(),lambda e:e[0].update(token_id=151645),lambda e:e[0].update(index=True),lambda e:next(v['statistics'] for v in e if v['event']=='done')['generation'].update(reason={'kind':'cancelled'}),lambda e:next(v['statistics'] for v in e if v['event']=='done')['generation'].update(evaluated_tokens=0),lambda e:next(v['statistics'] for v in e if v['event']=='done')['generation'].update(decode_seconds=float('nan'))):
   case,e=self.fixture();mutate(e)
   with self.assertRaises(ValueError):m.collect_response(case,e,self.admission())
 def test_independent_finalization_requires_exit_and_exact_text(self):
  r={'visible_token_ids':[1],'visible_text':'A','visible_text_sha256':'hash'}
  with patch.object(m.reference,'finalize') as decode:
   with self.assertRaisesRegex(ValueError,'exit/reaping'):m.finalized([r],Path('tokenizer.json'),model_exited=False)
   decode.assert_not_called();decode.return_value=[{'visible_text':'B','visible_text_sha256':'different'}]
   with self.assertRaisesRegex(ValueError,'differs'):m.finalized([r],Path('tokenizer.json'),model_exited=True)
 def test_unqualified_cli_rejects_before_model(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'qualification.json';path.write_text('{}')
   with patch.object(m,'QUALIFICATION',path),self.assertRaisesRegex(ValueError,'qualification differs'):m.qualified_cli()
  with patch.object(m.metal_engine,'digest',return_value='0'*64),self.assertRaisesRegex(ValueError,'binary differs'):m.qualified_cli()
 def test_receipt_replaced_after_parse_rejects_before_worker(self):
  original=m.metal_engine.digest;payload=m.QUALIFICATION.read_bytes();manifest=m.controlled.prepare.load_json(m.controlled.prepare.CONTRACT/'manifest.json');spec=m.controlled.MODELS['small'];source=m.ROOT/'models'/spec[0]
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);q=root/'qualification.json';q.write_bytes(payload)
   def digest(path):
    if path==m.ROOT/'artifacts/m5-cli-half-native-binary-v1/toklet':q.write_text('{}')
    if path==source/'model.safetensors':return spec[1]
    if path==source/'config.json':return spec[2]
    if path.parent==source:return m.cli.ASSET_PINS[path.name]
    if path==m.ROOT/manifest['quality']['tasks_file']:return m.tasks.TASK_SHA
    return original(path)
   with patch.object(m,'QUALIFICATION',q),patch.object(m.metal_engine,'digest',side_effect=digest),patch.object(m.tasks,'load_cases',return_value=inventory()[0]),patch.object(m,'child') as child:
    with self.assertRaisesRegex(ValueError,'changed after parsed'):m.run(root/'output.json','small',True)
   child.assert_not_called()
 def test_bad_ready_reaps_owned_worker(self):
  real=subprocess.Popen;owned=[]
  def launch(command,**kw):
   self.assertTrue(kw['start_new_session']);p=real([sys.executable,'-c','print(\'{"event":"wrong"}\',flush=True);import time;time.sleep(60)'],**kw);p.wait=Mock(wraps=p.wait);owned.append(p);return p
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   with (root/'raw').open('x') as raw,(root/'stderr').open('xb') as stderr,patch.object(m.subprocess,'Popen',launch),patch.object(m,'finalized') as decode:r=m.child(root/'binary',root,'small',inventory()[0],raw,stderr,True)
   decode.assert_not_called()
  self.assertFalse(r['passed']);self.assertFalse(r['model_exited']);self.assertEqual(r['cleanup_errors'],[]);self.assertIsNotNone(owned[0].returncode);owned[0].wait.assert_called_once_with(timeout=5)

if __name__=='__main__':unittest.main()
