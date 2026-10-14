"""Offline frozen workload, complete state/mode/timing and owned cleanup checks."""
import copy
import io
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import metal_controlled as ctl

class Controlled(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.manifest,cls.arrays=ctl.contract()
 def admission(self,model='small'):
  m={k:1 for k in ('weights_bytes','kv_bytes','scratch_bytes','history_bytes','logits_bytes','metadata_bytes','metadata_allowance_bytes')};m['weights_bytes']=ctl.MODELS[model][5];m['logits_bytes']=151936*8;m['admitted_bytes']=sum(m.values())
  e={'event':'ready','format':'toklet-metal-controlled-v1','debug_assertions':False,'backend':'metal','projection_mode':'HalfCompensated','normalization_mode':'Original','readback_mode':'FullLogits','weight_dtype':'FP16','kv_dtype':'FP16','half_compensated_execution_widths':[32,32],'hidden_size':ctl.MODELS[model][3],'num_layers':ctl.MODELS[model][4],'vocab_size':151936,'context_capacity':2048,'chunk_size':128,'application_bytes':64*1024**2,'memory':m,'memory_admitted_bytes':m['admitted_bytes']+64*1024**2,'load_seconds':1.,'conversion_seconds':.5}
  return e
 def fixture(self,name):
  setup,req,_=ctl.requests(name,self.arrays,self.manifest);base=list(self.arrays[f'prompt_{ctl.frozen.WORKLOADS[name]}']);reused=512 if name in ('append','cached','pending') else 0
  if name=='append':base+=self.arrays['append_64']
  elif name=='pending':base+=[self.manifest['continuation']['pending_sample']['pending_token']]
  elif name=='changed':base=self.arrays['changed_prompt_512']
  prefill=len(base)-reused;p={'prefill_seconds':.01,'ttft_seconds':.02,'decode_seconds':1.27,'total_seconds':1.30,'token_event_seconds':[.02+i/100 for i in range(128)],'emitted_tokens':128,'reused_tokens':reused,'prefill_tokens':prefill,'evaluated_tokens':len(base)+127,'pending_tokens':1};p['decode_seconds']=p['token_event_seconds'][-1]-p['token_event_seconds'][0]
  ids=base+self.arrays['continuation_128'];s={'history_tokens':len(ids),'evaluated_tokens':len(ids)-1,'pending_tokens':1,'history_sha256':ctl.history_sha(ids)};t={k:.01 for k in ('command_creation_seconds','clock_calibration_seconds','commit_seconds','host_encode_seconds','wait_seconds','gpu_start_seconds','gpu_end_seconds','readback_seconds')};t['command_count']=1
  return {'event':'sample','sample':p,'state':s,'memory':self.admission()['memory'],'last_completed_timing':t}
 def test_eight_frozen_workloads_and_explicit_changed_reset(self):
  for name in ctl.frozen.WORKLOADS:
   with self.subTest(name=name):ctl.sample(self.fixture(name),name,self.admission(),self.arrays,self.manifest)
   _,req,_=ctl.requests(name,self.arrays,self.manifest)
   if name=='changed':self.assertEqual(req['action'],'reset');e=self.fixture(name);e['sample']['reused_tokens']=128;e['sample']['prefill_tokens']=384
   else:continue
   with self.assertRaises(ValueError):ctl.sample(e,name,self.admission(),self.arrays,self.manifest)
 def test_actual_modes_typed_geometry_and_complete_budget(self):
  for model in ctl.MODELS:ctl.ready(self.admission(model),model)
  for key,value in [('projection_mode','Original'),('weight_dtype','FP32'),('context_capacity',True),('half_compensated_execution_widths',[32,64]),('memory_admitted_bytes',ctl.LIMIT+1)]:
   e=self.admission();e[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):ctl.ready(e,'small')
  e=self.admission();e['memory']['admitted_bytes']+=1
  with self.assertRaises(ValueError):ctl.ready(e,'small')
 def test_state_completion_timing_and_inventory_mutations_reject(self):
  for group,key,value in [('sample','pending_tokens',True),('sample','decode_seconds',float('nan')),('state','history_sha256','0'*64),('memory','kv_bytes',99),('last_completed_timing','command_count',0),('last_completed_timing','gpu_end_seconds',0)]:
   e=self.fixture('short');e[group][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):ctl.sample(e,'short',self.admission(),self.arrays,self.manifest)
 def test_result_schema_exact_repetitions_and_cached_throughput_null(self):
  obs={'compiler':'rustc','device':{'name':'MacBookPro','chip':'M1 Pro','physical_memory_bytes':16*1024**3,'os_version':'26','os_build':'test','power_mode':'AC','thermal_conditions':None}}
  setup,req,cache=ctl.requests('cached',self.arrays,self.manifest);row={'response':self.fixture('cached'),'process_peak_footprint_bytes':1};run={'workload':'cached','ready':self.admission(),'samples':[row]*13,'idle_usage':[{'worker':{'lifetime_max_phys_footprint':1}}],'cache_state':cache}
  result=ctl.result(run,'small',self.manifest,obs,'0'*40,True);self.assertIsNone(result['summary']['prefill_tokens_per_second_median']);self.assertEqual(len(result['samples']),10)
  run['samples']=run['samples'][:-1]
  with self.assertRaisesRegex(ValueError,'3 warmups/10'):ctl.result(run,'small',self.manifest,obs,'0'*40,True)
 def test_invalid_ready_reaps_owned_worker_preserves_primary_error(self):
  real=subprocess.Popen;owned=[]
  def launch(command,**kw):
   self.assertTrue(kw['start_new_session']);p=real([sys.executable,'-c','print(\'{"event":"wrong"}\',flush=True);import time;time.sleep(60)'],**kw);p.wait=Mock(wraps=p.wait);owned.append(p);return p
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   with (root/'raw').open('x') as raw,(root/'stderr').open('xb') as stderr,patch.object(ctl.subprocess,'Popen',launch):r=ctl.worker(root/'binary',root,'small','short',self.manifest,self.arrays,raw,stderr,True)
  self.assertFalse(r['passed']);self.assertIn('actual mode/geometry',r['error']);self.assertEqual(r['cleanup_errors'],[]);self.assertIsNotNone(owned[0].returncode);owned[0].wait.assert_called_once_with(timeout=5)

 def test_token_replacement_after_verify_rejects_before_child(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   for p in ctl.prepare.CONTRACT.iterdir():
    if p.is_file():shutil.copy2(p,root/p.name)
   def replace(_):
    p=root/self.manifest['arrays']['prompt_128']['file'];data=p.read_bytes();p.write_bytes(bytes(len(data)))
   with patch.object(ctl.prepare,'CONTRACT',root),patch.object(ctl.prepare,'verify',side_effect=replace),patch.object(ctl,'worker') as worker:
    with self.assertRaisesRegex(ValueError,'token bytes differ'):ctl.run(root/'binary',root/'output.json','small',True)
   worker.assert_not_called()

 def test_initial_snapshot_must_match_every_admitted_array(self):
  source=ctl.ROOT/'models'/ctl.MODELS['small'][0];m=self.manifest
  identities={str(ctl.prepare.CONTRACT/v['file']):v['sha256'] for v in m['arrays'].values()};identities.update({str(source/'model.safetensors'):ctl.MODELS['small'][1],str(source/'config.json'):ctl.MODELS['small'][2],str(source/'tokenizer.json'):'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',str(ctl.prepare.CONTRACT/'manifest.json'):ctl.CONTRACT_SHA,str(ctl.ROOT/'benchmarks/result.schema.json'):ctl.SCHEMA_SHA,str(ctl.ROOT/'benchmarks/sources.json'):m['sources_sha256'],str(ctl.ROOT/'tools/bench/requirements.lock'):m['python_dependencies_sha256']})
  ctl.initial_pins(identities,m,source,ctl.MODELS['small'])
  for key in identities:
   changed={**identities,key:'0'*64}
   with self.subTest(path=key),self.assertRaisesRegex(ValueError,'initial input'):ctl.initial_pins(changed,m,source,ctl.MODELS['small'])

if __name__=='__main__':unittest.main()
