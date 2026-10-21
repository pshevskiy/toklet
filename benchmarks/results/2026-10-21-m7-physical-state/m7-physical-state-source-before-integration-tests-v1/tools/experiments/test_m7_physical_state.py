import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import m7_physical_state as p
import test_m7_native_state_check as prior

def fixture():
 c,d=prior.fixture();c.update(metal=True,maximum=128);r=d['diagnostic']
 for row in r['rows']:row['environment']=dict(applicationState=0,thermalState=0,lowPowerMode=False,batteryState=2,batteryLevel=1)
 return c,r

def text(r,signed=None):
 return '\n'.join(['TOKLET_INSTALLATION_JSON='+json.dumps(dict(phase=phase,simulator=False,files=signed or {})) for phase in ('before','after')]+['TOKLET_STATE_JSON='+json.dumps(r)])
class Checks(unittest.TestCase):
 def reject(self,mutate):
  c,r=fixture();mutate(c,r)
  with self.assertRaises((ValueError,KeyError,TypeError)):p.validate_console(text(r),c,{})
 def test_complete_actual_result_and_prefix_witnesses(self):
  c,r=fixture();self.assertEqual(p.validate_console(text(r),c,{})['result'],r)
 def test_component_wrapper_not_used_as_device_result(self):
  c,r=fixture();_,wrapped=prior.fixture()
  with self.assertRaises((KeyError,ValueError)):p.validate_console(text(wrapped),c,{})
 def test_real_partial_prefill_and_capacity_required(self):
  self.reject(lambda c,r:r.update(prefillWitness=False))
  self.reject(lambda c,r:next(v for v in r['rows'] if v['name']=='prefill-cancelled')['state'].update(evaluated=512,pending=0))
  self.reject(lambda c,r:c.update(capacity=1151))
  self.reject(lambda c,r:c.update(chunk=64))
 def test_actual_environment_typed_and_foreground(self):
  for field,value in [('applicationState',None),('applicationState',False),('applicationState',2),('thermalState',True),('thermalState',4),('lowPowerMode',0),('batteryState',0),('batteryLevel',1.1)]:
   self.reject(lambda c,r:r['rows'][0]['environment'].update({field:value}))
 def test_install_order_identity_and_duplicates(self):
  c,r=fixture();raw=text(r)
  for value in (raw+'\nTOKLET_ERROR=failed',raw+'\nTOKLET_STATE_JSON='+json.dumps(r),raw.replace('"simulator": false','"simulator": true',1),raw.replace('"phase": "before"','"phase": "after"',1)):
   with self.assertRaises(ValueError):p.validate_console(value,c,{})
  with self.assertRaises(ValueError):p.validate_console(raw,c,{'TokletHarness':'a'*64})
 def test_job_pins_full_current_app_and_pure_replay_source(self):
  c,r=fixture();base=dict(c,version=1,modelFolder='M7Inputs/small/Model',packedFolder='')
  proof=dict(passed=True,app_integration=True,review='Astra accepted source and completed app component evidence with no findings',actual_os_lifecycle_qualified=False,physical_device_qualified=False,app_code_pins={'apple/Harness/NativeStateRun.swift':'a'*64})
  with patch.object(p.m,'job',return_value=(base,{},{})),patch.object(p.m.integration,'metadata',return_value=proof),patch.object(p.m,'verify_inputs') as verify:
   job,host,pins=p.job('small');self.assertEqual(set(job),{'version','modelFolder','packedFolder','files','metal','capacity','chunk','numericalBudget','processBudget','prompt','maximum','vocabulary'});verify.assert_called_once_with(proof['app_code_pins']);self.assertIn(p.APP_PROOF,pins);self.assertIn('tools/experiments/m7_native_state_check.py',pins)
 def test_stale_app_sources_or_unaccepted_proof_reject(self):
  c,r=fixture();proof=dict(passed=True,app_integration=True,review='Astra accepted source and completed app component evidence with no findings',actual_os_lifecycle_qualified=False,physical_device_qualified=False,app_code_pins={})
  with patch.object(p.m,'job',return_value=(c,{},{})),patch.object(p.m.integration,'metadata',return_value=proof),patch.object(p.m,'verify_inputs',side_effect=ValueError('Changed source')):
   with self.assertRaises(ValueError):p.job('small')
  for field,value in [('passed',False),('physical_device_qualified',True),('review','pending')]:
   changed=dict(proof);changed[field]=value
   with patch.object(p.m,'job',return_value=(c,{},{})),patch.object(p.m.integration,'metadata',return_value=changed),self.assertRaises(ValueError):p.job('small')
 def test_modes_reject_before_children(self):
  args=[Path('artifacts/x'),Path('artifacts/private'),'a'*64,Path('artifacts/prepared'),'b'*64,'small','512',Path('artifacts/out')]
  with patch.object(p.m.package,'owned') as child:
   for options in (dict(state=True,lifecycle=True),dict(state=True,tasks=True),dict(state=True,quality=True)):
    with self.assertRaises(ValueError):p.m.main(*args,**options)
   child.assert_not_called()
 def test_explicit_real_model_scope_before_reads(self):
  with patch.object(p.m,'job') as base:
   for model,workload in [('tiny-cpu','512'),('tiny-metal','512'),('small','short'),('larger','1024')]:
    with self.assertRaises(ValueError):p.job(model,workload)
   base.assert_not_called()
if __name__=='__main__':unittest.main()
