import copy,json,unittest,tempfile,plistlib
from contextlib import ExitStack
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
 def controller(self,reject_console=False):
  with tempfile.TemporaryDirectory(dir=p.ROOT/'artifacts') as f:
   root=Path(f);private=root/'private.json';private.write_text('{}');private_sha=p.m.package.sha(private)
   payload=root/'payload';payload.write_bytes(b'known');target='M7Inputs/small/Model/config.json';pin=p.m.package.sha(payload)
   c,result=fixture();c.update(version=1,modelFolder='M7Inputs/small/Model',packedFolder='',files={target:pin});result.update(filesBefore=c['files'],filesAfter=c['files'])
   host={target:str(payload.relative_to(p.ROOT))};inputs={str(payload.relative_to(p.ROOT)):pin};remote='/installed/TokletHarness.app';signed={'TokletHarness':'b'*64};selected=dict(device_identifier='selected');calls=[]
   def owned(args,out,err,env,timeout):
    label=out.stem;calls.append((label,args));data={}
    if label=='host-boot':out.write_text('00000000-0000-4000-8000-000000000001\n')
    elif label in ('profile','entitlements'):out.write_bytes(plistlib.dumps({}))
    elif label=='measurement':
     value=copy.deepcopy(result)
     if reject_console:value['rows'].pop();private.write_text('{"changed":true}')
     out.write_text(text(value,signed));Path(args[args.index('--json-output')+1]).write_text('{}')
    else:
     if label.startswith('process-'):data=dict(info=dict(outcome='success',commandType='devicectl.device.info.processes'),result=dict(deviceIdentifier='selected',runningProcesses=[]))
     elif label.startswith('current-install-'):data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='selected',matchingBundleIdentifier=p.m.physical.BUNDLE,apps=[dict(bundleIdentifier=p.m.physical.BUNDLE,builtByDeveloper=True,url=remote)]))
     elif label=='copied-inventory':data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='selected',domain='appDataContainer',domainIdentifier=p.m.physical.BUNDLE,files=[dict(name='Model/config.json',relativePath='Model/config.json',resources=dict(isDirectory=False,isSymbolicLink=False),metadata=dict(size=5))]))
     out.write_text(json.dumps(data))
    err.write_text('');return dict(command=args,exit_code=0,primary_error=None,cleanup_errors=[],pid=1,stdout=str(out.relative_to(p.ROOT)),stderr=str(err.relative_to(p.ROOT)))
   real_stage=p.m.stage;real_admit=p.m.admit_ready;real_validate=p.validate_console
   with ExitStack() as stack:
    for obj,name,value in [(p.m.physical,'selection',selected),(p.m.integration,'admit_build',{}),(p.m.physical,'admit_reuse',(root,{},remote,signed)),(p.m.physical,'admit_device',{}),(p.m.physical,'admit_profile',{}),(p.m.physical,'inventory',signed),(p.m.physical,'check_launch',{}),(p.m.storage,'admit_proof',({'device':{}},{})),(p.m.storage,'decision',{'checks':[]}),(p.m.storage,'fresh',{}),(p.m.storage,'admit_ready',({'checks':[]},{}))]:stack.enter_context(patch.object(obj,name,return_value=value))
    stack.enter_context(patch.object(p,'job',return_value=(c,host,inputs)));staged=stack.enter_context(patch.object(p,'stage',side_effect=lambda config,host,out,*_:real_stage(config,host,out)))
    admitted=stack.enter_context(patch.object(p.m,'admit_ready',wraps=real_admit));validated=stack.enter_context(patch.object(p,'validate_console',wraps=real_validate));stack.enter_context(patch.object(p.m.package,'owned',side_effect=owned));stack.enter_context(patch('builtins.print'))
    ready=root/'ready';code=p.m.main(root,private,private_sha,root,'prepared-sha','small','512',ready,prepare_only=True,storage_proof=root/'proof',storage_sha='proof-sha',native_sha=p.NATIVE_SHA,state=True)
    self.assertEqual(code,0);prepared=json.loads((ready/'measurement.json').read_text());self.assertEqual(prepared['format'],p.READY_FORMAT);self.assertIsNone(prepared['checked']);self.assertFalse(any(label=='measurement' for label,_ in calls));self.assertEqual(staged.call_count,1)
    self.assertEqual([label for label,_ in calls if label.startswith('copy-')],['copy-Model','copy-job']);before=len(calls);digest=p.m.package.sha(ready/'measurement.json')
    output=root/'run';code=p.m.main(root,private,private_sha,root,'prepared-sha','small','512',output,transfer=ready,transfer_sha=digest,native_sha=p.NATIVE_SHA,state=True)
    data=json.loads((output/'measurement.json').read_text());self.assertEqual(data['format'],p.RESULT_FORMAT);self.assertEqual(code,1 if reject_console else 0);self.assertEqual(admitted.call_count,1);self.assertEqual(admitted.call_args.args[-1],p.READY_FORMAT);self.assertEqual(staged.call_count,1);self.assertEqual(validated.call_count,1)
    later=calls[before:];self.assertFalse(any(label.startswith('copy-') or label in ('profile','signature-verify','entitlements','signature-details') for label,_ in later));launch=next(args for label,args in later if label=='measurement');self.assertEqual(launch[-4:],['--','--state',prepared['job_receipt']['relative_path'],prepared['job_receipt']['sha256']])
    natural=dict(absent=True,observations=[[]],signals=[],errors=[]);self.assertEqual(data['remote_cleanup'],[natural,natural]);self.assertEqual([label for label,_ in later if label.startswith('process-')],['process-1','process-2']);self.assertIn('device-after',[label for label,_ in later])
    if reject_console:self.assertTrue(data['primary_error']);self.assertTrue(data['endpoint_errors']);self.assertIsNone(data['checked'])
    else:self.assertTrue(data['passed']);self.assertEqual(data['endpoint_errors'],[]);self.assertEqual(data['checked']['result'],result)
 def test_shared_main_state_prepare_and_authenticated_reuse(self):self.controller()
 def test_shared_main_rejected_state_keeps_cleanup_and_endpoints(self):self.controller(reject_console=True)
 def test_explicit_real_model_scope_before_reads(self):
  with patch.object(p.m,'job') as base:
   for model,workload in [('tiny-cpu','512'),('tiny-metal','512'),('small','short'),('larger','1024')]:
    with self.assertRaises(ValueError):p.job(model,workload)
   base.assert_not_called()
if __name__=='__main__':unittest.main()
