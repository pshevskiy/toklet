from pathlib import Path
import importlib.util,tempfile,unittest,json
p=Path(__file__).with_name('probe.py');s=importlib.util.spec_from_file_location('probe',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class Checks(unittest.TestCase):
 def test_only_observed_exact_target_pid(self):
  data=dict(info=dict(outcome='success',commandType='devicectl.device.info.processes'),result=dict(deviceIdentifier='device',runningProcesses=[dict(processIdentifier=42,executable='/owned/TokletHarness')]))
  self.assertEqual(a.target_pid(data,dict(device_identifier='device'),'/owned'),42)
  data['result']['runningProcesses'][0]['executable']='/other/TokletHarness'
  with self.assertRaises(ValueError):a.target_pid(data,dict(device_identifier='device'),'/owned')
 def test_no_pid_or_wrong_device_rejected(self):
  data=dict(info=dict(outcome='success',commandType='devicectl.device.info.processes'),result=dict(deviceIdentifier='wrong',runningProcesses=[]))
  with self.assertRaises(ValueError):a.target_pid(data,dict(device_identifier='device'),'/owned')
 def test_bounded_attach_without_other_targets(self):
  args=a.trace_args('private',42);self.assertEqual(args[args.index('--attach')+1],'42');self.assertEqual(args[args.index('--time-limit')+1],'10s');self.assertIn('--no-prompt',args);self.assertNotIn('--all-processes',args)
  for p in (True,0,-1):
   with self.assertRaises(ValueError):a.trace_args('private',p)
 def test_complete_bound_marker(self):
  with tempfile.TemporaryDirectory() as f:
   p=Path(f)/'console';self.assertFalse(a.marker(p,{}));raw='TOKLET_INSTALLATION_JSON='+json.dumps(dict(phase='before',simulator=False,files={}))
   p.write_text(raw);self.assertFalse(a.marker(p,{}));p.write_text(raw+'\n');self.assertTrue(a.marker(p,{}))
   with self.assertRaises(ValueError):a.marker(p,{'changed':'a'*64})
class WholeProbeChecks(unittest.TestCase):
 def run_probe(self,kind='success'):
  import threading,hashlib,copy
  from contextlib import ExitStack
  from unittest.mock import patch
  with tempfile.TemporaryDirectory(dir=a.R/'artifacts') as f:
   root=Path(f);out=root/'probe';out.mkdir();own=out/'probe.py';own.write_bytes(p.read_bytes());own.with_name('checks.py').write_text('whole probe fixture');run=root/'native-run';trace=out/'private-trace.trace';done=threading.Event();live={'value':True};selected=dict(device_identifier='private-device',device_udid='private-udid',development_team='private-team');remote='/installed/TokletHarness.app'
   def owned(args,stdout,stderr,env,timeout):
    label=stdout.stem;error=None;exit=0;data={}
    if label=='native-state-child':
     (run/'private-logs').mkdir(parents=True);(run/'private-logs/measurement.stdout').write_text('TOKLET_INSTALLATION_JSON='+json.dumps(dict(phase='before',simulator=False,files={}))+'\n');done.wait(2)
     (root/'native-source').write_text('source');(run/'private-evidence').write_text('native raw');h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();r=dict(passed=kind=='success',bindings={'native-source':h(root/'native-source')},private_evidence_pins={'native-run/private-evidence':h(run/'private-evidence')});(run/'measurement.json').write_text(json.dumps(r))
     if kind=='success':live['value']=False
     elif kind=='interrupt':raise InterruptedError('Injected parent interruption')
     else:exit=-9;error='TimeoutExpired: injected child timeout'
    elif label=='power-profiler':trace.mkdir();(trace/'sample').write_bytes(b'private trace');done.set()
    elif 'signal' in args:
     if kind=='cleanup-failure':exit=1;error='tool exit 1'
     else:live['value']=False
    elif 'apps' in args:data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='private-device',matchingBundleIdentifier=a.m.physical.BUNDLE,apps=[dict(bundleIdentifier=a.m.physical.BUNDLE,builtByDeveloper=True,url=remote)]))
    elif 'processes' in args:data=dict(info=dict(outcome='success',commandType='devicectl.device.info.processes'),result=dict(deviceIdentifier='private-device',runningProcesses=[dict(processIdentifier=42,executable=remote+'/TokletHarness')] if live['value'] else []))
    stdout.write_text(json.dumps(data));stderr.write_bytes(b'');return dict(command=args,exit_code=exit,primary_error=error,cleanup_errors=[],pid=123,stdout=str(stdout.relative_to(root)),stderr=str(stderr.relative_to(root)))
   def verify(pins):
    for n,h in pins.items():self.assertEqual(hashlib.sha256((root/n).read_bytes()).hexdigest(),h,n)
   with ExitStack() as stack:
    for name,val in dict(R=root,OUT=out,OWN=own,RUN=run,TRACE=trace,PRIVATE=root/'private',PREP=root/'prepared',READY=root/'ready',NATIVE=root/'native').items():stack.enter_context(patch.object(a,name,val))
    stack.enter_context(patch.object(a.m.package,'owned',side_effect=owned));stack.enter_context(patch.object(a.m.package,'build_environment',return_value=({},[])));stack.enter_context(patch.object(a.m.physical,'selection',return_value=selected));stack.enter_context(patch.object(a.m.physical,'admit_reuse',return_value=(root/'app',{},remote,{})));stack.enter_context(patch.object(a.state,'job',return_value=({}, {}, {})));stack.enter_context(patch.object(a.m,'admit_ready',return_value=({},'job','a'*64)));stack.enter_context(patch.object(a.m,'verify_inputs',side_effect=verify));a.main()
   r=json.loads((out/'attempt.json').read_text());pr=json.loads((out/'private-execution.json').read_text());self.assertTrue(r['attempt_completed']);self.assertEqual(r['private_trace_files'],1);self.assertTrue(r['child_evidence'] is not None);self.assertIn('native-source',r['bindings']);self.assertIn('native-run/private-evidence',r['bindings']);self.assertFalse(r['energy_qualified']);self.assertIsNone(r['energy_joules']);self.assertEqual(len(pr['trace_inventory']),1)
   self.assertNotIn('private-device',json.dumps(r));self.assertNotIn('private-udid',json.dumps(r));return r,pr
 def test_success_binds_trace_native_and_independent_absence(self):
  r,_=self.run_probe();self.assertEqual(r['primary_errors'],[]);self.assertEqual(r['endpoint_errors'],[]);self.assertTrue(r['independent_phone_cleanup']['absent']);self.assertEqual(r['independent_phone_cleanup']['signals'],[])
 def test_timeout_independently_cleans_phone_and_preserves_native_failure(self):
  r,_=self.run_probe('timeout');self.assertTrue(r['independent_phone_cleanup']['absent']);self.assertEqual(len(r['independent_phone_cleanup']['signals']),1);self.assertGreaterEqual(len(r['primary_errors']),2);self.assertFalse(r['child_evidence']['passed'])
 def test_interruption_independently_cleans_phone_and_keeps_errors(self):
  r,_=self.run_probe('interrupt');self.assertTrue(r['independent_phone_cleanup']['absent']);self.assertTrue(any('Injected parent interruption' in x for x in r['primary_errors']));self.assertTrue(any('Native receipt failed' in x for x in r['primary_errors']))
 def test_cleanup_failure_does_not_claim_absence(self):
  r,_=self.run_probe('cleanup-failure');self.assertFalse(r['independent_phone_cleanup']['absent']);self.assertTrue(r['endpoint_errors'])
 def test_dangling_output_link_rejected_before_children(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as f:
   out=Path(f);run=out/'missing';run.symlink_to(out/'absent')
   with patch.object(a,'OUT',out),patch.object(a,'RUN',run),patch.object(a.m.package,'owned') as owned,self.assertRaises(FileExistsError):a.main()
   owned.assert_not_called()
if __name__=='__main__':unittest.main()
