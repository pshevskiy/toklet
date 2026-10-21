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
if __name__=='__main__':unittest.main()
