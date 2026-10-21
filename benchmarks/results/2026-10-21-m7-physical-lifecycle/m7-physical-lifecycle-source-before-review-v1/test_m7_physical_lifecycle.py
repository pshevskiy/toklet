import copy,json,tempfile,unittest,plistlib,hashlib,subprocess
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import m7_physical_lifecycle as p
import test_m7_native_lifecycle_app_check as prior

def fixture(returned=True):
 config,data=prior.fixture();r=data['diagnostic'];events=r['trace']['events']
 for e in events:
  e['applicationState']=2 if e['name']=='scene-background' else 1 if e['name']=='scene-inactive' else 0
  if e['name'].startswith('scene-'):e['origin']='scenePhase'
 r['ready']['applicationState']=0;r['environmentBefore']=dict(applicationState=0,thermalState=0,lowPowerMode=False);r['environmentAfter']=dict(r['environmentBefore'])
 if not returned:
  events.pop(4)
  for i,e in enumerate(events):e['index']=i
 for key,name,index in [('inactive','scene-inactive',2),('background','scene-background',3),('active','scene-active',5 if returned else 4)]:r['roundTrip'][key]=copy.deepcopy(events[index])
 r['roundTrip']['returnInactive']=copy.deepcopy(events[4]) if returned else None
 return config,r

def text(r,signed=None):
 return '\n'.join(['TOKLET_INSTALLATION_JSON='+json.dumps(dict(phase='before',simulator=False,files=signed or {})),
 'TOKLET_LIFECYCLE_READY_JSON='+json.dumps(r['ready']),'TOKLET_INSTALLATION_JSON='+json.dumps(dict(phase='after',simulator=False,files=signed or {})),
 'TOKLET_LIFECYCLE_JSON='+json.dumps(r)])
class Checks(unittest.TestCase):
 def reject(self,change):
  c,r=fixture();change(c,r)
  with self.assertRaises((ValueError,KeyError,TypeError,IndexError)):p.validate_result(r,c)
 def test_actual_os_direct_and_inactive_return(self):
  for returned in (False,True):
   c,r=fixture(returned);self.assertIs(p.validate_result(r,c),r);self.assertEqual(p.validate_console(text(r),c,{})['result'],r)
 def test_system_provenance_cannot_be_synthetic(self):
  def change(c,r):
   r['trace']['events'][2]['origin']='syntheticCheck';r['roundTrip']['inactive']=copy.deepcopy(r['trace']['events'][2])
  self.reject(change);self.reject(lambda c,r:r['trace']['events'][12].update(origin='systemNotification'))
 def test_ready_must_precede_busy_foreground_interruption(self):
  self.reject(lambda c,r:r['ready'].update(applicationState=2));self.reject(lambda c,r:r['ready'].update(eventIndex=True));self.reject(lambda c,r:r['ready'].update(observedUptimeNanoseconds=100))
  def stopped(c,r):
   r['trace']['events'][2]['before']['busy']=False;r['roundTrip']['inactive']=copy.deepcopy(r['trace']['events'][2])
  self.reject(stopped)
 def test_trace_binding_prefix_and_exact_replay(self):
  self.reject(lambda c,r:r['roundTrip']['active'].update(index=2));self.reject(lambda c,r:r.update(referenceIDs=[46]*3));self.reject(lambda c,r:r.update(rebuiltIDs=[46,47]));self.reject(lambda c,r:r['trace']['events'][6]['state'].update(hasSession=True))
 def test_memory_identity_and_close(self):
  self.reject(lambda c,r:r['checkpoints'].pop());self.reject(lambda c,r:r['checkpoints'][2]['footprint'].update(lifetimePeakPhysicalBytes=c['processBudget']+1));self.reject(lambda c,r:r.update(filesAfter={}));self.reject(lambda c,r:r['trace']['events'][-1]['state'].update(workerFinished=False));self.reject(lambda c,r:r['trace']['events'][0].update(applicationState=False))
 def test_marker_duplicates_unknown_and_ready_binding(self):
  c,r=fixture();raw=text(r)
  for changed in (raw+'\nTOKLET_ERROR=failed',raw+'\nTOKLET_LIFECYCLE_JSON='+json.dumps(r),raw.replace('"submittedUptimeNanoseconds": 1','"submittedUptimeNanoseconds": 2',1)):
   with self.assertRaises(ValueError):p.validate_console(changed,c,{})
 def test_partial_ready_and_duplicates(self):
  with tempfile.TemporaryDirectory() as f:
   path=Path(f)/'console';path.write_bytes(b'TOKLET_LIFECYCLE_READY_JSON={"index":1}')
   self.assertIsNone(p.peek_ready(path));path.write_bytes(path.read_bytes()+b'\n');self.assertEqual(p.peek_ready(path),dict(index=1))
   path.write_bytes(path.read_bytes()*2)
   with self.assertRaises(ValueError):p.peek_ready(path)
   path.write_bytes(b'TOKLET_LIFECYCLE_ERROR=failed\n')
   with self.assertRaises(ValueError):p.peek_ready(path)
 def test_same_pid_activation_and_settings_scope(self):
  selected=dict(device_identifier='device')
  def data(path,pid=42):return dict(info=dict(outcome='success',commandType='devicectl.device.process.launch'),result=dict(deviceIdentifier='device',process=dict(processIdentifier=pid,executable=path)))
  self.assertEqual(p.activation(data('/Applications/Preferences.app/Preferences'),selected),42)
  self.assertEqual(p.activation(data('/installed/TokletHarness.app/TokletHarness'),selected,'/installed/TokletHarness.app',42),42)
  for value in (data('/other/TokletHarness.app/TokletHarness'),data('/installed/TokletHarness.app/TokletHarness',43),data('/installed/TokletHarness.app/TokletHarness',True)):
   with self.assertRaises(ValueError):p.activation(value,selected,'/installed/TokletHarness.app',42)
  with self.assertRaises(ValueError):p.activation(data('/Applications/Other.app/Other'),selected)
 def test_modes_reject_before_children(self):
  args=(Path('/native'),Path('/private'),'a'*64,Path('/prepared'),'b'*64,'small','512',Path('/output'))
  with patch.object(p.m.package,'owned') as child:
   for options in (dict(lifecycle=True),dict(lifecycle=True,prepare_only=True,tasks=True),dict(lifecycle=True,prepare_only=True,quality=True)):
    with self.assertRaises(ValueError):p.m.main(*args,**options)
   child.assert_not_called()
 def test_job_is_exact_and_preserves_app_source_pins(self):
  c,_=fixture();base=dict(c,version=1,modelFolder='M7Inputs/small/Model',packedFolder='',chunk=3,expectedPrefix=[],warmups=3,repetitions=10,minimumNativeSeconds=0)
  with patch.object(p.m,'job',return_value=(base,{},{})):
   job,host,pins=p.job('small');self.assertEqual(set(job),{'version','modelFolder','packedFolder','files','metal','capacity','chunk','numericalBudget','processBudget','prompt','maximum','vocabulary'});self.assertEqual(job['vocabulary'],151936);self.assertIn(p.APP_PROOF,pins)
   proof=json.loads((p.ROOT/p.APP_PROOF).read_text());self.assertTrue(set(proof['code_pins']).issubset(pins))
  for model,workload in [('tiny-cpu','512'),('small','short'),('larger','1024')]:
   with self.assertRaises(ValueError):p.job(model,workload)
 def controller(self,fail=None):
  with tempfile.TemporaryDirectory(dir=p.ROOT/'artifacts') as f:
   root=Path(f);native=root/'native';native.mkdir();private=root/'private.json';private.write_text('{}');prepared=root/'prepared';prepared.mkdir();ready=root/'ready';ready.mkdir();logs=ready/'private-logs';logs.mkdir()
   for n in ('profile.stdout','entitlements.stdout'):(logs/n).write_bytes(plistlib.dumps({}))
   (logs/'signature-details.stderr').write_text('signature');output=root/'output'
   config,r=fixture();config.update(version=1,modelFolder='M7Inputs/small/Model',packedFolder='',chunk=3)
   selected=dict(device_identifier='device-private-123',device_udid='udid-private-456',development_team='team-private-789');remote='/installed/secret-UUID/TokletHarness.app';state=dict(alive=False,wait_failed=False);calls=[]
   def process_data():return dict(info=dict(outcome='success',commandType='devicectl.device.info.processes'),result=dict(deviceIdentifier=selected['device_identifier'],runningProcesses=[dict(processIdentifier=42,executable=remote+'/TokletHarness')] if state['alive'] else []))
   def owned(args,out,err,env,timeout):
    calls.append(args)
    if 'processes' in args:data=process_data()
    elif 'signal' in args:state['alive']=False;data={}
    elif 'launch' in args:
     system='com.apple.Preferences' in args;actual=43 if fail=='pid' and not system else 42
     data=dict(info=dict(outcome='success',commandType='devicectl.device.process.launch'),result=dict(deviceIdentifier=selected['device_identifier'],process=dict(processIdentifier=actual,executable='/Applications/Preferences.app/Preferences' if system else remote+'/TokletHarness')))
    else:data={}
    out.write_text(json.dumps(data));err.write_bytes(b'')
    return dict(command=args,exit_code=0,primary_error=None,cleanup_errors=[],pid=99999,stdout=str(out.relative_to(p.ROOT)),stderr=str(err.relative_to(p.ROOT)))
   class Process:
    pid=88888
    def __init__(self,args,**kwargs):
     calls.append(args);state['alive']=True;kwargs['stdout'].write(text(r).encode());kwargs['stdout'].flush()
     metadata=Path(args[args.index('--json-output')+1]);metadata.write_text(json.dumps(dict(info=dict(outcome='success'),result=dict(deviceIdentifier=selected['device_identifier'],process=dict(processIdentifier=42,executable=remote+'/TokletHarness'),terminationResult=dict(exitCode=0,wasCoreDumpCreated=False)))))
    def poll(self):return None if state['alive'] else 0
    def wait(self,timeout):
     if fail=='timeout' and not state['wait_failed']:
      state['wait_failed']=True;raise subprocess.TimeoutExpired(['devicectl',*selected.values(),remote],timeout)
     state['alive']=False;return 0
   with ExitStack() as stack:
    for target,value in [(p.m.package,'owned'),(p.m.package,'build_environment'),(p.m.integration,'admit_build'),(p.m.physical,'selection'),(p.m.physical,'admit_reuse'),(p.m,'admit_ready'),(p.m.integration,'metadata'),(p.m,'job'),(p,'job'),(p.m.storage,'admit_ready'),(p.m.physical,'admit_profile'),(p.m.physical,'admit_device'),(p.m.physical,'launch_verified'),(p.m.physical,'inventory'),(p.m.package,'drain_group')]:
     mock=stack.enter_context(patch.object(target,value))
     if value=='owned':mock.side_effect=owned
     elif value=='build_environment':mock.return_value=({},None)
     elif value=='admit_build':mock.return_value={}
     elif value=='selection':mock.return_value=selected
     elif value=='admit_reuse':mock.return_value=(prepared,{},remote,{})
     elif value=='admit_ready' and target is p.m:mock.return_value=({},'M7Jobs/'+'a'*64+'.json','a'*64)
     elif value=='metadata':mock.return_value={}
     elif value=='job':mock.return_value=(config,{},{} )
     elif value=='admit_ready':mock.return_value=({}, {})
     elif value in ('admit_profile','admit_device','inventory'):mock.return_value={}
     elif value=='launch_verified':mock.side_effect=lambda current,action,*_: (current(),action())[1]
    stack.enter_context(patch('subprocess.Popen',Process));stack.enter_context(patch.object(p.time,'sleep'))
    code=p.collect(native,private,hashlib.sha256(private.read_bytes()).hexdigest(),prepared,'b'*64,ready,'c'*64,'small',output)
   result=json.loads((output/'lifecycle.json').read_text());private_result=json.loads((output/'private-execution.json').read_text())
   public=json.dumps(result)
   for secret in (*selected.values(),remote):self.assertNotIn(secret,public)
   for call in calls:self.assertNotIn('--terminate-existing',call)
   for n,h in result['bindings'].items():self.assertEqual(p.m.package.sha(p.ROOT/n),h,n)
   return code,result,private_result,calls
 def test_controller_same_pid_roundtrip_and_clean_capture(self):
  code,r,private,calls=self.controller();self.assertEqual(code,0);self.assertTrue(r['passed']);self.assertEqual(r['checked']['launch']['pid'],42)
  self.assertEqual([v['phase'] for v in r['phase_observations']],['ready-consumed','ready-process','settings-activated','background-process','harness-activated'])
  self.assertEqual(len(r['remote_cleanup']),2);self.assertTrue(all(not v['signals'] for v in r['remote_cleanup']))
  self.assertEqual(len(calls),11);self.assertTrue(all(c['exit_code']==0 for c in private['commands']))
 def test_controller_rejects_new_foreground_pid_and_cleans_owned(self):
  code,r,private,calls=self.controller('pid');self.assertEqual(code,1);self.assertFalse(r['passed']);self.assertIn('activated process identity',r['primary_error']);self.assertTrue(r['remote_cleanup'][-1]['absent']);self.assertTrue(r['remote_cleanup'][-1]['signals'])
 def test_controller_timeout_is_private_and_owned_resources_clean(self):
  code,r,private,calls=self.controller('timeout');self.assertEqual(code,1);self.assertFalse(r['passed']);self.assertIn('device-private-123',private['primary_error']);self.assertIn('<private-device_identifier>',r['primary_error']);self.assertTrue(r['remote_cleanup'][-1]['absent'])
if __name__=='__main__':unittest.main()
