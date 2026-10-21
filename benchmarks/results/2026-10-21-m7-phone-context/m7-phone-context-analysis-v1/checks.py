from pathlib import Path
import contextlib,copy,hashlib,importlib.util,json,tempfile,unittest
from unittest.mock import patch
P=Path(__file__).with_name('analyze.py').resolve();spec=importlib.util.spec_from_file_location('context_analysis',P);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def cleanup_fixture(first=True):
 base=a.ROOT/'artifacts/test-cleanup';remote='/var/containers/Bundle/Application/owned/TokletHarness.app';selected=dict(device_identifier='fixture-device')
 names=['current-install-6','process-1','signal-2','process-3','copy-Model','measurement','process-4'] if first else ['process-1','process-2']
 natural=dict(absent=True,observations=[[]],signals=[],errors=[])
 reset=dict(absent=True,observations=[[42],[]],signals=[dict(pid=42,signal='SIGTERM')],errors=[])
 command=[a.m.physical.DEVICECTL,'device','process','signal','--pid','42','--signal','SIGTERM','--device','<private-device_identifier>','--timeout','90','--quiet','--json-output','-']
 record=dict(remote_cleanup=[reset,natural] if first else [natural,natural],commands=[dict(stdout=str(base/'private-logs'/(n+'.stdout')),command=command if n=='signal-2' else []) for n in names])
 metadata={}
 for n in names:
  if n.startswith('process-'):
   rows=[dict(executable=remote+'/TokletHarness',processIdentifier=42)] if first and n=='process-1' else []
   metadata[n]=dict(info=dict(outcome='success',commandType='devicectl.device.info.processes'),result=dict(deviceIdentifier='fixture-device',runningProcesses=rows))
 metadata['signal-2']=dict(info=dict(outcome='success',commandType='devicectl.device.process.signal'),result=dict(deviceIdentifier='fixture-device',process=dict(executable=remote+'/TokletHarness',processIdentifier=42),signal=dict(name='SIGTERM',value=15)))
 return (a.expected_stages()[0 if first else 1],record,base,metadata,remote,selected)

class AnalysisChecks(unittest.TestCase):
 def test_exact_hash_and_json_share_one_bounded_read(self):
  with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as folder:
   path=Path(folder)/'record.json';initial=b'{"original":true}';changed=b'{"changed":true}';path.write_bytes(initial);pins={};original=Path.open
   @contextlib.contextmanager
   def opened(target,*args,**kwargs):
    with original(target,*args,**kwargs) as stream:yield stream
    if target==path and args==('rb',):
     with original(path,'wb') as stream:stream.write(changed)
   with patch.object(Path,'open',opened):raw=a.read_bound(path,100,pins)
   self.assertEqual(json.loads(raw),dict(original=True))
   self.assertEqual(pins[str(path.relative_to(a.ROOT))],hashlib.sha256(initial).hexdigest())
   with self.assertRaises(ValueError):a.verify(pins)
   with self.assertRaises(ValueError):a.read_bound(path,1,{})
 def test_exact_order_paths_and_repeat_counts_required(self):
  good=dict(exit_code=0,primary_error=None,endpoint_errors=[],results=a.expected_stages(),
            driver_bindings={n:'0'*64 for n in a.expected_driver_inputs()},
            ready_receipts={r['path']+'/measurement.json':'0'*64 for r in a.expected_stages() if r['stage']=='prepare'})
  a.require_execution(good)
  for index,change in [(0,dict(exit_code=False)),(0,dict(workload='512')),(2,dict(path=good['results'][3]['path'])),(3,dict(stage='prepare'))]:
   record=json.loads(json.dumps(good));record['results'][index].update(change)
   with self.subTest(change=change),self.assertRaises(ValueError):a.require_execution(record)
  swapped=json.loads(json.dumps(good));swapped['results'][0],swapped['results'][1]=swapped['results'][1],swapped['results'][0]
  with self.assertRaises(ValueError):a.require_execution(swapped)
  unequal=json.loads(json.dumps(good));unequal['results'][3]=unequal['results'][7]
  with self.assertRaises(ValueError):a.require_execution(unequal)
  for field in ('driver_bindings','ready_receipts'):
   missing=json.loads(json.dumps(good));missing[field].pop(next(iter(missing[field])))
   with self.subTest(field=field),self.assertRaises(ValueError):a.require_execution(missing)
 def test_consistently_failed_commands_are_rejected(self):
  good=dict(exit_code=0,primary_error=None,cleanup_errors=[])
  a.require_commands([good])
  for change in (dict(exit_code=False),dict(exit_code=1),dict(primary_error='failed'),dict(cleanup_errors=['failed cleanup'])):
   private=[dict(good,**change)];public=json.loads(json.dumps(private));self.assertEqual(private,public)
   for commands in (private,public):
    with self.subTest(change=change),self.assertRaises(ValueError):a.require_commands(commands)
 def test_actual_install_guard_rejects_other_device_bundle_and_path(self):
  remote='/var/containers/Bundle/Application/owned/TokletHarness.app';selected=dict(device_identifier='test-device')
  item=dict(bundleIdentifier=a.m.physical.BUNDLE,builtByDeveloper=True,url='file:///private'+remote+'/')
  data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='test-device',matchingBundleIdentifier=a.m.physical.BUNDLE,apps=[item]))
  self.assertTrue(a.require_current(data,remote,selected))
  for mutation in ('device','bundle','path'):
   changed=json.loads(json.dumps(data))
   if mutation=='device':changed['result']['deviceIdentifier']='other'
   elif mutation=='bundle':changed['result']['apps'][0]['bundleIdentifier']='other'
   else:changed['result']['apps'][0]['url']='file:///var/other.app/'
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):a.require_current(changed,remote,selected)
 def test_wrong_configuration_or_installation_rejected(self):
  config=dict(capacity=1152,maximum=128,metal=True);signed=dict(app='a'*64);prepared=dict(path='expected',sha256='b'*64)
  record=dict(passed=True,primary_error=None,endpoint_errors=[],cleanup_errors=[],config=config,signed_bundle=signed,prepared_receipt=prepared)
  a.require_stage(record,config,signed,prepared)
  for field,value in [('config',dict(config,capacity=32)),('signed_bundle',dict(app='c'*64)),('prepared_receipt',dict(path='other',sha256='b'*64))]:
   changed=dict(record);changed[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):a.require_stage(changed,config,signed,prepared)
 def test_full_closure_conflict_and_mutation_fail(self):
  with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as folder:
   path=Path(folder)/'private-output';path.write_bytes(b'original');name=str(path.relative_to(a.ROOT));pins={}
   a.merge(pins,{name:a.sha(path)});a.verify(pins)
   with self.assertRaises(ValueError):a.merge(pins,{name:'0'*64})
   with self.assertRaises(ValueError):a.merge(pins,{'../outside':'0'*64})
   path.write_bytes(b'mutated private output')
   with self.assertRaises(ValueError):a.verify(pins)
 def test_metrics_use_completed_native_and_caller_intervals(self):
  events=[dict(nativeElapsedSeconds=2+i*0.5,callbackUptimeNanoseconds=10**9+int((3+i*0.5)*10**9)) for i in range(128)]
  rows=[dict(index=3,events=events,submittedUptimeNanoseconds=10**9,completedUptimeNanoseconds=68*10**9)]
  result=a.metrics(rows)['raw'][0]
  self.assertEqual(result['decode_tokens_per_second'],2);self.assertEqual(result['native_ttft_seconds'],2)
  self.assertEqual(result['callback_ttft_seconds'],3);self.assertEqual(result['completed_request_seconds'],67)
 def cleanup(self,fixture):
  stage,record,base,metadata,remote,selected=fixture
  return a.require_cleanups(stage,record,base,lambda path:metadata[path.stem],remote,selected)
 def test_exact_initial_reset_and_natural_cleanup_scopes(self):
  self.assertEqual(self.cleanup(cleanup_fixture()),dict(initial_preflight_resets=1,natural_absence_checks=1))
  self.assertEqual(self.cleanup(cleanup_fixture(False)),dict(initial_preflight_resets=0,natural_absence_checks=2))
 def test_reset_cannot_move_to_later_stage_or_hide_signal(self):
  fixture=cleanup_fixture()
  for stage in a.expected_stages()[1:]:
   changed=list(copy.deepcopy(fixture));changed[0]=stage
   with self.subTest(stage=stage),self.assertRaises(ValueError):self.cleanup(changed)
  changed=list(cleanup_fixture(False));changed[0]=a.expected_stages()[0]
  with self.assertRaises(ValueError):self.cleanup(changed)
 def test_signal_device_pid_executable_result_and_command_required(self):
  for field,value in [('deviceIdentifier','other'),('process',dict(executable='/var/other.app/Other',processIdentifier=42)),('process',dict(executable=cleanup_fixture()[4]+'/TokletHarness',processIdentifier=43)),('signal',dict(name='SIGKILL',value=9))]:
   fixture=cleanup_fixture();fixture[3]['signal-2']['result'][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):self.cleanup(fixture)
  fixture=cleanup_fixture();fixture[3]['signal-2']['info']['outcome']='failed'
  with self.assertRaises(ValueError):self.cleanup(fixture)
  fixture=cleanup_fixture();fixture[1]['commands'][2]['command'][5]='43'
  with self.assertRaises(ValueError):self.cleanup(fixture)
 def test_initial_signal_must_precede_copy_and_measurement(self):
  for swap in [(2,5),(3,4),(0,2)]:
   fixture=cleanup_fixture();commands=fixture[1]['commands'];commands[swap[0]],commands[swap[1]]=commands[swap[1]],commands[swap[0]]
   with self.subTest(swap=swap),self.assertRaises(ValueError):self.cleanup(fixture)
 def test_every_process_observation_requires_device_success_and_absence(self):
  for first,names in [(True,['process-1','process-3','process-4']),(False,['process-1','process-2'])]:
   for name in names:
    for field in ('device','outcome'):
     fixture=cleanup_fixture(first);data=fixture[3][name]
     if field=='device':data['result']['deviceIdentifier']='other'
     else:data['info']['outcome']='failed'
     with self.subTest(first=first,name=name,field=field),self.assertRaises(ValueError):self.cleanup(fixture)
  for name in ('process-3','process-4'):
   fixture=cleanup_fixture();fixture[3][name]['result']['runningProcesses']=[dict(executable=fixture[4]+'/TokletHarness',processIdentifier=42)]
   with self.subTest(name=name),self.assertRaises(ValueError):self.cleanup(fixture)
if __name__=='__main__':unittest.main()
