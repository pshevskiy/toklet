from pathlib import Path
import copy,importlib.util,tempfile,unittest
p=Path(__file__).resolve().with_name('analyze.py');s=importlib.util.spec_from_file_location('small_sustained_analysis',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
SOURCE={'sources':{'frozen/source':'a'*64}}
def driver():
 source={'sources':{'frozen/source':'a'*64}};bindings={n:'a'*64 for n in a.driver_inputs(source)};return dict(passed=True,primary_error=None,endpoint_errors=[],physical_device_qualified=False,m7_complete=False,driver_bindings=bindings,results=[dict(**r,exit_code=0,sha256=bindings[r['path']]) for r in a.stages()])
class Checks(unittest.TestCase):
 def test_exact_installed_guard_scope_without_auxiliary_payload(self):
  self.assertEqual([a.expected_installs(s['stage']) for s in a.stages()],[4,3,2])
  with self.assertRaises(KeyError):a.expected_installs('tasks')
 def test_exact_driver_and_three_stages(self):
  r=driver();a.require_driver(r,SOURCE);self.assertEqual(len(r['results']),3);self.assertEqual(sum(r['stage']=='sustained' for r in r['results']),1)
 def test_driver_order_and_count(self):
  for mutate in (lambda r:r['results'].reverse(),lambda r:r['results'].pop(),lambda r:r['results'][2].update(model='larger')):
   r=driver();mutate(r)
   with self.assertRaises(ValueError):a.require_driver(r,SOURCE)
 def test_driver_failure_scope_and_digest(self):
  for mutate in (lambda r:r.update(passed=False),lambda r:r.update(m7_complete=True),lambda r:r['results'][0].update(exit_code=True),lambda r:r['results'][0].update(sha256='b'*64),lambda r:r['driver_bindings'].pop(a.SOURCE)):
   r=driver();mutate(r)
   with self.assertRaises(ValueError):a.require_driver(r,SOURCE)
 def test_two_natural_absences(self):
  n=dict(absent=True,observations=[[]],signals=[],errors=[]);r=dict(remote_cleanup=[n,copy.deepcopy(n)],commands=[dict(stdout='process-1.stdout'),dict(stdout='process-2.stdout')]);selected=dict(device_identifier='device')
  def read(path):return dict(info=dict(outcome='success',commandType='devicectl.device.info.processes'),result=dict(deviceIdentifier='device',runningProcesses=[]))
  a.require_natural(r,Path('unused'),read,'/owned',selected)
  r['remote_cleanup'][0]['signals']=[dict(pid=1,signal='SIGTERM')]
  with self.assertRaises(ValueError):a.require_natural(r,Path('unused'),read,'/owned',selected)
 def test_absence_wrong_device_and_process(self):
  n=dict(absent=True,observations=[[]],signals=[],errors=[]);r=dict(remote_cleanup=[n,n],commands=[dict(stdout='process-1.stdout'),dict(stdout='process-2.stdout')]);selected=dict(device_identifier='device')
  for device,processes in [('other',[]),('device',[dict(executable='/owned/TokletHarness',processIdentifier=1)])]:
   def read(path):return dict(info=dict(outcome='success',commandType='devicectl.device.info.processes'),result=dict(deviceIdentifier=device,runningProcesses=processes))
   with self.assertRaises(ValueError):a.require_natural(r,Path('unused'),read,'/owned',selected)
 def reset_fixture(self):
  selected=dict(device_identifier='device');remote='/owned';base=a.ROOT/'artifacts/m7-phone-small-sustained-storage-proof-v1';pid=42
  natural=dict(absent=True,observations=[[]],signals=[],errors=[]);reset=dict(absent=True,observations=[[pid],[]],signals=[dict(pid=pid,signal='SIGTERM')],errors=[])
  record=dict(remote_cleanup=[reset,natural],commands=[dict(stdout=n+'.stdout') for n in ('current-install-6','process-1','signal-2','process-3','copy-Model','process-4')])
  record['commands'][2]['command']=[a.m.physical.DEVICECTL,'device','process','signal','--pid',str(pid),'--signal','SIGTERM','--device','<private-device_identifier>','--timeout','90','--quiet','--json-output','-']
  def query(rows):return dict(info=dict(outcome='success',commandType='devicectl.device.info.processes'),result=dict(deviceIdentifier='device',runningProcesses=rows))
  data={'process-1.stdout':query([dict(executable=remote+'/TokletHarness',processIdentifier=pid)]),'process-3.stdout':query([]),'process-4.stdout':query([]),
   'signal-2.stdout':dict(info=dict(outcome='success',commandType='devicectl.device.process.signal'),result=dict(deviceIdentifier='device',process=dict(processIdentifier=pid,executable=remote+'/TokletHarness'),signal=dict(name='SIGTERM',value=15)))}
  return record,base,data,remote,selected
 def test_exact_owned_preparation_reset(self):
  r,b,d,remote,selected=self.reset_fixture();self.assertEqual(a.require_natural(r,b,lambda p:d[p.name],remote,selected),(1,1))
 def test_reset_identity_signal_and_post_absence(self):
  for mutate in (lambda r,d:d['signal-2.stdout']['result']['process'].update(processIdentifier=43),lambda r,d:d['signal-2.stdout']['result'].update(deviceIdentifier='other'),lambda r,d:d['signal-2.stdout']['result']['process'].update(executable='/other/TokletHarness'),lambda r,d:d['signal-2.stdout']['result']['signal'].update(name='SIGKILL',value=9),lambda r,d:d['process-3.stdout']['result'].update(runningProcesses=d['process-1.stdout']['result']['runningProcesses']),lambda r,d:r['remote_cleanup'][0].update(errors=['failed']),lambda r,d:r['commands'].reverse()):
   r,b,d,remote,selected=self.reset_fixture();mutate(r,d)
   with self.assertRaises(ValueError):a.require_natural(r,b,lambda p:d[p.name],remote,selected)
 def test_owned_reset_command_and_install_order(self):
  for index,value in ((5,'43'),(7,'SIGKILL'),(9,'other-device')):
   r,b,d,remote,selected=self.reset_fixture();r['commands'][2]['command'][index]=value
   with self.assertRaises(ValueError):a.require_natural(r,b,lambda p:d[p.name],remote,selected)
  r,b,d,remote,selected=self.reset_fixture();r['commands'][0],r['commands'][1]=r['commands'][1],r['commands'][0]
  with self.assertRaises(ValueError):a.require_natural(r,b,lambda p:d[p.name],remote,selected)
 def test_reset_not_admitted_in_inference_or_other_preparation(self):
  for leaf in ('m7-phone-small-sustained-v1','m7-phone-small-sustained-ready-v1'):
   r,b,d,remote,selected=self.reset_fixture()
   with self.assertRaises(ValueError):a.require_natural(r,b.with_name(leaf),lambda p:d[p.name],remote,selected)
 def test_measured600_excludes_warmups(self):
  rows=[dict(tokens=[47]*128,events=[dict(nativeElapsedSeconds=600 if i<3 else 60)]) for i in range(13)]
  self.assertEqual(a.check_sustained(rows,dict(minimumNativeSeconds=600),dict(sumNativeLastEventSeconds=600)),600)
  with self.assertRaises(ValueError):a.check_sustained(rows,dict(minimumNativeSeconds=600),dict(sumNativeLastEventSeconds=2400))
  for r in rows[3:]:r['events'][-1]['nativeElapsedSeconds']=59
  with self.assertRaises(ValueError):a.check_sustained(rows,dict(minimumNativeSeconds=600),dict(sumNativeLastEventSeconds=590))
 def test_native_total_replays_producer_binary64_order(self):
  rows=[dict(tokens=[47]*128,events=[dict(nativeElapsedSeconds=60.1)]) for _ in range(13)];ordered=0.0
  for row in rows[3:]:ordered+=row['events'][-1]['nativeElapsedSeconds']
  self.assertEqual(a.check_sustained(rows,dict(minimumNativeSeconds=600),dict(sumNativeLastEventSeconds=ordered)),ordered)
  self.assertNotEqual(ordered,sum(r['events'][-1]['nativeElapsedSeconds'] for r in rows[3:]))
  with self.assertRaises(ValueError):a.check_sustained(rows,dict(minimumNativeSeconds=600),dict(sumNativeLastEventSeconds=sum(r['events'][-1]['nativeElapsedSeconds'] for r in rows[3:])))
 def test_count_and_exact_ids(self):
  rows=[dict(tokens=[47]*128,events=[dict(nativeElapsedSeconds=60)]) for _ in range(13)]
  with self.assertRaises(ValueError):a.check_sustained(rows[:-1],dict(minimumNativeSeconds=600),dict(sumNativeLastEventSeconds=600))
  rows[-1]['tokens'][0]=46
  with self.assertRaises(ValueError):a.check_sustained(rows,dict(minimumNativeSeconds=600),dict(sumNativeLastEventSeconds=600))
 def test_consumed_buffer_binding_and_conflict(self):
  with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as f:
   p=Path(f)/'input';p.write_bytes(b'original');pins={};raw=a.c.read_bound(p,8,pins);self.assertEqual(raw,b'original');a.c.verify(pins)
   p.write_bytes(b'replaced')
   with self.assertRaises(ValueError):a.c.verify(pins)
   with self.assertRaises(ValueError):a.c.read_bound(p,3,{})
  with self.assertRaises(ValueError):a.c.merge({'x':'a'},{'x':'b'})
 def prerequisites(self):
  source=dict(sources={'frozen/source':dict(sha256='a'*64)});driver=dict(driver_bindings={'frozen/source':'a'*64,a.QUALITY:a.QUALITY_SHA})
  quality=dict(format='toklet-m7-phone-tasks-qualification-v1',passed=True,task_quality_qualified=True,review=dict(accepted=True,reviewer='Astra'),full_cases_per_model=100,physical_device_qualified=False,m7_complete=False,new_generated_program_executions=0,new_vm_runs=0,models=[dict(model=n,cases=100,drop_percentage_points=0) for n in ('small','larger')])
  return source,driver,quality,{a.QUALITY:a.QUALITY_SHA}
 def test_prerequisites_reject_failed_partial_or_wrong_digest(self):
  values=self.prerequisites();a.admit_prerequisites(*values)
  for mutate in (lambda s,d,q,p:q.update(passed=False),lambda s,d,q,p:q['review'].update(accepted=False),lambda s,d,q,p:q['models'].pop(),lambda s,d,q,p:q['models'][0].update(cases=99),lambda s,d,q,p:d['driver_bindings'].update({a.QUALITY:'c'*64}),lambda s,d,q,p:p.update({a.QUALITY:'c'*64}),lambda s,d,q,p:q.update(new_vm_runs=1)):
   values=self.prerequisites();mutate(*values)
   with self.assertRaises(ValueError):a.admit_prerequisites(*values)
 def test_source_values_must_match_driver(self):
  values=self.prerequisites();values[0]['sources']['frozen/source']['sha256']='c'*64
  with self.assertRaises(ValueError):a.admit_prerequisites(*values)
if __name__=='__main__':unittest.main()
