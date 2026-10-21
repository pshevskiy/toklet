from pathlib import Path
import copy,importlib.util,tempfile,unittest
p=Path(__file__).resolve().with_name('analyze.py');s=importlib.util.spec_from_file_location('small_sustained_analysis',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
SOURCE={'sources':{'frozen/source':'a'*64}}
def driver():
 source={'sources':{'frozen/source':'a'*64}};bindings={n:'a'*64 for n in a.driver_inputs(source)};return dict(passed=True,primary_error=None,endpoint_errors=[],physical_device_qualified=False,m7_complete=False,driver_bindings=bindings,results=[dict(**r,exit_code=0,sha256=bindings[r['path']]) for r in a.stages()])
class Checks(unittest.TestCase):
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
 def test_measured600_excludes_warmups(self):
  rows=[dict(tokens=[47]*128,events=[dict(nativeElapsedSeconds=600 if i<3 else 60)]) for i in range(13)]
  self.assertEqual(a.check_sustained(rows,dict(minimumNativeSeconds=600),dict(sumNativeLastEventSeconds=600)),600)
  with self.assertRaises(ValueError):a.check_sustained(rows,dict(minimumNativeSeconds=600),dict(sumNativeLastEventSeconds=2400))
  for r in rows[3:]:r['events'][-1]['nativeElapsedSeconds']=59
  with self.assertRaises(ValueError):a.check_sustained(rows,dict(minimumNativeSeconds=600),dict(sumNativeLastEventSeconds=590))
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
if __name__=='__main__':unittest.main()
