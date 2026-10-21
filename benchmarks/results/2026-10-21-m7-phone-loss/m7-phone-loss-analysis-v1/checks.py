from pathlib import Path
import copy,importlib.util,math,tempfile,unittest
p=Path(__file__).resolve().with_name('analyze.py');s=importlib.util.spec_from_file_location('loss_analysis',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def driver():
 bindings={n:'a'*64 for n in a.driver_inputs()};return dict(passed=True,primary_error=None,endpoint_errors=[],physical_device_qualified=False,m7_complete=False,driver_bindings=bindings,results=[dict(**r,exit_code=0,sha256=bindings[r['path']]) for r in a.stages()])
def samples(value=1.):return [dict(target=dict(score=dict(negativeLogLikelihood=value))) for _ in range(10000)]
class Checks(unittest.TestCase):
 def test_exact_driver_and_fourteen_stages(self):
  r=driver();a.require_driver(r);self.assertEqual(len(r['results']),14);self.assertEqual(sum(r['stage']=='quality' for r in r['results']),4)
 def test_driver_order_and_count(self):
  for mutate in (lambda r:r['results'].reverse(),lambda r:r['results'].pop(),lambda r:r['results'][2].update(model='small')):
   r=driver();mutate(r)
   with self.assertRaises(ValueError):a.require_driver(r)
 def test_driver_failure_scope_and_digest(self):
  for mutate in (lambda r:r.update(passed=False),lambda r:r.update(m7_complete=True),lambda r:r['results'][0].update(exit_code=True),lambda r:r['results'][0].update(sha256='b'*64),lambda r:r['driver_bindings'].pop('tools/experiments/m7_physical_quality.py')):
   r=driver();mutate(r)
   with self.assertRaises(ValueError):a.require_driver(r)
 def test_gate_and_mean(self):
  good=a.aggregate(samples(1.),1.);self.assertEqual(good['scored_targets'],10000);self.assertTrue(good['loss_gate_passed']);self.assertEqual(good['increase_nats_per_token'],0.)
  self.assertTrue(a.aggregate(samples(1.04),1.)['loss_gate_passed']);self.assertFalse(a.aggregate(samples(1.06),1.)['loss_gate_passed'])
 def test_full_count_and_finite_loss(self):
  for rows,baseline in ((samples()[:-1],1.),(samples(),float('nan')),(samples(float('inf')),1.),(samples(True),1.),(samples(-1.),1.)):
   with self.assertRaises(ValueError):a.aggregate(rows,baseline)
 def test_fsum_independent_of_incremental_accumulation(self):
  rows=samples(1e-10);rows[0]['target']['score']['negativeLogLikelihood']=1e6
  self.assertEqual(a.aggregate(rows,1.)['mean_negative_log_likelihood'],math.fsum([1e6]+[1e-10]*9999)/10000)
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
 def test_independent_tiny_cpu_and_driver_record(self):
  oracle=a.independent_tiny((a.ROOT/'tests/fixtures/tiny-v1/cases.prompt.logits.f32le').read_bytes())
  rows=[dict(target=dict(score=dict(negativeLogLikelihood=x,greedy=i))) for i,x in enumerate(oracle)]
  expected=a.tiny_evidence(rows,'tiny-cpu',oracle);a.require_tiny(dict(tiny_oracle={'tiny-cpu':expected}),{'tiny-cpu':expected})
  rows[0]['target']['score']['negativeLogLikelihood']+=0.001
  with self.assertRaises(ValueError):a.tiny_evidence(rows,'tiny-cpu',oracle)
  with self.assertRaises(ValueError):a.require_tiny(dict(tiny_oracle={}),{'tiny-cpu':expected})
  with self.assertRaises(ValueError):a.independent_tiny(b'changed')
 def test_consumed_buffer_binding_and_conflict(self):
  with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as f:
   p=Path(f)/'input';p.write_bytes(b'original');pins={};raw=a.c.read_bound(p,8,pins);self.assertEqual(raw,b'original');a.c.verify(pins)
   p.write_bytes(b'replaced')
   with self.assertRaises(ValueError):a.c.verify(pins)
   with self.assertRaises(ValueError):a.c.read_bound(p,3,{})
  with self.assertRaises(ValueError):a.c.merge({'x':'a'},{'x':'b'})
if __name__=='__main__':unittest.main()
