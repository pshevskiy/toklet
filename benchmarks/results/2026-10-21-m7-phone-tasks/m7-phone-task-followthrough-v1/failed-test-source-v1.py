from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile,unittest
from unittest.mock import patch
path=Path(__file__).resolve().with_name('run.py');spec=importlib.util.spec_from_file_location('task_followthrough',path);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class Checks(unittest.TestCase):
 def test_complete_driver_and_failures(self):
  record=dict(passed=True,primary_error=None,endpoint_errors=[],results=[{}]*14);footer=dict(passed=True,stages=14,primary_error=None,endpoint_errors=[]);self.assertTrue(a.completed(record,footer))
  for mutate in (lambda r:r.update(passed=False),lambda r:r['results'].pop(),lambda r:r.update(endpoint_errors=['changed'])):
   r=copy.deepcopy(record);mutate(r)
   with self.assertRaises(ValueError):a.completed(r,footer)
  with self.assertRaises(ValueError):a.completed(record,dict(footer,stages=13))
 def test_dangling_output_prevents_children(self):
  with tempfile.TemporaryDirectory() as f:
   folder=Path(f);(folder/'execution.json').symlink_to(folder/'absent')
   with patch.object(a,'OUT',folder),patch.object(a.p,'owned') as child,self.assertRaises(FileExistsError):a.main()
   child.assert_not_called()
 def test_bounded_read_and_changed_guard(self):
  with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as f:
   path=Path(f)/'input';path.write_bytes(b'first');pins={str(path.relative_to(a.ROOT)):a.p.sha(path)};self.assertEqual(a.bounded(path,5),b'first');a.guard(pins);path.write_bytes(b'other')
   with self.assertRaises(ValueError):a.guard(pins)
   with self.assertRaises(ValueError):a.bounded(path,3)
 def test_unterminated_footer_waits_and_pins_complete_buffer(self):
  with tempfile.TemporaryDirectory() as f:
   root=Path(f);folder=root/'artifacts/m7-phone-task-followthrough-v1';folder.mkdir(parents=True)
   for n in ('run.py','checks.py'):(folder/n).write_bytes(path.with_name(n).read_bytes())
   spec=importlib.util.spec_from_file_location('isolated_followthrough',folder/'run.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.ROOT=root;b.DRIVER=root/'artifacts/m7-phone-tasks-driver-v1';b.DRIVER.mkdir()
   for dirname,files in [('m7-phone-tasks-analysis-v1',('analyze.py','checks.py')),('m7-phone-task-quality-v1',('quality.py','checks.py'))]:
    d=root/'artifacts'/dirname;d.mkdir()
    for n in files:(d/n).write_text('# source\n')
   d=root/'artifacts/m7-phone-tasks-source-v1';d.mkdir();(d/'source.json').write_text('{"sources":{}}')
   record=dict(passed=True,primary_error=None,endpoint_errors=[],results=[{}]*14);footer=dict(passed=True,stages=14,primary_error=None,endpoint_errors=[])
   (b.DRIVER/'execution.json').write_text(json.dumps(record)+'\n');stdout=b.DRIVER/'driver.stdout';partial=json.dumps(footer).encode();stdout.write_bytes(partial);calls=[];polls=[]
   def execute(args,out,err,env,timeout):
    calls.append(args);self.assertEqual(stdout.read_bytes(),partial+b'\n');out.write_bytes(b'checked\n');err.write_bytes(b'')
    if len(calls)==1:(root/'artifacts/m7-phone-tasks-analysis-v1/analysis.json').write_text(json.dumps(dict(passed=True,cases=206,task_quality_qualified=False))+'\n')
    else:(root/'artifacts/m7-phone-task-quality-v1/quality.json').write_text(json.dumps(dict(passed=True,quality_completed=True,models=[{},{}],new_vm_runs=0,new_generated_program_executions=0))+'\n')
    return dict(command=args,exit_code=0,primary_error=None,cleanup_errors=[],stdout=str(out.relative_to(root)),stderr=str(err.relative_to(root)))
   def polling(seconds):
    self.assertFalse(calls);polls.append(seconds);stdout.write_bytes(partial+b'\n')
   with patch.object(b.p,'owned',side_effect=execute),patch.object(b.p,'build_environment',return_value=({},None)),patch.object(b.time,'sleep',side_effect=polling):self.assertEqual(b.main(),0)
   self.assertEqual(polls,[1]);self.assertEqual(len(calls),2)
   result=json.loads((folder/'execution.json').read_text());self.assertEqual(result['bindings'][str(stdout.relative_to(root))],hashlib.sha256(partial+b'\n').hexdigest())
   expected=hashlib.sha256((root/'artifacts/m7-phone-tasks-analysis-v1/analysis.json').read_bytes()).hexdigest();self.assertEqual(calls[1][-1],expected)
if __name__=='__main__':unittest.main()
