from pathlib import Path
import copy,importlib.util,tempfile,unittest
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
if __name__=='__main__':unittest.main()
