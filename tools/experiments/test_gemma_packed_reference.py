"""Owned-capture error regressions with mocked children; no numerical imports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gemma_packed_reference as reference


class OwnedPacked(unittest.TestCase):
 def setup_tree(self,root):
  for name in ('tools/reference/gemma_packed.py','tools/reference/test_gemma_packed.py','tools/reference/test_reference_runtime.py','tools/experiments/gemma_packed_reference.py','tools/experiments/test_gemma_packed_reference.py','tools/experiments/m7_apple_package.py','target/release/toklet','target/release/examples/cpu_bench'):
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic source')
  (root/'artifacts').mkdir()

 def test_bad_destination_and_reuse_reject_without_child(self):
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary).resolve();self.setup_tree(root)
   with mock.patch.object(reference,'ROOT',root),mock.patch.object(reference.owner,'owned') as child:
    with self.assertRaises(ValueError):reference.collect(root/'escape')
    existing=root/'artifacts'/'existing';existing.mkdir()
    with self.assertRaises(FileExistsError):reference.collect(existing)
    child.assert_not_called()

 def test_failed_or_raised_child_retains_logs_attempt_and_changed_source(self):
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary).resolve();self.setup_tree(root);helper=root/'tools/experiments/m7_apple_package.py';runner=root/'tools/experiments/gemma_packed_reference.py'
   source=root/'tools/reference/gemma_packed.py'
   def failed(command,stdout,stderr,environment,timeout):
    self.assertEqual(list(Path(environment['PYTHONPYCACHEPREFIX']).iterdir()),[])
    stdout.write_bytes(b'partial stdout');stderr.write_bytes(b'partial stderr');source.write_bytes(b'changed')
    return dict(exit_code=1,primary_error='synthetic failed child',cleanup_errors=[])
   def raised(*args,**kwargs):failed(*args,**kwargs);raise OSError('synthetic owner raise')
   with mock.patch.object(reference,'ROOT',root),mock.patch.object(reference,'__file__',str(runner)),mock.patch.object(reference.owner,'__file__',str(helper)),mock.patch.object(reference.owner,'numerical_sources',return_value=[]),mock.patch.object(reference.runtime,'runtime_paths',side_effect=lambda selected:sorted(set(selected))):
    for label,operation in (('failed',failed),('raised',raised)):
     source.write_bytes(b'synthetic source');destination=root/'artifacts'/label
     with mock.patch.object(reference.owner,'owned',side_effect=operation):self.assertEqual(reference.collect(destination),1)
     result=json.loads((destination/'capture.json').read_bytes())
     self.assertFalse(result['capture_complete']);self.assertTrue(result['endpoint_errors']);self.assertEqual(len(result['commands']),1)
     self.assertEqual(set(result['outputs']),{'logs/owner.stdout','logs/owner.stderr'})
     self.assertEqual((destination/'logs/owner.stdout').read_bytes(),b'partial stdout')
     if label=='raised':self.assertIn('synthetic owner raise',result['commands'][0]['invocation_error'])
     else:self.assertEqual(result['commands'][0]['owned_result']['exit_code'],1)

 def test_replaced_admitted_source_buffer_rejects_before_child(self):
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary).resolve();self.setup_tree(root);helper=root/'tools/experiments/m7_apple_package.py';runner=root/'tools/experiments/gemma_packed_reference.py'
   def replace(path,maximum,expected):
    raw=path.read_bytes();path.write_bytes(b'substituted after capture');return raw
   with mock.patch.object(reference,'ROOT',root),mock.patch.object(reference,'__file__',str(runner)),mock.patch.object(reference.owner,'__file__',str(helper)),mock.patch.object(reference.owner,'numerical_sources',return_value=[]),mock.patch.object(reference.runtime,'runtime_paths',side_effect=lambda selected:sorted(set(selected))),mock.patch.object(reference.runtime,'captured',side_effect=replace),mock.patch.object(reference.owner,'owned') as child:
    destination=root/'artifacts'/'substitution';self.assertEqual(reference.collect(destination),1);child.assert_not_called()
   report=json.loads((destination/'capture.json').read_bytes())
   self.assertIn('source buffer changed',report['primary_error']);self.assertFalse(report['capture_complete'])


if __name__=='__main__':unittest.main()
