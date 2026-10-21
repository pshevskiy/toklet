from pathlib import Path
import hashlib,importlib.util,json,tempfile,unittest
from unittest.mock import patch
p=Path(__file__).resolve().with_name('driver.py');spec=importlib.util.spec_from_file_location('small_sustained_driver',p);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
class Checks(unittest.TestCase):
 def test_dangling_output_prevents_children(self):
  with tempfile.TemporaryDirectory() as f:
   out=Path(f);(out/'execution.json').symlink_to(out/'absent')
   with patch.object(d,'OUT',out),patch.object(d.m,'main') as child,self.assertRaises(FileExistsError):d.main()
   child.assert_not_called()
 def test_receipt_sha_and_json_from_same_bytes_and_later_replacement_rejected(self):
  with tempfile.TemporaryDirectory(dir=d.ROOT/'artifacts') as f:
   path=Path(f)/'measurement.json';raw=b'{"passed":true}\n';path.write_bytes(raw);pins={}
   parsed,digest=d.receipt(path,pins);self.assertEqual(parsed,dict(passed=True));self.assertEqual(digest,d.m.package.hashlib.sha256(raw).hexdigest())
   path.write_text('{"passed":false}');self.assertTrue(parsed['passed'])
   with self.assertRaises(ValueError):d.guard(pins)
 def test_input_bound(self):
  with tempfile.TemporaryDirectory() as f:
   p=Path(f)/'large';p.write_bytes(b'12345')
   with self.assertRaises(ValueError):d.bounded(p,4)
 def test_quality_gate_waits_complete_bytes_and_binds_exact_pair(self):
  with tempfile.TemporaryDirectory() as f:
   root=Path(f).resolve();workflow=root/'artifacts/m7-phone-task-followthrough-v1/execution.json';quality=root/'artifacts/m7-phone-task-quality-v1/quality.json';workflow.parent.mkdir(parents=True);quality.parent.mkdir(parents=True)
   value=dict(format='toklet-m7-phone-task-outcome-equivalence-v1',passed=True,quality_completed=True,primary_error=None,endpoint_errors=[],models=[dict(model=n,cases=100,drop_percentage_points=0) for n in ('small','larger')]);raw=(json.dumps(value)+'\n').encode();quality.write_bytes(raw[:-1])
   result=dict(passed=True,primary_error=None,endpoint_errors=[],commands=[{},{}],bindings={str(quality.relative_to(root)):hashlib.sha256(raw).hexdigest()});workflow.write_text(json.dumps(result)+'\n');pins={};polls=[]
   def complete(seconds):polls.append(seconds);quality.write_bytes(raw)
   with patch.object(d,'ROOT',root),patch.object(d.time,'sleep',side_effect=complete):d.await_quality(pins)
   self.assertEqual(polls,[1]);self.assertEqual(pins[str(quality.relative_to(root))],hashlib.sha256(raw).hexdigest())
   result['bindings'][str(quality.relative_to(root))]='a'*64;workflow.write_text(json.dumps(result)+'\n')
   with patch.object(d,'ROOT',root),self.assertRaises(ValueError):d.await_quality({})
   result.update(passed=False);workflow.write_text(json.dumps(result)+'\n')
   with patch.object(d,'ROOT',root),self.assertRaises(ValueError):d.await_quality({})
 def test_source_receipt_replacement_rejected_before_children(self):
  with tempfile.TemporaryDirectory(dir=d.ROOT/'artifacts') as f:
   path=Path(f)/'source.json';raw=b'{"sources":{}}\n';path.write_bytes(raw);pins={str(path.relative_to(d.ROOT)):hashlib.sha256(raw).hexdigest()};d.admit_source(path,pins)
   path.write_bytes(b'{"sources":{"replacement":{"sha256":"changed"}}}\n')
   with patch.object(d.m,'main') as child,self.assertRaisesRegex(ValueError,'consumed bytes differ'):d.admit_source(path,pins)
   child.assert_not_called();self.assertEqual(pins[str(path.relative_to(d.ROOT))],hashlib.sha256(raw).hexdigest())
if __name__=='__main__':unittest.main()
