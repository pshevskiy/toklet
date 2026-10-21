from pathlib import Path
import importlib.util,json,tempfile,unittest
from unittest.mock import patch
p=Path(__file__).resolve().with_name('driver.py');spec=importlib.util.spec_from_file_location('loss_driver',p);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
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
 def test_console_replacement_rejected_and_consumed_pins_persist(self):
  with tempfile.TemporaryDirectory(dir=d.ROOT/'artifacts') as f:
   root=Path(f);logs=root/'private-logs';logs.mkdir();stdout=logs/'measurement.stdout';stderr=logs/'measurement.stderr'
   stdout.write_text('\n'.join('TOKLET_QUALITY_SAMPLE_JSON='+json.dumps(dict(target=dict(score=dict(negativeLogLikelihood=1.,greedy=47)))) for _ in range(3)));stderr.write_bytes(b'')
   record=dict(private_evidence_pins={str(p.relative_to(d.ROOT)):d.m.package.sha(p) for p in (stdout,stderr)});pins={}
   self.assertEqual(d.tiny_scores(root,record,pins),([1.]*3,[47]*3));self.assertEqual(pins,record['private_evidence_pins'])
   stdout.write_text('replaced')
   with self.assertRaises(ValueError):d.tiny_scores(root,record,{})
   with self.assertRaises(ValueError):d.guard(pins)
 def test_input_bound(self):
  with tempfile.TemporaryDirectory() as f:
   p=Path(f)/'large';p.write_bytes(b'12345')
   with self.assertRaises(ValueError):d.bounded(p,4)
if __name__=='__main__':unittest.main()
