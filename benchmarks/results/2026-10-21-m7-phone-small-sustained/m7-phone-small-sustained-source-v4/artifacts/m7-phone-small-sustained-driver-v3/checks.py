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
 def quality(self):
  return dict(format='toklet-m7-phone-tasks-qualification-v1',passed=True,task_quality_qualified=True,review=dict(accepted=True,reviewer='Astra'),full_cases_per_model=100,physical_device_qualified=False,m7_complete=False,new_generated_program_executions=0,new_vm_runs=0,models=[dict(model=n,cases=100,drop_percentage_points=0) for n in ('small','larger')])
 def test_quality_gate_consumed_hash_and_policy(self):
  value=self.quality();raw=(json.dumps(value)+'\n').encode();h=hashlib.sha256(raw).hexdigest();self.assertEqual(d.require_quality(raw,h),value)
  with self.assertRaises(ValueError):d.require_quality(raw,'a'*64)
  for mutate in (lambda v:v.update(passed=False),lambda v:v['review'].update(accepted=False),lambda v:v['models'][0].update(cases=99),lambda v:v['models'][0].update(drop_percentage_points=3),lambda v:v.update(new_vm_runs=1)):
   v=self.quality();mutate(v);raw=json.dumps(v).encode()
   with self.assertRaises(ValueError):d.require_quality(raw,hashlib.sha256(raw).hexdigest())
 def test_accepted_qualification_is_pinned_and_replacement_rejected(self):
  with tempfile.TemporaryDirectory() as f:
   root=Path(f).resolve();path=root/d.GATE;path.parent.mkdir(parents=True);raw=(json.dumps(self.quality())+'\n').encode();path.write_bytes(raw);pins={}
   with patch.object(d,'ROOT',root),patch.object(d,'GATE_SHA',hashlib.sha256(raw).hexdigest()):d.await_quality(pins)
   self.assertEqual(pins[d.GATE],hashlib.sha256(raw).hexdigest());path.write_bytes(b'{}')
   with patch.object(d,'ROOT',root),self.assertRaises(ValueError):d.guard(pins)
 def test_source_receipt_replacement_rejected_before_children(self):
  with tempfile.TemporaryDirectory(dir=d.ROOT/'artifacts') as f:
   path=Path(f)/'source.json';raw=b'{"sources":{}}\n';path.write_bytes(raw);pins={str(path.relative_to(d.ROOT)):hashlib.sha256(raw).hexdigest()};d.admit_source(path,pins)
   path.write_bytes(b'{"sources":{"replacement":{"sha256":"changed"}}}\n')
   with patch.object(d.m,'main') as child,self.assertRaisesRegex(ValueError,'consumed bytes differ'):d.admit_source(path,pins)
   child.assert_not_called();self.assertEqual(pins[str(path.relative_to(d.ROOT))],hashlib.sha256(raw).hexdigest())
if __name__=='__main__':unittest.main()
