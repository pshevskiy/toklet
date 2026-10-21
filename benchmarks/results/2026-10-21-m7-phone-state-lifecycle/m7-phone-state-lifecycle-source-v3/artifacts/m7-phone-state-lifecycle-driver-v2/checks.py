from pathlib import Path
import importlib.util,json,tempfile,unittest,hashlib
from unittest.mock import patch
p=Path(__file__).resolve().with_name('driver.py');spec=importlib.util.spec_from_file_location('phone_state_life_driver',p);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
class Checks(unittest.TestCase):
 def test_nine_stage_order(self):
  rows=d.stages();self.assertEqual(len(rows),9);self.assertEqual([r['stage'] for r in rows],['storage-proof','lifecycle-prepare','lifecycle','storage-proof','state-prepare','state','storage-proof','lifecycle-prepare','lifecycle'])
 def test_dangling_output_prevents_children(self):
  with tempfile.TemporaryDirectory() as f:
   out=Path(f);(out/'execution.json').symlink_to(out/'absent')
   with patch.object(d,'OUT',out),patch.object(d.m,'main') as child,self.assertRaises(FileExistsError):d.main(*(['a'*64]*3))
   child.assert_not_called()
 def test_consumed_sha_bound_and_later_mutation(self):
  with tempfile.TemporaryDirectory(dir=d.ROOT/'artifacts') as f:
   path=Path(f)/'input';raw=b'{"passed":true}';path.write_bytes(raw);pins={};value,h=d.consume(path,hashlib.sha256(raw).hexdigest(),pins);self.assertTrue(value['passed']);d.guard(pins);path.write_bytes(b'{}')
   with self.assertRaises(ValueError):d.guard(pins)
   with self.assertRaises(ValueError):d.consume(path,h,{})
 def test_owned_stage_calls_exact_modes_and_reuses_ready(self):
  with tempfile.TemporaryDirectory() as f:
   root=Path(f).resolve();pins={};results=[];calls=[]
   def write(path):
    path.parent.mkdir(parents=True);path.write_text(json.dumps(dict(passed=True,primary_error=None,endpoint_errors=[],cleanup_errors=[])));return 0
   def tiny(native,native_sha,private,private_sha,out,**kw):calls.append(('tiny',kw));return write(out/'integration.json')
   def measure(*args,**kw):calls.append((args[5],args[6],kw));return write(args[7]/'measurement.json')
   def lifecycle(*args):calls.append(('life',args));return write(args[-1]/'lifecycle.json')
   with patch.object(d,'ROOT',root),patch.object(d,'guard'),patch.object(d.m.physical,'main',side_effect=tiny),patch.object(d.m,'main',side_effect=measure),patch.object(d.life,'collect',side_effect=lifecycle):d.collect('a'*64,pins,results)
   self.assertEqual(len(results),9);self.assertEqual(len(calls),9);self.assertEqual(sum(c[0]=='life' for c in calls),2)
   state=[c for c in calls if len(c)==3 and c[2].get('state')];self.assertEqual(len(state),2);self.assertEqual(sum(c[2].get('prepare_only',False) for c in state),1);self.assertTrue(all('transfer_sha' in c[2] for c in state if not c[2].get('prepare_only')))
 def test_stage_failure_stops_after_bound_receipt(self):
  with tempfile.TemporaryDirectory() as f:
   root=Path(f).resolve();results=[]
   def failed(*args,**kwargs):
    out=args[7];out.mkdir(parents=True);(out/'measurement.json').write_text(json.dumps(dict(passed=False,primary_error='failed',endpoint_errors=[],cleanup_errors=[])));return 1
   with patch.object(d,'ROOT',root),patch.object(d,'guard'),patch.object(d.m,'main',side_effect=failed) as child,patch.object(d.life,'collect') as life,self.assertRaises(ValueError):d.collect('a'*64,{},results)
   self.assertEqual(len(results),1);self.assertEqual(child.call_count,1);life.assert_not_called()
 def test_sustained_gate_exact_digest_and_policy(self):
  with tempfile.TemporaryDirectory() as f:
   root=Path(f).resolve();path=root/d.GATE;path.parent.mkdir(parents=True)
   value=dict(format='toklet-m7-phone-small-sustained-qualification-v1',passed=True,sustained_duration_qualified=True,review=dict(accepted=True,reviewer='Astra'),measured_native_seconds=600,measured_requests=10,warmups=3,hardware=dict(product_type='iPhone15,2',os_version='27.0.1'),physical_device_qualified=False,m7_complete=False)
   for field,new in ((None,None),('measured_native_seconds',599),('measured_native_seconds',float('nan')),('measured_requests',9),('warmups',2),('passed',False)):
    record=dict(value)
    if field:record[field]=new
    raw=json.dumps(record).encode();path.write_bytes(raw)
    with patch.object(d,'ROOT',root):
     if field:
      with self.assertRaises(ValueError):d.admit_sustained(hashlib.sha256(raw).hexdigest(),{})
     else:self.assertEqual(d.admit_sustained(hashlib.sha256(raw).hexdigest(),{}),record)
if __name__=='__main__':unittest.main()
