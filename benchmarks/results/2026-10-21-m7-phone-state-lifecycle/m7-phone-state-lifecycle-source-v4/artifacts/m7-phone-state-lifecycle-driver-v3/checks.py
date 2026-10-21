from pathlib import Path
import importlib.util,json,tempfile,unittest
from unittest.mock import patch
p=Path(__file__).with_name('driver.py');s=importlib.util.spec_from_file_location('taildriver',p);d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
class Checks(unittest.TestCase):
 def prior(self):return dict(passed=False,endpoint_errors=[],primary_error='ValueError: Physical workflow stage failed: state',results=[dict(exit_code=v) for v in [0,0,0,0,0,1]],driver_bindings={})
 def test_four_ordered_stages(self):self.assertEqual([v['stage'] for v in d.stages()],['state','storage-proof','lifecycle-prepare','lifecycle'])
 def test_reuse_exact_ready_and_separate_fresh_life(self):
  with tempfile.TemporaryDirectory() as f:
   root=Path(f);calls=[];results=[];original=d.consume
   def consume(path,expected,pins,*args):
    if path.name=='execution.json':return self.prior(),'a'*64
    return original(path,expected,pins,*args)
   def write(path):path.parent.mkdir(parents=True);path.write_text(json.dumps(dict(passed=True,primary_error=None,endpoint_errors=[],cleanup_errors=[])));return 0
   def m(*args,**kw):calls.append((args,kw));return write(args[7]/'measurement.json')
   def life(*args):calls.append((args,{}));return write(args[-1]/'lifecycle.json')
   with patch.object(d,'ROOT',root),patch.object(d,'guard'),patch.object(d,'consume',side_effect=consume),patch.object(d.m,'main',side_effect=m),patch.object(d.life,'collect',side_effect=life):d.collect('a'*64,{},results)
   self.assertEqual(len(calls),4);self.assertEqual(len(results),4);self.assertEqual(str(calls[0][1]['transfer'].relative_to(root)),'artifacts/m7-phone-state-larger-ready-v2');self.assertTrue(calls[0][1]['state']);self.assertTrue(calls[2][1]['lifecycle']);self.assertTrue(calls[2][1]['prepare_only'])
 def test_failed_prior_prevents_device_children(self):
  prior=self.prior();prior['results'][0]['exit_code']=1
  with patch.object(d,'guard'),patch.object(d,'consume',return_value=(prior,'a'*64)),patch.object(d.m,'main') as child,self.assertRaises(ValueError):d.collect('a'*64,{},[])
  child.assert_not_called()
 def test_existing_output_prevents_admission(self):
  with tempfile.TemporaryDirectory() as f:
   p=Path(f);(p/'execution.json').touch()
   with patch.object(d,'OUT',p),patch.object(d.m,'main') as child,self.assertRaises(FileExistsError):d.main(*(['a'*64]*3))
   child.assert_not_called()
if __name__=='__main__':unittest.main()
