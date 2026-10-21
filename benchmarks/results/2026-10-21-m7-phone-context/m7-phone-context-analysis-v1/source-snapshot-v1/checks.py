from pathlib import Path
import contextlib,hashlib,importlib.util,json,tempfile,unittest
from unittest.mock import patch
P=Path(__file__).with_name('analyze.py').resolve();spec=importlib.util.spec_from_file_location('context_analysis',P);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class AnalysisChecks(unittest.TestCase):
 def test_exact_hash_and_json_share_one_bounded_read(self):
  with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as folder:
   path=Path(folder)/'record.json';initial=b'{"original":true}';changed=b'{"changed":true}';path.write_bytes(initial);pins={};original=Path.open
   @contextlib.contextmanager
   def opened(target,*args,**kwargs):
    with original(target,*args,**kwargs) as stream:yield stream
    if target==path and args==('rb',):
     with original(path,'wb') as stream:stream.write(changed)
   with patch.object(Path,'open',opened):raw=a.read_bound(path,100,pins)
   self.assertEqual(json.loads(raw),dict(original=True))
   self.assertEqual(pins[str(path.relative_to(a.ROOT))],hashlib.sha256(initial).hexdigest())
   with self.assertRaises(ValueError):a.verify(pins)
   with self.assertRaises(ValueError):a.read_bound(path,1,{})
 def test_exact_order_paths_and_repeat_counts_required(self):
  good=dict(exit_code=0,primary_error=None,endpoint_errors=[],results=a.expected_stages(),
            driver_bindings={n:'0'*64 for n in a.expected_driver_inputs()},
            ready_receipts={r['path']+'/measurement.json':'0'*64 for r in a.expected_stages() if r['stage']=='prepare'})
  a.require_execution(good)
  for index,change in [(0,dict(exit_code=False)),(0,dict(workload='512')),(2,dict(path=good['results'][3]['path'])),(3,dict(stage='prepare'))]:
   record=json.loads(json.dumps(good));record['results'][index].update(change)
   with self.subTest(change=change),self.assertRaises(ValueError):a.require_execution(record)
  swapped=json.loads(json.dumps(good));swapped['results'][0],swapped['results'][1]=swapped['results'][1],swapped['results'][0]
  with self.assertRaises(ValueError):a.require_execution(swapped)
  unequal=json.loads(json.dumps(good));unequal['results'][3]=unequal['results'][7]
  with self.assertRaises(ValueError):a.require_execution(unequal)
  for field in ('driver_bindings','ready_receipts'):
   missing=json.loads(json.dumps(good));missing[field].pop(next(iter(missing[field])))
   with self.subTest(field=field),self.assertRaises(ValueError):a.require_execution(missing)
 def test_consistently_failed_commands_are_rejected(self):
  good=dict(exit_code=0,primary_error=None,cleanup_errors=[])
  a.require_commands([good])
  for change in (dict(exit_code=False),dict(exit_code=1),dict(primary_error='failed'),dict(cleanup_errors=['failed cleanup'])):
   private=[dict(good,**change)];public=json.loads(json.dumps(private));self.assertEqual(private,public)
   for commands in (private,public):
    with self.subTest(change=change),self.assertRaises(ValueError):a.require_commands(commands)
 def test_actual_install_guard_rejects_other_device_bundle_and_path(self):
  remote='/var/containers/Bundle/Application/owned/TokletHarness.app';selected=dict(device_identifier='test-device')
  item=dict(bundleIdentifier=a.m.physical.BUNDLE,builtByDeveloper=True,url='file:///private'+remote+'/')
  data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='test-device',matchingBundleIdentifier=a.m.physical.BUNDLE,apps=[item]))
  self.assertTrue(a.require_current(data,remote,selected))
  for mutation in ('device','bundle','path'):
   changed=json.loads(json.dumps(data))
   if mutation=='device':changed['result']['deviceIdentifier']='other'
   elif mutation=='bundle':changed['result']['apps'][0]['bundleIdentifier']='other'
   else:changed['result']['apps'][0]['url']='file:///var/other.app/'
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):a.require_current(changed,remote,selected)
 def test_wrong_configuration_or_installation_rejected(self):
  config=dict(capacity=1152,maximum=128,metal=True);signed=dict(app='a'*64);prepared=dict(path='expected',sha256='b'*64)
  record=dict(passed=True,primary_error=None,endpoint_errors=[],cleanup_errors=[],config=config,signed_bundle=signed,prepared_receipt=prepared)
  a.require_stage(record,config,signed,prepared)
  for field,value in [('config',dict(config,capacity=32)),('signed_bundle',dict(app='c'*64)),('prepared_receipt',dict(path='other',sha256='b'*64))]:
   changed=dict(record);changed[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):a.require_stage(changed,config,signed,prepared)
 def test_full_closure_conflict_and_mutation_fail(self):
  with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as folder:
   path=Path(folder)/'private-output';path.write_bytes(b'original');name=str(path.relative_to(a.ROOT));pins={}
   a.merge(pins,{name:a.sha(path)});a.verify(pins)
   with self.assertRaises(ValueError):a.merge(pins,{name:'0'*64})
   with self.assertRaises(ValueError):a.merge(pins,{'../outside':'0'*64})
   path.write_bytes(b'mutated private output')
   with self.assertRaises(ValueError):a.verify(pins)
 def test_metrics_use_completed_native_and_caller_intervals(self):
  events=[dict(nativeElapsedSeconds=2+i*0.5,callbackUptimeNanoseconds=10**9+int((3+i*0.5)*10**9)) for i in range(128)]
  rows=[dict(index=3,events=events,submittedUptimeNanoseconds=10**9,completedUptimeNanoseconds=68*10**9)]
  result=a.metrics(rows)['raw'][0]
  self.assertEqual(result['decode_tokens_per_second'],2);self.assertEqual(result['native_ttft_seconds'],2)
  self.assertEqual(result['callback_ttft_seconds'],3);self.assertEqual(result['completed_request_seconds'],67)
if __name__=='__main__':unittest.main()
