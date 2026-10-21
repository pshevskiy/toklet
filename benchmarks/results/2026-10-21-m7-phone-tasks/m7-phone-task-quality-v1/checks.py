from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile,unittest
from unittest.mock import Mock
from unittest.mock import patch
p=Path(__file__).resolve().with_name('quality.py');spec=importlib.util.spec_from_file_location('phone_task_quality',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def one():
 case=dict(id='literal/id',category='code',max_output_tokens=3)
 phone=dict(caseID='literal/id',response=dict(ids=[47,48,49],sampled=3,reason=1,history=7,evaluated=6,pending=1))
 old=dict(max_output_tokens=3,response=dict(id='literal/id',category='code',visible_token_ids=[47,48,49],visible_text='raw',visible_text_sha256=hashlib.sha256(b'raw').hexdigest(),stop_reason=dict(kind='output_limit'),statistics=dict(generation=dict(sampled_tokens=3,history_tokens=7,evaluated_tokens=6,pending_tokens=1))))
 return case,phone,old
class Checks(unittest.TestCase):
 def test_exact_limit_response(self):
  c,p,o=one();r=a.match_response(c,p,o,'raw');self.assertEqual(r['id'],'literal/id');self.assertEqual(r['visible_token_ids'],[47,48,49])
 def test_exact_hidden_stop(self):
  c,p,o=one();p['response'].update(reason=2,stopToken=62);o['response']['stop_reason']=dict(kind='stop_token',token_id=62)
  self.assertEqual(a.match_response(c,p,o,'raw')['stop_reason'],dict(kind='stop_token',token_id=62))
 def test_wrong_response_stop_cap_and_typed_counts(self):
  for mutate in (lambda c,p,o:p['response'].update(ids=[46,48,49]),lambda c,p,o:p['response'].update(reason=2,stopToken=62),lambda c,p,o:p['response'].update(reason=True),lambda c,p,o:p['response'].update(sampled=True),lambda c,p,o:p['response'].update(stopToken=62),lambda c,p,o:o.update(max_output_tokens=2),lambda c,p,o:p['response'].update(history=6),lambda c,p,o:p.update(caseID='other'),lambda c,p,o:o['response'].update(category='prose')):
   c,p,o=one();mutate(c,p,o)
   with self.assertRaises(ValueError):a.match_response(c,p,o,'raw')
 def test_text_is_exact_and_not_repaired(self):
  c,p,o=one()
  for text in ('raw\n',' raw','```raw```','different'):
   with self.assertRaises(ValueError):a.match_response(c,p,o,text)
 def grading(self):
  q=json.loads((a.ROOT/a.HISTORICAL).read_text());self.assertEqual(hashlib.sha256((a.ROOT/a.HISTORICAL).read_bytes()).hexdigest(),a.HISTORICAL_SHA)
  parent=q['models'][0];cases=a.tasks.load_cases();old=json.loads((a.ROOT/parent['collection']).read_text());responses=[copy.deepcopy(r['response']) for r in old['run']['cases']];g=json.loads((a.ROOT/parent['grading']).read_text());return cases,responses,g
 def test_actual_historical_100case_replay_is_read_only(self):
  cases,responses,g=self.grading();out,aggregate=a.replay_outcomes(cases,responses,g);self.assertEqual(out,g['outcomes']);self.assertEqual(aggregate['passed'],2)
 def test_historical_program_test_and_response_binding(self):
  for field in ('program_sha256','completion_sha256','id'):
   cases,responses,g=self.grading();g['runs'][0][field]='changed'
   with self.assertRaises(ValueError):a.replay_outcomes(cases,responses,g)
  cases,responses,g=self.grading();cases[0]=copy.deepcopy(cases[0]);cases[0]['tests']+='\n# changed'
  with self.assertRaises(ValueError):a.replay_outcomes(cases,responses,g)
 def test_unresolved_inventory_or_outcome_cannot_aggregate(self):
  for mutate in (lambda g:g.update(unresolved_case_ids=['literal/id']),lambda g:g.update(quality_completed=False),lambda g:g['runs'].pop(),lambda g:g['outcomes'][0].update(passed=True,status='model_pass'),lambda g:g['aggregate'].update(passed=3)):
   cases,responses,g=self.grading();mutate(g)
   with self.assertRaises(ValueError):a.replay_outcomes(cases,responses,g)
 def test_changed_classifier_result_rejected(self):
  cases,responses,g=self.grading()
  with patch.object(a.evidence,'classify',return_value={'passed':True}),self.assertRaises(ValueError):a.replay_outcomes(cases,responses,g)
 def test_typed_hidden_stop_rejected(self):
  c,p,o=one();p['response'].update(reason=2,stopToken=True);o['response']['stop_reason']=dict(kind='stop_token',token_id=1)
  with self.assertRaises(ValueError):a.match_response(c,p,o,'raw')
 def test_exclusive_outputs_reject_files_and_dangling_links(self):
  for name in ('quality.json','small-responses.json','larger-responses.json','private-execution.json','private-final-guard.json','private-logs'):
   with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as f:
    root=Path(f);p=root/name;p.symlink_to(root/'absent')
    with self.assertRaises(FileExistsError):a.exclusive_outputs(root)
    p.unlink();p.write_bytes(b'original')
    with self.assertRaises(FileExistsError):a.exclusive_outputs(root)
    self.assertEqual(p.read_bytes(),b'original')
 def test_exact_serialized_output_pin_and_replacement_guard(self):
  with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as f:
   path=Path(f)/'responses.json';pins={};raw=a.write_bound(path,dict(ids=[47]),pins)
   self.assertEqual(path.read_bytes(),raw);self.assertEqual(pins[str(path.relative_to(a.ROOT))],hashlib.sha256(raw).hexdigest())
   path.write_bytes(b'replaced')
   with self.assertRaises(ValueError):a.a.c.verify(pins)
   self.assertEqual(pins[str(path.relative_to(a.ROOT))],hashlib.sha256(raw).hexdigest())
   with self.assertRaises(FileExistsError):a.write_bound(path,dict(ids=[46]),{})
 def test_failed_absence_diagnostics_bound_before_rejection(self):
  with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as f:
   root=Path(f);out=root/'absence.stdout';err=root/'absence.stderr';out.write_bytes(b'failure stdout');err.write_bytes(b'failure stderr');pins={}
   row=dict(exit_code=1,primary_error='tool exit 1',cleanup_errors=[],stdout=str(out.relative_to(a.ROOT)),stderr=str(err.relative_to(a.ROOT)))
   tokenizer=Mock();grading=Mock()
   def read(path):return a.a.c.read_bound(path,1024,pins)
   with self.assertRaisesRegex(ValueError,'tool exit 1'):
    payload=a.consume_owned(row,read);tokenizer(payload);grading(payload)
   tokenizer.assert_not_called();grading.assert_not_called()
   self.assertEqual(set(pins),{row['stdout'],row['stderr']});a.a.c.verify(pins)
 def test_timeout_errors_private_and_public_redaction(self):
  import subprocess
  selection=dict(device_identifier='device-private-123',device_udid='udid-private-456',development_team='team-private-789');remote='/private/installed/secret-UUID/TokletHarness.app'
  timeout=str(subprocess.TimeoutExpired(['devicectl',*selection.values(),remote],90));record=dict(commands=[dict(primary_error=timeout)],primary_error=timeout,endpoint_errors=[timeout])
  with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts') as f:
   path=Path(f)/'private-execution.json';pins={};private=a.write_bound(path,record,pins)
   public=a.m.physical.redact_record(record,selection,remote);status=json.dumps(dict(primary_error=public['primary_error'],endpoint_errors=public['endpoint_errors']))
   for secret in (*selection.values(),remote):
    self.assertIn(secret.encode(),private);self.assertNotIn(secret,json.dumps(public));self.assertNotIn(secret,status)
if __name__=='__main__':unittest.main()
