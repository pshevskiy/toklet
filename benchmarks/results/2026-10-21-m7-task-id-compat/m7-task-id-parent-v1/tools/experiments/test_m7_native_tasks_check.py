import copy
import json
import hashlib
import tempfile
from unittest.mock import patch
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import m7_native_tasks_check as check
SHA='a'*64


def fixture():
 events=[]
 def mem():return dict(physicalBytes=10,lifetimePeakPhysicalBytes=10,returnedNaturalCount=93)
 def env():return dict(thermalState=0,lowPowerMode=False)
 def result(backend):
  return dict(backend=backend,passed=True,cleanupErrors=[],endpointErrors=[],filesBefore=check.EXPECTED_FILES,filesAfter=check.EXPECTED_FILES,
   tasksBeforeSHA256=SHA,tasksAfterSHA256=SHA,checkpoints=[dict(phase=p,footprint=mem()) for p in ['before-artifact-proof','after-artifact-proof','loaded','case-0','case-1','unloaded','closed','after-final-artifact-proof']],
   interruptionCount=0,loadAndSessionSeconds=0.1,loadedAdmittedBytes=100,chunksCompleted=2,casesCompleted=2,closed=True,released=True,fullQualityQualified=False,physicalDeviceQualified=False,m7Complete=False)
 for offset,backend in enumerate(('cpu','metal')):
  for index in (0,1):
   maximum=(3,2)[index];prompt=(3,4)[index];start=(offset*3+index+1)*1_000_000_000
   item=dict(index=0,requested=maximum,ids=[47]*maximum,tokenTimings=[dict(index=i,nativeElapsedSeconds=(i+1)/10,callbackUptimeNanoseconds=start+(i+1)*100_000_000) for i in range(maximum)],reason=1,history=prompt+maximum,evaluated=prompt+maximum-1,pending=1)
   common=dict(caseIndex=index,caseID=('first','second')[index],admittedBytes=100,footprint=mem(),environment=env())
   events.append(('TASKS_CHECK_CHUNK_JSON=',dict(**common,chunk=item,submittedUptimeNanoseconds=start,completedUptimeNanoseconds=start+1_000_000_000)))
   events.append(('TASKS_CHECK_CASE_JSON=',dict(**common,response=dict(ids=[47]*maximum,sampled=maximum,reason=1,history=prompt+maximum,evaluated=prompt+maximum-1,pending=1,chunks=1),resetHistory=0,resetEvaluated=0,resetPending=0)))
  events.append(('TASKS_CHECK_RESULT_JSON=',result(backend)))
 for label in ['overflow-count','output-chunk','output-case','model-endpoint','tasks-endpoint']:
  row=copy.deepcopy(result('cpu'));row.update(passed=False)
  if label=='overflow-count':
   row.update(error='Invalid policy',closed=False,casesCompleted=0,chunksCompleted=0,loadedAdmittedBytes=0,filesBefore={},filesAfter={},tasksBeforeSHA256=None,tasksAfterSHA256=None,loadAndSessionSeconds=0,checkpoints=[dict(phase='closed',footprint=mem())])
  elif label.startswith('output'):row.update(error='Injected sink',casesCompleted=1,chunksCompleted=1 if label=='output-chunk' else 2)
  else:row.update(endpointErrors=['Changed artifact'],tasksAfterSHA256=None,filesAfter={} if label=='model-endpoint' else check.EXPECTED_FILES)
  events.append(('TASKS_CHECK_FAILURE_JSON=',dict(label=label,result=row)))
 events.append(('TASKS_CHECKS_JSON=',dict(passed=True,cases=check.CASES,physicalDeviceQualified=False)))
 return events


def encode(events):return '\n'.join(marker+json.dumps(value) for marker,value in events)+'\n'


class TasksOutputChecks(unittest.TestCase):
 def validate(self,data):return check.validate_output(encode(data),SHA)
 def rejects(self,change):
  data=copy.deepcopy(fixture());change(data)
  with self.assertRaises((ValueError,KeyError,TypeError)):self.validate(data)
 def test_complete_output(self):self.assertEqual(self.validate(fixture())['case_count'],4)
 def test_exact_marker_sequence(self):
  self.rejects(lambda d:d.pop())
  self.rejects(lambda d:d.reverse())
  with self.assertRaises(ValueError):check.validate_output(encode(fixture())+'unexpected\n',SHA)
 def test_case_and_admission_identity(self):
  for key,value in [('caseIndex',True),('caseIndex',1),('caseID','second'),('admittedBytes',0),('admittedBytes',101)]:
   with self.subTest(key=key):self.rejects(lambda d:d[0][1].update({key:value}))
 def test_chunk_ids_prefix_and_reset(self):
  for key,value in [('ids',[47,47]),('ids',[True,47,47]),('requested',True),('history',5),('evaluated',6),('pending',0),('reason',3),('stopToken',47)]:
   with self.subTest(key=key):self.rejects(lambda d:d[0][1]['chunk'].update({key:value}))
  self.rejects(lambda d:d[1][1].update(resetHistory=1))
  self.rejects(lambda d:d[1][1]['response'].update(ids=[47]))
 def test_callback_clocks_coverage_and_order(self):
  self.rejects(lambda d:d[0][1]['chunk']['tokenTimings'].pop())
  for key,value in [('index',1),('nativeElapsedSeconds',True),('nativeElapsedSeconds',float('nan')),('nativeElapsedSeconds',-1),('nativeElapsedSeconds',2),('callbackUptimeNanoseconds',0),('callbackUptimeNanoseconds',3_000_000_000)]:
   with self.subTest(key=key):self.rejects(lambda d:d[0][1]['chunk']['tokenTimings'][0].update({key:value}))
  self.rejects(lambda d:d[0][1].update(completedUptimeNanoseconds=1))
 def test_completed_command_brackets_do_not_overlap(self):
  # Both new submissions remain after the prior last callback. Only the
  # preceding whole-command completion catches these one-nanosecond overlaps.
  self.rejects(lambda d:d[2][1].update(submittedUptimeNanoseconds=1_999_999_999))
  self.rejects(lambda d:d[5][1].update(submittedUptimeNanoseconds=2_999_999_999))
 def test_nested_metadata_hashes_consumed_buffer(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'record.json';old=b'{"passed":true}';changed=b'{"passed":false}'
   expected=hashlib.sha256(old).hexdigest();path.write_bytes(old)
   self.assertEqual(check.nested_metadata(path,expected),({'passed':True},expected))
   path.write_bytes(changed)
   # An independent path-hash observation cannot bless replacement bytes.
   with patch.object(check.package,'sha',return_value=expected):
    with self.assertRaises(ValueError):check.nested_metadata(path,expected)
   self.assertEqual(check.nested_metadata(path,hashlib.sha256(changed).hexdigest())[0],{'passed':False})
 def test_process_and_environment(self):
  self.rejects(lambda d:d[0][1]['footprint'].update(lifetimePeakPhysicalBytes=9))
  self.rejects(lambda d:d[0][1]['footprint'].update(physicalBytes=193*1024**2,lifetimePeakPhysicalBytes=193*1024**2))
  self.rejects(lambda d:d[0][1]['environment'].update(applicationState=0))
  self.rejects(lambda d:d[0][1]['environment'].update(thermalState=True))
  self.rejects(lambda d:d[4][1]['checkpoints'][0]['footprint'].update(lifetimePeakPhysicalBytes=11))
 def test_result_closure_and_proofs(self):
  for key,value in [('passed',False),('closed',False),('released',False),('loadedAdmittedBytes',True),('casesCompleted',1),('interruptionCount',1),('tasksAfterSHA256','b'*64),('filesAfter',{}),('loadAndSessionSeconds',True),('fullQualityQualified',True)]:
   with self.subTest(key=key):self.rejects(lambda d:d[4][1].update({key:value}))
 def test_failed_output_or_mutation_cannot_pass(self):
  self.rejects(lambda d:d[11][1]['result'].update(casesCompleted=2))
  self.rejects(lambda d:d[12][1]['result'].update(chunksCompleted=1))
  self.rejects(lambda d:d[13][1]['result'].update(endpointErrors=[]))
  self.rejects(lambda d:d[14][1]['result'].update(tasksAfterSHA256=SHA))
  self.rejects(lambda d:d[10][1]['result'].update(closed=True))
 def test_complete_case_inventory(self):self.rejects(lambda d:d[-1][1].update(cases=[]))
 def test_exact_task_sha_required(self):
  for sha in (None,'bad','A'*64):
   with self.assertRaises(ValueError):check.validate_output(encode(fixture()),sha)


if __name__=='__main__':unittest.main()
