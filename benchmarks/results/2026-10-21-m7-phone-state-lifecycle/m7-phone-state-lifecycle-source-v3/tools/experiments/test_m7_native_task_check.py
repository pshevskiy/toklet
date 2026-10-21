import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import m7_native_task_check as check


def fixture():
 def response(chunks):return dict(ids=[47]*8,sampled=8,reason=1,history=11,evaluated=10,pending=1,chunks=chunks)
 return dict(passed=True,backend='cpu',simulated=check.SIMULATED,oneCall=response(1),chunked=response(3),
  hiddenStop=dict(ids=[],sampled=1,reason=2,stopToken=47,history=4,evaluated=3,pending=1,chunks=1),
  lifecycleCleared=True,closed=True,released=True,physicalDeviceQualified=False)


class TaskOutputChecks(unittest.TestCase):
 def validate(self,data):return check.validate_output(json.dumps(data),'cpu',47)
 def rejects(self,change):
  data=copy.deepcopy(fixture());change(data)
  with self.assertRaises((ValueError,KeyError,TypeError)):self.validate(data)
 def test_complete_output(self):self.assertTrue(self.validate(fixture())['passed'])
 def test_exact_scope(self):
  for key,value in [('passed',False),('backend','fallback'),('simulated',[]),('lifecycleCleared',False),('closed',False),('released',False),('physicalDeviceQualified',True)]:
   with self.subTest(key=key):self.rejects(lambda d:d.update({key:value}))
 def test_bounded_native_output(self):
  for ids in ([47]*7,[47]*9,[64]*8,[True]*8,[62]*8):
   self.rejects(lambda d:d['oneCall'].update(ids=ids))
  self.rejects(lambda d:d['chunked'].update(ids=[47]*7+[46]))
 def test_integer_counts_and_prefix(self):
  for key,value in [('sampled',True),('history',12),('evaluated',11),('pending',0),('chunks',2),('reason',3)]:
   with self.subTest(key=key):self.rejects(lambda d:d['chunked'].update({key:value}))
 def test_hidden_stop_budget(self):
  for key,value in [('ids',[47]),('sampled',2),('stopToken',46),('reason',1),('history',5),('evaluated',4),('pending',0)]:
   with self.subTest(key=key):self.rejects(lambda d:d['hiddenStop'].update({key:value}))
  self.rejects(lambda d:d['oneCall'].update(stopToken=47))
 def test_independent_oracle_required(self):
  for oracle in (None,[],True,64,-1):
   with self.assertRaises(ValueError):check.validate_output(json.dumps(fixture()),'cpu',oracle)
  self.rejects(lambda d:d['oneCall']['ids'].__setitem__(0,46))
 def test_frozen_fixture_identity(self):
  with self.assertRaises(ValueError):check.independent_first(bytes(1024))
  with self.assertRaises(ValueError):check.independent_first(bytes(1025))


if __name__=='__main__':unittest.main()
