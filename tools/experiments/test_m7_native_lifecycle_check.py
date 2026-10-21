import copy,json,unittest
import m7_native_lifecycle_check as p

def state(saved=None,history=0,evaluated=0,pending=0,reason=0,status=0,owners=(False,False),closed=False):
 return dict(busy=False,closed=closed,workerFinished=closed,hasModel=owners[0],hasSession=owners[1],status=status,history=history,evaluated=evaluated,pending=pending,reason=reason,savedHistory=len(saved) if saved is not None else 0,savedPrefix=saved,ids=[])
def fixture():
 events=[]
 for i,n in enumerate(p.NAMES):
  expected=p.SETTLED.get(n)
  if expected is not None:
   saved,h,e,pen,r,s=expected;value=state(saved,h,e,pen,r,s,p.OWNERS[n],n=='closed')
  else:value=state(p.P+[47],5,4,1,3,owners=(True,True))
  before=copy.deepcopy(value)
  if n in ('inactive-request','background-request','warning-request'):
   count=2 if n=='warning-request' else 1;before=state(p.P+[47]*count,4+count,3+count,1,3 if count==1 else 1,owners=(True,True))
  events.append(dict(index=i,name=n,origin='syntheticCheck' if n in p.SYNTHETIC else 'coordinator',observedUptimeNanoseconds=i+1,recordedUptimeNanoseconds=i+2,
                     applicationState=p.APPLICATIONS[n],command=p.COMMANDS.get(n),before=before,state=value,footprint=dict(physicalBytes=100,lifetimePeakPhysicalBytes=200,returnedNaturalCount=93)))
 return dict(passed=True,backend='cpu',trace=dict(events=events,activeCommand=None,error=None,detachedAfterClose=True),cases=list(p.CASES),closed=True,released=True,physicalDeviceQualified=False,actualOSLifecycleQualified=False)
class Checks(unittest.TestCase):
 def reject(self,mutate):
  d=fixture();mutate(d)
  with self.assertRaises((ValueError,KeyError,TypeError)):p.validate_output(json.dumps(d),'cpu')
 def test_exact_synthetic_trace(self):self.assertEqual(len(p.validate_output(json.dumps(fixture()),'cpu')['trace']['events']),14)
 def test_provenance_order_and_scope(self):
  self.reject(lambda d:d['trace']['events'].reverse());self.reject(lambda d:d['trace']['events'][2].update(origin='systemNotification'));self.reject(lambda d:d.update(actualOSLifecycleQualified=True));self.reject(lambda d:d['cases'].pop())
 def test_exact_settled_state_and_replay(self):
  self.reject(lambda d:d['trace']['events'][6]['state'].update(history=0,evaluated=0,pending=0));self.reject(lambda d:d['trace']['events'][5]['state'].update(hasSession=True));self.reject(lambda d:d['trace']['events'][-1]['state'].update(workerFinished=False))
 def test_pre_action_prefix_and_command_identity(self):
  self.reject(lambda d:d['trace']['events'][3]['before'].update(savedPrefix=[]));self.reject(lambda d:d['trace']['events'][8]['before'].update(savedHistory=33));self.reject(lambda d:d['trace']['events'][0].update(command='generate'));self.reject(lambda d:d['trace']['events'][2].update(applicationState=True))
 def test_clocks_process_and_bounded_history(self):
  self.reject(lambda d:d['trace']['events'][0].update(observedUptimeNanoseconds=3));self.reject(lambda d:d['trace']['events'][3].update(recordedUptimeNanoseconds=1));self.reject(lambda d:d['trace']['events'][1]['footprint'].update(lifetimePeakPhysicalBytes=201));self.reject(lambda d:d['trace']['events'][0]['state'].update(savedHistory=True))
 def test_failure_and_detach_must_be_clean(self):
  self.reject(lambda d:d['trace'].update(error='overflow'));self.reject(lambda d:d['trace'].update(activeCommand='generate'));self.reject(lambda d:d['trace'].update(detachedAfterClose=False));self.reject(lambda d:d.update(released=False))
if __name__=='__main__':unittest.main()
