import copy,json,unittest
import m7_native_lifecycle_app_check as p

def state(saved=None,history=0,evaluated=0,pending=0,reason=0,status=0,model=False,session=False,busy=False,ids=None,closed=False):
 return dict(busy=busy,closed=closed,workerFinished=closed,hasModel=model,hasSession=session,status=status,history=history,evaluated=evaluated,pending=pending,reason=reason,savedHistory=0 if saved is None else len(saved),savedPrefix=saved,ids=[] if ids is None else ids)
def fixture():
 prompt=[1,7,13,3];rebuilt=[47,47];prefix=prompt+rebuilt;next_ids=[47]*3
 config=dict(files={'Model/config.json':'a'*64},numericalBudget=128*1024**2,processBudget=256*1024**2,prompt=prompt,maximum=25,capacity=32,vocabulary=64,metal=True)
 names=['loaded','generation-active','scene-inactive','scene-background','scene-inactive','scene-active','suspended','replayed','resumed','reference-reset','reference-prefix','reference','synthetic-memory-warning','warning-unloaded','replay-rejected','fresh-load','fresh-prompt','unloaded','closed']
 settled={
  'loaded':state([],model=True,session=True),'suspended':state(prefix,model=True,ids=rebuilt),
  'replayed':state(prefix,6,6,0,1,model=True,session=True),
  'resumed':state(prefix+next_ids,9,8,1,1,model=True,session=True,ids=next_ids),
  'reference-reset':state([],model=True,session=True),
  'reference-prefix':state(prefix,6,5,1,1,model=True,session=True,ids=rebuilt),
  'reference':state(prefix+next_ids,9,8,1,1,model=True,session=True,ids=next_ids),
  'warning-unloaded':state([]),'replay-rejected':state([],status=1),'fresh-load':state([],model=True,session=True),
  'fresh-prompt':state(prompt+next_ids,7,6,1,1,model=True,session=True,ids=next_ids),'unloaded':state([]),'closed':state(None,closed=True)}
 events=[]
 for i,n in enumerate(names):
  value=copy.deepcopy(settled.get(n,state(None,model=True,session=True,busy=True,ids=[47,47])))
  if n=='generation-active':value=state(None,model=True,session=True,busy=True,ids=[47])
  events.append(dict(index=i,name=n,origin='syntheticCheck' if n.startswith('scene-') or n=='synthetic-memory-warning' else 'coordinator',
   observedUptimeNanoseconds=10+10*i,recordedUptimeNanoseconds=11+10*i,applicationState=None,
   command='generate' if i<=6 and i>0 else n if n in settled and n!='closed' else None,
   before=copy.deepcopy(value),state=value,footprint=dict(physicalBytes=100,lifetimePeakPhysicalBytes=200,returnedNaturalCount=93)))
 trip={n:copy.deepcopy(events[i]) for n,i in [('inactive',2),('background',3),('returnInactive',4),('active',5)]}
 ready=dict(eventIndex=1,submittedUptimeNanoseconds=1,observedUptimeNanoseconds=events[1]['recordedUptimeNanoseconds'],visibleIDs=events[1]['before']['ids'],applicationState=None)
 result=dict(passed=True,backend='metal',cleanupErrors=[],endpointErrors=[],filesBefore=config['files'],filesAfter=config['files'],closed=True,released=True,
  actualOSLifecycleQualified=False,physicalDeviceQualified=False,m7Complete=False,loadedAdmittedBytes=100,ready=ready,roundTrip=trip,interruptedPrefix=prefix,rebuiltIDs=rebuilt,resumedIDs=next_ids,referenceIDs=next_ids,freshIDs=next_ids,
  trace=dict(events=events,detachedAfterClose=True))
 return config,dict(passed=True,backend='metal',kind='small',cases=['small-synthetic-roundtrip'],trips=[],diagnostic=result,actualOSLifecycleQualified=False,physicalDeviceQualified=False)
class Checks(unittest.TestCase):
 def reject(self,mutate):
  c,d=fixture();mutate(c,d)
  with self.assertRaises((ValueError,KeyError,TypeError,IndexError)):p.validate_output(json.dumps(d),'metal','small',c)
 def test_exact_synthetic_real_model_protocol(self):
  c,d=fixture();self.assertTrue(p.validate_output(json.dumps(d),'metal','small',c)['passed'])
 def test_trace_binding_and_ready_causality(self):
  self.reject(lambda c,d:d['diagnostic']['roundTrip']['inactive'].update(index=3));self.reject(lambda c,d:d['diagnostic']['ready'].update(observedUptimeNanoseconds=40));self.reject(lambda c,d:d['diagnostic']['ready'].update(visibleIDs=[46]))
 def test_exact_replay_reference_and_fresh_load(self):
  self.reject(lambda c,d:d['diagnostic'].update(referenceIDs=[46]*3));self.reject(lambda c,d:d['diagnostic'].update(freshIDs=[46]*3));self.reject(lambda c,d:d['diagnostic']['trace']['events'][7]['state'].update(history=0,evaluated=0,pending=0));self.reject(lambda c,d:d['diagnostic']['trace']['events'][6]['state'].update(hasSession=True))
 def test_original_prefix_and_source_identity(self):
  self.reject(lambda c,d:c.update(prompt=[2,7,13,3]));self.reject(lambda c,d:d['diagnostic'].update(filesAfter={}));self.reject(lambda c,d:d['diagnostic'].update(loadedAdmittedBytes=True))
 def test_provenance_process_and_close(self):
  self.reject(lambda c,d:d['diagnostic']['trace']['events'][2].update(origin='scenePhase'));self.reject(lambda c,d:d.update(actualOSLifecycleQualified=True));self.reject(lambda c,d:d['diagnostic']['trace']['events'][0]['footprint'].update(lifetimePeakPhysicalBytes=300*1024**2));self.reject(lambda c,d:d['diagnostic']['trace']['events'][-1]['state'].update(workerFinished=False))
 def test_typed_prefix_and_ordered_clocks(self):
  self.reject(lambda c,d:d['diagnostic']['trace']['events'][2]['before'].update(busy=1));self.reject(lambda c,d:d['diagnostic']['trace']['events'][3].update(recordedUptimeNanoseconds=1));self.reject(lambda c,d:d['diagnostic']['trace'].update(activeCommand='generate'))
if __name__=='__main__':unittest.main()
