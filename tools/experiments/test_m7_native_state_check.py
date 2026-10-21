import copy,json,unittest
import m7_native_state_check as p

def fixture(prefill=True):
 prompt=[1,7,13,3]*128 if prefill else [1,7,13,3]
 c=dict(prompt=prompt,capacity=1152 if prefill else 32,chunk=128 if prefill else 3,vocabulary=151936 if prefill else 64,files={'Model/config.json':'a'*64},numericalBudget=1500000000,processBudget=2500000000)
 primer=prompt[:16];target=primer[-1];pending=[47]*3;ap=primer+pending+prompt[:2];changed=primer.copy();changed[0]=0
 full=[prompt[i%len(prompt)] for i in range(c['capacity'])];rows=[]
 f=dict(physicalBytes=100,lifetimePeakPhysicalBytes=200,returnedNaturalCount=93)
 def state(tokens,tail=0,ids=None,reason=1):return dict(busy=False,closed=False,workerFinished=False,hasModel=True,hasSession=True,status=0,history=len(tokens),evaluated=len(tokens)-tail,pending=tail,reason=reason,savedHistory=len(tokens),savedPrefix=tokens,ids=[] if ids is None else ids)
 for i,n in enumerate(p.names(prefill)):
  if n=='loaded' or n.endswith('-reset'):s=state([],reason=0)
  elif n in ('primer','primer-score','fully-cached','cached-score'):s=state(primer)
  elif n=='pending-primer':s=state(primer+pending,1,pending)
  elif n in ('append-pending','append-reference'):s=state(ap+[47]*2,1,[47]*2)
  elif n in ('changed-prefix','changed-reference','usable-fresh-reference'):s=state(changed+[47]*3,1,[47]*3)
  elif n=='prefill-cancel-trigger':s=state([],reason=0);s.update(busy=True,savedPrefix=None)
  elif n=='prefill-cancelled':s=state(prompt,len(prompt)-128,reason=3)
  elif n=='prefill-completed':s=state(prompt)
  elif n in ('prefill-resumed','prefill-reference'):s=state(prompt+[47]*3,1,[47]*3)
  elif n=='closed':s=state([],reason=0);s.update(closed=True,workerFinished=True,hasModel=False,hasSession=False,savedPrefix=None)
  else:s=state(full,reason=1 if n in ('capacity-committed','overflow-continue-before-score') else 0)
  if n in ('overflow-continue','overflow-append','overflow-prompt'):s['status']=1
  score=dict(target=target,greedy=47,vocabulary=c['vocabulary'],evaluated=s['evaluated'],targetLogit=1.0,greedyLogit=2.0,negativeLogLikelihood=3.0) if 'score' in n else None
  rows.append(dict(name=n,state=s,scalarScore=score,callbackCount=len(s['ids']),admittedBytes=0 if n=='closed' else 1000000000,completedUptimeNanoseconds=i+1,footprint=copy.deepcopy(f),environment=dict(applicationState=None)))
 d=dict(passed=True,backend='metal',prefillWitnessRequired=prefill,prefillWitness=prefill,cleanupErrors=[],endpointErrors=[],closed=True,released=True,physicalDeviceQualified=False,m7Complete=False,filesBefore=c['files'],filesAfter=c['files'],rows=rows,trace=dict(events=[],detachedAfterClose=True),checkpoints=[dict(phase=n,footprint=copy.deepcopy(f)) for n in ('before-artifact-proof','after-artifact-proof','loaded','closed','after-final-artifact-proof')])
 error=dict(busy=False,closed=False,workerFinished=False,hasModel=False,hasSession=False,status=1,history=0,evaluated=0,pending=0,reason=0,savedHistory=0,savedPrefix=[],ids=[])
 after=copy.deepcopy(error);after.update(closed=True,workerFinished=True,status=0,savedPrefix=None)
 return c,dict(diagnostic=d,errorBeforeClose=error,errorAfterClose=after,errorCloseAdmission=0,errorCloseCallbacks=0,errorCloseMessage='')
class Checks(unittest.TestCase):
 def reject(self,mutate):
  c,d=fixture();mutate(c,d['diagnostic'])
  with self.assertRaises((ValueError,KeyError,TypeError)):p.validate_output(json.dumps(d),'metal','small',c)
 def row(self,d,n):return next(r for r in d['rows'] if r['name']==n)
 def test_error_to_close_regression(self):
  for field,value in [('errorCloseAdmission',1),('errorCloseCallbacks',1),('errorCloseMessage','stale')]:
   c,d=fixture();d[field]=value
   with self.assertRaises(ValueError):p.validate_output(json.dumps(d),'metal','small',c)
  c,d=fixture();d['errorAfterClose']['status']=1
  with self.assertRaises(ValueError):p.validate_output(json.dumps(d),'metal','small',c)
 def test_error_wrapper_status_types_and_busy(self):
  for name in ('errorBeforeClose','errorAfterClose'):
   for value in (True,False,1.0,0.0):
    c,d=fixture();d[name]['status']=value
    with self.assertRaises(ValueError):p.validate_output(json.dumps(d),'metal','small',c)
   c,d=fixture();d[name]['busy']=True
   with self.assertRaises(ValueError):p.validate_output(json.dumps(d),'metal','small',c)
 def test_complete_real_and_tiny(self):
  for prefill in (False,True):
   c,d=fixture(prefill);self.assertEqual(p.validate_output(json.dumps(d),'metal','small' if prefill else 'unit',c),d)
 def test_missing_partial_commit(self):
  self.reject(lambda c,d:self.row(d,'prefill-cancelled')['state'].update(evaluated=0,pending=512))
  self.reject(lambda c,d:self.row(d,'prefill-cancelled')['state'].update(evaluated=512,pending=0))
 def test_cancel_emission_or_late_trigger(self):
  self.reject(lambda c,d:self.row(d,'prefill-cancel-trigger')['state'].update(busy=False))
  self.reject(lambda c,d:self.row(d,'prefill-cancelled')['state'].update(ids=[47]))
  self.reject(lambda c,d:self.row(d,'prefill-cancelled')['state'].update(reason=1))
 def test_changed_prompt_reference(self):
  self.reject(lambda c,d:self.row(d,'changed-reference')['state']['savedPrefix'].__setitem__(0,1))
  self.reject(lambda c,d:self.row(d,'changed-reference')['state']['ids'].__setitem__(0,46))
 def test_append_pending_reference(self):
  self.reject(lambda c,d:self.row(d,'append-reference')['state']['savedPrefix'].__setitem__(16,46))
  self.reject(lambda c,d:self.row(d,'append-pending')['state'].update(evaluated=23,pending=0))
 def test_resume_reference(self):
  self.reject(lambda c,d:self.row(d,'prefill-reference')['state']['savedPrefix'].__setitem__(0,0))
  self.reject(lambda c,d:self.row(d,'prefill-reference')['state']['ids'].__setitem__(0,46))
 def test_cached_scalar(self):
  self.reject(lambda c,d:self.row(d,'cached-score')['scalarScore'].update(targetLogit=0.9))
  self.reject(lambda c,d:self.row(d,'cached-score')['scalarScore'].update(targetLogit=float('nan')))
 def test_each_overflow_preserves_state_and_scalar(self):
  for n in ('overflow-continue','overflow-append','overflow-prompt'):
   self.reject(lambda c,d:self.row(d,n)['state'].update(evaluated=1151,pending=1))
   self.reject(lambda c,d:self.row(d,n)['state']['savedPrefix'].__setitem__(0,0))
   self.reject(lambda c,d:self.row(d,n+'-after-score')['scalarScore'].update(negativeLogLikelihood=3.1))
   self.reject(lambda c,d:self.row(d,n).update(callbackCount=1))
 def test_forged_types_and_missing_rows(self):
  self.reject(lambda c,d:self.row(d,'capacity-committed')['state'].update(history=True))
  self.reject(lambda c,d:self.row(d,'capacity-committed').update(admittedBytes=True))
  self.reject(lambda c,d:d['rows'].pop(1))
 def test_memory_chronology_and_cap(self):
  self.reject(lambda c,d:d['rows'][0]['footprint'].update(lifetimePeakPhysicalBytes=c['processBudget']+1))
  self.reject(lambda c,d:d['checkpoints'][1]['footprint'].update(lifetimePeakPhysicalBytes=300))
  self.reject(lambda c,d:d['rows'][1].update(completedUptimeNanoseconds=1))
 def test_lifecycle_and_owner_release(self):
  self.reject(lambda c,d:d['trace'].update(events=[{}]))
  self.reject(lambda c,d:d.update(released=False))
  self.reject(lambda c,d:self.row(d,'closed')['state'].update(history=1,evaluated=1))
 def test_integer_widths_and_scalar_identity_types(self):
  for key in ('target','vocabulary','evaluated','greedy'):
   def mutate(c,d):
    for n in ('primer-score','cached-score'):
     score=self.row(d,n)['scalarScore'];score[key]=float(score[key])
   self.reject(mutate)
  self.reject(lambda c,d:d['rows'][-1].update(completedUptimeNanoseconds=2**64))
  self.reject(lambda c,d:d['rows'][0]['footprint'].update(returnedNaturalCount=2**32))
  self.reject(lambda c,d:d['rows'][0]['state'].update(status=2**32))
 def test_scope_and_file_endpoints(self):
  self.reject(lambda c,d:d.update(physicalDeviceQualified=True))
  self.reject(lambda c,d:d.update(prefillWitness=False))
  self.reject(lambda c,d:d.update(filesAfter={}))
if __name__=='__main__':unittest.main()
