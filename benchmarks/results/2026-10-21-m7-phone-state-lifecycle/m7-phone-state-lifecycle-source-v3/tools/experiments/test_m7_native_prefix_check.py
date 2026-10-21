import copy,json,unittest
import m7_native_prefix_check as p

def fixture():
 rows=[]
 for n,prefix in p.PREFIXES.items():
  count=p.DEFAULT_SAVED.get(n,0) if prefix is None else len(prefix)
  history,evaluated,pending,reason=p.STATES[n]
  rows.append(dict(name=n,prefix=prefix,saved=count,status=1 if n=='recoverable-failure' else 0,history=history,evaluated=evaluated,pending=pending,reason=reason))
 return dict(passed=True,backend='cpu',rows=rows,races=['unload','close'],publicationWitnesses=[dict(name=n,ownerAlive=True,prefixAbsent=True,scorePublished=False) for n in ('unload','close')],closed=True,released=True,physicalDeviceQualified=False)
class Checks(unittest.TestCase):
 def test_complete_bounded_snapshot_sequence(self):
  d=fixture();self.assertEqual(len(p.validate_output(json.dumps(d),'cpu')['rows']),26)
 def test_wrong_history_or_saved_prefix(self):
  for field,value in [('prefix',[47]),('saved',True),('history',1),('status',1)]:
   d=fixture();d['rows'][0][field]=value
   with self.assertRaises(ValueError):p.validate_output(json.dumps(d),'cpu')
 def test_default_and_close_cannot_publish(self):
  for row in fixture()['rows']:
   if row['name'].startswith('default-') or row['name'].endswith('close'):
    d=fixture();next(x for x in d['rows'] if x['name']==row['name'])['prefix']=[]
    with self.assertRaises(ValueError):p.validate_output(json.dumps(d),'cpu')
 def test_exact_pending_replay_and_suspension_states(self):
  for name in ('prompt','suspend','cancelled-replay','replay','cancel','default-prompt'):
   d=fixture();row=next(r for r in d['rows'] if r['name']==name);row.update(history=0,evaluated=0,pending=0,reason=1)
   with self.assertRaises(ValueError):p.validate_output(json.dumps(d),'cpu')
 def test_actual_post_publication_witness_is_mandatory(self):
  for field,value in [('prefixAbsent',False),('scorePublished',True),('ownerAlive',False),('prefixAbsent',1)]:
   d=fixture();d['publicationWitnesses'][0][field]=value
   with self.assertRaises(ValueError):p.validate_output(json.dumps(d),'cpu')
 def test_outcome_order_and_failure_status(self):
  for mutate in (lambda d:d['rows'].reverse(),lambda d:d.update(races=['close']),lambda d:d.update(released=False),lambda d:d.update(physicalDeviceQualified=True),lambda d:d['rows'][2].update(status=0)):
   d=fixture();mutate(d)
   with self.assertRaises(ValueError):p.validate_output(json.dumps(d),'cpu')
if __name__=='__main__':unittest.main()
