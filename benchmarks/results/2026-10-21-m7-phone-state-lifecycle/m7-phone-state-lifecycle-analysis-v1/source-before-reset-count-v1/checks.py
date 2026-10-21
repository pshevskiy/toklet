from pathlib import Path
import copy,importlib.util,unittest
p=Path(__file__).with_name('analyze.py');s=importlib.util.spec_from_file_location('state_life_analysis',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class Checks(unittest.TestCase):
 def phase(self):
  labels=['ready-consumed','ready-process','settings-activated','background-process','harness-activated'];phase=[dict(phase=n,observed_host_monotonic_nanoseconds=i+1) for i,n in enumerate(labels)];checked=dict(launch=dict(pid=42),ready=dict(visibleIDs=[47]));phase[0]['ready']=copy.deepcopy(checked['ready'])
  for i in (1,3,4):phase[i]['pid']=42
  return phase,checked
 def test_exact_original_phase(self):
  phase,checked=self.phase();self.assertEqual(a.require_phase(phase,checked),42)
 def test_missing_reordered_duplicate_phases(self):
  for kind in ('missing','reorder','duplicate'):
   phase,c=self.phase()
   if kind=='missing':phase.pop()
   elif kind=='reorder':phase[2],phase[3]=phase[3],phase[2]
   else:phase[-1]=phase[1]
   with self.assertRaises(ValueError):a.require_phase(phase,c)
 def test_phase_clocks_and_boolean(self):
  for value in (True,0,2):
   phase,c=self.phase();phase[-1]['observed_host_monotonic_nanoseconds']=value
   with self.assertRaises(ValueError):a.require_phase(phase,c)
 def test_changed_pid_or_ready(self):
  for value in (43,True):
   phase,c=self.phase();phase[3]['pid']=value
   with self.assertRaises(ValueError):a.require_phase(phase,c)
  phase,c=self.phase();phase[0]['ready']['visibleIDs']=[48]
  with self.assertRaises(ValueError):a.require_phase(phase,c)
 def test_activation_actual_positional_boundary(self):
  bundle='com.apple.Preferences';cmd=[a.m.physical.DEVICECTL,'device','process','launch','--device','<private-device_identifier>','--timeout','90','--quiet','--json-output','-','--activate',bundle];a.require_activation_argv(cmd,bundle)
  for v in ([*cmd[:4],*cmd[-2:],*cmd[4:-2]],[*cmd[:-1],'--terminate-existing',bundle],[*cmd[:-1],a.m.physical.BUNDLE]):
   with self.assertRaises(ValueError):a.require_activation_argv(v,bundle)
 def test_complete_both_models_and_modes(self):
  rows=[dict(model=m,mode=k) for m in ('small','larger') for k in ('state','lifecycle')];a.require_complete([None]*12,rows)
  for bad in (rows[:-1],rows[:3]+[rows[0]],rows+[rows[0]]):
   with self.assertRaises(ValueError):a.require_complete([None]*12,bad)
  with self.assertRaises(ValueError):a.require_complete([None]*11,rows)
 def test_prior_and_remaining_graphs(self):
  self.assertEqual(len(a.plans(1)),13);self.assertEqual(len(a.plans(2)),9);self.assertEqual(len(a.plans(3)),4)
  self.assertEqual(a.plans(1)[3]['stage'],'state');self.assertEqual(a.plans(2)[2]['stage'],'lifecycle');self.assertTrue(all('-v2/' in v['path'] for v in a.plans(2)))
class AdmissionChecks(unittest.TestCase):
 def test_exact_driver_binding_closure(self):
  source=dict(sources={'src':dict(snapshot='copy')});name='source';reused={'reuse':'a'*64};e=dict(results=[dict(path='stage')],driver_bindings={n:'b'*64 for n in [name,'src','copy','reuse','stage',a.GATE,a.PREPARED+'/integration.json']});e['driver_bindings'][a.GATE]=a.GATE_SHA;e['driver_bindings'][a.PREPARED+'/integration.json']=a.PREPARED_SHA;a.require_driver_inventory(e,source,name,reused)
  for key in list(e['driver_bindings']):
   bad=copy.deepcopy(e);bad['driver_bindings'].pop(key)
   with self.assertRaises(ValueError):a.require_driver_inventory(bad,source,name,reused)
  bad=copy.deepcopy(e);bad['driver_bindings']['foreign']='a'*64
  with self.assertRaises(ValueError):a.require_driver_inventory(bad,source,name,reused)
 def test_tail_prior_binding_is_required(self):
  source=dict(sources={'src':dict(snapshot='copy')});e=dict(results=[dict(path='tail')],driver_bindings={n:'a'*64 for n in ['manifest','src','copy','tail',a.GATE,a.PREPARED+'/integration.json','old-stage',a.PRIOR_EXECUTION]});e['driver_bindings'][a.GATE]=a.GATE_SHA;e['driver_bindings'][a.PREPARED+'/integration.json']=a.PREPARED_SHA;extra={'old-stage',a.PRIOR_EXECUTION};a.require_driver_inventory(e,source,'manifest',{},extra)
  for key in extra:
   bad=copy.deepcopy(e);bad['driver_bindings'].pop(key)
   with self.assertRaises(ValueError):a.require_driver_inventory(bad,source,'manifest',{},extra)
 def test_gate_digest_and_policy(self):
  gate=dict(format='toklet-m7-phone-small-sustained-qualification-v1',passed=True,sustained_duration_qualified=True,review=dict(accepted=True,reviewer='Astra'),measured_native_seconds=600,measured_requests=10,warmups=3,hardware=dict(product_type='iPhone15,2',os_version='27.0.1'),physical_device_qualified=False,m7_complete=False);a.require_gate(gate)
  for k,v in [('passed',False),('measured_native_seconds',599),('measured_native_seconds',float('nan')),('measured_requests',9),('m7_complete',True)]:
   bad=copy.deepcopy(gate);bad[k]=v
   with self.assertRaises(ValueError):a.require_gate(bad)
 def test_exact_command_order_no_missing_or_extra_rows(self):
  labels=['device-before','current-install-before','preflight-process','lifecycle','ready-process','activate-settings','background-process','current-install-return','activate-harness','cleanup-process-1','device-after'];rows=[dict(stdout='private/'+v+'.stdout') for v in labels];a.require_labels(rows,labels)
  for bad in (rows[:-1],rows[1:]+rows[:1],rows+[rows[0]]):
   with self.assertRaises(ValueError):a.require_labels(bad,labels)
 def test_reset_rows_reject_extra_signal_or_process(self):
  rows=[dict(stdout='private/'+v+'.stdout') for v in ['process-1','signal-2','process-3','process-4']];a.require_reset_labels(rows)
  for bad in (rows[:-1],rows+[dict(stdout='signal-5.stdout')],rows+[dict(stdout='process-6.stdout')],rows[::-1]):
   with self.assertRaises(ValueError):a.require_reset_labels(bad)
if __name__=='__main__':unittest.main()
