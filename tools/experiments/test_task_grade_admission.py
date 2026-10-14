"""Offline full-collection admission; never executes generated source or starts VM."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import task_grade_admission as admission
from test_task_cases import inventory,events
import test_metal_tasks as metal_fixtures
import test_cpu_task_reference as reference_fixtures

def current_inventory_fixture(record):
 # Historical native records predate the new v2 Rust module. Add its synthetic
 # identity only in this mocked unit fixture; never re-admit old native evidence.
 ids=dict(record['identities']);extra=str(admission.ROOT/'src/metal/q4_v2_projection.rs')
 assert extra not in ids
 ids[extra]='a'*64
 return ids

class GradingAdmission(unittest.TestCase):
 def reference_fixture(self):
  cases,_=inventory();ready=reference_fixtures.Reference().admission();items=[]
  empty={'history_tokens':0,'evaluated_tokens':0,'pending_tokens':0,'history_sha256':admission.reference.history_sha([])};error={'event':'error','state':empty,'memory':ready['memory']};raw=[ready,error]
  for case in cases:
   stream=events(case,projection='tiled')
   for e in stream:
    if e['event']=='token':del e['text']
   s=next(e['statistics'] for e in stream if e['event']=='done');ids=case['prompt_tokens']+[e['token_id'] for e in stream if e['event']=='token'];s.update(backend='cpu',tokenizer_resident=False,memory=ready['memory'],memory_admitted_bytes=ready['memory_admitted_bytes'],state={'history_tokens':len(ids),'evaluated_tokens':len(ids)-1,'pending_tokens':1,'history_sha256':admission.reference.history_sha(ids)})
   item=admission.reference.validate_response(case,stream,ready);item.update(visible_text='',visible_text_sha256=admission.reference.history_sha([]));items.append(item);raw+=stream
  usage={'worker':{'lifetime_max_phys_footprint':1},'driver':{'lifetime_max_phys_footprint':2},'joint_native_lifetime_peak_bytes':3}
  phase={'after_owned_model_exit':True,'model_resident_bytes':0,'tokenizer_allowance_bytes':admission.reference.TOKENIZER_ALLOWANCE,'application_allowance_bytes':admission.reference.APP,'admitted_bytes':admission.reference.TOKENIZER_ALLOWANCE+admission.reference.APP,'native_driver':{'lifetime_max_phys_footprint':1}}
  record={'format':'toklet-cpu-task-reference-collection-v1','passed':True,'collection_completed':True,'check_only':False,'quality_completed':False,'identities_unchanged':True,'endpoint_errors':[],'task_payload_sha256':admission.tasks.TASK_SHA,'task_ids':[c['id'] for c in cases],'run':{'passed':True,'model_exited':True,'worker_pid':1234567,'exit_code':0,'cleanup_errors':[],'ready':ready,'cases':items,'idle_usage':[usage]*102,'checks':[{'request':{'action':'generate','tokens':[1]*307,'max_new_tokens':256},'response':error}],'decoder_phase':phase}}
  return record,cases,raw

 def test_full_raw_reference_and_separate_decoder_phase(self):
  record,cases,raw=self.reference_fixture();r=admission.validate_records(record,cases,raw);self.assertEqual(r['identity']['backend'],'cpu');self.assertEqual(r['identity']['model'],'larger');self.assertEqual(len(r['responses']),100)
  for key,value in [('model_resident_bytes',1),('after_owned_model_exit',False),('admitted_bytes',admission.reference.APP)]:
   changed=copy.deepcopy(record);changed['run']['decoder_phase'][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):admission.validate_records(changed,cases,raw)
 def fixture(self):
  cases,_=inventory();ready=metal_fixtures.MetalTasks().admission();items=[];raw=[ready]
  for case in cases:
   stream=events(case,projection='tiled');s=next(e['statistics'] for e in stream if e['event']=='done');s.update({k:v for k,v in ready.items() if k not in ('event','context_capacity')});response=admission.metal.collect_response(case,stream,ready);items.append({'max_output_tokens':case['max_output_tokens'],'response':response,'native_events':stream});raw+=stream
  usage={'worker':{'lifetime_max_phys_footprint':1},'driver':{'lifetime_max_phys_footprint':2},'joint_native_lifetime_peak_bytes':3}
  record={'format':'toklet-metal-task-collection-v1','passed':True,'collection_completed':True,'check_only':False,'quality_completed':False,'identities_unchanged':True,'endpoint_errors':[],'task_payload_sha256':admission.tasks.TASK_SHA,'task_ids':[c['id'] for c in cases],'model':'small','backend':'metal','projection_mode':'HalfCompensated','normalization_mode':'Original','readback_mode':'FullLogits','run':{'passed':True,'model_exited':True,'worker_pid':1234567,'exit_code':0,'cleanup_errors':[],'ready':ready,'cases':items,'idle_usage':[usage]*101}}
  return record,cases,raw
 def test_full_metadata_and100frozen_caps_preserved(self):
  record,cases,raw=self.fixture();result=admission.validate_records(record,cases,raw);self.assertEqual(len(result['responses']),100);self.assertEqual(result['identity']['backend'],'metal');self.assertEqual(result['identity']['projection_mode'],'HalfCompensated')
 def test_published_protocol_checks_cannot_be_graded(self):
  cases,_=inventory()
  for name in ('small-half-tasks','larger-half-tasks','larger-cpu-task-reference'):
   path=admission.ROOT/f'benchmarks/results/2026-10-14-{name}-check-v1.json';record=json.loads(path.read_text())
   with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Complete frozen100'):admission.validate_records(record,cases,[])
 def test_incomplete_flags_ids_order_caps_and_worker_exit_reject(self):
  for mutate in (lambda p:p.update(collection_completed=False),lambda p:p.update(check_only=True),lambda p:p.update(quality_completed=True),lambda p:p['task_ids'].reverse(),lambda p:p['run']['cases'].reverse(),lambda p:p['run']['cases'].pop(),lambda p:p['run'].update(model_exited=False),lambda p:p['run'].update(exit_code=True),lambda p:p['run'].update(worker_pid=True),lambda p:p['run']['cases'][0].update(max_output_tokens=8)):
   record,cases,raw=self.fixture();mutate(record)
   with self.assertRaises(ValueError):admission.validate_records(record,cases,raw)
 def test_raw_coverage_and_native_charge_are_required(self):
  record,cases,raw=self.fixture()
  with self.assertRaisesRegex(ValueError,'raw event'):admission.validate_records(record,cases,raw[:-1])
  record['run']['idle_usage'][0]['joint_native_lifetime_peak_bytes']=True
  with self.assertRaisesRegex(ValueError,'native charge'):admission.validate_records(record,cases,raw)
 def test_cpu_mode_masquerade_and_bad_memory_reject(self):
  record,cases,raw=self.fixture();record['backend']='cpu'
  with self.assertRaisesRegex(ValueError,'Metal collection metadata'):admission.validate_records(record,cases,raw)
  record,cases,raw=self.fixture();record['run']['ready']['execution']['projection_mode']='Original'
  with self.assertRaises(ValueError):admission.validate_records(record,cases,raw)

 def test_every_producer_inventory_entry_is_required(self):
  manifest=admission.frozen_manifest()
  for name,kind,model,binary in (
   ('small-half-tasks','toklet-metal-task-collection-v1','small',admission.ROOT/'artifacts/m5-cli-half-native-binary-v1/toklet'),
   ('larger-cpu-task-reference','toklet-cpu-task-reference-collection-v1','larger',admission.ROOT/'artifacts/m5-cpu-task-reference-native-binary-v1/cpu_task_reference')):
   record=json.loads((admission.ROOT/f'benchmarks/results/2026-10-14-{name}-check-v1.json').read_text());ids=current_inventory_fixture(record);source=admission.ROOT/'models'/admission.controlled.MODELS[model][0]
   self.assertEqual(set(ids),admission.inventory_paths(kind,binary,source,manifest))
   with patch.object(admission.metal_engine,'digest',side_effect=lambda p:ids[str(p)]):
    admission.validate_inventory(ids,kind,binary,ids[str(binary)],source,manifest,model)
    for entry in ids:
     changed=dict(ids);del changed[entry]
     with self.subTest(backend=kind,entry=entry),self.assertRaisesRegex(ValueError,'Complete backend'):admission.validate_inventory(changed,kind,binary,ids[str(binary)],source,manifest,model)
    changed={**ids,str(admission.ROOT/'extra.py'):'a'*64}
    with self.assertRaisesRegex(ValueError,'Complete backend'):admission.validate_inventory(changed,kind,binary,ids[str(binary)],source,manifest,model)

 def test_binary_replacement_never_becomes_expected_identity(self):
  manifest=admission.frozen_manifest();source=admission.ROOT/'models'/admission.controlled.MODELS['small'][0];kind='toklet-metal-task-collection-v1';binary=admission.ROOT/'artifacts/m5-cli-half-native-binary-v1/toklet'
  record=json.loads((admission.ROOT/'benchmarks/results/2026-10-14-small-half-tasks-check-v1.json').read_text());ids=current_inventory_fixture(record);expected=ids[str(binary)];replacement='a'*64
  # Simulate replacement immediately after the earlier qualified source check.
  with patch.object(admission.metal,'qualified_cli',return_value=binary),patch.object(admission.metal_engine,'digest',return_value=replacement):
   self.assertEqual(admission.qualified_binary(kind),(binary,expected))
  replaced={**ids,str(binary):replacement}
  with patch.object(admission.metal_engine,'digest',side_effect=lambda p:replaced[str(p)]):
   with self.assertRaisesRegex(ValueError,'Required qualified'):admission.validate_inventory(replaced,kind,binary,expected,source,manifest,'small')
   with self.assertRaisesRegex(ValueError,'identity changed'):admission.validate_inventory(ids,kind,binary,expected,source,manifest,'small')

 def test_contract_replacement_rejects_before_parsing(self):
  with patch.object(admission,'bounded',return_value=b'{}'),patch.object(admission.evidence,'decode_event',side_effect=AssertionError('must not parse')):
   with self.assertRaisesRegex(ValueError,'Frozen task contract'):admission.frozen_manifest()

if __name__=='__main__':unittest.main()
