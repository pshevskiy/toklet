"""Portable exact-case resolution checks; no native or full replay execution."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'Cargo.toml').is_file() and (p/'tools/experiments').is_dir())
spec=importlib.util.spec_from_file_location('reviewed_closure',ROOT/'tools/experiments/m6_context_reviewed_closure.py')
closure=importlib.util.module_from_spec(spec);spec.loader.exec_module(closure)

class Resolution(unittest.TestCase):
 def setUp(self):
  self.receipt=json.loads((ROOT/closure.RECEIPT).read_text())
  original=json.loads((ROOT/'benchmarks/results/2026-10-16-m6-context-larger-quality-4096-v1.json').read_text())
  self.arrays={4096:list(struct.unpack('<4096I',(ROOT/'benchmarks/m6-context-v1/context_4096.u32le').read_bytes()))}
  self.loaded=[];pins={}
  for model in closure.typed.quality.MODELS:
   for plan in closure.typed.scope.PLANS:
    for depth in closure.typed.quality.DEPTHS:
     path=ROOT/f'artifacts/unit-{model}-{plan}-{depth}.json'
     record=dict(model=model,plan=plan,depth=depth,worker=dict(phases=[dict(events=[dict(event='comparison_begin'),dict(event='comparison_done')])]))
     sha='0'*64
     if (model,plan,depth)==('larger','quality',4096):record=copy.deepcopy(original);sha=self.receipt['original_quality_sha256']
     self.loaded.append((path,record,sha));pins[str(path.relative_to(ROOT))]=sha
  q=dict(complete_depth_target_coverage=True,frozen_nll_gates_passed=True,argmax_investigations_complete=False,models=dict(small=dict(argmax_change_count=0,scored_targets=512,pooled_gate_passed=True,each_depth_gate_passed=True),larger=dict(argmax_change_count=2,scored_targets=512,pooled_gate_passed=True,each_depth_gate_passed=True)))
  self.proof=dict(kind='context',passed=False,bindings=pins,derived=dict(complete_frozen_context_coverage=True,record_count=32,long_quality=q))
 def run_resolution(self):return closure.resolution(self.proof,self.loaded,self.receipt,self.arrays)
 def test_exact_real_reviewed_cases(self):
  before=copy.deepcopy((self.proof,self.loaded,self.receipt,self.arrays))
  rows=self.run_resolution();self.assertEqual([r['target_index'] for r in rows],[90,127]);self.assertEqual((self.proof,self.loaded,self.receipt,self.arrays),before)
 def test_each_manual_identity_mismatch(self):
  original=copy.deepcopy(self.receipt)
  for field in closure.FIELDS:
   with self.subTest(field=field):
    self.receipt=copy.deepcopy(original);row=self.receipt['investigated_cases'][0];row[field]=8192 if field=='depth' else 91 if field=='target_index' else 'changed'
    with self.assertRaises(ValueError):self.run_resolution()
 def test_manual_coverage_not_generic(self):
  original=copy.deepcopy(self.receipt)
  for rows in (original['investigated_cases'][:1],original['investigated_cases']*2,[original['investigated_cases'][0]]*2):
   self.receipt=copy.deepcopy(original);self.receipt['investigated_cases']=rows
   with self.assertRaises(ValueError):self.run_resolution()
 def test_review_and_quality_requirements(self):
  for group,key,value in ((None,'format','other'),(None,'passed',False),('review','accepted',False),('review','reviewer','other'),('investigation_assessment','complete',False),('investigation_assessment','reviewer','other'),(None,'m6_complete',True),(None,'full_long_quality_qualified',True),(None,'original_automatic_unresolved_results_preserved',False)):
   with self.subTest(group=group,key=key):
    original=copy.deepcopy(self.receipt);target=self.receipt if group is None else self.receipt[group];target[key]=value
    with self.assertRaises(ValueError):self.run_resolution()
    self.receipt=original
  self.proof['derived']['long_quality']['frozen_nll_gates_passed']=False
  with self.assertRaises(ValueError):self.run_resolution()
 def test_changed_row_and_prefix_must_match(self):
  index=next(i for i,(_,r,_) in enumerate(self.loaded) if (r['model'],r['plan'],r['depth'])==('larger','quality',4096));original=copy.deepcopy(self.loaded[index])
  for key,value in (('prefix_tokens',4059),('reference_logits_sha256','0'*64),('online_logits_sha256','0'*64),('target_index',91)):
   with self.subTest(key=key):
    self.loaded[index]=copy.deepcopy(original);self.loaded[index][1]['worker']['phases'][0]['events'][91][key]=value
    with self.assertRaises(ValueError):self.run_resolution()
 def test_unreviewed_additional_flip_rejected(self):
  for _,record,_ in self.loaded:
   if (record['model'],record['plan'],record['depth'])==('larger','quality',4096):record['worker']['phases'][0]['events'][1]['online_greedy']=(record['worker']['phases'][0]['events'][1]['reference_greedy']+1)%151936;break
  with self.assertRaises(ValueError):self.run_resolution()
 def test_automatic_flags_and_collection_sha_preserved(self):
  self.proof['passed']=True
  with self.assertRaises(ValueError):self.run_resolution()
  self.proof['passed']=False;self.proof['derived']['long_quality']['argmax_investigations_complete']=True
  with self.assertRaises(ValueError):self.run_resolution()
  self.proof['derived']['long_quality']['argmax_investigations_complete']=False
  path,record,_=next(x for x in self.loaded if (x[1]['model'],x[1]['plan'],x[1]['depth'])==('larger','quality',4096));self.proof['bindings'][str(path.relative_to(ROOT))]='f'*64
  with self.assertRaises(ValueError):self.run_resolution()
 def test_per_model_gates_cannot_be_waived(self):
  for model in ('small','larger'):
   for field,value in (('scored_targets',511),('pooled_gate_passed',False),('each_depth_gate_passed',False)):
    with self.subTest(model=model,field=field):
     original=self.proof['derived']['long_quality']['models'][model][field];self.proof['derived']['long_quality']['models'][model][field]=value
     with self.assertRaises(ValueError):self.run_resolution()
     self.proof['derived']['long_quality']['models'][model][field]=original
 def test_explicit_reviewed_receipt_sha_guard(self):
  with patch.object(closure.typed,'run',return_value=self.proof),patch.object(closure.typed.audit,'read_json',return_value=(self.receipt,'0'*64)):
   with self.assertRaisesRegex(ValueError,'Explicit reviewed winner receipt SHA differs'):closure.run()
 def test_bool_case_index_rejected(self):
  self.receipt['investigated_cases'][0]['target_index']=True
  with self.assertRaises(ValueError):self.run_resolution()

if __name__=='__main__':unittest.main()
