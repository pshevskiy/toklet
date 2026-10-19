"""Bind complete context replay to separately reviewed exact winner cases.

No generic winner-margin acceptance, modified automatic audit, KV decision or
milestone acceptance. Execute only after the native queue is completely idle.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'Cargo.toml').is_file() and (p/'tools/experiments').is_dir())
sys.path.insert(0,str(ROOT/'tools/experiments'))
import m6_fp32_context_audit as typed
RECEIPT='benchmarks/results/2026-10-16-m6-winner-investigation-qualification-v1.json'
RECEIPT_SHA='1d10ed435bcbe355af2ed2c204728795489624a311f17338e7fe8c839e57c735'
FIELDS=('model','depth','original_quality_sha256','target_index','prefix_sha256','reference_logits_sha256','online_logits_sha256')

def resolution(proof,loaded,receipt,arrays):
 derived=proof['derived'];q=derived['long_quality']
 if proof.get('kind')!='context' or derived.get('complete_frozen_context_coverage') is not True or derived.get('record_count')!=32 or q.get('complete_depth_target_coverage') is not True or q.get('frozen_nll_gates_passed') is not True:raise ValueError('Complete unchanged context/quality gates required')
 for model in typed.quality.MODELS:
  m=q['models'][model]
  if type(m.get('scored_targets')) is not int or m['scored_targets']!=512 or m.get('pooled_gate_passed') is not True or m.get('each_depth_gate_passed') is not True:raise ValueError('Full per-model depth and pooled gates required')
 if receipt.get('format')!='toklet-m6-winner-investigation-qualification-v1' or receipt.get('passed') is not True or receipt.get('review',{}).get('accepted') is not True or receipt.get('review',{}).get('reviewer')!='Astra' or receipt.get('investigation_assessment',{}).get('complete') is not True or receipt.get('investigation_assessment',{}).get('reviewer')!='Astra':raise ValueError('Accepted separate Astra exact-case review required')
 if receipt.get('m6_complete') is not False or receipt.get('full_long_quality_qualified') is not False or receipt.get('original_automatic_unresolved_results_preserved') is not True:raise ValueError('Original manual review scope differs')
 wanted=[]
 for path,record,sha in loaded:
  if record['plan']!='quality':continue
  if proof['bindings'].get(str(path.relative_to(ROOT)))!=sha:raise ValueError('Quality collection SHA not replayed')
  for row in record['worker']['phases'][0]['events'][1:-1]:
   if row['reference_greedy']==row['online_greedy']:continue
   wanted.append(dict(model=record['model'],depth=record['depth'],original_quality_sha256=sha,target_index=row['target_index'],prefix_sha256=typed.audit.base.shared.history_sha(arrays[record['depth']][:row['prefix_tokens']]),reference_logits_sha256=row['reference_logits_sha256'],online_logits_sha256=row['online_logits_sha256']))
 if len(wanted)!=2 or {(c['model'],c['depth'],c['target_index']) for c in wanted}!={('larger',4096,90),('larger',4096,127)}:raise ValueError('Only the two explicitly reviewed cases are admitted')
 actual=receipt.get('investigated_cases')
 if type(actual) is not list or len(actual)!=len(wanted):raise ValueError('Manual case coverage differs')
 def key(row):
  if any(k not in row for k in FIELDS):raise ValueError('Incomplete manual case key')
  if type(row['depth']) is not int or type(row['target_index']) is not int:raise ValueError('Manual case integer type differs')
  return tuple(row[k] for k in FIELDS)
 if {key(r) for r in actual}!={key(r) for r in wanted}:raise ValueError('Manual review does not match exact current cases')
 if sum(m['argmax_change_count'] for m in q['models'].values())!=len(wanted):raise ValueError('Derived changed-case coverage differs')
 if proof.get('passed') is not False or q.get('argmax_investigations_complete') is not False:raise ValueError('Original automatic unresolved flags must remain false')
 return wanted

def run():
 source=Path(__file__).resolve();source_sha=typed.audit.read_bytes(source)[1]
 paths=[ROOT/'artifacts'/f'm6-context-{model}-{plan}-{depth}-v{2 if (plan=="quality" and ((model=="small" and depth==16384) or (model=="larger" and depth==32768))) or (model=="small" and plan=="long" and depth==32768) else 1}.json' for model in typed.quality.MODELS for plan in typed.scope.PLANS for depth in typed.quality.DEPTHS]
 loaded=[(p,*typed.audit.read_json(p)) for p in paths]
 proof=typed.run(paths,'context')
 receipt_path=ROOT/RECEIPT;receipt,sha=typed.audit.read_json(receipt_path)
 if sha!=RECEIPT_SHA:raise ValueError('Explicit reviewed winner receipt SHA differs')
 _,arrays,_,_,_=typed.audit.base.inputs('larger')
 cases=resolution(proof,loaded,receipt,arrays)
 pins=dict(proof['bindings'])
 def merge(items):
  for p,value in items.items():
   if p in pins and pins[p]!=value:raise ValueError('Conflicting final context binding')
   pins[p]=value
 merge(receipt['source_pins']);merge(receipt['evidence_pins'])
 merge({RECEIPT:sha,str(source.relative_to(ROOT)):source_sha})
 typed.audit.verify_bindings(pins)
 return dict(format='toklet-m6-context-reviewed-closure-v1',passed=True,m6_complete=False,kv_retention_qualified=False,complete_frozen_context_coverage=True,frozen_nll_gates_passed=True,all_argmax_changes_investigated=True,reviewed_cases=cases,manual_review_receipt=RECEIPT,manual_review_sha256=sha,automatic_audit=proof,bindings=pins,scope='Complete32contextroles plus separately reviewed two exact original winner cases only; automatic unresolved audit preserved, final KV and overall M6 synthesis separate')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 if args.output.exists():raise ValueError('Exclusive reviewed context closure output exists')
 result=run()
 with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(passed=result['passed'],context_records=len(result['automatic_audit']['replay']['records']),raw_events=sum(r['raw_events'] for r in result['automatic_audit']['replay']['records']),bindings=len(result['bindings']),manual_cases=len(result['reviewed_cases']),m6_complete=False)))
