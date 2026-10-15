"""Partial saved-data compatibility only; not complete32-record qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/experiments'))
import m6_context_scope_audit as scope

def run():
 paths=[ROOT/'artifacts'/name for name in (
  'm6-context-small-quality-4096-v1.json','m6-context-small-quality-8192-v1.json',
  'm6-context-small-profile-32768-v1.json','m6-context-larger-profile-32768-v1.json')]
 source=Path(__file__).resolve(); source_sha=scope.audit.read_bytes(source)[1]
 module_pins={str(Path(m.__file__).resolve().relative_to(ROOT)):scope.audit.read_bytes(Path(m.__file__).resolve())[1] for m in (scope,scope.quality)}
 loaded=[(p,*scope.audit.read_json(p)) for p in paths]
 proof=scope.audit.audit(paths)
 expected={('small','quality',4096),('small','quality',8192),('small','profile',32768),('larger','profile',32768)}
 actual={(r['model'],r['plan'],r['depth']) for _,r,_ in loaded}
 if actual!=expected: raise ValueError('Partial compatibility roles differ')
 for p,_,sha in loaded:
  if proof['bindings'].get(str(p.relative_to(ROOT)))!=sha: raise ValueError('Parsed compatibility input differs')
 try: scope.summarize([r for _,r,_ in loaded])
 except ValueError as error:
  if str(error)!='Require all32 long-context model/plan/depth records': raise
 else: raise ValueError('Incomplete real coverage was accepted')
 profile_rows={r['model']:r['profile_medians'] for _,r,_ in loaded if r['plan']=='profile'}
 pins=dict(proof['bindings'])
 pins.update(module_pins)
 pins[str(source.relative_to(ROOT))]=source_sha
 scope.audit.verify_bindings(pins)
 return dict(format='toklet-m6-context-scope-compatibility-v1',passed=True,
  scope='Four real records replayed; incomplete32-record context synthesis correctly rejected; no full context/M6 acceptance',
  complete_frozen_context_coverage=False,m6_complete=False,records=4,
  raw_events=sum(r['raw_events'] for r in proof['records']),bindings=pins,replay=proof,
  max_depth_profile_medians=profile_rows,incomplete_coverage_rejected=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Exclusive compatibility evidence exists')
 result=run()
 with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:result[k] for k in ('passed','records','raw_events','incomplete_coverage_rejected','max_depth_profile_medians')}))
