"""Replay the bounded typed-native probe; no full context acceptance."""
import argparse
import json
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'Cargo.toml').is_file() and (p/'tools/experiments').is_dir())
sys.path.insert(0,str(ROOT/'tools/experiments'))
import m6_typed_quality_audit as audit


def run():
 source=Path(__file__).resolve(); source_sha=audit.read_bytes(source)[1]
 path=ROOT/'artifacts/m6-typed-quality-small-4096-probe-v2.json'
 record,sha=audit.read_json(path)
 proof=audit.audit([path])
 if proof['bindings'].get(str(path.relative_to(ROOT)))!=sha:raise ValueError('Parsed probe bytes differ')
 original_path=ROOT/'artifacts/m6-context-small-quality-4096-v1.json'
 original,original_sha=audit.read_json(original_path)
 a,b=record['worker']['phases'][0],original['worker']['phases'][0]
 if a!=b:raise ValueError('Typed probe changed original native request/events')
 if record['quality']!=original['quality']:raise ValueError('Typed probe changed numerical quality summary')
 pins=dict(proof['bindings'])
 pins[str(source.relative_to(ROOT))]=source_sha
 pins[str(original_path.relative_to(ROOT))]=original_sha
 audit.verify_bindings(pins)
 return dict(format='toklet-m6-typed-probe-replay-v1',passed=True,m6_complete=False,
  records=1,raw_events=sum(r['raw_events'] for r in proof['records']),bindings=pins,replay=proof,
  original_native_request_events_and_quality_identical=True,
  scope='Bounded Small4K typed collector qualification only; larger32K fresh typed retry/full quality remain pending')


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 if args.output.exists():raise ValueError('Exclusive typed probe replay evidence exists')
 result=run()
 with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps({k:result[k] for k in ('passed','records','raw_events','original_native_request_events_and_quality_identical')}))
