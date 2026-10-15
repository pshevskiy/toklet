"""Replay four isolated component collections; no GPU launch or I8 adoption."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'Cargo.toml').is_file())
sys.path.insert(0,str(ROOT/'tools/experiments'))
import m6_kv_component as kv
import m6_kv_component_build as build
BUILD=ROOT/'artifacts/m6-kv-probe-build-v1.json'
BUILD_SHA='845b65158b5b3cbe4f89a2e0e2fc7a4a4a766bbdca7bd4bf6c9495f4dbcad107'

def bounded(path,limit):
 with path.open('rb') as f:data=f.read(limit+1)
 if len(data)>limit:raise ValueError('Qualification input exceeds bound')
 return data,hashlib.sha256(data).hexdigest()

def run():
 binary,pins,check=kv.build_admission(BUILD,BUILD_SHA)
 for p in (Path(__file__).resolve(),Path(kv.__file__).resolve(),Path(build.__file__).resolve()):pins[p]=bounded(p,1024**2)[1]
 expected={str(p.relative_to(ROOT)):sha for p,sha in pins.items() if p!=Path(__file__).resolve()}
 rows=[]
 for model in ('small','larger'):
  for collection in ('primary','confirmation'):
   path=ROOT/f'artifacts/m6-kv-component-{model}-{collection}-v1.json'
   record,sha=kv.load(path,8*1024**2);pins[path]=sha
   raw_path=path.with_suffix('.raw.jsonl');data,sha=bounded(raw_path,8*1024**2);pins[raw_path]=sha
   def reject(v):raise ValueError('Nonfinite raw JSON: '+v)
   events=[json.loads(line,parse_constant=reject) for line in data.splitlines()]
   stderr=path.with_suffix('.stderr');pins[stderr]=bounded(stderr,1024**2)[1]
   for key,value in dict(format='toklet-m6-kv-component-collection-v1',model_shape=model,collection=collection,
      passed=True,m6_complete=False,adoption_qualified=False,exit_code=0,identities_unchanged=True).items():
    kv.exact(record.get(key),value,key)
   if any(record.get(key) for key in ('cleanup_errors','environment_errors','error','endpoint_error','rejected_event')):
    raise ValueError('Component errors remain')
   if record.get('cleanup_errors')!=[] or record.get('environment_errors')!=[]:raise ValueError('Cleanup/observation coverage absent')
   if record['command']!=[str(binary),model,collection] or record['build_sha256']!=BUILD_SHA or record['provenance']!=expected:
    raise ValueError('Component command/provenance differs')
   if record['events']!=events or len(events)!=29:raise ValueError('Raw event coverage differs')
   kv.ready(events[0],model,collection);kv.correctness(events[1],model)
   for pair in range(13):
    for ordinal in range(2):kv.sample(events[2+2*pair+ordinal],model,pair,ordinal)
   kv.done(events[-1])
   summary=kv.summaries(events)
   if summary!=record['summaries']:raise ValueError('All-sample summary differs')
   usage=record['usage']
   if len(usage)!=29:raise ValueError('Native sample coverage differs')
   for i,u in enumerate(usage,1):
    kv.exact(u['event'],i,'native event index')
    peaks=[u[owner].get('lifetime_max_phys_footprint') for owner in ('worker','driver')]
    if any(type(v) is not int or v<=0 for v in peaks):raise ValueError('Native charge unavailable')
    kv.exact(u['joint_native_lifetime_peak_bytes'],sum(peaks),'native sum')
    if sum(peaks)>kv.LIMIT:raise ValueError('Native memory outside6GiB')
   for endpoint in ('environment_before','environment_after'):
    observations=record[endpoint]
    if set(observations)!=set(('platform','hardware','chip','battery','policy','thermal')) or any(type(v) is not str or not v for v in observations.values()):
     raise ValueError('Environment observation coverage differs')
   rows.append(dict(model_shape=model,collection=collection,raw_events=29,admitted_bytes=events[0]['admitted_bytes'],
    max_joint_native_bytes=max(u['joint_native_lifetime_peak_bytes'] for u in usage),correctness=events[1]['result'],
    summaries=summary,environment_before=record['environment_before'],environment_after=record['environment_after']))
 check()
 for p,sha in pins.items():
  if build.digest(p)!=sha:raise ValueError('Qualification endpoint changed: '+str(p))
 return dict(format='toklet-m6-kv-component-replay-v1',passed=True,records=rows,raw_events=116,
  bindings={str(p.relative_to(ROOT)):sha for p,sha in sorted(pins.items())},build_sha256=BUILD_SHA,
  all_four_gpu_median_gains_at_least_five_percent=all(r['summaries']['gpu_median_gain_at_least_five_percent'] for r in rows),
  adoption_qualified=False,m6_complete=False,
  scope='Synthetic decode arithmetic/memory/timing only; no cache-write, model/task quality or engine adoption qualification')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Exclusive replay evidence exists')
 result=run()
 with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(passed=result['passed'],bindings=len(result['bindings']),raw_events=116,
  gpu_median_gains=[(r['model_shape'],r['collection'],r['summaries']['gpu_median_reduction_fraction']) for r in result['records']]),indent=2))
