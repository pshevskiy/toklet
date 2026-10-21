from pathlib import Path
import json,sys,time
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_measurement as m
NATIVE_SHA="06016a4e6eeea98c9c9c0f898cc4fd107c85415ed26cd3faa8de221386002971"
OUT=Path(__file__).resolve().parent
ARGS=(ROOT/'artifacts/m7-apple-package-v5',ROOT/'artifacts/m7-physical-admission-v1/private-selection.json','711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db',ROOT/'artifacts/m7-physical-prepared-v14','c5fd8e28d90413ece9d400f43c626e7b052f802241b6596c52d64229967e19b8')
def exclusive(path):
 if path.exists() or path.is_symlink():raise FileExistsError('Exclusive driver execution required')
def bounded(path,limit):
 with path.open('rb') as stream:raw=stream.read(limit+1)
 if len(raw)>limit:raise ValueError('Bounded driver input exceeded')
 return raw
def receipt(path,pins):
 raw=bounded(path,64*1024**2);digest=m.package.hashlib.sha256(raw).hexdigest();data=json.loads(raw)
 name=str(path.relative_to(ROOT));m.integration.merge(pins,{name:digest})
 return data,digest
def guard(pins):
 for n,h in pins.items():
  if m.package.sha(ROOT/n)!=h:raise ValueError('Driver endpoint changed: '+n)
def call(pins,results,stage,model,workload,destination,**options):
 guard(pins);code=m.main(*ARGS,model,workload,destination,native_sha=NATIVE_SHA,**options)
 path=destination/'measurement.json';data,digest=receipt(path,pins)
 results.append(dict(stage=stage,model=model,workload=workload,path=str(path.relative_to(ROOT)),sha256=digest,exit_code=code));guard(pins)
 if code:raise RuntimeError(stage+' failed: '+model+'/'+workload)
 return data,digest
def await_quality(pins):
 deadline=time.monotonic()+14400
 paths=[ROOT/'artifacts/m7-phone-task-followthrough-v1/execution.json',ROOT/'artifacts/m7-phone-task-quality-v1/quality.json']
 while True:
  if time.monotonic()>=deadline:raise TimeoutError('Current task quality completion deadline')
  records=[];raws=[]
  for path in paths:
   if not path.exists():break
   raw=bounded(path,64*1024**2)
   if not raw.endswith(b'\n'):break
   try:record=json.loads(raw)
   except json.JSONDecodeError:break
   records.append(record);raws.append(raw)
  if len(records)==2:
   workflow,quality=records
   if (workflow['passed'] is not True or quality.get('format')!='toklet-m7-phone-task-outcome-equivalence-v1' or workflow['primary_error'] or workflow['endpoint_errors'] or len(workflow['commands'])!=2
       or quality['passed'] is not True or quality['quality_completed'] is not True or quality['primary_error'] or quality['endpoint_errors']
       or len(quality['models'])!=2 or {r['model'] for r in quality['models']}!={'small','larger'}
       or any(r['cases']!=100 or r['drop_percentage_points']>2 for r in quality['models'])):raise ValueError('Complete exact phone task quality required before sustained run')
   if workflow['bindings'].get(str(paths[1].relative_to(ROOT)))!=m.package.hashlib.sha256(raws[1]).hexdigest():raise ValueError('Exact paired quality receipt required')
   for path,raw in zip(paths,raws):m.integration.merge(pins,{str(path.relative_to(ROOT)):m.package.hashlib.sha256(raw).hexdigest()})
   return
  time.sleep(1)
def admit_source(path,pins):
 raw=bounded(path,1024*1024);name=str(path.relative_to(ROOT))
 if m.package.hashlib.sha256(raw).hexdigest()!=pins[name]:raise ValueError('Frozen source receipt consumed bytes differ')
 source=json.loads(raw)
 for n,v in source['sources'].items():m.integration.merge(pins,{n:v['sha256']})
def main():
 execution=OUT/'execution.json';exclusive(execution)
 modules=(m,m.package,m.integration,m.physical,m.storage)
 paths=[Path(__file__).resolve(),OUT/'checks.py',ROOT/'artifacts/m7-phone-small-sustained-source-v2/source.json',*[Path(v.__file__).resolve() for v in modules]]
 pins={str(p.relative_to(ROOT)):m.package.sha(p) for p in paths};results=[];primary=None;endpoints=[]
 admit_source(ROOT/'artifacts/m7-phone-small-sustained-source-v2/source.json',pins)
 previous=m.package.signal.signal(m.package.signal.SIGTERM,m.package.interrupted)
 try:
  await_quality(pins)
  proof=ROOT/'artifacts/m7-phone-small-sustained-storage-proof-v1'
  _,proof_sha=call(pins,results,'storage-proof','tiny-cpu','short',proof)
  ready=ROOT/'artifacts/m7-phone-small-sustained-ready-v1'
  _,ready_sha=call(pins,results,'prepare','small','sustained',ready,prepare_only=True,storage_proof=proof,storage_sha=proof_sha)
  call(pins,results,'sustained','small','sustained',ROOT/'artifacts/m7-phone-small-sustained-v1',transfer=ready,transfer_sha=ready_sha)
 except BaseException as e:primary=type(e).__name__+': '+str(e)
 finally:
  m.package.signal.signal(m.package.signal.SIGTERM,previous)
  try:guard(pins)
  except BaseException as e:endpoints.append(type(e).__name__+': '+str(e))
  passed=primary is None and not endpoints and len(results)==3
  with execution.open('x') as f:json.dump(dict(passed=passed,driver_bindings=pins,results=results,primary_error=primary,endpoint_errors=endpoints,scope='Raw dedicated small-model sustained sequence; independent completed-duration/thermal analysis remains',physical_device_qualified=False,m7_complete=False),f,indent=2);f.write('\n')
  print(json.dumps(dict(passed=passed,stages=len(results),primary_error=primary,endpoint_errors=endpoints)))
 return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
