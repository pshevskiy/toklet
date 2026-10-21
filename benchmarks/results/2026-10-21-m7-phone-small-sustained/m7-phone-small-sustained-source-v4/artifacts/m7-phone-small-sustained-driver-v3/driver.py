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
GATE='benchmarks/results/2026-10-21-m7-phone-tasks-qualification-v1.json'
GATE_SHA='17ff7fa4c429aa2dff0240642a50f63c6fcc4ab46bd118526cc99eb24c4f5b95'
def require_quality(raw,expected):
 if m.package.hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Exact accepted task qualification digest required')
 record=json.loads(raw)
 if (record['format']!='toklet-m7-phone-tasks-qualification-v1' or record['passed'] is not True or record['task_quality_qualified'] is not True
     or record['review']['accepted'] is not True or record['review']['reviewer']!='Astra' or record['full_cases_per_model']!=100
     or record['physical_device_qualified'] is not False or record['m7_complete'] is not False
     or record['new_generated_program_executions']!=0 or record['new_vm_runs']!=0 or len(record['models'])!=2
     or {r['model'] for r in record['models']}!={'small','larger'} or any(r['cases']!=100 or r['drop_percentage_points']>2 for r in record['models'])):raise ValueError('Accepted complete phone task quality required before sustained run')
 return record
def await_quality(pins):
 path=ROOT/GATE;raw=bounded(path,16*1024**2);require_quality(raw,GATE_SHA)
 m.integration.merge(pins,{GATE:m.package.hashlib.sha256(raw).hexdigest()})
def admit_source(path,pins):
 raw=bounded(path,1024*1024);name=str(path.relative_to(ROOT))
 if m.package.hashlib.sha256(raw).hexdigest()!=pins[name]:raise ValueError('Frozen source receipt consumed bytes differ')
 source=json.loads(raw)
 for n,v in source['sources'].items():m.integration.merge(pins,{n:v['sha256']})
def main():
 execution=OUT/'execution.json';exclusive(execution)
 modules=(m,m.package,m.integration,m.physical,m.storage)
 paths=[Path(__file__).resolve(),OUT/'checks.py',ROOT/'artifacts/m7-phone-small-sustained-source-v4/source.json',*[Path(v.__file__).resolve() for v in modules]]
 pins={str(p.relative_to(ROOT)):m.package.sha(p) for p in paths};results=[];primary=None;endpoints=[]
 admit_source(ROOT/'artifacts/m7-phone-small-sustained-source-v4/source.json',pins)
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
