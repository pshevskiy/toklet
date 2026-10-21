from pathlib import Path
import json,sys
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_quality as q
import m7_native_teacher_check as teacher
m=q.measurement
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
 guard(pins);code=m.main(*ARGS,model,workload,destination,native_sha=q.NATIVE_SHA,**options)
 path=destination/'measurement.json';data,digest=receipt(path,pins)
 results.append(dict(stage=stage,model=model,workload=workload,path=str(path.relative_to(ROOT)),sha256=digest,exit_code=code));guard(pins)
 if code:raise RuntimeError(stage+' failed: '+model+'/'+workload)
 return data,digest
def tiny_scores(folder,record,pins):
 lines=[]
 for label in ('measurement.stdout','measurement.stderr'):
  path=folder/'private-logs'/label;name=str(path.relative_to(ROOT));expected=record['private_evidence_pins'][name]
  raw=bounded(path,64*1024**2)
  if m.package.hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Tiny console binding differs')
  m.integration.merge(pins,{name:expected});lines.extend(raw.decode().splitlines())
 samples=[json.loads(s.removeprefix('TOKLET_QUALITY_SAMPLE_JSON=')) for s in lines if s.startswith('TOKLET_QUALITY_SAMPLE_JSON=')]
 if len(samples)!=3:raise ValueError('Three tiny scores required')
 return [r['target']['score']['negativeLogLikelihood'] for r in samples],[r['target']['score']['greedy'] for r in samples]
def main():
 execution=OUT/'execution.json';exclusive(execution)
 modules=(q,m,m.package,m.integration,m.physical,m.storage,teacher)
 paths=[Path(__file__).resolve(),OUT/'checks.py',*[Path(v.__file__).resolve() for v in modules],*[ROOT/n for n in teacher.ORACLE]]
 pins={str(p.relative_to(ROOT)):m.package.sha(p) for p in paths};results=[];primary=None;endpoints=[];tiny={}
 try:
  oracle=teacher.independent_losses(bounded(ROOT/'tests/fixtures/tiny-v1/cases.prompt.logits.f32le',1024))
  for model in ('tiny-cpu','tiny-metal'):
   folder=ROOT/('artifacts/m7-phone-loss-'+model+'-v2');record,_=call(pins,results,'tiny-quality',model,'short',folder,quality=True)
   losses,greedy=tiny_scores(folder,record,pins)
   if model=='tiny-cpu' and any(abs(a-b)>=1e-5 for a,b in zip(losses,oracle)):raise ValueError('Independent tiny CPU loss differs')
   tiny[model]=dict(losses=losses,greedy=greedy,independent_cpu_oracle=oracle);guard(pins)
  for model in ('small','larger'):
   for workload in ('short','full'):
    stem='artifacts/m7-phone-loss-'+model+'-'+workload
    proof=ROOT/(stem+'-storage-proof-v1');_,proof_sha=call(pins,results,'storage-proof','tiny-cpu','short',proof)
    ready=ROOT/(stem+'-ready-v1');_,ready_sha=call(pins,results,'prepare',model,workload,ready,quality=True,prepare_only=True,storage_proof=proof,storage_sha=proof_sha)
    call(pins,results,'quality',model,workload,ROOT/(stem+'-v1'),quality=True,transfer=ready,transfer_sha=ready_sha)
 except BaseException as e:primary=type(e).__name__+': '+str(e)
 finally:
  try:guard(pins)
  except BaseException as e:endpoints.append(type(e).__name__+': '+str(e))
  passed=primary is None and not endpoints and len(results)==14
  with execution.open('x') as f:json.dump(dict(passed=passed,driver_bindings=pins,results=results,tiny_oracle=tiny,primary_error=primary,endpoint_errors=endpoints,scope='Raw physical loss collection; independent full quality analysis remains',physical_device_qualified=False,m7_complete=False),f,indent=2);f.write('\n')
  print(json.dumps(dict(passed=passed,stages=len(results),primary_error=primary,endpoint_errors=endpoints)))
 return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
