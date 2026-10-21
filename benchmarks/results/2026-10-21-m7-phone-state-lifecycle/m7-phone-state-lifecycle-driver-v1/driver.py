from pathlib import Path
import json,sys
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_measurement as m
import m7_physical_state as state
import m7_physical_lifecycle as life
OUT=Path(__file__).resolve().parent
NATIVE_SHA='06016a4e6eeea98c9c9c0f898cc4fd107c85415ed26cd3faa8de221386002971'
PRIVATE='artifacts/m7-physical-admission-v1/private-selection.json'
PRIVATE_SHA='711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db'
PREPARED='artifacts/m7-physical-prepared-v16'
SOURCE='artifacts/m7-phone-state-lifecycle-source-v2/source.json'
GATE='benchmarks/results/2026-10-21-m7-phone-small-sustained-qualification-v1.json'
def exclusive(path):
 if path.exists() or path.is_symlink():raise FileExistsError('Exclusive driver output required')
def consume(path,expected,pins,limit=64*1024**2):
 with path.open('rb') as f:raw=f.read(limit+1)
 if len(raw)>limit:raise ValueError('Bounded workflow input exceeded')
 h=m.package.hashlib.sha256(raw).hexdigest()
 if expected is not None and h!=expected:raise ValueError('Consumed workflow bytes differ')
 m.integration.merge(pins,{str(path.relative_to(ROOT)):h});return json.loads(raw),h

def admit_source(expected,pins):
 source,_=consume(ROOT/SOURCE,expected,pins,16*1024**2)
 if source['format']!='toklet-m7-phone-state-lifecycle-source-v1' or not source['sources']:raise ValueError('Exact nonempty source snapshot required')
 for n,v in source['sources'].items():
  m.integration.merge(pins,{n:v['sha256']});path=ROOT/v['snapshot']
  if not path.resolve().is_relative_to((ROOT/'artifacts/m7-phone-state-lifecycle-source-v2').resolve()) or path.is_symlink() or m.package.sha(path)!=v['sha256']:raise ValueError('Exact frozen source copy required')
  m.integration.merge(pins,{str(path.relative_to(ROOT)):v['sha256']})
 required={str(Path(v.__file__).resolve().relative_to(ROOT)) for v in (m,state,life,m.package,m.integration,m.physical,m.storage)}|{str(Path(__file__).resolve().relative_to(ROOT)),str(Path(__file__).with_name('checks.py').resolve().relative_to(ROOT))}
 if not required.issubset(source['sources']):raise ValueError('Workflow source closure missing')
 guard(pins)
def admit_sustained(expected,pins):
 gate,_=consume(ROOT/GATE,expected,pins)
 if (gate['format']!='toklet-m7-phone-small-sustained-qualification-v1' or gate['passed'] is not True or gate['sustained_duration_qualified'] is not True
     or gate['review']['accepted'] is not True or gate['review']['reviewer']!='Astra' or not m.nonnegative(gate['measured_native_seconds']) or gate['measured_native_seconds']<600
     or type(gate['measured_requests']) is not int or gate['measured_requests']<10 or gate['warmups']!=3
     or gate['hardware']['product_type']!='iPhone15,2' or gate['hardware']['os_version']!='27.0.1'
     or gate['physical_device_qualified'] is not False or gate['m7_complete'] is not False):raise ValueError('Accepted sustained evidence required before replacing app')
 return gate
def guard(pins):m.verify_inputs(pins)
def stages():
 result=[dict(stage='tiny-integration',model='tiny-cpu+metal',path='artifacts/m7-state-lifecycle-tiny-v1/integration.json')]
 for model in ('small','larger'):
  for mode in ('state','lifecycle'):
   stem='artifacts/m7-phone-'+mode+'-'+model
   result.extend([dict(stage='storage-proof',model='tiny-cpu',path=stem+'-storage-proof-v1/measurement.json'),dict(stage=mode+'-prepare',model=model,path=stem+'-ready-v1/measurement.json'),dict(stage=mode,model=model,path=stem+'-v1/'+('lifecycle.json' if mode=='lifecycle' else 'measurement.json'))])
 return result
def collect(prepared_sha,pins,results):
 args=(ROOT/'artifacts/m7-apple-package-v5',ROOT/PRIVATE,PRIVATE_SHA,ROOT/PREPARED,prepared_sha)
 plan=stages()
 def finish(stage,code):
  record,h=consume(ROOT/stage['path'],None,pins)
  results.append(dict(**stage,exit_code=code,sha256=h));guard(pins)
  if code!=0 or record['passed'] is not True or record['primary_error'] or record['endpoint_errors'] or record['cleanup_errors']:raise ValueError('Physical workflow stage failed: '+stage['stage'])
  return h
 guard(pins)
 code=m.physical.main(args[0],NATIVE_SHA,args[1],PRIVATE_SHA,(ROOT/plan[0]['path']).parent,reuse=args[3],reuse_sha=prepared_sha)
 finish(plan[0],code)
 for start in range(1,len(plan),3):
  proof,ready,run=plan[start:start+3];mode=run['stage'];model=run['model'];guard(pins)
  proof_sha=finish(proof,m.main(*args,'tiny-cpu','short',(ROOT/proof['path']).parent,native_sha=NATIVE_SHA))
  guard(pins);ready_path=(ROOT/ready['path']).parent
  ready_sha=finish(ready,m.main(*args,model,'512',ready_path,prepare_only=True,storage_proof=(ROOT/proof['path']).parent,storage_sha=proof_sha,native_sha=NATIVE_SHA,**{mode:True}))
  guard(pins)
  if mode=='state':code=m.main(*args,model,'512',(ROOT/run['path']).parent,transfer=ready_path,transfer_sha=ready_sha,native_sha=NATIVE_SHA,state=True)
  else:code=life.collect(args[0],args[1],PRIVATE_SHA,args[3],prepared_sha,ready_path,ready_sha,model,(ROOT/run['path']).parent)
  finish(run,code)
def main(prepared_sha,sustained_sha,source_sha):
 if not all(m.storage.digest(h) for h in (prepared_sha,sustained_sha,source_sha)):raise ValueError('Explicit prepared/sustained/source digests required')
 execution=OUT/'execution.json';exclusive(execution)
 for row in stages():exclusive((ROOT/row['path']).parent)
 pins={};results=[];primary=None;endpoints=[]
 previous=m.package.signal.signal(m.package.signal.SIGTERM,m.package.interrupted)
 try:
  admit_source(source_sha,pins);admit_sustained(sustained_sha,pins)
  prepared,_=consume(ROOT/PREPARED/'integration.json',prepared_sha,pins)
  if prepared['passed'] is not True or prepared['format']!='toklet-m7-physical-prepared-v1':raise ValueError('Exact prepared app required')
  selected=m.physical.selection(ROOT/PRIVATE,PRIVATE_SHA);_,extra,_,_=m.physical.admit_reuse(ROOT/PREPARED,prepared_sha,selected,NATIVE_SHA);m.integration.merge(pins,extra)
  collect(prepared_sha,pins,results)
 except BaseException as error:primary=type(error).__name__+': '+str(error)
 finally:
  m.package.signal.signal(m.package.signal.SIGTERM,previous)
  try:guard(pins)
  except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
  passed=primary is None and not endpoints and len(results)==13
  with execution.open('x') as f:json.dump(dict(passed=passed,driver_bindings=pins,results=results,primary_error=primary,endpoint_errors=endpoints,physical_device_qualified=False,m7_complete=False,scope='Raw new-app tiny integration and both-model state/actual OS lifecycle collection; independent analysis required'),f,indent=2);f.write('\n')
  print(json.dumps(dict(passed=passed,stages=len(results),primary_error=primary,endpoint_errors=endpoints)));return 0 if passed else 1
if __name__=='__main__':
 if len(sys.argv)!=4:raise SystemExit('Expected preparedSHA sustainedSHA sourceSHA')
 sys.exit(main(*sys.argv[1:]))
