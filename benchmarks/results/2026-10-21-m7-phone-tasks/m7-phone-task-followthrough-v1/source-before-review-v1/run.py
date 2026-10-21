"""Continue the active collection into reviewed saved-data analysis and quality."""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools/experiments'));import m7_apple_package as p
OUT=Path(__file__).resolve().parent
DRIVER=ROOT/'artifacts/m7-phone-tasks-driver-v1'
def exclusive(path):
 if path.exists() or path.is_symlink():raise FileExistsError('Exclusive follow-through output required')
def bounded(path,limit):
 with path.open('rb') as f:raw=f.read(limit+1)
 if len(raw)>limit:raise ValueError('Bounded follow-through input exceeded')
 return raw
def completed(record,footer):
 if record['passed'] is not True or record['primary_error'] or record['endpoint_errors'] or len(record['results'])!=14 or footer!=dict(passed=True,stages=14,primary_error=None,endpoint_errors=[]):raise ValueError('Successful complete task driver required')
 return True
def guard(pins):
 for n,h in pins.items():
  if p.sha(ROOT/n)!=h:raise ValueError('Changed follow-through endpoint: '+n)
def main():
 output=OUT/'execution.json';exclusive(output)
 analysis=ROOT/'artifacts/m7-phone-tasks-analysis-v1';quality=ROOT/'artifacts/m7-phone-task-quality-v1'
 for path in (analysis/'analysis.json',quality/'quality.json',OUT/'analysis.stdout',OUT/'analysis.stderr',OUT/'quality.stdout',OUT/'quality.stderr'):exclusive(path)
 paths=[Path(__file__).resolve(),OUT/'checks.py',analysis/'analyze.py',analysis/'checks.py',quality/'quality.py',quality/'checks.py',ROOT/'artifacts/m7-phone-tasks-source-v1/source.json']
 pins={str(path.relative_to(ROOT)):p.sha(path) for path in paths};source_raw=bounded(paths[-1],1024*1024)
 if hashlib.sha256(source_raw).hexdigest()!=pins[str(paths[-1].relative_to(ROOT))]:raise ValueError('Frozen source receipt consumed bytes differ')
 source=json.loads(source_raw)
 for n,v in source['sources'].items():
  if n in pins and pins[n]!=v['sha256']:raise ValueError('Conflicting frozen source')
  pins[n]=v['sha256']
 commands=[];primary=None;endpoints=[];driver_digest=None;analysis_digest=None
 previous=p.signal.signal(p.signal.SIGTERM,p.interrupted)
 try:
  guard(pins);deadline=time.monotonic()+14400
  while True:
   if time.monotonic()>=deadline:raise TimeoutError('Original task collection deadline')
   receipt=DRIVER/'execution.json';stdout=DRIVER/'driver.stdout'
   if receipt.exists() and stdout.exists():
    raw=bounded(receipt,1024*1024);stdout_raw=bounded(stdout,1024*1024);lines=stdout_raw.decode().splitlines()
    try:record=json.loads(raw);footer=json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:record=None;footer=None
    if record is not None and type(footer) is dict and set(footer)=={'passed','stages','primary_error','endpoint_errors'}:
     completed(record,footer);driver_digest=hashlib.sha256(raw).hexdigest();pins[str(receipt.relative_to(ROOT))]=driver_digest;pins[str(stdout.relative_to(ROOT))]=hashlib.sha256(stdout_raw).hexdigest();break
   time.sleep(1)
  guard(pins);env,_=p.build_environment()
  for args,label in [([sys.executable,str(analysis/'analyze.py')],'analysis'),([sys.executable,str(quality/'quality.py')],'quality')]:
   if label=='quality':
    raw=bounded(analysis/'analysis.json',64*1024**2);analysis_digest=hashlib.sha256(raw).hexdigest();record=json.loads(raw)
    if record['passed'] is not True or record['cases']!=206 or record['task_quality_qualified'] is not False:raise ValueError('Complete response replay required')
    pins[str((analysis/'analysis.json').relative_to(ROOT))]=analysis_digest;args.append(analysis_digest)
   guard(pins);row=p.owned(args,OUT/(label+'.stdout'),OUT/(label+'.stderr'),env,1800);commands.append(row)
   for f in ('stdout','stderr'):pins[row[f]]=p.sha(ROOT/row[f])
   if row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:raise ValueError('Owned follow-through stage failed: '+label)
   guard(pins)
  quality_raw=bounded(quality/'quality.json',64*1024**2);result=json.loads(quality_raw)
  if result['passed'] is not True or result['quality_completed'] is not True or len(result['models'])!=2 or result['new_vm_runs']!=0 or result['new_generated_program_executions']!=0:raise ValueError('Exact reviewed outcome-equivalence result required')
  pins[str((quality/'quality.json').relative_to(ROOT))]=hashlib.sha256(quality_raw).hexdigest();guard(pins)
 except BaseException as error:primary=type(error).__name__+': '+str(error)
 finally:
  p.signal.signal(p.signal.SIGTERM,previous)
  try:guard(pins)
  except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
  passed=primary is None and not endpoints and len(commands)==2
  with output.open('x') as f:json.dump(dict(passed=passed,bindings=pins,commands=commands,driver_sha256=driver_digest,analysis_sha256=analysis_digest,primary_error=primary,endpoint_errors=endpoints,physical_device_qualified=False,m7_complete=False,scope='Owned continuation of current collection into reviewed exact saved-data/quality checks'),f,indent=2);f.write('\n')
  print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)));return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
