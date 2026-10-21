from pathlib import Path
import hashlib,json,sys
ROOT=Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_measurement as m
own=Path(__file__).resolve();execution=own.parent/'execution.json'
if execution.exists() or execution.is_symlink():raise FileExistsError('Exclusive driver execution required')
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
pins={str(p.relative_to(ROOT)):sha(p) for p in [own,*[Path(v.__file__).resolve() for v in [m,m.package,m.integration,m.physical,m.storage]]]}
def guard():
 for name,digest in pins.items():assert sha(ROOT/name)==digest,name
args=(ROOT/'artifacts/m7-apple-package-v4',ROOT/'artifacts/m7-physical-admission-v1/private-selection.json',
      '711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db',ROOT/'artifacts/m7-physical-prepared-v11',
      '661010123ba4cc9539dee098aefaf9bb5399c184c57f84b3fa748d4563fd6577')
results=[];ready_receipts={};code=0;error=None;endpoints=[]
def call(stage,model,workload,output,*job,**options):
 guard()
 returned=m.main(*args,*job,output,**options)
 results.append(dict(stage=stage,model=model,workload=workload,path=str(output.relative_to(ROOT)),exit_code=returned))
 guard()
 if returned:raise RuntimeError(stage+' failed')
try:
 for model in ('small','larger'):
  for workload in ('128','512','1024'):
   stem='m7-phone-context-'+model+'-'+workload
   proof=ROOT/('artifacts/'+stem+'-storage-proof-v1');ready=ROOT/('artifacts/'+stem+'-ready-v1')
   call('storage-proof',model,workload,proof,'tiny-cpu','short')
   proof_name=str((proof/'measurement.json').relative_to(ROOT));proof_sha=sha(ROOT/proof_name);pins[proof_name]=proof_sha
   call('prepare',model,workload,ready,model,workload,prepare_only=True,storage_proof=proof,storage_sha=proof_sha)
   ready_name=str((ready/'measurement.json').relative_to(ROOT));ready_sha=sha(ROOT/ready_name)
   ready_receipts[ready_name]=ready_sha;pins[ready_name]=ready_sha
   for version in (1,2):
    output=ROOT/('artifacts/'+stem+'-v'+str(version))
    call('measure',model,workload,output,model,workload,transfer=ready,transfer_sha=ready_sha)
except BaseException as failure:
 error=type(failure).__name__+': '+str(failure);code=1
finally:
 try:guard()
 except BaseException as failure:
  endpoints.append(type(failure).__name__+': '+str(failure));code=1
 with execution.open('x') as stream:
  json.dump(dict(driver_bindings=pins,ready_receipts=ready_receipts,results=results,exit_code=code,primary_error=error,
                 endpoint_errors=endpoints,scope='Exploratory raw-ID context matrix; no quality or device qualification inferred'),stream,indent=2)
  stream.write('\n')
sys.exit(code)
