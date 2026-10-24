"""Own bounded packed-table witnesses and preserve every numerical rejection."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

import m7_apple_package as owner

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/reference'))
import gemma_packed as operator
import reference_runtime as runtime


def pin(path):
 if path.is_symlink() or not path.is_file():raise ValueError('Regular reference file required: '+str(path))
 digest=hashlib.sha256();count=0
 with path.open('rb') as stream:
  while chunk:=stream.read(128*1024):
   count+=len(chunk)
   if count>512*1024**2:raise ValueError('Oversize reference input')
   digest.update(chunk)
 return dict(bytes=count,sha256=digest.hexdigest())


def collect(destination):
 if (not destination.is_absolute() or destination.is_symlink() or not destination.parent.is_dir()
     or not destination.resolve().is_relative_to(ROOT/'artifacts')):raise ValueError('New absolute output below artifacts required')
 destination.mkdir();inputs={};outputs={};commands=[];primary=None;endpoints=[];capture_errors=[];observations={}
 previous={};runtime_files=None;selected=None;math_sha=None;numerical_files=None
 def interrupted(number,_):
  for value in (signal.SIGTERM,signal.SIGINT):signal.signal(value,signal.SIG_IGN)
  raise InterruptedError('Received signal '+str(number))
 def guard():
  if selected is not None and runtime.runtime_paths(selected)!=runtime_files:raise ValueError('Runtime inventory changed')
  if numerical_files is not None and owner.numerical_sources()!=numerical_files:raise ValueError('Engine source inventory changed')
  for name,value in inputs.items():
   if pin(Path(name))!=value:raise ValueError('Reference input changed: '+name)
  for name,value in outputs.items():
   if pin(destination/name)!=value:raise ValueError('Captured output changed: '+name)
 def retain(path,maximum):
  if path.is_symlink() or not path.is_file():raise ValueError('Regular output required')
  with path.open('rb') as stream:raw=stream.read(maximum+1)
  if len(raw)>maximum:raise ValueError('Oversize captured output')
  name=str(path.relative_to(destination));value=dict(bytes=len(raw),sha256=operator.sha(raw))
  if name in outputs and outputs[name]!=value:raise ValueError('Prior captured buffer changed')
  outputs[name]=value;return raw
 def invoke(label,command,expected=()):
  guard();cache=destination/(label+'-pycache');cache.mkdir()
  row=dict(label=label,command=command,owned_result=None,invocation_error=None);commands.append(row)
  try:row['owned_result']=owner.owned(command,destination/'logs'/(label+'.stdout'),destination/'logs'/(label+'.stderr'),dict(env,PYTHONPYCACHEPREFIX=str(cache)),1200)
  except (Exception,KeyboardInterrupt) as error:row['invocation_error']=type(error).__name__+': '+str(error)
  finally:
   for path,limit in [(destination/'logs'/(label+'.'+stream),8*1024**2) for stream in ('stdout','stderr')]+list(expected):
    try:retain(path,limit)
    except (Exception,KeyboardInterrupt) as error:capture_errors.append(label+': '+str(error))
  guard()
  result=row['owned_result']
  if row['invocation_error'] or result is None or result['exit_code']!=0 or result['primary_error'] or result['cleanup_errors'] or capture_errors:raise RuntimeError('Owned reference command failed: '+label)
 try:
  previous={value:signal.signal(value,interrupted) for value in (signal.SIGTERM,signal.SIGINT)}
  selected=[ROOT/'tools/reference/gemma_packed.py',ROOT/'tools/reference/test_gemma_packed.py',ROOT/'tools/reference/test_reference_runtime.py',Path(__file__).resolve(),ROOT/'tools/experiments/test_gemma_packed_reference.py',Path(owner.__file__).resolve()]
  selected=[x.resolve() for x in selected];runtime_files=runtime.runtime_paths(selected)
  numerical_files=owner.numerical_sources()
  for path in sorted(set([*runtime_files,*numerical_files,ROOT/'target/release/toklet',ROOT/'target/release/examples/cpu_bench'])):inputs[str(path)]=pin(path)
  snapshot=destination/'source-snapshot';snapshot.mkdir()
  for path in selected:
   target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
   raw=runtime.captured(path,512*1024,inputs[str(path)]['sha256'])
   with target.open('xb') as stream:stream.write(raw)
   admitted=dict(bytes=len(raw),sha256=operator.sha(raw))
   if pin(target)!=admitted or pin(path)!=admitted or inputs[str(path)]!=admitted:raise ValueError('Admitted source buffer changed before copying')
   inputs[str(target)]=admitted
  for name in ('logs','tmp'):(destination/name).mkdir()
  env=dict(HOME=str(Path.home()),PATH='/usr/bin:/bin:/usr/sbin:/sbin',LANG='en_US.UTF-8',TMPDIR=str(destination/'tmp'),PYTHONNOUSERSITE='1')
  for label,folder,pattern in (('owner','tools/experiments','test_gemma_packed_reference.py'),('portable','tools/reference','test_gemma_packed.py'),('admission','tools/reference','test_reference_runtime.py')):
   invoke(label,[sys.executable,'-s','-m','unittest','discover','-s',str(ROOT/folder),'-p',pattern,'-v'])
  script=ROOT/'tools/reference/gemma_packed.py';manifest=destination/'runtime.json'
  files=runtime.runtime_paths([script]);content=operator.encoded(dict(format='toklet-mlx-reference-runtime-v1',versions=runtime.VERSIONS,files={str(x):inputs[str(x)] for x in files}))
  with manifest.open('xb') as stream:stream.write(content)
  inputs[str(manifest)]=pin(manifest)
  math_path=destination/'math.json'
  invoke('export',[sys.executable,'-s',str(script),'export','--output',str(math_path)],[(math_path,operator.MAX_JSON),(destination/'math.bin',operator.MAX_RAW)])
  math_sha=outputs['math.json']['sha256'];operator.verify(math_path,math_sha)
  for path in (math_path,destination/'math.bin'):inputs[str(path)]=pin(path)
  for device in ('cpu','gpu'):
   output=destination/device;output.mkdir()
   invoke(device,[sys.executable,'-s',str(script),'compare','--output',str(output),'--device',device,'--reference',str(math_path),'--expected-reference-sha256',math_sha,'--runtime-manifest',str(manifest),'--expected-runtime-sha256',inputs[str(manifest)]['sha256']],[(output/'observation.json',operator.MAX_JSON),(output/'observed.bin',operator.MAX_RAW)])
   report=json.loads(retain(output/'observation.json',operator.MAX_JSON),object_pairs_hook=runtime.json_unique,parse_constant=runtime.bad_json_number)
   if (report['format']!='toklet-gemma-packed-observation-v1' or report['capture_complete'] is not True or report['requested_device']!=device
       or report['mathematical_reference_sha256']!=math_sha or report['admitted_runtime']['manifest_sha256']!=inputs[str(manifest)]['sha256']
       or report['binary']!=dict(path='observed.bin',**outputs[device+'/observed.bin']) or report['immutable_parameters'] is not True
       or any(report[x] is not False for x in ('native_composition_implemented','bf16_checkpoint_qualified','device_qualified'))):raise ValueError('Observation identity/scope differs')
   observations[device]=report
 except (Exception,KeyboardInterrupt) as error:primary=type(error).__name__+': '+str(error)
 finally:
  for value in previous:signal.signal(value,signal.SIG_IGN)
  try:guard()
  except (Exception,KeyboardInterrupt) as error:endpoints.append(type(error).__name__+': '+str(error))
 complete=primary is None and not endpoints and not capture_errors and len(commands)==6 and set(observations)=={'cpu','gpu'}
 report=dict(format='toklet-gemma-packed-owned-reference-v1',capture_complete=complete,provisional=True,fixture_accepted=False,
  mathematical_reference_sha256=math_sha,inputs=inputs,outputs=outputs,commands=commands,observations=observations,
  primary_error=primary,endpoint_errors=endpoints,capture_errors=capture_errors,numerical_rejections_preserved=True,
  scope='Packed BF16 coefficient/FP32 and BF16 module witnesses; numerical comparisons separate from capture completion',
  bf16_checkpoint_qualified=False,native_composition_implemented=False,device_qualified=False)
 try:
  with (destination/'capture.json').open('xb') as stream:stream.write(operator.encoded(report))
 finally:
  for value,handler in previous.items():signal.signal(value,handler)
 print(json.dumps(dict(capture_complete=complete,output=str(destination),primary_error=primary)))
 return 0 if complete else 1


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
 raise SystemExit(collect(parser.parse_args().output))
