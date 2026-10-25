"""Data-only acquisition of one immutable Gemma4 candidate.

Reuse the already reviewed bounded range/descriptor/assembly mechanism with an
explicit Gemma identity and 64MiB aggregate metadata allowance. No remote model
code executes; this does not grant numerical, app or device eligibility.
"""
import concurrent.futures as futures
import ctypes,hashlib,json,os,re,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_apple_package as package
OUT=ROOT/'artifacts/m8-gemma4-checkpoint-v1'
REPO='mlx-community/gemma-4-e2b-it-4bit';REV='238767527555cb75a05732a84dff5d6ba0dd6809'
SIZE=3550670554;SHA='038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e'
NAMES=['README.md','chat_template.jinja','config.json','generation_config.json','model.safetensors','model.safetensors.index.json','processor_config.json','tokenizer.json','tokenizer_config.json']
CAP=4*1024**3;METADATA_CAP=64*1024**2
STRIPE=16*1024**2;RESERVE=256*1024**2
STOP=threading.Event();DEADLINE=0.0

def ranges():return [(i,start,min(start+STRIPE,SIZE)-1) for i,start in enumerate(range(0,SIZE,STRIPE))]
def response(raw,status,start=None,end=None):
 if len(raw)>2*1024**2:raise ValueError('Oversize HTTP headers')
 blocks=[b for b in re.split(rb'\r?\n\r?\n',raw) if b.startswith(b'HTTP/')]
 if not blocks:raise ValueError('Missing final HTTP headers')
 lines=blocks[-1].splitlines();expected=b'206' if start is not None else b'200'
 if status.strip()!=expected or not re.fullmatch(rb'HTTP/\S+ '+expected+rb'(?: .*)?',lines[0]):raise ValueError('Final HTTP status differs')
 headers={}
 for line in lines[1:]:
  if not line or line[:1] in [b' ',b'\t'] or b':' not in line:raise ValueError('Invalid final header')
  key,value=line.split(b':',1);headers.setdefault(key.lower(),[]).append(value.strip())
 for key in [b'content-range',b'content-encoding',b'content-length']:
  if len(headers.get(key,[]))>1:raise ValueError('Duplicate range/encoding/length header')
 if headers.get(b'content-encoding',[b'identity'])[0].lower()!=b'identity':raise ValueError('Encoded response rejected')
 if start is not None:
  wanted=f'bytes {start}-{end}/{SIZE}'.encode()
  if headers.get(b'content-range')!=[wanted]:raise ValueError('Final Content-Range differs')
  if b'content-length' in headers and headers[b'content-length']!=[str(end-start+1).encode()]:raise ValueError('Range length differs')
 elif b'content-range' in headers:raise ValueError('Unexpected partial response')

def capture_all(paths):
 buffers={};captures=[];errors=[]
 for key,path in paths:
  try:
   with path.open('rb') as stream:raw=stream.read(2*1024**2+1)
   if len(raw)>2*1024**2:raise ValueError('Oversize command diagnostic')
   buffers[key]=raw;captures.append(dict(key=key,path=str(path.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
  except BaseException as error:errors.append(dict(key=key,path=str(path.relative_to(ROOT)),error=type(error).__name__+': '+str(error)))
 return buffers,captures,errors

def admitted_inventory(path,record):
 with path.open('rb') as stream:raw=stream.read(2*1024**2+1)
 if len(raw)>2*1024**2 or len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
  raise ValueError('Transferred inventory changed before consumption')
 return raw,json.loads(raw)

def check_pins(pins):
 errors=[]
 for name,pin in pins.items():
  try:
   if package.sha(ROOT/name)!=pin:errors.append('Changed source: '+name)
  except OSError as error:errors.append('Unavailable source: '+name+': '+str(error))
 return errors

def space(amount):
 if shutil.disk_usage(OUT).free<amount+RESERVE:raise ValueError('Insufficient output-volume storage')
def owned(command,stdout,stderr,env,pass_fds=()):
 record=dict(command=command,pid=None,exit_code=None,primary_error=None,cleanup_errors=[],stdout=str(stdout.relative_to(ROOT)),stderr=str(stderr.relative_to(ROOT)))
 process=None;limit=min(DEADLINE,time.monotonic()+620)
 try:
  with stdout.open('xb') as out,stderr.open('xb') as err:
   try:
    if STOP.is_set():raise RuntimeError('Acquisition cancelled before command')
    if time.monotonic()>=limit:raise TimeoutError('Owned command deadline expired before start')
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True,pass_fds=pass_fds);record['pid']=process.pid
    while process.poll() is None:
     if STOP.is_set():raise RuntimeError('Acquisition cancelled')
     if time.monotonic()>=limit:raise TimeoutError('Owned command deadline exceeded')
     time.sleep(.1)
    if time.monotonic()>=limit:raise TimeoutError('Owned command completed after deadline')
    record['exit_code']=process.returncode
    if process.returncode:raise RuntimeError('Owned command exit '+str(process.returncode))
   except BaseException as error:record['primary_error']=type(error).__name__+': '+str(error);STOP.set()
   finally:
    if process is not None:
     try:package.drain_group(process,record['cleanup_errors'],15)
     except BaseException as error:record['cleanup_errors'].append('drain: '+str(error))
     try:record['exit_code']=process.wait(timeout=15)
     except BaseException as error:record['cleanup_errors'].append('reap: '+str(error))
 except BaseException as error:
  if record['primary_error'] is None:record['primary_error']=type(error).__name__+': '+str(error)
  STOP.set()
 return record

def transfer(label,url,body,maximum,env,start=None,end=None):
 result=dict(label=label,passed=False,command=None,primary_error=None,sha256=None,bytes=None,body=str(body.relative_to(ROOT)),captures=[],capture_errors=[])
 header=OUT/'logs'/(label+'.headers');stdout=OUT/'logs'/(label+'.stdout');stderr=OUT/'logs'/(label+'.stderr')
 attempted=False;buffers={}
 try:
  space(4*maximum)
  # Inherited exclusive descriptors prevent curl reopening a replaced path.
  with body.open('xb') as sink,header.open('xb') as head:
   command=['curl','--disable','--http1.1','--fail','--location','--max-redirs','8','--silent','--show-error','--max-time','600','--max-filesize',str(max(maximum,4096)),'--proto','=https','--proto-redir','=https','--header','Accept-Encoding: identity','--dump-header',f'/dev/fd/{head.fileno()}','--write-out','%{http_code}','--output',f'/dev/fd/{sink.fileno()}']
   if start is not None:command+=['--range',f'{start}-{end}']
   command.append(url);attempted=True
   result['command']=owned(command,stdout,stderr,env,(head.fileno(),sink.fileno()))
 except BaseException as error:result['primary_error']=type(error).__name__+': '+str(error)
 finally:
  # Closing a descriptor can fail after a child returned. Bind each available
  # stream even then; keep command, close and capture causes in separate fields.
  if attempted:
   buffers,result['captures'],result['capture_errors']=capture_all([('headers',header),('stdout',stdout),('stderr',stderr)])
 try:
  if result['primary_error'] is not None:
   STOP.set();return result
  rec=result['command']
  if rec['primary_error'] or rec['cleanup_errors']:raise RuntimeError('Transfer command failed: '+str(rec['primary_error'])+' '+str(rec['cleanup_errors']))
  if result['capture_errors']:raise ValueError('Command diagnostic capture failed')
  response(buffers['headers'],buffers['stdout'],start,end)
  size=body.stat().st_size
  if size<=0 or size>maximum or (start is not None and size!=end-start+1):raise ValueError('Response body size differs')
  result.update(passed=True,bytes=size,sha256=package.sha(body))
 except BaseException as error:result['primary_error']=type(error).__name__+': '+str(error);STOP.set()
 return result


def assemble(chunks,destination):
 expected=ranges();ordered=sorted(chunks,key=lambda x:x['index'])
 if [x['index'] for x in ordered]!=[i for i,_,_ in expected] or any(x['bytes']!=end-start+1 for x,(_,start,end) in zip(ordered,expected)):
  raise ValueError('Chunk coverage differs')
 whole=hashlib.sha256();total=0
 with destination.open('xb') as out:
  for item in ordered:
   digest=hashlib.sha256();count=0;path=ROOT/item['body']
   with path.open('rb') as source:
    while raw:=source.read(256*1024):
     if STOP.is_set() or time.monotonic()>=DEADLINE:raise RuntimeError('Assembly cancelled/deadline')
     count+=len(raw)
     if count>item['bytes']:raise ValueError('Chunk grew during assembly')
     digest.update(raw);whole.update(raw);out.write(raw)
   if count!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise ValueError('Chunk changed during assembly')
   total+=count
 if total!=SIZE or whole.hexdigest()!=SHA:raise ValueError('Assembled weight identity differs')
 return total

def publish(source,destination):
 if STOP.is_set() or time.monotonic()>=DEADLINE:raise TimeoutError('Cancelled or deadline expired before publication')
 libc=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True);fn=libc.renamex_np
 fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
 if fn(os.fsencode(source),os.fsencode(destination),4):
  error=ctypes.get_errno();raise OSError(error,os.strerror(error),str(destination))

def inventory_rows(inventory):
 if inventory['id']!=REPO or inventory['sha']!=REV:raise ValueError('Immutable inventory differs')
 rows={r['rfilename']:r for r in inventory['siblings']}
 if len(rows)!=len(inventory['siblings']) or not set(NAMES)<=set(rows):raise ValueError('Required inventory differs')
 if any(type(rows[n]['size']) is not int or not 0<rows[n]['size']<=CAP for n in NAMES) or sum(rows[n]['size'] for n in NAMES)>CAP:raise ValueError('Inventory size exceeds cap')
 if sum(rows[n]['size'] for n in NAMES if n!='model.safetensors')>METADATA_CAP:raise ValueError('Metadata exceeds reserved space')
 if rows['model.safetensors']['size']!=SIZE or rows['model.safetensors']['lfs']['sha256']!=SHA:raise ValueError('Weight metadata differs')
 return rows


def main():
 global DEADLINE
 destination=ROOT/'models/MLX-Gemma4-E2B-4bit-2387675';staging=OUT/'staging';parts=OUT/'chunks';logs=OUT/'logs';receipt=OUT/'download.json'
 targets=[destination,staging,parts,logs,receipt,OUT/'inventory.json']
 targets += [OUT/'logs'/(label+'.'+suffix) for label in ['inventory',*[f'file-{i}' for i in range(len(NAMES))],*[f'chunk-{i:04d}' for i,_,_ in ranges()]] for suffix in ['headers','stdout','stderr']]
 targets += [parts/f'{i:04d}.bin' for i,_,_ in ranges()]
 if any(os.path.lexists(path) for path in targets):raise FileExistsError('Existing output refused before acquisition')
 env,_=package.build_environment();pins={str(Path(__file__).resolve().relative_to(ROOT)):package.sha(Path(__file__)),str(Path(package.__file__).resolve().relative_to(ROOT)):package.sha(Path(package.__file__))}
 space(2*SIZE+METADATA_CAP);assert os.stat(destination.parent).st_dev==os.stat(OUT).st_dev
 for folder in [staging,parts,logs]:folder.mkdir(mode=0o700)
 DEADLINE=time.monotonic()+7200;STOP.clear();previous={sig:signal.signal(sig,lambda *_:STOP.set()) for sig in [signal.SIGTERM,signal.SIGINT]};records=[];primary=None;files={};chunks=[]
 try:
  row=transfer('inventory',f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true',OUT/'inventory.json',2*1024**2,env);records.append(row)
  if not row['passed']:raise RuntimeError('Inventory transfer failed')
  raw,inventory=admitted_inventory(OUT/'inventory.json',row);metadata_sha=hashlib.sha256(raw).hexdigest()
  rows=inventory_rows(inventory)
  pending=iter(ranges())
  def submit(pool,spec):
   i,start,end=spec
   return pool.submit(transfer,f'chunk-{i:04d}',f'https://huggingface.co/{REPO}/resolve/{REV}/model.safetensors',parts/f'{i:04d}.bin',end-start+1,env,start,end)
  with futures.ThreadPoolExecutor(max_workers=4) as pool:
   running={}
   for _ in range(4):
    spec=next(pending,None)
    if spec is not None:running[submit(pool,spec)]=spec[0]
   while running:
    done,_=futures.wait(running,timeout=.2,return_when=futures.FIRST_COMPLETED)
    if time.monotonic()>=DEADLINE:STOP.set()
    for task in done:
     index=running.pop(task);row=task.result();row['index']=index;records.append(row)
     if row['passed']:chunks.append(row)
     else:STOP.set()
     if not STOP.is_set():
      spec=next(pending,None)
      if spec is not None:running[submit(pool,spec)]=spec[0]
  if STOP.is_set() or len(chunks)!=len(ranges()):raise RuntimeError('Range acquisition failed or cancelled; all workers joined')
  space(SIZE);assemble(chunks,staging/'model.safetensors')
  for i,name in enumerate(NAMES):
   if name!='model.safetensors':
    row=transfer(f'file-{i}',f'https://huggingface.co/{REPO}/resolve/{REV}/{name}',staging/name,rows[name]['size'],env);records.append(row)
    if not row['passed'] or row['bytes']!=rows[name]['size']:raise ValueError('Metadata transfer failed')
   path=staging/name;size=path.stat().st_size
   if 'lfs' in rows[name]:
    if package.sha(path)!=rows[name]['lfs']['sha256']:raise ValueError('LFS identity differs')
   else:
    digest=hashlib.sha1();digest.update(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
     while data:=stream.read(256*1024):digest.update(data)
    if digest.hexdigest()!=rows[name]['blobId']:raise ValueError('Git identity differs')
   files[name]=dict(bytes=size,sha256=package.sha(path))
  if package.sha(OUT/'inventory.json')!=metadata_sha:raise ValueError('Consumed inventory changed')
  if check_pins(pins):raise ValueError('Bound source changed or disappeared')
  if STOP.is_set():raise RuntimeError('Cancelled before publication')
  publish(staging,destination)
 except BaseException as error:primary=type(error).__name__+': '+str(error);STOP.set()
 finally:
  for sig,handler in previous.items():signal.signal(sig,handler)
  endpoints=check_pins(pins)
  result=dict(format='toklet-gemma4-range-acquisition-v1',passed=primary is None and not endpoints,repository=REPO,revision=REV,destination=str(destination.relative_to(ROOT)),files=files,commands=records,source_pins=pins,primary_error=primary,endpoint_errors=endpoints,inference_executed=False,range_bytes=STRIPE,maximum_workers=4)
  with receipt.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  print(json.dumps(dict(passed=result['passed'],chunks=len(chunks),primary_error=primary,endpoint_errors=endpoints)))
 return 0 if result['passed'] else 1
if __name__=='__main__':sys.exit(main())
