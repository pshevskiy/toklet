import importlib.util,tempfile,unittest,hashlib,time,sys,threading,json
from pathlib import Path
from unittest.mock import patch
import m8_qwen3_checkpoint as m
class Checks(unittest.TestCase):
 def test_close_failure_preserves_owned_cause_streams_and_prevents_publication(self):
  import os,io,contextlib
  original_open=Path.open
  class CloseFailure:
   def __init__(self,stream):self.stream=stream
   def __enter__(self):return self.stream
   def __exit__(self,*_):self.stream.close();raise OSError('synthetic body close failure')
  with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as tmp:
   root=Path(tmp);(root/'logs').mkdir();body=root/'body'
   def opening(path,*args,**kwargs):
    stream=original_open(path,*args,**kwargs)
    return CloseFailure(stream) if path==body and args==('xb',) else stream
   def child(command,stdout,stderr,env,pass_fds):
    os.write(pass_fds[0],b'HTTP/2 200\r\nContent-Length: 2\r\n\r\n');os.write(pass_fds[1],b'ab')
    stdout.write_bytes(b'200');stderr.write_bytes(b'original child failure')
    return dict(exit_code=42,primary_error='tool exit 42',cleanup_errors=[],pid=None)
   m.STOP.clear()
   with patch.object(m,'OUT',root),patch.object(m,'owned',side_effect=child),patch.object(Path,'open',opening),patch.object(m,'space'):
    result=m.transfer('close','https://huggingface.co/example',body,2,{})
   self.assertFalse(result['passed']);self.assertIn('synthetic body close failure',result['primary_error'])
   self.assertEqual(result['command']['primary_error'],'tool exit 42');self.assertEqual(result['capture_errors'],[])
   captured={row['key']:row for row in result['captures']}
   self.assertEqual(set(captured),{'headers','stdout','stderr'})
   self.assertEqual(captured['stderr']['sha256'],hashlib.sha256(b'original child failure').hexdigest())
   self.assertEqual(captured['stdout']['sha256'],hashlib.sha256(b'200').hexdigest());self.assertTrue(m.STOP.is_set())
   output=root/'failed-main';output.mkdir()
   with patch.object(m,'OUT',output),patch.object(m,'transfer',return_value=result),patch.object(m,'space'),patch.object(m,'publish') as publication,contextlib.redirect_stdout(io.StringIO()):
    self.assertEqual(m.main(),1);publication.assert_not_called()
   m.STOP.clear()

 def test_immutable_inventory_rejects_identity_size_duplicates_and_missing_files(self):
  rows=[dict(rfilename=n,size=1,blobId='f'*40) for n in m.NAMES]
  weight=next(r for r in rows if r['rfilename']=='model.safetensors')
  weight.update(size=m.SIZE,lfs=dict(sha256=m.SHA))
  original=dict(id=m.REPO,sha=m.REV,siblings=rows)
  self.assertEqual(set(m.inventory_rows(original)),set(m.NAMES))
  import copy
  for mode in ['repo','revision','missing','duplicate','negative','bool','float','total','weight','hash','metadata']:
   value=copy.deepcopy(original)
   if mode=='repo':value['id']='other'
   elif mode=='revision':value['sha']='other'
   elif mode=='missing':value['siblings'].pop()
   elif mode=='duplicate':value['siblings'].append(value['siblings'][0])
   elif mode=='negative':value['siblings'][0]['size']=-1
   elif mode=='bool':value['siblings'][0]['size']=True
   elif mode=='float':value['siblings'][0]['size']=1.5
   elif mode=='total':value['siblings'][0]['size']=m.CAP
   elif mode=='weight':next(r for r in value['siblings'] if r['rfilename']=='model.safetensors')['size']-=1
   elif mode=='hash':next(r for r in value['siblings'] if r['rfilename']=='model.safetensors')['lfs']['sha256']='0'*64
   elif mode=='metadata':value['siblings'][0]['size']=17*1024**2
   with self.subTest(mode=mode),self.assertRaises(ValueError):m.inventory_rows(value)

 def test_ranges_cover_exact_tail(self):
  with patch.object(m,'SIZE',5),patch.object(m,'STRIPE',2):self.assertEqual(m.ranges(),[(0,0,1),(1,2,3),(2,4,4)])
  rows=m.ranges();self.assertEqual(rows[0][1],0);self.assertEqual(rows[-1][2],m.SIZE-1)
  self.assertTrue(all(b[1]==a[2]+1 for a,b in zip(rows,rows[1:])))
 def test_final_response_identity_and_duplicate_rejection(self):
  with patch.object(m,'SIZE',5):
   prefix=b'HTTP/1.1 302 Found\r\nContent-Range: bytes 0-1/5\r\nLocation: https://hf.co/x\r\n\r\n'
   good=b'HTTP/2 206\r\nContent-Range: bytes 0-1/5\r\nContent-Length: 2\r\n\r\n'
   m.response(prefix+good,b'206',0,1)
   for raw,status in [(prefix+b'HTTP/2 200\r\n\r\n',b'200'),(good.replace(b'/5',b'/6'),b'206'),(good.replace(b'Content-Length: 2',b'Content-Length: 3'),b'206'),(good.replace(b'\r\n\r\n',b'\r\nContent-Range: bytes 0-1/5\r\n\r\n'),b'206'),(good.replace(b'\r\n\r\n',b'\r\nContent-Encoding: gzip\r\n\r\n'),b'206'),(good.replace(b'\r\n\r\n',b'\r\nContent-Encoding: identity\r\ncontent-encoding: identity\r\n\r\n'),b'206')]:
    with self.assertRaises(ValueError):m.response(raw,status,0,1)
 def test_inventory_buffer_is_admitted_before_parse(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as tmp:
   path=Path(tmp)/'inventory';raw=b'{"sha":"original"}';path.write_bytes(raw)
   rec=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
   self.assertEqual(m.admitted_inventory(path,rec),(raw,json.loads(raw)))
   path.write_bytes(b'{"sha":"replaced"}')
   with self.assertRaises(ValueError):m.admitted_inventory(path,rec)
 def test_missing_stdout_preserves_stderr_and_endpoint_failure(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as tmp:
   root=Path(tmp);err=root/'stderr';err.write_bytes(b'original diagnostic')
   buffers,captures,errors=m.capture_all([('stdout',root/'missing'),('stderr',err)])
   self.assertEqual(buffers['stderr'],b'original diagnostic');self.assertEqual(len(errors),1)
   self.assertEqual(captures[0]['sha256'],hashlib.sha256(buffers['stderr']).hexdigest())
   self.assertEqual(len(m.check_pins({str((root/'gone').relative_to(m.ROOT)):'a'*64})),1)
 def test_assembly_rechecks_coverage_chunk_and_whole_hash(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as tmp,patch.object(m,'SIZE',5),patch.object(m,'STRIPE',2),patch.object(m,'SHA',hashlib.sha256(b'abcde').hexdigest()),patch.object(m,'DEADLINE',time.monotonic()+60):
   m.STOP.clear();root=Path(tmp);items=[]
   for i,data in enumerate([b'ab',b'cd',b'e']):
    p=root/str(i);p.write_bytes(data);items.append(dict(index=i,body=str(p.relative_to(m.ROOT)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
   self.assertEqual(m.assemble(items,root/'good'),5);self.assertEqual((root/'good').read_bytes(),b'abcde')
   with self.assertRaises(ValueError):m.assemble(items[:-1],root/'missing')
   (root/'1').write_bytes(b'xx')
   with self.assertRaises(ValueError):m.assemble(items,root/'changed')
 def test_preflight_never_overwrites_metadata(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as tmp:
   root=Path(tmp);out=root/'output';out.mkdir();(out/'inventory.json').write_bytes(b'keep')
   with patch.object(m,'OUT',out),patch.object(m,'ROOT',root),patch.object(m.package,'build_environment',side_effect=AssertionError('Must not start')):
    with self.assertRaises(FileExistsError):m.main()
   self.assertEqual((out/'inventory.json').read_bytes(),b'keep');self.assertFalse((out/'staging').exists())
 def test_actual_owned_process_and_child_cancelled_and_reaped(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as tmp,patch.object(m,'DEADLINE',time.monotonic()+60):
   root=Path(tmp);stdout=root/'stdout';stderr=root/'stderr';results=[];m.STOP.clear()
   code='import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); print(p.pid,flush=True);time.sleep(60)'
   thread=threading.Thread(target=lambda:results.append(m.owned([sys.executable,'-c',code],stdout,stderr,dict(PATH='/usr/bin:/bin'))));thread.start()
   deadline=time.monotonic()+5
   while time.monotonic()<deadline and (not stdout.exists() or not stdout.read_bytes().strip()):time.sleep(.02)
   m.STOP.set();thread.join(50)
   self.assertFalse(thread.is_alive());self.assertEqual(len(results),1)
   rec=results[0];self.assertIn('cancelled',rec['primary_error']);self.assertEqual(rec['cleanup_errors'],[])
   self.assertFalse(m.package.group_exists(rec['pid']));m.STOP.clear()
if __name__=='__main__':unittest.main()
