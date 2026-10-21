from pathlib import Path
import json,sys,hashlib
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_apple_package as p
out=Path(__file__).resolve().parent;record=out/'checks.json'
if record.exists():raise FileExistsError('Exclusive checks receipt required')
paths=[*sorted((ROOT/'tools/experiments').glob('m7_*.py')),*sorted((ROOT/'tools/experiments').glob('test_m7_*.py')),Path(__file__).resolve()]
pins={str(f.relative_to(ROOT)):p.sha(f) for f in paths};commands=[];primary=None;endpoints=[]
try:
 for f in sorted((ROOT/'tools/experiments').glob('test_m7_*.py')):
  label=f.stem;row=p.owned(['/Users/coloner/work/rust/toklet/.venv/bin/python','-m','unittest','discover','-s','tools/experiments','-p',f.name,'-v'],out/(label+'.stdout'),out/(label+'.stderr'),{},180);commands.append(row)
  if row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:raise RuntimeError('Failed module: '+label)
except BaseException as e:primary=type(e).__name__+': '+str(e)
finally:
 for n,h in pins.items():
  try:
   if p.sha(ROOT/n)!=h:endpoints.append('Changed: '+n)
  except OSError:endpoints.append('Unavailable: '+n)
 with record.open('x') as stream:json.dump(dict(passed=primary is None and not endpoints,bindings=pins,commands=commands,primary_error=primary,endpoint_errors=endpoints),stream,indent=2);stream.write('\n')
 print(json.dumps(dict(passed=primary is None and not endpoints,commands=len(commands),primary_error=primary,endpoint_errors=endpoints)))
 sys.exit(0 if primary is None and not endpoints else 1)
