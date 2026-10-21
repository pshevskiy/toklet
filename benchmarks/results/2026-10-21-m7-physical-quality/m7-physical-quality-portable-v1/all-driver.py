from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/experiments'));import m7_apple_package as p
OUT=Path(__file__).resolve().parent
paths=[*sorted((ROOT/'tools/experiments').glob('m7_*.py')),*sorted((ROOT/'tools/experiments').glob('test_m7_*.py')),Path(__file__).resolve()]
pins={str(path.relative_to(ROOT)):p.sha(path) for path in paths}
def guard():
 for n,h in pins.items():assert p.sha(ROOT/n)==h,n
guard();env,_=p.build_environment();env['TMPDIR']=str(OUT)
r=p.owned([sys.executable,'-m','unittest','discover','-s','tools/experiments','-p','test_m7_*.py'],OUT/'all.stdout',OUT/'all.stderr',env,180)
endpoints=[]
try:guard()
except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
passed=r['exit_code']==0 and r['primary_error'] is None and not r['cleanup_errors'] and not endpoints
with (OUT/'all.json').open('x') as stream:json.dump(dict(passed=passed,bindings=pins,command=r,endpoint_errors=endpoints),stream,indent=2);stream.write('\n')
sys.exit(0 if passed else 1)
