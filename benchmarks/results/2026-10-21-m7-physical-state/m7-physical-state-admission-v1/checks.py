from pathlib import Path
import sys,json
R=Path.cwd();sys.path.insert(0,str(R/'tools/experiments'));import m7_apple_package as p
out=Path(__file__).resolve().parent;record=out/'checks.json'
if record.exists():raise FileExistsError('Exclusive check receipt required')
paths=[R/'tools/experiments'/n for n in ('m7_native_state_check.py','m7_physical_state.py','test_m7_physical_state.py','m7_physical_measurement.py','m7_physical_lifecycle.py','test_m7_physical_lifecycle.py')]+[out/'checks.py',out/'inventory.py',R/'benchmarks/results/2026-10-21-m7-state-app-qualification-v1.json']
pins={str(f.relative_to(R)):p.sha(f) for f in paths};commands=[];primary=None;endpoints=[]
try:
 env,_=p.build_environment();row=p.owned([sys.executable,str(out/'inventory.py')],out/'inventory.stdout',out/'inventory.stderr',env,600);commands.append(row)
 for field in ('stdout','stderr'):pins[row[field]]=p.sha(R/row[field])
 if row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:raise ValueError('Owned app/job/raw-replay admission failed')
 data=json.loads((R/row['stdout']).read_bytes())
 if data['passed'] is not True or set(data['jobs'])!={'small','larger'} or data['device_commands']!=0 or data['native_model_executions']!=0:raise ValueError('Exact offline job scope required')
 for j in data['jobs'].values():
  for field in ('bindings','lifecycle_bindings'):
   for n,h in j[field].items():
    if n in pins and pins[n]!=h:raise ValueError('Conflicting local job pin')
    pins[n]=h
except BaseException as e:primary=type(e).__name__+': '+str(e)
finally:
 for n,h in pins.items():
  try:
   if p.sha(R/n)!=h:endpoints.append('Changed: '+n)
  except OSError:endpoints.append('Unavailable: '+n)
 with record.open('x') as f:json.dump(dict(passed=primary is None and not endpoints,bindings=pins,commands=commands,primary_error=primary,endpoint_errors=endpoints,device_commands=0,native_model_executions=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=primary is None and not endpoints,primary_error=primary,endpoint_errors=endpoints)));sys.exit(0 if primary is None and not endpoints else 1)
