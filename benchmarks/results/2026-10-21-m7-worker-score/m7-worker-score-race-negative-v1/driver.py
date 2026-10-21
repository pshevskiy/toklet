from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/experiments'));import m7_apple_package as p
OUT=Path(__file__).resolve().parent;(OUT/'logs').mkdir();(OUT/'tmp').mkdir()
rpath=ROOT/'artifacts/m7-native-worker-score-v2/checks.json'
with rpath.open('rb') as stream:raw=stream.read(16*1024**2+1)
assert len(raw)<=16*1024**2;r=json.loads(raw);assert r['passed']
pins={**r['bindings'],str(rpath.relative_to(ROOT)):p.hashlib.sha256(raw).hexdigest(),str(Path(__file__).resolve().relative_to(ROOT)):p.sha(Path(__file__).resolve())}
original=ROOT/'apple/Harness/NativeWorker.swift';text=original.read_text();old='failure == nil && !closing && queued == nil ? engine.score : nil';assert text.count(old)==1
mutant=OUT/'NativeWorker.swift';mutant.write_text(text.replace(old,'failure == nil ? engine.score : nil'));pins[str(mutant.relative_to(ROOT))]=p.sha(mutant)
def guard():
 for n,h in pins.items():assert p.sha(ROOT/n)==h,n
commands=[];primary=None;endpoints=[]
previous=p.signal.signal(p.signal.SIGTERM,p.interrupted)
try:
 env=dict(r['controlled_environment']);env['TMPDIR']=str(OUT/'tmp');command=list(r['commands'][3]['command']);assert str(original) in command
 command[command.index(str(original))]=str(mutant);command[command.index('-module-cache-path')+1]=str(OUT/'ModuleCache');binary=OUT/'score-mutant';command[-1]=str(binary)
 guard();row=p.owned(command,OUT/'logs/compile.stdout',OUT/'logs/compile.stderr',env,180);commands.append(row)
 assert row['exit_code']==0 and row['primary_error'] is None and not row['cleanup_errors'];pins[str(binary.relative_to(ROOT))]=p.sha(binary)
 guard();row=p.owned([str(binary),str(ROOT/'artifacts/m7-native-tiny-v2')],OUT/'logs/cpu.stdout',OUT/'logs/cpu.stderr',env,180);commands.append(row)
 assert row['exit_code']==1 and row['primary_error']=='tool exit 1' and not row['cleanup_errors']
 assert 'TokletScoreChecks' in p.small(OUT/'logs/cpu.stderr')
 assert not p.small(OUT/'logs/cpu.stdout').strip()
except BaseException as error:primary=type(error).__name__+': '+str(error)
finally:
 p.signal.signal(p.signal.SIGTERM,previous)
 try:guard()
 except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
 for row in commands:
  for field in ('stdout','stderr'):pins[row[field]]=p.sha(ROOT/row[field])
 passed=primary is None and not endpoints and len(commands)==2
 with (OUT/'negative.json').open('x') as stream:json.dump(dict(format='toklet-m7-worker-score-negative-v1',passed=passed,mutation='Remove closing/queued suppression only',bindings=pins,commands=commands,primary_error=primary,endpoint_errors=endpoints,expected_child_exit=1,physical_device_qualified=False,m7_complete=False),stream,indent=2);stream.write('\n')
 print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)))
sys.exit(0 if passed else 1)
