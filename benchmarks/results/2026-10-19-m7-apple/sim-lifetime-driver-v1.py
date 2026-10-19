import json,os,sys
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_apple_integration as integration
import m7_apple_package as package
folder=ROOT/'artifacts/m7-apple-package-v3'
expected='4d2a5a329e57f67e3f3040dba2dedb3dae61600930c6bf852f07e7110e1a9d56'
simulator='66CC7053-4EFC-4007-8A25-9FF7450CB411'
rows=[];before={};primary=None;cleanup=None;owned=False;pid=None;alive_before=False;absent_after=False;endpoints=[]
previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
try:
 env,_=package.build_environment();env['TMPDIR']=str(folder/'tmp')
 before=integration.admit_build(folder,expected)
 integration.merge(before,integration.admit_simulator(simulator));owned=True
 for path in [Path(__file__).resolve(),ROOT/'tools/experiments/m7_apple_integration.py',ROOT/'apple/tests/simulator_lifetime.c']:
  before[str(path.relative_to(ROOT))]=package.sha(path)
 for name,pin in before.items():assert package.sha(ROOT/name)==pin,name
 sdk=next(s['sdk'] for s in integration.metadata(folder/'build.json',expected)['slices'] if s['target']=='aarch64-apple-ios-sim')
 executable=folder/'simulator-lifetime';marker=folder/'simulator-lifetime.pid'
 command=['xcrun','--sdk','iphonesimulator','clang','-target','arm64-apple-ios18.0-simulator','-isysroot',sdk,'-std=c11','-Wall','-Wextra','-Werror',str(ROOT/'apple/tests/simulator_lifetime.c'),'-o',str(executable)]
 row=package.owned(command,folder/'logs/lifetime-compile.stdout',folder/'logs/lifetime-compile.stderr',env,120);rows.append(row)
 if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(str(row))
 before[str(executable.relative_to(ROOT))]=package.sha(executable)
 row=package.owned(['xcrun','simctl','spawn',simulator,str(executable),str(marker)],folder/'logs/lifetime-spawn.stdout',folder/'logs/lifetime-spawn.stderr',env,2);rows.append(row)
 if not row['primary_error'] or 'TimeoutExpired' not in row['primary_error'] or row['cleanup_errors']:raise RuntimeError('Expected cleanly reaped launcher timeout')
 with marker.open() as f:text=f.read(33)
 if len(text)>32:raise ValueError('Oversize PID marker')
 pid=int(text)
 try:os.kill(pid,0);alive_before=True
 except ProcessLookupError:pass
except BaseException as error:primary=type(error).__name__+': '+str(error)
finally:
 if owned:
  try:cleanup=integration.shutdown_simulator(folder,simulator,env)
  except BaseException as error:cleanup=dict(errors=[str(error)],shutdown_verified=False)
 package.signal.signal(package.signal.SIGTERM,previous)
 if pid:
  try:os.kill(pid,0)
  except ProcessLookupError:absent_after=True
 for name,pin in before.items():
  try:
   if package.sha(ROOT/name)!=pin:endpoints.append(name)
  except OSError:endpoints.append(name)
passed=primary is None and cleanup is not None and cleanup['shutdown_verified'] and not cleanup['errors'] and absent_after and not endpoints
result=dict(passed=passed,primary_error=primary,bindings=before,workers=rows,simulator_cleanup=cleanup,witness_pid=pid,witness_alive_after_host_cleanup=alive_before,witness_absent_after_simulator_shutdown=absent_after,endpoint_errors=endpoints,physical_device_qualified=False)
with (folder/'simulator-lifetime.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({key:result[key] for key in ('passed','primary_error','witness_alive_after_host_cleanup','witness_absent_after_simulator_shutdown','endpoint_errors')}))
sys.exit(not passed)
