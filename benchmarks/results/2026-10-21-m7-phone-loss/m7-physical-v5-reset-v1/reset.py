from pathlib import Path
import json,sys
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_integration as p
pkg=p.package;out=Path(__file__).resolve().parent;logs=out/'private-logs';logs.mkdir();commands=[];before={};result=None;error=None;endpoints=[]
config=p.selection(ROOT/'artifacts/m7-physical-admission-v1/private-selection.json','711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db')
for f in (Path(__file__).resolve(),Path(p.__file__).resolve(),Path(pkg.__file__).resolve()):before[str(f.relative_to(ROOT))]=pkg.sha(f)
env,_=pkg.build_environment();counter=0
old=ROOT/'artifacts/m7-physical-prepared-v11';receipt=p.integration.metadata(old/'integration.json','661010123ba4cc9539dee098aefaf9bb5399c184c57f84b3fa748d4563fd6577')
install=old/'private-logs/install.stdout';pin=receipt['private_evidence_pins'][str(install.relative_to(ROOT))];remote=p.installed_path(p.integration.metadata(install,pin),config);before[str(install.relative_to(ROOT))]=pin
before[str((old/'integration.json').relative_to(ROOT))]='661010123ba4cc9539dee098aefaf9bb5399c184c57f84b3fa748d4563fd6577'
def command(parts,label):
 global counter
 counter+=1
 row=pkg.owned([p.DEVICECTL,'device',*parts,'--device',config['device_identifier'],'--timeout','45','--quiet','--json-output','-'],logs/(str(counter)+'-'+label+'.stdout'),logs/(str(counter)+'-'+label+'.stderr'),env,60);commands.append(row)
 if row['primary_error'] or row['cleanup_errors'] or row['exit_code']!=0:raise RuntimeError('Owned device command failed')
 data=json.loads((ROOT/row['stdout']).read_text())
 identity=data['result'].get('identifier') if label=='device' else data['result'].get('deviceIdentifier')
 if data['info']['outcome']!='success' or identity!=config['device_identifier']:raise ValueError('Command outcome/device differs')
 return data
try:
 p.admit_device(command(['info','details'],'device'),config)
 p.launch_verified(lambda:command(['info','apps','--bundle-id',p.BUNDLE],'installed'),lambda:None,remote,config)
 result=p.stop_remote(lambda:command(['info','processes','--search','TokletHarness'],'processes'),lambda pid,sig:command(['process','signal','--pid',str(pid),'--signal',sig],'signal'),remote)
 if not result['absent'] or result['errors']:raise ValueError('Owned process absence not confirmed')
except BaseException as e:error=type(e).__name__+': '+str(e)
finally:
 for n,h in before.items():
  try:
   if pkg.sha(ROOT/n)!=h:endpoints.append('Changed endpoint: '+n)
  except OSError as e:endpoints.append('Unavailable endpoint: '+n+': '+type(e).__name__)
 public=p.redact_record(dict(passed=error is None and not endpoints,bindings=before,commands=commands,cleanup=result,primary_error=error,endpoint_errors=endpoints,private_evidence_pins={str(f.relative_to(ROOT)):pkg.sha(f) for f in logs.iterdir()}),config,remote)
 with (out/'reset.json').open('x') as f:json.dump(public,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=public['passed'],primary_error=public['primary_error'],absent=result and result['absent'])))
 sys.exit(0 if public['passed'] else 1)
