from pathlib import Path
import json,sys
R=Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(R/'tools/experiments'));import m7_physical_integration as m;import m7_apple_package as p;import m7_apple_integration as i
out=R/'artifacts/m7-physical-preflight-stop-v6';logs=out/'private-logs';logs.mkdir();env,_=p.build_environment();env['TMPDIR']=str(out)
config=m.selection(R/'artifacts/m7-physical-admission-v1/private-selection.json','711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db')
origin=R/'artifacts/m7-physical-prepared-v9';receipt=i.metadata(origin/'integration.json','c7f5be741d76259bce3ac03820a4bd0ecda915c6761c1263947b2a465c41fbf7');assert receipt['passed'] and receipt['signing']
app=origin/'derived/Build/Products/Release-iphoneos/TokletHarness.app';assert m.inventory(app)==receipt['signed_bundle']
install=origin/'private-logs/install.stdout';remote=m.installed_path(i.metadata(install,receipt['private_evidence_pins'][str(install.relative_to(R))]),config)
pins={str((origin/'integration.json').relative_to(R)):p.sha(origin/'integration.json'),str(install.relative_to(R)):p.sha(install)}
pins.update({str((app/n).relative_to(R)):h for n,h in receipt['signed_bundle'].items()});commands=[];counter=0
pins[str(Path(__file__).relative_to(R))]=p.sha(Path(__file__))
for module in [m,p,i]:pins[str(Path(module.__file__).resolve().relative_to(R))]=p.sha(Path(module.__file__).resolve())
def execute(args,label):
 for name,h in pins.items():assert p.sha(R/name)==h
 row=p.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,60);commands.append(row)
 if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(label+' failed')
 return json.loads(p.small(R/row['stdout'])) if label!='signature' else None
def device(parts,label):return execute([m.DEVICECTL,'device',*parts,'--device',config['device_identifier'],'--timeout','45','--quiet','--json-output','-'],label)
def query():
 global counter
 counter+=1;return device(['info','processes','--search','TokletHarness'],'processes-'+str(counter))
def signal(pid,sig):
 global counter
 counter+=1;device(['process','signal','--pid',str(pid),'--signal',sig],'signal-'+str(counter))
execute(['codesign','--verify','--strict',str(app)],'signature')
m.admit_device(device(['info','details'],'device'),config)
m.launch_verified(lambda:device(['info','apps','--bundle-id',m.BUNDLE],'current-install'),lambda:True,remote,config)
result=m.stop_remote(query,signal,remote)
for name,h in pins.items():assert p.sha(R/name)==h
raw=dict(commands=commands,bindings=pins,result=result,reason='Close only exact historical verified signed install before replacing app; not inference reuse');(out/'private-execution.json').write_text(json.dumps(raw,indent=2)+'\n')
public=m.redact_record(raw,config,remote);public['private_pins']={str(f.relative_to(R)):p.sha(f) for f in [out/'private-execution.json',*sorted(logs.iterdir())]};(out/'stop.json').write_text(json.dumps(public,indent=2)+'\n');print(json.dumps(dict(absent=result['absent'],errors=result['errors'],signals=len(result['signals']))));sys.exit(not(result['absent'] and not result['errors']))
