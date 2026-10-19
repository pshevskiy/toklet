import json,sys
from pathlib import Path
sys.path.insert(0,'tools/experiments');import m7_apple_package as p
root=Path.cwd();folder=root/'artifacts/m7-swift-worker-native-v1';folder.mkdir();env,_=p.build_environment();env['TMPDIR']=str(folder)
paths=['apple/Harness/NativeWorker.swift','apple/tests/swift/WorkerChecks.swift','apple/tests/swift/main.swift','artifacts/m7-swift-worker-check-v4','artifacts/m7-history-target-v1/release/libtoklet.a','artifacts/m7-native-tiny-v2/config.json','artifacts/m7-native-tiny-v2/model.safetensors',str(Path(__file__).resolve().relative_to(root))]
before={name:p.sha(root/name) for name in paths};rows=[];results=[];primary=None;endpoints=[]
previous=p.signal.signal(p.signal.SIGTERM,p.interrupted)
try:
 for backend,extra in [('cpu',[]),('metal',['metal'])]:
  command=[str(root/'artifacts/m7-swift-worker-check-v4'),str(root/'artifacts/m7-native-tiny-v2'),*extra]
  row=p.owned(command,folder/(backend+'.stdout'),folder/(backend+'.stderr'),env,180);rows.append(row)
  if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(str(row))
  result=json.loads(p.small(root/row['stdout']));results.append(result)
  assert result['passed'] and result['backend']==backend and result['tokens']==[47]*4 and result['closed'] and result['released'] and not result['physicalDeviceQualified']
except BaseException as e:primary=type(e).__name__+': '+str(e)
finally:
 p.signal.signal(p.signal.SIGTERM,previous)
 for name,pin in before.items():
  if p.sha(root/name)!=pin:endpoints.append(name)
passed=primary is None and not endpoints and len(rows)==2
result=dict(passed=passed,bindings=before,workers=rows,results=results,primary_error=primary,endpoint_errors=endpoints,physical_device_qualified=False)
with (folder/'validation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)));sys.exit(not passed)
