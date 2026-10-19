import json,sys
from pathlib import Path
sys.path.insert(0,'tools/experiments');import m7_apple_package as p
root=Path.cwd();folder=root/'artifacts/m7-history-native-v1';folder.mkdir();env,_=p.build_environment();env['TMPDIR']=str(folder)
paths=['src/native.rs','src/native/tests.rs','apple/include/toklet.h','apple/tests/native_smoke.c','artifacts/m7-history-target-v1/release/libtoklet.a','artifacts/m7-native-tiny-v2/config.json','artifacts/m7-native-tiny-v2/model.safetensors',str(Path(__file__).relative_to(root))]
before={name:p.sha(root/name) for name in paths};rows=[];primary=None;endpoints=[]
try:
 sdk='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk'
 executable=folder/'native-client'
 cmd=['xcrun','--sdk','macosx','clang','-target','arm64-apple-macos15.0','-isysroot',sdk,'-std=c11','-Wall','-Wextra','-Werror','-I'+str(root/'apple/include'),str(root/'apple/tests/native_smoke.c'),str(root/'artifacts/m7-history-target-v1/release/libtoklet.a'),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security','-lobjc','-liconv','-lresolv','-lc++','-Wl,-fatal_warnings','-o',str(executable)]
 for label,command in [('compile',cmd),('cpu',[str(executable),str(root/'artifacts/m7-native-tiny-v2')]),('metal',[str(executable),str(root/'artifacts/m7-native-tiny-v2'),'metal'])]:
  row=p.owned(command,folder/(label+'.stdout'),folder/(label+'.stderr'),env,180);rows.append(row)
  if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(str(row))
  if label=='compile':before[str(executable.relative_to(root))]=p.sha(executable)
  else:assert json.loads(p.small(root/row['stdout']))==dict(ids=[47]*4,abi=1,passed=True)
except BaseException as e:primary=type(e).__name__+': '+str(e)
finally:
 for name,pin in before.items():
  if p.sha(root/name)!=pin:endpoints.append(name)
passed=primary is None and not endpoints and len(rows)==3
result=dict(passed=passed,bindings=before,workers=rows,primary_error=primary,endpoint_errors=endpoints,physical_device_qualified=False)
with (folder/'validation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)));sys.exit(not passed)
