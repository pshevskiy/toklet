from pathlib import Path
import json,sys
R=Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(R/'tools/experiments'));import m7_apple_package as p;import m7_apple_integration as integration
folder=R/'artifacts/m7-worker-timing-v1';folder.mkdir();env,_=p.build_environment();env['TMPDIR']=str(folder)
archive=R/'artifacts/m7-apple-package-v4/Toklet.xcframework/macos-arm64/libtoklet.a'
pins=integration.admit_build(R/'artifacts/m7-apple-package-v4','1185314043649692b7e41aff9b90e37355201eef71852965c20ef69513e6f13c')
paths=[R/n for n in ['apple/Harness/NativeWorker.swift','apple/tests/swift/WorkerChecks.swift','apple/tests/swift/main.swift','artifacts/m7-native-tiny-v2/config.json','artifacts/m7-native-tiny-v2/model.safetensors']]
pins.update({str(n.relative_to(R)):p.sha(n) for n in paths});rows=[];results=[];primary=None;errors=[]
try:
 cmd=['xcrun','--sdk','macosx','swiftc','-sdk','/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk','-module-cache-path',str(folder/'ModuleCache'),'-swift-version','5','-strict-concurrency=complete','-warnings-as-errors','-target','arm64-apple-macos15.0','-O','-I',str(R/'apple/include'),*[str(n) for n in paths[:3]],str(archive),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security','-lobjc','-liconv','-lresolv','-lc++','-o',str(folder/'worker-check')]
 for label,command in [('compile',cmd),('cpu',[str(folder/'worker-check'),str(R/'artifacts/m7-native-tiny-v2')]),('metal',[str(folder/'worker-check'),str(R/'artifacts/m7-native-tiny-v2'),'metal'])]:
  row=p.owned(command,folder/(label+'.stdout'),folder/(label+'.stderr'),env,180);rows.append(row)
  if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(str(row))
  if label!='compile':
   d=json.loads(p.small(R/row['stdout']));assert d['passed'] and d['backend']==label and d['closed'] and d['released'] and d['tokens']==[47]*4
   for c in d['timingChecks']:
    assert len(c['tokens'])==len(c['events'])
    for i,e in enumerate(c['events']):assert e['index']==i and c['submittedUptimeNanoseconds']<=e['callbackUptimeNanoseconds']<=c['completedUptimeNanoseconds']
   results.append(d)
except BaseException as e:primary=type(e).__name__+': '+str(e)
finally:
 for n,h in pins.items():
  if p.sha(R/n)!=h:errors.append(n)
passed=primary is None and not errors and len(results)==2
(folder/'validation.json').write_text(json.dumps(dict(passed=passed,bindings=pins,commands=rows,results=results,primary_error=primary,endpoint_errors=errors,physical_device_qualified=False),indent=2)+'\n');print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=errors)));sys.exit(not passed)
