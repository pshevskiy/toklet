"""Bounded synthetic lifecycle core checks; actual OS integration is separate."""
from pathlib import Path
import json,sys
import m7_native_prefix_check as prefix
package=prefix.package;integration=prefix.integration;ROOT=package.ROOT
NAMES=['loaded','cancelled','inactive-request','background-request','session-released','foreground-observed','replayed','resumed','warning-request','background-after-warning','warning-unloaded','replay-rejected','fresh-load','closed']
CASES=['event-bounds','prefix-bounds','bad-label','overlap','wrong-end','future-clock','zero-clock','application-state','premature-detach','separate-observation-clock','pre-action-bound-after-cleanup','exhausted-trace-background','failed-trace-warning']
P=[1,7,13,3]
SETTLED={'loaded':([],0,0,0,0,0),'cancelled':(P+[47],5,4,1,3,0),'session-released':(P+[47],0,0,0,0,0),
 'foreground-observed':(P+[47],0,0,0,0,0),'replayed':(P+[47],5,5,0,1,0),'resumed':(P+[47,47],6,5,1,1,0),
 'warning-unloaded':([],0,0,0,0,0),'replay-rejected':([],0,0,0,0,1),'fresh-load':([],0,0,0,0,0),'closed':(None,0,0,0,0,0)}
SYNTHETIC={'inactive-request','background-request','foreground-observed','warning-request','background-after-warning'}
COMMANDS={'loaded':'load','cancelled':'generate','replayed':'replay','resumed':'continue','closed':'close'}
APPLICATIONS=dict(zip(NAMES,[None,None,1,2,2,0,0,0,0,2,0,0,0,0]))
OWNERS={n:(True,True) for n in ('loaded','cancelled','replayed','resumed','fresh-load')}
OWNERS.update({n:(True,False) for n in ('session-released','foreground-observed')})
OWNERS.update({n:(False,False) for n in ('warning-unloaded','replay-rejected','closed')})
def state(value):
 for k in ('busy','closed','workerFinished','hasModel','hasSession'):
  if type(value[k]) is not bool:raise ValueError('Typed lifecycle owner state required')
 for k in ('status','history','evaluated','pending','reason','savedHistory'):
  if type(value[k]) is not int or not 0<=value[k]<=32:raise ValueError('Bounded lifecycle state required')
 if value['history']!=value['evaluated']+value['pending'] or value['pending']>1:raise ValueError('Lifecycle commit prefix differs')
 for tokens in (value.get('savedPrefix'),value['ids']):
  if tokens is not None and (len(tokens)>128 or any(type(t) is not int or not 0<=t<64 for t in tokens)):raise ValueError('Lifecycle raw IDs differ')
 if value.get('savedPrefix') is not None and len(value['savedPrefix'])!=value['savedHistory']:raise ValueError('Lifecycle saved prefix count differs')
def validate_output(text,backend):
 d=json.loads(text);trace=d['trace'];events=trace['events']
 if (d['passed'] is not True or d['backend']!=backend or d['cases']!=CASES or d['closed'] is not True or d['released'] is not True
     or d['physicalDeviceQualified'] is not False or d['actualOSLifecycleQualified'] is not False or trace.get('error') is not None
     or trace.get('activeCommand') is not None or trace['detachedAfterClose'] is not True or [e['name'] for e in events]!=NAMES):raise ValueError('Complete synthetic lifecycle outcome required')
 last=0;peak=0
 for index,event in enumerate(events):
  app=event.get('applicationState')
  if app!=APPLICATIONS[event['name']] or app is not None and type(app) is not int or event.get('command')!=COMMANDS.get(event['name']):raise ValueError('Lifecycle command/application identity differs')
  if type(event['index']) is not int or event['index']!=index or event['origin']!=('syntheticCheck' if event['name'] in SYNTHETIC else 'coordinator'):raise ValueError('Lifecycle event order/provenance differs')
  observed=event['observedUptimeNanoseconds'];recorded=event['recordedUptimeNanoseconds']
  if type(observed) is not int or type(recorded) is not int or not 0<observed<=recorded or recorded<last:raise ValueError('Lifecycle observation clocks differ')
  last=recorded;state(event['before']);state(event['state']);f=event['footprint']
  if any(type(f[k]) is not int or f[k]<=0 for k in ('physicalBytes','lifetimePeakPhysicalBytes','returnedNaturalCount')) or not f['physicalBytes']<=f['lifetimePeakPhysicalBytes'] or f['lifetimePeakPhysicalBytes']<peak:raise ValueError('Lifecycle process footprint differs')
  peak=f['lifetimePeakPhysicalBytes'];value=event['state'];expected=SETTLED.get(event['name'])
  if expected is not None:
   saved,history,evaluated,pending,reason,status=expected
   if (value['hasModel'],value['hasSession'])!=OWNERS[event['name']] or value['workerFinished']!=(event['name']=='closed') or value['closed']!=(event['name']=='closed'):raise ValueError('Exact settled lifecycle owner state differs')
   if value.get('savedPrefix')!=saved or tuple(value[k] for k in ('history','evaluated','pending','reason','status'))!=(history,evaluated,pending,reason,status) or value['busy']:raise ValueError('Exact settled lifecycle state differs')
  if event['name'] in ('inactive-request','background-request','warning-request'):
   before=event['before'];wanted=P+[47]*(2 if event['name']=='warning-request' else 1)
   if before.get('savedPrefix')!=wanted or not before['hasSession'] or before['busy']:raise ValueError('Lifecycle request lost pre-action prefix')
  if event['name'] in ('session-released','foreground-observed') and (value['hasSession'] or not value['hasModel']):raise ValueError('Foreground must not replay implicitly')
  if event['name'] in ('warning-unloaded','replay-rejected','closed') and (value['hasModel'] or value['hasSession']):raise ValueError('Lifecycle owners retained after release')
  if event['name']=='closed' and (not value['closed'] or not value['workerFinished']):raise ValueError('Completed close missing')
 return d

def main(native,expected,output):
 native=native.resolve();output=output.resolve()
 if any(not p.is_relative_to(ROOT/'artifacts') for p in (native,output)):raise ValueError('Repository artifact paths required')
 output.mkdir();logs=output/'logs';logs.mkdir();temporary=output/'tmp';temporary.mkdir();pins={};commands=[];checked={};nested=None;primary=None;endpoints=[]
 previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
 def merge(extra):integration.merge(pins,extra)
 def guard():
  for n,h in pins.items():
   if package.sha(ROOT/n)!=h:raise ValueError('Changed prefix-check endpoint: '+n)
 def execute(args,label,timeout=180):
  guard();row=package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row)
  if row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:raise RuntimeError('Owned prefix check failed: '+label)
  for field in ('stdout','stderr'):merge({row[field]:package.sha(ROOT/row[field])})
  guard();return ROOT/row['stdout']
 try:
  env,_=package.build_environment();env['TMPDIR']=str(temporary);merge(integration.admit_build(native,expected));merge(integration.FIXTURE)
  sources=[ROOT/'apple/Harness/NativeWorker.swift',ROOT/'apple/Harness/NativeMeasurement.swift',ROOT/'apple/Harness/NativeLifecycle.swift',ROOT/'apple/tests/swift/LifecycleMain.swift']
  helpers=[Path(v.__file__).resolve() for v in (prefix,package,integration)]+[Path(__file__).resolve(),Path(__file__).with_name('test_m7_native_lifecycle_check.py')]
  merge({str(p.relative_to(ROOT)):package.sha(p) for p in sources+helpers})
  execute(['xcodebuild','-version'],'xcode-version')
  sdk=package.small(execute(['xcrun','--sdk','macosx','--show-sdk-path'],'sdk-path')).strip()
  if sdk!='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk':raise ValueError('Unadmitted Mac SDK')
  execute(['xcrun','--sdk','macosx','swiftc','--version'],'swift-version')
  binary=output/'lifecycle-checks';archive=native/'Toklet.xcframework/macos-arm64/libtoklet.a'
  execute(['xcrun','--sdk','macosx','swiftc','-sdk',sdk,'-module-cache-path',str(output/'ModuleCache'),'-swift-version','5','-strict-concurrency=complete','-warnings-as-errors',
    '-target','arm64-apple-macos15.0','-O','-I',str(ROOT/'apple/include'),*[str(p) for p in sources],str(archive),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security','-lobjc','-liconv','-lresolv','-lc++','-o',str(binary)],'compile')
  merge({str(binary.relative_to(ROOT)):package.sha(binary)})
  for sdk_name,target in [('iphoneos','arm64-apple-ios18.0'),('iphonesimulator','arm64-apple-ios18.0-simulator')]:
   sdk_path=Path('/Applications/Xcode.app/Contents/Developer/Platforms')/('iPhoneOS.platform/Developer/SDKs/iPhoneOS27.1.sdk' if sdk_name=='iphoneos' else 'iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator27.1.sdk')
   execute(['xcrun','--sdk',sdk_name,'swiftc','-typecheck','-sdk',str(sdk_path),'-module-cache-path',str(output/('ModuleCache-'+sdk_name)),
       '-swift-version','5','-strict-concurrency=complete','-warnings-as-errors','-target',target,'-I',str(ROOT/'apple/include'),*[str(p) for p in sources]],sdk_name+'-typecheck')
  for backend,extra in [('cpu',[]),('metal',['metal'])]:checked[backend]=validate_output(package.small(execute([str(binary),str(ROOT/'artifacts/m7-native-tiny-v2'),*extra],backend)),backend)
  nested=output/'prefix-regression'
  execute([sys.executable,str(Path(prefix.__file__).resolve()),str(native),expected,str(nested)],'prefix-regression',1800)
  p=nested/'checks.json'
  with p.open('rb') as f:raw=f.read(16*1024**2+1)
  if len(raw)>16*1024**2:raise ValueError('Oversized nested receipt')
  d=json.loads(raw)
  if not d['passed'] or d['primary_error'] or d['endpoint_errors']:raise ValueError('Task regression failed')
  merge({str(p.relative_to(ROOT)):package.hashlib.sha256(raw).hexdigest()});merge(d['bindings'])
 except BaseException as e:primary=type(e).__name__+': '+str(e)
 finally:
  package.signal.signal(package.signal.SIGTERM,previous)
  try:guard()
  except BaseException as e:endpoints.append(type(e).__name__+': '+str(e))
 passed=primary is None and not endpoints and len(commands)==9 and set(checked)=={'cpu','metal'}
 result=dict(format='toklet-m7-native-lifecycle-check-v1',passed=passed,collection_root=str(ROOT),bindings=pins,commands=commands,checked=checked,nested_receipt=str((nested/'checks.json').relative_to(ROOT)) if nested else None,primary_error=primary,endpoint_errors=endpoints,physical_device_qualified=False,m7_complete=False)
 with (output/'checks.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)));return 0 if passed else 1
if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])))
