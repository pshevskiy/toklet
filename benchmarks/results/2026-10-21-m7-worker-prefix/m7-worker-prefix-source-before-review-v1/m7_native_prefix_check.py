"""Opt-in completed-prefix CPU/Metal checks plus full existing app regressions."""
from pathlib import Path
import json,sys
import m7_native_tasks_check as tasks
package=tasks.package;integration=tasks.integration;ROOT=package.ROOT
PROMPT=[1,7,13,3]
PREFIXES={'load':[],'prompt':PROMPT+[47]*2,'recoverable-failure':PROMPT+[47]*2,'continue':PROMPT+[47]*4,
 'suspend':PROMPT+[47]*4,'cancelled-replay':PROMPT+[47]*4,'replay':PROMPT+[47]*4,'after-replay':PROMPT+[47]*5,
 'reset':[],'cancel':PROMPT+[47],'interrupt-suspend':PROMPT+[47],'cancel-prefix-replay':PROMPT+[47],'resume':PROMPT+[47]*2,
 'reset-hidden':[],'hidden-stop':PROMPT+[47],'unload':[],'close':None,
 **{n:None for n in ('default-load','default-prompt','default-suspend','default-replay','default-reset','default-unload','default-close')},'race-unload':[],'race-close':None}
def validate_output(text,backend):
 d=json.loads(text)
 if (d['passed'] is not True or d['backend']!=backend or d['races']!=['unload','close'] or d['closed'] is not True or d['released'] is not True
     or d['physicalDeviceQualified'] is not False or [r['name'] for r in d['rows']]!=list(PREFIXES)):raise ValueError('Complete prefix checks required')
 for row in d['rows']:
  expected=PREFIXES[row['name']]
  if row.get('prefix')!=expected or row['status']!=(1 if row['name']=='recoverable-failure' else 0):raise ValueError('Exact prefix/status differs')
  for field in ('saved','status','history','evaluated','pending','reason'):
   if type(row[field]) is not int or not 0<=row[field]<=32:raise ValueError('Bounded prefix state required')
  if row['history']!=row['evaluated']+row['pending'] or row['pending']>1:raise ValueError('Prefix commit state differs')
  if expected is not None and (row['saved']!=len(expected) or any(type(x) is not int for x in row['prefix'])):raise ValueError('Prefix saved count differs')
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
  sources=[ROOT/'apple/Harness/NativeWorker.swift',ROOT/'apple/tests/swift/PrefixMain.swift']
  helpers=[Path(v.__file__).resolve() for v in (tasks,package,integration)]+[Path(__file__).resolve(),Path(__file__).with_name('test_m7_native_prefix_check.py')]
  merge({str(p.relative_to(ROOT)):package.sha(p) for p in sources+helpers})
  execute(['xcodebuild','-version'],'xcode-version')
  sdk=package.small(execute(['xcrun','--sdk','macosx','--show-sdk-path'],'sdk-path')).strip()
  if sdk!='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk':raise ValueError('Unadmitted Mac SDK')
  execute(['xcrun','--sdk','macosx','swiftc','--version'],'swift-version')
  binary=output/'prefix-checks';archive=native/'Toklet.xcframework/macos-arm64/libtoklet.a'
  execute(['xcrun','--sdk','macosx','swiftc','-sdk',sdk,'-module-cache-path',str(output/'ModuleCache'),'-swift-version','5','-strict-concurrency=complete','-warnings-as-errors',
    '-target','arm64-apple-macos15.0','-O','-I',str(ROOT/'apple/include'),*[str(p) for p in sources],str(archive),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security','-lobjc','-liconv','-lresolv','-lc++','-o',str(binary)],'compile')
  merge({str(binary.relative_to(ROOT)):package.sha(binary)})
  for backend,extra in [('cpu',[]),('metal',['metal'])]:checked[backend]=validate_output(package.small(execute([str(binary),str(ROOT/'artifacts/m7-native-tiny-v2'),*extra],backend)),backend)
  nested=output/'tasks-regression'
  execute([sys.executable,str(Path(tasks.__file__).resolve()),str(native),expected,str(nested)],'tasks-regression',1800)
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
 passed=primary is None and not endpoints and len(commands)==7 and set(checked)=={'cpu','metal'}
 result=dict(format='toklet-m7-native-prefix-check-v1',passed=passed,collection_root=str(ROOT),bindings=pins,commands=commands,checked=checked,nested_receipt=str((nested/'checks.json').relative_to(ROOT)) if nested else None,primary_error=primary,endpoint_errors=endpoints,physical_device_qualified=False,m7_complete=False)
 with (output/'checks.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)));return 0 if passed else 1
if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])))
