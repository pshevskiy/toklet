"""Check bounded task responses on Mac; no physical-device actions."""
import json
from pathlib import Path
import struct
import sys
import m7_apple_package as package
import m7_apple_integration as integration
ROOT=package.ROOT
SIMULATED=['bounded-256-continuation',*[f'hidden-stop-{n}' for n in (1,128,129,256)],
 'latched-between-chunks','final-callback-cancelled','continuation-failure',
 'sink-failure-after-valid-chunk','invalid-events-and-prefix','whole-request-preflight']
ORACLE={'tests/fixtures/tiny-v1/manifest.json':'1d40cec4f24ece5fdc7b727d3e96f41237621120908833ee6c7df83352acfdd4',
 'tests/fixtures/tiny-v1/cases.prompt.logits.f32le':'45b1def79d59fd9b37fa2b8504af2b5b2eea72d41a2c44187f9cd68e46b95c23'}

def independent_first(raw):
 if len(raw)!=1024 or package.hashlib.sha256(raw).hexdigest()!=ORACLE['tests/fixtures/tiny-v1/cases.prompt.logits.f32le']:
  raise ValueError('Independent tiny logits differ')
 row=struct.unpack('<256f',raw)[128:192]
 return max(range(64),key=lambda i:row[i])

def validate_output(text,backend,oracle):
 data=json.loads(text)
 if (data['passed'] is not True or data['backend']!=backend or data['simulated']!=SIMULATED
  or data['lifecycleCleared'] is not True or data['closed'] is not True or data['released'] is not True or data['physicalDeviceQualified'] is not False):
  raise ValueError('Task checks incomplete')
 if type(oracle) is not int or not 0<=oracle<64:raise ValueError('Independent first token required')
 for name,chunks in [('oneCall',1),('chunked',3),('hiddenStop',1)]:
  row=data[name];ids=row['ids']
  if any(type(x) is not int or not 0<=x<64 for x in ids):raise ValueError('Invalid task output IDs')
  for key in ('sampled','reason','history','evaluated','pending','chunks'):
   if type(row[key]) is not int:raise ValueError('Integer task state required')
  if row['chunks']!=chunks or row['pending']!=1:raise ValueError('Chunk or pending count differs')
  if name=='hiddenStop':
   if ids or row['sampled']!=1 or row['reason']!=2 or row['stopToken']!=oracle or type(row['stopToken']) is not int:
    raise ValueError('Hidden stop differs')
  elif (len(ids)!=8 or row['sampled']!=8 or row['reason']!=1 or row.get('stopToken') is not None
        or any(x in (62,63) for x in ids) or ids[0]!=oracle):raise ValueError('Output-limit response differs')
  if row['history']!=3+row['sampled'] or row['evaluated']!=row['history']-1:raise ValueError('Task prefix differs')
 if data['oneCall']['ids']!=data['chunked']['ids']:raise ValueError('Native chunking changed output')
 return data

def main(native,expected,output):
    native=native.resolve();output=output.resolve()
    if not native.is_relative_to(ROOT/'artifacts') or not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Repository artifacts required')
    output.mkdir();logs=output/'logs';logs.mkdir();(output/'tmp').mkdir()
    pins={};commands=[];checked={};primary=None;endpoints=[];env={};oracle=None
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    def merge(items):integration.merge(pins,items)
    def guard():
        for n,h in pins.items():
            if package.sha(ROOT/n)!=h:raise ValueError('Changed task-check input/output: '+n)
    def execute(args,label):
        guard();row=package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,180);commands.append(row)
        if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(label+': '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
        for key in ('stdout','stderr'):merge({row[key]:package.sha(ROOT/row[key])})
        guard();return ROOT/row['stdout']
    try:
        env,_=package.build_environment();env['TMPDIR']=str(output/'tmp')
        merge(integration.admit_build(native,expected));merge(integration.FIXTURE);merge(ORACLE)
        sources=[ROOT/'apple/Harness/NativeWorker.swift',ROOT/'apple/Harness/NativeTaskResponse.swift',ROOT/'apple/tests/swift/TaskResponseMain.swift']
        helpers=[Path(module.__file__).resolve() for module in (package,integration)]
        helpers.extend([Path(__file__).resolve(),Path(__file__).with_name('test_m7_native_task_check.py')])
        merge({str(p.relative_to(ROOT)):package.sha(p) for p in sources+helpers});guard()
        with (ROOT/'tests/fixtures/tiny-v1/cases.prompt.logits.f32le').open('rb') as stream:raw=stream.read(1025)
        oracle=independent_first(raw)
        execute(['xcodebuild','-version'],'xcode-version')
        execute(['xcrun','--sdk','macosx','swiftc','--version'],'swift-version')
        sdk=package.small(execute(['xcrun','--sdk','macosx','--show-sdk-path'],'mac-sdk')).strip()
        if sdk!='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk':raise ValueError('Unadmitted Mac SDK')
        common=['-swift-version','5','-strict-concurrency=complete','-warnings-as-errors','-O','-I',str(ROOT/'apple/include')]
        binary=output/'task';archive=native/'Toklet.xcframework/macos-arm64/libtoklet.a'
        execute(['xcrun','--sdk','macosx','swiftc','-sdk',sdk,'-module-cache-path',str(output/'MacModuleCache'),
                 '-target','arm64-apple-macos15.0',*common,*[str(p) for p in sources],str(archive),
                 '-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security',
                 '-lobjc','-liconv','-lresolv','-lc++','-o',str(binary)],'mac-compile')
        merge({str(binary.relative_to(ROOT)):package.sha(binary)})
        for backend,extra in [('cpu',[]),('metal',['metal'])]:
            stdout=execute([str(binary),str(ROOT/'artifacts/m7-native-tiny-v2'),*extra],backend)
            checked[backend]=validate_output(package.small(stdout),backend,oracle)
        for sdkname,target in [('iphoneos','arm64-apple-ios18.0'),('iphonesimulator','arm64-apple-ios18.0-simulator')]:
            sdk=package.small(execute(['xcrun','--sdk',sdkname,'--show-sdk-path'],sdkname+'-sdk')).strip()
            module=output/(sdkname+'.swiftmodule')
            execute(['xcrun','--sdk',sdkname,'swiftc','-sdk',sdk,'-module-cache-path',str(output/(sdkname+'-ModuleCache')),
                     '-target',target,*common,'-parse-as-library','-emit-module','-module-name','TokletTaskDiagnostics',
                     '-emit-module-path',str(module),str(sources[0]),str(sources[1])],sdkname+'-compile')
            for p in output.glob(sdkname+'.*'):merge({str(p.relative_to(ROOT)):package.sha(p)})
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        package.signal.signal(package.signal.SIGTERM,previous)
        try:guard()
        except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
    passed=primary is None and not endpoints and len(commands)==10 and len(checked)==2
    record=dict(format='toklet-m7-native-task-check-v1',passed=passed,collection_root=str(ROOT),bindings=pins,commands=commands,
                checked=checked,independent_tiny_first=oracle,controlled_environment=env,primary_error=primary,
                endpoint_errors=endpoints,full_quality_qualified=False,physical_device_qualified=False,m7_complete=False)
    with (output/'checks.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])))
