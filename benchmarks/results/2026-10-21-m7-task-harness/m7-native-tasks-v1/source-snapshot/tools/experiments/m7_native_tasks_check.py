"""Sealed bounded task-runner checks with quality/measurement/app regressions."""
import json
import math
from pathlib import Path
import struct
import sys
import m7_apple_package as package
import m7_apple_integration as integration
import m7_native_task_check as task
import m7_native_quality_check as quality
import m7_native_measurement_check as measurement
import m7_swift_harness_build as builder
ROOT=package.ROOT
EXPECTED_FILES={'Model/'+Path(name).name:sha for name,sha in integration.FIXTURE.items()}
EXPECTED_TASKS=dict(version=1,vocabulary=64,stops=[62,63],cases=[dict(id='first',promptTokens=[1,7,13],maximum=3),dict(id='second',promptTokens=[1,7],maximum=2)])
CASES=['cpu-complete','metal-complete',*('reject-'+key for key in ['version','overflow-count','capacity','chunk','budget','tasks-path','tasks-in-model','tasks-hash','inventory','extra-config','config-sha','set-version','vocabulary','duplicate-stops','stop-range','case-count','extra-set','duplicate-case','case-id','overflow-output','prompt-range','prompt-capacity','extra-case','tasks-symlink','tasks-size','output-chunk','output-case','model-endpoint','tasks-endpoint'])]


def integer(value,minimum=0,maximum=2**64-1):
 if type(value) is not int or not minimum<=value<=maximum:raise ValueError('Bounded integer required')
 return value

def footprint(value):
 a=integer(value['physicalBytes'],1,192*1024**2);b=integer(value['lifetimePeakPhysicalBytes'],a,192*1024**2)
 integer(value['returnedNaturalCount'],1,1024)
 return b

def environment(value):
 if (value.get('applicationState') is not None or type(value['thermalState']) is not int
  or value['thermalState'] not in range(4) or type(value['lowPowerMode']) is not bool
  or value.get('batteryState') is not None or value.get('batteryLevel') is not None):raise ValueError('Mac task environment differs')

def response(value,index):
 maximum=(3,2)[index];prompt=(3,2)[index]
 if value['ids']!=[47]*maximum or any(type(x) is not int for x in value['ids']):raise ValueError('Independent task IDs differ')
 for key,expected in [('sampled',maximum),('reason',1),('history',prompt+maximum),('evaluated',prompt+maximum-1),('pending',1),('chunks',1)]:
  if integer(value[key])!=expected:raise ValueError('Task response prefix/count differs')
 if value.get('stopToken') is not None:raise ValueError('Unexpected hidden task stop')

def validate_output(text,tasks_sha):
 if type(tasks_sha) is not str or len(tasks_sha)!=64 or any(c not in '0123456789abcdef' for c in tasks_sha):raise ValueError('Complete task artifact SHA required')
 markers=['TASKS_CHECK_CHUNK_JSON=','TASKS_CHECK_CASE_JSON=','TASKS_CHECK_RESULT_JSON=','TASKS_CHECK_FAILURE_JSON=','TASKS_CHECKS_JSON='];events=[]
 for line in text.splitlines():
  if not line.strip():continue
  matches=[m for m in markers if line.startswith(m)]
  if len(matches)!=1:raise ValueError('Unexpected task output/error marker')
  events.append((matches[0],json.loads(line[len(matches[0]):])))
 expected=([markers[0],markers[1]]*2+[markers[2]])*2+[markers[3]]*5+[markers[4]]
 if [e[0] for e in events]!=expected:raise ValueError('Exact task stream sequence differs')
 results=[];global_peak=0;last_completion=0
 for offset,backend in [(0,'cpu'),(5,'metal')]:
  group=events[offset:offset+5];result=group[4][1];peak=global_peak;last_callback=0;admission=integer(result['loadedAdmittedBytes'],1,128*1024**2)
  if (result['backend']!=backend or result['passed'] is not True or result.get('error') is not None or result['cleanupErrors'] or result['endpointErrors']
   or result['closed'] is not True or result['released'] is not True or result['fullQualityQualified'] is not False or result['physicalDeviceQualified'] is not False or result['m7Complete'] is not False
   or integer(result['interruptionCount'])!=0 or integer(result['chunksCompleted'])!=2 or integer(result['casesCompleted'])!=2
   or result['filesBefore']!=EXPECTED_FILES or result['filesAfter']!=EXPECTED_FILES or result.get('tasksBeforeSHA256')!=tasks_sha or result.get('tasksAfterSHA256')!=tasks_sha):raise ValueError('Task result/admission/proofs differ')
  elapsed=result['loadAndSessionSeconds']
  if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('Task load bracket differs')
  checkpoints=result['checkpoints']
  if [c['phase'] for c in checkpoints]!=['before-artifact-proof','after-artifact-proof','loaded','case-0','case-1','unloaded','closed','after-final-artifact-proof']:raise ValueError('Task memory phases differ')
  # Replay observations in chronological order: three initial checkpoints,
  # chunk+case+case checkpoint pairs, then unload/close/final proof.
  observations=[c['footprint'] for c in checkpoints[:3]]
  for index in (0,1):
   chunk=group[index*2][1];case=group[index*2+1][1];item=chunk['chunk'];maximum=(3,2)[index];prompt=(3,2)[index]
   for sample in (chunk,case):
    if integer(sample['caseIndex'])!=index or sample['caseID']!=('first','second')[index] or integer(sample['admittedBytes'],1,128*1024**2)!=admission:raise ValueError('Task sample identity/admission differs')
    environment(sample['environment'])
   for key,expected_value in [('index',0),('requested',maximum),('reason',1),('history',prompt+maximum),('evaluated',prompt+maximum-1),('pending',1)]:
    if integer(item[key])!=expected_value:raise ValueError('Task chunk prefix/budget differs')
   if item['ids']!=[47]*maximum or any(type(x) is not int for x in item['ids']) or item.get('stopToken') is not None:raise ValueError('Task chunk visible/hidden output differs')
   timings=item['tokenTimings'];start=integer(chunk['submittedUptimeNanoseconds'],1);end=integer(chunk['completedUptimeNanoseconds'],start)
   if start<last_completion or start<last_callback or len(timings)!=maximum:raise ValueError('Task caller clock/coverage differs')
   elapsed=0.0
   for n,timing in enumerate(timings):
    value=timing['nativeElapsedSeconds'];callback=integer(timing['callbackUptimeNanoseconds'],max(start,last_callback),end)
    if integer(timing['index'])!=n or type(value) not in (int,float) or not math.isfinite(value) or not elapsed<=value<=(end-start)/1e9:raise ValueError('Task callback clocks/order differ')
    elapsed=value;last_callback=callback
   last_completion=end
   response(case['response'],index)
   for key in ('resetHistory','resetEvaluated','resetPending'):
    if integer(case[key])!=0:raise ValueError('Task final reset differs')
   observations.extend([chunk['footprint'],case['footprint'],checkpoints[index+3]['footprint']])
  observations.extend(c['footprint'] for c in checkpoints[5:])
  for observed in observations:
   next_peak=footprint(observed)
   if next_peak<peak:raise ValueError('Task lifetime peak moved backward')
   peak=next_peak
  global_peak=peak;results.append(result)
 failures=[row[1] for row in events[10:15]]
 if [f['label'] for f in failures]!=['overflow-count','output-chunk','output-case','model-endpoint','tasks-endpoint']:raise ValueError('Task failure inventory differs')
 for row in failures:
  label=row['label'];result=row['result']
  for key in ('casesCompleted','chunksCompleted','loadedAdmittedBytes','interruptionCount'):integer(result[key])
  for checkpoint in result['checkpoints']:footprint(checkpoint['footprint'])
  if result['passed'] is not False or result['released'] is not True or result['fullQualityQualified'] is not False or result['physicalDeviceQualified'] is not False or result['m7Complete'] is not False:raise ValueError('Failed tasks qualified or retained owner')
  if label=='overflow-count':
   if (not result.get('error') or result['closed'] is not False or result['casesCompleted']!=0 or result['chunksCompleted']!=0 or result['loadedAdmittedBytes']!=0
    or result['filesBefore'] or result['filesAfter'] or result.get('tasksBeforeSHA256') is not None or result.get('tasksAfterSHA256') is not None
    or result['loadAndSessionSeconds']!=0 or [c['phase'] for c in result['checkpoints']]!=['closed']):raise ValueError('Invalid count issued task work')
  else:
   if result['closed'] is not True or result['cleanupErrors'] or result['filesBefore']!=EXPECTED_FILES or result.get('tasksBeforeSHA256')!=tasks_sha:raise ValueError('Failed task proof/cleanup differs')
   if label.startswith('output'):
    if (not result.get('error') or result['endpointErrors'] or result['casesCompleted']!=1 or result['chunksCompleted']!=(1 if label=='output-chunk' else 2)
     or result['filesAfter']!=EXPECTED_FILES or result.get('tasksAfterSHA256')!=tasks_sha):raise ValueError('Failed sink completed task')
   elif (result.get('error') is not None or not result['endpointErrors'] or result['casesCompleted']!=2 or result['chunksCompleted']!=2
     or result.get('tasksAfterSHA256') is not None or result['filesAfter']!=({} if label=='model-endpoint' else EXPECTED_FILES)):raise ValueError('Task endpoint mutation accepted')
 if events[-1][1]!=dict(passed=True,cases=CASES,physicalDeviceQualified=False):raise ValueError('Task admission case coverage differs')
 return dict(results=results,failures=failures,summary=events[-1][1],chunk_count=4,case_count=4)


def nested_metadata(path,expected_sha=None):
 with path.open('rb') as stream:raw=stream.read(16*1024**2+1)
 if len(raw)>16*1024**2:raise ValueError('Oversized nested task regression')
 digest=package.hashlib.sha256(raw).hexdigest()
 if expected_sha is not None and digest!=expected_sha:raise ValueError('Consumed nested receipt SHA differs')
 return json.loads(raw),digest


def main(native,expected,output):
 native=native.resolve();output=output.resolve()
 if not native.is_relative_to(ROOT/'artifacts') or not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Repository artifacts required')
 output.mkdir();logs=output/'logs';logs.mkdir();(output/'tmp').mkdir()
 pins={};commands=[];checked=None;primary=None;endpoints=[];env={};nested={};links=[];tasks_sha=None
 previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
 def merge(items):integration.merge(pins,items)
 def guard():
  for link in links:builder.verify_library_link(link)
  for name,digest in pins.items():
   if package.sha(ROOT/name)!=digest:raise ValueError('Changed tasks-check input/output: '+name)
 def execute(args,label,timeout=180):
  guard();row=package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row)
  if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(label+': '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
  for key in ('stdout','stderr'):merge({row[key]:package.sha(ROOT/row[key])})
  guard();return ROOT/row['stdout']
 def admit_nested(path,expected_sha=None,depth=0):
  if depth>2:raise ValueError('Unbounded nested task regression')
  record,digest=nested_metadata(path,expected_sha)
  if not record['passed'] or record['primary_error'] or record['endpoint_errors']:raise ValueError('Nested task regression failed')
  merge({str(path.relative_to(ROOT)):digest});merge(record['bindings'])
  if 'bundle_pins' in record:merge(record['bundle_pins'])
  if 'library_link' in record:links.append(record['library_link'])
  for row in record['commands']:
   if type(row['exit_code']) is not int or row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:raise ValueError('Nested task command failed')
   for key in ('stdout','stderr'):merge({row[key]:package.sha(ROOT/row[key])})
  nested[str(path.relative_to(ROOT))]=digest
  for name,expected_sha in record.get('nested_receipts',{}).items():
   child=(ROOT/name).resolve()
   if not child.is_relative_to(path.parent):raise ValueError('Nested task receipt path differs')
   admit_nested(child,expected_sha,depth+1)
  return record
 try:
  env,_=package.build_environment();env['TMPDIR']=str(output/'tmp')
  merge(integration.admit_build(native,expected));merge(integration.FIXTURE);merge(task.ORACLE)
  sources=[ROOT/'apple/Harness'/name for name in ['NativeWorker.swift','NativeMeasurement.swift','NativeTaskResponse.swift','NativeTasks.swift']]
  sources.append(ROOT/'apple/tests/swift/TasksMain.swift')
  helpers=[Path(m.__file__).resolve() for m in (package,integration,task,quality,measurement,builder)]
  helpers += [Path(__file__).resolve(),Path(__file__).with_name('test_m7_native_tasks_check.py'),ROOT/'apple/Harness/HarnessApp.swift',ROOT/'apple/TokletHarness.xcodeproj/project.pbxproj']
  merge({str(p.relative_to(ROOT)):package.sha(p) for p in sources+helpers});guard()
  with (ROOT/'tests/fixtures/tiny-v1/cases.prompt.logits.f32le').open('rb') as stream:oracle_raw=stream.read(1025)
  if task.independent_first(oracle_raw)!=47 or max(range(64),key=lambda i:struct.unpack('<256f',oracle_raw)[64+i])!=47:raise ValueError('Independent prompt first IDs differ')
  execute(['xcodebuild','-version'],'xcode-version');execute(['xcrun','--sdk','macosx','swiftc','--version'],'swift-version')
  sdk=package.small(execute(['xcrun','--sdk','macosx','--show-sdk-path'],'sdk-path')).strip()
  if sdk!='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk':raise ValueError('Unadmitted Mac SDK')
  binary=output/'tasks'
  execute(['xcrun','--sdk','macosx','swiftc','-sdk',sdk,'-module-cache-path',str(output/'ModuleCache'),'-swift-version','5',
   '-strict-concurrency=complete','-warnings-as-errors','-target','arm64-apple-macos15.0','-O','-I',str(ROOT/'apple/include'),
   *[str(p) for p in sources],str(native/'Toklet.xcframework/macos-arm64/libtoklet.a'),'-framework','Metal',
   '-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security','-lobjc','-liconv','-lresolv','-lc++','-o',str(binary)],'compile')
  merge({str(binary.relative_to(ROOT)):package.sha(binary)})
  stdout=execute([str(binary),str(ROOT/'artifacts/m7-native-tiny-v2'),str(output/'cases')],'tasks-checks',300)
  for backend in ('cpu','metal'):
   folder=output/'cases'/backend;path=folder/'tasks.json'
   with path.open('rb') as stream:raw=stream.read(512*1024+1)
   digest=package.hashlib.sha256(raw).hexdigest()
   if len(raw)>512*1024 or json.loads(raw)!=EXPECTED_TASKS or tasks_sha not in (None,digest):raise ValueError('Actual tiny task input differs')
   tasks_sha=digest;merge({str(path.relative_to(ROOT)):digest})
   path=folder/'job.json'
   with path.open('rb') as stream:raw=stream.read(512*1024+1)
   if len(raw)>512*1024:raise ValueError('Oversized actual tiny task config')
   config=json.loads(raw)
   if config!=dict(version=1,modelFolder='Model',packedFolder='',files=EXPECTED_FILES,metal=backend=='metal',capacity=6,chunk=3,
      numericalBudget=128*1024**2,processBudget=192*1024**2,tasksFile='tasks.json',tasksSHA256=tasks_sha,caseCount=2):raise ValueError('Actual tiny task config differs')
   merge({str(path.relative_to(ROOT)):package.hashlib.sha256(raw).hexdigest()})
  checked=validate_output(package.small(stdout),tasks_sha)
  destination=output/'quality-regression'
  execute([sys.executable,str(Path(quality.__file__).resolve()),str(native),expected,str(destination)],'quality-regression',1800)
  record=admit_nested(destination/'checks.json')
  oracle=quality.teacher.independent_losses(oracle_raw)
  if record['checked']!=quality.validate_output(package.small(destination/'logs/quality-checks.stdout'),oracle):raise ValueError('Quality raw regression replay differs')
 except BaseException as error:primary=type(error).__name__+': '+str(error)
 finally:
  package.signal.signal(package.signal.SIGTERM,previous)
  try:guard()
  except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
 passed=primary is None and not endpoints and len(commands)==6 and len(nested)==3 and checked is not None
 record=dict(format='toklet-m7-native-tasks-check-v1',passed=passed,collection_root=str(ROOT),bindings=pins,commands=commands,
  checked=checked,tasks_sha256=tasks_sha,nested_receipts=nested,controlled_environment=env,primary_error=primary,endpoint_errors=endpoints,
  full_quality_qualified=False,physical_device_qualified=False,m7_complete=False)
 with (output/'checks.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)))
 return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])))
