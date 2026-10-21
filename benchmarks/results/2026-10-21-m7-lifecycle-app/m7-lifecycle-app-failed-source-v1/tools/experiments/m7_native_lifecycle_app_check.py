"""Lifecycle runner and app checks; simulated scenes never qualify a device."""
from pathlib import Path
import json,sys
import m7_native_lifecycle_check as core
import m7_physical_measurement as measurement
package=core.package;integration=core.integration;ROOT=package.ROOT
CONFIG_CASES=['config-version','config-boolean','config-extra','config-capacity','config-chunk','config-output','config-reference-space','config-vocabulary','config-prompt-range','config-empty-prompt','config-budget-overflow','config-traversal','config-overlap','config-inventory','config-canonical-path','config-size']
UNIT_CASES=CONFIG_CASES+['direct-return-sequence','return-inactive-sequence','duplicate-inactive','background-before-inactive','active-before-background','duplicate-background','duplicate-return-inactive','missing-roundtrip','synthetic-not-system','ready-before-interruption','late-ready-index','late-ready-observation','background-ready','unknown-system-foreground','wrong-ready-prefix','independent-budget-and-mutation-failures','no-unadmitted-reproof','tiny-diagnostic-fails-closed']
def validate_output(text,backend,kind,config):
 d=json.loads(text);result=d['diagnostic']
 if (d['passed'] is not True or d['backend']!=backend or d['kind']!=kind or d['actualOSLifecycleQualified'] is not False
     or d['physicalDeviceQualified'] is not False or d['cases']!=(UNIT_CASES if kind=='unit' else ['small-synthetic-roundtrip'])
     or result['closed'] is not True or result['released'] is not True or result['actualOSLifecycleQualified'] is not False
     or result['physicalDeviceQualified'] is not False or result['m7Complete'] is not False or result['filesBefore']!=config['files'] or result['filesAfter']!=config['files']):raise ValueError('Complete lifecycle app test outcome required')
 if kind=='unit':
  if result['passed'] is not False or result['error'] not in ('No demonstrably active generation window','Forced READY sink failure') or len(d['trips'])!=2:raise ValueError('Tiny failure must remain closed and unqualified')
  for index,trip in enumerate(d['trips']):
   inactive=trip['inactive'];background=trip['background'];active=trip['active'];returned=trip.get('returnInactive')
   if (inactive['origin']!='syntheticCheck' or inactive['observedUptimeNanoseconds']!=1 or active['observedUptimeNanoseconds']!=4
       or not inactive['index']<background['index']<active['index'] or (returned is not None)!=(index==1)):
    raise ValueError('Simulated round trip observation differs')
   if any(e['origin']!='syntheticCheck' for e in (inactive,background,active,*([returned] if returned is not None else []))):raise ValueError('Unit phases must remain synthetic')
   if returned is not None and not background['index']<returned['index']<active['index']:raise ValueError('Return inactive order differs')
 else:
  trace=result['trace'];trip=result['roundTrip'];ready=result['ready'];prefix=result['interruptedPrefix'];rebuilt=result['rebuiltIDs']
  if (result['passed'] is not True or result.get('error') is not None or result['cleanupErrors'] or result['endpointErrors'] or trace.get('error') is not None
      or trace.get('activeCommand') is not None or trace['detachedAfterClose'] is not True or d['trips']
      or len(result['resumedIDs'])!=3 or result['resumedIDs']!=result['referenceIDs'] or result['freshIDs']!=(rebuilt+result['referenceIDs'])[:3]
      or not 1<=len(rebuilt)<128 or prefix[-len(rebuilt):]!=rebuilt or not ready['eventIndex']<trip['inactive']['index']
      or not ready['observedUptimeNanoseconds']<trip['inactive']['observedUptimeNanoseconds'] or not trip['inactive']['before']['busy']):raise ValueError('Exact real-model synthetic replay differs')
  events=trace['events'];last=0;peak=0
  for i,event in enumerate(events):
   if type(event['index']) is not int or event['index']!=i or event['origin'] not in ('coordinator','syntheticCheck'):raise ValueError('Synthetic event provenance/order differs')
   observed=event['observedUptimeNanoseconds'];recorded=event['recordedUptimeNanoseconds']
   if type(observed) is not int or type(recorded) is not int or not 0<observed<=recorded or recorded<last:raise ValueError('Synthetic event clocks differ')
   last=recorded;f=event['footprint']
   if type(f['physicalBytes']) is not int or type(f['lifetimePeakPhysicalBytes']) is not int or not 0<f['physicalBytes']<=f['lifetimePeakPhysicalBytes'] or f['lifetimePeakPhysicalBytes']<peak:raise ValueError('Synthetic event process footprint differs')
   peak=f['lifetimePeakPhysicalBytes']
  if len(events)>64 or any(e['origin']!='syntheticCheck' for e in (trip['inactive'],trip['background'],trip['active'],trip['returnInactive'])):raise ValueError('Synthetic lifecycle scope differs')
 if type(result['loadedAdmittedBytes']) is not int or not 0<result['loadedAdmittedBytes']<=config['numericalBudget']:raise ValueError('Actual lifecycle admission differs')
 trace=result['trace'];events=trace['events']
 closed=events[-1]['state']
 if events[-1]['name']!='closed' or closed['closed'] is not True or closed['workerFinished'] is not True or closed['hasModel'] is not False or closed['hasSession'] is not False or closed.get('savedPrefix') is not None or any(closed[k]!=0 or type(closed[k]) is not int for k in ('history','evaluated','pending','savedHistory','reason','status')):raise ValueError('Raw completed release state differs')
 if kind=='small':
  expected=['loaded','generation-active','scene-inactive','scene-background','scene-inactive','scene-active','suspended','replayed','resumed','reference-reset','reference-prefix','reference','synthetic-memory-warning','warning-unloaded','replay-rejected','fresh-load','fresh-prompt','unloaded','closed']
  if [e['name'] for e in events]!=expected:raise ValueError('Exact synthetic coordinator sequence differs')
  for field,name in [('inactive','scene-inactive'),('background','scene-background'),('returnInactive','scene-inactive'),('active','scene-active')]:
   event=trip[field]
   if type(event['index']) is not int or not 0<=event['index']<len(events) or events[event['index']]!=event or event['name']!=name:raise ValueError('Round trip must match consumed trace')
  if not trip['inactive']['index']<trip['background']['index']<trip['returnInactive']['index']<trip['active']['index']:raise ValueError('Round trip indices differ')
  active=events[ready['eventIndex']]
  if active['name']!='generation-active' or active['command']!='generate' or ready['visibleIDs']!=active['before']['ids'] or ready.get('applicationState')!=active.get('applicationState') or ready['observedUptimeNanoseconds']!=active['recordedUptimeNanoseconds'] or not 0<ready['submittedUptimeNanoseconds']<=ready['observedUptimeNanoseconds']:raise ValueError('READY differs from trace')
  if not ready['visibleIDs'] or trip['inactive']['before']['ids'][:len(ready['visibleIDs'])]!=ready['visibleIDs'] or prefix!=config['prompt']+rebuilt:raise ValueError('Original visible/saved prefix differs')
  for event in events:
   if event['footprint']['lifetimePeakPhysicalBytes']>config['processBudget']:raise ValueError('Lifecycle trial process cap exceeded')
   for value in (event['before'],event['state']):
    if any(type(value[k]) is not bool for k in ('busy','closed','workerFinished','hasModel','hasSession')):raise ValueError('Typed lifecycle owner state required')
    for field in ('history','evaluated','pending','savedHistory','reason','status'):
     if type(value[field]) is not int or not 0<=value[field]<=config['capacity']:raise ValueError('Bounded lifecycle prefix metadata required')
    if value['history']!=value['evaluated']+value['pending'] or value['pending']>1:raise ValueError('Lifecycle committed prefix differs')
    for ids in (value.get('savedPrefix'),value['ids']):
     if ids is not None and (len(ids)>config['capacity'] or any(type(t) is not int or not 0<=t<config['vocabulary'] for t in ids)):raise ValueError('Lifecycle raw prefix IDs differ')
    if value.get('savedPrefix') is not None and len(value['savedPrefix'])!=value['savedHistory']:raise ValueError('Lifecycle saved count differs')
  length=len(prefix);prompt=config['prompt'];n=len(prompt)
  required={
   'loaded':([],0,0,0,0,0,True,True),
   'suspended':(prefix,0,0,0,0,0,True,False),
   'replayed':(prefix,length,length,0,1,0,True,True),
   'resumed':(prefix+result['resumedIDs'],length+3,length+2,1,1,0,True,True),
   'reference-reset':([],0,0,0,0,0,True,True),
   'reference-prefix':(prefix,length,length-1,1,1,0,True,True),
   'reference':(prefix+result['referenceIDs'],length+3,length+2,1,1,0,True,True),
   'warning-unloaded':([],0,0,0,0,0,False,False),
   'replay-rejected':([],0,0,0,0,1,False,False),
   'fresh-load':([],0,0,0,0,0,True,True),
   'fresh-prompt':(prompt+result['freshIDs'],n+3,n+2,1,1,0,True,True),
   'unloaded':([],0,0,0,0,0,False,False),
   'closed':(None,0,0,0,0,0,False,False)}
  for event in events:
   if event['name'] not in required:continue
   value=event['state'];saved,h,e,p,r,status,model,session=required[event['name']]
   if value.get('savedPrefix')!=saved or tuple(value[k] for k in ('history','evaluated','pending','reason','status'))!=(h,e,p,r,status) or value['hasModel'] is not model or value['hasSession'] is not session or value['busy'] is not False:raise ValueError('Exact lifecycle coordinator state differs')
 return d

def main(native,expected,output,small=False):
 native=native.resolve();output=output.resolve()
 if any(not p.is_relative_to(ROOT/'artifacts') for p in (native,output)):raise ValueError('Repository artifacts required')
 output.mkdir();logs=output/'logs';logs.mkdir();(output/'tmp').mkdir();pins={};commands=[];checked={};nested=None;primary=None;endpoints=[];jobs={}
 previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
 def merge(extra):integration.merge(pins,extra)
 def guard():measurement.verify_inputs(pins)
 def execute(args,label,timeout=180):
  guard();row=package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row)
  if row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:raise RuntimeError('Owned lifecycle app check failed: '+label)
  for field in ('stdout','stderr'):merge({row[field]:package.sha(ROOT/row[field])})
  guard();return ROOT/row['stdout']
 try:
  env,_=package.build_environment();env['TMPDIR']=str(output/'tmp');merge(integration.admit_build(native,expected));merge(integration.FIXTURE)
  sources=[ROOT/'apple/Harness'/n for n in ('NativeWorker.swift','NativeMeasurement.swift','NativeLifecycle.swift','NativeLifecycleSignals.swift','NativeLifecycleRun.swift')]+[ROOT/'apple/tests/swift/LifecycleAppMain.swift']
  helpers=[Path(v.__file__).resolve() for v in (core,measurement,package,integration)]+[Path(__file__).resolve(),Path(__file__).with_name('test_m7_native_lifecycle_app_check.py')]
  merge({str(p.relative_to(ROOT)):package.sha(p) for p in sources+helpers})
  for model,kind in [('tiny-cpu','unit'),('tiny-metal','unit')]+([('small','small')] if small else []):
   base,host,extra=measurement.job(model,'short' if kind=='unit' else '512');merge(extra)
   config={k:base[k] for k in ('version','modelFolder','packedFolder','files','metal','capacity','chunk','numericalBudget','processBudget','prompt')}
   config.update(maximum=25 if kind=='unit' else 128,vocabulary=64 if kind=='unit' else 151936)
   folder=output/model;folder.mkdir();root,name,digest=measurement.stage(config,host,folder);jobs[model]=dict(root=str(root.relative_to(ROOT)),name=name,sha256=digest,kind=kind,config=config)
   merge({str(p.relative_to(ROOT)):package.sha(p) for p in root.rglob('*') if p.is_file()})
  execute(['xcodebuild','-version'],'xcode-version');sdk=package.small(execute(['xcrun','--sdk','macosx','--show-sdk-path'],'sdk-path')).strip()
  if sdk!='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk':raise ValueError('Unadmitted Mac SDK')
  execute(['xcrun','--sdk','macosx','swiftc','--version'],'swift-version')
  binary=output/'lifecycle-app-checks';archive=native/'Toklet.xcframework/macos-arm64/libtoklet.a'
  execute(['xcrun','--sdk','macosx','swiftc','-sdk',sdk,'-module-cache-path',str(output/'ModuleCache'),'-swift-version','5','-strict-concurrency=complete','-warnings-as-errors',
    '-target','arm64-apple-macos15.0','-O','-I',str(ROOT/'apple/include'),*[str(p) for p in sources],str(archive),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security','-lobjc','-liconv','-lresolv','-lc++','-o',str(binary)],'compile')
  merge({str(binary.relative_to(ROOT)):package.sha(binary)})
  for model,job in jobs.items():
   raw=package.small(execute([str(binary),str(ROOT/job['root']),job['name'],job['sha256'],job['kind']],model,1200))
   checked[model]=validate_output(raw,'cpu' if model=='tiny-cpu' else 'metal',job['kind'],job['config'])
  nested=output/'core-regression';execute([sys.executable,str(Path(core.__file__).resolve()),str(native),expected,str(nested)],'core-regression',2400)
  path=nested/'checks.json'
  with path.open('rb') as f:raw=f.read(16*1024**2+1)
  if len(raw)>16*1024**2:raise ValueError('Oversized nested receipt')
  data=json.loads(raw)
  if not data['passed'] or data['primary_error'] or data['endpoint_errors']:raise ValueError('Lifecycle core regression failed')
  merge({str(path.relative_to(ROOT)):package.hashlib.sha256(raw).hexdigest()});merge(data['bindings'])
 except BaseException as e:primary=type(e).__name__+': '+str(e)
 finally:
  package.signal.signal(package.signal.SIGTERM,previous)
  try:guard()
  except BaseException as e:endpoints.append(type(e).__name__+': '+str(e))
 passed=primary is None and not endpoints and len(commands)==(8 if small else 7) and set(checked)==({'tiny-cpu','tiny-metal','small'} if small else {'tiny-cpu','tiny-metal'})
 result=dict(format='toklet-m7-lifecycle-app-check-v1',passed=passed,collection_root=str(ROOT),bindings=pins,commands=commands,jobs=jobs,checked=checked,nested_receipt=str((nested/'checks.json').relative_to(ROOT)) if nested else None,primary_error=primary,endpoint_errors=endpoints,real_model_checked=small,actual_os_lifecycle_qualified=False,physical_device_qualified=False,m7_complete=False)
 with (output/'checks.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)));return 0 if passed else 1
if __name__=='__main__':
 if len(sys.argv) not in (4,5) or len(sys.argv)==5 and sys.argv[4]!='--small':raise SystemExit('Expected native SHA output [--small]')
 sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),small=len(sys.argv)==5))
