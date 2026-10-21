"""Exact-prefix, prefill-cancellation and bounded-capacity native witnesses."""
from pathlib import Path
import json,math,sys
import m7_native_lifecycle_app_check as core
import m7_physical_measurement as measurement
package=core.package;integration=core.integration;ROOT=package.ROOT

def names(prefill):
 return (['loaded','primer','primer-score','fully-cached','cached-score','pending-primer','append-pending','append-reference-reset','append-reference','changed-prefix','changed-reference-reset','changed-reference']
  + (['prefill-reset','prefill-cancel-trigger','prefill-cancelled','prefill-completed','prefill-resumed','prefill-reference-reset','prefill-reference'] if prefill else [])
  + ['capacity-reset','capacity-committed']
  + [name+suffix for name in ('overflow-continue','overflow-append','overflow-prompt') for suffix in ('-before-score','','-after-score')]
  + ['usable-reset','usable-fresh-reference','closed'])

def validate_output(text,backend,kind,config):
 outer=json.loads(text);d=outer['diagnostic'];prefill=kind=='small';cap=config['capacity'];vocab=config['vocabulary'];prompt=config['prompt'];primer=prompt[:16];target=primer[-1]
 def need(ok,message):
  if not ok:raise ValueError(message)
 def uint(value,bits=32):return type(value) is int and 0<=value<2**bits
 def ids(value,limit):return type(value) is list and len(value)<=limit and all(uint(v) and v<vocab for v in value)
 need(d['passed'] is True and d['backend']==backend and d['prefillWitnessRequired'] is prefill and d['prefillWitness'] is prefill,'Explicit witness result differs')
 need(d.get('error') is None and d['cleanupErrors']==[] and d['endpointErrors']==[] and d['closed'] is True and d['released'] is True,'Successful cleanup and endpoints required')
 need(d['physicalDeviceQualified'] is False and d['m7Complete'] is False and d['filesBefore']==config['files'] and d['filesAfter']==config['files'],'Component scope/artifact proof differs')
 before=outer['errorBeforeClose'];after=outer['errorAfterClose']
 need(uint(before['status']) and before['busy'] is False and before['status']==1 and before['closed'] is False and before['hasModel'] is False and before['hasSession'] is False,'Direct failing command witness required')
 need(uint(after['status']) and after['busy'] is False and after['status']==0 and after['closed'] is True and after['workerFinished'] is True and after['hasModel'] is False and after['hasSession'] is False and after.get('savedPrefix') is None and after['ids']==[] and all(type(after[k]) is int and after[k]==0 for k in ('history','evaluated','pending','savedHistory','reason')) and type(outer['errorCloseAdmission']) is int and outer['errorCloseAdmission']==0 and type(outer['errorCloseCallbacks']) is int and outer['errorCloseCallbacks']==0 and outer['errorCloseMessage']=='','Error followed by successful zero close required')
 trace=d['trace'];need(trace['events']==[] and trace.get('error') is None and trace.get('activeCommand') is None and trace['detachedAfterClose'] is True,'Unexpected lifecycle observations')
 rows=d['rows'];need([r['name'] for r in rows]==names(prefill),'Exact state command sequence required')
 by={r['name']:r for r in rows};last=0;peaks=[]
 def footprint(f):
  need(all(uint(f[k],32 if k=='returnedNaturalCount' else 64) and f[k]>0 for k in ('physicalBytes','lifetimePeakPhysicalBytes','returnedNaturalCount')) and f['physicalBytes']<=f['lifetimePeakPhysicalBytes']<=config['processBudget'],'Bounded typed footprint required')
  return f['lifetimePeakPhysicalBytes']
 for r in rows:
  clock=r['completedUptimeNanoseconds'];need(uint(clock,64) and clock>last,'Monotonic command clock required');last=clock;peaks.append(footprint(r['footprint']))
  e=r['environment'];need(e.get('applicationState') is None or type(e['applicationState']) is int and e['applicationState']==0,'Foreground observation required')
  s=r['state'];need(all(type(s[k]) is bool for k in ('busy','closed','workerFinished','hasModel','hasSession')),'Typed owner state required')
  need(all(uint(s[k]) for k in ('status','history','evaluated','pending','reason','savedHistory')) and s['history']==s['evaluated']+s['pending'] and s['history']<=cap and s['savedHistory']<=cap,'Bounded exact committed counts required')
  need(ids(s['ids'],128) and uint(r['callbackCount']) and r['callbackCount']==len(s['ids']),'Exact callback count required')
  saved=s.get('savedPrefix');need(saved is None or ids(saved,cap) and len(saved)==s['savedHistory'],'Bounded exact saved prefix required')
  trigger=r['name']=='prefill-cancel-trigger';closed=r['name']=='closed'
  need(s['busy'] is trigger and s['closed'] is closed and s['workerFinished'] is closed and s['hasModel'] is not closed and s['hasSession'] is not closed,'Exact command owner state differs')
  need(uint(r['admittedBytes'],64) and (r['admittedBytes']==0 if closed else 0<r['admittedBytes']<=config['numericalBudget']),'Native admission differs')
  need(s['status']==(1 if r['name'] in ('overflow-continue','overflow-append','overflow-prompt') else 0),'Exact status differs')
  score=r.get('scalarScore')
  if 'score' in r['name']:
   need(type(score) is dict and all(uint(score[k]) for k in ('target','greedy','vocabulary','evaluated')) and score['target']==target and score['vocabulary']==vocab and score['evaluated']==s['evaluated'] and uint(score['greedy']) and score['greedy']<vocab,'Scalar diagnostic identity differs')
   need(all(type(score[k]) in (int,float) and math.isfinite(score[k]) for k in ('targetLogit','greedyLogit','negativeLogLikelihood')) and score['greedyLogit']>=score['targetLogit'] and score['negativeLogLikelihood']>=0,'Finite scalar diagnostic required')
  else:need(score is None,'Unexpected scalar result')
 need(peaks==sorted(peaks),'Lifetime peak chronology differs')
 checkpoints=d['checkpoints'];need([r['phase'] for r in checkpoints]==['before-artifact-proof','after-artifact-proof','loaded','closed','after-final-artifact-proof'],'Exact endpoint memory sequence required')
 p=[footprint(r['footprint']) for r in checkpoints];need(p==sorted(p) and p[1]<=peaks[0]<=p[2]<=peaks[1] and peaks[-1]<=p[3]<=p[4],'Endpoint observation chronology differs')
 def exact(name,tokens,pending,emitted,reason=1):
  s=by[name]['state'];need(s.get('savedPrefix')==tokens and s['savedHistory']==len(tokens) and s['history']==len(tokens) and s['evaluated']==len(tokens)-pending and s['pending']==pending and s['ids']==emitted and s['reason']==reason,'Exact state differs: '+name)
 exact('loaded',[],0,[],0)
 for name in ('primer','primer-score','fully-cached','cached-score'):exact(name,primer,0,[])
 need(by['primer-score']['scalarScore']==by['cached-score']['scalarScore'],'Fully cached scalar diagnostics differ')
 pending=by['pending-primer']['state']['ids'];need(len(pending)==3,'Three pending-primer IDs required');exact('pending-primer',primer+pending,1,pending)
 ap=primer+pending+prompt[:2];out=by['append-pending']['state']['ids'];need(len(out)==2,'Two appended IDs required')
 for n in ('append-pending','append-reference'):exact(n,ap+out,1,out)
 changed=primer.copy();changed[0]=1 if changed[0]==0 else 0;out=by['changed-prefix']['state']['ids'];need(len(out)==3,'Three changed-prefix IDs required')
 for n in ('changed-prefix','changed-reference','usable-fresh-reference'):exact(n,changed+out,1,out)
 for n in ('append-reference-reset','changed-reference-reset','capacity-reset','usable-reset')+ (('prefill-reset','prefill-reference-reset') if prefill else ()):exact(n,[],0,[],0)
 if prefill:
  need(len(prompt)==512 and cap==1152 and config['chunk']==128,'Bounded physical witness workload required')
  trigger=by['prefill-cancel-trigger']['state'];need(trigger['ids']==[] and by['prefill-cancel-trigger']['callbackCount']==0,'No prefill trigger emissions allowed')
  cancelled=by['prefill-cancelled']['state'];e=cancelled['evaluated'];need(0<e<len(prompt),'Actual committed partial prefill required');exact('prefill-cancelled',prompt,len(prompt)-e,[],3)
  exact('prefill-completed',prompt,0,[]);out=by['prefill-resumed']['state']['ids'];need(len(out)==3,'Three resumed IDs required')
  for n in ('prefill-resumed','prefill-reference'):exact(n,prompt+out,1,out)
 full=[prompt[i%len(prompt)] for i in range(cap)];exact('capacity-committed',full,0,[])
 for n in ('overflow-continue','overflow-append','overflow-prompt'):
  for suffix in ('-before-score','','-after-score'):exact(n+suffix,full,0,[],1 if n=='overflow-continue' and suffix=='-before-score' else 0)
  need(by[n+'-before-score']['scalarScore']==by[n+'-after-score']['scalarScore'],'Rejected request changed scalar diagnostic')
 s=by['closed']['state'];need(s.get('savedPrefix') is None and all(s[k]==0 for k in ('history','evaluated','pending','savedHistory','reason','status')) and s['ids']==[],'Completed close must zero state')
 return outer

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
  sources=[ROOT/'apple/Harness'/n for n in ('NativeWorker.swift','NativeMeasurement.swift','NativeLifecycle.swift','NativeLifecycleSignals.swift','NativeLifecycleRun.swift','NativeStateRun.swift')]+[ROOT/'apple/tests/swift/StateMain.swift']
  helpers=[Path(v.__file__).resolve() for v in (core,measurement,package,integration)]+[Path(__file__).resolve(),Path(__file__).with_name('test_m7_native_state_check.py')]
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
  binary=output/'state-checks';archive=native/'Toklet.xcframework/macos-arm64/libtoklet.a'
  execute(['xcrun','--sdk','macosx','swiftc','-sdk',sdk,'-module-cache-path',str(output/'ModuleCache'),'-swift-version','5','-strict-concurrency=complete','-warnings-as-errors',
    '-target','arm64-apple-macos15.0','-O','-I',str(ROOT/'apple/include'),*[str(p) for p in sources],str(archive),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security','-lobjc','-liconv','-lresolv','-lc++','-o',str(binary)],'compile')
  merge({str(binary.relative_to(ROOT)):package.sha(binary)})
  for model,job in jobs.items():
   raw=package.small(execute([str(binary),str(ROOT/job['root']),job['name'],job['sha256'],job['kind']],model,1200))
   checked[model]=validate_output(raw,'cpu' if model=='tiny-cpu' else 'metal',job['kind'],job['config'])
  nested=output/'core-regression';execute([sys.executable,str(Path(core.__file__).resolve()),str(native),expected,str(nested),'--small'] if small else [sys.executable,str(Path(core.__file__).resolve()),str(native),expected,str(nested)],'core-regression',3600)
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
 result=dict(format='toklet-m7-state-check-v1',passed=passed,collection_root=str(ROOT),bindings=pins,commands=commands,jobs=jobs,checked=checked,nested_receipt=str((nested/'checks.json').relative_to(ROOT)) if nested else None,primary_error=primary,endpoint_errors=endpoints,real_model_checked=small,actual_os_lifecycle_qualified=False,physical_device_qualified=False,m7_complete=False)
 with (output/'checks.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)));return 0 if passed else 1
if __name__=='__main__':
 if len(sys.argv) not in (4,5) or len(sys.argv)==5 and sys.argv[4]!='--small':raise SystemExit('Expected native SHA output [--small]')
 sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),small=len(sys.argv)==5))
