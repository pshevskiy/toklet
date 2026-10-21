"""Sealed physical lifecycle jobs and same-process OS round trips."""
from pathlib import Path
import json,sys,time
import m7_physical_measurement as m
ROOT=m.ROOT
NATIVE_SHA='06016a4e6eeea98c9c9c0f898cc4fd107c85415ed26cd3faa8de221386002971'
APP_PROOF='benchmarks/results/2026-10-21-m7-lifecycle-app-qualification-v1.json'
APP_PROOF_SHA='40997d6fa69742477682db5fce105a37da3dc4863b8ccfafae33033b7c20ddcf'
READY_FORMAT='toklet-m7-physical-lifecycle-ready-v1'
RESULT_FORMAT='toklet-m7-physical-lifecycle-v1'

def job(model,workload='512',verify_sources=True):
 if model not in ('small','larger') or workload!='512':raise ValueError('Explicit real-model lifecycle512 job required')
 base,host,pins=m.job(model,workload,verify_sources=verify_sources)
 proof=m.integration.metadata(ROOT/APP_PROOF,APP_PROOF_SHA)
 if proof['passed'] is not True or proof['app_integration'] is not True or proof['review']!='Astra accepted source and completed app component evidence with no findings' or proof['actual_os_lifecycle_qualified'] is not False:raise ValueError('Accepted lifecycle app integration required')
 m.verify_inputs(proof['code_pins']);m.integration.merge(pins,proof['code_pins']);pins[APP_PROOF]=APP_PROOF_SHA
 config={k:base[k] for k in ('version','modelFolder','packedFolder','files','metal','capacity','chunk','numericalBudget','processBudget','prompt','maximum')};config['vocabulary']=151936
 return config,host,pins

def stage(config,host,output,model,workload):
 expected,_,_=job(model,workload,verify_sources=False)
 if config!=expected:raise ValueError('Frozen lifecycle config differs')
 return m.stage(config,host,output)

def validate_console(text,config,signed):
 markers={'TOKLET_INSTALLATION_JSON=':'proof','TOKLET_LIFECYCLE_READY_JSON=':'ready','TOKLET_LIFECYCLE_JSON=':'result'};rows=[]
 for line in text.splitlines():
  for prefix,kind in markers.items():
   if line.startswith(prefix):rows.append((kind,json.loads(line[len(prefix):])));break
  else:
   if line.startswith('TOKLET_'):raise ValueError('Unexpected lifecycle marker')
 if [k for k,_ in rows]!=['proof','ready','proof','result']:raise ValueError('Exact installed/READY/lifecycle stream required')
 proofs=[rows[0][1],rows[2][1]];result=rows[3][1];ready=rows[1][1]
 if any(p['simulator'] is not False or p['files']!=signed for p in proofs) or [p['phase'] for p in proofs]!=['before','after']:raise ValueError('Installed lifecycle bundle proof differs')
 validate_result(result,config)
 if ready!=result['ready']:raise ValueError('Consumed READY differs from final result')
 return dict(result=result,ready=ready,installed_proofs=proofs)

def validate_result(r,c):
 if (r['passed'] is not True or r.get('error') is not None or r['cleanupErrors'] or r['endpointErrors']
     or r['closed'] is not True or r['released'] is not True or any(r[k] is not False for k in ('actualOSLifecycleQualified','physicalDeviceQualified','m7Complete'))
     or r['backend']!='metal' or r['filesBefore']!=c['files'] or r['filesAfter']!=c['files']
     or type(r['loadedAdmittedBytes']) is not int or not 0<r['loadedAdmittedBytes']<=c['numericalBudget']):raise ValueError('Complete closed lifecycle outcome required')
 for environment in (r['environmentBefore'],r['environmentAfter']):
  if not m.uint(environment['applicationState'],32) or environment['applicationState']!=0 or type(environment['lowPowerMode']) is not bool or not m.uint(environment['thermalState'],32) or environment['thermalState']>3:raise ValueError('Foreground lifecycle endpoints required')
 trace=r['trace'];events=trace['events'];trip=r['roundTrip'];ready=r['ready'];prefix=r['interruptedPrefix'];rebuilt=r['rebuiltIDs'];returned=trip.get('returnInactive')
 if trace.get('error') is not None or trace.get('activeCommand') is not None or trace['detachedAfterClose'] is not True:raise ValueError('Completed lifecycle trace required')
 names=['loaded','generation-active','scene-inactive','scene-background']+(['scene-inactive'] if returned is not None else [])+['scene-active','suspended','replayed','resumed','reference-reset','reference-prefix','reference','synthetic-memory-warning','warning-unloaded','replay-rejected','fresh-load','fresh-prompt','unloaded','closed']
 if [e['name'] for e in events]!=names:raise ValueError('Exact physical lifecycle event sequence required')
 def footprint(value):
  if any(not m.uint(value[k]) or value[k]==0 for k in ('physicalBytes','lifetimePeakPhysicalBytes','returnedNaturalCount')) or not value['physicalBytes']<=value['lifetimePeakPhysicalBytes']<=c['processBudget']:raise ValueError('Bounded typed lifecycle footprint required')
  return value['lifetimePeakPhysicalBytes']
 peaks=[];last=0
 for i,event in enumerate(events):
  if not m.uint(event['index'],32) or event['index']!=i:raise ValueError('Indexed lifecycle trace required')
  observed=event['observedUptimeNanoseconds'];recorded=event['recordedUptimeNanoseconds']
  if not m.uint(observed) or not m.uint(recorded) or not 0<observed<=recorded or recorded<last:raise ValueError('Ordered callback/record clocks required')
  last=recorded;peaks.append(footprint(event['footprint']))
  if not m.uint(event['applicationState'],32) or event['applicationState']>2:raise ValueError('Typed physical application state required')
  expected='scenePhase' if event['name'].startswith('scene-') else 'syntheticCheck' if event['name']=='synthetic-memory-warning' else 'coordinator'
  if event['origin']!=expected:raise ValueError('Actual OS/synthetic provenance differs')
  for state in (event['before'],event['state']):
   if any(type(state[k]) is not bool for k in ('busy','closed','workerFinished','hasModel','hasSession')):raise ValueError('Typed lifecycle owner state required')
   if any(not m.uint(state[k],32) or state[k]>c['capacity'] for k in ('history','evaluated','pending','savedHistory','reason','status')) or state['history']!=state['evaluated']+state['pending'] or state['pending']>1:raise ValueError('Exact lifecycle committed state required')
   for ids in (state['ids'],state.get('savedPrefix')):
    if ids is not None and (type(ids) is not list or len(ids)>c['capacity'] or any(not m.uint(t,32) or t>=c['vocabulary'] for t in ids)):raise ValueError('Bounded lifecycle IDs required')
   if state.get('savedPrefix') is not None and len(state['savedPrefix'])!=state['savedHistory']:raise ValueError('Saved lifecycle prefix count differs')
 checkpoints=r['checkpoints']
 if [v['phase'] for v in checkpoints]!=['before-artifact-proof','after-artifact-proof','loaded','closed','after-final-artifact-proof']:raise ValueError('Exact lifecycle checkpoint sequence required')
 cp=[footprint(v['footprint']) for v in checkpoints]
 if cp!=sorted(cp) or peaks!=sorted(peaks) or not cp[1]<=peaks[0]<=cp[2]<=peaks[1]<=peaks[-1]<=cp[3]<=cp[4]:raise ValueError('Lifecycle memory chronology differs')
 fields=[('inactive','scene-inactive'),('background','scene-background')]+([('returnInactive','scene-inactive')] if returned is not None else [])+[('active','scene-active')]
 indices=[];observed=[]
 for field,name in fields:
  event=trip[field];i=event['index']
  if not m.uint(i,32) or i>=len(events) or events[i]!=event or event['name']!=name or event['origin']!='scenePhase':raise ValueError('Consumed actual scene phases required')
  indices.append(i);observed.append(event['observedUptimeNanoseconds'])
 if indices!=sorted(set(indices)) or observed!=sorted(observed) or trip['inactive']['applicationState'] not in (1,2) or trip['background']['applicationState']!=2 or trip['active']['applicationState']!=0:raise ValueError('Ordered OS scene phases required')
 if returned is not None and returned['applicationState'] not in (1,2):raise ValueError('Return inactive application state differs')
 if any(not m.uint(ready[k]) for k in ('eventIndex','submittedUptimeNanoseconds','observedUptimeNanoseconds')) or not ready['eventIndex']<trip['inactive']['index']:raise ValueError('Typed READY boundary required')
 active=events[ready['eventIndex']];initial=trip['inactive'];maximum=c['maximum']
 if (active['name']!='generation-active' or active['command']!='generate' or active['before']['busy'] is not True or active['applicationState']!=0
     or ready['applicationState']!=0 or ready['visibleIDs']!=active['before']['ids'] or not 0<len(ready['visibleIDs'])<maximum
     or ready['observedUptimeNanoseconds']!=active['recordedUptimeNanoseconds'] or not 0<ready['submittedUptimeNanoseconds']<=ready['observedUptimeNanoseconds']<initial['observedUptimeNanoseconds']
     or initial['command']!='generate' or initial['before']['busy'] is not True or not 0<len(initial['before']['ids'])<maximum
     or initial['before']['ids'][:len(ready['visibleIDs'])]!=ready['visibleIDs']):raise ValueError('Active foreground generation must precede actual interruption')
 if (not 0<len(rebuilt)<maximum or prefix!=c['prompt']+rebuilt or len(r['resumedIDs'])!=3 or r['resumedIDs']!=r['referenceIDs'] or r['freshIDs']!=(rebuilt+r['referenceIDs'])[:3]):raise ValueError('Original prefix/replay/reference/fresh oracle differs')
 n=len(c['prompt']);length=len(prefix)
 required={'loaded':([],0,0,0,0,0,True,True),'suspended':(prefix,0,0,0,0,0,True,False),'replayed':(prefix,length,length,0,1,0,True,True),
 'resumed':(prefix+r['resumedIDs'],length+3,length+2,1,1,0,True,True),'reference-reset':([],0,0,0,0,0,True,True),
 'reference-prefix':(prefix,length,length-1,1,1,0,True,True),'reference':(prefix+r['referenceIDs'],length+3,length+2,1,1,0,True,True),
 'warning-unloaded':([],0,0,0,0,0,False,False),'replay-rejected':([],0,0,0,0,1,False,False),'fresh-load':([],0,0,0,0,0,True,True),
 'fresh-prompt':(c['prompt']+r['freshIDs'],n+3,n+2,1,1,0,True,True),'unloaded':([],0,0,0,0,0,False,False),'closed':(None,0,0,0,0,0,False,False)}
 for event in events:
  if event['name'] not in required:continue
  state=event['state'];saved,h,e,p,reason,status,model,session=required[event['name']]
  if state.get('savedPrefix')!=saved or tuple(state[k] for k in ('history','evaluated','pending','reason','status'))!=(h,e,p,reason,status) or state['hasModel'] is not model or state['hasSession'] is not session or state['busy'] is not False:raise ValueError('Exact settled lifecycle state differs')
 closed=events[-1]['state']
 if closed['closed'] is not True or closed['workerFinished'] is not True:raise ValueError('Completed native close required')
 return r

def read_json(path,pins,limit=64*1024**2):
 with path.open('rb') as f:raw=f.read(limit+1)
 if len(raw)>limit:raise ValueError('Bounded lifecycle metadata exceeded')
 m.integration.merge(pins,{str(path.relative_to(ROOT)):m.package.hashlib.sha256(raw).hexdigest()})
 return json.loads(raw)

def peek_ready(path):
 if not path.exists():return None
 with path.open('rb') as f:raw=f.read(64*1024**2+1)
 if len(raw)>64*1024**2:raise ValueError('Bounded live lifecycle console exceeded')
 # A partial final line is not a complete observation.
 lines=raw.split(b'\n')[:-1];rows=[]
 for line in lines:
  if line.startswith(b'TOKLET_LIFECYCLE_READY_JSON='):rows.append(json.loads(line.split(b'=',1)[1]))
  elif line.startswith(b'TOKLET_LIFECYCLE_ERROR='):raise ValueError('App reported lifecycle failure before READY')
 if len(rows)>1:raise ValueError('Exactly one lifecycle READY required')
 return rows[0] if rows else None

def activation(data,selected,remote=None,pid=None):
 result=data['result'];process=result['process'];actual=process['processIdentifier'];path=m.physical.remote_path(process['executable'])
 if (data['info']['outcome']!='success' or data['info']['commandType']!='devicectl.device.process.launch' or result['deviceIdentifier']!=selected['device_identifier']
     or not m.uint(actual,32) or actual==0 or remote is not None and path!=remote+'/TokletHarness'
     or remote is None and not path.endswith('/Preferences.app/Preferences') or pid is not None and actual!=pid):raise ValueError('Exact activated process identity differs')
 return actual

def collect(native,private,private_sha,prepared,prepared_sha,ready,ready_sha,model,output):
 import plistlib,subprocess
 native=native.resolve();private=private.resolve();prepared=prepared.resolve();ready=ready.resolve();output=output.resolve()
 if any(not p.is_relative_to(ROOT/'artifacts') for p in (native,private,prepared,ready,output)):raise ValueError('Repository artifact paths required')
 output.mkdir();logs=output/'private-logs';logs.mkdir();env,_=m.package.build_environment();env['TMPDIR']=str(output)
 pins={};commands=[];selected={};remote=None;app=None;signed={};config={};profile=None;device=None;checked=None;primary=None;endpoints=[];cleanup_errors=[];cleanups=[];phase=[];process=None;live_row=None;out_handle=None;err_handle=None;launch_attempted=False;live_pid=None;ready_value=None;storage_admission=None
 previous=m.package.signal.signal(m.package.signal.SIGTERM,m.package.interrupted)
 def guard():m.verify_inputs(pins)
 def pin_log(path):
  with path.open('rb') as f:raw=f.read(64*1024**2+1)
  if len(raw)>64*1024**2:raise ValueError('Bounded lifecycle log exceeded')
  m.integration.merge(pins,{str(path.relative_to(ROOT)):m.package.hashlib.sha256(raw).hexdigest()});return raw
 def execute(args,label,timeout=90,full_guard=True):
  if full_guard:guard()
  row=m.package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row)
  raw=pin_log(ROOT/row['stdout']);pin_log(ROOT/row['stderr'])
  if row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:raise ValueError('Owned lifecycle command failed: '+str(row['primary_error']))
  if full_guard:guard()
  return raw
 def command(parts,label,full_guard=True):
  return json.loads(execute([m.physical.DEVICECTL,'device',*parts,'--device',selected['device_identifier'],'--timeout','90','--quiet','--json-output','-'],label,full_guard=full_guard))
 def current(label,full_guard=True):return command(['info','apps','--bundle-id',m.physical.BUNDLE],label,full_guard)
 def query(label):return command(['info','processes','--search','TokletHarness'],label,False)
 def same_pid(label):
  data=query(label);pids=m.physical.owned_processes(data,remote)
  if data['info']['outcome']!='success' or data['result']['deviceIdentifier']!=selected['device_identifier'] or len(pids)!=1 or live_pid is not None and pids!=[live_pid]:raise ValueError('Same live lifecycle PID required')
  phase.append(dict(phase=label,pid=pids[0],observed_host_monotonic_nanoseconds=time.monotonic_ns()));return pids[0]
 def cleanup():
  counter=0
  def signal(pid,sig):
   nonlocal counter
   counter+=1;command(['process','signal','--pid',str(pid),'--signal',sig],'cleanup-signal-'+str(counter),False)
  counter_query=0
  def query_cleanup():
   nonlocal counter_query
   counter_query+=1;return query('cleanup-process-'+str(counter_query))
  result=m.physical.stop_remote(query_cleanup,signal,remote);cleanups.append(result);cleanup_errors.extend(result['errors'])
  if not result['absent']:cleanup_errors.append('Lifecycle phone process absence unverified')
 try:
  selected=m.physical.selection(private,private_sha)
  m.integration.merge(pins,m.integration.admit_build(native,NATIVE_SHA));app,extra,remote,signed=m.physical.admit_reuse(prepared,prepared_sha,selected,NATIVE_SHA);m.integration.merge(pins,extra)
  for module in (m,m.package,m.integration,m.physical,m.storage):
   path=Path(module.__file__).resolve();m.integration.merge(pins,{str(path.relative_to(ROOT)):m.package.sha(path)})
  for name in ('test_m7_physical_measurement.py','test_m7_physical_storage.py','m7_physical_lifecycle.py','test_m7_physical_lifecycle.py'):
   path=ROOT/'tools/experiments'/name;m.integration.merge(pins,{str(path.relative_to(ROOT)):m.package.sha(path)})
  m.integration.merge(pins,{str(private.relative_to(ROOT)):private_sha});common=dict(pins)
  # Preparation's generic storage context excludes diagnostic-specific files.
  for name in ('m7_physical_lifecycle.py','test_m7_physical_lifecycle.py'):common.pop('tools/experiments/'+name)
  config,_,inputs=job(model,'512');m.integration.merge(pins,inputs)
  extra,name,digest=m.admit_ready(ready,ready_sha,config,inputs,prepared,prepared_sha,signed,READY_FORMAT);m.integration.merge(pins,extra)
  record=m.integration.metadata(ready/'measurement.json',ready_sha)
  tiny,_,tiny_inputs=m.job('tiny-cpu','short',verify_sources=False);m.integration.merge(common,tiny_inputs)
  context=dict(tiny_config=tiny,prepared_receipt=dict(path=str((prepared/'integration.json').relative_to(ROOT)),sha256=prepared_sha),signed_bundle=signed,selection_path=str(private.relative_to(ROOT)),selection_sha=private_sha,selected=selected,remote=remote,required_bindings=common)
  storage_admission,extra=m.storage.admit_ready(record,ready/'transfer',config,name,digest,context,m.validate_console);m.integration.merge(pins,extra)
  profile=m.physical.admit_profile(plistlib.loads(pin_log(ready/'private-logs/profile.stdout')),plistlib.loads(pin_log(ready/'private-logs/entitlements.stdout')),pin_log(ready/'private-logs/signature-details.stderr').decode(),selected)
  device=m.physical.admit_device(command(['info','details'],'device-before'),selected)
  m.physical.launch_verified(lambda:current('current-install-before'),lambda:True,remote,selected)
  preflight=query('preflight-process');pids=m.physical.owned_processes(preflight,remote)
  if preflight['info']['outcome']!='success' or preflight['result']['deviceIdentifier']!=selected['device_identifier'] or pids:raise ValueError('Natural preflight absence required; no reset during lifecycle admission')
  cleanups.append(dict(absent=True,observations=[[]],signals=[],errors=[]));guard()
  metadata=logs/'launch.json';args=[m.physical.DEVICECTL,'device','process','launch','--device',selected['device_identifier'],'--console','--timeout','900','--json-output',str(metadata),m.physical.BUNDLE,'--','--lifecycle',name,digest]
  # No --terminate-existing on the original launch or foreground activation.
  out_handle=(logs/'lifecycle.stdout').open('xb');err_handle=(logs/'lifecycle.stderr').open('xb')
  live_row=dict(command=args,exit_code=None,primary_error=None,cleanup_errors=[],pid=None,stdout=str((logs/'lifecycle.stdout').relative_to(ROOT)),stderr=str((logs/'lifecycle.stderr').relative_to(ROOT)));commands.append(live_row)
  process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out_handle,stderr=err_handle,start_new_session=True);live_row['pid']=process.pid;launch_attempted=True
  deadline=time.monotonic()+300
  while ready_value is None:
   ready_value=peek_ready(logs/'lifecycle.stdout')
   if ready_value is not None:break
   if process.poll() is not None:raise ValueError('Lifecycle child exited before active READY')
   if time.monotonic()>=deadline:raise TimeoutError('Active READY deadline')
   time.sleep(.02)
  phase.append(dict(phase='ready-consumed',observed_host_monotonic_nanoseconds=time.monotonic_ns(),ready=ready_value))
  # The app keeps generation running for at most128 IDs. Do no multi-GiB input
  # hashes between READY and system activation; full guards bracket the round trip.
  live_pid=same_pid('ready-process')
  activation(command(['process','launch','--activate','com.apple.Preferences'],'activate-settings',False),selected)
  phase.append(dict(phase='settings-activated',observed_host_monotonic_nanoseconds=time.monotonic_ns()))
  time.sleep(2)
  same_pid('background-process')
  m.physical.launch_verified(lambda:current('current-install-return',False),lambda:True,remote,selected)
  activation(command(['process','launch','--activate',m.physical.BUNDLE],'activate-harness',False),selected,remote,live_pid)
  phase.append(dict(phase='harness-activated',pid=live_pid,observed_host_monotonic_nanoseconds=time.monotonic_ns()))
  # A successful launch can run the remainder quickly; the activation metadata
  # itself proves the original PID, so no forced relaunch is used to inspect it.
  live_row['exit_code']=process.wait(timeout=600)
  if live_row['exit_code']!=0:live_row['primary_error']='tool exit '+str(live_row['exit_code']);raise ValueError('Lifecycle console child failed')
  guard();raw=pin_log(logs/'lifecycle.stdout').decode();err=pin_log(logs/'lifecycle.stderr').decode()
  checked=validate_console(raw+'\n'+err,config,signed);checked['launch']=m.physical.check_launch(read_json(metadata,pins),remote,selected)
  if checked['ready']!=ready_value or checked['launch']['pid']!=live_pid:raise ValueError('Final original process/READY identity differs')
 except BaseException as error:primary=type(error).__name__+': '+str(error)
 finally:
  if remote and launch_attempted:
   try:cleanup()
   except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
  if process is not None:
   try:m.package.drain_group(process,live_row['cleanup_errors'],15);live_row['exit_code']=process.wait(timeout=15)
   except BaseException as error:live_row['cleanup_errors'].append(type(error).__name__+': '+str(error))
   if live_row['exit_code']!=0 and live_row['primary_error'] is None:live_row['primary_error']='tool exit '+str(live_row['exit_code'])
  for handle in (out_handle,err_handle):
   if handle is not None:handle.close()
  for path in sorted(logs.iterdir()):
   if path.is_file():
    try:pin_log(path)
    except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
  if device is not None:
   try:
    if m.physical.admit_device(command(['info','details'],'device-after',False),selected)!=device:raise ValueError('Device endpoint differs')
   except BaseException as error:endpoints.append('Device endpoint: '+str(error))
  try:guard()
  except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
  if app is not None:
   try:
    if m.physical.inventory(app)!=signed:raise ValueError('Signed app endpoint differs')
   except BaseException as error:endpoints.append(str(error))
  m.package.signal.signal(m.package.signal.SIGTERM,previous)
 private_record=dict(commands=commands,primary_error=primary,endpoint_errors=endpoints,cleanup_errors=cleanup_errors)
 private_path=output/'private-execution.json';private_raw=(json.dumps(private_record,indent=2)+'\n').encode()
 with private_path.open('xb') as f:f.write(private_raw)
 m.integration.merge(pins,{str(private_path.relative_to(ROOT)):m.package.hashlib.sha256(private_raw).hexdigest()})
 natural=dict(absent=True,observations=[[]],signals=[],errors=[])
 passed=primary is None and not endpoints and not cleanup_errors and checked is not None and cleanups==[natural,natural] and all(c['exit_code']==0 and not c['primary_error'] and not c['cleanup_errors'] for c in commands)
 record=dict(format=RESULT_FORMAT,passed=passed,bindings=pins,config=config,device=device,signing=profile,signed_bundle=signed,prepared_receipt=dict(path=str((prepared/'integration.json').relative_to(ROOT)),sha256=prepared_sha),transfer_receipt=dict(path=str((ready/'measurement.json').relative_to(ROOT)),sha256=ready_sha),storage_admission=storage_admission,checked=checked,phase_observations=phase,commands=commands,remote_cleanup=cleanups,primary_error=primary,endpoint_errors=endpoints,cleanup_errors=cleanup_errors,actual_os_lifecycle_qualified=False,physical_device_qualified=False,m7_complete=False,scope='Raw actual OS lifecycle observation; independent analysis required; synthetic warning remains labelled')
 record=m.physical.redact_record(record,selected,remote)
 with (output/'lifecycle.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=passed,primary_error=record['primary_error'],endpoint_errors=record['endpoint_errors'],cleanup_errors=record['cleanup_errors'])));return 0 if passed else 1

if __name__=='__main__':
 if len(sys.argv)!=10:raise SystemExit('Expected native private privateSHA prepared preparedSHA ready readySHA model output')
 sys.exit(collect(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),sys.argv[5],Path(sys.argv[6]),sys.argv[7],sys.argv[8],Path(sys.argv[9])))
