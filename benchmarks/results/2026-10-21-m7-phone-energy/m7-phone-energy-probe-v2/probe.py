"""Bounded target-PID Power Profiler attempt; raw trace remains private."""
from pathlib import Path
import concurrent.futures,json,sys,time,threading
R=Path.cwd();sys.path.insert(0,str(R/'tools/experiments'))
import m7_physical_measurement as m
import m7_physical_state as state
OWN=Path(__file__).resolve();OUT=OWN.parent;PREP=R/'artifacts/m7-physical-prepared-v16';PREP_SHA='e1989a8d078a3660a9c516d7257b93147eca63640eb0d9feda1d63ea0bb4398e'
PRIVATE=R/'artifacts/m7-physical-admission-v1/private-selection.json';PRIVATE_SHA='711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db'
READY=R/'artifacts/m7-phone-state-small-ready-v1';READY_SHA='4065c91c7c3021cf370636efcdbb2a6cb03540a517853aaa8bf7f07dc03df007'
NATIVE=R/'artifacts/m7-apple-package-v5';NATIVE_SHA=state.NATIVE_SHA
RUN=R/'artifacts/m7-phone-energy-state-small-v2';TRACE=OUT/'private-trace.trace'
def target_pid(data,selected,remote):
 if data['info']['outcome']!='success' or data['info']['commandType']!='devicectl.device.info.processes' or data['result']['deviceIdentifier']!=selected['device_identifier']:raise ValueError('Exact target process query required')
 pids=m.physical.owned_processes(data,remote)
 if len(pids)!=1:raise ValueError('One exact installed target process required')
 return pids[0]
def marker(path,signed):
 if not path.exists():return False
 with path.open('rb') as f:raw=f.read(64*1024**2+1)
 if len(raw)>64*1024**2:raise ValueError('Bounded live console required')
 rows=[json.loads(v.split(b'=',1)[1]) for v in raw.split(b'\n')[:-1] if v.startswith(b'TOKLET_INSTALLATION_JSON=')]
 if not rows:return False
 if rows[0]!=dict(phase='before',simulator=False,files=signed):raise ValueError('Actual target installed-bundle marker required')
 return True
def trace_args(udid,pid):
 if type(pid) is not int or pid<=0:raise ValueError('Positive target PID required')
 # PID is freshly verified above; the name selector is the documented fallback.
 return ['xcrun','xctrace','record','--template','Power Profiler','--device',udid,'--attach','TokletHarness','--time-limit','10s','--output',str(TRACE),'--no-prompt']
def bind_child(receipt,pins):
 with receipt.open('rb') as f:raw=f.read(64*1024**2+1)
 if len(raw)>64*1024**2:raise ValueError('Bounded native receipt required')
 r=json.loads(raw);h=m.package.hashlib.sha256(raw).hexdigest()
 m.integration.merge(pins,r['bindings']);m.integration.merge(pins,r['private_evidence_pins']);m.integration.merge(pins,{str(receipt.relative_to(R)):h})
 return r,h

def main():
 for p in (OUT/'attempt.json',OUT/'private-execution.json',OUT/'private-trace-inventory.json',OUT/'private-logs',RUN,TRACE):
  if p.exists() or p.is_symlink():raise FileExistsError('Fresh exclusive instrumented output required')
 logs=OUT/'private-logs';logs.mkdir();pins={str(OWN.relative_to(R)):m.package.sha(OWN),str(OWN.with_name('checks.py').relative_to(R)):m.package.sha(OWN.with_name('checks.py'))};commands=[];primary=[];endpoints=[];probe=None;child=None;selected={};remote=None;cleanup=None;future=None;launched=False;stop=threading.Event();trace_inventory={};child_evidence=None
 env,_=m.package.build_environment();env['TMPDIR']=str(OUT)
 def execute(args,label,timeout=90):
  row=m.package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row);return row
 def query(parts,label):
  row=execute([m.physical.DEVICECTL,'device',*parts,'--device',selected['device_identifier'],'--timeout','60','--quiet','--json-output','-'],label,80)
  if row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:raise ValueError('Owned instrumentation identity query failed')
  data=json.loads((R/row['stdout']).read_text())
  if data['info']['outcome']!='success' or data['result']['deviceIdentifier']!=selected['device_identifier']:raise ValueError('Instrumentation query device/outcome differs')
  return data
 def sidecar(signed):
  deadline=time.monotonic()+180
  while not marker(RUN/'private-logs/measurement.stdout',signed):
   if stop.is_set():raise InterruptedError('Instrumentation admission cancelled')
   if (RUN/'measurement.json').exists():raise ValueError('Native child completed before instrumentation admission')
   if time.monotonic()>=deadline:raise TimeoutError('Bounded instrumentation launch admission')
   time.sleep(.05)
  m.physical.launch_verified(lambda:query(['info','apps','--bundle-id',m.physical.BUNDLE],'target-install'),lambda:True,remote,selected)
  pid=target_pid(query(['info','processes','--search','TokletHarness'],'target-process'),selected,remote)
  record=execute(trace_args(selected['device_udid'],pid),'power-profiler',60)
  return dict(target_pid=pid,recording_command=record,trace_exists=TRACE.exists(),recording_completed=record['exit_code']==0 and not record['primary_error'] and not record['cleanup_errors'])
 previous=m.package.signal.signal(m.package.signal.SIGTERM,m.package.interrupted)
 try:
  selected=m.physical.selection(PRIVATE,PRIVATE_SHA);_,extra,remote,signed=m.physical.admit_reuse(PREP,PREP_SHA,selected,NATIVE_SHA);m.integration.merge(pins,extra)
  config,_,inputs=state.job('small','512',verify_sources=False);extra,_,_=m.admit_ready(READY,READY_SHA,config,inputs,PREP,PREP_SHA,signed,state.READY_FORMAT);m.integration.merge(pins,extra);m.verify_inputs(pins)
  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
   future=pool.submit(sidecar,signed)
   try:
    launched=True;child=execute([sys.executable,str(OWN),'--child'],'native-state-child',360)
   finally:
    stop.set()
    try:probe=future.result(timeout=260)
    except BaseException as e:primary.append('Sidecar: '+type(e).__name__+': '+str(e))
  m.verify_inputs(pins)
  if child['exit_code']!=0 or child['primary_error'] or child['cleanup_errors']:primary.append('Native child: '+str(child['primary_error'])+' '+repr(child['cleanup_errors']))
 except BaseException as e:primary.append('Parent: '+type(e).__name__+': '+str(e))
 finally:
  # ThreadPool exits only after the bounded sidecar's owned command is drained.
  # Recheck/clean the admitted phone process independently of child cleanup.
  stop.set()
  if launched and remote is not None:
   try:
    m.physical.launch_verified(lambda:query(['info','apps','--bundle-id',m.physical.BUNDLE],'cleanup-install'),lambda:True,remote,selected)
    counter=0
    def clean_query():
     nonlocal counter
     counter+=1;return query(['info','processes','--search','TokletHarness'],'cleanup-process-'+str(counter))
    def clean_signal(pid,sig):query(['process','signal','--pid',str(pid),'--signal',sig],'cleanup-signal-'+str(pid)+'-'+sig)
    cleanup=m.physical.stop_remote(clean_query,clean_signal,remote)
    if not cleanup['absent'] or cleanup['errors']:endpoints.append('Independent phone cleanup did not verify absence: '+repr(cleanup['errors']))
   except BaseException as e:endpoints.append('Independent phone cleanup: '+type(e).__name__+': '+str(e))
  m.package.signal.signal(m.package.signal.SIGTERM,previous)
  try:
   receipt=RUN/'measurement.json'
   if receipt.exists():
    r,h=bind_child(receipt,pins);child_evidence=dict(path=str(receipt.relative_to(R)),sha256=h,passed=r['passed'])
    if r['passed'] is not True:primary.append('Native receipt failed')
   elif launched:primary.append('Native receipt unavailable')
  except BaseException as e:endpoints.append('Native evidence: '+type(e).__name__+': '+str(e))
  try:
   count=0;total=0
   if TRACE.exists():
    for p in sorted(TRACE.rglob('*')):
     if p.is_symlink():raise ValueError('Trace symlinks are not admitted')
     if not p.is_file():continue
     count+=1;total+=p.stat().st_size
     if count>10000 or total>1024**3:raise ValueError('Bounded private trace inventory exceeded')
     trace_inventory[str(p.relative_to(R))]=m.package.sha(p)
   raw=(json.dumps(trace_inventory,indent=2)+'\n').encode();p=OUT/'private-trace-inventory.json'
   with p.open('xb') as f:f.write(raw)
   pins[str(p.relative_to(R))]=m.package.hashlib.sha256(raw).hexdigest()
  except BaseException as e:endpoints.append('Trace evidence: '+type(e).__name__+': '+str(e))
  try:m.verify_inputs(pins);m.verify_inputs(trace_inventory)
  except BaseException as e:endpoints.append(type(e).__name__+': '+str(e))
  for p in logs.iterdir():
   if p.is_file():m.integration.merge(pins,{str(p.relative_to(R)):m.package.sha(p)})
  private=dict(commands=commands,primary_errors=primary,endpoint_errors=endpoints,probe=probe,cleanup=cleanup,child_evidence=child_evidence,trace_inventory=trace_inventory);p=OUT/'private-execution.json';raw=(json.dumps(private,indent=2)+'\n').encode()
  with p.open('xb') as f:f.write(raw)
  pins[str(p.relative_to(R))]=m.package.hashlib.sha256(raw).hexdigest()
  record=m.physical.redact_record(dict(format='toklet-m7-phone-energy-attempt-v1',attempt_completed=True,bindings=pins,commands=commands,primary_errors=primary,endpoint_errors=endpoints,probe=probe,independent_phone_cleanup=cleanup,child_evidence=child_evidence,private_trace_files=len(trace_inventory),native_child_exit=child['exit_code'] if child else None,energy_joules=None,energy_qualified=False,physical_device_qualified=False,m7_complete=False,scope='Instrumented separate bounded state workload; trace scope/units require inspection, no functional/performance qualification'),selected,remote)
  with (OUT/'attempt.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  print(json.dumps(dict(attempt_completed=True,recording_completed=probe['recording_completed'] if probe else False,primary_errors=record['primary_errors'],endpoint_errors=record['endpoint_errors'])))
if __name__=='__main__':
 if sys.argv[1:]==['--child']:sys.exit(m.main(NATIVE,PRIVATE,PRIVATE_SHA,PREP,PREP_SHA,'small','512',RUN,transfer=READY,transfer_sha=READY_SHA,native_sha=NATIVE_SHA,state=True))
 if sys.argv[1:]:raise SystemExit('No probe arguments expected')
 main()
