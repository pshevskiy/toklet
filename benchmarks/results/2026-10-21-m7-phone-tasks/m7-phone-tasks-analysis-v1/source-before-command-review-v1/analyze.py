"""Independent completed phone task response replay; no model load or device commands."""
from pathlib import Path
import hashlib,importlib.util,json,sys
ROOT=Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_tasks as q
m=q.measurement
HELPER=ROOT/'artifacts/m7-phone-context-analysis-v1/analyze.py'
spec=importlib.util.spec_from_file_location('context_replay',HELPER);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
DRIVER='artifacts/m7-phone-tasks-driver-v1'
PREPARED='artifacts/m7-physical-prepared-v14'
PREPARED_SHA='c5fd8e28d90413ece9d400f43c626e7b052f802241b6596c52d64229967e19b8'
PRIVATE='artifacts/m7-physical-admission-v1/private-selection.json'
PRIVATE_SHA='711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db'
def stages():
 rows=[dict(stage='tiny-tasks',model=model,workload='short',path='artifacts/m7-phone-tasks-'+model+'-v1/measurement.json') for model in ('tiny-cpu','tiny-metal')]
 for model in ('small','larger'):
  for workload in ('short','full'):
   stem='artifacts/m7-phone-tasks-'+model+'-'+workload
   rows.extend([dict(stage='storage-proof',model='tiny-cpu',workload='short',path=stem+'-storage-proof-v1/measurement.json'),dict(stage='prepare',model=model,workload=workload,path=stem+'-ready-v1/measurement.json'),dict(stage='tasks',model=model,workload=workload,path=stem+'-v1/measurement.json')])
 return rows
def driver_inputs():
 names={DRIVER+'/driver.py',DRIVER+'/checks.py',*('tools/experiments/'+n+'.py' for n in ('m7_physical_tasks','m7_physical_measurement','m7_apple_package','m7_apple_integration','m7_physical_integration','m7_physical_storage'))}
 names.update(r['path'] for r in stages())
 names.update('artifacts/m7-phone-tasks-'+model+'-v1/private-logs/measurement.'+suffix for model in ('tiny-cpu','tiny-metal') for suffix in ('stdout','stderr'))
 return names
def require_driver(record):
 if (record['passed'] is not True or record['primary_error'] or record['endpoint_errors'] or record['physical_device_qualified'] is not False or record['m7_complete'] is not False
     or set(record['driver_bindings'])!=driver_inputs() or len(record['results'])!=14):raise ValueError('Exact complete task driver required')
 for row,expected in zip(record['results'],stages()):
  if ({k:row[k] for k in ('stage','model','workload','path')}!=expected or type(row['exit_code']) is not int or row['exit_code']!=0
      or row['sha256']!=record['driver_bindings'][row['path']]):raise ValueError('Task driver stage order/digest differs')
def require_natural(record,base,metadata,remote,selected):
 natural=dict(absent=True,observations=[[]],signals=[],errors=[])
 exceptional=base==ROOT/'artifacts/m7-phone-tasks-small-short-ready-v1'
 names=[Path(row['stdout']).stem for row in record['commands']]
 filtered=[n for n in names if n.startswith(('process-','signal-'))]
 if not exceptional:
  if record['remote_cleanup']!=[natural,natural] or filtered!=['process-1','process-2']:raise ValueError('Two natural process absences required')
  indices=(1,2)
 else:
  if filtered!=['process-1','signal-2','process-3','process-4'] or not names.index('process-3')<names.index('copy-Model')<names.index('process-4'):raise ValueError('Exact pre-copy owned reset sequence required')
  initial=metadata(base/'private-logs/process-1.stdout');signal=metadata(base/'private-logs/signal-2.stdout')
  if initial['info']['outcome']!='success' or initial['info']['commandType']!='devicectl.device.info.processes' or initial['result']['deviceIdentifier']!=selected['device_identifier']:raise ValueError('Owned reset query identity differs')
  pids=m.physical.owned_processes(initial,remote)
  if len(pids)!=1:raise ValueError('Exactly one installed owned process required')
  pid=pids[0];result=signal['result'];process=result['process']
  if (signal['info']['outcome']!='success' or signal['info']['commandType']!='devicectl.device.process.signal'
      or result['deviceIdentifier']!=selected['device_identifier'] or type(process['processIdentifier']) is not int or process['processIdentifier']!=pid
      or m.physical.remote_path(process['executable'])!=remote+'/TokletHarness' or result['signal']!={'name':'SIGTERM','value':15}):raise ValueError('Exact installed-process successful SIGTERM required')
  reset=dict(absent=True,observations=[[pid],[]],signals=[dict(pid=pid,signal='SIGTERM')],errors=[])
  if record['remote_cleanup']!=[reset,natural]:raise ValueError('Owned preflight reset receipt differs')
  indices=(3,4)
 for index in indices:
  data=metadata(base/'private-logs'/('process-'+str(index)+'.stdout'))
  if (data['info']['outcome']!='success' or data['info']['commandType']!='devicectl.device.info.processes' or data['result']['deviceIdentifier']!=selected['device_identifier']
      or m.physical.owned_processes(data,remote)):raise ValueError('Owned process absence differs')
 return (1,1) if exceptional else (2,0)
def tiny_evidence(samples):
 if len(samples)!=2:raise ValueError('Two independent tiny cases required')
 responses=[r['response'] for r in samples]
 for row,maximum in zip(responses,(3,2)):
  if (any(type(row[k]) is not int for k in ('sampled','reason')) or any(type(t) is not int for t in row['ids'])
      or row['ids']!=[47]*maximum or row['sampled']!=maximum or row['reason']!=1 or row.get('stopToken') is not None):raise ValueError('Independent tiny response differs')
 return responses
def require_tiny(record,expected):
 if record['tiny_oracle']!=expected:raise ValueError('Driver tiny oracle differs from consumed rows')
def run():
 own=Path(__file__).resolve();out=own.parent/'analysis.json'
 if out.exists() or out.is_symlink():raise FileExistsError('Exclusive analysis required')
 pins={};buffers={}
 def raw(path,limit=64*1024**2):
  if path not in buffers:buffers[path]=c.read_bound(path,limit,pins)
  if len(buffers[path])>limit:raise ValueError('Oversized cached input')
  return buffers[path]
 def metadata(path):return json.loads(raw(path))
 def pin(path):c.merge(pins,{str(path.relative_to(ROOT)):c.sha(path)})
 for path in (own,own.parent/'checks.py',HELPER,*[Path(v.__file__).resolve() for v in (q,m,m.package,m.integration,m.physical,m.storage)]):pin(path)
 execution=metadata(ROOT/DRIVER/'execution.json');require_driver(execution);c.merge(pins,execution['driver_bindings']);c.verify(pins)
 selected=m.physical.selection(ROOT/PRIVATE,PRIVATE_SHA);prepared=ROOT/PREPARED;prepared_receipt=dict(path=PREPARED+'/integration.json',sha256=PREPARED_SHA)
 _,common,remote,signed=m.physical.admit_reuse(prepared,PREPARED_SHA,selected,q.NATIVE_SHA)
 m.integration.merge(common,m.integration.admit_build(ROOT/'artifacts/m7-apple-package-v5',q.NATIVE_SHA))
 for module in (m,m.package,m.integration,m.physical,m.storage):
  path=Path(module.__file__).resolve();m.integration.merge(common,{str(path.relative_to(ROOT)):c.sha(path)})
 for leaf in ('test_m7_physical_measurement.py','test_m7_physical_storage.py'):
  path=ROOT/'tools/experiments'/leaf;m.integration.merge(common,{str(path.relative_to(ROOT)):c.sha(path)})
 m.integration.merge(common,{PRIVATE:PRIVATE_SHA});tiny,_,tiny_inputs=m.job('tiny-cpu','short',verify_sources=False);m.integration.merge(common,tiny_inputs);c.merge(pins,common)
 context=dict(tiny_config=tiny,prepared_receipt=prepared_receipt,signed_bundle=signed,selection_path=PRIVATE,selection_sha=PRIVATE_SHA,selected=selected,remote=remote,required_bindings=common)
 tiny_results={};natural_absences=0;owned_preflight_resets=0
 admissions={};ready_shas={};runs=[];commands=0
 for row in execution['results']:
  path=ROOT/row['path'];base=path.parent;record=metadata(path)
  if pins[row['path']]!=row['sha256']:raise ValueError('Consumed stage digest differs')
  plain=row['stage']=='storage-proof';config,_,inputs=(m if plain else q).job(row['model'],row['workload'],verify_sources=False)
  c.require_stage(record,config,signed,prepared_receipt)
  wanted_format='toklet-m7-physical-measurement-v1' if plain else q.READY_FORMAT if row['stage']=='prepare' else q.RESULT_FORMAT
  if (record['format']!=wanted_format or record['physical_device_qualified'] is not False or record['m7_complete'] is not False or not plain and record['full_quality_qualified'] is not False):raise ValueError('Stage scope differs')
  for n,h in {**common,**inputs}.items():
   if record['bindings'].get(n)!=h:raise ValueError('Stage input absent: '+n)
  expected_private={str(p.relative_to(ROOT)) for p in (base/'private-execution.json',*(base/'private-logs').iterdir()) if p.is_file()}
  if set(record['private_evidence_pins'])!=expected_private:raise ValueError('Private evidence inventory differs')
  c.merge(pins,record['bindings']);c.merge(pins,record['private_evidence_pins']);c.require_commands(record['commands']);commands+=len(record['commands'])
  private=m.physical.redact_record(metadata(base/'private-execution.json'),selected,remote)
  for field in ('commands','primary_error','endpoint_errors','cleanup_errors'):
   if private[field]!=record[field]:raise ValueError('Private/public execution differs')
  installs=[r for r in record['commands'] if Path(r['stdout']).stem.startswith('current-install-')]
  expected_count=4 if plain else 5 if row['stage']=='tiny-tasks' or row['stage']=='prepare' and row['model']=='larger' else 4 if row['stage']=='prepare' else 2
  if len(installs)!=expected_count:raise ValueError('Current installed-app guard count differs')
  for command in installs:c.require_current(metadata(ROOT/command['stdout']),remote,selected)
  for leaf in ('device-before.stdout','device-after.stdout'):
   if m.physical.admit_device(metadata(base/'private-logs'/leaf),selected)!=record['device']:raise ValueError('Device endpoints differ')
  if m.storage.boot_token(raw(base/'private-logs/host-boot.stdout',1024).decode())!=record['host_boot_token']:raise ValueError('Host boot differs')
  natural_count,reset_count=require_natural(record,base,metadata,remote,selected);natural_absences+=natural_count;owned_preflight_resets+=reset_count
  key=(row['model'],row['workload'])
  if plain:
   _,extra=m.storage.admit_proof(base,row['sha256'],context,m.validate_console);c.merge(pins,extra)
  elif row['stage']=='prepare':
   extra,name,digest=m.admit_ready(base,row['sha256'],config,inputs,prepared,PREPARED_SHA,signed,q.READY_FORMAT);c.merge(pins,extra)
   admission,extra=m.storage.admit_ready(record,base/'transfer',config,name,digest,context,m.validate_console);c.merge(pins,extra);admissions[key]=admission;ready_shas[key]=dict(path=row['path'],sha256=row['sha256'])
   continue
  stdout=raw(base/'private-logs/measurement.stdout').decode();stderr=raw(base/'private-logs/measurement.stderr').decode()
  checked=(m if plain else q).validate_console(stdout+'\n'+stderr,config,signed);checked['launch']=m.physical.check_launch(metadata(base/'private-logs/launch.json'),remote,selected)
  if checked!=record['checked']:raise ValueError('Raw console/launch differs')
  if plain:continue
  samples=[json.loads(line.split('=',1)[1]) for line in (stdout+'\n'+stderr).splitlines() if line.startswith('TOKLET_TASK_CASE_JSON=')]
  if len(samples)!=config['caseCount']:raise ValueError('Raw case count differs')
  if row['stage']=='tiny-tasks':tiny_results[row['model']]=tiny_evidence(samples)
  if row['stage']=='tasks':
   if record['transfer_receipt']!=ready_shas[key] or record['storage_admission']!=admissions[key]:raise ValueError('Original ready/storage admission differs')
   if any(Path(r['stdout']).stem.startswith('copy-') for r in record['commands']):raise ValueError('Unexpected task copy')
  if checked['responses']!=samples:raise ValueError('Raw response rows differ')
  result=checked['result'];chunks=[json.loads(line.split('=',1)[1]) for line in (stdout+'\n'+stderr).splitlines() if line.startswith('TOKLET_TASK_CHUNK_JSON=')]
  observations=[s['footprint'] for s in samples+chunks]+[p['footprint'] for p in result['checkpoints']]
  _,payload,_=q.task_payload(row['model'],row['workload'])
  if [r['caseID'] for r in samples]!=[r['id'] for r in payload['cases']]:raise ValueError('Original ordered task IDs differ')
  runs.append(dict(model=row['model'],workload=row['workload'],receipt=row['path'],case_count=len(samples),chunk_count=len(chunks),
   emitted_ids=sum(len(s['response']['ids']) for s in samples),sampled_ids=sum(s['response']['sampled'] for s in samples),
   task_set_sha256=config['tasksSHA256'],loaded_admitted_bytes=result['loadedAdmittedBytes'],
   peak_physical_bytes=max(f['lifetimePeakPhysicalBytes'] for f in observations),load_and_session_seconds=result['loadAndSessionSeconds'],
   thermal_states=sorted({s['environment']['thermalState'] for s in samples+chunks}),foreground_states=sorted({s['environment']['applicationState'] for s in samples+chunks}),
   low_power_modes=sorted({s['environment']['lowPowerMode'] for s in samples+chunks}),energy=None))
 require_tiny(execution,tiny_results)
 if len(runs)!=6 or sum(r['case_count'] for r in runs)!=206 or natural_absences!=27 or owned_preflight_resets!=1:raise ValueError('Incomplete physical response scope')
 c.verify(pins)
 result=dict(format='toklet-m7-physical-task-response-analysis-v1',passed=True,bindings=pins,runs=runs,stages=14,commands=commands,cases=206,full_cases_per_model=100,natural_absence_checks=natural_absences,owned_preflight_resets=owned_preflight_resets,preflight_reset_stage='artifacts/m7-phone-tasks-small-short-ready-v1/measurement.json',task_quality_qualified=False,physical_device_qualified=False,m7_complete=False,energy=None,scope='Frozen raw phone task responses only; independent post-exit decoding/literal grading remains')
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=True,stages=14,runs=6,cases=206,task_quality_qualified=False)))
if __name__=='__main__':run()
