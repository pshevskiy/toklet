"""Independent completed dedicated small phone sustained replay; no model load or device commands."""
from pathlib import Path
import hashlib,importlib.util,json,sys
ROOT=Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_measurement as q
m=q
NATIVE_SHA="06016a4e6eeea98c9c9c0f898cc4fd107c85415ed26cd3faa8de221386002971"
HELPER=ROOT/'artifacts/m7-phone-context-analysis-v1/analyze.py'
spec=importlib.util.spec_from_file_location('context_replay',HELPER);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
DRIVER='artifacts/m7-phone-small-sustained-driver-v3'
PREPARED='artifacts/m7-physical-prepared-v14'
PREPARED_SHA='c5fd8e28d90413ece9d400f43c626e7b052f802241b6596c52d64229967e19b8'
PRIVATE='artifacts/m7-physical-admission-v1/private-selection.json'
PRIVATE_SHA='711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db'
SOURCE='artifacts/m7-phone-small-sustained-source-v4/source.json'
def stages():
 return [dict(stage=stage,model=model,workload=workload,path='artifacts/m7-phone-small-sustained'+suffix+'/measurement.json') for stage,model,workload,suffix in [('storage-proof','tiny-cpu','short','-storage-proof-v1'),('prepare','small','sustained','-ready-v1'),('sustained','small','sustained','-v1')]]
def driver_inputs(source):
 return {DRIVER+'/driver.py',DRIVER+'/checks.py',SOURCE,*source['sources'],*[r['path'] for r in stages()],
 QUALITY}
def require_driver(record,source):
 if (record['passed'] is not True or record['primary_error'] or record['endpoint_errors'] or record['physical_device_qualified'] is not False or record['m7_complete'] is not False
     or set(record['driver_bindings'])!=driver_inputs(source) or len(record['results'])!=3):raise ValueError('Exact complete task driver required')
 for row,expected in zip(record['results'],stages()):
  if ({k:row[k] for k in ('stage','model','workload','path')}!=expected or type(row['exit_code']) is not int or row['exit_code']!=0
      or row['sha256']!=record['driver_bindings'][row['path']]):raise ValueError('Task driver stage order/digest differs')
QUALITY='benchmarks/results/2026-10-21-m7-phone-tasks-qualification-v1.json'
QUALITY_SHA='17ff7fa4c429aa2dff0240642a50f63c6fcc4ab46bd118526cc99eb24c4f5b95'
def admit_prerequisites(source,driver,quality,pins):
 for n,value in source['sources'].items():
  if driver['driver_bindings'].get(n)!=value['sha256']:raise ValueError('Frozen source value differs from driver admission')
 if (pins[QUALITY]!=QUALITY_SHA or driver['driver_bindings'].get(QUALITY)!=QUALITY_SHA
     or quality['format']!='toklet-m7-phone-tasks-qualification-v1' or quality['passed'] is not True or quality['task_quality_qualified'] is not True
     or quality['review']['accepted'] is not True or quality['review']['reviewer']!='Astra' or quality['full_cases_per_model']!=100
     or quality['physical_device_qualified'] is not False or quality['m7_complete'] is not False or quality['new_generated_program_executions']!=0 or quality['new_vm_runs']!=0
     or len(quality['models'])!=2 or {r['model'] for r in quality['models']}!={'small','larger'}
     or any(type(r['cases']) is not int or r['cases']!=100 or r['drop_percentage_points']>2 for r in quality['models'])):raise ValueError('Exact accepted two-model task qualification required')
def require_natural(record,base,metadata,remote,selected):
 natural=dict(absent=True,observations=[[]],signals=[],errors=[])
 exceptional=base==ROOT/'artifacts/m7-phone-small-sustained-storage-proof-v1'
 names=[Path(row['stdout']).stem for row in record['commands']]
 filtered=[n for n in names if n.startswith(('process-','signal-'))]
 if not exceptional:
  if record['remote_cleanup']!=[natural,natural] or filtered!=['process-1','process-2']:raise ValueError('Two natural process absences required')
  indices=(1,2)
 else:
  if filtered!=['process-1','signal-2','process-3','process-4'] or not names.index('current-install-6')<names.index('process-1')<names.index('signal-2')<names.index('process-3')<names.index('copy-Model')<names.index('process-4'):raise ValueError('Exact pre-copy owned reset sequence required')
  initial=metadata(base/'private-logs/process-1.stdout');signal=metadata(base/'private-logs/signal-2.stdout')
  if initial['info']['outcome']!='success' or initial['info']['commandType']!='devicectl.device.info.processes' or initial['result']['deviceIdentifier']!=selected['device_identifier']:raise ValueError('Owned reset query identity differs')
  pids=m.physical.owned_processes(initial,remote)
  if len(pids)!=1:raise ValueError('Exactly one installed owned process required')
  pid=pids[0];result=signal['result'];process=result['process']
  command=next(row['command'] for row in record['commands'] if Path(row['stdout']).stem=='signal-2')
  expected=[m.physical.DEVICECTL,'device','process','signal','--pid',str(pid),'--signal','SIGTERM','--device','<private-device_identifier>','--timeout','90','--quiet','--json-output','-']
  if command!=expected:raise ValueError('Exact recorded owned SIGTERM command required')
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
def check_sustained(samples,config,result):
 if len(samples)<13:raise ValueError('Three warmups and at least10measured requests required')
 reference=samples[0]['tokens']
 if len(reference)!=128 or any(s['tokens']!=reference for s in samples):raise ValueError('Exact repeated128-ID sustained requests required')
 measured=samples[3:];duration=sum(s['events'][-1]['nativeElapsedSeconds'] for s in measured)
 if len(measured)<10 or config['minimumNativeSeconds']!=600 or duration!=result['sumNativeLastEventSeconds'] or duration<600:raise ValueError('At least600measured native seconds required')
 return duration
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
 source=metadata(ROOT/SOURCE);execution=metadata(ROOT/DRIVER/'execution.json');require_driver(execution,source);c.merge(pins,execution['driver_bindings']);c.verify(pins)
 quality=metadata(ROOT/QUALITY);admit_prerequisites(source,execution,quality,pins)
 selected=m.physical.selection(ROOT/PRIVATE,PRIVATE_SHA);prepared=ROOT/PREPARED;prepared_receipt=dict(path=PREPARED+'/integration.json',sha256=PREPARED_SHA)
 _,common,remote,signed=m.physical.admit_reuse(prepared,PREPARED_SHA,selected,NATIVE_SHA)
 m.integration.merge(common,m.integration.admit_build(ROOT/'artifacts/m7-apple-package-v5',NATIVE_SHA))
 for module in (m,m.package,m.integration,m.physical,m.storage):
  path=Path(module.__file__).resolve();m.integration.merge(common,{str(path.relative_to(ROOT)):c.sha(path)})
 for leaf in ('test_m7_physical_measurement.py','test_m7_physical_storage.py'):
  path=ROOT/'tools/experiments'/leaf;m.integration.merge(common,{str(path.relative_to(ROOT)):c.sha(path)})
 m.integration.merge(common,{PRIVATE:PRIVATE_SHA});tiny,_,tiny_inputs=m.job('tiny-cpu','short',verify_sources=False);m.integration.merge(common,tiny_inputs);c.merge(pins,common)
 context=dict(tiny_config=tiny,prepared_receipt=prepared_receipt,signed_bundle=signed,selection_path=PRIVATE,selection_sha=PRIVATE_SHA,selected=selected,remote=remote,required_bindings=common)
 admissions={};ready_shas={};runs=[];commands=0;natural_absences=0;owned_preflight_resets=0
 for row in execution['results']:
  path=ROOT/row['path'];base=path.parent;record=metadata(path)
  if pins[row['path']]!=row['sha256']:raise ValueError('Consumed stage digest differs')
  plain=row['stage']=='storage-proof';config,_,inputs=(m if plain else q).job(row['model'],row['workload'],verify_sources=False)
  c.require_stage(record,config,signed,prepared_receipt)
  wanted_format='toklet-m7-physical-measurement-ready-v1' if row['stage']=='prepare' else 'toklet-m7-physical-measurement-v1'
  if (record['format']!=wanted_format or record['physical_device_qualified'] is not False or record['m7_complete'] is not False):raise ValueError('Stage scope differs')
  for n,h in {**common,**inputs}.items():
   if record['bindings'].get(n)!=h:raise ValueError('Stage input absent: '+n)
  expected_private={str(p.relative_to(ROOT)) for p in (base/'private-execution.json',*(base/'private-logs').iterdir()) if p.is_file()}
  if set(record['private_evidence_pins'])!=expected_private:raise ValueError('Private evidence inventory differs')
  c.merge(pins,record['bindings']);c.merge(pins,record['private_evidence_pins']);c.require_commands(record['commands']);commands+=len(record['commands'])
  private=m.physical.redact_record(metadata(base/'private-execution.json'),selected,remote)
  for field in ('commands','primary_error','endpoint_errors','cleanup_errors'):
   if private[field]!=record[field]:raise ValueError('Private/public execution differs')
  installs=[r for r in record['commands'] if Path(r['stdout']).stem.startswith('current-install-')]
  expected_count=4 if plain or row['stage']=='prepare' else 2
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
   extra,name,digest=m.admit_ready(base,row['sha256'],config,inputs,prepared,PREPARED_SHA,signed,'toklet-m7-physical-measurement-ready-v1');c.merge(pins,extra)
   admission,extra=m.storage.admit_ready(record,base/'transfer',config,name,digest,context,m.validate_console);c.merge(pins,extra);admissions[key]=admission;ready_shas[key]=dict(path=row['path'],sha256=row['sha256'])
   continue
  stdout=raw(base/'private-logs/measurement.stdout').decode();stderr=raw(base/'private-logs/measurement.stderr').decode()
  checked=(m if plain else q).validate_console(stdout+'\n'+stderr,config,signed);checked['launch']=m.physical.check_launch(metadata(base/'private-logs/launch.json'),remote,selected)
  if checked!=record['checked']:raise ValueError('Raw console/launch differs')
  if plain:continue
  samples=[json.loads(line.split('=',1)[1]) for line in (stdout+'\n'+stderr).splitlines() if line.startswith('TOKLET_MEASUREMENT_SAMPLE_JSON=')]
  if record['transfer_receipt']!=ready_shas[key] or record['storage_admission']!=admissions[key]:raise ValueError('Original sustained ready/storage differs')
  if any(Path(r['stdout']).stem.startswith('copy-') for r in record['commands']):raise ValueError('Unexpected sustained copy')
  result=checked['result'];duration=check_sustained(samples,config,result);measured=samples[3:]
  observations=[s['footprint'] for s in samples]+[p['footprint'] for p in result['checkpoints']]
  runs.append(dict(model='small',workload='sustained512',receipt=row['path'],samples=len(samples),warmups=3,measured_samples=len(measured),
   completed_native_seconds_excluding_warmups=duration,completed_caller_seconds_excluding_warmups=sum((s['completedUptimeNanoseconds']-s['submittedUptimeNanoseconds'])/1e9 for s in measured),
   native_token_events=len(samples)*128,identical_all128_ids=True,warmup_metrics=c.metrics(samples[:3]),measured_metrics=c.metrics(measured),early_five_metrics=c.metrics(measured[:5]),late_five_metrics=c.metrics(measured[-5:]),
   numerical_admission_bytes=sorted({s['admittedBytes'] for s in samples}),peak_physical_bytes=max(f['lifetimePeakPhysicalBytes'] for f in observations),
   thermal_states=sorted({s['environment']['thermalState'] for s in samples}),foreground_states=sorted({s['environment']['applicationState'] for s in samples}),
   low_power_modes=sorted({s['environment']['lowPowerMode'] for s in samples}),energy=None))
 if len(runs)!=1 or natural_absences!=5 or owned_preflight_resets!=1:raise ValueError('One complete dedicated sustained sequence required')
 c.verify(pins)
 result=dict(format='toklet-m7-phone-small-sustained-analysis-v1',passed=True,bindings=pins,runs=runs,stages=3,commands=commands,natural_absence_checks=natural_absences,owned_preflight_resets=owned_preflight_resets,preflight_reset_stage='artifacts/m7-phone-small-sustained-storage-proof-v1/measurement.json',dedicated_small_duration_qualified=True,controlled_speedup_qualified=False,physical_device_qualified=False,m7_complete=False,energy=None,scope='Dedicated512 sustained completed native duration/memory/thermal observations; no controlled speedup or overall device/M7 acceptance')
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=True,stages=3,runs=1,measured_native_seconds=runs[0]['completed_native_seconds_excluding_warmups'])))
if __name__=='__main__':run()
