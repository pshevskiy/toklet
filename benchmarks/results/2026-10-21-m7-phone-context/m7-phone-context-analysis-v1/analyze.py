"""Replay completed exploratory phone trials and summarize completed work only."""
from pathlib import Path
import hashlib,json,math,statistics,sys
ROOT=Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_measurement as m

def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def merge(pins, incoming):
 for name,digest in incoming.items():
  if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Nonlocal binding path')
  if name in pins and pins[name]!=digest:raise ValueError('Conflicting input: '+name)
  pins[name]=digest
def verify(pins):
 for name,digest in pins.items():
  if sha(ROOT/name)!=digest:raise ValueError('Changed analysis endpoint: '+name)
def read_bound(path,limit,pins):
 with path.open('rb') as stream:raw=stream.read(limit+1)
 if len(raw)>limit:raise ValueError('Oversized analysis input')
 merge(pins,{str(path.relative_to(ROOT)):hashlib.sha256(raw).hexdigest()})
 return raw
def expected_stages():
 rows=[]
 for model in ('small','larger'):
  for workload in ('128','512','1024'):
   stem='artifacts/m7-phone-context-'+model+'-'+workload
   for stage,tail in [('storage-proof','-storage-proof-v1'),('prepare','-ready-v1'),('measure','-v1'),('measure','-v2')]:
    rows.append(dict(stage=stage,model=model,workload=workload,path=stem+tail,exit_code=0))
 return rows
def expected_driver_inputs():
 sources={'artifacts/m7-phone-context-driver-v1/driver.py',
          *('tools/experiments/'+name+'.py' for name in ['m7_physical_measurement','m7_apple_package','m7_apple_integration','m7_physical_integration','m7_physical_storage'])}
 receipts={row['path']+'/measurement.json' for row in expected_stages() if row['stage']!='measure'}
 return sources|receipts
def require_commands(commands):
 if not commands:raise ValueError('No saved commands')
 for command in commands:
  if type(command['exit_code']) is not int or command['exit_code']!=0 or command['primary_error'] is not None or command['cleanup_errors']:
   raise ValueError('Unsuccessful saved command')
def require_current(data,remote,selected):
 return m.physical.launch_verified(lambda:data,lambda:True,remote,selected)
def require_execution(record):
 if (type(record['exit_code']) is not int or record['exit_code']!=0 or record['primary_error'] or record['endpoint_errors']
     or record['results']!=expected_stages() or any(type(row['exit_code']) is not int for row in record['results'])):
  raise ValueError('Exact successful matrix sequence required')
 ready={row['path']+'/measurement.json' for row in expected_stages() if row['stage']=='prepare'}
 if set(record['driver_bindings'])!=expected_driver_inputs() or set(record['ready_receipts'])!=ready:
  raise ValueError('Incomplete driver/ready binding inventories')
def require_stage(record,config,signed,prepared):
 if (record['passed'] is not True or record['primary_error'] or record['endpoint_errors'] or record['cleanup_errors']
     or record['config']!=config or record['signed_bundle']!=signed or record['prepared_receipt']!=prepared):
  raise ValueError('Stage config/installation/status differs')


def summary(values):
 if not values or any(not math.isfinite(x) or x<=0 for x in values):raise ValueError('Invalid timing sample')
 center=statistics.median(values)
 return dict(median=center,minimum=min(values),maximum=max(values),median_absolute_deviation=statistics.median(abs(x-center) for x in values))
def metrics(requests):
 rows=[]
 for sample in requests:
  events=sample['events'];first=events[0];last=events[-1]
  if len(events)!=128 or not last['nativeElapsedSeconds']>first['nativeElapsedSeconds']:
   raise ValueError('Completed decode interval unavailable')
  rows.append(dict(index=sample['index'],native_ttft_seconds=first['nativeElapsedSeconds'],
                   callback_ttft_seconds=(first['callbackUptimeNanoseconds']-sample['submittedUptimeNanoseconds'])/1e9,
                   native_last_token_seconds=last['nativeElapsedSeconds'],
                   completed_request_seconds=(sample['completedUptimeNanoseconds']-sample['submittedUptimeNanoseconds'])/1e9,
                   decode_tokens_per_second=127/(last['nativeElapsedSeconds']-first['nativeElapsedSeconds'])))
 return dict(raw=rows,summary={name:summary([r[name] for r in rows]) for name in rows[0] if name!='index'})
def require_cleanups(stage,record,base,metadata,remote,selected):
 cleanups=record['remote_cleanup']
 if len(cleanups)!=2:raise ValueError('Incomplete absence observations')
 first=stage==expected_stages()[0]
 expected_sequence=['process-1','signal-2','process-3','process-4'] if first else ['process-1','process-2']
 names=[Path(c['stdout']).stem for c in record['commands']]
 if [n for n in names if n.startswith(('process-','signal-'))]!=expected_sequence:
  raise ValueError('Unexpected process/signal command sequence')
 def processes(index):
  data=metadata(base/'private-logs'/('process-'+str(index)+'.stdout'))
  if (data['info']['outcome']!='success' or data['info']['commandType']!='devicectl.device.info.processes'
   or data['result']['deviceIdentifier']!=selected['device_identifier']):raise ValueError('Process observation device/outcome differs')
  return m.physical.owned_processes(data,remote)
 natural=dict(absent=True,observations=[[]],signals=[],errors=[])
 if not first:
  if cleanups!=[natural,natural] or processes(1) or processes(2):raise ValueError('Natural absence differs')
  return dict(initial_preflight_resets=0,natural_absence_checks=2)
 # The sole admitted intervention is the exact first preflight reset. It
 # precedes input copying and even the tiny storage-proof launch, so no timed
 # model request includes forced termination. Every other cleanup stays natural.
 pids=processes(1)
 if len(pids)!=1:raise ValueError('Exact first owned preflight process required')
 pid=pids[0]
 expected=dict(absent=True,observations=[pids,[]],signals=[dict(pid=pid,signal='SIGTERM')],errors=[])
 if cleanups!=[expected,natural] or processes(3) or processes(4):raise ValueError('First reset or endpoint absence differs')
 data=metadata(base/'private-logs/signal-2.stdout');result=data['result'];process=result['process']
 if (data['info']['outcome']!='success' or data['info']['commandType']!='devicectl.device.process.signal'
  or result['deviceIdentifier']!=selected['device_identifier'] or type(process['processIdentifier']) is not int
  or process['processIdentifier']!=pid or m.physical.remote_path(process['executable'])!=remote+'/TokletHarness'
  or result['signal']!=dict(name='SIGTERM',value=15)):raise ValueError('Exact owned SIGTERM result differs')
 command=record['commands'][names.index('signal-2')]['command']
 if command!=[m.physical.DEVICECTL,'device','process','signal','--pid',str(pid),'--signal','SIGTERM','--device',
              '<private-device_identifier>','--timeout','90','--quiet','--json-output','-']:
  raise ValueError('Exact owned preflight signal command differs')
 order=['current-install-6','process-1','signal-2','process-3','copy-Model','measurement','process-4']
 if any(names.count(n)!=1 for n in order) or [names.index(n) for n in order]!=sorted(names.index(n) for n in order):
  raise ValueError('Initial reset must precede input copy and first launch')
 return dict(initial_preflight_resets=1,natural_absence_checks=1)

def run():
 own=Path(__file__).resolve();out=own.parent/'analysis.json'
 if out.exists() or out.is_symlink():raise FileExistsError('Exclusive analysis required')
 bindings={};buffers={}
 def pin(path):
  merge(bindings,{str(path.relative_to(ROOT)):sha(path)})
  return path
 def raw(path,limit=16*1024**2):
  if path not in buffers:buffers[path]=read_bound(path,limit,bindings)
  if len(buffers[path])>limit:raise ValueError('Oversized cached analysis input')
  return buffers[path]
 def metadata(path):return json.loads(raw(path))
 pin(own)
 for module in (m,m.package,m.integration,m.physical,m.storage):pin(Path(module.__file__).resolve())
 execution=metadata(ROOT/'artifacts/m7-phone-context-driver-v1/execution.json');require_execution(execution)
 merge(bindings,execution['driver_bindings']);verify(bindings)
 private=ROOT/'artifacts/m7-physical-admission-v1/private-selection.json'
 selection_sha='711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db'
 selected=m.physical.selection(private,selection_sha)
 prepared=ROOT/'artifacts/m7-physical-prepared-v11';prepared_sha='661010123ba4cc9539dee098aefaf9bb5399c184c57f84b3fa748d4563fd6577'
 prepared_receipt=dict(path=str((prepared/'integration.json').relative_to(ROOT)),sha256=prepared_sha)
 _,common,remote,signed=m.physical.admit_reuse(prepared,prepared_sha,selected,m.NATIVE_SHA)
 m.integration.merge(common,m.integration.admit_build(ROOT/'artifacts/m7-apple-package-v4',m.NATIVE_SHA))
 for module in (m,m.package,m.integration,m.physical,m.storage):
  path=Path(module.__file__).resolve();m.integration.merge(common,{str(path.relative_to(ROOT)):sha(path)})
 for name in ('test_m7_physical_measurement.py','test_m7_physical_storage.py'):
  path=ROOT/'tools/experiments'/name;m.integration.merge(common,{str(path.relative_to(ROOT)):sha(path)})
 m.integration.merge(common,{str(private.relative_to(ROOT)):selection_sha})
 tiny,_,tiny_inputs=m.job('tiny-cpu','short',verify_sources=False);m.integration.merge(common,tiny_inputs)
 merge(bindings,common)
 context=dict(tiny_config=tiny,prepared_receipt=prepared_receipt,signed_bundle=signed,selection_path=str(private.relative_to(ROOT)),
              selection_sha=selection_sha,selected=selected,remote=remote,required_bindings=common)
 admissions={}
 runs=[];tokens={};reset_count=0;natural_count=0
 for row in execution['results']:
  if row['exit_code']!=0:raise ValueError('Failed stage')
  base=ROOT/row['path'];receipt=base/'measurement.json';record=metadata(receipt)
  model,workload=('tiny-cpu','short') if row['stage']=='storage-proof' else (row['model'],row['workload'])
  config,_,inputs=m.job(model,workload,verify_sources=False);require_stage(record,config,signed,prepared_receipt)
  expected_format='toklet-m7-physical-measurement-ready-v1' if row['stage']=='prepare' else 'toklet-m7-physical-measurement-v1'
  if record['format']!=expected_format or record['physical_device_qualified'] is not False or record['m7_complete'] is not False:
   raise ValueError('Stage format or qualification scope differs')
  for name,digest in {**common,**inputs}.items():
   if record['bindings'].get(name)!=digest:raise ValueError('Missing required stage input: '+name)
  expected_private={str(path.relative_to(ROOT)) for path in [base/'private-execution.json',*(base/'private-logs').iterdir()] if path.is_file()}
  if set(record['private_evidence_pins'])!=expected_private:raise ValueError('Private evidence inventory differs')
  for field in ('bindings','private_evidence_pins'):merge(bindings,record[field])
  private_execution=m.physical.redact_record(metadata(base/'private-execution.json'),selected,remote)
  require_commands(private_execution['commands']);require_commands(record['commands'])
  install_observations=[command for command in record['commands'] if Path(command['stdout']).name.startswith('current-install-')]
  expected_observations=4 if row['stage']=='storage-proof' else (3 if row['model']=='small' else 4) if row['stage']=='prepare' else 2
  if len(install_observations)!=expected_observations:raise ValueError('Current installation guard sequence incomplete')
  for command in install_observations:require_current(metadata(ROOT/command['stdout']),remote,selected)
  for field in ('commands','primary_error','endpoint_errors','cleanup_errors'):
   if private_execution[field]!=record[field]:raise ValueError('Execution/summary correspondence differs')
  for leaf in ('device-before.stdout','device-after.stdout'):
   if m.physical.admit_device(metadata(base/'private-logs'/leaf),selected)!=record['device']:
    raise ValueError('Device endpoint differs')
  if m.storage.boot_token(raw(base/'private-logs/host-boot.stdout',1024).decode())!=record['host_boot_token']:
   raise ValueError('Recorded boot differs')
  cleanup_scope=require_cleanups(row,record,base,metadata,remote,selected)
  reset_count+=cleanup_scope['initial_preflight_resets'];natural_count+=cleanup_scope['natural_absence_checks']
  if row['stage']=='storage-proof':
   _,extra=m.storage.admit_proof(base,bindings[str(receipt.relative_to(ROOT))],context,m.validate_console)
   merge(bindings,extra)
  elif row['stage']=='prepare':
   digest=bindings[str(receipt.relative_to(ROOT))]
   if execution['ready_receipts'].get(str(receipt.relative_to(ROOT)))!=digest:raise ValueError('Ready digest differs')
   extra,name,job_sha=m.admit_ready(base,digest,config,inputs,prepared,prepared_sha,signed);merge(bindings,extra)
   admission,extra=m.storage.admit_ready(record,base/'transfer',config,name,job_sha,context,m.validate_console)
   merge(bindings,extra);admissions[(row['model'],row['workload'])]=admission
  if row['stage']=='prepare':continue
  stdout=raw(base/'private-logs/measurement.stdout',64*1024**2).decode()
  stderr=raw(base/'private-logs/measurement.stderr',64*1024**2).decode()
  checked=m.validate_console(stdout+'\n'+stderr,config,signed)
  checked['launch']=m.physical.check_launch(metadata(base/'private-logs/launch.json'),remote,selected)
  if checked!=record['checked']:raise ValueError('Raw console/launch differs')
  if row['stage']=='storage-proof':continue
  if record['storage_admission']!=admissions[(row['model'],row['workload'])]:raise ValueError('Original storage admission differs')
  samples=[json.loads(line.split('=',1)[1]) for line in stdout.splitlines() if line.startswith('TOKLET_MEASUREMENT_SAMPLE_JSON=')]
  if len(samples)!=13:raise ValueError('Incomplete reset request sequence')
  key=(row['model'],row['workload']);expected=tokens.setdefault(key,samples[0]['tokens'])
  if any(sample['tokens']!=expected for sample in samples):raise ValueError('Identical reset requests generated differing IDs; investigate numerical behavior')
  if len(expected)!=128:raise ValueError('Incomplete generated sequence')
  cfg=record['config'];result=record['checked']['result']
  ready_name=record['transfer_receipt']['path']
  canonical_ready='artifacts/m7-phone-context-'+row['model']+'-'+row['workload']+'-ready-v1/measurement.json'
  if ready_name!=canonical_ready:raise ValueError('Noncanonical ready-receipt path')
  if execution['ready_receipts'][ready_name]!=record['transfer_receipt']['sha256']:
   raise ValueError('Frozen ready receipt differs')
  if any('/copy-' in c['stdout'] for c in record['commands']):raise ValueError('Unexpected measurement copy')
  footprints=[s['footprint'] for s in samples]+[c['footprint'] for c in result['checkpoints']]
  runs.append(dict(model=row['model'],workload=row['workload'],receipt=str(receipt.relative_to(ROOT)),
      config=cfg,load_and_session_seconds=result['loadAndSessionSeconds'],warmups=metrics(samples[:3]),measured=metrics(samples[3:]),
      peak_physical_bytes=max(f['lifetimePeakPhysicalBytes'] for f in footprints),
      post_request_thermal_states=sorted(set(s['environment']['thermalState'] for s in samples)),
      post_request_power=[s['environment'] for s in samples],tokens_sha256=hashlib.sha256(json.dumps(expected,separators=(',',':')).encode()).hexdigest(),
      numerical_admission_bytes=sorted(set(s['admittedBytes'] for s in samples)),energy=None,isolated_prefill_seconds=None))
 if len(runs)!=12 or len(tokens)!=6 or reset_count!=1 or natural_count!=47:raise ValueError('Expected complete repeated matrix and exact cleanup scope')
 verify(bindings)
 record=dict(format='toklet-m7-exploratory-context-analysis-v1',passed=True,bindings=bindings,runs=runs,
             request_samples=156,warmup_samples=36,measured_samples=120,raw_token_events=19968,
             identical_generated_sequences=True,initial_preflight_resets=reset_count,natural_absence_checks=natural_count,energy=None,full_quality_qualified=False,physical_device_qualified=False,m7_complete=False,
             scope='Descriptive raw-ID reset workloads; native TTFT includes prefix setup/prefill/sampling; callback TTFT includes Swift scheduling; ten repetitions do not establish tail percentiles')
 with out.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(passed=True,runs=len(runs),request_samples=156,measured_samples=120,raw_token_events=19968)))
if __name__=='__main__':run()
