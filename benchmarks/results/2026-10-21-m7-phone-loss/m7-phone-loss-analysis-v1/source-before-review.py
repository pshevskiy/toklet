"""Independent completed phone loss replay; no model load or device commands."""
from pathlib import Path
import hashlib,importlib.util,json,math,sys
ROOT=Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_quality as q
m=q.measurement
HELPER=ROOT/'artifacts/m7-phone-context-analysis-v1/analyze.py'
spec=importlib.util.spec_from_file_location('context_replay',HELPER);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
DRIVER='artifacts/m7-phone-loss-driver-v2'
PREPARED='artifacts/m7-physical-prepared-v14'
PREPARED_SHA='c5fd8e28d90413ece9d400f43c626e7b052f802241b6596c52d64229967e19b8'
PRIVATE='artifacts/m7-physical-admission-v1/private-selection.json'
PRIVATE_SHA='711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db'
BASELINE='benchmarks/results/2026-10-15-m6-online-full-loss-qualification-v1.json'
# This reviewed result carries both unchanged matched FP32 baseline means.
BASELINE_SHA='7b87646805ac5c6053150b3d7f02a06ea37daf46c967222401d6735d3a64def9'
def stages():
 rows=[dict(stage='tiny-quality',model=model,workload='short',path='artifacts/m7-phone-loss-'+model+'-v2/measurement.json') for model in ('tiny-cpu','tiny-metal')]
 for model in ('small','larger'):
  for workload in ('short','full'):
   stem='artifacts/m7-phone-loss-'+model+'-'+workload
   rows.extend([dict(stage='storage-proof',model='tiny-cpu',workload='short',path=stem+'-storage-proof-v1/measurement.json'),dict(stage='prepare',model=model,workload=workload,path=stem+'-ready-v1/measurement.json'),dict(stage='quality',model=model,workload=workload,path=stem+'-v1/measurement.json')])
 return rows
def driver_inputs():
 names={DRIVER+'/driver.py',DRIVER+'/checks.py',*('tools/experiments/'+n+'.py' for n in ('m7_physical_quality','m7_physical_measurement','m7_apple_package','m7_apple_integration','m7_physical_integration','m7_physical_storage','m7_native_teacher_check')),
 'tests/fixtures/tiny-v1/manifest.json','tests/fixtures/tiny-v1/cases.prompt.logits.f32le'}
 names.update(r['path'] for r in stages())
 names.update('artifacts/m7-phone-loss-'+model+'-v2/private-logs/measurement.'+suffix for model in ('tiny-cpu','tiny-metal') for suffix in ('stdout','stderr'))
 return names
def require_driver(record):
 if (record['passed'] is not True or record['primary_error'] or record['endpoint_errors'] or record['physical_device_qualified'] is not False or record['m7_complete'] is not False
     or set(record['driver_bindings'])!=driver_inputs() or len(record['results'])!=14):raise ValueError('Exact complete loss driver required')
 for row,expected in zip(record['results'],stages()):
  if ({k:row[k] for k in ('stage','model','workload','path')}!=expected or type(row['exit_code']) is not int or row['exit_code']!=0
      or row['sha256']!=record['driver_bindings'][row['path']]):raise ValueError('Loss driver stage order/digest differs')
def require_natural(record,base,metadata,remote,selected):
 natural=dict(absent=True,observations=[[]],signals=[],errors=[])
 if record['remote_cleanup']!=[natural,natural]:raise ValueError('Two natural process absences required')
 names=[Path(row['stdout']).stem for row in record['commands']]
 if [n for n in names if n.startswith(('process-','signal-'))]!=['process-1','process-2']:raise ValueError('Unexpected process/signal sequence')
 for index in (1,2):
  data=metadata(base/'private-logs'/('process-'+str(index)+'.stdout'))
  if (data['info']['outcome']!='success' or data['info']['commandType']!='devicectl.device.info.processes' or data['result']['deviceIdentifier']!=selected['device_identifier']
      or m.physical.owned_processes(data,remote)):raise ValueError('Owned process absence differs')
def aggregate(samples,baseline):
 if len(samples)!=10000 or type(baseline) not in (int,float) or not math.isfinite(baseline) or baseline<=0:raise ValueError('Full matched loss required')
 losses=[sample['target']['score']['negativeLogLikelihood'] for sample in samples]
 if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in losses):raise ValueError('Invalid scored loss')
 mean=math.fsum(losses)/10000;delta=mean-baseline
 return dict(scored_targets=10000,mean_negative_log_likelihood=mean,matched_fp32_mean_negative_log_likelihood=baseline,increase_nats_per_token=delta,maximum_increase_nats_per_token=0.05,loss_gate_passed=delta<=0.05)
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
 baseline=metadata(ROOT/BASELINE)
 if pins[BASELINE]!=BASELINE_SHA or baseline['passed'] is not True or baseline['review']['accepted'] is not True:raise ValueError('Reviewed baseline identity differs')
 baselines={r['model']:r['baseline_mean_negative_log_likelihood'] for r in baseline['models']}
 selected=m.physical.selection(ROOT/PRIVATE,PRIVATE_SHA);prepared=ROOT/PREPARED;prepared_receipt=dict(path=PREPARED+'/integration.json',sha256=PREPARED_SHA)
 _,common,remote,signed=m.physical.admit_reuse(prepared,PREPARED_SHA,selected,q.NATIVE_SHA)
 m.integration.merge(common,m.integration.admit_build(ROOT/'artifacts/m7-apple-package-v5',q.NATIVE_SHA))
 for module in (m,m.package,m.integration,m.physical,m.storage):
  path=Path(module.__file__).resolve();m.integration.merge(common,{str(path.relative_to(ROOT)):c.sha(path)})
 for leaf in ('test_m7_physical_measurement.py','test_m7_physical_storage.py'):
  path=ROOT/'tools/experiments'/leaf;m.integration.merge(common,{str(path.relative_to(ROOT)):c.sha(path)})
 m.integration.merge(common,{PRIVATE:PRIVATE_SHA});tiny,_,tiny_inputs=m.job('tiny-cpu','short',verify_sources=False);m.integration.merge(common,tiny_inputs);c.merge(pins,common)
 context=dict(tiny_config=tiny,prepared_receipt=prepared_receipt,signed_bundle=signed,selection_path=PRIVATE,selection_sha=PRIVATE_SHA,selected=selected,remote=remote,required_bindings=common)
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
  expected_count=4 if plain else 5 if row['stage']=='tiny-quality' or row['stage']=='prepare' and row['model']=='larger' else 4 if row['stage']=='prepare' else 2
  if len(installs)!=expected_count:raise ValueError('Current installed-app guard count differs')
  for command in installs:c.require_current(metadata(ROOT/command['stdout']),remote,selected)
  for leaf in ('device-before.stdout','device-after.stdout'):
   if m.physical.admit_device(metadata(base/'private-logs'/leaf),selected)!=record['device']:raise ValueError('Device endpoints differ')
  if m.storage.boot_token(raw(base/'private-logs/host-boot.stdout',1024).decode())!=record['host_boot_token']:raise ValueError('Host boot differs')
  require_natural(record,base,metadata,remote,selected)
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
  samples=[json.loads(line.split('=',1)[1]) for line in (stdout+'\n'+stderr).splitlines() if line.startswith('TOKLET_QUALITY_SAMPLE_JSON=')]
  if len(samples)!=config['targets']:raise ValueError('Raw target count differs')
  if row['stage']=='quality':
   if record['transfer_receipt']!=ready_shas[key] or record['storage_admission']!=admissions[key]:raise ValueError('Original ready/storage admission differs')
   if any(Path(r['stdout']).stem.startswith('copy-') for r in record['commands']):raise ValueError('Unexpected quality copy')
  full=row['workload']=='full';result=checked['result'];observations=[s['footprint'] for s in samples]+[p['footprint'] for p in result['checkpoints']]
  loss=aggregate(samples,baselines[row['model']]) if full else None
  runs.append(dict(model=row['model'],workload=row['workload'],receipt=row['path'],sample_count=len(samples),loss=loss,
   raw_mean_negative_log_likelihood=math.fsum(s['target']['score']['negativeLogLikelihood'] for s in samples)/len(samples),
   peak_physical_bytes=max(f['lifetimePeakPhysicalBytes'] for f in observations),load_and_session_seconds=result['loadAndSessionSeconds'],
   thermal_states=sorted({s['environment']['thermalState'] for s in samples}),foreground_states=sorted({s['environment']['applicationState'] for s in samples}),
   low_power_modes=sorted({s['environment']['lowPowerMode'] for s in samples}),energy=None))
 if len(runs)!=6 or sum(r['sample_count'] for r in runs)!=20012:raise ValueError('Incomplete physical loss scope')
 c.verify(pins)
 result=dict(format='toklet-m7-physical-loss-analysis-v1',passed=True,bindings=pins,runs=runs,stages=14,commands=commands,scored_targets=20012,full_targets_per_model=10000,natural_absence_checks=28,loss_gates_passed=all(r['loss']['loss_gate_passed'] for r in runs if r['loss']),task_quality_qualified=False,physical_device_qualified=False,m7_complete=False,energy=None,scope='Matched frozen loss gate only; all other device and quality gates remain separate')
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=True,stages=14,runs=6,scored_targets=20012,loss_gates_passed=result['loss_gates_passed'])))
if __name__=='__main__':run()
