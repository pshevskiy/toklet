"""Replay completed current-app state and actual OS lifecycle; no device actions."""
from pathlib import Path
import hashlib,importlib.util,json,plistlib,subprocess,sys
R=Path.cwd();sys.path.insert(0,str(R/'tools/experiments'))
import m7_physical_measurement as m
import m7_physical_state as state
import m7_physical_lifecycle as life
HELPER=R/'artifacts/m7-phone-context-analysis-v1/analyze.py'
spec=importlib.util.spec_from_file_location('context_replay',HELPER);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
PREPARED='artifacts/m7-physical-prepared-v16';PREPARED_SHA='e1989a8d078a3660a9c516d7257b93147eca63640eb0d9feda1d63ea0bb4398e'
PRIVATE='artifacts/m7-physical-admission-v1/private-selection.json';PRIVATE_SHA='711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db'
NATIVE_SHA=state.NATIVE_SHA
OLD_SOURCE='artifacts/m7-phone-state-lifecycle-source-v2/source.json';OLD_SHA='ae5212770ad4e539d75d22240e3762fb8492fd23fee73ff7411fdf1a3390f7f6'
NEW_SOURCE='artifacts/m7-phone-state-lifecycle-source-v3/source.json';NEW_SHA='887a0f0df3223fd05fa3142d75f4b08dd5532745e5461b0eabc8f2b4167563ce'
TAIL_SOURCE='artifacts/m7-phone-state-lifecycle-source-v4/source.json';TAIL_SHA='59f3347223795d4f4a375abb05c087762655431d74d65518ad1424c61b61cc01'
PRIOR_EXECUTION='artifacts/m7-phone-state-lifecycle-driver-v2/execution.json';PRIOR_SHA='135d0061f7ef694a6727bbd27181744db4571d0a4b2b93dceaa21aff0c7b1cd6'
NATURAL=dict(absent=True,observations=[[]],signals=[],errors=[])
def need(ok,message):
 if not ok:raise ValueError(message)
def plans(version):
 if version==3:
  stem='artifacts/m7-phone-lifecycle-larger'
  return [dict(stage='state',model='larger',path='artifacts/m7-phone-state-larger-v3/measurement.json'),dict(stage='storage-proof',model='tiny-cpu',path=stem+'-storage-proof-v3/measurement.json'),dict(stage='lifecycle-prepare',model='larger',path=stem+'-ready-v3/measurement.json'),dict(stage='lifecycle',model='larger',path=stem+'-v3/lifecycle.json')]
 rows=[]
 if version==1:rows.append(dict(stage='tiny-integration',model='tiny-cpu+metal',path='artifacts/m7-state-lifecycle-tiny-v1/integration.json'))
 pairs=[('small','state'),('small','lifecycle'),('larger','state'),('larger','lifecycle')] if version==1 else [('small','lifecycle'),('larger','state'),('larger','lifecycle')]
 for model,mode in pairs:
  stem='artifacts/m7-phone-'+mode+'-'+model
  rows.extend([dict(stage='storage-proof',model='tiny-cpu',path=stem+'-storage-proof-v'+str(version)+'/measurement.json'),dict(stage=mode+'-prepare',model=model,path=stem+'-ready-v'+str(version)+'/measurement.json'),dict(stage=mode,model=model,path=stem+'-v'+str(version)+'/'+('lifecycle.json' if mode=='lifecycle' else 'measurement.json'))])
 return rows

def require_phase(phase,checked):
 need([v['phase'] for v in phase]==['ready-consumed','ready-process','settings-activated','background-process','harness-activated'],'Exact host roundtrip phases required')
 times=[v['observed_host_monotonic_nanoseconds'] for v in phase];need(all(type(v) is int and v>0 for v in times) and times==sorted(set(times)),'Ordered host phases required');pid=checked['launch']['pid']
 need(type(pid) is int and pid>0 and phase[0]['ready']==checked['ready'] and all(type(v['pid']) is int and v['pid']==pid for v in [phase[1],phase[3],phase[4]]),'Same original PID phase binding differs')
 return pid

def require_activation_argv(cmd,bundle):
 need(cmd==[m.physical.DEVICECTL,'device','process','launch','--device','<private-device_identifier>','--timeout','90','--quiet','--json-output','-','--activate',bundle],'Exact activation argv differs')

def require_complete(accepted,runs):
 need(len(accepted)==12 and len(runs)==4 and {(v['model'],v['mode']) for v in runs}=={(a,b) for a in ('small','larger') for b in ('state','lifecycle')},'Both complete model/mode witnesses required')

GATE='benchmarks/results/2026-10-21-m7-phone-small-sustained-qualification-v1.json'
GATE_SHA='8f6ced905719614568195b0eca6832e8e774d0df2156bea41f626b4560bc3ead'
def require_gate(gate):
 need(gate['format']=='toklet-m7-phone-small-sustained-qualification-v1' and gate['passed'] is True and gate['sustained_duration_qualified'] is True and gate['review']['accepted'] is True and gate['review']['reviewer']=='Astra' and type(gate['measured_native_seconds']) in (int,float) and gate['measured_native_seconds']>=600 and type(gate['measured_requests']) is int and gate['measured_requests']>=10 and gate['warmups']==3 and gate['hardware']['product_type']=='iPhone15,2' and gate['hardware']['os_version']=='27.0.1' and gate['physical_device_qualified'] is False and gate['m7_complete'] is False,'Accepted sustained prerequisite required')

def require_driver_inventory(e,source,source_name,reused,extra=()):
 wanted={*extra,source_name,GATE,PREPARED+'/integration.json',*reused,*source['sources'],*(v['snapshot'] for v in source['sources'].values()),*(r['path'] for r in e['results'])}
 need(set(e['driver_bindings'])==wanted and e['driver_bindings'][GATE]==GATE_SHA and e['driver_bindings'][PREPARED+'/integration.json']==PREPARED_SHA,'Exact driver prerequisite/source/stage closure required')

def require_labels(commands,wanted):
 need([Path(v['stdout']).stem for v in commands]==wanted,'Exact ordered command rows required')

def require_reset_labels(commands):
 labels=[Path(v['stdout']).stem for v in commands]
 need([v for v in labels if v.startswith(('process-','signal-'))]==['process-1','signal-2','process-3','process-4'],'Exact reset process/signal rows required')

def run():
 own=Path(__file__).resolve();out=own.parent/'analysis.json';need(not out.exists() and not out.is_symlink(),'Exclusive analysis output required')
 pins={};buffers={};history={}
 def raw(path,limit=64*1024**2):
  if path not in buffers:buffers[path]=c.read_bound(path,limit,pins)
  need(len(buffers[path])<=limit,'Bounded analysis input required');return buffers[path]
 def metadata(path):return json.loads(raw(path))
 def add(extra):
  for n,h in extra.items():
   if n in history and c.sha(R/n)!=h:
    n=history[n]['snapshot'];need(c.sha(R/n)==h,'Historical binding differs')
   c.merge(pins,{n:h})
 for path in [own,own.with_name('checks.py'),HELPER,*[Path(v.__file__).resolve() for v in (m,state,life,m.package,m.integration,m.physical,m.storage,state.replay)]]:raw(path)
 sources=[]
 for name,expected,old in [(OLD_SOURCE,OLD_SHA,True),(NEW_SOURCE,NEW_SHA,False),(TAIL_SOURCE,TAIL_SHA,False)]:
  s=metadata(R/name);need(pins[name]==expected and len(s['sources'])==111 and sum(v['matches_collection_head'] for v in s['sources'].values())==109,'Exact source manifest required')
  for n,v in s['sources'].items():
   path=R/v['snapshot'];need(path.resolve().is_relative_to((R/name).parent.resolve()) and not path.is_symlink(),'Confined exact source required')
   value=raw(path);need(c.sha(path)==v['sha256'],'Snapshot digest differs')
   if v['matches_collection_head']:need(subprocess.check_output(['git','show',s['engine_revision']+':'+n])==value,'Collection HEAD source differs')
   if c.sha(R/n)!=v['sha256']:
    need(old and n in ('tools/experiments/m7_physical_lifecycle.py','tools/experiments/test_m7_physical_lifecycle.py'),'Unexpected historical source change');history[n]=v
   else:add({n:v['sha256']})
  sources.append(s)
 selected=m.physical.selection(R/PRIVATE,PRIVATE_SHA);prepared=R/PREPARED;prepared_ref=dict(path=PREPARED+'/integration.json',sha256=PREPARED_SHA)
 _,reused,remote,signed=m.physical.admit_reuse(prepared,PREPARED_SHA,selected,NATIVE_SHA)
 gate=metadata(R/GATE);need(pins[GATE]==GATE_SHA,'Frozen sustained prerequisite bytes differ');require_gate(gate)
 executions=[]
 for version,source in [(1,sources[0]),(2,sources[1]),(3,sources[2])]:
  d='artifacts/m7-phone-state-lifecycle-driver-v'+str(version);e=metadata(R/d/'execution.json');expected=plans(version)[:7] if version==1 else plans(version)[:6] if version==2 else plans(version)
  need(len(e['results'])==len(expected) and not e['endpoint_errors'] and e['physical_device_qualified'] is False and e['m7_complete'] is False,'Exact driver completion scope required')
  error='ValueError: Physical workflow stage failed: lifecycle' if version==1 else 'ValueError: Physical workflow stage failed: state' if version==2 else None
  need(e['passed'] is (version==3) and e['primary_error']==error,'Driver failure/success differs')
  for i,(row,wanted) in enumerate(zip(e['results'],expected)):
   need({k:row[k] for k in wanted}==wanted and type(row['exit_code']) is int and row['exit_code']==(1 if (version,i) in ((1,6),(2,5)) else 0),'Exact driver stage sequence required')
   need(e['driver_bindings'].get(row['path'])==row['sha256'],'Driver stage binding differs');need(c.sha(R/row['path'])==row['sha256'],'Stage digest differs')
  for n,v in source['sources'].items():need(e['driver_bindings'].get(n)==v['sha256'] and e['driver_bindings'].get(v['snapshot'])==v['sha256'],'Source driver binding differs')
  extra={*executions[1]['driver_bindings'],PRIOR_EXECUTION} if version==3 else set()
  require_driver_inventory(e,source,[OLD_SOURCE,NEW_SOURCE,TAIL_SOURCE][version-1],reused,extra)
  if version==3:need(e['driver_bindings'].get(PRIOR_EXECUTION)==PRIOR_SHA and pins[PRIOR_EXECUTION]==PRIOR_SHA,'Exact preserved prior install-guard failure required')
  add(e['driver_bindings']);executions.append(e)
 common=dict(reused);add(common)
 m.integration.merge(common,m.integration.admit_build(R/'artifacts/m7-apple-package-v5',NATIVE_SHA))
 for v in (m,m.package,m.integration,m.physical,m.storage):
  p=Path(v.__file__).resolve();m.integration.merge(common,{str(p.relative_to(R)):c.sha(p)})
 for leaf in ('test_m7_physical_measurement.py','test_m7_physical_storage.py'):
  p=R/'tools/experiments'/leaf;m.integration.merge(common,{str(p.relative_to(R)):c.sha(p)})
 m.integration.merge(common,{PRIVATE:PRIVATE_SHA});tiny,_,tiny_inputs=m.job('tiny-cpu','short',verify_sources=False);m.integration.merge(common,tiny_inputs);add(common)
 context=dict(tiny_config=tiny,prepared_receipt=prepared_ref,signed_bundle=signed,selection_path=PRIVATE,selection_sha=PRIVATE_SHA,selected=selected,remote=remote,required_bindings=common)
 def private_equal(record,base):
  private=m.physical.redact_record(metadata(base/'private-execution.json'),selected,remote)
  for k in ('commands','primary_error','endpoint_errors','cleanup_errors'):need(private[k]==record[k],'Private/public command record differs')
 def endpoints(record,base):
  for leaf in ('device-before.stdout','device-after.stdout'):need(m.physical.admit_device(metadata(base/'private-logs'/leaf),selected)==record['device'],'Device endpoints differ')
 def process(path):
  d=metadata(path);need(d['info']['outcome']=='success' and d['info']['commandType']=='devicectl.device.info.processes' and d['result']['deviceIdentifier']==selected['device_identifier'],'Process query identity differs');return m.physical.owned_processes(d,remote)
 def require_natural(record,base,lifecycle=False):
  need(record['remote_cleanup']==[NATURAL,NATURAL],'Two natural absences required')
  for label in (('preflight-process','cleanup-process-1') if lifecycle else ('process-1','process-2')):need(process(base/'private-logs'/(label+'.stdout'))==[],'Natural absence raw query differs')
  names=[Path(v['stdout']).stem for v in record['commands']];need(not any(v.startswith(('signal-','cleanup-signal-')) for v in names),'Unexpected phone signals')
 # Separate prepared tiny CPU/Metal integration, retained from the first driver.
 row=executions[0]['results'][0];base=(R/row['path']).parent;r=metadata(R/row['path'])
 need(r['passed'] is True and r['reviewed_native_package_sha256']==NATIVE_SHA and r['signed_bundle']==signed and not r['primary_error'] and not r['endpoint_errors'] and not r['cleanup_errors'],'Tiny status differs')
 need(r['format']=='toklet-m7-physical-tiny-integration-v1' and r['physical_device_qualified'] is False and r['m7_complete'] is False and r['private_selection_sha256']==PRIVATE_SHA and r['reused_install_receipt']==prepared_ref,'Exact tiny scope/installation lineage required')
 tiny_expected=dict(reused)
 for name in ('tools/experiments/m7_physical_integration.py','tools/experiments/test_m7_physical_integration.py'):tiny_expected[name]=c.sha(R/name)
 need(r['bindings']==tiny_expected,'Exact tiny required source/input closure differs')
 wanted={str(p.relative_to(R)) for p in [base/'private-execution.json',*(base/'private-logs').iterdir()] if p.is_file()};need(set(r['private_evidence_pins'])==wanted,'Exact tiny private inventory differs')
 c.require_commands(r['commands']);add(r['bindings']);add(r['private_evidence_pins']);private_equal(r,base);endpoints(r,base)
 require_labels(r['commands'],['device-before','processes-1','architecture','deployment','signature-verify','provisioning','entitlements','signature-details','device-preinstall','installed-cpu','cpu-self-test','processes-2','installed-metal','metal-self-test','processes-3','processes-4','device-after'])
 need(m.physical.admit_device(metadata(base/'private-logs/device-preinstall.stdout'),selected)==r['device'],'Tiny preinstall endpoint differs')
 for label in ('installed-cpu','installed-metal'):c.require_current(metadata(base/'private-logs'/(label+'.stdout')),remote,selected)
 signing=m.physical.admit_profile(plistlib.loads(raw(base/'private-logs/provisioning.stdout')),plistlib.loads(raw(base/'private-logs/entitlements.stdout')),raw(base/'private-logs/signature-details.stderr').decode(),selected);need(signing==r['signing'],'Tiny signing differs')
 for backend in ('cpu','metal'):
  checked=m.physical.check_console(raw(base/'private-logs'/(backend+'-self-test.stdout')).decode()+'\n'+raw(base/'private-logs'/(backend+'-self-test.stderr')).decode(),backend,signed)
  checked['launch']=m.physical.check_launch(metadata(base/'private-logs'/(backend+'-launch.json')),remote,selected);need(checked==r['results'][backend],'Tiny raw result differs')
 need(r['remote_cleanup']==[NATURAL]*3,'Tiny natural endpoints differ')
 for i in (1,2,3,4):need(process(base/'private-logs'/('processes-'+str(i)+'.stdout'))==[],'Tiny natural absence differs')
 commands=len(r['commands']);natural=4;resets=0;runs=[];ready={};admissions={};accepted=executions[0]['results'][1:4]+executions[1]['results'][:5]+executions[2]['results']
 for row in accepted:
  path=R/row['path'];base=path.parent;r=metadata(path);plain=row['stage']=='storage-proof';prepare=row['stage'].endswith('-prepare');mode='state' if row['stage'].startswith('state') else 'lifecycle';diag=state if mode=='state' else life
  config,_,inputs=m.job('tiny-cpu','short',verify_sources=False) if plain else diag.job(row['model'],'512',verify_sources=False)
  c.require_stage(r,config,signed,prepared_ref);need(r['physical_device_qualified'] is False and r['m7_complete'] is False,'Raw collection scope differs')
  need(r['format']==('toklet-m7-physical-measurement-v1' if plain else diag.READY_FORMAT if prepare else diag.RESULT_FORMAT),'Stage format differs')
  for n,h in {**common,**inputs}.items():need(r['bindings'].get(n)==h,'Stage source absent: '+n)
  c.require_commands(r['commands']);commands+=len(r['commands']);add(r['bindings']);private_equal(r,base);endpoints(r,base)
  if 'private_evidence_pins' in r:
   wanted={str(p.relative_to(R)) for p in [base/'private-execution.json',*(base/'private-logs').iterdir()] if p.is_file()};need(set(r['private_evidence_pins'])==wanted,'Private evidence inventory differs');add(r['private_evidence_pins'])
  else:
   for p in [base/'private-execution.json',*(base/'private-logs').iterdir()]:need(r['bindings'].get(str(p.relative_to(R)))==c.sha(p),'Lifecycle private binding absent')
  if row['path'] in ('artifacts/m7-phone-lifecycle-small-storage-proof-v2/measurement.json','artifacts/m7-phone-state-larger-v3/measurement.json'):
   require_reset_labels(r['commands']);initial=process(base/'private-logs/process-1.stdout');need(len(initial)==1,'Exact first preflight owned PID required');pid=initial[0]
   reset=dict(absent=True,observations=[[pid],[]],signals=[dict(pid=pid,signal='SIGTERM')],errors=[])
   need(r['remote_cleanup']==[reset,NATURAL] and process(base/'private-logs/process-3.stdout')==[] and process(base/'private-logs/process-4.stdout')==[],'Owned pre-copy reset differs')
   signal=metadata(base/'private-logs/signal-2.stdout');proc=signal['result']['process']
   need(signal['info']['outcome']=='success' and signal['info']['commandType']=='devicectl.device.process.signal' and signal['result']['deviceIdentifier']==selected['device_identifier'] and proc['processIdentifier']==pid and m.physical.remote_path(proc['executable'])==remote+'/TokletHarness' and signal['result']['signal']==dict(name='SIGTERM',value=15),'Exact owned signal raw result differs')
   cmd=next(v['command'] for v in r['commands'] if Path(v['stdout']).stem=='signal-2');need(cmd==[m.physical.DEVICECTL,'device','process','signal','--pid',str(pid),'--signal','SIGTERM','--device','<private-device_identifier>','--timeout','90','--quiet','--json-output','-'],'Exact owned signal argv differs')
   labels=[Path(v['stdout']).stem for v in r['commands']]
   if plain:order=['current-install-6','process-1','signal-2','process-3','copy-Model','measurement','process-4']
   else:order=['current-install-2','process-1','signal-2','process-3','current-install-6','measurement','process-4']
   need(all(labels.count(v)==1 for v in order) and [labels.index(v) for v in order]==sorted(labels.index(v) for v in order),'Owned reset must precede launch/copy');natural+=1;resets+=1
  else:require_natural(r,base,lifecycle=not plain and not prepare and mode=='lifecycle');natural+=2
  installs=[v for v in r['commands'] if Path(v['stdout']).stem.startswith('current-install-')];count=4 if plain else (4 if row['model']=='larger' else 3) if prepare else 2
  need(len(installs)==count,'Exact installed guard count differs')
  for v in installs:c.require_current(metadata(R/v['stdout']),remote,selected)
  if 'host_boot_token' in r:need(m.storage.boot_token(raw(base/'private-logs/host-boot.stdout',1024).decode())==r['host_boot_token'],'Host boot differs')
  key=(row['model'],mode)
  if plain:
   _,extra=m.storage.admit_proof(base,row['sha256'],context,m.validate_console);add(extra)
   checked=m.validate_console(raw(base/'private-logs/measurement.stdout').decode()+'\n'+raw(base/'private-logs/measurement.stderr').decode(),config,signed);checked['launch']=m.physical.check_launch(metadata(base/'private-logs/launch.json'),remote,selected);need(checked==r['checked'],'Tiny storage raw differs');continue
  if prepare:
   extra,name,digest=m.admit_ready(base,row['sha256'],config,inputs,prepared,PREPARED_SHA,signed,diag.READY_FORMAT);add(extra)
   admission,extra=m.storage.admit_ready(r,base/'transfer',config,name,digest,context,m.validate_console);add(extra);ready[key]=(base,row);admissions[key]=admission;continue
  readybase,readyrow=ready[key];need(r['transfer_receipt']==dict(path=readyrow['path'],sha256=readyrow['sha256']) and r['storage_admission']==admissions[key],'Original ready/storage binding differs')
  need(not any(Path(v['stdout']).stem.startswith('copy-') for v in r['commands']),'Unexpected run copy')
  signing=m.physical.admit_profile(plistlib.loads(raw(readybase/'private-logs/profile.stdout')),plistlib.loads(raw(readybase/'private-logs/entitlements.stdout')),raw(readybase/'private-logs/signature-details.stderr').decode(),selected);need(signing==r['signing'],'Signing observation differs')
  label='measurement' if mode=='state' else 'lifecycle';checked=diag.validate_console(raw(base/'private-logs'/(label+'.stdout')).decode()+'\n'+raw(base/'private-logs'/(label+'.stderr')).decode(),config,signed);checked['launch']=m.physical.check_launch(metadata(base/'private-logs/launch.json'),remote,selected);need(checked==r['checked'],'Raw complete result/launch differs')
  result=checked['result']
  if mode=='state':
   by={v['name']:v for v in result['rows']};cancel=by['prefill-cancelled']['state'];footprints=[v['footprint'] for v in result['rows']]+[v['footprint'] for v in result['checkpoints']];environments=[v['environment'] for v in result['rows']]
   summary=dict(states=len(by),partial_prefill=dict(history=cancel['history'],evaluated=cancel['evaluated'],pending=cancel['pending']),fully_committed_capacity=by['capacity-committed']['state']['evaluated'],overflow_rejections=3,unchanged_scalar_diagnostics=True,exact_fresh_reference_ids=True,closed_zero_state=True)
  else:
   require_labels(r['commands'],['device-before','current-install-before','preflight-process','lifecycle','ready-process','activate-settings','background-process','current-install-return','activate-harness','cleanup-process-1','device-after'])
   need(r['actual_os_lifecycle_qualified'] is False,'Raw lifecycle scope differs');phase=r['phase_observations'];pid=require_phase(phase,checked)
   for label in ('ready-process','background-process'):need(process(base/'private-logs'/(label+'.stdout'))==[pid],'Same live raw PID differs')
   life.activation(metadata(base/'private-logs/activate-settings.stdout'),selected);life.activation(metadata(base/'private-logs/activate-harness.stdout'),selected,remote,pid)
   for label,bundle in [('activate-settings','com.apple.Preferences'),('activate-harness',m.physical.BUNDLE)]:
    cmd=next(v['command'] for v in r['commands'] if Path(v['stdout']).stem==label);require_activation_argv(cmd,bundle)
   events=result['trace']['events'];footprints=[v['footprint'] for v in events]+[v['footprint'] for v in result['checkpoints']];environments=[result['environmentBefore'],result['environmentAfter']]
   summary=dict(events=len(events),same_original_process=True,active_foreground_interruption=True,actual_scene_roundtrip=True,retained_sampled_ids=len(result['rebuiltIDs']),replayed_prefix_tokens=len(result['interruptedPrefix']),exact_continuation_reference=True,synthetic_warning_only=True,closed_and_released=True)
  runs.append(dict(model=row['model'],mode=mode,receipt=row['path'],summary=summary,peak_physical_bytes=max(v['lifetimePeakPhysicalBytes'] for v in footprints),thermal_states=sorted({v['thermalState'] for v in environments}),low_power_modes=sorted({v['lowPowerMode'] for v in environments}),energy=None))
 require_complete(accepted,runs)
 c.verify(pins)
 result=dict(format='toklet-m7-phone-state-lifecycle-analysis-v1',passed=True,bindings=pins,historical_source_bindings=history,runs=runs,accepted_stages=13,accepted_commands=commands,natural_absence_checks=natural,owned_preflight_resets=resets,preserved_failed_driver_stage_counts=[7,6],state_witnesses_qualified=True,actual_os_lifecycle_qualified=True,system_memory_warning_delivery_qualified=False,jetsam_recovery_qualified=False,physical_device_qualified=False,m7_complete=False,energy=None,scope='Current signed app bounded state/actual scene interruption/replay; synthetic warning handler only; independent base-device/M7 qualification remains open')
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=True,stages=13,runs=4,commands=commands,natural_absence_checks=natural)))
if __name__=='__main__':run()
