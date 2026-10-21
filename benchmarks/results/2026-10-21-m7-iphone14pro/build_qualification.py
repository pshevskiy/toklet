from pathlib import Path
import hashlib,json,math,subprocess
R=Path.cwd();stem='2026-10-21-m7-iphone14pro';O=R/'benchmarks/results'/stem;O.mkdir();pins={};copies={};children={}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def consume(n):
 raw=(R/n).read_bytes();h=hashlib.sha256(raw).hexdigest();assert n not in pins or pins[n]==h;pins[n]=h;return json.loads(raw)
for role,suffix in [('context','phone-context'),('loss','phone-loss'),('tasks','phone-tasks'),('sustained','phone-small-sustained'),('state_lifecycle','phone-state-lifecycle'),('energy','phone-energy')]:
 n=f'benchmarks/results/2026-10-21-m7-{suffix}-qualification-v1.json';q=consume(n);assert q['passed'] is True
 if role=='context':assert q['review'].startswith('Astra accepted') and q['identical_generated_sequences'] is True and q['request_samples']==156 and q['runs']==12
 else:assert q['review']['accepted'] is True and q['review']['reviewer']=='Astra'
 assert q['hardware']['product_type']=='iPhone15,2' and q['hardware']['os_version']=='27.0.1';children[role]=dict(path=n,sha256=pins[n],data=q)
loss=children['loss']['data'];tasks=children['tasks']['data'];sustain=children['sustained']['data'];state=children['state_lifecycle']['data'];energy=children['energy']['data'];context=children['context']['data']
assert loss['loss_quality_qualified'] and loss['full_targets_per_model']==10000 and len(loss['full_losses'])==2
assert tasks['task_quality_qualified'] and tasks['full_cases_per_model']==100 and {(r['model'],r['passed'],r['fp32_passed']) for r in tasks['models']}=={('small',2,2),('larger',11,11)} and all(r['drop_percentage_points']==0 for r in tasks['models'])
assert sustain['sustained_duration_qualified'] and sustain['measured_native_seconds']>=600 and sustain['measured_requests']==24 and sustain['warmups']==3
assert state['state_witnesses_qualified'] and state['actual_os_lifecycle_qualified'] and state['accepted_stages']==13 and state['accepted_commands']==191 and state['natural_absence_checks']==26 and state['owned_preflight_resets']==2
assert energy['bounded_observation_complete'] and not energy['actual_recording_available_in_tested_setup'] and energy['energy_joules'] is None and energy['energy_qualified'] is False and [r['instrument_exit'] for r in energy['attempts']]==[21,19]
analysis_path=next(v['origin'] for v in context['exact_published_copies'].values() if v['origin']=='artifacts/m7-phone-context-analysis-v1/analysis.json');a=consume(analysis_path);assert pins[analysis_path]==context['evidence_pins'][analysis_path] and a['passed']
durations=[]
for row in a['runs']:
 if row['model']=='larger' and row['workload'] in ('512','1024'):
  seconds=math.fsum(s['native_last_token_seconds'] for s in row['measured']['raw']);assert seconds>=600;durations.append(dict(workload=row['workload'],receipt=row['receipt'],measured_native_seconds=seconds))
assert len(durations)==4
models={}
for r in state['runs']:
 if r['mode']!='state':continue
 receipt=consume(r['receipt']);assert pins[r['receipt']]==state['evidence_pins'][r['receipt']];config=receipt['config'];assert config['capacity']==1152 and config['chunk']==128 and config['maximum']==128 and len(config['prompt'])==512
 assert all(v['config']['files']==config['files'] for v in a['runs'] if v['model']==r['model'])
 models[r['model']]=dict(name='Qwen2.5-0.5B-Instruct' if r['model']=='small' else 'Qwen2.5-1.5B-Instruct',artifact_files=config['files'],context_state_capacity=1152,context_state_chunk=128,measured_prompt_tokens=[128,512,1024],context_response_ids=128,numerical_admission_cap_bytes=config['numericalBudget'],process_trial_cap_bytes=config['processBudget'],attention='Online',kv='FP16',readback='FullLogits',weights='compensated Half / Original normalization' if r['model']=='small' else 'frozen group8 Q4v2 compensated tiled / Compensated normalization')
assert len(models)==2
# Existing numerical source is identical from full phone-quality collection to now.
current=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();numerical=subprocess.check_output(['git','ls-files','src','Cargo.toml','Cargo.lock'],text=True).splitlines();lineage={}
for n in numerical:
 raw=(R/n).read_bytes();assert subprocess.check_output(['git','show','721fbea:'+n])==raw and subprocess.check_output(['git','show','c2e44fb:'+n])==raw;lineage[n]=hashlib.sha256(raw).hexdigest();pins[n]=lineage[n]
 # Matrix package differs only through additive native diagnostics/tests.
 if n not in ('src/native.rs','src/native/tests.rs'):assert subprocess.check_output(['git','show','b2dee4a:'+n])==raw
peak=max(context['peak_physical_bytes'],loss['peak_physical_bytes'],tasks['peak_physical_bytes'],sustain['run_summary']['peak_physical_bytes'],*(r['peak_physical_bytes'] for r in state['runs']));assert peak==1944701920 and peak<2560*1024**2
src=Path(__file__).resolve();dst=O/'build_qualification.py';dst.write_bytes(src.read_bytes());h=sha(dst);copies[str(dst.relative_to(R))]=dict(origin=str(src),sha256=h);pins[str(dst.relative_to(R))]=h
q=dict(format='toklet-m7-iphone14pro-qualification-v1',passed=True,hardware=dict(marketing_model='iPhone14Pro',product_type='iPhone15,2',os_version='27.0.1'),decision_revision=current,source_qualification_receipts={k:{x:v[x] for x in ('path','sha256')} for k,v in children.items()},evidence_pins=pins,exact_published_copies=copies,numerical_source_pins=lineage,models=models,full_loss_targets_per_model=10000,full_task_cases_per_model=100,task_counts=dict(small=2,larger=11),small_sustained_measured_native_seconds=sustain['measured_native_seconds'],larger_sustained_observations=durations,largest_uninstrumented_process_peak_bytes=peak,quality_policies=dict(loss=dict(capacity=1025,chunk=1,target_block=1024),tasks=dict(capacity=562,chunk=128,category_caps=[256,32,16],maximum_ids_per_command=128)),state_and_actual_scene_lifecycle_qualified=True,synthetic_memory_warning_handler_tested=True,real_system_memory_warning_delivery_qualified=False,jetsam_recovery_qualified=False,energy_observation_complete=True,energy_joules=None,energy_qualified=False,controlled_speedup_qualified=False,supported_trial_limits_qualified=True,physical_device_qualified=True,original_plan_target='base iPhone14',original_plan_target_qualified=False,m7_complete=False,review=dict(accepted=False,reviewer='Astra',scope='Joined Pro-only supported trial limits; original base-device milestone remains open pending user scope choice'))
(R/'benchmarks/results'/f'{stem}-qualification-v1.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(dict(passed=True,children=len(children),numerical_sources=len(lineage),peak=peak,larger_sustained_observations=len(durations),m7_complete=False)))
