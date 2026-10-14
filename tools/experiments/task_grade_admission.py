"""Full backend-aware frozen task admission. No generated-code execution here."""
import hashlib
from pathlib import Path
import sys
import cpu_task_reference as reference
import metal_tasks as metal
import metal_controlled as controlled
import metal_engine
sys.path.insert(0,str(controlled.ROOT/'tools/bench'))
import task_cases as tasks
import linux_task_evidence as evidence
import linux_code_isolation_probe as boundary
ROOT=controlled.ROOT
REFERENCE_QUAL=ROOT/'benchmarks/results/2026-10-14-cpu-task-reference-qualification-v1.json'
REFERENCE_QUAL_SHA='c9a9ff08e4fef7143ba0af4c4e3bd6e7e023d89b1f114504926fd3bbb12ff09a'
MAX_SUMMARY=32*1024**2
MAX_RAW=8*1024**2


def bounded(path,limit):
 with Path(path).open('rb') as stream:data=stream.read(limit+1)
 if len(data)>limit:raise ValueError('Task grading evidence exceeds bound')
 return data


def validate_records(record,cases,events):
 if len(cases)!=100 or record.get('passed') is not True or record.get('collection_completed') is not True or record.get('check_only') is not False or record.get('quality_completed') is not False or record.get('identities_unchanged') is not True or record.get('endpoint_errors')!=[] or record.get('task_payload_sha256')!=tasks.TASK_SHA or record.get('task_ids')!=[v['id'] for v in cases]:raise ValueError('Complete frozen100-case collection required')
 run=record.get('run',{})
 if run.get('passed') is not True or run.get('model_exited') is not True or type(run.get('exit_code')) is not int or run['exit_code']!=0 or run.get('cleanup_errors')!=[] or len(run.get('cases',[]))!=100:raise ValueError('Complete naturally exited owned model required')
 if type(run.get('worker_pid')) is not int or run['worker_pid']<=1:raise ValueError('Owned model PID differs')
 kind=record.get('format');responses=[];expected=[run['ready']]
 if kind=='toklet-metal-task-collection-v1':
  model=record.get('model')
  if model not in controlled.MODELS or record.get('backend')!='metal' or record.get('projection_mode')!='HalfCompensated' or record.get('normalization_mode')!='Original' or record.get('readback_mode')!='FullLogits':raise ValueError('Actual Metal collection metadata differs')
  metal.ready(run['ready'],model)
  for case,item in zip(cases,run['cases'],strict=True):
   if type(item.get('max_output_tokens')) is not int or item['max_output_tokens']!=case['max_output_tokens']:raise ValueError('Full Metal task cap differs')
   response=metal.collect_response(case,item['native_events'],run['ready']);expected.extend(item['native_events'])
   if response!=item['response']:raise ValueError('Metal saved response differs from native stream')
   responses.append(response)
  identity={'backend':'metal','model':model,'projection_mode':'HalfCompensated','normalization_mode':'Original','readback_mode':'FullLogits'}
  if len(run['idle_usage'])!=101:raise ValueError('Metal idle charge coverage differs')
 elif kind=='toklet-cpu-task-reference-collection-v1':
  reference.ready(run['ready']);checks=run.get('checks')
  if type(checks) is not list or len(checks)!=1 or checks[0]['request']!={'action':'generate','tokens':[1]*307,'max_new_tokens':256}:raise ValueError('Raw reference overcapacity preflight differs')
  e=checks[0]['response'];empty={'history_tokens':0,'evaluated_tokens':0,'pending_tokens':0,'history_sha256':reference.history_sha([])}
  if e.get('event')!='error' or e.get('state')!=empty or e.get('memory')!=run['ready']['memory']:raise ValueError('Raw reference rejected input changed state')
  expected.append(e)
  for case,item in zip(cases,run['cases'],strict=True):
   response=reference.validate_response(case,item['native_events'],run['ready']);expected.extend(item['native_events'])
   if any(item.get(k)!=v for k,v in response.items()):raise ValueError('Raw reference saved response differs')
   response.update(visible_text=item['visible_text'],visible_text_sha256=item['visible_text_sha256']);responses.append(response)
  phase=run.get('decoder_phase',{})
  if phase.get('after_owned_model_exit') is not True or type(phase.get('model_resident_bytes')) is not int or phase['model_resident_bytes']!=0 or phase.get('tokenizer_allowance_bytes')!=reference.TOKENIZER_ALLOWANCE or phase.get('application_allowance_bytes')!=reference.APP or phase.get('admitted_bytes')!=reference.TOKENIZER_ALLOWANCE+reference.APP:raise ValueError('Raw reference decoder phase differs')
  peak=phase.get('native_driver',{}).get('lifetime_max_phys_footprint')
  if type(peak) is not int or not 0<peak<=controlled.LIMIT:raise ValueError('Raw reference decoder charge differs')
  identity={'backend':'cpu','model':'larger','projection_mode':'tiled','normalization_mode':'OrderedFP64','readback_mode':'FullLogits'}
  if len(run['idle_usage'])!=102:raise ValueError('Raw reference idle charge coverage differs')
 else:raise ValueError('Unsupported task backend collection')
 if events!=expected:raise ValueError('Complete raw event stream differs')
 for usage in run['idle_usage']:
  a,b=(usage[key].get('lifetime_max_phys_footprint') for key in ('worker','driver'))
  if type(a) is not int or type(b) is not int or type(usage.get('joint_native_lifetime_peak_bytes')) is not int or not 0<a or not 0<b or usage['joint_native_lifetime_peak_bytes']!=a+b or not a+b<=controlled.LIMIT:raise ValueError('Task model-phase native charge differs')
 return {'identity':identity,'responses':responses,'worker_pid':run.get('worker_pid'),'record':record}


def reference_binary():
 payload=bounded(REFERENCE_QUAL,2*1024**2)
 if hashlib.sha256(payload).hexdigest()!=REFERENCE_QUAL_SHA:raise ValueError('Frozen raw reference qualification differs')
 q=evidence.decode_event(payload.decode());binary=ROOT/'artifacts/m5-cpu-task-reference-native-binary-v1/cpu_task_reference'
 if q.get('passed') is not True or q.get('format')!='toklet-cpu-task-reference-qualification-v1' or metal_engine.digest(binary)!=q['bindings'][str(binary.relative_to(ROOT))]['sha256']:raise ValueError('Qualified raw reference executable differs')
 prefix='artifacts/m5-cpu-task-reference-source-v1/'
 for name,value in q['bindings'].items():
  if name.startswith(prefix):
   if metal_engine.digest(ROOT/name)!=value['sha256']:raise ValueError('Compiled reference snapshot differs')
   if name.startswith(prefix+'src/') and metal_engine.digest(ROOT/name[len(prefix):])!=value['sha256']:raise ValueError('Current reference numerical source differs')
 return binary,q['bindings'][str(binary.relative_to(ROOT))]['sha256']


def frozen_manifest():
 payload=bounded(controlled.prepare.CONTRACT/'manifest.json',1024**2)
 if hashlib.sha256(payload).hexdigest()!=controlled.CONTRACT_SHA:raise ValueError('Frozen task contract differs')
 return evidence.decode_event(payload.decode())


def qualified_binary(kind):
 if kind=='toklet-cpu-task-reference-collection-v1':return reference_binary()
 payload=bounded(metal.QUALIFICATION,2*1024**2)
 if hashlib.sha256(payload).hexdigest()!=metal.QUALIFICATION_SHA:raise ValueError('Frozen CLI qualification differs')
 q=evidence.decode_event(payload.decode());binary=metal.qualified_cli()
 return binary,q['bindings'][str(binary.relative_to(ROOT))]['sha256']


def inventory_paths(kind,binary,source,manifest):
 """Exact producer inventory, including transitive helper and source dependencies."""
 common=[binary,ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/manifest['quality']['tasks_file'],ROOT/'benchmarks/sources.json',ROOT/'benchmarks/result.schema.json',ROOT/'tools/bench/requirements.lock',controlled.prepare.CONTRACT/'manifest.json']
 modules=(tasks,tasks.prepare,controlled,controlled.frozen,controlled.common,controlled.metal_prefill,controlled.metal_continue,metal_engine,reference.native_usage)
 files=common+[Path(v.__file__) for v in modules]+list((ROOT/'src').rglob('*.rs'))+[controlled.prepare.CONTRACT/v['file'] for v in manifest['arrays'].values()]+[source/name for name in ('model.safetensors','config.json',*metal.cli.ASSET_PINS)]
 if kind=='toklet-metal-task-collection-v1':files += [Path(metal.__file__),Path(metal.cli.__file__),Path(reference.__file__),metal.QUALIFICATION]+list((ROOT/'src').rglob('*.metal'))
 else:files += [Path(reference.__file__),ROOT/'examples/cpu_task_reference.rs']
 return {str(p) for p in files}


def validate_inventory(identities,kind,binary,binary_sha,source,manifest,model):
 if type(identities) is not dict or set(identities)!=inventory_paths(kind,binary,source,manifest):raise ValueError('Complete backend task provenance inventory required')
 if any(not Path(name).resolve().is_relative_to(ROOT) or type(sha) is not str or len(sha)!=64 for name,sha in identities.items()):raise ValueError('Collection inventory escapes frozen root')
 controlled.initial_pins(identities,manifest,source,controlled.MODELS[model])
 required={binary:binary_sha,ROOT/manifest['quality']['tasks_file']:tasks.TASK_SHA,**{source/name:sha for name,sha in metal.cli.ASSET_PINS.items()}}
 if kind=='toklet-metal-task-collection-v1':required[metal.QUALIFICATION]=metal.QUALIFICATION_SHA
 if any(identities.get(str(file))!=sha for file,sha in required.items()):raise ValueError('Required qualified task provenance differs')
 for name,sha in identities.items():
  if metal_engine.digest(Path(name))!=sha:raise ValueError('Collection source/input identity changed')


def validate_collection(path):
 path=Path(path).resolve();payload=bounded(path,MAX_SUMMARY);record=evidence.decode_event(payload.decode());raw=bounded(path.with_suffix('.raw.jsonl'),MAX_RAW);events=[evidence.decode_event(line) for line in raw.decode().splitlines()];cases=tasks.load_cases();validated=validate_records(record,cases,events)
 kind=record['format'];model=validated['identity']['model'];source=ROOT/'models'/controlled.MODELS[model][0];manifest=frozen_manifest()
 binary,binary_sha=qualified_binary(kind)
 command=([str(binary),'--model',str(source),'--backend','metal','--metal-projection','half-compensated','--interactive','--json','--context-capacity','2048','--prefill-chunk','128'] if kind=='toklet-metal-task-collection-v1' else [str(binary),str(source)])
 if record['run']['command']!=command:raise ValueError('Qualified task executable/model command differs')
 validate_inventory(record.get('identities'),kind,binary,binary_sha,source,manifest,model)
 # Interpret text only after the model is proved naturally exited in its record.
 if kind=='toklet-metal-task-collection-v1':metal.finalized(validated['responses'],source/'tokenizer.json',model_exited=True)
 else:
  copies=reference.finalize([{'visible_token_ids':v['visible_token_ids']} for v in validated['responses']],source/'tokenizer.json',model_exited=True)
  if any(a['visible_text']!=b['visible_text'] or a['visible_text_sha256']!=b['visible_text_sha256'] for a,b in zip(copies,validated['responses'],strict=True)):raise ValueError('Raw reference text differs from independent batch decode')
 validated.update(collection_path=str(path),collection_sha256=hashlib.sha256(payload).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest(),cases=cases)
 return validated
