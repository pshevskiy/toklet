"""Exact response/outcome equivalence to accepted M6 grading; no code execution."""
from pathlib import Path
import hashlib,importlib.util,json,sys
ROOT=Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(ROOT/'tools/experiments'));sys.path.insert(0,str(ROOT/'tools/bench'))
import m7_physical_tasks as q
import linux_task_evidence as evidence
import task_cases as tasks
m=q.measurement
SPEC=importlib.util.spec_from_file_location('phone_tasks_analysis',ROOT/'artifacts/m7-phone-tasks-analysis-v1/analyze.py');a=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(a)
HISTORICAL='benchmarks/results/2026-10-15-m6-online-task-quality-qualification-v1.json'
HISTORICAL_SHA='ca4a63ab48b934436c8b05cea16817e04355ab9cfd584a48db41c44199513842'
SOURCE_MAPS={n:'artifacts/m6-context-source-v1/'+n for n in ('Cargo.toml','src/lib.rs','src/metal/memory.rs','src/weights/mod.rs')}
def exclusive_outputs(folder):
 for name in ('quality.json','small-responses.json','larger-responses.json','private-execution.json','private-final-guard.json','private-logs'):
  p=folder/name
  if p.exists() or p.is_symlink():raise FileExistsError('Exclusive quality output required: '+name)
def write_bound(path,value,pins):
 raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
 with path.open('xb') as stream:stream.write(raw)
 a.c.merge(pins,{str(path.relative_to(ROOT)):hashlib.sha256(raw).hexdigest()})
 return raw
def consume_owned(row,read):
 outputs={};errors=[]
 for field in ('stdout','stderr'):
  try:outputs[field]=read(ROOT/row[field])
  except BaseException as error:errors.append(field+': '+type(error).__name__+': '+str(error))
 if type(row['exit_code']) is not int or row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:
  raise ValueError('Owned fresh absence guard failed: '+str(row['primary_error'])+'; diagnostic errors='+repr(errors))
 if errors:raise ValueError('Owned diagnostic binding failed: '+repr(errors))
 return json.loads(outputs['stdout'])
def match_response(case,phone,old,visible):
 row=phone['response'];reference=old['response'];g=reference['statistics']['generation']
 if type(row['reason']) is not int or row['reason'] not in (1,2) or row.get('stopToken') is not None and type(row['stopToken']) is not int:raise ValueError('Typed physical stop policy required')
 reason={'kind':'output_limit'} if row['reason']==1 else {'kind':'stop_token','token_id':row.get('stopToken')} if row['reason']==2 else None
 if (phone['caseID']!=case['id'] or reference['id']!=case['id'] or reference['category']!=case['category']
     or type(old['max_output_tokens']) is not int or old['max_output_tokens']!=case['max_output_tokens']
     or row['ids']!=reference['visible_token_ids'] or any(type(t) is not int for t in row['ids'])
     or reason!=reference['stop_reason'] or row['reason']==1 and row.get('stopToken') is not None
     or type(row['sampled']) is not int or row['sampled']!=g['sampled_tokens']
     or visible!=reference['visible_text'] or hashlib.sha256(visible.encode()).hexdigest()!=reference['visible_text_sha256']):raise ValueError('Exact historical response equivalence required')
 for key,old_key in [('history','history_tokens'),('evaluated','evaluated_tokens'),('pending','pending_tokens')]:
  if type(row[key]) is not int or row[key]!=g[old_key]:raise ValueError('Exact historical sampled/committed state differs')
 return dict(id=case['id'],category=case['category'],visible_token_ids=row['ids'],visible_text=visible,visible_text_sha256=hashlib.sha256(visible.encode()).hexdigest(),stop_reason=reason)
def replay_outcomes(cases,responses,grading):
 if (grading['completed'] is not True or grading['quality_completed'] is not True or grading['unresolved_case_ids']
     or len(cases)!=100 or len(responses)!=100 or len(grading['runs'])!=40 or grading['task_payload_sha256']!=tasks.TASK_SHA):raise ValueError('Complete resolved historical grading required')
 runs=iter(grading['runs']);outcomes=[]
 for case,response in zip(cases,responses,strict=True):
  if response['id']!=case['id'] or response['category']!=case['category']:raise ValueError('Original response ID/category differs')
  if case['category']=='code':
   run=next(runs);program=tasks.human_eval_program(case,response['visible_text'])
   payload=dict(case_id=case['id'],program=program,program_sha256=hashlib.sha256(program.encode()).hexdigest())
   if (run['id']!=case['id'] or run['completion_sha256']!=response['visible_text_sha256']
       or run['program_sha256']!=payload['program_sha256']):raise ValueError('Exact historical literal program/test bytes required')
   resolved=evidence.classify(run['raw'],payload,grading['runtime'],grading['python_version'])
   if resolved!=run['resolved']:raise ValueError('Historical classifier outcome differs')
   passed=resolved['passed']
  else:passed=tasks.grade_text(case,response['visible_text'])
  outcomes.append(dict(id=case['id'],category=case['category'],passed=passed,status='model_pass' if passed else 'model_fail'))
 if next(runs,None) is not None or outcomes!=grading['outcomes']:raise ValueError('Complete historical outcomes differ')
 aggregate=tasks.aggregate(cases,outcomes)
 if aggregate!=grading['aggregate']:raise ValueError('Historical aggregate differs')
 return outcomes,aggregate

def run(expected_analysis_sha):
 out=Path(__file__).resolve().parent;result_path=out/'quality.json'
 exclusive_outputs(out)
 logs=out/'private-logs';logs.mkdir();pins={};commands=[];models=[];snapshots={};primary=None;endpoints=[]
 def raw(path,expected=None,limit=64*1024**2):
  data=a.c.read_bound(path,limit,pins)
  if expected is not None and hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Consumed quality input differs: '+str(path.relative_to(ROOT)))
  return data
 def metadata(path,expected=None):return json.loads(raw(path,expected))
 def guard():a.c.verify(pins)
 def execute(args,label):
  guard();row=m.package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,120);commands.append(row)
  data=consume_owned(row,raw);guard();return data
 def absence(label):
  current=execute([m.physical.DEVICECTL,'device','info','apps','--bundle-id',m.physical.BUNDLE,'--device',selected['device_identifier'],'--timeout','90','--quiet','--json-output','-'],label+'-install')
  a.c.require_current(current,remote,selected)
  data=execute([m.physical.DEVICECTL,'device','info','processes','--search','TokletHarness','--device',selected['device_identifier'],'--timeout','90','--quiet','--json-output','-'],label+'-absence')
  if (data['info']['outcome']!='success' or data['info']['commandType']!='devicectl.device.info.processes' or data['result']['deviceIdentifier']!=selected['device_identifier'] or m.physical.owned_processes(data,remote)):raise ValueError('Fresh phone model absence required')
 try:
  env,_=m.package.build_environment();env['TMPDIR']=str(out)
  for module in (q,m,m.package,m.integration,m.physical,m.storage,evidence,tasks,tasks.prepare):raw(Path(module.__file__).resolve())
  raw(Path(__file__).resolve());raw(out/'checks.py');raw(Path(a.__file__).resolve());raw(a.HELPER)
  analysis=metadata(ROOT/'artifacts/m7-phone-tasks-analysis-v1/analysis.json',expected_analysis_sha)
  if (analysis['passed'] is not True or analysis['stages']!=14 or analysis['cases']!=206 or analysis['task_quality_qualified'] is not False or analysis['physical_device_qualified'] is not False or analysis['m7_complete'] is not False):raise ValueError('Complete raw phone response replay required')
  a.c.merge(pins,analysis['bindings']);guard()
  historical=metadata(ROOT/HISTORICAL,HISTORICAL_SHA)
  if historical['passed'] is not True or historical['quality_completed'] is not True or historical['review']['accepted'] is not True or historical['infrastructure_unresolved']!=0 or historical['resolved_outcomes']!=200:raise ValueError('Accepted exact historical outcomes required')
  old_pins=historical['bindings']
  for n,v in old_pins.items():
   expected=v['sha256'];path=ROOT/n
   if a.c.sha(path)==expected:a.c.merge(pins,{n:expected})
   else:
    replacement=SOURCE_MAPS.get(n)
    if replacement is None or old_pins.get(replacement)!=v or a.c.sha(ROOT/replacement)!=expected:raise ValueError('Historical closure endpoint differs: '+n)
    snapshots[n]=dict(snapshot=replacement,sha256=expected);a.c.merge(pins,{replacement:expected})
  selected=m.physical.selection(ROOT/a.PRIVATE,a.PRIVATE_SHA)
  _,extra,remote,signed=m.physical.admit_reuse(ROOT/a.PREPARED,a.PREPARED_SHA,selected,q.NATIVE_SHA);a.c.merge(pins,extra)
  cases,extra=q.frozen_tasks();a.c.merge(pins,extra)
  if cases!=tasks.load_cases():raise ValueError('Exact full frozen prompts/caps/test policy required')
  # No tokenizer construction or generated program classification before fresh absence.
  absence('before-decode');guard()
  from tokenizers import Tokenizer,__version__ as tokenizer_version
  for model in ('small','larger'):
   parent=next(r for r in historical['models'] if r['model']==model)
   old=metadata(ROOT/parent['collection'],old_pins[parent['collection']]['sha256'])
   grading=metadata(ROOT/parent['grading'],old_pins[parent['grading']]['sha256'])
   window=metadata(ROOT/parent['window'],old_pins[parent['window']]['sha256'])
   if (old['task_ids']!=[c['id'] for c in cases] or old['task_payload_sha256']!=tasks.TASK_SHA or len(old['run']['cases'])!=100
       or window['completed'] is not True or window['quality_completed'] is not True or window['collection_sha256']!=old_pins[parent['collection']]['sha256']
       or grading['collection_sha256']!=old_pins[parent['collection']]['sha256'] or grading['native_raw_sha256']!=old_pins[str(Path(parent['collection']).with_suffix('.raw.jsonl'))]['sha256']):raise ValueError('Historical collection/grading window differs')
   for n,h in grading['source_sha256'].items():
    if old_pins.get(n)!={'sha256':h}:raise ValueError('Historical classifier closure not accepted')
    a.c.merge(pins,{n:h})
   guard()
   path=ROOT/('artifacts/m7-phone-tasks-'+model+'-full-v1/measurement.json');receipt=metadata(path,analysis['bindings'][str(path.relative_to(ROOT))])
   text=raw(path.parent/'private-logs/measurement.stdout',receipt['private_evidence_pins'][str((path.parent/'private-logs/measurement.stdout').relative_to(ROOT))]).decode()
   err=raw(path.parent/'private-logs/measurement.stderr',receipt['private_evidence_pins'][str((path.parent/'private-logs/measurement.stderr').relative_to(ROOT))]).decode()
   checked=q.validate_console(text+'\n'+err,receipt['config'],signed);rows=checked['responses']
   if len(rows)!=100:raise ValueError('All physical responses required')
   source='models/Qwen2.5-'+('0.5B' if model=='small' else '1.5B')+'-Instruct/tokenizer.json';tokenizer=Tokenizer.from_str(raw(ROOT/source,old_pins[source]['sha256'],16*1024**2).decode());responses=[]
   for case,phone,prior in zip(cases,rows,old['run']['cases'],strict=True):responses.append(match_response(case,phone,prior,tokenizer.decode(phone['response']['ids'],skip_special_tokens=True)))
   outcomes,aggregate=replay_outcomes(cases,responses,grading);baseline=2 if model=='small' else 11
   if parent['fp32_passed']!=baseline or parent['unchanged_allowed_drop_percentage_points']!=2 or baseline-aggregate['passed']>2:raise ValueError('Unchanged full task gate differs')
   response_file=out/(model+'-responses.json');write_bound(response_file,responses,pins)
   models.append(dict(model=model,cases=100,passed=aggregate['passed'],fp32_passed=baseline,drop_percentage_points=baseline-aggregate['passed'],maximum_drop_percentage_points=2,aggregate=aggregate,outcomes=outcomes,decoded_responses=str(response_file.relative_to(ROOT)),identical_visible_ids=True,identical_hidden_stops=True,identical_text=True,historical_literal_cases=40,recomputed_text_cases=60,historical_grading=parent['grading'],historical_execution_started_utc=grading['started_utc'],historical_execution_completed_utc=grading['completed_utc'],historical_runtime=grading['runtime'],historical_python_version=grading['python_version'],new_vm_runs=0,new_generated_program_executions=0))
  absence('after-quality');guard()
 except BaseException as e:primary=type(e).__name__+': '+str(e)
 finally:
  try:guard()
  except BaseException as e:endpoints.append(type(e).__name__+': '+str(e))
  private=dict(commands=commands,primary_error=primary,endpoint_errors=endpoints);private_path=out/'private-execution.json';write_bound(private_path,private,pins)
  try:guard()
  except BaseException as e:endpoints.append(type(e).__name__+': '+str(e))
  write_bound(out/'private-final-guard.json',dict(primary_error=primary,endpoint_errors=endpoints),pins)
  passed=primary is None and not endpoints and len(models)==2 and len(commands)==4
  result=dict(format='toklet-m7-phone-task-outcome-equivalence-v1',passed=passed,quality_completed=passed,bindings=pins,historical_qualification=dict(path=HISTORICAL,sha256=HISTORICAL_SHA),historical_source_snapshots=snapshots,models=models,commands=m.physical.redact_record(private,locals().get('selected',{}),locals().get('remote',''))['commands'],primary_error=primary,endpoint_errors=endpoints,new_vm_runs=0,new_generated_program_executions=0,physical_device_qualified=False,m7_complete=False,scope='Exact phone response equivalence and independently replayed historical literal outcomes; no fresh generated program execution or broader mobile qualification')
  result=m.physical.redact_record(result,locals().get('selected',{}),locals().get('remote',''))
  with result_path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  print(json.dumps(dict(passed=passed,models=len(models),primary_error=result['primary_error'],endpoint_errors=result['endpoint_errors'])));return 0 if passed else 1
if __name__=='__main__':
 if len(sys.argv)!=2:raise SystemExit('Expected completed raw analysis SHA256')
 sys.exit(run(sys.argv[1]))
