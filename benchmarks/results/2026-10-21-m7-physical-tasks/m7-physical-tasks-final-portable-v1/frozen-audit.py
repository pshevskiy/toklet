from pathlib import Path
import sys,json
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools/experiments'));import m7_physical_tasks as t
m=t.measurement;out=Path(__file__).resolve().parent;before={};jobs={};primary=None;endpoints=[]
try:
 original,pins=t.frozen_tasks();m.integration.merge(before,pins)
 raw,data,pins=t.task_payload('small','full');m.integration.merge(before,pins)
 assert len(data['cases'])==100 and [r['id'] for r in data['cases']]==[r['id'] for r in original]
 assert [(r['promptTokens'],r['maximum']) for r in data['cases']]==[(r['prompt_tokens'],r['max_output_tokens']) for r in original]
 assert data['stops']==list(t.task_cases.EOS_IDS)
 for model in ('tiny-cpu','tiny-metal','small','larger'):
  for workload in (('short',) if model.startswith('tiny') else ('short','full')):
   c,_,pins=t.job(model,workload,verify_sources=False);m.integration.merge(before,pins)
   jobs[model+'/'+workload]=c
 for n in ('tools/experiments/m7_physical_tasks.py','tools/experiments/test_m7_physical_tasks.py',str(Path(__file__).resolve().relative_to(ROOT))):before[n]=m.package.sha(ROOT/n)
 m.verify_inputs({n:h for n,h in before.items() if not n.startswith(('models/','artifacts/m5-larger-q4-v2/'))})
except BaseException as e:primary=type(e).__name__+': '+str(e)
finally:
 for n,h in before.items():
  if n.startswith(('models/','artifacts/m5-larger-q4-v2/')):continue
  try:
   if m.package.sha(ROOT/n)!=h:endpoints.append('Changed: '+n)
  except OSError:endpoints.append('Unavailable: '+n)
 result=dict(passed=primary is None and not endpoints,bindings=before,jobs=jobs,case_count=100,full_payload_bytes=len(raw),full_payload_sha256=m.package.hashlib.sha256(raw).hexdigest(),literal_ids_unchanged=True,prompts_caps_unchanged=True,native_model_loads=False,primary_error=primary,endpoint_errors=endpoints)
 with (out/'frozen-audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=result['passed'],case_count=100,full_payload_bytes=len(raw),full_payload_sha256=result['full_payload_sha256'])))
 sys.exit(0 if result['passed'] else 1)
