"""Offline exact replay of two full32K applications; no numerical/model launch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'Cargo.toml').is_file())
sys.path.insert(0,str(ROOT/'tools/experiments'))
import m6_application as app
import m6_context_audit as audit

def run():
 source=Path(__file__).resolve();pins={str(source.relative_to(ROOT)):audit.read_bytes(source)[1]}
 pins[str(Path(app.__file__).resolve().relative_to(ROOT))]=audit.read_bytes(Path(app.__file__).resolve())[1]
 pins[str(Path(audit.__file__).resolve().relative_to(ROOT))]=audit.read_bytes(Path(audit.__file__).resolve())[1]
 rows=[]
 for model in ('small','larger'):
  binary,model_path,packed,identities=app.producer.provenance(model)
  identities[str(Path(app.__file__).resolve().relative_to(ROOT))]=pins[str(Path(app.__file__).resolve().relative_to(ROOT))]
  for path,sha in identities.items():
   if path in pins and pins[path]!=sha:raise ValueError('Conflicting application identity')
   pins[path]=sha
  path=ROOT/f'artifacts/m6-application-{model}-full-v1.json'
  record,sha=audit.read_json(path);pins[str(path.relative_to(ROOT))]=sha
  raw_path=path.with_suffix('.raw.jsonl');raw,sha=audit.read_json(raw_path,lines=True);pins[str(raw_path.relative_to(ROOT))]=sha
  stderr=path.with_suffix('.stderr');pins[str(stderr.relative_to(ROOT))]=audit.read_bytes(stderr)[1]
  for key,value in dict(format='toklet-m6-max-context-application-v1',passed=True,collection_completed=True,
   check_only=False,model=model,context_capacity=32768,prompt_tokens=32640,output_tokens=128,
   identities_unchanged=True,independent_text_after_owned_exit=True).items():app.context.exact(record.get(key),value,key)
  if record.get('endpoint_errors')!=[] or record.get('error'):raise ValueError('Application endpoint/error remains')
  if record['identities']!=identities:raise ValueError('Application consumed inventory differs')
  worker=record['worker']
  for key,value in dict(passed=True,exit_code=0).items():app.context.exact(worker.get(key),value,key)
  if worker.get('cleanup_errors')!=[] or worker.get('error'):raise ValueError('Owned application cleanup/error remains')
  projection,extra=app.producer.arguments(model,packed)
  command=[str(binary),'--model',str(model_path),'--backend','metal','--metal-projection',projection,
   '--interactive','--json','--context-capacity','32768','--prefill-chunk','128','--no-eos',*extra]
  if worker['command']!=command:raise ValueError('Application command differs')
  ready=worker['ready'];memory=app.producer.ready(ready,model,32768)
  layers,dimension=(24,64) if model=='small' else (28,128)
  for key,value in dict(kv_bytes=(2*32768*dimension+7)*4*layers,history_bytes=32768*4,logits_bytes=151936*8).items():
   app.context.exact(memory[key],value,key)
  _,arrays,_,_,_=app.context.inputs(model);prompt=arrays[32768][:-128]
  phases=worker['phases']
  if len(phases)!=2:raise ValueError('Application phase coverage differs')
  if phases[0]['request']!=dict(action='generate',tokens=prompt,max_new_tokens=128) or any(type(t) is not int for t in phases[0]['request']['tokens']):
   raise ValueError('Application frozen input differs')
  app.context.exact(phases[0]['request']['max_new_tokens'],128,'output count')
  if phases[1]['request']!=dict(action='reset'):raise ValueError('Final reset request differs')
  if raw!=[ready,*(e for phase in phases for e in phase['events'])]:raise ValueError('Raw application event coverage differs')
  response=app.response(phases[0]['events'],phases[1]['events'],ready,prompt,128)
  if response!=record['response']:raise ValueError('Application response differs')
  app.producer.shared.finalized([response],model_path/'tokenizer.json',model_exited=True)
  usage=worker['usage']
  if len(usage)!=3:raise ValueError('Application native phase coverage differs')
  for i,u in enumerate(usage):
   app.context.exact(u['phase'],'ready' if i==0 else f'phase_{i}','native phase')
   values=[u[owner]['lifetime_max_phys_footprint'] for owner in ('worker','driver')]
   if any(type(v) is not int or v<=0 for v in values):raise ValueError('Native lifetime unavailable')
   app.context.exact(u['joint_native_lifetime_peak_bytes'],sum(values),'native sum')
   if sum(values)>app.context.LIMIT:raise ValueError('Application native charge outside6GiB')
  decoder=record['decoder_phase'];app.context.exact(decoder['after_owned_exit'],True,'decoder phase')
  app.context.exact(decoder['admitted_bytes'],247772624,'decoder admission')
  charge=decoder['driver']['lifetime_max_phys_footprint']
  if type(charge) is not int or not 0<charge<=app.context.LIMIT:raise ValueError('Decoder native charge outside6GiB')
  app.context.finite(record['driver_submission_through_done_validation_seconds'],'driver completed duration',positive=True)
  for endpoint in ('power_before','power_after'):
   if set(record[endpoint])!={'battery','policy','thermal'} or any(type(v) is not str or not v for v in record[endpoint].values()):
    raise ValueError('Power endpoint observation coverage differs')
  rows.append(dict(model=model,raw_events=len(raw),prompt_tokens=32640,output_tokens=128,
   memory_admitted_bytes=ready['memory_admitted_bytes'],max_joint_native_bytes=max(u['joint_native_lifetime_peak_bytes'] for u in usage),
   decoder_peak_bytes=charge,history_sha256=response['history_sha256'],visible_text_sha256=response['visible_text_sha256'],
   generation=response['statistics']['generation'],power_before=record['power_before'],power_after=record['power_after']))
 audit.verify_bindings(pins)
 return dict(format='toklet-m6-full-application-replay-v1',passed=True,models=rows,bindings=pins,m6_complete=False,
  scope='Two full tokenizer-owning fixed-ID32K applications; exact derived history, state/memory/publishedtext and postexit independent decode; no latency ranking or natural long-document quality')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Exclusive application replay evidence exists')
 result=run()
 with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'passed':True,'bindings':len(result['bindings']),'models':[{k:r[k] for k in ('model','raw_events','memory_admitted_bytes','max_joint_native_bytes','decoder_peak_bytes')} for r in result['models']]},indent=2))
