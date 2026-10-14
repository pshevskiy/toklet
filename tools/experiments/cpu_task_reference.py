"""Bounded raw-ID larger FP32 task reference; detokenize only after model exit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import metal_engine
import metal_controlled as controlled
import native_usage
sys.path.insert(0,str(controlled.ROOT/'tools/bench'))
import task_cases as tasks

ROOT=controlled.ROOT
LIMIT=controlled.LIMIT
APP=64*1024**2
CAPACITY=562
CHUNK=8
WEIGHTS=6174857216
TOKENIZER_ALLOWANCE=113554896

def history_sha(ids):return controlled.history_sha(ids)

def ready(e):
 expected={'event':'ready','format':'toklet-cpu-task-reference-v1','debug_assertions':False,'hidden_size':1536,'num_layers':28,'vocab_size':tasks.VOCAB,'backend':'cpu','cpu_projection':'tiled','weight_dtype':'FP32','kv_dtype':'FP32','normalization_mode':'OrderedFP64','readback_mode':'FullLogits','context_capacity':CAPACITY,'chunk_size':CHUNK,'tokenizer_resident':False,'eos_ids':[151643,151645],'application_bytes':APP}
 if any(type(e.get(k)) is not type(v) or e[k]!=v for k,v in expected.items()) or any(type(x) is not int for x in e.get('eos_ids',[])):raise ValueError('Raw reference actual mode/geometry differs')
 m=e.get('memory');keys=('weights_bytes','kv_bytes','scratch_bytes','history_bytes','logits_bytes','metadata_bytes','metadata_allowance_bytes')
 if type(m) is not dict or set(m)!=set(keys)|{'admitted_bytes'} or any(type(m[k]) is not int or m[k]<0 for k in m) or sum(m[k] for k in keys)!=m['admitted_bytes']:raise ValueError('Raw reference complete memory differs')
 # Exact current CPU representation; capacities and both logits are explicit.
 expected_memory={'weights_bytes':WEIGHTS,'kv_bytes':57344*CAPACITY,'scratch_bytes':147456*CHUNK+8*CAPACITY,'history_bytes':4*CAPACITY,'logits_bytes':tasks.VOCAB*8,'metadata_bytes':9408,'metadata_allowance_bytes':APP}
 if any(m[k]!=v for k,v in expected_memory.items()):raise ValueError('Raw reference admitted representation differs')
 if type(e.get('memory_admitted_bytes')) is not int or e['memory_admitted_bytes']!=m['admitted_bytes']+APP or not 0<e['memory_admitted_bytes']<=LIMIT:raise ValueError('Raw reference admission exceeds6GiB')
 return m

def validate_response(case,events,admission,probe_cap=None):
 tasks.validate_case(case);cap=case['max_output_tokens'] if probe_cap is None else probe_cap
 if probe_cap is not None and (type(probe_cap) is not int or not 1<=probe_cap<=8):raise ValueError('Reference probe cap outside1..8')
 if len(case['prompt_tokens'])+cap>CAPACITY:raise ValueError('Reference task exceeds actual capacity')
 ids=[];elapsed=[];done=None;reset=False
 for e in events:
  if type(e) is not dict:raise ValueError('Raw reference event differs')
  kind=e.get('event')
  if kind=='token' and done is None:
   if set(e)!= {'event','token_id','index','elapsed_seconds'} or type(e['index']) is not int or e['index']!=len(ids) or type(e['token_id']) is not int or not 0<=e['token_id']<tasks.VOCAB or e['token_id'] in tasks.EOS_IDS or len(ids)>=cap or type(e['elapsed_seconds']) not in (int,float) or not math.isfinite(e['elapsed_seconds']) or e['elapsed_seconds']<0 or (elapsed and e['elapsed_seconds']<elapsed[-1]):raise ValueError('Raw reference token coverage differs')
   ids.append(e['token_id']);elapsed.append(e['elapsed_seconds'])
  elif kind=='done' and done is None:done=e
  elif kind=='reset' and done is not None and not reset and e=={'event':'reset'}:reset=True
  else:raise ValueError('Raw reference generation/reset order differs')
  if reset and e is not events[-1]:raise ValueError('Raw reference events after reset')
 if done is None or not reset:raise ValueError('Raw reference lacks completed reset')
 s=done.get('statistics',{});g=s.get('generation',{});reason=g.get('reason')
 if s.get('backend')!='cpu' or s.get('cpu_projection')!='tiled' or s.get('tokenizer_resident') is not False or s.get('memory')!=admission['memory'] or type(s.get('memory_admitted_bytes')) is not int or s['memory_admitted_bytes']!=admission['memory_admitted_bytes']:raise ValueError('Raw reference statistics/admission differ')
 if reason=={'kind':'output_limit'}:
  if len(ids)!=cap:raise ValueError('Raw reference cap exhaustion differs')
  sampled=cap;history=case['prompt_tokens']+ids
 elif type(reason) is dict and set(reason)=={'kind','token_id'} and reason.get('kind')=='stop_token' and type(reason['token_id']) is int and reason['token_id'] in tasks.EOS_IDS:
  sampled=len(ids)+1;history=case['prompt_tokens']+ids+[reason['token_id']]
  if sampled>cap:raise ValueError('Raw reference hidden stop exceeds cap')
 else:raise ValueError('Raw reference EOS/stop policy differs')
 prompt=len(case['prompt_tokens']);expected={'reused_tokens':0,'prefill_tokens':prompt,'sampled_tokens':sampled,'emitted_tokens':len(ids),'history_tokens':prompt+sampled,'evaluated_tokens':prompt+sampled-1,'pending_tokens':1}
 if any(type(g.get(k)) is not int or g[k]!=v for k,v in expected.items()):raise ValueError('Raw reference committed/pending state differs')
 controlled.state({'state':s.get('state')},history,len(history)-1)
 for k in ('prefill_seconds','decode_seconds','sampling_seconds','total_seconds'):
  if type(g.get(k)) not in (int,float) or not math.isfinite(g[k]) or g[k]<0 or g[k]>g['total_seconds']:raise ValueError('Raw reference generation timing invalid')
 if g.get('first_token_seconds')!=(elapsed[0] if elapsed else None) or (elapsed and elapsed[-1]>g['total_seconds']):raise ValueError('Raw reference first/last token timing differs')
 return {'id':case['id'],'category':case['category'],'visible_token_ids':ids,'stop_reason':reason,'statistics':s,'native_events':events,'max_output_tokens':cap}

def finalize(items,tokenizer_path,*,model_exited):
 if model_exited is not True:raise ValueError('Tokenizer requires owned model exit/reaping')
 from tokenizers import Tokenizer
 tokenizer=Tokenizer.from_file(str(tokenizer_path))
 for item in items:
  text=tokenizer.decode(item['visible_token_ids'],skip_special_tokens=True)
  if len(text.encode())>tasks.MAX_RESPONSE_BYTES:raise ValueError('Raw reference finalized text exceeds bound')
  item['visible_text']=text;item['visible_text_sha256']=hashlib.sha256(text.encode()).hexdigest()
 return items

def child(binary,source,cases,raw,stderr,check):
 record={'passed':False,'model_exited':False,'worker_tokenizer_resident':False,'cases':[],'idle_usage':[],'checks':[],'cleanup_errors':[]};worker=None;transport=None
 try:
  command=[str(binary),str(source)];record['command']=command
  worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True);record['worker_pid']=worker.pid
  transport=metal_engine.Transport(worker,raw,stderr,stdout_limit=8*1024**2,event_limit=16000,request_timeout=600,lifetime_timeout=24*3600)
  admission=transport.read();ready(admission);record['ready']=admission
  def usage(label):
   own=native_usage.usage(worker.pid);driver=native_usage.usage();joint=own['lifetime_max_phys_footprint']+driver['lifetime_max_phys_footprint']
   if not 0<joint<=LIMIT:raise ValueError('Raw reference joint model-phase native charge exceeds6GiB')
   record['idle_usage'].append({'phase':label,'worker':own,'driver':driver,'joint_native_lifetime_peak_bytes':joint})
  usage('ready')
  def send(value):worker.stdin.write((json.dumps(value)+'\n').encode());worker.stdin.flush()
  def rejection(value,expected_state):
   send(value);e=transport.read()
   if e.get('event')!='error' or e.get('memory')!=admission['memory'] or e.get('state')!=expected_state:raise ValueError('Raw reference rejection changed valid prefix')
   record['checks'].append({'request':value,'response':e});usage('rejection')
  empty={'history_tokens':0,'evaluated_tokens':0,'pending_tokens':0,'history_sha256':history_sha([])}
  rejection({'action':'generate','tokens':[1]*307,'max_new_tokens':256},empty)
  selected=[cases[i] for i in (0,40,70)] if check else cases
  for case in selected:
   cap=min(8,case['max_output_tokens']) if check else case['max_output_tokens'];send({'action':'generate','tokens':case['prompt_tokens'],'max_new_tokens':cap});events=[]
   while True:
    e=transport.read();events.append(e)
    if e.get('event')=='done':break
    if e.get('event')!='token':raise ValueError('Raw reference generation failed')
   if check:
    rejection({'action':'generate','tokens':[tasks.VOCAB],'max_new_tokens':1},events[-1]['statistics']['state'])
   send({'action':'reset'});events.append(transport.read());response=validate_response(case,events,admission,cap if check else None);record['cases'].append(response);usage(f'case_{len(record["cases"])}');print(f'Raw FP32 {case["id"]}: {len(response["visible_token_ids"])} emitted',flush=True)
  worker.stdin.close();record['exit_code']=transport.finish()
  expected_events=1+len(record['checks'])+sum(len(v['native_events']) for v in record['cases'])
  if record['exit_code']!=0 or transport.pending or transport.events!=expected_events:raise ValueError('Raw reference finalization coverage differs')
  worker.wait(timeout=5);record['model_exited']=True
  # No tokenizer construction (or full model copy) before this ownership barrier.
  finalize(record['cases'],source/'tokenizer.json',model_exited=record['model_exited'])
  record['decoder_phase']={'after_owned_model_exit':True,'model_resident_bytes':0,'tokenizer_allowance_bytes':TOKENIZER_ALLOWANCE,'application_allowance_bytes':APP,'admitted_bytes':TOKENIZER_ALLOWANCE+APP,'native_driver':native_usage.usage()}
  if record['decoder_phase']['native_driver']['lifetime_max_phys_footprint']>LIMIT:raise ValueError('Raw reference decoder phase exceeds6GiB')
  record['passed']=True
 except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
 finally:
  def cleanup(label,action):
   try:action()
   except BaseException as error:record['cleanup_errors'].append(f'{label}: {type(error).__name__}: {error}')
  if worker:
   cleanup('kill owned group',lambda:metal_engine.kill_owned_group(worker));cleanup('reap',lambda:worker.wait(timeout=5))
   for stream in (worker.stdin,worker.stdout,worker.stderr):cleanup('close pipe',stream.close)
  if transport:cleanup('close transport',transport.close)
  if record['cleanup_errors']:record['passed']=False
 return record

def run(binary,output,check=False):
 paths=[output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr')]
 if any(p.exists() for p in paths):raise ValueError('Exclusive reference output exists')
 manifest,_=controlled.contract();cases=tasks.load_cases()
 if max(len(v['prompt_tokens'])+v['max_output_tokens'] for v in cases)!=CAPACITY:raise ValueError('Frozen raw-reference capacity differs')
 spec=controlled.MODELS['larger'];source=ROOT/'models'/spec[0]
 files=[binary,Path(__file__),ROOT/'examples/cpu_task_reference.rs',ROOT/'Cargo.toml',ROOT/'Cargo.lock',Path(tasks.__file__),Path(tasks.prepare.__file__),Path(controlled.__file__),Path(controlled.frozen.__file__),Path(controlled.common.__file__),Path(controlled.metal_prefill.__file__),Path(controlled.metal_continue.__file__),Path(metal_engine.__file__),Path(native_usage.__file__),ROOT/manifest['quality']['tasks_file'],ROOT/'benchmarks/sources.json',ROOT/'benchmarks/result.schema.json',ROOT/'tools/bench/requirements.lock',controlled.prepare.CONTRACT/'manifest.json']+list((ROOT/'src').rglob('*.rs'))+[controlled.prepare.CONTRACT/v['file'] for v in manifest['arrays'].values()]+[source/name for name in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json','generation_config.json')]
 identities={str(p):metal_engine.digest(p) for p in files};controlled.initial_pins(identities,manifest,source,spec)
 if identities[str(ROOT/manifest['quality']['tasks_file'])]!=tasks.TASK_SHA:raise ValueError('Reference task payload changed after admission')
 if identities[str(source/'tokenizer_config.json')]!='5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583' or identities[str(source/'generation_config.json')]!='e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6':raise ValueError('Raw reference fixed tokenizer/EOS assets differ')
 record={'format':'toklet-cpu-task-reference-collection-v1','passed':False,'check_only':check,'collection_completed':False,'quality_completed':False,'scope':'Matched larger raw-ID FP32 response reference only; no literal task-grade or latency acceptance','revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'identities':identities,'task_ids':[v['id'] for v in cases],'task_payload_sha256':tasks.TASK_SHA,'phase_policy':'No tokenizer object during resident model; decode after owned exit/reaping'}
 with paths[0].open('x') as summary,paths[1].open('x') as raw,paths[2].open('xb') as stderr:
  try:
   record['run']=child(binary,source,cases,raw,stderr,check)
   if not record['run']['passed']:raise ValueError('Raw reference worker failed')
   record['passed']=True;record['collection_completed']=not check
  except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
  finally:
   record['endpoint_errors']=[];record['identities_unchanged']=False
   try:record['identities_unchanged']=all(metal_engine.digest(Path(p))==sha for p,sha in identities.items())
   except BaseException as error:record['endpoint_errors'].append(f'{type(error).__name__}: {error}')
   if record['endpoint_errors'] or not record['identities_unchanged']:record['passed']=record['collection_completed']=False
   json.dump(record,summary,indent=2,allow_nan=False);summary.write('\n')
 if not record['passed']:raise ValueError('Raw reference collection failed; see retained record')
 return record

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();run(a.binary.resolve(),a.output.resolve(),a.check)
