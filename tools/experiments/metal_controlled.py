"""Frozen eight-workload completed Metal worker; checks are distinct from samples."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import metal_engine
import metal_prefill
import metal_continue
import native_usage

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
import cpu as frozen
import mlx_baseline as common
import prepare

LIMIT=6*1024**3
CONTRACT_SHA='3f09dcf70f6d9620ce5172dcbaa6428f3487a3edf412f2bfce57937faab72c66'
SCHEMA_SHA='503047553e2a79bbc637da756497494e2fc916c70cdfef1fcf0b687a7a4be852'
MODELS={
 'small':('Qwen2.5-0.5B-Instruct','fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe','18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45',896,24,988069596,'7ae557604adf67be50417f59c2c2f167def9a775'),
 'larger':('Qwen2.5-1.5B-Instruct','dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee','98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670',1536,28,3087433340,'989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
}

def contract():
 path=prepare.CONTRACT/'manifest.json'
 with path.open('rb') as stream:data=stream.read(1024**2+1)
 if len(data)>1024**2 or hashlib.sha256(data).hexdigest()!=CONTRACT_SHA:raise ValueError('Frozen controlled contract differs')
 prepare.verify(prepare.CONTRACT);manifest=json.loads(data)
 arrays={k:read_array(prepare.CONTRACT/v['file'],v) for k,v in manifest['arrays'].items() if k!='held_out_loss'}
 return manifest,arrays

def read_array(path,item):
 count=item['count']
 if type(count) is not int or not 0<count<=2048:raise ValueError('Controlled token count outside bound')
 with path.open('rb') as stream:data=stream.read(count*4+1)
 if len(data)!=count*4 or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Frozen controlled token bytes differ')
 values=list(struct.unpack(f'<{count}I',data))
 if any(v>=151936 for v in values):raise ValueError('Controlled token outside vocabulary')
 return values

def initial_pins(identities,manifest,source,spec):
 pins={source/'model.safetensors':spec[1],source/'config.json':spec[2],source/'tokenizer.json':'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',prepare.CONTRACT/'manifest.json':CONTRACT_SHA,ROOT/'benchmarks/result.schema.json':SCHEMA_SHA,ROOT/'benchmarks/sources.json':manifest['sources_sha256'],ROOT/'tools/bench/requirements.lock':manifest['python_dependencies_sha256']}
 pins.update({prepare.CONTRACT/v['file']:v['sha256'] for v in manifest['arrays'].values()})
 if any(identities.get(str(path))!=sha for path,sha in pins.items()):raise ValueError('Fixed controlled initial input hashes differ')

def requests(name,arrays,manifest):
 setup,request,cache=frozen.requests(name,arrays,manifest)
 if name=='changed':request={**request,'action':'reset'}
 return setup,request,cache

def history_sha(ids):return hashlib.sha256(struct.pack(f'<{len(ids)}I',*ids)).hexdigest()

def state(event,ids,evaluated):
 expected={'history_tokens':len(ids),'evaluated_tokens':evaluated,'pending_tokens':len(ids)-evaluated,'history_sha256':history_sha(ids)}
 actual=event.get('state')
 if actual!=expected or any(type(actual.get(k)) is not int for k in ('history_tokens','evaluated_tokens','pending_tokens')):raise ValueError('Controlled complete prefix state differs')

def ready(e,model):
 spec=MODELS[model];expected={'event':'ready','format':'toklet-metal-controlled-v1','debug_assertions':False,'backend':'metal','projection_mode':'HalfCompensated','normalization_mode':'Original','readback_mode':'FullLogits','weight_dtype':'FP16','kv_dtype':'FP16','hidden_size':spec[3],'num_layers':spec[4],'vocab_size':151936,'context_capacity':2048,'chunk_size':128,'application_bytes':64*1024**2}
 if any(type(e.get(k)) is not type(v) or e[k]!=v for k,v in expected.items()):raise ValueError('Controlled actual mode/geometry differs')
 widths=e.get('half_compensated_execution_widths')
 if type(widths) is not list or len(widths)!=2 or any(type(w) is not int or not 0<w<=0xffffffff for w in widths) or widths[0]!=widths[1]:raise ValueError('Controlled queried widths differ')
 m=e.get('memory');keys=('weights_bytes','kv_bytes','scratch_bytes','history_bytes','logits_bytes','metadata_bytes','metadata_allowance_bytes')
 if type(m) is not dict or set(m)!=set(keys)|{'admitted_bytes'} or any(type(m[k]) is not int or m[k]<0 for k in m) or sum(m[k] for k in keys)!=m['admitted_bytes'] or m['weights_bytes']!=spec[5] or m['logits_bytes']!=151936*8:raise ValueError('Controlled complete memory differs')
 if type(e.get('memory_admitted_bytes')) is not int or e['memory_admitted_bytes']!=m['admitted_bytes']+e['application_bytes'] or not 0<e['memory_admitted_bytes']<=LIMIT:raise ValueError('Controlled admission exceeds6GiB')

def sample(e,name,admission,arrays,manifest):
 if e.get('event')!='sample' or e.get('memory')!=admission['memory']:raise ValueError('Controlled sample/persistent inventory differs')
 p=e['sample'];base=arrays[f'prompt_{frozen.WORKLOADS[name]}'];fixed=arrays['continuation_128']
 if name=='append':prefix=base+arrays['append_64'];reused=512;prefill=64
 elif name=='pending':prefix=base+[manifest['continuation']['pending_sample']['pending_token']];reused=512;prefill=1
 elif name=='changed':prefix=arrays['changed_prompt_512'];reused=0;prefill=512
 elif name=='cached':prefix=base;reused=512;prefill=0
 else:prefix=base;reused=0;prefill=len(base)
 expected={'emitted_tokens':128,'reused_tokens':reused,'prefill_tokens':prefill,'evaluated_tokens':len(prefix)+127,'pending_tokens':1}
 if any(type(p.get(k)) is not int or p[k]!=v for k,v in expected.items()):raise ValueError('Controlled workload state differs')
 state(e,prefix+fixed,len(prefix)+127)
 times=p['token_event_seconds']
 if len(times)!=128 or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in times) or any(b<a for a,b in zip(times,times[1:])):raise ValueError('Controlled token timestamps differ')
 for k in ('prefill_seconds','ttft_seconds','decode_seconds','total_seconds'):
  if type(p.get(k)) not in (int,float) or not math.isfinite(p[k]) or p[k]<0:raise ValueError('Controlled completed timing invalid')
 if not p['prefill_seconds']<=p['ttft_seconds']==times[0]<=times[-1]<=p['total_seconds'] or p['decode_seconds']!=times[-1]-times[0]:raise ValueError('Controlled timing boundary differs')
 t=e['last_completed_timing']
 if type(t.get('command_count')) is not int or t['command_count']!=1:raise ValueError('Controlled completed command differs')
 for k in ('command_creation_seconds','clock_calibration_seconds','commit_seconds','host_encode_seconds','wait_seconds','gpu_start_seconds','gpu_end_seconds','readback_seconds'):
  if type(t.get(k)) not in (int,float) or not math.isfinite(t[k]) or t[k]<0:raise ValueError('Controlled command completion invalid')
 if t['gpu_end_seconds']<t['gpu_start_seconds']:raise ValueError('Controlled GPU completion order differs')
 return p

def worker(binary,source,model,name,manifest,arrays,raw,stderr,check,*,admission_check=None,extra_args=()):
 run={'workload':name,'passed':False,'phases':[],'idle_usage':[],'samples':[],'cleanup_errors':[]};child=None;transport=None
 try:
  command=[str(binary),str(source),*extra_args];run['command']=command
  child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
  transport=metal_engine.Transport(child,raw,stderr,stdout_limit=2*1024**2,event_limit=256,request_timeout=600,lifetime_timeout=14400)
  initial=transport.read();(admission_check or ready)(initial,model);run['ready']=initial
  def usage(label):
   own=native_usage.usage(child.pid);driver=native_usage.usage();joint=own['lifetime_max_phys_footprint']+driver['lifetime_max_phys_footprint']
   if not 0<joint<=LIMIT:raise ValueError('Controlled joint native charge exceeds6GiB')
   run['idle_usage'].append({'phase':label,'worker':own,'driver':driver,'joint_native_lifetime_peak_bytes':joint});return own['lifetime_max_phys_footprint']
  usage('ready')
  def send(value):
   child.stdin.write((json.dumps(value)+'\n').encode());child.stdin.flush();e=transport.read();run['phases'].append({'request':value,'response':e});peak=usage(f'phase_{len(run["phases"])}')
   if e.get('memory')!=initial['memory']:raise ValueError('Controlled inventory changed')
   return e,peak
  rejection,_=send({'action':'reset','tokens':arrays['prompt_1921'],'continuation':arrays['continuation_128']})
  if rejection.get('event')!='error':raise ValueError('Controlled overcapacity accepted')
  state(rejection,[],0)
  setup,request,cache=requests(name,arrays,manifest);run.update(cache_state=cache,setup_requests=setup,measured_request=request)
  for i in range(1 if check else 13):
   for prep in setup:
    e,_=send(prep)
    if e.get('event')!='prepared':raise ValueError('Controlled preparation failed')
    ids=arrays['prompt_512']+([manifest['continuation']['pending_sample']['pending_token']] if prep['action']=='stage' else [])
    state(e,ids,512)
   e,peak=send(request);p=sample(e,name,initial,arrays,manifest);run['samples'].append({'phase':'check' if check else 'warmup' if i<3 else 'sample','index':i,'response':e,'process_peak_footprint_bytes':peak})
   print(f'{model}/{name} {i+1}: {p["total_seconds"]:.6f}s',flush=True)
  if name=='boundary':
   final=request['tokens']+request['continuation'];e,_=send({'action':'prepare','tokens':final})
   if e.get('event')!='prepared':raise ValueError('Controlled final boundary not committed')
   state(e,final,2048);e,_=send({'action':'stage','tokens':[1]})
   if e.get('event')!='error':raise ValueError('Controlled beyond final capacity accepted')
   state(e,final,2048)
  child.stdin.close();run['exit_code']=transport.finish()
  if run['exit_code']!=0 or transport.events!=len(run['phases'])+1 or transport.pending:raise ValueError('Controlled finalization coverage differs')
  run['passed']=True
 except BaseException as error:run['error']=f'{type(error).__name__}: {error}'
 finally:
  def cleanup(label,action):
   try:action()
   except BaseException as error:run['cleanup_errors'].append(f'{label}: {type(error).__name__}: {error}')
  if child:
   cleanup('kill owned group',lambda:metal_engine.kill_owned_group(child));cleanup('reap',lambda:child.wait(timeout=5))
   for stream in (child.stdin,child.stdout,child.stderr):cleanup('close pipe',stream.close)
  if transport:cleanup('close transport',transport.close)
  if run['cleanup_errors']:run['passed']=False
 return run

def result(run,model,manifest,observations,revision,dirty,*,representation=None,runtime="Toklet HalfCompensated Metal",execution_note=None):
 name=run['workload'];admission=run['ready'];spec=MODELS[model];rows=run['samples'][3:]
 samples=[]
 for row in rows:
  p=row['response']['sample'];s={k:p[k] for k in ('prefill_seconds','ttft_seconds','decode_seconds','total_seconds','token_event_seconds','emitted_tokens')};s.update(first_visible_text_seconds=None,process_peak_footprint_bytes=row['process_peak_footprint_bytes'],gpu_peak_allocation_bytes=None);samples.append(s)
 if len(samples)!=10:raise ValueError('Controlled needs3 warmups/10 samples')
 prompt='append_64' if name=='append' else 'changed_prompt_512' if name=='changed' else f'prompt_{frozen.WORKLOADS[name]}'
 value={'schema_version':1,'contract_id':manifest['contract_id'],'comparison_track':'controlled_engine','runtime':runtime,'identity':{'engine_commit':revision,'working_tree_dirty':dirty,'baseline_versions':{},'compiler':observations['compiler'],'build_flags':['CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --example metal_bench'],'model_revision':spec[6],'tokenizer_revision':spec[6],'model_sha256':spec[1],'tokenizer_sha256':'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539','device':observations['device']},'representation':representation or {'weight_dtype':'FP16','kv_dtype':'FP16','packed_format_version':None,'group_size':None},'workload':{'name':name+'_teacher_forced_128','prompt_tokens':manifest['arrays'][prompt]['count'],'output_steps':128,'context_capacity':2048,'used_context_before_request':0 if run['cache_state']=='empty' else 513 if name=='pending' else 512,'prompt_sha256':manifest['arrays'][prompt]['sha256'],'continuation_sha256':manifest['arrays']['continuation_128']['sha256'],'stop_policy':'fixed_length_teacher_forced_ignore_eos','cache_state':run['cache_state']},'timing_scope':{'includes_tokenization':False,'includes_sampling':False,'includes_detokenization':False,'completion_boundary':'synchronous completed Metal GPU vocabulary projection and host FP32 logits readback; token timestamps before staging next fixed ID','first_visible_text_policy':'not measured; fixed events only'},'setup':{'model_load_seconds':admission['load_seconds'],'conversion_seconds':admission['conversion_seconds'],'compilation_seconds':None,'tokenization_seconds':None,'warmup_repetitions':3},'samples':samples,'summary':common.summarize(samples,rows[0]['response']['sample']['prefill_tokens']),'memory':{'engine_working_set_limit_bytes':LIMIT,'logical_weights_bytes':admission['memory']['weights_bytes'],'allocated_kv_bytes':admission['memory']['kv_bytes'],'scratch_bytes':admission['memory']['scratch_bytes'],'peak_process_bytes':max(u['worker']['lifetime_max_phys_footprint'] for u in run['idle_usage']),'system_memory_pressure':None,'measurement_instrument':'owned idle worker libproc v4 lifetime native charge plus complete logical admission; shared physical total unavailable'},'energy':{'joules':None,'instrument':None,'scope':None},'notes':['Exact source/binary/contract identities in parent evidence.', 'Prepare is reset-and-evaluate outside timing; changed row explicitly resets inside timing. Prefix-reuse optimization is a separate workload.',execution_note or f'Actual HalfCompensated/Original/FullLogits, queried widths{admission["half_compensated_execution_widths"]}; complete admission{admission["memory_admitted_bytes"]}.','No sampling, tokenization, text, energy or continuous thermal/system-pressure result; no quality or speed-ranking acceptance.']}
 from jsonschema import Draft202012Validator
 schema_path=ROOT/'benchmarks/result.schema.json'
 if metal_engine.digest(schema_path)!=SCHEMA_SHA:raise ValueError('Frozen result schema differs')
 Draft202012Validator(json.loads(schema_path.read_text())).validate(value)
 return value

def run(binary,output,model,check=False):
 paths=[output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr')]
 if any(p.exists() for p in paths):raise ValueError('Exclusive controlled evidence exists')
 manifest,arrays=contract();spec=MODELS[model];source=ROOT/'models'/spec[0]
 files=[binary,Path(__file__),ROOT/'examples/metal_bench.rs',ROOT/'Cargo.toml',ROOT/'Cargo.lock',prepare.CONTRACT/'manifest.json',ROOT/'benchmarks/result.schema.json',ROOT/'benchmarks/sources.json',ROOT/'tools/bench/requirements.lock',Path(frozen.__file__),Path(prepare.__file__),Path(common.__file__),Path(metal_engine.__file__),Path(native_usage.__file__),Path(metal_prefill.__file__),Path(metal_continue.__file__),source/'model.safetensors',source/'config.json',source/'tokenizer.json']+list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))+[prepare.CONTRACT/v['file'] for v in manifest['arrays'].values()]
 identities={str(p):metal_engine.digest(p) for p in files}
 initial_pins(identities,manifest,source,spec)
 observation=metal_prefill.observation
 observations={'compiler':observation(['rustc','--version']),'device':{'name':observation(['/usr/sbin/sysctl','-n','hw.model']),'chip':observation(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string']),'physical_memory_bytes':int(observation(['/usr/sbin/sysctl','-n','hw.memsize'])),'os_version':observation(['/usr/bin/sw_vers','-productVersion']),'os_build':observation(['/usr/bin/sw_vers','-buildVersion']),'power_mode':observation(['/usr/bin/pmset','-g','batt'])+'\n'+observation(['/usr/bin/pmset','-g','custom']),'thermal_conditions':observation(['/usr/bin/pmset','-g','therm'])}}
 record={'format':'toklet-metal-controlled-collection-v1','passed':False,'check_only':check,'measurements_completed':False,'scope':'Frozen controlled Half workloads only; no task/application or release ranking','model':model,'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'identities':identities,'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'observations':observations,'runs':[],'results':[]}
 with paths[0].open('x') as summary,paths[1].open('x') as raw,paths[2].open('xb') as stderr:
  try:
   for name in frozen.WORKLOADS:
    item=worker(binary,source,model,name,manifest,arrays,raw,stderr,check);record['runs'].append(item)
    if not item['passed']:raise ValueError(f'Controlled {name} failed')
    if not check:record['results'].append(result(item,model,manifest,observations,record['revision'],record['working_tree_dirty']))
   record['passed']=True;record['measurements_completed']=not check
  except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
  finally:
   record['endpoint_errors']=[];record['identities_unchanged']=False
   try:record['identities_unchanged']=all(metal_engine.digest(Path(p))==sha for p,sha in identities.items())
   except BaseException as error:record['endpoint_errors'].append(f'{type(error).__name__}: {error}')
   try:record['power_after']=metal_prefill.endpoints()
   except BaseException as error:record['endpoint_errors'].append(f'{type(error).__name__}: {error}')
   if record['endpoint_errors'] or not record['identities_unchanged']:record['passed']=record['measurements_completed']=False
   json.dump(record,summary,indent=2,allow_nan=False);summary.write('\n')
 if not record['passed']:raise ValueError('Controlled qualification failed; see retained record')
 return record

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();run(a.binary.resolve(),a.output.resolve(),a.model,a.check)
