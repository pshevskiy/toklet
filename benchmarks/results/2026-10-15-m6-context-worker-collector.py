import hashlib,json,math,pathlib,shutil,subprocess,sys
ROOT=pathlib.Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(ROOT/'tools/experiments'))
import metal_engine,native_usage

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(ids,e):return {'history_tokens':len(ids),'evaluated_tokens':e,'pending_tokens':len(ids)-e,'history_sha256':hashlib.sha256(b''.join(x.to_bytes(4,'little') for x in ids)).hexdigest()}
def main():
 out=ROOT/'artifacts/m6-context-worker-native-v1';out.mkdir(exist_ok=False)
 binary=out/'metal_context';shutil.copyfile(ROOT/'artifacts/m5-checks/release/examples/metal_context',binary);binary.chmod(0o755)
 shutil.copyfile(pathlib.Path(__file__),out/'collector.py')
 source=ROOT/'artifacts/m6-context-cli-native-v1/tiny'
 paths=[binary,out/'collector.py',source/'config.json',source/'model.safetensors',ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'examples/metal_context.rs']
 paths+=list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))+[pathlib.Path(m.__file__) for m in (metal_engine,native_usage)]
 pins={str(p.relative_to(ROOT)):sha(p) for p in paths}
 result={'passed':False,'scope':'Paired-worker tiny functional/profiling component only; no real-model performance or long-context qualification','pins':pins,'runs':[]}
 with (out/'raw.jsonl').open('x') as raw,(out/'stderr').open('xb') as stderr:
  try:
   for profiling in [False,True]:
    assert all(sha(ROOT/p)==v for p,v in pins.items())
    command=[str(binary),str(source),'-','32','3']+(['profile'] if profiling else [])
    worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    run={'command':command,'phases':[],'usage':[],'passed':False,'cleanup_errors':[]};result['runs'].append(run)
    transport=metal_engine.Transport(worker,raw,stderr,stdout_limit=4*1024**2,event_limit=4096,request_timeout=120,lifetime_timeout=900)
    try:
     ready=transport.read();run['ready']=ready
     assert ready['event']=='ready' and ready['format']=='toklet-m6-context-v1' and ready['context_capacity']==32 and ready['chunk_size']==3 and ready['profiling'] is profiling
     assert ready['debug_assertions'] is False and ready['hidden_size']==32 and ready['num_layers']==2 and ready['vocab_size']==64
     assert ready['unique_memory_admitted_bytes']==ready['reference_memory']['admitted_bytes']+ready['online_memory']['admitted_bytes']-ready['shared_weights_bytes']+ready['application_bytes']
     assert ready['unique_memory_admitted_bytes']<=6*1024**3
     memories=[ready['reference_memory'],ready['online_memory']]
     for m in memories:assert m['admitted_bytes']==sum(v for k,v in m.items() if k!='admitted_bytes')
     assert memories[0]['scratch_bytes']>memories[1]['scratch_bytes']
     def usage():
      a=native_usage.usage(worker.pid);b=native_usage.usage();charge=a['lifetime_max_phys_footprint']+b['lifetime_max_phys_footprint'];assert 0<charge<=6*1024**3
      run['usage'].append({'worker':a,'driver':b,'joint_native_bytes':charge})
     usage()
     def call(request,final):
      worker.stdin.write((json.dumps(request)+'\n').encode());worker.stdin.flush();events=[]
      while True:
       e=transport.read();events.append(e)
       if e['event'] in (final,'error'):break
      run['phases'].append({'request':request,'events':events});usage()
      return events
     if not profiling:
      for mode,index in [('reference',0),('online',1)]:
       prefix=[1,7,13,3];continuation=[47,47]
       e=call({'action':'reset','mode':mode,'tokens':prefix,'continuation':continuation},'sample')[-1]
       assert e['event']=='sample' and e['state']==state(prefix+continuation,5) and e['memory']==memories[index]
       s=e['sample'];assert (s['reused_tokens'],s['prefill_tokens'],s['emitted_tokens'],s['pending_tokens'])==(0,4,2,1)
       assert len(s['token_event_seconds'])==2 and s['ttft_seconds']==s['token_event_seconds'][0] and s['decode_seconds']==s['token_event_seconds'][1]-s['token_event_seconds'][0]
       assert 0<s['ttft_seconds']<s['total_seconds'] and s['decode_seconds']>0
       e=call({'action':'prompt','mode':mode,'tokens':prefix+[47],'continuation':continuation},'sample')[-1]
       assert (e['sample']['reused_tokens'],e['sample']['prefill_tokens'])==(5,0) and e['state']==state(prefix+[47]+continuation,6)
       before=e['state'];e=call({'action':'reset','mode':mode,'tokens':[1]*32,'continuation':[47]},'sample')[-1]
       assert e['event']=='error' and e['states'][index]==before and e['memories']==memories
       e=call({'action':'prepare','mode':mode,'tokens':prefix},'prepared')[-1];assert e['state']==state(prefix,4)
       e=call({'action':'stage','mode':mode,'tokens':[47]},'prepared')[-1];assert e['state']==state(prefix+[47],4)
       e=call({'action':'continue','mode':mode,'continuation':continuation},'sample')[-1]
       assert (e['sample']['reused_tokens'],e['sample']['prefill_tokens'])==(4,1) and e['state']==state(prefix+[47]+continuation,6)
       e=call({'action':'prepare','mode':mode,'tokens':prefix},'prepared')[-1]
       e=call({'action':'append','mode':mode,'tokens':[47,11],'continuation':continuation},'sample')[-1]
       assert (e['sample']['reused_tokens'],e['sample']['prefill_tokens'])==(4,2) and e['state']==state(prefix+[47,11]+continuation,7)
      events=call({'action':'compare','tokens':[1,7,13,3],'continuation':[47,47]},'comparison_done')
      assert [e['event'] for e in events]==['comparison_begin','comparison_row','comparison_row','comparison_done']
      for i,row in enumerate(events[1:-1]):
       assert row['target_index']==i and row['prefix_tokens']==4+i and row['vocabulary_values_compared']==64
       assert row['reference_greedy']==47 and row['online_greedy']==47
       assert math.isfinite(row['reference_nll']) and math.isfinite(row['online_nll']) and row['max_absolute_logit_delta']>=row['rms_logit_delta']>=0
       assert abs(row['online_nll']-row['reference_nll'])<=.05
      assert events[-1]['states']==[state([1,7,13,3,47,47],5)]*2 and events[-1]['memories']==memories
     else:
      prefix=[1,7,13,3]+[47]*27
      for mode,index in [('reference',0),('online',1)]:
       for i in range(3):
        events=call({'action':'profile','mode':mode,'tokens':prefix,'continuation':[47]},'profile_done')
        assert events[0]['event']=='profile_begin' and events[-1]['event']=='profile_done'
        assert events[0]['reused_tokens']==(0 if i==0 else 30) and events[0]['prefill_tokens']==(31 if i==0 else 1)
        ops=events[1:-1];assert len(ops)==38 and events[-1]['operator_count']==38 and [e['index'] for e in ops]==list(range(38))
        assert all(e['event']=='profile_operator' for e in ops)
        names=[e['profile']['name'] for e in ops];assert names.count('attention' if index==0 else 'attention_decode_online')==2
        assert events[-1]['state']==state(prefix+[47],32) and events[-1]['memory']==memories[index]
        t=events[-1]['timing'];assert t['command_count']==1 and t['gpu_end_seconds']>t['gpu_start_seconds']
     worker.stdin.close();assert transport.finish()==0 and not transport.pending and transport.stdout_closed
     run['passed']=True;run['exit_code']=worker.returncode
    finally:
     try:metal_engine.kill_owned_group(worker);worker.wait(timeout=5)
     except BaseException as e:run['cleanup_errors'].append(str(e))
     transport.close()
     for stream in (worker.stdin,worker.stdout,worker.stderr):stream.close()
   result['passed']=all(x['passed'] and not x['cleanup_errors'] for x in result['runs'])
  finally:
   result['endpoints_unchanged']=all(sha(ROOT/p)==v for p,v in pins.items());result['passed'] &= result['endpoints_unchanged']
   with (out/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 assert result['passed'];print('paired ordinary/profile tiny workers passed; cache, pending, append, rejection, quality rows and same-depth profile reuse validated')
if __name__=='__main__':main()
