"""Real immutable tied-head/session replay from prior independent source assets.

All caller/staging and final shared GPU scratch bytes are retained. Overwritten
intermediate child outputs and private lookup scratch are explicitly excluded.
"""
from pathlib import Path
import hashlib,json,re,struct,sys
import check_gemma_real_head as original
same=original.same
N,K,B,TILE=original.N,original.K,original.B,original.TILE
S,G=original.SENTINEL,original.GUARDS
WEIGHTS,WORKSPACE,LOOKUP=226495664,4743496,16851208
RAW_BYTES=54772416
TEST='metal::gemma_affine::projection::tests::gpu::real::real_tied_head_reuses_immutable_weights_and_fresh_projection_scratch'

def sha(raw):return hashlib.sha256(raw).hexdigest()

def policy():
    categories=dict(resident_weights=WEIGHTS,projection_workspace=WORKSPACE,lookup_policy=LOOKUP,source_staging=14155848,
                    two_contexts=134217728,head_reference=4218880,lookup_reference=167936,frozen_metadata=131072,header_bytes=318228,
                    metadata_policy=8388608,caller=4194304,input=24576,lookup_caller=24576,serialization=8388608,
                    diagnostic_state=16777216,hash_io_stack=131072)
    return dict(categories=categories,total_policy_bytes=sum(categories.values()),native_process_allowance_bytes=1024**3,exact_physical_residency=False)

class Assets:
    def __init__(self,model,frozen,lookup):
        self.original=original.Assets(model,frozen)
        try:
            self.input=self.original.input;self.expected=self.original.expected
            with (lookup/'arrays.bin').open('rb') as stream:raw=stream.read(167937)
            if len(raw)!=167936 or sha(raw)!='aea89e9e83d266ae5e648e1c4bebb01509370ef29ef777dea7018b5822cbf469':raise ValueError('original lookup reference differs')
            self.lookup=raw[:24576];self.spans=[];self.fills=[[] for _ in range(16)]
            with model.open('rb') as source:
                for role,prior in zip(('codes','scales','biases'),self.original.spans,strict=True):
                    stride=prior['bytes']//N
                    self.spans.append(dict(role=role,**{k:prior[k] for k in ('name','dtype','shape','offset','bytes')},stride=stride))
                    for index in range(16):
                        offset=prior['offset']+index*TILE*stride;source.seek(offset);remaining=TILE*stride;h=hashlib.sha256()
                        while remaining:
                            part=source.read(min(65536,remaining))
                            if not part:raise ValueError('short original source chunk')
                            remaining-=len(part);h.update(part)
                        self.fills[index].append(dict(offset=offset,bytes=TILE*stride,sha256=h.hexdigest()))
        except BaseException:self.close();raise
    def close(self):self.original.close()

def parse(raw):
    if len(raw)>8*1024**2:raise ValueError('bounded transcript required')
    events=[];wrapper=[]
    for line in raw.decode().splitlines():
        if len(line)>65536:raise ValueError('bounded record required')
        if 'GEMMA_PROJECTION_HEAD_' in line:
            prefix,body=line.split('GEMMA_PROJECTION_HEAD_',1)
            if prefix not in ('','test '+TEST+' ... '):raise ValueError('unexpected test prefix')
            label,text=body.split(' ',1);value=original.base.strict_json(text)
            if type(value)is not dict:raise ValueError('object record required')
            events.append((label,value))
        elif line.strip():wrapper.append(line)
    if len(wrapper)!=3 or wrapper[:2]!=['running 1 test','ok'] or not re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s',wrapper[2]):raise ValueError('exact successful wrapper required')
    return events

def initial():return dict(input=bytes(24604),output=bytes(262172),status=bytes(32),staged=bytearray(4194304),chunk=bytearray(262144))

def descriptor(raw,address):return dict(allocation=address,words=len(raw)//4,bytes=len(raw),contents=dict(sha256=sha(raw),guards=list(struct.unpack('<7I',raw[-28:]))))

def workspace(state,addresses):
    return dict(owned_bytes=WORKSPACE,staged_capacity=1048576,chunk_capacity=65536,poisoned=False,pending=False,shared_read_safe=True,
                staged_sha256=sha(state['staged']),chunk_sha256=sha(state['chunk']),
                **{role:descriptor(state[role],addresses[role]) for role in ('input','output','status')})

def evolve(state,assets,tokens,index):
    inputs=assets.input if tokens==4 else assets.input[index*K*4:(index+1)*K*4]
    expected=assets.expected if tokens==4 else assets.expected[index*N*4:(index+1)*N*4]
    for first in range(0,N,TILE):
        values=b''.join(expected[(token*N+first)*4:(token*N+first+TILE)*4] for token in range(tokens))
        state['input']=inputs+S*(6151-len(inputs)//4)
        state['output']=values+S*(65543-len(values)//4);state['status']=bytes(4)+G
        state['chunk'][:len(values)]=values
        for token in range(tokens):
            start=(token*N+first)*4;state['staged'][start:start+TILE*4]=values[token*TILE*4:(token+1)*TILE*4]
    return inputs,expected

def verify(events,binary,assets):
    if len(binary)!=RAW_BYTES:raise ValueError('complete exact raw capture required')
    stream=iter(events);offset=0;arrays=0;live=set()
    def take(label,wanted=None):
        try:actual,value=next(stream)
        except StopIteration as error:raise ValueError('missing '+label) from error
        if actual!=label:raise ValueError('unexpected '+actual+'/'+label)
        if wanted is not None:same(value,wanted)
        return value
    def pointer(value):
        if type(value)is not str or not re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}',value):raise ValueError('native identity required')
        return value
    def enroll(value):
        pointer(value)
        if value in live:raise ValueError('live buffer alias')
        live.add(value);return value
    def array(key,role,dtype,expected):
        nonlocal offset,arrays
        raw=memoryview(binary)[offset:offset+len(expected)]
        item=dict(key=key,role=role,dtype=dtype,words=len(expected)//4,bytes=len(expected),offset=offset,sha256=sha(raw))
        take('ARRAY_ATTEMPT',item);take('ARRAY',item)
        if raw!=expected:raise ValueError('complete '+key+'/'+role+' bytes differ')
        offset+=len(expected);arrays+=1;return item
    def capture(key,state):return [array(key,role,'U32' if role in ('input','output','status') else 'F32',state[role]) for role in ('input','output','status','staged','chunk')]
    def lookup(index):
        take('LOOKUP_ATTEMPT',dict(index=index,ids=[2,100,257,100]))
        take('LOOKUP_OUTCOME',dict(index=index,completed=True,error=None,caller_sha256=sha(assets.lookup)))
        item=array(f'lookup/{index}','caller','F32',assets.lookup)
        take('LOOKUP',dict(index=index,array=item,private_lookup_scratch_captured=False))
    take('PLAN',policy())
    context=take('CONSTRUCTION_CONTEXT');device_name=context.get('device');device=pointer(context.get('identity'))
    if type(device_name)is not str or not 1<=len(device_name)<=256:raise ValueError('named device required')
    same(context,dict(device=device_name,identity=device));take('SPANS',dict(entries=assets.spans))
    take('UPLOAD_ATTEMPT',dict(working_budget=240651512,maximum_rows=TILE));addresses=None
    for index in range(16):
        row=index*TILE;lengths=[TILE*K//2,TILE*K//32,TILE*K//32]
        actual=take('FILL_ATTEMPT')
        if addresses is None:
            addresses=actual.get('addresses');intervals=[]
            if type(addresses)is not list or len(addresses)!=3:raise ValueError('three source arrays required')
            for value,length in zip(addresses,lengths,strict=True):
                start=int(pointer(value),16);end=start+length
                if end>2**64 or any(start<right and left<end for left,right in intervals):raise ValueError('source intervals overlap or wrap')
                intervals.append((start,end))
        same(actual,dict(first_row=row,rows=TILE,addresses=addresses,lengths=lengths))
        take('FILL',dict(first_row=row,consumed=assets.fills[index]))
    take('UPLOAD_OUTCOME',dict(completed=True,error=None));actual=take('WEIGHTS');chunks=[]
    for original_chunk,observed in zip(assets.original.buffers,actual['chunks'],strict=True):
        child=dict(rows=TILE)
        for role in ('codes','scales','biases'):
            old=original_chunk[role];content=dict(sha256=old['sha256'],guards=old['guards'])
            child[role]=dict(allocation=enroll(observed[role]['allocation']),words=old['words'],bytes=old['bytes'],contents=content)
        chunks.append(child)
    weights=dict(rows=N,columns=K,owned_bytes=WEIGHTS,owner_slots=16,chunk_struct_bytes=112,device=device,chunks=chunks);same(actual,weights)
    take('CONSTRUCTION_CONTEXT_DROPPED',dict(owners=1,device=device))
    take('EXECUTION_CONTEXT',dict(device=device_name,identity=device,same_device=True))
    take('WORKSPACE_ATTEMPT',dict(projection_budget=WORKSPACE,lookup_budget=LOOKUP))
    take('PROJECTION_OUTCOME',dict(completed=True,error=None));take('GATHER_OUTCOME',dict(completed=True,error=None))
    state=initial();actual=take('COLD_OBSERVATION');addresses={role:enroll(actual[role]['allocation']) for role in ('input','output','status')}
    same(actual,workspace(state,addresses));items=capture('cold',state);take('COLD',dict(arrays=items));lookup(0)
    for phase,index,tokens in [('repeat',0,4),('repeat',1,4),('part',0,1),('part',1,1),('part',2,1),('part',3,1),('resume',0,4)]:
        key=f'{phase}/{index}';inputs,expected=evolve(state,assets,tokens,index)
        take('STEP_ATTEMPT',dict(key=key,tokens=tokens,input_sha256=sha(inputs)))
        take('STEP_OUTCOME',dict(key=key,completed=True,error=None,caller_sha256=sha(expected),workspace=workspace(state,addresses),weights=weights))
        items=capture(key,state);caller=array(key,'caller','F32',expected)
        take('STEP',dict(key=key,arrays=items,caller=caller,overwritten_child_device_outputs_separately_captured=False,captured_scratch_is_failed_request_result=False))
    lookup(1);take('FINAL_WEIGHTS',weights)
    result=dict(steps=7,lookups=2,arrays=49,binary_bytes=RAW_BYTES,chunks=16,construction_context_dropped=True,weights_shared_with_lookup=True,
                overwritten_child_device_outputs_separately_captured=False,private_lookup_scratch_captured=False,source_cpu_table_retained=False,
                foreign_device_qualified=False,whole_graph_qualified=False,original_bf16_source_qualified=False,performance_qualified=False)
    take('RESULT',result)
    try:next(stream)
    except StopIteration:pass
    else:raise ValueError('unexpected trailing record')
    same([offset,arrays,len(live)],[RAW_BYTES,49,51])
    return dict(passed=True,**result,returned_projection_values=4194304,returned_lookup_values=12288,observed_gpu_allocations=51,
                aggregate_workload_policy_bytes=policy()['total_policy_bytes'])

if __name__=='__main__':
    if len(sys.argv)!=6:raise SystemExit('LOG BINARY MODEL FROZEN_HEAD FROZEN_LOOKUP')
    log,binary,model,head,lookup=map(Path,sys.argv[1:]);assets=Assets(model,head,lookup)
    try:
        with log.open('rb') as stream:raw=stream.read(8*1024**2+1)
        with binary.open('rb') as stream:body=stream.read(64*1024**2+1)
        print(json.dumps(verify(parse(raw),body,assets),allow_nan=False))
    finally:assets.close()
