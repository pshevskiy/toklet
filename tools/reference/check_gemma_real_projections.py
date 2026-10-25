"""Stream real GPU outputs and reconstruct whole-buffer integrity independently.

Original GPU weight buffers are retained as cryptographic observations, not
plaintext copies. Complete active outputs and native ordered rows are retained.
"""
from pathlib import Path
import contextlib,hashlib,json,math,re,struct,sys

MODEL_BYTES=3550670554
MODEL_SHA='038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e'
SENTINEL=struct.pack('<I',0x5a170031)
GUARDS=SENTINEL*7

def digest(path,expected_bytes):
    result=hashlib.sha256();total=0
    with path.open('rb') as stream:
        while True:
            block=stream.read(min(65536,expected_bytes-total+1))
            if not block:break
            total+=len(block)
            if total>expected_bytes:raise ValueError('authenticated file exceeds exact byte bound')
            result.update(block)
    if total!=expected_bytes:raise ValueError('authenticated file is short')
    return result.hexdigest()

def same(a,b):
    if type(a) is not type(b):raise ValueError('typed observation differs')
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('observation fields differ')
        for key in a:same(a[key],b[key])
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('observation length differs')
        for x,y in zip(a,b,strict=True):same(x,y)
    elif a!=b:raise ValueError('observation value differs')

def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON field')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('finite JSON required')))

def load(path,limit,sha=None):
    with path.open('rb') as stream:raw=stream.read(limit+1)
    if len(raw)>limit or sha is not None and hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('bounded authenticated document required')
    return strict_json(raw)

def identity(raw):
    if len(raw)%4 or len(raw)<28:raise ValueError('complete guarded buffer required')
    return dict(words=len(raw)//4,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),guards=[0x5a170031]*7)

class Assets:
    """Authenticated cold files; no expanded model or full result tensor held."""
    def __init__(self,model,frozen):
        if model.stat().st_size!=MODEL_BYTES or digest(model,MODEL_BYTES)!=MODEL_SHA:raise ValueError('immutable checkpoint differs')
        self.reference=load(frozen/'observation.json',2*1024**2,'6ac7c122fb89c3afadb3bef39b4a15118100cff990130851129b3771beff87e4')
        load(frozen/'manifest.json',65536,'89827c6b24c6e626bc96d95e63cae787a8b5fecd4754880d8176d191cdd2113c')
        if (frozen/'arrays.bin').stat().st_size!=26333184 or digest(frozen/'arrays.bin',26333184)!='037ca2048d2ee9f1fea58bd3ea1ba6008543dc8a13e7a064e2d755817c7f021e':raise ValueError('immutable reference binary differs')
        same(self.reference['passed'],True)
        self.handles=contextlib.ExitStack()
        try:
            self.model=self.handles.enter_context(model.open('rb'))
            self.frozen=self.handles.enter_context((frozen/'arrays.bin').open('rb'))
            prefix=self.model.read(8)
            if len(prefix)!=8 or struct.unpack('<Q',prefix)[0]!=318220:raise ValueError('checkpoint header length differs')
            raw=self.model.read(318220)
            if hashlib.sha256(raw).hexdigest()!='4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7':raise ValueError('checkpoint header differs')
            self.header=strict_json(raw);self.base=318228
            self.cases=[row for row in self.reference['records'] if row['status']!='geometry-rejected']
            if len(self.cases)!=276:raise ValueError('complete fitting projection inventory required')
        except BaseException:
            self.close();raise
    def close(self):
        self.handles.close()
    def array(self,entry,width):
        if entry['dtype']!='F32' or entry['shape']!=[4,width] or entry['bytes']!=width*16 or entry['completed'] is not True or not 0<entry['bytes']<=196608:raise ValueError('complete reference array shape differs')
        self.frozen.seek(entry['offset']);raw=self.frozen.read(entry['bytes'])
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('reference array bytes differ')
        return raw
    def weights(self,case):
        n=case['rows'];k=case['columns'];stem='language_model.model.'+case['name'][:-7]
        specifications=[(stem+'.weight',[n,k//8],'U32',n*k//2),
                        (stem+'.scales',[n,k//64],'BF16',n*k//32),
                        (stem+'.biases',[n,k//64],'BF16',n*k//32)]
        spans=[];buffers={}
        for role,(name,shape,dtype,count) in zip(('codes','scales','biases'),specifications,strict=True):
            tensor=self.header[name]
            if tensor['shape']!=shape or tensor['dtype']!=dtype or tensor['data_offsets'][1]-tensor['data_offsets'][0]!=count:raise ValueError('independent tensor geometry differs')
            offset=self.base+tensor['data_offsets'][0]
            if not 0<count<=128*1024**2 or offset+count>MODEL_BYTES:raise ValueError('bounded source span required')
            self.model.seek(offset);plain=hashlib.sha256();remaining=count
            while remaining:
                raw=self.model.read(min(65536,remaining))
                if not raw:raise ValueError('short checkpoint span')
                remaining-=len(raw);plain.update(raw)
            gpu=plain.copy();gpu.update(b'\0'*((-count)%4));gpu.update(GUARDS)
            words=(count+3)//4+7
            buffers[role]=dict(words=words,bytes=words*4,sha256=gpu.hexdigest(),guards=[0x5a170031]*7)
            spans.append(dict(name=name,shape=shape,dtype=dtype,offset=offset,bytes=count,sha256=plain.hexdigest()))
        return spans,buffers

def parse(raw):
    if len(raw)>32*1024**2:raise ValueError('bounded native transcript required')
    rows={}
    for line in raw.decode().splitlines():
        if 'GEMMA_REAL_' not in line:continue
        if len(line)>65536:raise ValueError('bounded observation line required')
        prefix,_,tail=line.partition('GEMMA_REAL_')
        if prefix not in ('','test metal::gemma_affine::tests::real::real_checkpoint_projections ... '):raise ValueError('unexpected test transcript prefix')
        label,body=tail.split(' ',1);rows.setdefault(label,[]).append(strict_json(body))
    if not re.search(rb'test result: ok\. 1 passed; 0 failed; 0 ignored;',raw):raise ValueError('completed exact native test required')
    return rows

def verify(rows,binary,assets,*,overrides=None):
    """overrides is used only by explicit post-capture mutation tests."""
    same({key:len(value) for key,value in rows.items()},dict(META=1,CASE_ATTEMPT=276,CASE=276,ORDERED_ATTEMPT=276,ORDERED_ROW_ATTEMPT=1104,ORDERED_ROW=1104,ORDERED=276,
        ARRAY_ATTEMPT=3036,ARRAY=3036,STEP_ATTEMPT=1932,STEP_OUTCOME=1932,STEP=1932,HEAD_REJECTION=1,FINAL=1))
    same(rows['ARRAY_ATTEMPT'],rows['ARRAY'])
    meta=rows['META'][0]
    same(sorted(meta),sorted(('device','simd_width','owner_struct_bytes','context_allowance_bytes','binary_cap_bytes','whole_model_resident','native_activation_parity','reference_contract')))
    if type(meta['device']) is not str or not meta['device'] or type(meta['simd_width']) is not int or not 1<=meta['simd_width']<=65535:raise ValueError('named device/capability required')
    owner=meta['owner_struct_bytes']
    if type(owner) is not int or not 0<owner<=4096:raise ValueError('owner size required')
    same(meta['context_allowance_bytes'],67108864);same(meta['binary_cap_bytes'],128*1024**2)
    same(meta['whole_model_resident'],False);same(meta['native_activation_parity'],False)
    same(meta['reference_contract'],assets.reference['expected_contract'])
    if binary.stat().st_size!=76021760:raise ValueError('complete raw output size required')
    array_index=step_index=ordered_index=0;offset=0;outputs=comparisons=ordered_comparisons=0
    max_error=maximum_fraction=ordered_error=0.;max_owned=0
    with binary.open('rb') as stream:
        def payload(entry,key,role,tokens,n):
            nonlocal array_index,offset
            same(entry,rows['ARRAY'][array_index]);array_index+=1
            count=tokens*n*4
            if not 0<count<=196608:raise ValueError('bounded complete raw result required')
            stream.seek(offset);raw=stream.read(count)
            if overrides is not None and offset in overrides:raw=overrides[offset]
            expected=dict(key=key,role=role,dtype='F32',shape=[tokens,n],offset=offset,bytes=count,sha256=hashlib.sha256(raw).hexdigest())
            same(entry,expected)
            if len(raw)!=count:raise ValueError('short captured array')
            offset+=count
            if any(not math.isfinite(v) for (v,) in struct.iter_unpack('<f',raw)):raise ValueError('nonfinite numerical result')
            return raw
        for index,case in enumerate(assets.cases):
            name=case['name'];n=case['rows'];k=case['columns'];evidence=case['input_evidence']
            same(rows['CASE_ATTEMPT'][index],dict(name=name,rows=n,columns=k,evidence=evidence))
            spans,buffers=assets.weights(case)
            owned=(sum(item['words'] for item in buffers.values())+4*k+7+4*n+7+8)*4+owner
            max_owned=max(max_owned,owned)
            declared=rows['CASE'][index]
            same(declared,dict(name=name,rows=n,columns=k,batch=4,evidence=evidence,reference_admission=case['status'],owned_bytes=owned,loaded_spans=spans,host_tensor_bytes=declared['host_tensor_bytes']))
            if type(declared['host_tensor_bytes']) is not int or declared['host_tensor_bytes']<sum(v['bytes'] for v in spans) or declared['host_tensor_bytes']>128*1024**2:raise ValueError('bounded original host tensor accounting required')
            input_raw=assets.array(case['arrays']['input'],k);math_raw=assets.array(case['arrays']['expected_staged_fp32'],n)
            same(rows['ORDERED_ATTEMPT'][index],dict(name=name,contract='native ordered FP64 dot, output rounded FP32'))
            ordered_parts=[];ordered_raw=b''
            for token in range(4):
                same(rows['ORDERED_ROW_ATTEMPT'][ordered_index],dict(name=name,token=token))
                record=rows['ORDERED_ROW'][ordered_index];ordered_index+=1
                same({key:value for key,value in record.items() if key!='array'},dict(name=name,token=token,completed=True,error=None))
                raw=payload(record['array'],f'{name}/ordered/{token}','native-ordered-fp32',1,n)
                ordered_parts.append(record['array']);ordered_raw+=raw
            same(rows['ORDERED'][index],dict(name=name,arrays=ordered_parts,contract='native ordered FP64 dot, output rounded FP32'))
            for (a,),(b,) in zip(struct.iter_unpack('<f',ordered_raw),struct.iter_unpack('<f',math_raw),strict=True):
                error=abs(a-b);gate=1e-5+1e-4*abs(b)
                if error>gate:raise ValueError(f'native ordered/mathematical gate failed: {name}')
                ordered_comparisons+=1;ordered_error=max(ordered_error,error)
            baseline=None
            for phase,phase_index,tokens in [('repeat',0,4),('repeat',1,4),*[('part',token,1) for token in range(4)],('resume',0,4)]:
                key=f'{name}/{phase}/{phase_index}'
                active_input=input_raw if tokens==4 else input_raw[phase_index*k*4:(phase_index+1)*k*4]
                same(rows['STEP_ATTEMPT'][step_index],dict(key=key,name=name,phase=phase,index=phase_index,tokens=tokens,input_sha256=hashlib.sha256(active_input).hexdigest()))
                result=rows['STEP'][step_index];outcome=rows['STEP_OUTCOME'][step_index];step_index+=1
                same(outcome,{field:value for field,value in result.items() if field!='output'})
                raw=payload(result['output'],key,'device-output',tokens,n)
                expected_buffers=dict(buffers,input=identity(active_input+SENTINEL*((4-tokens)*k)+GUARDS),
                                      output=identity(raw+SENTINEL*((4-tokens)*n)+GUARDS),status=identity(b'\0'*4+GUARDS))
                same(result,dict(key=key,name=name,phase=phase,index=phase_index,tokens=tokens,completed=True,error=None,poisoned=False,pending=False,owned_bytes=owned,
                                 caller_sha256=hashlib.sha256(raw).hexdigest(),device=expected_buffers,output=result['output']))
                if phase=='repeat':
                    outputs+=len(raw)//4
                    for contract,reference in [('rounded-mathematical',math_raw),('native-ordered-fp32',ordered_raw)]:
                        for (a,),(b,) in zip(struct.iter_unpack('<f',raw),struct.iter_unpack('<f',reference),strict=True):
                            error=abs(a-b);gate=1e-5+1e-4*abs(b)
                            if error>gate:raise ValueError(f'GPU {contract} gate failed: {name}/{phase_index}')
                            comparisons+=1;max_error=max(max_error,error);maximum_fraction=max(maximum_fraction,error/gate)
                    if phase_index==0:baseline=raw
                    elif raw!=baseline:raise ValueError('repeat output bits differ')
                elif phase=='part':
                    if raw!=baseline[phase_index*n*4:(phase_index+1)*n*4]:raise ValueError('partition output bits differ')
                elif raw!=baseline:raise ValueError('full resume output bits differ')
        if (array_index,step_index,ordered_index,offset)!=(3036,1932,1104,76021760):raise ValueError('complete output ownership inventory differs')
    same(rows['HEAD_REJECTION'],[dict(name='embed_tokens.weight',rejected=True,error='Gemma affine fixed geometry/format outside bounds',large_spans_read=False)])
    same(rows['FINAL'],[dict(cases=276,captured_state_inputs=206,derived_recipe_inputs=70,binary_bytes=76021760,head_geometry_rejected=True,whole_graph_qualified=False)])
    return dict(passed=True,cases=276,captured_state_inputs=206,derived_recipe_inputs=70,repeats=2,partitions=1104,resumes=276,
                numerical_output_values=outputs,gpu_contract_comparisons=comparisons,native_ordered_math_comparisons=ordered_comparisons,
                maximum_absolute_error=max_error,maximum_gate_fraction=maximum_fraction,maximum_native_ordered_math_error=ordered_error,
                maximum_owned_bytes=max_owned,owner_struct_bytes=owner,raw_binary_bytes=offset,device=meta['device'],simd_width=meta['simd_width'],
                whole_device_buffer_digests_checked=True,whole_gpu_weights_retained_as_plaintext=False,native_activation_parity=False,whole_graph_qualified=False)

if __name__=='__main__':
    if len(sys.argv)!=5:raise SystemExit('native-stdout native-output-binary checkpoint frozen-reference-directory')
    stdout,binary,model,frozen=[Path(v) for v in sys.argv[1:]]
    with stdout.open('rb') as stream:raw=stream.read(32*1024**2+1)
    assets=Assets(model,frozen)
    try:print(json.dumps(verify(parse(raw),binary,assets)))
    finally:assets.close()
