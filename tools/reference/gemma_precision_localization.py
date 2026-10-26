"""Bounded counterfactual analysis of the existing real Gemma precision gap.

No Toklet or MLX execution; no replacement reference contract. Original gates
remain, and same-input endpoint replay must succeed before attribution.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import signal
import stat
import struct
import sys
import check_gemma_hybrid_tiny as safe

REFERENCE = {'observation.json':'ca86bd9836c88d5084f770b9ae84ecdb6e1e5a91d775e7cda315702450e918e9',
             'arrays.bin':'352385699238b422b430b6f535c2eac99b6e90543becb4c46caa0c59f930ed93'}
INVENTORY='a3dff3b9d825fcf4fa6f391262732c5f429345682d683d6a39bc551ddacd68e2'
MODEL='038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e'
PREFIX='language_model.model.'
EPSILON=struct.unpack('<f',struct.pack('<f',1e-6))[0]

def f32(value): return struct.unpack('<f',struct.pack('<f',value))[0]

def order():
    names=['embedding','scaled_embedding','ple_lookup','ple_projection','ple_norm']
    for layer in range(35):
        stages=['input','ple_input','input_layernorm','self_attn.q_proj','self_attn.q_norm','q_rotary']
        if layer<15:stages+=['self_attn.k_proj','self_attn.v_proj','self_attn.k_norm','self_attn.v_norm','k_rotary']
        stages+=['context','self_attn.o_proj','post_attention_layernorm','pre_feedforward_layernorm','mlp.gate_proj','mlp.up_proj','mlp.down_proj','post_feedforward_layernorm','per_layer_input_gate','per_layer_projection','post_per_layer_input_norm','output']
        names += [f'layer.{layer}.'+stage for stage in stages]
    return names+['final_norm','head','logits']

def authenticated(reference,inventory):
    raw={name:safe.read(reference/name,128*1024**2) for name in REFERENCE}
    for name,pin in REFERENCE.items():safe.require(hashlib.sha256(raw[name]).hexdigest()==pin,'frozen reference identity')
    data=safe.document(raw['observation.json']);safe.same(data['gate'],dict(atol=1e-5,rtol=1e-4));safe.same(data['tokens'],[2,100,257,100])
    names=order();safe.require(len(names)==713,'complete execution plan')
    specs={}
    for contract in ['math_fp64','staged_fp32']:
        items=[item for item in data['arrays'] if item['contract']==contract]
        safe.same([item['name'] for item in items],names)
        for item in items:
            safe.require(item['completed'] is True,'complete original state')
            block=memoryview(raw['arrays.bin'])[item['offset']:item['offset']+item['bytes']]
            safe.require(hashlib.sha256(block).hexdigest()==item['sha256'],'complete state bytes')
            specs[contract,item['name']]=item
    body=safe.read(inventory,1024**2);safe.require(hashlib.sha256(body).hexdigest()==INVENTORY,'original source inventory')
    return specs,raw['arrays.bin'],safe.document(body)

def values(specs,binary,contract,name):
    spec=specs[contract,name];block=memoryview(binary)[spec['offset']:spec['offset']+spec['bytes']]
    return (v for (v,) in struct.iter_unpack('<d' if contract=='math_fp64' else '<f',block))

def row(specs,binary,contract,name,position):
    spec=specs[contract,name];width=math.prod(spec['shape'][2:]);safe.require(spec['shape'][0:2]==[1,4] and 0<=position<4,'token-major complete row')
    size=8 if contract=='math_fp64' else 4;start=spec['offset']+position*width*size
    return [v for (v,) in struct.iter_unpack('<d' if size==8 else '<f',memoryview(binary)[start:start+width*size])]

def metrics(specs,binary):
    output=[]
    for name in order():
        ideal=values(specs,binary,'math_fp64',name);staged=values(specs,binary,'staged_fp32',name)
        unequal=0;first=None;worst=None;failed=[];count=0
        for index,(a,b) in enumerate(zip(ideal,staged,strict=True)):
            safe.require(math.isfinite(a) and math.isfinite(b),'finite source arrays')
            delta=b-a;ratio=abs(delta)/(1e-5+1e-4*abs(a));value=dict(index=index,ideal=a,staged=b,delta=delta,gate_ratio=ratio)
            if a!=b:
                unequal+=1
                if first is None:first=value
            if worst is None or ratio>worst['gate_ratio']:worst=value
            if ratio>1:failed.append(value)
            count+=1
        output.append(dict(name=name,shape=specs['math_fp64',name]['shape'],values=count,unequal_values=unequal,first_inequality=first,worst=worst,failed=failed))
    return output

def file_identity(s):return s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns

class Source:
    """Whole-container authentication and same-consumption component excerpts."""
    def __init__(self,path,inventory,journal):
        self.path=path;self.inventory=inventory;self.journal=journal
        self.file=os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb')
        self.before=file_identity(os.fstat(self.file.fileno()))
        self.consumed=[]
    def authenticate(self):
        path=self.path;inventory=self.inventory
        safe.require(stat.S_ISREG(self.before[2]) and self.before==file_identity(path.lstat()) and self.before[3]==3550670554,'original bounded file identity')
        safe.require(hashlib.file_digest(self.file,'sha256').hexdigest()==MODEL,'original full container digest')
        self.file.seek(0);safe.require(self.file.read(8)==struct.pack('<Q',318220),'original header extent')
        header=self.file.read(318220);safe.require(hashlib.sha256(header).hexdigest()==inventory['metadata']['header.json']['sha256'],'original header digest')
        self.header=safe.document(header)
    def component(self,name,dtype,shape,selections):
        spec=self.inventory['source_components'][name];header=self.header[name]
        safe.same(header['dtype'],dtype);safe.same(header['shape'],shape)
        safe.same([header['data_offsets'][0]+318228,header['data_offsets'][1]-header['data_offsets'][0]],[spec['offset'],spec['bytes']])
        collected=[bytearray() for _ in selections];hashing=hashlib.sha256();position=0;self.file.seek(spec['offset'])
        attempt=dict(name=name,expected_bytes=spec['bytes'],expected_sha256=spec['sha256'],completed=False)
        self.consumed.append(attempt)
        for start,count in selections:safe.require(type(start)is int and type(count)is int and 0<=start<start+count<=spec['bytes'],'bounded excerpt')
        try:
            while position<spec['bytes']:
                block=self.file.read(min(65536,spec['bytes']-position));safe.require(bool(block),'complete component read');hashing.update(block)
                for out,(start,count) in zip(collected,selections,strict=True):
                    first=max(start,position);last=min(start+count,position+len(block))
                    if first<last:out.extend(block[first-position:last-position])
                position+=len(block)
        finally:
            attempt.update(bytes=position,sha256=hashing.hexdigest(),selections=[dict(offset=start,bytes=count,raw=bytes(raw).hex()) for raw,(start,count) in zip(collected,selections,strict=True)])
            self.journal.write(json.dumps(dict(kind='source_component_attempt',value=attempt))+'\n');self.journal.flush()
        safe.require(hashing.hexdigest()==spec['sha256'],'same consumed original component hash')
        safe.require(all(len(raw)==size for raw,(_,size) in zip(collected,selections,strict=True)),'complete selected source bytes')
        attempt['completed']=True

        return list(map(bytes,collected))
    def finish(self):
        try:
            safe.require(stat.S_ISREG(self.before[2]),'regular source required')
            self.file.seek(0);safe.require(hashlib.file_digest(self.file,'sha256').hexdigest()==MODEL,'consumed container endpoint digest')
            safe.require(file_identity(os.fstat(self.file.fileno()))==self.before==file_identity(self.path.lstat()),'source FD/path unchanged')
        finally:self.file.close()

def packed_rows(source,name,selected):
    matrix=next(v for v in source.inventory['matrices'] if v['name']==name);columns=matrix['columns'];rows=matrix['rows']
    safe.require(columns==1536 and all(0<=v<rows for v in selected),'named matrix row selection')
    components=[]
    for key,dtype,shape,stride in [(name,'U32',[rows,columns//8],columns//2),(name.removesuffix('.weight')+'.scales','BF16',[rows,columns//64],columns//32),(name.removesuffix('.weight')+'.biases','BF16',[rows,columns//64],columns//32)]:
        components.append(source.component(key,dtype,shape,[(index*stride,stride) for index in selected]))
    result={}
    for i,index in enumerate(selected):
        codes,scales,biases=(v[i] for v in components)
        scale=[struct.unpack('<f',struct.pack('<I',v<<16))[0] for (v,) in struct.iter_unpack('<H',scales)]
        bias=[struct.unpack('<f',struct.pack('<I',v<<16))[0] for (v,) in struct.iter_unpack('<H',biases)]
        words=struct.unpack('<'+'I'*(columns//8),codes)
        weights=[f32(((words[j//8]>>(4*(j%8)))&15)*scale[j//64]+bias[j//64]) for j in range(columns)]
        result[index]=dict(codes=codes.hex(),scales=scales.hex(),biases=biases.hex(),columns=columns,weights=weights)
    return result

def contribution(ideal,staged,parts):
    return dict(ideal=ideal,staged=staged,signed_delta=staged-ideal,parts=parts,residual=math.fsum([staged,-ideal,*[-v for v in parts.values()]]))

def assert_endpoints(actual,ideal,staged):
    safe.require(abs(actual[0]-ideal)<=1e-13+1e-13*abs(ideal),'same-input ideal endpoint replay')
    safe.require(struct.pack('<f',actual[1])==struct.pack('<f',staged),'same-input staged endpoint replay')

def analyze(specs,binary,inventory,source,result,journal):
    summary=metrics(specs,binary);result['states']=summary;failed={v['name']:v['failed'] for v in summary if v['failed']}
    safe.same({k:len(v) for k,v in failed.items()},{'layer.14.post_feedforward_layernorm':1,'layer.15.post_feedforward_layernorm':1,'layer.26.self_attn.q_proj':1,'head':27,'logits':27})
    cases=result['cases'];e=row(specs,binary,'math_fp64','embedding',0)[0];ideal=row(specs,binary,'math_fp64','scaled_embedding',0)[0];staged=row(specs,binary,'staged_fp32','scaled_embedding',0)[0]
    base=e*math.sqrt(1536);constant=e*f32(math.sqrt(1536));stored=f32(constant)
    cases.append(dict(kind='embedding_scale',index=0,input=e,ideal_constant=math.sqrt(1536),staged_constant=f32(math.sqrt(1536)),
        output_only_rounding=f32(base),replayed=dict(ideal=base,staged=stored),decomposition=contribution(ideal,staged,dict(constant_rounding=constant-base,output_storage=stored-constant))))
    for layer in [14,15]:
        name=f'layer.{layer}.post_feedforward_layernorm';index=failed[name][0]['index'];position,feature=divmod(index,1536)
        x=row(specs,binary,'math_fp64',f'layer.{layer}.mlp.down_proj',position);y=row(specs,binary,'staged_fp32',f'layer.{layer}.mlp.down_proj',position)
        vector=PREFIX+f'layers.{layer}.post_feedforward_layernorm.weight'
        raw=source.component(vector,'BF16',[1536],[(0,3072)])[0];scale=struct.unpack('<f',struct.pack('<I',struct.unpack_from('<H',raw,2*feature)[0]<<16))[0]
        a=math.sqrt(math.fsum(v*v for v in x)/1536+EPSILON);b=math.sqrt(math.fsum(v*v for v in y)/1536+EPSILON)
        baseline=x[feature]/a*scale;numerator=y[feature]/a*scale;combined=y[feature]/b*scale;stored=f32(combined)
        ideal=row(specs,binary,'math_fp64',name,position)[feature];staged=row(specs,binary,'staged_fp32',name,position)[feature]
        cases.append(dict(kind='rms',name=name,index=index,position=position,feature=feature,input_name=f'layer.{layer}.mlp.down_proj',
            source_vector=vector,source_bf16=raw.hex(),scale=scale,epsilon=EPSILON,ideal_denominator=a,staged_input_denominator=b,
            ideal_input_output_rounding=f32(baseline),replayed=dict(ideal=baseline,staged=stored),decomposition=contribution(ideal,staged,dict(numerator_input=numerator-baseline,denominator_input=combined-numerator,output_storage=stored-combined))))
    for name,input_name,matrix in [('layer.26.self_attn.q_proj','layer.26.input_layernorm',PREFIX+'layers.26.self_attn.q_proj.weight'),('head','final_norm',PREFIX+'embed_tokens.weight')]:
        width=specs['math_fp64',name]['shape'][-1];indices=failed[name];selected=sorted({v['index']%width for v in indices});weights=packed_rows(source,matrix,selected)
        for item in indices:
            position,feature=divmod(item['index'],width);x=row(specs,binary,'math_fp64',input_name,position);y=row(specs,binary,'staged_fp32',input_name,position);w=weights[feature]['weights']
            baseline=math.fsum(a*b for a,b in zip(x,w,strict=True));changed=math.fsum(a*b for a,b in zip(y,w,strict=True));stored=f32(changed)
            ideal=item['ideal'];staged=item['staged']
            case=dict(kind='projection',name=name,index=item['index'],position=position,feature=feature,input_name=input_name,source_matrix=matrix,
                packed={k:v for k,v in weights[feature].items() if k!='weights'},ideal_input_output_rounding=f32(baseline),replayed=dict(ideal=baseline,staged=stored),
                decomposition=contribution(ideal,staged,dict(inherited_input=changed-baseline,output_storage=stored-changed)))
            if name=='head':
                ideal_logit=row(specs,binary,'math_fp64','logits',position)[feature];staged_logit=row(specs,binary,'staged_fp32','logits',position)[feature]
                a=math.tanh(ideal/30)*30;b=math.tanh(staged/30)*30;c=math.tanh(f32(staged/30))*30;d=f32(math.tanh(f32(staged/30)))*30;end=f32(d)
                case['softcap_replayed']=dict(ideal=a,staged=end)
                case['softcap']=contribution(ideal_logit,staged_logit,dict(inherited_head=b-a,division_storage=c-b,tanh_storage=d-c,final_storage=end-d))
            cases.append(case)
    safe.require(len(cases)==31,'complete localization case set')
    for index,case in enumerate(cases):
        journal.write(json.dumps(dict(kind='counterfactual_attempt',index=index,value=case))+'\n');journal.flush()
        actual=case['replayed'];saved=case['decomposition']
        assert_endpoints((actual['ideal'],actual['staged']),saved['ideal'],saved['staged'])
        if 'softcap' in case:
            actual=case['softcap_replayed'];saved=case['softcap']
            assert_endpoints((actual['ideal'],actual['staged']),saved['ideal'],saved['staged'])
        case['same_input_endpoints_passed']=True
    result.update(first_inequality=next(dict(name=v['name'],value=v['first_inequality']) for v in summary if v['unequal_values']),
        first_tolerance_failure=next(dict(name=v['name'],value=v['failed'][0]) for v in summary if v['failed']),failed_values=sum(len(v['failed']) for v in summary))

if __name__=='__main__':
    output,reference,inventory_path,model=map(Path,sys.argv[1:]);output.mkdir()
    report=dict(format='toklet-gemma-precision-localization-v1',passed=False,primary_error=None,endpoint_errors=[],scope='Frozen reference counterfactual analysis only; not a native/GPU defect or product precision decision.',decomposition_policy='Ordered counterfactual path; nonlinear attribution depends on order. Reconstruction residual includes frozen BLAS versus fsum reduction differences, not production GPU error.',
        original_gates_changed=False,original_failures_waived=False,native_engine_executed=False,performance_qualified=False,full_model_window_qualified=False,app_eligibility_changed=False)
    source=None;journal=(output/'capture.jsonl').open('x');report['analysis']=dict(cases=[])
    def stop(number,_):
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
        raise InterruptedError(str(number))
    old={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        specs,binary,inventory=authenticated(reference,inventory_path);source=Source(model,inventory,journal)
        report['consumed_components']=source.consumed;source.authenticate();analyze(specs,binary,inventory,source,report['analysis'],journal);source.finish();source=None;report['passed']=True
    except BaseException as error:report['primary_error']=type(error).__name__+': '+str(error)
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        if source is not None and not source.file.closed:
            try:source.finish()
            except BaseException as error:report['endpoint_errors'].append(type(error).__name__+': '+str(error))
        journal.close()
        try:
            with (output/'observation.json').open('x')as file:json.dump(report,file,indent=2,allow_nan=False);file.write('\n')
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
    print(json.dumps(dict(passed=report['passed'],primary_error=report['primary_error'],endpoint_errors=report['endpoint_errors'])))
    sys.exit(0 if report['passed'] and not report['endpoint_errors'] else 1)
