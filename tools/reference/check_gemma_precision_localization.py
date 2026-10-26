"""Independent frozen-state, source-excerpt and counterfactual replay.

No producer or engine imports. Explicitly preserves the original 57 failures.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import stat
import struct
import sys
import check_gemma_hybrid_tiny as safe

JSON_SHA='ca86bd9836c88d5084f770b9ae84ecdb6e1e5a91d775e7cda315702450e918e9'
BINARY_SHA='352385699238b422b430b6f535c2eac99b6e90543becb4c46caa0c59f930ed93'
INVENTORY_SHA='a3dff3b9d825fcf4fa6f391262732c5f429345682d683d6a39bc551ddacd68e2'
PREFIX='language_model.model.'
R=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]

class Frozen:
    def __init__(self,directory):
        data=safe.read(directory/'observation.json',2*1024**2);self.raw=safe.read(directory/'arrays.bin',128*1024**2)
        safe.require(hashlib.sha256(data).hexdigest()==JSON_SHA and hashlib.sha256(self.raw).hexdigest()==BINARY_SHA,'original full reference identity')
        record=safe.document(data);safe.same(record['gate'],dict(atol=1e-5,rtol=1e-4))
        self.items={(v['contract'],v['name']):v for v in record['arrays']};self.order=[v['name'] for v in record['arrays'] if v['contract']=='math_fp64']
        safe.require(len(self.items)==1426 and len(self.order)==len(set(self.order))==713,'all original states')
        for item in self.items.values():
            safe.require(item['completed'] is True and hashlib.sha256(self.block(item)).hexdigest()==item['sha256'],'complete frozen state')
    def block(self,item):return memoryview(self.raw)[item['offset']:item['offset']+item['bytes']]
    def pairs(self,name):
        return zip(struct.iter_unpack('<d',self.block(self.items['math_fp64',name])),struct.iter_unpack('<f',self.block(self.items['staged_fp32',name])),strict=True)
    def at(self,contract,name,index):
        item=self.items[contract,name];code,step=('<d',8) if contract=='math_fp64' else ('<f',4)
        return struct.unpack_from(code,self.raw,item['offset']+index*step)[0]
    def row(self,contract,name,position):
        item=self.items[contract,name];safe.same(item['shape'][:2],[1,4]);width=math.prod(item['shape'][2:])
        return [self.at(contract,name,position*width+i) for i in range(width)]

def state_summary(frozen):
    result=[]
    for name in frozen.order:
        first=worst=None;failed=[];different=total=0
        for index,((a,),(b,)) in enumerate(frozen.pairs(name)):
            safe.require(math.isfinite(a) and math.isfinite(b),'finite source state')
            d=b-a;q=abs(d)/(1e-5+1e-4*abs(a));record=dict(index=index,ideal=a,staged=b,delta=d,gate_ratio=q)
            if a!=b:
                different+=1
                if first is None:first=record
            if worst is None or q>worst['gate_ratio']:worst=record
            if q>1:failed.append(record)
            total+=1
        result.append(dict(name=name,shape=frozen.items['math_fp64',name]['shape'],values=total,unequal_values=different,first_inequality=first,worst=worst,failed=failed))
    return result

def split(ideal,staged,parts):
    return dict(ideal=ideal,staged=staged,signed_delta=staged-ideal,parts=parts,residual=math.fsum([staged,-ideal,*[-v for v in parts.values()]]))

def weights(packed):
    safe.same(set(packed),{'codes','scales','biases','columns'});safe.same(packed['columns'],1536)
    raw={key:bytes.fromhex(packed[key]) for key in ['codes','scales','biases']}
    safe.require([len(raw[k]) for k in raw]==[768,48,48],'complete original packed row')
    scales=[struct.unpack('<f',b'\0\0'+part)[0] for part in [raw['scales'][i:i+2] for i in range(0,48,2)]]
    biases=[struct.unpack('<f',b'\0\0'+part)[0] for part in [raw['biases'][i:i+2] for i in range(0,48,2)]]
    codes=[(value>>shift)&15 for value in raw['codes'] for shift in (0,4)]
    return [R(float(code)*scales[i//64]+biases[i//64]) for i,code in enumerate(codes)]

def endpoints(replayed,saved):
    safe.require(abs(replayed['ideal']-saved['ideal'])<=1e-13+1e-13*abs(saved['ideal']),'same-input FP64 endpoint')
    safe.require(struct.pack('<f',replayed['staged'])==struct.pack('<f',saved['staged']),'same-input FP32 endpoint')

def case_math(frozen,case):
    kind=case['kind']
    if kind=='embedding_scale':
        a=frozen.at('math_fp64','scaled_embedding',0);b=frozen.at('staged_fp32','scaled_embedding',0);x=frozen.at('math_fp64','embedding',0)
        base=x*math.sqrt(1536);constant=x*R(math.sqrt(1536));end=R(constant)
        wanted=dict(kind=kind,index=0,input=x,ideal_constant=math.sqrt(1536),staged_constant=R(math.sqrt(1536)),output_only_rounding=R(base),
            replayed=dict(ideal=base,staged=end),decomposition=split(a,b,dict(constant_rounding=constant-base,output_storage=end-constant)))
    elif kind=='rms':
        name=case['name'];index=case['index'];position,feature=divmod(index,1536);input_name=name.replace('post_feedforward_layernorm','mlp.down_proj')
        x=frozen.row('math_fp64',input_name,position);y=frozen.row('staged_fp32',input_name,position);raw=bytes.fromhex(case['source_bf16']);safe.require(len(raw)==3072,'complete learned norm')
        scale=struct.unpack('<f',b'\0\0'+raw[feature*2:feature*2+2])[0];epsilon=R(1e-6)
        a=math.sqrt(math.fsum(v*v for v in x)/len(x)+epsilon);b=math.sqrt(math.fsum(v*v for v in y)/len(y)+epsilon)
        ideal=x[feature]/a*scale;numerator=y[feature]/a*scale;changed=y[feature]/b*scale;end=R(changed)
        saved_a=frozen.at('math_fp64',name,index);saved_b=frozen.at('staged_fp32',name,index)
        layer=name.split('.')[1]
        wanted=dict(kind=kind,name=name,index=index,position=position,feature=feature,input_name=input_name,source_vector=PREFIX+f'layers.{layer}.post_feedforward_layernorm.weight',
            source_bf16=raw.hex(),scale=scale,epsilon=epsilon,ideal_denominator=a,staged_input_denominator=b,ideal_input_output_rounding=R(ideal),
            replayed=dict(ideal=ideal,staged=end),decomposition=split(saved_a,saved_b,dict(numerator_input=numerator-ideal,denominator_input=changed-numerator,output_storage=end-changed)))
    elif kind=='projection':
        name=case['name'];index=case['index'];width=frozen.items['math_fp64',name]['shape'][-1];position,feature=divmod(index,width)
        input_name='final_norm' if name=='head' else 'layer.26.input_layernorm';matrix=PREFIX+('embed_tokens.weight' if name=='head' else 'layers.26.self_attn.q_proj.weight')
        x=frozen.row('math_fp64',input_name,position);y=frozen.row('staged_fp32',input_name,position);w=weights(case['packed'])
        a=math.fsum(w[i]*x[i] for i in range(1536));b=math.fsum(w[i]*y[i] for i in range(1536));end=R(b)
        ideal=frozen.at('math_fp64',name,index);staged=frozen.at('staged_fp32',name,index)
        wanted=dict(kind=kind,name=name,index=index,position=position,feature=feature,input_name=input_name,source_matrix=matrix,packed=case['packed'],ideal_input_output_rounding=R(a),
            replayed=dict(ideal=a,staged=end),decomposition=split(ideal,staged,dict(inherited_input=b-a,output_storage=end-b)))
        if name=='head':
            baseline=30*math.tanh(ideal/30);inherited=30*math.tanh(staged/30);division=30*math.tanh(R(staged/30));tanh=30*R(math.tanh(R(staged/30)));final=R(tanh)
            ia=frozen.at('math_fp64','logits',index);ib=frozen.at('staged_fp32','logits',index)
            wanted['softcap_replayed']=dict(ideal=baseline,staged=final)
            wanted['softcap']=split(ia,ib,dict(inherited_head=inherited-baseline,division_storage=division-inherited,tanh_storage=tanh-division,final_storage=final-tanh))
    else:raise ValueError('unknown counterfactual kind')
    wanted['same_input_endpoints_passed']=True
    safe.same(case,wanted);endpoints(wanted['replayed'],wanted['decomposition'])
    if 'softcap' in wanted:endpoints(wanted['softcap_replayed'],wanted['softcap'])
    return wanted

def source_checks(report,inventory,model):
    requests=[];cases=report['analysis']['cases']
    for case in cases:
        if case['kind']=='rms':requests.append((case['source_vector'],[(0,3072,bytes.fromhex(case['source_bf16']))]))
    for matrix in [PREFIX+'layers.26.self_attn.q_proj.weight',PREFIX+'embed_tokens.weight']:
        selected=sorted((c for c in cases if c.get('source_matrix')==matrix),key=lambda c:c['feature'])
        for role,name,stride in [('codes',matrix,768),('scales',matrix.removesuffix('.weight')+'.scales',48),('biases',matrix.removesuffix('.weight')+'.biases',48)]:
            requests.append((name,[(c['feature']*stride,stride,bytes.fromhex(c['packed'][role])) for c in selected]))
    safe.require(len(requests)==len(report['consumed_components'])==8,'all selected components')
    def ident(s):return s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns
    with os.fdopen(os.open(model,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb')as file:
        original=ident(os.fstat(file.fileno()));safe.require(stat.S_ISREG(original[2]) and original==ident(model.lstat()) and original[3]==3550670554,'original regular source identity')
        safe.require(file.read(8)==struct.pack('<Q',318220),'original header length')
        safe.require(hashlib.sha256(file.read(318220)).hexdigest()==inventory['metadata']['header.json']['sha256'],'original header identity')
        for (name,selections),actual in zip(requests,report['consumed_components'],strict=True):
            spec=inventory['source_components'][name];file.seek(spec['offset']);remaining=spec['bytes'];position=0;digest=hashlib.sha256();copied=[bytearray() for _ in selections]
            while remaining:
                block=file.read(min(131072,remaining));safe.require(bool(block),'complete source component');digest.update(block)
                for i,(start,count,_) in enumerate(selections):
                    left=max(position,start);right=min(position+len(block),start+count)
                    if left<right:copied[i].extend(block[left-position:right-position])
                remaining-=len(block);position+=len(block)
            safe.require(digest.hexdigest()==spec['sha256'],'original consumed component SHA')
            for raw,(_,_,expected) in zip(copied,selections,strict=True):safe.require(raw==expected,'source-backed excerpt identity')
            safe.same(actual,dict(name=name,expected_bytes=spec['bytes'],expected_sha256=spec['sha256'],completed=True,bytes=spec['bytes'],sha256=spec['sha256'],
                selections=[dict(offset=start,bytes=count,raw=raw.hex()) for start,count,raw in selections]))
        safe.require(original==ident(os.fstat(file.fileno()))==ident(model.lstat()),'source descriptor/path stable')

def replay(report,journal,frozen,inventory,model):
    safe.same(set(report),{'format','passed','primary_error','endpoint_errors','scope','original_gates_changed','original_failures_waived','native_engine_executed','performance_qualified','full_model_window_qualified','app_eligibility_changed','analysis','consumed_components','decomposition_policy'})
    safe.same(report['format'],'toklet-gemma-precision-localization-v1');safe.same(report['passed'],True);safe.same(report['primary_error'],None);safe.same(report['endpoint_errors'],[])
    safe.same(report['scope'],'Frozen reference counterfactual analysis only; not a native/GPU defect or product precision decision.')
    safe.same(report['decomposition_policy'],'Ordered counterfactual path; nonlinear attribution depends on order. Reconstruction residual includes frozen BLAS versus fsum reduction differences, not production GPU error.')
    for key in ['original_gates_changed','original_failures_waived','native_engine_executed','performance_qualified','full_model_window_qualified','app_eligibility_changed']:safe.same(report[key],False)
    summary=state_summary(frozen);a=report['analysis'];safe.same(set(a),{'cases','states','first_inequality','first_tolerance_failure','failed_values'});safe.same(a['states'],summary)
    safe.same(a['first_inequality'],next(dict(name=s['name'],value=s['first_inequality']) for s in summary if s['unequal_values']))
    safe.same(a['first_tolerance_failure'],next(dict(name=s['name'],value=s['failed'][0]) for s in summary if s['failed']))
    safe.same(a['failed_values'],57)
    expected=[('embedding_scale',None,0)]
    for name in ['layer.14.post_feedforward_layernorm','layer.15.post_feedforward_layernorm','layer.26.self_attn.q_proj','head']:
        state=next(s for s in summary if s['name']==name)
        expected += [('rms' if 'post_feedforward' in name else 'projection',name,v['index']) for v in state['failed']]
    safe.same([(c['kind'],c.get('name'),c['index']) for c in a['cases']],expected)
    for case in a['cases']:case_math(frozen,case)
    source_checks(report,inventory,model)
    safe.require(len(journal)==39,'complete durable attempts')
    for actual,source in zip(journal[:8],report['consumed_components'],strict=True):safe.same(actual,dict(kind='source_component_attempt',value=dict(source,completed=False)))
    for i,(actual,case) in enumerate(zip(journal[8:],a['cases'],strict=True)):
        safe.same(actual,dict(kind='counterfactual_attempt',index=i,value={k:v for k,v in case.items() if k!='same_input_endpoints_passed'}))
    return dict(passed=True,states=713,failed_values_retained=57,localized_cases=len(a['cases']),same_input_endpoints=True,source_components=8,
        first_inequality=a['first_inequality'],first_tolerance_failure=a['first_tolerance_failure'],original_gates_changed=False,original_failures_waived=False,
        original_bf16_gpu_rotary_gaps_closed=False,native_engine_executed=False,full_model_window_qualified=False,app_eligibility_changed=False)

if __name__=='__main__':
    observation,reference,inventory_path,model=map(Path,sys.argv[1:])
    body=safe.read(inventory_path,1024**2);safe.require(hashlib.sha256(body).hexdigest()==INVENTORY_SHA,'original inventory identity');inventory=safe.document(body)
    raw=safe.read(observation/'capture.jsonl',2*1024**2);journal=[safe.document(line) for line in raw.splitlines()]
    result=replay(safe.document(safe.read(observation/'observation.json',4*1024**2)),journal,Frozen(reference),inventory,model)
    print(json.dumps(result))
