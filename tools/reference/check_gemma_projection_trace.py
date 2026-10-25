"""Independent full projection mapping, raw-array and FP32 recipe replay."""
from fractions import Fraction
from pathlib import Path
import hashlib,json,math,struct,sys
import check_gemma_affine_admission as source_check
import check_gemma_affine_l1_math as proof_check

def add32(a,b):
    result=source_check.rounded(Fraction(a)+Fraction(b))
    if a==b==0 and math.copysign(1.,a)<0 and math.copysign(1.,b)<0:return -0.
    return result

def mapping():
    result=[]
    def add(name,rows,columns,kind,sources,output):
        result.append(dict(name=name,rows=rows,columns=columns,kind=kind,sources=sources,output=output))
    add('per_layer_model_projection.weight',8960,1536,'direct',['scaled_embedding'],'ple_projection')
    for layer in range(35):
        stem='layer.'+str(layer)+'.';matrix='layers.'+str(layer)+'.'
        dimension=512 if layer in (4,9,14,19,24,29,34) else 256
        ff=6144 if layer<=14 else 12288
        specifications=[('self_attn.q_proj',8*dimension,1536,'direct',[stem+'input_layernorm'])]
        if layer<=14:
            specifications.extend((suffix,dimension,1536,'direct',[stem+'input_layernorm']) for suffix in ('self_attn.k_proj','self_attn.v_proj'))
        specifications.extend([
            ('self_attn.o_proj',1536,8*dimension,'direct',[stem+'context']),
            ('mlp.gate_proj',ff,1536,'direct',[stem+'pre_feedforward_layernorm']),
            ('mlp.up_proj',ff,1536,'direct',[stem+'pre_feedforward_layernorm']),
            ('mlp.down_proj',1536,ff,'product-envelope',[stem+'mlp.gate_proj',stem+'mlp.up_proj']),
            ('per_layer_projection',1536,256,'product-envelope',[stem+'per_layer_input_gate',stem+'ple_input']),
            ('per_layer_input_gate',256,1536,'exact-residual',[stem+'input',stem+'post_attention_layernorm',stem+'post_feedforward_layernorm'])])
        for suffix,n,k,kind,names in specifications:add(matrix+suffix+'.weight',n,k,kind,names,stem+suffix)
    add('embed_tokens.weight',262144,1536,'direct',['final_norm'],'head')
    if len(result)!=277 or len({row['name'] for row in result})!=277:raise ValueError('complete unique named inventory required')
    return result

def reconstruct(item,reader):
    names=item['sources'];k=item['columns']
    if item['kind']=='exact-residual':
        parts=[reader(name) for name in names]
        if any(shape!=[1,4,k] or len(values)!=4*k for values,shape in parts):raise ValueError('complete residual shapes required')
        values=[add32(add32(a,b),c)
                for a,b,c in zip(*(v for v,_ in parts),strict=True)]
    elif item['kind']=='product-envelope':
        a,shape=reader(names[0]);b,other=reader(names[1])
        if shape!=[1,4,k] or shape!=other or len(a)!=4*k or len(b)!=len(a):raise ValueError('complete paired product shapes required')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in a+b):raise ValueError('finite typed product values required')
        values=[]
        for gate,up in zip(a,b,strict=True):
            cube=gate**3
            argument=math.sqrt(2/math.pi)*(gate+.044715*cube)
            unrounded=.5*gate*(1+math.tanh(argument))
            activation=source_check.rounded(Fraction(unrounded)) if unrounded else math.copysign(0.,unrounded)
            result=source_check.rounded(Fraction(activation)*Fraction(up))
            if activation==0 or up==0:result=math.copysign(0.,math.copysign(1.,activation)*math.copysign(1.,up))
            values.append(result)
    else:
        values,shape=reader(names[0])
        if names[0].endswith('.context'):
            if shape!=[1,8,4,k//8] or len(values)!=4*k:raise ValueError('head-major context shape differs')
            values=[values[(head*4+token)*(k//8)+column] for token in range(4) for head in range(8) for column in range(k//8)]
        elif shape!=[1,4,k] or len(values)!=4*k:raise ValueError('complete direct shape differs')
    if len(values)!=4*k or any(not math.isfinite(v) for v in values):raise ValueError('complete finite inputs required')
    return values

def replay(coordinate,l1_receipt,trace,raw_trace,compact,recipe,directory):
    wanted=['2d082715177522d4ea353d699a8415b792fa52ffa943b9d083d8aee832c7d821',
            '14d479c1f218f3531512568e286f008a869375959cec2c7c31fdeac3e5964d44',
            'ca86bd9836c88d5084f770b9ae84ecdb6e1e5a91d775e7cda315702450e918e9',
            '352385699238b422b430b6f535c2eac99b6e90543becb4c46caa0c59f930ed93',
            'f60c9f355b9008cfba6acd5d39c89c108de2d1ed5a3c8013a3a595576d8f910b',
            '6c04a81b93c95edfad4a4de646da3731b71a6dd175163fb856d2009bd103b096']
    arguments=[coordinate,l1_receipt,trace,raw_trace,compact,recipe]
    for path,digest in zip(arguments,wanted,strict=True):
        if source_check.digest(path)!=digest:raise ValueError('authenticated prerequisite differs')
    coordinate_data=source_check.load(coordinate,2*1024**2);l1_data=source_check.load(l1_receipt,2*1024**2)
    if coordinate_data['passed'] is not True or l1_data['passed'] is not True:raise ValueError('accepted prerequisite required')
    observation=source_check.load(trace,4*1024**2);sample=source_check.load(compact,2*1024**2)
    ledger={(v['contract'],v['name']):v for v in observation['arrays']};expected={(v['contract'],v['name']):v for v in sample['arrays']}
    if len(ledger)!=len(observation['arrays']) or len(ledger)!=len(expected) or len(ledger)!=1426:raise ValueError('complete unique trace required')
    matrices={row['name']:row for row in coordinate_data['matrices']}
    manifest=source_check.load(directory/'manifest.json',65536)
    source_check.same(manifest['format'],'toklet-gemma-projection-manifest-v1');source_check.same(manifest['passed'],True)
    source_check.same(manifest['producer_sha256'],source_check.digest(Path(__file__).with_name('gemma_projection_trace.py')))
    if set(manifest['files'])!={'observation.json','arrays.bin'}:raise ValueError('exact capture files required')
    for name in manifest['files']:
        source_check.same(manifest['files'][name],dict(bytes=(directory/name).stat().st_size,sha256=source_check.digest(directory/name)))
    if manifest['files']['arrays.bin']['bytes']!=26333184:raise ValueError('fixed complete binary size required')
    records=[];offset=0;in_bytes=out_bytes=0;captured=derived=rejected=0;statuses={}
    with raw_trace.open('rb') as source,(directory/'arrays.bin').open('rb') as frozen:
        def read(name):
            key=('staged_fp32',name);entry=ledger[key];spec=expected[key]
            if entry['dtype']!='<f4' or entry['shape']!=spec['source_shape'] or entry['sha256']!=spec['source_array_sha256'] or not 0<entry['bytes']<=196608 or entry['bytes']!=math.prod(entry['shape'])*4:raise ValueError('bounded source state identity differs')
            source.seek(entry['offset']);raw=source.read(entry['bytes'])
            if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('source bytes differ')
            values=[v for (v,) in struct.iter_unpack('<f',raw)]
            if any(not math.isfinite(v) for v in values):raise ValueError('finite raw values required')
            return values,entry['shape']
        for item in mapping():
            name=item['name'];n=item['rows'];k=item['columns'];m=matrices[name]
            if (m['rows'],m['columns'])!=(n,k):raise ValueError('independent matrix shape differs')
            record=dict(name=name,input_kind=item['kind'],sources=[dict(name=s,sha256=ledger[('staged_fp32',s)]['sha256']) for s in item['sources']],
                        output_source=item['output'],rows=n,columns=k,arrays={},completed=True,
                        input_evidence='derived-mathematical-gelu-product' if item['kind']=='product-envelope' else 'captured-state-reconstruction')
            records.append(record)
            if n>16384:
                if name!='embed_tokens.weight' or m['geometry_admitted'] is not False:raise ValueError('full head rejection required')
                record['status']='geometry-rejected';rejected+=1;continue
            if m['geometry_admitted'] is not True:raise ValueError('admitted geometry required')
            values=reconstruct(item,read);output,shape=read(item['output'])
            if shape!=[1,4,n] or len(output)!=4*n:raise ValueError('complete pre-norm output required')
            sums=[proof_check.accumulated(abs(v) for v in values[token*k:(token+1)*k]) for token in range(4)]
            old=source_check.bounds(k,m['maximum_weight'],max(map(abs,values)));new=proof_check.bound(k,m['maximum_weight'],max(sums))
            if not(old['admitted'] or new['admitted']):raise ValueError('actual max-first/L1 proof fails')
            record.update(maximum_weight=m['maximum_weight'],per_token_l1=sums,legacy_max=old,l1=new,status='proved-max' if old['admitted'] else 'proved-l1')
            statuses[record['status']]=statuses.get(record['status'],0)+1
            for role,array,width in [('input',values,k),('expected_staged_fp32',output,n)]:
                body=struct.pack('<'+'f'*len(array),*array)
                if frozen.read(len(body))!=body:raise ValueError('complete frozen array differs')
                record['arrays'][role]=dict(dtype='F32',shape=[4,width],offset=offset,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),completed=True)
                offset+=len(body)
                if role=='input':in_bytes+=len(body)
                else:out_bytes+=len(body)
            if item['kind']=='product-envelope':derived+=1
            else:captured+=1
        if frozen.read(1):raise ValueError('trailing binary bytes')
    if (captured,derived,rejected,in_bytes,out_bytes,offset)!=(206,70,1,11128832,15204352,26333184):raise ValueError('fixed complete inventory differs')
    result=dict(format='toklet-gemma-projection-trace-v1',passed=True,records=records,primary_error=None,
        expected_contract='Acknowledged mathematical graph: rounded staged-FP32 linear output over reconstructed FP32 weights',
        tokens=[2,100,257,100],positions=[0,1,2,3],binary_bytes=offset,input_bytes=in_bytes,output_bytes=out_bytes,
        captured_state_inputs=captured,derived_recipe_inputs=derived,geometry_rejections=rejected,
        native_activation_parity=False,actual_gpu_executed=False,whole_graph_qualified=False,
        source_or_precision_failures_waived=False,app_or_phone_qualified=False,
        prerequisites=[dict(path=str(path),sha256=digest) for path,digest in zip(arguments,wanted,strict=True)])
    source_check.same(source_check.load(directory/'observation.json',2*1024**2),result)
    return dict(passed=True,records=277,captured_state_inputs=206,derived_recipe_inputs=70,arrays=552,binary_bytes=offset,statuses=statuses,full_head_geometry_rejected=True)

if __name__=='__main__':
    if len(sys.argv)!=8:raise SystemExit('coordinate l1-receipt trace-json trace-bin compact-json recipe-source capture-directory')
    print(json.dumps(replay(*(Path(v) for v in sys.argv[1:]))))
