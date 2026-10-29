"""Independent streamed real LFM2 NativeHalf source; unexecuted private draft.

No native engine/whole-graph reference is imported. Original matrix stripes use
one exact affine center; only qualified standalone RMS/Conv helpers are reused.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import numpy as np
import lfm2_exact_affine as affine
import lfm2_half_rms as rms_leaf
import causal_conv3_half as conv_leaf

ROOT=Path(__file__).resolve().parents[2]
H,I,V,Q,KV,D=1024,4608,65536,16,8,64
ATTENTION=(2,5,8,10,12,14)
MODEL_BYTES=199527820
MODEL_SHA='558353ad098c1230a8e2ea508cd34ef311a1b97a2bc4f47498f4df8229dc5c0f'
HEADER_BYTES=36740
HEADER_SHA='7c5e241d377d693fac694426ed10fb79df470864a9ec31d0ca82f3044171506d'
DATA=8+HEADER_BYTES
PINS={'config.json':'95e6906d77ac647e8bf93c2d2bbd2f37eb9bf3fac8732a563653236ad81072af',
      'model.safetensors.index.json':'9921c33945951c541dfb7de7a11c6ff3b0bb8fb58009be615f851a7de89bcc38'}
FORMAT='toklet-lfm2-real-half-reference-v1'
PROTOCOL_SHA='49005e6e0342adc1e05177063623d149c2319b1966b52d0174ea00d46e6ce59f'
MAX_ARRAYS,MAX_JSON=128*1024**2,16*1024**2
F32=np.float32


def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,
                      parse_constant=lambda x:(_ for _ in()).throw(ValueError(x)))


def read_pinned(path,sha,cap):
    with path.open('rb')as stream:raw=stream.read(cap+1)
    if len(raw)>cap or hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Pinned file differs: '+str(path))
    return raw


def expected_inventory():
    logical={'model.embed_tokens':('affine',(V,H)),
             'model.embedding_norm.weight':('direct',(H,))}
    for layer in range(16):
        base=f'model.layers.{layer}.'
        for name in ('operator_norm','ffn_norm'):
            logical[base+name+'.weight']=('direct',(H,))
        if layer in ATTENTION:
            for name,rows in [('q_proj',Q*D),('k_proj',KV*D),('v_proj',KV*D),('out_proj',H)]:
                logical[base+'self_attn.'+name]=('affine',(rows,H))
            for name in ('q_layernorm','k_layernorm'):
                logical[base+'self_attn.'+name+'.weight']=('direct',(D,))
        else:
            logical[base+'conv.in_proj']=('affine',(3*H,H))
            logical[base+'conv.out_proj']=('affine',(H,H))
            logical[base+'conv.conv.weight']=('direct',(H,3,1))
        for name,shape in [('w1',(I,H)),('w2',(H,I)),('w3',(I,H))]:
            logical[base+'feed_forward.'+name]=('affine',shape)
    physical={}
    for name,(kind,shape)in logical.items():
        if kind=='direct':physical[name]=('BF16',shape)
        else:
            rows,columns=shape
            physical[name+'.weight']=('U32',(rows,columns//8))
            for suffix in ('scales','biases'):physical[name+'.'+suffix]=('BF16',(rows,columns//64))
    return logical,physical


def validate_header(header,index):
    logical,physical=expected_inventory()
    if set(header)!={'__metadata__',*physical}or header['__metadata__']!={'format':'mlx'}:
        raise ValueError('Original exact tensor inventory')
    if index.get('metadata')!={'total_size':199491072,'total_parameters':354483968}:
        raise ValueError('Original index totals')
    if index.get('weight_map')!={name:'model.safetensors'for name in physical}:
        raise ValueError('Original shard map')
    spans=[]
    for name,(dtype,shape)in physical.items():
        item=header[name]
        if set(item)!={'dtype','shape','data_offsets'}or item['dtype']!=dtype or item['shape']!=list(shape):
            raise ValueError('Original tensor dtype/shape '+name)
        offsets=item['data_offsets']
        if not isinstance(offsets,list)or len(offsets)!=2 or any(type(x)is not int for x in offsets):
            raise ValueError('Original tensor offsets')
        a,b=offsets
        if a<0 or b-a!=math.prod(shape)*(4 if dtype=='U32' else 2)or b>MODEL_BYTES-DATA:
            raise ValueError('Original tensor extent')
        spans.append((a,b))
    end=0
    for a,b in sorted(spans):
        if a!=end:raise ValueError('Original payload gap/overlap')
        end=b
    if end!=199491072 or len(physical)!=334 or len(logical)!=148:
        raise ValueError('Original payload completeness')
    return logical


class Source:
    """One full-file readonly mapping; only owning decoded stripes/small vectors."""
    def __init__(self,directory):
        self.file=None;self.mapping=None
        self.directory=Path(directory)
        self.counts=dict(outputs=0,fp64=0,int64=0,zero=0,blocked_outputs=0,certified_blocks=0,scalar_blocks=0,calls=0,max_stripe_fp32_bytes=0)
        config=strict_json(read_pinned(self.directory/'config.json',PINS['config.json'],65536))
        index=strict_json(read_pinned(self.directory/'model.safetensors.index.json',PINS['model.safetensors.index.json'],65536))
        if config['hidden_size']!=H or config['full_attn_idxs']!=list(ATTENTION):raise ValueError('Original config identity')
        path=self.directory/'model.safetensors'
        try:
            self.file=path.open('rb')
            if os.fstat(self.file.fileno()).st_size!=MODEL_BYTES:raise ValueError('Whole file size')
            if hashlib.file_digest(self.file,'sha256').hexdigest()!=MODEL_SHA:raise ValueError('Whole file SHA256')
            self.mapping=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
            if struct.unpack('<Q',self.mapping[:8])[0]!=HEADER_BYTES:raise ValueError('Original header length')
            raw=self.mapping[8:DATA]
            if hashlib.sha256(raw).hexdigest()!=HEADER_SHA:raise ValueError('Original header identity')
            self.header=strict_json(raw);self.logical=validate_header(self.header,index)
        except BaseException:
            self.close();raise

    def close(self):
        if self.mapping is not None:self.mapping.close();self.mapping=None
        if self.file is not None:self.file.close();self.file=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def verify(self):
        # No mapping-backed views escape; authenticate consumed original bytes
        # again before successful export publication.
        self.file.seek(0)
        if os.fstat(self.file.fileno()).st_size!=MODEL_BYTES or hashlib.file_digest(self.file,'sha256').hexdigest()!=MODEL_SHA:
            raise ValueError('Original container changed during source execution')
        for name,sha in PINS.items():read_pinned(self.directory/name,sha,65536)

    def layout(self,name):
        kind,shape=self.logical[name]
        if kind!='affine':raise ValueError('Affine parent name')
        return affine.Layout(*shape,*(DATA+self.header[name+'.'+suffix]['data_offsets'][0]
                                     for suffix in('weight','scales','biases')),MODEL_BYTES).validate()

    def small(self,name):
        kind,shape=self.logical[name]
        if kind!='direct' or math.prod(shape)>3*H:raise ValueError('Small direct owner')
        a,b=self.header[name]['data_offsets']
        raw=self.mapping[DATA+a:DATA+b]  # Copy; no mmap-backed NumPy view escapes.
        value=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view('<f4').reshape(shape)
        if not np.isfinite(value).all():raise ValueError('Nonfinite original direct coefficient')
        return value

    def gather(self,tokens):
        if not isinstance(tokens,list)or not 1<=len(tokens)<=64 or any(type(t)is not int or not 0<=t<V for t in tokens):
            raise ValueError('Bounded raw model token IDs')
        layout=self.layout('model.embed_tokens');output=np.empty((len(tokens),H),dtype='<f2')
        for row,token in enumerate(tokens):
            generator=affine.read_stripes(self.mapping,layout,token,token+1)
            try:
                _,weight=next(generator);output[row]=half_store(weight[0])
            finally:generator.close()
        return output

    def project(self,name,inputs,destination='half'):
        if inputs.dtype!=np.dtype('<f2')or inputs.ndim!=2 or not 1<=inputs.shape[0]<=64:
            raise ValueError('Projection stored inputs')
        if destination not in ('half','f32') or (destination=='f32')!=(name=='model.embed_tokens'):
            raise ValueError('Only original tied head stores FP32')
        if name=='model.embed_tokens' and len(inputs)>4:
            raise ValueError('Source head selected rows only')
        layout=self.layout(name)
        if inputs.shape[1]!=layout.columns:raise ValueError('Projection input columns')
        result=np.empty((len(inputs),layout.rows),dtype='<f2'if destination=='half'else'<f4')
        generator=affine.read_stripes(self.mapping,layout)
        try:
            for row,weights in generator:
                bits,counts=affine.apply(weights,inputs,destination)
                result[:,row:row+len(weights)]=bits.view(result.dtype)
                self.counts['calls']+=1
                self.counts['max_stripe_fp32_bytes']=max(self.counts['max_stripe_fp32_bytes'],counts['stripe_fp32_bytes'])
                for key in('outputs','fp64','int64','zero','blocked_outputs','certified_blocks','scalar_blocks'):
                    self.counts[key]+=counts[key]
                del bits,weights
        finally:generator.close()
        return result


def finite(values):
    if not np.isfinite(values).all():raise ValueError('Nonfinite operand/store')


def half_store(values):
    finite(values)
    with np.errstate(over='ignore',invalid='ignore'):result=np.asarray(values,dtype='<f2')
    finite(result)
    return result


def norm(inputs,gains,profile):
    if profile not in('main','wide')or inputs.dtype!=np.dtype('<f2')or inputs.ndim!=2:
        raise ValueError('RMS profile/input')
    width=inputs.shape[1]
    if width not in(64,H)or gains.dtype!=np.dtype('<f4')or gains.shape!=(width,):raise ValueError('Direct RMS shape')
    output=np.empty_like(inputs)
    limit=512 if width==64 else 32
    for start in range(0,len(inputs),limit):
        part=np.ascontiguousarray(inputs[start:start+limit])
        words=rms_leaf.rms(part.view('<u2').ravel().tolist(),gains.tolist(),len(part),width,width,profile)
        output[start:start+len(part)]=np.array(words,dtype='<u2').view('<f2').reshape(part.shape)
    return output


def product(left,right):
    if left.shape!=right.shape or left.dtype!=np.dtype('<f2')or right.dtype!=left.dtype:raise ValueError('Half product shapes')
    finite(left);finite(right)
    return half_store(np.multiply(left.astype('<f4'),right.astype('<f4'),dtype='<f4'))


def residual(left,right,profile):
    if left.shape!=right.shape or left.dtype!=np.dtype('<f2')or right.dtype!=left.dtype or profile not in('main','wide'):
        raise ValueError('Residual shape/profile')
    finite(left);finite(right)
    kind='<f4'if profile=='main'else'<f8'
    return half_store(np.add(left.astype(kind),right.astype(kind),dtype=kind))


def silu(inputs,profile):
    finite(inputs)
    if inputs.dtype!=np.dtype('<f2')or profile not in('main','wide'):raise ValueError('SiLU profile')
    kind='<f4'if profile=='main'else'<f8';values=inputs.astype(kind)
    exp=np.fromiter((math.exp(-abs(float(x)))for x in values.ravel()),dtype='<f8',count=values.size).reshape(values.shape).astype(kind)
    denominator=np.add(1.,exp,dtype=kind)
    sigmoid=np.where(values>=0,np.divide(1.,denominator,dtype=kind),np.divide(exp,denominator,dtype=kind))
    return half_store(np.multiply(values,sigmoid,dtype=kind))


def frequencies(profile):
    if profile=='main':return np.array([rms_leaf.f32(math.pow(rms_leaf.f32(1e6),rms_leaf.f32(-2*j/D)))for j in range(D//2)],dtype='<f4')
    if profile!='wide':raise ValueError('RoPE profile')
    candidates=[]
    for precision in(80,120):
        with localcontext()as context:
            context.prec=precision;logarithm=Decimal(1000000).ln()
            candidates.append([float((logarithm*Decimal(-2*j)/D).exp())for j in range(D//2)])
    if candidates[0]!=candidates[1]or candidates[0][0]!=1. or candidates[0][16]!=.001:
        raise ValueError('RoPE frequency same-method instability')
    return np.array(candidates[0],dtype='<f8')


def rotary(inputs,heads,profile):
    if inputs.dtype!=np.dtype('<f2')or inputs.ndim!=2 or not 1<=len(inputs)<=64 or inputs.shape[1]!=heads*D or heads not in(Q,KV):
        raise ValueError('RoPE head shape')
    finite(inputs);kind='<f4'if profile=='main'else'<f8'
    values=inputs.reshape(len(inputs),heads,D).astype(kind)
    angle=np.multiply(np.arange(len(inputs),dtype=kind)[:,None],frequencies(profile)[None,:],dtype=kind)
    cosine=np.fromiter((math.cos(float(x))for x in angle.ravel()),dtype='<f8',count=angle.size).reshape(angle.shape).astype(kind)[:,None,:]
    sine=np.fromiter((math.sin(float(x))for x in angle.ravel()),dtype='<f8',count=angle.size).reshape(angle.shape).astype(kind)[:,None,:]
    first,last=values[:,:,:D//2],values[:,:,D//2:]
    left=np.subtract(np.multiply(first,cosine,dtype=kind),np.multiply(last,sine,dtype=kind),dtype=kind)
    right=np.add(np.multiply(last,cosine,dtype=kind),np.multiply(first,sine,dtype=kind),dtype=kind)
    return half_store(np.concatenate((left,right),axis=2).reshape(inputs.shape))


def convolution(inputs,coefficients,profile):
    if inputs.dtype!=np.dtype('<f2')or inputs.ndim!=2 or not 1<=len(inputs)<=64 or inputs.shape[1]!=H or coefficients.dtype!=np.dtype('<f4')or coefficients.shape!=(H,3,1):
        raise ValueError('Raw Conv3 original shape')
    finite(inputs);finite(coefficients)
    n=len(inputs);output=np.empty_like(inputs)
    joined=np.zeros((n+2,H),dtype='<f2');joined[2:]=inputs
    before=np.empty((n,2,H),dtype='<f2');after=np.empty_like(before)
    for row in range(n):before[row]=joined[row:row+2];after[row]=joined[row+1:row+3]
    for start in range(0,n,32):
        part=np.ascontiguousarray(inputs[start:start+32])
        words,candidate=conv_leaf.conv(part.view('<u2').ravel().tolist(),
            joined[start:start+2].view('<u2').ravel().tolist(),coefficients.ravel().tolist(),len(part),H,H,profile)
        output[start:start+len(part)]=np.array(words,dtype='<u2').view('<f2').reshape(part.shape)
        if candidate!=after[start+len(part)-1].view('<u2').ravel().tolist():
            raise ValueError('Raw u history suffix')
    return output,before,after


def attention(query,keys,values,profile):
    n=len(query)
    if profile not in('main','wide')or query.shape!=(n,Q*D)or keys.shape!=(n,KV*D)or values.shape!=keys.shape or not 1<=n<=64:
        raise ValueError('Full short GQA geometry')
    for operand in(query,keys,values):
        if operand.dtype!=np.dtype('<f2'):raise ValueError('Stored GQA operand')
        finite(operand)
    scores=np.empty((Q,n,n),dtype='<f4');probabilities=np.empty_like(scores)
    result=np.empty((n,Q*D),dtype='<f2')
    root=F32(math.sqrt(D))if profile=='main'else math.sqrt(D)
    for head in range(Q):
        q=query[:,head*D:(head+1)*D]
        kv=head//(Q//KV);k=keys[:,kv*D:(kv+1)*D];v=values[:,kv*D:(kv+1)*D]
        if profile=='main':
            q32,k32=q.astype('<f4'),k.astype('<f4');total=np.zeros((n,n),dtype='<f4')
            for feature in range(D):
                product_=np.multiply(q32[:,feature,None],k32[None,:,feature],dtype='<f4')
                total=np.add(total,product_,dtype='<f4')
            scores[head]=np.divide(total,root,dtype='<f4')
        else:
            for token in range(n):
                for position in range(n):
                    scores[head,token,position]=math.fsum(float(q[token,j])*float(k[position,j])for j in range(D))/root
        finite(scores[head])
        for token in range(n):
            row=scores[head,token];maximum=max(float(x)for x in row[:token+1])
            exponentials=np.zeros(n,dtype='<f4'if profile=='main'else'<f8');mass=0.
            for position in range(token+1):
                delta=F32(row[position]-F32(maximum))if profile=='main'else float(row[position])-maximum
                exp=math.exp(float(delta));value=F32(exp)if profile=='main'else exp
                exponentials[position]=value
                mass=F32(F32(mass)+value)if profile=='main'else mass+value
            if not math.isfinite(float(mass))or mass<=0:raise ValueError('Softmax mass')
            probability=np.divide(exponentials,mass).astype('<f4')
            probabilities[head,token]=probability
            if not np.all(probability[token+1:].view('<u4')==0):raise ValueError('Masked positive zero')
            if profile=='main':
                total=np.zeros(D,dtype='<f4')
                for position in range(token+1):
                    product_=np.multiply(probability[position],v[position].astype('<f4'),dtype='<f4')
                    total=np.add(total,product_,dtype='<f4')
            else:
                total=np.array([math.fsum(float(probability[position])*float(v[position,feature])
                                         for position in range(token+1))for feature in range(D)],dtype='<f8')
            result[token,head*D:(head+1)*D]=half_store(total)
    finite(probabilities)
    return scores,probabilities,result


class Capture:
    """Append-only bounded source bytes; WIDE compares exact saved MAIN bytes."""
    def __init__(self,directory,protocol):
        self.directory=Path(directory);self.directory.mkdir(exist_ok=False)
        self.file=(self.directory/'arrays.bin').open('x+b',buffering=0)
        self.progress=(self.directory/'progress.jsonl').open('xb',buffering=0)
        self.offset=0;self.journal_bytes=0;self.arrays={};self.expected={}
        self.profile_counts={'main':dict(arrays=0,values=0,bytes=0),'wide':dict(arrays=0,values=0,bytes=0)}
        for case in protocol['cases']:
            prefix='cases.'+case['name']+'.'
            for row in case['selected_rows']:
                for spec in case['stages']:
                    self.expected[prefix+f'row_{row}.'+spec['name']]=(spec['dtype'],[spec['width']],spec['route'])
            n=len(case['tokens'])
            for layer in range(16):
                layer_prefix=prefix+f'layers.{layer}.'
                if layer in ATTENTION:
                    for name in('kv_keys_producer','kv_values_producer'):
                        self.expected[layer_prefix+name]=('f16',[KV,n,D],'state_producer_sideband')
                else:
                    for name in('history_before','history_after'):
                        self.expected[layer_prefix+name]=('f16',[n,2,H],'state_producer_sideband')
                    self.expected[layer_prefix+'u_producer']=('f16',[n,H],'state_producer_sideband')
        for teacher in protocol['teacher']:
            self.expected[teacher['name']]=('f32',[V],'coupled')
        if len(self.expected)!=3102:raise ValueError('Exact source group inventory')

    def read(self,name,profile):
        group=self.arrays[name];item=group[profile]
        raw=os.pread(self.file.fileno(),item['bytes'],item['offset'])
        if len(raw)!=item['bytes']or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Saved source bytes changed')
        return np.frombuffer(raw,dtype='<f2'if group['dtype']=='f16'else'<f4').reshape(group['shape'])

    def close(self):
        self.progress.close();self.file.close()

    def add(self,name,profile,value):
        if name not in self.expected or profile not in('main','wide'):raise ValueError('Unexpected capture group')
        dtype,shape,route=self.expected[name]
        if value.dtype!=np.dtype('<f2'if dtype=='f16'else'<f4')or list(value.shape)!=shape:raise ValueError('Capture shape/dtype '+name)
        finite(value)
        group=self.arrays.setdefault(name,dict(dtype=dtype,shape=shape,route=route))
        if profile in group:raise ValueError('Duplicate capture '+name)
        raw=np.ascontiguousarray(value).tobytes()
        if self.offset+len(raw)>MAX_ARRAYS:raise ValueError('Binary source cap')
        descriptor=dict(offset=self.offset,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if self.file.write(raw)!=len(raw):raise OSError('Short capture write')
        self.offset+=len(raw);group[profile]=descriptor
        counts=self.profile_counts[profile];counts['arrays']+=1;counts['values']+=value.size;counts['bytes']+=len(raw)
        if profile=='wide':
            if 'main'not in group:raise ValueError('MAIN must precede WIDE')
            first=group['main'];old=os.pread(self.file.fileno(),first['bytes'],first['offset'])
            if len(old)!=first['bytes']or hashlib.sha256(old).hexdigest()!=first['sha256']:raise ValueError('Saved MAIN changed')
            main=np.frombuffer(old,dtype=value.dtype).astype('<f8').reshape(value.shape)
            envelope=float(np.max(np.abs(main-value.astype('<f8')),initial=0.))
            group['envelope']=envelope;group['gate']=dict(atol=max(1e-4,2*envelope),rtol=1e-3)
            group['gate_usage']='diagnostic sideband'if route=='state_producer_sideband'else'fixed MAIN coefficient; route declared prospectively'
        event=(json.dumps(dict(name=name,profile=profile,**descriptor),sort_keys=True,allow_nan=False)+'\n').encode()
        if self.journal_bytes+len(event)>MAX_JSON:raise ValueError('Journal source cap')
        if self.progress.write(event)!=len(event):raise OSError('Short journal write')
        self.journal_bytes+=len(event)

    def selected(self,case,stage,profile,value):
        for row in case['selected_rows']:
            if stage.endswith(('attention_scores','attention_probabilities')):
                selected=np.ascontiguousarray(value[:,row,:]).reshape(-1)
            else:selected=np.ascontiguousarray(value[row]).reshape(-1)
            self.add(f"cases.{case['name']}.row_{row}.{stage}",profile,selected)

    def side(self,case,stage,profile,value):
        self.add(f"cases.{case['name']}.{stage}",profile,value)

    def complete(self,protocol):
        if set(self.arrays)!=set(self.expected)or self.offset!=protocol['counts']['arrays_bytes_both_profiles']:
            raise ValueError('Source full capture extent')
        for name,group in self.arrays.items():
            if not all(key in group for key in('main','wide','gate','envelope')):raise ValueError('Missing paired group '+name)
        for profile in('main','wide'):
            if self.profile_counts[profile]!=dict(arrays=3102,values=11036032,bytes=24024576):
                raise ValueError('Source profile counts')


def forward(source,tokens,profile,capture=None,case=None,teacher_name=None):
    """Full16 layers; selected diagnostics emitted before scratch references die."""
    if profile not in('main','wide')or (case is None)==(teacher_name is None):
        raise ValueError('One source case or forced teacher')
    def save(name,value):
        finite(value)
        if capture is not None and case is not None:capture.selected(case,name,profile,value)
        return value
    hidden=save('embedding_lookup',source.gather(tokens))
    for layer in range(16):
        base=f'model.layers.{layer}.';stage=f'layers.{layer}.'
        normalized=save(stage+'operator_norm',norm(hidden,source.small(base+'operator_norm.weight'),profile))
        if layer not in ATTENTION:
            bcx=save(stage+'bcx_projection',source.project(base+'conv.in_proj',normalized))
            b=save(stage+'b',bcx[:,:H]);c=save(stage+'c',bcx[:,H:2*H]);x=save(stage+'x',bcx[:,2*H:])
            u=save(stage+'u',product(b,x))
            if capture is not None and case is not None:capture.side(case,stage+'u_producer',profile,u)
            z,before,after=convolution(u,source.small(base+'conv.conv.weight'),profile)
            if capture is not None and case is not None:
                capture.side(case,stage+'history_before',profile,before)
                capture.side(case,stage+'history_after',profile,after)
            save(stage+'convolution',z)
            gated=save(stage+'gated_conv',product(c,z))
            projection=save(stage+'operator_projection',source.project(base+'conv.out_proj',gated))
            del bcx,b,c,x,u,z,before,after,gated
        else:
            q=save(stage+'q_projection',source.project(base+'self_attn.q_proj',normalized))
            k=save(stage+'k_projection',source.project(base+'self_attn.k_proj',normalized))
            v=save(stage+'v_projection',source.project(base+'self_attn.v_proj',normalized))
            q=save(stage+'q_norm',norm(q.reshape(-1,D),source.small(base+'self_attn.q_layernorm.weight'),profile).reshape(len(tokens),Q*D))
            k=save(stage+'k_norm',norm(k.reshape(-1,D),source.small(base+'self_attn.k_layernorm.weight'),profile).reshape(len(tokens),KV*D))
            q=save(stage+'q_rotary',rotary(q,Q,profile));k=save(stage+'k_rotary',rotary(k,KV,profile))
            if capture is not None and case is not None:
                capture.side(case,stage+'kv_keys_producer',profile,np.ascontiguousarray(k.reshape(-1,KV,D).transpose(1,0,2)))
                capture.side(case,stage+'kv_values_producer',profile,np.ascontiguousarray(v.reshape(-1,KV,D).transpose(1,0,2)))
            scores,probabilities,context=attention(q,k,v,profile)
            save(stage+'attention_scores',scores);save(stage+'attention_probabilities',probabilities)
            save(stage+'attention_context',context)
            projection=save(stage+'operator_projection',source.project(base+'self_attn.out_proj',context))
            del q,k,v,scores,probabilities,context
        hidden=save(stage+'operator_residual',residual(hidden,projection,profile))
        del projection,normalized
        normalized=save(stage+'ffn_norm',norm(hidden,source.small(base+'ffn_norm.weight'),profile))
        gate=save(stage+'gate_projection',source.project(base+'feed_forward.w1',normalized))
        up=save(stage+'up_projection',source.project(base+'feed_forward.w3',normalized))
        active=save(stage+'gate_silu',silu(gate,profile))
        gated=save(stage+'ffn_gated',product(active,up))
        down=save(stage+'down_projection',source.project(base+'feed_forward.w2',gated))
        hidden=save(stage+'output',residual(hidden,down,profile))
        del normalized,gate,up,active,gated,down
    final=save('final_norm',norm(hidden,source.small('model.embedding_norm.weight'),profile))
    rows=case['selected_rows']if case is not None else[len(tokens)-1]
    logits=source.project('model.embed_tokens',np.ascontiguousarray(final[rows]),'f32')
    if capture is not None:
        if case is not None:
            for position,row in enumerate(rows):capture.add(f"cases.{case['name']}.row_{row}.logits",profile,logits[position])
        else:capture.add(teacher_name,profile,logits[0])
    return logits.copy()


def stage_inventory(tokens):
    """Independent semantic inventory; no target strings/runtime imported."""
    result=[dict(name='embedding_lookup',dtype='f16',width=H,route='actual_input')]
    for layer in range(16):
        stages=[('operator_norm',H)]
        if layer in ATTENTION:
            stages += [('q_projection',Q*D),('k_projection',KV*D),('v_projection',KV*D),
                       ('q_norm',Q*D),('k_norm',KV*D),('q_rotary',Q*D),('k_rotary',KV*D),
                       ('attention_scores',Q*tokens),('attention_probabilities',Q*tokens),('attention_context',Q*D)]
        else:
            stages += [('bcx_projection',3*H),('b',H),('c',H),('x',H),('u',H),('convolution',H),('gated_conv',H)]
        stages += [('operator_projection',H),('operator_residual',H),('ffn_norm',H),
                   ('gate_projection',I),('up_projection',I),('gate_silu',I),('ffn_gated',I),('down_projection',H),('output',H)]
        for name,width in stages:
            result.append(dict(name=f'layers.{layer}.{name}',dtype='f32'if name in('attention_scores','attention_probabilities')else'f16',
                               width=width,route='coupled'if name in('attention_scores','attention_context')else'actual_input'))
    result += [dict(name='final_norm',dtype='f16',width=H,route='actual_input'),
               dict(name='logits',dtype='f32',width=V,route='coupled')]
    if len(result)!=293:raise ValueError('Semantic stage count')
    return result


def load_protocol(path):
    raw=read_pinned(Path(path),PROTOCOL_SHA,MAX_JSON);protocol=strict_json(raw)
    for case in protocol['cases']:
        if case['stages']!=stage_inventory(len(case['tokens'])):raise ValueError('Protocol stage inventory')
    if len(protocol['cases'])!=4 or len(protocol['teacher'])!=4:raise ValueError('Bounded protocol cases')
    return protocol


def words_equal(left,right):
    if left.shape!=right.shape or left.dtype!=right.dtype:return False
    dtype='<u2'if left.dtype==np.dtype('<f2')else'<u4'
    return np.array_equal(left.view(dtype),right.view(dtype))


def saved_invariants(capture,protocol):
    """Read only retained bytes; never another forward or synthetic native state."""
    cases={case['name']:case for case in protocol['cases']}
    small=cases['plain_ids8'];large=cases['future_tail11'];prefix=len(small['tokens'])
    if small['tokens']!=large['tokens'][:prefix]:raise ValueError('Frozen causal prefix')
    result=dict(causal_groups=0,causal_values=0,sideband_prefix_groups=0,
                source_history_words=0,masked_positive_zero_values=0,teacher_prefix_values=0)
    for profile in('main','wide'):
        for row in small['selected_rows']:
            for stage in stage_inventory(prefix):
                name=stage['name'];left=capture.read(f"cases.{small['name']}.row_{row}.{name}",profile)
                right=capture.read(f"cases.{large['name']}.row_{row}.{name}",profile)
                if name.endswith(('attention_scores','attention_probabilities')):
                    left=left.reshape(Q,prefix);right=right.reshape(Q,len(large['tokens']))[:,:prefix]
                if not words_equal(left,right):raise ValueError('Source causal prefix '+name)
                result['causal_groups']+=1;result['causal_values']+=left.size
        for layer in range(16):
            suffixes=('kv_keys_producer','kv_values_producer')if layer in ATTENTION else('u_producer','history_before','history_after')
            for suffix in suffixes:
                left=capture.read(f"cases.{small['name']}.layers.{layer}.{suffix}",profile)
                right=capture.read(f"cases.{large['name']}.layers.{layer}.{suffix}",profile)
                right=right[:,:prefix,:]if layer in ATTENTION else right[:prefix]
                if not words_equal(left,right):raise ValueError('Source sideband prefix')
                result['sideband_prefix_groups']+=1
        for case in protocol['cases']:
            n=len(case['tokens'])
            for layer in range(16):
                prefix_name=f"cases.{case['name']}.layers.{layer}."
                if layer not in ATTENTION:
                    u=capture.read(prefix_name+'u_producer',profile)
                    before=capture.read(prefix_name+'history_before',profile)
                    after=capture.read(prefix_name+'history_after',profile)
                    joined=np.zeros((n+2,H),dtype='<f2');joined[2:]=u
                    for row in range(n):
                        if not words_equal(before[row],joined[row:row+2])or not words_equal(after[row],joined[row+1:row+3]):
                            raise ValueError('Actual source raw-u suffix history')
                    result['source_history_words']+=before.size+after.size
                    del joined,u,before,after
                else:
                    for row in case['selected_rows']:
                        probability=capture.read(f"cases.{case['name']}.row_{row}.layers.{layer}.attention_probabilities",profile).reshape(Q,n)
                        masked=probability[:,row+1:]
                        if not np.all(masked.view('<u4')==0):raise ValueError('Saved masked probability +0')
                        if not np.all((probability>=0)&(probability<=1)):raise ValueError('Saved probability domain')
                        result['masked_positive_zero_values']+=masked.size
        source=capture.read('cases.plain_ids8.row_7.logits',profile)
        teacher=capture.read('teacher.prefix_0.logits',profile)
        if not words_equal(source,teacher):raise ValueError('Fixed teacher identical prefix')
        result['teacher_prefix_values']+=V
    return result


def component_controls():
    """Small prospective witnesses; never derive expected words from target data."""
    gain=np.zeros(H,dtype='<f4');gain[0]=2**-25
    inp=np.zeros((1,H),dtype='<f2');inp[0,0]=1
    norm_words={}
    for profile in('main','wide'):
        norm_words[profile]=int(norm(inp,gain,profile).view('<u2')[0,0])
    if norm_words!={'main':16,'wide':16}:raise ValueError('Direct tiny-gain RMS store')
    # One real channel repeated, original BCx stores remain separate.
    b=np.full((1,H),0x3c01,dtype='<u2').view('<f2')
    u=product(b,b);coeff=np.zeros((H,3,1),dtype='<f4');coeff[:,2,0]=1.25
    z,before,after=convolution(u,coeff,'main');cz=product(b,z)
    observed=[int(x.view('<u2')[0,0])for x in(u,z,cz)]
    if observed!=[0x3c02,0x3d02,0x3d03]:raise ValueError('BCx/Conv/Cz store witness')
    if not words_equal(after[0,1],u[0])or not np.all(before.view('<u2')==0):raise ValueError('Raw source history witness')
    q=np.zeros((1,Q*D),dtype='<f2');k=np.zeros((1,KV*D),dtype='<f2')
    v=np.empty_like(k)
    for head in range(KV):v[:,head*D:(head+1)*D]=head+1
    for profile in('main','wide'):
        scores,probabilities,context=attention(q,k,v,profile)
        if not np.all(scores==0)or not np.all(probabilities==1):raise ValueError('One-key attention')
        for head in range(Q):
            if not np.all(context[:,head*D:(head+1)*D]==head//2+1):raise ValueError('GQA grouped owner')
    return dict(direct_gain_norm_first_words=norm_words,bcx_conv_cz_words=observed,
                one_key_gqa_heads=Q,one_key_kv_owners=KV,
                projection_contract='qualified lfm2_exact_affine; no uncertified FP64 rounding',
                frequency_claim='Decimal80/120 same-method binary64 stability, not certified transcendental rounding')


def file_pin(path):
    path=Path(path)
    with path.open('rb')as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(bytes=path.stat().st_size,sha256=sha)


def write_json(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if len(raw)>MAX_JSON:raise ValueError('Metadata source cap')
    with Path(path).open('xb')as stream:
        if stream.write(raw)!=len(raw):raise OSError('Short metadata write')


def export(directory,model,protocol_path):
    protocol=load_protocol(protocol_path)
    dependencies={str(Path(module.__file__).resolve()):file_pin(module.__file__)
                  for module in(affine,rms_leaf,conv_leaf,rms_leaf.helper)}
    dependencies[str(Path(__file__).resolve())]=file_pin(__file__)
    dependencies[str(Path(protocol_path).resolve())]=file_pin(protocol_path)
    capture=None;last=dict(profile=None,case=None)
    try:
        capture=Capture(directory,protocol)
        components=component_controls()
        counters={}
        with Source(model)as source:
            for profile in('main','wide'):
                before=dict(source.counts)
                for case in protocol['cases']:
                    last=dict(profile=profile,case=case['name'])
                    forward(source,case['tokens'],profile,capture,case=case)
                for teacher in protocol['teacher']:
                    last=dict(profile=profile,case=teacher['name'])
                    forward(source,teacher['tokens'],profile,capture,teacher_name=teacher['name'])
                counters[profile]={key:source.counts[key]-before[key]for key in source.counts if key!='max_stripe_fp32_bytes'}
                counters[profile]['max_stripe_fp32_bytes']=source.counts['max_stripe_fp32_bytes']
            source.verify()
        capture.complete(protocol)
        components['saved_invariants']=saved_invariants(capture,protocol)
        components['affine_counters']=counters
        for path,pin in dependencies.items():
            if file_pin(path)!=pin:raise ValueError('Source dependency changed')
        capture.close()
        write_json(Path(directory)/'components.json',components)
        manifest=dict(format=FORMAT,protocol_sha256=PROTOCOL_SHA,protocol=protocol,
            artifact=dict(protocol['artifact'],header_bytes=HEADER_BYTES,header_sha256=HEADER_SHA,
                          data_offset=DATA,mapped_whole_file_bytes=MODEL_BYTES,physical_tensor_count=334,logical_owner_count=148),
            profiles=dict(main='exact original-F32 affine/direct RNE; qualified FP32 stores and serial Reference attention',
                          wide='same exact affine on own inputs; qualified wider standalone stores; diagnostic only'),
            counts=capture.profile_counts,arrays=capture.arrays,dependencies=dependencies,
            execution=dict(source_graph_executed=True,target_gpu_executed=False,template_executed=False,codec_executed=False),
            files={name:file_pin(Path(directory)/name)for name in('arrays.bin','progress.jsonl','components.json')},
            gate=protocol['gate'],policy=protocol['policy'],
            ownership=dict(original_mapping=MODEL_BYTES,numeric_reserve=128*1024**2,
                exact_affine_included_reserve=32*1024**2,head_row_maximum=4,
                persistent_expanded_model=False,process_cap=6*1024**3,
                label='accounted reserve, not measured process peak; runtime owner records actual observations'),
            scope='independent64-token source trajectory; capacity512 and full128000 runtime are not source-qualified')
        write_json(Path(directory)/'manifest.json',manifest)
        return manifest
    except BaseException as error:
        if capture is not None:
            capture.close()
            failure=dict(format=FORMAT,passed=False,error=repr(error),last=last,
                         arrays_written=len(capture.arrays),bytes_written=capture.offset,
                         source_graph_attempted=last['case']is not None,target_gpu_executed=False)
            failure_path=Path(directory)/'failure.json'
            if not failure_path.exists():write_json(failure_path,failure)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args()
    export(args.output,args.model,args.protocol)
