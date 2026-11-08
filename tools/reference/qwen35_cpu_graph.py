"""Independent packed Qwen3.5 hybrid fixture, without Toklet/MLX imports.

The mathematical and explicit FP32-store profiles are frozen before the Rust
composition. This is not emulation of a native BF16 GPU reduction graph.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

P = 'language_model.model.'
GEOMETRY = dict(hidden=64, intermediate=128, vocab=32, layers=4,
                query_heads=4, kv_heads=2, head_dim=32, rotated_dims=8,
                key_heads=2, value_heads=4, key_dim=16, value_dim=16,
                max_positions=4096, rope_theta=10000000.0, rms_epsilon=1e-6)
TOKENS = (1,7,13,3,9,5,11,2,17,4,6,10,19,8,21,23,29)
CASES = [('one',(1,),0),('context17',TOKENS,0),('offset17',TOKENS,4079),
         ('future_changed17',TOKENS[:-1]+(31,),0)]
GATE = dict(atol=1e-5, rtol=1e-4)


def f32(x):
    value = struct.unpack('<f',struct.pack('<f',x))[0]
    if not math.isfinite(value):raise ValueError('Nonfinite FP32 store')
    return value


def bf16(x):
    word = struct.unpack('<I',struct.pack('<f',x))[0]
    bits = (word + 0x7fff + ((word>>16)&1))>>16
    return bits, struct.unpack('<f',struct.pack('<I',bits<<16))[0]


def specs(geometry=None):
    g=GEOMETRY if geometry is None else geometry
    h=g['hidden'];i=g['intermediate'];q=g['query_heads']*g['head_dim'];k=g['kv_heads']*g['head_dim']
    dk=g['key_heads']*g['key_dim'];dv=g['value_heads']*g['value_dim']
    out={P+'embed_tokens.weight':('matrix',[g['vocab'],h]),P+'norm.weight':('bf16',[h])}
    for layer in range(g['layers']):
        b=P+f'layers.{layer}.'
        for name in ['input_layernorm','post_attention_layernorm']:out[b+name+'.weight']=('bf16',[h])
        for name,shape in [('gate_proj',[i,h]),('up_proj',[i,h]),('down_proj',[h,i])]:out[b+'mlp.'+name+'.weight']=('matrix',shape)
        if layer%4==3:
            for name,shape in [('q_proj',[2*q,h]),('k_proj',[k,h]),('v_proj',[k,h]),('o_proj',[h,q])]:out[b+'self_attn.'+name+'.weight']=('matrix',shape)
            for name in ['q_norm','k_norm']:out[b+'self_attn.'+name+'.weight']=('bf16',[g['head_dim']])
        else:
            for name,shape in [('in_proj_a',[g['value_heads'],h]),('in_proj_b',[g['value_heads'],h]),('in_proj_qkv',[2*dk+dv,h]),('in_proj_z',[dv,h]),('out_proj',[h,dv])]:out[b+'linear_attn.'+name+'.weight']=('matrix',shape)
            for name in ['A_log','dt_bias']:out[b+'linear_attn.'+name]=('bf16',[g['value_heads']])
            out[b+'linear_attn.norm.weight']=('bf16',[g['value_dim']])
            out[b+'linear_attn.conv1d.weight']=('bf16',[2*dk+dv,4,1])
    return out


def artifact():
    raw=bytearray();manifest={};matrices={};vectors={};backing=0
    def save(data):
        offset=len(raw);raw.extend(data)
        return dict(offset=offset,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    for role,(name,(kind,shape)) in enumerate(sorted(specs().items())):
        if kind=='matrix':
            rows,cols=shape;codes=bytearray();scales=bytearray();biases=bytearray();dense=[]
            for row in range(rows):
                values=[]
                for group in range(cols//64):
                    scale=[0.015625,-0.0078125,0.00390625][(row+group+role)%3]
                    bias=-7.0*scale + ((row+role)%3-1)*0.00390625
                    sb,scale=bf16(scale);bb,bias=bf16(bias)
                    scales.extend(struct.pack('<H',sb));biases.extend(struct.pack('<H',bb))
                    nibbles=[(row*11+col*7+(row^(col>>2))*3+role*5)%16 for col in range(group*64,(group+1)*64)]
                    codes.extend(bytes(nibbles[j]|(nibbles[j+1]<<4) for j in range(0,64,2)))
                    values.extend(f32(scale*n+bias) for n in nibbles)
                dense.append(values)
            manifest[name]=dict(kind=kind,shape=shape,codes=save(codes),scales=save(scales),biases=save(biases));matrices[name]=dense
            backing+=len(codes)+len(scales)+len(biases)
        else:
            count=math.prod(shape)
            if name.endswith('A_log'):values=[-1.,-.5,0.,.5]
            elif name.endswith('dt_bias'):values=[.125,-.25,.375,-.5]
            elif 'conv1d' in name:values=[[-.125,.25,-.5,1.][j%4]+((j//4+role)%3-1)*.015625 for j in range(count)]
            else:values=[1.+((j+role)%5-2)*.03125 for j in range(count)]
            encoded=[bf16(v) for v in values];data=b''.join(struct.pack('<H',b) for b,_ in encoded)
            manifest[name]=dict(kind=kind,shape=shape,data=save(data));vectors[name]=[v for _,v in encoded];backing+=4*len(encoded)
    return bytes(raw),manifest,matrices,vectors,backing


class Math:
    def __init__(self,staged,geometry=None):
        self.staged=staged;self.geometry=GEOMETRY if geometry is None else geometry
    def r(self,x):return f32(x) if self.staged else x
    def dot(self,a,b):
        if not self.staged:return math.fsum(x*y for x,y in zip(a,b,strict=True))
        acc=0.
        for x,y in zip(a,b,strict=True):acc=f32(acc+f32(x*y))
        return acc
    def linear(self,x,m):return [self.r(math.fsum(a*b for a,b in zip(x,row,strict=True))) for row in m]
    def norm(self,x,gamma,width):
        result=[]
        for start in range(0,len(x),width):
            row=x[start:start+width];denom=math.sqrt(math.fsum(v*v for v in row)/width+f32(self.geometry['rms_epsilon']))
            result.extend(self.r(v/denom*w) for v,w in zip(row,gamma,strict=True))
        return result
    def sigmoid(self,x):
        ex=math.exp(-abs(x));return self.r(1./(1.+ex) if x>=0 else ex/(1.+ex))
    def silu(self,x):
        if not self.staged:return x*self.sigmoid(x)
        ex=f32(math.exp(-abs(x)));den=f32(1.+ex);sig=f32(1./den) if x>=0 else f32(ex/den)
        return f32(x*sig)
    def rotary(self,x,heads,dim,rot,position):
        out=list(x)
        for head in range(heads):
            for j in range(rot//2):
                angle=position*self.geometry['rope_theta']**(-2.*j/rot);co=self.r(math.cos(angle));si=self.r(math.sin(angle))
                a=head*dim+j;b=a+rot//2
                out[a]=self.r(self.r(x[a]*co)-self.r(x[b]*si));out[b]=self.r(self.r(x[b]*co)+self.r(x[a]*si))
        return out


def forward(matrices,vectors,tokens,offset,staged,*,geometry=None,math_type=Math,sink=None):
    g=GEOMETRY if geometry is None else geometry
    m=math_type(staged,g);trace=[];states={};histories={};keys={};values={}
    dk=g['key_heads']*g['key_dim'];dv=g['value_heads']*g['value_dim'];channels=2*dk+dv
    for layer in range(g['layers']):
        if layer%4!=3:states[layer]=[0.]*(g['value_heads']*g['value_dim']*g['key_dim']);histories[layer]=[[0.]*channels for _ in range(3)]
        else:keys[layer]=[];values[layer]=[]
    def save(t,layer,stage,x):
        if any(not math.isfinite(v) for v in x):raise ValueError('Nonfinite reference')
        row=dict(token=t,layer=layer,stage=stage,values=list(x))
        if sink is None:trace.append(row)
        else:sink(row)
    for t,token in enumerate(tokens):
        x=list(matrices[P+'embed_tokens.weight'][token]);save(t,None,'embedding',x)
        for layer in range(g['layers']):
            b=P+f'layers.{layer}.';u=m.norm(x,vectors[b+'input_layernorm.weight'],g['hidden']);save(t,layer,'input_norm',u)
            def linear(role,v):return m.linear(v,matrices[b+role+'.weight'])
            if layer%4!=3:
                p='linear_attn.';qkv=linear(p+'in_proj_qkv',u);save(t,layer,'delta_qkv',qkv)
                z=linear(p+'in_proj_z',u);a=linear(p+'in_proj_a',u);beta_raw=linear(p+'in_proj_b',u)
                save(t,layer,'delta_z',z);save(t,layer,'delta_a',a);save(t,layer,'delta_b',beta_raw)
                joined=histories[layer]+[qkv];coeff=vectors[b+p+'conv1d.weight']
                conv=[m.r(math.fsum(joined[tap][c]*coeff[4*c+tap] for tap in range(4))) for c in range(channels)]
                conv=[m.silu(v) for v in conv];histories[layer]=joined[1:];save(t,layer,'delta_convolved',conv)
                q,k,v=conv[:dk],conv[dk:2*dk],conv[2*dk:];qn=[];kn=[]
                for h in range(g['key_heads']):
                    qr=q[h*g['key_dim']:(h+1)*g['key_dim']];kr=k[h*g['key_dim']:(h+1)*g['key_dim']]
                    qden=math.sqrt(math.fsum(a*a for a in qr)+1e-6);kden=math.sqrt(math.fsum(a*a for a in kr)+1e-6)
                    qn.extend(m.r(a/qden/math.sqrt(g['key_dim'])) for a in qr);kn.extend(m.r(a/kden) for a in kr)
                save(t,layer,'delta_q',qn);save(t,layer,'delta_k',kn);save(t,layer,'delta_v',v)
                alog=vectors[b+p+'A_log'];dt=vectors[b+p+'dt_bias'];decay=[];beta=[]
                for h in range(g['value_heads']):
                    az=a[h]+dt[h];soft=max(az,0.)+math.log1p(math.exp(-abs(az)))
                    decay.append(m.r(math.exp(-math.exp(alog[h])*soft)));beta.append(m.sigmoid(beta_raw[h]))
                save(t,layer,'delta_decay',decay);save(t,layer,'delta_beta',beta)
                old=states[layer];new=[0.]*len(old);out=[]
                for h in range(g['value_heads']):
                    kh=h//(g['value_heads']//g['key_heads']);key=kn[kh*g['key_dim']:(kh+1)*g['key_dim']];query=qn[kh*g['key_dim']:(kh+1)*g['key_dim']]
                    for d in range(g['value_dim']):
                        base=(h*g['value_dim']+d)*g['key_dim'];decayed=[m.r(old[base+j]*decay[h]) for j in range(g['key_dim'])]
                        correction=m.r(m.r(v[h*g['value_dim']+d]-m.dot(decayed,key))*beta[h])
                        updated=[m.r(s+m.r(k*correction)) for s,k in zip(decayed,key,strict=True)];new[base:base+g['key_dim']]=updated;out.append(m.dot(updated,query))
                states[layer]=new;save(t,layer,'delta_state',new);save(t,layer,'delta_history',[v for row in histories[layer] for v in row]);save(t,layer,'delta_output',out)
                out=m.norm(out,vectors[b+p+'norm.weight'],g['value_dim']);save(t,layer,'delta_norm',out)
                out=[m.r(v*m.silu(gate)) for v,gate in zip(out,z,strict=True)];save(t,layer,'delta_gated',out);op=linear(p+'out_proj',out)
            else:
                p='self_attn.';dim=g['head_dim'];qh=g['query_heads'];kvh=g['kv_heads'];qgate=linear(p+'q_proj',u);save(t,layer,'attention_q_gate',qgate)
                q=[v for h in range(qh) for v in qgate[2*h*dim:(2*h+1)*dim]];gate=[v for h in range(qh) for v in qgate[(2*h+1)*dim:(2*h+2)*dim]]
                k=linear(p+'k_proj',u);v=linear(p+'v_proj',u);save(t,layer,'attention_q',q);save(t,layer,'attention_k',k);save(t,layer,'attention_v',v)
                q=m.norm(q,vectors[b+p+'q_norm.weight'],dim);k=m.norm(k,vectors[b+p+'k_norm.weight'],dim);save(t,layer,'attention_q_norm',q);save(t,layer,'attention_k_norm',k)
                q=m.rotary(q,qh,dim,g['rotated_dims'],offset+t);k=m.rotary(k,kvh,dim,g['rotated_dims'],offset+t);save(t,layer,'attention_q_rot',q);save(t,layer,'attention_k_rot',k)
                keys[layer].append(k);values[layer].append(v);context=[];scale=m.r(1./math.sqrt(dim))
                for head in range(kvh):
                    save(t,layer,f'attention_k_cache.{head}',[x for row in keys[layer] for x in row[head*dim:(head+1)*dim]])
                    save(t,layer,f'attention_v_cache.{head}',[x for row in values[layer] for x in row[head*dim:(head+1)*dim]])
                for head in range(qh):
                    kh=head//(qh//kvh);query=q[head*dim:(head+1)*dim]
                    scores=[m.r(m.dot(query,key[kh*dim:(kh+1)*dim])*scale) for key in keys[layer]]
                    maximum=max(scores);ex=[math.exp(v-maximum) for v in scores];mass=math.fsum(ex);prob=[m.r(v/mass) for v in ex]
                    save(t,layer,f'attention_scores.{head}',scores);save(t,layer,f'attention_probabilities.{head}',prob)
                    context.extend(m.dot(prob,[row[kh*dim+d] for row in values[layer]]) for d in range(dim))
                save(t,layer,'attention_context',context);out=[m.r(v*m.sigmoid(gate)) for v,gate in zip(context,gate,strict=True)];save(t,layer,'attention_gated',out);op=linear(p+'o_proj',out)
            save(t,layer,'operator_output',op);x=[m.r(a+b) for a,b in zip(x,op,strict=True)];save(t,layer,'operator_residual',x)
            u=m.norm(x,vectors[b+'post_attention_layernorm.weight'],g['hidden']);save(t,layer,'post_norm',u)
            gate=linear('mlp.gate_proj',u);up=linear('mlp.up_proj',u);save(t,layer,'mlp_gate',gate);save(t,layer,'mlp_up',up)
            activated=[m.r(m.silu(a)*b) for a,b in zip(gate,up,strict=True)];save(t,layer,'mlp_activated',activated);out=linear('mlp.down_proj',activated);save(t,layer,'mlp_output',out)
            x=[m.r(a+b) for a,b in zip(x,out,strict=True)];save(t,layer,'layer_output',x)
        x=m.norm(x,vectors[P+'norm.weight'],g['hidden']);save(t,None,'final_norm',x);save(t,None,'logits',m.linear(x,matrices[P+'embed_tokens.weight']))
    return trace


def generate(destination):
    destination.mkdir(exist_ok=False);weights,table,matrices,vectors,backing=artifact();(destination/'weights.bin').write_bytes(weights)
    body=bytearray();cases=[]
    for name,tokens,offset in CASES:
        profiles={}
        for profile,staged in [('ideal_fp64',False),('staged_fp32',True)]:
            arrays=[]
            for row in forward(matrices,vectors,tokens,offset,staged):
                values=row.pop('values');raw=struct.pack('<'+('f' if staged else 'd')*len(values),*values)
                arrays.append(dict(**row,offset=len(body),bytes=len(raw),count=len(values),sha256=hashlib.sha256(raw).hexdigest()));body.extend(raw)
            profiles[profile]=arrays
        cases.append(dict(name=name,tokens=list(tokens),position_offset=offset,profiles=profiles))
    if len(body)>16*1024**2:raise ValueError('Reference capture exceeds bound')
    (destination/'reference.bin').write_bytes(body)
    manifest=dict(format='toklet-qwen35-cpu-graph-v1',geometry=GEOMETRY,gate=GATE,weights=table,backing_bytes=backing,weights_bytes=len(weights),weights_sha256=hashlib.sha256(weights).hexdigest(),reference_bytes=len(body),reference_sha256=hashlib.sha256(body).hexdigest(),cases=cases,fixture_only=True,real_model_qualified=False)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return dict(weights_bytes=len(weights),reference_bytes=len(body),arrays=sum(len(a) for c in cases for a in c['profiles'].values()),manifest_sha256=hashlib.sha256((destination/'manifest.json').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args();print(json.dumps(generate(args.output)))
