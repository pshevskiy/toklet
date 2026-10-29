"""Fixed selected-stage Gemma3 conformance on authenticated actual operands.

Not a forward graph: all dependencies are observed native rows. No producer
module imports, absent-stage substitution, target-dependent tolerances or KV
reconstruction. Original weights are read one small logical matrix at a time.
"""
from pathlib import Path
import hashlib
import json
import math
import struct
import numpy as np
import gemma3_conformance_math as m

F = np.float32
POLICY_SHA = '415d493f1aac05c2cc387867e89001e0a7dc9e2f41c2b96d5e27af3b01111ba8'
WEIGHT_SHA = 'bf8b442cff5dd7c9ac6b5695310d5b66a4d0a4c3780f5c774fb90cd0c352bd9d'
HEADER_SHA = '994efdf1aa3b4f6bc39d7935b1286380c6f9f49e926e02405fed6ed72f6e578f'
LAYER_STAGES = ('input_norm','q_projection','k_projection','v_projection','q_norm','k_norm','q_rotary','k_rotary','attention_scores','attention_probabilities','attention_context','attention_projection','post_attention_norm','attention_residual','pre_ff_norm','gate_projection','up_projection','gate_gelu','gated','down_projection','post_ff_norm','output')
STAGES = ('embedding_lookup','embedding') + tuple(f'layers.{l}.{s}' for l in range(18) for s in LAYER_STAGES) + ('final_norm','logits')
COUPLED = {'logits','attention_scores','attention_context'}
PROJECT = {'q_projection':('self_attn.q_proj','input_norm',1024,640),
           'k_projection':('self_attn.k_proj','input_norm',256,640),
           'v_projection':('self_attn.v_proj','input_norm',256,640),
           'attention_projection':('self_attn.o_proj','attention_context',640,1024),
           'gate_projection':('mlp.gate_proj','pre_ff_norm',2048,640),
           'up_projection':('mlp.up_proj','pre_ff_norm',2048,640),
           'down_projection':('mlp.down_proj','gated',640,2048)}
NORM = {'input_norm':('input_layernorm.weight',None,640),
        'q_norm':('self_attn.q_norm.weight','q_projection',256),
        'k_norm':('self_attn.k_norm.weight','k_projection',256),
        'post_attention_norm':('post_attention_layernorm.weight','attention_projection',640),
        'pre_ff_norm':('pre_feedforward_layernorm.weight','attention_residual',640),
        'post_ff_norm':('post_feedforward_layernorm.weight','down_projection',640)}


def route(stage):
    m.require(stage in STAGES, 'unknown selected stage')
    return 'coupled' if stage.split('.')[-1] in COUPLED else 'local'


def expected_shape(stage, keys):
    route(stage)
    leaf=stage.split('.')[-1]
    if stage in ('embedding_lookup','embedding','final_norm'):return (1,640)
    if stage=='logits':return (1,262144)
    if leaf in ('attention_scores','attention_probabilities'):
        m.require(1<=keys<=544,'materialized key count')
        return (4,1,keys)
    if leaf in ('q_projection','q_norm','q_rotary'):return (1,4,256)
    if leaf in ('k_projection','v_projection','k_norm','k_rotary'):return (1,1,256)
    return (1,2048 if leaf in ('gate_projection','up_projection','gate_gelu','gated') else 1024 if leaf=='attention_context' else 640)


def numeric_gate(actual, center, gate):
    m.require(actual.shape == center.shape, 'local gate shape')
    m.finite(actual); m.finite(center)
    m.require(set(gate) == {'atol','rtol','fatal_reference'} and gate['fatal_reference']=='main' and math.isfinite(gate['atol']) and gate['atol']>=1e-4 and gate['rtol']==.001, 'literal local gate')
    error = np.abs(actual.astype(np.float64)-center.astype(np.float64))
    bound = gate['atol'] + gate['rtol']*np.abs(center.astype(np.float64))
    return error <= bound, float(error.max(initial=0)), float((error/bound).max(initial=0))


class Weights:
    """Closed by caller; one FP32 logical matrix cached, selected embedding rows."""
    def __init__(self, path, reader):
        self.path = Path(path).resolve()
        reader.pin(self.path,151*1024**2,{'bytes':150939130,'sha256':WEIGHT_SHA})
        self.file = self.path.open('rb'); self.cached = None; self.name = None
        try:
            size = struct.unpack('<Q',self.file.read(8))[0]
            m.require(size == 53490, 'original header length')
            raw = self.file.read(size)
            m.require(hashlib.sha256(raw).hexdigest()==HEADER_SHA, 'original header identity')
            self.header=json.loads(raw); self.base=8+size
        except BaseException:
            self.file.close(); raise
        self.ranges = {}; self.max_numeric_bytes=0
    def close(self):
        self.cached=None; self.file.close()
    def raw(self,name,shape,dtype,row=None):
        d=self.header[name]; m.require(d['shape']==list(shape) and d['dtype']==dtype, 'original tensor shape/dtype')
        size=4 if dtype=='U32' else 2; lo,hi=d['data_offsets']
        m.require(0<=lo<=hi and hi+self.base<=150939130 and hi-lo==math.prod(shape)*size, 'original tensor extent')
        if row is not None:
            m.require(len(shape)==2 and type(row)is int and 0<=row<shape[0], 'original row index')
            width=shape[1]; lo+=row*width*size; hi=lo+width*size; shape=(1,width)
        m.require(hi-lo<=2*1024**2, 'bounded packed read')
        self.file.seek(self.base+lo); raw=self.file.read(hi-lo)
        m.require(len(raw)==hi-lo, 'truncated original payload')
        self.ranges[name+':'+str(row)]={'offset':self.base+lo,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        return np.frombuffer(raw,dtype='<u4'if dtype=='U32'else'<u2').copy().reshape(shape)
    @staticmethod
    def bf(words, metadata=False):
        values=(words.astype(np.uint32)<<16).view(F); m.finite(values)
        # Individual zero coefficients are allowed, subnormal metadata are not.
        if metadata:m.require(np.all((((words>>7)&255)!=0)|((words&0x7fff)==0)), 'subnormal coefficient metadata')
        return values
    def norm(self,name,width):
        return F(F(1)+self.bf(self.raw(name,(width,),'BF16')).reshape(-1))
    def decode(self,name,out,width,row=None):
        m.require(width%64==0 and width<=2048 and (row is not None or out<=2048), 'matrix decode bound')
        codes=self.raw(name+'.weight',(out,width//8),'U32',row)
        scales=self.bf(self.raw(name+'.scales',(out,width//64),'BF16',row),True)
        biases=self.bf(self.raw(name+'.biases',(out,width//64),'BF16',row),True)
        # Independent original nibble layout. A FINITE Q4xBF16 product fits
        # FP32 exactly. Requiring that intermediate makes separate addition
        # equivalent to original FMA reconstruction; exotic overflow/cancel
        # cases are unsupported rather than incorrectly reconstructed.
        result=np.empty((codes.shape[0],width),dtype=F)
        for shift in range(8):
            q=((codes>>(shift*4))&15).astype(F)
            with np.errstate(over='ignore',invalid='ignore'):
                product=F(q*np.repeat(scales,8,axis=1))
                m.finite(product)
                result[:,shift::8]=F(product+np.repeat(biases,8,axis=1))
        m.finite(result); m.require(np.all((result==0)|(np.abs(result)>=F(2**-79))), 'MLX reconstructed weight domain')
        self.max_numeric_bytes=max(self.max_numeric_bytes,result.nbytes+codes.nbytes+scales.nbytes+biases.nbytes)
        return result
    def matrix(self,name,out,width):
        if self.name!=name:
            self.cached=None
            self.cached=self.decode(name,out,width); self.name=name
        m.require(self.cached.shape==(out,width), 'cached matrix shape')
        return self.cached
    def embedding(self,token):
        return m.half(self.decode('model.embed_tokens',262144,640,token)).reshape(1,640)


class Validator:
    def __init__(self, weights, progress):
        self.weights=weights; self.progress=progress; self.saved={}; self.context=None
        self.groups=0; self.values=0; self.semantic_failures=0; self.numeric_failures=0
        self.max_abs=0.; self.max_fraction=0.; self.fallbacks=0; self.first_failure=None
        self.expected=set(); self.seen=set()
    def begin(self,label,tokens,selected):
        m.require(self.context is None, 'unclosed local schedule')
        m.require(len(tokens)<=544 and len(selected)<=4 and len(set(selected))==len(selected) and all(0<=r<len(tokens)for r in selected), 'local schedule bound')
        self.context=(label,tuple(tokens)); self.saved={}; self.seen=set()
        self.expected={(row,stage)for row in selected for stage in STAGES}
    def end(self):
        m.require(self.seen==self.expected, 'missing selected local stage/operand')
        self.context=None; self.saved={}
    def get(self,row,stage):
        m.require((row,stage)in self.saved, 'missing same-schedule actual operand: '+stage)
        return self.saved[row,stage][0]
    def observe(self,label,row,stage,actual,record,gate):
        m.require(self.context is not None and label==self.context[0], 'wrong local schedule/case')
        m.require((row,stage)in self.expected and (row,stage)not in self.seen, 'unknown/duplicate row/stage')
        m.require(record['label']==label and record['row']==row and record['stage']==stage and list(actual.shape)==record['shape'], 'local descriptor identity/shape')
        keys=actual.shape[-1]if stage.split('.')[-1]in ('attention_scores','attention_probabilities')else 0
        m.require(actual.shape==expected_shape(stage,keys),'fixed local stage shape')
        if not keys:m.require(record['absolute_key_start']==0,'non-attention key origin')
        self.seen.add((row,stage)); m.finite(actual)
        if route(stage)=='coupled':
            # Keep only actual context/scores needed by next operators, not head.
            if stage!='logits':self.saved[row,stage]=(actual.copy(),dict(record))
            return None
        original_shape=actual.shape; leaf=stage.split('.')[-1]
        if leaf!='attention_probabilities':m.half_input(actual)
        own=lambda name:self.get(row,name)
        center=None; semantic=None
        if stage=='embedding_lookup':
            center=self.weights.embedding(self.context[1][row])
        elif stage=='embedding':center=m.half(F(own('embedding_lookup')*F(25.25)))
        elif stage=='final_norm':center=m.rms(own('layers.17.output'),self.weights.norm('model.norm.weight',640))
        else:
            layer=int(stage.split('.')[1]); prefix=f'layers.{layer}.'; model=f'model.layers.{layer}.'
            previous='embedding'if layer==0 else f'layers.{layer-1}.output'
            if leaf in NORM:
                field,inp,width=NORM[leaf]
                center=m.rms(own(previous if inp is None else prefix+inp),self.weights.norm(model+field,width))
            elif leaf in PROJECT:
                field,inp,out,width=PROJECT[leaf]
                center,semantic,count=m.projection(own(prefix+inp),self.weights.matrix(model+field,out,width),actual)
                center=center.reshape(original_shape); semantic=semantic.reshape(original_shape); self.fallbacks+=count
            elif leaf in ('q_rotary','k_rotary'):
                center,semantic=m.rotary(own(prefix+leaf[0]+'_norm'),actual,row,1000000 if layer%6==5 else 10000)
            elif leaf=='attention_probabilities':
                scores=own(prefix+'attention_scores'); d=self.saved[row,prefix+'attention_scores'][1]
                m.require(record['absolute_key_start']==d['absolute_key_start'] and record['shape']==d['shape'], 'softmax key-column identity')
                center,semantic=m.softmax(scores,actual,row,d['absolute_key_start'],None if layer%6==5 else 512)
            elif leaf=='gate_gelu':center,semantic=m.gelu(own(prefix+'gate_projection'),actual)
            elif leaf=='gated':center=m.half(F(own(prefix+'gate_gelu')*own(prefix+'up_projection')))
            elif leaf in ('attention_residual','output'):
                a=own(previous if leaf=='attention_residual'else prefix+'attention_residual')
                b=own(prefix+('post_attention_norm'if leaf=='attention_residual'else'post_ff_norm'))
                center=m.half(np.clip(F(a+b),F(-65504),F(65504)))
            else:raise m.ContractError('unhandled local stage: '+stage)
        m.require(center is not None and center.shape==actual.shape, 'local oracle shape')
        if semantic is None:semantic=m.equal_numeric(actual,center)
        passed,absolute,fraction=numeric_gate(actual,center,gate)
        self.groups+=1; self.values+=actual.size
        self.semantic_failures+=int(np.count_nonzero(~semantic)); self.numeric_failures+=int(np.count_nonzero(~passed))
        self.max_abs=max(self.max_abs,absolute); self.max_fraction=max(self.max_fraction,fraction)
        bad=~semantic|~passed
        if np.any(bad)and self.first_failure is None:
            index=int(np.flatnonzero(bad.reshape(-1))[0])
            self.first_failure=dict(label=label,row=row,stage=stage,index=index,
                native_bits=m.bits(actual.flat[index]),own_input_point_bits=m.bits(center.flat[index]),
                semantic_pass=bool(semantic.flat[index]),local_numeric_pass=bool(passed.flat[index]),
                gate=dict(gate),raw_descriptor=dict(record),
                operand_descriptors={name:dict(d)for (r,name),(a,d)in self.saved.items()if r==row})
        self.saved[row,stage]=(actual.copy(),dict(record))
        # Only this/previous layer is needed. Release old scratch as soon as the
        # next layer's actual input_norm has consumed previous.output.
        if leaf=='output':
            layer=int(stage.split('.')[1])
            self.saved={k:v for k,v in self.saved.items()if k[0]!=row or k[1]==stage or not k[1].startswith('layers.')}
        m.require(sum(a.nbytes for a,d in self.saved.values())<=4*1024**2,'actual operand-cache cap')
        self.progress.update(self.summary())
        return semantic,passed
    def summary(self):
        return dict(groups=self.groups,values=self.values,semantic_failures=self.semantic_failures,
                    numeric_failures=self.numeric_failures,max_abs=self.max_abs,max_fraction=self.max_fraction,
                    projection_exact_fallbacks=self.fallbacks,first_failure=self.first_failure,
                    numeric_owner_cap_bytes=64*1024**2,provider_check='fixed published admissible sets; softmax interval enclosure, not hidden-intrinsic reconstruction')
