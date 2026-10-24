"""Pure control-flow witness for Gemma4 KV routing, not numerical execution.

Selected captured bodies execute with strict shape/identity-only stand-ins.
Their constructors, numerical operators, PLE, MoE and learned scalars are not
qualified by this witness. Native stale-input rejection is a separate policy.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

REVISION='0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
SOURCE_SHA='73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d'
FORMAT='toklet-gemma-shared-kv-routing-expected-v1'
MAX_JSON=2*1024*1024
LOCAL,FULL='sliding_attention','full_attention'
SPECS=(
    ('actual35',tuple(([LOCAL]*4+[FULL])*7),20,512,(511,1,1,2,1)),
    ('alternating6',(LOCAL,FULL,LOCAL,FULL,LOCAL,FULL),3,3,(2,2,1,3)),
    ('no_sharing',(LOCAL,FULL,LOCAL,FULL),0,3,(1,4,1)),
    ('irregular',(FULL,LOCAL,LOCAL,FULL,LOCAL),2,1,(2,1,2)),
)


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def mapping(types,shared):
    if (not isinstance(types,(tuple,list)) or not 1<=len(types)<=35 or any(type(t) is not str or t not in (LOCAL,FULL) for t in types)
            or type(shared) is not int or not 0<=shared<len(types)):
        raise ValueError('Bounded supported layer types and at least one producer required')
    producers=len(types)-shared
    result=[]
    for layer,kind in enumerate(types):
        candidates=[i for i in range(min(layer+1,producers)) if types[i]==kind]
        if not candidates:raise ValueError('No preceding producer for shared layer type')
        result.append(layer if layer<producers else max(candidates))
    return result


def export():
    cases=[]
    for name,types,shared,window,chunks in SPECS:
        route=mapping(types,shared);owned=len(types)-shared;steps=[];offset=0
        for index,count in enumerate(chunks):
            layers=[]
            for layer,producer in enumerate(route):
                local=types[layer]==LOCAL;start=max(0,offset-window+1) if local else 0
                own=layer<owned
                layers.append(dict(layer=layer,producer=producer,layer_type=types[layer],cache_present=own,
                    shared_identity_current_step=not own,shared_kv_present=not own,shared_offset=None if own else offset,
                    returned_offset=offset,key_start=start,key_end=offset+count,
                    ops=dict(q_proj=1,q_norm=1,q_rope=1,k_proj=int(own),v_proj=int(own),k_norm=int(own),v_norm=int(own),k_rope=int(own),append=int(own))))
            steps.append(dict(index=index,before=offset,count=count,after=offset+count,layers=layers,
                cache_offsets=[offset+count]*owned,cache_updates=[index+1]*owned))
            offset+=count
        cases.append(dict(name=name,types=list(types),shared=shared,window=window,chunks=list(chunks),producer_map=route,steps=steps))
    return encoded(dict(format=FORMAT,source_revision=REVISION,source_sha256=SOURCE_SHA,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases,
        numerical_execution_qualified=False,native_stale_rejection_qualified=False,graph_qualified=False,device_qualified=False))


def captured(path,expected,maximum=MAX_JSON):
    if path.is_symlink() or not path.is_file():raise ValueError('Regular captured routing input required')
    with path.open('rb') as stream:raw=stream.read(maximum+1)
    if len(raw)>maximum or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Captured routing input identity differs')
    return raw


def source_nodes(raw):
    if len(raw)>128*1024 or hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Pinned Gemma4 source identity differs')
    selected={'Attention':('__call__',),'DecoderLayer':('__call__',),'Gemma4TextModel':('__init__','_make_masks','__call__')}
    result=[]
    for name,methods in selected.items():
        classes=[node for node in ast.parse(raw).body if isinstance(node,ast.ClassDef) and node.name==name]
        if len(classes)!=1:raise ValueError('Unique selected routing class required')
        nodes=[node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name in methods]
        if len(nodes)!=len(methods) or {node.name for node in nodes}!=set(methods):raise ValueError('Unique selected routing methods required')
        # Bases and selected method bodies are captured; constructors not listed
        # above are deliberately replaced by the documented shape-only setup.
        result.append(ast.ClassDef(name=name,bases=classes[0].bases,keywords=classes[0].keywords,body=nodes,decorator_list=[]))
    return result


@dataclass(frozen=True)
class Tensor:
    shape: tuple
    role: str
    owner: int|None=None
    step: int|None=None
    start: int|None=None
    end: int|None=None

    def reshape(self,*shape):
        if any(type(x) is not int or x==0 or x< -1 for x in shape) or shape.count(-1)>1:raise ValueError('Unsupported shape-only reshape')
        count=math.prod(self.shape)
        if -1 in shape:
            known=math.prod(x for x in shape if x!=-1)
            if count%known:raise ValueError('Shape-only reshape changes count')
            shape=tuple(count//known if x==-1 else x for x in shape)
        if math.prod(shape)!=count:raise ValueError('Shape-only reshape changes count')
        return Tensor(tuple(shape),self.role,self.owner,self.step,self.start,self.end)

    def transpose(self,*axes):
        if sorted(axes)!=list(range(len(self.shape))):raise ValueError('Unsupported shape-only transpose')
        return Tensor(tuple(self.shape[i] for i in axes),self.role,self.owner,self.step,self.start,self.end)

    def __mul__(self,scale):
        if type(scale) is not float or scale!=math.sqrt(1536):raise ValueError('Only declared model embedding scale supported')
        return self

    def __add__(self,other):
        if not isinstance(other,Tensor) or other.shape!=self.shape or self.role!='hidden' or other.role!='hidden':raise ValueError('Only shape-matched hidden residual supported')
        return self


class Module:
    pass


class Identity:
    def __init__(self,*args,**kwargs):
        if (args,kwargs) not in (((),{}),((1536,),dict(eps=1e-6))):raise ValueError('Only declared hidden norm construction supported')
    def __call__(self,value):
        if not isinstance(value,Tensor):raise ValueError('Shape-only identity requires tensor')
        return value


class Embedding:
    def __init__(self,vocabulary,hidden):
        if (vocabulary,hidden)!=(8,1536):raise ValueError('Only declared unused embedding construction supported')
    def __call__(self,*args):raise ValueError('Token embedding execution excluded')


class Replay:
    def __init__(self,nodes):
        self.step=0;self.offset=0;self.count=0;self.calls=[];self.returned={};self.active=None
        self.env=dict(nn=SimpleNamespace(Module=Module,Embedding=Embedding,RMSNorm=Identity),mx=SimpleNamespace(array=self.scalar),create_attention_mask=self.mask,scaled_dot_product_attention=self.attention)
        deferred=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        module=ast.fix_missing_locations(ast.Module(body=[deferred,*nodes],type_ignores=[]))
        exec(compile(module,'authenticated-gemma-routing-selected-source','exec'),self.env)
        self.decoder_class=self.env['DecoderLayer'];self.env['DecoderLayer']=self.layer

    @staticmethod
    def scalar(value):
        if type(value) is not int or value<0:raise ValueError('Only nonnegative offset scalar supported')
        return value

    def mask(self,h,cache,window_size=None):
        if not isinstance(h,Tensor) or cache is None:raise ValueError('Only producer mask requests supported')
        if window_size!=cache.window:raise ValueError('Mask kind/producer differs')
        return (self.step,h.shape[1],cache.offset,window_size)

    def op(self,name,value):
        if self.active is None:raise ValueError('No active routing layer')
        self.active['ops'][name]+=1
        if not isinstance(value,Tensor):raise ValueError('Shape-only operator requires tensor')
        return value

    def projection(self,layer,kind,width):
        def apply(value):
            self.op(kind+'_proj',value)
            if len(value.shape)!=3 or value.shape[-1]!=1536:raise ValueError('Projection input shape differs')
            return Tensor((*value.shape[:2],width),kind,layer,self.step)
        return apply

    def rope(self,value,offset):
        if len(value.shape)!=4 or type(offset) is not int or offset!=self.offset:raise ValueError('Source rotary offset differs')
        return self.op('q_rope' if value.role=='q' else 'k_rope',value)

    def attention(self,q,k,v,cache,scale,mask):
        if (not all(isinstance(x,Tensor) for x in (q,k,v)) or scale!=1. or len(q.shape)!=4 or len(k.shape)!=4
                or v.shape!=k.shape or q.shape[:1]!=k.shape[:1] or q.shape[3]!=k.shape[3] or q.shape[1]%k.shape[1]
                or mask!=(self.step,self.count,self.offset,self.config.sliding_window if self.config.layer_types[self.active['layer']]==LOCAL else None)):
            raise ValueError('Unsupported shape-only attention call')
        self.active.update(producer=k.owner,key_start=k.start,key_end=k.end)
        return Tensor(q.shape,'context')

    def layer(self,config,layer_idx):
        self.config=config
        obj=object.__new__(self.decoder_class);attention=object.__new__(self.env['Attention'])
        kind=config.layer_types[layer_idx];own=layer_idx<config.num_hidden_layers-config.num_kv_shared_layers
        dims=256 if kind==LOCAL else 512
        attention.layer_idx=layer_idx;attention.has_kv=own;attention.use_k_eq_v=False;attention.n_heads=8;attention.n_kv_heads=1;attention.head_dim=dims;attention.scale=1.
        attention.q_proj=self.projection(layer_idx,'q',8*dims)
        attention.q_norm=lambda x:self.op('q_norm',x)
        attention.rope=self.rope
        if own:
            attention.k_proj=self.projection(layer_idx,'k',dims);attention.v_proj=self.projection(layer_idx,'v',dims)
            attention.k_norm=lambda x:self.op('k_norm',x);attention.v_norm=lambda x:self.op('v_norm',x)
        def output(value):
            if value.shape!=(1,self.count,8*dims):raise ValueError('Attention output shape differs')
            return Tensor((1,self.count,1536),'hidden')
        attention.o_proj=output
        def forward(x,mask,cache,shared_kv,offset):
            call=dict(layer=layer_idx,layer_type=kind,cache_present=cache is not None,shared_kv_present=shared_kv is not None,
                shared_identity_current_step=False,shared_offset=offset,ops={name:0 for name in ('q_proj','q_norm','q_rope','k_proj','v_proj','k_norm','v_norm','k_rope','append')})
            self.calls.append(call);self.active=call
            if shared_kv is not None:
                producer=shared_kv[0].owner
                previous=self.returned.get(producer)
                call['shared_identity_current_step']=previous is not None and shared_kv is previous[0] and offset==previous[1] and shared_kv[0].step==self.step and shared_kv[1].step==self.step
            result=attention(x,mask,cache,shared_kv=shared_kv,offset=offset)
            call['returned_offset']=result[2];self.returned[layer_idx]=(result[1],result[2])
            return result
        obj.self_attn=forward;obj.layer_type=kind;obj.input_layernorm=Identity();obj.post_attention_layernorm=Identity();obj.pre_feedforward_layernorm=Identity();obj.post_feedforward_layernorm=Identity();obj.mlp=Identity()
        obj.enable_moe=False;obj.per_layer_input_gate=None;obj.hidden_size_per_layer_input=0;obj.layer_scalar=None
        return obj

    def cache(self,layer,window):
        replay=self
        class Cache:
            def __init__(self):self.offset=0;self.updates=0;self.window=window
            def update_and_fetch(self,k,v):
                replay.op('append',k)
                if k.owner!=layer or v.owner!=layer or k.step!=replay.step or v.step!=replay.step or self.offset!=replay.offset or k.shape!=(1,1,replay.count,k.shape[-1]) or v.shape!=k.shape:
                    raise ValueError('Unsupported shape-only cache append')
                begin=0 if self.window is None else max(0,self.offset-self.window+1)
                self.offset+=replay.count;self.updates+=1
                shape=(1,1,self.offset-begin,k.shape[-1])
                return (Tensor(shape,'k',layer,replay.step,begin,self.offset),Tensor(shape,'v',layer,replay.step,begin,self.offset))
        return Cache()


def compare(data,sources):
    nodes=source_nodes(sources);rows=[];primary=None
    try:
        for case in data['cases']:
            replay=Replay(nodes)
            config=SimpleNamespace(vocab_size=8,hidden_size=1536,sliding_window=case['window'],sliding_window_pattern=5,num_hidden_layers=len(case['types']),num_kv_shared_layers=case['shared'],layer_types=case['types'],hidden_size_per_layer_input=0,rms_norm_eps=1e-6)
            model=replay.env['Gemma4TextModel'](config)
            row=dict(name=case['name'],producer_map=model.previous_kvs,steps=[],completed=False);rows.append(row)
            caches=[replay.cache(i,case['window'] if kind==LOCAL else None) for i,kind in enumerate(case['types'][:len(case['types'])-case['shared']])]
            for step in case['steps']:
                replay.step=step['index'];replay.offset=step['before'];replay.count=step['count'];replay.calls=[];replay.returned={}
                trial=dict(index=replay.step,layers=replay.calls,completed=False);row['steps'].append(trial)
                h=model(input_embeddings=Tensor((1,replay.count,1536),'hidden'),cache=caches)
                trial.update(cache_offsets=[c.offset for c in caches],cache_updates=[c.updates for c in caches],completed=True)
                if h.shape!=(1,replay.count,1536):raise ValueError('Model hidden routing shape differs')
                expected={key:step[key] for key in ('index','layers','cache_offsets','cache_updates')}
                if {key:trial[key] for key in expected}!=expected:raise ValueError('Source routing/identity observation differs from independent witness')
            if model.previous_kvs!=case['producer_map']:raise ValueError('Source producer map differs')
            row['completed']=True
    except Exception as error:primary=type(error).__name__+': '+str(error)
    return dict(format='toklet-gemma-shared-kv-routing-observation-v1',passed=primary is None and len(rows)==len(SPECS) and all(row['completed'] for row in rows),
        cases=rows,primary_error=primary,source_revision=REVISION,source_sha256=SOURCE_SHA,interruption_partial_observation_qualified=False,
        selected_source_definitions=[dict(name=node.name,methods=[dict(name=method.name,line=method.lineno,source=ast.unparse(method)) for method in node.body]) for node in nodes],
        annotation_adaptation='from __future__ import annotations; selected executable bodies unchanged',
        stand_ins='Strict shape/identity-only projections, norms, rotary, attention, cache and containers; Attention/DecoderLayer constructors excluded; PLE, MoE and learned scalars disabled',
        numerical_execution_qualified=False,native_stale_rejection_qualified=False,graph_qualified=False,device_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['export','compare']);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source',type=Path);parser.add_argument('--reference',type=Path);parser.add_argument('--expected-reference-sha256');options=parser.parse_args()
    if options.mode=='export':raw=export()
    else:
        if options.source is None or options.reference is None or options.expected_reference_sha256 is None:parser.error('Explicit captured source/reference identities required')
        raw_reference=captured(options.reference,options.expected_reference_sha256)
        # The owned producer binds the exact fixture; strict byte equality also
        # prevents coercion of metadata types or substitution of expected maps.
        if raw_reference!=export():raise ValueError('Frozen routing witness differs from independent producer')
        report=compare(json.loads(raw_reference),captured(options.source,SOURCE_SHA,128*1024));raw=encoded(report)
    with options.output.open('xb') as stream:stream.write(raw)
    if options.mode=='compare':raise SystemExit(0 if report['passed'] else 1)
