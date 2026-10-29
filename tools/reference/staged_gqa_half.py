"""Independent seeded multi-KV old-span/staged Reference attention.

All active queries/heads/features/materialized keys, including the deep32 case.
Only generic word/serialization helpers are imported. No target, graph or model.
Physical head strides are retention for old and fixed32 for staged, never activeN.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

_path=Path(__file__).with_name('gemma3_staged_attention.py')
_spec=importlib.util.spec_from_file_location('_staged_gqa_words',_path)
helper=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(helper)
f32,half_word,half_value=helper.f32,helper.half_word,helper.half_value
encoded,identity,word_identity=helper.encoded,helper.identity,helper.word_identity
FORMAT='toklet-staged-gqa-half-v1'
PROTOCOL='55a1f4227dfed70a08df601044778a51d7f80d5e742ad6976f469050f5784455'
POLICY=dict(fatal_reference='main',atol_floor=1e-4,envelope_factor=2,rtol=1e-3)
MAX_CHUNK,GUARD,PAD=32,7,0x7e55
MAX_BINARY,MAX_JSON=32*1024**2,1024**2
CASES=tuple(dict(name=name,query_heads=q,kv_heads=kv,dim=64,capacity=cap,
    committed=c,tokens=n,window=w) for name,q,kv,cap,c,n,w in (
    ('tiny_single',4,2,64,0,1,None),('tiny_partial',4,2,64,1,3,None),
    ('tiny_full32',4,2,64,0,32,None),('tiny_wrap',4,2,64,5,7,4),
    ('real_decode',16,8,544,31,1,None),('real_partial',16,8,544,32,3,None),
    ('real_full32',16,8,544,0,32,None),('real_512',16,8,544,511,1,None),
    ('real_513',16,8,544,512,1,None),('real_deep32',16,8,544,512,32,None),
    ('real_wrap',16,8,544,513,2,512)))


def shape(c):
    names=('committed','tokens','dim','capacity','query_heads','kv_heads')
    if any(type(c[k]) is not int for k in names):raise ValueError('Integer geometry')
    start,n,d,cap,q,kv=(c[k] for k in names);w=c['window']
    if (q,kv,d,cap) not in ((4,2,64,64),(16,8,64,544)) or q%kv:
        raise ValueError('Fixed GQA geometry')
    if start<0 or not 1<=n<=MAX_CHUNK or start+n>cap:
        raise ValueError('Absolute position domain')
    if w is not None and (type(w) is not int or w!=(4 if q==4 else 512)):
        raise ValueError('Fixed local compatibility window')
    if c['name'] not in {item['name'] for item in CASES}:raise ValueError('Fixed named case')
    return start,n,d,cap,w,q,kv


def view(c):
    committed,n,d,cap,window,q,kv=shape(c)
    first=max(0,committed-(window-1)) if window else 0
    retention=window or cap;old=committed-first;slot=first%retention
    first_count=min(old,retention-slot);spans=[]
    if first_count:spans.append(dict(absolute_start=first,slot=slot,rows=first_count))
    if old>first_count:spans.append(dict(absolute_start=first+first_count,slot=0,rows=old-first_count))
    return dict(start=first,end=committed+n,old_spans=spans,new_start=committed,new_rows=n,retention=retention)


def copy_plan(c):
    committed,n,d,cap,window,q,kv=shape(c)
    skip=max(0,n-window) if window else 0;keep=n-skip
    slot=(committed+skip)%window if window else committed
    retention=window or cap;first=min(keep,retention-slot)
    runs=[dict(source_row=skip,destination_row=slot,rows=first)]
    if keep>first:runs.append(dict(source_row=skip+first,destination_row=0,rows=keep-first))
    return runs


def intervals(c):
    """Pure geometry evidence, not an ownership/readiness implementation."""
    _,n,d,_,_,_,kv=shape(c);retention=view(c)['retention']
    active=[];copies=[]
    for kind in ('k','v'):
        for head in range(kv):
            active.append(dict(kind=kind,head=head,byte_offset=head*MAX_CHUNK*d*2,bytes=n*d*2))
            for run in copy_plan(c):
                copies.append(dict(kind=kind,head=head,
                    source_byte_offset=(head*MAX_CHUNK+run['source_row'])*d*2,
                    destination_byte_offset=(head*retention+run['destination_row'])*d*2,
                    bytes=run['rows']*d*2))
    return dict(active_stage_fill_and_validation=active,phase_b_copy=copies,
                old_head_stride_bytes=retention*d*2,staged_head_stride_bytes=MAX_CHUNK*d*2)


def row(c,position,head,kind):
    d=c['dim']
    if kind=='k':values=[((position*5+head*7+feature*3+1)%17-8)/32 for feature in range(d)]
    elif kind=='v':values=[((position*11+head*13+feature*5+7)%31-15)/16 for feature in range(d)]
    else:raise ValueError('KV kind')
    result=[half_word(x) for x in values]
    if c['name']=='tiny_single' and kind=='v' and position==0:
        result[0]=0x0001 if head==0 else 0x8001
    return result


def physical(c,committed,kind):
    _,_,d,cap,window,_,kv=shape(c);retention=window or cap
    if type(committed) is not int or not 0<=committed<=cap:raise ValueError('Physical prefix')
    words=[PAD]*(kv*retention*d+GUARD)
    first=max(0,committed-retention) if window else 0
    for head in range(kv):
        for position in range(first,committed):
            slot=position%retention if window else position
            offset=(head*retention+slot)*d
            words[offset:offset+d]=row(c,position,head,kind)
    return words


def operands(c):
    committed,n,d,cap,window,qh,kv=shape(c)
    old={kind:physical(c,committed,kind) for kind in ('k','v')}
    staged={kind:[PAD]*(kv*MAX_CHUNK*d+GUARD) for kind in ('k','v')}
    query=[PAD]*(MAX_CHUNK*qh*d+GUARD)
    for token in range(n):
        for kind in ('k','v'):
            for head in range(kv):
                offset=(head*MAX_CHUNK+token)*d
                staged[kind][offset:offset+d]=row(c,committed+token,head,kind)
        for head in range(qh):
            for feature in range(d):
                value=((7*(committed+token)+5*head+3*feature)%13-6)/32
                query[(token*qh+head)*d+feature]=half_word(value)
    return old,staged,query


def decoded_view(c,old,staged,query):
    committed,n,d,_,_,qh,kv=shape(c);description=view(c);retention=description['retention']
    if len(query)!=MAX_CHUNK*qh*d+GUARD:raise ValueError('Q extent')
    for kind in ('k','v'):
        if len(old[kind])!=kv*retention*d+GUARD or len(staged[kind])!=kv*MAX_CHUNK*d+GUARD:
            raise ValueError('KV extent')
    selected={kind:[[] for _ in range(kv)] for kind in ('k','v')}
    for head in range(kv):
        for position in range(description['start'],description['end']):
            for kind in ('k','v'):
                if position<committed:
                    offset=(head*retention+position%retention)*d;words=old[kind][offset:offset+d]
                else:
                    offset=(head*MAX_CHUNK+position-committed)*d;words=staged[kind][offset:offset+d]
                values=[half_value(w) for w in words]
                if any(not math.isfinite(x) for x in values):raise ValueError('Nonfinite active KV')
                selected[kind][head].append(values)
    queries=[[half_value(w) for w in query[token*qh*d:(token+1)*qh*d]] for token in range(n)]
    if any(not math.isfinite(x) for values in queries for x in values):raise ValueError('Nonfinite active Q')
    return description,queries,selected['k'],selected['v']


def attention(c,old,staged,query,profile):
    if profile not in ('main','wide'):raise ValueError('Precision profile')
    committed,n,d,_,window,qh,kv=shape(c)
    description,queries,head_keys,head_values=decoded_view(c,old,staged,query)
    first=description['start'];count=description['end']-first
    scores,probs=[],[];context=[0]*(n*qh*d)
    root=f32(math.sqrt(d)) if profile=='main' else math.sqrt(d)
    for head in range(qh):
        selected_head=head//(qh//kv)
        keys,values=head_keys[selected_head],head_values[selected_head]
        for token,q in enumerate(queries):
            position=committed+token
            first_visible=max(first,position-(window-1)) if window else first
            included=range(first_visible-first,position-first+1)
            row_scores=[]
            for key in keys:
                if profile=='main':
                    total=0.0
                    for feature in range(d):total=f32(total+f32(q[head*d+feature]*key[feature]))
                else:total=math.fsum(q[head*d+feature]*key[feature] for feature in range(d))
                row_scores.append(f32(total/root))
            maximum=max(row_scores[i] for i in included)
            exponential=[0.0]*count;mass=0.0
            for i in range(count):
                if i in included:
                    difference=f32(row_scores[i]-maximum) if profile=='main' else row_scores[i]-maximum
                    e=math.exp(difference);exponential[i]=f32(e) if profile=='main' else e
                mass=f32(mass+exponential[i]) if profile=='main' else mass+exponential[i]
            if not math.isfinite(mass) or mass<=0:raise ValueError('Softmax mass')
            row_probs=[f32(e/mass) for e in exponential]
            for feature in range(d):
                if profile=='main':
                    total=0.0
                    for i in included:total=f32(total+f32(row_probs[i]*values[i][feature]))
                else:total=math.fsum(row_probs[i]*values[i][feature] for i in included)
                context[(token*qh+head)*d+feature]=half_word(total)
            scores.extend(row_scores);probs.extend(row_probs)
    if any(not math.isfinite(x) for x in scores+probs) or any(not math.isfinite(half_value(x)) for x in context):
        raise ValueError('Nonfinite attention output')
    return dict(scores=scores,probabilities=probs,context=context)


def commit_reference(c,old,staged):
    _,_,d,_,_,_,kv=shape(c);retention=view(c)['retention']
    result={kind:words[:] for kind,words in old.items()}
    for head in range(kv):
        for run in copy_plan(c):
            source=(head*MAX_CHUNK+run['source_row'])*d
            target=(head*retention+run['destination_row'])*d;length=run['rows']*d
            for kind in ('k','v'):result[kind][target:target+length]=staged[kind][source:source+length]
    return result


def numerical_checks(c,output):
    committed,n,d,_,window,qh,kv=shape(c);desc=view(c);keys=desc['end']-desc['start']
    if len(output['context'])!=n*qh*d or any(len(output[k])!=qh*n*keys for k in ('scores','probabilities')):
        raise ValueError('Numerical shape')
    visible=[[max(desc['start'],committed+t-window+1) if window else desc['start'],committed+t+1] for t in range(n)]
    masked=0
    for head in range(qh):
        for token,(start,end) in enumerate(visible):
            for i in range(keys):
                if not start<=desc['start']+i<end:
                    value=output['probabilities'][(head*n+token)*keys+i]
                    if struct.pack('<f',value)!=b'\0'*4:raise ValueError('Masked probability not positive zero')
                    masked+=1
    if c['name']=='tiny_single':
        if output['probabilities']!=[1.]*qh:raise ValueError('One-key probability')
        for head in range(qh):
            if output['context'][head*d:(head+1)*d]!=row(c,0,head//(qh//kv),'v'):
                raise ValueError('One-key GQA V identity/subnormal')
    return visible,masked


def export(directory):
    producer=identity(Path(__file__).read_bytes());helpers=identity(Path(helper.__file__).read_bytes())
    directory.mkdir(parents=False,exist_ok=False)
    cases=[];chain=None;counts=dict(context=0,scores=0,probabilities=0,physical_words=0,masked_probability_values=0)
    try:
        with (directory/'arrays.bin').open('xb') as binary,(directory/'progress.jsonl').open('x') as progress:
            def save(values,kind,dimensions):
                raw=b''.join(struct.pack('<'+kind,x) for x in values);offset=binary.tell()
                if offset+len(raw)>MAX_BINARY:raise ValueError('Binary bound')
                binary.write(raw)
                return dict(file='arrays.bin',dtype='u16le' if kind=='H' else 'f32le',offset=offset,shape=dimensions,**identity(raw))
            for c in CASES:
                progress.write(json.dumps(dict(case=c['name'],started=True))+'\n');progress.flush()
                old,staged,q=operands(c)
                pins={name:word_identity(values) for name,values in [('q',q),*[(f'old_{k}',v) for k,v in old.items()],*[(f'new_{k}',v) for k,v in staged.items()]]}
                if c['name']=='real_513' and old!=chain:raise ValueError('512-to513 physical chain')
                main=attention(c,old,staged,q,'main');wide=attention(c,old,staged,q,'wide')
                visible,masked=numerical_checks(c,main)
                if numerical_checks(c,wide)!=(visible,masked):raise ValueError('Profile mask differs')
                counts['masked_probability_values']+=2*masked
                after=commit_reference(c,old,staged)
                committed,n,d,cap,window,qh,kv=shape(c)
                expected={kind:physical(c,committed+n,kind) for kind in ('k','v')}
                if after!=expected:raise ValueError('Per-head copies differ from absolute rebuild')
                if c['name']=='real_512':chain=after
                for name,values in [('q',q),*[(f'old_{k}',v) for k,v in old.items()],*[(f'new_{k}',v) for k,v in staged.items()]]:
                    if word_identity(values)!=pins[name]:raise ValueError('Source mutated input')
                inputs={'q':save(q,'H',[len(q)])};counts['physical_words']+=len(q)
                for kind in ('k','v'):
                    for prefix,words in [('old_',old[kind]),('new_',staged[kind]),('expected_after_',after[kind])]:
                        inputs[prefix+kind]=save(words,'H',[len(words)]);counts['physical_words']+=len(words)
                desc=view(c);arrays={}
                for stage in ('scores','probabilities','context'):
                    is_context=stage=='context'
                    a=[half_value(x) for x in main[stage]] if is_context else main[stage]
                    b=[half_value(x) for x in wide[stage]] if is_context else wide[stage]
                    envelope=max(abs(x-y) for x,y in zip(a,b,strict=True))
                    dimensions=[n,qh,d] if is_context else [qh,n,desc['end']-desc['start']]
                    kind='H' if is_context else 'f'
                    arrays[stage]=dict(main=save(main[stage],kind,dimensions),wide=save(wide[stage],kind,dimensions),envelope=envelope,
                        gate=dict(atol=max(POLICY['atol_floor'],POLICY['envelope_factor']*envelope),rtol=POLICY['rtol']))
                    counts[stage]+=len(a)
                cases.append(dict(**c,view=desc,visible_ranges=visible,copy_runs=copy_plan(c),intervals=intervals(c),
                    chunk_capacity=MAX_CHUNK,score_key_capacity=cap,selected_query_rows=list(range(n)),all_active_queries=True,
                    inputs=inputs,arrays=arrays,seeded_state_only=True,source_input_immutability=pins,
                    exact_one_key_context=c['name']=='tiny_single'))
                binary.flush();progress.write(json.dumps(dict(case=c['name'],complete=True,arrays_end=binary.tell()))+'\n');progress.flush()
            if binary.tell()!=22405366:raise ValueError('Frozen exact binary bytes')
        if (counts['context'],counts['scores'],counts['probabilities'],counts['physical_words'])!=(84736,334348,334348,8358427):
            raise ValueError('Frozen all-query counts')
        files={'arrays.bin':identity((directory/'arrays.bin').read_bytes())}
        if identity(Path(__file__).read_bytes())!=producer or identity(Path(helper.__file__).read_bytes())!=helpers:
            raise ValueError('Source endpoint changed')
        if identity((directory/'arrays.bin').read_bytes())!=files['arrays.bin']:raise ValueError('Output endpoint changed')
        manifest=dict(format=FORMAT,protocol_sha256=PROTOCOL,producer=producer,helper=dict(path=str(Path(helper.__file__).resolve()),**helpers),
            files=files,cases=cases,policy=POLICY,guard=GUARD,input_padding_half_word=PAD,
            numerical_profile_arrays=66,physical_state_arrays=77,stored_arrays=143,context_values_per_profile=counts['context'],
            score_values_per_profile=counts['scores'],probability_values_per_profile=counts['probabilities'],physical_half_words=counts['physical_words'],
            masked_positive_zero_values_both_profiles=counts['masked_probability_values'],source_reference_executed=True,target_gpu_executed=False,full_graph_executed=False,
            runtime=dict(python=sys.version,executable=str(Path(sys.executable).resolve())),
            scope='Full-query seeded GQA Reference arithmetic/copy-interval expectations; no runtime Ready, owner publication, mixed graph or model/quality/performance qualification')
        raw=encoded(manifest)
        if len(raw)>MAX_JSON:raise ValueError('Manifest bound')
        (directory/'manifest.json').write_bytes(raw);return identity(raw)
    except BaseException as error:
        (directory/'failure.json').write_bytes(encoded(dict(complete=False,error=type(error).__name__+': '+str(error))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    print(json.dumps(export(parser.parse_args().output_dir)))
