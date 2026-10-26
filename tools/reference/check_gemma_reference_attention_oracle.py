"""Independent stdlib input recipe and sequential-dot replay before fast code."""
from array import array
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct
import sys
import check_gemma_hybrid_tiny as safe

CASES = [
    dict(name='single',queries=1,keys=1,heads=2,kv_heads=1,head_dim=3,window=None,key_start=0,query_start=0),
    dict(name='tiny_local',queries=3,keys=7,heads=2,kv_heads=1,head_dim=64,window=4,key_start=0,query_start=4),
    dict(name='trimmed_local512',queries=5,keys=516,heads=8,kv_heads=1,head_dim=256,window=512,key_start=509,query_start=1020),
    dict(name='global1025_tail',queries=9,keys=1025,heads=8,kv_heads=1,head_dim=512,window=None,key_start=0,query_start=1016),
    dict(name='local512_tail',queries=9,keys=1025,heads=8,kv_heads=1,head_dim=512,window=512,key_start=0,query_start=1016),
    dict(name='local4_long',queries=9,keys=1025,heads=2,kv_heads=1,head_dim=128,window=4,key_start=0,query_start=1016),
    dict(name='translated',queries=3,keys=7,heads=2,kv_heads=1,head_dim=3,window=4,key_start=17,query_start=21),
    dict(name='cancellation',queries=3,keys=513,heads=2,kv_heads=1,head_dim=16,window=None,key_start=0,query_start=510),
]
SHAPES = ('q','k','v','scores','probabilities','context')


def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def rational(case,kind,head,position,feature):
    if case=='cancellation':
        if kind=='q':return Fraction((1024,1,-1024,-1)[feature%4])
        if kind=='k':
            return Fraction(1)+Fraction(position%2,2**20)+(Fraction(position%7,2**24)if feature%4==1 else 0)
        return Fraction(0 if position==512 else(1024,1,-1024,-1)[position%4])+Fraction(feature%7,2**8)
    return Fraction((head*31+position*17+feature*7+dict(q=3,k=11,v=23)[kind])%53-26,13)


def dimensions(case,name):
    h,q,k,d=case['heads'],case['queries'],case['keys'],case['head_dim']
    return dict(q=[1,h,q,d],k=[1,1,k,d],v=[1,1,k,d],scores=[h,q,k],
                probabilities=[h,q,k],context=[1,h,q,d])[name]


def values(raw,item):
    body=raw[item['offset']:item['offset']+item['bytes']]
    safe.require(len(body)==item['bytes']and digest(body)==item['sha256'],'attention oracle extent/digest differs')
    result=array('d'if item['dtype']=='<f8'else'f');result.frombytes(body)
    if sys.byteorder!='little':result.byteswap()
    safe.require(all(math.isfinite(v)for v in result),'finite attention oracle required')
    return result


def gate(actual,expected,label,report):
    safe.same(len(actual),len(expected));maximum=0.
    for a,b in zip(actual,expected,strict=True):
        ratio=abs(a-b)/(1e-5+1e-4*abs(b));maximum=max(maximum,ratio)
    report.append(dict(label=label,values=len(actual),maximum_gate_fraction=maximum,passed=maximum<=1.))
    safe.require(maximum<=1.,'independent attention '+label+' differs')


def verify(directory,manifest_sha,report):
    body=safe.read(directory/'manifest.json',256*1024)
    safe.require(digest(body)==manifest_sha,'external attention oracle differs');manifest=safe.document(body)
    safe.same(set(manifest),{'format','passed','primary_error','endpoint_errors','last_attempt','cases','arrays','binary','arithmetic',
                             'new_fast_reference_implemented','engine_executed','real_graph_qualified','device_qualified'})
    safe.same(manifest['format'],'toklet-reference-attention-fsum-oracle-v1')
    safe.same(manifest['passed'],True);safe.same(manifest['primary_error'],None);safe.same(manifest['endpoint_errors'],[])
    safe.same(manifest['cases'],CASES);safe.same(manifest['last_attempt'],dict(case='cancellation',contract='staged_fp32',stage='complete'))
    safe.same(manifest['arithmetic'],'FP64 scalar products/fsum; scalar exp/fsum softmax; explicit score/probability/context stage casts')
    for key in('new_fast_reference_implemented','engine_executed','real_graph_qualified','device_qualified'):safe.same(manifest[key],False)
    raw=safe.read(directory/'arrays.bin',64*1024**2)
    safe.same(manifest['binary'],dict(bytes=len(raw),sha256=digest(raw)));safe.same(len(manifest['arrays']),96)
    cursor=0;index=0;report['comparisons']=[];report['inputs']=[]
    for case in CASES:
        heads,queries,keys,dims=case['heads'],case['queries'],case['keys'],case['head_dim']
        for contract,dtype,size in [('math','<f8',8),('staged_fp32','<f4',4)]:
            def rounded(value):return struct.unpack('<f',struct.pack('<f',value))[0]if size==4 else value
            data={}
            for name in SHAPES:
                item=manifest['arrays'][index];index+=1;shape=dimensions(case,name);count=math.prod(shape)*size
                expected=dict(case=case['name'],contract=contract,name=name,shape=shape,dtype=dtype,offset=cursor,bytes=count,captured=True)
                safe.same(set(item),set(expected)|{'sha256'});safe.same({key:item[key]for key in expected},expected)
                data[name]=values(raw,item);cursor+=count
                if name in('q','k','v'):
                    origin=case['query_start']if name=='q'else case['key_start']
                    n=queries if name=='q'else keys;h=heads if name=='q'else 1
                    rebuilt=array('d'if size==8 else'f',(float(rational(case['name'],name,head,origin+position,feature))for head in range(h)for position in range(n)for feature in range(dims)))
                    safe.require(rebuilt==data[name],'independent attention input recipe differs')
                    report['inputs'].append(dict(case=case['name'],contract=contract,name=name,values=len(rebuilt),passed=True))
            scores=[];probabilities=[0.]*(heads*queries*keys);context=[]
            for head in range(heads):
                for query in range(queries):
                    qbase=(head*queries+query)*dims
                    for key in range(keys):
                        # Deliberately sequential FP64 addition, independent of
                        # the producer's fsum product iterator.
                        total=0.
                        for feature in range(dims):total+=float(data['q'][qbase+feature])*float(data['k'][key*dims+feature])
                        scores.append(rounded(total))
                    absolute=case['query_start']+query;origin=case['key_start']
                    start=0 if case['window']is None else max(0,absolute-case['window']+1-origin)
                    end=absolute-origin+1;base=(head*queries+query)*keys
                    selected=data['scores'][base+start:base+end];maximum=max(selected)
                    exponent=[math.exp(float(value)-float(maximum))for value in selected];total=math.fsum(exponent)
                    for key,value in enumerate(exponent):probabilities[base+start+key]=rounded(value/total)
                    for feature in range(dims):
                        total=0.
                        for key in range(start,end):total+=float(data['probabilities'][base+key])*float(data['v'][key*dims+feature])
                        context.append(rounded(total))
            prefix=case['name']+'/'+contract+'/'
            gate(data['scores'],scores,prefix+'scores',report['comparisons'])
            gate(data['probabilities'],probabilities,prefix+'probabilities',report['comparisons'])
            gate(data['context'],context,prefix+'context',report['comparisons'])
    safe.same(cursor,len(raw));report.update(passed=True,cases=8,input_arrays=48,output_arrays=48,
                                            compared_values=sum(v['values']for v in report['comparisons']))
    safe.require(digest(safe.read(directory/'manifest.json',256*1024))==manifest_sha,'attention manifest endpoint differs')
    safe.require(digest(safe.read(directory/'arrays.bin',64*1024**2))==manifest['binary']['sha256'],'attention binary endpoint differs')
    return manifest,raw


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--manifest-sha',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=dict(passed=False,primary_error=None)
    try:verify(args.capture,args.manifest_sha,report)
    except BaseException as error:report.update(passed=False,primary_error=type(error).__name__+': '+str(error))
    finally:
        with args.output.open('xb')as file:file.write(encoded(report))
    print(json.dumps({key:report[key]for key in('passed','primary_error')}));sys.exit(0 if report['passed']else 1)
