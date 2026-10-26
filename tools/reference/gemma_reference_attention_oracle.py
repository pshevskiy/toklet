"""Pre-target scalar fsum attention data; no NumPy or fast implementation import."""
from array import array
from pathlib import Path
import argparse
import hashlib
import json
import math
import signal
import struct
import sys
import check_gemma_hybrid_tiny as safe

SPECS = [
    ('single',1,1,2,3,None,0),
    ('tiny_local',3,7,2,64,4,0),
    ('trimmed_local512',5,516,8,256,512,509),
    ('global1025_tail',9,1025,8,512,None,0),
    ('local512_tail',9,1025,8,512,512,0),
    ('local4_long',9,1025,2,128,4,0),
    ('translated',3,7,2,3,4,17),
    ('cancellation',3,513,2,16,None,0),
]


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()


def recipe(case,kind,head,position,feature):
    if case=='cancellation':
        if kind=='q':return (1024.,1.,-1024.,-1.)[feature%4]
        if kind=='k':return 1.+(position%2)*2**-20+((position%7)*2**-24 if feature%4==1 else 0.)
        return (0. if position==512 else (1024.,1.,-1024.,-1.)[position%4])+(feature%7)*2**-8
    shift={'q':3,'k':11,'v':23}[kind]
    return (((head*31+position*17+feature*7+shift)%53)-26)/13.


def export(destination):
    destination.mkdir();entries=[];cursor=0;primary=None;attempt=None
    try:
        with(destination/'arrays.bin').open('xb')as file:
            def write(case,contract,name,value,shape):
                nonlocal cursor
                data=array(value.typecode,value)
                if sys.byteorder!='little':data.byteswap()
                raw=data.tobytes();assert cursor+len(raw)<=64*1024**2
                item=dict(case=case,contract=contract,name=name,shape=shape,
                          dtype='<f8'if value.typecode=='d'else'<f4',offset=cursor,bytes=len(raw),sha256=digest(raw),captured=False)
                entries.append(item);assert file.write(raw)==len(raw);file.flush();item['captured']=True;cursor+=len(raw)
            for case,queries,keys,heads,dims,window,origin in SPECS:
                query_start=origin+keys-queries
                for contract,code in [('math','d'),('staged_fp32','f')]:
                    attempt=dict(case=case,contract=contract,stage='input')
                    def rounded(value):return struct.unpack('<f',struct.pack('<f',value))[0]if code=='f'else value
                    q=array(code,(recipe(case,'q',head,query_start+position,d)for head in range(heads)for position in range(queries)for d in range(dims)))
                    k=array(code,(recipe(case,'k',0,origin+position,d)for position in range(keys)for d in range(dims)))
                    v=array(code,(recipe(case,'v',0,origin+position,d)for position in range(keys)for d in range(dims)))
                    for name,value,shape in [('q',q,[1,heads,queries,dims]),('k',k,[1,1,keys,dims]),('v',v,[1,1,keys,dims])]:
                        write(case,contract,name,value,shape)
                    attempt['stage']='computing scalar reductions'
                    scores=array(code,[0.])*(heads*queries*keys)
                    probabilities=array(code,[0.])*(heads*queries*keys)
                    context=array(code,[0.])*(heads*queries*dims)
                    for head in range(heads):
                        for row in range(queries):
                            absolute=query_start+row;begin=max(origin,absolute-window+1)if window is not None else origin
                            base=(head*queries+row)*keys
                            query=q[(head*queries+row)*dims:(head*queries+row+1)*dims]
                            for position in range(keys):
                                scores[base+position]=rounded(math.fsum(float(query[d])*float(k[position*dims+d])for d in range(dims)))
                            selected=scores[base+begin-origin:base+absolute-origin+1];maximum=max(selected)
                            exponent=[math.exp(float(value)-float(maximum))for value in selected];total=math.fsum(exponent)
                            for index,value in enumerate(exponent):probabilities[base+begin-origin+index]=rounded(value/total)
                            for d in range(dims):
                                context[(head*queries+row)*dims+d]=rounded(math.fsum(float(probabilities[base+position])*float(v[position*dims+d])for position in range(begin-origin,absolute-origin+1)))
                    for name,value,shape in [('scores',scores,[heads,queries,keys]),('probabilities',probabilities,[heads,queries,keys]),('context',context,[1,heads,queries,dims])]:
                        write(case,contract,name,value,shape)
                        assert all(math.isfinite(x)for x in value)
                    attempt['stage']='complete'
    except BaseException as error:
        primary=type(error).__name__+': '+str(error)
    finally:
        previous={sig:signal.signal(sig,signal.SIG_IGN)for sig in(signal.SIGINT,signal.SIGTERM)}
        binary=None;endpoint_errors=[]
        try:
            raw=safe.read(destination/'arrays.bin',64*1024**2)
            binary=dict(bytes=len(raw),sha256=digest(raw))
        except BaseException as error:endpoint_errors.append(type(error).__name__+': '+str(error))
        manifest=dict(format='toklet-reference-attention-fsum-oracle-v1',passed=primary is None and not endpoint_errors,primary_error=primary,endpoint_errors=endpoint_errors,last_attempt=attempt,cases=[dict(name=name,queries=q,keys=k,heads=h,kv_heads=1,head_dim=d,window=w,key_start=o,query_start=o+k-q)for name,q,k,h,d,w,o in SPECS],
                      arrays=entries,binary=binary,
                      arithmetic='FP64 scalar products/fsum; scalar exp/fsum softmax; explicit score/probability/context stage casts',
                      new_fast_reference_implemented=False,engine_executed=False,real_graph_qualified=False,device_qualified=False)
        with(destination/'manifest.json').open('xb')as file:file.write(encoded(manifest))
        for sig,handler in previous.items():signal.signal(sig,handler)
    return dict(passed=manifest['passed'],primary_error=primary,endpoint_errors=endpoint_errors,manifest=dict(bytes=len(encoded(manifest)),sha256=digest(encoded(manifest))),binary=manifest['binary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    def stop(number,_):
        for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
        raise InterruptedError(str(number))
    previous={sig:signal.signal(sig,stop)for sig in(signal.SIGINT,signal.SIGTERM)}
    try:result=export(args.output)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps(result));sys.exit(0 if result['passed']else 1)
