"""Independent full trace replay from frozen Gemma graph arrays; no native import."""
from pathlib import Path
import hashlib,json,math,os,re,stat,struct,sys
TEST='model::gemma_packed::tests::hybrid::frozen_tiny_gpu_products_and_gathers_share_cpu_graph_and_cache_transactions'
PINS={'math.json':'fd32f9f5dfb3791a31cc595ac527e10d7eb2af9329120aae8f4da70d2ac2f87e','math.bin':'a597c8d965a48f7f5e93003834f258002059c8bf487f42220d88445592089832'}

def require(ok,message):
    if not ok:raise ValueError(message)

def same(a,b):
    require(type(a)is type(b),'typed field differs')
    if isinstance(a,dict):
        require(a.keys()==b.keys(),'record fields differ')
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        require(len(a)==len(b),'record length differs')
        for v,w in zip(a,b,strict=True):same(v,w)
    else:require(a==b,'record value differs')

def read(path,limit):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    def identity(s):return s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns
    with os.fdopen(fd,'rb') as f:
        before=os.fstat(f.fileno());require(stat.S_ISREG(before.st_mode) and before.st_size<=limit,'regular bounded input required')
        raw=f.read(limit+1);require(len(raw)==before.st_size and len(raw)<=limit,'input length differs')
        require(identity(before)==identity(os.fstat(f.fileno()))==identity(path.lstat()),'input changed')
    return raw

def document(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON field');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('finite JSON required')))

def fixture(path):
    raw={name:read(path/name,8*1024**2) for name in PINS}
    for name,pin in PINS.items():require(hashlib.sha256(raw[name]).hexdigest()==pin,'frozen fixture identity: '+name)
    return document(raw['math.json']),raw['math.bin']

def signatures(data):
    out=[]
    for schedule in data['schedules']:
        offset=0
        for count in schedule['chunks']:
            for layer in range(4):
                base=f'layer.{layer}.'
                for stage in ['output','self_attn.o_proj','post_per_layer_input_norm','per_layer_projection','per_layer_input_gate','mlp.up_proj','self_attn.q_norm','q_rotary','context']:
                    out.append((base+stage,offset,count))
            for name in ['logits','final_norm','layer.3.output','layer.3.self_attn.o_proj','layer.3.post_per_layer_input_norm','layer.3.per_layer_projection','layer.3.per_layer_input_gate','layer.3.mlp.up_proj','ple_lookup','ple_projection','layer.3.q_rotary','layer.3.context']:
                out.append((name,offset,count))
            after=offset+count
            for owner,capacity in [(0,4),(1,32)]:
                for stage in ['k_rotary','self_attn.v_norm']:out.append((f'layer.{owner}.{stage}',offset,count))
                for slot in range(capacity):
                    positions=[p for p in range(max(0,after-capacity),after) if p%capacity==slot]
                    for p in positions:
                        for stage in ['k_rotary','self_attn.v_norm']:out.append((f'layer.{owner}.{stage}',p,1))
            offset=after
        require(offset==32,'frozen schedule capacity')
        out.append(('logits',0,32))
    return out

def values(data,binary,contract,name,before,count):
    spec=data[contract][name];shape=spec['shape'];axis=2 if name.endswith(('rotary','context')) else 1
    require(0<=before<before+count<=shape[axis],'observation position bound')
    stride=math.prod(shape[axis+1:]);code,size=('d',8) if spec['dtype']=='<f8' else ('f',4)
    raw=binary[spec['offset']:spec['offset']+spec['bytes']]
    require(len(raw)==math.prod(shape)*size and hashlib.sha256(raw).hexdigest()==spec['sha256'],'complete frozen array')
    return [v for index,(v,) in enumerate(struct.iter_unpack('<'+code,raw)) if before<=index//stride%shape[axis]<before+count]

def records(raw):
    require(len(raw)<=16*1024**2,'bounded transcript')
    events=[];wrapper=[]
    for line in raw.decode().splitlines():
        require(len(line.encode())<=4096,'bounded record')
        if 'GEMMA_HYBRID_' in line:
            before,rest=line.split('GEMMA_HYBRID_',1)
            require(before in ('','test '+TEST+' ... '),'exact test prefix')
            name,body=rest.split(' ',1)
            if name=='TINY_COMPLETE':
                require(body=='frozen_graph=true scalar_order_unchanged=true full_e2b_qualified=false','scope differs');value=None
            else:value=document(body)
            events.append((name,value))
        elif line.strip():wrapper.append(line)
    require(len(wrapper)==3 and wrapper[:2]==['running 1 test','ok'] and re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s',wrapper[2]),'complete exact successful wrapper')
    return events

def replay(events,raw,data,binary):
    expected=signatures(data);require(len(events)==2*len(expected)+2,'complete event count')
    offset=0;comparisons=0;maximum={'math':0.,'staged_fp32':0.}
    for number,(name,before,count) in enumerate(expected):
        attempted,attempt=events[2*number];finished,actual=events[2*number+1]
        require(attempted=='ARRAY_ATTEMPT' and finished=='ARRAY','ordered observation pair')
        wanted=values(data,binary,'staged_fp32',name,before,count);size=len(wanted)*4
        header=dict(record=number,name=name,offset=offset,bytes=size,before=before,count=count,values=len(wanted))
        same(attempt,header)
        require(type(actual)is dict and set(actual)==set(header)|{'sha256'},'observation descriptor')
        same({k:v for k,v in actual.items() if k!='sha256'},header)
        block=raw[offset:offset+size];require(len(block)==size and hashlib.sha256(block).hexdigest()==actual['sha256'],'complete native array')
        output=[v for (v,) in struct.iter_unpack('<f',block)];require(all(map(math.isfinite,output)),'nonfinite native observation')
        for contract in maximum:
            target=wanted if contract=='staged_fp32' else values(data,binary,contract,name,before,count)
            ratios=[abs(v-w)/(1e-5+1e-4*abs(w)) for v,w in zip(output,target,strict=True)]
            worst=max(ratios,default=0.);maximum[contract]=max(maximum[contract],worst)
            require(worst<=1.,f'numerical gate: {contract}/{name}/{before}+{count}: {worst}')
            comparisons+=len(output)
        offset+=size
    require(events[-2][0]=='CAPTURE','capture terminal label')
    same(events[-2][1],dict(records=len(expected),bytes=offset,sha256=hashlib.sha256(raw).hexdigest(),whole_e2b_qualified=False))
    require(events[-1]==('TINY_COMPLETE',None) and len(raw)==offset,'complete binary and terminal marker')
    return dict(passed=True,records=len(expected),binary_bytes=offset,compared_values=comparisons,maximum_gate_ratio=maximum,whole_e2b_qualified=False,performance_qualified=False,phone_qualified=False)

if __name__=='__main__':
    capture,arrays,directory=map(Path,sys.argv[1:]);data,binary=fixture(directory)
    print(json.dumps(replay(records(read(capture,16*1024**2)),read(arrays,64*1024**2),data,binary)))
