"""Independent original-byte oracle and complete real resident-table replay.

Preparation precedes target execution. Exact rational BF16 reconstruction uses
an unchanged pre-target oracle; full source chunks stream through a64KiB window.
No target output participates in constructing the expected rows.
"""
from pathlib import Path
import hashlib,json,os,re,struct,sys
import check_gemma_real_projections as base
import check_gemma_table_math as mathref

N,TILE,CAP=262144,16384,32
TABLES=[('embedding','embed_tokens',1536),('ple_lookup','embed_tokens_per_layer',8960)]
ORIGINAL=[2,100,257,100]
BOUNDARY=[i*TILE+j for i in range(16) for j in (0,TILE-1)]
SELECTED=sorted(set(ORIGINAL+BOUNDARY))
TEST='metal::gemma_affine::partitioned::gather::tests::real::real_streamed_tables_remain_resident_across_lookups'
SENTINEL=0x5a170031
GUARDS=struct.pack('<7I',*[SENTINEL]*7)
same=base.same

def sha(raw):return hashlib.sha256(raw).hexdigest()
def guarded(raw):return dict(words=(len(raw)+28)//4,bytes=len(raw)+28,sha256=sha(raw+GUARDS),guards=[SENTINEL]*7)
def geometry(k):
    child=sum((n+7)*4 for n in (TILE*k//8,TILE*k//128,TILE*k//128,k,TILE,1))+240
    resident=16*child+(N+TILE)*4+112
    gather=(32+7+32*k+7+1+7)*4+2*32*k*4+160+16777216
    source=TILE*k//2+2*TILE*k//32+72
    return dict(child=child,resident=resident,gather=gather,source=source,working=resident+source)

def policy():
    g=[geometry(k) for _,_,k in TABLES]
    # Includes loading staging even though released before two gather pipelines.
    categories=dict(resident_owners=sum(x['resident'] for x in g),gather_workspaces=sum(x['gather'] for x in g),
                    source_staging=max(x['source'] for x in g),context_policy=67108864,
                    header_bytes=318228,header_and_reference_parser_policy=8388608,frozen_lookup=167936,
                    frozen_metadata_policy=131072,caller_f32=32*8960*4,serialization_policy=8388608,
                    hash_and_io_stack=131072,diagnostic_metadata_policy=8388608)
    total=sum(categories.values())
    if total>3*1024**3:raise ValueError('aggregate policy exceeds native cap')
    return dict(categories=categories,total_policy_bytes=total,native_process_allowance_bytes=3*1024**3,
                mac_working_set_ceiling_bytes=6*1024**3,exact_physical_residency=False,phone_qualified=False)

def schedules():return [ORIGINAL,BOUNDARY,list(reversed(BOUNDARY)),[N-1],[TILE]*32,ORIGINAL]

def spans(header,stem,k):
    result=[]
    for role,suffix,dtype,shape,stride in [('codes','weight','U32',[N,k//8],k//2),('scales','scales','BF16',[N,k//64],k//32),('biases','biases','BF16',[N,k//64],k//32)]:
        name='language_model.model.'+stem+'.'+suffix
        e=header[name];same(e['dtype'],dtype);same(e['shape'],shape)
        first,end=e['data_offsets']
        if type(first)is not int or type(end)is not int or not 0<=first<end<=base.MODEL_BYTES-318228 or end-first!=N*stride:raise ValueError('original tensor range differs')
        result.append(dict(role=role,name=name,dtype=dtype,shape=shape,offset=318228+first,bytes=end-first,stride=stride))
    return result

def authenticate_inputs(model,frozen):
    if base.digest(model,base.MODEL_BYTES)!=base.MODEL_SHA:raise ValueError('checkpoint identity differs')
    base.load(frozen/'observation.json',65536,'b1818331b5d341444fbc4ea22e018fc99241bd2d4fe5edbca7df9222739546a1')
    base.load(frozen/'manifest.json',65536,'6651c175ea8d90e9759ea40b8746e6c816a8c867a969b9075e5d83e50c860a95')
    reference=mathref.read(frozen/'arrays.bin',167936)
    if len(reference)!=167936 or sha(reference)!='aea89e9e83d266ae5e648e1c4bebb01509370ef29ef777dea7018b5822cbf469':raise ValueError('complete original lookup reference differs')
    return reference

def read_header(source):
    if source.read(8)!=struct.pack('<Q',318220):raise ValueError('header length differs')
    headerraw=source.read(318220)
    if sha(headerraw)!='4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7':raise ValueError('header identity differs')
    return base.strict_json(headerraw)

def build_oracle(model,frozen,*,progress=None,storage=None):
    report={} if progress is None else progress
    rawrows=bytearray() if storage is None else storage
    report.update(dict(format='toklet-gemma-real-table-oracle-v1',passed=False,primary_error=None,attempt=dict(phase='authenticate'),
                model_bytes=base.MODEL_BYTES,model_sha256=base.MODEL_SHA,
                selected_ids=SELECTED,requests=schedules(),tables=[],policy=policy(),engine_executed=False,gpu_executed=False))
    reference=authenticate_inputs(model,frozen)
    with model.open('rb') as source:
        report['attempt']=dict(phase='header')
        header=read_header(source)
        for role,stem,k in TABLES:
            ss=spans(header,stem,k);chunks=[dict(first_row=i*TILE,rows=TILE,sources={},buffers={}) for i in range(N//TILE)]
            table=dict(role=role,stem=stem,columns=k,spans=ss,chunks=chunks,geometry=geometry(k),selected_rows_completed=[])
            report['tables'].append(table)
            for span in ss:
                source.seek(span['offset'])
                for chunk in chunks:
                    report['attempt']=dict(phase='source-chunk',table=role,first_row=chunk['first_row'],tensor=span['role'],consumed_bytes=0)
                    plain=hashlib.sha256();left=TILE*span['stride']
                    while left:
                        raw=source.read(min(65536,left))
                        if not raw:raise ValueError('short original tensor')
                        left-=len(raw);plain.update(raw);report['attempt']['consumed_bytes']+=len(raw)
                    device=plain.copy();device.update(GUARDS);count=TILE*span['stride']
                    chunk['sources'][span['role']]=dict(offset=span['offset']+chunk['first_row']*span['stride'],bytes=count,sha256=plain.hexdigest())
                    chunk['buffers'][span['role']]=dict(words=count//4+7,bytes=count+28,sha256=device.hexdigest(),guards=[SENTINEL]*7)
            rowstart=len(rawrows)
            selected={}
            for row in SELECTED:
                report['attempt']=dict(phase='selected-row',table=role,row=row)
                arrays=[]
                for span in ss:
                    source.seek(span['offset']+row*span['stride']);body=source.read(span['stride'])
                    if len(body)!=span['stride']:raise ValueError('short selected original row')
                    arrays.append(body)
                codes,scales,biases=arrays;out=bytearray()
                # BF16*integer has an exact small significand; use exact Fraction
                # plus one IEEE32 rounding, preserving a negative-zero sum only
                # when both zero operands have negative signs.
                for col in range(k):
                    code=(codes[col//2]>>(4*(col%2)))&15
                    sb=struct.unpack_from('<H',scales,(col//64)*2)[0];bb=struct.unpack_from('<H',biases,(col//64)*2)[0]
                    s=mathref.rational(sb,7,8);b=mathref.rational(bb,7,8)
                    out.extend(struct.pack('<I',mathref.rounded(s*code+b,23,8,sb>>15==1 and bb==0x8000 and s*code==0)))
                selected[row]=bytes(out);rawrows.extend(out);table['selected_rows_completed'].append(row)
            external=b''.join(selected[row] for row in ORIGINAL)
            expected=reference[:len(ORIGINAL)*TABLES[0][2]*4] if role==TABLES[0][0] else reference[len(ORIGINAL)*TABLES[0][2]*4:]
            if external!=expected:raise ValueError('original full-trace lookup rows differ')
            table.update(rows_offset=rowstart,rows_bytes=len(SELECTED)*k*4,rows_sha256=sha(rawrows[rowstart:]),full_trace_values=len(external)//4)
    report['binary_bytes']=len(rawrows);report['binary_sha256']=sha(rawrows)
    report['passed']=True;report['attempt']=None
    return report,bytes(rawrows)

def prepare(model,frozen,destination):
    destination.mkdir()
    report={};storage=bytearray();primary=None
    def failed(error):
        nonlocal primary
        report['passed']=False
        if primary is None:primary=error;report['primary_error']=type(error).__name__+': '+str(error)
        else:primary.add_note('Preparation retention failed: '+str(error))
    try:
        build_oracle(model,frozen,progress=report,storage=storage)
    except BaseException as error:failed(error)
    finally:
        try:
            with (destination/'rows.bin').open('xb') as out:
                if out.write(storage)!=len(storage):raise OSError('short oracle write')
                out.flush()
        except BaseException as error:failed(error)
        report['binary_bytes']=len(storage);report['binary_sha256']=sha(storage)
        try:
            with (destination/'oracle.json').open('x') as out:json.dump(report,out,indent=2);out.write('\n')
        except BaseException as error:failed(error)
        manifest=dict(passed=primary is None,files={},file_errors=[],preparation_state=report)
        for name in ('rows.bin','oracle.json'):
            if os.path.lexists(destination/name):
                try:
                    raw=mathref.read(destination/name,2*1024**2)
                    manifest['files'][name]=dict(bytes=len(raw),sha256=sha(raw))
                except BaseException as error:failed(error);manifest['passed']=False;manifest['file_errors'].append(name+': '+str(error))
        manifest['primary_error']=report.get('primary_error');manifest['passed']=primary is None
        try:
            with (destination/'manifest.json').open('x') as out:json.dump(manifest,out,indent=2);out.write('\n')
        except BaseException as error:failed(error)
    if primary is not None:raise primary.with_traceback(primary.__traceback__)
    return dict(passed=True,tables=len(report['tables']),unique_rows=len(SELECTED),values=len(storage)//4,binary_sha256=sha(storage),full_trace_values=sum(t['full_trace_values'] for t in report['tables']),engine_executed=False,gpu_executed=False)

def parse(raw):
    if len(raw)>8*1024**2:raise ValueError('bounded native log required')
    events=[];wrapper=[]
    for line in raw.decode().splitlines():
        if len(line)>65536:raise ValueError('bounded record required')
        if 'GEMMA_TABLES_' in line:
            prefix,tail=line.split('GEMMA_TABLES_',1)
            if prefix not in ('','test '+TEST+' ... '):raise ValueError('unexpected test prefix')
            label,body=tail.split(' ',1);events.append((label,base.strict_json(body)))
        elif line.strip():wrapper.append(line)
    if len(wrapper)!=3 or wrapper[:2]!=['running 1 test','ok'] or not re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s',wrapper[2]):raise ValueError('exact successful test wrapper required')
    return events

def compare_oracle(folder,expected,raw):
    same(expected['binary_bytes'],len(raw));same(expected['binary_sha256'],sha(raw))
    same(base.load(folder/'oracle.json',1024**2),expected)
    if mathref.read(folder/'rows.bin',2*1024**2)!=raw:raise ValueError('immutable pre-capture row oracle differs')

def verify(events,binary,oracle,rowbytes):
    same(oracle['format'],'toklet-gemma-real-table-oracle-v1');same(oracle['selected_ids'],SELECTED)
    same(oracle['requests'],schedules());same(oracle['policy'],policy())
    same(oracle['engine_executed'],False);same(oracle['gpu_executed'],False)
    same(oracle['model_bytes'],base.MODEL_BYTES);same(oracle['model_sha256'],base.MODEL_SHA)
    same(len(rowbytes),oracle['binary_bytes']);same(sha(rowbytes),oracle['binary_sha256'])
    if len(binary)>64*1024**2:raise ValueError('bounded raw capture required')
    rows=iter(events);allocations=set();offset=0;arrays=0
    def take(label,expected=None):
        try:actual,item=next(rows)
        except StopIteration as error:raise ValueError('missing '+label) from error
        if actual!=label:raise ValueError('event order differs '+actual+'/'+label)
        if expected is not None:same(item,expected)
        return item
    def enroll(value):
        if type(value)is not str or not re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}',value) or value in allocations:raise ValueError('distinct live allocations required')
        allocations.add(value);return value
    def array(key,role,expected):
        nonlocal offset,arrays
        item=take('ARRAY_ATTEMPT');take('ARRAY',item)
        raw=binary[offset:offset+len(expected)]
        same(item,dict(key=key,role=role,offset=offset,bytes=len(expected),words=len(expected)//4,sha256=sha(raw)))
        if raw!=expected:raise ValueError('complete '+role+' bytes differ')
        offset+=len(expected);arrays+=1;return item
    take('PLAN',policy())
    meta=take('META');device=meta.get('device')
    if type(device)is not str or not 1<=len(device)<=256:raise ValueError('named hardware required')
    same(meta,dict(device=device,owner_struct_bytes=112,child_struct_bytes=240,gather_struct_bytes=160,
                   header_capacity=318220,reference_capacity=167936,binary_cap_bytes=64*1024**2,whole_graph_qualified=False,performance_qualified=False))
    states={};workspaces={};selected={}
    for table,(role,stem,k) in zip(oracle['tables'],TABLES,strict=True):
        same(table['role'],role);same(table['stem'],stem);same(table['columns'],k);same(table['geometry'],geometry(k))
        body=rowbytes[table['rows_offset']:table['rows_offset']+table['rows_bytes']]
        same(len(body),len(SELECTED)*k*4);same(sha(body),table['rows_sha256'])
        selected[role]={row:body[i*k*4:(i+1)*k*4] for i,row in enumerate(SELECTED)}
        g=geometry(k)
        take('UPLOAD_ATTEMPT',dict(role=role,columns=k,maximum_rows=TILE,capacity=1,working_budget=g['working'],spans=table['spans']))
        pointers=None
        for index,chunk in enumerate(table['chunks']):
            same(chunk['first_row'],index*TILE);same(chunk['rows'],TILE)
            attempt=take('FILL_ATTEMPT');addresses=attempt.get('addresses');lengths=[TILE*k//2,TILE*k//32,TILE*k//32]
            if type(addresses)is not list or len(addresses)!=3:raise ValueError('three source arrays required')
            if pointers is None:
                intervals=[]
                for address,length in zip(addresses,lengths,strict=True):
                    if type(address)is not str or not re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}',address):raise ValueError('source address differs')
                    start=int(address,16);end=start+length
                    if end>2**64:raise ValueError('source interval overflows')
                    if any(start<e and a<end for a,e in intervals):raise ValueError('source arrays overlap')
                    intervals.append((start,end))
                pointers=addresses
            same(attempt,dict(role=role,first_row=index*TILE,rows=TILE,addresses=pointers,lengths=lengths))
            take('FILL',dict(role=role,first_row=index*TILE,consumed=[chunk['sources'][name] for name in ('codes','scales','biases')]))
        same(len(table['chunks']),16)
        take('UPLOAD_OUTCOME',dict(role=role,completed=True,error=None))
        cold=take('MATRIX');actual=cold['observation'];children=[]
        for index,chunk in enumerate(table['chunks']):
            buffers=dict(chunk['buffers'])
            for name,size in [('input',k),('output',TILE),('status',1)]:
                raw=bytes((size+7)*4);buffers[name]=dict(words=size+7,bytes=len(raw),sha256=sha(raw),guards=[0]*7)
            for name,identity in buffers.items():buffers[name]=dict(identity,allocation=enroll(actual['children'][index]['device'][name]['allocation']))
            children.append(dict(rows=TILE,owned_bytes=g['child'],poisoned=False,pending=False,shared_read_safe=True,device=buffers))
        state=dict(rows=N,columns=k,capacity=1,owned_bytes=g['resident'],poisoned=False,owner_slots=16,
                   full_scratch_capacity=N,chunk_scratch_capacity=TILE,staged_sha256=sha(bytes(N*4)),chunk_scratch_sha256=sha(bytes(TILE*4)),children=children)
        same(cold,dict(role=role,source_cpu_bytes_retained=0,observation=state));states[role]=state
    for role,_,k in TABLES:
        take('GATHER_ATTEMPT',dict(role=role,columns=k,capacity=CAP,budget=geometry(k)['gather']))
        take('GATHER_OUTCOME',dict(role=role,completed=True,error=None))
    identities={}
    def workspace(role,key,caller,staged,chunk,ids,out,status,*,cold=False):
        k=next(k for r,_,k in TABLES if r==role)
        descriptor=take('SNAPSHOT')
        addresses=[d['identity']['allocation'] for d in descriptor['device']]
        if cold:identities[role]=[enroll(address) for address in addresses]
        same(addresses,identities[role])
        hostmeta=[dict(role=name,words=len(raw)//4,sha256=sha(raw)) for name,raw in [('caller',caller),('staged',staged),('chunk',chunk)]]
        devmeta=[]
        for (name,raw),address in zip([('ids',ids),('output',out),('status',status)],addresses,strict=True):
            devmeta.append(dict(role=name,identity=dict(words=len(raw)//4,bytes=len(raw),sha256=sha(raw),guards=list(struct.unpack('<7I',raw[-28:])),allocation=address)))
        same(descriptor,dict(key=key,columns=k,capacity=CAP,admitted_bytes=geometry(k)['gather'],poisoned=False,pending=False,shared_read_safe=True,
                             full_scratch_capacity=CAP*k,chunk_scratch_capacity=CAP*k,host=hostmeta,device=devmeta))
        host=[array(key,name,raw) for name,raw in [('caller',caller),('staged',staged),('chunk',chunk)]]
        device=[]
        for name,raw in [('ids',ids),('output',out),('status',status)]:device.append(dict(role=name,array=array(key,name,raw)))
        # Allocations occur in the enclosing snapshot after all raw arrays.
        return dict(columns=k,capacity=CAP,admitted_bytes=geometry(k)['gather'],poisoned=False,pending=False,shared_read_safe=True,
                    full_scratch_capacity=CAP*k,chunk_scratch_capacity=CAP*k,arrays=host,device=device)
    expected={}
    for role,_,k in TABLES:
        expected[role]=workspace(role,role+'/cold',b'',bytes(CAP*k*4),bytes(CAP*k*4),bytes((CAP+7)*4),bytes((CAP*k+7)*4),bytes(32),cold=True)
        workspaces[role]=dict(staged=bytearray(CAP*k*4),chunk=bytearray(CAP*k*4))
    cold=take('WORKSPACES')
    for role in expected:
        for exp,address in zip(expected[role]['device'],identities[role],strict=True):exp['allocation']=address
    same(cold,expected)
    compared=0
    for index,request in enumerate(schedules()):
        for role,_,k in TABLES:
            key=f'{role}/{index}';data=workspaces[role];last=None
            for child in range(16):
                slots=[(slot,row) for slot,row in enumerate(request) if child*TILE<=row<(child+1)*TILE]
                if not slots:continue
                values=b''.join(selected[role][row] for _,row in slots)
                data['chunk'][:len(values)]=values
                for slot,row in slots:data['staged'][slot*k*4:(slot+1)*k*4]=selected[role][row]
                local=[row-child*TILE for _,row in slots]
                last=(struct.pack('<'+'I'*CAP,*local,*([SENTINEL]*(CAP-len(local))))+GUARDS,
                      values+struct.pack('<I',SENTINEL)*(CAP*k-len(slots)*k+7),struct.pack('<I',0)+GUARDS)
            caller=b''.join(selected[role][row] for row in request);compared+=len(caller)//4
            take('STEP_ATTEMPT',dict(key=key,role=role,index=index,ids=request,caller_capacity=len(request)*k))
            take('STEP_OUTCOME',dict(key=key,completed=True,error=None,poisoned=False))
            state=workspace(role,key,caller,data['staged'],data['chunk'],*last)
            for d,address in zip(state['device'],identities[role],strict=True):d['allocation']=address
            take('STEP',dict(key=key,observation=state,workspace_output_contract='last executed child; inactive tail sentinel; CPU chunk tail may be stale'))
    for role,_,_ in TABLES:take('MATRIX_FINAL',dict(role=role,observation=states[role]))
    result=dict(tables=2,chunks=32,callbacks=32,steps=12,binary_bytes=offset,actual_table_lookup_qualified=True,whole_graph_qualified=False,
                source_or_precision_failures_waived=False,phone_qualified=False,performance_qualified=False)
    take('RESULT',result)
    try:next(rows)
    except StopIteration:pass
    else:raise ValueError('unexpected extra event')
    same(offset,len(binary))
    return dict(passed=True,**result,arrays=arrays,compared_values=compared,full_trace_values=41984,
                aggregate_workload_policy_bytes=policy()['total_policy_bytes'],live_device_allocations=len(allocations),source_cpu_table_retained=False)

if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='prepare':
        print(json.dumps(prepare(Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]))))
    elif len(sys.argv)==7 and sys.argv[1]=='replay':
        log,binary,model,frozen,folder=map(Path,sys.argv[2:])
        expected,raw=build_oracle(model,frozen)
        compare_oracle(folder,expected,raw)
        print(json.dumps(verify(parse(mathref.read(log,8*1024**2)),mathref.read(binary,64*1024**2),expected,raw)))
    else:raise SystemExit('prepare MODEL FROZEN NEW_ORACLE_DIR | replay LOG BINARY MODEL FROZEN ORACLE_DIR')
