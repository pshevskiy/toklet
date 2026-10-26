"""Replay real immutable GPU weights using the pre-existing complete byte oracle.

The earlier oracle retains its original Affine policy metadata as historical
identity; only its authenticated source/row data inform the new numerical gate.
New immutable-owner and two-Context admission are reconstructed separately.
"""
from pathlib import Path
import json,re,struct,sys
import check_gemma_real_tables as original
base=original.base
mathref=original.mathref
same=original.same
sha=original.sha
N,TILE,CAP=original.N,original.TILE,original.CAP
TABLES,SELECTED=original.TABLES,original.SELECTED
schedules=original.schedules
GUARDS=original.GUARDS
SENTINEL=original.SENTINEL
TEST='metal::gemma_affine::partitioned::gather::tests::real::immutable::immutable_real_tables_outlive_context_and_share_lookup_workspaces'

def geometry(k):
    source=original.geometry(k)['source']
    resident=N*k//2+2*N*k//32+16*3*28+16*112+112
    return dict(resident=resident,gather=original.geometry(k)['gather'],source=source,working=resident+source)

def policy():
    categories=dict(original.policy()['categories'])
    del categories['context_policy']
    categories['two_context_policies']=2*67108864
    categories['resident_owners']=sum(geometry(k)['resident'] for _,_,k in TABLES)
    total=sum(categories.values())
    if total>3*1024**3:raise ValueError('aggregate immutable workload exceeds process cap')
    return dict(categories=categories,total_policy_bytes=total,native_process_allowance_bytes=3*1024**3,
                mac_working_set_ceiling_bytes=6*1024**3,exact_physical_residency=False,phone_qualified=False)

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

def verify(events,binary,oracle,rowbytes):
    same(oracle['format'],'toklet-gemma-real-table-oracle-v1');same(oracle['selected_ids'],SELECTED)
    same(oracle['requests'],schedules());same(oracle['policy'],original.policy())
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
    device_identity=meta.get('device_identity')
    if type(device_identity)is not str or not re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}',device_identity):raise ValueError('retained device identity required')
    same(meta,dict(device=device,device_identity=device_identity,owner_struct_bytes=112,chunk_struct_bytes=112,gather_struct_bytes=160,
                   header_capacity=318220,reference_capacity=167936,binary_cap_bytes=64*1024**2,whole_graph_qualified=False,performance_qualified=False))
    states={};workspaces={};selected={}
    for table,(role,stem,k) in zip(oracle['tables'],TABLES,strict=True):
        same(table['role'],role);same(table['stem'],stem);same(table['columns'],k);same(table['geometry'],original.geometry(k))
        body=rowbytes[table['rows_offset']:table['rows_offset']+table['rows_bytes']]
        same(len(body),len(SELECTED)*k*4);same(sha(body),table['rows_sha256'])
        selected[role]={row:body[i*k*4:(i+1)*k*4] for i,row in enumerate(SELECTED)}
        g=geometry(k)
        take('UPLOAD_ATTEMPT',dict(role=role,columns=k,maximum_rows=TILE,working_budget=g['working'],spans=table['spans']))
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
        cold=take('IMMUTABLE');actual=cold['observation'];children=[]
        for index,chunk in enumerate(table['chunks']):
            buffers=dict(chunk['buffers'])
            for name,identity in buffers.items():buffers[name]=dict(identity,allocation=enroll(actual['children'][index][name]['allocation']))
            children.append(dict(rows=TILE,**buffers))
        state=dict(rows=N,columns=k,owned_bytes=g['resident'],owner_slots=16,chunk_struct_bytes=112,device=device_identity,children=children)
        same(cold,dict(role=role,source_cpu_bytes_retained=0,observation=state));states[role]=state
    take('CONSTRUCTION_CONTEXT_DROPPED',dict(weights=2,devices=[device_identity,device_identity]))
    take('LOOKUP_CONTEXT',dict(device=device,device_identity=device_identity,same_devices=[True,True]))
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
    for role,_,_ in TABLES:take('IMMUTABLE_FINAL',dict(role=role,observation=states[role]))
    result=dict(tables=2,chunks=32,callbacks=32,steps=12,binary_bytes=offset,construction_context_dropped=True,actual_table_lookup_qualified=True,whole_graph_qualified=False,
                source_or_precision_failures_waived=False,foreign_device_qualified=False,phone_qualified=False,performance_qualified=False)
    take('RESULT',result)
    try:next(rows)
    except StopIteration:pass
    else:raise ValueError('unexpected extra event')
    same(offset,len(binary))
    return dict(passed=True,**result,arrays=arrays,compared_values=compared,full_trace_values=41984,
                aggregate_workload_policy_bytes=policy()['total_policy_bytes'],live_device_allocations=len(allocations),source_cpu_table_retained=False)

if __name__=='__main__':
    if len(sys.argv)!=7 or sys.argv[1]!='replay':raise SystemExit('replay LOG BINARY MODEL FROZEN ORIGINAL_ORACLE_DIR')
    log,binary,model,frozen,folder=map(Path,sys.argv[2:])
    expected,raw=original.build_oracle(model,frozen)
    original.compare_oracle(folder,expected,raw)
    print(json.dumps(verify(parse(mathref.read(log,8*1024**2)),mathref.read(binary,64*1024**2),expected,raw)))
