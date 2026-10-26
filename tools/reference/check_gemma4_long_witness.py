"""Stdlib planning/replay of the long tiny full-sequence oracle; no graph import."""
from array import array
from decimal import Decimal, localcontext
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
import check_gemma_hybrid_tiny as safe

ROOT = Path(__file__).resolve().parents[2]
CAPACITY = 1025
LEDGER_MIB = dict(runtime_blas=320, tile_conversion=64, active_graph=64,
                  capture=16, comparison=16, metadata_weights_margin=32)
SCOPE = ('incremental_reference_implemented', 'native_engine_executed',
         'real_e2b_boundary_qualified', 'quality_scored', 'model_fit_qualified',
         'phone_qualified', 'full_model_window_qualified')
POSITIONS = [0,1,2,3,4,7,8,16,17,31,32,510,511,512,513,1023,1024]


def digest(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def plan():
    old, _ = safe.fixture(ROOT/'tests/fixtures/gemma-packed-graph-v1')
    ids = old['token_ids'] + [(position*7+3)%17 for position in range(32, CAPACITY)]
    tokens = dict(original=ids, future=ids[:-1]+[(ids[-1]+1)%17])
    captures = []
    for sequence in tokens:
        cursor = 0
        for contract in ('math','staged_fp32'):
            for name, item in old[contract].items():
                axis = 2 if name.endswith(('rotary','context')) else 1
                shape = list(item['shape']); safe.same(shape[axis], 32); shape[axis]=CAPACITY
                count = math.prod(shape)*(8 if contract=='math' else 4)
                captures.append(dict(sequence=sequence,contract=contract,name=name,
                                     shape=shape,dtype=item['dtype'],offset=cursor,bytes=count))
                cursor += count
        safe.same(cursor,109051800)
    safe.same(sum(LEDGER_MIB.values()),512)
    return dict(format='toklet-gemma-long-witness-plan-v1',tokens=tokens,captures=captures,
                file_bytes=109051800,total_array_bytes=218103600,capacity=CAPACITY,
                selected_positions=POSITIONS,gate=dict(atol=1e-5,rtol=1e-4),
                model_window=4,projection_chunk_rows=32,memory_ledger_mib=LEDGER_MIB,
                **{key:False for key in SCOPE})


def load_plan(path, sha):
    raw=safe.read(path,1024**2);safe.require(digest(raw)==sha,'external long plan differs')
    result=safe.document(raw);safe.same(result,plan());return result


def state(directory, item):
    path=directory/(item['sequence']+'.bin')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb')as file:
        file.seek(item['offset']);raw=file.read(item['bytes'])
    safe.require(len(raw)==item['bytes']and digest(raw)==item['sha256'],'saved long state differs')
    result=array('d'if item['dtype']=='<f8'else'f');result.frombytes(raw)
    if sys.byteorder!='little':result.byteswap()
    safe.require(all(math.isfinite(v)for v in result),'finite long state required')
    return result


def rows(values,item,positions):
    shape=item['shape'];axis=2 if item['name'].endswith(('rotary','context'))else 1
    stride=math.prod(shape[axis+1:]);heads=math.prod(shape[:axis]);n=shape[axis]
    return array(values.typecode,(values[(head*n+position)*stride+feature]
                  for head in range(heads)for position in positions for feature in range(stride)))


def comparison(actual, expected):
    safe.same(len(actual),len(expected));failed=0;maximum=0.
    for a,b in zip(actual,expected,strict=True):
        fraction=abs(a-b)/(1e-5+1e-4*abs(b));maximum=max(maximum,fraction);failed+=fraction>1.
    return dict(values=len(actual),failed_values=failed,maximum_gate_fraction=maximum,passed=failed==0)


def boundaries(directory, entries, old, binary, report):
    report['boundaries']=[]
    def check(label, actual, expected):
        result=dict(label=label,**comparison(actual,expected));report['boundaries'].append(result)
        safe.require(result['passed'],'independent long boundary differs: '+label)
    for sequence,ids in plan()['tokens'].items():
        for contract in ('math','staged_fp32'):
            def rounded(x):return struct.unpack('<f',struct.pack('<f',x))[0]if contract=='staged_fp32'else x
            def get(name):return state(directory,entries[(sequence,contract,name)])
            label=sequence+'/'+contract+'/'
            for name,tensor,width in [('embedding','model.embed_tokens.weight',64),
                                      ('ple_lookup','model.embed_tokens_per_layer.weight',256)]:
                item=old['weights'][tensor];body=binary[item['offset']:item['offset']+item['bytes']]
                safe.require(digest(body)==item['sha256'],'frozen long source row differs')
                weights=array('f');weights.frombytes(body)
                if sys.byteorder!='little':weights.byteswap()
                expected=array('d',(weights[token*width+column]for token in ids for column in range(width)))
                check(label+name,get(name),expected)
                if name=='embedding':
                    check(label+'scaled_embedding',get('scaled_embedding'),array('d',(rounded(v*rounded(math.sqrt(64)))for v in expected)))
            for layer in range(4):
                full=layer%2==1;dims=128 if full else 64;heads=2
                norm=get(f'layer.{layer}.self_attn.q_norm');rotated=get(f'layer.{layer}.q_rotary')
                actual=[];expected=[]
                with localcontext()as context:
                    context.prec=80
                    for head in range(heads):
                        for position in (511,512,1024):
                            row=[norm[(position*heads+head)*dims+column]for column in range(dims)]
                            for pair in range((32 if full else dims)//2):
                                frequency=Decimal(1000000 if full else 10000)**(Decimal(2*pair)/Decimal(dims))
                                angle=float(Decimal(position)/frequency)
                                cosine=rounded(math.cos(angle));sine=rounded(math.sin(angle))
                                a,b=row[pair],row[pair+dims//2]
                                row[pair]=rounded(rounded(a*cosine)-rounded(b*sine))
                                row[pair+dims//2]=rounded(rounded(a*sine)+rounded(b*cosine))
                            expected.extend(row)
                            actual.extend(rotated[(head*CAPACITY+position)*dims:(head*CAPACITY+position+1)*dims])
                check(label+f'layer.{layer}.late_rotary',actual,expected)
                producer=1 if full else 0
                keys=get(f'layer.{producer}.k_rotary');values=get(f'layer.{producer}.self_attn.v_norm')
                target=get(f'layer.{layer}.context');actual=[];expected=[]
                for head in range(heads):
                    for position in (511,512,1024):
                        start=0 if full else max(0,position-3)
                        query=rotated[(head*CAPACITY+position)*dims:(head*CAPACITY+position+1)*dims]
                        scores=[rounded(math.fsum(float(query[d])*float(keys[key*dims+d])for d in range(dims)))for key in range(start,position+1)]
                        maximum=max(scores);exponents=[math.exp(score-maximum)for score in scores];total=math.fsum(exponents)
                        probabilities=[rounded(value/total)for value in exponents]
                        expected.extend(rounded(math.fsum(probabilities[key-start]*float(values[key*dims+d])for key in range(start,position+1)))for d in range(dims))
                        actual.extend(target[(head*CAPACITY+position)*dims:(head*CAPACITY+position+1)*dims])
                check(label+f'layer.{layer}.late_attention',actual,expected)
            expected=array('d',(rounded(rounded(math.tanh(rounded(v/30.)))*30.)for v in get('head')))
            check(label+'softcap',get('logits'),expected)
    safe.same(len(report['boundaries']),48)
    report['independent_selected_boundaries_passed']=True


def identity(path, cap):
    def stamp(s): return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb')as file:
        before=os.fstat(file.fileno())
        safe.require(stat.S_ISREG(before.st_mode)and before.st_size<=cap,'bounded regular long input required')
        hasher=hashlib.sha256();count=0
        while True:
            body=file.read(min(65536,cap-count+1))
            if not body:break
            count+=len(body);safe.require(count<=cap,'long input grew');hasher.update(body)
        safe.same(count,before.st_size)
        safe.same(stamp(os.fstat(file.fileno())),stamp(before))
        safe.same(stamp(path.lstat()),stamp(before))
    return dict(bytes=count,sha256=hasher.hexdigest())


def verify(directory, specification, manifest_sha, report):
    raw=safe.read(directory/'observation.json',1024**2)
    safe.require(digest(raw)==manifest_sha,'external long capture differs')
    capture=safe.document(raw)
    safe.same(set(capture),{'format','passed','primary_error','endpoint_errors','arrays','files',
                           'runtime_admission','runtime_endpoint','memory_ledger_mib','tokens',*SCOPE})
    safe.same(capture['format'],'toklet-gemma-long-witness-capture-v1')
    safe.same(capture['passed'],True);safe.same(capture['primary_error'],None);safe.same(capture['endpoint_errors'],[])
    safe.same(capture['memory_ledger_mib'],LEDGER_MIB);safe.same(capture['tokens'],specification['tokens'])
    for key in SCOPE:safe.same(capture[key],False)
    safe.same(len(capture['arrays']),360)
    for sequence in specification['tokens']:
        actual=identity(directory/(sequence+'.bin'),specification['file_bytes'])
        safe.same(actual,capture['files'][sequence]);safe.same(actual['bytes'],specification['file_bytes'])
    safe.same(set(capture['files']),set(specification['tokens']))
    entries={}
    for actual,wanted in zip(capture['arrays'],specification['captures'],strict=True):
        safe.same(set(actual),set(wanted)|{'sha256','captured'})
        safe.same({key:actual[key]for key in wanted},wanted);safe.same(actual['captured'],True)
        safe.require(type(actual['sha256'])is str and len(actual['sha256'])==64,'state digest required')
        entries[(actual['sequence'],actual['contract'],actual['name'])]=actual
    old,binary=safe.fixture(ROOT/'tests/fixtures/gemma-packed-graph-v1')
    report.update(short_prefix=[],future_prefix=[],cross_precision=[],selected_states=[])
    for contract in ('math','staged_fp32'):
        for name in old[contract]:
            entry=entries[('original',contract,name)];original=state(directory,entry)
            future=state(directory,entries[('future',contract,name)])
            expected=safe.values(old,binary,contract,name,0,32)
            for sequence,values in [('original',original),('future',future)]:
                checked=comparison(rows(values,entry,range(32)),expected)
                report['short_prefix'].append(dict(sequence=sequence,contract=contract,name=name,**checked))
                safe.require(checked['passed'],'frozen short long-witness prefix differs')
            checked=comparison(rows(future,entry,range(CAPACITY-1)),rows(original,entry,range(CAPACITY-1)))
            report['future_prefix'].append(dict(contract=contract,name=name,**checked))
            safe.require(checked['passed'],'future token changed long causal prefix')
            selected=rows(original,entry,POSITIONS)
            report['selected_states'].append(dict(contract=contract,name=name,values=len(selected),
                                                  sha256=digest(selected.tobytes())))
    for sequence in ('original','future'):
        for name in old['math']:
            ideal=state(directory,entries[(sequence,'math',name)])
            staged=state(directory,entries[(sequence,'staged_fp32',name)])
            report['cross_precision'].append(dict(sequence=sequence,name=name,**comparison(staged,ideal)))
    boundaries(directory,entries,old,binary,report)
    for sequence in specification['tokens']:
        safe.same(identity(directory/(sequence+'.bin'),specification['file_bytes']),capture['files'][sequence])
    safe.require(digest(safe.read(directory/'observation.json',1024**2))==manifest_sha,'long capture endpoint differs')
    report.update(passed=True,complete_states_checked=360,short_prefix_comparisons=360,
                  causal_prefix_comparisons=180,**{key:False for key in SCOPE})
    return entries


def freeze(directory, entries, destination, report):
    destination.mkdir();descriptors=[];cursor=0
    complete={'logits','layer.0.k_rotary','layer.0.self_attn.v_norm',
              'layer.1.k_rotary','layer.1.self_attn.v_norm'}
    with(destination/'arrays.bin').open('xb')as file:
        for key,item in entries.items():
            positions=list(range(CAPACITY))if item['name']in complete else POSITIONS
            # Future full logits and last-position intermediate states suffice;
            # its unchanged complete causal prefix is separately replayed.
            if item['sequence']=='future'and item['name']!='logits':positions=[CAPACITY-1]
            selected=rows(state(directory,item),item,positions)
            if sys.byteorder!='little':selected.byteswap()
            raw=selected.tobytes();safe.require(cursor+len(raw)<=8*1024**2,'bounded compact long fixture')
            descriptor={k:item[k]for k in('sequence','contract','name','dtype')}
            descriptor.update(positions=positions,source_shape=item['shape'],offset=cursor,bytes=len(raw),sha256=digest(raw))
            safe.same(file.write(raw),len(raw));file.flush();cursor+=len(raw);descriptors.append(descriptor)
    binary=identity(destination/'arrays.bin',8*1024**2)
    manifest=dict(format='toklet-gemma-long-witness-v1',capacity=CAPACITY,model_window=4,
                  selected_positions=POSITIONS,tokens=plan()['tokens'],arrays=descriptors,binary=binary,
                  preparation_passed=True,complete_full_sequence_capture_replayed=True,
                  short_prefix_comparisons=report['short_prefix_comparisons'],
                  causal_prefix_comparisons=report['causal_prefix_comparisons'],
                  cross_precision_failed_values=sum(v['failed_values']for v in report['cross_precision']),
                  **{key:False for key in SCOPE})
    with(destination/'manifest.json').open('xb')as file:file.write(encoded(manifest))
    return dict(manifest=identity(destination/'manifest.json',1024**2),binary=binary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',type=Path)
    parser.add_argument('--plan',type=Path);parser.add_argument('--plan-sha')
    parser.add_argument('--capture',type=Path);parser.add_argument('--manifest-sha')
    parser.add_argument('--freeze',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=dict(passed=False,primary_error=None)
    try:
        if args.prepare:
            with args.prepare.open('xb')as file:file.write(encoded(plan()))
            result.update(passed=True,plan=identity(args.prepare,1024**2))
        else:
            specification=load_plan(args.plan,args.plan_sha)
            entries=verify(args.capture,specification,args.manifest_sha,result)
            if args.freeze:result['fixture']=freeze(args.capture,entries,args.freeze,result)
    except BaseException as error:result.update(passed=False,primary_error=type(error).__name__+': '+str(error))
    finally:
        with args.output.open('xb')as file:file.write(encoded(result))
    print(json.dumps({key:result[key]for key in('passed','primary_error')}))
    sys.exit(0 if result['passed']else 1)
