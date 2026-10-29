"""Independent stdlib raw causal Conv3 Half reference; no target/graph imports.

Generic word/serialization helpers only are reused. MAIN has three separately
rounded FP32 products, two FP32 adds and one Half store. Old history is immutable;
returned candidate is a bit-exact raw-input suffix, not a publication operation.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

_path = Path(__file__).with_name('gemma3_half_rms.py')
_spec = importlib.util.spec_from_file_location('_conv3_word_helpers', _path)
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
f32, half_word, half_value = helper.f32, helper.half_word, helper.half_value
float_word, float_value = helper.float_word, helper.float_value
encoded, identity = helper.encoded, helper.identity
FORMAT = 'toklet-causal-conv3-half-v1'
PROTOCOL = '251197ef28690222f29cc1e2315a5c2fa9a245a8efc1da02daf5a41de09b11b3'
POLICY = dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3, fatal_reference='main')
CASES = (('hand',5,5,'hand'),('tail1',1,5,'seed'),('tail2',2,5,'seed'),
         ('tail3',3,5,'seed'),('tail32',32,5,'seed'),('tiny1',1,256,'seed'),
         ('tiny32',32,256,'seed'),('real2',2,1024,'seed'),('real3',3,1024,'seed'),
         ('real32',32,1024,'seed'),('stores',1,5,'stores'),('zero_coeff',3,5,'zero'))
GUARD, INPUT_PADDING, CANARY, COEFFICIENT_GUARD = 7, 0x7e55, 0x5a17, 0x5a170031
MAX_BINARY, MAX_JSON = 2*1024**2, 512*1024
HAND_OUTPUT = [317,38,1,9,-6, 631,-25,2,0,-12, 1262,15,3,6,-2,
               2126,-34,4,8,0, 3212,-10,5,10,2]
STORE_WORDS = [0x3c01,0x0002,0x0001,0x8001,0x8000]


def shape(rows, width):
    if type(rows) is not int or type(width) is not int or not 1 <= rows <= 32 or width not in (5,256,1024):
        raise ValueError('Conv3 shape profile')


def coefficients(words):
    values = [float_value(w << 16) for w in words]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite BF16 coefficient')
    return values


def operands(rows, width, pattern):
    shape(rows,width)
    if pattern not in ('hand','seed','stores','zero'):
        raise ValueError('Fixed operand pattern')
    stride = width+3
    u = [INPUT_PADDING]*(rows*stride+GUARD)
    old = [half_word(((13*r+5*c)%29-14)/8) for r in range(2) for c in range(width)]
    gamma = [(0x3f80,0xbe80,0x3f01,0xbf40,0x3d80,0x3980)[(3*c+t)%6]
             for c in range(width) for t in range(3)]
    for r in range(rows):
        for c in range(width):u[r*stride+c]=half_word(((17*r+7*c)%33-16)/8)
    if pattern == 'hand':
        if (rows,width)!=(5,5):raise ValueError('Hand geometry')
        old = list(map(half_word,[7,13,3,2,-2,11,-7,5,4,-4]))
        gamma = [float_word(float(v)) >> 16 for v in [1,10,100,3,-2,5,0,0,1,-1,2,-3,2,0,-1]]
        raw = [2,-3,1,-1,2,6,-2,2,-2,4,12,4,3,-3,6,20,-4,4,-4,8,30,-6,5,-5,10]
        for r in range(rows):u[r*stride:r*stride+width]=list(map(half_word,raw[r*width:(r+1)*width]))
    elif pattern == 'stores':
        if (rows,width)!=(1,5):raise ValueError('Store geometry')
        old = list(map(half_word,[1,1,0,0,-0.0,2**-11,1,0,0,-0.0]))
        u[:5] = list(map(half_word,[2**-11,1,2**-24,-2**-24,-0.0]))
        gamma = [0x3f80]*3+[0x3300]*3+[0,0,0x3f80]*2+[0x3f80]*3
    elif pattern == 'zero':gamma=[0]*(width*3)
    return u,old+[CANARY]*GUARD,gamma,coefficients(gamma),stride


def checked_f32(value):
    value = f32(value)
    if not math.isfinite(value):raise ValueError('Nonfinite FP32 intermediate')
    return value


def conv(u, old, scales, rows, width, stride, profile):
    shape(rows,width)
    if profile not in ('main','wide') or type(stride) is not int or stride < width:
        raise ValueError('Conv3 profile/stride')
    if len(u)<(rows-1)*stride+width or len(old)<2*width or len(scales)!=3*width:
        raise ValueError('Conv3 extent')
    # Prevalidate every active operand before numerical evaluation. Padding and
    # guards are deliberately outside the numerical domain.
    if any(not math.isfinite(half_value(old[i])) for i in range(2*width)) or any(
            not math.isfinite(half_value(u[r*stride+c])) for r in range(rows) for c in range(width)):
        raise ValueError('Nonfinite active Half')
    if any(not math.isfinite(v) or f32(v)!=v for v in scales):
        raise ValueError('Nonfinite/non-FP32 coefficient')
    output=[]
    for r in range(rows):
        for c in range(width):
            x=[half_value(old[(r+t)*width+c] if r+t<2 else u[(r+t-2)*stride+c]) for t in range(3)]
            k=scales[3*c:3*c+3]
            p=[checked_f32(x[t]*k[t]) for t in range(3)]
            main=checked_f32(checked_f32(p[0]+p[1])+p[2])
            main_word=half_word(main)  # Shared domain includes finite Half store.
            if profile=='main':output.append(main_word)
            else:
                # Explicit ordered FP64 operations, NOT fsum or reassociation.
                a=x[0]*k[0];b=x[1]*k[1];d=x[2]*k[2]
                output.append(half_word((a+b)+d))
    candidate=(list(old[width:2*width])+list(u[:width]) if rows==1 else
               list(u[(rows-2)*stride:(rows-2)*stride+width])+list(u[(rows-1)*stride:(rows-1)*stride+width]))
    return output,candidate


def partition(rows, maximum):
    result=[]
    while rows:
        count=min(rows,maximum);result.append(count);rows-=count
    return result


def partition_checks(u, old, scales, rows, width, stride, main):
    schedules=[partition(rows,n) for n in (1,2,3,7,32)]
    schedules=list(dict.fromkeys(tuple(s) for s in schedules))
    records=[]
    for schedule in schedules:
        state=list(old[:2*width]);combined=[];at=0;boundaries=[]
        for count in schedule:
            before=state[:]
            values,state=conv(u[at*stride:],state,scales,count,width,stride,'main')
            # Independent absolute raw suffix; no call to the candidate routine.
            prefix=list(old[:2*width])
            for r in range(at+count):prefix.extend(u[r*stride:r*stride+width])
            if state!=prefix[-2*width:]:raise ValueError('Absolute candidate suffix')
            combined.extend(values);at+=count
            boundaries.append(dict(end=at,before=identity(struct.pack('<'+'H'*len(before),*before)),
                                   after=identity(struct.pack('<'+'H'*len(state),*state))))
        if combined!=main:raise ValueError('Partition output words')
        records.append(dict(chunks=list(schedule),boundaries=boundaries))
    return records


def errors():
    u,old,gamma,_,stride=operands(1,5,'stores')
    result=[]
    for owner,index in [('u',0),('old',0)]:
        for name,word in [('nan',0x7e00),('infinity',0x7c00)]:
            a,b,g=u[:],old[:],gamma[:]
            (a if owner=='u' else b)[index]=word
            result.append(dict(name=owner+'_'+name,u_words=a,old_words=b,gamma_words=g))
    for name,word in [('coefficient_nan',0x7fc0),('coefficient_infinity',0x7f80)]:
        g=gamma[:];g[0]=word
        result.append(dict(name=name,u_words=u,old_words=old,gamma_words=g))
    a,b,g=u[:],old[:],gamma[:];b[0]=half_word(65504.);g[0]=0x7f7f
    result.append(dict(name='product_fp32_overflow',u_words=a,old_words=b,gamma_words=g))
    a,b,g=u[:],old[:],gamma[:];b[0]=b[5]=a[0]=half_word(1.);g[:3]=[0x7f00,0x7f00,0xff00]
    result.append(dict(name='partial_fp32_overflow_before_cancellation',u_words=a,old_words=b,gamma_words=g))
    a,b,g=u[:],old[:],gamma[:];b[0]=b[5]=a[0]=half_word(65504.);g[:3]=[0x3f80]*3
    result.append(dict(name='half_output_overflow',u_words=a,old_words=b,gamma_words=g))
    for item in result:item.update(rows=1,width=5,input_stride=stride,status_nonzero=True)
    return result


def export(directory):
    producer=identity(Path(__file__).read_bytes());helpers=identity(Path(helper.__file__).read_bytes())
    directory.mkdir(parents=False,exist_ok=False)
    cases=[]
    try:
        with (directory/'arrays.bin').open('xb') as binary,(directory/'progress.jsonl').open('x') as progress:
            def save(values,kind,shape_):
                raw=b''.join(struct.pack('<'+kind,v) for v in values);offset=binary.tell()
                if offset+len(raw)>MAX_BINARY:raise ValueError('Binary bound')
                binary.write(raw)
                return dict(file='arrays.bin',offset=offset,shape=shape_,dtype={'H':'u16le','I':'u32le'}[kind],**identity(raw))
            for name,rows,width,pattern in CASES:
                progress.write(json.dumps(dict(case=name,started=True))+'\n');progress.flush()
                u,old,gamma,scales,stride=operands(rows,width,pattern)
                main,candidate=conv(u,old,scales,rows,width,stride,'main')
                wide,wide_candidate=conv(u,old,scales,rows,width,stride,'wide')
                if candidate!=wide_candidate:raise ValueError('Profile state differs')
                exact=[]
                if pattern=='hand':exact=list(map(half_word,HAND_OUTPUT))
                if pattern=='stores':exact=STORE_WORDS
                if exact and (main!=exact or wide!=exact):raise ValueError('Frozen exact arithmetic witness')
                if pattern=='zero' and (any(w&0x7fff for w in main+wide) or not any(w&0x7fff for w in candidate)):
                    raise ValueError('Zero coefficient/raw state witness')
                envelope=max(abs(half_value(a)-half_value(b)) for a,b in zip(main,wide,strict=True))
                schedules=partition_checks(u,old,scales,rows,width,stride,main)
                descriptors=dict(u=save(u,'H',[len(u)]),old=save(old,'H',[len(old)]),
                    gamma=save(gamma,'H',[width,3]),scales=save([float_word(v) for v in scales]+[COEFFICIENT_GUARD]*GUARD,'I',[3*width+GUARD]),
                    main=save(main,'H',[rows,width]),wide=save(wide,'H',[rows,width]),candidate=save(candidate,'H',[2,width]))
                cases.append(dict(name=name,rows=rows,width=width,input_stride=stride,output_stride=width+5,
                    output_elements=rows*(width+5)+GUARD,candidate_elements=2*width+GUARD,arrays=descriptors,
                    envelope=envelope,gate=dict(atol=max(POLICY['atol_floor'],POLICY['envelope_factor']*envelope),rtol=POLICY['rtol']),
                    exact_words=exact,partitions=schedules))
                binary.flush();progress.write(json.dumps(dict(case=name,complete=True,arrays_end=binary.tell()))+'\n');progress.flush()
        invalid=errors()
        for item in invalid:
            for profile in ('main','wide'):
                try:conv(item['u_words'],item['old_words'],coefficients(item['gamma_words']),item['rows'],item['width'],item['input_stride'],profile)
                except (ValueError,OverflowError):continue
                raise ValueError('Invalid control admitted')
        raw=encoded(dict(controls=invalid,output_bits_not_a_gate=True))
        (directory/'errors.json').write_bytes(raw)
        files={name:identity((directory/name).read_bytes()) for name in ('arrays.bin','errors.json')}
        if producer!=identity(Path(__file__).read_bytes()) or helpers!=identity(Path(helper.__file__).read_bytes()):raise ValueError('Source endpoint changed')
        manifest=dict(format=FORMAT,protocol_sha256=PROTOCOL,producer=producer,
            helper=dict(path=str(Path(helper.__file__).resolve()),**helpers),policy=POLICY,guard=GUARD,
            input_padding_word=INPUT_PADDING,output_canary_word=CANARY,coefficient_guard_word=COEFFICIENT_GUARD,
            cases=cases,files=files,finite_output_values=sum(r*h for _,r,h,_ in CASES),
            candidate_words=sum(2*h for _,_,h,_ in CASES),source_reference_executed=True,target_gpu_executed=False,
            runtime=dict(python=sys.version,executable=str(Path(sys.executable).resolve())),
            scope='Standalone raw Conv3 Half arithmetic and exact candidate recipe; no fused gates, resident transaction, graph, checkpoint or performance claim')
        raw=encoded(manifest)
        if len(raw)>MAX_JSON:raise ValueError('Manifest bound')
        (directory/'manifest.json').write_bytes(raw)
        return identity(raw)
    except BaseException as error:
        (directory/'failure.json').write_bytes(encoded(dict(complete=False,error=type(error).__name__+': '+str(error))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    print(json.dumps(export(parser.parse_args().output_dir)))
