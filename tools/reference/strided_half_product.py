"""Independent stdlib strided Half product reference, not a graph/target import.

Shared word/serialization helpers only. One physical BCx parent serves both
readonly input views; this reference never expands it into stored B/C/x arrays.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

_path = Path(__file__).with_name('gemma3_half_rms.py')
_spec = importlib.util.spec_from_file_location('_product_word_helpers', _path)
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
f32, half_word, half_value = helper.f32, helper.half_word, helper.half_value
identity, encoded = helper.identity, helper.encoded
FORMAT = 'toklet-strided-half-product-v1'
PROTOCOL = 'eb77913b363e5ba0b58fb7bebfb84adf1137da412c48bf98f2202801b79da8d6'
POLICY = dict(atol_floor=1e-4,envelope_factor=2,rtol=1e-3,fatal_reference='main')
CASES = (('tiny_bx1',1,256,'bx',0,3),('tiny_cz3',3,256,'cz',3,7),
         ('tiny_words2',2,256,'words',5,1),('real_bx32',32,1024,'bx',0,0),
         ('real_cz32',32,1024,'cz',3,3))
GUARD, PAD, CANARY, U32 = 7, 0x7e55, 0x5a17, 2**32-1
MAX_BINARY, MAX_JSON, MAX_CONTROLS = 2*1024**2,128*1024,64*1024
WITNESS_A = [0x0000,0x8000,0x0001,0x0001,0x0003,0x8001,0x8001,0x7bff,0x3c01,0xbc00]
WITNESS_B = [0xbc00,0xbc00,0x3c00,0x3800,0x3800,0x3c00,0x3800,0x3c00,0x3c01,0x0000]
WITNESS = [0x8000,0x0000,0x0001,0x0000,0x0002,0x8001,0x8000,0x7bff,0x3c02,0x8000]


def integer(value):
    if type(value) is not int or not 0 <= value <= U32:raise ValueError('Unsigned UInt32 value')
    return value


def extent(rows,width,offset,stride,output=False):
    if any(type(v) is not int for v in (rows,width,offset,stride)):
        raise ValueError('Integral shape')
    if not 1<=rows<=32 or width not in (256,1024) or stride<width:
        raise ValueError('Half product shape/stride')
    integer(offset);integer(stride)
    end=offset+((rows*stride) if output else (rows-1)*stride+width)
    if end>U32 or end+GUARD>U32 or end+GUARD>sys.maxsize//2:raise ValueError('Half product UInt32 extent')
    return end+GUARD


def aliases(a,b,out,status):
    if any(not isinstance(v,str) or not v for v in (a,b,out,status)):
        raise ValueError('Physical owner identity')
    if out in (a,b,status) or status in (a,b):raise ValueError('Writable physical alias')


def operands(case):
    name,rows,width,kind,out_offset,extra=case
    extent(rows,width,out_offset,width+extra,True)
    if kind not in ('bx','cz','words'):raise ValueError('Fixed product kind')
    parent=[PAD]*(rows*3*width+GUARD)
    for row in range(rows):
        for field in range(3):
            for column in range(width):
                parent[row*3*width+field*width+column]=half_word(((17*row+11*column+7*field)%37-18)/16)
    physical={'bcx':parent}
    if kind=='cz':
        zoffset,zstride=2,width+5
        z=[PAD]*(zoffset+rows*zstride+GUARD)
        for row in range(rows):
            for column in range(width):z[zoffset+row*zstride+column]=half_word(((19*row+5*column)%31-15)/8)
        physical['z']=z
        a=dict(owner='bcx',offset=width,stride=3*width)
        b=dict(owner='z',offset=zoffset,stride=zstride)
    else:
        a=dict(owner='bcx',offset=0,stride=3*width)
        b=dict(owner='bcx',offset=2*width,stride=3*width)
    if kind=='words':
        parent[:len(WITNESS_A)]=WITNESS_A
        parent[2*width:2*width+len(WITNESS_B)]=WITNESS_B
    return physical,a,b,dict(owner='output',offset=out_offset,stride=width+extra)


def check_views(physical,a,b,out,rows,width,output_elements):
    aliases(a['owner'],b['owner'],out['owner'],'status')
    required=extent(rows,width,out['offset'],out['stride'],True)
    if type(output_elements)is not int or output_elements<required:raise ValueError('Output extent')
    for view in (a,b):
        required=extent(rows,width,view['offset'],view['stride'])
        if view['owner']not in physical or len(physical[view['owner']])<required:
            raise ValueError('Input extent')
    # Finite validation of BOTH active views precedes multiplication, including
    # a zero counterpart. Inactive parent fields/padding/guards may be NaN.
    for view in (a,b):
        values=physical[view['owner']]
        for row in range(rows):
            for column in range(width):
                if not math.isfinite(half_value(values[view['offset']+row*view['stride']+column])):
                    raise ValueError('Nonfinite active Half')


def multiply(physical,a,b,out,rows,width,profile,output_elements=None):
    if profile not in ('main','wide'):raise ValueError('Reference profile')
    if output_elements is None:output_elements=extent(rows,width,out['offset'],out['stride'],True)
    check_views(physical,a,b,out,rows,width,output_elements)
    result=[]
    for row in range(rows):
        for column in range(width):
            x=half_value(physical[a['owner']][a['offset']+row*a['stride']+column])
            y=half_value(physical[b['owner']][b['offset']+row*b['stride']+column])
            # Half*Half has <=22 significand bits, exponent range [-48,32].
            # Thus the product is exact in FP32 as well as FP64, even when its
            # subsequent Half store overflows. No contraction or reduction.
            product=f32(x*y) if profile=='main' else x*y
            word=half_word(product)
            if not math.isfinite(half_value(word)):raise ValueError('Nonfinite Half store')
            result.append(word)
    return result


def output_words(values,rows,width,out):
    if len(values)!=rows*width:raise ValueError('Active output shape')
    result=[CANARY]*extent(rows,width,out['offset'],out['stride'],True)
    for row in range(rows):
        start=out['offset']+row*out['stride'];result[start:start+width]=values[row*width:(row+1)*width]
    return result


def raw_words(values):return b''.join(struct.pack('<H',v)for v in values)


def partitions(physical,a,b,out,rows,width,main):
    schedules=[]
    for maximum in (1,3,32):
        left=rows;s=[]
        while left:n=min(maximum,left);s.append(n);left-=n
        if s not in schedules:schedules.append(s)
    for schedule in schedules:
        at=0;combined=[]
        for n in schedule:
            av=dict(a,offset=a['offset']+at*a['stride'])
            bv=dict(b,offset=b['offset']+at*b['stride'])
            combined+=multiply(physical,av,bv,out,n,width,'main');at+=n
        if combined!=main:raise ValueError('Partition words differ')
    return schedules


def controls():
    # Scalar patches describe corruption of the fully pinned finite tiny_words2
    # buffers. Native invalid output words are not an equality oracle.
    bad=[]
    for view in ('a','b'):
        for name,word in [('nan',0x7e01),('positive_inf',0x7c00),('negative_inf',0xfc00)]:
            bad.append(dict(name=view+'_'+name,view=view,row=1,column=255,word=word,other_word=0))
    bad.append(dict(name='positive_half_overflow',view='a',row=1,column=255,word=0x7bff,other_word=0x4000))
    bad.append(dict(name='negative_half_overflow',view='a',row=1,column=255,word=0xfbff,other_word=0x4000))
    return dict(exact_a=WITNESS_A,exact_b=WITNESS_B,exact_result=WITNESS,
                invalid_base_case='tiny_words2',invalid=bad,invalid_output_bits_not_a_gate=True)


def apply_control(item):
    physical,a,b,out=operands(CASES[2]);views={'a':a,'b':b};v=views[item['view']];other=views['b'if item['view']=='a'else'a']
    for view,word in [(v,item['word']),(other,item['other_word'])]:
        physical[view['owner']][view['offset']+item['row']*view['stride']+item['column']]=word
    return physical,a,b,out


def export(directory):
    producer=identity(Path(__file__).read_bytes());helpers=identity(Path(helper.__file__).read_bytes())
    directory.mkdir(parents=False,exist_ok=False);cases=[];arrays=0
    try:
        with (directory/'arrays.bin').open('xb')as binary,(directory/'progress.jsonl').open('x')as progress:
            def save(values,shape):
                nonlocal arrays
                raw=raw_words(values);offset=binary.tell()
                if offset+len(raw)>MAX_BINARY:raise ValueError('Binary bound')
                binary.write(raw);arrays+=1
                return dict(file='arrays.bin',dtype='u16le',offset=offset,shape=shape,**identity(raw))
            for case in CASES:
                name,rows,width,kind,_,_=case;progress.write(json.dumps(dict(case=name,started=True))+'\n');progress.flush()
                physical,a,b,out=operands(case);before={key:identity(raw_words(v))for key,v in physical.items()}
                main=multiply(physical,a,b,out,rows,width,'main');wide=multiply(physical,a,b,out,rows,width,'wide')
                if main!=wide:raise ValueError('Exact finite Half product profiles differ')
                if kind=='words' and main[:len(WITNESS)]!=WITNESS:raise ValueError('Exact store witnesses')
                if before!={key:identity(raw_words(v))for key,v in physical.items()}:raise ValueError('Input mutation')
                envelope=max(abs(half_value(x)-half_value(y))for x,y in zip(main,wide,strict=True))
                inputs={key:save(v,[len(v)])for key,v in physical.items()}
                descriptors=dict(main=save(main,[rows,width]),wide=save(wide,[rows,width]))
                cases.append(dict(name=name,rows=rows,width=width,inputs=inputs,a=a,b=b,output=out,
                    output_elements=extent(rows,width,out['offset'],out['stride'],True),
                    physical_input_bytes=sum(len(v)*2 for v in physical.values()),
                    readonly_same_parent=a['owner']==b['owner'],arrays=descriptors,
                    expected_physical_output=identity(raw_words(output_words(main,rows,width,out))),
                    envelope=envelope,gate=dict(atol=max(POLICY['atol_floor'],2*envelope),rtol=POLICY['rtol']),
                    exact_prefix_words=WITNESS if kind=='words'else[],partitions=partitions(physical,a,b,out,rows,width,main)))
                binary.flush();progress.write(json.dumps(dict(case=name,complete=True,arrays_end=binary.tell()))+'\n');progress.flush()
        controls_raw=encoded(controls())
        if len(controls_raw)>MAX_CONTROLS:raise ValueError('Controls bound')
        for item in controls()['invalid']:
            physical,a,b,out=apply_control(item)
            for profile in ('main','wide'):
                try:multiply(physical,a,b,out,2,256,profile)
                except(ValueError,OverflowError):continue
                raise ValueError('Invalid source input admitted')
        (directory/'controls.json').write_bytes(controls_raw)
        files={n:identity((directory/n).read_bytes())for n in ('arrays.bin','controls.json')}
        if files['arrays.bin']['bytes']!=738248 or arrays!=17:raise ValueError('Frozen descriptor counts')
        if producer!=identity(Path(__file__).read_bytes())or helpers!=identity(Path(helper.__file__).read_bytes()):raise ValueError('Source endpoint changed')
        m=dict(format=FORMAT,protocol_sha256=PROTOCOL,producer=producer,helper=dict(path=str(Path(helper.__file__).resolve()),**helpers),
            policy=POLICY,cases=cases,stored_arrays=arrays,finite_output_values=67072,files=files,
            guard_words=GUARD,input_padding_word=PAD,output_canary_word=CANARY,
            source_reference_executed=True,target_gpu_executed=False,
            runtime=dict(python=sys.version,executable=str(Path(sys.executable).resolve())),
            scope='Standalone strided Half multiply and physical-view oracle; no activation, projection, graph, model, transaction or performance claim')
        raw=encoded(m)
        if len(raw)>MAX_JSON:raise ValueError('Manifest bound')
        (directory/'manifest.json').write_bytes(raw);return identity(raw)
    except BaseException as e:
        (directory/'failure.json').write_bytes(encoded(dict(complete=False,error=type(e).__name__+': '+str(e))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    print(json.dumps(export(parser.parse_args().output_dir)))
