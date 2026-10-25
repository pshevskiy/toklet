"""Pre-target packed row/gather/partition witnesses; no engine or GPU imports."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,math,struct,sys

LIMIT=64*1024
U64=(1<<64)-1

def f32(value):
    try:return struct.unpack('<f',struct.pack('<f',value))[0]
    except OverflowError:return math.copysign(math.inf,value)

def bf16(bits):return struct.unpack('<f',struct.pack('<I',bits<<16))[0]
def raw32(values):return b''.join(struct.pack('<f',v) for v in values)
def sha(raw):return hashlib.sha256(raw).hexdigest()

def recipe(rows,k):
    scales=[];biases=[];codes=[]
    ss=[0x3e00,0xbe80,0x3f00,0x3f80,0xbf80,0x4000,0x0000]
    bs=[0x3e80,0xbf00,0x3f80,0xc000,0x3d80,0x0000]
    for row in range(rows):
        for group in range(k//64):
            scales.append(0x8000 if row==0 else ss[(row+3*group)%len(ss)])
            biases.append(0x8000 if row==0 else bs[(2*row+group)%len(bs)])
        codes.extend((row*7+column*5+column//64)%16 for column in range(k))
    packed=bytes(codes[i]|codes[i+1]<<4 for i in range(0,len(codes),2))
    return packed,b''.join(struct.pack('<H',v) for v in scales),b''.join(struct.pack('<H',v) for v in biases)

def reconstruct(rows,k,packed,scales,biases):
    result=[]
    for row in range(rows):
        for column in range(k):
            group=(row*k+column)//64
            scale=bf16(struct.unpack_from('<H',scales,group*2)[0])
            bias=bf16(struct.unpack_from('<H',biases,group*2)[0])
            code=(packed[(row*k+column)//2]>>((column%2)*4))&15
            result.append(f32(float(scale)*code+float(bias)))
    return result

def gather(rows,k,weights,ids,destination):
    # Tiny witness policy, not a production model/table admission capability.
    if type(rows) is not int or not 1<=rows<=32 or type(k) is not int or k not in (64,128) or len(weights)!=rows*k:raise ValueError('gather geometry or source')
    if not 1<=len(ids)<=32 or len(destination)!=len(ids)*k or any(type(i) is not int or not 0<=i<rows for i in ids):raise ValueError('gather index or destination')
    # The reference preflights every selected row before publishing any row.
    if any(not math.isfinite(v) for i in ids for v in weights[i*k:(i+1)*k]):raise ValueError('gather reconstruction overflow')
    destination[:]=[v for i in ids for v in weights[i*k:(i+1)*k]]

def interval(rows,start,end):
    if not all(type(v) is int and 0<=v<=U64 for v in (rows,start,end)) or not 0<=start<end<=rows:raise ValueError('row interval')
    return start,end-start

def partition(rows,maximum):
    if type(rows) is not int or not 0<rows<=32 or type(maximum) is not int or not 1<=maximum<=16384:raise ValueError('partition bound')
    return [(start,min(maximum,rows-start)) for start in range(0,rows,maximum)]

def checked_layout(rows,k):
    if type(rows) is not int or type(k) is not int or rows<=0 or k<=0 or k%64:raise ValueError('table layout')
    sizes=(rows*k//2,rows*(k//64)*2)
    if any(v>U64 for v in sizes) or sizes[0]+sizes[1]*2>U64:raise ValueError('table allocation overflow')
    return sizes

class Capture:
    def __init__(self,path):self.path=path;self.binary=(path/'math.bin').open('xb');self.bytes=0;self.arrays=[];self.attempt=None
    def add(self,name,dtype,shape,raw):
        width={'U8':1,'BF16':2,'F32':4,'F64':8}[dtype]
        if math.prod(shape)*width!=len(raw) or self.bytes+len(raw)>LIMIT:raise ValueError('bounded complete array')
        item=dict(name=name,dtype=dtype,shape=shape,offset=self.bytes,bytes=len(raw),sha256=sha(raw))
        self.attempt=item
        if self.binary.write(raw)!=len(raw):raise OSError('short raw array write')
        self.binary.flush();self.bytes+=len(raw);self.arrays.append(item);self.attempt=None
        return item

def witnesses(capture,report):
    cases=report['cases']
    for rows,k in ((19,128),(11,64)):
        name=f'rows{rows}-k{k}';attempt=dict(name=name,completed=False);report['attempts'].append(attempt)
        packed,scales,biases=recipe(rows,k);weights=reconstruct(rows,k,packed,scales,biases)
        inputs=[-0. if c%2 else 0. for c in range(k)]+[(c%13-6)/8 for c in range(k)]+[0.]*k
        for c,value in ((0,1.),(31,-2.),(63,.5),(k-1,.25)):inputs[2*k+c]=value
        if k>64:inputs[2*k+64]=-.75
        ordered=[];exact=[]
        for token in range(3):
            for row in range(rows):
                total=0.;dot=Fraction()
                for column in range(k):
                    x=inputs[token*k+column];w=weights[row*k+column]
                    total+=float(x)*float(w);dot+=Fraction(x)*Fraction(w)
                ordered.append(total);exact.append(dict(numerator=str(dot.numerator),denominator=str(dot.denominator)))
        ids=[0,6,7,rows-1,7];selected=[42.]*(len(ids)*k);gather(rows,k,weights,ids,selected)
        arrays={}
        for role,dtype,shape,raw in [('codes','U8',[rows,k//2],packed),('scales','BF16',[rows,k//64],scales),('biases','BF16',[rows,k//64],biases),
            ('reconstructed','F32',[rows,k],raw32(weights)),('input','F32',[3,k],raw32(inputs)),('gather','F32',[len(ids),k],raw32(selected)),
            ('ordered_fp64','F64',[3,rows],b''.join(struct.pack('<d',v) for v in ordered)),('output_f32','F32',[3,rows],raw32(map(f32,ordered)))]:
            arrays[role]=capture.add(name+'/'+role,dtype,shape,raw)
        partitions=[]
        for maximum in (1,7,8,19):
            chunks=[]
            for start,count in partition(rows,maximum):
                views={}
                for role,stride,source in [('codes',k//2,packed),('scales',k//32,scales),('biases',k//32,biases)]:
                    body=source[start*stride:(start+count)*stride]
                    views[role]=dict(source_array=arrays[role]['name'],source_offset=start*stride,bytes=len(body),sha256=sha(body),zero_padding_bytes=(-len(body))%4,padded_sha256=sha(body+b'\0'*((-len(body))%4)))
                chunks.append(dict(first_row=start,rows=count,views=views,output_indices=[token*rows+row for token in range(3) for row in range(start,start+count)]))
            partitions.append(dict(maximum_rows=maximum,chunks=chunks))
        cases.append(dict(name=name,rows=rows,columns=k,bits=4,group_size=64,metadata='BF16',tokens=3,gather_ids=ids,arrays=arrays,exact_rational_dots=exact,partitions=partitions,
            shared_embedding_head_source=True,source_bytes_counted_once=len(packed)+len(scales)+len(biases),whole_head_gpu_qualified=False))
        attempt['completed']=True
    # Finite BF16 source with a later selected row that reconstructs to infinity.
    attempt=dict(name='late-gather-overflow',completed=False);report['attempts'].append(attempt)
    rows=3;k=64;packed=bytes([0xff])*(rows*k//2);scales=b'\0\0'*2+struct.pack('<H',0x7f7f);biases=b'\0\0'*2+struct.pack('<H',0x7f7f)
    weights=reconstruct(rows,k,packed,scales,biases);destination=[42.]*(4*k);before=raw32(destination)
    try:gather(rows,k,weights,[1,0,2,1],destination)
    except ValueError as error:reason=str(error)
    else:raise AssertionError('late selected-row overflow must reject')
    assert raw32(destination)==before
    overflow=dict(name='late-gather-overflow',rows=rows,columns=k,gather_ids=[1,0,2,1],rejected=True,error=reason,destination_untouched=True,arrays={})
    report['overflow']=overflow
    for role,dtype,shape,raw in [('codes','U8',[rows,k//2],packed),('scales','BF16',[rows,1],scales),('biases','BF16',[rows,1],biases),('destination','F32',[4,k],before)]:
        overflow['arrays'][role]=capture.add(overflow['name']+'/'+role,dtype,shape,raw)
    attempt['completed']=True
    rejected=report['rejections']
    for label,operation,args in [('zero-tile',partition,(19,0)),('excessive-tile',partition,(19,16385)),('empty-interval',interval,(19,1,1)),('reversed-interval',interval,(19,7,6)),('outside-interval',interval,(19,19,20)),('overflow-interval',interval,(19,U64+1,U64+2)),('allocation-overflow',checked_layout,(U64,128)),('invalid-gather-id',gather,(3,64,[0.]*192,[0,U64],[42.]*128)),('excessive-materialized-rows',partition,(U64,1)),('truncated-gather-source',gather,(3,64,[0.]*128,[0,2],[42.]*128))]:
        try:operation(*args)
        except ValueError as error:rejected.append(dict(name=label,rejected=True,error=str(error)))
        else:raise AssertionError(label)
    return cases,overflow,rejected

def capture(directory):
    directory.mkdir();writer=Capture(directory);report=dict(format='toklet-gemma-table-math-v1',passed=False,cases=[],overflow=None,rejections=[],attempts=[],arrays=writer.arrays,binary_cap_bytes=LIMIT,engine_executed=False,gpu_executed=False)
    original=None
    try:
        witnesses(writer,report);report['passed']=True
    except BaseException as error:original=error;report['error']=type(error).__name__+': '+str(error)
    finally:
        try:writer.binary.close()
        except BaseException as error:
            if original is None:original=error
        report['binary_bytes']=writer.bytes;report['array_attempt']=writer.attempt
        if original is not None:report['passed']=False
        try:
            with (directory/'math.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
        except BaseException as error:
            if original is None:original=error
        files={};file_errors=[]
        for name in ('math.json','math.bin'):
            try:
                path=directory/name
                with path.open('rb') as stream:raw=stream.read(512*1024+1)
                if len(raw)>512*1024:raise ValueError('bounded frozen document')
                files[name]=dict(bytes=len(raw),sha256=sha(raw))
            except BaseException as error:
                file_errors.append(dict(name=name,error=type(error).__name__+': '+str(error)))
                if original is None:original=error
        try:
            manifest=dict(format='toklet-gemma-table-math-manifest-v1',passed=original is None and report['passed'],files=files,file_errors=file_errors,primary_error=None if original is None else type(original).__name__+': '+str(original))
            with (directory/'manifest.json').open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
        except BaseException as error:
            if original is None:original=error
    if original is not None:raise original
    return dict(passed=True,cases=len(report['cases']),arrays=len(report['arrays']),binary_bytes=writer.bytes,rejections=len(report['rejections']),overflow_atomic=True)

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('exclusive output directory')
    print(json.dumps(capture(Path(sys.argv[1]))))
