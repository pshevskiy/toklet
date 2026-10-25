"""Independent packed-table byte, IEEE rounding and Decimal dot replay."""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal,localcontext
import hashlib,json,math,struct,sys

def same(a,b):
    if type(a) is not type(b):raise ValueError('typed field differs')
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('record fields differ')
        for key in a:same(a[key],b[key])
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('record length differs')
        for x,y in zip(a,b,strict=True):same(x,y)
    elif a!=b:raise ValueError('record value differs')

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path,limit):
    with path.open('rb') as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise ValueError('bounded fixture required')
    return raw

def document(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate field')
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('finite JSON required')))

def rational(bits,mantissa,exponent):
    sign=-1 if bits>>(mantissa+exponent) else 1
    e=(bits>>mantissa)&((1<<exponent)-1);m=bits&((1<<mantissa)-1);bias=(1<<(exponent-1))-1
    if e==(1<<exponent)-1:raise ValueError('finite source required')
    power=(1-bias if e==0 else e-bias)-mantissa
    if e:m+=1<<mantissa
    return sign*Fraction(m)*(Fraction(2)**power)

def rounded(value,mantissa,exponent,negative_zero=False):
    sign=int(value<0 or value==0 and negative_zero);value=abs(value)
    if not value:return sign<<(mantissa+exponent)
    bias=(1<<(exponent-1))-1
    e=value.numerator.bit_length()-value.denominator.bit_length()
    if value<Fraction(2)**e:e-=1
    effective=max(e,1-bias)
    scaled=value*Fraction(2)**(mantissa-effective)
    q,r=divmod(scaled.numerator,scaled.denominator)
    if 2*r>scaled.denominator or 2*r==scaled.denominator and q%2:q+=1
    if q>=1<<(mantissa+1):q>>=1;effective+=1
    if effective>bias:return sign<<(mantissa+exponent)|((1<<exponent)-1)<<mantissa
    field=0 if q<1<<mantissa else effective+bias
    return sign<<(mantissa+exponent)|field<<mantissa|(q&((1<<mantissa)-1))

def bits32(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def ieee32(value,negative_zero=False):return struct.pack('<I',rounded(value,23,8,negative_zero))

def replay(directory):
    raw=read(directory/'math.bin',65536);report_raw=read(directory/'math.json',512*1024);manifest=document(read(directory/'manifest.json',65536));data=document(report_raw)
    same(manifest,dict(format='toklet-gemma-table-math-manifest-v1',passed=True,files={name:dict(bytes=len(body),sha256=sha(body)) for name,body in [('math.json',report_raw),('math.bin',raw)]},file_errors=[],primary_error=None))
    arrays=[];offset=0;expected_cases=[];negative_zeros=0;unaligned=0
    def array(entry,name,dtype,shape,expected):
        nonlocal offset
        same(entry,dict(name=name,dtype=dtype,shape=shape,offset=offset,bytes=len(expected),sha256=sha(expected)))
        if raw[offset:offset+len(expected)]!=expected:raise ValueError('complete array bytes differ')
        arrays.append(entry);offset+=len(expected)
    for case,(rows,k) in zip(data['cases'],[(19,128),(11,64)],strict=True):
        name=f'rows{rows}-k{k}';groups=k//64
        ss=[Fraction(1,8),Fraction(-1,4),Fraction(1,2),Fraction(1),Fraction(-1),Fraction(2),Fraction()]
        bs=[Fraction(1,4),Fraction(-1,2),Fraction(1),Fraction(-2),Fraction(1,16),Fraction()]
        code_values=[];scale_values=[];bias_values=[];weight_bits=[];weights=[]
        for row in range(rows):
            for group in range(groups):
                s=ss[(row+group*3)%7] if row else Fraction();b=bs[(2*row+group)%6] if row else Fraction()
                sb=rounded(s,7,8,row==0);bb=rounded(b,7,8,row==0)
                scale_values.append(sb);bias_values.append(bb)
                for column in range(group*64,(group+1)*64):
                    code=(7*row+5*column+group)&15;code_values.append(code)
                    bits=rounded(s*code+b,23,8,row==0);weight_bits.append(bits);weights.append(rational(bits,23,8))
                    negative_zeros+=bits==0x80000000
        packed=bytes(a+16*b for a,b in zip(code_values[::2],code_values[1::2],strict=True));scales=b''.join(struct.pack('<H',v) for v in scale_values);biases=b''.join(struct.pack('<H',v) for v in bias_values)
        input_values=[Fraction()]*(3*k);input_bits=[0x80000000 if c%2 else 0 for c in range(k)]
        for c in range(k):input_values[k+c]=Fraction(c%13-6,8)
        for c,value in ((0,1),(31,-2),(63,Fraction(1,2)),(k-1,Fraction(1,4))):input_values[2*k+c]=Fraction(value)
        if k==128:input_values[2*k+64]=Fraction(-3,4)
        input_bits += [rounded(v,23,8) for v in input_values[k:]]
        outputs=[];exact=[];ordered=[]
        with localcontext() as context:
            context.prec=400
            for token in range(3):
                for row in range(rows):
                    total=Decimal(0);current=Fraction()
                    for column in range(k):
                        product=weights[row*k+column]*input_values[token*k+column]
                        total+=Decimal(product.numerator)/Decimal(product.denominator)
                        current=rational(rounded(current+product,52,11),52,11)
                    dot=Fraction(total);exact.append(dict(numerator=str(dot.numerator),denominator=str(dot.denominator)))
                    ordered.append(rounded(current,52,11));outputs.append(rounded(current,23,8))
        ids=[0,6,7,rows-1,7];gathered=[v for i in ids for v in weight_bits[i*k:(i+1)*k]]
        expected_arrays=[('codes','U8',[rows,k//2],packed),('scales','BF16',[rows,groups],scales),('biases','BF16',[rows,groups],biases),
            ('reconstructed','F32',[rows,k],b''.join(struct.pack('<I',v) for v in weight_bits)),('input','F32',[3,k],b''.join(struct.pack('<I',v) for v in input_bits)),
            ('gather','F32',[5,k],b''.join(struct.pack('<I',v) for v in gathered)),('ordered_fp64','F64',[3,rows],b''.join(struct.pack('<Q',v) for v in ordered)),('output_f32','F32',[3,rows],b''.join(struct.pack('<I',v) for v in outputs))]
        same(sorted(case['arrays']),sorted(v[0] for v in expected_arrays))
        for role,dtype,shape,expected in expected_arrays:array(case['arrays'][role],name+'/'+role,dtype,shape,expected)
        plans=[]
        for maximum in (1,7,8,19):
            chunks=[];start=0
            while start<rows:
                count=maximum if rows-start>=maximum else rows-start;views={}
                for role,source,stride in [('codes',packed,k//2),('scales',scales,groups*2),('biases',biases,groups*2)]:
                    first=start*stride;body=source[first:first+count*stride];padding=(4-len(body)%4)%4
                    if role!='codes' and first%4==2:unaligned+=1
                    views[role]=dict(source_array=name+'/'+role,source_offset=first,bytes=count*stride,sha256=sha(body),zero_padding_bytes=padding,padded_sha256=sha(body+bytes(padding)))
                indices=[]
                for token in range(3):indices.extend(range(token*rows+start,token*rows+start+count))
                chunks.append(dict(first_row=start,rows=count,views=views,output_indices=indices));start+=count
            plans.append(dict(maximum_rows=maximum,chunks=chunks))
        expected_case=dict(name=name,rows=rows,columns=k,bits=4,group_size=64,metadata='BF16',tokens=3,gather_ids=ids,arrays=case['arrays'],exact_rational_dots=exact,partitions=plans,
            shared_embedding_head_source=True,source_bytes_counted_once=len(packed)+len(scales)+len(biases),whole_head_gpu_qualified=False)
        same(case,expected_case);expected_cases.append(expected_case)
    overflow=data['overflow'];oa=overflow['arrays']
    same(sorted(oa),['biases','codes','destination','scales'])
    for role,dtype,shape,expected in [('codes','U8',[3,32],bytes([255])*96),('scales','BF16',[3,1],bytes.fromhex('000000007f7f')),('biases','BF16',[3,1],bytes.fromhex('000000007f7f')),('destination','F32',[4,64],struct.pack('<f',42.)*256)]:
        array(oa[role],'late-gather-overflow/'+role,dtype,shape,expected)
    same(overflow,dict(name='late-gather-overflow',rows=3,columns=64,gather_ids=[1,0,2,1],rejected=True,error='gather reconstruction overflow',destination_untouched=True,arrays=oa))
    # Independently prove the failing selected row reconstructs beyond finite FP32.
    maximum=rational(0x7f7f,7,8)
    if rounded(maximum*16,23,8)!=0x7f800000:raise ValueError('late row is not an overflow witness')
    rejects=[dict(name=name,rejected=True,error=error) for name,error in [('zero-tile','partition bound'),('excessive-tile','partition bound'),('empty-interval','row interval'),('reversed-interval','row interval'),('outside-interval','row interval'),('overflow-interval','row interval'),('allocation-overflow','table allocation overflow'),('invalid-gather-id','gather index or destination'),('excessive-materialized-rows','partition bound'),('truncated-gather-source','gather geometry or source')]]
    same(data,dict(format='toklet-gemma-table-math-v1',passed=True,cases=expected_cases,overflow=overflow,rejections=rejects,attempts=[dict(name=n,completed=True) for n in ['rows19-k128','rows11-k64','late-gather-overflow']],arrays=arrays,binary_cap_bytes=65536,engine_executed=False,gpu_executed=False,binary_bytes=offset,array_attempt=None))
    if len(raw)!=offset or offset!=22664 or len(arrays)!=20 or not unaligned or negative_zeros!=192:raise ValueError('complete boundary inventory differs')
    return dict(passed=True,cases=2,arrays=20,binary_bytes=offset,rejections=10,overflow_atomic=True,negative_zero_weights=negative_zeros,two_byte_metadata_views=unaligned,engine_executed=False,gpu_executed=False)

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('frozen fixture directory')
    print(json.dumps(replay(Path(sys.argv[1]))))
