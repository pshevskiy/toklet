"""Synthetic only: no artifact, tokenizer, graph or GPU imports/operands."""
from fractions import Fraction
import io
import math
import struct
import unittest
import numpy as np
import lfm2_exact_affine as a


def fraction_bits(bits, kind):
    p, eb, bias = (10, 5, 15) if kind == 'half' else (23, 8, 127)
    sign = -1 if bits >> (p+eb) else 1
    field = (bits >> p) & ((1 << eb)-1)
    if field == (1 << eb)-1:
        raise ValueError('nonfinite oracle word')
    mantissa = bits & ((1 << p)-1)
    exponent = 1-bias-p
    if field:
        mantissa += 1 << p;exponent = field-bias-p
    return sign*Fraction(mantissa)*(Fraction(2)**exponent)


def oracle(value, kind):
    """Independent finite-word binary search, exact Fraction distances/ties."""
    signbit, largest = (15, 0x7bff) if kind == 'half' else (31, 0x7f7fffff)
    sign = (1 << signbit) if value < 0 else 0
    value = abs(value)
    low, high = 0, largest
    while low < high:
        middle = (low+high+1)//2
        if fraction_bits(middle, kind) <= value:
            low = middle
        else:
            high = middle-1
    bottom = fraction_bits(low, kind)
    if bottom == value:
        return sign | low
    if low == largest:
        # Overflow midpoint uses the next binade endpoint, not infinity distance.
        next_binade = Fraction(2)**(16 if kind == 'half' else 128)
        if value >= (bottom+next_binade)/2:
            raise ValueError('oracle overflow')
        return sign | low
    top = fraction_bits(low+1, kind)
    if value-bottom < top-value or (value-bottom == top-value and not low & 1):
        return sign | low
    return sign | (low+1)


def stats():
    return dict(zero=0, fp64=0, int64=0, blocked_outputs=0, certified_blocks=0, scalar_blocks=0)


def expected(w, x, kind):
    return [oracle(sum((Fraction.from_float(float(v))*Fraction.from_float(float(t))
                        for v, t in zip(row, x, strict=True)), Fraction()), kind) for row in w]


def operands(columns=64):
    return np.zeros((1, columns), dtype='<f4'), np.ones((1, columns), dtype='<f2')


class ExactAffine(unittest.TestCase):
    def test_direct_midpoints_subnormals_signs_and_overflow(self):
        probes=[(0, -173),(1,-25),(3,-25),(-1,-26),(-3,-25),(2049,-11),(2051,-11),
                (65504,0),(65519,0),(65520,0),(1,-150),(-1,-151),((1<<24)+1,-24)]
        for kind in ('half','f32'):
            for n,e in probes:
                value=Fraction(n)*(Fraction(2)**e)
                try:word=oracle(value,kind)
                except ValueError:
                    with self.assertRaises(ValueError):a.round_dyadic(n,e,kind)
                else:self.assertEqual(a.round_dyadic(n,e,kind),word)
        self.assertEqual(a.round_dyadic(0,-173,'half'),0)
        self.assertEqual(a.round_dyadic(-1,-26,'half'),0x8000)
        with self.assertRaises(ValueError):a.round_dyadic(True,0,'half')

    def test_double_rounding_counterexamples(self):
        for kind,p,tail in [('half',11,80),('f32',24,100)]:
            exact=(1<<tail)+(1<<(tail-p))+1
            word=a.round_dyadic(exact,-tail,kind)
            self.assertEqual(word,oracle(Fraction(exact,1<<tail),kind))
            floating=float(Fraction(exact,1<<tail))
            self.assertNotEqual(word,oracle(Fraction.from_float(floating),kind))
            w,x=operands();w[0,:3]=[1.,2.**-p,2.**-tail]
            actual,path=a.apply(w,x,kind)
            self.assertEqual(int(actual[0,0]),word)
            self.assertGreater(path['scalar_blocks'],0)

    def test_integer_grid_exact_values_and_wide_domain(self):
        for dtype in ('<f2','<f4'):
            row=np.array([0.,-0.,1.,-.5,3.25],dtype=dtype)
            values,e=a.grid(row)
            for x,n in zip(row,values,strict=True):
                self.assertEqual(Fraction.from_float(float(x)),Fraction(int(n))*(Fraction(2)**e))
        self.assertIsNone(a.grid(np.array([2.**100,1.],dtype='<f4')))
        for word in (0x7c00,0xfc00,0x7e01):
            with self.assertRaises(ValueError):a.grid(np.array([word],dtype='<u2').view('<f2'))

    def test_threshold_certificates_before_arithmetic(self):
        for bound,route in [(a.F53-1,'fp64'),(a.F53,'fp64'),(a.F53+1,'int64'),(a.I63-1,'int64'),(a.I63,'int64')]:
            w=np.array([bound-1,1],dtype=np.int64);x=np.ones(2,dtype=np.int64);s=stats()
            self.assertEqual(a.product_bound(w,x),bound)
            self.assertEqual(a.certified_integer_dot(w,x,bound,s),bound)
            self.assertEqual(s[route],1)
        w=np.array([a.I63,1],dtype=np.int64);x=np.ones(2,dtype=np.int64)
        self.assertIsNone(a.product_bound(w,x))
        with self.assertRaises(ValueError):a.certified_integer_dot(w,x,a.I63,stats())
        with self.assertRaises(ValueError):a.product_bound(np.array([-(1<<63)],dtype=np.int64),np.ones(1,dtype=np.int64))
        self.assertEqual(a.unsigned_sum(np.array([a.I63,a.I63],dtype=np.uint64),a.U64),2*a.I63)
        self.assertIsNone(a.unsigned_sum(np.array([a.I63,a.I63],dtype=np.uint64)))

    def test_wrong_certificates_and_zero_extreme_operands(self):
        w=np.array([a.I63,3],dtype=np.int64);x=np.array([0,2],dtype=np.int64)
        self.assertEqual(a.product_bound(w,x),6)
        self.assertEqual(a.certified_integer_dot(w,x,6,stats()),6)
        original,half=operands();original[0,0]=np.finfo(np.float32).max
        original[0,1]=1.;half[:]=0;half[0,1]=1.
        for kind in ('half','f32'):
            actual,_=a.apply(original,half,kind)
            self.assertEqual(int(actual[0,0]),oracle(Fraction(1),kind))
        for bad in [5,7,None,True]:
            with self.assertRaises(ValueError):a.certified_integer_dot(w,x,bad,stats())
        x[:]=0;s=stats();self.assertEqual(a.certified_integer_dot(w,x,0,s),0);self.assertEqual(s['zero'],1)

    def test_stripe_fast_exact_fraction_and_zero(self):
        w=np.array([[(i%13-6)/8 for i in range(64)],[(i%7-3)/16 for i in range(64)]],dtype='<f4')
        x=np.array([[(i%9-4)/4 for i in range(64)],[0.]*64],dtype='<f2')
        for kind in ('half','f32'):
            result,s=a.apply(w,x,kind)
            self.assertEqual(result.tolist(),[expected(w,row,kind)for row in x])
            self.assertGreater(s['fp64'],0);self.assertGreater(s['zero'],0)
            self.assertEqual(s['blocked_outputs'],0)

    def test_whole_int64_cancellation_not_f64(self):
        w,x=operands();w[0,:3]=[2.**60,1.,-2.**60]
        result,s=a.apply(w,x)
        self.assertEqual(int(result[0,0]),0x3c00)
        self.assertEqual(s['int64'],1);self.assertEqual(s['blocked_outputs'],0)
        naive=0.
        for v in [2.**60,1.,-2.**60]:naive+=v
        self.assertEqual(naive,0.)

    def test_group_subtotals_different_grids(self):
        w,x=operands(192);w[0,[0,64,128]]=[2.**100,1.,-2.**100]
        result,s=a.apply(w,x)
        self.assertEqual(int(result[0,0]),0x3c00)
        self.assertEqual(s['blocked_outputs'],1);self.assertEqual(s['certified_blocks'],3);self.assertEqual(s['scalar_blocks'],0)
        self.assertEqual(result[0,0],expected(w,x[0],'half')[0])

    def test_bounded_scalar_block_fallback_and_exact_zero(self):
        w,x=operands();w[0,:3]=[2.**100,1.,-2.**100]
        result,s=a.apply(w,x)
        self.assertEqual(int(result[0,0]),0x3c00);self.assertEqual(s['scalar_blocks'],1)
        x[0,1]=0
        result,s=a.apply(w,x);self.assertEqual(int(result[0,0]),0)
        w[0,:3]=[-2.**-126,0.,0.];x[0,:]=0;x[0,0]=np.float16(2.**-24)
        result,_=a.apply(w,x);self.assertEqual(int(result[0,0]),0x8000)
        result,_=a.apply(w,x,'f32');self.assertEqual(int(result[0,0]),0x80000000)
        for kind in ('half','f32'):
            w[:]=0;w[0,:2]=[1.,-1.];x[:]=np.float16(-0.0)
            self.assertEqual(int(a.apply(w,x,kind)[0][0,0]),0)
            x[0,:2]=[1.,1.]
            self.assertEqual(int(a.apply(w,x,kind)[0][0,0]),0)

    def test_typed_shapes_nonfinite_and_destination_overflow(self):
        w,x=operands()
        for left,right in [(w.astype('<f8'),x),(w,x.astype('<f4')),(w[:,::2],x[:,::2]),(w[:0],x),(w,x[:0]),(w,np.ones((65,64),dtype='<f2'))]:
            with self.assertRaises(ValueError):a.apply(left,right)
        with self.assertRaises(ValueError):a.apply(w,x,'bf16')
        for invalid in [math.nan,math.inf,-math.inf]:
            w[0,0]=invalid;x[:]=0
            with self.assertRaises(ValueError):a.apply(w,x)
        w[:]=0;w[0,0]=2.;x[:]=0;x[0,0]=65504
        with self.assertRaises(ValueError):a.apply(w,x)
        self.assertEqual(a.apply(w,x,'f32')[0][0,0],oracle(Fraction(131008),'f32'))

    def test_original_bf16_q4_decode_and_unsupported_intermediate(self):
        codes=np.array([[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]*4],dtype=np.uint8)
        # Unaligned original BF16 metadata is legal; no aligned cast assumption.
        scales=np.frombuffer(b'x'+struct.pack('<H',0x3e80),dtype='<u2',offset=1).reshape(1,1)
        biases=np.frombuffer(b'x'+struct.pack('<H',0xbf00),dtype='<u2',offset=1).reshape(1,1)
        result=a.decode(codes,scales,biases,64)
        self.assertEqual(result.tolist(),[[float(i%16)/4-.5 for i in range(64)]])
        for bad in [0x7f80,0x7fc1]:
            with self.assertRaises(ValueError):a.decode(codes,np.array([[bad]],dtype='<u2'),biases,64)
        with self.assertRaises(ValueError):a.decode(codes,np.array([[0x7f7f]],dtype='<u2'),np.array([[0xff7f]],dtype='<u2'),64)
        with self.assertRaises(ValueError):a.decode(codes,scales.astype('<u4'),biases,64)

    def test_stream_row_partition_truncation_and_owned_outputs(self):
        n,c=3,128;packed=bytes((i*17+3)%256 for i in range(n*c//2))
        scales=struct.pack('<6H',0x3e80,0x3f00,0x3f80,0xbe80,0x3e00,0x3f40)
        biases=struct.pack('<6H',0xbf00,0x3e80,0,0x3f80,0xbe80,0x3e00)
        raw=b'odd'+packed+scales+biases
        layout=a.Layout(n,c,3,3+len(packed),3+len(packed)+len(scales),len(raw))
        one=list(a.read_stripes(io.BytesIO(raw),layout,stripe_rows=1))
        many=list(a.read_stripes(io.BytesIO(raw),layout,stripe_rows=3))
        self.assertTrue(np.array_equal(np.concatenate([x for _,x in one]),many[0][1]))
        selected=list(a.read_stripes(io.BytesIO(raw),layout,1,3,stripe_rows=2))[0][1]
        self.assertTrue(np.array_equal(selected,many[0][1][1:]))
        stream=io.BytesIO(raw);generator=a.read_stripes(stream,layout);_,owned=next(generator);stream.close();generator.close()
        self.assertTrue(owned.flags.owndata);self.assertTrue(np.array_equal(owned,many[0][1]))
        for bad in [a.Layout(n,c,3,3,3+len(packed),len(raw)),a.Layout(n,c,3,3+len(packed),a.MAX_OFFSET,len(raw)),a.Layout(n,65,3,20,30,len(raw))]:
            with self.assertRaises(ValueError):list(a.read_stripes(io.BytesIO(raw),bad))
        with self.assertRaises(ValueError):list(a.read_stripes(io.BytesIO(raw[:-1]),layout))
        with self.assertRaises(ValueError):list(a.read_stripes(io.BytesIO(raw),layout,2,2))
        self.assertTrue(np.array_equal(list(a.read_stripes(io.BytesIO(raw),layout))[0][1],owned))

    def test_row_and_token_partition_invariance(self):
        w=np.array([[(i%11-r)/32 for i in range(128)]for r in range(5)],dtype='<f4')
        x=np.array([[(i%5-t)/8 for i in range(128)]for t in range(3)],dtype='<f2')
        whole,_=a.apply(w,x)
        by_rows=np.concatenate([a.apply(w[start:start+2],x)[0]for start in range(0,5,2)],axis=1)
        by_tokens=np.concatenate([a.apply(w,x[start:start+1])[0]for start in range(3)],axis=0)
        self.assertTrue(np.array_equal(whole,by_rows));self.assertTrue(np.array_equal(whole,by_tokens))
        self.assertEqual(whole.tolist(),[expected(w,row,'half')for row in x])


if __name__=='__main__':
    unittest.main()
