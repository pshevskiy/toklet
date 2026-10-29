"""Portable controls; no downloaded artifact, graph export or Apple GPU required."""
import copy
import io
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import gemma3_real_half as g
import mlx_gemma3_real as p
import gemma3_half_rms as old_rms
import gemma3_half_pointwise as old_pointwise


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def word(x):return struct.unpack('<H',struct.pack('<e',x))[0]
def words(a):return a.astype('<f2',copy=False).view('<u2').reshape(-1).tolist()
def bfword(x):return struct.unpack('<I',struct.pack('<f',x))[0]>>16


class RealGemmaControls(unittest.TestCase):
    def test_original_rows_unaligned_and_nibbles(self):
        code=np.array([[sum(((r*3+j*8+k)%16)<<(4*k) for k in range(8)) for j in range(8)] for r in range(3)],dtype='<u4')
        scales=np.array([[bfword(v)] for v in (.125,.25,.5)],dtype='<u2')
        biases=np.array([[bfword(v)] for v in (-.5,-1.,-2.)],dtype='<u2')
        data=b'abcde'+code.tobytes()+b'z'
        stream=io.BytesIO(data);desc=dict(dtype='U32',shape=[3,8],data_offsets=[0,96])
        selected=p.rows_from_stream(stream,5,desc,1,3)
        self.assertTrue(selected.flags.owndata or selected.base.flags.owndata)
        np.testing.assert_array_equal(selected,code[1:3]);stream.close()
        actual=g.decode(selected,scales[1:3],biases[1:3],64)
        expected=np.array([[f32(((r*3+j)%16)*(.125,.25,.5)[r]+(-.5,-1.,-2.)[r]) for j in range(64)] for r in (1,2)],dtype=np.float32)
        np.testing.assert_array_equal(actual,expected)
        for begin,end in ((-1,1),(1,1),(0,4),(True,2)):
            with self.assertRaises(ValueError):p.rows_from_stream(io.BytesIO(data),5,desc,begin,end)
        with self.assertRaises(ValueError):p.rows_from_stream(io.BytesIO(data[:-2]),5,desc,1,3)
        bad=dict(desc,data_offsets=[0,95])
        with self.assertRaises(ValueError):p.rows_from_stream(io.BytesIO(data),5,bad,0,1)
        with self.assertRaises(ValueError):g.decode(code,scales,biases,32)
        malformed=scales.copy();malformed[0,0]=0x7f80
        with self.assertRaises(ValueError):g.decode(code,malformed,biases,64)

    def test_packed_head_is_original_f32_and_detached(self):
        matrix=np.array([[1+2**-12],[2]],dtype=np.float32);matrix.flags.writeable=False
        lookup=g.half(matrix[[0]])
        self.assertEqual(float(lookup[0,0]),1.)
        for profile in ('main','wide'):
            output=g.projection(np.array([[1]],dtype=np.float16),matrix,profile,output_float=True)
            self.assertEqual(float(output[0,0]),1+2**-12)
            self.assertFalse(np.shares_memory(output,matrix))
            np.testing.assert_array_equal(g.projection(np.array([[1]],dtype=np.float16),matrix,profile),np.array([[1,2]],dtype=np.float16))
        self.assertFalse(matrix.flags.writeable)
        with self.assertRaises(ValueError):g.projection(lookup,matrix.astype(np.float64),'main')

    def test_rms_column_order_and_exact_witnesses(self):
        for width in (32,640):
            source=[word((j%9-4)/8) for j in range(width*2)]
            x=np.array(source,dtype='<u2').view('<f2').reshape(2,width)
            scale=np.array([1+2**-12 if j%3==0 else .75 for j in range(width)],dtype=np.float32)
            expected=old_rms.rms(source,scale.tolist(),2,width,width,'main')
            self.assertEqual(words(g.norm(x,scale,'main')),expected)
        x=np.ones((1,32),dtype=np.float16);x[0,0]=2
        scale=np.ones(32,dtype=np.float32);scale[0]=1+2**-12
        for profile in ('main','wide'):
            self.assertEqual(words(g.norm(x,scale,profile))[0],0x3fa7)
            self.assertEqual(words(g.norm(x,scale.astype(np.float16).astype(np.float32),profile))[0],0x3fa6)
            for sign,want in ((0x0001,0x03e8),(0x8001,0x83e8)):
                tiny=np.zeros((1,32),dtype=np.float16);tiny.view('<u2')[0,0]=sign
                self.assertEqual(words(g.norm(tiny,np.ones(32,dtype=np.float32),profile)),[want]+[0]*31)

    def test_gelu_store_and_embedding_scale(self):
        for profile in ('main','wide'):
            activation=g.gelu(np.array([[.25]],dtype=np.float16),profile)
            product=g.half(activation.astype(g.dtype(profile))*g.dtype(profile)(5))
            self.assertEqual((words(activation)[0],words(product)[0]),(0x30ca,0x39fc))
            self.assertEqual((words(activation)[0],words(product)[0]),old_pointwise.gelu_product(word(.25),word(5),profile))
            self.assertNotEqual(words(product)[0],0x39fd)
            raw=[0x0001,0x8001,word(.25),word(1),word(-1)]
            x=np.array(raw,dtype='<u2').view('<f2')
            scaled=g.half(x.astype(g.dtype(profile))*g.dtype(profile)(25.25))
            self.assertEqual(words(scaled),[old_pointwise.scale_word(w,25.25,profile) for w in raw])
        # Actual source BF16-rounded sqrt640, not a narrowed FP16 coefficient.
        self.assertEqual(bfword(math.sqrt(640)),bfword(25.25))

    def test_gqa_one_key_visibility_and_future(self):
        for profile in ('main','wide'):
            q=np.ones((1,4,32),dtype=np.float16);k=np.ones((1,1,32),dtype=np.float16)
            v=np.arange(32,dtype=np.float16).reshape(1,1,32)
            _,prob,ctx=g.attention(q,k,v,None,profile)
            np.testing.assert_array_equal(prob,np.ones((4,1,1),dtype=np.float32))
            np.testing.assert_array_equal(ctx,np.tile(v.reshape(1,32),(1,4)))
            q=np.zeros((7,4,32),dtype=np.float16);k=np.zeros((7,1,32),dtype=np.float16)
            v=np.broadcast_to(np.arange(7,dtype=np.float16)[:,None,None],(7,1,32)).copy()
            _,prob,ctx=g.attention(q,k,v,4,profile)
            for row in range(7):
                lo=max(0,row-3);expected=np.float16(sum(range(lo,row+1))/(row-lo+1))
                np.testing.assert_array_equal(ctx[row],np.full(128,expected,dtype=np.float16))
                self.assertTrue(np.all(prob[:,row,:lo]==0) and np.all(prob[:,row,row+1:]==0))
            changed=v.copy();changed[-1]=123
            _,_,again=g.attention(q,k,changed,4,profile)
            np.testing.assert_array_equal(ctx[:-1],again[:-1])
        mask=g.visibility(544,512)
        for row,start in ((511,0),(512,1),(513,2),(543,32)):
            self.assertEqual(np.flatnonzero(mask[row]).tolist(),list(range(start,row+1)))
        self.assertEqual(np.flatnonzero(g.visibility(544,None)[543]).tolist(),list(range(544)))

    def test_rotary_position_zero_and_invalid_storage(self):
        for dim in (32,256):
            x=np.array([[(j%13-6)/8 for j in range(dim)]],dtype=np.float16).reshape(1,1,dim)
            for profile in ('main','wide'):
                np.testing.assert_array_equal(g.rotary(x,10000,profile),x)
        for profile in ('unknown',None):
            with self.assertRaises(ValueError):g.dtype(profile)
        with self.assertRaises(ValueError):g.half(np.array([float('inf')]))
        with self.assertRaises(ValueError):g.half(np.array([65536.]))
        with self.assertRaises(ValueError):g.visibility(545,512)
        class Empty: vocab=64
        for tokens,selected in (([],[0]),([64],[0]),([1],[1]),([1],[0,0])):
            with self.assertRaises(ValueError):g.forward(Empty(),tokens,selected,'main',lambda *args:None)

    def test_writer_legacy_limits_wide_identity_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=p.Writer(tmp)
            try:
                x=np.array([[1.,2.]],dtype=np.float16)
                w.emit('case','main',0,'embedding',x);x[0,0]=9
                w.emit('case','wide',0,'embedding',np.array([[1.,2.25]],dtype=np.float16))
                name='case.row_0.embedding'
                legacy={'arrays':copy.deepcopy(w.arrays)}
                gate=dict(atol=.125,rtol=.001,fatal_reference='main')
                legacy['arrays'][name]['gate']=dict(gate)
                limits={'gates':{name:dict(gate)}}
                w.gates(legacy,limits);group=w.arrays[name]
                self.assertEqual(group['envelope'],.25)
                # The new envelope would imply .5 under old formula: forbidden.
                self.assertEqual(group['gate'],gate)
                self.assertEqual(group['gate_basis'],'legacy_fixed_limits_new_native_store_center_v1')
                changed=copy.deepcopy(limits);changed['gates'][name]['atol']=.5
                with self.assertRaisesRegex(ValueError,'literal numerical'):w.gates(legacy,changed)
                changed=copy.deepcopy(legacy);changed['arrays'][name]['wide']['sha256']='0'*64
                with self.assertRaisesRegex(ValueError,'WIDE identity'):w.gates(changed,limits)
                changed=copy.deepcopy(legacy);changed['arrays'][name]['main']['shape']=[2,1]
                with self.assertRaisesRegex(ValueError,'shape/store'):w.gates(changed,limits)
                with self.assertRaisesRegex(ValueError,'inventory'):w.gates(legacy,{'gates':{}})
                with self.assertRaises(ValueError):w.emit('case','main',0,'embedding',x)
                with patch.object(p,'MAX_ARRAYS',w.raw.tell()):
                    with self.assertRaises(ValueError):w.emit('other','main',0,'embedding',x)
                # Remove the failed append's empty map to exercise actual raw hash.
                w.arrays.pop('other.row_0.embedding',None)
                with (Path(tmp)/'arrays.bin').open('r+b') as f:f.write(b'xx')
                with self.assertRaisesRegex(ValueError,'Raw capture changed'):w.gates(legacy,limits)
            finally:w.close()
        with tempfile.TemporaryDirectory() as tmp:
            w=p.Writer(tmp)
            try:
                w.emit('case','main',0,'logits',np.array([[0]],dtype=np.float32))
                name='case.row_0.logits'
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    w.gates({'arrays':{name:{}}},{'gates':{name:{}}})
            finally:w.close()

    def test_fixed_inventory_missing_extra_shape_and_memory(self):
        matrices,norms,physical=p.specifications()
        self.assertEqual((len(matrices),len(norms),len(physical)),(127,109,490))
        self.assertEqual(sum(r*c*4 for r,c in matrices.values()),1072168960)
        self.assertEqual(sum(norms.values())*4,223744)
        cases=[dict(name='a',tokens=[2]*6,selected_rows=[0,5]),dict(name='b',tokens=[2]*11,selected_rows=[0,10]),dict(name='w',tokens=[2]*544,selected_rows=[511,512,513,543])]
        stages=['embedding_lookup','embedding','final_norm','logits']+[f'layers.{i}.{s}' for i in range(18) for s in p.LAYER_STAGES]
        arrays={}
        def add(name,shape,dtype):
            d=dict(shape=shape,dtype=dtype,bytes=math.prod(shape)*(2 if dtype=='u16le' else 4))
            arrays[name]={'main':dict(d),'wide':dict(d)}
        for case in cases:
            for row in case['selected_rows']:
                for stage in stages:add(f'{case["name"]}.row_{row}.{stage}',*p.capture_shape(stage,len(case['tokens'])))
        teacher=[dict(prompt='a',steps=[dict(prefix=[2]*6)]),dict(prompt='b',steps=[dict(prefix=[2]*11)])]
        add('teacher.a.step_0.row_5.logits',[1,262144],'f32le');add('teacher.b.step_0.row_10.logits',[1,262144],'f32le')
        self.assertEqual(p.validate_inventory(arrays,cases,teacher),dict(graph_stage_groups=3200,teacher_stage_groups=2,total_arrays=6404))
        first=next(iter(arrays));saved=arrays.pop(first)
        with self.assertRaises(ValueError):p.validate_inventory(arrays,cases,teacher)
        arrays[first]=saved;arrays['unexpected']=saved
        with self.assertRaises(ValueError):p.validate_inventory(arrays,cases,teacher)
        del arrays['unexpected'];arrays[first]['wide']['shape']=[1]
        with self.assertRaises(ValueError):p.validate_inventory(arrays,cases,teacher)

    def test_small_graph_selected_capture_and_causal_prefix(self):
        class Tiny:
            vocab=64;layers=1;dim=32
            matrices={};norms={}
        w=Tiny();w.matrices={};w.norms={}
        shapes={'model.embed_tokens':(64,64),'model.layers.0.self_attn.q_proj':(128,64),
                'model.layers.0.self_attn.k_proj':(32,64),'model.layers.0.self_attn.v_proj':(32,64),
                'model.layers.0.self_attn.o_proj':(64,128),'model.layers.0.mlp.gate_proj':(128,64),
                'model.layers.0.mlp.up_proj':(128,64),'model.layers.0.mlp.down_proj':(64,128)}
        for name,(rows,cols) in shapes.items():
            a=np.fromfunction(lambda r,c:((r*3+c*5)%11-5)/1024,(rows,cols),dtype=int).astype(np.float32)
            a.flags.writeable=False;w.matrices[name]=a
        for name in ('input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm'):
            w.norms['model.layers.0.'+name+'.weight']=np.ones(64,dtype=np.float32)
        for name in ('q_norm','k_norm'):w.norms['model.layers.0.self_attn.'+name+'.weight']=np.ones(32,dtype=np.float32)
        w.norms['model.norm.weight']=np.ones(64,dtype=np.float32)
        for profile in ('main','wide'):
            captures={}
            out=g.forward(w,[1,7,13],[0,2],profile,lambda row,stage,a:captures.__setitem__((row,stage),a))
            self.assertEqual(len(captures),52);self.assertEqual(out.shape,(2,64))
            expected={'embedding_lookup','embedding','final_norm','logits'}|{'layers.0.'+s for s in p.LAYER_STAGES}
            self.assertEqual({name for row,name in captures},expected)
            self.assertTrue(all(a.flags.owndata for a in captures.values()))
            np.testing.assert_array_equal(out[0],captures[(0,'logits')][0])
            altered={};g.forward(w,[1,7,17],[0,2],profile,lambda row,stage,a:altered.__setitem__((row,stage),a))
            np.testing.assert_array_equal(captures[(0,'logits')],altered[(0,'logits')])
            # Output and capture are detached: no retained full vocabulary matrix view.
            out[0,0]=123
            self.assertNotEqual(float(captures[(0,'logits')][0,0]),123)

    def test_residual_clip_extremes_and_saved_witness(self):
        # Independently specified FP32 add/finite clamp/nearest-even Half oracle.
        # The real witness is retained in the failed-v1 evidence, not a fitted gate.
        pairs=[(64864.,14520.),(-64864.,-14520.),(65504.,65504.),(-65504.,-65504.),
               (65504.,16.),(-65504.,-16.),(32752.,32752.),(-32752.,-32752.),
               (65504.,-65504.),(2**-24,2**-24),(-2**-24,-2**-24),
               (1.,.5),(1.,2**-11),(1.,-2**-11),(.25,-.125),(0.,-0.),(-0.,-0.)]
        left=np.array([a for a,b in pairs],dtype=np.float16)
        right=np.array([b for a,b in pairs],dtype=np.float16)
        expected=[]
        for a,b in pairs:
            total=f32(a+b)
            if total>65504.:total=65504.
            elif total< -65504.:total=-65504.
            expected.append(word(total))
        actual=g.residual(left,right)
        self.assertEqual(words(actual),expected)
        self.assertEqual(words(actual)[:6],[0x7bff,0xfbff,0x7bff,0xfbff,0x7bff,0xfbff])
        self.assertEqual(words(actual)[9:11],[0x0002,0x8002])
        self.assertEqual(words(actual)[-2:],[0x0000,0x8000])
        # Finite nonoverflow behavior preserves the previous checked operation.
        inside=np.abs(left.astype(np.float32)+right.astype(np.float32))<=65504.
        np.testing.assert_array_equal(actual[inside].view('<u2'),g.half(left[inside].astype(np.float32)+right[inside].astype(np.float32)).view('<u2'))

    def test_residual_rejects_invalid_before_clamp(self):
        ordinary=np.ones((2,3),dtype=np.float16)
        for bad_word in (0x7c00,0xfc00,0x7e00,0x7d00):
            bad=ordinary.copy();bad.view('<u2')[0,1]=bad_word
            for left,right in ((bad,ordinary),(ordinary,bad)):
                with self.assertRaises(ValueError):g.residual(left,right)
        for left,right in ((ordinary.astype(np.float32),ordinary),(ordinary,ordinary.astype(np.float32)),(ordinary,ordinary[:1])):
            with self.assertRaises(ValueError):g.residual(left,right)
        # Saturation is local to residual, never a globally permissive Half cast.
        with self.assertRaises(ValueError):g.half(np.array([79384.],dtype=np.float32))

    def test_binary64_destination_cast_is_direct(self):
        # Certification endpoint conversion must not double-round through FP32.
        endpoints=np.array([1+2**-11+2**-40,-1-2**-11-2**-40],dtype=np.float64)
        self.assertEqual(words(endpoints.astype(np.float16)),[0x3c01,0xbc01])
        self.assertEqual(words(endpoints.astype(np.float32).astype(np.float16)),[0x3c00,0xbc00])
        floats=np.array([1+2**-24+2**-48,-1-2**-24-2**-48],dtype=np.float64).astype(np.float32)
        self.assertEqual(floats.view(np.uint32).tolist(),[0x3f800001,0xbf800001])

    def test_certified_projection_midpoints_cancellation_and_direct_rounding(self):
        # Literal exact-dyadic answers: a tiny positive tail beyond a midpoint
        # is lost by binary64 summation but must choose the upper destination.
        examples=[([1.,2**-11,2**-80],False,word(1+2**-10)),
                  ([1.,2**-11,0.],False,word(1.)),
                  ([1+2**-10,2**-11,0.],False,word(1+2**-9)),
                  ([-1.,-2**-11,-2**-80],False,word(-1-2**-10)),
                  ([2**100,1.,-2**100],False,word(1.)),
                  ([1.,2**-24,2**-100],True,0x3f800001),
                  ([1.,2**-24,0.],True,0x3f800000),
                  ([1+2**-23,2**-24,0.],True,0x3f800002),
                  ([-1.,-2**-24,-2**-100],True,0xbf800001),
                  ([2**100,-1.,-2**100],True,0xbf800000)]
        for coefficients,output_float,expected in examples:
            g.reset_projection_audit()
            x=np.ones((1,3),dtype=np.float16);matrix=np.array([coefficients],dtype=np.float32)
            got=g.projection(x,matrix,'main',output_float=output_float)
            raw=int(got.view(np.uint32 if output_float else np.uint16)[0,0])
            self.assertEqual(raw,expected)
            self.assertEqual(g.PROJECTION_AUDIT['exact_fallback'],1)
            # An independent Fraction check verifies nearest-even directly;
            # no production integer-unit/rounding helper is the test oracle.
            from fractions import Fraction
            exact=sum((Fraction(float(a))*Fraction(float(b)) for a,b in zip(x[0],matrix[0])),Fraction())
            center=float(got[0,0]);previous=float(np.nextafter(got[0,0],got.dtype.type(-np.inf)))
            following=float(np.nextafter(got[0,0],got.dtype.type(np.inf)))
            distance=abs(exact-Fraction(center))
            self.assertLessEqual(distance,abs(exact-Fraction(previous)))
            self.assertLessEqual(distance,abs(exact-Fraction(following)))
            if distance in (abs(exact-Fraction(previous)),abs(exact-Fraction(following))):self.assertEqual(raw&1,0)

    def test_certified_projection_subnormal_signed_zero_and_overflow(self):
        for sign in (1.,-1.):
            for tail,want in ((0.,0),(2**-100,1)):
                x=np.array([[2**-24,2**-24]],dtype=np.float16)
                matrix=np.array([[sign*.5,sign*tail]],dtype=np.float32)
                actual=g.projection(x,matrix,'main')
                self.assertEqual(words(actual),[want|(0x8000 if sign<0 else 0)])
            for tail,want in ((0.,0),(2**-149,1)):
                matrix=np.array([[sign*2**-126,sign*tail]],dtype=np.float32)
                actual=g.projection(x,matrix,'main',output_float=True)
                self.assertEqual(actual.view(np.uint32).tolist(),[[want|(0x80000000 if sign<0 else 0)]])
        zero=g.projection(np.array([[-0.,0.]],dtype=np.float16),np.ones((1,2),dtype=np.float32),'main')
        self.assertEqual(words(zero),[0])
        for coefficients in ([65504.,16.],[-65504.,-16.]):
            with self.assertRaisesRegex(ValueError,'projection store'):
                g.projection(np.ones((1,2),dtype=np.float16),np.array([coefficients],dtype=np.float32),'main')
        with self.assertRaisesRegex(ValueError,'projection store'):
            g.projection(np.array([[2.]],dtype=np.float16),np.array([[np.finfo(np.float32).max]],dtype=np.float32),'main',output_float=True)
        for bad in (np.nan,np.inf):
            with self.assertRaises(ValueError):g.projection(np.ones((1,2),dtype=np.float16),np.array([[bad,1.]],dtype=np.float32),'main')
        with self.assertRaises(ValueError):g.projection(np.ones((1,2049),dtype=np.float16),np.ones((1,2049),dtype=np.float32),'main')

    def test_projection_stripes_certificate_owner_and_basis(self):
        g.reset_projection_audit()
        x=np.array([[1.,2.,3.],[-2.,1.,.5]],dtype=np.float16)
        matrix=np.array([[.25,.5,1.],[1.,2.,3.],[-1.,.25,.5],[.125,-.5,2.],[.75,1.,-.125]],dtype=np.float32)
        matrix.flags.writeable=False
        with patch.object(g,'PROJECTION_STRIPE',2):out=g.projection(x,matrix,'main')
        from fractions import Fraction
        expected=np.array([[float(sum((Fraction(float(a))*Fraction(float(b))for a,b in zip(row,w)),Fraction()))for w in matrix]for row in x],dtype=np.float16)
        np.testing.assert_array_equal(out,expected)
        self.assertEqual(g.PROJECTION_AUDIT['stripes'],3)
        self.assertEqual(g.PROJECTION_AUDIT['max_converted_bytes'],2*3*8)
        self.assertEqual(g.PROJECTION_AUDIT['certified'],10)
        self.assertEqual(g.PROJECTION_AUDIT['exact_fallback'],0)
        self.assertTrue(out.flags.owndata);self.assertFalse(np.shares_memory(out,matrix))
        self.assertFalse(matrix.flags.writeable)

    def test_reference_attention_explicit_serial_cancellation(self):
        # Exact products2^24,+1,-2^24: serialFP32 is0, accurate sum is1.
        q=np.zeros((3,4,256),dtype=np.float16);k=np.zeros((3,1,256),dtype=np.float16)
        q[:,:,:3]=[4096.,1.,4096.];k[:,:,:3]=[4096.,1.,-4096.]
        v=np.zeros_like(k);v[:,:,0]=[[2048.],[1.],[-2048.]]
        scores,prob,context=g.attention(q,k,v,None,'main')
        self.assertTrue(np.all(scores==0))
        wide,_,_=g.attention(q,k,v,None,'wide')
        self.assertTrue(np.all(wide==np.float32(1/16)))
        for row in range(3):
            mass=f32(row+1);expected=f32(1/mass)
            for head in range(4):
                self.assertEqual(prob[head,row,:row+1].tolist(),[expected]*(row+1))
                self.assertEqual(prob[head,row,row+1:].tolist(),[0.]*(2-row))
                total=0.
                for key in range(row+1):total=f32(total+f32(expected*float(v[key,0,0])))
                self.assertEqual(word(float(context[row,head*256])),word(total))

    def test_real_first_layer_midpoint_input_semantics(self):
        # Source-only original embedding row, independently reconstructed before
        # successor design. This is an arithmetic branch control, not target data.
        original_hex='d5b4d5b8d5383bc0d534403b403bd5b840bf0b3ed538d5b40000d538d5b8d534d534d534d53c0000d5b40000d534d538d534403bd534d5b8d5b8d538d534d5340000d538d5b8d5b840bbd5bc40bbd538403bd540d534d5bcd5b400000000d538d5b80000d5b80b3e0000d5380000d53840bbd5b8d5b80bbed5b40000d53cd5b4501e93b9b8353b40b835501e31bcb8353e3cac3986b5b83599bd93b9501e50c23e3c501eb83586b5501e86b5b835b835b835b8353b4086b53e3cb835ac3993b90d3fb83593b901bf99bdb83593b93e3c501eb835501e501ea63d01bfac39ac3931bcac3986b586b5b835ac3931bc93b9501eb83586b5b8353e3ca63d501e31bc083d9c44b5bab5ba0000b5b60000b536084100000000b5b60000b5b6b5ba083db5bab5bab5b6b5be0000b53608bdb5360000b536b536b53ab5360000b53ab53e0000b5b6b53ade41b536b5b6b5bab5b60000b53608bd0000b53eb5b6b5b60000b53ab5b6b5ba0000b5bab536b5b6b53a0000083db5360000b5b6b5ba000000003940c2b6c23ac2b6c23ac23a11bdc236e941000011bd0000c2b6000011bdc2bac23ac23ac236c2b6c236c23a113dc23ac23a3940c2bac2bac2b6c236113dc2ba113d0000c236c23ac2c20000c2b6c23ec236c2b6c2bac23611bd0000c2b6e941c236c2bac2b60000c23ac23ac23a11bdc23600000000c23a11bd0000c2b6000050ba823a502250ba5022823a5022763e5022502250ba823a5022823a50225022823a763e502250ba823a5022823a763e823a5022502250225022763e5dbe5022502234c95022763ec9c0823a823a763e50ba5022823a5dbe5022502250225dbe502250ba823a823a823a5022823a5dbe823a502250ba50ba763e502250ba502224342434373c3138509e69ba44bc313857b43138243431384ab844bc57b463be373c44bc509e313857b444bc313863be57b43138509e69ba4ab8373c509e69ba54bd3138503a63be57b4373c44bc44bc503a373c2434473d473d563e563e24344ab824343138c240243457b457b463be57b457b43138503a243463be509e313810c1de35bc1c5b40de35de35b8b5bc1cde3556c0c2b954bcbc1c56c0de35d539d539b8b5de35bc1c82c2de35b8b53abfd539b8b53abfc2b9de35d53954bcd03d5d3cde35bc1c5b40b8b5b8b5bc1c3abfd03dde35de35c2b9bc1cbc1c54bcd539de35de35c7bd5d3cb8b5b8b554bcde35d539b8b554bcd03dbc1cde355b40b8b5b5369cb6501a2b40501a9cb6af3afbbca5bea2ba9cb6af3a501aaf3a501aa2bab536a2ba9cb6af3aac3eb536b5369cb6a2bab536ac3eb536af3a9cb6a2bab536013db536501a013d501ab536b536501ab536b536501a9cb6b53628c0af3a501a013daf3a501afdc09cb628c02b40501aa2bafbbcb5367f43501ab536501a501a14ba1e361e361a3a1a3a1e361a3a933c933c11b69f3f501690bc11b6501611b614ba11b650161e36db42501690bc501650161e36501691c014ba11b614ba50161a3a9f3f11b61e361a3a11b61a3a501611b690bc1e3614ba11b611b61e36933c14ba933c11b61e36501614ba9f3f14ba5016933c11b650161e361e361a3a1e3614b9509a1bb51bb51bb50839083901358f3b8f3b013514b98f3b0b3d14b90b3d9bbb0839509a8f3b509a9bbb55be01359bbb923f55be01351bb59bbb4e3e08390135013511bd14b9013501351bb5013501350135509a9bbb01351bb511bd11bd013508390135923f1bb59bbb509a14b90839509a509a14b9509a1bb508395042'
        x=np.frombuffer(bytes.fromhex(original_hex),dtype='<f2').copy().reshape(1,640)
        scale=np.ones(640,dtype=np.float32);scale[3]=41.5
        self.assertEqual(float(x[0,3]),-2.115234375)
        self.assertEqual(words(g.norm(x,scale,'main'))[3],0xd54e)
        self.assertEqual(words(g.norm(x,scale,'wide'))[3],0xd54f)
        # Independent scalar structFP32 RMS produces an exact Half midpoint.
        total=correction=0.
        for value in x[0]:
            square=f32(float(value)*float(value));following=f32(total+square)
            error=f32(f32(total-following)+square)if abs(total)>=abs(square)else f32(f32(square-following)+total)
            correction=f32(correction+error);total=following
        inverse=f32(1/f32(math.sqrt(f32(f32(f32(total+correction)/640)+f32(1e-6)))))
        pre=f32(f32(float(x[0,3])*inverse)*41.5)
        self.assertEqual(pre,-84.90625);self.assertEqual(word(pre),0xd54e)

    def test_cleanup_preserves_primary_and_releases_owners(self):
        class Broken(io.BytesIO):
            def close(self):super().close();raise OSError('close failure')
        weights=p.Weights('.',{});weights.stream=Broken(b'x');weights.matrices['x']=np.zeros(1,dtype=np.float32);weights.norms['x']=np.ones(1,dtype=np.float32)
        original=ValueError('primary numerical failure')
        with patch.object(weights,'endpoints',side_effect=RuntimeError('endpoint failure')):
            self.assertFalse(weights.__exit__(type(original),original,None))
        self.assertEqual(str(original),'primary numerical failure');self.assertEqual(len(original.__notes__),2)
        self.assertFalse(weights.matrices or weights.norms);self.assertIsNone(weights.stream)
        with self.assertRaisesRegex(RuntimeError,'cleanup only'):p.preserve_cleanup(None,[RuntimeError('cleanup only')])
        with tempfile.TemporaryDirectory() as tmp:
            bad=p.Weights(tmp,{'missing':dict(bytes=1,sha256='0'*64)})
            with self.assertRaises(FileNotFoundError):bad.__enter__()
            self.assertFalse(bad.matrices or bad.norms);self.assertIsNone(bad.stream)


if __name__=='__main__':unittest.main()
