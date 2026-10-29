"""Portable controls; no downloaded artifact, graph export or Apple GPU required."""
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

    def test_writer_raw_pair_gate_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=p.Writer(tmp)
            try:
                x=np.array([[1.,2.]],dtype=np.float16)
                w.emit('case','main',0,'embedding',x);x[0,0]=9
                w.emit('case','wide',0,'embedding',np.array([[1.,2.25]],dtype=np.float16))
                w.gates();group=w.arrays['case.row_0.embedding']
                self.assertEqual(group['envelope'],.25);self.assertEqual(group['gate'],dict(atol=.5,rtol=.001,fatal_reference='main'))
                with self.assertRaises(ValueError):w.emit('case','main',0,'embedding',x)
                with patch.object(p,'MAX_ARRAYS',w.raw.tell()):
                    with self.assertRaises(ValueError):w.emit('other','main',0,'embedding',x)
                with (Path(tmp)/'arrays.bin').open('r+b') as f:f.write(b'xx')
                with self.assertRaises(ValueError):w.gates()
            finally:w.close()
        with tempfile.TemporaryDirectory() as tmp:
            w=p.Writer(tmp)
            try:
                w.emit('case','main',0,'logits',np.array([[0]],dtype=np.float32))
                with self.assertRaises(ValueError):w.gates()
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
