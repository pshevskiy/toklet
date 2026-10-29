"""Portable synthetic controls; no target, MLX, Torch, checkpoint or GPU."""
import hashlib
import json
import math
from pathlib import Path
import struct
import unittest
import lfm2_cpu_graph as ref


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=ref.make_artifact()
        cls.matrices,cls.norms,cls.convs,cls.start=ref.decode(cls.raw)

    def test_original_packing_direct_gamma_tied_owner_and_unaligned_payload(self):
        self.assertEqual(len(ref.shapes()),29)
        self.assertEqual((len(self.matrices),len(self.norms),len(self.convs)),(18,9,2))
        self.assertEqual(self.start%4,2)
        n=struct.unpack('<Q',self.raw[:8])[0];header=json.loads(self.raw[8:8+n])
        self.assertEqual(len(header)-1,65)
        self.assertNotIn('lm_head.weight',header)
        name='model.layers.0.conv.in_proj.weight';spec=header[name]
        self.assertEqual(spec['shape'],[768,32])
        parent=self.matrices[name]
        for row,col in [(0,0),(255,255),(256,63),(511,64),(512,127),(767,191)]:
            base=name.removesuffix('.weight');group=row*4+col//64
            code_offset=self.start+spec['data_offsets'][0]+(row*32+col//8)*4
            packed=int.from_bytes(self.raw[code_offset:code_offset+4],'little');q=(packed>>(4*(col%8)))&15
            def coefficient(suffix):
                offset=self.start+header[base+'.'+suffix]['data_offsets'][0]+group*2
                return struct.unpack('<f',b'\0\0'+self.raw[offset:offset+2])[0]
            self.assertEqual(parent[row][col],ref.f32(q*coefficient('scales')+coefficient('biases')))
        self.assertNotEqual(parent[0],parent[256]);self.assertNotEqual(parent[256],parent[512])
        self.assertNotEqual(self.convs['model.layers.0.conv.conv.weight'],self.convs['model.layers.2.conv.conv.weight'])
        for projection in ('k','v'):
            matrix=self.matrices['model.layers.1.self_attn.'+projection+'_proj.weight']
            self.assertNotEqual(matrix[0],matrix[64])
        for values in self.norms.values():self.assertEqual(values[:3],[0.,-.5,1.5])
        for name in self.convs:self.assertEqual(header[name]['shape'],[256,3,1])

    def test_corrupt_inventory_shape_dtype_overlap_nonfinite_and_truncation(self):
        n=struct.unpack('<Q',self.raw[:8])[0];header=json.loads(self.raw[8:8+n]);payload=self.raw[8+n:]
        def changed(h):
            raw=json.dumps(h,separators=(',',':')).encode();return struct.pack('<Q',len(raw))+raw+payload
        for kind in range(5):
            h=json.loads(json.dumps(header));name='model.layers.0.conv.conv.weight'
            if kind==0:del h[name]
            elif kind==1:h[name]['shape']=[256,1,3]
            elif kind==2:h[name]['dtype']='F16'
            elif kind==3:h['lm_head.weight']=h['model.embed_tokens.weight']
            else:h[name]['data_offsets']=h['model.layers.2.conv.conv.weight']['data_offsets']
            with self.assertRaises(ValueError):ref.decode(changed(h))
        for raw in [self.raw[:7],self.raw[:-1],self.raw+b'\0']:
            with self.assertRaises(ValueError):ref.decode(raw)
        h=header['model.embedding_norm.weight'];bad=bytearray(self.raw)
        bad[self.start+h['data_offsets'][0]:self.start+h['data_offsets'][0]+2]=b'\x80\x7f'
        with self.assertRaises(ValueError):ref.decode(bytes(bad))
        with self.assertRaises(ValueError):ref.strict_json(b'{"x":1,"x":2}')

    def test_direct_norm_fp32_pointwise_and_no_gemma_clamp(self):
        out=ref.norm([[1.,2.,3.]],[0.,-.5,1.5])[0]
        d=math.sqrt(14/3+ref.f32(1e-5))
        self.assertEqual(out,[0.,ref.f32(-1/d),ref.f32(4.5/d)])
        self.assertNotEqual(out,ref.norm([[1.,2.,3.]],[1.,.5,2.5])[0])
        for x in [-20.,-2.,0.,2.,20.]:
            ideal=x/(1+math.exp(-x));self.assertLessEqual(abs(ref.silu(x)-ideal),1e-5+1e-4*abs(ideal))
        self.assertEqual(ref.silu(1000.),1000.)
        self.assertEqual(ref.multiply([[2.,-3.]],[[4.,5.]]),[[8.,-15.]])
        self.assertEqual(ref.residual([[65504.]],[[65504.]]),[[131008.]])
        with self.assertRaises(ValueError):ref.residual([[3.4e38]],[[3.4e38]])
        with self.assertRaises(ValueError):ref.norm([[float('nan')]],[1.])

    def test_exact_three_taps_raw_history_and_no_convolution_activation(self):
        u=[[2.,-3.],[6.,-2.],[12.,4.],[20.,-4.],[30.,-6.]];w=[[1.,10.,100.],[3.,-2.,5.]]
        z,before,after=ref.convolution(u,w)
        self.assertEqual(z,[[200.,-15.],[620.,-4.],[1262.,15.],[2126.,-34.],[3212.,-10.]])
        self.assertEqual(before[0],[[0.,0.],[0.,0.]])
        self.assertEqual(after[0],[[0.,0.],u[0]])
        for i in range(1,len(u)):
            self.assertEqual(after[i],[u[i-1],u[i]])
            self.assertEqual(before[i],after[i-1])
        self.assertEqual(ref.multiply(z,[[1.,2.],[-1.,1.],[2.,-2.],[-2.,3.],[3.,-1.]]),[[200.,-30.],[-620.,-4.],[2524.,-30.],[-4252.,-102.],[9636.,10.]])
        self.assertNotEqual(ref.convolution(u,[x[::-1] for x in w])[0],z)
        with self.assertRaises(ValueError):ref.convolution([[float('inf'),0.]],w)

    def test_gqa_two_heads_causal_mask_and_raw_future_scores(self):
        q=[[0.]*4]*3;k=[[0.]*2]*3;v=[[1.,2.],[3.,6.],[5.,10.]]
        scores,prob,context=ref.attention(q,k,v,4,2,1)
        self.assertEqual(context[0],[1.,1.,2.,2.]);self.assertEqual(context[1],[2.,2.,4.,4.])
        for h in range(4):
            self.assertEqual(prob[h][0],[1.,0.,0.]);self.assertEqual(prob[h][1],[.5,.5,0.])
            self.assertEqual([struct.pack('<f',x) for x in prob[h][0][1:]],[b'\0'*4]*2)
        q=[[1.]*4]*3;k=[[1.]*2,[2.]*2,[3.]*2]
        a=ref.attention(q,k,v,4,2,1);k[-1]=[100.,200.];v[-1]=[-7.,-8.]
        b=ref.attention(q,k,v,4,2,1)
        self.assertEqual(a[2][:2],b[2][:2]);self.assertNotEqual(a[0][0][0][-1],b[0][0][0][-1]);self.assertEqual(b[1][0][0][-1],0.)
        k[-1][0]=float('nan')
        with self.assertRaises(ValueError):ref.attention(q,k,v,4,2,1)

    def test_rotary_f32_products_offsets_head_mapping_and_shape(self):
        x=[[1.,2.,3.,4.,5.,6.,7.,8.]]
        self.assertEqual(ref.rotary(x,2,4),x)
        y=ref.rotary(x,2,4,1)[0]
        co=ref.f32(math.cos(1.));si=ref.f32(math.sin(1.))
        self.assertEqual(y[0],ref.f32(ref.f32(1*co)-ref.f32(3*si)))
        self.assertEqual(y[2],ref.f32(ref.f32(3*co)+ref.f32(1*si)))
        self.assertEqual(ref.rotary([x[0][4:]],1,4,1)[0],y[4:])
        last=ref.rotary([[.25]*64],1,64,11)[0]
        self.assertLess(abs(math.fsum(v*v for v in last)-4.),1e-5)
        for args in [(x,2,3,0),(x,2,4,12),(x,2,4,-1)]:
            with self.assertRaises(ValueError):ref.rotary(*args)

    def test_fixed_stage_inventory_causal_counterexamples_and_state_layout(self):
        self.assertEqual([len(ref.contracts(len(ids))) for _,ids in ref.CASES],[63]*4)
        expected=ref.contracts(12)
        self.assertEqual(expected['layers.1.kv_keys_after'],[12,2,12,64])
        def zero():return {name:[0.]*math.prod(shape) for name,shape in expected.items()}
        a=zero();b=zero();b['layers.1.attention_scores'][11]=3.
        ref.causal_equal(a,b,12) # unobserved future diagnostic score is not causal output
        b['layers.1.attention_scores'][0]=1.
        with self.assertRaises(ValueError):ref.causal_equal(a,b,12)
        for stage in ['layers.0.u','layers.0.history_after','layers.1.attention_probabilities','layers.1.kv_keys_after','logits']:
            b=zero();b[stage][0]=1.
            with self.assertRaises(ValueError):ref.causal_equal(a,b,12)
        rows=[[float(i*128+j) for j in range(128)] for i in range(3)]
        states=ref.kv_snapshots(rows)
        for t in range(3):
            for h in range(2):
                for k in range(12):self.assertEqual(states[t][h][k],rows[k][h*64:(h+1)*64] if k<=t else [0.]*64)
        for ids in [[],[-1],[64],[True],list(range(13))]:
            with self.assertRaises(ValueError):ref.forward({}, {}, {}, ids)

    def test_projection_orientation_and_fixture_size_bounds(self):
        self.assertEqual(ref.linear([[1.,2.,3.]],[[.5,-1.,2.],[2.,0.,-1.]]),[[4.5,-1.]])
        self.assertEqual(ref.linear([[1.,1.,1.]],[[2.**60,1.,-2.**60]]),[[1.]])
        with self.assertRaises(ValueError):ref.linear([[1.,2.]],[[1.]])
        self.assertLess(len(self.raw),ref.MAX_ARTIFACT)
        total=sum(math.prod(shape)*4 for _,ids in ref.CASES for shape in ref.contracts(len(ids)).values())
        self.assertLess(total,ref.MAX_ARRAYS)
        self.assertEqual(ref.CPU_GATE,dict(atol=1e-5,rtol=1e-4))


if __name__=='__main__':unittest.main()
