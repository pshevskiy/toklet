"""Synthetic/cached-metadata controls; no real checkpoint or graph execution."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import lfm2_real_half as source

ROOT=Path(__file__).resolve().parents[2]


class RealHalfControls(unittest.TestCase):
    def test_original_metadata_inventory_and_corruption(self):
        fixture=ROOT/'tests/fixtures/lfm2-350m-metadata-v1'
        header=source.strict_json((fixture/'header.json').read_bytes())
        index=source.strict_json((fixture/'index.json').read_bytes())
        logical=source.validate_header(header,index)
        self.assertEqual(len(logical),148)
        self.assertEqual(sum(kind=='affine'for kind,_ in logical.values()),93)
        name='model.layers.0.conv.conv.weight'
        for change in ('dtype','shape','overlap','missing','extra'):
            wrong=copy.deepcopy(header)
            if change=='dtype':wrong[name]['dtype']='F16'
            elif change=='shape':wrong[name]['shape']=[1024,1,3]
            elif change=='overlap':wrong[name]['data_offsets']=[0,6144]
            elif change=='missing':del wrong[name]
            else:wrong['model.lm_head.weight']=copy.deepcopy(wrong['model.embed_tokens.weight'])
            with self.subTest(change=change),self.assertRaises(ValueError):source.validate_header(wrong,index)
        wrong=copy.deepcopy(index);wrong['metadata']['total_parameters']-=1
        with self.assertRaises(ValueError):source.validate_header(header,wrong)

    def test_json_and_pinned_metadata_fail_closed(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'):
            with self.assertRaises(ValueError):source.strict_json(raw)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'sample';path.write_bytes(b'abcdef')
            sha=source.hashlib.sha256(b'abcdef').hexdigest()
            self.assertEqual(source.read_pinned(path,sha,6),b'abcdef')
            for digest,cap in ((sha,5),('0'*64,6)):
                with self.assertRaises(ValueError):source.read_pinned(path,digest,cap)

    def test_exact_stage_routes_and_accounting(self):
        stages=source.stage_inventory(64)
        self.assertEqual(len(stages),293);self.assertEqual(len({s['name']for s in stages}),293)
        self.assertEqual(sum(s['route']=='coupled'for s in stages),13)
        names={s['name']for s in stages}
        self.assertNotIn('embedding_norm',names)
        self.assertNotIn('layers.2.v_norm',names)
        self.assertNotIn('layers.2.post_attention_norm',names)
        self.assertEqual([s['width']for s in stages if s['name']=='layers.2.attention_scores'],[1024])
        self.assertEqual(stages[-1],dict(name='logits',dtype='f32',width=65536,route='coupled'))

    def test_direct_rms_gain_epsilon_and_chunk_boundary(self):
        x=np.zeros((33,1024),dtype='<f2');x[:,0]=1
        gain=np.zeros(1024,dtype='<f4');gain[0]=2**-25
        for profile in ('main','wide'):
            output=source.norm(x,gain,profile).view('<u2')
            self.assertTrue(np.all(output[:,0]==16));self.assertTrue(np.all(output[:,1:]==0))
            self.assertTrue(np.array_equal(output[0],output[32]))
        gain[:]=0
        self.assertTrue(np.all(source.norm(np.ones_like(x),gain,'main')==0))
        with self.assertRaises(ValueError):source.norm(x,np.zeros(1023,dtype='<f4'),'main')

    def test_product_signed_zero_subnormal_and_nonfinite(self):
        words=[0x0000,0x8000,1,1,3,0x7bff]
        left=np.array(words,dtype='<u2').view('<f2')
        right=np.array([-1.,-1.,1.,.5,.5,1.],dtype='<f2')
        self.assertEqual(source.product(left,right).view('<u2').tolist(),[0x8000,0,1,0,2,0x7bff])
        with self.assertRaises(ValueError):source.product(np.array([65504],dtype='<f2'),np.array([2],dtype='<f2'))
        for value in(float('nan'),float('inf')):
            with self.assertRaises(ValueError):source.product(np.array([value],dtype='<f2'),np.zeros(1,dtype='<f2'))

    def test_residual_is_ordinary_not_gemma_clipped(self):
        for profile in ('main','wide'):
            with self.assertRaises(ValueError):source.residual(np.array([64864],dtype='<f2'),np.array([14520],dtype='<f2'),profile)
            x=np.array([1,-1,2**-24],dtype='<f2');y=np.array([-.5,1,2**-24],dtype='<f2')
            self.assertEqual(source.residual(x,y,profile).view('<u2').tolist(),[0x3800,0,2])

    def test_silu_stable_domain_and_store(self):
        x=np.array([-65504.,-0.,0.,65504.],dtype='<f2')
        for profile in ('main','wide'):
            words=source.silu(x,profile).view('<u2').tolist()
            self.assertEqual(words,[0x8000,0x8000,0,0x7bff])
        with self.assertRaises(ValueError):source.silu(np.array([float('inf')],dtype='<f2'),'main')

    def test_convolution_raw_history_not_filtered_output(self):
        u=np.arange(33*1024,dtype=np.int32).reshape(33,1024)%17
        u=u.astype('<f2');coeff=np.zeros((1024,3,1),dtype='<f4')
        z,before,after=source.convolution(u,coeff,'main')
        self.assertTrue(np.all(z==0));self.assertTrue(np.all(before[0].view('<u2')==0))
        self.assertTrue(source.words_equal(after[0,1],u[0]))
        self.assertTrue(source.words_equal(before[32],u[30:32]))
        self.assertTrue(source.words_equal(after[32],u[31:33]))
        self.assertFalse(source.words_equal(after[-1],z[-2:]))

    def test_bcx_intermediate_store_witness_and_components(self):
        result=source.component_controls()
        self.assertEqual(result['bcx_conv_cz_words'],[0x3c02,0x3d02,0x3d03])
        self.assertEqual(result['direct_gain_norm_first_words'],dict(main=16,wide=16))

    def test_gqa_mapping_causality_and_raw_future_scores(self):
        q=np.ones((3,1024),dtype='<f2');k=np.ones((3,512),dtype='<f2');v=np.empty_like(k)
        for head in range(8):v[:,head*64:(head+1)*64]=head+1
        k[2]=2;v[2]=20
        for profile in('main','wide'):
            scores,probabilities,context=source.attention(q,k,v,profile)
            self.assertTrue(np.all(scores[:,0,2]==16)) # Retain actual future raw dot.
            self.assertTrue(np.all(probabilities[:,0,1:].view('<u4')==0))
            for head in range(16):self.assertTrue(np.all(context[0,head*64:(head+1)*64]==head//2+1))
            a,b,c=source.attention(q[:2],k[:2],v[:2],profile)
            self.assertTrue(source.words_equal(scores[:,:2,:2],a))
            self.assertTrue(source.words_equal(probabilities[:,:2,:2],b))
            self.assertTrue(source.words_equal(context[:2],c))

    def test_rotary_position_zero_and_prefix_offsets(self):
        values=((np.arange(64*1024).reshape(64,1024)%17)-8).astype('<f2')
        for profile in('main','wide'):
            f=source.frequencies(profile)
            self.assertEqual(float(f[0]),1.)
            output=source.rotary(values,16,profile)
            self.assertTrue(np.array_equal(output[0],values[0]))
            self.assertTrue(source.words_equal(output[:33],source.rotary(values[:33],16,profile)))
            self.assertFalse(np.array_equal(output[32],values[32]))
        with self.assertRaises(ValueError):source.rotary(values,8,'main')

    def test_projection_selected_head_and_owned_stripes(self):
        fake=object.__new__(source.Source);fake.mapping=io.BytesIO()
        fake.counts=dict(outputs=0,fp64=0,int64=0,zero=0,blocked_outputs=0,certified_blocks=0,scalar_blocks=0,calls=0,max_stripe_fp32_bytes=0)
        fake.layout=lambda name:source.affine.Layout(2,64,0,64,68,72)
        weights=np.zeros((2,64),dtype='<f4');weights[0,0]=2;weights[1,1]=3
        x=np.zeros((1,64),dtype='<f2');x[0,:2]=[4,5]
        closed=[]
        def stripes(*args,**kwargs):
            try:yield 0,weights
            finally:closed.append(True)
        with patch.object(source.affine,'read_stripes',stripes):
            output=fake.project('some.parent',x)
            head=fake.project('model.embed_tokens',x,'f32')
            self.assertEqual(output.tolist(),[[8,15]]);self.assertEqual(head.tolist(),[[8,15]])
            weights[:]=99
            self.assertEqual(output.tolist(),[[8,15]])
            with self.assertRaises(ValueError):fake.project('model.embed_tokens',np.zeros((5,64),dtype='<f2'),'f32')
            with self.assertRaises(ValueError):fake.project('some.parent',x,'f32')
        self.assertEqual(len(closed),2)

    def test_capture_rejects_wrong_shape_duplicate_nonfinite_and_saved_corruption(self):
        # Minimal bounded capture via real serializer with synthetic expected inventory.
        fake=object.__new__(source.Capture)
        with tempfile.TemporaryDirectory()as directory:
            fake.file=(Path(directory)/'arrays').open('x+b',buffering=0)
            fake.progress=(Path(directory)/'progress').open('xb',buffering=0)
            fake.offset=0;fake.journal_bytes=0;fake.arrays={}
            fake.expected={'x':('f16',[2],'actual_input')}
            fake.profile_counts={p:dict(arrays=0,values=0,bytes=0)for p in('main','wide')}
            try:
                x=np.array([1,2],dtype='<f2');fake.add('x','main',x);fake.add('x','wide',x)
                self.assertEqual(fake.arrays['x']['gate'],dict(atol=1e-4,rtol=1e-3))
                self.assertTrue(source.words_equal(fake.read('x','main'),x))
                with self.assertRaises(ValueError):fake.add('x','main',x)
                with self.assertRaises(ValueError):fake.add('unknown','main',x)
                fake.expected['y']=('f16',[2],'actual_input')
                with self.assertRaises(ValueError):fake.add('y','main',np.zeros(1,dtype='<f2'))
                with self.assertRaises(ValueError):fake.add('y','main',np.array([float('nan'),0],dtype='<f2'))
                source.os.pwrite(fake.file.fileno(),b'\0\0',0)
                with self.assertRaises(ValueError):fake.read('x','main')
            finally:fake.close()


if __name__=='__main__':unittest.main()
