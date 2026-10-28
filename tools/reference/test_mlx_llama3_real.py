"""Proposed portable controls. No real weights, downloads, Metal or MLX execution."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import mlx_llama3_real as oracle
import llama3_fp16 as graph


def manifest():
    arrays={};cursor=0
    for name,(shape,dtype) in oracle.expected_arrays().items():
        size=int(np.prod(shape))*(2 if dtype=='<f2' else 4)
        arrays[name]=dict(shape=shape,dtype=dtype,offset=cursor,bytes=size,sha256='0'*64,wide_sha256='0'*64,
                          gate=dict(atol=1e-4,rtol=1e-3,pooled_fp32_fp64_max_abs=0,equivalent_prefix_max_abs=0))
        cursor+=size
    return dict(format='toklet-mlx-llama3-real-reference-v1',policy=oracle.POLICY,cases=oracle.CASES,layers=oracle.LAYERS,
                model=dict(repository=oracle.REPO,revision=oracle.REVISION,file_pins=oracle.PINS),numpy='2.3.2',
                numerical=oracle.NUMERICAL,scaling=oracle.SCALING,native_profile=oracle.NATIVE_PROFILE,
                admission=oracle.admission(),producer_pins={p:'0'*64 for p in oracle.PRODUCER_NAMES},
                package_pins={'.venv/lib/numpy/__init__.py':'0'*64,'.venv/lib/numpy/core.so':'0'*64},
                array_sha256='0'*64,wide_array_sha256='0'*64,arrays=arrays,array_bytes=cursor,
                greedy=dict(prompt=oracle.CASES['short']['tokens'],teacher_forcing='Independent FP64-accumulation greedy history',
                            fp32_tokens=[1]*4,fp64_tokens=[2]*4,fp64_logit_margins=[.1]*4),
                target_gpu_executed=False,held_out_quality_qualified=False,physical_device_qualified=False)


class Reader(unittest.TestCase):
    def test_affine_unsigned_octets_f16_signed_metadata_f32_rounding(self):
        words=np.array([[0xff7f0100,0xfefd0302,0x12345678,0x90abcdef]*4],dtype='<u4')
        for scale,bias in [(.003,.3333),(-.2,1.3),(0,-.7)]:
            a=np.array([[scale]],dtype='<f2');b=np.array([[bias]],dtype='<f2')
            expected=[((int(word)//256**index)%256)*float(a[0,0])+float(b[0,0]) for word in words[0] for index in range(4)]
            self.assertEqual(oracle.decode(words,a,b,64).tobytes(),struct.pack('<64f',*expected))
        for args in [(words.astype('<i4'),a,b,64),(words,a.astype('<f4'),b,64),(words,a,b,True),
                     (words,a,np.full_like(b,np.inf),64)]:
            with self.assertRaises(ValueError):oracle.decode(*args)

    def test_tied_gather_and_head_have_distinct_storage_semantics(self):
        table=np.array([[.3333431,-.009007],[1.12345,.77777]],dtype=np.float32)
        tied=oracle.TiedEmbedding(table)
        self.assertTrue(np.shares_memory(tied[:],table))
        self.assertEqual(tied[[0,1]].dtype,np.float16)
        self.assertFalse(np.array_equal(tied[:],tied[[0,1]].astype(np.float32)))
        x=np.array([[1.,-.5]],dtype=np.float16)
        got=graph.linear(x,tied,None,np.float64,np.float32)
        expected=np.array([[sum(float(a)*float(b) for a,b in zip(x[0],row)) for row in table]],dtype=np.float32)
        np.testing.assert_array_equal(got,expected)

    def test_fixed_inventory_and_gate_mutations(self):
        m=manifest();oracle.validate_manifest(m)
        self.assertEqual(len(m['arrays']),349)
        key='cases.offset509.layers.15.q_rotary'
        self.assertEqual(m['arrays'][key]['shape'],[3,32,64])
        self.assertLess(m['admission']['admitted_bytes'],6*1024**3)
        mutations=[lambda x:x['arrays'].pop(key),lambda x:x['arrays'][key].update(shape=[3,2048]),
                   lambda x:x['arrays'][key].update(offset=0),lambda x:x['arrays'][key].update(wide_sha256='bad'),
                   lambda x:x['arrays'][key]['gate'].update(atol=.1),lambda x:x['native_profile'].update(attention='CompensatedReference'),
                   lambda x:x['scaling'].update(factor=8),lambda x:x['admission'].update(admitted_bytes=1),
                   lambda x:x['model'].update(revision='new'),lambda x:x['producer_pins'].pop('tools/reference/llama3_rotary.py')]
        for change in mutations:
            altered=copy.deepcopy(m);change(altered)
            with self.assertRaises(ValueError):oracle.validate_manifest(altered)
        # Prefix variability is a diagnostic, not an additive tolerance source.
        for item in m['arrays'].values():item['gate']['equivalent_prefix_max_abs']=100
        oracle.validate_manifest(m)

    def test_descriptor_identity_and_cleanup_preserve_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a';path.write_bytes(b'original')
            with path.open('rb') as stream:
                stream.seek(3);other=Path(tmp)/'b';other.write_bytes(b'new');other.replace(path)
                self.assertEqual(oracle.descriptor_sha(stream),hashlib.sha256(b'original').hexdigest())
                self.assertEqual(stream.tell(),3)
        w=object.__new__(oracle.Weights);w.embedding=None
        w.mapping=mock.Mock();w.stream=mock.Mock();w.mapping.close.side_effect=RuntimeError('cleanup')
        stream_owner=w.stream
        primary=ValueError('numerical');w.__exit__(ValueError,primary,None)
        self.assertEqual(str(primary),'numerical');self.assertIn('cleanup',primary.__notes__[0]);stream_owner.close.assert_called_once()
        self.assertIsNone(w.stream);self.assertIsNone(w.mapping)

    def test_dual_bit_reader_recomputes_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);a=struct.pack('<2f',1,2);b=struct.pack('<2f',1,2.25)
            (folder/'arrays.bin').write_bytes(a);(folder/'wide.bin').write_bytes(b)
            m=dict(array_bytes=8,array_sha256=hashlib.sha256(a).hexdigest(),wide_array_sha256=hashlib.sha256(b).hexdigest(),
                   arrays={'cases.short.logits':dict(bytes=8,dtype='<f4',sha256=hashlib.sha256(a).hexdigest(),wide_sha256=hashlib.sha256(b).hexdigest(),
                                                     gate=dict(pooled_fp32_fp64_max_abs=.25))})
            def save(): (folder/'manifest.json').write_text(json.dumps(m))
            save()
            with mock.patch.object(oracle,'validate_manifest'),mock.patch.object(oracle,'saved_greedy'):
                oracle.verify(folder)
                m['arrays']['cases.short.logits']['gate']['pooled_fp32_fp64_max_abs']=.5;save()
                with self.assertRaisesRegex(ValueError,'Envelope'):oracle.verify(folder)
                (folder/'wide.bin').write_bytes(a)
                with self.assertRaisesRegex(ValueError,'hash'):oracle.verify(folder)


class Graph(unittest.TestCase):
    def data(self):
        table=np.linspace(-.3,.4,16*16,dtype=np.float32).reshape(16,16)
        w={'embed_tokens':oracle.TiedEmbedding(table),'norm':np.ones(16,dtype=np.float16)}
        for name in ['input_layernorm','post_attention_layernorm']:w['layers.0.'+name]=np.ones(16,dtype=np.float16)
        for name,rows,columns in [('q',16,16),('k',8,16),('v',8,16),('o',16,16),('gate',32,16),('up',32,16),('down',16,32)]:
            w['layers.0.'+name+'_proj.weight']=np.linspace(-.1,.2,rows*columns,dtype=np.float32).reshape(rows,columns)
        return w,dict(num_layers=1,num_query_heads=4,num_kv_heads=2,head_dim=4,rms_norm_eps=1e-5,rope_theta=500000.)

    def test_frequency_profiles_and_scaled_position_boundary(self):
        a=graph.frequencies(64,500000.,np.float32,oracle.SCALING)
        b=graph.frequencies(64,500000.,np.float64,oracle.SCALING)
        self.assertEqual(len(a),32);self.assertTrue(np.all(a>0))
        np.testing.assert_allclose(a,b,rtol=1e-7,atol=0)
        # Independent scalar wavelength formula for wide frequency and split-half rotation.
        import math
        x=np.array([[[.25,-.5,.75,1.,-.75,.5,-.25,1.5]]],dtype=np.float16)
        got=graph.rotary(x,511,500000.,np.float64,oracle.SCALING)
        expected=[];nu=[]
        for j in range(4):
            v=500000.**(-2*j/8);wave=math.tau/v
            if wave<2048:z=v
            elif wave>8192:z=v/32
            else:
                t=(8192/wave-1)/3;z=(1-t)*v/32+t*v
            nu.append(z)
        for j,z in enumerate(nu):expected.append(float(x[0,0,j])*math.cos(511*z)-float(x[0,0,j+4])*math.sin(511*z))
        for j,z in enumerate(nu):expected.append(float(x[0,0,j+4])*math.cos(511*z)+float(x[0,0,j])*math.sin(511*z))
        np.testing.assert_array_equal(got,np.array([[expected]],dtype=np.float16))
        for offset in [-1,512]:
            with self.assertRaises(ValueError):graph.rotary(x,offset,500000.,np.float32,oracle.SCALING)

    def test_causality_gqa_zero_offset_and_graph_inventory(self):
        w,c=self.data()
        for precision in [np.float32,np.float64]:
            a=graph.forward([1,3,5],509,w,c,precision,[0],oracle.SCALING)
            b=graph.forward([1,3,7],509,w,c,precision,[0],oracle.SCALING)
            np.testing.assert_array_equal(a['logits'][:2],b['logits'][:2])
            self.assertEqual(len(a),21)
            self.assertEqual(a['layers.0.attention_scores'].dtype,np.float32)
            self.assertEqual(a['layers.0.attention_context'].dtype,np.float16)
            one=graph.forward([1],0,w,c,precision,[0],oracle.SCALING)
            np.testing.assert_array_equal(one['layers.0.q_rotary'].reshape(1,16),one['layers.0.q_projection'])
            values=one['layers.0.v_projection'].reshape(1,2,4)
            np.testing.assert_array_equal(one['layers.0.attention_context'],np.repeat(values,2,axis=1).reshape(1,16))
        w['layers.0.q_proj.bias']=np.zeros(16,dtype=np.float16)
        with self.assertRaises(ValueError):graph.forward([1],0,w,c,np.float64,[],oracle.SCALING)


if __name__=='__main__':unittest.main()
