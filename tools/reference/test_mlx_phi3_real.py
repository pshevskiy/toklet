"""Portable bounded controls; synthetic original packed rows, no real model/GPU."""
from contextlib import contextmanager
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np
import mlx_phi3_real as oracle
import phi3_fp16 as graph
import qwen_fp16


def manifest():
    arrays={};cursor=0
    for name,(shape,dtype) in oracle.expected_arrays().items():
        size=math.prod(shape)*(2 if dtype=='<f2' else 4)
        arrays[name]=dict(shape=shape,dtype=dtype,offset=cursor,bytes=size,sha256='0'*64,wide_sha256='0'*64,
                         gate=dict(atol=1e-4,rtol=1e-3,exact_array_fp32_fp64_max_abs=0))
        cursor+=size
    identity=dict(path='/synthetic/runtime',bytes=1,sha256='0'*64)
    return dict(format='toklet-mlx-phi3-real-reference-v1',policy=copy.deepcopy(oracle.POLICY),cases=copy.deepcopy(oracle.CASES),layers=oracle.LAYERS,
                model=dict(repository=oracle.REPO,revision=oracle.REVISION,file_pins=oracle.PINS),numpy='2.3.2',
                numerical=oracle.NUMERICAL,source_profile=copy.deepcopy(oracle.SOURCE_PROFILE),native_profile=copy.deepcopy(oracle.NATIVE_PROFILE),
                design_protocol_sha256=oracle.DESIGN_PROTOCOL_SHA256,source_cpu_graph_executed=True,
                admission=oracle.admission(),producer_pins={p:'0'*64 for p in oracle.PRODUCER_NAMES},
                package_pins={'.venv/lib/numpy/__init__.py':'0'*64,'.venv/lib/numpy/core.so':'0'*64},
                runtime_identity=dict(python_version='synthetic',executable=identity,venv_config=identity),
                array_sha256='0'*64,wide_array_sha256='0'*64,arrays=arrays,array_bytes=cursor,
                greedy=dict(prompt=oracle.CASES['short']['tokens'],teacher_forcing='Independent FP64-accumulation greedy history',
                            fp32_tokens=[1]*4,fp64_tokens=[2]*4,fp64_logit_margins=[.1]*4),
                target_gpu_executed=False,held_out_quality_qualified=False,physical_device_qualified=False)


def synthetic():
    """Build original fused file and independently decoded complete tiny owners."""
    config=copy.deepcopy(oracle.MODEL_CONFIG)
    config.update(hidden_size=128,intermediate_size=128,vocab_size=8,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
    # Literal physical shapes are independent of specifications()/logical map.
    shapes={'model.embed_tokens.weight':[8,128],'lm_head.weight':[8,128],'model.norm.weight':[128],
            'model.layers.0.input_layernorm.weight':[128],'model.layers.0.post_attention_layernorm.weight':[128],
            'model.layers.0.self_attn.qkv_proj.weight':[256,128],'model.layers.0.self_attn.o_proj.weight':[128,128],
            'model.layers.0.mlp.gate_up_proj.weight':[256,128],'model.layers.0.mlp.down_proj.weight':[128,128]}
    tensors={};expanded={}
    for seed,(name,shape) in enumerate(sorted(shapes.items())):
        if len(shape)==1:
            tensors[name]=('F16',shape,struct.pack('<128e',*[1+(i%7-3)/256 for i in range(128)]));continue
        rows,columns=shape;codes=bytearray();scales=bytearray();biases=bytearray();expected=[]
        for row in range(rows):
            co=[(seed+3*row+7*c+c//8)%16 for c in range(columns)]
            for start in range(0,columns,8):codes.extend(struct.pack('<I',sum(co[start+j]<<(4*j) for j in range(8))))
            scale_values=[];bias_values=[]
            for group in range(columns//64):
                a=struct.pack('<H',0x1801+2*((seed+row+group)%127))
                b=struct.pack('<e',(row%11-5)/4096+seed/32768)
                scales.extend(a);biases.extend(b)
                scale_values.append(struct.unpack('<e',a)[0]);bias_values.append(struct.unpack('<e',b)[0])
            expected.append([float(np.float32(co[c]*scale_values[c//64]+bias_values[c//64])) for c in range(columns)])
        module=name.removesuffix('.weight')
        tensors[name]=('U32',[rows,columns//8],bytes(codes))
        tensors[module+'.scales']=('F16',[rows,columns//64],bytes(scales))
        tensors[module+'.biases']=('F16',[rows,columns//64],bytes(biases))
        expanded[name]=np.array(expected,dtype=np.float32)
    body=bytearray();header={}
    for name,(dtype,shape,raw) in sorted(tensors.items()):
        begin=len(body);body.extend(raw)
        header[name]=dict(dtype=dtype,shape=shape,data_offsets=[begin,len(body)])
    raw_header=json.dumps(header,separators=(',',':')).encode()
    while (8+len(raw_header))%4!=3:raw_header+=b' '
    raw=struct.pack('<Q',len(raw_header))+raw_header+body
    index={'weight_map':{name:'model.safetensors' for name in header}}
    return config,header,raw_header,raw,index,expanded


@contextmanager
def opened_synthetic():
    config,header,raw_header,raw,index,expanded=synthetic()
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        (folder/'config.json').write_text(json.dumps(config))
        (folder/'model.safetensors.index.json').write_text(json.dumps(index))
        (folder/'model.safetensors').write_bytes(raw)
        pins={name:oracle.sha(folder/name) for name in oracle.PINS}
        with mock.patch.multiple(oracle,MODEL_CONFIG=config,PINS=pins,WEIGHT_BYTES=len(raw),HEADER_BYTES=len(raw_header),
                                 HEADER_SHA=hashlib.sha256(raw_header).hexdigest()):
            with oracle.Weights(folder) as weights:yield weights,expanded,header,index,folder


class Reader(unittest.TestCase):
    def test_four_bit_order_and_f16_metadata_independent_scalar(self):
        words=np.array([[0xfedcba98,0x76543210]*4],dtype='<u4')
        for scale,bias in [(.003,.3333),(-.2,1.3),(0,-.7)]:
            a=np.array([[scale]],dtype='<f2');b=np.array([[bias]],dtype='<f2')
            expected=[((int(word)>>(4*j))&15)*float(a[0,0])+float(b[0,0]) for word in words[0] for j in range(8)]
            self.assertEqual(oracle.decode(words,a,b,64).tobytes(),struct.pack('<64f',*expected))
        for args in [(words.astype('<i4'),a,b,64),(words,a.astype('<f4'),b,64),(words,a,b,True),
                     (words,a,np.full_like(b,np.inf),64),(words[:,:7],a,b,64)]:
            with self.assertRaises(ValueError):oracle.decode(*args)

    def test_original_nonzero_spans_gather_and_distinct_untied_head(self):
        with opened_synthetic() as (w,expected,_,__,___):
            self.assertEqual(w.origin%4,3)
            x=np.linspace(-.2,.3,128,dtype=np.float16).reshape(1,128)
            qkv=expected['model.layers.0.self_attn.qkv_proj.weight'];fused=expected['model.layers.0.mlp.gate_up_proj.weight']
            for key,target in [('q',qkv[:128]),('k',qkv[128:192]),('v',qkv[192:256]),('gate',fused[:128]),('up',fused[128:])]:
                name=f'layers.0.{key}_proj.weight'
                np.testing.assert_array_equal(w._matrix(name),target)
                np.testing.assert_array_equal(w.project(name,x,np.float64),qwen_fp16.linear(x,target,None,np.float64))
            gathered=w.lookup([3,0,3]);np.testing.assert_array_equal(gathered,expected['model.embed_tokens.weight'][[3,0,3]].astype(np.float16))
            self.assertEqual(gathered.dtype,np.float16);self.assertFalse(hasattr(w,'embedding'))
            head=expected['lm_head.weight']
            self.assertTrue(np.any(head!=head.astype(np.float16).astype(np.float32)))
            self.assertFalse(np.array_equal(head,expected['model.embed_tokens.weight']))
            np.testing.assert_array_equal(w.project('lm_head.weight',x,np.float64,np.float32),qwen_fp16.linear(x,head,None,np.float64,np.float32))
            for ids in [[],[True],[-1],[8],[0]*513]:
                with self.assertRaises(ValueError):w.lookup(ids)
            w.verify()

    def test_no_mmap_view_escape_and_projection_owner_release_on_failure(self):
        with opened_synthetic() as (w,_,__,___,____):
            # A retained copied stripe cannot prevent closing the mapping.
            owned=w._raw_copy('model.norm.weight',0,128)
            self.assertTrue(owned.flags.owndata)
            references=[];original=w._matrix
            def track(key):
                a=original(key);references.append(weakref.ref(a));return a
            x=np.ones((1,128),dtype=np.float16)
            with mock.patch.object(w,'_matrix',side_effect=track):
                w.project('layers.0.q_proj.weight',x,np.float32)
                self.assertIsNone(references[-1]())
                def fail(_x,matrix,*args):
                    self.assertIsNotNone(references[-1]());raise RuntimeError('injected linear failure')
                with mock.patch.object(qwen_fp16,'linear',new=fail):
                    try:w.project('layers.0.q_proj.weight',x,np.float32)
                    except RuntimeError as error:
                        # Keep the exception alive: completed traceback frames
                        # must not retain the expanded matrix.
                        self.assertEqual(str(error),'injected linear failure')
                        gc.collect();self.assertIsNone(references[-1]())
                    else:self.fail('Expected injected failure')
                self.assertFalse(w.active)
                w.active=True
                with self.assertRaisesRegex(ValueError,'Overlapping'):w.project('layers.0.q_proj.weight',x,np.float32)
                w.active=False
            w.mapping.close();w.mapping=None
            self.assertTrue(np.isfinite(owned).all())

    def test_metadata_and_file_corruption_rejected(self):
        with opened_synthetic() as (w,_,header,index,folder):
            expected=w.expected
            mutations=[lambda x:x.pop(next(iter(x))),lambda x:x[next(iter(x))].update(dtype='F32'),
                       lambda x:x[next(iter(x))]['data_offsets'].__setitem__(0,True),
                       lambda x:x[next(iter(x))]['data_offsets'].__setitem__(1,10**20),
                       lambda x:x[next(iter(x))]['shape'].__setitem__(0,True)]
            for change in mutations:
                h=copy.deepcopy(header);change(h)
                with self.assertRaises(ValueError):oracle.validate_header(h,index,expected,w.origin,oracle.WEIGHT_BYTES)
            bad=copy.deepcopy(index);bad['weight_map'][next(iter(bad['weight_map']))]='other.safetensors'
            with self.assertRaises(ValueError):oracle.validate_header(header,bad,expected,w.origin,oracle.WEIGHT_BYTES)
            for begin,end in [(-1,1),(0,0),(True,1),(0,129)]:
                with self.assertRaises(ValueError):w._raw_copy('model.norm.weight',begin,end)
            original=(folder/'config.json').read_bytes();(folder/'config.json').write_bytes(b'{}')
            with self.assertRaises(ValueError):oracle.Weights(folder)
            (folder/'config.json').write_bytes(original)
            # Replace pathname without altering the mapped descriptor.
            p=folder/'replacement';p.write_bytes(b'truncated');p.replace(folder/'model.safetensors')
            with self.assertRaises(ValueError):w.verify()
            other=oracle.Weights(folder)
            with self.assertRaises(ValueError):other.__enter__()
            self.assertIsNone(other.stream);self.assertIsNone(other.mapping)
        with self.assertRaises(ValueError):json.loads('{"x":1,"x":2}',object_pairs_hook=oracle.unique)

    def test_selected_metadata_and_cleanup_primary_preserved(self):
        with opened_synthetic() as (w,_,__,___,____):
            raw=w._raw_copy
            def corrupt(name,start,end):
                value=raw(name,start,end)
                if name.endswith('.scales'):value.flat[0]=np.inf
                return value
            with mock.patch.object(w,'_raw_copy',side_effect=corrupt):
                with self.assertRaises(ValueError):w.project('layers.0.k_proj.weight',np.ones((1,128),dtype=np.float16),np.float32)
            self.assertFalse(w.active)
        w=object.__new__(oracle.Weights);w.mapping=mock.Mock();w.stream=mock.Mock()
        w.mapping.close.side_effect=RuntimeError('cleanup');stream=w.stream
        primary=ValueError('numerical');w.__exit__(ValueError,primary,None)
        self.assertEqual(str(primary),'numerical');self.assertIn('cleanup',primary.__notes__[0]);stream.close.assert_called_once()
        self.assertIsNone(w.mapping);self.assertIsNone(w.stream)

    def test_manifest_exact_array_gates_and_inventory(self):
        m=manifest();oracle.validate_manifest(m)
        self.assertEqual(len(m['arrays']),349);self.assertEqual(m['array_bytes'],19025664)
        self.assertEqual(m['admission']['admitted_bytes'],4489855655)
        key='cases.offset509.layers.31.q_rotary';self.assertEqual(m['arrays'][key]['shape'],[3,32,96])
        for change in [lambda x:x['arrays'].pop(key),lambda x:x['arrays'][key].update(shape=[3,3072]),
                       lambda x:x['arrays'][key].update(offset=0),lambda x:x['arrays'][key]['gate'].update(atol=.1),
                       lambda x:x['arrays'][key]['gate'].update(equivalent_prefix_max_abs=1),
                       lambda x:x.update(source_cpu_graph_executed=False),lambda x:x['source_profile'].update(tied_head=True),
                       lambda x:x['native_profile'].update(attention='Online'),lambda x:x['admission'].update(admitted_bytes=1)]:
            altered=copy.deepcopy(m);change(altered)
            with self.assertRaises(ValueError):oracle.validate_manifest(altered)
        # Two same-stage arrays may have distinct independently derived envelopes.
        m['arrays']['cases.short.logits']['gate'].update(exact_array_fp32_fp64_max_abs=.02,atol=.04)
        oracle.validate_manifest(m)

    def test_replay_recomputes_each_array_and_rejects_hash_or_envelope_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);a=struct.pack('<4f',1,2,3,4);b=struct.pack('<4f',1,2.25,3,4.5)
            (folder/'arrays.bin').write_bytes(a);(folder/'wide.bin').write_bytes(b)
            arrays={}
            for i,(case,env) in enumerate([('short',.25),('single',.5)]):
                x=a[8*i:8*i+8];y=b[8*i:8*i+8]
                arrays[f'cases.{case}.logits']=dict(bytes=8,dtype='<f4',sha256=hashlib.sha256(x).hexdigest(),wide_sha256=hashlib.sha256(y).hexdigest(),gate=dict(exact_array_fp32_fp64_max_abs=env))
            m=dict(array_bytes=16,array_sha256=hashlib.sha256(a).hexdigest(),wide_array_sha256=hashlib.sha256(b).hexdigest(),arrays=arrays)
            def save():(folder/'manifest.json').write_text(json.dumps(m))
            save()
            with mock.patch.object(oracle,'validate_manifest'),mock.patch.object(oracle,'saved_greedy'):
                oracle.verify(folder)
                arrays['cases.short.logits']['gate']['exact_array_fp32_fp64_max_abs']=.5;save()
                with self.assertRaisesRegex(ValueError,'Envelope'):oracle.verify(folder)
                (folder/'wide.bin').write_bytes(a)
                with self.assertRaisesRegex(ValueError,'hash'):oracle.verify(folder)

    def test_final_distribution_copy_releases_complete_matrix(self):
        refs=[]
        def forward(*args):
            full=np.arange(4*8,dtype=np.float32).reshape(4,8).copy();refs.append(weakref.ref(full));return {'logits':full}
        row=oracle.last_distribution(forward,[1],0,None,np.float32)
        self.assertTrue(row.flags.owndata);self.assertIsNone(refs[0]());np.testing.assert_array_equal(row,np.arange(24,32,dtype=np.float32))


class Graph(unittest.TestCase):
    def test_ordinary_head96_rotary_and_offsets(self):
        x=np.linspace(-1,1,96,dtype=np.float16).reshape(1,1,96)
        for precision in [np.float32,np.float64]:
            np.testing.assert_array_equal(graph.rotary(x,0,10000.,precision),x)
        expected=[]
        for j in range(48):
            a=float(x[0,0,j]);b=float(x[0,0,j+48]);angle=511*10000.**(-2*j/96)
            expected.append(a*math.cos(angle)-b*math.sin(angle))
        for j in range(48):
            a=float(x[0,0,j]);b=float(x[0,0,j+48]);angle=511*10000.**(-2*j/96)
            expected.append(b*math.cos(angle)+a*math.sin(angle))
        np.testing.assert_array_equal(graph.rotary(x,511,10000.,np.float64),np.array([[expected]],dtype=np.float16))
        for offset in [-1,512]:
            with self.assertRaises(ValueError):graph.rotary(x,offset,10000.,np.float32)
        for dim in [0,95,True]:
            with self.assertRaises(ValueError):graph.frequencies(dim,10000.,np.float64)

    def test_real_reader_small_graph_causal_gqa_head_and_storage(self):
        c=dict(num_layers=1,num_query_heads=2,num_kv_heads=1,head_dim=64,rms_norm_eps=1e-5,rope_theta=10000.)
        with opened_synthetic() as (w,expected,_,__,___):
            for precision in [np.float32,np.float64]:
                a=graph.forward([1,3,5],509,w,c,precision,[0]);b=graph.forward([1,3,7],509,w,c,precision,[0])
                np.testing.assert_array_equal(a['logits'][:2],b['logits'][:2])
                self.assertEqual(len(a),21)
                self.assertEqual(a['layers.0.attention_scores'].dtype,np.float32)
                self.assertEqual(a['layers.0.attention_probabilities'].dtype,np.float32)
                self.assertEqual(a['layers.0.attention_context'].dtype,np.float16)
                np.testing.assert_array_equal(a['logits'],qwen_fp16.linear(a['final_norm'],expected['lm_head.weight'],None,precision,np.float32))
                one=graph.forward([1],0,w,c,precision,[0])
                np.testing.assert_array_equal(one['layers.0.q_rotary'].reshape(1,128),one['layers.0.q_projection'])
                v=one['layers.0.v_projection'].reshape(1,1,64)
                np.testing.assert_array_equal(one['layers.0.attention_context'],np.repeat(v,2,axis=1).reshape(1,128))
            with self.assertRaises(ValueError):graph.forward([1,2],511,w,c,np.float32,[])


if __name__=='__main__':unittest.main()
