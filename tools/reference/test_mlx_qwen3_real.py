"""Portable independent reconstruction/schema tests; no model, MLX or GPU required."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import mlx_qwen3_real as oracle


def manifest():
    arrays = {};cursor = 0
    for name,(shape,dtype) in oracle.expected_arrays().items():
        size = int(np.prod(shape))*(2 if dtype == '<f2' else 4)
        arrays[name] = dict(shape=shape,dtype=dtype,offset=cursor,bytes=size,sha256='0'*64,
                            gate=dict(atol=1e-4,rtol=1e-3,pooled_fp32_fp64_max_abs=0,equivalent_prefix_max_abs=0))
        cursor += size
    return dict(format='toklet-mlx-qwen3-real-reference-v1',policy=oracle.POLICY,cases=oracle.CASES,layers=oracle.LAYERS,
                model=dict(repository=oracle.REPO,revision=oracle.REVISION,file_pins=oracle.PINS),numpy='2.3.2',tokenizers='0.22.2',tokenization=oracle.TOKENIZATION,
                producer_pins={key:'0'*64 for key in ['tools/reference/mlx_qwen3_real.py','tools/reference/mlx_qwen3_checkpoint.py','tools/reference/qwen3_fp16.py','tools/reference/qwen_fp16.py',
                    'tools/reference/qwen.py','tools/bench/prepare.py','tools/experiments/native_usage.py']},
                package_pins={'.venv/lib/numpy/__init__.py':'0'*64,'.venv/lib/numpy/core.so':'0'*64,'.venv/lib/tokenizers/__init__.py':'0'*64,'.venv/lib/tokenizers/tokenizers.so':'0'*64},array_sha256='0'*64,arrays=arrays,array_bytes=cursor,
                greedy=dict(prompt=oracle.CASES['short']['tokens'],teacher_forcing='Independent FP64-accumulation greedy history',
                            fp32_tokens=[1]*4,fp64_tokens=[2]*4,fp32_logit_margins=[0.2]*4,fp64_logit_margins=[0.1]*4),
                target_gpu_executed=False,held_out_quality_qualified=False,physical_device_qualified=False)


class Qwen3Oracle(unittest.TestCase):
    def test_bf16_bits_signed_zero_subnormal_normal_and_nonfinite(self):
        words=np.array([0,0x8000,1,0x8001,0x0080,0x8080,0x3f80,0xbf80,0x7f7f],dtype='<u2')
        raw=b''.join(struct.pack('<I',int(word)<<16) for word in words)
        self.assertEqual(oracle.bf16(words).tobytes(),raw)
        self.assertEqual(np.signbit(oracle.bf16(words))[1],True)
        self.assertEqual(float(oracle.bf16(words)[6]),1.0)
        for bits in [0x7f80,0xff80,0x7fc0,0x7f81]:
            with self.assertRaises(ValueError):oracle.bf16(np.array([bits],dtype='<u2'))
        with self.assertRaises(ValueError):oracle.bf16(words.astype('<f2'))

    def test_literal_unsigned_octets_signed_bf16_and_fp32_boundary(self):
        words=np.array([[0xff7f0100,0xfefd0302,0x12345678,0x90abcdef]*4],dtype='<u4')
        for scale,bias in [(0x3b45,0x3eab),(0xbe4d,0x3fa6),(0,0xbf33)]:
            scales=np.array([[scale]],dtype='<u2');biases=np.array([[bias]],dtype='<u2')
            scale_value=struct.unpack('<f',struct.pack('<I',scale<<16))[0]
            bias_value=struct.unpack('<f',struct.pack('<I',bias<<16))[0]
            expected=[]
            for word in words[0]:
                for index in range(4):
                    value=((int(word)//(256**index))%256)*scale_value+bias_value
                    expected.append(struct.unpack('<f',struct.pack('<f',value))[0])
            actual=oracle.decode(words,scales,biases,64)
            self.assertEqual(actual.dtype,np.dtype('<f4'))
            self.assertEqual(actual.tobytes(),struct.pack('<64f',*expected))

    def test_reconstruction_rejects_invalid_domains(self):
        words=np.zeros((1,16),dtype='<u4');scales=np.full((1,1),0x3f80,dtype='<u2');bias=np.zeros_like(scales)
        variants=[(words.astype('<i4'),scales,bias,64),(words,scales.astype('<f4'),bias,64),
                  (words,scales,bias,32),(words,scales,bias,True),(words[:0],scales[:0],bias[:0],64),
                  (words,scales.reshape(1),bias,64),(words,scales,np.full_like(bias,0x7f80),64)]
        for args in variants:
            with self.subTest(args=[getattr(x,'shape',x) for x in args]):
                with self.assertRaises(ValueError):oracle.decode(*args)

    def test_tied_gather_rounds_but_output_rows_preserve_original(self):
        table=np.array([[0.3333431,-0.009007],[1.12345,0.77777]],dtype='<f4');weights=oracle.TiedEmbedding(table)
        self.assertEqual(len(weights),2)
        self.assertEqual(weights[[1,0]].dtype,np.dtype('<f2'))
        np.testing.assert_array_equal(weights[[1,0]],table[[1,0]].astype(np.float16))
        self.assertTrue(np.shares_memory(weights[:],table))
        self.assertFalse(np.array_equal(weights[:],weights[[0,1]].astype(np.float32)))
        with self.assertRaises(ValueError):weights[0]

    def test_exact_schema_and_mutation_rejection(self):
        original=manifest();oracle.validate_manifest(original)
        self.assertEqual(len(original['arrays']),385)
        key='cases.short.layers.27.q_rotary'
        self.assertEqual(original['arrays'][key]['shape'],[4,16,128])
        mutations=[lambda m:m['arrays'].pop(key),lambda m:m['arrays'][key].update(dtype='<f4'),
            lambda m:m['arrays'][key].update(shape=[4,1024]),
            lambda m:m['arrays']['cases.single.embedding'].update(shape=[True,1024]),lambda m:m['arrays'][key].update(bytes=True),
            lambda m:m['arrays'][key].update(offset=0),lambda m:m['arrays'][key]['gate'].update(atol=99),
            lambda m:m['arrays'][key]['gate'].update(rtol=0.1),lambda m:m['arrays'][key]['gate'].update(extra=1),
            lambda m:[item['gate'].update(pooled_fp32_fp64_max_abs=1e308,atol=float('inf')) for name,item in m['arrays'].items() if name.endswith('logits')],
            lambda m:m['arrays'][key]['gate'].update(pooled_fp32_fp64_max_abs=float('nan')),
            lambda m:m['greedy']['fp64_tokens'].pop(),lambda m:m['greedy']['fp32_tokens'].__setitem__(0,151936),
            lambda m:m['greedy']['fp64_logit_margins'].__setitem__(0,-1),
            lambda m:m['model'].update(revision='other'),lambda m:m.update(target_gpu_executed=True),
            lambda m:m['producer_pins'].pop('tools/reference/qwen.py'),
            lambda m:m.update(package_pins={'../escape.so':'0'*64})]
        for index,change in enumerate(mutations):
            altered=copy.deepcopy(original);change(altered)
            with self.subTest(index=index):
                with self.assertRaises(ValueError):oracle.validate_manifest(altered)

    def test_descriptor_hash_restores_position_and_uses_consumed_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input';path.write_bytes(b'original')
            with path.open('rb') as stream:
                stream.seek(3)
                replacement=Path(directory)/'new';replacement.write_bytes(b'replaced');replacement.replace(path)
                self.assertEqual(oracle.descriptor_sha(stream),hashlib.sha256(b'original').hexdigest())
                self.assertEqual(stream.tell(),3)
                self.assertNotEqual(oracle.descriptor_sha(stream),oracle.sha(path))

    def test_reader_maps_authenticated_descriptor_and_rejects_changed_path(self):
        import mlx_qwen3_checkpoint as checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);weight=root/'model.safetensors'
            header={};payload=bytearray()
            for name,dtype,shape,data in [
                ('model.embed_tokens.weight','U32',[1,16],struct.pack('<16I',*[0x03020100]*16)),
                ('model.embed_tokens.scales','BF16',[1,1],struct.pack('<H',0x3d00)),
                ('model.embed_tokens.biases','BF16',[1,1],struct.pack('<H',0xbf00))]:
                header[name]=dict(dtype=dtype,shape=shape,data_offsets=[len(payload),len(payload)+len(data)]);payload.extend(data)
            raw=json.dumps(header).encode();original=struct.pack('<Q',len(raw))+raw+payload;weight.write_bytes(original)
            for name in ['config.json','model.safetensors.index.json']:(root/name).write_bytes(b'{}')
            pins={p.name:oracle.sha(p) for p in root.iterdir()}
            opening=Path.open
            def substitute(path,*args,**kwargs):
                stream=opening(path,*args,**kwargs)
                if path==weight and args==('rb',):
                    changed=root/'replacement';changed.write_bytes(b'x'*len(original));changed.replace(weight)
                return stream
            with mock.patch.object(oracle,'PINS',pins),mock.patch.object(oracle,'WEIGHT_BYTES',len(original)),mock.patch.object(checkpoint,'validate_header',return_value=dict(dtype_counts={'U32':197,'BF16':507})),mock.patch.object(checkpoint,'logical_shapes',return_value={'model.embed_tokens.weight':[1,64]}):
                with mock.patch.object(Path,'open',substitute):
                    weights=oracle.Weights(root).__enter__()
                try:
                    self.assertEqual(weights.embedding.table.dtype,np.dtype('<f4'))
                    np.testing.assert_array_equal(weights.embedding.table[0,:4],np.array([-.5,-.46875,-.4375,-.40625],dtype=np.float32))
                    with self.assertRaisesRegex(ValueError,'Checkpoint path changed'):weights.verify()
                finally:weights.__exit__(None,None,None)
                self.assertIsNone(weights.stream);self.assertIsNone(weights.mapping)

    def test_array_verifier_uses_one_descriptor_and_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'arrays.bin';raw=struct.pack('<2f',1,2)
            path.write_bytes(raw)
            def write_manifest(data):
                value=dict(array_bytes=len(data),array_sha256=hashlib.sha256(data).hexdigest(),
                           arrays={'value':dict(bytes=len(data),dtype='<f4',sha256=hashlib.sha256(data).hexdigest())})
                (root/'manifest.json').write_text(json.dumps(value));return value
            write_manifest(raw)
            with self.assertRaisesRegex(ValueError,'manifest SHA'):oracle.verify(root,'0'*64)
            real_open=Path.open;opens=[]
            def opening(item,*args,**kwargs):
                stream=real_open(item,*args,**kwargs)
                if item == path:
                    opens.append(item)
                    replacement=root/'replacement';replacement.write_bytes(b'x'*8);replacement.replace(path)
                return stream
            # Replacement after open cannot substitute the bytes being verified.
            with mock.patch.object(oracle,'validate_manifest'),mock.patch.object(oracle,'saved_greedy',return_value=None),mock.patch.object(Path,'open',opening):
                oracle.verify(root)
            self.assertEqual(len(opens),1)
            data=struct.pack('<2f',1,float('inf'));path.write_bytes(data);write_manifest(data)
            with mock.patch.object(oracle,'validate_manifest'):
                with self.assertRaisesRegex(ValueError,'nonfinite'):oracle.verify(root)

    def test_malformed_mapped_metadata_keeps_original_rejection_and_closes(self):
        import mlx_qwen3_checkpoint as checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);header={};payload=bytearray()
            for name,dtype,shape,data in [
                ('model.embed_tokens.weight','U32',[1,16],struct.pack('<16I',*[0]*16)),
                ('model.embed_tokens.scales','BF16',[1,1],struct.pack('<H',0x7f80)),
                ('model.embed_tokens.biases','BF16',[1,1],struct.pack('<H',0))]:
                header[name]=dict(dtype=dtype,shape=shape,data_offsets=[len(payload),len(payload)+len(data)]);payload.extend(data)
            raw=json.dumps(header).encode();raw=struct.pack('<Q',len(raw))+raw+payload
            (root/'model.safetensors').write_bytes(raw)
            for name in ['config.json','model.safetensors.index.json']:(root/name).write_bytes(b'{}')
            pins={path.name:oracle.sha(path) for path in root.iterdir()}
            with mock.patch.object(oracle,'PINS',pins),mock.patch.object(oracle,'WEIGHT_BYTES',len(raw)),mock.patch.object(checkpoint,'validate_header',return_value=dict(dtype_counts={'U32':197,'BF16':507})),mock.patch.object(checkpoint,'logical_shapes',return_value={'model.embed_tokens.weight':[1,64]}):
                weights=oracle.Weights(root)
                try:weights.__enter__()
                except ValueError as failure:
                    # Keep the original traceback live while checking cleanup.
                    self.assertIn('Nonfinite BF16 metadata',str(failure))
                    self.assertIsNotNone(failure.__traceback__)
                    self.assertIsNone(weights.mapping);self.assertIsNone(weights.stream);self.assertIsNone(weights.embedding)
                else:self.fail('Malformed metadata must reject')

    def test_duplicate_hash_and_publication_preflight(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):json.loads('{"a":1,"a":2}',object_pairs_hook=oracle.unique)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad';path.write_bytes(b'{}')
            with self.assertRaises(ValueError):oracle.bounded(path,1,oracle.sha(path))
            with self.assertRaises(ValueError):oracle.bounded(path,10,'0'*64)
            with mock.patch.dict('sys.modules',{'numpy':None}):
                with self.assertRaises(FileExistsError):oracle.export(path)
                dangling=Path(directory)/'dangling';dangling.symlink_to(Path(directory)/'absent')
                with self.assertRaises(FileExistsError):oracle.export(dangling)




class Qwen3Graph(unittest.TestCase):
    def data(self):
        table=np.linspace(-.3,.4,16*16,dtype=np.float32).reshape(16,16)
        weights={'embed_tokens':oracle.TiedEmbedding(table),'norm':np.linspace(.8,1.2,16,dtype=np.float16)}
        for name in ['input_layernorm','post_attention_layernorm']:weights['layers.0.'+name]=np.ones(16,dtype=np.float16)
        weights['layers.0.q_norm']=np.linspace(.6,1.4,6,dtype=np.float16)
        weights['layers.0.k_norm']=np.linspace(1.3,.7,6,dtype=np.float16)
        for name,rows,columns in [('q',24,16),('k',12,16),('v',12,16),('o',16,24),('gate',32,16),('up',32,16),('down',16,32)]:
            weights['layers.0.'+name+'_proj.weight']=np.linspace(-.1,.2,rows*columns,dtype=np.float32).reshape(rows,columns)
        return weights,dict(num_layers=1,num_query_heads=4,num_kv_heads=2,head_dim=6,rms_norm_eps=1e-6,rope_theta=1e6),table

    def test_unequal_width_qk_norm_graph_matches_ordered_scalar(self):
        import qwen3_fp16,mlx_qwen3
        weights,config,table=self.data()
        actual=qwen3_fp16.forward([1,3,5],5,weights,config,np.float64,[0])
        expected=mlx_qwen3.forward([1,3,5],5,dict(weights,output=table),config,np.float64,np.float16)
        self.assertEqual(actual['layers.0.q_projection'].shape,(3,24))
        self.assertEqual(actual['layers.0.q_norm'].shape,(3,4,6))
        self.assertEqual(actual['layers.0.k_norm'].shape,(3,2,6))
        for name,value in actual.items():
            np.testing.assert_allclose(value.reshape(-1),expected[name].reshape(-1),atol=1e-6,rtol=1e-6,err_msg=name)
        bypassed=mlx_qwen3.forward([1,3,5],5,dict(weights,output=table),config,np.float64,np.float16,normalize_qk=False)
        self.assertGreater(float(np.max(np.abs(actual['logits']-bypassed['logits']))),1e-3)

    def test_causality_grouped_values_and_bias_rejection(self):
        import qwen3_fp16
        weights,config,_=self.data()
        a=qwen3_fp16.forward([1,3,5],5,weights,config,np.float64,[0])
        b=qwen3_fp16.forward([1,3,7],5,weights,config,np.float64,[0])
        np.testing.assert_array_equal(a['logits'][:2],b['logits'][:2])
        single=qwen3_fp16.forward([1],0,weights,config,np.float64,[0])
        values=single['layers.0.v_projection'].reshape(1,2,6)
        np.testing.assert_array_equal(single['layers.0.attention_context'],np.repeat(values,2,axis=1).reshape(1,24))
        weights['layers.0.q_proj.bias']=np.zeros(24,dtype=np.float16)
        with self.assertRaises(ValueError):qwen3_fp16.forward([1],0,weights,config,np.float64,[0])

if __name__=='__main__':unittest.main()
