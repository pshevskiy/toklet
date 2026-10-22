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
import mlx_qwen_real as oracle


def manifest():
    arrays = {};cursor = 0
    for name,(shape,dtype) in oracle.expected_arrays().items():
        size = int(np.prod(shape))*(2 if dtype == '<f2' else 4)
        arrays[name] = dict(shape=shape,dtype=dtype,offset=cursor,bytes=size,sha256='0'*64,
                            gate=dict(atol=1e-4,rtol=1e-3,pooled_fp32_fp64_max_abs=0,equivalent_prefix_max_abs=0))
        cursor += size
    return dict(format='toklet-mlx-qwen-real-reference-v1',policy=oracle.POLICY,cases=oracle.CASES,layers=oracle.LAYERS,
                model=dict(repository=oracle.REPO,revision=oracle.REVISION,file_pins=oracle.PINS),numpy='2.3.2',
                producer_pins={key:'0'*64 for key in ['tools/reference/mlx_qwen_real.py','tools/reference/qwen_fp16.py',
                    'tools/reference/qwen.py','tools/bench/prepare.py','tools/experiments/native_usage.py']},
                numpy_binary_pins={'.venv/numpy.so':'0'*64},array_sha256='0'*64,arrays=arrays,array_bytes=cursor,
                greedy=dict(prompt=oracle.CASES['short']['tokens'],teacher_forcing='Independent FP64-accumulation greedy history',
                            fp32_tokens=[1]*4,fp64_tokens=[2]*4,fp64_logit_margins=[0.1]*4),
                target_gpu_executed=False,held_out_quality_qualified=False,physical_device_qualified=False)


class RealOracle(unittest.TestCase):
    def test_literal_unsigned_nibbles_signed_metadata_and_fp32_boundary(self):
        words = np.array([[0xfedcba98,0x76543210]*4],dtype='<u4')
        for scale,bias in [(0.003,0.3333),(-0.2,1.3),(0,-0.7)]:
            scales = np.array([[scale]],dtype='<f2');biases=np.array([[bias]],dtype='<f2')
            expected=[]
            for word in words[0]:
                for index in range(8):
                    value=((int(word)//(16**index))%16)*float(scales[0,0])+float(biases[0,0])
                    expected.append(struct.unpack('<f',struct.pack('<f',value))[0])
            actual=oracle.decode(words,scales,biases,64)
            self.assertEqual(actual.dtype,np.dtype('<f4'))
            self.assertEqual(actual.tobytes(),struct.pack('<64f',*expected))

    def test_reconstruction_rejects_invalid_domains(self):
        words=np.zeros((1,8),dtype='<u4');scales=np.ones((1,1),dtype='<f2');bias=np.zeros_like(scales)
        variants=[(words.astype('<i4'),scales,bias,64),(words,scales.astype('<f4'),bias,64),
                  (words,scales,bias,32),(words,scales,bias,True),(words[:0],scales[:0],bias[:0],64),
                  (words,scales.reshape(1),bias,64),(words,scales,np.full_like(bias,np.inf),64)]
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
        self.assertEqual(len(original['arrays']),346)
        key='cases.short.layers.23.q_rotary'
        self.assertEqual(original['arrays'][key]['shape'],[4,14,64])
        mutations=[lambda m:m['arrays'].pop(key),lambda m:m['arrays'][key].update(dtype='<f4'),
            lambda m:m['arrays'][key].update(shape=[4,896]),
            lambda m:m['arrays']['cases.single.embedding'].update(shape=[True,896]),lambda m:m['arrays'][key].update(bytes=True),
            lambda m:m['arrays'][key].update(offset=0),lambda m:m['arrays'][key]['gate'].update(atol=99),
            lambda m:m['arrays'][key]['gate'].update(rtol=0.1),lambda m:m['arrays'][key]['gate'].update(extra=1),
            lambda m:[item['gate'].update(pooled_fp32_fp64_max_abs=1e308,atol=float('inf')) for name,item in m['arrays'].items() if name.endswith('logits')],
            lambda m:m['arrays'][key]['gate'].update(pooled_fp32_fp64_max_abs=float('nan')),
            lambda m:m['greedy']['fp64_tokens'].pop(),lambda m:m['greedy']['fp32_tokens'].__setitem__(0,151936),
            lambda m:m['greedy']['fp64_logit_margins'].__setitem__(0,-1),
            lambda m:m['model'].update(revision='other'),lambda m:m.update(target_gpu_executed=True),
            lambda m:m['producer_pins'].pop('tools/reference/qwen.py'),
            lambda m:m.update(numpy_binary_pins={'../escape.so':'0'*64})]
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
            with mock.patch.object(oracle,'validate_manifest'),mock.patch.object(Path,'open',opening):
                oracle.verify(root)
            self.assertEqual(len(opens),1)
            data=struct.pack('<2f',1,float('inf'));path.write_bytes(data);write_manifest(data)
            with mock.patch.object(oracle,'validate_manifest'):
                with self.assertRaisesRegex(ValueError,'nonfinite'):oracle.verify(root)

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


if __name__=='__main__':unittest.main()
