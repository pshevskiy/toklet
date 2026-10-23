"""Portable independent-reference checks; no model download or Apple GPU."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import numpy as np
import mlx_mistral_real as m
import mistral_fp16

def manifest():
    arrays = {};cursor = 0
    for name,(shape,dtype) in m.expected_arrays().items():
        size = int(np.prod(shape))*(2 if dtype == '<f2' else 4)
        arrays[name] = dict(shape=shape,dtype=dtype,offset=cursor,bytes=size,sha256='0'*64,
                            gate=dict(atol=1e-4,rtol=1e-3,pooled_fp32_fp64_max_abs=0,equivalent_prefix_max_abs=0))
        cursor += size
    return dict(format='toklet-mlx-mistral-real-reference-v1',policy=m.POLICY,cases=m.CASES,layers=m.LAYERS,
                model=dict(repository=m.REPO,revision=m.REVISION,file_pins=m.PINS),numpy='2.3.2',tokenizers='0.22.2',tokenization=m.TOKENIZATION,
                producer_pins={key:'0'*64 for key in ['tools/reference/mlx_mistral_real.py','tools/reference/mistral_fp16.py','tools/reference/qwen_fp16.py',
                    'tools/reference/qwen.py','tools/bench/prepare.py','tools/experiments/native_usage.py']},
                package_pins={'.venv/lib/numpy/__init__.py':'0'*64,'.venv/lib/numpy/core.so':'0'*64,'.venv/lib/tokenizers/__init__.py':'0'*64,'.venv/lib/tokenizers/tokenizers.so':'0'*64},array_sha256='0'*64,arrays=arrays,array_bytes=cursor,
                greedy=dict(prompt=m.CASES['short']['tokens'],teacher_forcing='Independent FP64-accumulation greedy history',
                            fp32_tokens=[1]*4,fp64_tokens=[2]*4,fp64_logit_margins=[0.1]*4),
                target_gpu_executed=False,held_out_quality_qualified=False,physical_device_qualified=False)


class Checks(unittest.TestCase):
    def test_handpacked_low_nibble_first_groups(self):
        packed = np.array([[0x76543210,0xfedcba98]*8],dtype='<u4')
        scales = np.array([[0.5,2]],dtype='<f2')
        biases = np.array([[-1,3]],dtype='<f2')
        actual = m.decode(packed,scales,biases,128)
        expected = np.concatenate([np.tile(np.arange(16),4)*0.5-1,np.tile(np.arange(16),4)*2+3]).astype(np.float32)[None,:]
        np.testing.assert_array_equal(actual,expected)
        for args in [(packed.astype(np.int32),scales,biases,128),(packed,scales.astype(np.float32),biases,128),(packed,scales,biases,64)]:
            with self.assertRaises(ValueError):m.decode(*args)

    def test_row_view_bounds_and_gather_order(self):
        class Owner:
            logical={'w':[4096,64]}
            def __init__(self):self.calls=[]
            def matrix(self,name,start,end):
                self.calls.append((start,end));return np.repeat(np.arange(start,end,dtype=np.float32)[:,None],64,axis=1)
        owner=Owner();view=m.Matrix(owner,'w')
        np.testing.assert_array_equal(view[[3,1,3]][:,0],[3,1,3])
        self.assertEqual(owner.calls,[(3,4),(1,2),(3,4)])
        self.assertEqual(view[10:20].shape,(10,64))
        for key in [slice(0,1025),slice(0,10,2),[],[-1],[4096],1]:
            with self.assertRaises(ValueError):view[key]

    def test_full_attention_graph_uses_untied_head(self):
        weights={'embed_tokens':np.ones((5,8),dtype=np.float16),'norm':np.ones(8,dtype=np.float16),'lm_head':np.zeros((5,8),dtype=np.float32)}
        weights['lm_head'][2]=2
        p='layers.0.'
        for name in ['input_layernorm','post_attention_layernorm']:weights[p+name]=np.ones(8,dtype=np.float16)
        for name,rows in [('q',8),('k',4),('v',4),('o',8),('gate',8),('up',8),('down',8)]:weights[p+name+'_proj.weight']=np.zeros((rows,8),dtype=np.float32)
        config=dict(num_layers=1,num_query_heads=2,num_kv_heads=1,head_dim=4,rms_norm_eps=1e-5,rope_theta=10000)
        for precision in [np.float32,np.float64]:
            result=mistral_fp16.forward([1,3],5,weights,config,precision,[0])
            np.testing.assert_array_equal(result['logits'],[[0,0,16,0,0],[0,0,16,0,0]])
            np.testing.assert_array_equal(result['final_norm'],np.ones((2,8),dtype=np.float16))

    def test_explicit_reference_coverage(self):
        arrays=m.expected_arrays()
        self.assertEqual(len(arrays),349)
        self.assertEqual(arrays['cases.tile17.logits'],([17,32768],'<f4'))
        self.assertEqual(arrays['cases.offset.layers.31.q_rotary'],([3,32,128],'<f2'))
        self.assertEqual(arrays['cases.offset.layers.31.k_projection'],([3,1024],'<f2'))
        self.assertEqual(m.POLICY['atol_floor'],1e-4)
        self.assertEqual(m.POLICY['envelope_multiplier'],2)
        self.assertEqual(m.POLICY['rtol'],1e-3)

    def test_exact_schema_and_mutation_rejection(self):
        original=manifest();m.validate_manifest(original)
        self.assertEqual(len(original['arrays']),349)
        key='cases.short.layers.31.q_rotary'
        self.assertEqual(original['arrays'][key]['shape'],[4,32,128])
        mutations=[lambda m:m['arrays'].pop(key),lambda m:m['arrays'][key].update(dtype='<f4'),
            lambda m:m['arrays'][key].update(shape=[4,4096]),
            lambda m:m['arrays']['cases.single.embedding'].update(shape=[True,4096]),lambda m:m['arrays'][key].update(bytes=True),
            lambda m:m['arrays'][key].update(offset=0),lambda m:m['arrays'][key]['gate'].update(atol=99),
            lambda m:m['arrays'][key]['gate'].update(rtol=0.1),lambda m:m['arrays'][key]['gate'].update(extra=1),
            lambda m:[item['gate'].update(pooled_fp32_fp64_max_abs=1e308,atol=float('inf')) for name,item in m['arrays'].items() if name.endswith('logits')],
            lambda m:m['arrays'][key]['gate'].update(pooled_fp32_fp64_max_abs=float('nan')),
            lambda m:m['greedy']['fp64_tokens'].pop(),lambda m:m['greedy']['fp32_tokens'].__setitem__(0,32768),
            lambda m:m['greedy']['fp64_logit_margins'].__setitem__(0,-1),
            lambda m:m['model'].update(revision='other'),lambda m:m.update(target_gpu_executed=True),
            lambda m:m['producer_pins'].pop('tools/reference/qwen.py'),
            lambda m:m.update(package_pins={'../escape.so':'0'*64})]
        for index,change in enumerate(mutations):
            altered=copy.deepcopy(original);change(altered)
            with self.subTest(index=index):
                with self.assertRaises(ValueError):m.validate_manifest(altered)

    def test_descriptor_hash_restores_position_and_uses_consumed_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input';path.write_bytes(b'original')
            with path.open('rb') as stream:
                stream.seek(3)
                replacement=Path(directory)/'new';replacement.write_bytes(b'replaced');replacement.replace(path)
                self.assertEqual(m.descriptor_sha(stream),hashlib.sha256(b'original').hexdigest())
                self.assertEqual(stream.tell(),3)
                self.assertNotEqual(m.descriptor_sha(stream),m.sha(path))

    def test_array_verifier_uses_one_descriptor_and_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'arrays.bin';raw=struct.pack('<2f',1,2)
            path.write_bytes(raw)
            def write_manifest(data):
                value=dict(array_bytes=len(data),array_sha256=hashlib.sha256(data).hexdigest(),
                           arrays={'value':dict(bytes=len(data),dtype='<f4',sha256=hashlib.sha256(data).hexdigest())})
                (root/'manifest.json').write_text(json.dumps(value));return value
            write_manifest(raw)
            with self.assertRaisesRegex(ValueError,'manifest SHA'):m.verify(root,'0'*64)
            real_open=Path.open;opens=[]
            def opening(item,*args,**kwargs):
                stream=real_open(item,*args,**kwargs)
                if item == path:
                    opens.append(item)
                    replacement=root/'replacement';replacement.write_bytes(b'x'*8);replacement.replace(path)
                return stream
            # Replacement after open cannot substitute the bytes being verified.
            with mock.patch.object(m,'validate_manifest'),mock.patch.object(m,'saved_greedy',return_value=None),mock.patch.object(Path,'open',opening):
                m.verify(root)
            self.assertEqual(len(opens),1)
            path.write_bytes(raw[:4]);write_manifest(raw)
            with mock.patch.object(m,'validate_manifest'):
                with self.assertRaisesRegex(ValueError,'size'):m.verify(root)
            path.write_bytes(raw);value=write_manifest(raw)
            value['arrays']['value']['bytes']=12
            (root/'manifest.json').write_text(json.dumps(value))
            with mock.patch.object(m,'validate_manifest'):
                with self.assertRaisesRegex(ValueError,'truncated'):m.verify(root)
            for field in ['entry','whole']:
                value=write_manifest(raw)
                if field=='entry':value['arrays']['value']['sha256']='0'*64
                else:value['array_sha256']='0'*64
                (root/'manifest.json').write_text(json.dumps(value))
                with mock.patch.object(m,'validate_manifest'):
                    with self.assertRaisesRegex(ValueError,'differs|differ'):m.verify(root)
            data=struct.pack('<2f',1,float('inf'));path.write_bytes(data);write_manifest(data)
            with mock.patch.object(m,'validate_manifest'):
                with self.assertRaisesRegex(ValueError,'nonfinite'):m.verify(root)

    def test_duplicate_hash_and_publication_preflight(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):json.loads('{"a":1,"a":2}',object_pairs_hook=m.unique)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad';path.write_bytes(b'{}')
            with self.assertRaises(ValueError):m.bounded(path,1,m.sha(path))
            with self.assertRaises(ValueError):m.bounded(path,10,'0'*64)
            with mock.patch.dict('sys.modules',{'numpy':None}):
                with self.assertRaises(FileExistsError):m.export(path)
                dangling=Path(directory)/'dangling';dangling.symlink_to(Path(directory)/'absent')
                with self.assertRaises(FileExistsError):m.export(dangling)

    def test_linear_composes_bounded_rows_across_boundary(self):
        import qwen_fp16
        rows,columns=1031,128
        # Scalar integer nibble extraction is independent of vectorized decode.
        packed=np.arange(rows*(columns//8),dtype='<u4').reshape(rows,-1)*np.uint32(918273)
        scales=np.resize(np.array([0.125,-0.25,0.0625],dtype='<f2'),(rows,2))
        biases=np.resize(np.array([-0.5,0.25,1],dtype='<f2'),(rows,2))
        dense=np.empty((rows,columns),dtype=np.float32)
        for row in range(rows):
            for col in range(columns):
                code=(int(packed[row,col//8])//(16**(col%8)))%16
                dense[row,col]=code*float(scales[row,col//64])+float(biases[row,col//64])
        class Owner:
            logical={'w':[rows,columns]}
            def __init__(self):self.calls=[]
            def matrix(self,name,start,end):
                self.calls.append((start,end))
                return m.decode(packed[start:end],scales[start:end],biases[start:end],columns)
        x=(np.arange(-192,192,dtype=np.float16)/np.float16(1024)).reshape(3,columns)
        for precision in [np.float32,np.float64]:
            owner=Owner()
            actual=qwen_fp16.linear(x,m.Matrix(owner,'w'),None,precision,np.float32)
            expected=(x.astype(precision)@dense.astype(precision).T).astype(np.float32)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(owner.calls,[(0,1024),(1024,1031)])

if __name__=='__main__':unittest.main()
