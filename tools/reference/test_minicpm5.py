"""CPU-only independent reader/graph checks; no model download or GPU."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import minicpm5 as subject
import mlx_llama as frozen
import minicpm5_reference as exporter


class Reader(unittest.TestCase):
    def test_bf16_literal_bit_patterns_and_affine_group_lane_order(self):
        values=subject.bf16(bytes.fromhex('c03f10c000000080'),[4])
        np.testing.assert_array_equal(values,np.array([1.5,-2.25,0.,-0.],dtype=np.float32))
        self.assertTrue(np.signbit(values[-1]))
        codes=np.arange(128,dtype=np.uint32).reshape(1,128)
        words=sum(codes[:,lane::4] << (8*lane) for lane in range(4)).astype('<u4')
        result=subject.decode(words,np.array([[1.5,.25]],dtype=np.float32),
                              np.array([[-2.25,1.]],dtype=np.float32),128)
        expected=np.array([[float(i)*1.5-2.25 if i<64 else float(i)*.25+1 for i in range(128)]],dtype=np.float32)
        np.testing.assert_array_equal(result,expected)
        np.testing.assert_array_equal(result[0,[0,1,2,3,63,64,127]],[-2.25,-.75,.75,2.25,92.25,17,32.75])

    def test_nonfinite_wrong_dtypes_and_incomplete_groups_reject(self):
        with self.assertRaises(ValueError):subject.bf16(bytes.fromhex('807f'),[1])
        with self.assertRaises(ValueError):subject.bf16(b'\0',[1])
        packed=np.zeros((1,16),dtype='<u4');scales=np.ones((1,1),dtype=np.float32);bias=np.zeros_like(scales)
        for args in [(packed.astype('<i4'),scales,bias,64),(packed,scales.astype(np.float16),bias,64),
                     (packed,scales,bias[:,:0],64),(packed,scales,bias,63),
                     (packed,scales*np.float32(float('nan')),bias,64)]:
            with self.assertRaises(ValueError):subject.decode(*args)

    def test_literal_complete_model_inventory_and_distinct_output(self):
        shapes=subject.logical_shapes()
        self.assertEqual(len(shapes),381)
        self.assertEqual(sum(np.prod(s) for s in shapes.values()),2516756480)
        self.assertEqual(shapes['lm_head.weight'],[130560,2048])
        self.assertEqual(shapes['model.layers.41.self_attn.k_proj.weight'],[256,2048])
        self.assertEqual(shapes['model.layers.0.mlp.down_proj.weight'],[2048,6144])
        self.assertEqual(subject.POLICY,dict(atol_floor=1e-4,envelope_multiplier=2,rtol=1e-3))
        self.assertEqual(subject.CASES['offset509'],dict(tokens=[7,13,3],position_offset=509))

    def test_unaligned_source_origin_and_exact_pread_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unaligned.bin'
            path.write_bytes(bytes.fromhex('abcdc03f10c0003f0040'))
            source=subject.Source(directory)
            source.origin=2
            source.header={'norm':dict(dtype='BF16',shape=[2,2],data_offsets=[0,8])}
            with path.open('rb') as stream:
                source.stream=stream
                np.testing.assert_array_equal(source.raw('norm',1,2),[[.5,2.]])
                with self.assertRaises(ValueError):source.raw('norm',True,2)
                source.origin=4
                with self.assertRaisesRegex(ValueError,'Truncated'):source.raw('norm',1,2)

    def test_prospective_array_coverage_and_bounded_reference_ledger(self):
        arrays=exporter.expected_arrays()
        self.assertEqual(len(arrays),349)
        self.assertEqual(arrays['cases.tile17.layers.41.k_projection'],([17,256],'<f2'))
        self.assertEqual(arrays['cases.offset509.layers.1.attention_scores'],([16,3,3],'<f4'))
        self.assertEqual(arrays['greedy.step_3.logits'],([130560],'<f4'))
        self.assertEqual(exporter.admission()['admitted_bytes'],2448*1024**2)
        self.assertLess(sum(np.prod(shape)*np.dtype(dtype).itemsize for shape,dtype in arrays.values()),128*1024**2)


def tiny_weights(variant):
    """Independent scalar extraction of the unchanged tiny publisher fixture."""
    folder=frozen.DEFAULT/variant
    raw=(folder/'model.safetensors').read_bytes();length=int.from_bytes(raw[:8],'little')
    header=json.loads(raw[8:8+length]);origin=8+length
    config=json.loads((folder/'config.json').read_text());quant=config['quantization']
    def tensor(name):
        item=header[name];a,b=item['data_offsets'];data=raw[origin+a:origin+b]
        fmt={'F16':'e','F32':'f','U32':'I'}[item['dtype']]
        return list(struct.unpack('<'+fmt*(len(data)//struct.calcsize(fmt)),data))
    weights={}
    for name,item in header.items():
        if name=='__metadata__' or name.endswith(('.scales','.biases')):continue
        shape=item['shape']
        if item['dtype']=='U32':
            module=name.removesuffix('.weight');options=quant.get(module,True)
            bits=options.get('bits',quant['bits']) if isinstance(options,dict) else quant['bits']
            group=options.get('group_size',quant['group_size']) if isinstance(options,dict) else quant['group_size']
            rows=shape[0];cols=shape[1]*32//bits;words=tensor(name)
            scales,bias=tensor(module+'.scales'),tensor(module+'.biases')
            value=np.empty((rows,cols),dtype=np.float32)
            for row in range(rows):
                for col in range(cols):
                    code=(words[row*shape[1]+col*bits//32]>>(col*bits%32))&((1<<bits)-1)
                    metadata=row*(cols//group)+col//group
                    value[row,col]=float(code)*scales[metadata]+bias[metadata]
        else:value=np.array(tensor(name),dtype=np.float16).reshape(shape)
        key=name.removeprefix('model.').replace('.self_attn.','.').replace('.mlp.','.')
        if key=='embed_tokens.weight':key='embed_tokens'
        elif key=='lm_head.weight':key='output'
        elif key=='norm.weight':key='norm'
        elif key.endswith(('input_layernorm.weight','post_attention_layernorm.weight')):key=key.removesuffix('.weight')
        weights[key]=value
    if config['tie_word_embeddings']:weights['output']=weights['embed_tokens']
    weights['embed_tokens']=weights['embed_tokens'].astype(np.float16)
    return weights


class FrozenGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=frozen.verify(frozen.DEFAULT,'824d2ee2ae9dd5b623c5aa9db10c13b2e4dee32647ea5fa9d78274fb3529a1f5')
        cls.data=(frozen.DEFAULT/'arrays.bin').read_bytes()

    def test_tied_and_distinct_head_all_frozen_half_stages_cases_and_profiles(self):
        compared=0
        for variant in ['tied','untied']:
            weights=tiny_weights(variant);config=self.manifest['variants'][variant]['geometry']
            for case,spec in frozen.CASES.items():
                for precision in [np.float32,np.float64]:
                    trace=subject.forward(spec['tokens'],spec['position_offset'],weights,config,precision,[0,1])
                    self.assertEqual(len(trace),39)
                    for stage,actual in trace.items():
                        item=self.manifest['arrays'][f'{variant}.half.cases.{case}.{stage}']
                        expected=np.frombuffer(self.data[item['offset']:item['offset']+item['bytes']],dtype=item['dtype'])
                        self.assertEqual(actual.size,expected.size)
                        self.assertEqual(actual.dtype,np.float32 if stage.endswith(('attention_scores','attention_probabilities','logits')) else np.float16,stage)
                        gate=item['gate']
                        error=np.abs(actual.ravel().astype(np.float64)-expected.astype(np.float64))
                        self.assertTrue(np.all(error<=gate['atol']+gate['rtol']*np.abs(expected)),f'{variant}.{case}.{stage}.{precision}')
                        compared+=1
        self.assertEqual(compared,780)

    def test_wrong_head_changes_distribution_and_future_tokens_preserve_prefix(self):
        weights=tiny_weights('untied');config=self.manifest['variants']['untied']['geometry']
        actual=subject.forward([1,7,13,3],0,weights,config,np.float64,[0,1])
        wrong=subject.forward([1,7,13,3],0,dict(weights,output=weights['embed_tokens']),config,np.float64,[0,1])
        self.assertNotEqual(int(np.argmax(actual['logits'][-1])),int(np.argmax(wrong['logits'][-1])))
        changed=subject.forward([1,7,13,17],0,weights,config,np.float64,[0,1])
        for name in ['embedding','layers.1.output','logits']:
            np.testing.assert_array_equal(actual[name][:3],changed[name][:3])


if __name__=='__main__':unittest.main()
