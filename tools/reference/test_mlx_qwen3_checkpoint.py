"""Portable candidate-inventory tests, independent of model downloads and GPUs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import struct
import mlx_qwen3_checkpoint as m


class Inventory(unittest.TestCase):
    def fixture(self, dtype='BF16'):
        header = {};cursor = 0
        def add(name, kind, shape):
            nonlocal cursor
            count = 1
            for width in shape:count *= width
            end = cursor + count * (4 if kind in ['U32', 'F32'] else 2)
            header[name] = dict(dtype=kind, shape=shape, data_offsets=[cursor, end]);cursor = end
        for name, shape in m.logical_shapes().items():
            if len(shape) == 1:add(name, dtype, shape)
            else:
                rows, columns = shape
                add(name, 'U32', [rows, columns // 4])
                for suffix in ['scales', 'biases']:
                    add(name.removesuffix('.weight') + '.' + suffix, dtype, [rows, columns // 64])
        index = dict(weight_map={name:'model.safetensors' for name in header}, metadata=dict(total_size=cursor))
        return header, index, cursor

    def test_complete_shape_inventory_separates_query_width_and_qk_norms(self):
        for dtype in ['BF16', 'F16', 'F32']:
            header, index, size = self.fixture(dtype)
            result = m.validate_header(header, index, size)
            self.assertEqual((result['tensors'], result['packed_matrices'], result['floating_vectors']), (704, 197, 113))
            self.assertEqual(result['dtype_counts'], dict(U32=197, **{dtype:507}))
            self.assertEqual(header['model.layers.0.self_attn.q_proj.weight']['shape'], [2048, 256])
            self.assertEqual(header['model.layers.0.self_attn.o_proj.weight']['shape'], [1024, 512])
            self.assertEqual(header['model.layers.27.self_attn.q_norm.weight']['shape'], [128])

    def test_missing_extra_wrong_norm_width_dtype_spans_and_index_reject(self):
        for mode in ['missing', 'extra', 'norm', 'dtype', 'metadata_dtype', 'gap', 'overlap', 'tail', 'index', 'total', 'bool']:
            h, i, size = self.fixture()
            norm = 'model.layers.0.self_attn.q_norm.weight';scales = 'model.layers.0.self_attn.q_proj.scales'
            if mode == 'missing':h.pop(norm)
            elif mode == 'extra':h['lm_head.weight'] = copy.deepcopy(h[norm])
            elif mode == 'norm':h[norm]['shape'] = [1024]
            elif mode == 'dtype':h[norm]['dtype'] = 'F64'
            elif mode == 'metadata_dtype':h[scales]['dtype'] = 'F16'
            elif mode == 'gap':h[norm]['data_offsets'] = [n + 1 for n in h[norm]['data_offsets']]
            elif mode == 'overlap':h[norm]['data_offsets'] = [n - 1 for n in h[norm]['data_offsets']]
            elif mode == 'tail':size += 1
            elif mode == 'index':i['weight_map'][norm] = 'other.safetensors'
            elif mode == 'total':i['metadata']['total_size'] -= 1
            elif mode == 'bool':h[norm]['data_offsets'][0] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError):m.validate_header(h, i, size)

    def test_bounded_json_hash_and_duplicate_fields_reject(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'data.json';raw = b'{"a":1}';path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(m.document(path, len(raw), digest), dict(a=1))
            for maximum, expected in [(len(raw)-1, digest), (len(raw), '0'*64)]:
                with self.assertRaises(ValueError):m.document(path, maximum, expected)
            raw = b'{"a":1,"a":2}';path.write_bytes(raw)
            with self.assertRaises(ValueError):m.document(path, len(raw), hashlib.sha256(raw).hexdigest())

    def test_header_authenticates_consumed_descriptor_during_path_substitution(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'weights';other=Path(root)/'replacement'
            raw=b'{"a":1}';original=struct.pack('<Q',len(raw))+raw+b'x'
            changed=struct.pack('<Q',len(raw))+b'{"b":2}'+b'x'
            path.write_bytes(original);other.write_bytes(changed)
            digest=hashlib.sha256(original).hexdigest()
            self.assertEqual(m.weight_header(path,len(original),digest),raw)
            opening=Path.open
            def substituted(candidate,*args,**kwargs):
                return opening(other if candidate==path and args==('rb',) else candidate,*args,**kwargs)
            before=m.sha(path)
            with patch.object(Path,'open',substituted):
                with self.assertRaisesRegex(ValueError,'Consumed weight descriptor'):m.weight_header(path,len(original),digest)
            self.assertEqual(m.sha(path),before)
            with path.open('rb') as stream:
                stream.seek(3);self.assertEqual(m.descriptor_sha(stream),digest);self.assertEqual(stream.tell(),3)

    def test_tokenizer_is_data_only_and_ambiguous_ids_reject(self):
        tokenizer = dict(model=dict(type='BPE', vocab={'a':0, 'b':1}), added_tokens=[dict(content='<stop>', id=2)])
        settings = dict(chat_template="{{ raise_exception('must not execute') }} enable_thinking", eos_token='<stop>', pad_token='<pad>', bos_token=None)
        config = dict(bos_token_id=0, eos_token_id=2)
        result = m.tokenizer_inventory(tokenizer, settings, config)
        self.assertFalse(result['chat_template_qualified']);self.assertFalse(result['tokenizer_execution_qualified'])
        self.assertTrue(result['template_mentions_enable_thinking'])
        for item in [dict(content='other', id=0), dict(content='other', id=True), dict(content='other', id=151936), dict(content='<stop>', id=3), dict(content='other', id=2)]:
            altered = copy.deepcopy(tokenizer);altered['added_tokens'].append(item)
            with self.assertRaises(ValueError):m.tokenizer_inventory(altered, settings, config)


if __name__ == '__main__':unittest.main()
