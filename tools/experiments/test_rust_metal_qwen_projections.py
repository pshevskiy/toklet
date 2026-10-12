"""Pinned Qwen input provenance, projection gates and fail-closed paths."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import rust_metal_qwen_projections as q
from test_rust_metal_projection import fixture


class QwenProjection(unittest.TestCase):
    def test_original_twelve_source_gates_and_repeat_policy(self):
        inputs,anchors=q.cases();self.assertEqual(len(inputs),12)
        for item in inputs:
            item={**item,'bias_bits':[0]*item['outputs']};event=fixture([item],anchors);q.validate(event,item,anchors[item['id']])
        item=inputs[0];event=fixture([item],anchors);event['result']['cases'][0]['rows'][-1]['fp32_bits'][0]^=1
        with self.assertRaisesRegex(ValueError,'repeated'):q.validate(event,item,anchors[item['id']])
        bad=copy.deepcopy(anchors[item['id']]);bad['source'][0]+=1000
        with self.assertRaisesRegex(ValueError,'Frozen projection'):q.validate(fixture([item],anchors),item,bad)

    def test_bounded_metadata_and_nonfinite_bf16_fail_before_conversion(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'fake.safetensors'
            for meta in [{}, {'model.layers.0.self_attn.q_proj.weight':{'dtype':'F32','shape':[896,896],'data_offsets':[0,2]}}, {'model.layers.0.self_attn.q_proj.weight':{'dtype':'BF16','shape':[896,896],'data_offsets':[0,2]}}]:
                header=json.dumps(meta).encode();data=struct.pack('<Q',len(header))+header+b'\0\0';path.write_bytes(data)
                with self.assertRaises(ValueError):q.weights(path,hashlib.sha256(data).hexdigest(),len(data))
            meta={'model.layers.0.self_attn.q_proj.weight':{'dtype':'BF16','shape':[896,896],'data_offsets':[0,896*896*2]}}
            header=json.dumps(meta).encode();data=struct.pack('<Q',len(header))+header+struct.pack('<H',0x7f80)+b'\0\0'*(896*896-1);path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'Nonfinite'):q.weights(path,hashlib.sha256(data).hexdigest(),len(data))

    def test_path_aliases_rejected_before_frozen_or_weight_reads(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)
            with patch.object(q,'cases')as read:
                for directory in(root,root/'out.json',root/'build',root/'build/inputs'):
                    with self.assertRaises(ValueError):q.run(root/'out.json',root/'build',directory)
                read.assert_not_called()

if __name__=='__main__':unittest.main()
