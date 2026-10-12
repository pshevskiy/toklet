"""Frozen attention projection inventory, gates, batch bounds and path safety."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_attention_projections as attention
from test_rust_metal_projection import fixture


class AttentionProjections(unittest.TestCase):
    def test_all_layers_cases_shapes_and_original_gates_are_required(self):
        groups,anchors=attention.batches()
        self.assertEqual([name for name,_ in groups],['k_layer_0','k_layer_1','v_layer_0','v_layer_1','o_layer_0','o_layer_1'])
        self.assertEqual(len(anchors),30)
        self.assertEqual(sum(c['rows']*c['outputs'] for _,cases in groups for c in cases),5632)
        for name,cases in groups:
            self.assertEqual(len(cases),5)
            self.assertTrue(all((c['bias_bits']==[0]*c['outputs']) == name.startswith('o') for c in cases))
            expected=(32,32) if name.startswith('o') else (32,16)
            self.assertEqual({(c['inputs'],c['outputs']) for c in cases},{expected})
            event=fixture(cases,anchors);attention.projection.validate(event,cases,anchors)
        event=fixture(groups[0][1],anchors)
        anchors[groups[0][1][0]['id']]['source'][0]+=100
        with self.assertRaisesRegex(ValueError,'Frozen projection gate'):
            attention.projection.validate(event,groups[0][1],anchors)

    def test_corrupt_or_mis_shaped_payload_is_rejected_before_native_work(self):
        original=attention.rms.payload
        def changed(root,manifest,key,dtype):
            data,item=original(root,manifest,key,dtype)
            if key=='weights.layers.0.k_proj.weight':item={**item,'shape':[32,16]}
            return data,item
        with patch.object(attention.rms,'payload',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'weight shape'):attention.batches()
        def nonfinite(root,manifest,key,dtype):
            data,item=original(root,manifest,key,dtype)
            if key=='cases.capacity.layers.0.input_norm':data=b'\x00\x7c'+data[2:]
            return data,item
        with patch.object(attention.rms,'payload',side_effect=nonfinite):
            with self.assertRaisesRegex(ValueError,'nonfinite'):attention.batches()

    def test_aliases_and_protected_fixtures_fail_before_source_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'out.json';build=root/'build'
            with patch.object(attention,'batches') as read:
                for inputs in (output,build,build/'inputs',root):
                    with self.assertRaises(ValueError):attention.run(output,build,inputs)
                for source in (attention.rms.FP16,attention.rms.FP32):
                    with self.assertRaises(ValueError):attention.run(source/'new.json',build,root/'inputs')
                read.assert_not_called()

if __name__=='__main__':unittest.main()
