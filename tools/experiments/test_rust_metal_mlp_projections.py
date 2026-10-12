"""Frozen MLP projection inventory, gates, batch bounds and path safety."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_mlp_projections as mlp
from test_rust_metal_projection import fixture


class MlpProjections(unittest.TestCase):
    def test_all_layers_cases_shapes_and_original_gates_are_required(self):
        groups,anchors=mlp.batches()
        self.assertEqual([name for name,_ in groups],['gate_layer_0','gate_layer_1','up_layer_0','up_layer_1','down_layer_0','down_layer_1'])
        self.assertEqual(len(anchors),30)
        self.assertEqual(sum(c['rows']*c['outputs'] for _,cases in groups for c in cases),14080)
        for name,cases in groups:
            self.assertEqual(len(cases),5)
            self.assertTrue(all(c['bias_bits']==[0]*c['outputs'] for c in cases))
            expected=(64,32) if name.startswith('down') else (32,64)
            self.assertEqual({(c['inputs'],c['outputs']) for c in cases},{expected})
            event=fixture(cases,anchors);mlp.projection.validate(event,cases,anchors)
        event=fixture(groups[0][1],anchors)
        anchors[groups[0][1][0]['id']]['source'][0]+=100
        with self.assertRaisesRegex(ValueError,'Frozen projection gate'):
            mlp.projection.validate(event,groups[0][1],anchors)

    def test_corrupt_or_mis_shaped_payload_is_rejected_before_native_work(self):
        original=mlp.rms.payload
        def changed(root,manifest,key,dtype):
            data,item=original(root,manifest,key,dtype)
            if key=='weights.layers.0.gate_proj.weight':item={**item,'shape':[32,64]}
            return data,item
        with patch.object(mlp.rms,'payload',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'weight shape'):mlp.batches()
        def nonfinite(root,manifest,key,dtype):
            data,item=original(root,manifest,key,dtype)
            if key=='cases.capacity.layers.0.gated':data=b'\x00\x7c'+data[2:]
            return data,item
        with patch.object(mlp.rms,'payload',side_effect=nonfinite):
            with self.assertRaisesRegex(ValueError,'nonfinite'):mlp.batches()

    def test_aliases_and_protected_fixtures_fail_before_source_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'out.json';build=root/'build'
            with patch.object(mlp,'batches') as read:
                for inputs in (output,build,build/'inputs',root):
                    with self.assertRaises(ValueError):mlp.run(output,build,inputs)
                for source in (mlp.rms.FP16,mlp.rms.FP32):
                    with self.assertRaises(ValueError):mlp.run(source/'new.json',build,root/'inputs')
                read.assert_not_called()

if __name__=='__main__':unittest.main()
