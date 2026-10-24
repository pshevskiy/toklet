"""Portable routing/identity witnesses; no numerical or Apple runtime import."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gemma_shared_kv as oracle

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/m8-gemma4-source-v1/mlx-lm/gemma4_text.py'


class SharedRouting(unittest.TestCase):
    def test_actual_producer_map_and_independent_small_maps(self):
        actual=[*range(15),*[13 if i%5!=4 else 14 for i in range(15,35)]]
        self.assertEqual(oracle.mapping(oracle.SPECS[0][1],20),actual)
        self.assertEqual(oracle.mapping((oracle.LOCAL,oracle.FULL,oracle.LOCAL,oracle.FULL,oracle.LOCAL,oracle.FULL),3),[0,1,2,1,2,1])
        self.assertEqual(oracle.mapping((oracle.LOCAL,oracle.FULL,oracle.LOCAL,oracle.FULL),0),[0,1,2,3])
        self.assertEqual(oracle.mapping((oracle.FULL,oracle.LOCAL,oracle.LOCAL,oracle.FULL,oracle.LOCAL),2),[0,1,2,0,2])

    def test_invalid_mapping_and_missing_preceding_type_reject(self):
        for types,shared in [((),0),((oracle.LOCAL,),1),((oracle.LOCAL,),True),((oracle.LOCAL,),-1),((oracle.LOCAL,oracle.FULL),1),(('unknown',),0),((True,),0),((oracle.LOCAL,)*36,0),('text',0)]:
            with self.assertRaises(ValueError):oracle.mapping(types,shared)

    def test_expected_layers_positions_ops_and_append_counts(self):
        data=json.loads(oracle.export());self.assertEqual(len(data['cases']),4)
        calls=0
        for case in data['cases']:
            offset=0;owned=len(case['types'])-case['shared']
            for index,step in enumerate(case['steps']):
                self.assertEqual(step['before'],offset);offset+=case['chunks'][index];self.assertEqual(step['after'],offset)
                self.assertEqual(step['cache_offsets'],[offset]*owned);self.assertEqual(step['cache_updates'],[index+1]*owned)
                for layer in step['layers']:
                    own=layer['layer']<owned;self.assertEqual(layer['cache_present'],own)
                    self.assertEqual(layer['shared_identity_current_step'],not own);self.assertEqual(layer['shared_kv_present'],not own)
                    self.assertEqual(layer['ops']['append'],int(own))
                    for op in ('k_proj','v_proj','k_norm','v_norm','k_rope'):self.assertEqual(layer['ops'][op],int(own))
                    for op in ('q_proj','q_norm','q_rope'):self.assertEqual(layer['ops'][op],1)
                    calls+=1
        self.assertEqual(calls,226)

    def test_strict_shape_only_stand_ins_reject_unsupported_operations(self):
        tensor=oracle.Tensor((1,2,3),'hidden')
        for operation in [lambda:tensor.reshape(1,5),lambda:tensor.reshape(-1,-1),lambda:tensor.reshape(0,6),lambda:tensor.transpose(1,1,0),lambda:tensor*1.,lambda:tensor+oracle.Tensor((1,2,4),'hidden'),lambda:oracle.Identity(9),lambda:oracle.Identity()(7),lambda:oracle.Embedding(8,1536)(None)]:
            with self.assertRaises(ValueError):operation()
        self.assertEqual(tensor.reshape(1,3,2).transpose(0,2,1).shape,tensor.shape)

    def test_pinned_source_selection_and_missing_shared_source_guard(self):
        raw=SOURCE.read_bytes();nodes=oracle.source_nodes(raw)
        self.assertEqual([(n.name,[m.name for m in n.body]) for n in nodes],[('Attention',['__call__']),('DecoderLayer',['__call__']),('Gemma4TextModel',['__init__','_make_masks','__call__'])])
        with self.assertRaises(ValueError):oracle.source_nodes(raw+b'\n')
        replay=oracle.Replay(nodes)
        config=oracle.SimpleNamespace(layer_types=[oracle.LOCAL,oracle.FULL,oracle.LOCAL],num_hidden_layers=3,num_kv_shared_layers=1)
        layer=replay.layer(config,2);replay.count=1
        with self.assertRaisesRegex(ValueError,'KV-shared layer but received no shared_kv'):
            layer.self_attn(oracle.Tensor((1,1,1536),'hidden'),None,None,None,None)
        self.assertEqual(replay.calls[0]['ops']['append'],0)

    def test_normal_selected_source_routing_and_current_step_identity(self):
        data=json.loads(oracle.export());observed=oracle.compare(data,SOURCE.read_bytes())
        self.assertTrue(observed['passed'],observed['primary_error'])
        self.assertFalse(observed['numerical_execution_qualified']);self.assertFalse(observed['native_stale_rejection_qualified'])
        for case in observed['cases']:
            self.assertTrue(case['completed'])
            for step in case['steps']:
                for layer in step['layers']:
                    if layer['shared_kv_present']:
                        self.assertFalse(layer['cache_present']);self.assertTrue(layer['shared_identity_current_step'])
                        self.assertEqual(layer['returned_offset'],layer['shared_offset'])

    def test_captured_identity_and_partial_observations_survive_failure(self):
        raw=SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'source';path.write_bytes(raw)
            self.assertEqual(oracle.captured(path,oracle.SOURCE_SHA,128*1024),raw)
            for maximum,digest in [(1,oracle.SOURCE_SHA),(128*1024,'0'*64)]:
                with self.assertRaises(ValueError):oracle.captured(path,digest,maximum)
            alias=path.parent/'alias';alias.symlink_to(path)
            with self.assertRaises(ValueError):oracle.captured(alias,oracle.SOURCE_SHA,128*1024)
        original=oracle.Replay.attention
        def failure(replay,*args,**kwargs):
            if replay.active['layer']==15:raise RuntimeError('synthetic consumer failure')
            return original(replay,*args,**kwargs)
        with mock.patch.object(oracle.Replay,'attention',failure):observed=oracle.compare(json.loads(oracle.export()),raw)
        self.assertFalse(observed['passed']);self.assertIn('synthetic consumer failure',observed['primary_error'])
        first=observed['cases'][0]['steps'][0];self.assertFalse(first['completed']);self.assertEqual(len(first['layers']),16)
        self.assertTrue(first['layers'][15]['shared_identity_current_step']);self.assertEqual(first['layers'][14]['ops']['append'],1)


if __name__=='__main__':unittest.main()
