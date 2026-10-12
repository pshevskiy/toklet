"""Lossless layout oracle, saved cache policy, capacity and untouched-state checks."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import rust_metal_cache as cache


def fixture(inputs,anchors):
    cases=[]
    for item in inputs:
        count=item['rows']*16;rounded=math.ceil(count/32)*32;shared=2*((rounded+7)*2+519*2)
        cases.append({'id':item['id'],'rows_count':item['rows'],'offset':item['offset'],'heads':2,'dimension':8,'capacity':32,
            'layout':'heads_capacity_dimension','input_guard_values':rounded-count+7,'cache_guard_values':7,'k_guard_bits':cache.K_GUARD,'v_guard_bits':cache.V_GUARD,
            'committed_position_advanced':False,'allocations_reused_across_repetitions':True,'threadgroup_width':32,'threadgroups':math.ceil(count/32),
            'shared_buffer_bytes':shared,'host_baseline_readback_bytes':2*shared,'admitted_bytes':3*shared+cache.rms.probe.ALLOWANCE,
            'rows':[{'phase':'warmup' if i<3 else 'sample','repetition':i+1 if i<3 else i-2,'command_completed':True,
                     'inputs_targets_untouched_and_guards_valid':True,'encode_commit_wait_seconds':.001,
                     'cache_k_bits':anchors[item['id']]['k']+[cache.K_GUARD]*7,'cache_v_bits':anchors[item['id']]['v']+[cache.V_GUARD]*7} for i in range(13)]})
    return {'event':'idle_result','passed':True,'result':{'operation':'kv_half_token_to_head_major_write','cases':cases,
        'metadata':{'unified_memory':True,'language_version':'2.3','math_mode':'safe','math_functions':'precise','minimum_macos_major':15,
                    'maximum_admitted_bytes':max(c['admitted_bytes'] for c in cases),'execution_width':32,'pipeline_max_threads':1024,'device_max_threads':[1024,1024,1024]}}}


class Cache(unittest.TestCase):
    def test_frozen_layout_manual_edges_and_independent_head_mapping(self):
        inputs,anchors=cache.cases();self.assertEqual(len(inputs),14)
        self.assertEqual({c['offset'] for c in inputs},{0,7,31})
        self.assertEqual([a['rotary_position_offset'] for a in anchors.values() if 'rotary_position_offset' in a].count(5),2)
        cache.validate(fixture(inputs,anchors),inputs,anchors)
        c=inputs[-4];k,v=cache.expected(c)
        self.assertEqual(k[31*8:32*8],c['k_bits'][:8]);self.assertEqual(k[63*8:64*8],c['k_bits'][8:])
        self.assertEqual(v[:8],c['initial_v_bits'][:8]);self.assertEqual(k[32*8:33*8],c['initial_k_bits'][32*8:33*8])

    def test_targets_prefix_tail_guards_and_commit_marker_fail_closed(self):
        inputs,anchors=cache.cases();inputs=inputs[-4:-3]
        for index in (0,31*8,32*8,63*8,512,-1):
            e=fixture(inputs,anchors);e['result']['cases'][0]['rows'][-1]['cache_k_bits'][index]^=1
            with self.subTest(index=index),self.assertRaises(ValueError):cache.validate(e,inputs,anchors)
        e=fixture(inputs,anchors);e['result']['cases'][0]['committed_position_advanced']=True
        with self.assertRaises(ValueError):cache.validate(e,inputs,anchors)
        with self.assertRaises(ValueError):cache.rejected({'event':'idle_result','passed':True})

    def test_aliases_and_source_shapes_fail_before_native_work(self):
        original=cache.rms.payload
        def changed(root,manifest,key,dtype):
            data,item=original(root,manifest,key,dtype)
            if key=='cases.offset.layers.0.cache_k':item={**item,'shape':[32,2,8]}
            return data,item
        with patch.object(cache.rms,'payload',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'shape'):cache.cases()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'out.json';build=root/'build'
            with patch.object(cache,'cases') as read:
                for inputs in (output,build,build/'inputs',root):
                    with self.assertRaises(ValueError):cache.run(output,build,inputs)
                read.assert_not_called()

if __name__=='__main__':unittest.main()
