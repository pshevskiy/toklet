"""Frozen rotary offsets, head mapping, guards and source-gate failure checks."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_rotary as rotary


def fixture(inputs,anchors):
    cases=[]
    for item in inputs:
        count=item['rows']*item['heads']*8;rounded=math.ceil(count/2/32)*64;guards=rounded-count+7;size=(rounded+7)*2
        cases.append({'id':item['id'],'rows_count':item['rows'],'heads':item['heads'],'dimension':8,'theta':1_000_000.0,
            'offset':item['offset'],'pairing':'split_half','guards':guards,'output_guard_bits':rotary.GUARD,
            'allocations_reused_across_repetitions':True,'threadgroup_width':32,'threadgroups':math.ceil(count/2/32),
            'shared_buffer_bytes':size*2,'host_input_and_readback_bytes':size*3,'admitted_bytes':size*5+rotary.rms.probe.ALLOWANCE,
            'rows':[{'phase':'warmup' if i<3 else 'sample','repetition':i+1 if i<3 else i-2,'command_completed':True,
                     'immutable_input_and_guards':True,'encode_commit_wait_seconds':.001,
                     'output_bits':anchors[item['id']]['bits']+[rotary.GUARD]*guards} for i in range(13)]})
    return {'event':'idle_result','passed':True,'result':{'operation':'rotary_split_half_fp32_half_store','cases':cases,
        'metadata':{'unified_memory':True,'language_version':'2.3','math_mode':'safe','math_functions':'precise','minimum_macos_major':15,
                    'maximum_admitted_bytes':max(c['admitted_bytes'] for c in cases),'execution_width':32,'pipeline_max_threads':1024,'device_max_threads':[1024,1024,1024]}}}


class Rotary(unittest.TestCase):
    def test_frozen_offsets_heads_and_original_source_gates(self):
        inputs,anchors,_=rotary.cases()
        self.assertEqual(len(inputs),21);self.assertEqual(sum(c['rows']*c['heads']*8 for c in inputs),4240)
        self.assertEqual({c['offset'] for c in inputs},{0,5});self.assertEqual({c['heads'] for c in inputs},{2,4})
        event=fixture(inputs,anchors);rotary.validate(event,inputs,anchors)
        anchors[inputs[0]['id']]['reference'][0]+=100
        with self.assertRaisesRegex(ValueError,'source gate'):rotary.validate(fixture(inputs,anchors),inputs,anchors)

    def test_offset_pairing_inactive_guards_and_exact_manual_fail_closed(self):
        inputs,anchors,_=rotary.cases();inputs=inputs[-1:]
        for key,value in [('offset',1),('pairing','interleaved'),('heads',4),('theta',10000),('guards',7),('admitted_bytes',1)]:
            event=fixture(inputs,anchors);event['result']['cases'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):rotary.validate(event,inputs,anchors)
        for index in (0,16,-1):
            event=fixture(inputs,anchors);event['result']['cases'][0]['rows'][-1]['output_bits'][index]^=1
            with self.subTest(index=index),self.assertRaises(ValueError):rotary.validate(event,inputs,anchors)
        self.assertFalse(rotary.overflow_rejected({'event':'idle_result','passed':False,'error':{'stage':'validation','message':'completed rotary contains nonfinite Half; result discarded'}}))
        with self.assertRaises(ValueError):rotary.overflow_rejected({'event':'idle_result','passed':True})

    def test_source_shapes_and_path_aliases_are_rejected(self):
        original=rotary.rms.payload
        def changed(root,manifest,key,dtype):
            data,item=original(root,manifest,key,dtype)
            if key=='cases.offset.layers.0.q_rotary':item={**item,'shape':[3,2,16]}
            return data,item
        with patch.object(rotary.rms,'payload',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'shape'):rotary.cases()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'out.json';build=root/'build'
            with patch.object(rotary,'cases') as read:
                for inputs in (output,build,build/'inputs',root):
                    with self.assertRaises(ValueError):rotary.run(output,build,inputs)
                read.assert_not_called()

if __name__=='__main__':unittest.main()
