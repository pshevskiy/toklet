"""Original GQA intermediate gates, causal masking and admission failure checks."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import rust_metal_attention as attention


def fixture(item,anchor):
    rows=item['rows'];units=math.ceil(rows*4/32)*32;h=units*8+7;f=units*rows+7;shared=h*4+519*4+f*8
    score,probability,context=[s['bits'] for s in anchor['stages']]
    return {'event':'idle_result','passed':True,'result':{'operation':'tiny_causal_gqa_fp32_half_context','id':item['id'],'rows_count':rows,'query_heads':4,'kv_heads':2,'dimension':8,'capacity':32,
        'metadata':{'unified_memory':True,'language_version':'2.3','math_mode':'safe','math_functions':'precise','minimum_macos_major':15,'execution_width':32,'pipeline_max_threads':1024,'device_max_threads':[1024]*3,'maximum_admitted_bytes':shared*3+attention.rms.probe.ALLOWANCE},
        'threadgroup_width':32,'threadgroups':math.ceil(rows*4/32),'half_storage_elements':h,'fp32_storage_elements':f,'half_guard_bits':attention.HALF_GUARD,'fp32_guard_bits':attention.FLOAT_GUARD,
        'allocations_reused':True,'shared_buffer_bytes':shared,'host_baseline_frames_readback_bytes':shared*2,'admitted_bytes':shared*3+attention.rms.probe.ALLOWANCE,
        'first_score_bits':score+[attention.FLOAT_GUARD]*(f-len(score)),'first_probability_bits':probability+[attention.FLOAT_GUARD]*(f-len(probability)),
        'first_context_bits':context+[attention.HALF_GUARD]*(h-len(context)),
        'rows':[{'phase':'warmup' if i<3 else 'sample','repetition':i+1 if i<3 else i-2,'command_completed':True,'inputs_guards_intermediates_valid':True,'commit_wait_pairs':1,'encode_commit_wait_seconds':.001,'context_bits':context.copy()} for i in range(13)]}}


class Attention(unittest.TestCase):
    def test_all_frozen_scores_probabilities_context_gates_are_required(self):
        inputs,anchors=attention.cases();self.assertEqual(len(inputs),10)
        for item in inputs:attention.validate(fixture(item,anchors[item['id']]),item,anchors[item['id']])
        item=inputs[0];anchor=anchors[item['id']];anchor['stages'][0]['reference'][0]+=100
        with self.assertRaisesRegex(ValueError,'source gate'):attention.validate(fixture(item,anchor),item,anchor)

    def test_masked_probability_signed_zero_guards_repetitions_and_memory_fail(self):
        inputs,anchors=attention.cases();item=next(c for c in inputs if c['rows']==4);anchor=anchors[item['id']]
        for kind in ('masked_zero','guard','context','commit','memory','headmap'):
            event=fixture(item,anchor);r=event['result']
            if kind=='masked_zero':r['first_probability_bits'][1]=0x80000000
            elif kind=='guard':r['first_score_bits'][-1]^=1
            elif kind=='context':r['rows'][-1]['context_bits'][0]^=1
            elif kind=='commit':r['rows'][-1]['commit_wait_pairs']=2
            elif kind=='memory':r['admitted_bytes']-=1
            elif kind=='headmap':r['kv_heads']=4
            with self.subTest(kind=kind),self.assertRaises(ValueError):attention.validate(event,item,anchor)

    def test_source_shapes_and_path_aliases_rejected(self):
        original=attention.rms.payload
        def changed(root,manifest,key,dtype):
            data,item=original(root,manifest,key,dtype)
            if key=='cases.offset.layers.0.cache_k':item={**item,'shape':[32,2,8]}
            return data,item
        with patch.object(attention.rms,'payload',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'shape'):attention.cases()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(attention,'cases') as read:
                for directory in (root,root/'out.json',root/'build',root/'build/inputs'):
                    with self.assertRaises(ValueError):attention.run(root/'out.json',root/'build',directory)
                read.assert_not_called()

if __name__=='__main__':unittest.main()
