"""Independent slice layout, valid-prefix masking and stale-capacity tests."""
import copy
import math
import unittest
import rust_metal_cached_attention as cached
from test_rust_metal_attention import fixture as full_fixture


def fixture(item,anchor):
    # Build a full-shape evidence shell, then resize each rectangular diagnostic
    # from the independent saved slice and actual cached shape.
    shell={**item,'valid':item['rows'],'offset':0};event=full_fixture(shell,anchor);r=event['result']
    rows=item['rows'];valid=item['valid'];units=math.ceil(rows*4/32)*32;h=units*8+7;f=units*valid+7;shared=h*4+519*4+f*8
    r.update(offset=item['offset'],valid=valid,fp32_storage_elements=f,shared_buffer_bytes=shared,host_baseline_frames_readback_bytes=2*shared,admitted_bytes=3*shared+cached.rms.probe.ALLOWANCE)
    r['metadata']['maximum_admitted_bytes']=r['admitted_bytes']
    r['first_score_bits']=anchor['stages'][0]['bits']+[cached.attention.FLOAT_GUARD]*(f-4*rows*valid)
    r['first_probability_bits']=anchor['stages'][1]['bits']+[cached.attention.FLOAT_GUARD]*(f-4*rows*valid)
    return event


class CachedAttention(unittest.TestCase):
    def test_irregular_chunk_decode_gates_and_stale_capacity_pairs(self):
        inputs,anchors=cached.cases();self.assertEqual(len(inputs),24)
        self.assertEqual({c['offset'] for c in inputs},{0,1,4,11,24,31})
        self.assertEqual({c['rows'] for c in inputs},{1,3,7,13,8})
        for i in range(0,len(inputs),2):
            a,b=inputs[i:i+2];self.assertEqual(a['q_bits'],b['q_bits']);self.assertEqual(anchors[a['id']]['stages'],anchors[b['id']]['stages'])
            cached.attention.validate(fixture(a,anchors[a['id']]),a,anchors[a['id']]);cached.attention.validate(fixture(b,anchors[b['id']]),b,anchors[b['id']])
        item=inputs[-2];self.assertEqual((item['offset'],item['rows'],item['valid']),(31,1,32))

    def test_absolute_mask_offsets_and_parent_gates_are_required(self):
        inputs,anchors=cached.cases();item=inputs[2];anchor=anchors[item['id']]
        event=fixture(item,anchor);event['result']['offset']=0
        with self.assertRaises(ValueError):cached.attention.validate(event,item,anchor)
        reference=copy.deepcopy(anchor);reference['stages'][2]['reference'][0]+=100
        with self.assertRaisesRegex(ValueError,'source gate'):cached.attention.validate(fixture(item,anchor),item,reference)
        event=fixture(item,anchor);event['result']['first_probability_bits'][item['offset']+1]=0x80000000
        with self.assertRaises(ValueError):cached.attention.validate(event,item,anchor)

if __name__=='__main__':unittest.main()
