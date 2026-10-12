"""Propagated stage gates, actual K/V copying and graph admission checks."""
import copy
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import rust_metal_layer as graph


def fixture(item,anchor):
    rows=item['rows'];units=math.ceil(rows*4/32)*32;qcap=max(math.ceil(units*8/32),math.ceil(rows/32)*32);wcap=math.ceil(math.ceil(rows*64/32)*32/64);kcap=math.ceil(math.ceil(rows*8/32)*64/16)
    def layout(r,c,s,capacity):return {'rows':r,'columns':c,'stride':c,'elements':capacity*c+7,'bytes':(capacity*c+7)*s}
    il=[layout(*a) for a in [(rows,32,2,rows),(32,32,2,32),(16,32,2,16),(16,32,2,16),(32,32,2,32),(1,32,2,1),(1,16,2,1),(1,16,2,1),(1,64,2,1),(1,32,2,1),(1,32,2,1),(64,32,2,64),(64,32,2,64),(32,64,2,32)]]
    hl=[layout(*a) for a in [(rows,32,2,qcap),(rows,16,2,kcap),(rows,16,2,kcap),(rows,32,2,qcap),(rows,16,2,kcap),(2,256,2,2),(2,256,2,2),(rows,32,2,qcap),(rows,32,2,qcap)]+[(rows,32,2,qcap)]*3+[(rows,64,2,wcap)]*4+[(rows,32,2,qcap)]*2]
    fl=[layout(*a) for a in [(rows,32,4,qcap),(rows,16,4,kcap),(rows,16,4,kcap),(rows*4,rows,4,units),(rows*4,rows,4,units),(rows,32,4,qcap)]+[(rows,64,4,wcap)]*2+[(rows,32,4,qcap)]]
    halves=[s['bits'].copy() for s in anchor['stages'][:18]]
    def single(bits):return list(struct.unpack(f'<{len(bits)}I',struct.pack(f'<{len(bits)}f',*struct.unpack(f'<{len(bits)}e',struct.pack(f'<{len(bits)}H',*bits)))))
    floats=[single(halves[0]),single(halves[1]),single(halves[2]),anchor['stages'][18]['bits'].copy(),anchor['stages'][19]['bits'].copy(),single(halves[8]),single(halves[12]),single(halves[14]),single(halves[16])]
    shared=sum(l['bytes'] for group in (il,hl,fl) for l in group)
    return {'event':'idle_result','passed':True,'result':{'operation':'resident_tiny_layer_half_boundaries','id':item['id'],'rows_count':rows,'position':item['position'],
        'metadata':{'mps_device_supported':True,'unified_memory':True,'language_version':'2.3','math_mode':'safe','math_functions':'precise','minimum_macos_major':15,
                    'execution_width':32,'pipeline_max_threads':1024,'device_max_threads':[1024]*3,'maximum_admitted_bytes':3*shared+graph.rms.probe.ALLOWANCE,
                    'graph_pipeline_caps':[{'name':n,'execution_width':32,'max_threads':1024} for n in ('toklet_projection_bias','toklet_rotary','toklet_cache_write','toklet_attention','toklet_rms_norm','toklet_residual','toklet_silu','toklet_gating')]},
        'threadgroup_width':32,'input_layouts':il,'half_layouts':hl,'float_layouts':fl,'shared_buffer_bytes':shared,'host_snapshot_readback_bytes':2*shared,'admitted_bytes':3*shared+graph.rms.probe.ALLOWANCE,
        'allocations_reused':True,'first_half_intermediates':halves,'first_fp32_intermediates':floats,
        'rows':[{'phase':'warmup' if i<3 else 'sample','repetition':i+1 if i<3 else i-2,'command_completed':True,'commit_wait_pairs':1,'inputs_guards_and_intermediates_unchanged':True,'encode_commit_wait_seconds':.001,'output_bits':halves[17].copy()} for i in range(13)]}}


class Layer(unittest.TestCase):
    def test_all_original_eighteen_source_gates_and_two_cache_layouts(self):
        inputs,anchors,_=graph.cases();self.assertEqual(len(inputs),10)
        for item in inputs:graph.validate(fixture(item,anchors[item['id']]),item,anchors[item['id']])
        item=inputs[0];anchor=copy.deepcopy(anchors[item['id']]);anchor['stages'][7]['source'][0]+=100
        with self.assertRaisesRegex(ValueError,'source gate'):graph.validate(fixture(item,anchors[item['id']]),item,anchor)

    def test_actual_gpu_operands_half_boundaries_layout_and_wait_fail_closed(self):
        inputs,anchors,_=graph.cases();item=inputs[0];anchor=anchors[item['id']]
        for kind in ('cache','cast','float_guard_shape','wait','memory','type','residual','gating'):
            event=fixture(item,anchor);r=event['result']
            if kind=='cache':r['first_half_intermediates'][5][0]^=1
            elif kind=='cast':r['first_fp32_intermediates'][0][0]=0x3f800000
            elif kind=='float_guard_shape':r['float_layouts'][3]['elements']-=1
            elif kind=='wait':r['rows'][-1]['commit_wait_pairs']=2
            elif kind=='memory':r['admitted_bytes']-=1
            elif kind=='type':r['first_half_intermediates'][0][0]=True
            elif kind=='residual':r['first_half_intermediates'][10][0]^=1
            elif kind=='gating':r['first_half_intermediates'][15][0]^=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):graph.validate(event,item,anchor)

    def test_path_aliases_rejected_before_fixture_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(graph,'cases') as read:
                for directory in (root,root/'out.json',root/'build',root/'build/inputs'):
                    with self.assertRaises(ValueError):graph.run(root/'out.json',root/'build',directory)
                read.assert_not_called()

if __name__=='__main__':unittest.main()
