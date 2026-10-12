"""Independent full intermediate gates and composed MLP graph contracts."""
import copy
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_mlp as mlp


def fixture(case,anchor):
    rows=case['rows'];width=32
    h=[a['bits'].copy()for a in anchor['stages']]
    floats=[]
    for index in (0,2,4):
        values=struct.unpack(f'<{len(h[index])}e',struct.pack(f'<{len(h[index])}H',*h[index]))
        floats.append(list(struct.unpack(f'<{len(values)}I',struct.pack(f'<{len(values)}f',*values))))
    inp=(rows*32+7)*2+2*(64*32+7)*2+(32*64+7)*2+(64+7)*2
    out=4*(rows*64+7)*2+(rows*32+7)*2+2*(rows*64+7)*4+(rows*32+7)*4
    shared=inp+out;total=shared*3+mlp.rms.probe.ALLOWANCE
    return {'event':'idle_result','passed':True,'result':{'operation':'resident_tiny_mlp_half_boundaries','id':case['id'],'rows_count':rows,
        'hidden':32,'intermediate':64,'threadgroup_width':width,'capacity_rows64':rows,'capacity_rows32':rows,
        'input_stride':32,'gate_weight_stride':32,'down_weight_stride':64,'bias_stride':64,'half_intermediate_stride':64,
        'half_output_stride':32,'float_intermediate_stride':64,'float_output_stride':32,'shared_buffer_bytes':shared,
        'host_snapshot_readback_bytes':shared*2,'admitted_bytes':total,'allocations_reused':True,
        'first_half_intermediates':h,'first_fp32_projection_bits':floats,
        'rows':[{'phase':'warmup'if i<3 else'sample','repetition':i+1 if i<3 else i-2,'command_completed':True,'commit_wait_pairs':1,
                 'inputs_guards_and_intermediates_unchanged':True,'encode_commit_wait_seconds':.001,'output_bits':h[4].copy()}for i in range(13)],
        'metadata':{'mps_device_supported':True,'unified_memory':True,'language_version':'2.3','math_mode':'safe','math_functions':'precise',
                    'minimum_macos_major':15,'maximum_admitted_bytes':total,'execution_width':32,'pipeline_max_threads':1024,'device_max_threads':[1024]*3,
                    'graph_pipeline_caps':[{'name':name,'execution_width':32,'max_threads':1024}for name in ('toklet_projection_bias','toklet_silu','toklet_gating')]}}}


class Mlp(unittest.TestCase):
    def test_all_five_independent_stages_and_the_half_gating_boundary(self):
        inputs,anchors,overflow=mlp.cases();self.assertEqual(len(inputs),10)
        self.assertEqual(overflow['gate_weight_bits'][0],0x7bff)
        for case in inputs:
            anchor=anchors[case['id']];event=fixture(case,anchor);mlp.validate(event,case,anchor)
            self.assertEqual([q['stage']for q in event['result']['qualification']],list(mlp.HALF_STAGES))
            self.assertTrue(all(q['source_fp32_max_gate_fraction']<=1 for q in event['result']['qualification']))
        case=inputs[0];anchor=copy.deepcopy(anchors[case['id']]);event=fixture(case,anchor)
        anchor['stages'][1]['source'][0]+=10
        with self.assertRaisesRegex(ValueError,'Frozen composed MLP'):mlp.validate(event,case,anchor)

    def test_graph_waits_memory_and_stored_half_intermediates_cannot_be_weakened(self):
        cases,anchors,_=mlp.cases();case=cases[-1];anchor=anchors[case['id']]
        for kind in ('gating','cast','nan','guard','wait','output','shared','host','budget','capacity','contiguous','stride','pipeline'):
            event=fixture(case,anchor);r=event['result']
            if kind=='gating':r['first_half_intermediates'][3][0]^=1
            elif kind=='cast':r['first_fp32_projection_bits'][0][0]=0
            elif kind=='nan':r['first_fp32_projection_bits'][0][0]=0x7fc00000
            elif kind=='guard':r['rows'][-1]['inputs_guards_and_intermediates_unchanged']=False
            elif kind=='wait':r['rows'][-1]['commit_wait_pairs']=2
            elif kind=='output':r['rows'][-1]['output_bits'][0]^=1
            elif kind=='shared':r['shared_buffer_bytes']-=1
            elif kind=='host':r['host_snapshot_readback_bytes']-=1
            elif kind=='budget':r['admitted_bytes']-=1
            elif kind=='capacity':r['capacity_rows64']+=1
            elif kind=='contiguous':r['half_intermediate_stride']=65
            elif kind=='stride':r['input_stride']=31
            elif kind=='pipeline':r['metadata']['graph_pipeline_caps'][0]['name']='other'
            with self.subTest(kind=kind),self.assertRaises(ValueError):mlp.validate(event,case,anchor)

    def test_completed_overflow_discards_entire_graph(self):
        good={'passed':False,'error':{'stage':'validation','message':'completed MLP nonfinite; result discarded'}}
        self.assertIs(mlp.overflow_rejected(good),False)
        with self.assertRaises(ValueError):mlp.overflow_rejected({**good,'passed':True})
        with self.assertRaises(ValueError):mlp.overflow_rejected({**good,'error':{'stage':'admission','message':good['error']['message']}})

    def test_paths_are_disjoint_and_frozen_inputs_protected_before_read(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);output=root/'out.json';build=root/'build'
            with patch.object(mlp,'cases')as read:
                for inputs in (output,build,build/'inputs',root):
                    with self.assertRaises(ValueError):mlp.run(output,build,inputs)
                for source in (mlp.rms.FP16,mlp.rms.FP32):
                    with self.assertRaises(ValueError):mlp.run(source/'new.json',build,root/'inputs')
                read.assert_not_called()

if __name__=='__main__':unittest.main()
