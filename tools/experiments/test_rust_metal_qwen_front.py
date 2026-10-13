"""Propagated original gates, combined admission and completion policy."""
import copy
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import rust_metal_qwen_front as front


def fixture(item,anchor):
    n=item['rows'];width=32
    def l(rows,cols,size,capacity):return {'rows':rows,'columns':cols,'stride':cols,'elements':capacity*cols+7,'bytes':(capacity*cols+7)*size}
    il=[l(*x)for x in[(n,896,2,n),(1,896,2,1),(896,896,2,896),(128,896,2,128),(128,896,2,128),(1,896,2,1),(1,128,2,1),(1,128,2,1)]]
    hl=[l(n,896,2,math.ceil(n/width)*width)]+[l(n,c,2,n)for c in(896,128,128)];fl=[l(n,c,4,n)for c in(896,128,128)];shared=sum(x['bytes']for group in(il,hl,fl)for x in group)
    halves=[a['bits'].copy()for a in anchor['stages']];floats=[list(struct.unpack(f'<{len(a)}I',struct.pack(f'<{len(a)}f',*front.qp.decoder.unpack(a,'H'))))for a in halves[1:]]
    return {'event':'idle_result','passed':True,'result':{'operation':'resident_qwen_layer0_input_norm_qkv','id':item['id'],'rows_count':n,'threadgroup_width':width,'metadata':{'unified_memory':True,'mps_device_supported':True,'math_mode':'safe','math_functions':'precise','language_version':'2.3','minimum_macos_major':15,'execution_width':32,'pipeline_max_threads':1024,'device_max_threads':[1024]*3,'graph_pipeline_caps':[{'name':name,'execution_width':32,'max_threads':1024}for name in('toklet_projection_bias','toklet_rotary','toklet_cache_write','toklet_attention','toklet_rms_norm','toklet_residual','toklet_silu','toklet_gating')],'maximum_admitted_bytes':3*shared+front.probe.ALLOWANCE},'input_layouts':il,'half_layouts':hl,'float_layouts':fl,'shared_buffer_bytes':shared,'host_snapshot_readback_bytes':2*shared,'admitted_bytes':3*shared+front.probe.ALLOWANCE,'allocations_reused':True,'first_half_intermediates':halves,'first_fp32_intermediates':floats,'rows':[{'phase':'warmup'if i<3 else'sample','repetition':i+1 if i<3 else i-2,'command_completed':True,'commit_wait_pairs':1,'inputs_guards_and_intermediates_unchanged':True,'encode_commit_wait_seconds':.001,'output_bits':copy.deepcopy(halves[1:])}for i in range(13)]}}


class Front(unittest.TestCase):
    def test_all_four_original_gates_on_four_prefixes(self):
        inputs,anchors=front.cases();self.assertEqual(len(inputs),4)
        for item in inputs:front.validate(fixture(item,anchors[item['id']]),item,anchors[item['id']])
        item=inputs[0];a=copy.deepcopy(anchors[item['id']]);a['stages'][0]['source'][0]+=100
        with self.assertRaisesRegex(ValueError,'source gate'):front.validate(fixture(item,anchors[item['id']]),item,a)

    def test_storage_half_boundary_and_one_wait_fail_closed(self):
        inputs,anchors=front.cases();item=inputs[-1];a=anchors[item['id']]
        for kind in('layout','budget','cast','type','repeat','wait','norm_count'):
            e=fixture(item,a);r=e['result']
            if kind=='layout':r['half_layouts'][0]['elements']-=1
            elif kind=='budget':r['admitted_bytes']-=1
            elif kind=='cast':r['first_fp32_intermediates'][0][0]=1.0.hex()
            elif kind=='type':r['first_half_intermediates'][0][0]=True
            elif kind=='repeat':r['rows'][0]['output_bits'][0][0]^=1
            elif kind=='wait':r['rows'][0]['commit_wait_pairs']=2
            elif kind=='norm_count':r['first_half_intermediates'][0].pop()
            with self.subTest(kind=kind),self.assertRaises(ValueError):front.validate(e,item,a)

    def test_cli_main_name_still_binds_its_own_driver(self):
        with patch.dict(front.sys.modules,{'__main__':front}):
            front.sys.modules.pop('rust_metal_qwen_front',None)
            identities=front.driver_identities()
        path=Path(front.__file__).resolve()
        self.assertEqual(identities[str(path)],front.probe.common.digest(path))

    def test_path_aliases_rejected_before_reads(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)
            with patch.object(front,'cases')as read:
                for directory in(root,root/'out.json',root/'build',root/'build/inputs'):
                    with self.assertRaises(ValueError):front.run(root/'out.json',root/'build',directory)
                read.assert_not_called()

if __name__=='__main__':unittest.main()
