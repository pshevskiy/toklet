"""Offline Qwen SiLU row coverage, source gates and checkpoint-free checks."""
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_qwen_silu as qs
from test_rust_metal_silu import fixture


def synthetic(root):
    hroot,sroot=root/"half",root/"single";hroot.mkdir();sroot.mkdir()
    half,single={"arrays":{}},{"arrays":{}}
    def save(directory,manifest,key,rows,values,dtype):
        data=struct.pack(f"<{len(values)}"+("e" if dtype=="f16le" else "f"),*values)
        path=directory/f"{key}.{dtype}";path.write_bytes(data)
        item={"file":path.name,"shape":[rows,qs.WIDTH],"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
        if dtype=="f16le":item["dtype"]=dtype
        manifest["arrays"][key]=item;return item
    for name,rows in qs.qwen.CASE_ROWS.items():
        for layer in qs.LAYERS:
            prefix=f"cases.{name}.layers.{layer}";values=[];outputs=[]
            for row in range(rows):
                value=(0.0,1.0,-1.0,20.0)[(row+layer)%4]
                e=math.exp(-abs(value));answer=value*(1/(1+e) if value>=0 else e/(1+e))
                rounded=struct.unpack("<e",struct.pack("<e",answer))[0]
                values.extend([value]*qs.WIDTH);outputs.extend([rounded]*qs.WIDTH)
            save(hroot,half,f"{prefix}.gate_projection",rows,values,"f16le")
            key=f"{prefix}.gate_silu";oracle=save(hroot,half,key,rows,outputs,"f16le")
            reference=save(sroot,single,key,rows,[value+.0001 for value in outputs],"f32le")
            oracle.update(source_fp32_full_sha256=reference["sha256"],gate_against_source_fp32={"atol":.001,"rtol":.001})
    return half,single,hroot,sroot


class QwenMetalSilu(unittest.TestCase):
    def test_all_selected_rows_cover_complete_arrays_and_original_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            source=synthetic(Path(directory));inputs,anchors=qs.assemble(*source);half,single,hroot,sroot=source
            self.assertEqual(len(inputs),36);self.assertEqual(sum(item["rows"]*item["width"] for item in inputs),175104)
            stages={anchor["source_stage"] for anchor in anchors.values()};self.assertEqual(len(stages),12)
            for stage in stages:
                group=[item for item in inputs if anchors[item["id"]]["source_stage"]==stage]
                self.assertEqual([anchors[item["id"]]["row_index"] for item in group],list(range(len(group))))
                input_data=b''.join(struct.pack(f'<{qs.WIDTH}H',*item['input_bits']) for item in group)
                oracle_data=b''.join(struct.pack(f'<{qs.WIDTH}H',*anchors[item['id']]['bits']) for item in group)
                source_data=b''.join(struct.pack(f'<{qs.WIDTH}f',*anchors[item['id']]['reference']) for item in group)
                projection=stage.replace('.gate_silu','.gate_projection')
                self.assertEqual(input_data,(hroot/half['arrays'][projection]['file']).read_bytes())
                self.assertEqual(oracle_data,(hroot/half['arrays'][stage]['file']).read_bytes())
                self.assertEqual(source_data,(sroot/single['arrays'][stage]['file']).read_bytes())
                for item in group:
                    self.assertLessEqual(item['rows']*item['width'],8192)
                    self.assertEqual(anchors[item['id']]['gate'],half['arrays'][stage]['gate_against_source_fp32'])
                    event=fixture([item],anchors);qs.silu.validate(event,[item],anchors)
                    self.assertTrue(all(row['qualification']['exact_independent_fp16_bits'] for row in event['result']['cases'][0]['rows']))

    def test_exact_half_output_cannot_bypass_frozen_source_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs,anchors=qs.assemble(*synthetic(Path(directory)));group=inputs[:1]
            anchors[group[0]['id']]['reference'][0]+=10
            with self.assertRaisesRegex(ValueError,'FP32 source gate'):qs.silu.validate(fixture(group,anchors),group,anchors)

    def test_full_payload_shape_dtype_anchor_and_nonfinite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source=synthetic(Path(directory));half,single,hroot,sroot=source
            key='cases.single.layers.0.gate_silu';item=half['arrays'][key]
            for field,bad in [('shape',[qs.WIDTH]),('dtype','f32le'),('bytes',1),('source_fp32_full_sha256','0'*64)]:
                original=item[field];item[field]=bad
                with self.subTest(field=field),self.assertRaises(ValueError):qs.assemble(*source)
                item[field]=original
            path=hroot/item['file'];data=path.read_bytes();path.write_bytes(b'\x00\x7c'+data[2:])
            with self.assertRaisesRegex(ValueError,'payload identity'):qs.assemble(*source)
            item['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'nonfinite'):qs.assemble(*source)

    def test_payload_reads_are_bounded_and_cannot_escape_reference_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'x';path.write_bytes(b'\0\0'*qs.WIDTH)
            item={'shape':[1,qs.WIDTH],'dtype':'f16le','bytes':qs.WIDTH*2,'file':'x','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(len(qs.payload(root,item,'f16le',1)),qs.WIDTH*2)
            with self.assertRaisesRegex(ValueError,'shape'):qs.payload(root,item,'f16le',5)
            path.write_bytes(path.read_bytes()+b'\0')
            with self.assertRaisesRegex(ValueError,'identity'):qs.payload(root,item,'f16le',1)
            item['file']='../x'
            with self.assertRaisesRegex(ValueError,'escapes'):qs.payload(root,item,'f16le',1)

    def test_driver_keeps_tiny_regression_and_rechecks_inputs_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);half,single,hroot,sroot=synthetic(root);inputs,anchors=qs.assemble(half,single,hroot,sroot)
            tiny=[{'id':'tiny','rows':1,'width':64,'input_bits':[0]*64}]
            tiny_anchors={'tiny':{'bits':[0]*64,'reference':[0.0]*64,'gate':None}}
            contexts=[]
            def native(output,build,extra_runs,extra_context):
                contexts.append(extra_context);self.assertEqual(len(extra_runs),37)
                for label,options,validator in extra_runs:
                    self.assertEqual(options[0],'--silu-input');group=json.loads(Path(options[1]).read_text())['cases']
                    expected=tiny_anchors if label=='tiny_silu_regression' else anchors
                    validator(fixture(group,expected))
            with patch.object(qs.qwen,'HALF',hroot),patch.object(qs.qwen,'SINGLE',sroot),\
                    patch.object(qs,'checked_references',return_value=(half,single)) as read,\
                    patch.object(qs.silu,'cases',return_value=(tiny,tiny_anchors)),\
                    patch.object(qs.qwen,'norm_weights',side_effect=AssertionError('checkpoint read')) as weights,\
                    patch.object(qs.qwen,'stream_digest',side_effect=AssertionError('checkpoint hash')) as stream,\
                    patch.object(qs.qwen.rms.probe,'run',side_effect=native):
                qs.run(root/'result.json',root/'build',root/'inputs')
                self.assertEqual(read.call_count,2);weights.assert_not_called();stream.assert_not_called()
            self.assertFalse(contexts[0]['checkpoint_reads'])
            self.assertEqual(len(contexts[0]['input_sha256']),37)
            self.assertEqual(len(contexts[0]['qualification_driver_sha256']),6)

    def test_path_aliases_fail_before_loading_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'result.json';build=root/'build'
            with patch.object(qs,'checked_references') as read:
                for inputs in (output,build,build/'inputs',root):
                    with self.subTest(inputs=inputs),self.assertRaises(ValueError):qs.run(output,build,inputs)
                inputs=root/'inputs';inputs.mkdir()
                with self.assertRaisesRegex(ValueError,'overwrite'):qs.run(output,build,inputs)
                with self.assertRaisesRegex(ValueError,'source fixtures'):qs.run(qs.qwen.HALF/'new.json',build,root/'new')
                read.assert_not_called()


if __name__=='__main__':unittest.main()
