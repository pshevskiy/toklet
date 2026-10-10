"""Synthetic Qwen residual pairings, anchors and checkpoint-free evidence."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_qwen_residual as qres
from test_rust_metal_residual import fixture


def synthetic(root):
    hroot,sroot=root/"half",root/"single";hroot.mkdir();sroot.mkdir()
    half,single={"arrays":{}},{"arrays":{}}
    values={"embedding":1,"layers.0.attention_projection":2,"layers.0.attention_residual":3,
        "layers.0.down_projection":4,"layers.0.output":7,"layers.1.attention_projection":5,
        "layers.1.attention_residual":12,"layers.1.down_projection":6,"layers.1.output":18,
        "layers.23.attention_residual":7,"layers.23.down_projection":8,"layers.23.output":15}
    def save(directory,manifest,key,rows,value,dtype):
        count=rows*896;data=struct.pack(f"<{count}"+("e" if dtype=="f16le" else "f"),*([value]*count))
        path=directory/f"{key}.{dtype}";path.write_bytes(data)
        item={"file":path.name,"shape":[rows,896],"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
        if dtype=="f16le":item["dtype"]=dtype
        manifest["arrays"][key]=item;return item
    for name,rows in qres.qwen.CASE_ROWS.items():
        for tensor,value in values.items():save(hroot,half,f"cases.{name}.{tensor}",rows,value,"f16le")
        for operation,_,_ in qres.OPERATIONS:
            key=f"cases.{name}.{operation}";ref=save(sroot,single,key,rows,values[operation]+.0001,"f32le")
            half["arrays"][key].update(source_fp32_full_sha256=ref["sha256"],gate_against_source_fp32={"atol":.001,"rtol":.001})
    return half,single,hroot,sroot


class QwenMetalResidual(unittest.TestCase):
    def test_selected_saved_pairs_batch_without_loss_and_keep_exact_source_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            source=synthetic(Path(directory));inputs,anchors=qres.assemble(*source)
            self.assertEqual(len(inputs),20);self.assertEqual(sum(item["rows"]*896 for item in inputs),53760)
            self.assertFalse(any(item["id"].endswith("layers.23.attention_residual") for item in inputs))
            self.assertEqual([item["id"] for item in inputs[:5]],[f"cases.future_changed.{op}" for op,_,_ in qres.OPERATIONS])
            groups=qres.qwen.batches(inputs);self.assertEqual(len(groups),8)
            self.assertEqual([item for group in groups for item in group],inputs)
            for group in groups:
                self.assertLessEqual(sum(item["rows"]*896 for item in group),8192)
                native=fixture(group,anchors);qres.residual.validate(native,group,anchors)
                for case in native["result"]["cases"]:
                    self.assertEqual(case["anchors"]["gate"],source[0]["arrays"][case["id"]]["gate_against_source_fp32"])
                    self.assertTrue(all(row["qualification"]["exact_independent_fp16_bits"] for row in case["rows"]))

    def test_exact_native_half_bits_do_not_replace_original_source_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs,anchors=qres.assemble(*synthetic(Path(directory)));group=qres.qwen.batches(inputs)[0]
            anchors[group[0]["id"]]["reference"][0]+=10
            with self.assertRaisesRegex(ValueError,"FP32 source gate"):qres.residual.validate(fixture(group,anchors),group,anchors)

    def test_source_anchor_payload_shape_and_independent_sum_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            source=synthetic(Path(directory));half,single,hroot,sroot=source;key="cases.single.layers.1.output"
            item=half["arrays"][key];original=item["source_fp32_full_sha256"]
            item["source_fp32_full_sha256"]="0"*64
            with self.assertRaisesRegex(ValueError,"source shape"):qres.assemble(*source)
            item["source_fp32_full_sha256"]=original
            item["shape"]=[896]
            with self.assertRaisesRegex(ValueError,"source shape"):qres.assemble(*source)
            item["shape"]=[1,896]
            path=hroot/item["file"];data=bytes([1])+path.read_bytes()[1:];path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,"identity"):qres.assemble(*source)
            item["sha256"]=hashlib.sha256(data).hexdigest()
            with self.assertRaisesRegex(ValueError,"independent Half sum"):qres.assemble(*source)

    def test_checkpoint_free_driver_revalidates_all_inputs_and_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);half,single,hroot,sroot=synthetic(root);inputs,anchors=qres.assemble(half,single,hroot,sroot)
            contexts=[]
            def native(output,build,extra_runs,extra_context):
                contexts.append(extra_context);self.assertEqual(len(extra_runs),8)
                for _,options,validator in extra_runs:
                    self.assertEqual(options[0],"--residual-input")
                    group=json.loads(Path(options[1]).read_text())["cases"]
                    validator(fixture(group,anchors))
            with patch.object(qres.qwen,"HALF",hroot),patch.object(qres.qwen,"SINGLE",sroot),\
                    patch.object(qres.qwen,"checked_manifests",return_value=(half,single,1e-6)) as read,\
                    patch.object(qres.qwen,"norm_weights",side_effect=AssertionError("checkpoint read")) as weights,\
                    patch.object(qres.qwen,"stream_digest",side_effect=AssertionError("checkpoint hash")) as stream,\
                    patch.object(qres.qwen.rms.probe,"run",side_effect=native):
                qres.run(root/"report.json",root/"build",root/"inputs")
                self.assertEqual(read.call_count,2);weights.assert_not_called();stream.assert_not_called()
            self.assertFalse(contexts[0]["checkpoint_reads"])
            self.assertEqual(len(contexts[0]["input_sha256"]),8)
            self.assertEqual(len(contexts[0]["qualification_driver_sha256"]),4)

    def test_path_aliases_fail_before_any_fixture_or_model_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/"report.json";build=root/"build"
            with patch.object(qres.qwen,"checked_manifests") as read:
                for inputs in (output,build,build/"inputs",root):
                    with self.subTest(inputs=inputs),self.assertRaises(ValueError):qres.run(output,build,inputs)
                inputs=root/"inputs";inputs.mkdir()
                with self.assertRaisesRegex(ValueError,"overwrite"):qres.run(output,build,inputs)
                with self.assertRaisesRegex(ValueError,"source fixtures"):qres.run(qres.qwen.HALF/"new.json",build,root/"new")
                read.assert_not_called()


if __name__=="__main__":unittest.main()
