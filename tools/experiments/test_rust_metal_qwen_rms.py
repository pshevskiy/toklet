"""Offline archive, batch and gate checks; no model downloads or GPU required."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_qwen_rms as qwen
from test_rust_metal_rms import fixture


def archive(root,values,metadata=None):
    data=struct.pack(f"<{len(values)}H",*values)
    if metadata is None:
        metadata={"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},
                  "b":{"dtype":"BF16","shape":[4],"data_offsets":[8,16]}}
    header=json.dumps(metadata,separators=(",",":")).encode()
    payload=struct.pack("<Q",len(header))+header+data
    path=root/"model.safetensors"
    path.write_bytes(payload)
    return path,hashlib.sha256(payload).hexdigest(),len(payload)


def synthetic(root):
    half_root,single_root=root/"half",root/"single"
    half_root.mkdir();single_root.mkdir()
    half,single={"arrays":{}},{"arrays":{}}
    weights={name:{"bits":[0x3c00]*896,"fp16_sha256":"synthetic-half",
                   "source_bf16_sha256":"synthetic-bf16"} for _,_,name in qwen.OPERATIONS}
    def save(directory,manifest,key,dtype,rows):
        data=bytes(rows*896*(2 if dtype=="f16le" else 4))
        name=f"{key}.{dtype}"
        (directory/name).write_bytes(data)
        item={"file":name,"dtype":dtype,"shape":[rows,896],"bytes":len(data),
              "sha256":hashlib.sha256(data).hexdigest()}
        manifest["arrays"][key]=item
        return item
    for case,rows in qwen.CASE_ROWS.items():
        for operation,input_name,_ in qwen.OPERATIONS:
            save(half_root,half,f"cases.{case}.{input_name}","f16le",rows)
            key=f"cases.{case}.{operation}"
            source=save(single_root,single,key,"f32le",rows)
            # Exercise original full source's global FP32 dtype convention.
            del source["dtype"]
            out=save(half_root,half,key,"f16le",rows)
            out.update(source_fp32_full_sha256=source["sha256"],
                       gate_against_source_fp32={"atol":1e-4,"rtol":1e-4})
    return half,single,half_root,single_root,weights


class QwenMetalRms(unittest.TestCase):
    def test_bf16_conversion_preserves_signed_zero_and_half_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            path,sha,size=archive(Path(directory),[0,0x8000,0x3f80,0xc000,0x3300,0x3380,0x33c0,0x477f])
            weights=qwen.norm_weights(path,sha,size,["a","b"],hidden=4)
            self.assertEqual(weights["a"]["bits"],[0,0x8000,0x3c00,0xc000])
            self.assertEqual(weights["b"]["bits"],[0,1,2,0x7bf8])
            for item in weights.values():
                self.assertEqual(item["fp16_sha256"],hashlib.sha256(struct.pack("<4H",*item["bits"])).hexdigest())

    def test_archive_identity_nonfinite_and_half_overflow_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path,sha,size=archive(root,[0x3f80]*8)
            with self.assertRaisesRegex(ValueError,"identity"):
                qwen.norm_weights(path,"0"*64,size,["a","b"],4)
            for bits in (0x7fc0,0x7f80,0x4780):
                path,sha,size=archive(root,[bits]+[0x3f80]*7)
                with self.subTest(bits=bits),self.assertRaisesRegex(ValueError,"Nonfinite|Half range"):
                    qwen.norm_weights(path,sha,size,["a","b"],4)

    def test_header_dtype_shape_offsets_overlap_and_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for kind in ("dtype","shape","offset","overlap","boolean","alignment"):
                metadata={"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},
                          "b":{"dtype":"BF16","shape":[4],"data_offsets":[8,16]}}
                if kind=="dtype":metadata["a"]["dtype"]="F16"
                elif kind=="shape":metadata["a"]["shape"]=[2,2]
                elif kind=="offset":metadata["b"]["data_offsets"]=[16,24]
                elif kind=="overlap":metadata["b"]["data_offsets"]=[0,8]
                elif kind=="boolean":metadata["a"]["data_offsets"]=[False,8]
                elif kind=="alignment":metadata["a"]["data_offsets"]=[1,9]
                path,sha,size=archive(root,[0x3f80]*8,metadata)
                with self.subTest(kind=kind),self.assertRaises(ValueError):qwen.norm_weights(path,sha,size,["a","b"],4)
            for header in (b'{"a":{},"a":{}}',b"[]"):
                data=struct.pack("<Q",len(header))+header
                path=root/"duplicate.safetensors";path.write_bytes(data)
                with self.assertRaises(ValueError):qwen.norm_weights(path,hashlib.sha256(data).hexdigest(),len(data),["a"],4)
            data=struct.pack("<Q",1024*1024+1)
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,"header exceeds"):
                qwen.norm_weights(path,hashlib.sha256(data).hexdigest(),len(data),["a"],4)

    def test_all_selected_stages_batch_without_loss_and_keep_source_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            source=synthetic(Path(directory))
            inputs,anchors=qwen.assemble(*source,1e-6)
            self.assertEqual(len(inputs),24)
            self.assertEqual(sum(item["rows"]*item["hidden"] for item in inputs),64512)
            self.assertFalse(any("layers.23.input_norm" in item["id"] for item in inputs))
            groups=qwen.batches(inputs)
            self.assertEqual(len(groups),9)
            self.assertEqual([item for group in groups for item in group],inputs)
            for group in groups:
                self.assertLessEqual(sum(item["rows"]*item["hidden"] for item in group),8192)
                native=fixture(group,anchors)
                qwen.rms.validate(native,group,anchors)
                self.assertTrue(all(row["qualification"]["exact_independent_fp16_bits"] for case in native["result"]["cases"] for row in case["rows"]))
                for item in group:
                    self.assertEqual(anchors[item["id"]]["gate"],source[0]["arrays"][item["id"]]["gate_against_source_fp32"])
            with self.assertRaises(ValueError):qwen.batches([{"rows":10,"hidden":896}])

    def test_activation_identity_shape_and_source_anchor_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source=synthetic(Path(directory))
            half,single,hroot,sroot,weights=source
            key="cases.single.final_norm"
            half["arrays"][key]["source_fp32_full_sha256"]="0"*64
            with self.assertRaisesRegex(ValueError,"activation shape"):qwen.assemble(*source,1e-6)
            half["arrays"][key]["source_fp32_full_sha256"]=single["arrays"][key]["sha256"]
            path=hroot/half["arrays"][key]["file"]
            path.write_bytes(bytes([1])+path.read_bytes()[1:])
            with self.assertRaisesRegex(ValueError,"identity"):qwen.assemble(*source,1e-6)

    def test_output_overlap_and_existing_inputs_fail_before_source_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output,build,inputs=root/"result.json",root/"build",root/"inputs"
            with patch.object(qwen,"checked_manifests") as read:
                for candidate in (output,build,build/"inputs",root):
                    with self.subTest(candidate=candidate),self.assertRaises(ValueError):qwen.run(output,build,candidate)
                inputs.mkdir()
                with self.assertRaisesRegex(ValueError,"overwrite"):qwen.run(output,build,inputs)
                with self.assertRaisesRegex(ValueError,"source fixtures"):
                    qwen.check_paths(qwen.HALF/"new.json",build,root/"new")
                read.assert_not_called()


if __name__=="__main__":unittest.main()
