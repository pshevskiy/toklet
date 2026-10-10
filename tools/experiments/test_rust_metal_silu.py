"""Offline frozen-source, stable-edge and native-evidence failure checks."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_silu as silu


def fixture(inputs,anchors):
    cases=[]
    for item in inputs:
        count=item["rows"]*item["width"];size=(count+7)*2
        cases.append({"id":item["id"],"rows_count":item["rows"],"width":item["width"],"guards":7,
            "output_guard_bits":silu.GUARD,"allocations_reused_across_repetitions":True,
            "threadgroup_width":32,"threadgroups":math.ceil(count/32),"shared_buffer_bytes":size*2,
            "host_input_and_readback_bytes":size*3,"admitted_bytes":size*5+silu.rms.probe.ALLOWANCE,
            "rows":[{"phase":"warmup" if i<3 else "sample","repetition":i+1 if i<3 else i-2,
                "command_completed":True,"immutable_input_and_guards":True,"encode_commit_wait_seconds":.001,
                "output_bits":anchors[item["id"]]["bits"]+[silu.GUARD]*7} for i in range(13)]})
    return {"event":"idle_result","passed":True,"result":{"operation":"silu_fp16_storage_stable_fp32","cases":cases,
        "metadata":{"unified_memory":True,"language_version":"2.3","math_mode":"safe","math_functions":"precise",
            "minimum_macos_major":15,"maximum_admitted_bytes":max(case["admitted_bytes"] for case in cases),
            "execution_width":32,"pipeline_max_threads":1024,"device_max_threads":[1024,1024,1024]}}}


class MetalSilu(unittest.TestCase):
    def test_ten_frozen_stages_and_preselected_manual_edges_keep_source_gates(self):
        inputs,anchors=silu.cases()
        self.assertEqual(len(inputs),11);self.assertEqual(sum(item["rows"]*item["width"] for item in inputs),6720)
        self.assertEqual(anchors[inputs[-1]["id"]]["bits"][:8],[0,0x8000,0x7bff,0x8000,2,0x8002,0x4d00,0x8001])
        event=fixture(inputs,anchors);silu.validate(event,inputs,anchors)
        for case in event["result"]["cases"]:
            for row in case["rows"]:
                self.assertNotIn("output_bits",row)
                self.assertTrue(row["qualification"]["exact_independent_fp16_bits"])
                self.assertLessEqual(row["qualification"]["against_source_fp32_max_gate_fraction"],1)

    def test_manual_bits_guards_completion_and_ownership_fail_closed(self):
        inputs,anchors=silu.cases();inputs=inputs[-1:]
        for kind in ("sign","subnormal","negative_branch","nonfinite","guard","integer","immutable","completed","memory","launch","repetition","math"):
            event=fixture(inputs,anchors);case=event["result"]["cases"][0];row=case["rows"][-1]
            if kind=="sign":row["output_bits"][1]=0
            elif kind=="subnormal":row["output_bits"][4]=0
            elif kind=="negative_branch":row["output_bits"][7]=0x8000
            elif kind=="nonfinite":row["output_bits"][0]=0x7c00
            elif kind=="guard":row["output_bits"][-1]^=1
            elif kind=="integer":row["output_bits"][0]=0.0
            elif kind=="immutable":row["immutable_input_and_guards"]=False
            elif kind=="completed":row["command_completed"]=False
            elif kind=="memory":case["admitted_bytes"]-=1
            elif kind=="launch":case["threadgroups"]+=1
            elif kind=="repetition":row["repetition"]=9
            elif kind=="math":event["result"]["metadata"]["math_mode"]="fast"
            with self.subTest(kind=kind),self.assertRaises(ValueError):silu.validate(event,inputs,anchors)

    def test_frozen_source_gate_is_required_even_when_half_bits_match(self):
        inputs,anchors=silu.cases();event=fixture(inputs,anchors)
        anchors[inputs[0]["id"]]["reference"][0]+=10
        with self.assertRaisesRegex(ValueError,"FP32 source gate"):silu.validate(event,inputs,anchors)

    def test_frozen_payload_and_manifest_corruption_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"manifest.json").write_text('{}')
            with self.assertRaisesRegex(ValueError,"manifest identity"):silu.rms.manifest(root,silu.rms.FP16_MANIFEST_SHA)
            (root/"x.bin").write_bytes(b'\0\0')
            record={"arrays":{"x":{"dtype":"f16le","shape":[1],"bytes":2,"file":"x.bin","sha256":"0"*64}}}
            with self.assertRaisesRegex(ValueError,"payload identity"):silu.rms.payload(root,record,"x","f16le")

    def test_evidence_aliases_and_existing_inputs_fail_before_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/"result.json";build=root/"build"
            with patch.object(silu,"cases") as read:
                for inputs in (output,build,build/"inputs",root):
                    with self.subTest(inputs=inputs),self.assertRaises(ValueError):silu.run(output,build,inputs)
                inputs=root/"inputs";inputs.mkdir()
                with self.assertRaisesRegex(ValueError,"overwrite"):silu.run(output,build,inputs)
                read.assert_not_called()


if __name__=="__main__":unittest.main()
