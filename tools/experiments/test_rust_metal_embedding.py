"""Offline exact embedding oracles, evidence and corruption checks."""
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_embedding as embedding


def fixture(source,anchors):
    h,v=source["hidden"],source["vocab"];table=(h*v+7)*2;cases=[]
    for item in source["cases"]:
        rows=len(item["tokens"]);count=rows*h;ids=(rows+7)*4;output=(count+7)*2
        cases.append({"id":item["id"],"rows_count":rows,"hidden":h,"vocab":v,
            "guards":7,"output_guard_bits":embedding.GUARD,"threadgroup_width":32,"threadgroups":math.ceil(count/32),
            "shared_buffer_bytes":table+ids+output,"host_upload_and_readback_bytes":2*table+2*ids+output,
            "admitted_bytes":3*table+3*ids+2*output+embedding.rms.probe.ALLOWANCE,"allocations_reused_across_repetitions":True,
            "rows":[{"phase":"warmup" if i<3 else "sample","repetition":i+1 if i<3 else i-2,
                "command_completed":True,"immutable_table_tokens_and_guards":True,"encode_commit_wait_seconds":.001,
                "output_bits":anchors[item["id"]]["bits"]+[embedding.GUARD]*7} for i in range(13)]})
    return {"event":"idle_result","passed":True,"result":{"operation":"embedding_fp16_bit_copy",
        "table_allocation_reused_across_cases":True,"cases":cases,"metadata":{"unified_memory":True,
        "language_version":"2.3","math_mode":"safe","math_functions":"precise","minimum_macos_major":15,
        "maximum_admitted_bytes":max(case["admitted_bytes"] for case in cases),
        "execution_width":32,"pipeline_max_threads":1024,"device_max_threads":[1024,1024,1024]}}}


class MetalEmbedding(unittest.TestCase):
    def test_five_frozen_cases_and_manual_bits_keep_original_gates(self):
        sources,anchors=embedding.cases()
        self.assertEqual(len(sources[0]["cases"]),5)
        self.assertEqual(sum(len(item["tokens"])*32 for item in sources[0]["cases"]),1408)
        self.assertEqual(len(sources[1]["cases"][0]["tokens"]),17)
        self.assertTrue({0,0x8000,1,0x8001,0x7bff,0xfbff}.issubset(anchors["manual.bit_copy"]["bits"]))
        for source in sources:
            native=fixture(source,anchors)
            embedding.validate(native,source,anchors)
            for case in native["result"]["cases"]:
                for row in case["rows"]:
                    self.assertNotIn("output_bits",row)
                    self.assertTrue(row["qualification"]["exact_independent_fp16_bits"])
                    self.assertLessEqual(row["qualification"]["against_source_fp32_max_gate_fraction"],1)
                    bits=anchors[case["id"]]["bits"]
                    self.assertEqual(row["qualification"]["output_sha256"],hashlib.sha256(struct.pack(f"<{len(bits)}H",*bits)).hexdigest())

    def test_manual_signedzero_subnormal_lastrow_and_guard_corruption_fail(self):
        sources,anchors=embedding.cases();source=sources[1]
        for kind in ("zero","subnormal","lastrow","guard","integer","immutable","completion","table_reuse","memory","launch"):
            native=fixture(source,anchors);case=native["result"]["cases"][0];row=case["rows"][-1]
            if kind=="zero":row["output_bits"][33]=0
            elif kind=="subnormal":row["output_bits"][34]=0
            elif kind=="lastrow":row["output_bits"][0]^=1
            elif kind=="guard":row["output_bits"][-1]^=1
            elif kind=="integer":row["output_bits"][0]=float(row["output_bits"][0])
            elif kind=="immutable":row["immutable_table_tokens_and_guards"]=False
            elif kind=="completion":row["command_completed"]=False
            elif kind=="table_reuse":native["result"]["table_allocation_reused_across_cases"]=False
            elif kind=="memory":case["admitted_bytes"]-=1
            elif kind=="launch":case["threadgroups"]+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):embedding.validate(native,source,anchors)

    def test_source_fp32_gate_is_required_after_exact_bits_pass(self):
        sources,anchors=embedding.cases();source=sources[0]
        native=fixture(source,anchors)
        anchors[source["cases"][0]["id"]]["reference"][0]+=10
        with self.assertRaisesRegex(ValueError,"FP32 source gate"):
            embedding.validate(native,source,anchors)

    def test_frozen_table_to_embedding_identity_is_checked_before_native_work(self):
        payload=embedding.rms.payload
        def changed(root,manifest,name,dtype):
            data,item=payload(root,manifest,name,dtype)
            if name=="weights.embed_tokens":data=bytes([data[0]^1])+data[1:]
            return data,item
        with patch.object(embedding.rms,"payload",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"table gather"):embedding.cases()

    def test_evidence_aliases_and_existing_inputs_fail_before_oracle_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/"result.json";build=root/"build"
            with patch.object(embedding,"cases") as read:
                for inputs in (output,build,build/"inputs",root):
                    with self.subTest(inputs=inputs),self.assertRaises(ValueError):embedding.run(output,build,inputs)
                inputs=root/"inputs";inputs.mkdir()
                with self.assertRaisesRegex(ValueError,"overwrite"):embedding.run(output,build,inputs)
                read.assert_not_called()


if __name__=="__main__":unittest.main()
