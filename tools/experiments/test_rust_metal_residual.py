"""Offline residual rounding, immutable-source and failure-policy checks."""
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_residual as residual


def fixture(inputs,anchors):
    cases=[]
    for item in inputs:
        count=item["rows"]*item["hidden"];size=(count+7)*2
        cases.append({"id":item["id"],"rows_count":item["rows"],"hidden":item["hidden"],"guards":7,
            "output_guard_bits":residual.GUARD,"in_place_state":True,"baseline_restored_before_each_repetition":True,
            "allocations_reused_across_repetitions":True,"threadgroup_width":32,"threadgroups":math.ceil(count/32),
            "shared_buffer_bytes":size*2,"host_baseline_and_readback_bytes":size*4,"admitted_bytes":size*6+residual.rms.probe.ALLOWANCE,
            "rows":[{"phase":"warmup" if i<3 else "sample","repetition":i+1 if i<3 else i-2,"command_completed":True,
                "immutable_delta_and_state_guards":True,"encode_commit_wait_seconds":.001,
                "output_bits":anchors[item["id"]]["bits"]+[residual.GUARD]*7} for i in range(13)]})
    return {"event":"idle_result","passed":True,"result":{"operation":"residual_in_place_fp16_storage_fp32_add","cases":cases,
        "metadata":{"unified_memory":True,"language_version":"2.3","math_mode":"safe","math_functions":"precise","minimum_macos_major":15,
            "maximum_admitted_bytes":max(case["admitted_bytes"] for case in cases),"execution_width":32,"pipeline_max_threads":1024,"device_max_threads":[1024,1024,1024]}}}


class MetalResidual(unittest.TestCase):
    def test_twenty_frozen_stages_and_manual_rne_keep_bits_and_source_gates(self):
        inputs,anchors,overflow=residual.cases()
        self.assertEqual(len(inputs),21);self.assertEqual(sum(item["rows"]*item["hidden"] for item in inputs),6176)
        manual=inputs[-1];bits=anchors[manual["id"]]["bits"]
        self.assertEqual(bits[:16],[0,0x8000,0,0x3c00,0x3c02,0xbc00,0xbc02,2,0x8002,0,0,0x7bff,0x7bff,0xfbff,0,0x03ff])
        self.assertEqual(overflow["state_bits"],[0x7bff]*32)
        with self.assertRaises(OverflowError):residual.half_sum(0x7bff,0x7bff)
        native=fixture(inputs,anchors);residual.validate(native,inputs,anchors)
        for case in native["result"]["cases"]:
            bits=anchors[case["id"]]["bits"]
            for row in case["rows"]:
                self.assertNotIn("output_bits",row)
                q=row["qualification"];self.assertTrue(q["exact_independent_fp16_bits"])
                self.assertLessEqual(q["against_source_fp32_max_gate_fraction"],1)
                self.assertEqual(q["output_sha256"],hashlib.sha256(struct.pack(f"<{len(bits)}H",*bits)).hexdigest())

    def test_in_place_repetition_guards_completion_and_rounding_corruption_fail(self):
        inputs,anchors,_=residual.cases();inputs=inputs[-1:]
        for kind in ("negative_zero","tie","subnormal","guard","integer","immutable","completed","reset","in_place","memory","launch","repetition"):
            native=fixture(inputs,anchors);case=native["result"]["cases"][0];row=case["rows"][-1]
            if kind=="negative_zero":row["output_bits"][1]=0
            elif kind=="tie":row["output_bits"][3]+=1
            elif kind=="subnormal":row["output_bits"][7]=0
            elif kind=="guard":row["output_bits"][-1]^=1
            elif kind=="integer":row["output_bits"][0]=0.0
            elif kind=="immutable":row["immutable_delta_and_state_guards"]=False
            elif kind=="completed":row["command_completed"]=False
            elif kind=="reset":case["baseline_restored_before_each_repetition"]=False
            elif kind=="in_place":case["in_place_state"]=False
            elif kind=="memory":case["admitted_bytes"]-=1
            elif kind=="launch":case["threadgroups"]+=1
            elif kind=="repetition":row["repetition"]=9
            with self.subTest(kind=kind),self.assertRaises(ValueError):residual.validate(native,inputs,anchors)

    def test_source_gate_and_independent_sum_are_both_required(self):
        inputs,anchors,_=residual.cases();native=fixture(inputs,anchors)
        anchors[inputs[0]["id"]]["reference"][0]+=10
        with self.assertRaisesRegex(ValueError,"FP32 source gate"):residual.validate(native,inputs,anchors)
        payload=residual.rms.payload
        def changed(root,manifest,name,dtype):
            data,item=payload(root,manifest,name,dtype)
            if name=="cases.capacity.layers.0.attention_residual" and dtype=="f16le":
                data=bytes([data[0]^1])+data[1:]
            return data,item
        with patch.object(residual.rms,"payload",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"independent Half sum"):residual.cases()

    def test_completed_overflow_rejects_success_wrong_stage_or_recovery_claim(self):
        message="completed residual contains nonfinite Half; in-place result invalid and discarded"
        good={"event":"idle_result","passed":False,"error":{"stage":"validation","message":message}}
        self.assertIs(residual.overflow_rejected(good),False)
        for bad in ({**good,"passed":True},{**good,"error":{"stage":"fixture","message":message}},
                {**good,"error":{"stage":"validation","message":"rolled back"}}):
            with self.assertRaisesRegex(ValueError,"overflow rejection"):residual.overflow_rejected(bad)

    def test_evidence_aliases_and_existing_inputs_fail_before_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/"result.json";build=root/"build"
            with patch.object(residual,"cases") as read:
                for inputs in (output,build,build/"inputs",root):
                    with self.subTest(inputs=inputs),self.assertRaises(ValueError):residual.run(output,build,inputs)
                inputs=root/"inputs";inputs.mkdir()
                with self.assertRaisesRegex(ValueError,"overwrite"):residual.run(output,build,inputs)
                read.assert_not_called()


if __name__=="__main__":unittest.main()
