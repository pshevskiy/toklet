"""Offline frozen gates, complete diagnostics and source/evidence integrity."""
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_rms as rms


def fixture(inputs,anchors):
    cases=[]
    for source in inputs:
        count=source["rows"]*source["hidden"]
        x,w=(count+7)*2,(source["hidden"]+7)*2
        values=anchors[source["id"]]["oracle_bits"]+[rms.OUTPUT_GUARD]*7
        cases.append({"id":source["id"],"rows_count":source["rows"],"hidden":source["hidden"],
            "epsilon_bits":source["epsilon_bits"],"guards":7,"output_guard_bits":rms.OUTPUT_GUARD,
            "shared_buffer_bytes":2*x+w,"host_upload_and_readback_bytes":3*x+2*w,
            "admitted_bytes":5*x+3*w+rms.probe.ALLOWANCE,"threadgroup_width":32,
            "threadgroups":math.ceil(source["rows"]/32),"allocations_reused_across_repetitions":True,
            "rows":[{"phase":"warmup" if i<3 else "sample","repetition":i+1 if i<3 else i-2,
                "command_completed":True,"immutable_input_weight_and_guards":True,
                "encode_commit_wait_seconds":0.001,"output_bits":values.copy()} for i in range(13)]})
    return {"event":"idle_result","passed":True,"result":{
        "operation":"rms_norm_fp16_storage_fp32_serial","cases":cases,"metadata":{
            "unified_memory":True,"language_version":"2.3","math_mode":"safe","math_functions":"precise",
            "minimum_macos_major":15,"maximum_admitted_bytes":max(case["admitted_bytes"] for case in cases),
            "execution_width":32,"pipeline_max_threads":1024,"device_max_threads":[1024,1024,1024]}}}


class RustMetalRms(unittest.TestCase):
    def test_changed_manifest_gate_and_oversized_manifest_fail_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/"manifest.json"
            data=(rms.FP16/"manifest.json").read_bytes()
            path.write_bytes(data)
            rms.manifest(root,rms.FP16_MANIFEST_SHA)
            changed=json.loads(data)
            changed["arrays"]["cases.single.final_norm"]["gate_against_source_fp32"]["atol"]*=2
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"manifest identity"):
                rms.manifest(root,rms.FP16_MANIFEST_SHA)
            path.write_bytes(b" "*((1 << 20)+1))
            with self.assertRaisesRegex(ValueError,"manifest identity"):
                rms.manifest(root,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_full_frozen_case_inventory_uses_original_gates_and_compacts_checked_output(self):
        inputs,anchors=rms.cases()
        self.assertEqual(len(inputs),28)
        self.assertEqual(sum(case["rows"]*case["hidden"] for case in inputs),7136)
        native=fixture(inputs,anchors)
        rms.validate(native,inputs,anchors)
        for case in native["result"]["cases"]:
            for row in case["rows"]:
                self.assertNotIn("output_bits",row)
                proof=row["qualification"]
                self.assertLessEqual(proof["against_source_fp32_max_gate_fraction"],1)
                self.assertTrue(proof["exact_independent_fp16_bits"])
                raw=struct.pack(f"<{len(anchors[case['id']]['oracle_bits'])}H",*anchors[case["id"]]["oracle_bits"])
                self.assertEqual(proof["output_sha256"],hashlib.sha256(raw).hexdigest())

    def test_corruption_in_values_guards_completion_and_accounting_is_rejected(self):
        inputs,anchors=rms.cases()
        inputs=[next(case for case in inputs if case["id"]=="cases.single.final_norm")]
        for kind in ("gate","nonfinite","guard","count","input","completed","phase","time","accounting","launch"):
            native=fixture(inputs,anchors)
            case=native["result"]["cases"][0]
            row=case["rows"][-1]
            if kind=="gate":row["output_bits"][0]=0x7bff
            elif kind=="nonfinite":row["output_bits"][0]=0x7c00
            elif kind=="guard":row["output_bits"][-1]^=1
            elif kind=="count":case["rows"].pop()
            elif kind=="input":row["immutable_input_weight_and_guards"]=False
            elif kind=="completed":row["command_completed"]=False
            elif kind=="phase":row["phase"]="warmup"
            elif kind=="time":row["encode_commit_wait_seconds"]=float("nan")
            elif kind=="accounting":case["admitted_bytes"]-=1
            elif kind=="launch":case["threadgroup_width"]=2048
            with self.subTest(kind=kind),self.assertRaises(ValueError):rms.validate(native,inputs,anchors)

    def test_changed_fixture_payload_and_metadata_fail_bounded_read(self):
        inputs,_=rms.cases()
        source=inputs[0]
        count=source["rows"]*source["hidden"]
        raw=struct.pack(f"<{count}H",*source["input_bits"])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            item={"file":"input.f16le","dtype":"f16le","shape":[source["rows"],source["hidden"]],
                  "bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
            manifest={"arrays":{"input":item}}
            path=root/item["file"]
            path.write_bytes(raw)
            rms.payload(root,manifest,"input","f16le")
            path.write_bytes(bytes([raw[0]^1])+raw[1:])
            with self.assertRaisesRegex(ValueError,"identity"):rms.payload(root,manifest,"input","f16le")
            path.write_bytes(raw)
            item["bytes"]+=1
            with self.assertRaisesRegex(ValueError,"metadata"):rms.payload(root,manifest,"input","f16le")

    def test_manual_subnormal_requires_exact_half_bits_even_inside_source_tolerance(self):
        inputs,anchors=rms.cases()
        inputs=[next(case for case in inputs if case["id"]=="manual.minimum_half_subnormal")]
        native=fixture(inputs,anchors)
        native["result"]["cases"][0]["rows"][0]["output_bits"][0]+=1
        with self.assertRaisesRegex(ValueError,"Manual exact Half"):
            rms.validate(native,inputs,anchors)

    def test_input_evidence_aliases_and_existing_build_fail_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output,build=root/"result.json",root/"build"
            with patch.object(rms.probe.subprocess,"Popen") as launch:
                for input_path in (output,build/"input.json"):
                    with self.assertRaisesRegex(ValueError,"distinct"):
                        rms.run(output,build,input_path)
                    self.assertFalse(input_path.exists())
                build.mkdir()
                with self.assertRaisesRegex(ValueError,"overwrite"):
                    rms.run(output,build,root/"input.json")
                self.assertFalse((root/"input.json").exists())
                launch.assert_not_called()


if __name__=="__main__":unittest.main()
