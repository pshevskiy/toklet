"""Offline independent bit validation and bounded native-protocol checks."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_probe as probe


def fixture():
    cases = []
    for count in probe.SHAPES:
        input_bits = [probe.float_bits((i % 29)*0.25-3.5) for i in range(count)]+[probe.INPUT_GUARD]*7
        output_bits = [probe.float_bits((i % 29)*0.25-2.5) for i in range(count)]+[probe.OUTPUT_GUARD]*7
        rows = [{"phase":"warmup" if i<3 else "sample", "repetition":i+1 if i<3 else i-2,
                 "command_completed":True,"encode_commit_wait_seconds":0.001,
                 "input_bits":input_bits,"output_bits":output_bits} for i in range(13)]
        cases.append({"count":count,"guards":7,"threadgroup_width":32,"threadgroups":(count+31)//32,
                      "shared_buffer_bytes":(count+7)*8,"host_input_and_readback_bytes":(count+7)*12,
                      "admitted_bytes":(count+7)*20+probe.ALLOWANCE,
                      "allocations_reused_across_repetitions":True,"rows":rows})
    return {"event":"idle_result","passed":True,"result":{
        "metadata":{"unified_memory":True,"language_version":"2.3","math_mode":"safe",
                    "math_functions":"precise","minimum_macos_major":15,
                    "maximum_admitted_bytes":cases[-1]["admitted_bytes"],
                    "execution_width":32,"pipeline_max_threads":1024,
                    "device_max_threads":[1024,1024,64]}, "cases":cases}}


class RustMetal(unittest.TestCase):
    def test_custom_negative_fixture_requires_validated_failure_and_exit_one(self):
        result={"exit_code":1,"event":{"passed":False}}
        probe.validate_run("overflow",result,lambda event:False)
        for passed,code in ((True,1),(False,0),(True,0)):
            with self.subTest(passed=passed,code=code),self.assertRaisesRegex(ValueError,"exit policy"):
                probe.validate_run("overflow",{"exit_code":code,"event":{"passed":passed}},lambda event:False)
        probe.validate_run("primitive",{"exit_code":0,"event":{"passed":True}},lambda event:None)
        with self.assertRaisesRegex(ValueError,"exit policy"):
            probe.validate_run("primitive",result,lambda event:None)

    def test_inherited_cargo_target_directory_is_overridden_for_every_check(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)/"isolated build"
            external = str(Path(directory)/"external-target")
            with patch.dict(os.environ,{"CARGO_TARGET_DIR":external}):
                commands = probe.build_commands(build)
                self.assertEqual(len(commands),4)
                for command in commands:
                    self.assertEqual(command[0],"env")
                    self.assertIn(f"CARGO_TARGET_DIR={(build/'target').resolve()}",command)
                    self.assertNotIn(f"CARGO_TARGET_DIR={external}",command)
                    self.assertIn(str((build/"Cargo.toml").resolve()),command)
                self.assertEqual(os.environ["CARGO_TARGET_DIR"],external)

    def test_exact_value_input_output_guard_and_completed_work_are_required(self):
        good = fixture()
        probe.validate_result(good)
        for kind in ("input_value","input_guard","output_value","output_guard","count","phase","completed","nan","accounting","device_width"):
            bad = copy.deepcopy(good)
            case = bad["result"]["cases"][-1]
            row = case["rows"][-1]
            if kind == "input_value": row["input_bits"][0] ^= 1
            elif kind == "input_guard": row["input_bits"][-1] ^= 1
            elif kind == "output_value": row["output_bits"][0] ^= 1
            elif kind == "output_guard": row["output_bits"][-1] ^= 1
            elif kind == "count": case["rows"].pop()
            elif kind == "phase": row["phase"] = "warmup"
            elif kind == "completed": row["command_completed"] = False
            elif kind == "nan": row["encode_commit_wait_seconds"] = float("nan")
            elif kind == "accounting": case["admitted_bytes"] -= 1
            elif kind == "device_width": bad["result"]["metadata"]["device_max_threads"][0] = 16
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                probe.validate_result(bad)

    def test_retry_and_build_evidence_alias_never_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/"run.json"
            output.with_suffix(".json.native.jsonl").write_text("interrupted evidence\n")
            with patch.object(probe.subprocess,"Popen") as launch:
                with self.assertRaisesRegex(ValueError,"overwrite"):
                    probe.run(output,root/"build")
                with self.assertRaisesRegex(ValueError,"outside"):
                    probe.run(root/"build/Cargo.toml",root/"build")
                launch.assert_not_called()

    def test_child_remains_alive_for_memory_and_expected_failure_exits_after_ack(self):
        script = ("import json,sys; "
                  "print(json.dumps({'event':'idle_result','passed':False,'error':{'stage':'admission','message':'expected'}}),flush=True); "
                  "ack=sys.stdin.readline(); sys.exit(1 if ack=='done\\n' else 9)")
        def peak(pid):
            os.kill(pid,0)  # only the child created by this test
            return 123
        raw = io.StringIO()
        with tempfile.TemporaryFile(mode="w+") as stderr, patch.object(probe.memory,"footprint_peak",side_effect=peak) as read_memory:
            result = probe.run_child([sys.executable,"-c",script],raw,stderr,5)
        self.assertEqual(read_memory.call_count,1)
        self.assertEqual(result["exit_code"],1)
        self.assertEqual(result["physical_process_charge_peak_bytes"],123)
        self.assertFalse(result["watchdog_killed"])
        self.assertEqual(len(raw.getvalue().splitlines()),1)

    def test_unfinished_native_protocol_is_bounded_and_reaped(self):
        # Child waits forever after malformed output unless cleanup kills its
        # owned group. Successful return from this test proves it was reaped.
        script = "import sys,time; print('malformed',flush=True); time.sleep(30)"
        with tempfile.TemporaryFile(mode="w+") as stderr, patch.object(probe.memory,"footprint_peak") as read_memory:
            with self.assertRaises(json.JSONDecodeError):
                probe.run_child([sys.executable,"-c",script],io.StringIO(),stderr,5)
            read_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
