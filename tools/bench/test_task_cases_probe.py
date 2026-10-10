"""Exercise task transport with a tiny fake CLI, never a downloaded model."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import task_cases_probe as probe
from test_task_cases import events, inventory


class TaskTransport(unittest.TestCase):
    def run_fake(self, variant="normal", timer=None):
        cases = [inventory()[0][index] for index in (40, 70)]
        for case in cases:
            case["expected_first_token"] = 200
        by_cap = {str(case["max_output_tokens"]): events(case, text="authored")[:-1] for case in cases}
        body = """import json, sys
mode=sys.argv[-1]
ready={'event':'ready','backend':'cpu','context_capacity':2048,'cpu_projection':mode,'memory_admitted_bytes':1000}
variant=VARIANT
data=json.loads(DATA)
if variant=='nonfinite':
    print('{"event":"ready","number":NaN}',flush=True)
else:
    print(json.dumps(ready),flush=True)
for line in sys.stdin:
    command=json.loads(line)
    if command['action']=='reset':
        if variant=='missing_reset': sys.exit(0)
        print('{"event":"reset"}',flush=True)
    elif variant=='error':
        print('{"event":"error","message":"admission failed"}',flush=True)
    else:
        for event in data[str(command['max_new_tokens'])]:
            if event['event']=='done': event['statistics']['cpu_projection']=mode
            print(json.dumps(event),flush=True)
if variant=='trailing': print('{"event":"reset"}',flush=True)
""".replace("VARIANT", repr(variant)).replace("DATA", repr(json.dumps(by_cap)))
        raw = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory)/"fake-cli"
            binary.write_text(f"#!{sys.executable}\n"+body)
            binary.chmod(0o700)
            with tempfile.TemporaryFile(mode="w+t") as errors, patch.object(probe.common,"footprint_peak",return_value=1000):
                if timer:
                    with patch.object(probe.threading,"Timer",timer):
                        result=probe.child(binary,cases,"tiled",raw,errors)
                else:
                    result=probe.child(binary,cases,"tiled",raw,errors)
        return result,raw.getvalue()

    def test_persistent_fake_cli_preserves_raw_ids_and_completed_responses(self):
        result, raw = self.run_fake()
        self.assertTrue(result["passed"])
        self.assertEqual(result["exit_code"],0)
        self.assertFalse(result["watchdog_killed"])
        self.assertFalse(result["process_group_killed"])
        self.assertEqual(result["cleanup_errors"],[])
        self.assertEqual([len(case["response"]["visible_token_ids"]) for case in result["cases"]],[32,16])
        self.assertTrue(all(case["request"]["tokens"]==[12,34] for case in result["cases"]))
        self.assertTrue(all(case["conservative_joint_physical_peak_bytes"]==2000 for case in result["cases"]))
        self.assertEqual(len(raw.splitlines()),53)

    def test_error_missing_reset_nonfinite_and_trailing_output_never_pass(self):
        for variant in ("error","missing_reset","nonfinite","trailing"):
            with self.subTest(variant=variant):
                result, raw = self.run_fake(variant)
                self.assertFalse(result["passed"])
                self.assertIn("failure",result)
                self.assertFalse(result["watchdog_killed"])
                self.assertIsNotNone(result["exit_code"])
                self.assertEqual(result["cleanup_errors"],[])
                self.assertTrue(raw)

    def test_watchdog_kills_and_reaps_only_its_owned_group(self):
        class ImmediateTimer:
            def __init__(self,delay,callback):self.callback=callback
            def start(self):self.callback()
            def cancel(self):pass
        result,_=self.run_fake(timer=ImmediateTimer)
        self.assertFalse(result["passed"])
        self.assertTrue(result["watchdog_killed"])
        self.assertTrue(result["process_group_killed"])
        self.assertIsNotNone(result["exit_code"])
        self.assertEqual(result["cleanup_errors"],[])

    def test_empty_group_permission_error_needs_proof_of_no_live_members(self):
        class Worker:
            pid = 123
            def poll(self): return 0
        for snapshot, expected_errors in (("123 Z\n", 0), ("124 S\n", 0), ("123 S\n", 1)):
            errors = []
            with patch.object(probe.os, "killpg", side_effect=PermissionError(1, "Operation not permitted")), patch.object(probe.subprocess, "check_output", return_value=snapshot):
                self.assertFalse(probe.signal_owned_group(Worker(), errors))
            self.assertEqual(len(errors), expected_errors)

    def test_probe_inventory_uses_only_independent_saved_numerical_ids(self):
        cases=probe.probe_cases()
        self.assertEqual([case["id"] for case in cases],["probe/future_changed","probe/short"])
        self.assertEqual([case["max_output_tokens"] for case in cases],[32,16])
        self.assertTrue(all(len(case["prompt_tokens"])==4 for case in cases))
        with patch.object(probe.prepare,"digest",return_value="changed"),self.assertRaisesRegex(ValueError,"identity"):
            probe.probe_cases()

    def test_mode_comparison_requires_complete_matching_finalized_responses(self):
        tiled, _ = self.run_fake()
        scalar = copy.deepcopy(tiled)
        scalar["projection"] = "scalar"
        probe.compare_modes([scalar, tiled])
        for key, value in (("visible_token_ids", [1]), ("visible_text", "changed"),
                           ("stop_reason", {"kind": "stop_token", "token_id": 151645})):
            changed = copy.deepcopy(tiled)
            changed["cases"][0]["response"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "finalized response"):
                probe.compare_modes([scalar, changed])
        with self.assertRaisesRegex(ValueError, "coverage"):
            probe.compare_modes([scalar])

    def test_idle_requirement_rejects_live_or_unheld_numerical_work(self):
        proof={"completed_workload":"cached","measured_requests":10,"held_orchestration_pid":10,
               "defunct_transport_pid":11,"exited_numerical_worker_pid":12}
        held="PID PPID STAT TIME COMMAND\n10 1 Ts 0:00 parent\n11 10 Z 0:00 defunct\n"
        with patch.object(probe.subprocess,"check_output",return_value=held):self.assertEqual(probe.idle(proof),held)
        for snapshot in (held+"12 11 R 0:00 numerical\n",held.replace("Ts","Ss"),held.replace("11 10 Z","11 10 S")):
            with patch.object(probe.subprocess,"check_output",return_value=snapshot),self.assertRaisesRegex(ValueError,"has not exited"):
                probe.idle(proof)


if __name__ == "__main__":
    unittest.main()
