"""Offline exact-oracle/probe lifecycle checks; no Apple framework required."""
import io
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import mps_oracle
import mps_probe


class MpsChecks(unittest.TestCase):
    def test_exact_integer_oracle_and_manual_bias_range(self):
        fixture = mps_oracle.verify()
        self.assertEqual(len(fixture["cases"]), 13)
        for case in fixture["cases"]:
            if case["kind"] == "synthetic":
                self.assertLess(case["maximum_absolute_partial_sum_numerator"], 2**24)
                d = case["shape"][1]
                for row, out in ((0, 0), (4, 2), (12, 16)):
                    expected = sum(((row*7+i)%13-6)*((out+3*i)%17-8) for i in range(d))
                    self.assertEqual(case["integer_table"][row][out], expected)
            else:
                for row, values in enumerate(case["left"]):
                    for out, weights in enumerate(case["right"]):
                        self.assertEqual(sum(a*b for a,b in zip(values,weights))+case["bias"][out], case["expected"][row][out])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"manifest.json"; mps_oracle.export(path)
            self.assertEqual(mps_oracle.verify(path), fixture)
            edited = json.loads(path.read_text()); edited["cases"][0]["integer_table"][0][0] += 1; path.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError, "oracle differ"):
                mps_oracle.verify(path)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                mps_oracle.export(path)

    def test_existing_evidence_or_build_refused_before_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in ("", ".compile.stderr.txt", ".f16.jsonl", ".f16-to-f32.stderr.txt"):
                output, build = root/"output.json", root/"build"
                existing = output.with_suffix(output.suffix+suffix); existing.touch()
                with self.assertRaisesRegex(ValueError, "Refusing"):
                    mps_probe.run(output,build)
                existing.unlink()
            build.mkdir()
            with self.assertRaisesRegex(ValueError, "Refusing"):
                mps_probe.run(output,build)

    def test_idle_complete_handshake_peak_and_partial_failures(self):
        code = "import json,sys; print(json.dumps({'event':'complete'}),flush=True); assert input()=='quit'"
        with tempfile.TemporaryFile(mode="w+") as errors, patch.object(mps_probe.memory,"footprint_peak",return_value=123) as peak:
            raw = io.StringIO(); result = mps_probe.run_child([sys.executable,"-c",code],raw,errors,timeout=3)
            self.assertTrue(result["completed"]); self.assertEqual(result["physical_process_charge_peak_bytes"],123); peak.assert_called_once()
            code = "import json,sys; print(json.dumps({'event':'case_started','name':'bad'}),flush=True); sys.exit(7)"
            raw = io.StringIO(); result = mps_probe.run_child([sys.executable,"-c",code],raw,errors,timeout=3)
            self.assertFalse(result["completed"]); self.assertEqual(result["exit_code"],7); self.assertIn('bad',raw.getvalue())
            code = "import json,sys; print(json.dumps({'event':'failure','message':'rejected'}),flush=True); assert input()=='quit'; sys.exit(1)"
            result = mps_probe.run_child([sys.executable,"-c",code],io.StringIO(),errors,timeout=3)
            self.assertFalse(result["completed"]); self.assertEqual(result["physical_process_charge_peak_bytes"],123)

    def test_native_result_inventory_skips_counts_and_quality_are_strict(self):
        cases = []
        for case in mps_oracle.verify()["cases"] + [{"name":"single_cast_bias_edge","shape":[1,2,1]}]:
            cases.append({"name":case["name"],"shape":case["shape"],"samples":[{"sample_index":i,"completed_host_seconds":1} for i in range(10)],
                          "exact_outputs":True,"guards_untouched":True,"logical_buffer_bytes":128})
        names = json.loads((mps_probe.TINY_HALF/"manifest.json").read_text())["cases"]
        result = {"mode":"f32","warmups":3,"repetitions":10,"cases":cases,
                  "tiny_projection_checks":[{"case":name,"maximum_tolerance_fraction":.5,"guards_untouched":True} for name in names]}
        mps_probe.validate_result(result,"f32")
        mutations = [lambda r:r['cases'].pop(), lambda r:r['cases'].__setitem__(0,{"name":"manual_bias","skipped":"bad"}),
                     lambda r:r['cases'][0]['samples'].pop(), lambda r:r['cases'][0]['samples'][0].update(completed_host_seconds=float('nan')),
                     lambda r:r['tiny_projection_checks'][0].update(maximum_tolerance_fraction=1.01)]
        for mutate in mutations:
            edited = copy.deepcopy(result); mutate(edited)
            with self.assertRaises(ValueError): mps_probe.validate_result(edited,"f32")

    def test_watchdog_kills_only_its_owned_group_and_reaps(self):
        with tempfile.TemporaryFile(mode="w+") as errors, patch.object(mps_probe.os,"killpg",wraps=os.killpg) as kill:
            result = mps_probe.run_child([sys.executable,"-c","import time; time.sleep(60)"],io.StringIO(),errors,timeout=.1)
            self.assertFalse(result["completed"]); self.assertTrue(result["watchdog_killed"])
            kill.assert_called_once(); pid,sig = kill.call_args.args; self.assertEqual(sig,signal.SIGKILL)
            with self.assertRaises(ChildProcessError): os.waitpid(pid,os.WNOHANG)


if __name__ == "__main__":
    unittest.main()
