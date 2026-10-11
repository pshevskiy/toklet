"""Offline collector fixtures; no models or generated code."""
import io
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import held_out_tasks as collector
from test_task_cases import inventory, events


class PhaseTimer:
    instances = []

    def __init__(self, delay, callback):
        self.delay = delay
        self.callback = callback
        self.started = self.cancelled = self.joined = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def join(self):
        self.joined = True


class FakeWorker:
    pid = 12345

    def __init__(self, body):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(body)
        self.returncode = None

    def wait(self, timeout):
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode


class Collection(unittest.TestCase):
    def collect(self, cases, immediate=False, mutate=None, trailing="", timer_type=PhaseTimer, survivors=None, group_observations=None, kill_result=False):
        ready = {"event": "ready", "backend": "cpu", "context_capacity": 2048,
                 "cpu_projection": "tiled", "memory_admitted_bytes": 1000}
        streams = [events(case, count=0 if immediate else 1, eos=True,
                          text="", tail="" if immediate else "tail Ω", projection="tiled") for case in cases]
        if mutate:
            mutate(streams)
        lines = [json.dumps(ready)] + [json.dumps(event) for stream in streams for event in stream]
        worker = FakeWorker("\n".join(lines) + "\n" + trailing)
        PhaseTimer.instances = []
        raw = io.StringIO()
        with patch.object(collector.subprocess, "Popen", return_value=worker), \
                patch.object(collector.threading, "Timer", timer_type), \
                patch.object(collector.common, "footprint_peak", return_value=1000), \
                patch.object(collector.time, "sleep"), \
                patch.object(collector.boundary, "live_group_members", return_value=survivors or [], side_effect=group_observations), \
                patch.object(collector, "signal_owned_group", return_value=kill_result):
            result = collector.child(Path("fake"), cases, "tiled", raw, io.StringIO())
        return result, raw.getvalue()

    def test_all_100_cases_accept_immediate_hidden_eos_and_reset(self):
        cases = inventory()[0]
        result, raw = self.collect(cases, immediate=True)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["cases"]), 100)
        self.assertTrue(all(item["response"]["visible_token_ids"] == [] for item in result["cases"]))
        self.assertTrue(all(item["response"]["visible_text"] == "" for item in result["cases"]))
        self.assertEqual(len(raw.splitlines()), 201)
        self.assertEqual([timer.delay for timer in PhaseTimer.instances], [1200] + [1800] * 100 + [20])
        self.assertTrue(all(timer.started and timer.cancelled and timer.joined for timer in PhaseTimer.instances))
        self.assertEqual(result["surviving_group_members"], [])
        self.assertEqual(result["exit_code"], 0)

    def test_collector_does_not_require_authored_first_token_and_keeps_tail(self):
        cases = [inventory()[0][index] for index in (0, 40, 70)]
        result, raw = self.collect(cases)
        self.assertTrue(result["passed"])
        self.assertTrue(all(item["response"]["visible_token_ids"] == [200] for item in result["cases"]))
        self.assertTrue(all(item["response"]["visible_text"] == "tail Ω" for item in result["cases"]))
        self.assertEqual(len(raw.splitlines()), 13)

    def test_missing_reset_and_invalid_state_abort_without_inventing_response(self):
        case = inventory()[0][40]
        def missing_reset(streams):
            streams[0].pop()
        def bad_state(streams):
            streams[0][-2]["statistics"]["generation"]["pending_tokens"] = 0
        def late_token(streams):
            streams[0].insert(-1, {"event": "token", "index": 1, "token_id": 201, "text": "", "elapsed_seconds": 1})
        for mutate in (missing_reset, bad_state, late_token):
            with self.subTest(mutate=mutate.__name__):
                result, raw = self.collect([case], mutate=mutate)
                self.assertFalse(result["passed"])
                self.assertEqual(result["cases"], [])
                self.assertTrue(raw)
                self.assertEqual(result["exit_code"], 0)

    def test_trailing_output_fails_completed_responses(self):
        case = inventory()[0][70]
        result, _ = self.collect([case], trailing='{"event":"reset"}\n')
        self.assertFalse(result["passed"])
        self.assertIn("trailing_stdout", result)

    def test_output_limit_keeps_finalization_at_reset(self):
        case = inventory()[0][70]
        def output_limit(streams):
            streams[0][:] = events(case, tail="reset Ω", projection="tiled")
        result, raw = self.collect([case], mutate=output_limit)
        self.assertTrue(result["passed"])
        self.assertEqual(result["cases"][0]["response"]["visible_text"], "reset Ω")
        self.assertEqual(len(result["cases"][0]["response"]["visible_token_ids"]), 16)
        self.assertEqual(len(raw.splitlines()), 20)

    def test_surviving_owned_group_prohibits_success_after_normal_leader_exit(self):
        result, _ = self.collect([inventory()[0][70]], survivors=[{"pid": 12346}])
        self.assertFalse(result["passed"])
        self.assertEqual(result["cleanup_errors"][0]["type"], "SurvivingGroup")

    def test_descendants_after_normal_exit_are_killed_and_rechecked(self):
        result, _ = self.collect([inventory()[0][70]], group_observations=[[12346], []], kill_result=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["initial_surviving_group_members"], [12346])
        self.assertEqual(result["surviving_group_members"], [])
        self.assertTrue(result["process_group_killed"])
        self.assertEqual(result["cleanup_errors"], [])
        self.assertEqual(result["failure"]["type"], "UnexpectedDescendants")

    def test_shutdown_deadline_firing_during_join_cannot_preserve_success(self):
        class LateShutdownTimer(PhaseTimer):
            def join(self):
                super().join()
                if self.delay == 20:
                    self.callback()
        result, _ = self.collect([inventory()[0][70]], timer_type=LateShutdownTimer)
        self.assertFalse(result["passed"])
        self.assertTrue(result["watchdog_killed"])
        self.assertEqual(result["failure"]["type"], "WatchdogExpired")

    def test_expired_case_timer_cannot_arm_next_phase(self):
        class ExpiringTimer(PhaseTimer):
            def start(self):
                self.started = True
                if self.delay == 1800:
                    self.callback()
        result, raw = self.collect([inventory()[0][70]], timer_type=ExpiringTimer)
        self.assertFalse(result["passed"])
        self.assertTrue(result["watchdog_killed"])
        self.assertIn("already expired", result["failure"]["message"])

    def test_real_persistent_fake_cli_consumes_all_100_requests_and_exits(self):
        cases = inventory()[0]
        bodies = [events(case, count=0, eos=True, projection="tiled") for case in cases]
        body = """import json, sys
streams=json.loads(DATA)
mode=sys.argv[-1]
print(json.dumps({'event':'ready','backend':'cpu','context_capacity':2048,
                  'cpu_projection':mode,'memory_admitted_bytes':1000}),flush=True)
index=0
for line in sys.stdin:
    command=json.loads(line)
    if command['action']=='generate':
        assert command['tokens']==[12,34]
        for event in streams[index][:-1]:print(json.dumps(event),flush=True)
    elif command['action']=='reset':
        print(json.dumps(streams[index][-1]),flush=True)
        index+=1
    else:raise ValueError('unknown request')
assert index==100
""".replace("DATA", repr(json.dumps(bodies)))
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/"fake-cli"
            binary.write_text("#!"+sys.executable+"\n"+body)
            binary.chmod(0o700)
            raw=io.StringIO()
            with tempfile.TemporaryFile(mode="w+t") as errors, \
                    patch.object(collector.common,"footprint_peak",return_value=1000):
                result=collector.child(binary,cases,"tiled",raw,errors)
            self.assertTrue(result["passed"], result.get("failure"))
            self.assertEqual(len(result["cases"]),100)
            self.assertEqual(len(raw.getvalue().splitlines()),201)
            self.assertEqual(result["cleanup_errors"],[])
            self.assertEqual(result["surviving_group_members"],[])
            self.assertFalse(result["watchdog_killed"])

    def test_duplicate_case_or_bad_mode_rejected_before_spawn(self):
        case = inventory()[0][70]
        for cases, mode in (([case, case], "tiled"), ([case], "other"), ([], "scalar")):
            with patch.object(collector.subprocess, "Popen") as spawned:
                with self.assertRaises(ValueError):
                    collector.child(Path("fake"), cases, mode, io.StringIO(), io.StringIO())
                spawned.assert_not_called()


if __name__ == "__main__":
    unittest.main()
