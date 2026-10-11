"""Mocked persistent startup ownership; no VM or subprocess is executed."""
import base64
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import podman_machine_lifecycle as lifecycle


class MachineLifecycleTests(unittest.TestCase):
    def exercise(self, behavior):
        state = {"clock": 0.0, "reads": 0,
                 "returncode": 0 if behavior in ["success", "late", "queued", "flood"] else 1 if behavior == "failed" else None}
        class Stream(io.BytesIO):
            def fileno(self): return 10 if self is stdout else 11
        class Selector:
            def __init__(self): self.keys = {}
            def register(self, stream, _, name): self.keys[stream] = SimpleNamespace(fileobj=stream, data=name)
            def unregister(self, stream): del self.keys[stream]
            def get_map(self): return self.keys
            def select(self, _):
                if state.get("shutdown") and behavior == "flood": return [(key, 1) for key in self.keys.values()]
                if state["reads"] and behavior != "flood": return []
                return [(next(key for key in self.keys.values() if key.data == "stdout"), 1)]
            def close(self): self.keys.clear()
        stdout, stderr = Stream(), Stream()
        def wait(**_): return state["returncode"]
        # returncode on a real Popen follows wait/poll; expose it dynamically.
        class Worker:
            pid = 123
            @property
            def returncode(self): return state["returncode"]
            def poll(self): return state["returncode"]
            def wait(self, **kwargs): return wait(**kwargs)
        owned = Worker(); owned.stdout = stdout; owned.stderr = stderr
        def read(_, size):
            self.assertEqual(size, 4096)
            state["reads"] += 1
            if state.get("shutdown") and behavior == "flood": return b""
            if behavior == "timeout": state["clock"] = 121.0
            return b"x" * 4096 if behavior == "flood" else b"helper output\n" if state.get("late") else b"started\n"
        def kill(_, errors):
            self.assertIs(_, owned)
            state["returncode"] = -9
            return True
        with patch.object(lifecycle.selectors, "DefaultSelector", Selector), \
             patch.object(lifecycle.subprocess, "Popen", return_value=owned) as popen, \
             patch.object(lifecycle.os, "set_blocking", side_effect=OSError("setup") if behavior == "setup" else None), \
             patch.object(lifecycle.os, "read", side_effect=read), \
             patch.object(lifecycle.threading, "Thread", return_value=SimpleNamespace(start=lambda: None, join=lambda **_: None, is_alive=lambda: False)), \
             patch.object(lifecycle.time, "monotonic", side_effect=lambda: state["clock"]), \
             patch.object(lifecycle, "signal_owned_group", side_effect=kill) as signal, \
             patch.object(lifecycle, "live_group_members", return_value=[] if behavior in ["timeout", "flood", "setup"] else [456]):
            result, plumbing = lifecycle.start_machine("/verified/podman")
            self.assertFalse(stdout.closed or stderr.closed)
            if behavior == "late":
                state["late"] = True; state["reads"] = 0
                plumbing.drain(0)
                self.assertEqual(bytes(plumbing.captured["stdout"]), b"started\nhelper output\n")
                self.assertFalse(stdout.closed or stderr.closed)
            with self.assertRaises(ValueError):
                plumbing.finish([{"Name": lifecycle.MACHINE, "Running": True, "Starting": False}])
            self.assertFalse(stdout.closed or stderr.closed)
            with patch.object(lifecycle, "live_group_members", return_value=[456]), self.assertRaises(ValueError):
                plumbing.finish([{"Name": lifecycle.MACHINE, "Running": False, "Starting": False}])
            self.assertFalse(stdout.closed or stderr.closed)
            if behavior == "queued":
                state["late"] = True; state["reads"] = 0
            state["shutdown"] = True
            with patch.object(lifecycle, "live_group_members", return_value=[]):
                plumbing.finish([{"Name": lifecycle.MACHINE, "Running": False, "Starting": False}])
        self.assertTrue(stdout.closed and stderr.closed)
        self.assertEqual(popen.call_args.args[0], ["/verified/podman", "machine", "start", "podman-machine-default"])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        return result, signal

    def test_success_preserves_helper_even_with_pipe_writer_still_alive(self):
        result, signal = self.exercise("success")
        self.assertTrue(result["client_succeeded"])
        self.assertTrue(result["machine_restoration_required"])
        self.assertEqual(result["live_startup_group_members"], [456])
        self.assertFalse(result["process_group_killed"])
        signal.assert_not_called()
        self.assertEqual(base64.b64decode(result["stdout_base64"]), b"started\n")

    def test_later_helper_output_keeps_readers_alive_until_verified_stop(self):
        result, signal = self.exercise("late")
        self.assertTrue(result["client_succeeded"])
        self.assertEqual(base64.b64decode(result["stdout_base64"]), b"started\nhelper output\n")
        self.assertTrue(result["helper_pipe_lifetime_completed"])
        signal.assert_not_called()

    def test_queued_final_helper_output_is_drained_after_verified_shutdown(self):
        result, signal = self.exercise("queued")
        self.assertEqual(base64.b64decode(result["stdout_base64"]), b"started\nhelper output\n")
        self.assertTrue(result["helper_pipe_lifetime_completed"])
        signal.assert_not_called()

    def test_failed_client_does_not_claim_machine_restoration(self):
        result, signal = self.exercise("failed")
        self.assertFalse(result["client_succeeded"])
        self.assertTrue(result["machine_restoration_required"])
        self.assertEqual(result["live_startup_group_members"], [456])
        signal.assert_not_called()

    def test_startup_timeout_kills_only_owned_group_and_requires_machine_cleanup(self):
        result, signal = self.exercise("timeout")
        self.assertTrue(result["wall_timed_out"])
        self.assertTrue(result["process_group_killed"])
        self.assertFalse(result["client_succeeded"])
        self.assertTrue(result["machine_restoration_required"])
        signal.assert_called_once()

    def test_output_bound_and_setup_failure_never_count_as_success(self):
        for mode in ["flood", "setup"]:
            with self.subTest(mode=mode):
                result, signal = self.exercise(mode)
                self.assertFalse(result["client_succeeded"])
                self.assertTrue(result["machine_restoration_required"])
                self.assertTrue(result["process_group_killed"])
                signal.assert_called_once()
                self.assertLessEqual(result["stdout_captured_bytes"] + result["stderr_captured_bytes"], 262144)
                if mode == "flood": self.assertTrue(result["output_limit_exceeded"])
                else: self.assertEqual(result["failure"]["message"], "setup")

    def test_publication_failure_cannot_bypass_unverified_owner_restoration(self):
        record = {}
        seen = []
        def finish(inventory):
            seen.append(inventory)
            if inventory != "stopped": raise ValueError("still running")
            record["helper_pipe_lifetime_completed"] = True
        owner = SimpleNamespace(record=record, finish=finish)
        observations = iter(["running", "stopped"])
        pauses = []
        with self.assertRaisesRegex(OSError, "publication failed"):
            try:
                raise OSError("publication failed")
            finally:
                lifecycle.await_stopped_owner(owner, lambda: next(observations), pauses.append)
        self.assertEqual(seen, ["running", "stopped"])
        self.assertEqual(pauses, [30])
        self.assertTrue(record["helper_pipe_lifetime_completed"])

    def test_failed_observation_or_already_restored_owner_never_drops_live_readers(self):
        record = {}; calls = []
        def observe():
            calls.append(1)
            if len(calls) == 1: raise RuntimeError("inventory unavailable")
            return "stopped"
        def finish(_): record["helper_pipe_lifetime_completed"] = True
        owner = SimpleNamespace(record=record, finish=finish)
        lifecycle.await_stopped_owner(owner, observe, lambda seconds: self.assertEqual(seconds, 30))
        self.assertEqual(len(calls), 2)
        lifecycle.await_stopped_owner(owner, lambda: self.fail("restored owner was observed again"))
        lifecycle.await_stopped_owner(None, lambda: self.fail("absent owner was observed"))

    def test_fixed_limits_reject_changes_before_any_process(self):
        with patch.object(lifecycle.subprocess, "Popen") as popen:
            for kwargs in [{"wall_seconds": 121}, {"output_bytes": 262145}]:
                with self.assertRaises(ValueError): lifecycle.start_machine("podman", **kwargs)
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
