"""Offline isolation evidence/transport tests; no sandbox or model required."""
import base64
import copy
import errno
import hashlib
import io
import json
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import code_isolation_probe as probe


def ready(mode):
    return {"event": "isolated", "mode": mode, "policy": probe.child.POLICY.copy(),
            "profile": "authored-fixture", "limits": {name: [value, value] for name, (_, value) in probe.child.limits().items()},
            "python": "offline fixture", "isolated_flag": 1, "no_site_flag": 1, "optimize_flag": 0,
            "dont_write_bytecode": True, "environment_keys": sorted(probe.ENVIRONMENT)}


def capture(mode, result=None, code=0, tail=b"", stderr=b"", **updates):
    output = (json.dumps(ready(mode)) + "\n").encode()
    if result is not None:
        output += (json.dumps({"event": "probe_result", "mode": mode, "result": result}) + "\n").encode()
    output += tail
    record = {"mode": mode, "exit_code": code, "wall_timed_out": False, "output_limit_exceeded": False,
              "process_group_killed": False, "cleanup_errors": [], "surviving_group_members": [], "elapsed_seconds": 1.0}
    for name, payload in (("stdout", output), ("stderr", stderr)):
        record[name + "_base64"] = base64.b64encode(payload).decode()
        record[name + "_captured_bytes"] = len(payload)
        record[name + "_captured_sha256"] = hashlib.sha256(payload).hexdigest()
    record.update(updates)
    return record


class IsolationEvidence(unittest.TestCase):
    def test_active_admission_and_complete_benign_check_are_required(self):
        good = capture("benign", {"check_returned": True})
        self.assertEqual(probe.check_record(good), ready("benign"))
        for result in (None, {"check_returned": False}, {"check_returned": 1}, {"check_returned": True, "extra": True}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                probe.check_record(capture("benign", result))
        bad = copy.deepcopy(good)
        bad["stdout_base64"] = base64.b64encode(b'{"event":"setup_error"}\n').decode()
        with self.assertRaises(ValueError):
            probe.check_record(bad)

    def test_failed_assertion_and_early_zero_exit_are_distinct_evidence(self):
        probe.check_record(capture("assertion_failure", {"check_returned": False, "exception": "AssertionError"}))
        probe.check_record(capture("early_exit"))
        with self.assertRaises(ValueError):
            probe.check_record(capture("benign"))
        with self.assertRaises(ValueError):
            probe.check_record(capture("assertion_failure", {"check_returned": True}))
        with self.assertRaises(ValueError):
            probe.check_record(capture("early_exit", {"check_returned": True}))

    def test_refusal_is_not_network_denial_and_memory_probes_bind_the_request(self):
        for mode in ("file_read", "file_write", "network", "fork", "exec"):
            probe.check_record(capture(mode, {"denied": True, "errno": errno.EPERM, "exception": "PermissionError"}))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                probe.check_record(capture(mode, {"denied": True, "errno": errno.ECONNREFUSED}))
        probes = {
            "data_limit": {"bounded": True, "exception": "MemoryError", "requested_bytes": 512 * 1024**2},
            "address_space_limit": {"bounded": True, "exception": "OSError", "errno": errno.ENOMEM, "requested_bytes": 2 * 1024**3},
        }
        for mode, result in probes.items():
            probe.check_record(capture(mode, result))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                probe.check_record(capture(mode, {**result, "requested_bytes": 1}))
        with patch.object(probe.child.os, "fork", return_value=123), \
             patch.object(probe.child.os, "waitpid", side_effect=PermissionError(errno.EPERM, "authored wait denial")):
            result = probe.child.probe("fork", "unused", "unused")
        self.assertIs(result["denied"], False)
        handle = Mock()
        handle.settimeout.side_effect = PermissionError(errno.EPERM, "authored timeout setup denial")
        with patch.object(probe.child.socket, "socket", return_value=handle):
            result = probe.child.probe("network", "unused", "unused")
        self.assertIs(result["denied"], False)
        handle.connect.assert_not_called()
        handle.close.assert_called_once()

    def test_resource_termination_requires_the_expected_cause_and_owned_cleanup(self):
        probe.check_record(capture("cpu_timeout", code=-signal.SIGKILL))
        probe.check_record(capture("wall_timeout", code=-signal.SIGKILL, wall_timed_out=True, process_group_killed=True))
        for record in (capture("cpu_timeout", code=-signal.SIGKILL, wall_timed_out=True),
                       capture("cpu_timeout", code=-signal.SIGKILL, process_group_killed=True),
                       capture("wall_timeout", code=0, wall_timed_out=True, process_group_killed=True),
                       capture("wall_timeout", code=-signal.SIGKILL)):
            with self.subTest(mode=record["mode"]), self.assertRaises(ValueError):
                probe.check_record(record)
        for mode in ("stdout_flood", "stderr_flood"):
            initial = (json.dumps(ready(mode)) + "\n").encode()
            remaining = probe.child.POLICY["output_bytes"] - len(initial)
            kwargs = {"tail": b"x" * remaining} if mode == "stdout_flood" else {"stderr": b"x" * remaining}
            good = capture(mode, code=-signal.SIGKILL, output_limit_exceeded=True, process_group_killed=True, **kwargs)
            probe.check_record(good)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                probe.check_record({**good, "process_group_killed": False})

    def test_corruption_unbounded_capture_and_failed_cleanup_never_qualify(self):
        good = capture("benign", {"check_returned": True})
        for update in ({"stdout_captured_sha256": "changed"}, {"stdout_captured_bytes": True},
                       {"cleanup_errors": [{"type": "PermissionError"}]}, {"surviving_group_members": [123]},
                       {"exit_code": False}, {"elapsed_seconds": float("nan")}, {"wall_timed_out": 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                probe.check_record({**good, **update})
        with self.assertRaises(ValueError):
            probe.check_record(capture("benign", {"check_returned": True}, tail=b"x" * 65536))

    def test_idle_guard_rejects_incomplete_or_live_numerical_work(self):
        proof = {"completed_workload": "pending", "measured_requests": 10, "held_orchestration_pid": 10,
                 "defunct_transport_pid": 11, "exited_numerical_worker_pid": 12}
        held = "PID PPID STAT TIME COMMAND\n10 1 Ts 0:00 parent\n11 10 Z 0:00 defunct\n"
        with patch.object(probe.subprocess, "check_output", return_value=held):
            self.assertEqual(probe.idle(proof), held)
        for snapshot in (held + "12 11 R 0:00 worker\n", held.replace("Ts", "Ss"), held.replace("11 10 Z", "11 10 S")):
            with patch.object(probe.subprocess, "check_output", return_value=snapshot), self.assertRaises(ValueError):
                probe.idle(proof)
        with self.assertRaises(ValueError):
            probe.idle({**proof, "measured_requests": 9})

    def execute_fake(self, mode, behavior):
        class Stream(io.BytesIO):
            def fileno(self):
                return 10 if self is stdout else 11

        class Selector:
            def __init__(self): self.keys = {}
            def register(self, stream, _, name): self.keys[stream] = SimpleNamespace(fileobj=stream, data=name)
            def unregister(self, stream): del self.keys[stream]
            def get_map(self): return self.keys
            def select(self, _):
                wanted = "stderr" if mode == "stderr_flood" and state["read"] else "stdout"
                return [(next(key for key in self.keys.values() if key.data == wanted), 1)]
            def close(self): self.keys.clear()

        stdout, stderr = Stream(), Stream()
        exit_code = -signal.SIGKILL if behavior in ("flood", "wall") else 0
        worker = SimpleNamespace(pid=123, stdout=stdout, stderr=stderr, returncode=exit_code,
                                 wait=lambda **_: exit_code)
        initial = (json.dumps(ready(mode)) + "\n").encode()
        state = {"read": 0, "clock": 0.0}

        def read(_, size):
            self.assertEqual(size, 4096)
            state["read"] += 1
            if state["read"] == 1:
                if behavior == "wall": state["clock"] = 7.0
                return initial
            return b"x" * 4096

        with patch.object(probe.selectors, "DefaultSelector", Selector), patch.object(probe.subprocess, "Popen", return_value=worker), \
             patch.object(probe.os, "set_blocking", side_effect=OSError("authored setup error") if behavior == "setup" else None), \
             patch.object(probe.os, "read", side_effect=read), patch.object(probe.time, "monotonic", side_effect=lambda: state["clock"]), \
             patch.object(probe, "signal_owned_group", return_value=behavior != "setup"), \
             patch.object(probe, "live_group_members", return_value=[]):
            return probe.execute(mode, "owned-read", "owned-write", ".")

    def test_controller_combined_output_and_wall_bounds_without_a_native_child(self):
        for mode in ("stdout_flood", "stderr_flood"):
            flood = self.execute_fake(mode, "flood")
            self.assertEqual(flood["stdout_captured_bytes"] + flood["stderr_captured_bytes"], 65536)
            probe.check_record(flood)
        wall = self.execute_fake("wall_timeout", "wall")
        self.assertTrue(wall["wall_timed_out"])
        probe.check_record(wall)

    def test_pipe_setup_failure_preserves_failure_and_reaps_its_owned_child(self):
        record = self.execute_fake("benign", "setup")
        self.assertEqual(record["failure"]["message"], "authored setup error")
        self.assertEqual(record["surviving_group_members"], [])
        self.assertEqual(record["exit_code"], 0)
        with self.assertRaises(ValueError):
            probe.check_record(record)


if __name__ == "__main__":
    unittest.main()
