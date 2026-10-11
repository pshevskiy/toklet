"""Mocked input-pipe lifecycle checks; never start or signal a real process."""
import base64
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import linux_task_transport as transport


class SelectorFixture:
    def __init__(self, plan):
        self.plan = list(plan)
        self.keys = {}
        self.closed = False

    def register(self, stream, _event, name):
        self.keys[name] = SimpleNamespace(fileobj=stream, data=name)

    def unregister(self, stream):
        for name, key in list(self.keys.items()):
            if key.fileobj is stream:
                del self.keys[name]
                return
        raise AssertionError("Mocked stream was not registered")

    def get_map(self):
        return self.keys

    def select(self, _timeout):
        if not self.plan:
            raise AssertionError("Mocked event inventory exhausted")
        return [(self.keys[name], 0) for name in self.plan.pop(0)]

    def close(self):
        self.closed = True
        self.keys.clear()


def worker_fixture(code=0):
    streams = {name: SimpleNamespace(fileno=Mock(return_value=fd), close=Mock())
               for name, fd in (("stdin", 10), ("stdout", 11), ("stderr", 12))}
    return SimpleNamespace(pid=424242, returncode=code, wait=Mock(return_value=code), **streams)


class InputTransportTests(unittest.TestCase):
    def run_mocked(self, plan, reads=(), writes=(3,), times=(0,), code=0,
                   worker=None, wait_failures=None, block_failures=None):
        selector = SelectorFixture(plan)
        worker = worker or worker_fixture(code)
        if wait_failures is not None:
            worker.wait.side_effect = wait_failures
        clock = list(times)

        def time_value():
            return clock.pop(0) if len(clock) > 1 else clock[0]

        with patch.object(transport.selectors, "DefaultSelector", return_value=selector), \
                patch.object(transport.subprocess, "Popen", return_value=worker) as spawn, \
                patch.object(transport.os, "set_blocking", side_effect=block_failures), \
                patch.object(transport.os, "read", side_effect=list(reads)), \
                patch.object(transport.os, "write", side_effect=list(writes)) as write, \
                patch.object(transport.time, "monotonic", side_effect=time_value), \
                patch.object(transport.boundary, "signal_owned_group", return_value=code == -9) as kill, \
                patch.object(transport.boundary, "live_group_members", return_value=[]):
            record = transport.capture_input(["authored-fixed-podman-command"], b"abc")
        self.assertTrue(selector.closed)
        self.assertEqual(record["surviving_group_members"], [])
        self.assertTrue(all(stream.close.called for stream in (worker.stdin, worker.stdout, worker.stderr)))
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        return record, worker, write, kill

    def test_partial_writes_and_eagain_finish_with_eof(self):
        record, worker, write, _ = self.run_mocked(
            [("stdin",), ("stdin",), ("stdin",), ("stdout", "stderr"), ("stdout",)],
            reads=(b"output", b"", b""), writes=(1, BlockingIOError(), 2))
        self.assertNotIn("failure", record)
        self.assertTrue(record["input_completed"])
        self.assertEqual(record["input_sent_bytes"], 3)
        self.assertEqual(base64.b64decode(record["stdout_base64"]), b"output")
        self.assertEqual([call.args[1] for call in write.call_args_list], [b"abc", b"bc", b"bc"])
        self.assertEqual(worker.stdin.close.call_count, 2)

    def test_broken_stdin_retains_failure_and_closes_owned_pipes(self):
        record, _, _, kill = self.run_mocked([("stdin",)], writes=(BrokenPipeError("authored"),))
        self.assertEqual(record["failure"]["type"], "BrokenPipeError")
        self.assertFalse(record["input_completed"])
        self.assertTrue(kill.called)

    def test_pipe_setup_failure_still_reaps_and_closes(self):
        record, worker, _, kill = self.run_mocked([], block_failures=(None, OSError("authored setup")))
        self.assertEqual(record["failure"]["type"], "OSError")
        self.assertTrue(worker.wait.called)
        self.assertTrue(kill.called)

    def test_wall_stop_reaps_under_cleanup_bound(self):
        record, worker, _, _ = self.run_mocked([], times=(0, 7, 8), code=-9)
        self.assertTrue(record["wall_timed_out"])
        self.assertTrue(record["process_group_killed"])
        self.assertEqual(worker.wait.call_args_list[0].kwargs, {"timeout": 20})

    def test_saturation_captures_exact_bound_before_killing(self):
        cap = transport.child.isolation.POLICY["output_bytes"]
        self.assertEqual(cap % 4096, 0)
        record, _, _, _ = self.run_mocked(
            [("stdin",)] + [("stdout",)] * (cap // 4096 + 1),
            reads=[b"x" * 4096] * (cap // 4096) + [b"x"], code=-9)
        self.assertTrue(record["output_limit_exceeded"])
        self.assertFalse(record["wall_timed_out"])
        self.assertEqual(record["stdout_captured_bytes"], cap)
        self.assertTrue(record["input_completed"])

    def test_interrupted_read_records_failure_and_still_cleans(self):
        record, _, _, kill = self.run_mocked([("stdout",)], reads=(InterruptedError("authored"),))
        self.assertEqual(record["failure"]["type"], "InterruptedError")
        self.assertTrue(kill.called)

    def test_cleanup_pipe_failure_is_retained(self):
        worker = worker_fixture()
        worker.stderr.close.side_effect = OSError("authored close")
        record, _, _, _ = self.run_mocked([("stdin",), ("stdout", "stderr")],
                                        reads=(b"", b""), worker=worker)
        self.assertEqual(record["cleanup_errors"][0]["type"], "OSError")

    def test_wait_timeout_is_killed_and_reaped_separately(self):
        exception = transport.subprocess.TimeoutExpired("authored", 6)
        record, worker, _, _ = self.run_mocked([("stdin",), ("stdout", "stderr")],
                                             reads=(b"", b""), code=-9,
                                             wait_failures=(exception, -9, -9))
        self.assertTrue(record["wall_timed_out"])
        self.assertEqual(worker.wait.call_args_list[1].kwargs, {"timeout": 20})

    def test_spawn_failure_closes_selector_without_signaling(self):
        selector = SelectorFixture([])
        with patch.object(transport.selectors, "DefaultSelector", return_value=selector), \
                patch.object(transport.subprocess, "Popen", side_effect=OSError("authored spawn")), \
                patch.object(transport.boundary, "signal_owned_group") as kill:
            with self.assertRaises(OSError):
                transport.capture_input(["authored-fixed-podman-command"], b"abc")
            kill.assert_not_called()
        self.assertTrue(selector.closed)

    def test_invalid_input_never_spawns(self):
        with patch.object(transport.subprocess, "Popen") as spawn:
            for data in (b"", "text", b"x" * (transport.child.INPUT_BYTES + 1)):
                with self.subTest(kind=type(data).__name__), self.assertRaises(ValueError):
                    transport.capture_input(["authored-fixed-podman-command"], data)
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
