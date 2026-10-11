"""Bound the fixed authored VM startup while preserving its successful helpers.

The finite-command capture helper intentionally kills surviving descendants.
VM startup instead owns persistent helpers until an explicit machine stop.
Callers must verify initial machine state, API readiness and final restoration.
This module neither pulls images nor runs caller-supplied programs.
"""
import base64
import hashlib
import os
import selectors
import subprocess
import threading
import time

from linux_code_isolation_probe import live_group_members, signal_owned_group

MACHINE = "podman-machine-default"


def await_stopped_owner(owner, observe, pause=None):
    """Retain helper readers through cleanup, even if evidence publication fails.

    The internal observer supplies machine inventory only. Failed observations
    retain one bounded diagnostic and never release an unverified owner.
    """
    if owner is None:
        return
    if pause is None:
        pause = time.sleep
    while not owner.record.get("helper_pipe_lifetime_completed"):
        owner.record["restoration_observations"] = owner.record.get("restoration_observations", 0) + 1
        try:
            owner.finish(observe())
        except BaseException as error:
            owner.record["last_restoration_failure"] = {"type": type(error).__name__, "message": str(error)[:4096]}
            pause(30)


class _HelperPipes:
    """Keep inherited pipe writers usable until verified machine shutdown."""
    def __init__(self, worker, selector, captured, output_bytes, record):
        self.worker = worker
        self.selector = selector
        self.captured = captured
        self.output_bytes = output_bytes
        self.record = record
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.pump, daemon=True)
        self.thread_started = False
        self.discarded_bytes = 0

    def drain(self, timeout):
        ready = self.selector.select(timeout)
        for key, _ in ready:
            payload = os.read(key.fileobj.fileno(), 4096)
            if not payload:
                self.selector.unregister(key.fileobj)
                continue
            available = self.output_bytes - sum(len(value) for value in self.captured.values())
            self.captured[key.data].extend(payload[:available])
            self.discarded_bytes += max(0, len(payload) - available)
        return bool(ready)

    def pump(self):
        try:
            while not self.stop.is_set():
                if self.selector.get_map():
                    self.drain(0.05)
                else:
                    self.stop.wait(0.05)
        except BaseException as error:
            self.record["helper_capture_failure"] = {"type": type(error).__name__, "message": str(error)}

    def finish(self, machines):
        matches = [item for item in machines if item.get("Name") == MACHINE]
        if (len(matches) != 1 or matches[0].get("Running") is not False
                or matches[0].get("Starting") is not False):
            raise ValueError("Owned machine shutdown is not verified")
        if live_group_members(self.worker.pid) != []:
            raise ValueError("Startup helper group remains live after machine stop")
        self.stop.set()
        if self.thread_started:
            self.thread.join(timeout=20)
            if self.thread.is_alive():
                raise RuntimeError("Helper pipe capture did not stop")
        try:
            while self.selector.get_map() and self.drain(0):
                pass
        except BaseException as error:
            self.record.setdefault("helper_capture_failure", {"type": type(error).__name__, "message": str(error)})
        finally:
            self.selector.close()
            for stream in [self.worker.stdout, self.worker.stderr]:
                stream.close()
        self.record["helper_output_discarded_bytes"] = self.discarded_bytes
        self.record["helper_pipe_lifetime_completed"] = True
        for name, payload in self.captured.items():
            self.record[name + "_base64"] = base64.b64encode(payload).decode("ascii")
            self.record[name + "_captured_bytes"] = len(payload)
            self.record[name + "_captured_sha256"] = hashlib.sha256(payload).hexdigest()
        return self.record


def start_machine(podman, wall_seconds=120, output_bytes=256 * 1024):
    """Return (record, pipe owner); explicit verified stop must finish the owner."""
    if wall_seconds != 120 or output_bytes != 256 * 1024:
        raise ValueError("Authored startup limits are fixed")
    command = [podman, "machine", "start", MACHINE]
    started = time.monotonic()
    deadline = started + wall_seconds
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    record = {"command": command, "wall_timed_out": False,
              "output_limit_exceeded": False, "helper_lifetime": "until explicit owned machine stop"}
    cleanup_errors = []
    selector = selectors.DefaultSelector()
    worker = None
    killed = False
    try:
        worker = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  start_new_session=True, close_fds=True)
        record["startup_process_group"] = worker.pid
        for stream, name in [(worker.stdout, "stdout"), (worker.stderr, "stderr")]:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                record["wall_timed_out"] = True
                break
            # Helpers may retain inherited pipe writers. After the direct
            # client exits, drain immediately available bytes without waiting
            # for daemon EOF. Keep readers drained for the whole VM lifetime.
            exited = worker.poll() is not None
            ready = selector.select(0 if exited else min(0.05, remaining)) if selector.get_map() else []
            if exited and not ready:
                break
            if not selector.get_map():
                worker.wait(timeout=remaining)
                break
            for key, _ in sorted(ready, key=lambda item: item[0].data != "stdout"):
                payload = os.read(key.fileobj.fileno(), 4096)
                if not payload:
                    selector.unregister(key.fileobj)
                    continue
                available = output_bytes - sum(len(value) for value in captured.values())
                captured[key.data].extend(payload[:available])
                if len(payload) > available:
                    record["output_limit_exceeded"] = True
                    break
            if record["output_limit_exceeded"]:
                break
        if record["wall_timed_out"] or record["output_limit_exceeded"]:
            killed = signal_owned_group(worker, cleanup_errors)
        worker.wait(timeout=20 if killed else max(0.001, deadline - time.monotonic()))
    except BaseException as error:
        if isinstance(error, subprocess.TimeoutExpired):
            record["wall_timed_out"] = True
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        if worker is not None and worker.poll() is None:
            killed = signal_owned_group(worker, cleanup_errors) or killed
    finally:
        if worker is not None:
            try:
                worker.wait(timeout=20)
            except BaseException as error:
                cleanup_errors.append({"type": type(error).__name__, "message": str(error)})
            try:
                record["live_startup_group_members"] = live_group_members(worker.pid)
            except BaseException as error:
                record["live_startup_group_members"] = None
                cleanup_errors.append({"type": type(error).__name__, "message": str(error)})
            record["exit_code"] = worker.returncode
        else:
            record["exit_code"] = None
            record["live_startup_group_members"] = None
        record.update(process_group_killed=killed, cleanup_errors=cleanup_errors,
                      elapsed_seconds=time.monotonic() - started)
        record["client_succeeded"] = (record["exit_code"] == 0 and not record.get("failure")
                                      and not record["wall_timed_out"] and not record["output_limit_exceeded"]
                                      and not cleanup_errors)
        record["machine_restoration_required"] = worker is not None
        for name, payload in captured.items():
            record[name + "_base64"] = base64.b64encode(payload).decode("ascii")
            record[name + "_captured_bytes"] = len(payload)
            record[name + "_captured_sha256"] = hashlib.sha256(payload).hexdigest()
    if worker is None:
        selector.close()
        plumbing = None
    else:
        plumbing = _HelperPipes(worker, selector, captured, output_bytes, record)
        try:
            plumbing.thread.start()
            plumbing.thread_started = True
        except BaseException as error:
            record["helper_capture_failure"] = {"type": type(error).__name__, "message": str(error)}
            record["client_succeeded"] = False
    return record, plumbing
