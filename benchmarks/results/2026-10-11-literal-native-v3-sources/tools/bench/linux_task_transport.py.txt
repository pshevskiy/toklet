"""Bounded input-pipe capture for a future owned evaluator controller.

No container is created here. Callers must construct and inspect one fixed
Podman invocation; arbitrary program data belongs only in the stdin payload.
Owned container waiting/removal and VM restoration remain caller obligations.
"""
import base64
import hashlib
import os
import selectors
import subprocess
import time

import linux_code_isolation_probe as boundary
import linux_task_child as child


def capture_input(command, data):
    if not isinstance(data, bytes) or not 0 < len(data) <= child.INPUT_BYTES:
        raise ValueError("Evaluator input pipe exceeds its bound")
    started = time.monotonic()
    deadline = started + child.isolation.POLICY["wall_seconds"]
    output_bytes = child.isolation.POLICY["output_bytes"]
    selector = selectors.DefaultSelector()
    try:
        worker = subprocess.Popen(command, cwd=boundary.ROOT, env=None, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  start_new_session=True, close_fds=True)
    except BaseException:
        selector.close()
        raise
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    sent = 0
    cleanup_errors = []
    record = {"command": command, "wall_timed_out": False, "output_limit_exceeded": False,
              "input_bytes": len(data), "input_sha256": hashlib.sha256(data).hexdigest()}
    killed = False
    try:
        for stream, name, event in ((worker.stdout, "stdout", selectors.EVENT_READ),
                                    (worker.stderr, "stderr", selectors.EVENT_READ),
                                    (worker.stdin, "stdin", selectors.EVENT_WRITE)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, event, name)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                record["wall_timed_out"] = True
                break
            for key, _ in sorted(selector.select(min(0.05, remaining)),
                                 key=lambda item: (item[0].data == "stdin", item[0].data != "stdout")):
                if key.data == "stdin":
                    try:
                        sent += os.write(key.fileobj.fileno(), data[sent:sent + 4096])
                    except BlockingIOError:
                        continue
                    if sent == len(data):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()  # EOF, not an open interactive input stream.
                    continue
                try:
                    payload = os.read(key.fileobj.fileno(), 4096)
                except BlockingIOError:
                    continue
                if not payload:
                    selector.unregister(key.fileobj)
                    continue
                available = output_bytes - sum(map(len, captured.values()))
                captured[key.data].extend(payload[:available])
                if len(payload) > available:
                    record["output_limit_exceeded"] = True
                    break
            if record["output_limit_exceeded"]:
                break
        if record["wall_timed_out"] or record["output_limit_exceeded"]:
            killed = boundary.signal_owned_group(worker, cleanup_errors)
            worker.wait(timeout=20)
        else:
            try:
                worker.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                record["wall_timed_out"] = True
                killed = boundary.signal_owned_group(worker, cleanup_errors) or killed
                worker.wait(timeout=20)
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)[:4096]}
    finally:
        selector.close()
        killed = boundary.signal_owned_group(worker, cleanup_errors) or killed
        try:
            worker.wait(timeout=20)
        except BaseException as error:
            cleanup_errors.append({"type": type(error).__name__, "message": str(error)[:4096]})
        for stream in (worker.stdin, worker.stdout, worker.stderr):
            try:
                stream.close()
            except BaseException as error:
                cleanup_errors.append({"type": type(error).__name__, "message": str(error)[:4096]})
        try:
            surviving = boundary.live_group_members(worker.pid)
        except BaseException as error:
            surviving = None
            cleanup_errors.append({"type": type(error).__name__, "message": str(error)[:4096]})
        record.update(transport_exit_code=worker.returncode, process_group_killed=killed,
                      cleanup_errors=cleanup_errors, surviving_group_members=surviving,
                      input_sent_bytes=sent, input_completed=sent == len(data),
                      elapsed_seconds=time.monotonic() - started)
        for name, payload in captured.items():
            record[name + "_base64"] = base64.b64encode(payload).decode("ascii")
            record[name + "_captured_bytes"] = len(payload)
            record[name + "_captured_sha256"] = hashlib.sha256(payload).hexdigest()
    return record
