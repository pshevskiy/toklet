"""Authored capability probes only; never accept dataset or generated code."""
import ctypes
import errno
import json
import math
import mmap
import os
import resource
import socket
import sys
import time
import typing

POLICY = {
    "id": "toklet-authored-code-isolation-policy-v1",
    "api": "sandbox_init(SANDBOX_NAMED, kSBXProfilePureComputation)",
    "library": "/usr/lib/system/libsystem_sandbox.dylib",
    "data_bytes": 256 * 1024**2,
    "address_space_bytes": 1024**3,
    "cpu_seconds": 4,
    "wall_seconds": 6,
    "output_bytes": 65536,
    "core_bytes": 0,
    "open_files": 64,
    "network": "denied",
    "filesystem": "denied after trusted initialization",
    "child_processes": "denied",
}
MODES = (
    "benign", "assertion_failure", "early_exit", "file_read", "file_write",
    "network", "fork", "exec", "stdout_flood", "stderr_flood", "cpu_timeout",
    "wall_timeout", "data_limit", "address_space_limit",
)


def emit(value):
    # Parent bounds both inherited pipes together. All qualification records
    # originate in this trusted static program, before any future code runner.
    print(json.dumps(value, allow_nan=False), flush=True)


def limits():
    return {
        "RLIMIT_DATA": (resource.RLIMIT_DATA, POLICY["data_bytes"]),
        "RLIMIT_AS": (resource.RLIMIT_AS, POLICY["address_space_bytes"]),
        "RLIMIT_CPU": (resource.RLIMIT_CPU, POLICY["cpu_seconds"]),
        "RLIMIT_CORE": (resource.RLIMIT_CORE, POLICY["core_bytes"]),
        "RLIMIT_NOFILE": (resource.RLIMIT_NOFILE, POLICY["open_files"]),
    }


def enter():
    """Use the SDK's named-profile API; never use reserved custom flags."""
    # The installed SDK's export stub assigns these public legacy symbols to
    # libsystem_sandbox, distinct from libsandbox's profile compilation API.
    library = ctypes.CDLL(POLICY["library"])
    initialize = library.sandbox_init
    initialize.argtypes = (ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p))
    initialize.restype = ctypes.c_int
    release = library.sandbox_free_error
    release.argtypes = (ctypes.c_void_p,)
    release.restype = None
    # The exported const char[] is a static C profile name. Keep the library
    # owner alive while borrowing its first byte and NUL-terminated name.
    first = ctypes.c_char.in_dll(library, "kSBXProfilePureComputation")
    profile = ctypes.cast(ctypes.addressof(first), ctypes.c_char_p)
    name = profile.value.decode("ascii")
    admitted = {}
    for label, (key, value) in limits().items():
        try:
            resource.setrlimit(key, (value, value))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Resource admission failed for {label}: {error}") from error
        pair = resource.getrlimit(key)
        if pair != (value, value):
            raise ValueError(f"Resource admission differs: {label}")
        admitted[label] = list(pair)
    error = ctypes.c_void_p()
    result = initialize(profile, 1, ctypes.byref(error))
    try:
        # Only the API's owned error string is dereferenced; free exactly once
        # with its documented allocator. No user pointer enters this call.
        message = ctypes.string_at(error.value).decode("utf-8", errors="replace")[:4096] if error.value else None
    finally:
        if error.value:
            release(error)
    if result != 0 or message is not None:
        raise RuntimeError(f"Named sandbox admission failed: {result}, {message}")
    return name, admitted


def denial(error):
    return {"denied": error.errno in (errno.EPERM, errno.EACCES), "errno": error.errno,
            "exception": type(error).__name__}


def denied(operation):
    try:
        operation()
    except OSError as error:
        return denial(error)
    return {"denied": False, "errno": None, "exception": None}


def probe(mode, read_path, write_path):
    if mode == "benign":
        # Test definition plus explicit invocation and post-return evidence.
        def candidate(values: typing.List[int]):
            return sum(values)

        def check(function):
            assert function([1, 2, 3]) == 6
            assert function([]) == 0
            assert math.sqrt(81) == 9

        check(candidate)
        return {"check_returned": True}
    if mode == "assertion_failure":
        try:
            assert 1 == 2
        except AssertionError:
            return {"check_returned": False, "exception": "AssertionError"}
        raise RuntimeError("Authored assertion unexpectedly passed")
    if mode == "early_exit":
        os._exit(0)
    if mode == "file_read":
        try:
            stream = open(read_path, "rb")
        except OSError as error:
            return denial(error)
        try:
            return denied(lambda: stream.read(1))
        finally:
            # A close failure must not be mistaken for denied reading.
            stream.close()
    if mode == "file_write":
        def write():
            with open(write_path, "xb") as stream:
                stream.write(b"OWNED ISOLATION PROBE ONLY")
        return denied(write)
    if mode == "network":
        try:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except OSError as error:
            return denial(error)
        try:
            try:
                handle.settimeout(1)
            except OSError as error:
                return {"denied": False, "setup_error": type(error).__name__}
            # ECONNREFUSED / timeout do not prove isolation; neither does a
            # failed timeout configuration when connect was never attempted.
            return denied(lambda: handle.connect(("127.0.0.1", 9)))
        finally:
            handle.close()
    if mode == "fork":
        try:
            child = os.fork()
        except OSError as error:
            return denial(error)
        if child == 0:
            os._exit(97)
        result = {"denied": False, "child_pid": child}
        try:
            os.waitpid(child, 0)
        except OSError as error:
            result["wait_error"] = type(error).__name__
        return result
    if mode == "exec":
        return denied(lambda: os.execv("/usr/bin/true", ["true"]))
    if mode in ("stdout_flood", "stderr_flood"):
        descriptor = 1 if mode == "stdout_flood" else 2
        while True:
            os.write(descriptor, b"x" * 4096)
    if mode == "cpu_timeout":
        while True:
            pass
    if mode == "wall_timeout":
        time.sleep(60)
        return {"unexpected_return": True}
    if mode == "data_limit":
        # This request is below the 1 GiB address limit but above RLIMIT_DATA.
        # Successful small allocation demonstrates allocation remains usable.
        small = bytearray(1024)
        assert len(small) == 1024
        try:
            large = bytearray(512 * 1024**2)
        except MemoryError:
            return {"bounded": True, "exception": "MemoryError", "requested_bytes": 512 * 1024**2}
        return {"bounded": False, "allocated_bytes": len(large)}
    if mode == "address_space_limit":
        try:
            handle = mmap.mmap(-1, 2 * 1024**3)
        except OSError as error:
            return {"bounded": error.errno == errno.ENOMEM, "exception": type(error).__name__,
                    "errno": error.errno, "requested_bytes": 2 * 1024**3}
        handle.close()
        return {"bounded": False}
    raise ValueError("Unsupported authored isolation mode")


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in MODES:
        raise ValueError("Only fixed authored isolation probes are admitted")
    mode, read_path, write_path = sys.argv[1:]
    try:
        profile, admitted = enter()
    except BaseException as error:
        emit({"event": "setup_error", "type": type(error).__name__, "message": str(error)[:4096]})
        return 2
    emit({"event": "isolated", "mode": mode, "policy": POLICY, "profile": profile,
          "limits": admitted, "python": sys.version, "isolated_flag": sys.flags.isolated,
          "no_site_flag": sys.flags.no_site, "optimize_flag": sys.flags.optimize,
          "dont_write_bytecode": sys.dont_write_bytecode,
          "environment_keys": sorted(os.environ)})
    try:
        result = probe(mode, read_path, write_path)
    except BaseException as error:
        emit({"event": "probe_error", "mode": mode, "type": type(error).__name__, "message": str(error)[:4096]})
        return 3
    emit({"event": "probe_result", "mode": mode, "result": result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
