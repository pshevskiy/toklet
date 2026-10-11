"""Trusted child preparation for literal source-test evaluation; not adopted.

The host must supply the literal task_cases.human_eval_program() result and
qualify runtime, ownership, transport, cleanup and restoration separately.
Container isolation bounds external effects; Python status events cannot
attest against deliberate manipulation of the shared grading interpreter.
"""
import hashlib
import importlib
import io
import json
import sys

import linux_code_isolation_child as isolation

PROGRAM_BYTES = 128 * 1024
INPUT_BYTES = 1024 * 1024
PRELOADS = (
    "math", "random", "typing", "collections", "collections.abc", "itertools",
    "functools", "heapq", "bisect", "re", "string", "statistics", "fractions",
    "decimal", "datetime", "copy", "operator", "json", "sys", "os", "resource",
    "ctypes", "io",
)
PROFILE = "toklet-linux-literal-source-tests-v1"


def read_payload(stream):
    """Read bounded source data from a binary input pipe, never an expression."""
    data = stream.read(INPUT_BYTES + 1)
    if not isinstance(data, bytes) or not 0 < len(data) <= INPUT_BYTES:
        raise ValueError("Serialized literal program exceeds its input bound")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Literal payload repeats a field")
            value[key] = item
        return value

    payload = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    admit_payload(payload)
    return payload


def admit_payload(payload):
    """Bound source data without importing, compiling or executing it."""
    if not isinstance(payload, dict) or set(payload) != {"case_id", "program", "program_sha256"}:
        raise ValueError("Literal program payload fields differ")
    case_id, program = payload["case_id"], payload["program"]
    if not isinstance(case_id, str) or not 0 < len(case_id.encode()) <= 256:
        raise ValueError("Literal program case identity exceeds its bound")
    if not isinstance(program, str):
        raise ValueError("Literal program must be source text")
    raw = program.encode("utf-8")
    if not 0 < len(raw) <= PROGRAM_BYTES:
        raise ValueError("Literal program source exceeds its bound")
    if hashlib.sha256(raw).hexdigest() != payload["program_sha256"]:
        raise ValueError("Literal program source identity differs")
    return {"case_id": case_id, "program_sha256": payload["program_sha256"], "program_bytes": len(raw)}


def preload():
    """Use a fixed compatibility inventory; caller source never selects it."""
    for name in PRELOADS:
        importlib.import_module(name)
    return list(PRELOADS)


def import_compatibility_failure(error):
    # ImportError can originate in candidate code too. Conservatively leave
    # such a case unresolved rather than attributing a harness limitation to
    # the model. No exceptions to the filesystem-denial policy are requested.
    if isinstance(error, ImportError):
        return True
    if not isinstance(error, PermissionError):
        return False
    trace = error.__traceback__
    while trace is not None:
        filename = trace.tb_frame.f_code.co_filename
        if filename.startswith("<frozen importlib."):
            return True
        trace = trace.tb_next
    return False


def evaluate(program):
    """Run the whole literal program, including its source-test invocation.

    No names from trusted preloading are injected into candidate globals.
    Compilation is inside the active boundary, and optimize=0 keeps asserts.
    A caught exit or exception never takes the normal completion path.
    """
    namespace = {"__name__": "__toklet_candidate__"}
    try:
        compiled = compile(program, "literal_task_program", "exec", optimize=0)
        exec(compiled, namespace)
    except BaseException as error:
        outcome = ("infrastructure_failure" if import_compatibility_failure(error) or isinstance(error, io.UnsupportedOperation)
                   else "resource_failure" if isinstance(error, MemoryError)
                   else "model_failure")
        return {"outcome": outcome, "normal_completion": False,
                "exception": type(error).__name__[:128]}
    return {"outcome": "passed", "normal_completion": True, "exception": None}


class FramedOutput(io.TextIOBase):
    """Frame ordinary Python output; the host bounds total serialized bytes."""

    def __init__(self, emit, channel):
        self.emit = emit
        self.channel = channel

    @property
    def encoding(self):
        return "utf-8"

    @property
    def buffer(self):
        raise io.UnsupportedOperation("Literal evaluator binary stream compatibility is not qualified")

    def writable(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return True

    def write(self, text):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if not isinstance(text, str):
            raise TypeError("Program output must be text")
        # Bounded frames avoid retaining a second full emitted string. Source
        # code can still use os.write directly; unexpected protocol bytes must
        # leave the host case unresolved, never count as a successful check.
        for offset in range(0, len(text), 1024):
            self.emit({"event": "program_output", "channel": self.channel,
                       "text": text[offset:offset + 1024]})
        return len(text)

    def flush(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return None

    def isatty(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return False


def main(payload=None):
    # Hold the controller stream outside the exec namespace. This protects
    # against accidental rebinding, not deliberate Python introspection.
    control_write, control_flush = sys.stdout.write, sys.stdout.flush
    serialize = json.dumps

    def emit(value):
        control_write(serialize(value, allow_nan=False) + "\n")
        control_flush()

    try:
        if payload is None:
            payload = read_payload(sys.stdin.buffer)
        binding = admit_payload(payload)
        modules = preload()
        removed = isolation.normalize_environment()
        limits, runtime = isolation.enter()
        if sys.flags.optimize != 0 or sys.flags.isolated != 1 or sys.flags.no_site != 1 or not sys.dont_write_bytecode:
            raise ValueError("Literal evaluator interpreter flags differ")
    except BaseException as error:
        emit({"event": "setup_error", "exception": type(error).__name__[:128]})
        return 2
    emit({"event": "isolated", "profile": PROFILE, "binding": binding,
          "policy": isolation.POLICY, "limits": limits, "runtime": runtime,
          "preloaded_modules": modules, "environment_keys": sorted(isolation.os.environ),
          "removed_environment_keys": removed, "python": sys.version,
          "isolated_flag": sys.flags.isolated, "no_site_flag": sys.flags.no_site,
          "optimize_flag": sys.flags.optimize, "dont_write_bytecode": sys.dont_write_bytecode})
    original_out, original_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = FramedOutput(emit, "stdout"), FramedOutput(emit, "stderr")
        result = evaluate(payload["program"])
    finally:
        sys.stdout, sys.stderr = original_out, original_err
    emit({"event": "program_result", "binding": binding, **result})
    return 0
