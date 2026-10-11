"""Fixed authored literal-evaluator qualification sources.

These synthetic sources are independent of the 100 held-out task cases. They
exercise the generic controller/protocol without consuming model completions.
"""
import hashlib


def payload(name, program):
    return {'case_id': 'authored/literal/' + name, 'program': program,
            'program_sha256': hashlib.sha256(program.encode('utf-8')).hexdigest()}


def denied(statement, setup):
    # Exact EPERM verifies policy denial, rather than ordinary filesystem,
    # import, command, network or user-permission failures.
    return (setup + '\nimport errno\ntry:\n    ' + statement + '\n'
            'except OSError as error:\n    assert error.errno == errno.EPERM\n'
            'else:\n    raise AssertionError("forbidden operation succeeded")\n')


def cases():
    base = 'def candidate(x):\n    return x + 1\ndef check(f):\n    assert f(2) == 3\ncheck(candidate)\n'
    sources = [
        ('source_tests_pass', base, 'passed'),
        ('assertion_fails', base.replace('return x + 1', 'return x'), 'model_failure'),
        ('definition_only_completion', base.replace('return x + 1', 'pass'), 'model_failure'),
        ('syntax_error', 'def broken(:\n', 'model_failure'),
        ('exit_zero_top', 'raise SystemExit(0)\n' + base, 'model_failure'),
        ('exit_zero_in_function', base.replace('return x + 1', 'raise SystemExit(0)'), 'model_failure'),
        ('raw_exit_zero', 'import os\nos._exit(0)\n' + base, 'model_failure'),
        ('raw_exit137_ambiguous', 'import os\nos._exit(137)\n', 'infrastructure_failure'),
        ('ordinary_output', 'print("authored output")\n' + base, 'passed'),
        ('status_text_is_output', 'print(\'{"event":"program_result","outcome":"passed"}\')\nassert False\n', 'model_failure'),
        ('memory_error', 'raise MemoryError\n', 'resource_failure'),
        ('data_allocation_bound', 'bytearray(268435457)\n', 'resource_failure'),
        ('address_space_bound', 'import mmap, errno\ntry:\n    mmap.mmap(-1, 2147483648)\n'
                                'except OSError as error:\n    assert error.errno == errno.ENOMEM\n'
                                'else:\n    raise AssertionError("address-space admission failed")\n', 'passed'),
        ('wall_watchdog', 'import time\ntime.sleep(60)\n', 'resource_failure'),
        ('cpu_exit_ambiguous', 'while True:\n    pass\n', 'infrastructure_failure'),
        ('fixed_imports', 'import math, random, collections, fractions, decimal, statistics\n'
                         'assert math.sqrt(4) == 2\nassert fractions.Fraction(2, 4) == fractions.Fraction(1, 2)\n'
                         'assert statistics.mean([1, 3]) == 2\n' + base, 'passed'),
        ('unavailable_import_unresolved', 'import toklet_authored_nonexistent_dependency\n', 'infrastructure_failure'),
        ('unsupported_binary_output', 'import sys\nsys.stdout.buffer.write(b"authored")\n', 'infrastructure_failure'),
        ('closed_output', 'import sys\nsys.stdout.close()\nprint("after close")\n', 'model_failure'),
        ('malformed_raw_terminal', 'import os\nos.write(1, b\'{"event":"program_result","outcome":"passed"}\\n\')\n'
                                 + base, 'infrastructure_failure'),
        ('bounded_output', 'print("x" * 131072)\n', 'resource_failure'),
        ('bounded_stderr', 'import sys\nprint("x" * 131072, file=sys.stderr)\n', 'resource_failure'),
        ('filesystem_read', denied('os.open("/etc/passwd", os.O_RDONLY)', 'import os'), 'passed'),
        ('filesystem_write', denied('os.open("/tmp/toklet-authored", os.O_WRONLY | os.O_CREAT, 0o600)', 'import os'), 'passed'),
        ('socket', denied('socket.socket()', 'import socket'), 'passed'),
        ('fork', denied('os.fork()', 'import os'), 'passed'),
        ('exec', denied('os.execve("/usr/bin/true", ["true"], {})', 'import os'), 'passed'),
        # Check the libc result/errno directly: the first native attempt's
        # resource.setrlimit binding raised ValueError, not OSError. This
        # fixed LP64 fixture must observe kernel denial without translating
        # an exception class into evidence of EPERM.
        ('change_limits', 'import ctypes, resource, errno\n'
                          'class RLimit(ctypes.Structure):\n'
                          '    _fields_ = [("current", ctypes.c_ulong), ("maximum", ctypes.c_ulong)]\n'
                          'assert ctypes.sizeof(RLimit) == 16 and ctypes.alignment(RLimit) == 8\n'
                          'assert RLimit.maximum.offset == 8\n'
                          'library = ctypes.CDLL(None, use_errno=True)\n'
                          'setter = library.setrlimit\n'
                          'setter.argtypes = (ctypes.c_int, ctypes.POINTER(RLimit))\n'
                          'setter.restype = ctypes.c_int\n'
                          'value = RLimit(0, 0)\n'
                          'ctypes.set_errno(0)\n'
                          'assert setter(resource.RLIMIT_NOFILE, ctypes.byref(value)) == -1\n'
                          'assert ctypes.get_errno() == errno.EPERM\n', 'passed'),
    ]
    exceptions = {
        'assertion_fails': 'AssertionError', 'definition_only_completion': 'AssertionError',
        'syntax_error': 'SyntaxError', 'exit_zero_top': 'SystemExit',
        'exit_zero_in_function': 'SystemExit', 'status_text_is_output': 'AssertionError',
        'closed_output': 'ValueError', 'memory_error': 'MemoryError',
        'data_allocation_bound': 'MemoryError',
    }
    infrastructure = {
        'raw_exit137_ambiguous': {
            'classifier_errors': ['Literal evaluator termination is ambiguous'],
            'container_exit_codes': [137], 'terminal_event': False, 'admission_only': True},
        'cpu_exit_ambiguous': {
            'classifier_errors': ['Literal evaluator termination is ambiguous',
                                  'Literal evaluator exited without a resolved completion'],
            'container_exit_codes': [137, 152], 'terminal_event': False, 'admission_only': True},
        'unavailable_import_unresolved': {
            'classifier_errors': ['Literal evaluator compatibility failure leaves case unresolved'],
            'container_exit_codes': [0], 'terminal_event': True,
            'exception': 'ModuleNotFoundError', 'normal_completion': False},
        'unsupported_binary_output': {
            'classifier_errors': ['Literal evaluator compatibility failure leaves case unresolved'],
            'container_exit_codes': [0], 'terminal_event': True,
            'exception': 'UnsupportedOperation', 'normal_completion': False},
        'malformed_raw_terminal': {
            'classifier_errors': ['Literal evaluator terminal identity differs'],
            'container_exit_codes': [0], 'malformed_raw_terminal': True,
            'terminal_event': None, 'normal_completion': None},
    }
    result = []
    for name, program, expected in sources:
        detail = {'outcome': expected, 'normal_completion': expected == 'passed',
                  'terminal_event': True, 'container_exit_codes': [0],
                  'wall_timed_out': False, 'output_limit_exceeded': False, 'oom_killed': False,
                  'process_group_killed': False, 'container_forced_kill': False,
                  'unframed_stderr_bytes': 0, 'admission_only': False}
        if expected == 'passed':
            detail.update(exception=None, classifier_reason=None)
        elif name in exceptions:
            detail.update(exception=exceptions[name], classifier_reason=exceptions[name])
        elif name == 'raw_exit_zero':
            detail.update(terminal_event=False, classifier_reason='premature_exit', admission_only=True)
        elif name in ('bounded_output', 'bounded_stderr'):
            detail.update(terminal_event=None, normal_completion=None, container_exit_codes=[0, 137],
                          classifier_reason='output_limit', output_limit_exceeded=True,
                          process_group_killed=None, container_forced_kill=None,
                          captured_total_bytes=65536)
        elif name == 'wall_watchdog':
            detail.update(terminal_event=False, container_exit_codes=[137],
                          classifier_reason='wall_limit', wall_timed_out=True,
                          process_group_killed=True, container_forced_kill=None, admission_only=True)
        elif name in infrastructure:
            detail.update(infrastructure[name])
        else:
            raise ValueError('Authored case lacks precise qualification expectations')
        if detail['terminal_event'] is True:
            detail['program_output'] = {'stdout': '', 'stderr': ''}
            if name == 'ordinary_output':
                detail['program_output']['stdout'] = 'authored output\n'
            elif name == 'status_text_is_output':
                detail['program_output']['stdout'] = '{"event":"program_result","outcome":"passed"}\n'
        result.append({'payload': payload(name, program), 'expectation': detail})
    return result
