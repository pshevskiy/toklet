"""Replay of precise authored native expectations."""
import base64

import linux_task_evidence as evidence


def check(record, case, runtime, python_version):
    expected = case['expectation']
    error = None
    try:
        resolved = evidence.classify(record, case['payload'], runtime, python_version)
    except ValueError as failure:
        # These specific messages arise only after the shared classifier's
        # input/admission/capture/termination/cleanup validation. A setup,
        # malformed admission or cleanup failure must never satisfy a fixture
        # intended to demonstrate an unresolved candidate termination/import.
        error = str(failure)
        if expected['outcome'] != 'infrastructure_failure' or error not in expected.get('classifier_errors', []):
            raise
        resolved = {'outcome': 'infrastructure_failure', 'passed': None,
                    'classifier_error': error}
    if resolved['outcome'] != expected['outcome']:
        raise ValueError('Authored evaluator outcome differs')
    if expected['outcome'] == 'infrastructure_failure':
        if error is None:
            raise ValueError('Authored unresolved case unexpectedly resolved')
    elif resolved.get('reason') != expected['classifier_reason']:
        raise ValueError('Authored evaluator classifier reason differs')
    state = record['container_state']
    if (state['ExitCode'] not in expected['container_exit_codes']
            or record['wall_timed_out'] is not expected['wall_timed_out']
            or record['output_limit_exceeded'] is not expected['output_limit_exceeded']
            or state['OOMKilled'] is not expected['oom_killed']):
        raise ValueError('Authored evaluator independent resource/status evidence differs')
    for flag in ('process_group_killed', 'container_forced_kill'):
        if expected[flag] is not None and record[flag] is not expected[flag]:
            raise ValueError('Authored evaluator forced termination evidence differs')
    if 'captured_total_bytes' in expected:
        if record['stdout_captured_bytes'] + record['stderr_captured_bytes'] != expected['captured_total_bytes']:
            raise ValueError('Authored evaluator output saturation differs')
    stderr = base64.b64decode(record['stderr_base64'], validate=True)
    if len(stderr) != expected['unframed_stderr_bytes']:
        raise ValueError('Authored evaluator unexpected controller stderr')
    terminal_expected = expected['terminal_event']
    raw = base64.b64decode(record['stdout_base64'], validate=True)
    if expected['admission_only']:
        events = [evidence.decode_event(line) for line in raw.splitlines()]
        if len(events) != 1 or events[0].get('event') != 'isolated':
            raise ValueError('Authored evaluator emitted unexpected post-admission output')
    if terminal_expected is not None:
        events = [evidence.decode_event(line) for line in raw.splitlines()]
        terminals = [event for event in events if event.get('event') == 'program_result']
        if terminal_expected:
            if len(terminals) != 1:
                raise ValueError('Authored evaluator lacks one observed terminal result')
            terminal = terminals[0]
            if (terminal['normal_completion'] is not expected['normal_completion']
                    or terminal['exception'] != expected['exception']):
                raise ValueError('Authored evaluator exception or normal completion differs')
        elif terminals:
            raise ValueError('Authored premature exit unexpectedly emitted a terminal event')
    if 'program_output' in expected:
        events = [evidence.decode_event(line) for line in raw.splitlines()]
        observed = {channel: ''.join(event['text'] for event in events
                                     if event.get('event') == 'program_output'
                                     and event['channel'] == channel)
                    for channel in ('stdout', 'stderr')}
        if observed != expected['program_output']:
            raise ValueError('Authored evaluator framed program output differs')
    if expected.get('malformed_raw_terminal'):
        events = [evidence.decode_event(line) for line in raw.splitlines()]
        malformed = [event for event in events if event.get('event') == 'program_result'
                     and set(event) == {'event', 'outcome'} and event['outcome'] == 'passed']
        if (len(malformed) != 1 or len(events) != 3 or events[0].get('event') != 'isolated'
                or events[1] != malformed[0]
                or evidence.canonical(events[2]) != evidence.canonical({'event': 'program_result',
                                  'binding': evidence.child.admit_payload(case['payload']),
                                  'outcome': 'passed', 'normal_completion': True, 'exception': None})):
            raise ValueError('Authored malformed terminal fixture was not observed')
    return resolved
