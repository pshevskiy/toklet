"""Fixed owned-container controller for bounded literal source execution.

VM admission/restoration, image/runtime identity and qualification belong to
an enclosing runtime window. This module never accepts candidate CLI commands.
"""
import copy
import json
import re
import time
import uuid

import linux_code_isolation_probe as boundary
import linux_task_evidence as evidence
import linux_task_transport as transport

PYTHON_PATH = '/usr/local/bin/python3'
BOOTSTRAP_ARGUMENT_BYTES = 128 * 1024 - 1
EXIT_WAIT_ATTEMPTS = 3
EXIT_WAIT_BACKOFF_SECONDS = 0.1


def bundled_source():
    # Only fixed trusted files become executable OCI arguments. Generated
    # program data arrives on bounded stdin and is compiled after policy entry.
    modules = [('code_isolation_child', 'code_isolation_child.py'),
               ('linux_code_isolation_child', 'linux_code_isolation_child.py'),
               ('linux_task_child', 'linux_task_child.py')]
    source = 'import sys, types\n'
    for name, filename in modules:
        text = (boundary.ROOT / 'tools/bench' / filename).read_text()
        source += (f'module = types.ModuleType({name!r})\n'
                   'sys.modules[module.__name__] = module\n'
                   f'exec(compile({text!r}, {("trusted_"+name)!r}, "exec"), module.__dict__)\n')
    return source + 'sys.exit(sys.modules["linux_task_child"].main())\n'


def admit_bootstrap(source):
    # Linux exec arguments include a terminating NUL and have a per-string
    # bound. Validate the fixed trusted bundle before any container operation.
    if (not isinstance(source, str) or source != bundled_source() or '\x00' in source
            or not 0 < len(source.encode('utf-8')) <= BOOTSTRAP_ARGUMENT_BYTES):
        raise ValueError('Literal evaluator trusted bootstrap identity or bound differs')


def create_command(podman, image_id, name, source):
    admit_bootstrap(source)
    # Keep the qualified resource/security declarations. The literal evaluator
    # changes only trusted bootstrap arguments and interactive input admission.
    command = boundary.create_command(podman, image_id, PYTHON_PATH, name, source, 'benign')
    assert command[-1] == 'benign'
    command.pop()
    command.insert(len(boundary.podman_prefix(podman)) + 1, '--interactive')
    return command


def validate_configuration(value, source):
    config = value.get('Config', {})
    if (config.get('OpenStdin') is not True or config.get('Tty') is not False
            or config.get('Cmd') != ['-I', '-S', '-B', '-c', source]):
        raise ValueError('Literal evaluator interactive input or bootstrap differs')
    # The shared validator checks every unchanged qualified security/resource
    # field. Verify the two differences above before adapting a private copy;
    # retain the original inspected configuration as execution evidence.
    adapted = copy.deepcopy(value)
    adapted['Config']['OpenStdin'] = False
    adapted['Config']['Cmd'].append('benign')
    summary = boundary.validate_configuration(adapted, PYTHON_PATH, source, 'benign')
    return {**summary, 'open_stdin': True, 'tty': False, 'literal_source_input': 'stdin'}


def wait_owned_exit(podman, identifier, name, image_id, record):
    # This follows execution termination; it never extends the six-second
    # candidate budget. Each control call already has its own 20-second bound.
    # Only the literal unknown status -1 may retry, at most three times. Never
    # normalize it, conceal a control failure or accept a nonnegative mismatch.
    for attempt in range(EXIT_WAIT_ATTEMPTS):
        waited = boundary.successful([*boundary.podman_prefix(podman), 'wait', '--condition=exited', identifier])
        record.setdefault('container_exit_waits', []).append(waited)
        record['container_exit_wait'] = waited
        text = boundary.output(waited).strip()
        unknown = text == '-1'
        if (waited['stderr_captured_bytes'] or not unknown
                and (re.fullmatch(r'[0-9]{1,3}', text) is None or int(text) > 255)):
            raise ValueError('Literal evaluator exit wait lacks completed status')
        stopped = boundary.inspect_owned(podman, identifier, name, image_id)
        state = stopped.get('State', {})
        record.setdefault('container_exit_inspections', []).append({
            'Id': stopped.get('Id'), 'Name': stopped.get('Name'), 'Image': stopped.get('Image'),
            'Labels': stopped.get('Config', {}).get('Labels'), 'State': copy.deepcopy(state)})
        if (state.get('Running') is not False or type(state.get('ExitCode')) is not int
                or not -1 <= state['ExitCode'] <= 255 or type(state.get('OOMKilled')) is not bool
                or state.get('Error') not in (None, '')):
            raise ValueError('Literal evaluator exit wait and inspection differ')
        if not unknown:
            if state['ExitCode'] != int(text):
                raise ValueError('Literal evaluator exit wait and inspection differ')
            return stopped
        if attempt + 1 < EXIT_WAIT_ATTEMPTS:
            time.sleep(EXIT_WAIT_BACKOFF_SECONDS)
    raise ValueError('Literal evaluator exit wait status remains unknown')


def execute(podman, image_id, source, payload):
    # Bound and bind program data before creating anything. A caller may not
    # serialize expressions or pass generated code in the OCI command vector.
    data = evidence.serialized_payload(payload)
    admit_bootstrap(source)
    name = 'toklet-authored-' + uuid.uuid4().hex
    if boundary.exists(podman, name):
        raise ValueError('Fresh evaluator container name unexpectedly exists')
    identifier = None
    record = {'container_name': name, 'container_cleanup_errors': []}
    try:
        created = boundary.successful(create_command(podman, image_id, name, source))
        record['container_create'] = created
        identifier = boundary.normalized_id(boundary.output(created).strip())
        record['container_id'] = identifier
        before = boundary.inspect_owned(podman, identifier, name, image_id)
        record['container_inspect_before'] = {key: before.get(key) for key in
            ['Id', 'Image', 'Config', 'HostConfig', 'Mounts', 'EffectiveCaps', 'BoundingCaps']}
        record['configuration'] = validate_configuration(before, source)
        record.update(transport.capture_input([*boundary.podman_prefix(podman), 'start',
                                              '--interactive', '--attach', identifier], data))
        stopped = boundary.inspect_owned(podman, identifier, name, image_id)
        record['container_state_after_attach'] = copy.deepcopy(stopped.get('State', {}))
        record['container_forced_kill'] = False
        if stopped.get('State', {}).get('Running') is True:
            record['container_kill'] = boundary.successful(
                [*boundary.podman_prefix(podman), 'kill', '--signal=KILL', identifier])
            record['container_forced_kill'] = True
        stopped = wait_owned_exit(podman, identifier, name, image_id, record)
        state = stopped['State']
        record['container_exit_code'] = state['ExitCode']
        record['container_state'] = {key: state.get(key) for key in
            ['Running', 'OOMKilled', 'ExitCode', 'Error', 'StartedAt', 'FinishedAt']}
        # Preserve status literally: arbitrary programs can deliberately exit
        # 137. Only independent transport watchdog/OOM evidence resolves it.
    except BaseException as error:
        record['failure'] = {'type': type(error).__name__, 'message': str(error)[:4096]}
        if isinstance(error, boundary.ControlError):
            record['failed_control'] = error.record
    finally:
        try:
            if identifier is None and boundary.exists(podman, name):
                inspected = json.loads(boundary.output(boundary.successful(
                    [*boundary.podman_prefix(podman), 'container', 'inspect', name])))
                if not isinstance(inspected, list) or len(inspected) != 1:
                    raise ValueError('Failed-create evaluator ownership differs')
                identifier = boundary.normalized_id(inspected[0].get('Id'))
                boundary.inspect_owned(podman, identifier, name, image_id)
            if identifier is not None:
                boundary.inspect_owned(podman, identifier, name, image_id)
                record['container_remove'] = boundary.successful(
                    [*boundary.podman_prefix(podman), 'rm', '--force', identifier])
                if boundary.exists(podman, identifier):
                    raise ValueError('Owned evaluator survives removal')
            if boundary.exists(podman, name):
                raise ValueError('Owned evaluator name survives cleanup')
            record['container_removed'] = True
        except BaseException as error:
            record['container_cleanup_errors'].append(
                {'type': type(error).__name__, 'message': str(error)[:4096]})
            if isinstance(error, boundary.ControlError):
                record['failed_cleanup_control'] = error.record
    return record
