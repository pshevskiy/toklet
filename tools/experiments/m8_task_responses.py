"""Post-exit admission/decoding of frozen native-app tasks; never executes responses."""
import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import signal
import sys

import m7_physical_tasks as tasks
import m8_physical_mlx as transport

ROOT = transport.ROOT
package = transport.package
physical = transport.physical
QUAL = 'benchmarks/results/2026-10-22-native-mlx8-full-task-collection-v1.json'
QUAL_SHA = 'a76eb73788a6748d2598e67b813fe9269ce951c49697d25865b40ece8e9a97dc'
TOKENIZER = 'models/Qwen2.5-0.5B-Instruct/tokenizer.json'
TOKENIZER_SHA = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
PLATFORMS = ('mac', 'iphone14pro')
MARKERS = ('TOKLET_INSTALLATION_JSON=', 'TOKLET_TASK_CHUNK_JSON=',
           'TOKLET_TASK_CASE_JSON=', 'TOKLET_TASKS_JSON=')


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def require(value, message):
    if not value:
        raise ValueError(message)


def merge(pins, extra):
    transport.integration.merge(pins, extra)


def capture(path, pins, expected=None, maximum=32*1024**2):
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(maximum+1)
    digest = hashlib.sha256(raw).hexdigest()
    require(len(raw) <= maximum and (expected is None or digest == expected),
            'Bounded consumed input differs: '+str(path))
    merge(pins, {str(path.relative_to(ROOT)): digest})
    return raw


def guard(pins):
    tasks.measurement.verify_inputs(pins)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def raw_derivative(text):
    """Explicit app-marker derivative preserving each original JSON substring."""
    rows = []
    for line in text.splitlines():
        for marker in MARKERS:
            if line.startswith(marker):
                rows.append(encoded(dict(marker=marker[:-1], original_json=line[len(marker):])))
                break
        else:
            require(not line.startswith('TOKLET_'), 'Unexpected app marker')
    require(bool(rows), 'No app task events')
    return b''.join(rows)


def admit_source(pins):
    q = json.loads(capture(ROOT/QUAL, pins, QUAL_SHA))
    require(q['passed'] is True and q['task_quality_qualified'] is False,
            'Exact accepted raw collection required')
    merge(pins, q['source_pins'])
    execution = q['execution']
    stage_ref = execution['stages']['qwen8-tasks-full']
    stage_path = ROOT/stage_ref['folder']/'staged.json'
    stage = json.loads(capture(stage_path, pins, stage_ref['sha256']))
    config = stage['config']
    task_set = json.loads(capture(stage_path.parent/'transfer'/config['tasksFile'],
                                 pins, config['tasksSHA256'], 512*1024))
    cases, extra = tasks.frozen_tasks()
    merge(pins, extra)
    require(len(cases) == 100 and task_set == dict(version=1, vocabulary=151936,
        stops=[151645, 151643], cases=[dict(id=c['id'], promptTokens=c['prompt_tokens'],
        maximum=c['max_output_tokens']) for c in cases]), 'Original full tasks required')
    driver = ROOT/'artifacts/m8-mlx8-full-task-driver-v1'
    packets = {}
    for platform in PLATFORMS:
        if platform == 'mac':
            paths = [driver/('mac-tasks-full.'+s) for s in ('stdout', 'stderr')]
            saved = execution['mac_checked']
            signed = None
        else:
            phone = execution['nested_receipts']['qwen8-tasks-full-run']
            paths = [driver/'qwen8-tasks-full-run/private-logs'/('measurement.'+s)
                     for s in ('stdout', 'stderr')]
            signed = phone['signed_bundle']
            saved = dict(phone['checked'])
            launch = saved.pop('launch')
            require(launch['executable_matches'] is True and launch['reported_exit_code'] == 0
                    and launch['reported_core_dump'] is False, 'Phone collection did not exit')
        text = '\n'.join(capture(path, pins, pins[str(path.relative_to(ROOT))],
                                 64*1024**2).decode() for path in paths)
        checked = tasks.check_console(text, config, signed, task_set,
                                      require_installation=platform != 'mac')
        require(checked == saved and checked['case_count'] == 100, 'Full source replay differs')
        packets[platform] = dict(checked=checked, raw=raw_derivative(text))
    commands = [*execution['commands']]
    for data in execution['nested_receipts'].values():
        commands.extend(data['commands'])
    require(all(c['exit_code'] == 0 and not c['primary_error'] and not c['cleanup_errors']
                and type(c['pid']) is int and c['pid'] > 1 for c in commands),
            'Naturally exited owned local producers required')
    # These are host controller/runner PIDs. Never insert remote launch PIDs here.
    pids = sorted({c['pid'] for c in commands})
    guard(pins)
    return q, cases, packets, pids


def decode_responses(cases, checked, decoder):
    require(len(cases) == len(checked['responses']) == 100, 'Complete response inventory required')
    result = []
    for case, sample in zip(cases, checked['responses'], strict=True):
        row = sample['response']
        require(sample['caseID'] == case['id'] and type(row['reason']) is int and row['reason'] in (1, 2)
                and all(type(t) is int and 0 <= t < 151936 for t in row['ids']),
                'Literal ordered response differs')
        require((row['reason'] == 1 and row.get('stopToken') is None) or
                (row['reason'] == 2 and type(row.get('stopToken')) is int
                 and row['stopToken'] in (151645, 151643)), 'Literal hidden stop differs')
        text = decoder.decode(row['ids'], skip_special_tokens=True)
        require(type(text) is str, 'Decoder must return literal text')
        reason = dict(kind='output_limit') if row['reason'] == 1 else dict(
            kind='stop_token', token_id=row['stopToken'])
        result.append(dict(id=case['id'], category=case['category'],
            visible_token_ids=row['ids'], visible_text=text,
            visible_text_sha256=hashlib.sha256(text.encode()).hexdigest(), stop_reason=reason,
            state={k: row[k] for k in ('sampled', 'history', 'evaluated', 'pending', 'chunks')}))
    return result


def tokenizer_inputs(pins):
    raw = capture(ROOT/TOKENIZER, pins, TOKENIZER_SHA, 16*1024**2)
    spec = importlib.util.find_spec('tokenizers')
    require(spec is not None and spec.origin is not None, 'Installed tokenizer runtime required')
    folder = Path(spec.origin).resolve().parent
    require(folder.is_relative_to(ROOT), 'Repository-local tokenizer runtime required')
    runtime = {}
    for path in sorted(folder.rglob('*')):
        if '__pycache__' in path.parts or not path.is_file():
            continue
        require(not path.is_symlink(), 'Tokenizer runtime symlink rejected')
        runtime[str(path.relative_to(ROOT))] = package.sha(path)
    require(any(name.endswith('.so') for name in runtime), 'Tokenizer binary inventory required')
    merge(pins, runtime)
    return raw.decode(), dict(version=importlib.metadata.version('tokenizers'), files=runtime)


class IdleBarrier:
    """Read-only exact phone identity/absence plus all local producer/group absence."""
    def __init__(self, output, pins, pids, signed):
        self.output, self.pins, self.pids = Path(output), pins, pids
        require(not os.path.lexists(self.output), 'Exclusive private idle logs required')
        self.output.mkdir()
        self.commands, self.observations = [], []
        self.env, _ = package.build_environment()
        self.env['TMPDIR'] = str(self.output)
        self.selected = physical.selection(transport.SELECTION, transport.SELECTION_SHA)
        merge(pins, {str(transport.SELECTION.relative_to(ROOT)): transport.SELECTION_SHA})
        merge(pins, transport.integration.admit_build(transport.NATIVE, transport.NATIVE_SHA))
        self.app, extra, self.remote, actual = physical.admit_reuse(
            transport.PREPARED, transport.PREPARED_SHA, self.selected, transport.NATIVE_SHA)
        merge(pins, extra)
        require(actual == signed, 'Exact collection signed bundle required')
        self.signed = dict(signed)
        prepared = json.loads(capture(transport.PREPARED/'integration.json', pins, transport.PREPARED_SHA))
        buffers = []
        for name in ('provisioning.stdout', 'entitlements.stdout', 'signature-details.stderr'):
            path = transport.PREPARED/'private-logs'/name
            buffers.append(capture(path, pins, prepared['private_evidence_pins'][str(path.relative_to(ROOT))]))
        self.profile = physical.admit_profile(plistlib.loads(buffers[0]), plistlib.loads(buffers[1]),
                                             buffers[2].decode(), self.selected)

    def command(self, args, label, expected=0):
        guard(self.pins)
        row = package.owned(args, self.output/(label+'.stdout'), self.output/(label+'.stderr'),
                            self.env, 110)
        self.commands.append(row)
        buffers = {}
        row['capture_errors'] = []
        for field in ('stdout', 'stderr'):
            try:
                buffers[field] = capture(ROOT/row[field], self.pins)
            except BaseException as error:
                row['capture_errors'].append(field+': '+type(error).__name__+': '+str(error))
        require(row['exit_code'] == expected and not row['cleanup_errors']
                and row['primary_error'] == (None if expected == 0 else 'tool exit '+str(expected)),
                'Read-only idle command failed: '+str(row['primary_error'])+
                '; cleanup='+repr(row['cleanup_errors'])+'; captures='+repr(row['capture_errors']))
        require(not row['capture_errors'], 'Idle diagnostic binding failed: '+repr(row['capture_errors']))
        guard(self.pins)
        return buffers['stdout'], buffers['stderr']

    def device(self, parts, label):
        out, _ = self.command([physical.DEVICECTL, 'device', *parts, '--device',
            self.selected['device_identifier'], '--timeout', '90', '--quiet', '--json-output', '-'], label)
        data = json.loads(out)
        if parts == ['info', 'details']:
            require(data['info']['outcome'] == 'success', 'Device details command failed')
            physical.admit_device(data, self.selected)
            return data
        return transport.admit_command(data, self.selected)

    def check(self, _binding=None):
        expiry = datetime.datetime.fromisoformat(self.profile['expires_utc'])
        require(expiry.tzinfo is not None and expiry > utc_now(), 'Admitted signing profile expired')
        index = len(self.observations)
        require(bool(self.pids) and all(type(p) is int and p > 1 for p in self.pids),
                'Actual owned host producer PIDs required')
        out, err = self.command(['ps', '-p', ','.join(map(str, self.pids)), '-o', 'pid='],
                                str(index)+'-local', expected=1)
        require(not out and not err and not any(package.group_exists(p) for p in self.pids),
                'Owned local producer/group remains live')
        device = physical.admit_device(self.device(['info', 'details'], str(index)+'-device'), self.selected)
        current = self.device(['info', 'apps', '--bundle-id', physical.BUNDLE], str(index)+'-install')
        physical.launch_verified(lambda: current, lambda: True, self.remote, self.selected)
        processes = self.device(['info', 'processes', '--search', 'TokletHarness'], str(index)+'-absence')
        require(not physical.owned_processes(processes, self.remote), 'Phone inference remains live')
        require(physical.inventory(self.app) == self.signed, 'Local signed bundle changed')
        self.observations.append(dict(local_pids=self.pids, local_groups_absent=True,
            remote_absent=True, installed_identity_matches=True, device=device))
        guard(self.pins)

    def public(self):
        return physical.redact_record(dict(commands=self.commands, observations=self.observations,
            profile=self.profile), self.selected, self.remote)


def write_bound(path, value, pins, raw=False):
    data = value if raw else encoded(value)
    with Path(path).open('xb') as stream:
        stream.write(data)
    merge(pins, {str(Path(path).relative_to(ROOT)): hashlib.sha256(data).hexdigest()})


def normalize(output):
    output = Path(output).absolute()
    require(output.resolve().is_relative_to(ROOT/'artifacts') and not os.path.lexists(output),
            'Fresh repository artifact output required')
    output.mkdir()
    pins = {str(Path(__file__).resolve().relative_to(ROOT)): package.sha(Path(__file__).resolve())}
    barrier = None
    primary, endpoints, decoded = None, [], {}
    previous = signal.signal(signal.SIGTERM, package.interrupted)
    try:
        q, cases, packets, pids = admit_source(pins)
        tokenizer_json, runtime = tokenizer_inputs(pins)
        signed = q['execution']['nested_receipts']['qwen8-tasks-full-run']['signed_bundle']
        barrier = IdleBarrier(output/'private-logs', pins, pids, signed)
        barrier.check()
        write_bound(output/'before-decode.json', barrier.public(), pins)
        # Import/construct only after fresh owned-process/group and remote absence.
        from tokenizers import Tokenizer
        decoder = Tokenizer.from_str(tokenizer_json)
        for platform in PLATFORMS:
            decoded[platform] = decode_responses(cases, packets[platform]['checked'], decoder)
        common = dict(pins)
        for platform in PLATFORMS:
            raw_path = output/(platform+'.raw.jsonl')
            write_bound(raw_path, packets[platform]['raw'], pins, raw=True)
            record = dict(format='toklet-m8-app-task-response-derivative-v1', platform=platform,
                source_qualification=dict(path=QUAL, sha256=QUAL_SHA), tokenizer=dict(path=TOKENIZER,
                sha256=TOKENIZER_SHA, skip_special_tokens=True, runtime=runtime),
                bindings=common, raw_sha256=pins[str(raw_path.relative_to(ROOT))],
                local_producer_pids=pids, responses=decoded[platform], quality_completed=False)
            write_bound(output/(platform+'.json'), record, pins)
        barrier.check()
        write_bound(output/'after-decode.json', barrier.public(), pins)
        guard(pins)
    except BaseException as error:
        primary = type(error).__name__+': '+str(error)
    finally:
        signal.signal(signal.SIGTERM, previous)
        try:
            guard(pins)
        except BaseException as error:
            endpoints.append(type(error).__name__+': '+str(error))
        private = dict(primary_error=primary, endpoint_errors=endpoints,
                       commands=barrier.commands if barrier else [])
        write_bound(output/'private-execution.json', private, pins)
        try:
            guard(pins)
        except BaseException as error:
            endpoints.append(type(error).__name__+': '+str(error))
        passed = primary is None and not endpoints and set(decoded) == set(PLATFORMS)
        result = dict(format='toklet-m8-app-task-response-normalization-v1', passed=passed,
            source_qualification=dict(path=QUAL, sha256=QUAL_SHA), bindings=pins,
            idle=barrier.public() if barrier else None, primary_error=primary, endpoint_errors=endpoints,
            task_quality_qualified=False, new_vm_runs=0, new_generated_program_executions=0)
        if barrier:
            result = physical.redact_record(result, barrier.selected, barrier.remote)
        with (output/'normalization.json').open('xb') as stream:
            stream.write(encoded(result))
        print(json.dumps(dict(passed=passed, primary_error=result['primary_error'],
                             endpoint_errors=result['endpoint_errors'])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    sys.exit(0 if normalize(arguments.output)['passed'] else 1)
