#!/usr/bin/env python3
"""Offline screening for the fixed official Gemma CLI384/128 capture.

Consumes sealed evidence; never launches an engine, fetches a model, or accepts
an optimization. A/B are instrumentation modes within each revision. The CLI
selects disabled mode and resamples independent complete five-request owners.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import statistics

from request_cost_report import digest, parse, require
from gemma_regression import SCREENING_THRESHOLD_PCT, classify_metric, classify_comparison

META_LIMIT = 2 * 1024**2
RAW_LIMIT = 8 * 1024**2
SOURCE_LIMIT = 32 * 1024**2
BINARY_LIMIT = 64 * 1024**2
POLICIES = ('per-tile', 'batched-single-row')
RAW_NAMES = {'command.json', 'stdout.bin', 'stderr.bin', 'frames.jsonl', 'requests.jsonl', 'session.json'}
COLD_FUNCTIONS = {'checkpoint', 'admit_conformance', 'admit_source3', 'prerequisites'}
COLD_GLOBALS = {'W', 'W2', 'SOURCE_SHA', 'SOURCE2_SHA', 'CHECKS_SHA', 'AUTH_SHA'}
MODEL_REVISION = '70af34e20bd4b7a91f0de6b22675850c43922a03'
ZERO = dict(history_tokens=0, committed_tokens=0, pending_tokens=0)
FULL = dict(history_tokens=512, committed_tokens=511, pending_tokens=1)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin_shape(item):
    require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}
            and isinstance(item['path'], str) and Path(item['path']).is_absolute()
            and type(item['bytes']) is int and item['bytes'] >= 0 and digest(item['sha256'], 64), 'Invalid exact pin')
    return item


def state_equal(value, expected):
    return (type(value) is dict and set(value) == set(expected)
            and all(type(value[k]) is int and value[k] == expected[k] for k in expected))


def _read_regular(path, limit, retain=True, expected=None):
    """Admit and bound one opened descriptor before reading or hashing bytes."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit, 'Bounded regular metadata required')
        require(expected is None or before.st_size == expected['bytes'], 'Pinned file type/size differs')
        count = 0
        checksum = hashlib.sha256()
        pieces = []
        while True:
            block = stream.read(min(65536, limit + 1 - count))
            if not block:
                break
            count += len(block)
            require(count <= limit, 'Pinned file exceeds bound')
            checksum.update(block)
            if retain:
                pieces.append(block)
        after = os.fstat(stream.fileno())
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                and count == before.st_size, 'Pinned file content changed')
    item = dict(path=str(path), bytes=count, sha256=checksum.hexdigest())
    require(expected is None or item == expected, 'Pinned file content changed')
    return (b''.join(pieces) if retain else None), item


def pinned(item, limit=META_LIMIT, retain=True):
    """Nonblocking regular-file read: no symlink/FIFO or unbounded allocation."""
    pin_shape(item)
    require(item['bytes'] <= limit, 'Pinned file exceeds bound')
    return _read_regular(item['path'], limit, retain, expected=item)[0]


def json_value(raw):
    value = parse(raw)
    stack = [(value, 0)]
    while stack:
        node, depth = stack.pop()
        require(depth <= 40, 'JSON nesting exceeds bound')
        if isinstance(node, float):
            require(math.isfinite(node), 'Nonfinite JSON number')
        elif isinstance(node, dict):
            stack.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, list):
            stack.extend((v, depth + 1) for v in node)
    return value


def document(item):
    return json_value(pinned(item))


def local_pin(path):
    path = Path(path).absolute()
    return _read_regular(str(path), META_LIMIT, retain=False)[1]


def integer(value, message='Invalid integer clock'):
    require(type(value) is int and 0 <= value < 2**64, message)
    return value


def positive(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'Positive finite duration required')
    return value


def relative(path):
    require(isinstance(path, str) and path and len(path) <= 512 and not Path(path).is_absolute()
            and all(part not in ('', '.', '..') for part in path.split('/')), 'Unsafe relative source path')
    return path


def source_files(freeze):
    require(freeze['format'] == 'toklet-gemma-cli-source-freeze-v1'
            and type(freeze['files']) is dict and 0 < len(freeze['files']) <= 8192, 'Invalid source freeze')
    for path, value in freeze['files'].items():
        relative(path)
        require(digest(value, 64), 'Invalid frozen source digest')
    return freeze['files']


def apply_diff(raw, originals):
    """Apply bounded ordinary UTF8 unified hunks; reject unsupported formats."""
    raw.decode('utf-8')
    lines = []
    for line in raw.splitlines(keepends=True):
        if line.startswith(b'\\ No newline at end of file'):
            require(lines and lines[-1].endswith(b'\n'), 'Invalid no-newline marker')
            lines[-1] = lines[-1][:-1]
        else:
            lines.append(line)
    i = 0
    results = {}
    while i < len(lines):
        if lines[i].startswith((b'diff --git ', b'index ', b'new file mode ', b'deleted file mode ')):
            i += 1
            continue
        require(lines[i].startswith(b'--- ') and i + 1 < len(lines) and lines[i+1].startswith(b'+++ '),
                'Unsupported or incomplete unified diff')
        def name(line, prefix):
            value = line[4:].rstrip(b'\n').split(b'\t')[0].decode('utf-8')
            if value == '/dev/null':
                return None
            require(value.startswith(prefix), 'Unified diff path prefix differs')
            return relative(value[2:])
        before = name(lines[i], 'a/')
        after = name(lines[i+1], 'b/')
        path = before or after
        require(path and (before is None or after is None or before == after)
                and path in originals and path not in results, 'Unified diff path inventory differs')
        old = originals[path]
        require((old is None) == (before is None), 'Unified diff addition differs')
        old_lines = (old or b'').splitlines(keepends=True)
        cursor = 0
        output = []
        hunks = 0
        i += 2
        while i < len(lines) and lines[i].startswith(b'@@ '):
            match = re.match(rb'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', lines[i])
            require(match is not None, 'Invalid unified hunk')
            start, count, new_start, new_count = [int(x) if x is not None else 1 for x in match.groups()]
            offset = start - 1 if count else start
            require(cursor <= offset <= len(old_lines), 'Overlapping/out-of-order unified hunk')
            output.extend(old_lines[cursor:offset])
            cursor = offset
            require(len(output) == (new_start - 1 if new_count else new_start), 'Unified new offset differs')
            used = added = 0
            i += 1
            while i < len(lines) and (used < count or added < new_count):
                line = lines[i]
                require(line[:1] in (b' ', b'+', b'-'), 'Incomplete unified hunk')
                if line[:1] != b'+':
                    require(cursor < len(old_lines) and old_lines[cursor] == line[1:], 'Unified source content differs')
                    cursor += 1
                    used += 1
                if line[:1] != b'-':
                    output.append(line[1:])
                    added += 1
                i += 1
            require(used == count and added == new_count, 'Unified hunk counts differ')
            hunks += 1
        require(hunks > 0, 'Missing unified content hunk')
        output.extend(old_lines[cursor:])
        result = b''.join(output)
        require(after is not None or result == b'', 'Deleted file retained content')
        results[path] = result if after is not None else None
    require(set(results) == set(originals), 'Unified diff/content archive inventory differs')
    return results


def clean_guard(value):
    require(value['passed'] is True and type(value['exit_code']) is int and value['exit_code'] == 0 and value['failure'] is None
            and value['cleanup_errors'] == [] and value['owned_live_after_cleanup'] == [], 'Guard did not finish cleanly')


def ci_proof(item, commit):
    value = document(item)
    require(value['format'] == 'toklet-gemma-cli-exact-head-ci-v1' and value['commit'] == commit
            and value['remote_ref']['object']['sha'] == commit and value['all_complete_success'] is True,
            'Actual published commit CI differs')
    runs = value['runs']
    groups = value['jobs']
    require(0 < len(runs) == value['workflow_count'] <= 64 and len(groups) == len(runs), 'CI workflow inventory differs')
    ids = {r['id'] for r in runs}
    require(len(ids) == len(runs) and all(type(i) is int for i in ids)
            and {g['run_id'] for g in groups} == ids, 'CI run identities differ')
    require(all(r['head_sha'] == commit and r['status'] == 'completed' and r['conclusion'] == 'success' for r in runs),
            'CI run incomplete or failed')
    jobs = [job for group in groups for job in group['jobs']]
    require(0 < len(jobs) == value['job_count'] <= 512 and len({j['id'] for j in jobs}) == len(jobs)
            and all(j['run_id'] == g['run_id'] and j['status'] == 'completed' and j['conclusion'] == 'success'
                    for g in groups for j in g['jobs']), 'CI job incomplete or failed')


def revision(proof):
    require(proof['format'] == 'toklet-gemma-cli-revision-proof-v1' and proof['kind'] in ('published', 'snapshot'),
            'Unsupported revision proof')
    frozen = source_files(document(proof['source_freeze']))
    build = document(proof['build'])
    checks = document(proof['checks'])
    require(build['format'] == 'toklet-gemma-cli-release-seal-v1'
            and {k: build[k] for k in ('path', 'bytes', 'sha256')} == proof['binary']
            and build['source_freeze'] == proof['source_freeze']
            and build['source_freeze_sha256'] == proof['source_freeze']['sha256']
            and build['required_checks'] == proof['checks'], 'Measured build/source/check pins differ')
    pinned(proof['binary'], BINARY_LIMIT, retain=False)
    artifact = build['cargo_artifact']
    profile = artifact['profile']
    require(artifact['target']['kind'] == ['bin'] and artifact['target']['name'] == 'toklet'
            and profile['opt_level'] == '3' and profile['debug_assertions'] is False
            and profile['overflow_checks'] is False and profile['test'] is False
            and type(artifact['features']) is list and 'metal' in artifact['features']
            and isinstance(build['compiler'], str) and build['compiler'], 'Optimized Metal build identity differs')
    clean_guard(document(build['compile_guard']))
    plan = document(build['compile_plan'])
    require(all(flag in plan['command'] for flag in ('--release', '--locked', '--offline', '--bin', 'toklet')),
            'Optimized offline compile command differs')
    for key in ('compile_stdout', 'compile_stderr'):
        pinned(build[key], RAW_LIMIT, retain=False)
    require(checks['format'] == 'toklet-gemma-cli-required-checks-v1' and checks['completed'] is True
            and checks['passed'] is True and checks['source_unchanged'] is True
            and checks['source_freeze_sha256'] == proof['source_freeze']['sha256']
            and checks['source_files'] == len(frozen) and 0 < len(checks['records']) <= 64, 'Required checks incomplete or stale')
    for record in checks['records']:
        require(record['source_unchanged'] is True, 'Check source changed')
        clean_guard(record['result'])
    commit = base = snapshot = None
    if proof['kind'] == 'published':
        commit = proof['commit']
        require(digest(commit, 40), 'Full published commit required')
        publication = document(proof['publication'])
        mapping = document(proof['source_content'])
        require(publication['format'] == 'toklet-gemma-cli-code-publication-v1'
                and publication['commit'] == commit and publication['pushed'] is True
                and publication['exact_head_ci_passed'] is True and publication['all_source_blobs_match'] is True,
                'Actual publication incomplete')
        for key, item in [('source_freeze', proof['source_freeze']), ('release', proof['build']),
                          ('required_checks', proof['checks']), ('commit_source_map', proof['source_content'])]:
            require(publication['evidence'][key] == item, 'Publication measured input differs: ' + key)
        require(mapping['format'] == 'toklet-gemma-cli-commit-source-map-v1' and mapping['commit'] == commit
                and mapping['source_freeze'] == proof['source_freeze'] and mapping['all_source_blobs_match'] is True
                and mapping['files'] == len(frozen) and mapping['source_binary_receipt'] == proof['build'],
                'Published source-content mapping differs')
        require(publication['evidence']['remote_ci'] == proof['ci'], 'Publication CI pin differs')
        ci_proof(proof['ci'], commit)
    else:
        require(proof['candidate_commit'] is None and digest(proof['base_commit'], 40),
                'Uncommitted snapshot requires null candidate commit and full base')
        base = proof['base_commit']
        archive = document(proof['archive'])
        require(archive['format'] == 'toklet-gemma-cli-snapshot-content-v1' and archive['base_commit'] == base
                and 0 < len(archive['files']) <= 256, 'Snapshot content archive differs')
        reconstructed = dict(source_files(document(archive['base_freeze'])))
        before = {}
        expected = {}
        total = 0
        for entry in archive['files']:
            path = relative(entry['relative'])
            require(path not in before and (entry['before'] is not None or entry['after'] is not None),
                    'Duplicate/empty snapshot change')
            contents = []
            for key in ('before', 'after'):
                item = entry[key]
                raw = None if item is None else pinned(item, SOURCE_LIMIT)
                total += len(raw or b'')
                require(total <= SOURCE_LIMIT, 'Snapshot delta content exceeds bound')
                contents.append(raw)
            require((entry['before'] is None and path not in reconstructed)
                    or (entry['before'] is not None and reconstructed.get(path) == entry['before']['sha256']),
                    'Snapshot base content differs')
            before[path], expected[path] = contents
            require(contents[0] != contents[1], 'Snapshot archive includes unchanged content')
            if entry['after'] is None:
                reconstructed.pop(path)
            else:
                reconstructed[path] = entry['after']['sha256']
        require(apply_diff(pinned(proof['diff'], SOURCE_LIMIT), before) == expected and reconstructed == frozen,
                'Snapshot diff/archive cannot reconstruct frozen source')
        snapshot = sha(canonical(dict(base_commit=base, source_files=frozen,
                                     diff_sha256=proof['diff']['sha256'], archive_sha256=proof['archive']['sha256'])))
        require(proof['snapshot_id'] == snapshot, 'Snapshot content-proof digest differs')
    engine = {p: d for p, d in frozen.items() if p.startswith(('src/', 'metal/', 'shaders/'))
              or p in ('Cargo.toml', 'Cargo.lock', 'build.rs')}
    require(engine, 'Missing measured engine sources')
    return dict(kind=proof['kind'], commit=commit, base_commit=base, snapshot_id=snapshot, source_files=frozen,
                engine_source_sha256=sha(canonical(engine)), source_freeze=proof['source_freeze'], build=proof['build'],
                binary=proof['binary'], checks=proof['checks'], build_identity=dict(compiler=build['compiler'],
                features=sorted(artifact['features']), profile=profile, compile_command=plan['command'])), build


def observer_scope(proof, bindings):
    require(proof['format'] == 'toklet-gemma-cli-observer-proof-v1' and proof['common'] == bindings['common']
            and proof['controller'] == bindings['controller'], 'Observer binding differs')
    pinned(proof['common'])
    raw = pinned(proof['controller'])
    reference = pinned(proof['reference_controller'])
    def timed(source):
        tree = ast.parse(source)
        kept = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in COLD_FUNCTIONS:
                continue
            if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in COLD_GLOBALS for t in node.targets):
                continue
            kept.append(node)
        return ast.dump(ast.Module(body=kept, type_ignores=[]), include_attributes=False)
    require(timed(raw) == timed(reference), 'Timed observer/capture semantics changed')
    if raw == reference:
        require(proof['delta_review'] is None, 'Unnecessary observer delta waiver')
    else:
        review = document(proof['delta_review'])
        require(review['format'] == 'toklet-gemma-cli-observer-delta-v1'
                and review['before'] == proof['reference_controller'] and review['after'] == proof['controller']
                and review['reviewed'] is True and review['cold_admission_only'] is True
                and review['timed_semantics_unchanged'] is True, 'Unreviewed observer provenance delta')
        delta = pinned(review['diff'], SOURCE_LIMIT)
        # Controller diffs are one ordinary path; its artifact filename may change.
        paths = [line[4:].rstrip(b'\n').split(b'\t')[0].decode()[2:]
                 for line in delta.splitlines(keepends=True) if line.startswith(b'--- ')]
        require(len(paths) == 1 and apply_diff(delta, {paths[0]: reference}) == {paths[0]: raw},
                'Observer delta does not reconstruct current controller')
    return dict(common_sha256=proof['common']['sha256'], reference_controller_sha256=proof['reference_controller']['sha256'],
                timed_semantics_sha256=sha(timed(raw).encode()))


def condition(row):
    require(type(row) is dict, 'Condition must be an object')
    power = row.get('power')
    lines = power.splitlines() if isinstance(power, str) else []
    require(lines and lines[0] == "Now drawing from 'AC Power'" and row['passed'] is True
            and row['lowPowerMode'] is False and type(row['thermalState']) is int and row['thermalState'] == 0,
            'AC/LPM-off/thermal-zero condition required')
    return integer(row['checked_uptime_ns'])


def chain(files):
    value = hashlib.sha256()
    for path, content in sorted(files.items()):
        value.update(path.encode() + b'\0' + bytes.fromhex(content))
    return value.hexdigest()


def checkpoint(value, files, auth):
    require(value['source_files'] == len(files) and value['source_chain_sha256'] == chain(files)
            and value['model_endpoints'] == auth['model_endpoints'], 'Capture source/model checkpoint differs')
    return integer(value['checked_ns'])


def owner(block, policy, mode, bindings, oracle):
    enabled = mode == 'B'
    require(block['mode'] == mode and block['metrics'] is enabled and block['warmups'] == 3 and block['measured'] == 5,
            'Owner mode/warmup inventory differs')
    require(len(block['samples']) == 8 and set(block['raw_pins']) == RAW_NAMES, 'Incomplete complete-owner raw inventory')
    folder = Path(block['directory'])
    require(folder.is_absolute(), 'Absolute owner directory required')
    raw = {}
    for name, item in block['raw_pins'].items():
        require(Path(item['path']) == folder/name, 'Owner raw pin escaped its directory')
        raw[name] = pinned(item, RAW_LIMIT)
    command = json_value(raw['command.json'])
    closed = json_value(raw['session.json'])
    require(closed == block['close'] and closed['closed'] is True and closed['exitcode'] == 0
            and closed['aborted'] is False and closed['failure'] is None and type(closed['extra_events']) is list
            and closed['cleanup']['child_created'] is True and closed['cleanup']['direct_child_reaped'] is True
            and closed['cleanup']['unrelated_process_signals'] is False
            and closed['stdout_bytes'] == len(raw['stdout.bin']) and closed['stderr_bytes'] == len(raw['stderr.bin'])
            and raw['stderr.bin'] == b'', 'Incomplete owner lifecycle or raw output')
    identity = block['child_identity']
    started = integer(block['started_ns'])
    finished = integer(closed['finished_ns'])
    require(type(identity['pid']) is int and identity['pid'] > 0 and closed['child_identity'] == identity
            and identity['spawn_monotonic_ns'] == started == command['started_ns']
            and started < integer(block['ready_ns']) < finished, 'Owner identity/clock differs')
    args = [bindings['binary']['path'], '--model', bindings['_model_directory'], '--model-profile', 'gemma4-bf16',
            '--backend', 'metal', '--context-capacity', '512', '--prefill-chunk', '32', '--submission', policy,
            '--interactive', '--json']
    if enabled:
        args += ['--request-metrics', 'summary', '--metrics-output', str(folder/'metrics.jsonl')]
    args += ['--no-eos', '--max-new-tokens', '128']
    require(command['command'] == block['command'] == args
            and command['timestamp_scope'] == 'parent monotonic receive; completed newline before JSON interpretation'
            and identity['command_sha256'] == sha((json.dumps(args, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n').encode()), 'CLI command/observer scope differs')
    stable = block['stable_conditions']
    require(len(stable) == 10, 'Ten fresh stable condition checks required')
    clocks = [condition(row) for row in stable]
    require(integer(block['before']['checked_ns']) <= clocks[0] and clocks == sorted(set(clocks))
            and clocks[-1] <= integer(block['launch_call_ns']) <= started,
            'Stable conditions not observed within current owner admission')
    frames = [json_value(line) for line in raw['frames.jsonl'].splitlines()]
    requests = [json_value(line) for line in raw['requests.jsonl'].splitlines()]
    require(len(requests) == 16 and 1 <= len(frames) <= 4096, 'Raw request/frame count differs')
    offset = 0
    clock = 0
    for frame in frames:
        require(integer(frame['offset']) == offset and type(frame['length']) is int and 0 < frame['length'] <= 65536,
                'Invalid raw stdout extent')
        line = raw['stdout.bin'][offset:offset+frame['length']]
        require(line.endswith(b'\n') and len(line) == frame['length'] and sha(line) == frame['sha256']
                and json_value(line) == frame['event'] and clock <= integer(frame['received_ns']) <= finished,
                'Raw frame/stdout/hash/clock join differs')
        clock = frame['received_ns']
        offset += frame['length']
    require(offset == len(raw['stdout.bin']) and frames[0]['event'] == block['ready']
            and frames[0]['received_ns'] == block['ready_ns'], 'Raw ready or stdout coverage differs')
    ready = block['ready']
    require(ready['event'] == 'ready' and ready['model_profile'] == 'gemma4-bf16' and ready['backend'] == 'metal'
            and ready['submission'] == policy and ready['context_capacity'] == 512, 'Ready route differs')
    position = 1
    observations = []
    all_observations = []
    finished_text = []
    for sequence, sample in enumerate(block['samples']):
        reset, submitted = requests[sequence*2:sequence*2+2]
        require(json_value(bytes.fromhex(reset['raw_hex'])) == {'action': 'reset'}
                and json_value(bytes.fromhex(submitted['raw_hex'])) == dict(action='generate', tokens=oracle['input_ids'], max_new_tokens=128),
                'Exact reset/input384/output128 request differs')
        reset_clock = integer(reset['sent_ns'])
        sent = integer(submitted['sent_ns'])
        require((block['ready_ns'] if sequence == 0 else all_observations[-1]['terminal_ns']) <= reset_clock < sent,
                'Reset/request chronology differs')
        tail = ''
        while position < len(frames) and frames[position]['event']['event'] == 'text':
            frame = frames[position]
            require(reset_clock <= frame['received_ns'] <= sent, 'Reset finalization clock differs')
            tail += frame['event']['text']
            position += 1
        if sequence:
            finished_text[-1] += tail
        else:
            require(tail == '', 'Unexpected initial text')
        require(position < len(frames) and frames[position]['event'] == dict(event='reset', request_id=sequence*2+1, state=ZERO)
                and reset_clock <= frames[position]['received_ns'] <= sent, 'Reset did not return exact empty state')
        position += 1
        tokens = []
        text = ''
        token_clocks = []
        while position < len(frames) and frames[position]['event']['event'] != 'done':
            frame = frames[position]
            event = frame['event']
            require(frame['received_ns'] >= sent and event['event'] in ('token', 'text'), 'Generation failed or raw event differs')
            if event['event'] == 'token':
                require(event['index'] == len(tokens), 'Token callback index differs')
                tokens.append(event['token_id'])
                token_clocks.append(frame['received_ns'])
            text += event['text']
            position += 1
        require(position < len(frames) and tokens == oracle['output_ids'] and len(tokens) == 128, 'Actual128 output IDs differ')
        terminal = frames[position]
        position += 1
        term = terminal['event']
        stats = term['statistics']
        core = stats['generation']
        require(term['event'] == 'done' and term['request_id'] == sequence*2+2 and stats['model_profile'] == 'gemma4-bf16'
                and stats['backend'] == 'metal' and stats['submission'] == policy
                and type(stats['decode_evaluations']) is int and stats['decode_evaluations'] == 127
                and state_equal(stats['state'], FULL) and core['reason'] == {'kind': 'output_limit'}
                and [core[k] for k in ('reused_tokens', 'prefill_tokens', 'sampled_tokens', 'emitted_tokens',
                    'history_tokens', 'evaluated_tokens', 'pending_tokens')] == [0, 384, 128, 128, 512, 511, 1],
                'Completed core work/state differs')
        require(all(type(core[k]) is int for k in ('reused_tokens', 'prefill_tokens', 'sampled_tokens', 'emitted_tokens',
                'history_tokens', 'evaluated_tokens', 'pending_tokens')), 'Integer completed work required')
        end = terminal['received_ns']
        clocks = [sent, token_clocks[0], token_clocks[-1], end]
        require(clocks == sorted(clocks) and sent < token_clocks[0] and 0 < end-sent <= 180*10**9,
                'External receive clocks invalid')
        for key, clock_value in zip(('sent_ns', 'first_token_ns', 'last_token_ns', 'terminal_ns'), clocks):
            require(type(sample[key]) is int and sample[key] == clock_value, 'Saved/raw receive clock differs')
        require(sample['terminal'] == term and sample['sequence'] == sequence and sample['warmup'] is (sequence < 3)
                and sample['measured_index'] == (None if sequence < 3 else sequence-3)
                and sample['output_ids_sha256'] == oracle['output_ids_u32le_sha256']
                and sample['text_before_finish_sha256'] == sha(text.encode())
                and sample['finish_scope'] == ('following_reset' if sequence < 7 else 'process_eof'),
                'Saved sample/raw work/phase differs')
        for key, expected in [('external_ttft_seconds', (clocks[1]-sent)/1e9),
                              ('external_complete_request_seconds', (end-sent)/1e9)]:
            require(math.isclose(positive(sample[key]), expected, rel_tol=1e-12, abs_tol=1e-12), 'Saved duration/raw clocks differ')
        observation = dict(sent_ns=sent, first_token_ns=clocks[1], last_token_ns=clocks[2], terminal_ns=end,
                           ttft_ms=(clocks[1]-sent)/1e6, total_ms=(end-sent)/1e6,
                           input_sha256=oracle['input_ids_u32le_sha256'], output_sha256=oracle['output_ids_u32le_sha256'])
        all_observations.append(observation)
        if sequence >= 3:
            observations.append(observation)
        finished_text.append(text)
    eof_tail = ''
    for frame in frames[position:]:
        require(frame['event']['event'] == 'text' and all_observations[-1]['terminal_ns'] <= frame['received_ns'] <= finished,
                'Unexpected event after final generation')
        eof_tail += frame['event']['text']
    require(closed['tail_text'] == eof_tail and closed['extra_events'] == frames[position:], 'EOF tail publication differs')
    finished_text[-1] += eof_tail
    require(all(value == oracle['canonical_decode_skip_special'] for value in finished_text)
            and all(sample['finalized_text_sha256'] == sha(text.encode())
                    for sample, text in zip(block['samples'], finished_text)), 'Completed text/reset/EOF oracle differs')
    manifest = None
    if enabled:
        metric_pin = closed['metrics_pin']
        require(metric_pin['path'] == str(folder/'metrics.jsonl') and block['metrics_export']['pin'] == metric_pin,
                'Summary export scope differs')
        rows = [json_value(line) for line in pinned(metric_pin, RAW_LIMIT).splitlines()]
        require(len(rows) == 18 and all(0 < len(canonical(row))+1 <= 4096 for row in rows), 'Summary fixed framing differs')
        manifest, footer = rows[0], rows[-1]
        facts = block['metrics_export']['facts']
        require(manifest['event'] == 'request_metrics_manifest' and manifest['schema_version'] == 1
                and facts['manifest'] == manifest and facts['export_status'] == footer
                and facts['validated_complete_export'] is True and facts['telemetry_complete'] is True
                and facts['retained_records'] == facts['attempted_requests'] == 16
                and footer['event'] == 'request_metrics_export' and footer['complete'] is True and footer['retained'] == 16
                and all(footer[k] == 0 for k in ('telemetry_dropped', 'invalid_records', 'export_errors'))
                and footer['counter_overflow'] is False, 'Summary framing/loss differs')
        for index, row in enumerate(rows[1:-1]):
            reset = index % 2 == 0
            require(row['request_id'] == row['attempt_id'] == index+1 and row['owner_generation'] == 1
                    and row['request_kind'] == ('reset' if reset else 'generate') and row['terminal_status'] == 'success'
                    and state_equal(row['before'], ZERO if index == 0 or not reset else FULL)
                    and state_equal(row['after'], ZERO if reset else FULL) and row['replay_required'] is False,
                    'Summary request/state inventory differs')
            if not reset:
                require([row['work'][k] for k in ('prompt_tokens', 'reused_prefix_tokens', 'prefill_rows', 'decode_evaluations',
                    'sampled_tokens', 'emitted_tokens', 'callback_tokens', 'replayed_rows')] == [384, 0, 384, 127, 128, 128, 128, 0],
                    'Summary completed work differs')
    else:
        require(closed['metrics_pin'] is None and block['metrics_export'] is None, 'Disabled owner exported metrics')
    return dict(identity=identity, observations=observations, warmups=3), all_observations, manifest, finished


def admit_cli_bundle(capture_pin, revision_proof, selector, observer_proof):
    require(type(selector) is dict and set(selector) == {'policy', 'mode'} and selector['policy'] in POLICIES
            and selector['mode'] == 'disabled', 'Select one policy and disabled mode explicitly')
    measured_revision, build = revision(revision_proof)
    capture = document(capture_pin)
    require(capture['format'] == 'toklet-gemma-cli-overhead-capture-v1' and capture['completed'] is True
            and capture['failure'] is None and capture['source_freeze_pin'] == revision_proof['source_freeze']
            and capture['binary_pin'] == revision_proof['binary'], 'Capture incomplete or revision scope differs')
    bindings = document(capture['bindings_pin'])
    require(bindings['format'] == 'toklet-gemma-cli-overhead-bindings-v1'
            and bindings['source_freeze'] == revision_proof['source_freeze'] and bindings['binary'] == revision_proof['binary']
            and bindings['build_receipt'] == revision_proof['build'] and bindings['required_checks'] == revision_proof['checks']
            and bindings['protocol'] == capture['protocol_pin'], 'Capture measured input bindings differ')
    observer = observer_scope(observer_proof, bindings)
    require(observer_proof['guard_plan'] == capture['outer_guard_plan'], 'Capture guard plan differs')
    guard_plan = document(observer_proof['guard_plan'])
    guard = document(observer_proof['guard_result'])
    enforcement = document(observer_proof['enforcement'])
    clean_guard(guard)
    require(guard_plan['limit_bytes'] == 6*1024**3 and guard_plan['helper_reserve_bytes'] == 32*1024**2
            and guard_plan['configured_guest_reserve_bytes'] == 0 and guard_plan['disk_floor_bytes'] == 2*1024**3
            and 7200 <= guard_plan['timeout_seconds'] <= 14400 and guard_plan['cwd'] == bindings['source_root']
            and guard['maximum_observed_charge_with_reserve'] <= guard_plan['limit_bytes']
            and guard['terminal_direct_pair_rss_charge_with_reserve'] <= guard_plan['limit_bytes']
            and guard['minimum_free_bytes'] >= guard_plan['disk_floor_bytes'] and guard['sample_count'] > 0,
            'Capture resource guard policy differs')
    command = guard_plan['command']
    require(type(command) is list and len(command) == 8
            and all(isinstance(arg, str) and arg for arg in command), 'Guard command must be eight nonempty strings')
    expected_command = [command[0], observer_proof['controller']['path'], '--bindings', capture['bindings_pin']['path'],
                        '--bindings-sha256', capture['bindings_pin']['sha256'], '--outer-guard-plan', observer_proof['guard_plan']['path']]
    require(guard_plan['command'] == expected_command and enforcement['format'] == 'toklet-gemma-cli-overhead-enforcement-v1'
            and enforcement['passed'] is True and enforcement['source_and_model_endpoints_unchanged'] is True
            and enforcement['source_freeze'] == revision_proof['source_freeze'] and enforcement['bindings_pin'] == capture['bindings_pin']
            and enforcement['result'] == guard, 'Capture guard/source enforcement differs')
    runtime_terminal_clock = condition(guard['last_runtime_gate'])
    if revision_proof['kind'] == 'published':
        publication = document(revision_proof['publication'])
        for key, item in [('capture', capture_pin), ('guard', observer_proof['guard_result']), ('enforcement', observer_proof['enforcement'])]:
            require(publication['evidence'][key] == item, 'Published measured capture differs: ' + key)
    auth = document(bindings['authenticated_inputs'])
    require(auth['format'] == 'toklet-gemma-cli-authenticated-inputs-v1'
            and set(auth['files']) == {'model.safetensors', 'config.json', 'tokenizer_config.json', 'generation_config.json',
                                      'chat_template.jinja', 'tokenizer.json'}
            and build['compiler'] == auth['rustc'] and type(auth['hardware_os']) is dict and auth['hardware_os'],
            'Authenticated model/toolchain/hardware identity missing')
    for item in auth['files'].values():
        pin_shape(item)  # Content authentication receipt; deliberately never read model weights.
    for key in ('machdep.cpu.brand_string', 'hw.model', 'kern.osrelease', 'kern.osversion', 'kern.osproductversion'):
        require(isinstance(auth['hardware_os'][key], str) and auth['hardware_os'][key], 'Named hardware/OS missing')
    oracle_receipt = document(bindings['codec_oracles'])
    require(oracle_receipt['format'] == 'toklet-gemma-cli-codec-oracles-v1' and oracle_receipt['completed'] is True
            and oracle_receipt['available'] is True and oracle_receipt['provenance_unchanged'] is True,
            'Independent codec oracle incomplete')
    oracle = oracle_receipt['primary128']
    import struct
    for key, count in [('input_ids', 384), ('output_ids', 128)]:
        require(len(oracle[key]) == count and all(type(t) is int and 0 <= t < 262144 for t in oracle[key])
                and sha(struct.pack('<' + str(count) + 'I', *oracle[key])) == oracle[key+'_u32le_sha256'],
                'Codec exact fixed token work differs')
    require(oracle['source']['requests_identical'] == 26
            and sha(oracle['canonical_decode_skip_special'].encode()) == oracle['canonical_decode_skip_special_utf8_sha256']
            and sha(oracle['canonical_decode_keep_special'].encode()) == oracle['canonical_decode_keep_special_utf8_sha256'],
            'Canonical codec text proof differs')
    protocol = document(capture['protocol_pin'])
    require(protocol['workload']['capacity'] == 512 and protocol['workload']['chunk'] == 32
            and protocol['workload']['max_new_tokens'] == 128
            and protocol['workload']['input_ids']['sha256'] == oracle['input_ids_u32le_sha256']
            and protocol['workload']['input_ids']['bytes'] == 1536, 'Fixed protocol work/codec bridge differs')
    require(protocol['totals'] == dict(generation_requests=128, measured_generation_requests=80, warmup_generation_requests=48,
            reset_controls=128, processes=16) and protocol['process_blocks']['warmup_generations_per_process'] == 3
            and protocol['process_blocks']['measured_generations_per_process'] == 5, 'Fixed sampling protocol differs')
    require(capture['inventory'] == dict(owners=16, generation_requests=128, measured=80, warmup=48, reset_controls=128,
            summary_exports=8, records_per_summary_export=16), 'Complete capture inventory differs')
    wanted = [(p, c, list('ABBA' if c == 1 else 'BAAB')) for p in POLICIES for c in (1, 2)]
    require([(c['policy'], c['cohort'], c['order']) for c in capture['cohorts']] == wanted
            and [(c['policy'], c['cohort'], c['order']) for c in protocol['cohorts']] == wanted, 'Fixed policy/cohort schedule differs')
    files = measured_revision['source_files']
    initial = checkpoint(capture['initial_checkpoint'], files, auth)
    final = checkpoint(capture['cohorts'][-1]['blocks'][-1]['after'], files, auth)
    require(initial < final <= runtime_terminal_clock, 'Capture/terminal guard boundary order differs')
    bindings = dict(bindings, _model_directory=auth['model_directory'])
    cohorts = []
    observations = []
    excluded = []
    manifests = []
    previous = initial
    identities = set()
    for cohort in capture['cohorts']:
        require(len(cohort['blocks']) == 4 and [b['block'] for b in cohort['blocks']] == [0, 1, 2, 3]
                and [b['mode'] for b in cohort['blocks']] == cohort['order'], 'Incomplete owner schedule')
        if cohort['policy'] != selector['policy']:
            continue
        owners = []
        for block, mode in zip(cohort['blocks'], cohort['order']):
            before = checkpoint(block['before'], files, auth)
            after = checkpoint(block['after'], files, auth)
            admitted, raw_observations, manifest, finish = owner(block, selector['policy'], mode, bindings, oracle)
            key = (admitted['identity']['pid'], admitted['identity']['spawn_monotonic_ns'])
            require(key not in identities and previous <= before <= block['launch_call_ns'] <= block['started_ns'] < finish <= after <= final,
                    'Overlapping, duplicate or incomplete owner chronology')
            identities.add(key)
            previous = after
            if mode == 'A':
                owners.append(admitted)
                observations.extend(raw_observations)
            else:
                manifests.append(manifest)
                excluded.append(dict(identity=admitted['identity'], raw_pins=block['raw_pins'], metrics_pin=block['close']['metrics_pin']))
        require(len(owners) == 2, 'Two independent disabled owners required per cohort')
        cohorts.append(dict(cohort=cohort['cohort'], owners=owners))
    require(len(cohorts) == 2 and len(manifests) == 4, 'Two confirmation cohorts required')
    route = None
    for manifest in manifests:
        current = {k: manifest[k] for k in ('model_profile', 'backend', 'model_repository', 'model_revision', 'weight_dtype', 'kv_dtype',
                                            'sampling', 'thinking_enabled', 'capacity', 'prefill_chunk_size', 'submission')}
        require(current == dict(model_profile='gemma4-bf16', backend='metal', model_repository='google/gemma-4-E2B-it',
                model_revision=MODEL_REVISION, weight_dtype='bf16', kv_dtype='fp32', sampling='greedy', thinking_enabled=False,
                capacity=512, prefill_chunk_size=32, submission=selector['policy'])
                and (route is None or route == current), 'Mixed model/precision/cache/sampling policy')
        route = current
    observation_ids = sorted(sha(canonical(row)) for row in observations)
    require(len(observation_ids) == 32 and len(set(observation_ids)) == 32, 'Repeated raw generation observations')
    identity = dict(route=route, model_files={k: {f: p[f] for f in ('bytes', 'sha256')} for k, p in auth['files'].items()},
                    hardware_os=auth['hardware_os'], compiler=auth['rustc'], cargo=auth['cargo'],
                    build=measured_revision['build_identity'], input_sha256=oracle['input_ids_u32le_sha256'],
                    output_sha256=oracle['output_ids_u32le_sha256'], text_sha256=oracle['canonical_decode_skip_special_utf8_sha256'])
    return dict(format='toklet-gemma-cli-admission-v1', revision=measured_revision, identity=identity, observer=observer, cohorts=cohorts,
                observation_ids=observation_ids, collection_sha256=sha(canonical(sorted(observation_ids))),
                inputs=dict(capture_pin=capture_pin, revision_proof=revision_proof, selector=selector, observer_proof=observer_proof),
                excluded_summary_owners=excluded, comparison_qualified=False, accepted_optimization=False,
                identity_scope='Named configuration and sealed authentication receipts; unique physical host identity is not established.')


def bootstrap_owner_difference(baseline_owners, candidate_owners):
    for owners in (baseline_owners, candidate_owners):
        require(len(owners) == 2 and all(len(rows) == 5 for rows in owners), 'Exactly two whole five-row owners required')
        for rows in owners:
            for value in rows:
                positive(value)
    rng = random.Random(1)
    values = []
    for _ in range(10000):
        after = [v for rows in rng.choices(candidate_owners, k=2) for v in rows]
        before = [v for rows in rng.choices(baseline_owners, k=2) for v in rows]
        values.append(100*(statistics.median(after)/statistics.median(before)-1))
    values.sort()
    return [values[250], values[9749]]


def compare_cli(baseline, candidate):
    # Reauthenticate public normalized values: caller edits cannot bypass admission.
    baseline = admit_cli_bundle(**baseline['inputs'])
    candidate = admit_cli_bundle(**candidate['inputs'])
    old, new = baseline['revision'], candidate['revision']
    require(old['kind'] == 'published', 'Baseline requires actual published proof')
    require(old['engine_source_sha256'] != new['engine_source_sha256']
            and old['binary']['sha256'] != new['binary']['sha256'], 'Distinct measured engine source and binary variants required')
    require(baseline['identity'] == candidate['identity'] and baseline['observer'] == candidate['observer'],
            'Model/hardware/build/codec or timed observer identity differs')
    require(baseline['collection_sha256'] != candidate['collection_sha256']
            and set(baseline['observation_ids']).isdisjoint(candidate['observation_ids']),
            'The same or overlapping raw collection cannot compare itself')
    batches = []
    for before, after in zip(baseline['cohorts'], candidate['cohorts']):
        require(before['cohort'] == after['cohort'], 'Confirmation cohort order differs')
        metrics = {}
        for key in ('ttft_ms', 'total_ms'):
            old_owners = [[row[key] for row in owner['observations']] for owner in before['owners']]
            new_owners = [[row[key] for row in owner['observations']] for owner in after['owners']]
            old_values = [v for rows in old_owners for v in rows]
            new_values = [v for rows in new_owners for v in rows]
            change = 100*(statistics.median(new_values)/statistics.median(old_values)-1)
            interval = bootstrap_owner_difference(old_owners, new_owners)
            metrics[key] = dict(baseline_median=statistics.median(old_values), candidate_median=statistics.median(new_values),
                change_pct=change, independent_owner_bootstrap_95_pct=interval, status=classify_metric(change, interval),
                baseline_range=[min(old_values), max(old_values)], candidate_range=[min(new_values), max(new_values)])
        batches.append(dict(batch=before['cohort'], metrics=metrics))
    states = [value['status'] for batch in batches for value in batch['metrics'].values()]
    primary = [batch['metrics']['total_ms']['status'] for batch in batches]
    return dict(format='toklet-gemma-cli-regression-v1', status=classify_comparison(states, primary),
        primary_metric='disabled_mode_caller_total_ms', screening_threshold_pct=SCREENING_THRESHOLD_PCT,
        baseline_commit=old['commit'], candidate_commit=new['commit'], candidate_base_commit=new['base_commit'],
        candidate_snapshot_id=new['snapshot_id'], candidate_revision=new, batches=batches,
        repeatable_primary_speed_signal=all(state == 'speed_signal' for state in primary), accepted_optimization=False,
        observer_overhead_qualified=False, sampling_unit='Independent whole five-request owner blocks; two owners per revision/cohort',
        evidence=dict(baseline=baseline['inputs'], candidate=candidate['inputs'], baseline_measurements_sha256=baseline['collection_sha256'],
                      candidate_measurements_sha256=candidate['collection_sha256'], observer=baseline['observer'], identity=baseline['identity']),
        limitations=['One fixed384/128 CLI workload and two confirmation cohorts; no tails or1% qualification.',
                     'G17A/B are instrumentation modes, not revision roles; summary observations are excluded.',
                     'Numerical/task quality, broader workloads, memory, UI, adoption and Astra remain separate gates.',
                     'Named hardware/OS equality does not prove a unique physical host.'])


def bind_publication(comparison_pin, actual_publication_pin):
    compared = document(comparison_pin)
    require(compared['format'] == 'toklet-gemma-cli-regression-v1' and compared['accepted_optimization'] is False
            and compared['candidate_commit'] is None and digest(compared['candidate_snapshot_id'], 64),
            'An immutable precommit comparison is required')
    measured, _ = revision(compared['evidence']['candidate']['revision_proof'])
    require(measured == compared['candidate_revision'] and measured['snapshot_id'] == compared['candidate_snapshot_id']
            and measured['kind'] == 'snapshot', 'Measured precommit snapshot proof differs')
    proof = document(actual_publication_pin)
    require(proof['kind'] == 'published', 'Actual published revision proof required')
    published, _ = revision(proof)
    require(published['source_files'] == measured['source_files'] and published['binary']['sha256'] == measured['binary']['sha256']
            and published['binary']['bytes'] == measured['binary']['bytes']
            and published['build_identity'] == measured['build_identity'], 'Published source/build differs from measured snapshot')
    publication = document(proof['publication'])
    require(publication['evidence']['capture'] == compared['evidence']['candidate']['capture_pin'], 'Publication refers to other measured capture')
    return dict(format='toklet-gemma-cli-publication-binding-v1', comparison_pin=comparison_pin,
                actual_publication_pin=actual_publication_pin, actual_commit=published['commit'], ci=proof['ci'],
                measured_snapshot_id=compared['candidate_snapshot_id'], relevant_content_matches=True,
                accepted_optimization=False, scope='New publication binding only; original comparison and all measurements remain immutable.')


def markdown(report):
    rows = ['# Gemma CLI regression screening', '', 'Status: '+report['status']+'. Negative change is faster.', '',
            '| Cohort | Metric | Baseline ms | Candidate ms | Change % | Owner bootstrap95% | Status |',
            '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for batch in report['batches']:
        for key, metric in batch['metrics'].items():
            interval = ', '.join(f'{v:.2f}' for v in metric['independent_owner_bootstrap_95_pct'])
            rows.append(f"| {batch['batch']} | {key} | {metric['baseline_median']:.3f} | {metric['candidate_median']:.3f} | {metric['change_pct']:.2f} | [{interval}] | {metric['status']} |")
    return '\n'.join(rows + ['', 'Accepted optimization: false.', '', *report['limitations'], ''])


def write_new(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--policy', choices=POLICIES)
    parser.add_argument('--bind-publication', action='store_true')
    args = parser.parse_args(argv)
    require(len(args.paths) == 3, 'Three input/output paths required')
    first, second, output = args.paths
    if args.bind_publication:
        result = bind_publication(local_pin(first), local_pin(second))
        write_new(output, canonical(result)+b'\n')
        return 0
    require(args.policy is not None, 'Explicit submission policy required')
    output.mkdir(parents=True, exist_ok=False)
    try:
        admissions = []
        for path in (first, second):
            value = document(local_pin(path))
            admissions.append(admit_cli_bundle(value['capture_pin'], value['revision_proof'],
                dict(policy=args.policy, mode='disabled'), value['observer_proof']))
        report = compare_cli(*admissions)
        write_new(output/'comparison.json', canonical(report)+b'\n')
        write_new(output/'comparison.md', markdown(report).encode())
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OSError, ArithmeticError, RecursionError, SyntaxError) as error:
        write_new(output/'failure.json', canonical(dict(status='incomparable', reason=str(error)[:2048], accepted_optimization=False))+b'\n')
        return 1
    return {'pass': 0, 'regression': 2, 'noisy': 3}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
