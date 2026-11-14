#!/usr/bin/env python3
"""Offline validation of six fixed ordinary Gemma submission measurements.

Only files are read. Root-supplied provenance references are hash-bound, never
independently certified by this report. No performance or adoption promotion.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import statistics
import struct

MIB = 1024 * 1024
LIMIT, CALLER, HELPER = 6 * 1024 * MIB, 16 * MIB, 32 * MIB
REPO = 'google/gemma-4-E2B-it'
REVISION = '70af34e20bd4b7a91f0de6b22675850c43922a03'
CODEC_SHA = '654dc6dfd66fd893a24f0faa39355b71d5e7ffc3f53c2dac394dfcfd9ed95298'
WORKLOAD_SHA = 'de4ebba10f7bcf6a721c973da65ca8c0a246d04cf7471e159a18e460b4bcc4fd'
PRIMARY_INPUT_SHA = '4ef45f4b84d488ee83a5e3c3ba701b13e1afc25e676ce076d55bd97b42a0f566'
TOKENS_SHA = '20e4789c70ba16a364591f621b274e3532f09ccb1859a51b227cc5b9c14b2503'
P = [2, 105, 2364, 107, 1567, 506, 5279, 529, 7001, 236761, 106, 107, 105, 4368, 107]
E, Q = P[:-1] + [2364], [818, 5279, 529, 7001]
ARMS = ('per-tile', 'batched-single-row')
CASES = ('reset', 'append', 'fully_cached', 'pending', 'edited_prefix',
         'short_128_128', 'boundary_prefill_512_0', 'teacher_128_128', 'teacher_384_128')
CORE = ('reason reused_tokens prefill_tokens sampled_tokens emitted_tokens history_tokens '
        'evaluated_tokens pending_tokens first_token_seconds prefill_seconds decode_seconds '
        'decode_evaluations sampling_seconds total_seconds').split()


def require(value, message):
    if not value:
        raise ValueError(message)


def keys(value, expected, name):
    require(type(value) is dict and set(value) == set(expected), name + ': exact keys')


def parse(raw):
    # Same duplicate/nonfinite policy as request_cost_report.parse.
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'duplicate JSON key: ' + key)
            value[key] = item
        return value
    def finite(token):
        value = float(token)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value
    return json.loads(raw, object_pairs_hook=unique, parse_float=finite,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def integer(value, name, maximum=LIMIT):
    require(type(value) is int and 0 <= value <= maximum, name + ': integer range')
    return value


def equal_int(value, wanted, name):
    require(integer(value, name) == wanted, name + ': count differs')


def number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), name + ': finite range')
    return value


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'SHA256 syntax')
    return value


def text(value, name):
    require(type(value) is str and 0 < len(value) <= 4096, name + ': text')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def token_ids(value, name, length=None):
    require(type(value) is list and len(value) <= 1025 and
            all(type(x) is int and 0 <= x < 262144 for x in value), name + ': token IDs')
    require(length is None or len(value) == length, name + ': token count')
    return value


def ids_sha(ids):
    return sha(b''.join(struct.pack('<I', value) for value in ids))


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _read(path, bound, expected=None, retain=True):
    path = Path(path)
    require(path.is_absolute(), 'pin path must be absolute')
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= bound, 'bounded regular file')
        require(_identity(os.lstat(path)) == _identity(before), 'path/fd identity before read')
        hasher, chunks, total = hashlib.sha256(), [], 0
        while True:
            block = os.read(fd, min(MIB, bound + 1 - total))
            if not block:
                break
            total += len(block)
            require(total <= bound, 'file exceeds bound')
            hasher.update(block)
            if retain:
                chunks.append(block)
        require(total == before.st_size and _identity(os.fstat(fd)) == _identity(before)
                and _identity(os.lstat(path)) == _identity(before), 'EOF/identity changed')
        if expected:
            require(total == expected['bytes'] and hasher.hexdigest() == expected['sha256'],
                    'pin bytes/SHA differ: ' + str(path))
        return b''.join(chunks) if retain else None
    finally:
        os.close(fd)


def read_pin(pin, bound, *, retain=True):
    keys(pin, ('path', 'bytes', 'sha256'), 'pin')
    text(pin['path'], 'pin path')
    integer(pin['bytes'], 'pin bytes', bound)
    digest(pin['sha256'])
    return _read(pin['path'], bound, pin, retain)


def _json_pin(pin, bound=MIB):
    return parse(read_pin(pin, bound))


def _same(value, wanted, name):
    require(value == wanted and type(value) is type(wanted), name + ': identity differs')


def _state(value, history, committed, pending):
    keys(value, ('history', 'committed', 'pending', 'history_sha256'), 'state')
    for key, wanted in zip(('history', 'committed', 'pending'), (len(history), committed, pending)):
        equal_int(value[key], wanted, key)
    require(value['history_sha256'] == ids_sha(history), 'full history hash')


def _core(value, reused, prefill, sampled, emitted, history, committed, pending, decode,
          reason='output_limit', stop=None):
    keys(value, CORE, 'core Report')
    wanted = {'kind': reason}
    if stop is not None:
        wanted['token_id'] = stop
    keys(value['reason'], wanted, 'stop reason')
    require(value['reason'] == wanted, 'stop reason differs')
    if stop is not None:
        equal_int(value['reason']['token_id'], stop, 'stop ID')
    for key, target in zip(('reused_tokens', 'prefill_tokens', 'sampled_tokens', 'emitted_tokens',
                            'history_tokens', 'evaluated_tokens', 'pending_tokens', 'decode_evaluations'),
                           (reused, prefill, sampled, emitted, history, committed, pending, decode)):
        equal_int(value[key], target, key)
    for key in ('prefill_seconds', 'decode_seconds', 'sampling_seconds', 'total_seconds'):
        number(value[key], key)
    require(sum(value[k] for k in ('prefill_seconds', 'decode_seconds', 'sampling_seconds'))
            <= value['total_seconds'] + 1e-9, 'core phases exceed total')
    for count, phase in ((prefill, 'prefill_seconds'), (decode, 'decode_seconds'),
                          (sampled, 'sampling_seconds')):
        require(count > 0 or value[phase] == 0, 'zero work has a nonzero phase: ' + phase)
    if emitted:
        require(value['first_token_seconds'] is not None, 'missing core first token')
        require(number(value['first_token_seconds'], 'core first token') <= value['total_seconds'],
                'core first token exceeds total')
    else:
        require(value['first_token_seconds'] is None, 'zero emissions have a first token')


def _callbacks(row, count):
    elapsed = number(row['complete_request_seconds'], 'completed request')
    events = row['token_event_seconds']
    require(type(events) is list and len(events) == count, 'callback count')
    require(all(number(x, 'callback') <= elapsed for x in events)
            and all(a <= b for a, b in zip(events, events[1:])), 'callback clocks')
    for key, wanted in (('first_callback_seconds', events[0] if events else None),
                        ('last_callback_seconds', events[-1] if events else None)):
        _same(row[key], wanted, key)
    require(row['report']['total_seconds'] <= elapsed, 'core total exceeds caller interval')


def _admission(row, load, reserved=None):
    expected = ('session_admitted_bytes owner_reserved_bytes caller_allowance_bytes '
                'helper_allowance_bytes combined_session_admitted_bytes maximum_with_load_admitted_bytes '
                'limit_bytes').split()
    values = {key: integer(row[key], key) for key in expected}
    equal_int(values['caller_allowance_bytes'], CALLER, 'caller allowance')
    equal_int(values['helper_allowance_bytes'], HELPER, 'helper allowance')
    equal_int(values['limit_bytes'], LIMIT, 'memory limit')
    require(values['session_admitted_bytes'] > 0, 'missing session admission')
    if reserved is not None:
        equal_int(values['owner_reserved_bytes'], reserved, 'owner reserve')
    combined = values['session_admitted_bytes'] + values['owner_reserved_bytes'] + CALLER + HELPER
    require(combined <= LIMIT and values['combined_session_admitted_bytes'] == combined,
            'combined admission')
    require(values['maximum_with_load_admitted_bytes'] == max(combined, load + CALLER + HELPER)
            <= LIMIT, 'load/session maximum')


def _pair_position(row, cohort, warmup, trial, within, *, matrix=False):
    _same(row['warmup'], warmup, 'warmup')
    equal_int(row['trial' if matrix else 'pair_index'], trial, 'pair index')
    equal_int(row['within_pair'], within, 'within pair')
    first = (trial + cohort) % 2 == 1
    order = ARMS if first else ARMS[::-1]
    _same(row['order'], 'AB' if first else 'BA', 'pair order')
    _same(row['arm'], order[within], 'pair arm')


def _base_header(header, *, matrix=False):
    _same(header['event'], 'ready', 'ready event')
    _same(header['repository'], REPO, 'repository')
    _same(header['revision'], REVISION, 'revision')
    equal_int(header['capacity'], 512, 'capacity')
    equal_int(header['chunk'], 32, 'chunk')
    _same(header['profiling'], False, 'profiling')
    _same(header['selected_logit_witness'], False, 'witness')
    for key, wanted in (('caller_allowance_bytes', CALLER), ('helper_allowance_bytes', HELPER),
                        ('limit_bytes', LIMIT)):
        equal_int(header[key], wanted, key)
    load = integer(header['owner_load_admitted_bytes'], 'owner load')
    require(0 < load + CALLER + HELPER <= LIMIT, 'owner load admission')
    for key in ('load_seconds', 'tokenization_seconds'):
        number(header[key], key)
    if matrix:
        _same(header['model_and_codec_manifest_sha256'], CODEC_SHA, 'model/codec')
    return load


def _fingerprint(rows):
    # Ignore order/relabel metadata, retain all work/clock observations per arm.
    fields = ('arm case warmup timing_kind complete_request_seconds first_callback_seconds '
              'last_callback_seconds token_event_seconds report output_ids output_sha256 '
              'history_sha256 state after completed_numerical_seconds prefill prefill_logits_sha256 '
              'prefill_greedy_id decode').split()
    observations = [{k: row[k] for k in fields if k in row} for row in rows]
    encoded = sorted(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)
                     for row in observations)
    return sha(('\n'.join(encoded)).encode())


def validate_primary(rows, manifest, input_ids, output_ids, *, cohort=None, sustained_arm=None):
    token_ids(input_ids, 'primary inputs', 384)
    token_ids(output_ids, 'primary outputs', 128)
    keys(manifest, ('format passed complete configuration requests measured_requests '
                    'measured_complete_request_seconds maximum_with_load_admitted_bytes events '
                    'performance_qualified engine_adoption_qualified scope').split(), 'primary manifest')
    _same(manifest['format'], 'toklet-gemma-submission-bench-manifest-v1', 'primary manifest format')
    _same(manifest['passed'], True, 'manifest passed')
    _same(manifest['complete'], True, 'manifest complete')
    _same(manifest['performance_qualified'], False, 'no performance promotion')
    _same(manifest['engine_adoption_qualified'], False, 'no adoption')
    text(manifest['scope'], 'scope')
    require(type(rows) is list and 2 <= len(rows) <= 257, 'primary journal bounds')
    header = rows[0]
    keys(header, ('event format mode repository revision workload_sha256 input_sha256 input_ids '
                  'expected_output_sha256 input_tokens maximum_output_tokens capacity chunk cache_policy '
                  'profiling selected_logit_witness load_seconds model_setup_seconds tokenization_seconds '
                  'owner_load_admitted_bytes owner_reserved_bytes caller_allowance_bytes helper_allowance_bytes '
                  'limit_bytes clock_scope source_build_hardware_guard_provenance').split(), 'primary ready')
    _same(header['format'], 'toklet-gemma-submission-bench-v1', 'primary format')
    require(json.dumps(manifest['configuration'], sort_keys=True, allow_nan=False) ==
            json.dumps(header, sort_keys=True, allow_nan=False), 'typed manifest/header join')
    load = _base_header(header)
    number(header['model_setup_seconds'], 'model setup')
    for field in ('clock_scope', 'source_build_hardware_guard_provenance'):
        text(header[field], field)
    _same(header['workload_sha256'], WORKLOAD_SHA, 'primary workload')
    _same(header['input_sha256'], ids_sha(input_ids), 'primary input hash')
    keys(header['input_ids'], ('file', 'bytes', 'sha256'), 'ready input bridge')
    _same(header['input_ids']['file'], 'input_ids.u32le', 'input bridge name')
    equal_int(header['input_ids']['bytes'], 1536, 'input bridge bytes')
    _same(header['input_ids']['sha256'], ids_sha(input_ids), 'input bridge SHA')
    _same(header['expected_output_sha256'], ids_sha(output_ids), 'primary output hash')
    equal_int(header['input_tokens'], 384, 'input count')
    equal_int(header['maximum_output_tokens'], 128, 'output cap')
    _same(header['cache_policy'], 'new_session_reset_each_request', 'cache policy')
    reserve = integer(header['owner_reserved_bytes'], 'owner reserve')
    paired = cohort is not None
    require(paired != (sustained_arm is not None), 'choose paired or sustained validation')
    if paired:
        require(type(cohort) is int and cohort in (1, 2), 'cohort')
        keys(header['mode'], ('kind', 'cohort'), 'paired mode')
        _same(header['mode']['kind'], 'paired', 'mode')
        equal_int(header['mode']['cohort'], cohort, 'cohort')
        require(len(rows) == 27, 'primary fixed26 requests')
    else:
        require(sustained_arm in ARMS, 'sustained arm')
        keys(header['mode'], ('kind', 'arm'), 'sustained mode')
        require(header['mode'] == {'kind': 'sustained', 'arm': sustained_arm}, 'sustained mode')
    requests, measured = rows[1:], []
    fields = ('event arm passed generation_error setup_seconds complete_request_seconds first_callback_seconds '
              'last_callback_seconds token_event_seconds report output_ids output_sha256 history_sha256 state '
              'state_matches output_matches postcheck_seconds session_admitted_bytes owner_reserved_bytes '
              'combined_session_admitted_bytes maximum_with_load_admitted_bytes caller_allowance_bytes '
              'helper_allowance_bytes limit_bytes sequence warmup pair_index within_pair order').split()
    for sequence, row in enumerate(requests):
        keys(row, fields, 'primary request')
        equal_int(row['sequence'], sequence, 'sequence')
        for key in ('passed', 'state_matches', 'output_matches'):
            _same(row[key], True, key)
        _same(row['event'], 'request', 'request event')
        require(row['generation_error'] is None, 'generation failed')
        for key in ('setup_seconds', 'postcheck_seconds'):
            number(row[key], key)
        if paired:
            warmup, index = (True, sequence) if sequence < 6 else (False, sequence - 6)
            _pair_position(row, cohort, warmup, index // 2, index % 2)
        else:
            _same(row['arm'], sustained_arm, 'sustained arm')
            _same(row['warmup'], sequence < 3, 'sustained warmup')
            require(all(row[key] is None for key in ('pair_index', 'within_pair', 'order')), 'sustained pairing')
        token_ids(row['output_ids'], 'output IDs', 128)
        require(row['output_ids'] == output_ids and row['output_sha256'] == ids_sha(output_ids)
                and row['history_sha256'] == ids_sha(input_ids + output_ids), 'primary IDs/history')
        keys(row['state'], ('history', 'committed', 'pending'), 'primary state')
        for key, target in (('history', 512), ('committed', 511), ('pending', 1)):
            equal_int(row['state'][key], target, key)
        _core(row['report'], 0, 384, 128, 128, 512, 511, 1, 127)
        _callbacks(row, 128)
        _admission(row, load, reserve)
        if not row['warmup']:
            measured.append(row)
    if paired:
        require(len(measured) == 20, 'primary measured20')
    else:
        require(4 <= len(requests) <= 256, 'sustained request cap')
        durations = [row['complete_request_seconds'] for row in measured]
        require(sum(durations) >= 600 and sum(durations[:-1]) < 600, 'sustained fixed stopping policy')
    equal_int(manifest['requests'], len(requests), 'manifest requests')
    equal_int(manifest['measured_requests'], len(measured), 'manifest measured')
    require(math.isclose(number(manifest['measured_complete_request_seconds'], 'manifest duration'),
                         sum(row['complete_request_seconds'] for row in measured), rel_tol=1e-12, abs_tol=1e-9),
            'manifest duration sum')
    equal_int(manifest['maximum_with_load_admitted_bytes'],
              max(row['maximum_with_load_admitted_bytes'] for row in requests), 'maximum admission')
    return {'header': header, 'requests': requests, 'measured': measured,
            'measurement_sha256': _fingerprint(requests)}


def _goldens(goldens, tokens):
    keys(goldens, ('format repository revision model_and_codec_manifest_sha256 tokens_sha256 '
                  'prompt_sha256 edited_input_sha256 short_input_sha256 edited short '
                  'baseline_candidate_parity').split(), 'goldens')
    for key, expected in (('format', 'toklet-gemma-submission-matrix-goldens-v1'), ('repository', REPO),
                          ('revision', REVISION), ('model_and_codec_manifest_sha256', CODEC_SHA),
                          ('tokens_sha256', ids_sha(tokens)), ('prompt_sha256', ids_sha(P)),
                          ('edited_input_sha256', ids_sha(E)), ('short_input_sha256', ids_sha(tokens[:128]))):
        _same(goldens[key], expected, key)
    _same(goldens['baseline_candidate_parity'], True, 'golden parity assertion')
    for case, count, maximum in (('edited', 15, 4), ('short', 128, 128)):
        value = goldens[case]
        keys(value, ('input_tokens maximum ids output_sha256 history committed pending').split(), 'golden case')
        token_ids(value['ids'], 'golden IDs', maximum)
        for key, target in (('input_tokens', count), ('maximum', maximum), ('history', count + maximum),
                            ('committed', count + maximum - 1), ('pending', 1)):
            equal_int(value[key], target, 'golden ' + key)
        require(value['output_sha256'] == ids_sha(value['ids']), 'golden output hash')


def _matrix_seed(row, case):
    source, output = {'reset': (P, Q), 'edited_prefix': (P, Q),
                      'append': (P[:14], []), 'fully_cached': (P, []),
                      'pending': (P, Q[:1])}.get(case, ([], []))
    pending = int(bool(output))
    _state(row['before'], source + output, len(source + output) - pending, pending)
    if source:
        _core(row['seed_report'], 0, len(source), len(output), len(output), len(source + output),
              len(source + output) - pending, pending, max(0, len(output) - 1))
    else:
        require(row['seed_report'] is None, 'unseeded case has seed Report')


def _matrix_controls(rows):
    names = ('capacity_385_128', 'pre_cancel', 'callback_cancel', 'hidden_stop')
    require(len(rows) == 8, 'eight controls')
    for arm_index, arm in enumerate(ARMS):
        for index, name in enumerate(names):
            row = rows[arm_index * 4 + index]
            fields = ('event case arm passed before after' if index == 0 else
                      'event case arm passed report after').split()
            keys(row, fields, 'control')
            require(row['event'] == 'control' and row['case'] == name and row['arm'] == arm
                    and row['passed'] is True, 'control identity')
            if index == 0:
                _state(row['before'], P + Q, 18, 1)
                _state(row['after'], P + Q, 18, 1)
            elif index == 1:
                _state(row['after'], P + Q, 18, 1)
                _core(row['report'], 0, 0, 0, 0, 19, 18, 1, 0, 'cancelled')
            else:
                _state(row['after'], P + Q[:1], 15, 1)
                _core(row['report'], 0, 15, 1, 1 if index == 2 else 0, 16, 15, 1, 0,
                      'cancelled' if index == 2 else 'stop_token', None if index == 2 else 818)


def _teacher(row, tokens, count):
    _same(row['timing_kind'], 'teacher_forced_evaluations', 'teacher clock')
    require(row['callback_seconds'] is None, 'teacher callback unavailable')
    text(row['scope'], 'teacher scope')
    prefill, decode = row['prefill'], row['decode']
    require(type(prefill) is list and len(prefill) == count // 32, 'teacher prefill count')
    require(type(decode) is list and len(decode) == 128, 'teacher decode count')
    durations = []
    for index, step in enumerate(prefill):
        keys(step, ('rows', 'position', 'seconds'), 'teacher prefill step')
        equal_int(step['rows'], 32, 'teacher prefill rows')
        equal_int(step['position'], 32 * (index + 1), 'teacher prefill position')
        durations.append(number(step['seconds'], 'teacher prefill seconds'))
    digest(row['prefill_logits_sha256'])
    integer(row['prefill_greedy_id'], 'prefill greedy', 262143)
    for index, step in enumerate(decode):
        keys(step, ('step', 'token_id', 'position', 'seconds', 'logits_sha256', 'greedy_id'), 'teacher decode step')
        for key, expected in (('step', index), ('token_id', tokens[count + index]), ('position', count + index + 1)):
            equal_int(step[key], expected, 'teacher ' + key)
        durations.append(number(step['seconds'], 'teacher decode seconds'))
        digest(step['logits_sha256'])
        integer(step['greedy_id'], 'decode greedy', 262143)
    total = number(row['completed_numerical_seconds'], 'teacher numerical seconds')
    require(math.isclose(total, sum(durations), rel_tol=1e-12, abs_tol=1e-9), 'teacher numerical sum')
    require(number(row['wall_with_posttimer_hash_seconds'], 'teacher wall') >= total, 'teacher wall enclosure')
    _state(row['after'], tokens[:count + 128], count + 128, 0)


def _matrix_request(row, case, tokens, goldens, load, positioned):
    common = 'event case arm passed setup_seconds before seed_report admission'.split()
    generation = ('timing_kind complete_request_seconds token_event_seconds first_callback_seconds '
                  'last_callback_seconds report output_ids output_sha256 after').split()
    teacher = ('timing_kind completed_numerical_seconds wall_with_posttimer_hash_seconds prefill '
               'prefill_logits_sha256 prefill_greedy_id decode after callback_seconds scope').split()
    keys(row, common + (teacher if case.startswith('teacher_') else generation) +
         (['warmup', 'trial', 'within_pair', 'order'] if positioned else []), 'matrix request')
    require(row['event'] == 'request' and row['case'] == case and row['passed'] is True
            and row['arm'] in ARMS, 'matrix request identity')
    number(row['setup_seconds'], 'matrix setup')
    keys(row['admission'], ('session_admitted_bytes owner_reserved_bytes caller_allowance_bytes '
                           'helper_allowance_bytes combined_session_admitted_bytes '
                           'maximum_with_load_admitted_bytes limit_bytes').split(), 'matrix admission')
    _admission(row['admission'], load)
    _matrix_seed(row, case)
    if case.startswith('teacher_'):
        _teacher(row, tokens, 128 if case == 'teacher_128_128' else 384)
        return
    base, output, reused, prefill, decode = {
        'reset': (P, Q, 0, 15, 3), 'append': (P, Q, 14, 1, 3),
        'fully_cached': (P, Q, 15, 0, 3), 'pending': (P + Q[:1], Q[1:], 15, 1, 2),
        'edited_prefix': (E, goldens['edited']['ids'], 0, 15, 3),
        'short_128_128': (tokens[:128], goldens['short']['ids'], 0, 128, 127),
        'boundary_prefill_512_0': (tokens[:512], [], 0, 512, 0)}[case]
    _same(row['timing_kind'], 'generation', 'generation clock')
    token_ids(row['output_ids'], 'matrix IDs', len(output))
    require(row['output_ids'] == output and row['output_sha256'] == ids_sha(output), 'matrix output IDs/hash')
    history = base + output
    pending = int(bool(output))
    _state(row['after'], history, len(history) - pending, pending)
    _core(row['report'], reused, prefill, len(output), len(output), len(history),
          len(history) - pending, pending, decode)
    _callbacks(row, len(output))


def _matrix_pair(left, right, case):
    require(left['before'] == right['before'] and left['after'] == right['after'], 'matrix pair state')
    if case.startswith('teacher_'):
        require(left['prefill_logits_sha256'] == right['prefill_logits_sha256'] and
                left['prefill_greedy_id'] == right['prefill_greedy_id'], 'teacher prefill parity')
        for x, y in zip(left['decode'], right['decode']):
            require(all(x[key] == y[key] for key in ('step', 'token_id', 'position', 'logits_sha256', 'greedy_id')),
                    'teacher complete decode parity')
    else:
        require(left['output_ids'] == right['output_ids'] and left['output_sha256'] == right['output_sha256'],
                'matrix generation pair')


def validate_matrix(rows, manifest, tokens, goldens, *, cohort=None, prepare=False):
    token_ids(tokens, 'T fixture', 1025)
    _goldens(goldens, tokens)
    keys(manifest, ('format passed complete prepare_goldens cohort requests cases warmup_pairs measured_pairs '
                    'events goldens_sha256 performance_qualified engine_adoption_qualified scope').split(), 'matrix manifest')
    _same(manifest['format'], 'toklet-gemma-submission-matrix-manifest-v1', 'matrix manifest format')
    for key, expected in (('passed', True), ('complete', True), ('prepare_goldens', prepare),
                          ('performance_qualified', False), ('engine_adoption_qualified', False)):
        _same(manifest[key], expected, key)
    require(manifest['cases'] == list(CASES), 'matrix case inventory')
    text(manifest['scope'], 'matrix manifest scope')
    require(type(rows) is list and len(rows) == (5 if prepare else 243), 'matrix journal fixed count')
    header = rows[0]
    keys(header, ('event format prepare_goldens cohort repository revision model_and_codec_manifest_sha256 '
                  'tokens_sha256 prompt_ids prompt_sha256 edited_input_sha256 short_input_sha256 goldens_sha256 '
                  'capacity chunk stop_policy load_seconds tokenization_seconds owner_load_admitted_bytes '
                  'caller_allowance_bytes helper_allowance_bytes limit_bytes profiling selected_logit_witness scope').split(), 'matrix ready')
    _same(header['format'], 'toklet-gemma-submission-matrix-v1', 'matrix format')
    load = _base_header(header, matrix=True)
    _same(header['prepare_goldens'], prepare, 'prepare mode')
    for key, wanted in (('tokens_sha256', ids_sha(tokens)), ('prompt_sha256', ids_sha(P)),
                        ('edited_input_sha256', ids_sha(E)), ('short_input_sha256', ids_sha(tokens[:128]))):
        _same(header[key], wanted, key)
    token_ids(header['prompt_ids'], 'P', 15)
    require(header['prompt_ids'] == P, 'prompt IDs')
    _same(header['stop_policy'], 'empty for fixed numerical cases; hidden-stop control818 only', 'stop policy')
    text(header['scope'], 'matrix scope')
    require(manifest['cohort'] == header['cohort'] and manifest['goldens_sha256'] == header['goldens_sha256'],
            'matrix header/manifest join')
    requests, measured = [], []
    if prepare:
        require(cohort is None and header['cohort'] is None and header['goldens_sha256'] is None, 'prepare arguments')
        for index, case in enumerate(('edited_prefix', 'short_128_128')):
            pair = rows[1 + 2 * index:3 + 2 * index]
            for within, row in enumerate(pair):
                _same(row['arm'], ARMS[within], 'prepare arm')
                _matrix_request(row, case, tokens, goldens, load, False)
                requests.append(row)
            _matrix_pair(*pair, case)
    else:
        require(type(cohort) is int and cohort in (1, 2), 'matrix cohort')
        equal_int(header['cohort'], cohort, 'matrix cohort')
        equal_int(manifest['cohort'], cohort, 'manifest cohort')
        digest(header['goldens_sha256'])
        _matrix_controls(rows[1:9])
        offset = 9
        for warmup, count in ((True, 3), (False, 10)):
            for trial in range(count):
                for case in CASES:
                    pair = rows[offset:offset + 2]
                    for within, row in enumerate(pair):
                        _pair_position(row, cohort, warmup, trial, within, matrix=True)
                        _matrix_request(row, case, tokens, goldens, load, True)
                        requests.append(row)
                        if not warmup:
                            measured.append(row)
                    _matrix_pair(*pair, case)
                    offset += 2
    for key, target in (('requests', 4 if prepare else 234), ('warmup_pairs', 0 if prepare else 3),
                        ('measured_pairs', 0 if prepare else 10)):
        equal_int(manifest[key], target, key)
    return {'header': header, 'requests': requests, 'measured': measured,
            'measurement_sha256': _fingerprint(requests)}


def summarize(values):
    require(type(values) is list and values and all(type(x) in (int, float) and math.isfinite(x) for x in values),
            'finite nonempty statistic')
    median = statistics.median(values)
    return {'n': len(values), 'raw': values, 'median': median,
            'mad': statistics.median(abs(x - median) for x in values), 'range': [min(values), max(values)]}


def statistics_for_pairs(a, b):
    require(type(a) is list and type(b) is list and len(a) == len(b) == 10
            and all(number(x, 'paired latency', True) for x in a + b), 'ten complete pairs')
    reductions = [100 * (1 - y / x) for x, y in zip(a, b)]
    rng = random.Random(1)
    boot = sorted(statistics.median(rng.choices(reductions, k=10)) for _ in range(10000))
    return {'per_tile': summarize(a), 'batched_single_row': summarize(b),
            'marginal_median_reduction_pct': 100 * (1 - statistics.median(b) / statistics.median(a)),
            'paired_reduction_pct': reductions, 'paired_median_reduction_pct': statistics.median(reductions),
            'paired_median_bootstrap_95_pct': [boot[250], boot[9749]]}


def _metrics(row):
    if row.get('timing_kind') == 'teacher_forced_evaluations':
        prefill = sum(x['seconds'] for x in row['prefill'])
        decode = sum(x['seconds'] for x in row['decode'])
        return {'completed_numerical_seconds': row['completed_numerical_seconds'],
                'prefill_evaluation_seconds': prefill, 'decode_evaluation_seconds': decode,
                'prefill_rows_per_second': (sum(x['rows'] for x in row['prefill']) / prefill if prefill > 0 else None),
                'decode_evaluations_per_second': (128 / decode if decode > 0 else None),
                'first_callback_seconds': None, 'callback_interval_tokens_per_second': None}
    core, events = row['report'], row['token_event_seconds']
    return {'complete_request_seconds': row['complete_request_seconds'],
            'first_callback_seconds': row['first_callback_seconds'],
            'callback_interval_tokens_per_second': ((len(events) - 1) / (events[-1] - events[0])
                if len(events) >= 2 and events[-1] > events[0] else None),
            'decode_evaluations_per_second': (core['decode_evaluations'] / core['decode_seconds']
                if core['decode_evaluations'] > 0 and core['decode_seconds'] > 0 else None),
            'prefill_tokens_per_second': (core['prefill_tokens'] / core['prefill_seconds']
                if core['prefill_tokens'] > 0 and core['prefill_seconds'] > 0 else None)}


def _paired_report(rows):
    arms = {arm: [row for row in rows if row['arm'] == arm] for arm in ARMS}
    require(all(len(v) == 10 for v in arms.values()), 'ten samples per arm')
    metrics = {arm: [_metrics(row) for row in value] for arm, value in arms.items()}
    result = {}
    for name in metrics[ARMS[0]][0]:
        a, b = ([m[name] for m in metrics[arm]] for arm in ARMS)
        if any(x is None or x <= 0 for x in a + b):
            result[name] = {'available': False, 'reason': 'zero work/denominator, unavailable callback, or nonpositive latency',
                            'per_tile_raw': a, 'batched_raw': b}
        else:
            result[name] = {'available': True, **statistics_for_pairs(a, b),
                            'interpretation': ('positive reduction is less time' if name.endswith('seconds')
                                               else 'positive reduction is lower rate; higher rate is better')}
    return result


def _guard(bundle, binary, expected_args):
    guard = bundle['guard']
    keys(guard, ('plan', 'result', 'memory', 'stdout', 'stderr'), 'guard pins')
    plan, result = _json_pin(guard['plan']), _json_pin(guard['result'])
    for key, target in (('limit_bytes', LIMIT), ('helper_reserve_bytes', HELPER),
                        ('configured_guest_reserve_bytes', 0), ('total_reserved_bytes', HELPER)):
        equal_int(plan[key], target, 'guard plan ' + key)
    number(plan['timeout_seconds'], 'guard timeout', True)
    equal_int(plan['log_limit_bytes'], 16 * MIB, 'guard log limit')
    equal_int(plan['disk_floor_bytes'], 2 * 1024 * MIB, 'guard disk floor')
    require(type(plan['command']) is list and plan['command'] and plan['command'][0] == binary['path'],
            'guard binary path differs from pinned build')
    text(plan['cwd'], 'guard cwd')
    require(Path(plan['cwd']).is_absolute(), 'guard cwd must be absolute')
    require(plan['command'][1:] == expected_args, 'guard exact argv differs')
    _same(result['passed'], True, 'guard passed')
    equal_int(result['exit_code'], 0, 'guard exit')
    require(result['failure'] is None and result['owned_live_after_cleanup'] == []
            and result['cleanup_errors'] == [], 'guard failure/cleanup')
    for key, target in (('helper_reserve_bytes', HELPER), ('configured_guest_reserve_bytes', 0)):
        equal_int(result[key], target, key)
    number(result['elapsed_seconds'], 'guard elapsed', True)
    require(integer(result['sample_count'], 'guard samples') > 0, 'missing guard samples')
    for key in ('maximum_observed_charge_with_reserve', 'terminal_direct_pair_rss_charge_with_reserve'):
        require(integer(result[key], key) >= HELPER, 'guard charge missing reserve')
    gate = result['last_runtime_gate']
    require(type(gate) is dict and gate['passed'] is True and gate['lowPowerMode'] is False
            and gate['power'] == "Now drawing from 'AC Power'", 'last runtime gate')
    equal_int(gate['thermalState'], 0, 'last thermal state')
    require(integer(gate['checked_uptime_ns'], 'last checked uptime', 2**64 - 1) > 0, 'runtime gate timestamp')
    for key in ('stdout', 'stderr'):
        read_pin(guard[key], 16 * MIB, retain=False)
        require(guard[key]['sha256'] == result[key + '_sha256'], 'guard log hash join')
    read_pin(guard['memory'], 64 * MIB, retain=False)
    read_pin(bundle['quiet_barrier'], MIB, retain=False)
    return {'last_runtime_gate': gate, 'maximum_observed_charge_with_reserve':
            result['maximum_observed_charge_with_reserve'], 'pins': guard,
            'quiet_barrier': bundle['quiet_barrier'], 'enforcement_independently_verified': False}


def _bundle(bundle, *, matrix=False):
    keys(bundle, ('manifest', 'events', 'guard', 'quiet_barrier'), 'bundle')
    manifest = _json_pin(bundle['manifest'])
    require(Path(bundle['manifest']['path']).name == 'manifest.json'
            and Path(bundle['events']['path']).name == 'events.jsonl'
            and Path(bundle['manifest']['path']).parent == Path(bundle['events']['path']).parent,
            'bundle fixed paths')
    raw = read_pin(bundle['events'], (8 if matrix else 2) * MIB)
    keys(manifest['events'], ('file', 'bytes', 'sha256'), 'manifest events')
    require(manifest['events'] == {'file': 'events.jsonl', 'bytes': len(raw), 'sha256': sha(raw)}, 'manifest/journal join')
    equal_int(manifest['events']['bytes'], len(raw), 'events byte count')
    require(raw.endswith(b'\n'), 'journal complete newline')
    lines = raw.splitlines()
    require(len(lines) <= (256 if matrix else 257) and all(0 < len(line) <= (32768 if matrix else 7168)
                                                       for line in lines), 'journal line/row bounds')
    return [parse(line) for line in lines], manifest


def report(inputs_path, *, expected_sha256):
    digest(expected_sha256)
    raw = _read(Path(inputs_path).absolute(), MIB)
    require(sha(raw) == expected_sha256, 'sealed input SHA differs')
    spec = parse(raw)
    keys(spec, ('format', 'root_provenance', 'inputs', 'golden_preparation', 'bundles'), 'sealed input')
    _same(spec['format'], 'toklet-gemma-submission-report-inputs-v1', 'input format')
    provenance = spec['root_provenance']
    keys(provenance, ('source_manifest', 'primary_build', 'matrix_build', 'hardware', 'runtime_enforcement'), 'root provenance')
    # Hardware/enforcement/source/quiet receipts are root-owned pinned references.
    # Their future schemas are not invented here; only the known binary receipt
    # fields below are interpreted for a concrete guard-command join.
    for pin in provenance.values():
        read_pin(pin, MIB, retain=False)
    binaries = {}
    for name in ('primary', 'matrix'):
        value = _json_pin(provenance[name + '_build'])
        require(type(value) is dict and all(key in value for key in ('path', 'bytes', 'sha256')),
                'existing release binary receipt requires path/bytes/sha256')
        binary = {key: value[key] for key in ('path', 'bytes', 'sha256')}
        read_pin(binary, 64 * MIB, retain=False)
        if 'source_freeze_sha256' in value:
            require(value['source_freeze_sha256'] == provenance['source_manifest']['sha256'], 'build/source pin join')
        binaries[name] = binary
    data = spec['inputs']
    keys(data, ('primary_workload', 'primary_input_ids', 'matrix_tokens', 'matrix_goldens'), 'input pins')
    workload_raw = read_pin(data['primary_workload'], 16384)
    primary_raw = read_pin(data['primary_input_ids'], 1536)
    token_raw = read_pin(data['matrix_tokens'], 4100)
    require(sha(workload_raw) == WORKLOAD_SHA and len(primary_raw) == 1536
            and sha(primary_raw) == PRIMARY_INPUT_SHA and len(token_raw) == 4100
            and sha(token_raw) == TOKENS_SHA, 'fixed input authentication')
    workload = parse(workload_raw)
    input_ids = list(struct.unpack('<384I', primary_raw))
    tokens = list(struct.unpack('<1025I', token_raw))
    output_ids = token_ids(workload['expected_output']['tokens'], 'frozen expected IDs', 128)
    require(workload['input_tokens_sha256'] == ids_sha(input_ids), 'descriptor/primary bridge join')
    goldens = _json_pin(data['matrix_goldens'], 65536)
    prep_rows, prep_manifest = _bundle(spec['golden_preparation'], matrix=True)
    prepared = validate_matrix(prep_rows, prep_manifest, tokens, goldens, prepare=True)
    require(Path(data['matrix_goldens']['path']) == Path(spec['golden_preparation']['manifest']['path']).parent / 'goldens.json',
            'prepared golden output path')
    bundles = spec['bundles']
    slots = ('primary_1', 'primary_2', 'matrix_1', 'matrix_2', 'sustained_per_tile', 'sustained_batched')
    keys(bundles, slots, 'six measurement bundles')
    paths = [bundle['events']['path'] for bundle in bundles.values()]
    require(len(set(paths)) == 6, 'reused bundle path')
    validated, guard_reports = {}, {}
    # Guard argv follows the frozen binaries' existing parsers, including required argument order.
    for slot in ('golden_preparation',) + slots:
        bundle = spec[slot] if slot == 'golden_preparation' else bundles[slot]
        matrix = slot.startswith('matrix') or slot == 'golden_preparation'
        rows, manifest = (prep_rows, prep_manifest) if slot == 'golden_preparation' else _bundle(bundle, matrix=matrix)
        plan = _json_pin(bundle['guard']['plan'])
        command = plan['command']
        require(type(command) is list and all(type(x) is str for x in command), 'guard command strings')
        # Model directory is root supplied but must be identical across every launch.
        require(len(command) >= 3 and command[1] == '--model', 'fixed model argument')
        model = command[2]
        require(Path(model).is_absolute(), 'absolute model directory')
        if validated:
            require(model == next(iter(validated.values()))['model_directory'], 'different model directories')
        output = str(Path(bundle['manifest']['path']).parent)
        if matrix:
            args = ['--model', model, '--tokens', data['matrix_tokens']['path'], '--output', output]
            if slot == 'golden_preparation':
                args += ['--prepare-goldens']
                value = prepared
            else:
                cohort = int(slot[-1])
                args += ['--goldens', data['matrix_goldens']['path'], '--goldens-sha256',
                         data['matrix_goldens']['sha256'], '--cohort', str(cohort)]
                value = validate_matrix(rows, manifest, tokens, goldens, cohort=cohort)
                require(value['header']['goldens_sha256'] == data['matrix_goldens']['sha256'], 'capture/golden raw pin')
        else:
            args = ['--model', model, '--workload', data['primary_workload']['path'], '--output', output]
            bridge = {'path': str(Path(output) / 'input_ids.u32le'), 'bytes': 1536,
                      'sha256': data['primary_input_ids']['sha256']}
            require(read_pin(bridge, 1536) == primary_raw, 'per-bundle exported bridge differs')
            if slot.startswith('primary'):
                cohort = int(slot[-1])
                args += ['--cohort', str(cohort)]
                value = validate_primary(rows, manifest, input_ids, output_ids, cohort=cohort)
            else:
                arm = ARMS[0] if slot == 'sustained_per_tile' else ARMS[1]
                args += ['--sustained-arm', arm]
                value = validate_primary(rows, manifest, input_ids, output_ids, sustained_arm=arm)
        guard_reports[slot] = _guard(bundle, binaries['matrix' if matrix else 'primary'], args)
        value['model_directory'] = model
        validated[slot] = value
    for family in ('primary', 'matrix'):
        require(validated[family + '_1']['measurement_sha256'] != validated[family + '_2']['measurement_sha256'],
                'same normalized measurement collection cannot confirm itself')
    paired = {}
    for family in ('primary', 'matrix'):
        for cohort in (1, 2):
            slot = f'{family}_{cohort}'
            rows = validated[slot]['measured']
            paired[slot] = {case: _paired_report([row for row in rows if family == 'primary' or row['case'] == case])
                            for case in (('primary_384_128',) if family == 'primary' else CASES)}
    sustained = {}
    for slot in ('sustained_per_tile', 'sustained_batched'):
        metrics = [_metrics(row) for row in validated[slot]['measured']]
        sustained[slot] = {key: (summarize([m[key] for m in metrics]) if all(m[key] is not None for m in metrics)
                                 else {'available': False, 'reason': 'unavailable denominator/work',
                                       'raw': [m[key] for m in metrics]}) for key in metrics[0]}
    sustained_ratios = {}
    for metric, a in sustained['sustained_per_tile'].items():
        b = sustained['sustained_batched'][metric]
        sustained_ratios[metric] = ({'marginal_median_reduction_pct': 100 * (1 - b['median'] / a['median']),
                                     'paired': False, 'confidence_interval': None}
                                    if 'median' in a and 'median' in b and a['median'] > 0 else
                                    {'available': False, 'reason': 'unavailable metric/zero denominator'})
    primary_signals = {f'cohort{n}': paired[f'primary_{n}']['primary_384_128']['complete_request_seconds']
                       ['marginal_median_reduction_pct'] >= 5 for n in (1, 2)}
    return {'format': 'toklet-gemma-submission-report-v1', 'verification_complete': True,
            'verification_scope': 'Fixed data/schema/pin and recorded guard-result joins only.',
            'performance_qualified': False, 'engine_adoption_qualified': False, 'full_retention_qualified': False,
            'inputs_sha256': expected_sha256, 'inputs': spec,
            'root_provenance': {'pins': provenance, 'independently_verified': False,
                'unestablished_semantics': ['Actual host enforcement, continuous power/thermal conditions, quiet launch and hardware/source/build authenticity require root and final Astra review; pins alone do not establish these.',
                    'Guard memory is sampled; short-lived/unobserved peaks and continuous allocator caps are not proved.',
                    'Logical row identities and whole-collection fingerprints cannot prove absence of every possible partial timing reuse.']},
            'guard_recorded_results': guard_reports, 'paired': paired, 'sustained': sustained,
            'sustained_marginal_comparison': sustained_ratios,
            'validated_inventory': {key: {'requests': len(value['requests']),
                'measured_requests': len(value['measured']),
                'controls': 8 if key.startswith('matrix_') else 0,
                'measurement_scope': ('teacher evaluations and generation kept distinct' if key.startswith('matrix_')
                                      else 'ordinary completed generation requests')}
                for key, value in validated.items()},
            'primary_marginal_median_ge5_observed': primary_signals,
            'measurement_sha256': {key: value['measurement_sha256'] for key, value in validated.items()},
            'limits': ['Warmups, preparation and controls excluded from statistics; cohorts/cases never pooled.',
                       'No new noise threshold/N, automatic retention, tail/UI/first-visible/energy claim.',
                       'Teacher clocks are completed evaluations, not generation TTFT. Rates have explicit work/denominator scope.',
                       'Original numerical/full quality/required workload/sustained/observer/adoption/Astra gates remain separate; observer TTFT1% stays unqualified/default off.']}


def markdown(value):
    lines = ['# Gemma ordinary submission measurements', '',
             'Data/schema/pin validation complete. Performance, adoption and full retention remain unqualified.', '',
             'Root hardware, enforcement and quiet conditions are separately pinned and not independently verified here.', '',
             '| Cohort | Case | Completed metric | Per-tile median | Batched median | Reduction % |',
             '| --- | --- | --- | ---: | ---: | ---: |']
    for cohort, cases in value['paired'].items():
        for case, metrics in cases.items():
            name = 'completed_numerical_seconds' if case.startswith('teacher_') else 'complete_request_seconds'
            metric = metrics[name]
            if metric['available']:
                lines.append(f"| {cohort} | {case} | {name} | {metric['per_tile']['median']:.6g} | "
                             f"{metric['batched_single_row']['median']:.6g} | {metric['marginal_median_reduction_pct']:.4f} |")
    lines += ['', 'JSON retains unscaled MAD, raw pairs and paired-bootstrap intervals. Sustained arms remain separate descriptive series.', '',
              *['- ' + item for item in value['limits']], '']
    return '\n'.join(lines)


def _publish(path, raw, bound):
    require(len(raw) <= bound, 'output byte bound')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    created = False
    try:
        args.output.mkdir(exist_ok=False)
        created = True
        value = report(args.inputs, expected_sha256=args.inputs_sha256)
        raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
        rendered = markdown(value).encode()
        require(len(raw) <= MIB and len(rendered) <= 65536, 'bounded final outputs')
        _publish(args.output / 'report.json', raw, MIB)
        _publish(args.output / 'report.md', rendered, 65536)
    except (ValueError, TypeError, KeyError, OSError, ArithmeticError, RecursionError) as error:
        if created:
            try:
                _publish(args.output / 'failure.json', (json.dumps({'verification_complete': False,
                          'reason': str(error)[:2048], 'performance_qualified': False}) + '\n').encode(), 4096)
            except OSError:
                pass
        raise SystemExit(2) from error
    print(json.dumps({'verification_complete': True, 'performance_qualified': False}))


if __name__ == '__main__':
    main()
