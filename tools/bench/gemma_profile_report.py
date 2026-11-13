"""Validate one official Gemma BF16 diagnostic bundle; aggregate no overlaps."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

MIB = 1024 ** 2
LIMIT = 6 * 1024 ** 3
U64 = 2 ** 64 - 1
REPOSITORY = 'google/gemma-4-E2B-it'
REVISION = '70af34e20bd4b7a91f0de6b22675850c43922a03'
HOST_FIELDS = ('host_staging_seconds', 'host_encoding_seconds', 'host_commit_seconds',
               'host_wait_seconds', 'host_validation_and_copy_seconds')
COUNTER_FIELDS = ('dispatches_encoded', 'commands_submitted', 'host_wait_calls')
# Existing instrument policies: resident/profile.rs frame partition, and
# tools/experiments/metal_trace_analysis.py paired-clock serialization replay.
HOST_TOL = 1e-6
CLOCK_REL_TOL, CLOCK_ABS_TOL = 1e-13, 1e-12
LAYOUT = ('row-major [out,in], original little-endian BF16, two coefficients/u32; '
          'FP32 activations/output')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected, label):
    need(type(value) is dict and set(value) == set(expected.split()), label + ' schema')


def uint(value, maximum=U64, minimum=0):
    need(type(value) is int and minimum <= value <= maximum, 'unsigned integer bound')
    return value


def seconds(value):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0,
         'finite nonnegative seconds required')
    return value


def _sum(values):
    try:
        value = math.fsum(values)
    except OverflowError as error:
        raise ValueError('duration sum overflow') from error
    need(math.isfinite(value), 'nonfinite aggregate')
    return value


def digest(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'SHA256 format')
    return value


def exact(value, expected, label):
    need(type(value) is type(expected) and value == expected, label)


def product(*values):
    result = 1
    for value in values:
        result *= uint(value)
        need(result <= U64, 'byte arithmetic overflow')
    return result


def _object(items):
    result = {}
    for key, value in items:
        need(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _float(text):
    value = float(text)
    need(math.isfinite(value), 'nonfinite JSON number')
    return value


def _read(folder, name, bound):
    file = folder / name
    try:
        before = file.lstat()
        need(stat.S_ISREG(before.st_mode) and before.st_size <= bound, 'bounded regular file required')
        descriptor = os.open(file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            need(identity(before) == identity(os.fstat(stream.fileno())), 'file changed before read')
            raw = stream.read(bound + 1)
            need(len(raw) == before.st_size and len(raw) <= bound, 'file length changed')
            need(identity(before) == identity(os.fstat(stream.fileno())) == identity(file.lstat()),
                 'file changed during read')
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_object, parse_float=_float,
                           parse_constant=lambda text: need(False, 'nonfinite JSON: ' + text))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('invalid bundle file: ' + name) from error
    return value, dict(file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def interval_union(intervals):
    merged = []
    for begin, end in sorted(intervals):
        seconds(begin)
        seconds(end)
        need(begin <= end, 'reversed interval')
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    return _sum(end - begin for begin, end in merged)


def _specifications():
    """Fixed source geometry; inventory values are checked, not self-authenticating."""
    prefix = 'model.language_model.'
    entries = [('token_embedding', None, 'embed_tokens.weight', 262144, 1536),
               ('per_layer_embedding', None, 'embed_tokens_per_layer.weight', 262144, 8960),
               ('per_layer_input_projection', None, 'per_layer_model_projection.weight', 8960, 1536)]
    for layer in range(35):
        dim = 512 if layer % 5 == 4 else 256
        ff = 6144 if layer < 15 else 12288
        values = [('query', 'self_attn.q_proj', 8 * dim, 1536)]
        if layer < 15:
            values += [('key', 'self_attn.k_proj', dim, 1536), ('value', 'self_attn.v_proj', dim, 1536)]
        values += [('attention_output', 'self_attn.o_proj', 1536, 8 * dim),
                   ('feed_forward_gate', 'mlp.gate_proj', ff, 1536),
                   ('feed_forward_up', 'mlp.up_proj', ff, 1536),
                   ('feed_forward_down', 'mlp.down_proj', 1536, ff),
                   ('per_layer_gate', 'per_layer_input_gate', 256, 1536),
                   ('per_layer_projection', 'per_layer_projection', 1536, 256)]
        entries += [(role, layer, f'layers.{layer}.{suffix}.weight', rows, columns)
                    for role, suffix, rows, columns in values]
    entries.append(('output_head', None, 'embed_tokens.weight', 262144, 1536))
    slots = {name: index for index, name in enumerate(sorted({row[2] for row in entries}))}
    return {(role, layer): (prefix + name, slots[name], rows, columns)
            for role, layer, name, rows, columns in entries}


def _id(value):
    keys(value, 'role layer', 'operation ID')
    need(type(value['role']) is str, 'operation role')
    layer = value['layer']
    if layer is not None:
        uint(layer, 34)
    return value['role'], layer


def _inventory(value):
    keys(value, 'payload_authenticated resident_matrix_owners resident_gpu_tile_owners '
         'vector_owners retained_weight_owner_bytes operations', 'inventory')
    for name, expected in [('payload_authenticated', True), ('resident_matrix_owners', 277),
                           ('resident_gpu_tile_owners', 349), ('vector_owners', 262)]:
        exact(value[name], expected, 'inventory owner count/availability')
    uint(value['retained_weight_owner_bytes'], LIMIT, 1)
    expected = _specifications()
    need(type(value['operations']) is list and len(value['operations']) == 279, 'operation inventory')
    result = {}
    for op in value['operations']:
        keys(op, 'id source_matrix_slot source_tensor source_rows source_columns source_bytes '
             'prefill_rows decode_rows execution coefficient_layout kernel_entry projection_tiles '
             'predicted_commands_per_chunk predicted_dispatches_per_chunk predicted_host_waits_per_chunk '
             'modelled_prefill_output_host_bytes modelled_decode_output_host_bytes', 'selected operation')
        identity = _id(op['id'])
        need(identity in expected and identity not in result, 'unknown/duplicate operation')
        name, slot, rows, columns = expected[identity]
        role, layer = identity
        gather = role in ('token_embedding', 'per_layer_embedding')
        tiles = 0 if gather else (rows + 8191) // 8192
        prefill = 1 if role == 'output_head' or layer is not None and layer >= 15 else 32
        execution = ('resident_bf16_cpu_gather' if role == 'token_embedding' else
                     'authenticated_file_bf16_cpu_gather' if gather else 'tiled_bf16_metal_projection')
        values = dict(source_matrix_slot=slot, source_tensor=name, source_rows=rows,
                      source_columns=columns, source_bytes=product(rows, columns, 2),
                      prefill_rows=prefill, decode_rows=1, execution=execution,
                      coefficient_layout=LAYOUT, kernel_entry=None if gather else 'gemma_bf16_product',
                      projection_tiles=tiles, predicted_commands_per_chunk=tiles,
                      predicted_dispatches_per_chunk=tiles, predicted_host_waits_per_chunk=tiles,
                      modelled_prefill_output_host_bytes=0 if gather else product(rows, prefill, 4),
                      modelled_decode_output_host_bytes=0 if gather else product(rows, 4))
        for key, target in values.items():
            exact(op[key], target, 'inventory ' + key)
        result[identity] = op
    need(set(result) == set(expected), 'complete operation identities')
    return result


def _manifest(value):
    keys(value, 'format passed diagnostic_only timing_qualified model workload capture parity plain '
         'profiled inventory profile memory clock_scope', 'manifest')
    for key, expected in [('format', 'toklet-gemma-profile-v1'), ('passed', True),
                          ('diagnostic_only', True), ('timing_qualified', False)]:
        exact(value[key], expected, 'manifest ' + key)
    model = value['model']
    keys(model, 'repository revision', 'model')
    exact(model['repository'], REPOSITORY, 'selected model')
    exact(model['revision'], REVISION, 'selected revision')
    need(type(value['clock_scope']) is str and 0 < len(value['clock_scope'].encode()) <= 4096, 'clock scope')
    w = value['workload']
    keys(w, 'sha256 conversation_sha256 input_tokens_sha256 input_tokens maximum_output_tokens capacity chunk', 'workload')
    for key in ('sha256', 'conversation_sha256', 'input_tokens_sha256'):
        digest(w[key])
    input_rows = uint(w['input_tokens'], 512, 1)
    maximum = uint(w['maximum_output_tokens'], 256, 1)
    need(input_rows + maximum <= 512, 'workload capacity')
    exact(w['capacity'], 512, 'capacity')
    exact(w['chunk'], 32, 'chunk')
    core_keys = ('reason reused_tokens prefill_tokens sampled_tokens emitted_tokens history_tokens '
                 'evaluated_tokens pending_tokens first_token_seconds prefill_seconds decode_seconds '
                 'decode_evaluations sampling_seconds total_seconds')
    for label in ('plain', 'profiled'):
        core = value[label]
        keys(core, core_keys, 'completed report')
        for key in core_keys.split():
            if key.endswith('_tokens') or key == 'decode_evaluations':
                uint(core[key], 512)
        reason = core['reason']
        need(type(reason) is dict and reason.get('kind') in ('output_limit', 'stop_token'), 'completed reason')
        emitted = core['emitted_tokens']
        need(emitted > 0, 'nonempty frozen output required')
        if reason['kind'] == 'output_limit':
            keys(reason, 'kind', 'output limit')
            need(emitted == maximum and core['sampled_tokens'] == emitted, 'output limit counts')
        else:
            keys(reason, 'kind token_id', 'hidden stop')
            need(uint(reason['token_id'], 262143) in (1, 106, 50) and emitted < maximum
                 and core['sampled_tokens'] == emitted + 1, 'hidden stop counts')
        for key, target in dict(reused_tokens=0, prefill_tokens=input_rows,
                                decode_evaluations=core['sampled_tokens'] - 1,
                                history_tokens=input_rows + core['sampled_tokens'],
                                evaluated_tokens=input_rows + core['sampled_tokens'] - 1,
                                pending_tokens=1).items():
            exact(core[key], target, 'core ' + key)
        for key in ('prefill_seconds', 'decode_seconds', 'sampling_seconds', 'total_seconds'):
            seconds(core[key])
        need(_sum(core[k] for k in ('prefill_seconds', 'decode_seconds', 'sampling_seconds'))
             <= core['total_seconds'] + 1e-9, 'core phase enclosure')
        need(seconds(core['first_token_seconds']) <= core['total_seconds'], 'first-token enclosure')
    a, b = value['plain'], value['profiled']
    for key in core_keys.split():
        if not key.endswith('_seconds'):
            need(a[key] == b[key], 'plain/profiled core parity')
    cap = value['capture']
    keys(cap, 'evaluation_ordinal request_id position input_rows output_rows committed_positions', 'capture')
    ordinal = uint(cap['evaluation_ordinal'], 512, 1)
    need(type(cap['request_id']) is list and len(cap['request_id']) == 16, 'request ID')
    for byte in cap['request_id']:
        uint(byte, 255)
    prefill_chunks = (input_rows + 31) // 32
    evaluations = prefill_chunks + a['decode_evaluations']
    need(ordinal <= evaluations, 'unavailable selected evaluation')
    position = 32 * (ordinal - 1) if ordinal <= prefill_chunks else input_rows + ordinal - prefill_chunks - 1
    rows = min(32, input_rows - position) if ordinal <= prefill_chunks else 1
    for key, target in dict(position=position, input_rows=rows, output_rows=1, committed_positions=position + rows).items():
        exact(cap[key], target, 'capture ' + key)
    parity = value['parity']
    keys(parity, 'selected_logits_bitwise selected_logits_count selected_logits_sha256 output_ids history report_counts', 'parity')
    for key in ('selected_logits_bitwise', 'output_ids', 'history', 'report_counts'):
        exact(parity[key], True, 'producer parity witness')
    exact(parity['selected_logits_count'], 262144, 'full selected logits')
    digest(parity['selected_logits_sha256'])
    memory = value['memory']
    keys(memory, 'limit_bytes owner_limit_bytes caller_allowance_bytes helper_allowance_bytes diagnostic_allowance_bytes '
         'plain_admitted_bytes profiled_admitted_bytes owner_reserved_bytes combined_profiled_admitted_bytes', 'memory')
    for amount in memory.values():
        uint(amount, LIMIT)
    for key, target in dict(limit_bytes=LIMIT, owner_limit_bytes=LIMIT - 48 * MIB,
                            caller_allowance_bytes=16 * MIB, helper_allowance_bytes=32 * MIB,
                            diagnostic_allowance_bytes=64 * MIB).items():
        exact(memory[key], target, 'memory ' + key)
    need(0 < memory['plain_admitted_bytes'] <= memory['owner_limit_bytes']
         and memory['profiled_admitted_bytes'] == memory['plain_admitted_bytes'] + 64 * MIB,
         'profile constructor admission')
    need(memory['combined_profiled_admitted_bytes'] == memory['profiled_admitted_bytes']
         + memory['owner_reserved_bytes'] + 48 * MIB <= LIMIT, 'combined admission')
    return evaluations


def _frame(frame, manifest, inventory, evaluations):
    keys(frame, 'schema_version request_id evaluation_ordinal evaluation status clocks tiles cpu_gathers '
         'frame_wall_seconds exclusive_host_seconds clock_bracket_seconds unattributed_host_seconds '
         'diagnostic_allowance_bytes', 'frame')
    exact(frame['schema_version'], 1, 'frame version')
    exact(frame['diagnostic_allowance_bytes'], 64 * MIB, 'diagnostic allowance')
    cap = manifest['capture']
    need(frame['request_id'] == cap['request_id'] and type(frame['request_id']) is list
         and all(type(x) is int for x in frame['request_id']), 'frame request ID')
    exact(frame['evaluation_ordinal'], cap['evaluation_ordinal'], 'frame ordinal')
    keys(frame['evaluation'], 'position input_rows output_rows committed_positions', 'evaluation')
    for key, value in frame['evaluation'].items():
        exact(value, cap[key], 'frame evaluation')
    status = frame['status']
    keys(status, 'state evaluations_seen tile_capacity tiles_recorded gathers_recorded overflowed', 'capture status')
    for key, value in dict(state='complete', evaluations_seen=evaluations, tile_capacity=349,
                           tiles_recorded=349, gathers_recorded=2, overflowed=False).items():
        exact(status[key], value, 'complete capture')
    wall, host, bracket, remainder = [seconds(frame[key]) for key in
        ('frame_wall_seconds', 'exclusive_host_seconds', 'clock_bracket_seconds', 'unattributed_host_seconds')]
    clocks = frame['clocks']
    keys(clocks, 'start end host_clock_sample_bracket_seconds events', 'clocks')
    for key in ('start', 'end'):
        keys(clocks[key], 'cpu_nanoseconds gpu_ticks', 'paired clocks')
        for value in clocks[key].values():
            uint(value)
    start, end = clocks['start'], clocks['end']
    cpu, gpu = end['cpu_nanoseconds'] - start['cpu_nanoseconds'], end['gpu_ticks'] - start['gpu_ticks']
    need(cpu > 0 and gpu > 0, 'positive calibration spans')
    need(seconds(clocks['host_clock_sample_bracket_seconds']) <= bracket <= wall, 'clock brackets')
    need(type(frame['tiles']) is list and len(frame['tiles']) == 349
         and type(clocks['events']) is list and len(clocks['events']) == 349, 'tile/event coverage')
    groups, intervals, seen = {}, [], set()
    completed_counters = {key: 0 for key in COUNTER_FIELDS}
    for slot, tile in enumerate(frame['tiles']):
        keys(tile, 'record event', 'attributed tile')
        record, event = tile['record'], tile['event']
        keys(record, 'counter_slot attribution ' + ' '.join(HOST_FIELDS) +
             ' dispatches_encoded commands_submitted host_wait_calls input_copy_bytes output_copy_bytes', 'tile record')
        exact(record['counter_slot'], slot, 'counter slot')
        a = record['attribution']
        keys(a, 'operation source_matrix_slot tile_index source_first_row source_rows input_rows output_features input_features', 'tile attribution')
        identity = _id(a['operation'])
        need(identity in inventory and inventory[identity]['projection_tiles'] > 0, 'projection identity')
        op = inventory[identity]
        index = uint(a['tile_index'], op['projection_tiles'] - 1)
        need((identity, index) not in seen, 'duplicate tile'); seen.add((identity, index))
        rows = cap['input_rows'] if op['prefill_rows'] == 32 else 1
        expected = dict(source_matrix_slot=op['source_matrix_slot'], source_first_row=index * 8192,
                        source_rows=min(8192, op['source_rows'] - index * 8192), input_rows=rows,
                        output_features=op['source_rows'], input_features=op['source_columns'])
        for key, value in expected.items():
            exact(a[key], value, 'attribution ' + key)
        for key in COUNTER_FIELDS:
            exact(record[key], 1, 'completed tile counters')
            completed_counters[key] = uint(completed_counters[key] + record[key])
        exact(record['input_copy_bytes'], product(rows, a['input_features'], 4), 'logical input copy')
        exact(record['output_copy_bytes'], product(rows, a['source_rows'], 4), 'logical output copy')
        for key in HOST_FIELDS:
            seconds(record[key])
        keys(event, 'name host_begin_seconds host_end_seconds gpu_begin_ticks gpu_end_ticks gpu_begin_seconds gpu_end_seconds', 'profile event')
        exact(event['name'], 'gemma_bf16_product', 'kernel entry')
        mirror = clocks['events'][slot]
        keys(mirror, ' '.join(event), 'clock event')
        for key, value in event.items():
            exact(mirror[key], value, 'event/counter binding')
        begin, finish = uint(event['gpu_begin_ticks']), uint(event['gpu_end_ticks'])
        need(begin not in (0, U64) and finish not in (0, U64)
             and start['gpu_ticks'] <= begin <= finish <= end['gpu_ticks'], 'timestamp samples')
        for key, tick in [('gpu_begin_seconds', begin), ('gpu_end_seconds', finish)]:
            expected_seconds = float(tick - start['gpu_ticks']) / float(gpu) * float(cpu) / 1e9
            need(math.isclose(seconds(event[key]), expected_seconds,
                              rel_tol=CLOCK_REL_TOL, abs_tol=CLOCK_ABS_TOL), 'calibrated timestamp replay')
        hb, he = seconds(event['host_begin_seconds']), seconds(event['host_end_seconds'])
        need(hb <= he <= wall + HOST_TOL and he - hb <= record['host_encoding_seconds'] + HOST_TOL,
             'nested encoder host scope')
        interval = (event['gpu_begin_seconds'], event['gpu_end_seconds'])
        intervals.append(interval)
        group_key = identity + (a['source_matrix_slot'], rows, a['input_features'], a['output_features'])
        group = groups.setdefault(group_key, dict(role=identity[0], layer=identity[1],
            source_matrix_slot=a['source_matrix_slot'], input_rows=rows, input_features=a['input_features'],
            output_features=a['output_features'], tiles=0, host_seconds={k: [] for k in HOST_FIELDS},
            completed_counters={k: 0 for k in COUNTER_FIELDS},
            input_copy_bytes=0, output_copy_bytes=0, intervals=[]))
        group['tiles'] += 1
        for key in COUNTER_FIELDS:
            group['completed_counters'][key] = uint(group['completed_counters'][key] + record[key])
        for key in HOST_FIELDS:
            group['host_seconds'][key].append(record[key])
        for key in ('input_copy_bytes', 'output_copy_bytes'):
            group[key] += record[key]
            uint(group[key])
        group['intervals'].append(interval)
    need(seen == {(identity, i) for identity, op in inventory.items() for i in range(op['projection_tiles'])},
         'complete tile identities')
    gathers = frame['cpu_gathers']
    need(type(gathers) is list and len(gathers) == 2, 'gather coverage')
    gather_ids = set()
    for gather in gathers:
        keys(gather, 'operation source_matrix_slot rows columns active_source_bytes host_begin_seconds host_end_seconds', 'CPU gather')
        identity = _id(gather['operation'])
        need(identity in inventory and inventory[identity]['projection_tiles'] == 0 and identity not in gather_ids, 'gather identity')
        gather_ids.add(identity)
        for key, target in dict(source_matrix_slot=inventory[identity]['source_matrix_slot'], rows=cap['input_rows'],
                                columns=inventory[identity]['source_columns']).items():
            exact(gather[key], target, 'gather ' + key)
        exact(gather['active_source_bytes'], product(gather['rows'], gather['columns'], 2), 'logical gather bytes')
        need(seconds(gather['host_begin_seconds']) <= seconds(gather['host_end_seconds']) <= wall + HOST_TOL,
             'gather interval')
    need(gather_ids == {('token_embedding', None), ('per_layer_embedding', None)}, 'complete gather IDs')
    host_sums = {key: _sum(tile['record'][key] for tile in frame['tiles']) for key in HOST_FIELDS}
    gather_sum = _sum(g['host_end_seconds'] - g['host_begin_seconds'] for g in gathers)
    calculated = _sum(host_sums.values()) + gather_sum
    remaining = wall - calculated - bracket
    need(abs(host - calculated) <= HOST_TOL and remaining >= -HOST_TOL
         and abs(remainder - max(0., remaining)) <= HOST_TOL
         and abs(wall - host - bracket - remainder) <= HOST_TOL, 'exclusive host partition')
    operations, roles = [], {}
    for group in groups.values():
        own_intervals = group.pop('intervals')
        group['host_seconds'] = {key: _sum(v) for key, v in group['host_seconds'].items()}
        group['exclusive_host_seconds'] = _sum(group['host_seconds'].values())
        group['gpu_encoder_union_seconds'] = interval_union(own_intervals)
        operations.append(group)
        role = roles.setdefault(group['role'], dict(role=group['role'], host_seconds={k: [] for k in HOST_FIELDS},
            completed_counters={k: 0 for k in COUNTER_FIELDS}, intervals=[]))
        for key in HOST_FIELDS:
            role['host_seconds'][key].append(group['host_seconds'][key])
        for key in COUNTER_FIELDS:
            role['completed_counters'][key] = uint(role['completed_counters'][key] + group['completed_counters'][key])
        role['intervals'] += own_intervals
    for role in roles.values():
        role['host_seconds'] = {key: _sum(v) for key, v in role['host_seconds'].items()}
        role['exclusive_host_seconds'] = _sum(role['host_seconds'].values())
        role['gpu_encoder_union_seconds'] = interval_union(role.pop('intervals'))
    operations.sort(key=lambda g: (-g['exclusive_host_seconds'], g['role'], -1 if g['layer'] is None else g['layer']))
    role_totals = sorted(roles.values(), key=lambda g: (-g['exclusive_host_seconds'], g['role']))
    union = interval_union(intervals)
    span = max(b for _, b in intervals) - min(a for a, _ in intervals)
    return dict(coverage=dict(logical_operations=279, projection_operations=len(groups), tiles=349, cpu_gathers=2),
        completed_counters=completed_counters,
        frame=dict(wall_seconds=wall, exclusive_host_seconds=host, clock_bracket_seconds=bracket,
                   unattributed_host_seconds=remainder), host_buckets_seconds=host_sums, cpu_gather_seconds=gather_sum,
        gpu=dict(encoder_union_seconds=union, observed_span_seconds=span,
                 gaps_between_attributed_samples_seconds=max(0., span - union)),
        operations=operations, role_totals=role_totals, cpu_gathers=gathers)


def report_bundle(folder):
    folder = Path(folder)
    need(stat.S_ISDIR(folder.lstat().st_mode), 'regular bundle directory required')
    manifest, manifest_file = _read(folder, 'manifest.json', 64 * 1024)
    evaluations = _manifest(manifest)
    data, files = {}, dict(manifest=manifest_file)
    for key in ('profile', 'inventory'):
        entry = manifest[key]
        keys(entry, 'file bytes sha256', 'file receipt')
        exact(entry['file'], key + '.json', 'fixed bundle leaf')
        uint(entry['bytes'], MIB, 1)
        digest(entry['sha256'])
        data[key], files[key] = _read(folder, key + '.json', MIB)
        need(entry == files[key], 'file SHA/length binding')
    result = _frame(data['profile'], manifest, _inventory(data['inventory']), evaluations)
    result.update(format='toklet-gemma-profile-cost-v1', diagnostic_only=True, timing_qualified=False,
                  availability='complete', model=manifest['model'], workload=manifest['workload'],
                  capture=manifest['capture'], parity=manifest['parity'], memory=manifest['memory'],
                  plain=manifest['plain'], profiled=manifest['profiled'], files=files,
                  units=dict(host='seconds', gpu='calibrated seconds', bytes='logical active payload'),
                  scope=['One selected evaluation; no whole-request extrapolation or acceptance timing.',
                         'Five host buckets are exclusive; GPU unions are separate overlapping scopes.',
                         'Encoder host intervals are nested; wait minus GPU is not measured queue latency.',
                         'Copy/gather bytes are logical active payload, not measured DRAM or page traffic.',
                         'Unattributed host includes CPU math/control/scatter; GPU gaps do not prove physical idleness.',
                         'Logit/output/history parity is producer-attested; this consumer authenticates its bundle bytes.',
                         manifest['clock_scope']])
    return result


def markdown(report):
    def ms(value):
        return f'{seconds(value * 1000):.6f}'

    counts = report['completed_counters']
    lines = ['# Gemma diagnostic profile', '', 'One selected evaluation. Diagnostic timing only.', '',
             f"Completed dispatches: {counts['dispatches_encoded']}; commands: {counts['commands_submitted']}; waits: {counts['host_wait_calls']}.", '',
             'Host and GPU columns overlap. Bytes are logical payload, not DRAM traffic.', '',
             '| Operation | Layer | Tiles | Host ms | Wait ms | Validation/copy ms | Encode ms | GPU union ms |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['operations'][:20]:
        host = row['host_seconds']
        numbers = [row['exclusive_host_seconds'], host['host_wait_seconds'],
                   host['host_validation_and_copy_seconds'], host['host_encoding_seconds'], row['gpu_encoder_union_seconds']]
        lines.append('| {} | {} | {} | {} |'.format(row['role'], row['layer'] if row['layer'] is not None else '-',
                     row['tiles'], ' | '.join(ms(n) for n in numbers)))
    lines += ['', '## Projection role totals', '', '| Role | Host ms | GPU union ms |', '|---|---:|---:|']
    for row in report['role_totals']:
        lines.append(f"| {row['role']} | {ms(row['exclusive_host_seconds'])} | {ms(row['gpu_encoder_union_seconds'])} |")
    lines += ['', 'CPU gathers, calibration and unattributed host remain separate in JSON.',
              'GPU union totals are recomputed from intervals; never add per-operation unions.', '']
    text = '\n'.join(lines)
    need(len(text.encode()) <= 64 * 1024, 'Markdown bound')
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args(argv)
    try:
        report = report_bundle(args.bundle)
        raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        need(len(raw) <= MIB, 'JSON report bound')
        outputs = [(args.json, raw)]
        if args.markdown is not None:
            outputs.append((args.markdown, markdown(report).encode()))
        need(len({p.absolute() for p, _ in outputs}) == len(outputs), 'distinct report paths required')
        need(all(not p.exists() and not p.is_symlink() for p, _ in outputs), 'report output already exists')
        for file, body in outputs:
            with file.open('xb') as stream:
                stream.write(body)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
