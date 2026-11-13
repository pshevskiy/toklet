"""Offline complete saved-byte Ministral comparison under frozen reference gates.

Uses the reviewed bounded descriptor, FP64 predicate and lossless byte-stream
mechanisms from the Granite checker without changing its geometry or decisions.
This module imports no numerical package; the supplied NumPy runtime is admitted
by the new reference driver before compare() is called.
"""
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path

import check_granite_reference as saved
import ministral3_graph_protocol as protocol

need = protocol.need
REPEATS = ('initial', 'reset_replay')
ENGINE_LIMIT = 6 * 1024**3 - 32 * 1024**2


def receipt(value, expected_sha=None, expected_bytes=None):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256', 'before_identity', 'after_identity', 'after_sha256'},
         'Complete native input receipt required')
    need(type(value['path']) is str and 0 < len(value['path']) <= 4096
         and '\x00' not in value['path'] and Path(value['path']).is_absolute(), 'Absolute native input path required')
    saved.integer(value['bytes'], 1)
    saved.sha(value['sha256'])
    need(value['after_sha256'] == value['sha256'], 'Native input final-byte hash differs')
    identity = value['before_identity']
    need(type(identity) is dict and set(identity) == {'dev', 'ino', 'bytes', 'mtime', 'mtime_nsec', 'ctime', 'ctime_nsec', 'mode', 'nlink'},
         'Complete native file identity required')
    for field, item in identity.items():
        saved.integer(item, -saved.MAX_INTEGER if field in ('mtime', 'ctime') else 0)
    need(identity['bytes'] == value['bytes'], 'Native input descriptor extent differs')
    saved.exact(value['after_identity'], identity, 'Native input endpoint identity changed')
    if expected_sha is not None:
        need(value['sha256'] == expected_sha, 'Native input external SHA256 differs')
    if expected_bytes is not None:
        need(value['bytes'] == expected_bytes, 'Native input external extent differs')
    return value


def validate_protocol(value):
    need(type(value) is dict and set(value) == {'format', 'model', 'cases', 'chunks'}
         and value['format'] == 'toklet-ministral3-native-graph-protocol-v1', 'Exact Ministral graph protocol required')
    need(type(value['model']) is dict and set(value['model']) == {'data_sha256', 'config_sha256'},
         'Complete child/config identities required')
    for pin in value['model'].values():
        saved.sha(pin)
    expected = [dict(name=name, tokens=tokens, offset=offset)
                for name, tokens, offset in protocol.TINY_CASES if not name.startswith('diagnostic_')]
    saved.exact(value['cases'], expected, 'Exact five prospective case/token/offset identities required')
    saved.exact(value['chunks'], [1, 3], 'Exact two chunk schedules required')
    for case in value['cases']:
        protocol.geometry(protocol.REAL, case['tokens'], case['offset'])
    return value


def array_spec(value, shape, native=False):
    fields = {'path', 'dtype', 'shape', 'bytes', 'sha256', 'name', 'layer', 'stage'} if native else {'file', 'dtype', 'shape', 'bytes', 'sha256'}
    need(type(value) is dict and set(value) == fields, 'Complete exact array descriptor required')
    saved.exact(value['shape'], shape, 'Exact array shape required')
    dtype = value['dtype']
    need(dtype in ('<f2', '<f4', 'f16le', 'f32le'), 'Explicit little-endian stored dtype required')
    dtype = {'f16le': '<f2', 'f32le': '<f4'}.get(dtype, dtype)
    need(type(value['bytes']) is int and value['bytes'] == math.prod(shape) * saved.DTYPES[dtype], 'Exact array byte extent required')
    saved.relative(value['path' if native else 'file'])
    saved.sha(value['sha256'])
    return dict(dtype=dtype, shape=shape, bytes=value['bytes'])


def stage_name(entry):
    need(type(entry.get('stage')) is str and type(entry.get('name')) is str, 'Explicit stage identity required')
    if entry.get('layer') is None:
        need(entry['stage'] in ('embedding', 'final_norm', 'logits') and entry['name'] == entry['stage'],
             'Exact global stage name required')
        return entry['stage']
    layer = saved.integer(entry['layer'], 0, 25)
    need(entry['stage'] in protocol.STAGES, 'Exact selected layer stage required')
    need(entry['name'] == f'layer{layer:02}.{entry["stage"]}', 'Exact physical capture stage name required')
    return f'layers.{layer}.{entry["stage"]}'


def reference_arrays(reference, prospective, protocol_sha):
    need(reference.get('format') == 'toklet-ministral3-independent-reference-v1'
         and reference.get('mode') == 'capture' and reference.get('passed') is True
         and reference.get('protocol_sha256') == protocol_sha
         and reference.get('source_consumption', {}).get('after_sha256') == prospective['model']['data_sha256'],
         'Complete pinned independent reference capture required')
    saved.exact(reference.get('geometry'), protocol.REAL, 'Exact independent geometry required')
    saved.exact(reference.get('numerical_policy'), protocol.POLICY, 'Exact independent numerical policy required')
    saved.exact(reference.get('model'), prospective['model'], 'Exact independent child identities required')
    summaries = reference.get('cases')
    need(type(summaries) is list and len(summaries) == 10, 'Complete reference profile summaries required')
    expected_summaries = {(case['name'], profile) for case in prospective['cases'] for profile in ('main', 'wide')}
    seen = set()
    for summary in summaries:
        need(type(summary) is dict and set(summary) == {'name', 'profile', 'arrays', 'diagnostic_graph_and_publication_ns',
                                                       'final_logits_sha256', 'top_two_ids', 'margin'}, 'Exact reference profile summary required')
        key = summary['name'], summary['profile']
        need(key in expected_summaries and key not in seen, 'Unique complete reference profile summary required')
        seen.add(key)
        saved.exact(summary['arrays'], 419, 'Complete reference stage summary required')
        saved.integer(summary['diagnostic_graph_and_publication_ns'])
        saved.sha(summary['final_logits_sha256'])
        ids = summary['top_two_ids']
        need(type(ids) is list and len(ids) == 2 and ids[0] != ids[1]
             and all(type(i) is int and 0 <= i < 131072 for i in ids)
             and type(summary['margin']) in (int, float) and math.isfinite(summary['margin']) and summary['margin'] >= 0,
             'Concrete reference rank/margin summary required')
    arrays = reference.get('arrays')
    need(type(arrays) is dict, 'Complete reference array map required')
    expected = {}
    paths = set()
    for case in prospective['cases']:
        for profile in ('main', 'wide'):
            for name, shape in protocol.contracts(protocol.REAL, len(case['tokens'])).items():
                key = case['name'] + '.' + profile + '.' + name
                need(key in arrays, 'Missing independent reference stage: ' + key)
                spec = array_spec(arrays[key], shape)
                need(spec['dtype'] == ('<f4' if name == 'logits' else '<f2'), 'Actual stored reference precision differs')
                need(arrays[key]['file'] not in paths, 'Reference array path alias')
                paths.add(arrays[key]['file'])
                expected[key] = arrays[key]
    need(set(arrays) == set(expected), 'No missing or extra independent reference array allowed')
    need(type(reference.get('published_bytes')) is int
         and reference['published_bytes'] == sum(item['bytes'] for item in arrays.values()), 'Complete reference byte coverage required')
    return expected


def native_runs(native, prospective, protocol_sha):
    need(native.get('format') == 'toklet-ministral3-native-graph-capture-v1'
         and native.get('completed') is True and native.get('graph_qualified') is False
         and native.get('error') is None and native.get('active_run') is None, 'Complete selected native graph capture required')
    receipt(native.get('protocol'), protocol_sha)
    receipt(native.get('model_data'), prospective['model']['data_sha256'], 2662082312)
    receipt(native.get('model_config'), prospective['model']['config_sha256'])
    receipt(native.get('build_manifest'))
    receipt(native.get('binary'))
    need(type(native.get('capture_source')) is dict and set(native['capture_source']) == {'bytes', 'sha256'}
         and type(native['capture_source']['bytes']) is int and 1 <= native['capture_source']['bytes'] <= 128 * 1024,
         'Explicit capture source pin required')
    saved.sha(native['capture_source']['sha256'])
    saved.exact(native.get('checks'), dict(all_input_descriptor_path_and_byte_endpoints_unchanged=True,
                                         complete_frozen_case_chunk_stage_inventory=True), 'Complete native endpoint/inventory checks required')
    need(type(native.get('engine_budget_bytes')) is int and native['engine_budget_bytes'] == ENGINE_LIMIT
         and type(native.get('helper_limit_bytes')) is int and native['helper_limit_bytes'] == 32 * 1024**2
         and type(native.get('helper_high_water_bytes')) is int
         and 0 <= native['helper_high_water_bytes'] <= native['helper_limit_bytes'], 'Exact native engine/helper budgets required')
    need(type(native.get('device')) is str and 0 < len(native['device']) <= 256
         and type(native.get('load_seconds')) in (int, float) and math.isfinite(native['load_seconds'])
         and native['load_seconds'] >= 0, 'Explicit native device and diagnostic load clock required')
    runs = native.get('runs')
    need(type(runs) is list and len(runs) == 22, 'Complete schedules/replays plus two changed-prefix suffix captures required')
    indexed, paths = {}, set()
    expected = {(case['name'], chunk, repeat) for case in prospective['cases']
                for chunk in prospective['chunks'] for repeat in REPEATS}
    expected.update({('future_changed', chunk, 'changed_prefix') for chunk in prospective['chunks']})
    cases = {case['name']: case for case in prospective['cases']}
    for run in runs:
        need(type(run) is dict and type(run.get('case')) is str
             and type(run.get('chunk')) is int and type(run.get('repeat')) is str,
             'Typed native run identity required')
        key = run['case'], run['chunk'], run['repeat']
        need(key in expected and key not in indexed, 'Unique exact native schedule/repeat required')
        case = cases[run['case']]
        saved.exact(run.get('tokens'), case['tokens'], 'Exact native run token IDs required')
        saved.exact(run.get('offset'), case['offset'], 'Exact native absolute position required')
        row_offset = 3 if run['repeat'] == 'changed_prefix' else 0
        saved.exact(run.get('row_offset'), row_offset, 'Exact native changed-prefix row origin required')
        shapes = protocol.contracts(protocol.REAL, len(case['tokens']) - row_offset)
        entries = run.get('arrays')
        need(type(entries) is list and len(entries) == len(shapes), 'Complete native graph stage inventory required')
        selected = {}
        for entry in entries:
            name = stage_name(entry)
            need(name in shapes and name not in selected, 'Unique declared native stage required')
            spec = array_spec(entry, shapes[name], True)
            need(spec['dtype'] == ('<f4' if name == 'logits' else '<f2'), 'Actual stored native precision differs')
            need(entry['path'] not in paths, 'Native array path alias')
            paths.add(entry['path'])
            selected[name] = entry
        need(set(selected) == set(shapes), 'Every native stage required')
        saved.sha(run.get('final_logits_sha256'))
        chunks = run.get('chunks')
        count = len(case['tokens'])
        schedule = [(first, min(run['chunk'], count - first)) for first in range(row_offset, count, run['chunk'])]
        need(type(chunks) is list and len(chunks) == len(schedule), 'Complete native chunk receipt inventory required')
        for chunk, (first, rows) in zip(chunks, schedule, strict=True):
            need(type(chunk) is dict and type(chunk.get('start')) is int and chunk['start'] == first
                 and type(chunk.get('rows')) is int and chunk['rows'] == rows
                 and type(chunk.get('absolute_start')) is int and chunk['absolute_start'] == case['offset'] + first,
                 'Exact native completed chunk coverage required')
            for field in ('wall_seconds', 'diagnostics_seconds'):
                need(type(chunk.get(field)) in (int, float) and math.isfinite(chunk[field]) and chunk[field] >= 0,
                     'Finite completed diagnostic clock required')
            timing = chunk.get('timing')
            fields = {'command_creation_seconds', 'clock_calibration_seconds', 'commit_seconds', 'host_encode_seconds',
                      'wait_seconds', 'gpu_start_seconds', 'gpu_end_seconds', 'readback_seconds', 'command_count'}
            need(type(timing) is dict and set(timing) == fields, 'Complete completed-command timing receipt required')
            for field in fields - {'command_count'}:
                need(type(timing[field]) in (int, float) and math.isfinite(timing[field]) and timing[field] >= 0,
                     'Finite completed command clock required')
            need(type(timing['command_count']) is int and timing['command_count'] == 1
                 and timing['gpu_end_seconds'] >= timing['gpu_start_seconds'], 'One ordered completed command per chunk required')
            memory = chunk.get('memory')
            fields = {'weights_bytes', 'positional_bytes', 'kv_bytes', 'scratch_bytes', 'history_bytes',
                      'logits_bytes', 'metadata_bytes', 'metadata_allowance_bytes', 'admitted_bytes'}
            need(type(memory) is dict and set(memory) == fields, 'Complete selected positional/session memory receipt required')
            for value in memory.values():
                saved.integer(value)
            need(memory['positional_bytes'] > 0 and sum(value for field, value in memory.items() if field != 'admitted_bytes')
                 <= memory['admitted_bytes'] <= ENGINE_LIMIT, 'Complete counted engine owners exceed native budget')
        expected_checks = {'fully_cached_prompt_zero_work', 'invalid_inputs_preserve_committed_prefix'} if run['repeat'] == 'initial' else {
            'reset_replay_all_stage_bytes_equal'} if run['repeat'] == 'reset_replay' else {'changed_prefix_all_suffix_bytes_equal_fresh'}
        applicable = len(case['tokens']) < 512 - case['offset']
        saved.exact(run.get('pending_token_applicable'), applicable, 'Exact pending-token applicability required')
        if run['repeat'] == 'initial' and applicable:
            expected_checks.add('pending_token_uncommitted_discard_restores_distribution')
        saved.exact(run.get('checks'), dict.fromkeys(expected_checks, True), 'Complete explicit native state checks required')
        need(run.get('diagnostic_final_row_equals_logits') is True
             and type(run.get('final_greedy_token')) is int and 0 <= run['final_greedy_token'] < 131072,
             'Completed distribution and concrete greedy token required')
        indexed[key] = dict(run=run, arrays=selected)
    need(set(indexed) == expected, 'Complete native schedule identity inventory required')
    return indexed


def prefix_hash(source, count):
    digest = hashlib.sha256()
    for offset in range(0, count, saved.BLOCK):
        digest.update(source.read(offset, min(saved.BLOCK, count - offset)))
    return digest.hexdigest()


def final_distribution(source, shape, np):
    start = (shape[0] - 1) * shape[1] * 4
    size, digest, top = shape[1] * 4, hashlib.sha256(), []
    for offset in range(0, size, saved.BLOCK):
        raw = source.read(start + offset, min(saved.BLOCK, size - offset))
        digest.update(raw)
        values = np.frombuffer(raw, dtype='<f4').astype(np.float64)
        need(np.isfinite(values).all(), 'Finite complete reference final distribution required')
        order = np.lexsort((np.arange(len(values)), -values))[:2]
        top.extend((float(values[i]), offset // 4 + int(i)) for i in order)
        top = sorted(top, key=lambda value: (-value[0], value[1]))[:2]
    need(len(top) == 2, 'Complete reference final rank required')
    return dict(final_logits_sha256=digest.hexdigest(), top_two_ids=[top[0][1], top[1][1]],
                margin=top[0][0] - top[1][0])


class Span:
    """Bounded row suffix view over an already authenticated full reference."""
    def __init__(self, source, offset):
        need(0 <= offset < source.size, 'Exact nonempty reference suffix required')
        self.source, self.offset, self.size = source, offset, source.size - offset

    def read(self, offset, count):
        saved.integer(offset, 0, self.size)
        need(0 <= count <= self.size - offset, 'Suffix read exceeds array extent')
        return self.source.read(self.offset + offset, count)

    def endpoint(self):
        self.source.endpoint()


def compare(np, prospective, protocol_sha, reference, native, reference_folder, native_folder, output):
    validate_protocol(prospective)
    refs = reference_arrays(reference, prospective, protocol_sha)
    indexed = native_runs(native, prospective, protocol_sha)
    failed, coordinates, records, exact_checks = 0, 0, 0, 0
    prefixes, reference_prefixes = {}, {}
    output = Path(output)
    with saved.Files(reference_folder) as rf, saved.Files(native_folder) as nf, (output / 'records.jsonl').open('x') as journal:
        for case in prospective['cases']:
            schedules = [(case['name'], chunk, repeat) for chunk in prospective['chunks'] for repeat in REPEATS]
            if case['name'] == 'future_changed':
                schedules.extend((case['name'], chunk, 'changed_prefix') for chunk in prospective['chunks'])
            for stage, shape in protocol.contracts(protocol.REAL, len(case['tokens'])).items():
                main, wide = (refs[case['name'] + '.' + profile + '.' + stage] for profile in ('main', 'wide'))
                spec = array_spec(main, shape)
                with ExitStack() as stack:
                    a = stack.enter_context(rf.open(main['file'], main['bytes'], main['sha256']))
                    b = stack.enter_context(rf.open(wide['file'], wide['bytes'], wide['sha256']))
                    gate, envelope = saved.gate_from_files(a, b, spec['dtype'], np)
                    if stage == 'logits':
                        for profile, source in (('main', a), ('wide', b)):
                            summary = next(row for row in reference['cases'] if row['name'] == case['name'] and row['profile'] == profile)
                            actual_summary = final_distribution(source, shape, np)
                            saved.exact({key: summary[key] for key in actual_summary}, actual_summary,
                                        'Reference raw final distribution/rank differs from its summary')
                    if case['name'] in ('prompt', 'future_changed'):
                        prefix_bytes = 3 * shape[1] * saved.DTYPES[spec['dtype']]
                        reference_prefixes[case['name'], stage] = (prefix_hash(a, prefix_bytes), prefix_hash(b, prefix_bytes))
                    base_entry = indexed[schedules[0]]['arrays'][stage]
                    baseline = stack.enter_context(nf.open(base_entry['path'], base_entry['bytes'], base_entry['sha256']))
                    for key in schedules:
                        entry = indexed[key]['arrays'][stage]
                        actual = baseline if key == schedules[0] else stack.enter_context(nf.open(entry['path'], entry['bytes'], entry['sha256']))
                        row_offset = indexed[key]['run']['row_offset']
                        source_offset = row_offset * shape[1] * saved.DTYPES[spec['dtype']]
                        selected_spec = dict(spec, shape=[shape[0] - row_offset, shape[1]], bytes=actual.size)
                        left, right = (Span(a, source_offset), Span(b, source_offset)) if row_offset else (a, b)
                        equal = True
                        digest = hashlib.sha256()

                        def blocks():
                            nonlocal equal
                            for offset in range(0, actual.size, saved.BLOCK):
                                raw = actual.read(offset, min(saved.BLOCK, actual.size - offset))
                                digest.update(raw)
                                if key != schedules[0]:
                                    equal = equal and raw == baseline.read(source_offset + offset, len(raw))
                                yield offset, raw
                            need(digest.hexdigest() == actual.expected, 'Complete actual native SHA256 differs')
                            actual.endpoint()

                        result = saved.compare_blocks(blocks(), left, right, selected_spec, gate, np, stage == 'logits')
                        if key != schedules[0]:
                            exact_checks += 1
                            need(equal, 'Native chunk/reset graph bytes differ: ' + str(key) + ':' + stage)
                        if case['name'] in ('prompt', 'future_changed') and key == schedules[0]:
                            prefixes[case['name'], stage] = prefix_hash(actual, prefix_bytes)
                        if stage == 'logits':
                            need(result['final_logits_sha256'] == indexed[key]['run']['final_logits_sha256'],
                                 'Actual final-logit bytes differ from native state receipt')
                            need(result['final_greedy_token'] == indexed[key]['run']['final_greedy_token'],
                                 'Actual full-logit greedy selection differs from native state receipt')
                        current = sum(item['failed_values'] for item in result['references'].values())
                        failed += current
                        coordinates += 2 * math.prod(selected_spec['shape'])
                        records += 1
                        journal.write(json.dumps(dict(case=key[0], chunk=key[1], repeat=key[2], stage=stage,
                            gate=gate, independent_precision_envelope=envelope, native_sha256=actual.expected,
                            reference_row_offset=row_offset, gate_scope='full prescribed case/array',
                            numerical=result), separators=(',', ':'), allow_nan=False) + '\n')
                    a.endpoint()
                    b.endpoint()
                    baseline.endpoint()
        rf.endpoint()
        nf.endpoint()
    for stage in protocol.contracts(protocol.REAL, 4):
        need(prefixes['prompt', stage] == prefixes['future_changed', stage], 'Native causal graph prefix changed: ' + stage)
        need(reference_prefixes['prompt', stage] == reference_prefixes['future_changed', stage],
             'Independent causal graph prefix changed: ' + stage)
    need(records == 22 * 419 and exact_checks == (5 * 3 + 2) * 419, 'Complete native-array and exact-replay coverage required')
    return dict(passed=failed == 0, failed_profile_coordinates=failed, profile_coordinates=coordinates,
                array_comparisons=records, exact_schedule_or_reset_array_checks=exact_checks,
                causal_stage_prefix_checks=419, both_profiles_required=True,
                short_prefix_graph_qualified=failed == 0, full_context_graph_qualified=False,
                quality_qualified=False, performance_qualified=False, managed_adoption_qualified=False)
