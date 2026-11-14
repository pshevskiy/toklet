"""Independent data-only bundle controls; no model, numerical library or GPU."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import gemma_profile_report as report


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(input_count=15, maximum=2, emitted=2, stop=None, ordinal=1, zero=False):
    # Source shapes come from the frozen independent source-layout fixture,
    # not the consumer's geometry routine or a native execution.
    root = Path(__file__).resolve().parents[2]
    layout = json.loads((root / 'tests/fixtures/gemma-unquantized-header-v1/layout.json').read_bytes())
    matrices = {name: row for name, row in layout['selected_text'].items() if len(row['shape']) == 2}
    suffixes = {'self_attn.q_proj': 'query', 'self_attn.k_proj': 'key', 'self_attn.v_proj': 'value',
                'self_attn.o_proj': 'attention_output', 'mlp.gate_proj': 'feed_forward_gate',
                'mlp.up_proj': 'feed_forward_up', 'mlp.down_proj': 'feed_forward_down',
                'per_layer_input_gate': 'per_layer_gate', 'per_layer_projection': 'per_layer_projection'}
    operations = []
    coefficient_layout = ('row-major [out,in], original little-endian BF16, two coefficients/u32; '
                          'FP32 activations/output')
    for slot, (name, spec) in enumerate(sorted(matrices.items())):
        short = name.removeprefix('model.language_model.').removesuffix('.weight')
        if short.startswith('layers.'):
            _, layer, tail = short.split('.', 2)
            identities = [(suffixes[tail], int(layer))]
        else:
            identities = {'embed_tokens': [('token_embedding', None), ('output_head', None)],
                          'embed_tokens_per_layer': [('per_layer_embedding', None)],
                          'per_layer_model_projection': [('per_layer_input_projection', None)]}[short]
        for role, layer in identities:
            rows, columns = spec['shape']
            gather = role in ('token_embedding', 'per_layer_embedding')
            tiles = 0 if gather else (rows + 8191) // 8192
            count = 1 if role == 'output_head' or layer is not None and layer >= 15 else 32
            execution = ('resident_bf16_cpu_gather' if role == 'token_embedding' else
                         'authenticated_file_bf16_cpu_gather' if gather else 'tiled_bf16_metal_projection')
            operations.append(dict(id=dict(role=role, layer=layer), source_matrix_slot=slot,
                source_tensor=name, source_rows=rows, source_columns=columns, source_bytes=spec['bytes'],
                prefill_rows=count, decode_rows=1, execution=execution, coefficient_layout=coefficient_layout,
                kernel_entry=None if gather else 'gemma_bf16_product', projection_tiles=tiles,
                predicted_commands_per_chunk=tiles, predicted_dispatches_per_chunk=tiles,
                predicted_host_waits_per_chunk=tiles,
                modelled_prefill_output_host_bytes=0 if gather else rows * count * 4,
                modelled_decode_output_host_bytes=0 if gather else rows * 4))
    inventory = dict(payload_authenticated=True, resident_matrix_owners=277,
                     resident_gpu_tile_owners=349, vector_owners=262,
                     retained_weight_owner_bytes=3_000_000_000, operations=operations)
    sampled = emitted + (stop is not None)
    chunks = (input_count + 31) // 32
    # Build actual evaluation sizes independently as a schedule, then index it.
    sizes = [min(32, input_count - start) for start in range(0, input_count, 32)] + [1] * (sampled - 1)
    position, rows = sum(sizes[:ordinal - 1]), sizes[ordinal - 1]
    capture = dict(evaluation_ordinal=ordinal, request_id=list(range(16)), position=position,
                   input_rows=rows, output_rows=1, committed_positions=position + rows)
    events, tiles, gathers = [], [], []
    for operation in operations:
        if operation['projection_tiles'] == 0:
            gathers.append(dict(operation=operation['id'], source_matrix_slot=operation['source_matrix_slot'],
                rows=rows, columns=operation['source_columns'], active_source_bytes=rows * operation['source_columns'] * 2,
                host_begin_seconds=0.0, host_end_seconds=0.0 if zero else .001))
            continue
        n = rows if operation['prefill_rows'] == 32 else 1
        for index in range(operation['projection_tiles']):
            # All intervals overlap deliberately: adding tile/group GPU times
            # would be wrong even though every native clock sample is valid.
            event = dict(name='gemma_bf16_product', host_begin_seconds=0.0, host_end_seconds=0.0,
                gpu_begin_ticks=1100, gpu_end_ticks=1100 if zero else 1300,
                gpu_begin_seconds=.01, gpu_end_seconds=.01 if zero else .03)
            a = dict(operation=operation['id'], source_matrix_slot=operation['source_matrix_slot'], tile_index=index,
                source_first_row=index * 8192, source_rows=min(8192, operation['source_rows'] - index * 8192),
                input_rows=n, output_features=operation['source_rows'], input_features=operation['source_columns'])
            record = dict(counter_slot=len(events), attribution=a,
                host_staging_seconds=0.0 if zero else .0001, host_encoding_seconds=0.0,
                host_commit_seconds=0.0, host_wait_seconds=0.0 if zero else .0002,
                host_validation_and_copy_seconds=0.0 if zero else .0003,
                dispatches_encoded=1, commands_submitted=1, host_wait_calls=1,
                input_copy_bytes=n * operation['source_columns'] * 4, output_copy_bytes=n * a['source_rows'] * 4)
            tiles.append(dict(record=record, event=event)); events.append(copy.deepcopy(event))
    host = 0.0 if zero else 349 * .0006 + .002
    frame = dict(schema_version=1, request_id=capture['request_id'], evaluation_ordinal=ordinal,
        evaluation={k: capture[k] for k in ['position', 'input_rows', 'output_rows', 'committed_positions']},
        status=dict(state='complete', evaluations_seen=len(sizes), tile_capacity=349,
                    tiles_recorded=349, gathers_recorded=2, overflowed=False),
        clocks=dict(start=dict(cpu_nanoseconds=10**18, gpu_ticks=1000),
                    end=dict(cpu_nanoseconds=10**18 + 100_000_000, gpu_ticks=2000),
                    host_clock_sample_bracket_seconds=.0005, events=events),
        tiles=tiles, cpu_gathers=gathers, frame_wall_seconds=1.0, exclusive_host_seconds=host,
        clock_bracket_seconds=.001, unattributed_host_seconds=1.0 - host - .001,
        diagnostic_allowance_bytes=64 * 1024**2)
    core = dict(reason=dict(kind='output_limit') if stop is None else dict(kind='stop_token', token_id=stop),
        reused_tokens=0, prefill_tokens=input_count, sampled_tokens=sampled, emitted_tokens=emitted,
        history_tokens=input_count + sampled, evaluated_tokens=input_count + sampled - 1, pending_tokens=1,
        first_token_seconds=.1 if emitted else None, prefill_seconds=.1, decode_seconds=.1 if sampled > 1 else 0.0,
        decode_evaluations=sampled - 1, sampling_seconds=.1, total_seconds=.5)
    manifest = dict(format='toklet-gemma-profile-v1', passed=True, diagnostic_only=True, timing_qualified=False,
        model=dict(repository='google/gemma-4-E2B-it', revision='70af34e20bd4b7a91f0de6b22675850c43922a03'),
        workload=dict(sha256=sha(b'workload'), conversation_sha256=sha(b'conversation'),
                      input_tokens_sha256=sha(b'tokens'), input_tokens=input_count,
                      maximum_output_tokens=maximum, capacity=512, chunk=32),
        capture=capture, parity=dict(selected_logits_bitwise=True, selected_logits_count=262144,
                                    selected_logits_sha256=sha(b'full-selected-row'), output_ids=True,
                                    history=True, report_counts=True), plain=core, profiled=copy.deepcopy(core),
        inventory={}, profile={}, memory=dict(limit_bytes=6 * 1024**3, owner_limit_bytes=6 * 1024**3 - 48 * 1024**2,
            caller_allowance_bytes=16 * 1024**2, helper_allowance_bytes=32 * 1024**2,
            diagnostic_allowance_bytes=64 * 1024**2, plain_admitted_bytes=1000 * 1024**2,
            profiled_admitted_bytes=1064 * 1024**2, owner_reserved_bytes=10 * 1024**2,
            combined_profiled_admitted_bytes=1122 * 1024**2),
        clock_scope='Core times include selected full-logit witness; diagnostics only.')
    assert chunks + sampled - 1 == len(sizes)
    return manifest, frame, inventory


def write_bundle(folder, manifest, frame, inventory):
    for name, value in [('profile', frame), ('inventory', inventory)]:
        raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
        (folder / (name + '.json')).write_bytes(raw)
        manifest[name] = dict(file=name + '.json', bytes=len(raw), sha256=sha(raw))
    (folder / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':'), allow_nan=False))


def batched_fixture(**kwargs):
    manifest, frame, inventory = fixture(**kwargs)
    manifest.update(format='toklet-gemma-profile-v2', submission_policy='batched_single_row')
    inventory.update(schema_version=2, submission_policy='batched_single_row')
    by_id = {}
    for op in inventory['operations']:
        identity = (op['id']['role'], op['id']['layer'])
        by_id[identity] = op
        # Fixed chunk32 initialized buffers from the independent source shapes.
        fits = op['projection_tiles'] > 1 and op['source_rows'] <= 262144 and op['source_columns'] <= 393216
        prefill = 1 if fits and op['prefill_rows'] == 1 else op['projection_tiles']
        op.update(predicted_commands_per_chunk=prefill, predicted_host_waits_per_chunk=prefill,
                  predicted_decode_commands_per_chunk=1 if fits else op['projection_tiles'],
                  predicted_decode_host_waits_per_chunk=1 if fits else op['projection_tiles'])
    groups = []
    shared = ['host_staging_seconds', 'host_commit_seconds', 'host_wait_seconds', 'host_validation_and_copy_seconds']
    first = 0
    while first < len(frame['tiles']):
        record = frame['tiles'][first]['record']
        a = record['attribution']
        op = by_id[(a['operation']['role'], a['operation']['layer'])]
        count = op['projection_tiles']
        if count > 1 and a['input_rows'] == 1 and op['source_rows'] <= 262144 and op['source_columns'] <= 393216:
            groups.append(dict(group_id=len(groups), first_counter_slot=first, tile_count=count,
                owner_counter_slot=first, operation=copy.deepcopy(op['id']), source_matrix_slot=op['source_matrix_slot'],
                input_rows=1, input_features=op['source_columns'], output_features=op['source_rows']))
            record['input_copy_bytes'] = op['source_columns'] * 4
            record['output_copy_bytes'] = op['source_rows'] * 4
            for tile in frame['tiles'][first + 1:first + count]:
                other = tile['record']
                for key in shared: other[key] = 0.0
                other.update(commands_submitted=0, host_wait_calls=0, input_copy_bytes=0, output_copy_bytes=0)
        first += count
    frame.update(schema_version=2, submission_groups=groups)
    host = sum(tile['record'][key] for tile in frame['tiles'] for key in shared + ['host_encoding_seconds'])
    host += sum(g['host_end_seconds'] - g['host_begin_seconds'] for g in frame['cpu_gathers'])
    frame.update(exclusive_host_seconds=host,
                 unattributed_host_seconds=frame['frame_wall_seconds'] - host - frame['clock_bracket_seconds'])
    return manifest, frame, inventory


class Reports(unittest.TestCase):
    def validate(self, data):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_bundle(folder, *copy.deepcopy(data))
            return report.report_bundle(folder)

    def test_synthetic_overlap_uses_unions_and_keeps_host_partition_separate(self):
        value = self.validate(fixture())
        self.assertEqual(value['format'], 'toklet-gemma-profile-cost-v1')
        self.assertEqual(value['coverage'], dict(logical_operations=279, projection_operations=277, tiles=349, cpu_gathers=2))
        self.assertAlmostEqual(value['gpu']['encoder_union_seconds'], .02)
        self.assertAlmostEqual(value['gpu']['observed_span_seconds'], .02)
        self.assertEqual(value['gpu']['gaps_between_attributed_samples_seconds'], 0)
        self.assertGreater(sum(g['gpu_encoder_union_seconds'] for g in value['operations']), .02)
        self.assertAlmostEqual(value['host_buckets_seconds']['host_wait_seconds'], 349 * .0002)
        self.assertAlmostEqual(value['cpu_gather_seconds'], .002)
        self.assertAlmostEqual(value['frame']['exclusive_host_seconds'], 349 * .0006 + .002)
        self.assertEqual(value['completed_counters'],
                         dict(dispatches_encoded=349, commands_submitted=349, host_wait_calls=349))
        by_operation = {(row['role'], row['layer']): row for row in value['operations']}
        # Independent source geometry: 262144-row head /8192 =>32 tiles;
        # layer15 gate12288 =>2, layer0 query2048 =>1. Each raw record has1.
        for identity, count in [(('output_head', None), 32), (('feed_forward_gate', 15), 2), (('query', 0), 1)]:
            self.assertEqual(by_operation[identity]['completed_counters'],
                             dict(dispatches_encoded=count, commands_submitted=count, host_wait_calls=count))
        by_role = {row['role']: row for row in value['role_totals']}
        self.assertEqual(by_role['query']['completed_counters'],
                         dict(dispatches_encoded=35, commands_submitted=35, host_wait_calls=35))
        self.assertEqual(by_role['feed_forward_gate']['completed_counters'],
                         dict(dispatches_encoded=55, commands_submitted=55, host_wait_calls=55))
        for key in ('dispatches_encoded', 'commands_submitted', 'host_wait_calls'):
            self.assertEqual(sum(row['completed_counters'][key] for row in value['operations']), 349)
        self.assertFalse(value['timing_qualified'])

    def test_full_descriptor_range_tail_prefill_last_decode_and_hidden_stop(self):
        for data, expected in [(fixture(33, 2, 2, ordinal=2), (32, 1, 33)),
                               (fixture(384, 128, 128, ordinal=139), (510, 1, 511)),
                               (fixture(256, 256, 256, ordinal=263), (510, 1, 511)),
                               (fixture(511, 1, 1, ordinal=16), (480, 31, 511)),
                               (fixture(1, 2, 1, stop=50), (0, 1, 1))]:
            value = self.validate(data)
            cap = value['capture']
            self.assertEqual((cap['position'], cap['input_rows'], cap['committed_positions']), expected)
        for stop in (1, 106, 50):
            self.validate(fixture(15, 2, 1, stop=stop))
        # The original frozen O1 descriptor requires a nonempty golden reply.
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.validate(fixture(1, 1, 0, stop=50))

    def test_available_zero_is_retained_but_missing_or_failed_capture_is_not_zero(self):
        value = self.validate(fixture(maximum=1, emitted=1, zero=True))
        self.assertEqual(value['availability'], 'complete')
        self.assertEqual(value['gpu']['encoder_union_seconds'], 0)
        self.assertEqual(value['host_buckets_seconds']['host_wait_seconds'], 0)
        self.assertEqual(value['profiled']['decode_evaluations'], 0)
        for target in ('state', 'overflowed'):
            data = fixture(zero=True)
            data[1]['status'][target] = 'invalid' if target == 'state' else True
            with self.assertRaises(ValueError): self.validate(data)
        data = fixture(zero=True); del data[1]['tiles'][0]['event']['gpu_begin_ticks']
        with self.assertRaises(ValueError): self.validate(data)

    def test_identity_parity_and_memory_manifest_cannot_be_relabelled(self):
        changes = [(['model', 'revision'], '0' * 40), (['parity', 'selected_logits_count'], 1),
                   (['parity', 'selected_logits_bitwise'], False), (['timing_qualified'], True),
                   (['capture', 'request_id'], [True] * 16), (['capture', 'evaluation_ordinal'], 3),
                   (['workload', 'capacity'], 513), (['memory', 'owner_reserved_bytes'], 0),
                   (['memory', 'profiled_admitted_bytes'], 6 * 1024**3),
                   (['profiled', 'decode_evaluations'], 0), (['plain', 'reason'], {'kind': 'cancelled'})]
        for path, value in changes:
            data = fixture(); node = data[0]
            for key in path[:-1]: node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError): self.validate(data)

    def test_current_inventory_geometry_and_semantic_source_ids_are_binding(self):
        for key, value in [('source_rows', 1), ('source_matrix_slot', 1), ('prefill_rows', 1),
                           ('source_bytes', True), ('execution', 'unknown'), ('kernel_entry', 'other')]:
            data = fixture(); data[2]['operations'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(data)
        for modification in ('duplicate', 'missing', 'foreign', 'owner'):
            data = fixture(); inventory = data[2]
            if modification == 'duplicate': inventory['operations'][-1] = copy.deepcopy(inventory['operations'][0])
            if modification == 'missing': inventory['operations'].pop()
            if modification == 'foreign': inventory['operations'][0]['id']['role'] = 'unknown'
            if modification == 'owner': inventory['resident_gpu_tile_owners'] = 350
            with self.subTest(modification=modification), self.assertRaises(ValueError): self.validate(data)

    def test_complete_slots_shapes_copy_counters_gathers_and_frame_partition(self):
        changes = [(['tiles', 0, 'record', 'counter_slot'], True),
                   (['tiles', 0, 'record', 'attribution', 'source_first_row'], 1),
                   (['tiles', 0, 'record', 'attribution', 'input_rows'], 32),
                   (['tiles', 0, 'record', 'input_copy_bytes'], 0),
                   (['tiles', 0, 'record', 'dispatches_encoded'], 0),
                   (['tiles', 0, 'record', 'commands_submitted'], 0),
                   (['tiles', 0, 'record', 'host_wait_calls'], 0),
                   (['cpu_gathers', 0, 'active_source_bytes'], 1),
                   (['cpu_gathers', 0, 'columns'], 1), (['exclusive_host_seconds'], 0.0),
                   (['unattributed_host_seconds'], 0.0), (['status', 'tiles_recorded'], 348)]
        for path, value in changes:
            data = fixture(); node = data[1]
            for key in path[:-1]: node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError): self.validate(data)
        data = fixture(); data[1]['tiles'].pop()
        with self.assertRaises(ValueError): self.validate(data)

    def test_paired_clock_formula_sentinel_wrap_and_duplicate_event_bindings(self):
        for damage in ('cpu-span', 'gpu-span', 'sentinel', 'outside', 'conversion', 'mirror', 'host'):
            data = fixture(); frame = data[1]; event = frame['tiles'][0]['event']
            if damage == 'cpu-span': frame['clocks']['end']['cpu_nanoseconds'] = 10**18
            if damage == 'gpu-span': frame['clocks']['end']['gpu_ticks'] = 999
            if damage == 'sentinel': event['gpu_begin_ticks'] = 2**64 - 1
            if damage == 'outside': event['gpu_begin_ticks'] = 999
            if damage == 'conversion': event['gpu_end_seconds'] = .0300001
            if damage == 'mirror': frame['clocks']['events'][0]['gpu_end_ticks'] = 1400
            if damage == 'host': event['host_end_seconds'] = 2.0
            if damage != 'mirror': frame['clocks']['events'][0] = copy.deepcopy(event)
            with self.subTest(damage=damage), self.assertRaises(ValueError): self.validate(data)

    def test_overflowing_host_aggregation_is_rejected_without_a_partial_report(self):
        data = fixture()
        for tile in data[1]['tiles']:
            tile['record']['host_staging_seconds'] = 1e308
        with self.assertRaisesRegex(ValueError, 'overflow'):
            self.validate(data)

    def test_unknown_schema_duplicate_nonfinite_oversize_symlink_and_corrupt_raw_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); data = fixture(); write_bundle(folder, *data)
            original = (folder / 'manifest.json').read_bytes()
            bad = [original[:-1] + b',"format":"duplicate"}', original[:-1] + b',"unknown":1}',
                   original.replace(b'"timing_qualified":false', b'"timing_qualified":NaN'),
                   original.replace(b'"total_seconds":0.5', b'"total_seconds":1e999'), b' ' * (64 * 1024 + 1)]
            for raw in bad:
                (folder / 'manifest.json').write_bytes(raw)
                with self.assertRaises(ValueError): report.report_bundle(folder)
            (folder / 'manifest.json').write_bytes(original)
            profile = folder / 'profile.json'; saved = profile.read_bytes()
            profile.write_bytes(saved + b' ')
            with self.assertRaises(ValueError): report.report_bundle(folder)
            profile.unlink(); (folder / 'target.json').write_bytes(saved); profile.symlink_to('target.json')
            with self.assertRaises(ValueError): report.report_bundle(folder)
            profile.unlink(); profile.write_bytes(b' ' * (1024**2 + 1))
            with self.assertRaises(ValueError): report.report_bundle(folder)

    def test_cli_roundtrip_bounded_markdown_and_existing_outputs_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); write_bundle(folder, *fixture())
            out, md = folder / 'cost.json', folder / 'cost.md'
            self.assertEqual(report.main([str(folder), '--json', str(out), '--markdown', str(md)]), 0)
            value = json.loads(out.read_bytes())
            self.assertEqual(value, report.report_bundle(folder))
            self.assertLess(len(out.read_bytes()), 1024**2)
            self.assertLess(len(md.read_bytes()), 64 * 1024)
            table = md.read_text().split('## Projection role totals')[0]
            self.assertEqual(sum(line.startswith('| ') for line in table.splitlines()), 21)
            before = out.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                report.main([str(folder), '--json', str(out)])
            self.assertEqual(failure.exception.code, 2)
            self.assertEqual(out.read_bytes(), before)

    def test_invalid_cli_bundle_creates_no_report_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); data = fixture()
            data[0]['parity']['report_counts'] = False
            write_bundle(folder, *data)
            out, md = folder / 'cost.json', folder / 'cost.md'
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                report.main([str(folder), '--json', str(out), '--markdown', str(md)])
            self.assertEqual(failure.exception.code, 2)
            self.assertFalse(out.exists())
            self.assertFalse(md.exists())

    def test_v2_prefill_decode_and_single_row_tail_groups_are_capture_specific(self):
        for kwargs, groups, commands in [({}, 41, 278), ({'ordinal': 2}, 42, 277),
                                         ({'input_count': 33, 'ordinal': 2}, 42, 277),
                                         ({'input_count': 384, 'maximum': 128, 'emitted': 128, 'ordinal': 12}, 41, 278),
                                         ({'input_count': 384, 'maximum': 128, 'emitted': 128, 'ordinal': 139}, 42, 277)]:
            value = self.validate(batched_fixture(**kwargs))
            self.assertEqual(value['format'], 'toklet-gemma-profile-cost-v2')
            self.assertEqual(value['submission_policy'], 'batched_single_row')
            self.assertEqual(len(value['submission_groups']), groups)
            self.assertEqual(value['completed_counters'],
                             dict(dispatches_encoded=349, commands_submitted=commands, host_wait_calls=commands))
            head = next(row for row in value['operations'] if row['role'] == 'output_head')
            self.assertEqual(head['completed_counters'], dict(dispatches_encoded=32, commands_submitted=1, host_wait_calls=1))
            self.assertEqual((head['input_copy_bytes'], head['output_copy_bytes']), (6144, 1048576))
            self.assertAlmostEqual(value['gpu']['encoder_union_seconds'], .02)
        zero = self.validate(batched_fixture(maximum=1, emitted=1, zero=True))
        self.assertEqual(zero['gpu']['encoder_union_seconds'], 0)
        self.assertEqual(zero['availability'], 'complete')

    def test_v1_wire_does_not_silently_gain_group_or_policy_fields(self):
        data = fixture()
        self.assertEqual(len(data[2]), 6)
        self.assertNotIn('submission_groups', data[1])
        value = self.validate(data)
        self.assertNotIn('submission_groups', value)
        self.assertNotIn('submission_policy', value)
        self.assertEqual(value['format'], 'toklet-gemma-profile-cost-v1')
        for part, key, value in [(0, 'submission_policy', 'batched_single_row'),
                                 (1, 'submission_groups', []), (2, 'schema_version', 2)]:
            data = fixture(); data[part][key] = value
            with self.subTest(part=part), self.assertRaises(ValueError): self.validate(data)

    def test_v2_rejects_old_trace_relabel_and_mixed_manifest_frame_inventory_versions(self):
        for part in ('manifest', 'frame', 'inventory', 'missing-groups', 'old-records'):
            data = batched_fixture()
            if part == 'manifest': data[0]['format'] = 'toklet-gemma-profile-v1'
            if part == 'frame': data[1]['schema_version'] = 1
            if part == 'inventory': del data[2]['schema_version']
            if part == 'missing-groups': data[1]['submission_groups'] = []
            if part == 'old-records': data[1]['tiles'] = fixture()[1]['tiles']
            with self.subTest(part=part), self.assertRaises(ValueError): self.validate(data)

    def test_v2_sparse_groups_require_exact_ranges_owners_shapes_and_eligible_set(self):
        changes = [('group_id', True), ('owner_counter_slot', 1), ('first_counter_slot', 348),
                   ('tile_count', 1), ('input_rows', 2), ('input_features', 1),
                   ('output_features', 8192), ('source_matrix_slot', 277)]
        for key, value in changes:
            data = batched_fixture(); data[1]['submission_groups'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(data)
        for damage in ('duplicate-id', 'overlap', 'missing', 'extra', 'unknown', 'noncontiguous-tile'):
            data = batched_fixture(); groups = data[1]['submission_groups']
            if damage == 'duplicate-id': groups[1]['group_id'] = 0
            if damage == 'overlap': groups[1]['first_counter_slot'] = groups[0]['first_counter_slot']
            if damage == 'missing': groups.pop()
            if damage == 'extra': groups.append(copy.deepcopy(groups[-1]))
            if damage == 'unknown': groups[0]['operation']['role'] = 'unknown'
            if damage == 'noncontiguous-tile': data[1]['tiles'][1]['record']['attribution']['tile_index'] = 2
            with self.subTest(damage=damage), self.assertRaises(ValueError): self.validate(data)

    def test_v2_shared_counts_bytes_and_host_cost_belong_only_to_first_owner(self):
        for key, value in [('commands_submitted', 1), ('host_wait_calls', 1), ('input_copy_bytes', 6144),
                           ('output_copy_bytes', 32768), ('host_staging_seconds', .0001),
                           ('host_commit_seconds', .0001), ('host_wait_seconds', .0001),
                           ('host_validation_and_copy_seconds', .0001), ('dispatches_encoded', 0)]:
            data = batched_fixture(); slot = data[1]['submission_groups'][0]['first_counter_slot'] + 1
            data[1]['tiles'][slot]['record'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(data)
        for key in ('commands_submitted', 'host_wait_calls', 'input_copy_bytes', 'output_copy_bytes'):
            data = batched_fixture(); slot = data[1]['submission_groups'][0]['owner_counter_slot']
            data[1]['tiles'][slot]['record'][key] = 0
            with self.subTest(owner=key), self.assertRaises(ValueError): self.validate(data)
        data = batched_fixture()
        fallback = next(tile['record'] for tile in data[1]['tiles'] if tile['record']['attribution']['operation'] == dict(role='query', layer=0))
        fallback['commands_submitted'] = 0
        with self.assertRaises(ValueError): self.validate(data)

    def test_v2_does_not_expand_frozen_cli_chunk_or_accept_false_decode_predictions(self):
        for damage in ('chunk', 'scratch-fit', 'decode-wait', 'configured-prefill'):
            data = batched_fixture()
            if damage == 'chunk': data[0]['workload']['chunk'] = 3
            op = next(op for op in data[2]['operations'] if op['id']['role'] == 'output_head')
            if damage == 'scratch-fit': op['predicted_decode_commands_per_chunk'] = 32
            if damage == 'decode-wait': op['predicted_decode_host_waits_per_chunk'] = 32
            if damage == 'configured-prefill': op['predicted_commands_per_chunk'] = 32
            with self.subTest(damage=damage), self.assertRaises(ValueError): self.validate(data)


if __name__ == '__main__':
    unittest.main()
