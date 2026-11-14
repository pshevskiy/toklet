"""Independent bounded wire fixtures; synthetic timings are never measurements.

The joined synthetic fixture may patch trusted input-pin constants only. It does
not patch validators or claim authentic codec/model/runtime qualification. An
unpatched frozen-input replay remains ROOT work after the cold384-ID export.
"""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('submission_report_under_test', HERE / 'gemma_submission_report.py')
REPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORTER
SPEC.loader.exec_module(REPORTER)

LIMIT = 6 * 1024**3
CALLER, HELPER = 16 * 1024**2, 32 * 1024**2
LOAD, SESSION, RESERVED = 4_000_000_000, 5_000_000_000, 1_000_000
MAXIMUM = max(LOAD + CALLER + HELPER, SESSION + RESERVED + CALLER + HELPER)
REPOSITORY = 'google/gemma-4-E2B-it'
REVISION = '70af34e20bd4b7a91f0de6b22675850c43922a03'
CODEC = '654dc6dfd66fd893a24f0faa39355b71d5e7ffc3f53c2dac394dfcfd9ed95298'
WORKLOAD_SHA = 'de4ebba10f7bcf6a721c973da65ca8c0a246d04cf7471e159a18e460b4bcc4fd'
P = [2, 105, 2364, 107, 1567, 506, 5279, 529, 7001, 236761, 106, 107, 105, 4368, 107]
E = P[:-1] + [2364]
Q = [818, 5279, 529, 7001]
ARMS = ('per-tile', 'batched-single-row')
CASES = ('reset', 'append', 'fully_cached', 'pending', 'edited_prefix',
         'short_128_128', 'boundary_prefill_512_0', 'teacher_128_128', 'teacher_384_128')
# Deliberately synthetic IDs, disjoint from numerical/tokenizer qualification.
PRIMARY_INPUT = list(range(1000, 1384))
PRIMARY_OUTPUT = list(range(2000, 2128))
TOKENS = [3000 + i for i in range(1025)]
EDITED_OUTPUT = [4001, 4002, 4003, 4004]
SHORT_OUTPUT = list(range(5000, 5128))


def raw_json(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def journal_bytes(rows):
    return b''.join(raw_json(row) + b'\n' for row in rows)


def id_bytes(ids):
    return struct.pack('<' + 'I' * len(ids), *ids)


def ids_sha(ids):
    return sha(id_bytes(ids))


def state(ids, pending=0):
    return dict(history=len(ids), committed=len(ids) - pending, pending=pending,
                history_sha256=ids_sha(ids))


def core(ids, *, reused=0, prefill=0, sampled=0, emitted=None, decode=0,
         pending=0, seconds=1.0, reason='output_limit', stop=None):
    if emitted is None:
        emitted = sampled
    why = dict(kind=reason)
    if stop is not None:
        why['token_id'] = stop
    return dict(reason=why, reused_tokens=reused, prefill_tokens=prefill,
                sampled_tokens=sampled, emitted_tokens=emitted,
                history_tokens=len(ids), evaluated_tokens=len(ids) - pending,
                pending_tokens=pending, first_token_seconds=seconds * .3 if emitted else None,
                prefill_seconds=seconds * .25 if prefill else 0.0,
                decode_seconds=seconds * .5 if decode else 0.0,
                sampling_seconds=seconds * .05 if sampled else 0.0,
                total_seconds=seconds * .9, decode_evaluations=decode)


def admission():
    return dict(session_admitted_bytes=SESSION, owner_reserved_bytes=RESERVED,
                caller_allowance_bytes=CALLER, helper_allowance_bytes=HELPER,
                combined_session_admitted_bytes=SESSION + RESERVED + CALLER + HELPER,
                maximum_with_load_admitted_bytes=MAXIMUM, limit_bytes=LIMIT)


def callbacks(count, seconds):
    return [seconds * (.35 + .5 * i / max(1, count - 1)) for i in range(count)]


def primary_fixture(cohort=1, sustained_arm=None):
    mode = dict(kind='paired', cohort=cohort) if sustained_arm is None else dict(kind='sustained', arm=sustained_arm)
    ready = dict(event='ready', format='toklet-gemma-submission-bench-v1', mode=mode,
                 repository=REPOSITORY, revision=REVISION, workload_sha256=WORKLOAD_SHA,
                 input_sha256=ids_sha(PRIMARY_INPUT), expected_output_sha256=ids_sha(PRIMARY_OUTPUT),
                 input_ids=dict(file='input_ids.u32le', bytes=1536, sha256=ids_sha(PRIMARY_INPUT)),
                 input_tokens=384, maximum_output_tokens=128, capacity=512, chunk=32,
                 cache_policy='new_session_reset_each_request', profiling=False,
                 selected_logit_witness=False, load_seconds=1.0, model_setup_seconds=.1,
                 tokenization_seconds=.01, owner_load_admitted_bytes=LOAD,
                 owner_reserved_bytes=RESERVED, caller_allowance_bytes=CALLER,
                 helper_allowance_bytes=HELPER, limit_bytes=LIMIT,
                 clock_scope='Instant around completed generate; callbacks same caller origin. Core Report has its own origin. Setup/tokenization/check/hash/export excluded.',
                 source_build_hardware_guard_provenance='ROOT external seals required; no UI/tails/energy/first-visible or adoption claim')
    rows = [ready]

    def request(arm, warmup, trial, within, order, duration):
        times = callbacks(128, duration)
        row = dict(event='request', arm=arm, passed=True, generation_error=None,
                   setup_seconds=.01, complete_request_seconds=duration,
                   first_callback_seconds=times[0], last_callback_seconds=times[-1],
                   token_event_seconds=times,
                   report=core(PRIMARY_INPUT + PRIMARY_OUTPUT, prefill=384, sampled=128,
                               decode=127, pending=1, seconds=duration),
                   output_ids=PRIMARY_OUTPUT[:], output_sha256=ids_sha(PRIMARY_OUTPUT),
                   history_sha256=ids_sha(PRIMARY_INPUT + PRIMARY_OUTPUT),
                   state=dict(history=512, committed=511, pending=1),
                   state_matches=True, output_matches=True, postcheck_seconds=.01,
                   sequence=len(rows) - 1, warmup=warmup, pair_index=trial,
                   within_pair=within, order=order, **admission())
        rows.append(row)

    if sustained_arm is None:
        for warmup, count in ((True, 3), (False, 10)):
            for trial in range(count):
                arms = ARMS if (trial + cohort) % 2 else ARMS[::-1]
                order = 'AB' if arms[0] == ARMS[0] else 'BA'
                for within, arm in enumerate(arms):
                    duration = (90.0 if warmup else 40.0 + trial / 10 + cohort / 100)
                    if arm == ARMS[1]:
                        duration *= .9
                    request(arm, warmup, trial, within, order, duration)
    else:
        for i, duration in enumerate((50.0, 51.0, 52.0, 301.0, 302.0)):
            if sustained_arm == ARMS[1]:
                duration += .5
            request(sustained_arm, i < 3, None, None, None, duration)
    measured = [row for row in rows[1:] if not row['warmup']]
    manifest = dict(format='toklet-gemma-submission-bench-manifest-v1', passed=True, complete=True,
                    configuration=copy.deepcopy(ready), requests=len(rows) - 1,
                    measured_requests=len(measured),
                    measured_complete_request_seconds=sum(row['complete_request_seconds'] for row in measured),
                    maximum_with_load_admitted_bytes=MAXIMUM,
                    events=dict(file='events.jsonl', bytes=0, sha256='0' * 64),
                    performance_qualified=False, engine_adoption_qualified=False,
                    scope='Ordinary numerical requests only; paired/sustained arms never pool. Root validates original performance/quality/continuation/observer gates.')
    raw = journal_bytes(rows)
    manifest['events'].update(bytes=len(raw), sha256=sha(raw))
    return rows, manifest


def retime_primary(row, seconds):
    times = callbacks(128, seconds)
    row.update(complete_request_seconds=seconds, token_event_seconds=times,
               first_callback_seconds=times[0], last_callback_seconds=times[-1],
               report=core(PRIMARY_INPUT + PRIMARY_OUTPUT, prefill=384, sampled=128,
                           decode=127, pending=1, seconds=seconds))


def golden_fixture(tokens=TOKENS):
    def expected(input_count, ids):
        return dict(input_tokens=input_count, maximum=len(ids), ids=ids[:],
                    output_sha256=ids_sha(ids), history=input_count + len(ids),
                    committed=input_count + len(ids) - 1, pending=1)
    return dict(format='toklet-gemma-submission-matrix-goldens-v1', repository=REPOSITORY,
                revision=REVISION, model_and_codec_manifest_sha256=CODEC,
                tokens_sha256=ids_sha(tokens), prompt_sha256=ids_sha(P),
                edited_input_sha256=ids_sha(E), short_input_sha256=ids_sha(tokens[:128]),
                edited=expected(15, EDITED_OUTPUT), short=expected(128, SHORT_OUTPUT),
                baseline_candidate_parity=True)


def matrix_request(case, arm, duration, tokens=TOKENS):
    seeds = {
        'reset': (P, Q), 'edited_prefix': (P, Q), 'append': (P[:14], []),
        'fully_cached': (P, []), 'pending': (P, Q[:1])}
    if case in seeds:
        seed_input, seed_output = seeds[case]
        before_ids = seed_input + seed_output
        before = state(before_ids, int(bool(seed_output)))
        seed_report = core(before_ids, prefill=len(seed_input), sampled=len(seed_output),
                           decode=max(0, len(seed_output) - 1), pending=int(bool(seed_output)), seconds=.1)
    else:
        before, seed_report = state([]), None
    row = dict(event='request', case=case, arm=arm, passed=True, setup_seconds=.01,
               before=before, seed_report=seed_report, admission=admission())
    if case.startswith('teacher_'):
        count = 128 if case == 'teacher_128_128' else 384
        scale = duration / 10
        prefill = [dict(rows=32, position=32 * (i + 1), seconds=.25 * scale)
                   for i in range(count // 32)]
        decode = [dict(step=i, token_id=tokens[count + i], position=count + i + 1,
                       seconds=.015 * scale,
                       logits_sha256=sha(('fixture-logits-%s' % (count + i + 1)).encode()),
                       greedy_id=(count + i + 1) % 262144) for i in range(128)]
        numerical = sum(item['seconds'] for item in prefill + decode)
        row.update(timing_kind='teacher_forced_evaluations', completed_numerical_seconds=numerical,
                   wall_with_posttimer_hash_seconds=numerical + .5, prefill=prefill,
                   prefill_logits_sha256=sha(('fixture-logits-%s' % count).encode()),
                   prefill_greedy_id=count, decode=decode,
                   after=state(tokens[:count + 128]), callback_seconds=None,
                   scope='Completed evaluation clocks only; append/hash/comparison excluded. No generation TTFT.')
    else:
        spec = {
            'reset': (P, Q, 0, 15, 3), 'append': (P, Q, 14, 1, 3),
            'fully_cached': (P, Q, 15, 0, 3), 'pending': (P + Q[:1], Q[1:], 15, 1, 2),
            'edited_prefix': (E, EDITED_OUTPUT, 0, 15, 3),
            'short_128_128': (tokens[:128], SHORT_OUTPUT, 0, 128, 127),
            'boundary_prefill_512_0': (tokens[:512], [], 0, 512, 0)}
        base, output, reused, prefill, decode = spec[case]
        pending = int(bool(output))
        times = callbacks(len(output), duration)
        row.update(timing_kind='generation', complete_request_seconds=duration,
                   token_event_seconds=times, first_callback_seconds=times[0] if times else None,
                   last_callback_seconds=times[-1] if times else None,
                   report=core(base + output, reused=reused, prefill=prefill,
                               sampled=len(output), decode=decode, pending=pending, seconds=duration),
                   output_ids=output[:], output_sha256=ids_sha(output), after=state(base + output, pending))
    return row


def matrix_controls(arm):
    before = state(P + Q, 1)
    rows = [dict(event='control', case='capacity_385_128', arm=arm, passed=True,
                 before=copy.deepcopy(before), after=copy.deepcopy(before))]
    rows.append(dict(event='control', case='pre_cancel', arm=arm, passed=True,
                     report=core(P + Q, pending=1, seconds=.05, reason='cancelled'),
                     after=copy.deepcopy(before)))
    rows.append(dict(event='control', case='callback_cancel', arm=arm, passed=True,
                     report=core(P + Q[:1], prefill=15, sampled=1, decode=0,
                                 pending=1, seconds=.05, reason='cancelled'),
                     after=state(P + Q[:1], 1)))
    rows.append(dict(event='control', case='hidden_stop', arm=arm, passed=True,
                     report=core(P + Q[:1], prefill=15, sampled=1, emitted=0, decode=0,
                                 pending=1, seconds=.05, reason='stop_token', stop=818),
                     after=state(P + Q[:1], 1)))
    return rows


def matrix_fixture(cohort=1, prepare=False, tokens=TOKENS):
    golden_sha = sha(raw_json(golden_fixture(tokens)))
    ready = dict(event='ready', format='toklet-gemma-submission-matrix-v1',
                 prepare_goldens=prepare, cohort=None if prepare else cohort,
                 repository=REPOSITORY, revision=REVISION,
                 model_and_codec_manifest_sha256=CODEC, tokens_sha256=ids_sha(tokens),
                 prompt_ids=P[:], prompt_sha256=ids_sha(P), edited_input_sha256=ids_sha(E),
                 short_input_sha256=ids_sha(tokens[:128]), goldens_sha256=None if prepare else golden_sha,
                 capacity=512, chunk=32,
                 stop_policy='empty for fixed numerical cases; hidden-stop control818 only',
                 load_seconds=1.0, tokenization_seconds=.01, owner_load_admitted_bytes=LOAD,
                 caller_allowance_bytes=CALLER, helper_allowance_bytes=HELPER, limit_bytes=LIMIT,
                 profiling=False, selected_logit_witness=False,
                 scope='Callbacks time completed generation; teacher-forced clocks time completed evaluations with hash/check outside. No UI/tails/adoption; ROOT provenance/physical/power guards required.')
    rows = [ready]
    if prepare:
        for case in ('edited_prefix', 'short_128_128'):
            for arm in ARMS:
                rows.append(matrix_request(case, arm, 9.0, tokens))
    else:
        for arm in ARMS:
            rows.extend(matrix_controls(arm))
        for warmup, count in ((True, 3), (False, 10)):
            for trial in range(count):
                for case in CASES:
                    arms = ARMS if (trial + cohort) % 2 else ARMS[::-1]
                    for within, arm in enumerate(arms):
                        duration = 90.0 if warmup else 10.0 + cohort / 10 + trial / 100
                        if arm == ARMS[1]:
                            duration *= .9
                        row = matrix_request(case, arm, duration, tokens)
                        row.update(warmup=warmup, trial=trial, within_pair=within,
                                   order='AB' if arms[0] == ARMS[0] else 'BA')
                        rows.append(row)
    manifest = dict(format='toklet-gemma-submission-matrix-manifest-v1', passed=True, complete=True,
                    prepare_goldens=prepare, cohort=None if prepare else cohort,
                    requests=4 if prepare else 234, cases=list(CASES),
                    warmup_pairs=0 if prepare else 3, measured_pairs=0 if prepare else 10,
                    events=dict(file='events.jsonl', bytes=0, sha256='0' * 64),
                    goldens_sha256=None if prepare else golden_sha,
                    performance_qualified=False, engine_adoption_qualified=False,
                    scope='Fixed matrix only; primary/sustained separate. Unsupported512/1024/1920+128 admission not truncated or waived; ROOT retains full original quality/performance gates.')
    raw = journal_bytes(rows)
    manifest['events'].update(bytes=len(raw), sha256=sha(raw))
    return rows, manifest


class SubmissionReport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='toklet-submission-report-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, name, raw):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return dict(path=str(path), bytes=len(raw), sha256=sha(raw))

    def primary(self, rows=None, manifest=None, **kwargs):
        if rows is None:
            rows, manifest = primary_fixture(**kwargs)
        return REPORTER.validate_primary(rows, manifest, PRIMARY_INPUT, PRIMARY_OUTPUT, **kwargs)

    def matrix(self, rows=None, manifest=None, **kwargs):
        if rows is None:
            rows, manifest = matrix_fixture(**kwargs)
        return REPORTER.validate_matrix(rows, manifest, TOKENS, golden_fixture(), **kwargs)

    def rewrite_bundle(self, spec, slot, rows, manifest):
        bundle = spec['golden_preparation'] if slot == 'golden_preparation' else spec['bundles'][slot]
        prefix = Path(bundle['manifest']['path']).parent.relative_to(self.directory)
        raw = journal_bytes(rows)
        manifest['events'].update(bytes=len(raw), sha256=sha(raw))
        bundle['events'] = self.write(str(prefix / 'events.jsonl'), raw)
        bundle['manifest'] = self.write(str(prefix / 'manifest.json'), raw_json(manifest))

    def sealed_fixture(self):
        """A complete synthetic byte join, not authenticated runtime evidence."""
        conversation = '[{"role":"user","content":"synthetic offline fixture"}]'
        workload = dict(schema_version=1, workload_id='synthetic-offline-test',
                        conversation_json=conversation, conversation_sha256=sha(conversation.encode()),
                        input_token_count=384, input_tokens_sha256=ids_sha(PRIMARY_INPUT),
                        maximum_output_tokens=128, capacity=512, chunk=32,
                        cache_policy='reset_chat_each_request',
                        expected_output=dict(tokens=PRIMARY_OUTPUT, history=512, committed=511,
                                             pending=1, reason=1, stop_token=2**32 - 1))
        workload_pin = self.write('inputs/workload.json', raw_json(workload))
        input_pin = self.write('inputs/primary.u32le', id_bytes(PRIMARY_INPUT))
        token_pin = self.write('inputs/tokens.u32le', id_bytes(TOKENS))
        golden_pin = self.write('golden_preparation/goldens.json', raw_json(golden_fixture()))
        source_pin = self.write('proof/source.json', raw_json(dict(synthetic=True)))
        binaries = {kind: self.write('proof/' + kind + '.bin', b'synthetic bytes; never executed: ' + kind.encode())
                    for kind in ('primary', 'matrix')}
        provenance = dict(source_manifest=source_pin)
        for kind, binary in binaries.items():
            receipt = dict(binary, source_freeze_sha256=source_pin['sha256'])
            provenance[kind + '_build'] = self.write('proof/' + kind + '-build.json', raw_json(receipt))
        for name in ('hardware', 'runtime_enforcement'):
            # These are opaque pinned references, not an invented proof schema.
            provenance[name] = self.write('proof/' + name + '.json', raw_json(dict(synthetic=True)))
        spec = dict(format='toklet-gemma-submission-report-inputs-v1', root_provenance=provenance,
                    inputs=dict(primary_workload=workload_pin, primary_input_ids=input_pin,
                                matrix_tokens=token_pin, matrix_goldens=golden_pin),
                    golden_preparation=None, bundles={})
        model_directory = str(self.directory / 'model-is-never-opened')
        rows_by_slot = {}
        for slot in ('golden_preparation', 'primary_1', 'primary_2', 'matrix_1', 'matrix_2',
                     'sustained_per_tile', 'sustained_batched'):
            matrix = slot.startswith('matrix') or slot == 'golden_preparation'
            if matrix:
                prepare = slot == 'golden_preparation'
                cohort = 1 if prepare else int(slot[-1])
                rows, manifest = matrix_fixture(cohort=cohort, prepare=prepare)
            else:
                if slot.startswith('primary'):
                    rows, manifest = primary_fixture(cohort=int(slot[-1]))
                else:
                    arm = ARMS[0] if slot == 'sustained_per_tile' else ARMS[1]
                    rows, manifest = primary_fixture(sustained_arm=arm)
                rows[0]['workload_sha256'] = workload_pin['sha256']
                manifest['configuration'] = copy.deepcopy(rows[0])
                self.write(slot + '/input_ids.u32le', id_bytes(PRIMARY_INPUT))
            folder = self.directory / slot
            args = ['--model', model_directory]
            if matrix:
                args += ['--tokens', token_pin['path'], '--output', str(folder)]
                if slot == 'golden_preparation':
                    args += ['--prepare-goldens']
                else:
                    args += ['--goldens', golden_pin['path'], '--goldens-sha256', golden_pin['sha256'],
                             '--cohort', slot[-1]]
            else:
                args += ['--workload', workload_pin['path'], '--output', str(folder)]
                args += ['--cohort', slot[-1]] if slot.startswith('primary') else ['--sustained-arm', arm]
            guard_name = slot + '-guard/'
            plan = dict(command=[binaries['matrix' if matrix else 'primary']['path']] + args,
                        cwd=str(self.directory), limit_bytes=LIMIT, helper_reserve_bytes=HELPER,
                        configured_guest_reserve_bytes=0, total_reserved_bytes=HELPER,
                        timeout_seconds=100000.0, log_limit_bytes=16 * 1024**2,
                        disk_floor_bytes=2 * 1024**3)
            stdout, stderr = b'synthetic offline fixture\n', b''
            result = dict(passed=True, failure=None, exit_code=0, elapsed_seconds=10000.0,
                          maximum_observed_charge_with_reserve=MAXIMUM,
                          terminal_direct_pair_rss_charge_with_reserve=MAXIMUM,
                          minimum_free_bytes=10 * 1024**3, sample_count=2,
                          owned_live_after_cleanup=[], cleanup_errors=[],
                          helper_reserve_bytes=HELPER, configured_guest_reserve_bytes=0,
                          stdout_sha256=sha(stdout), stderr_sha256=sha(stderr),
                          last_runtime_gate=dict(passed=True, power="Now drawing from 'AC Power'",
                                                 thermalState=0, lowPowerMode=False, checked_uptime_ns=12345),
                          memory_scope='Synthetic pinned record only; no enforcement qualification.')
            guard = dict(plan=self.write(guard_name + 'plan.json', raw_json(plan)),
                         result=self.write(guard_name + 'result.json', raw_json(result)),
                         memory=self.write(guard_name + 'memory.jsonl', b'{"synthetic":true}\n'),
                         stdout=self.write(guard_name + 'stdout.log', stdout),
                         stderr=self.write(guard_name + 'stderr.log', stderr))
            bundle = dict(manifest=dict(path=str(folder / 'manifest.json')),
                          events=dict(path=str(folder / 'events.jsonl')), guard=guard,
                          quiet_barrier=self.write(slot + '-quiet.json', raw_json(dict(synthetic=True))))
            if slot == 'golden_preparation':
                spec['golden_preparation'] = bundle
            else:
                spec['bundles'][slot] = bundle
            self.rewrite_bundle(spec, slot, rows, manifest)
            rows_by_slot[slot] = (rows, manifest)
        trusted = dict(WORKLOAD_SHA=workload_pin['sha256'], PRIMARY_INPUT_SHA=input_pin['sha256'],
                       TOKENS_SHA=token_pin['sha256'])
        return spec, trusted, rows_by_slot

    def joined_report(self, spec, trusted):
        pin = self.write('sealed-inputs.json', raw_json(spec))
        # Explicit ROOT-authorized fixture scope: trusted cold hash constants only.
        with mock.patch.multiple(REPORTER, **trusted):
            return REPORTER.report(pin['path'], expected_sha256=pin['sha256'])

    def rewrite_json_pin(self, pin, mutate):
        value = json.loads(Path(pin['path']).read_bytes())
        mutate(value)
        return self.write(str(Path(pin['path']).relative_to(self.directory)), raw_json(value))

    def test_actual_primary_matrix_preparation_and_sustained_wire_shapes(self):
        for cohort in (1, 2):
            with self.subTest(kind='primary', cohort=cohort):
                self.primary(cohort=cohort)
            with self.subTest(kind='matrix', cohort=cohort):
                self.matrix(cohort=cohort)
        self.matrix(prepare=True)
        for arm in ARMS:
            with self.subTest(kind='sustained', arm=arm):
                self.primary(sustained_arm=arm)

    def test_primary_work_state_hashes_and_callback_clocks_are_authoritative(self):
        mutations = (
            lambda row: row['report'].__setitem__('decode_evaluations', 126),
            lambda row: row['state'].__setitem__('committed', 510),
            lambda row: row['output_ids'].__setitem__(0, 99),
            lambda row: row.__setitem__('history_sha256', '0' * 64),
            lambda row: row['token_event_seconds'].__setitem__(-1, row['complete_request_seconds'] + 1),
            lambda row: row['report'].__setitem__('first_token_seconds', None),
            lambda row: row['report'].__setitem__('reason', dict(kind='cancelled')),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                rows, manifest = primary_fixture()
                mutate(rows[7])
                with self.assertRaises((ValueError, TypeError)):
                    self.primary(rows, manifest, cohort=1)

    def test_primary_exact_fixed_pair_schedule_and_terminal_marker(self):
        for label in ('sequence', 'phase_reset', 'within_pair', 'order', 'reversal', 'partial', 'unknown'):
            with self.subTest(label=label):
                rows, manifest = primary_fixture()
                cohort = 1
                if label == 'sequence':
                    rows[8]['sequence'] = rows[7]['sequence']
                elif label == 'phase_reset':
                    rows[7]['pair_index'] = 3
                elif label == 'within_pair':
                    rows[8]['within_pair'] = 0
                elif label == 'order':
                    rows[7]['order'] = 'BA'
                elif label == 'reversal':
                    cohort = 2
                    rows[0]['mode']['cohort'] = 2
                    manifest['configuration']['mode']['cohort'] = 2
                elif label == 'partial':
                    manifest['complete'] = False
                else:
                    rows[7]['invented_metric'] = 1
                with self.assertRaises((ValueError, TypeError)):
                    self.primary(rows, manifest, cohort=cohort)

    def test_matrix_cache_seed_and_control_state_cannot_be_relabelled(self):
        for label in ('cached', 'pending', 'seed', 'capacity', 'pre_cancel', 'hidden_stop', 'missing_control'):
            with self.subTest(label=label):
                rows, manifest = matrix_fixture()
                requests = {case: next(row for row in rows if row.get('event') == 'request'
                                       and row['case'] == case) for case in CASES}
                controls = {row['case']: row for row in rows if row.get('event') == 'control'}
                if label == 'cached':
                    requests['fully_cached']['report']['prefill_tokens'] = 15
                elif label == 'pending':
                    requests['pending']['report']['decode_evaluations'] = 3
                elif label == 'seed':
                    requests['append']['seed_report']['history_tokens'] = 15
                elif label == 'capacity':
                    controls['capacity_385_128']['after']['history_sha256'] = '0' * 64
                elif label == 'pre_cancel':
                    controls['pre_cancel']['report']['prefill_tokens'] = 15
                elif label == 'hidden_stop':
                    controls['hidden_stop']['report']['first_token_seconds'] = .01
                else:
                    rows.pop(1)
                with self.assertRaises((ValueError, TypeError)):
                    self.matrix(rows, manifest, cohort=1)

    def test_teacher_requires_complete_schedule_and_full_logit_pair_parity(self):
        for label in ('prefill_count', 'prefill_rows', 'prefill_position', 'decode_token',
                      'decode_position', 'logits', 'greedy', 'numerical_sum', 'callback'):
            with self.subTest(label=label):
                rows, manifest = matrix_fixture()
                teacher = next(row for row in rows if row.get('case') == 'teacher_384_128')
                if label == 'prefill_count':
                    teacher['prefill'].pop()
                elif label == 'prefill_rows':
                    teacher['prefill'][0]['rows'] = 31
                elif label == 'prefill_position':
                    teacher['prefill'][1]['position'] = 63
                elif label == 'decode_token':
                    teacher['decode'][0]['token_id'] += 1
                elif label == 'decode_position':
                    teacher['decode'][0]['position'] = 384
                elif label == 'logits':
                    teacher['decode'][50]['logits_sha256'] = '0' * 64
                elif label == 'greedy':
                    teacher['decode'][50]['greedy_id'] += 1
                elif label == 'numerical_sum':
                    teacher['completed_numerical_seconds'] += .25
                else:
                    teacher['callback_seconds'] = 0
                with self.assertRaises((ValueError, TypeError)):
                    self.matrix(rows, manifest, cohort=1)

    def test_teacher_timings_may_differ_while_logit_proofs_stay_equal(self):
        rows, manifest = matrix_fixture()
        first = next(row for row in rows if row.get('case') == 'teacher_128_128')
        first['prefill'][0]['seconds'] += .25
        first['decode'][0]['seconds'] += .5
        first['completed_numerical_seconds'] += .75
        first['wall_with_posttimer_hash_seconds'] += .75
        self.matrix(rows, manifest, cohort=1)

    def test_admission_integer_types_and_exact_six_gib_arithmetic(self):
        for label in ('bool_count', 'float_count', 'negative_clock', 'nan_clock', 'owner_twice', 'over_budget'):
            with self.subTest(label=label):
                rows, manifest = primary_fixture()
                row = rows[7]
                if label == 'bool_count':
                    row['report']['decode_evaluations'] = True
                elif label == 'float_count':
                    row['state']['history'] = 512.0
                elif label == 'negative_clock':
                    row['setup_seconds'] = -.01
                elif label == 'nan_clock':
                    row['setup_seconds'] = float('nan')
                elif label == 'owner_twice':
                    row['combined_session_admitted_bytes'] += RESERVED
                else:
                    row['session_admitted_bytes'] = LIMIT
                    row['combined_session_admitted_bytes'] = LIMIT + RESERVED + CALLER + HELPER
                    row['maximum_with_load_admitted_bytes'] = row['combined_session_admitted_bytes']
                with self.assertRaises((ValueError, TypeError)):
                    self.primary(rows, manifest, cohort=1)

    def test_sustained_requires_first_six_hundred_second_crossing_and_one_arm(self):
        for label in ('too_short', 'continued_after_crossing', 'pair_fields', 'other_arm', 'warmup_count'):
            with self.subTest(label=label):
                rows, manifest = primary_fixture(sustained_arm=ARMS[0])
                if label == 'too_short':
                    retime_primary(rows[4], 299.0)
                    retime_primary(rows[5], 299.0)
                elif label == 'continued_after_crossing':
                    retime_primary(rows[4], 601.0)
                    retime_primary(rows[5], 1.0)
                elif label == 'pair_fields':
                    rows[4]['pair_index'] = 0
                elif label == 'other_arm':
                    rows[4]['arm'] = ARMS[1]
                else:
                    rows[3]['warmup'] = False
                measured = [row for row in rows[1:] if not row['warmup']]
                manifest['measured_requests'] = len(measured)
                manifest['measured_complete_request_seconds'] = sum(row['complete_request_seconds'] for row in measured)
                with self.assertRaises((ValueError, TypeError)):
                    self.primary(rows, manifest, sustained_arm=ARMS[0])

    def test_strict_json_rejects_nested_duplicates_nonfinite_and_invalid_utf8(self):
        self.assertEqual(REPORTER.parse(b'{"sample": [0,1.25,null,true]}'),
                         dict(sample=[0, 1.25, None, True]))
        for raw in (b'{"x":1,"x":2}', b'{"outer":{"x":1,"x":2}}',
                    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e999}', b'{"x":"\xff"}'):
            with self.subTest(raw=raw):
                with self.assertRaises((ValueError, TypeError, UnicodeError)):
                    REPORTER.parse(raw)

    def test_pin_exact_hash_size_type_and_bound(self):
        pin = self.write('small.json', b'{"fixture":1}')
        self.assertEqual(REPORTER.read_pin(pin, 64), b'{"fixture":1}')
        for label in ('hash', 'size', 'bool', 'uppercase', 'bound', 'extra'):
            with self.subTest(label=label):
                bad = copy.deepcopy(pin)
                bound = 64
                if label == 'hash':
                    bad['sha256'] = '0' * 64
                elif label == 'size':
                    bad['bytes'] += 1
                elif label == 'bool':
                    bad['bytes'] = True
                elif label == 'uppercase':
                    bad['sha256'] = pin['sha256'].upper()
                elif label == 'bound':
                    bound = 1
                else:
                    bad['optional'] = None
                with self.assertRaises((ValueError, TypeError, OSError)):
                    REPORTER.read_pin(bad, bound)

    def test_pin_rejects_symlink_and_fifo_without_a_blocking_open(self):
        pin = self.write('target.json', b'{}')
        link = self.directory / 'link.json'
        link.symlink_to(pin['path'])
        fifo = self.directory / 'pipe.json'
        os.mkfifo(fifo)
        real_open = os.open

        def checked_open(path, flags, *args, **kwargs):
            self.assertEqual(flags & (os.O_NONBLOCK | os.O_NOFOLLOW), os.O_NONBLOCK | os.O_NOFOLLOW)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(REPORTER.os, 'open', side_effect=checked_open):
            for path in (link, fifo):
                with self.subTest(path=path.name):
                    bad = dict(path=str(path), bytes=0, sha256=sha(b''))
                    with self.assertRaises((ValueError, TypeError, OSError)):
                        REPORTER.read_pin(bad, 64)

    def test_statistics_keep_paired_marginal_and_unscaled_mad_distinct(self):
        summary = REPORTER.summarize([1, 2, 3, 100])
        self.assertEqual(summary, dict(n=4, raw=[1, 2, 3, 100], median=2.5, mad=1.0, range=[1, 100]))
        paired = REPORTER.statistics_for_pairs(list(range(1, 11)),
                                             [.9, 1.6, 2.1, 2.4, 2.5, 2.4, 2.1, 1.6, .9, .5])
        self.assertAlmostEqual(paired['marginal_median_reduction_pct'], 66.36363636363636)
        self.assertEqual(paired['paired_median_reduction_pct'], 55.0)
        self.assertEqual(paired['per_tile']['n'], 10)
        for actual, expected in zip(paired['paired_reduction_pct'], [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(paired['paired_median_bootstrap_95_pct'], [30, 80]):
            self.assertAlmostEqual(actual, expected)
        for invalid in ([], [True], [float('inf')], [float('nan')]):
            with self.subTest(summary=invalid):
                with self.assertRaises((ValueError, TypeError)):
                    REPORTER.summarize(invalid)
        for left, right in (([1] * 9, [1] * 9), ([1] * 10, [1] * 9),
                            ([0] * 10, [1] * 10), ([True] * 10, [1] * 10)):
            with self.subTest(left=left, right=right):
                with self.assertRaises((ValueError, TypeError)):
                    REPORTER.statistics_for_pairs(left, right)

    def test_joined_synthetic_cli_report_stays_explicitly_unqualified(self):
        spec, trusted, captures = self.sealed_fixture()
        rows, manifest = captures['primary_1']
        for row in rows[1:]:
            row['token_event_seconds'] = [row['first_callback_seconds']] * 128
            row['last_callback_seconds'] = row['first_callback_seconds']
        self.rewrite_bundle(spec, 'primary_1', rows, manifest)
        sealed = self.write('sealed-inputs.json', raw_json(spec))
        output = self.directory / 'report-output'
        argv = ['gemma_submission_report.py', '--inputs', sealed['path'],
                '--inputs-sha256', sealed['sha256'], '--output', str(output)]
        with mock.patch.multiple(REPORTER, **trusted), mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(sys, 'stdout', io.StringIO()):
            REPORTER.main()
        self.assertEqual({path.name for path in output.iterdir()}, {'report.json', 'report.md'})
        self.assertLessEqual((output / 'report.json').stat().st_size, 1024**2)
        self.assertLessEqual((output / 'report.md').stat().st_size, 65536)
        value = json.loads((output / 'report.json').read_bytes())
        self.assertIs(value['verification_complete'], True)
        for field in ('performance_qualified', 'engine_adoption_qualified', 'full_retention_qualified'):
            self.assertIs(value[field], False)
        self.assertIs(value['root_provenance']['independently_verified'], False)
        self.assertEqual(set(value['paired']), {'primary_1', 'primary_2', 'matrix_1', 'matrix_2'})
        primary = value['paired']['primary_1']['primary_384_128']['complete_request_seconds']
        self.assertEqual(primary['per_tile']['n'], 10)
        self.assertEqual(primary['per_tile']['raw'], [40.01 + i / 10 for i in range(10)])
        self.assertAlmostEqual(primary['paired_median_reduction_pct'], 10.0)
        self.assertIs(value['paired']['primary_1']['primary_384_128']
                      ['callback_interval_tokens_per_second']['available'], False)
        teacher = value['paired']['matrix_1']['teacher_128_128']
        self.assertIs(teacher['first_callback_seconds']['available'], False)
        boundary = value['paired']['matrix_1']['boundary_prefill_512_0']
        self.assertIs(boundary['first_callback_seconds']['available'], False)
        sustained = value['sustained']['sustained_per_tile']['complete_request_seconds']
        self.assertEqual(sustained['raw'], [301.0, 302.0])
        self.assertNotIn('paired_median_bootstrap_95_pct', sustained)

    def test_zero_teacher_intervals_are_unavailable_rates_not_division_errors(self):
        spec, trusted, captures = self.sealed_fixture()
        rows, manifest = captures['matrix_1']
        for row in rows:
            if row.get('case') != 'teacher_128_128':
                continue
            for item in row['prefill']:
                item['seconds'] = 0.0
            row['completed_numerical_seconds'] = sum(item['seconds'] for item in row['decode'])
        self.rewrite_bundle(spec, 'matrix_1', rows, manifest)
        value = self.joined_report(spec, trusted)
        metrics = value['paired']['matrix_1']['teacher_128_128']
        self.assertIs(metrics['prefill_rows_per_second']['available'], False)
        self.assertIs(metrics['decode_evaluations_per_second']['available'], True)
        self.assertIs(metrics['completed_numerical_seconds']['available'], True)

    def test_synthetic_bridge_fails_with_its_real_trusted_hash_unpatched(self):
        spec, trusted, _ = self.sealed_fixture()
        trusted.pop('PRIMARY_INPUT_SHA')
        with self.assertRaisesRegex(ValueError, 'fixed input authentication'):
            self.joined_report(spec, trusted)

    def test_joined_report_rejects_changed_pins_guard_policy_and_bridge(self):
        spec, trusted, _ = self.sealed_fixture()
        # Reuse immutable fixture files, changing one independently repinned
        # document at a time; validation must fail for its contents, not its SHA.
        original = copy.deepcopy(spec)
        mutations = ('guard_pass', 'cleanup', 'exit_bool', 'charge', 'thermal_bool', 'battery',
                     'disk_zero', 'disk_weakened', 'log_bound', 'argv', 'stdout_join', 'pin', 'bridge')
        for label in mutations:
            with self.subTest(label=label):
                spec = copy.deepcopy(original)
                guard = spec['bundles']['primary_1']['guard']
                result_path = Path(guard['result']['path'])
                plan_path = Path(guard['plan']['path'])
                saved_result, saved_plan = result_path.read_bytes(), plan_path.read_bytes()
                bridge = self.directory / 'primary_1/input_ids.u32le'
                saved_bridge = bridge.read_bytes()
                try:
                    if label in ('guard_pass', 'cleanup', 'exit_bool', 'charge', 'thermal_bool', 'battery'):
                        def change(value):
                            if label == 'guard_pass':
                                value['passed'] = False
                            elif label == 'cleanup':
                                value['owned_live_after_cleanup'] = [123]
                            elif label == 'exit_bool':
                                value['exit_code'] = False
                            elif label == 'charge':
                                value['maximum_observed_charge_with_reserve'] = LIMIT + 1
                            elif label == 'thermal_bool':
                                value['last_runtime_gate']['thermalState'] = False
                            else:
                                value['last_runtime_gate']['power'] = "Now drawing from 'Battery Power'"
                        guard['result'] = self.rewrite_json_pin(guard['result'], change)
                    elif label in ('disk_zero', 'disk_weakened', 'log_bound', 'argv'):
                        def change(value):
                            if label == 'disk_zero':
                                value['disk_floor_bytes'] = 0
                            elif label == 'disk_weakened':
                                value['disk_floor_bytes'] = 2 * 1024**3 - 1
                            elif label == 'log_bound':
                                value['log_limit_bytes'] = 16 * 1024**2 - 1
                            else:
                                value['command'][-1] = '2'
                        guard['plan'] = self.rewrite_json_pin(guard['plan'], change)
                    elif label == 'stdout_join':
                        guard['stdout'] = self.write('primary_1-guard/stdout-other.log', b'changed')
                    elif label == 'pin':
                        spec['root_provenance']['hardware']['sha256'] = '0' * 64
                    else:
                        bridge.write_bytes(id_bytes(PRIMARY_INPUT[:-1] + [999]))
                    with self.assertRaises((ValueError, TypeError, OSError)):
                        self.joined_report(spec, trusted)
                finally:
                    result_path.write_bytes(saved_result)
                    plan_path.write_bytes(saved_plan)
                    bridge.write_bytes(saved_bridge)

    def test_confirmation_cannot_reuse_a_normalized_collection_with_new_metadata(self):
        for family in ('primary', 'matrix'):
            with self.subTest(family=family):
                # A separate directory for each subtest keeps every raw Pin and
                # capture path distinct; only the work/clocks are reused.
                if family == 'matrix':
                    self.directory = self.directory / 'second-case'
                    self.directory.mkdir()
                spec, trusted, captures = self.sealed_fixture()
                first = captures[family + '_1'][0]
                second, manifest = captures[family + '_2']
                for row in second:
                    if row.get('event') != 'request':
                        continue
                    if family == 'primary':
                        match = next(item for item in first if item.get('event') == 'request'
                                     and (item['arm'], item['warmup'], item['pair_index'])
                                     == (row['arm'], row['warmup'], row['pair_index']))
                        for key in ('complete_request_seconds', 'first_callback_seconds', 'last_callback_seconds',
                                    'token_event_seconds', 'report'):
                            row[key] = copy.deepcopy(match[key])
                    else:
                        match = next(item for item in first if item.get('event') == 'request'
                                     and (item['case'], item['arm'], item['warmup'], item['trial'])
                                     == (row['case'], row['arm'], row['warmup'], row['trial']))
                        for key in ('complete_request_seconds', 'first_callback_seconds', 'last_callback_seconds',
                                    'token_event_seconds', 'report', 'completed_numerical_seconds',
                                    'wall_with_posttimer_hash_seconds', 'prefill', 'decode'):
                            if key in match:
                                row[key] = copy.deepcopy(match[key])
                if family == 'primary':
                    manifest['measured_complete_request_seconds'] = sum(
                        row['complete_request_seconds'] for row in second[1:] if not row['warmup'])
                self.rewrite_bundle(spec, family + '_2', second, manifest)
                with self.assertRaisesRegex(ValueError, 'cannot confirm itself'):
                    self.joined_report(spec, trusted)

    def test_cli_malformed_input_publishes_only_bounded_failure(self):
        pin = self.write('malformed.json', b'{}')
        output = self.directory / 'failure-output'
        argv = ['gemma_submission_report.py', '--inputs', pin['path'], '--inputs-sha256', pin['sha256'],
                '--output', str(output)]
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(sys, 'stdout', io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                REPORTER.main()
        self.assertEqual(error.exception.code, 2)
        self.assertEqual({path.name for path in output.iterdir()}, {'failure.json'})
        self.assertLessEqual((output / 'failure.json').stat().st_size, 4096)
        self.assertIs(json.loads((output / 'failure.json').read_bytes())['verification_complete'], False)
        before = (output / 'failure.json').read_bytes()
        with mock.patch.object(sys, 'argv', argv):
            with self.assertRaises(SystemExit) as error:
                REPORTER.main()
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((output / 'failure.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
