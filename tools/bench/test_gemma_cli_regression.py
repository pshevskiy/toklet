"""Independent synthetic raw receipts and fixed arithmetic; no engine execution."""
import copy
import difflib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import unittest
from unittest import mock

import gemma_cli_regression as consumer

FIXTURES = Path(__file__).resolve().parents[2] / 'tests/fixtures/gemma-cli-regression-v1'
WORK = json.loads((FIXTURES / 'workload.json').read_bytes())
EXPECTED = json.loads((FIXTURES / 'expected.json').read_bytes())
ZERO = dict(history_tokens=0, committed_tokens=0, pending_tokens=0)
FINAL = dict(history_tokens=512, committed_tokens=511, pending_tokens=1)
COMMIT = 'a' * 40
COMPILER = 'synthetic rustc1.90.0; not an executed compiler'
COLLECTOR_SCOPE = ('Logical Collector value plus retained ring capacity after construction, fixed until drop. '
                   'Excludes allocator overhead, temporary stack/copy/export storage and native/OS memory; not RSS or process peak.')
CANCEL_REASON = 'signal_timestamp_not_observed; cancellation_outcome_and_state_do_not_measure_signal_to_ack_latency'
LOGICAL = dict(weights_bytes=4560100024, kv_bytes=20054016, scratch_bytes=17514584,
               history_bytes=4096, logits_bytes=67108864, metadata_bytes=8441640,
               metadata_allowance_bytes=67108864, admitted_bytes=4740332088)


def encode(value):
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class Receipt:
    """Finite native-format fixture author, independent of the module under test."""
    def __init__(self, root, name, variant=1, epoch=1000000000000,
                 durations=None, snapshot=False, base=None, add_file=False):
        self.root = root / name
        self.root.mkdir()
        self.variant = variant
        self.epoch = epoch
        self.durations = durations or (lambda cohort, owner, sequence: (20, 100))
        self.source_files = {'Cargo.toml': digest(b'[package]\nname="synthetic"\n'),
                             'src/lib.rs': digest(f'pub const VERSION: u8 = {variant};\n'.encode())}
        self.add_file = add_file
        if add_file:
            self.source_files['src/added.rs'] = digest(b'pub const ADDED: bool = true;\n')
        self.freeze = self.write('freeze.json', dict(format='toklet-gemma-cli-source-freeze-v1', files=self.source_files))
        self.binary = self.bytes('binary-not-executable', f'synthetic measured binary{variant}\n'.encode())
        self.clean = dict(passed=True, failure=None, exit_code=0, elapsed_seconds=200,
                          maximum_observed_charge_with_reserve=5000000000,
                          terminal_direct_pair_rss_charge_with_reserve=4900000000,
                          minimum_free_bytes=3000000000, sample_count=2,
                          owned_live_after_cleanup=[], cleanup_errors=[],
                          helper_reserve_bytes=33554432, configured_guest_reserve_bytes=0,
                          last_runtime_gate=self.condition(epoch))
        self.checks_doc = dict(format='toklet-gemma-cli-required-checks-v1', completed=True,
                               passed=True, source_unchanged=True, source_freeze_sha256=self.freeze['sha256'],
                               source_files=len(self.source_files), records=[dict(name='synthetic',
                               source_unchanged=True, result=copy.deepcopy(self.clean))])
        self.checks = self.write('checks.json', self.checks_doc)
        self.compile_guard = self.write('compile-guard.json', self.clean)
        self.compile_plan = self.write('compile-plan.json', dict(command=['cargo', 'build', '--release',
                                      '--locked', '--offline', '--bin', 'toklet', '--features', 'metal']))
        self.empty = self.bytes('empty.log', b'')
        self.build_doc = dict(format='toklet-gemma-cli-release-seal-v1', **self.binary,
                              source_freeze=self.freeze, source_freeze_sha256=self.freeze['sha256'],
                              compiler=COMPILER, required_checks=self.checks, compile_guard=self.compile_guard,
                              compile_plan=self.compile_plan, compile_stdout=self.empty, compile_stderr=self.empty,
                              cargo_artifact=dict(target=dict(kind=['bin'], name='toklet'),
                              profile=dict(opt_level='3', debug_assertions=False, overflow_checks=False, test=False),
                              features=['default', 'metal']))
        self.build = self.write('build.json', self.build_doc)
        names = ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'generation_config.json', 'chat_template.jinja']
        self.endpoints = {name: [1, n + 1, 33188, 1, 1, 1] for n, name in enumerate(names)}
        self.endpoints['.'] = [1, 100, 16832, 0, 1, 1]
        self.model = str(root / 'model-is-never-opened')
        self.auth_doc = dict(format='toklet-gemma-cli-authenticated-inputs-v1', model_directory=self.model,
                             model_endpoints=self.endpoints, rustc=COMPILER, cargo='synthetic cargo1.90',
                             hardware_os={k: 'SYNTHETIC' for k in ('machdep.cpu.brand_string', 'hw.model',
                                 'kern.osrelease', 'kern.osversion', 'kern.osproductversion')},
                             files={name: dict(path=self.model + '/' + name, bytes=1,
                                    sha256=digest(name.encode())) for name in names})
        self.auth = self.write('auth.json', self.auth_doc)
        self.oracle = self.write('oracle.json', dict(format='toklet-gemma-cli-codec-oracles-v1',
            completed=True, available=True, provenance_unchanged=True,
            primary128=dict(input_ids=WORK['input_ids'], output_ids=WORK['output_ids'],
                            input_ids_u32le_sha256=WORK['input_sha256'],
                            output_ids_u32le_sha256=WORK['output_sha256'],
                            canonical_decode_skip_special=WORK['canonical_text'],
                            canonical_decode_skip_special_utf8_sha256=WORK['text_sha256'],
                            canonical_decode_keep_special=WORK['canonical_keep_special'],
                            canonical_decode_keep_special_utf8_sha256=WORK['keep_special_sha256'],
                            source=dict(requests_identical=26))))
        self.common = self.bytes('common.py', b'TIMESTAMP_SCOPE = "parent monotonic receive before JSON"\n')
        self.controller = self.bytes('controller.py', b'SOURCE_SHA = "audited"\ndef checkpoint():\n    return 1\ndef run():\n    return "timed"\n')
        self.reference = self.controller
        self.delta_review = None
        self.protocol_doc = dict(format='toklet-gemma-cli-overhead-protocol-v1',
            policies=['per-tile', 'batched-single-row'], modes={'A': 'disabled', 'B': 'summary'},
            cohorts=[dict(policy=p, cohort=c, order=list('ABBA' if c == 1 else 'BAAB'))
                     for p in ['per-tile', 'batched-single-row'] for c in [1, 2]],
            process_blocks=dict(fresh_processes_per_cohort=4, warmup_generations_per_process=3,
                measured_generations_per_process=5, measured_generations_per_mode_per_cohort=10,
                warmup_generations_per_mode_per_cohort=6, reset_before_each_generation=True),
            totals=dict(generation_requests=128, measured_generation_requests=80,
                        warmup_generation_requests=48, reset_controls=128, processes=16),
            workload=dict(capacity=512, chunk=32, max_new_tokens=128,
                cache='reset_before_each_generation', stop_policy='--no-eos',
                input_ids=self.bytes('input.u32le', b''.join(x.to_bytes(4, 'little') for x in WORK['input_ids'])),
                primary_workload=self.write('workload.json', dict(synthetic=True))),
            clock=dict(instrument='Parent monotonic perf_counter_ns at bounded stdout framing before JSON interpretation',
                ttft='stdin submission/flush start to first received token event',
                complete_request='same start to received terminal done event, including collector and terminal publication'))
        self.protocol = self.write('protocol.json', self.protocol_doc)
        self.guard_doc = copy.deepcopy(self.clean)
        self.guard = self.write('guard.json', self.guard_doc)
        self.guard_plan_doc = dict(command=['python', str(self.root / 'controller.py')],
            limit_bytes=6442450944, helper_reserve_bytes=33554432, configured_guest_reserve_bytes=0,
            total_reserved_bytes=33554432, disk_floor_bytes=2147483648, timeout_seconds=10800)
        self.guard_plan = self.write('guard-plan.json', self.guard_plan_doc)
        chain = hashlib.sha256()
        for relative, value in sorted(self.source_files.items()):
            chain.update(relative.encode() + b'\0' + bytes.fromhex(value))
        self.checkpoint = dict(source_files=len(self.source_files), source_chain_sha256=chain.hexdigest(),
                               model_endpoints=self.endpoints, checked_ns=epoch - 20000000)
        self.blocks = []
        cohorts = []
        for policy in ['per-tile', 'batched-single-row']:
            for cohort in [1, 2]:
                order = list('ABBA' if cohort == 1 else 'BAAB')
                blocks = [self.owner(policy, cohort, number, mode, len(self.blocks))
                          for number, mode in enumerate(order)]
                cohorts.append(dict(policy=policy, cohort=cohort, order=order, blocks=blocks))
        self.guard_doc['last_runtime_gate'] = self.condition(cohorts[-1]['blocks'][-1]['after']['checked_ns'] + 1)
        self.guard = self.write('guard.json', self.guard_doc)
        self.capture_doc = dict(format='toklet-gemma-cli-overhead-capture-v1', completed=True, failure=None,
            protocol_pin=self.protocol, source_freeze_pin=self.freeze, binary_pin=self.binary,
            outer_guard_plan=self.guard_plan, initial_checkpoint=self.checkpoint, cohorts=cohorts,
            inventory=dict(owners=16, generation_requests=128, measured=80, warmup=48, reset_controls=128,
                           summary_exports=8, records_per_summary_export=16),
            observer_overhead_qualified=False, performance_qualified=False, adoption_qualified=False)
        self.binding_doc = dict(format='toklet-gemma-cli-overhead-bindings-v1', source_root=str(self.root),
            output_root=str(self.root), source_freeze=self.freeze, binary=self.binary, build_receipt=self.build,
            required_checks=self.checks, authenticated_inputs=self.auth, codec_oracles=self.oracle,
            protocol=self.protocol, controller=self.controller, common=self.common)
        self.binding = self.write('bindings.json', self.binding_doc)
        self.capture_doc['bindings_pin'] = self.binding
        self.guard_plan_doc.update(cwd=str(self.root), command=[sys.executable, self.controller['path'],
            '--bindings', self.binding['path'], '--bindings-sha256', self.binding['sha256'],
            '--outer-guard-plan', self.guard_plan['path']])
        self.guard_plan = self.write('guard-plan.json', self.guard_plan_doc)
        self.capture_doc['outer_guard_plan'] = self.guard_plan
        self.enforcement_doc = dict(format='toklet-gemma-cli-overhead-enforcement-v1', passed=True,
            source_and_model_endpoints_unchanged=True, source_freeze=self.freeze,
            bindings_pin=self.binding, result=self.guard_doc)
        self.enforcement = self.write('enforcement.json', self.enforcement_doc)
        self.revision = dict(format='toklet-gemma-cli-revision-proof-v1', kind='published', commit=COMMIT,
            source_freeze=self.freeze, build=self.build, binary=self.binary, checks=self.checks)
        if snapshot:
            self.snapshot(base)
        self.reseal()

    @staticmethod
    def condition(ns):
        return dict(passed=True, thermalState=0, lowPowerMode=False,
                    power="Now drawing from 'AC Power'", checked_uptime_ns=ns)

    def bytes(self, relative, raw):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return dict(path=str(path), bytes=len(raw), sha256=digest(raw))

    def write(self, relative, value):
        return self.bytes(relative, encode(value))

    def memory(self, enabled, valid):
        return dict(weights_owned_bytes=LOGICAL['weights_bytes'], kv_allocated_bytes=LOGICAL['kv_bytes'],
            scratch_bytes=LOGICAL['scratch_bytes'], kv_valid_positions=valid, owner_reserved_bytes=1073741888,
            load_admitted_bytes=5890864866 if enabled else 5889799906,
            session_admitted_bytes=5987105400 if enabled else 5986040440,
            app_allowance_bytes=134217728, stream_allowance_bytes=4194304,
            telemetry_allowance_bytes=1064960 if enabled else 0, helper_reserve_bytes=33554432)

    def metric(self, sid, rid, reset, sequence, ttft, total, generation):
        first = None if reset else ttft - .25
        last = None if reset else total - .251
        span = 0 if reset else last - first
        timing = dict(request_total_ms=.1 if reset else total - .1,
            native_total_ms=None if reset else generation['total_seconds'] * 1000,
            tokenization_ms=None, prefill_ms=None if reset else generation['prefill_seconds'] * 1000,
            decode_ms=None if reset else generation['decode_seconds'] * 1000,
            sampling_ms=None if reset else generation['sampling_seconds'] * 1000,
            ttft_event_ms=first, first_text_publication_ms=first, last_token_event_ms=last,
            token_interval_count=0 if reset else 127, token_interval_min_ms=None if reset else span / 127,
            token_interval_max_ms=None if reset else span / 127, token_interval_sum_ms=span,
            output_decode_ms=0, output_publication_ms=0, cancel_to_terminal_ms=None)
        work = dict(prompt_tokens=None if reset else 384, reused_prefix_tokens=None if reset else 0,
            prefill_rows=None if reset else 384, decode_evaluations=None if reset else 127,
            sampled_tokens=None if reset else 128, emitted_tokens=None if reset else 128,
            callback_tokens=0 if reset else 128, replayed_rows=None if reset else 0)
        return dict(event='request_summary', schema_version=1, session_id=sid, request_id=rid,
            attempt_id=rid, owner_generation=1, request_kind='reset' if reset else 'generate',
            terminal_status='success', failed_phase=None, error_category=None,
            cache_action='reset' if reset else 'fresh', replay_required=False,
            before=copy.deepcopy(ZERO if not reset or sequence == 0 else FINAL),
            after=copy.deepcopy(ZERO if reset else FINAL), work=work, timing=timing,
            memory=self.memory(True, 0 if reset else 511),
            prefill_tokens_per_s=None if reset else 384000 / timing['prefill_ms'],
            decode_tokens_per_s=None if reset else 127000 / span,
            decode_evaluations_per_s=None if reset else 127000 / timing['decode_ms'],
            token_interval_mean_ms=None if reset else span / 127,
            ordinary_observations={k: None for k in ('gpu_ms host_submit_ms host_wait_ms command_count '
                'dispatch_count wait_count readback_bytes buffer_growth_count pipeline_compile_count '
                'allocation_count process_footprint_bytes ui_first_visible_ms energy_joules').split()},
            availability=dict(ordinary_observations='not_observed_in_ordinary_request',
                process_footprint_bytes='admission_is_not_sampled_process_footprint',
                ui_first_visible_ms='stdout_publication_is_not_ui_visibility', energy_joules='not_measured'))

    def header(self, sid, policy):
        return dict(event='request_metrics_manifest', schema_version=1, session_id=sid,
            capacity=512, prefill_chunk_size=32, submission=policy, load_ms=10., route_start_to_ready_ms=11.,
            load_admitted_bytes=5890864866, session_admitted_bytes=5987105400, logical_memory=LOGICAL,
            numerical_session_admitted_bytes=4740332088, additional_owner_and_external_admitted_bytes=1246773312,
            logical_memory_scope='Numerical component sum equals logical_memory.admitted_bytes. Whole session admission is numerical plus additional owner/external admission; do not add logical_memory twice. Admission is not measured process footprint.',
            collector_fixed_bytes=8192, retained_ring_capacity_bytes=1032192, collector_peak_bytes=1040384,
            telemetry_admitted_bytes=1064960, collector_memory_scope=COLLECTOR_SCOPE,
            model_profile='gemma4-bf16', backend='metal', model_repository='google/gemma-4-E2B-it',
            model_revision='70af34e20bd4b7a91f0de6b22675850c43922a03', weight_dtype='bf16',
            weight_dtype_scope='Original BF16 matrix coefficients and file-backed PLE; resident learned vectors are FP32.',
            kv_dtype='fp32', sampling='greedy', thinking_enabled=False, instrumentation='summary',
            engine_crate_version='0.1.0', engine_revision=None, build_flags=None, device_identity=None,
            device_power=None, model_content_sha256=None, tokenizer_content_sha256=None,
            external_provenance_reason='external_capture_manifest_required', conversion_ms=None, compile_ms=None,
            process_start_to_ready_ms=None, conversion_availability_reason='not_separately_timed_within_owner_load',
            compile_availability_reason='context_compilation_overlaps_owner_load; phase_not_separately_timed',
            process_start_to_ready_availability_reason='process_start_not_observed; route_start_to_ready_ms_is_handled_route_clock',
            cancel_to_terminal_availability_reason=CANCEL_REASON,
            route_start_to_ready_scope='Handled Gemma route entry to ready; process startup and stdout/UI receipt excluded.',
            load_timing_scope='Completed OfficialBf16::load owner construction; context/pipeline and authenticated asset preparation overlap; conversion/compile components unavailable.')

    def owner(self, policy, cohort, number, mode, ordinal):
        prefix = f'{policy}-c{cohort}-b{number}-{mode}'
        enabled = mode == 'B'
        started = self.epoch + ordinal * 10000000000
        ready_ns = started + 20000000
        command = [self.binary['path'], '--model', self.model, '--model-profile', 'gemma4-bf16',
            '--backend', 'metal', '--context-capacity', '512', '--prefill-chunk', '32',
            '--submission', policy, '--interactive', '--json']
        if enabled:
            command += ['--request-metrics', 'summary', '--metrics-output', str(self.root / prefix / 'metrics.jsonl')]
        command += ['--no-eos', '--max-new-tokens', '128']
        identity = dict(pid=1000 + ordinal, spawn_monotonic_ns=started, command_sha256=digest(encode(command)))
        ready = dict(event='ready', model_profile='gemma4-bf16', backend='metal', context_capacity=512,
                     submission=policy, memory_admitted_bytes=self.memory(enabled, 0)['session_admitted_bytes'], logical_memory=LOGICAL)
        stdout = bytearray()
        frames = []
        def frame(event, ns):
            raw = encode(event)
            frames.append(dict(received_ns=ns, offset=len(stdout), length=len(raw), sha256=digest(raw), event=event))
            stdout.extend(raw)
        frame(ready, ready_ns)
        requests, samples, metric_rows = [], [], []
        sid = f'gemma-{identity["pid"]}-synthetic'
        clock = ready_ns
        mode_owner = sum(x == mode for x in ('ABBA' if cohort == 1 else 'BAAB')[:number])
        for sequence in range(8):
            sent = clock + 1000000
            requests.append(dict(sent_ns=sent, raw_hex=encode(dict(action='reset')).hex()))
            frame(dict(event='reset', request_id=sequence * 2 + 1, state=ZERO), sent + 1000000)
            if enabled:
                metric_rows.append(self.metric(sid, sequence * 2 + 1, True, sequence, 0, 0, None))
            sent += 2000000
            ttft, total = self.durations(cohort, mode_owner, sequence)
            first = sent + round(ttft * 1000000)
            terminal_ns = sent + round(total * 1000000)
            last = terminal_ns - 1000
            requests.append(dict(sent_ns=sent, raw_hex=encode(dict(action='generate', tokens=WORK['input_ids'], max_new_tokens=128)).hex()))
            for index, token in enumerate(WORK['output_ids']):
                ns = first + (last - first) * index // 127
                lo = len(WORK['canonical_text']) * index // 128
                hi = len(WORK['canonical_text']) * (index + 1) // 128
                frame(dict(event='token', token_id=token, index=index,
                      elapsed_seconds=(ns - sent - 1000000) / 1e9,
                      text=WORK['canonical_text'][lo:hi]), ns)
            generation = dict(reason=dict(kind='output_limit'), reused_tokens=0, prefill_tokens=384,
                sampled_tokens=128, emitted_tokens=128, history_tokens=512, evaluated_tokens=511, pending_tokens=1,
                first_token_seconds=(ttft - 1) / 1000, prefill_seconds=(ttft - 2) / 1000,
                decode_seconds=(total - ttft - 1) / 1000, sampling_seconds=.001, total_seconds=(total - 1) / 1000)
            terminal = dict(event='done', request_id=sequence * 2 + 2,
                statistics=dict(model_profile='gemma4-bf16', backend='metal', submission=policy,
                  generation=generation, decode_evaluations=127, state=FINAL,
                  memory_admitted_bytes=self.memory(enabled, 511)['session_admitted_bytes'], logical_memory=LOGICAL))
            frame(terminal, terminal_ns)
            samples.append(dict(sent_ns=sent, first_token_ns=first, last_token_ns=last, terminal_ns=terminal_ns,
                external_ttft_seconds=ttft / 1000, external_complete_request_seconds=total / 1000,
                terminal=terminal, sequence=sequence, warmup=sequence < 3,
                measured_index=None if sequence < 3 else sequence - 3, output_ids_sha256=WORK['output_sha256'],
                text_before_finish_sha256=WORK['text_sha256'], finalized_text_sha256=WORK['text_sha256'],
                finish_scope='following_reset' if sequence < 7 else 'process_eof'))
            if enabled:
                metric_rows.append(self.metric(sid, sequence * 2 + 2, False, sequence, ttft, total, generation))
            clock = terminal_ns
        metrics = None
        if enabled:
            footer = dict(event='request_metrics_export', schema_version=1, complete=True, retained=16,
                telemetry_dropped=0, export_errors=0, invalid_records=0, counter_overflow=False)
            metrics = self.bytes(prefix + '/metrics.jsonl', b''.join(encode(r) for r in [self.header(sid, policy)] + metric_rows + [footer]))
        close = dict(exitcode=0, closed=True, aborted=False, failure=None, child_identity=identity,
            extra_events=[], tail_text='', stdout_bytes=len(stdout), stderr_bytes=0, metrics_pin=metrics,
            finished_ns=clock + 2000000, cleanup=dict(child_created=True, direct_child_reaped=True, unrelated_process_signals=False))
        raw_pins = {'command.json': self.write(prefix + '/command.json', dict(command=command, started_ns=started,
            timestamp_scope='parent monotonic receive; completed newline before JSON interpretation')),
            'stdout.bin': self.bytes(prefix + '/stdout.bin', bytes(stdout)),
            'stderr.bin': self.bytes(prefix + '/stderr.bin', b''),
            'frames.jsonl': self.bytes(prefix + '/frames.jsonl', b''.join(encode(x) for x in frames)),
            'requests.jsonl': self.bytes(prefix + '/requests.jsonl', b''.join(encode(x) for x in requests)),
            'session.json': self.write(prefix + '/session.json', close)}
        before = copy.deepcopy(self.checkpoint)
        before['checked_ns'] = started - 20000000
        after = copy.deepcopy(self.checkpoint)
        after['checked_ns'] = close['finished_ns'] + 1000000
        block = dict(block=number, mode=mode, metrics=enabled, warmups=3, measured=5, samples=samples,
            stable_conditions=[self.condition(started - 10000000 + i * 100000) for i in range(10)],
            before=before, after=after, launch_call_ns=started - 1, directory=str(self.root / prefix),
            ready=ready, started_ns=started, ready_ns=ready_ns, cold_start_to_ready_seconds=.02,
            child_identity=identity, command=command, close=close,
            metrics_export=None if not enabled else dict(pin=metrics, binary_pin=self.binary,
                facts=dict(manifest=self.header(sid, policy), export_status=footer,
                    validated_complete_export=True, telemetry_complete=True, retained_records=16, attempted_requests=16)),
            raw_pins=raw_pins)
        self.blocks.append(block)
        return block

    def published(self, commit):
        ci_doc = dict(format='toklet-gemma-cli-exact-head-ci-v1', commit=commit,
            remote_ref=dict(object=dict(sha=commit)), all_complete_success=True, workflow_count=1, job_count=1,
            runs=[dict(id=1, head_sha=commit, status='completed', conclusion='success')],
            jobs=[dict(run_id=1, jobs=[dict(id=2, run_id=1, status='completed', conclusion='success')])])
        ci = self.write('ci-' + commit[0] + '.json', ci_doc)
        content = self.write('source-content-' + commit[0] + '.json', dict(
            format='toklet-gemma-cli-commit-source-map-v1', commit=commit, source_freeze=self.freeze,
            all_source_blobs_match=True, files=len(self.source_files), source_binary_receipt=self.build))
        publication = self.write('publication-' + commit[0] + '.json', dict(
            format='toklet-gemma-cli-code-publication-v1', commit=commit, pushed=True,
            exact_head_ci_passed=True, all_source_blobs_match=True,
            evidence=dict(source_freeze=self.freeze, release=self.build, required_checks=self.checks,
                commit_source_map=content, capture=self.capture, guard=self.guard,
                enforcement=self.enforcement, remote_ci=ci)))
        return dict(format='toklet-gemma-cli-revision-proof-v1', kind='published', commit=commit,
            source_freeze=self.freeze, build=self.build, binary=self.binary, checks=self.checks,
            source_content=content, publication=publication, ci=ci)

    def snapshot(self, base):
        self.base = base
        old = b'pub const VERSION: u8 = 1;\n'
        new = f'pub const VERSION: u8 = {self.variant};\n'.encode()
        before = self.bytes('snapshot/before.rs', old)
        after = self.bytes('snapshot/after.rs', new)
        diff = ''.join(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True),
                                           fromfile='a/src/lib.rs', tofile='b/src/lib.rs'))
        files = [dict(relative='src/lib.rs', before=before, after=after)]
        if self.add_file:
            added = b'pub const ADDED: bool = true;\n'
            files.append(dict(relative='src/added.rs', before=None,
                              after=self.bytes('snapshot/added.rs', added)))
            diff += ''.join(difflib.unified_diff([], added.decode().splitlines(True),
                           fromfile='/dev/null', tofile='b/src/added.rs'))
        self.delta = self.bytes('snapshot/change.patch', diff.encode())
        self.archive = self.write('snapshot/archive.json', dict(format='toklet-gemma-cli-snapshot-content-v1',
            base_commit=COMMIT, base_freeze=base.freeze, files=files))
        snapshot_id = digest(canonical(dict(base_commit=COMMIT, source_files=self.source_files,
            diff_sha256=self.delta['sha256'], archive_sha256=self.archive['sha256'])))
        self.revision = dict(format='toklet-gemma-cli-revision-proof-v1', kind='snapshot', candidate_commit=None,
            base_commit=COMMIT, snapshot_id=snapshot_id, source_freeze=self.freeze,
            build=self.build, binary=self.binary, checks=self.checks, diff=self.delta, archive=self.archive)

    def reseal(self):
        self.capture = self.write('capture.json', self.capture_doc)
        if self.revision['kind'] == 'published':
            self.revision = self.published(self.revision.get('commit', COMMIT))
        self.observer = dict(format='toklet-gemma-cli-observer-proof-v1', common=self.common,
            controller=self.controller, reference_controller=self.reference, delta_review=self.delta_review,
            guard_plan=self.guard_plan, guard_result=self.guard, enforcement=self.enforcement)

    def admit(self, policy='per-tile'):
        return consumer.admit_cli_bundle(self.capture, self.revision, dict(policy=policy, mode='disabled'), self.observer)

    def edit_raw(self, role, change, block=0):
        owner = self.blocks[block]
        pin = owner['raw_pins'][role]
        path = Path(pin['path'])
        raw = change(path.read_bytes())
        owner['raw_pins'][role] = self.bytes(str(path.relative_to(self.root)), raw)
        self.reseal()


    def rewrite_events(self, transform, block=0):
        owner = self.blocks[block]
        frames = [json.loads(line) for line in Path(owner['raw_pins']['frames.jsonl']['path']).read_bytes().splitlines()]
        transform(frames)
        stdout = bytearray()
        for frame in frames:
            raw = encode(frame['event'])
            frame.update(offset=len(stdout), length=len(raw), sha256=digest(raw))
            stdout.extend(raw)
        prefix = str(Path(owner['directory']).relative_to(self.root))
        owner['raw_pins']['stdout.bin'] = self.bytes(prefix + '/stdout.bin', bytes(stdout))
        owner['raw_pins']['frames.jsonl'] = self.bytes(prefix + '/frames.jsonl', b''.join(encode(x) for x in frames))
        owner['close']['stdout_bytes'] = len(stdout)
        owner['raw_pins']['session.json'] = self.write(prefix + '/session.json', owner['close'])
        for row in frames:
            event = row['event']
            if event['event'] == 'done':
                owner['samples'][event['request_id'] // 2 - 1]['terminal'] = event
        self.reseal()

    def observer_delta(self, timed=False):
        before = Path(self.controller['path']).read_bytes()
        after = before.replace(b'return "timed"', b'return "changed"') if timed else before.replace(b'return 1', b'return 2')
        self.reference = self.controller
        self.controller = self.bytes('candidate-controller.py', after)
        delta = ''.join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                           fromfile='a/controller.py', tofile='b/controller.py'))
        diff_pin = self.bytes('observer-change.patch', delta.encode())
        self.delta_review = self.write('observer-delta.json', dict(format='toklet-gemma-cli-observer-delta-v1',
            before=self.reference, after=self.controller, diff=diff_pin, reviewed=True,
            cold_admission_only=True, timed_semantics_unchanged=True))
        self.binding_doc['controller'] = self.controller
        self.binding = self.write('bindings.json', self.binding_doc)
        self.capture_doc['bindings_pin'] = self.binding
        self.guard_plan_doc['command'] = [sys.executable, self.controller['path'], '--bindings', self.binding['path'],
            '--bindings-sha256', self.binding['sha256'], '--outer-guard-plan', self.guard_plan['path']]
        self.guard_plan = self.write('guard-plan.json', self.guard_plan_doc)
        self.capture_doc['outer_guard_plan'] = self.guard_plan
        self.enforcement_doc['bindings_pin'] = self.binding
        self.enforcement = self.write('enforcement.json', self.enforcement_doc)
        self.reseal()


    def rebind(self):
        self.binding = self.write('bindings.json', self.binding_doc)
        self.capture_doc['bindings_pin'] = self.binding
        self.guard_plan_doc['command'] = [sys.executable, self.controller['path'], '--bindings', self.binding['path'],
            '--bindings-sha256', self.binding['sha256'], '--outer-guard-plan', self.guard_plan['path']]
        self.guard_plan = self.write('guard-plan.json', self.guard_plan_doc)
        self.capture_doc['outer_guard_plan'] = self.guard_plan
        self.enforcement_doc['bindings_pin'] = self.binding
        self.enforcement = self.write('enforcement.json', self.enforcement_doc)
        self.reseal()


class CliRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='toklet-cli-regression-test-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def baseline(self, durations=None):
        return Receipt(self.root, 'baseline', durations=durations)

    def candidate(self, base, durations=None, snapshot=True, epoch=100000000000000):
        return Receipt(self.root, 'candidate', variant=2, epoch=epoch, durations=durations, snapshot=snapshot, base=base)

    def compare(self, base, candidate):
        return consumer.compare_cli(base.admit(), candidate.admit())

    def reject_admission(self, receipt):
        with self.assertRaises((ValueError, TypeError, KeyError, OSError, ArithmeticError)):
            receipt.admit()

    def assert_interval(self, actual, expected):
        self.assertEqual(len(actual), 2)
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=10)

    def test_baseline_admission_is_descriptive_not_a_comparison_pass(self):
        value = self.baseline().admit()
        self.assertEqual(value['format'], 'toklet-gemma-cli-admission-v1')
        self.assertNotIn('status', value)
        self.assertNotIn('repeatable_primary_speed_signal', value)
        self.assertIsNot(value.get('accepted_optimization'), True)
        self.assertEqual([c['cohort'] for c in value['cohorts']], [1, 2])
        self.assertEqual([len(c['owners']) for c in value['cohorts']], [2, 2])
        self.assertTrue(all(len(o['observations']) == 5 and o['warmups'] == 3
                            for c in value['cohorts'] for o in c['owners']))
        self.assertEqual(value['revision']['commit'], COMMIT)

    def test_uncommitted_same_base_different_patch_has_repeatable_ten_percent_signal(self):
        base = self.baseline()
        candidate = self.candidate(base, lambda c, o, s: (18, 90))
        result = self.compare(base, candidate)
        self.assertEqual(result['status'], 'pass')
        self.assertIsNone(result['candidate_commit'])
        self.assertEqual(result['candidate_base_commit'], COMMIT)
        self.assertEqual(result['candidate_snapshot_id'], candidate.revision['snapshot_id'])
        self.assertTrue(result['repeatable_primary_speed_signal'])
        self.assertFalse(result['accepted_optimization'])
        self.assertEqual(result['screening_threshold_pct'], 5)
        for batch in result['batches']:
            for metric in batch['metrics'].values():
                self.assertAlmostEqual(metric['change_pct'], -10)
                self.assert_interval(metric['independent_owner_bootstrap_95_pct'], [-10, -10])
                self.assertEqual(metric['status'], 'speed_signal')

    def test_published_candidate_and_two_percent_change_keep_legacy_five_percent_policy(self):
        base = self.baseline()
        candidate = self.candidate(base, lambda c, o, s: (20.4, 102), snapshot=False)
        candidate.revision = candidate.published('b' * 40)
        result = self.compare(base, candidate)
        self.assertEqual(result['candidate_commit'], 'b' * 40)
        self.assertEqual(result['status'], 'pass')
        self.assertFalse(result['repeatable_primary_speed_signal'])
        self.assertFalse(result['accepted_optimization'])
        self.assertEqual(result['screening_threshold_pct'], 5)

    def test_regression_and_ttft_guard_are_actionable_without_accepting_primary_gain(self):
        for index, (ttft, total) in enumerate([(22, 110), (22, 90)]):
            folder = self.root / str(index)
            folder.mkdir()
            base = Receipt(folder, 'baseline')
            candidate = Receipt(folder, 'candidate', variant=2, epoch=100000000000000,
                durations=lambda c, o, s, a=ttft, b=total: (a, b), snapshot=True, base=base)
            result = self.compare(base, candidate)
            self.assertEqual(result['status'], 'regression')
            self.assertFalse(result['accepted_optimization'])
            self.assertEqual(result['batches'][0]['metrics']['ttft_ms']['status'], 'regression')

    def test_disagreeing_confirmation_is_noisy(self):
        base = self.baseline()
        candidate = self.candidate(base, lambda c, o, s: (18, 90) if c == 1 else (20, 100))
        result = self.compare(base, candidate)
        self.assertEqual(result['status'], 'noisy')
        self.assertFalse(result['repeatable_primary_speed_signal'])
        self.assertFalse(result['accepted_optimization'])

    def test_owner_correlated_uncertainty_is_not_false_paired_or_request_iid(self):
        a = EXPECTED['owner_correlated']['baseline_owners_ms']
        b = EXPECTED['owner_correlated']['candidate_owners_ms']
        interval = consumer.bootstrap_owner_difference(a, b)
        self.assert_interval(interval, [-55, 80])
        paired = [100 * (y / x - 1) for x, y in zip(sum(a, []), sum(b, []))]
        self.assertTrue(all(math.isclose(x, -10, abs_tol=1e-12) for x in paired))
        # Deliberately wrong request-IID reference: useful only to prove the
        # production method preserves a different sampling unit.
        rng = random.Random(1)
        iid = []
        for _ in range(10000):
            after = statistics.median(rng.choices(sum(b, []), k=10))
            before = statistics.median(rng.choices(sum(a, []), k=10))
            iid.append(100 * (after / before - 1))
        iid.sort()
        self.assertNotEqual(interval, [iid[250], iid[9749]])
        base = self.baseline(lambda c, o, s: ((a[o][max(s - 3, 0)]) / 5, a[o][max(s - 3, 0)]))
        candidate = self.candidate(base, lambda c, o, s: (b[o][max(s - 3, 0)] / 5, b[o][max(s - 3, 0)]))
        result = self.compare(base, candidate)
        self.assertEqual(result['status'], 'noisy')
        self.assertFalse(result['repeatable_primary_speed_signal'])
        for batch in result['batches']:
            metric = batch['metrics']['total_ms']
            self.assertEqual(metric['baseline_median'], 150)
            self.assertEqual(metric['candidate_median'], 135)
            self.assert_interval(metric['independent_owner_bootstrap_95_pct'], [-55, 80])

    def test_self_and_relabelled_overlapping_raw_clocks_cannot_confirm_themselves(self):
        base = self.baseline()
        admitted = base.admit()
        with self.assertRaises((ValueError, TypeError)):
            consumer.compare_cli(admitted, copy.deepcopy(admitted))
        candidate = self.candidate(base, epoch=base.epoch)
        with self.assertRaises((ValueError, TypeError)):
            self.compare(base, candidate)

    def test_instrumentation_modes_and_policy_are_not_revision_roles(self):
        base = self.baseline()
        for selector in [dict(policy='per-tile', mode='summary'), dict(policy='unknown', mode='disabled')]:
            with self.assertRaises((ValueError, TypeError)):
                consumer.admit_cli_bundle(base.capture, base.revision, selector, base.observer)
        candidate = self.candidate(base)
        with self.assertRaises((ValueError, TypeError)):
            consumer.compare_cli(base.admit('per-tile'), candidate.admit('batched-single-row'))

    def test_truncated_and_failing_owners_are_incomparable(self):
        for index, change in enumerate(['incomplete', 'close', 'guard']):
            folder = self.root / str(index)
            folder.mkdir()
            item = Receipt(folder, 'baseline')
            if change == 'incomplete':
                item.capture_doc['completed'] = False
            elif change == 'close':
                item.blocks[0]['close']['cleanup']['direct_child_reaped'] = False
                item.blocks[0]['raw_pins']['session.json'] = item.write('bad-session.json', item.blocks[0]['close'])
            else:
                item.guard_doc['passed'] = False
                item.guard = item.write('guard.json', item.guard_doc)
                item.enforcement_doc['result'] = item.guard_doc
                item.enforcement = item.write('enforcement.json', item.enforcement_doc)
            item.reseal()
            self.reject_admission(item)

    def test_raw_frame_hash_offset_and_work_corruption_are_rejected_after_repinning(self):
        for index, field in enumerate(['sha256', 'offset', 'clock']):
            folder = self.root / str(index)
            folder.mkdir()
            item = Receipt(folder, 'baseline')
            def corrupt(raw, field=field):
                rows = [json.loads(x) for x in raw.splitlines()]
                if field == 'sha256': rows[2]['sha256'] = '0' * 64
                elif field == 'offset': rows[2]['offset'] += 1
                else: rows[2]['received_ns'] = 0
                return b''.join(encode(x) for x in rows)
            item.edit_raw('frames.jsonl', corrupt)
            self.reject_admission(item)

    def test_duplicate_nonfinite_metadata_and_explicit_sha_bounds_are_rejected(self):
        item = self.baseline()
        valid = encode(item.capture_doc)
        malformed = [valid.replace(b'"completed":true', b'"completed":true,"completed":true', 1),
                     valid.replace(b'"external_ttft_seconds":0.02', b'"external_ttft_seconds":NaN', 1)]
        for index, raw in enumerate(malformed):
            self.assertNotEqual(raw, valid)
            item.capture = item.bytes(f'malformed-{index}.json', raw)
            item.revision = item.published(COMMIT)
            with self.assertRaises((ValueError, TypeError, KeyError)):
                consumer.admit_cli_bundle(item.capture, item.revision, dict(policy='per-tile', mode='disabled'), item.observer)
        bad = copy.deepcopy(item.capture)
        bad['sha256'] = '0' * 64
        with self.assertRaises((ValueError, TypeError)):
            consumer.admit_cli_bundle(bad, item.revision, dict(policy='per-tile', mode='disabled'), item.observer)
        huge = item.bytes('too-large.json', b' ' * (2 * 1024**2 + 1))
        with self.assertRaises((ValueError, TypeError)):
            consumer.admit_cli_bundle(huge, item.revision, dict(policy='per-tile', mode='disabled'), item.observer)

    def test_bad_source_archive_build_check_and_ci_pins_are_not_accepted(self):
        base = self.baseline()
        candidate = self.candidate(base)
        for role in ['archive', 'diff', 'build', 'checks']:
            revision = copy.deepcopy(candidate.revision)
            revision[role]['sha256'] = '0' * 64
            with self.subTest(role=role), self.assertRaises((ValueError, TypeError)):
                consumer.admit_cli_bundle(candidate.capture, revision, dict(policy='per-tile', mode='disabled'), candidate.observer)
        published = copy.deepcopy(base.revision)
        published['ci']['sha256'] = '0' * 64
        with self.assertRaises((ValueError, TypeError)):
            consumer.admit_cli_bundle(base.capture, published, dict(policy='per-tile', mode='disabled'), base.observer)

    def test_late_publication_binds_snapshot_without_rewriting_comparison(self):
        base = self.baseline()
        candidate = self.candidate(base, lambda c, o, s: (18, 90))
        result = self.compare(base, candidate)
        comparison = candidate.write('precommit-comparison.json', result)
        before = Path(comparison['path']).read_bytes()
        publication = candidate.write('actual-proof.json', candidate.published('c' * 40))
        receipt = consumer.bind_publication(comparison, publication)
        self.assertEqual(receipt['format'], 'toklet-gemma-cli-publication-binding-v1')
        self.assertEqual(receipt['actual_commit'], 'c' * 40)
        self.assertEqual(receipt['measured_snapshot_id'], candidate.revision['snapshot_id'])
        self.assertFalse(receipt['accepted_optimization'])
        self.assertEqual(Path(comparison['path']).read_bytes(), before)
        self.assertIsNone(result['candidate_commit'])


    def test_cold_observer_delta_is_allowed_but_review_label_cannot_waive_timed_change(self):
        base = self.baseline()
        candidate = self.candidate(base)
        candidate.observer_delta()
        result = self.compare(base, candidate)
        self.assertEqual(result['status'], 'pass')
        self.assertFalse(result['accepted_optimization'])
        broken = Receipt(self.root, 'changed-observer', variant=2, epoch=200000000000000,
                         snapshot=True, base=base)
        broken.observer_delta(timed=True)
        self.reject_admission(broken)

    def test_valid_snapshot_reconstructs_new_files_and_wrong_after_bytes_are_rejected(self):
        base = self.baseline()
        candidate = Receipt(self.root, 'candidate', variant=2, epoch=100000000000000,
                            snapshot=True, base=base, add_file=True)
        admitted = candidate.admit()
        self.assertIn('src/added.rs', admitted['revision']['source_files'])
        archive = json.loads(Path(candidate.revision['archive']['path']).read_bytes())
        archive['files'][1]['after'] = candidate.bytes('snapshot/wrong-after.rs', b'pub const ADDED: bool = false;\n')
        candidate.revision['archive'] = candidate.write('snapshot/archive.json', archive)
        candidate.revision['snapshot_id'] = digest(canonical(dict(base_commit=COMMIT,
            source_files=candidate.source_files, diff_sha256=candidate.revision['diff']['sha256'],
            archive_sha256=candidate.revision['archive']['sha256'])))
        self.reject_admission(candidate)

    def test_coherent_raw_work_cache_and_stop_mismatches_are_rejected(self):
        for index, change in enumerate(['cache', 'state', 'stop']):
            folder = self.root / str(index)
            folder.mkdir()
            item = Receipt(folder, 'baseline')
            def mutate(frames, change=change):
                event = next(f['event'] for f in frames if f['event']['event'] == 'done')
                core = event['statistics']['generation']
                if change == 'cache':
                    core['reused_tokens'], core['prefill_tokens'] = 1, 383
                elif change == 'state':
                    event['statistics']['state']['committed_tokens'] -= 1
                else:
                    core['reason'] = dict(kind='stop_token')
            item.rewrite_events(mutate)
            self.reject_admission(item)

    def test_cached_float_cannot_override_raw_receive_clock_and_duplicate_tokens_fail(self):
        item = self.baseline()
        item.blocks[0]['samples'][3]['external_ttft_seconds'] = .0001
        item.reseal()
        self.reject_admission(item)
        other = Receipt(self.root, 'duplicate')
        def duplicate(frames):
            frame = copy.deepcopy(next(f for f in frames if f['event']['event'] == 'token'))
            frames.insert(3, frame)
        other.rewrite_events(duplicate)
        self.reject_admission(other)

    def test_power_and_owner_overlap_cannot_be_cured_by_relabelled_receipts(self):
        item = self.baseline()
        item.blocks[0]['stable_conditions'][4]['lowPowerMode'] = True
        item.reseal()
        self.reject_admission(item)
        other = Receipt(self.root, 'overlap')
        other.blocks[1]['started_ns'] = other.blocks[0]['started_ns']
        other.reseal()
        self.reject_admission(other)

    def test_late_publication_rejects_different_measured_content_even_with_green_ci(self):
        base = self.baseline()
        candidate = self.candidate(base, lambda c, o, s: (18, 90))
        comparison = candidate.write('comparison.json', self.compare(base, candidate))
        other = Receipt(self.root, 'different-content', variant=3, epoch=300000000000000)
        wrong = other.write('actual-proof.json', other.published('d' * 40))
        with self.assertRaises((ValueError, TypeError)):
            consumer.bind_publication(comparison, wrong)


    def test_identity_change_is_incomparable_after_valid_raw_admission(self):
        base = self.baseline()
        candidate = self.candidate(base)
        candidate.auth_doc['hardware_os']['hw.model'] = 'OTHER-SYNTHETIC-HOST'
        candidate.auth = candidate.write('auth.json', candidate.auth_doc)
        candidate.binding_doc['authenticated_inputs'] = candidate.auth
        candidate.rebind()
        admitted = candidate.admit()
        self.assertEqual(admitted['identity']['hardware_os']['hw.model'], 'OTHER-SYNTHETIC-HOST')
        with self.assertRaises((ValueError, TypeError)):
            consumer.compare_cli(base.admit(), admitted)

    def test_summary_dtype_mismatch_is_not_hidden_by_disabled_selection(self):
        item = self.baseline()
        block = item.blocks[1]
        rows = [json.loads(line) for line in Path(block['close']['metrics_pin']['path']).read_bytes().splitlines()]
        rows[0]['kv_dtype'] = 'f16'
        prefix = str(Path(block['directory']).relative_to(item.root))
        changed = item.bytes(prefix + '/metrics.jsonl', b''.join(encode(r) for r in rows))
        block['close']['metrics_pin'] = changed
        block['metrics_export']['pin'] = changed
        block['metrics_export']['facts']['manifest'] = rows[0]
        block['raw_pins']['session.json'] = item.write(prefix + '/session.json', block['close'])
        item.reseal()
        self.reject_admission(item)

    def test_owner_resampling_rejects_changed_n_zero_nonfinite_and_boolean_samples(self):
        valid = [[100] * 5, [100] * 5]
        for owners in [[[100] * 5], [[100] * 4, [100] * 5], [[0] * 5, [100] * 5],
                       [[float('nan')] * 5, [100] * 5], [[True] * 5, [100] * 5]]:
            with self.subTest(owners=owners), self.assertRaises((ValueError, TypeError)):
                consumer.bootstrap_owner_difference(valid, owners)


    def test_fresh_conditions_cannot_be_reused_from_an_earlier_owner(self):
        item = self.baseline()
        item.blocks[1]['stable_conditions'] = copy.deepcopy(item.blocks[0]['stable_conditions'])
        self.assertLess(item.blocks[1]['stable_conditions'][-1]['checked_uptime_ns'],
                        item.blocks[1]['before']['checked_ns'])
        item.reseal()
        with self.assertRaises(ValueError):
            item.admit()

    def test_first_condition_cannot_precede_its_own_before_checkpoint(self):
        item = self.baseline()
        owner = item.blocks[1]
        owner['stable_conditions'][0]['checked_uptime_ns'] = owner['before']['checked_ns'] - 1
        times = [row['checked_uptime_ns'] for row in owner['stable_conditions']]
        self.assertEqual(times, sorted(set(times)))
        self.assertLess(times[-1], owner['launch_call_ns'])
        item.reseal()
        with self.assertRaises(ValueError):
            item.admit()

    def test_cli_malformed_sequences_produce_only_incomparable_receipts(self):
        for case in ('empty-power', 'wrong-power-type', 'empty-guard-command'):
            with self.subTest(case=case):
                folder = self.root / case
                folder.mkdir()
                baseline = Receipt(folder, 'baseline')
                candidate = Receipt(folder, 'candidate', variant=2, epoch=100000000000000,
                                    snapshot=True, base=baseline)
                if case == 'empty-guard-command':
                    baseline.guard_plan_doc['command'] = []
                    baseline.guard_plan = baseline.write('guard-plan.json', baseline.guard_plan_doc)
                    baseline.capture_doc['outer_guard_plan'] = baseline.guard_plan
                else:
                    baseline.blocks[0]['stable_conditions'][4]['power'] = '' if case == 'empty-power' else 123
                baseline.reseal()
                old = baseline.write('cli-input.json', dict(capture_pin=baseline.capture,
                    revision_proof=baseline.revision, observer_proof=baseline.observer))
                new = candidate.write('cli-input.json', dict(capture_pin=candidate.capture,
                    revision_proof=candidate.revision, observer_proof=candidate.observer))
                output = folder / 'screen'
                code = consumer.main([old['path'], new['path'], str(output), '--policy', 'per-tile'])
                self.assertEqual(code, 1)
                self.assertEqual(sorted(p.name for p in output.iterdir()), ['failure.json'])
                failure = json.loads((output / 'failure.json').read_bytes())
                self.assertEqual(failure['status'], 'incomparable')
                self.assertIs(failure['accepted_optimization'], False)
                self.assertIsInstance(failure['reason'], str)
                self.assertLessEqual(len(failure['reason']), 2048)

    def _assert_metadata_substitution_rejected(self, kind):
        victim = self.root / 'local-input.json'
        victim.write_bytes(b'{"v":1}\n')
        target = self.root / 'substitution-target.json'
        target.write_bytes(b'{"v":2}\n')
        real_open, real_io_open, real_read_bytes = os.open, io.open, Path.read_bytes
        swapped = False
        opens = []

        def selected(path):
            return isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(os.fspath(path)) == str(victim)

        def substitute():
            nonlocal swapped
            if not swapped:
                victim.unlink()
                if kind == 'symlink':
                    victim.symlink_to(target)
                else:
                    os.mkfifo(victim, 0o600)
                swapped = True

        def descriptor_open(path, flags, *args, **kwargs):
            if selected(path):
                substitute()
                opens.append(flags)
                self.assertEqual(flags & (os.O_NOFOLLOW | os.O_NONBLOCK), os.O_NOFOLLOW | os.O_NONBLOCK)
            return real_open(path, flags, *args, **kwargs)

        def forbidden_path_read(path):
            if selected(path):
                substitute()
                raise AssertionError('initial metadata used an unbounded pathname read after substitution')
            return real_read_bytes(path)

        def no_unsafe_path_open(path, *args, **kwargs):
            if selected(path):
                substitute()
                raise AssertionError('initial metadata opened a substituted pathname without admitted descriptor flags')
            return real_io_open(path, *args, **kwargs)

        # Old Path.read_bytes is trapped before any link traversal or FIFO open.
        # The admitted descriptor path may open a FIFO only with NONBLOCK and
        # must reject its nonregular fstat before reading any metadata content.
        with mock.patch.object(consumer.os, 'open', new=descriptor_open), \
             mock.patch.object(Path, 'read_bytes', new=forbidden_path_read), \
             mock.patch.object(io, 'open', new=no_unsafe_path_open):
            with self.assertRaises((ValueError, OSError)):
                consumer.local_pin(victim)
        self.assertTrue(swapped)
        self.assertEqual(len(opens), 1)

    def test_local_metadata_symlink_substitution_is_rejected_without_path_read(self):
        self._assert_metadata_substitution_rejected('symlink')

    def test_local_metadata_fifo_substitution_is_nonblocking_and_rejected(self):
        self._assert_metadata_substitution_rejected('fifo')

    def test_local_metadata_growth_after_inspection_is_bounded_and_rejected(self):
        victim = self.root / 'growing-input.json'
        victim.write_bytes(b'12345678')
        real_open, real_read, real_fstat = os.open, os.read, os.fstat
        real_fdopen, real_io_open, real_read_bytes = os.fdopen, io.open, Path.read_bytes
        opened = []
        inspected = []
        reads = []
        grew = False

        def selected(path):
            return isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(os.fspath(path)) == str(victim)

        def append_four_bytes():
            nonlocal grew
            if not grew:
                with real_io_open(victim, 'ab', buffering=0) as stream:
                    stream.write(b'grew')
                grew = True

        def before_read(size):
            self.assertIs(type(size), int)
            self.assertGreater(size, 0)
            self.assertLessEqual(size, 2 * 1024**2 + 1)
            self.assertTrue(any(row.st_size == 8 for row in inspected), 'descriptor must be inspected before reading')
            reads.append(size)
            append_four_bytes()

        def descriptor_open(path, flags, *args, **kwargs):
            if selected(path):
                self.assertEqual(flags & (os.O_NOFOLLOW | os.O_NONBLOCK), os.O_NOFOLLOW | os.O_NONBLOCK)
                fd = real_open(path, flags, *args, **kwargs)
                opened.append(fd)
                return fd
            return real_open(path, flags, *args, **kwargs)

        def descriptor_stat(fd):
            result = real_fstat(fd)
            if fd in opened:
                inspected.append(result)
            return result

        def descriptor_read(fd, size):
            if fd in opened:
                before_read(size)
            return real_read(fd, size)

        class Stream:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def __getattr__(self, name):
                return getattr(self.stream, name)
            def read(self, size=-1):
                before_read(size)
                return self.stream.read(size)

        def descriptor_stream(fd, *args, **kwargs):
            stream = real_fdopen(fd, *args, **kwargs)
            return Stream(stream) if fd in opened else stream

        def forbidden_path_read(path):
            if selected(path):
                append_four_bytes()
                raise AssertionError('initial metadata attempted an unbounded pathname read after file growth')
            return real_read_bytes(path)

        def no_unsafe_path_open(path, *args, **kwargs):
            if selected(path):
                raise AssertionError('initial metadata bypassed descriptor admission')
            return real_io_open(path, *args, **kwargs)

        with mock.patch.object(consumer.os, 'open', new=descriptor_open), \
             mock.patch.object(consumer.os, 'fstat', new=descriptor_stat), \
             mock.patch.object(consumer.os, 'read', new=descriptor_read), \
             mock.patch.object(consumer.os, 'fdopen', new=descriptor_stream), \
             mock.patch.object(Path, 'read_bytes', new=forbidden_path_read), \
             mock.patch.object(io, 'open', new=no_unsafe_path_open):
            with self.assertRaises((ValueError, OSError)):
                consumer.local_pin(victim)
        self.assertTrue(grew)
        self.assertEqual(len(opened), 1)
        self.assertTrue(reads)
        self.assertEqual(victim.stat().st_size, 12)


if __name__ == '__main__':
    unittest.main()
