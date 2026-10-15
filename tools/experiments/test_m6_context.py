"""Portable M6 collection admission, numerical receipts and owned lifetimes."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import m6_context as m6


def admission(model='small', capacity=4096, profiling=False):
    large = model == 'larger'
    layers, heads, dimension, weights = (28, 12, 128, 1777247428) if large else (24, 14, 64, 988069596)
    memory = dict(weights_bytes=weights, kv_bytes=(2*capacity*dimension+7)*4*layers,
                  scratch_bytes=100000, history_bytes=capacity*4, logits_bytes=151936*8,
                  metadata_bytes=500, metadata_allowance_bytes=(128 if profiling else 64)*1024**2)
    online = {**memory, 'admitted_bytes': sum(memory.values())}
    memory['scratch_bytes'] += heads*128*capacity*8-8
    reference = {**memory, 'admitted_bytes': sum(memory.values())}
    return dict(event='ready', format='toklet-m6-context-v1', debug_assertions=False,
                device='fixture', context_capacity=capacity, chunk_size=128, profiling=profiling,
                hidden_size=1536 if large else 896, num_layers=layers, vocab_size=151936,
                projection_mode='Q4CompensatedTiled' if large else 'HalfCompensated',
                normalization_mode='Compensated' if large else 'Original', readback_mode='FullLogits',
                kv_dtype='FP16', weight_dtype='Q4 v2 + FP16 exclusions' if large else 'FP16',
                packed_format_version=2 if large else None, group_size=8 if large else None,
                quantizer='toklet-q4-affine-f32-source-half-metadata-rne-g8-v2' if large else None,
                q4_execution_widths=[32, 32], half_execution_widths=[32, 32], shared_weights_bytes=weights,
                application_bytes=64*1024**2, reference_memory=reference, online_memory=online,
                unique_memory_admitted_bytes=reference['admitted_bytes']+online['admitted_bytes']-weights+64*1024**2,
                load_seconds=.1)


def state(ids, evaluated):
    return dict(history_tokens=len(ids), evaluated_tokens=evaluated, pending_tokens=len(ids)-evaluated,
                history_sha256=m6.shared.history_sha(ids))


def comparison(ids, memories):
    rows = [dict(event='comparison_begin', scored_rows=128, initial_prefix_tokens=len(ids)-128)]
    for i, token in enumerate(ids[-128:]):
        rows.append(dict(event='comparison_row', target_index=i, target_token=token, prefix_tokens=len(ids)-128+i,
                         reference_nll=1., online_nll=1.01, reference_greedy=1, online_greedy=1,
                         reference_margin=.5, reference_winner_logits=[2., 2.01], online_winner_logits=[2., 2.01],
                         max_absolute_logit_delta=.02, rms_logit_delta=.01, vocabulary_values_compared=151936,
                         reference_logits_sha256='a'*64, online_logits_sha256='b'*64))
    rows.append(dict(event='comparison_done', scored_rows=128, memories=memories, states=[state(ids, len(ids)-1)]*2))
    return rows


class Context(unittest.TestCase):
    def test_ready_exact_modes_geometry_complete_memory_all_depths(self):
        for model in ('small', 'larger'):
            for capacity in (4096, 8192, 16384, 32768):
                for profiling in (False, True):
                    m6.ready(admission(model, capacity, profiling), model, capacity, 128, profiling)
        event = admission()
        for key, value in [('debug_assertions', True), ('profiling', 0), ('projection_mode', 'Original'),
                           ('context_capacity', 4096.), ('kv_dtype', 'I8'), ('group_size', 8),
                           ('q4_execution_widths', [32, 64]), ('load_seconds', float('nan')),
                           ('unique_memory_admitted_bytes', m6.LIMIT+1), ('application_bytes', True)]:
            changed = copy.deepcopy(event); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m6.ready(changed, 'small', 4096, 128, False)
        for field in ('kv_bytes', 'weights_bytes', 'scratch_bytes', 'history_bytes', 'admitted_bytes', 'metadata_allowance_bytes'):
            changed = copy.deepcopy(event); changed['online_memory'][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                m6.ready(changed, 'small', 4096, 128, False)

    def test_bounded_bytes_reject_truncation_replacement_and_excess(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'bytes'; p.write_bytes(b'ok')
            sha = m6.hashlib.sha256(b'ok').hexdigest()
            self.assertEqual(m6.bounded(p, 2, sha), b'ok')
            for data in (b'o', b'no', b'okx'):
                p.write_bytes(data)
                with self.assertRaises(ValueError): m6.bounded(p, 2, sha)

    def test_quality_coverage_states_gate_and_near_ties(self):
        ids = [1]*256; memories = [admission()[f'{m}_memory'] for m in m6.MODES]
        rows = comparison(ids, memories)
        value = m6.comparison(rows, ids, memories)
        self.assertTrue(value['gate_passed']); self.assertAlmostEqual(value['mean_nll_increase'], .01)
        for key, value in [('target_index', 1), ('target_token', 2), ('prefix_tokens', 99),
                           ('vocabulary_values_compared', 1), ('reference_nll', True),
                           ('rms_logit_delta', 1.), ('online_logits_sha256', 'x'*64),
                           ('reference_winner_logits', [2., 5.]), ('reference_greedy', -1)]:
            changed = copy.deepcopy(rows); changed[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): m6.comparison(changed, ids, memories)
        with self.assertRaises(ValueError): m6.comparison(rows[:-1], ids, memories)
        rows[-1]['states'][0] = state(ids, len(ids))
        with self.assertRaises(ValueError): m6.comparison(rows, ids, memories)
        rows = comparison(ids, memories)
        rows[1].update(reference_margin=.01, online_greedy=2, online_winner_logits=[1.995, 2.011])
        changed = m6.comparison(rows, ids, memories)['argmax_changes']
        self.assertTrue(changed[0]['within_error_envelope'])
        for row in rows[1:-1]: row['online_nll'] = 1.1
        self.assertFalse(m6.comparison(rows, ids, memories)['gate_passed'])

    def test_union_counts_overlaps_once_and_keeps_gaps(self):
        self.assertEqual(m6.union([(4., 5.), (1., 3.), (2., 4.), (8., 9.)]), 5.)
        with self.assertRaises(ValueError): m6.union([(2., 1.)])

    def test_profile_conversion_and_denominator_use_matching_completed_gpu(self):
        p = dict(name='embedding', host_begin_seconds=.0001, host_end_seconds=.0002,
                 gpu_begin_ticks=101, gpu_end_ticks=201,
                 gpu_begin_seconds=1e-8, gpu_end_seconds=1.01e-6)
        begin = dict(event='profile_begin', mode='reference',
                     start=dict(cpu_nanoseconds=1000, gpu_ticks=100),
                     end=dict(cpu_nanoseconds=10001000, gpu_ticks=1000100),
                     host_clock_sample_bracket_seconds=1e-7)
        timing = dict(command_count=1, command_creation_seconds=0., clock_calibration_seconds=0.,
                      commit_seconds=0., host_encode_seconds=.001, wait_seconds=.01,
                      gpu_start_seconds=100., gpu_end_seconds=100.005, readback_seconds=0.)
        done = dict(event='profile_done', mode='reference', timing=timing)
        ops = []
        for i in range(412):
            operator = copy.deepcopy(p); operator['name'] = 'attention' if i < 24 else 'other'
            ops.append(dict(event='profile_operator', index=i, profile=operator))
        ids = [1]*4096; memory = admission(profiling=True)['reference_memory']
        events = [begin, *ops, done]
        events[0].update(reused_tokens=0, prefill_tokens=4095)
        events[-1].update(operator_count=412, state=state(ids, 4096), memory=memory)
        value = m6.profile(events, 'reference', memory, 'small', ids, False)
        self.assertAlmostEqual(value['attention_fraction'], (p['gpu_end_seconds']-p['gpu_begin_seconds'])/(events[-1]['timing']['gpu_end_seconds']-events[-1]['timing']['gpu_start_seconds']))
        events[0]['reused_tokens'] = 4094; events[0]['prefill_tokens'] = 1
        m6.profile(events, 'reference', memory, 'small', ids, True)
        for mutation in ('conversion', 'coverage', 'clock', 'pending', 'reuse', 'denominator'):
            changed = copy.deepcopy(events)
            if mutation == 'conversion': changed[1]['profile']['gpu_begin_seconds'] += 1
            if mutation == 'coverage': changed.pop(1)
            if mutation == 'clock': changed[0]['end']['gpu_ticks'] = changed[0]['start']['gpu_ticks']
            if mutation == 'pending': changed[-1]['state']['pending_tokens'] = 1
            if mutation == 'reuse': changed[0]['reused_tokens'] = 4095
            if mutation == 'denominator': changed[-1]['timing']['gpu_end_seconds'] = changed[-1]['timing']['gpu_start_seconds']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): m6.profile(changed, 'reference', memory, 'small', ids, True)

    def test_launch_reassertion_precedes_popen(self):
        def reject(): raise ValueError('changed')
        w = m6.Worker(Path('bin'), Path('source'), None, 2048, 128, False, 'small', io.StringIO(), io.BytesIO(), reject)
        with patch.object(m6.subprocess, 'Popen') as launch, self.assertRaisesRegex(ValueError, 'changed'):
            with w: pass
        launch.assert_not_called()

    def test_primary_sends_exact_frozen_512_128_alternating_three_plus_ten_pairs(self):
        manifest, arrays = m6.shared.contract()
        class Fake:
            model = 'small'
            admission = {f'{m}_memory': {} for m in m6.MODES}
            calls = []
            def call(self, request, final):
                self.calls.append(request)
                return [dict(mode=request['mode'])]
        for plan in ('primary', 'confirmation'):
            worker = Fake(); worker.calls = []; result = {}
            with patch.object(m6.shared, 'sample', return_value={k: 1. for k in ('prefill_seconds', 'ttft_seconds', 'decode_seconds', 'total_seconds')}), patch('builtins.print'):
                m6.collect(worker, plan, [], manifest, arrays, result)
            self.assertEqual(len(worker.calls), 26)
            for i in range(13):
                self.assertEqual([r['mode'] for r in worker.calls[i*2:i*2+2]], list(m6.MODES if i % 2 == 0 else m6.MODES[::-1]))
            for request in worker.calls:
                self.assertEqual(request['action'], 'reset')
                self.assertEqual(request['tokens'], arrays['prompt_512'])
                self.assertEqual(request['continuation'], arrays['continuation_128'])
            self.assertEqual([r['phase'] for r in result['workloads'][0]['samples']].count('warmup'), 6)
        matrix = m6.measurement_plan('matrix')
        self.assertEqual([v for v in matrix if v[1:] == (3, 10)], [('medium', 3, 10)])
        self.assertEqual(set(v[0] for v in matrix), set(m6.shared.frozen.WORKLOADS))

    def test_finalization_error_survives_cleanup_error_and_all_actions_run(self):
        worker = m6.Worker(Path('bin'), Path('source'), None, 2048, 128, False, 'small', None, None, lambda: None)
        worker.child = Mock(); worker.transport = Mock()
        worker.transport.finish.side_effect = TimeoutError('primary finish timeout')
        worker.child.wait.side_effect = RuntimeError('cleanup reap failed')
        with patch.object(m6.shared.metal_engine, 'kill_owned_group') as kill, self.assertRaisesRegex(TimeoutError, 'primary finish timeout'):
            worker.__exit__(None, None, None)
        self.assertEqual(worker.record['error'], 'TimeoutError: primary finish timeout')
        self.assertEqual(worker.record['cleanup_errors'], ['cleanup reap failed'])
        self.assertFalse(worker.record['passed']); kill.assert_called_once()
        worker.transport.close.assert_called_once()
        for pipe in (worker.child.stdin, worker.child.stdout, worker.child.stderr):
            pipe.close.assert_called()

    def test_controlled_pins_bind_original_not_recaptured_mutations(self):
        manifest, _ = m6.shared.contract()
        pins = m6.controlled_pins(manifest)
        self.assertEqual(pins[m6.shared.prepare.CONTRACT/'manifest.json'], m6.shared.CONTRACT_SHA)
        for item in manifest['arrays'].values():
            self.assertEqual(pins[m6.shared.prepare.CONTRACT/item['file']], item['sha256'])
        for name, sha in manifest['authored_text_sha256'].items():
            self.assertEqual(pins[m6.shared.prepare.CONTRACT/name], sha)
        self.assertEqual(pins[m6.ROOT/'benchmarks/sources.json'], manifest['sources_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); p = root/'input'; p.write_bytes(b'original')
            frozen = {p: m6.shared.metal_engine.digest(p)}
            with patch.object(m6, 'ROOT', root):
                m6.capture_identities([p], frozen)
                p.write_bytes(b'replacement after admission before capture')
                with patch.object(m6.subprocess, 'Popen') as launch, self.assertRaisesRegex(ValueError, 'Consumed frozen bytes'):
                    m6.capture_identities([p], frozen)
                launch.assert_not_called()

    def test_failed_startup_reaps_only_owned_worker(self):
        real = subprocess.Popen; owned = []
        def launch(command, **kwargs):
            worker = real([sys.executable, '-c', 'print("{}", flush=True); import time; time.sleep(60)'], **kwargs)
            owned.append(worker); return worker
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory)/'raw').open('w') as raw, (Path(directory)/'err').open('wb') as err:
                w = m6.Worker(Path('bin'), Path('source'), None, 2048, 128, False, 'small', raw, err, lambda: None)
                with patch.object(m6.subprocess, 'Popen', launch), self.assertRaises(ValueError):
                    with w: pass
        self.assertIsNotNone(owned[0].returncode)
        self.assertFalse(w.record['passed']); self.assertEqual(w.record['cleanup_errors'], [])


if __name__ == '__main__': unittest.main()
