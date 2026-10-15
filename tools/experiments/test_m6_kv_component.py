"""Portable protocol, accounting and owned-lifetime checks; no Metal execution."""
import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import m6_kv_component as kv
import m6_kv_component_build as build


def ready(model='small'):
    layers, heads, d = kv.GEOMETRIES[model]
    return dict(event='ready', format='toklet-m6-kv-component-v1', model_shape=model,
                collection='primary', layers=layers, query_heads=heads, kv_heads=2, head_dim=d,
                capacity=32768, debug_assertions=False, m6_complete=False,
                admitted_bytes=kv.admitted(model), framework_allowance_bytes=512*1024**2,
                driver_allowance_bytes=64*1024**2, scratch_allowance_bytes=1024**2,
                oracle_score_bytes=32768*8, oracle_output_bytes=heads*d*16,
                device='portable fixture', execution_width=32,
                input_hashes=[[f'{i*7+j:064x}' for j in range(7)] for i in range(layers)],
                quantization=[dict(layer=i, **{name: dict(values=2*32768*d,
                    max_reconstruction_error=.01, rms_reconstruction_error=.001,
                    clamped_values=2) for name in ('k', 'v')}) for i in range(layers)])


def events(model='small'):
    layers, heads, d = kv.GEOMETRIES[model]
    rows = [ready(model), dict(event='correctness', result=dict(compared_outputs=3*layers*heads*d,
        expected_outputs=3*layers*heads*d, every_layer_checked=True, rejected_invalid_cases=7,
        exact_word_values_checked=256, valid_attention_outputs_checked=6*heads*d,
        quantization_error_is_not_arithmetic_error=True, max_arithmetic_error=.0001,
        quantization_attention_max_error=.01, quantization_attention_rms_error=.001))]
    for pair in range(13):
        for fmt in kv.FORMATS if pair % 2 == 0 else kv.FORMATS[::-1]:
            rows.append(dict(event='sample', pair=pair, phase='warmup' if pair < 3 else 'sample',
                sample=dict(format=fmt, operator_count=layers, command_count=1, status_zero=True,
                    output_guard_unchanged=True, completed_gpu_seconds=1. if fmt == 'fp16' else .9,
                    completed_host_seconds=1.1, host_encode_seconds=.001)))
    rows.append(dict(event='done', samples=26, inputs_unchanged=True, every_layer_checked=True,
                     cache_write_cost_included=False, m6_complete=False))
    return rows


class Component(unittest.TestCase):
    def test_complete_geometry_and_guarded_physical_accounting(self):
        # Independent payload + exact guards/queries/status/oracle/allowances.
        for model, payload, layers, query in [('small', 384*1024**2, 24, 896),
                                               ('larger', 896*1024**2, 28, 1536)]:
            expected = (payload + payload*17//32 + layers*(query*2+7*7*2)
                        + (query+7)*2+32 + 577*1024**2 + 32768*8 + query*16)
            self.assertEqual(kv.admitted(model), expected)
            self.assertLess(expected, kv.LIMIT)
            event = ready(model)
            kv.ready(event, model, 'primary')
            for key, value in [('admitted_bytes', expected-1), ('capacity', 32768.),
                               ('debug_assertions', 0), ('m6_complete', True), ('execution_width', 16)]:
                altered = copy.deepcopy(event); altered[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    kv.ready(altered, model, 'primary')
        event = ready(); event['input_hashes'][1] = event['input_hashes'][0]
        with self.assertRaises(ValueError): kv.ready(event, 'small', 'primary')

    def test_witness_coverage_and_no_quantization_gate_substitution(self):
        event = events()[1]
        kv.correctness(event, 'small')
        # Reconstruction quality remains recorded separately; no invented gate.
        event['result']['quantization_attention_max_error'] = 100.
        kv.correctness(event, 'small')
        for key, value in [('compared_outputs', 1), ('exact_word_values_checked', 128),
                           ('rejected_invalid_cases', 6), ('max_arithmetic_error', float('nan'))]:
            altered = copy.deepcopy(event); altered['result'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): kv.correctness(altered, 'small')

    def test_alternation_all_samples_and_descriptive_gain(self):
        rows = events()
        for pair in range(13):
            for ordinal in range(2):
                kv.sample(rows[2+pair*2+ordinal], 'small', pair, ordinal)
        summary = kv.summaries(rows)
        self.assertAlmostEqual(summary['gpu_median_reduction_fraction'], .1)
        self.assertFalse(summary['adoption_qualified'])
        # A slow outlier must survive and break range separation.
        rows[8]['sample']['completed_gpu_seconds'] = 2.
        summary = kv.summaries(rows)
        self.assertEqual(summary['i8']['completed_gpu_seconds']['maximum'], 2.)
        self.assertFalse(summary['gpu_observed_ranges_separate'])
        with self.assertRaises(ValueError): kv.summaries(rows[:-2])
        with self.assertRaises(ValueError): kv.sample(rows[2], 'small', 1, 0)

    def test_bounded_record_and_repository_path_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input'
            for content in (b'', b'NaN', b'{"bad":Infinity}'):
                path.write_bytes(content)
                with self.assertRaises(ValueError): kv.load(path)
            path.write_bytes(b'{}')
            with self.assertRaises(ValueError): kv.load(path, 1)
        for path in ('../escape', '/tmp/escape'):
            with self.assertRaises(ValueError): build.relative(path)

    def test_bounded_read_uses_one_descriptor_and_hashes_parsed_bytes(self):
        # No earlier stat/reopen can admit a different, larger replacement.
        path = Mock()
        source = Mock(wraps=io.BytesIO(b'{}extra'))
        path.open.return_value.__enter__ = Mock(return_value=source)
        path.open.return_value.__exit__ = Mock(return_value=False)
        with self.assertRaises(ValueError): kv.load(path, 2)
        source.read.assert_called_once_with(3)
        path.open.assert_called_once_with('rb')
        path.stat.assert_not_called()
        path = Mock()
        path.open.return_value.__enter__ = Mock(return_value=io.BytesIO(b'{"value":7}'))
        path.open.return_value.__exit__ = Mock(return_value=False)
        value, sha = kv.load(path)
        self.assertEqual(value, {'value': 7})
        self.assertEqual(sha, kv.hashlib.sha256(b'{"value":7}').hexdigest())

    def test_baseline_receipt_checks_the_same_buffer_it_parses(self):
        # Stop after receipt admission, before any native/source inventory work.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt = root/build.BASE_RECEIPT
            receipt.parent.mkdir(parents=True)
            receipt.write_bytes(b'{"passed":false}')
            sha = build.hashlib.sha256(receipt.read_bytes()).hexdigest()
            with patch.object(build, 'ROOT', root), patch.object(build, 'BASE_SHA', sha), \
                 patch.object(build, 'digest', side_effect=AssertionError('must not rehash receipt')):
                with self.assertRaisesRegex(ValueError, 'review missing'): build.baseline()
            receipt.write_bytes(b'{"passed":true}')
            with patch.object(build, 'ROOT', root), patch.object(build, 'BASE_SHA', sha):
                with self.assertRaisesRegex(ValueError, 'receipt differs'): build.baseline()

    def test_build_rejects_configuration_and_toolchain_on_actual_cwd_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot, home = root/'artifacts/snapshot', root/'cargo-home'
            snapshot.mkdir(parents=True); home.mkdir()
            build.reject_ambient_configuration(snapshot, home)
            for path in (root/'artifacts/.cargo/config.toml', snapshot/'.cargo/config',
                         root/'rust-toolchain', root/'artifacts/rust-toolchain.toml', home/'config.toml'):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text('unexpected')
                with self.subTest(path=path), self.assertRaises(ValueError):
                    build.reject_ambient_configuration(snapshot, home)
                path.unlink()

    def test_source_closure_rejects_drift_missing_files_and_unbound_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot = root/'artifacts/snapshot'
            snapshot.mkdir(parents=True)
            binary = root/'artifacts/probe'; binary.write_bytes(b'executable fixture')
            original, copied = {'Cargo.toml': 'a'*64}, {'Cargo.toml': 'a'*64, 'src/probe.rs': 'b'*64}
            record = dict(format='toklet-m6-kv-component-build-v1', passed=True,
                snapshot='artifacts/snapshot', root_sources=original, snapshot_sources=copied,
                tools_and_contract={}, binary=dict(path='artifacts/probe', sha256=build.digest(binary)))
            def verify(value, actual_copy=None):
                def inv(path): return original if path == root else copied if actual_copy is None else actual_copy
                with patch.object(build, 'ROOT', root), patch.object(build, 'baseline'), \
                     patch.object(build, 'inventory', side_effect=inv), \
                     patch.object(build, 'expected_snapshot', return_value=copied), \
                     patch.object(build, 'tool_paths', return_value=[]):
                    return build.verify_record(value)
            self.assertIn(binary, verify(record))
            for field in ('root_sources', 'snapshot_sources', 'tools_and_contract'):
                altered = copy.deepcopy(record); altered[field] = {'unexpected': 'c'*64}
                with self.subTest(field=field), self.assertRaises(ValueError): verify(altered)
            with self.assertRaises(ValueError): verify(record, {'Cargo.toml': 'a'*64})
            binary.write_bytes(b'changed executable')
            with self.assertRaises(ValueError): verify(record)

    def exercise(self, directory, rows, *, memory=1024, endpoint_failure=False, finish=0,
                 cleanup_failure=False, environment_failure=False):
        output = Path(directory)/'result.json'
        args = argparse.Namespace(output=output, build=kv.ROOT/'artifacts/fake-build.json',
            build_sha256='a'*64, model='small', collection='primary')
        child = Mock()
        child.stdin, child.stdout, child.stderr = io.BytesIO(), io.BytesIO(), io.BytesIO()
        child.wait.return_value = 0
        if cleanup_failure:
            child.wait.side_effect = RuntimeError('reap failed')
        transport = Mock(pending=b'', stdout_closed=True, events=29)
        transport.read.side_effect = rows
        transport.finish.return_value = finish
        contract = dict(format='toklet-m6-i8-kv-component-contract-v1')
        checks = [None, ValueError('endpoint changed') if endpoint_failure else None]
        def check():
            value = checks.pop(0)
            if value: raise value
        bindings = {kv.ROOT/kv.CONTRACT: 'b'*64}
        with patch.object(kv, 'build_admission', return_value=(kv.ROOT/'artifacts/fake', bindings, check)), \
             patch.object(kv, 'load', return_value=(contract, 'b'*64)), \
             patch.object(kv.subprocess, 'Popen', return_value=child), \
             patch.object(kv.metal_engine, 'Transport', return_value=transport), \
             patch.object(kv.metal_engine, 'kill_owned_group') as kill, \
             patch.object(kv, 'observe_environment', side_effect=[{}, RuntimeError('power endpoint failed')]
                          if environment_failure else [{}, {}]), \
             patch.object(kv.native_usage, 'usage', return_value={'lifetime_max_phys_footprint': memory}) as usage:
            try:
                kv.run(args)
            except BaseException as error:
                failure = error
            else:
                failure = None
            return json.loads(output.read_text()), failure, kill.call_count, usage.call_count

    def test_owned_success_records_all_events_and_samples_memory_before_final_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            result, failure, kills, samples = self.exercise(directory, events())
        self.assertIsNone(failure)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['events']), 29)
        self.assertEqual(len(result['usage']), 29)
        self.assertEqual(samples, 58)
        self.assertEqual(kills, 1)
        self.assertEqual(result['exit_code'], 0)
        self.assertFalse(result['m6_complete'])

    def test_failure_preserves_evidence_and_cleans_owned_child(self):
        cases = [dict(rows=events(), memory=kv.LIMIT), dict(rows=events(), finish=1),
                 dict(rows=events(), endpoint_failure=True), dict(rows=events(), cleanup_failure=True),
                 dict(rows=[ready(), TimeoutError('bounded event timeout')])]
        for case in cases:
            with self.subTest(case=list(case)), tempfile.TemporaryDirectory() as directory:
                result, failure, kills, _ = self.exercise(directory, **case)
            self.assertIsNotNone(failure)
            self.assertFalse(result['passed'])
            self.assertEqual(kills, 1)
            self.assertGreaterEqual(len(result['events']), 1)

    def test_nonfinite_events_and_endpoint_errors_preserve_primary_failure_and_report(self):
        for number in (float('nan'), float('inf'), -float('inf')):
            rows = events(); rows[1]['result']['max_arithmetic_error'] = number
            with self.subTest(number=number), tempfile.TemporaryDirectory() as directory:
                result, failure, kills, _ = self.exercise(directory, rows, environment_failure=True)
            self.assertIsInstance(failure, ValueError)
            self.assertIn('Nonfinite component event', str(failure))
            self.assertFalse(result['passed'])
            self.assertEqual(len(result['events']), 1)
            self.assertEqual(result['rejected_event']['ordinal'], 2)
            self.assertEqual(kills, 1)
            self.assertIn('power endpoint failed', result['environment_errors'][0])


if __name__ == '__main__':
    unittest.main()
