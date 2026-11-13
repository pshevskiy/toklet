"""Portable complete-scope schema and rejection controls; no model/NumPy."""
import copy
import inspect
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import ministral3_graph_check as checker
import ministral3_graph_protocol as protocol
import ministral3_graph_reference as driver

SHA = '1' * 64


def prospective():
    return dict(format='toklet-ministral3-native-graph-protocol-v1',
                model=dict(data_sha256='2' * 64, config_sha256='3' * 64),
                cases=[dict(name=name, tokens=tokens, offset=offset) for name, tokens, offset in protocol.TINY_CASES
                       if not name.startswith('diagnostic_')], chunks=[1, 3])


def receipt(size=1, digest=SHA):
    identity = dict(dev=1, ino=2, bytes=size, mtime=3, mtime_nsec=4, ctime=5, ctime_nsec=6, mode=33188, nlink=1)
    return dict(path='/absolute/test/file', bytes=size, sha256=digest,
                before_identity=identity, after_identity=dict(identity), after_sha256=digest)


def capture_manifest():
    p = prospective()
    runs = []
    for case in p['cases']:
        for chunk in p['chunks']:
            for repeat in checker.REPEATS + (('changed_prefix',) if case['name'] == 'future_changed' else ()):
                row_offset = 3 if repeat == 'changed_prefix' else 0
                arrays = []
                for name, shape in protocol.contracts(protocol.REAL, len(case['tokens']) - row_offset).items():
                    if name.startswith('layers.'):
                        _, layer, stage = name.split('.')
                        layer = int(layer)
                        actual = f'layer{layer:02}.{stage}'
                    else:
                        actual, layer, stage = name, None, name
                    dtype = '<f4' if name == 'logits' else '<f2'
                    arrays.append(dict(name=actual, layer=layer, stage=stage,
                                       path=f"{case['name']}/{chunk}/{repeat}/{name}.bin", dtype=dtype,
                                       shape=shape, bytes=shape[0] * shape[1] * (4 if dtype == '<f4' else 2), sha256=SHA))
                chunks = []
                for first in range(row_offset, len(case['tokens']), chunk):
                    memory = dict(weights_bytes=100, positional_bytes=20, kv_bytes=30, scratch_bytes=40,
                                  history_bytes=50, logits_bytes=60, metadata_bytes=70, metadata_allowance_bytes=80)
                    memory['admitted_bytes'] = sum(memory.values())
                    timing = {name: 0.0 for name in ('command_creation_seconds', 'clock_calibration_seconds',
                              'commit_seconds', 'host_encode_seconds', 'wait_seconds', 'gpu_start_seconds',
                              'gpu_end_seconds', 'readback_seconds')}
                    timing['command_count'] = 1
                    chunks.append(dict(start=first, absolute_start=case['offset'] + first,
                                       rows=min(chunk, len(case['tokens']) - first), wall_seconds=0.1,
                                       diagnostics_seconds=0.01, timing=timing, memory=memory))
                applicable = len(case['tokens']) < 512 - case['offset']
                if repeat == 'initial':
                    checks = dict(fully_cached_prompt_zero_work=True, invalid_inputs_preserve_committed_prefix=True)
                    if applicable:
                        checks['pending_token_uncommitted_discard_restores_distribution'] = True
                elif repeat == 'reset_replay':
                    checks = dict(reset_replay_all_stage_bytes_equal=True)
                else:
                    checks = dict(changed_prefix_all_suffix_bytes_equal_fresh=True)
                runs.append(dict(case=case['name'], tokens=case['tokens'][:], offset=case['offset'],
                                 chunk=chunk, repeat=repeat, row_offset=row_offset, arrays=arrays, chunks=chunks,
                                 final_logits_sha256=SHA, final_greedy_token=1,
                                 diagnostic_final_row_equals_logits=True, checks=checks,
                                 pending_token_applicable=applicable))
    return dict(format='toklet-ministral3-native-graph-capture-v1', completed=True,
                graph_qualified=False, error=None, active_run=None,
                protocol=receipt(digest=SHA), model_data=receipt(2662082312, p['model']['data_sha256']),
                model_config=receipt(digest=p['model']['config_sha256']), build_manifest=receipt(), binary=receipt(),
                capture_source=dict(bytes=1000, sha256=SHA), device='Portable metadata fixture', load_seconds=0.1,
                engine_budget_bytes=checker.ENGINE_LIMIT, helper_limit_bytes=32 * 1024**2, helper_high_water_bytes=1024,
                checks=dict(all_input_descriptor_path_and_byte_endpoints_unchanged=True,
                            complete_frozen_case_chunk_stage_inventory=True), runs=runs)


def reference_manifest():
    p = prospective()
    arrays = {}
    cases = []
    for case in p['cases']:
        for profile in ('main', 'wide'):
            for name, shape in protocol.contracts(protocol.REAL, len(case['tokens'])).items():
                key = case['name'] + '.' + profile + '.' + name
                dtype = '<f4' if name == 'logits' else '<f2'
                arrays[key] = dict(file=key + '.bin', dtype=dtype, shape=shape,
                                   bytes=shape[0] * shape[1] * (4 if dtype == '<f4' else 2), sha256=SHA)
            cases.append(dict(name=case['name'], profile=profile, arrays=419,
                              diagnostic_graph_and_publication_ns=1, final_logits_sha256=SHA,
                              top_two_ids=[1, 2], margin=0.0))
    return dict(format='toklet-ministral3-independent-reference-v1', mode='capture', passed=True,
                protocol_sha256=SHA, geometry=dict(protocol.REAL), numerical_policy=dict(protocol.POLICY),
                model=p['model'], source_consumption=dict(after_sha256=p['model']['data_sha256']),
                cases=cases, arrays=arrays, published_bytes=sum(a['bytes'] for a in arrays.values()))


class Protocol(unittest.TestCase):
    def test_exact_selected_protocol_has_five_cases_and_two_schedules(self):
        self.assertEqual(checker.validate_protocol(prospective()), prospective())

    def test_changed_case_tokens_offsets_schedules_and_extra_fields_reject(self):
        for change in ('tokens', 'offset', 'case', 'chunk', 'bool', 'extra'):
            value = prospective()
            if change == 'tokens': value['cases'][0]['tokens'] = [9]
            if change == 'offset': value['cases'][0]['offset'] = 1
            if change == 'case': value['cases'].pop()
            if change == 'chunk': value['chunks'] = [1, 2]
            if change == 'bool': value['chunks'] = [True, 3]
            if change == 'extra': value['extra'] = 1
            with self.subTest(change=change):
                with self.assertRaises(ValueError): checker.validate_protocol(value)

    def test_raw_descriptor_shape_dtype_extent_and_path_are_bound(self):
        good = dict(file='a.bin', dtype='<f2', shape=[1, 128], bytes=256, sha256=SHA)
        self.assertEqual(checker.array_spec(good, [1, 128])['bytes'], 256)
        for key, changed in [('shape', [True, 128]), ('bytes', True), ('bytes', 512),
                             ('dtype', '<f8'), ('file', '../escape'), ('sha256', 'x' * 64), ('extra', 1)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError): checker.array_spec(dict(good, **{key: changed}), [1, 128])


class Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.valid = capture_manifest()

    def value(self): return copy.deepcopy(self.valid)

    def test_complete_actual_twenty_two_run_inventory(self):
        indexed = checker.native_runs(self.value(), prospective(), SHA)
        self.assertEqual(len(indexed), 22)
        self.assertEqual(sum(len(v['arrays']) for v in indexed.values()), 9218)
        self.assertEqual(indexed['future_changed', 3, 'changed_prefix']['arrays']['logits']['shape'], [1, 131072])

    def test_missing_duplicate_or_unqualified_run_reject(self):
        for change in ('missing', 'duplicate', 'extra', 'self_qualified', 'partial', 'error'):
            value = self.value()
            if change == 'missing': value['runs'].pop()
            if change == 'duplicate': value['runs'][-1] = value['runs'][0]
            if change == 'extra': value['runs'][0]['repeat'] = 'other'
            if change == 'self_qualified': value['graph_qualified'] = True
            if change == 'partial': value['completed'] = False
            if change == 'error': value['error'] = 'incomplete'
            with self.subTest(change=change):
                with self.assertRaises(ValueError): checker.native_runs(value, prospective(), SHA)

    def test_native_missing_stage_duplicate_name_wrong_independent_query_shape_reject(self):
        for change in ('missing', 'duplicate', 'query_shape', 'path_alias', 'wrong_name'):
            value = self.value()
            run = value['runs'][0]
            if change == 'missing': run['arrays'].pop()
            if change == 'duplicate': run['arrays'][-1] = run['arrays'][0]
            if change == 'query_shape': next(a for a in run['arrays'] if a['name'] == 'layer00.q')['shape'] = [1, 3072]
            if change == 'path_alias': run['arrays'][-1]['path'] = run['arrays'][0]['path']
            if change == 'wrong_name': run['arrays'][0]['name'] = 'other'
            with self.subTest(change=change):
                with self.assertRaises(ValueError): checker.native_runs(value, prospective(), SHA)

    def test_changed_prefix_cannot_claim_fresh_or_wrong_reference_rows(self):
        for key, bad in [('row_offset', 0), ('offset', 1), ('tokens', [1]), ('chunk', True)]:
            value = self.value()
            run = next(r for r in value['runs'] if r['repeat'] == 'changed_prefix')
            run[key] = bad
            with self.subTest(key=key):
                with self.assertRaises(ValueError): checker.native_runs(value, prospective(), SHA)

    def test_completion_and_state_controls_are_typed_and_complete(self):
        for change in ('missing_state', 'false_state', 'unknown_state', 'pending', 'final_row', 'greedy'):
            value = self.value()
            run = value['runs'][0]
            if change == 'missing_state': run['checks'].popitem()
            if change == 'false_state': run['checks']['fully_cached_prompt_zero_work'] = False
            if change == 'unknown_state': run['checks']['unknown'] = True
            if change == 'pending': run['pending_token_applicable'] = 1
            if change == 'final_row': run['diagnostic_final_row_equals_logits'] = False
            if change == 'greedy': run['final_greedy_token'] = True
            with self.subTest(change=change):
                with self.assertRaises(ValueError): checker.native_runs(value, prospective(), SHA)

    def test_chunk_coverage_clocks_and_complete_memory_owners_are_required(self):
        for change in ('rows', 'absolute', 'commands', 'nan', 'gpu_order', 'memory', 'owner', 'positional'):
            value = self.value()
            chunk = value['runs'][0]['chunks'][0]
            if change == 'rows': chunk['rows'] = 0
            if change == 'absolute': chunk['absolute_start'] = 1
            if change == 'commands': chunk['timing']['command_count'] = 2
            if change == 'nan': chunk['wall_seconds'] = float('nan')
            if change == 'gpu_order': chunk['timing']['gpu_start_seconds'] = 1.0
            if change == 'memory': chunk['memory']['admitted_bytes'] = checker.ENGINE_LIMIT + 1
            if change == 'owner': chunk['memory']['admitted_bytes'] = 1
            if change == 'positional': del chunk['memory']['positional_bytes']
            with self.subTest(change=change):
                with self.assertRaises(ValueError): checker.native_runs(value, prospective(), SHA)

    def test_input_endpoints_and_fixed_engine_helper_budget_are_bound(self):
        for change in ('before', 'after_sha', 'model', 'protocol', 'engine', 'helper'):
            value = self.value()
            if change == 'before': value['binary']['before_identity']['ino'] = 9
            if change == 'after_sha': value['binary']['after_sha256'] = '4' * 64
            if change == 'model': value['model_data']['sha256'] = value['model_data']['after_sha256'] = '4' * 64
            if change == 'protocol': value['protocol']['sha256'] = value['protocol']['after_sha256'] = '4' * 64
            if change == 'engine': value['engine_budget_bytes'] += 1
            if change == 'helper': value['helper_high_water_bytes'] = value['helper_limit_bytes'] + 1
            with self.subTest(change=change):
                with self.assertRaises(ValueError): checker.native_runs(value, prospective(), SHA)


class Reference(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.valid = reference_manifest()

    def test_both_profiles_cover_every_declared_stage_coordinate(self):
        self.assertEqual(len(checker.reference_arrays(self.valid, prospective(), SHA)), 4190)

    def test_missing_profile_path_alias_wrong_gate_and_rank_reject(self):
        for change in ('missing', 'path_alias', 'gate', 'shape', 'rank', 'missing_summary'):
            value = copy.deepcopy(self.valid)
            first, second = list(value['arrays'])[:2]
            if change == 'missing': del value['arrays'][first]
            if change == 'path_alias': value['arrays'][second]['file'] = value['arrays'][first]['file']
            if change == 'gate': value['numerical_policy']['rtol'] = 0.01
            if change == 'shape': value['arrays'][second]['shape'] = [1, 1]
            if change == 'rank': value['cases'][0]['top_two_ids'] = [1, 1]
            if change == 'missing_summary': value['cases'].pop()
            with self.subTest(change=change):
                with self.assertRaises(ValueError): checker.reference_arrays(value, prospective(), SHA)


class Driver(unittest.TestCase):
    def run_actual_cli(self, mode, passed):
        """Execute the real __main__ entry point with only numerical work mocked."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'output'
            arguments = [str(Path(driver.__file__)), '--mode', mode, '--environment-root', '/missing-environment',
                         '--runtime-manifest', str(root / 'runtime.json'), '--runtime-sha256', SHA,
                         '--fresh-output', str(output)]
            if mode == 'compare':
                for name in ('protocol', 'reference', 'native'):
                    arguments.extend(['--' + name, str(root / 'inputs' / (name + '.json')),
                                      '--' + name + '-sha256', SHA])

            class Lease:
                def __enter__(self): return self
                def __exit__(self, *_): pass
                def document(self, *_): return prospective()
                def endpoint(self): return []

            numerical_result = dict(passed=passed, failures=0 if passed else 1,
                                    scope='Portable mocked numerical result')

            def admit(*_):
                # run_path creates a fresh CLI namespace. Replace its numerical
                # dispatch only after main reaches the real admission boundary;
                # publication, stdout, return and SystemExit remain real code.
                frame = inspect.currentframe()
                try:
                    namespace = frame.f_back.f_globals
                    namespace['controls'] = mock.Mock(return_value=dict(numerical_result))
                    namespace['InputLease'] = Lease
                finally:
                    del frame
                return {}, lambda: {}

            def comparison(*args):
                # The checker normally creates this scientific rejection journal
                # before returning False. Retain it through real CLI publication.
                (Path(args[-1]) / 'records.jsonl').write_text(json.dumps(numerical_result) + '\n')
                return dict(numerical_result)

            with mock.patch.dict(sys.modules, {'numpy': ModuleType('numpy')}), \
                    mock.patch.dict(driver.os.environ, {'MallocLargeCache': '0', 'MallocMediumZone': '0'}), \
                    mock.patch.object(driver.runtime, 'bind_environment'), \
                    mock.patch.object(driver.runtime, 'admit', new=admit), \
                    mock.patch.object(checker, 'compare', side_effect=comparison), \
                    mock.patch.object(sys, 'argv', arguments), \
                    mock.patch('sys.stdout', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(str(Path(driver.__file__)), run_name='__main__')
            self.assertEqual(stopped.exception.code, 0 if passed else 1)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertIs(manifest['passed'], passed)
            self.assertEqual(manifest['failures'], 0 if passed else 1)
            self.assertEqual(manifest['mode'], mode)
            self.assertEqual(manifest['format'], 'toklet-ministral3-independent-reference-v1')
            if mode == 'compare':
                self.assertIs(json.loads((output / 'records.jsonl').read_text())['passed'], passed)

    def test_actual_controls_cli_publishes_failed_gate_then_exits_one(self):
        for passed in (False, True):
            with self.subTest(passed=passed): self.run_actual_cli('controls', passed)

    def test_actual_compare_cli_retains_journal_and_failure_manifest_before_exit(self):
        for passed in (False, True):
            with self.subTest(passed=passed): self.run_actual_cli('compare', passed)

    def test_missing_external_inputs_and_occupied_outputs_reject_before_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'fresh'
            command = [sys.executable, '-s', '-B', str(Path(driver.__file__)), '--mode', 'capture',
                       '--environment-root', '/missing-environment', '--fresh-output', str(output)]
            result = subprocess.run(command, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Complete externally pinned reference arguments', result.stderr)
            self.assertFalse(output.exists())
            output.write_bytes(b'preserve')
            command[command.index('capture')] = 'controls'
            command.extend(['--runtime-manifest', '/missing-runtime', '--runtime-sha256', SHA])
            result = subprocess.run(command, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Fresh reference output required', result.stderr)
            self.assertEqual(output.read_bytes(), b'preserve')

    def test_closure_covers_new_and_reused_executed_sources_without_numerical_imports(self):
        selected = {path.name for path in driver.sources()}
        self.assertIn('ministral3_graph_check.py', selected)
        self.assertIn('ministral3_graph_reference.py', selected)
        self.assertIn('check_granite_reference.py', selected)
        self.assertIn('ministral3_convert_audit.py', selected)
        self.assertTrue(all(path.is_file() for path in driver.sources()))
        self.assertFalse(any(name.split('.')[0] in ('numpy', 'mlx', 'torch') for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
