"""Capture protocol/provenance/launch guards; no native model required."""
import copy
import argparse
import contextlib
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import m6_winner_capture as probe
from test_m6_context_scope_audit import records


def fixture():
    original = records()[0]
    events = original['worker']['phases'][0]['events']
    events[2].update(online_greedy=2, reference_logits_sha256='0'*64, online_logits_sha256='1'*64)
    ids = [7]*4096
    capture = dict(event='capture', target_index=1, prefix_tokens=3969,
        prefix_sha256=probe.base.shared.history_sha(ids[:3969]), vocabulary_values=151936,
        reference_file='reference-target0001.fp32le', online_file='online-target0001.fp32le',
        reference_logits_sha256=events[2].get('reference_logits_sha256'),
        online_logits_sha256=events[2].get('online_logits_sha256'))
    return original, ids, capture


class CaptureGuards(unittest.TestCase):
    def test_select_only_original_sorted_unique_changed_indices(self):
        original, _, _ = fixture()
        self.assertEqual(probe.selected_indices(original, None), [1])
        self.assertEqual(probe.selected_indices(original, [1]), [1])
        for indices in ([], [0], [1,1], [1,0], [True], list(range(9))):
            with self.subTest(indices=indices), self.assertRaises(ValueError): probe.selected_indices(original, indices)

    def test_exact_original_events_hashes_and_prefix_required(self):
        original, ids, capture = fixture()
        events = [capture, *original['worker']['phases'][0]['events']]
        self.assertEqual(probe.validate_events(events, original, [1], ids), [capture])
        mutations = [lambda e: e[0].update(reference_file='../escape'),
                     lambda e: e[0].update(prefix_sha256='wrong'),
                     lambda e: e[0].update(reference_logits_sha256='wrong'),
                     lambda e: e[0].update(extra=True),
                     lambda e: e[0].update(target_index=True),
                     lambda e: e.pop(),
                     lambda e: e[2].update(online_nll=99)]
        for mutate in mutations:
            changed = copy.deepcopy(events); mutate(changed)
            with self.assertRaises(ValueError): probe.validate_events(changed, original, [1], ids)

    def test_reviewed_build_sha_rejected_before_source_verification(self):
        with patch.object(probe.audit, 'read_json', return_value=({}, 'actual')), \
             patch.object(probe.build, 'verify_record') as verify, \
             self.assertRaisesRegex(ValueError, 'reviewed diagnostic build SHA'):
            probe.admission(Path('ignored'), 'different')
        verify.assert_not_called()

    def test_original_sha_and_replayed_consumed_bytes_required(self):
        path = probe.ROOT/'artifacts/anchor-synthetic.json'
        with patch.object(probe.audit, 'read_json', return_value=({}, 'actual')), \
             patch.object(probe.audit, 'audit') as replay, \
             self.assertRaisesRegex(ValueError, 'original quality record SHA'):
            probe.anchor_admission(path, 'different')
        replay.assert_not_called()
        with patch.object(probe.audit, 'read_json', return_value=({}, 'actual')), \
             patch.object(probe.audit, 'audit', return_value={'bindings': {'artifacts/anchor-synthetic.json':'other'}}), \
             self.assertRaisesRegex(ValueError, 'parsed versus replayed original SHA'):
            probe.anchor_admission(path, 'actual')

    def test_file_inventory_rejects_extras_before_analysis(self):
        with tempfile.TemporaryDirectory(dir=probe.ROOT/'artifacts') as temporary:
            directory=Path(temporary); (directory/'unexpected').write_bytes(b'')
            with patch.object(probe.analysis, 'analyze') as analyze, self.assertRaisesRegex(ValueError, 'inventory'):
                probe.verify_files(directory, [], {})
            analyze.assert_not_called()

    def test_snapshot_original_tools_and_binary_are_all_rechecked(self):
        root = probe.ROOT
        snapshot = root/'artifacts/synthetic-capture-source'
        binary = root/'artifacts/synthetic-capture-binary'
        original = {'Cargo.toml':'original'}
        expected = {**original, 'examples/metal_context_capture.rs':'asset'}
        record = dict(format='toklet-m6-winner-capture-build-v1', passed=True,
            snapshot=str(snapshot.relative_to(root)), root_sources=original,
            snapshot_sources=expected, tools_and_contract={},
            binary=dict(path=str(binary.relative_to(root)), sha256='binary'))
        for mutation in (None, 'root', 'snapshot', 'tools', 'binary'):
            altered = copy.deepcopy(record)
            if mutation == 'root': altered['root_sources']['Cargo.toml'] = 'changed'
            if mutation == 'snapshot': altered['snapshot_sources']['Cargo.toml'] = 'changed'
            if mutation == 'tools': altered['tools_and_contract'] = {'unexpected':'changed'}
            if mutation == 'binary': altered['binary']['sha256'] = 'changed'
            with self.subTest(mutation=mutation), \
                 patch.object(probe.build, 'baseline'), \
                 patch.object(probe.build, 'inventory', side_effect=lambda p: original if p == root else expected), \
                 patch.object(probe.build, 'expected_snapshot', return_value=expected), \
                 patch.object(probe.build, 'tool_paths', return_value=[]), \
                 patch.object(probe.build, 'digest', return_value='binary'):
                if mutation is None:
                    self.assertEqual(probe.build.verify_record(altered)[binary], 'binary')
                else:
                    with self.assertRaises(ValueError): probe.build.verify_record(altered)

    def test_orchestration_analyzes_after_exit_and_retains_failed_endpoints(self):
        for failure in (None, 'analysis', 'replacement', 'deleted', 'extra', 'symlink'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=probe.ROOT/'artifacts') as temporary:
                label = Path(temporary).name
                output = probe.ROOT/f'artifacts/{label}-result.json'
                directory = probe.ROOT/f'artifacts/{label}-captures'
                paths = [output, output.with_suffix('.raw.jsonl'), output.with_suffix('.stderr')]
                original, ids, capture = fixture()
                payload = b'\x00'*(151936*4)
                sha = hashlib.sha256(payload).hexdigest()
                for mode in probe.base.MODES:
                    original['worker']['phases'][0]['events'][2][f'{mode}_logits_sha256'] = sha
                    capture[f'{mode}_logits_sha256'] = sha
                events = [capture, *original['worker']['phases'][0]['events']]
                alive = [False]
                class FakeWorker:
                    def __init__(self, *args, **kwargs): self.record = dict(passed=False, cleanup_errors=[])
                    def __enter__(self):
                        directory.mkdir(); alive[0] = True
                        for mode in probe.base.MODES: (directory/capture[f'{mode}_file']).write_bytes(payload)
                        return self
                    def call(self, request, final): return copy.deepcopy(events)
                    def __exit__(self, *args):
                        alive[0] = False
                        self.record.update(passed=True, exit_code=0)
                def analyze_files(*args):
                    self.assertFalse(alive[0], 'Full-vector analysis must follow owned native exit')
                    if failure == 'analysis': raise ValueError('injected analysis failure')
                    return [dict(investigation_complete=False)], probe.capture_bindings(*args)
                def usage():
                    path = directory/capture['reference_file']
                    if failure == 'replacement': path.write_bytes(b'x'+payload[1:])
                    if failure in ('deleted', 'symlink'): path.unlink()
                    if failure == 'symlink': path.symlink_to(directory/capture['online_file'])
                    if failure == 'extra': (directory/'extra').write_bytes(b'')
                    return dict(lifetime_max_phys_footprint=1024)
                args = argparse.Namespace(output=str(output.relative_to(probe.ROOT)),
                    capture_dir=str(directory.relative_to(probe.ROOT)), build='artifacts/fake-build.json',
                    build_sha256='reviewed', anchor='artifacts/fake-anchor.json', anchor_sha256='original', indices=None)
                try:
                    with contextlib.ExitStack() as stack:
                        for target, value in ((probe, 'admission'), (probe, 'anchor_admission')):
                            result = (probe.ROOT/'artifacts/fake-binary', {}) if value == 'admission' else (original, {})
                            stack.enter_context(patch.object(target, value, return_value=result))
                        stack.enter_context(patch.object(probe.base, 'inputs', return_value=({}, {4096:ids}, Path('model'), None, {})))
                        stack.enter_context(patch.object(probe.base, 'capture_identities', return_value={}))
                        stack.enter_context(patch.object(probe.base.shared.metal_continue, 'endpoints', return_value={}))
                        stack.enter_context(patch.object(probe.base.shared.native_usage, 'usage', side_effect=usage))
                        stack.enter_context(patch.object(probe.build, 'verify_record', return_value={}))
                        stack.enter_context(patch.object(probe.audit, 'read_json', return_value=({}, 'reviewed')))
                        stack.enter_context(patch.object(probe, 'Worker', FakeWorker))
                        stack.enter_context(patch.object(probe, 'verify_files', side_effect=analyze_files))
                        if failure is None:
                            result = probe.run(args)
                            self.assertTrue(result['passed'])
                            self.assertTrue(result['capture_endpoints_unchanged'])
                            self.assertFalse(result['investigation_complete'])
                        else:
                            with self.assertRaisesRegex(ValueError, 'immutable attempted evidence'): probe.run(args)
                            result = json.loads(output.read_text())
                            self.assertFalse(result['passed'])
                            self.assertFalse(result['collection_completed'])
                            if failure != 'analysis': self.assertTrue(result['endpoint_errors'])
                    self.assertTrue(all(path.exists() for path in paths))
                    self.assertTrue(directory.exists())
                    self.assertFalse(alive[0])
                finally:
                    for path in paths: path.unlink(missing_ok=True)
                    if directory.exists(): shutil.rmtree(directory)

    def test_unsafe_paths_rejected_and_binary_source_closure_rechecked(self):
        for path in ('../elsewhere', '/tmp/absolute'):
            with self.assertRaises(ValueError): probe.build.relative(path)
        with patch.object(probe.build, 'baseline') as baseline, \
             self.assertRaisesRegex(ValueError, 'Completed capture build'):
            probe.build.verify_record({'format':'wrong','passed':True})
        baseline.assert_called_once()


if __name__ == '__main__':
    unittest.main()
