"""Corruption/replay controls using synthetic tokens independently of the corpus."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import prepare_qwen3_context8192 as subject


class Inputs8192(unittest.TestCase):
    def values(self):
        original = [9707, 11, 1879, 0] * 128
        complete = list(range(22289))
        a = original * 8
        return original, complete, complete[:10001], {
            'A.u32le': a, 'B.u32le': a[:-1] + [1], 'Q.u32le': complete[10001:14097]}

    def test_exact_new_span_preserves_both_old_regions_and_independent_replay(self):
        arguments = self.values()
        first = subject.choose(*arguments)
        self.assertEqual(first, subject.replay_selection(*arguments))
        self.assertEqual(first['Q.u32le'], list(range(14097, 22289)))
        self.assertEqual(first['A.u32le'][:4096], arguments[-1]['A.u32le'])
        self.assertEqual(first['B.u32le'][:-1], first['A.u32le'][:-1])
        self.assertEqual(first['B.u32le'][-1], 1)
        self.assertEqual(first['Q.u32le'][7168:], list(range(21265, 22289)))
        self.assertEqual(len(first['Q.u32le'][7168:]), 1024)

    def test_old_span_changes_short_corpus_nonintegers_or_structural_changes_reject(self):
        for change in ('original', 'old10001', 'old4096', 'old_a', 'old_b', 'short', 'boolean', 'vocabulary'):
            original, complete, previous, old = self.values()
            if change == 'original': original[0] += 1
            if change == 'old10001': complete[10000] += 1
            if change == 'old4096': complete[14096] += 1
            if change == 'old_a': old['A.u32le'][4095] += 1
            if change == 'old_b': old['B.u32le'][4095] = 0
            if change == 'short': complete.pop()
            if change == 'boolean': complete[-1] = True
            if change == 'vocabulary': complete[-1] = 151936
            with self.subTest(change=change), self.assertRaises(ValueError):
                subject.choose(original, complete, previous, old)

    def admission(self):
        # Synthetic retained-receipt pin is a unit fixture, never a proposed real identity.
        return dict(format='toklet-qwen3-context8192-parent-admission-v1', retention_approved=True,
            retained_schedule='ParallelOutputsExperimental', parent_receipts={
                'f2_result': dict(path='/synthetic/f2.json', bytes=1, sha256=subject.profile.F2_RESULT),
                'retained_parallel_result': dict(path='/synthetic/retained.json', bytes=1, sha256='a' * 64)})

    def test_retention_is_explicit_and_cannot_be_inferred_from_placeholder_paths(self):
        good = self.admission()
        subject.validate_admission(good)
        for field, value in [('retention_approved', False), ('retention_approved', 1),
                             ('retained_schedule', 'Serial'), ('format', 'old-format')]:
            with self.assertRaises(ValueError):
                subject.validate_admission(dict(good, **{field: value}))
        for field, value in [('path', 'relative.json'), ('bytes', True), ('bytes', 16 * 1024**2 + 1),
                             ('sha256', 'pending'), ('sha256', 'A' * 64)]:
            bad = copy.deepcopy(good)
            bad['parent_receipts']['retained_parallel_result'][field] = value
            with self.assertRaises(ValueError):
                subject.validate_admission(bad)

    def fake_derive(self, replay, report, path, digest):
        report.update(admission_sha256=digest, parent_receipts={
            key: {name: value[name] for name in ('bytes', 'sha256')}
            for key, value in self.admission()['parent_receipts'].items()})
        values = (subject.replay_selection if replay else subject.choose)(*self.values())
        return {name: struct.pack('<8192I', *ids) for name, ids in values.items()}

    def test_publication_replay_exact_bytes_and_changed_inputs_reject(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(subject, 'derive', side_effect=self.fake_derive):
            folder = Path(temporary) / 'inputs'
            first = subject.run(folder, '/synthetic/admission', 'c' * 64)
            self.assertTrue(first['passed'], first)
            replay = subject.run(folder, '/synthetic/admission', 'c' * 64, True, first['manifest_sha256'])
            self.assertTrue(replay['passed'], replay)
            self.assertTrue(all(parent['endpoint_checked'] for parent in replay['parents']))
            self.assertFalse(subject.run(folder, '/synthetic/admission', 'c' * 64)['passed'])
            raw = bytearray((folder / 'Q.u32le').read_bytes())
            raw[0] ^= 1
            (folder / 'Q.u32le').write_bytes(raw)
            self.assertFalse(subject.run(folder, '/synthetic/admission', 'c' * 64, True, first['manifest_sha256'])['passed'])

    def test_partial_files_and_manifest_remain_after_second_write_failure(self):
        ordinary_open = Path.open
        def fail_second(path, *args, **kwargs):
            if path.name == 'B.u32le' and args and args[0] == 'xb':
                raise OSError('injected publication failure')
            return ordinary_open(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, patch.object(subject, 'derive', side_effect=self.fake_derive):
            folder = Path(temporary) / 'inputs'
            with patch.object(Path, 'open', fail_second):
                result = subject.run(folder, '/synthetic/admission', 'c' * 64)
            self.assertFalse(result['passed'])
            self.assertIn('injected', result['error'])
            partial = json.loads((folder / 'manifest.json').read_bytes())
            self.assertFalse(partial['complete'])
            self.assertEqual(set(partial['files']), {'A.u32le'})
            self.assertEqual((folder / 'A.u32le').stat().st_size, 32768)
            with self.assertRaises(ValueError):
                subject.profile.validate(partial)


if __name__ == '__main__':
    unittest.main()
