"""Offline ownership/raw-row tests; numerical conversion runs cold explicitly."""
from pathlib import Path
import hashlib
import json
import os
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import bf16_source as target

FIXTURE = Path(__file__).resolve().parents[2] / 'tests/fixtures/bf16-source-v1'


class Checks(unittest.TestCase):
    def setUp(self):
        self.deadline = time.monotonic() + 30
        self.stop = threading.Event()

    def source(self, path, raw):
        path.write_bytes(raw)
        return target.Source(path, len(raw), hashlib.sha256(raw).hexdigest(), deadline=self.deadline)

    def test_frozen_raw_matrix_scalar_gather_and_consumed_byte_chain(self):
        raw = (FIXTURE / 'source.bin').read_bytes()
        manifest = json.loads((FIXTURE / 'manifest.json').read_bytes())
        with target.Source(FIXTURE / 'source.bin', len(raw), manifest['assets']['source.bin']['sha256'], deadline=self.deadline) as source:
            rows = target.Rows(source, 4, 3, 8, self.deadline)
            self.assertEqual(rows.read_raw(1, 3), raw[14:26])
            self.assertEqual(rows.gather_raw([3, 0, 3, 1]), raw[26:32] + raw[8:14] + raw[26:32] + raw[14:20])
            scalar = target.Rows(source, 1, 1, 32, self.deadline)
            self.assertEqual(scalar.read_raw(0, 1), b'\x80\x3f')
            chain = hashlib.sha256(b'toklet-bf16-source-reads-v1\0')
            spans = [(14, 12), (26, 6), (8, 6), (26, 6), (14, 6), (32, 2)]
            for index, (offset, count) in enumerate(spans, 1):
                chain.update(struct.pack('<QQQ', index, offset, count))
                chain.update(hashlib.sha256(raw[offset:offset + count]).digest())
            self.assertEqual(source.audit()['read_chain_sha256'], chain.hexdigest())
            self.assertEqual(source.audit()['reads'], 6)
            self.assertEqual(source.audit()['read_bytes'], 38)

    def test_wrong_size_sha_and_nonregular_paths_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'source'; path.write_bytes(b'abcd')
            for size, sha in [(5, hashlib.sha256(b'abcd').hexdigest()), (4, '0' * 64), (True, '0' * 64)]:
                with self.assertRaises(ValueError): target.Source(path, size, sha, deadline=self.deadline)
            link = root / 'link'; link.symlink_to(path)
            fifo = root / 'fifo'; os.mkfifo(fifo)
            for path in (link, fifo, root):
                with self.assertRaises((OSError, ValueError)): target.Source(path, 4, '0' * 64, deadline=self.deadline)

    def test_bad_row_and_gather_requests_do_not_poison_or_read(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.source(Path(folder) / 'source', b'0123456789abcdef') as source:
                rows = target.Rows(source, 4, 2, 0, self.deadline)
                for start, end in [(True, 2), (-1, 1), (1, 1), (0, 5), (0, 1.)]:
                    with self.assertRaises(ValueError): rows.read_raw(start, end)
                for ids in [[], [True], [-1], [4], [0] * 129, (0,)]:
                    with self.assertRaises(ValueError): rows.gather_raw(ids)
                self.assertEqual(source.audit()['reads'], 0); self.assertFalse(source.poisoned)
                self.assertEqual(len(rows.gather_raw([0] * 128)), 512)

    def test_views_reject_odd_outside_and_overflow_extents(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.source(Path(folder) / 'source', b'01234567') as source:
                for rows, columns, offset in [(1, 1, 1), (5, 1, 0), (1, 12289, 0), (True, 1, 0), (1, 0, 0)]:
                    with self.assertRaises(ValueError): target.Rows(source, rows, columns, offset, self.deadline)
                self.assertFalse(source.poisoned)

    def test_path_replacement_and_restoration_cannot_resurrect_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'source'; saved = root / 'saved'
            source = self.source(path, b'abcd')
            path.rename(saved); path.write_bytes(b'abcd')
            with self.assertRaises(target.SourceChanged): source.read(0, 2, deadline=self.deadline)
            path.unlink(); saved.rename(path)
            with self.assertRaises(target.SourceChanged): source.read(0, 2, deadline=self.deadline)
            self.assertTrue(source.poisoned); self.assertEqual(source.audit()['reads'], 0)
            source.close(); source.close()

    def test_mutation_restore_and_truncation_poison_original_identity(self):
        for mode in ('restore', 'truncate'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'source'; source = self.source(path, b'abcd')
                with path.open('r+b') as file:
                    if mode == 'restore': file.write(b'WXYZ'); file.flush(); file.seek(0); file.write(b'abcd')
                    else: file.truncate(2)
                with self.assertRaises(target.SourceChanged): source.read(0, 2, deadline=self.deadline)
                self.assertTrue(source.poisoned); source.close()

    def test_short_read_and_io_failure_are_sticky_and_not_audited(self):
        for mode in ('short', 'io'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                source = self.source(Path(folder) / 'source', b'abcd')
                kwargs = dict(return_value=b'a') if mode == 'short' else dict(side_effect=OSError('synthetic IO'))
                with patch.object(target.os, 'pread', **kwargs):
                    with self.assertRaises(target.SourceChanged): source.read(0, 2, deadline=self.deadline)
                with self.assertRaises(target.SourceChanged): source.read(0, 2, deadline=self.deadline)
                self.assertEqual(source.audit()['reads'], 0); source.close()

    def test_mid_read_mutation_discards_bytes_before_chain_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source'; source = self.source(path, b'abcd'); original = os.pread
            def changing(fd, count, offset):
                raw = original(fd, count, offset)
                with path.open('r+b') as file: file.write(b'WXYZ')
                return raw
            with patch.object(target.os, 'pread', side_effect=changing):
                with self.assertRaises(target.SourceChanged): source.read(0, 2, deadline=self.deadline)
            self.assertEqual(source.audit()['reads'], 0); self.assertTrue(source.poisoned); source.close()

    def test_mutation_during_initial_authentication_closes_original_descriptor(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source'; path.write_bytes(b'abcd')
            original = os.pread; descriptors = []
            def changing(fd, count, offset):
                raw = original(fd, count, offset)
                if not descriptors:
                    descriptors.append(fd)
                    with path.open('r+b') as file: file.write(b'WXYZ')
                return raw
            with patch.object(target.os, 'pread', side_effect=changing):
                with self.assertRaises(target.SourceChanged):
                    target.Source(path, 4, hashlib.sha256(b'abcd').hexdigest(), deadline=self.deadline)
            self.assertEqual(len(descriptors), 1)
            with self.assertRaises(OSError): os.fstat(descriptors[0])

    def test_cancellation_after_pread_leaves_audit_prefix_and_source_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source(Path(folder) / 'source', b'abcd'); original = os.pread
            def cancelled(fd, count, offset):
                raw = original(fd, count, offset); self.stop.set(); return raw
            before = source.audit()
            with patch.object(target.os, 'pread', side_effect=cancelled):
                with self.assertRaises(InterruptedError):
                    source.read(0, 2, deadline=self.deadline, stop=self.stop)
            self.assertEqual(source.audit(), before); self.assertFalse(source.poisoned)
            self.stop.clear()
            self.assertEqual(source.read(0, 2, deadline=self.deadline, stop=self.stop), b'ab')
            self.assertEqual(source.audit()['reads'], 1); source.close()

    def test_cancellation_deadline_and_closed_owner_reject_without_source_poison(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source(Path(folder) / 'source', b'abcd'); self.stop.set()
            with self.assertRaises(InterruptedError): source.read(0, 2, deadline=self.deadline, stop=self.stop)
            with self.assertRaises(TimeoutError): source.read(0, 2, deadline=time.monotonic() - 1)
            self.assertFalse(source.poisoned); self.assertEqual(source.audit()['reads'], 0)
            source.close(); source.close()
            with self.assertRaisesRegex(ValueError, 'closed'): source.read(0, 2, deadline=self.deadline)

    def test_cleanup_mutation_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source'; source = self.source(path, b'abcd')
            with self.assertRaisesRegex(RuntimeError, 'original') as error:
                with source:
                    path.write_bytes(b'WXYZ')
                    raise RuntimeError('original')
            self.assertTrue(source.closed); self.assertTrue(source.poisoned)
            self.assertIn('BF16 source cleanup', error.exception.__notes__[0])

    def test_official_adapter_rejects_unbound_source_even_with_real_layout(self):
        root = FIXTURE.parents[2]
        layout = (root / 'tests/fixtures/gemma-unquantized-header-v1/layout.json').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            with self.source(Path(folder) / 'source', b'abcd') as source:
                with self.assertRaisesRegex(ValueError, 'official source/layout'):
                    target.official_rows(source, 'model.language_model.norm.weight', layout, deadline=self.deadline)


if __name__ == '__main__':
    unittest.main()
