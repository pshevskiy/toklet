"""Offline owner, bounded diagnostic, source-FD and publication controls."""
from pathlib import Path
import hashlib
import io
import contextlib
import os
import resource
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import m8_gemma4_unquantized_checkpoint as acquire


class Checks(unittest.TestCase):
    def setUp(self):
        self.stop = threading.Event()
        acquire.transport.STOP.clear()

    def factory(self, code):
        def make(header, body, _):
            text = f'import os,sys,time;header={header};body={body};' + code
            return [sys.executable, '-I', '-S', '-B', '-c', text]
        return make

    def owned(self, code, size=16, timeout=10, journal_exists=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'body'
            if journal_exists:
                (root / 'download.child.json').write_bytes(b'existing')
            original = resource.getrlimit(resource.RLIMIT_FSIZE)
            with path.open('x+b') as file:
                result = acquire.run_owned(self.factory(code), file, root, size,
                                           time.monotonic() + timeout, self.stop,
                                           dict(PATH='/usr/bin:/bin'))
                body = path.read_bytes()
            logs = {name: (root / ('download.' + name)).read_bytes()
                    for name in ('headers', 'stdout', 'stderr')}
            self.assertEqual(resource.getrlimit(resource.RLIMIT_FSIZE), original)
            if result['pid'] is not None:
                self.assertFalse(acquire.transport.package.group_exists(result['pid']))
                with self.assertRaises(ProcessLookupError): os.kill(result['pid'], 0)
                if journal_exists:
                    self.assertEqual((root / 'download.child.json').read_bytes(), b'existing')
                else:
                    record = acquire.layout.document((root / 'download.child.json').read_bytes())
                    self.assertEqual(record['pid'], result['pid'])
                    self.assertEqual(record['command'], result['command'])
            return result, body, logs

    def test_real_child_success_has_bounded_three_streams_and_reaps(self):
        result, body, logs = self.owned("os.write(header,b'HTTP/1.1 200 OK\\r\\nContent-Length: 4\\r\\n\\r\\n');os.write(body,b'abcd');os.write(1,b'200');os.write(2,b'diagnostic')")
        self.assertIsNone(result['primary_error']); self.assertEqual(result['cleanup_errors'], [])
        self.assertEqual(result['exit_code'], 0); self.assertEqual(body, b'abcd')
        self.assertEqual(logs['stdout'], b'200'); self.assertEqual(logs['stderr'], b'diagnostic')
        acquire.full_response(logs['headers'], logs['stdout'], 4)

    def test_each_diagnostic_pipe_has_a_real_disk_cap(self):
        for channel in ('header', '1', '2'):
            with self.subTest(channel=channel), patch.object(acquire, 'DIAGNOSTIC_CAP', 1024):
                result, _, logs = self.owned(f"os.write({channel},b'x'*4096);time.sleep(30)")
                self.assertIn('diagnostic cap exceeded', result['primary_error'])
                self.assertTrue(result['diagnostic_overflow'])
                self.assertTrue(all(len(raw) <= 1024 for raw in logs.values()))

    def test_body_file_size_limit_is_inherited_and_parent_restored(self):
        result, body, _ = self.owned("os.write(body,b'abcd');os.write(body,b'excess')", size=4)
        self.assertIsNotNone(result['primary_error']); self.assertLessEqual(len(body), 4)

    def test_deadline_after_launch_closes_and_reaps_owned_group(self):
        result, _, _ = self.owned("os.write(2,b'launched');time.sleep(30)", timeout=.2)
        self.assertIsNotNone(result['pid']); self.assertIn('deadline', result['primary_error'])

    def test_cancellation_before_launch_preserves_empty_logs(self):
        self.stop.set()
        result, _, logs = self.owned("raise AssertionError('must not run')")
        self.assertIsNone(result['pid']); self.assertIn('cancelled', result['primary_error'])
        self.assertTrue(all(not raw for raw in logs.values()))

    def test_journal_write_failure_still_closes_and_reaps_child(self):
        result, _, _ = self.owned('time.sleep(30)', journal_exists=True)
        self.assertIsNotNone(result['pid'])
        self.assertIn('FileExistsError', result['primary_error'])
        self.assertEqual(result['cleanup_errors'], [])

    def test_full_response_final_content_length_and_redirects(self):
        for ending in [b'\r\n', b'\n']:
            prefix = ending.join([b'HTTP/1.1 302 Found', b'Content-Length: 19', b'', b''])
            good = ending.join([b'HTTP/1.1 200 OK', b'Content-Length: 4', b'', b''])
            acquire.full_response(prefix + good, b'200', 4)
            for raw in [good.replace(b'Length: 4', b'Length: 5'),
                        good.replace(b'200 OK', b'206 Partial'),
                        good.replace(b'Length: 4', b'Length: 4' + ending + b'Content-Length: 4'),
                        good.replace(b'Length: 4', b'Length: 4' + ending + b'Content-Encoding: gzip')]:
                with self.assertRaises(ValueError): acquire.full_response(raw, b'200', 4)

    def test_same_descriptor_hash_and_prefix_do_not_expand_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'body'
            with path.open('x+b') as file:
                file.write(b'abcdef'); file.flush()
                value, prefix, identity = acquire.hash_owned(file, path, 6, 3, time.monotonic() + 10, self.stop)
                self.assertEqual(value, dict(bytes=6, sha256=hashlib.sha256(b'abcdef').hexdigest()))
                self.assertEqual(prefix, b'abc'); self.assertEqual(identity, acquire.identity(path.stat()))
                with self.assertRaisesRegex(ValueError, 'complete body size'):
                    acquire.hash_owned(file, path, 7, 3, time.monotonic() + 10, self.stop)
                with self.assertRaisesRegex(ValueError, 'bounded integer'):
                    acquire.hash_owned(file, path, 6, acquire.layout.MAX_HEADER + 9, time.monotonic() + 10, self.stop)

    def test_path_replacement_and_mutation_during_hash_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'body'
            with path.open('x+b') as file:
                file.write(b'abcd'); file.flush(); path.unlink(); path.write_bytes(b'abcd')
                with self.assertRaisesRegex(ValueError, 'descriptor/path'):
                    acquire.hash_owned(file, path, 4, 1, time.monotonic() + 10, self.stop)
            path.unlink()
            with path.open('x+b') as file:
                file.write(b'abcd'); file.flush()
                original = acquire.deadline_check; calls = []
                def changing(deadline, stop):
                    original(deadline, stop); calls.append(None)
                    if len(calls) == 1:
                        with path.open('r+b') as writer: writer.write(b'WXYZ')
                with patch.object(acquire, 'deadline_check', side_effect=changing):
                    with self.assertRaisesRegex(ValueError, 'changed during hashing'):
                        acquire.hash_owned(file, path, 4, 1, time.monotonic() + 10, self.stop)

    def test_hash_deadline_and_cancellation_apply_after_download(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'body'
            with path.open('x+b') as file:
                file.write(b'abcd'); file.flush()
                with self.assertRaises(TimeoutError):
                    acquire.hash_owned(file, path, 4, 1, time.monotonic() - 1, self.stop)
                self.stop.set()
                with self.assertRaises(InterruptedError):
                    acquire.hash_owned(file, path, 4, 1, time.monotonic() + 10, self.stop)

    def test_existing_destination_rejects_before_any_acquisition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root / 'existing'; target.mkdir()
            with patch.object(acquire, 'DESTINATION', target), patch.object(acquire, 'run_owned') as child:
                with self.assertRaisesRegex(ValueError, 'existing output'): acquire.main(root / 'out')
                child.assert_not_called()
            self.assertFalse((root / 'out').exists())

    def test_outside_output_rejects_before_inputs_or_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(acquire, 'DESTINATION', root / 'absent'), patch.object(acquire, 'run_owned') as child:
                with self.assertRaisesRegex(ValueError, 'inside repository'):
                    acquire.main(root / 'output')
                child.assert_not_called()
            self.assertFalse((root / 'output').exists())

    def test_exclusive_publication_never_replaces_existing_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'staging'; source.mkdir()
            destination = root / 'existing'; destination.mkdir(); (destination / 'sentinel').write_bytes(b'keep')
            acquire.transport.STOP.clear()
            with patch.object(acquire.transport, 'DEADLINE', time.monotonic() + 10):
                with self.assertRaises(OSError): acquire.transport.publish(source, destination)
            self.assertEqual((destination / 'sentinel').read_bytes(), b'keep')
            self.assertTrue(source.is_dir())

    def test_source_endpoint_rejection_precedes_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); staging = root / 'staging'; staging.mkdir()
            body = staging / 'model.safetensors'; body.write_bytes(b'abcd')
            pin = root / 'source'; pin.write_bytes(b'original')
            authenticated = acquire.identity(body.stat())
            directory = (staging.stat().st_dev, staging.stat().st_ino)
            pins = {'source': hashlib.sha256(b'original').hexdigest()}; pin.write_bytes(b'changed!')
            result = dict(published=False)
            with patch.object(acquire, 'ROOT', root), patch.object(acquire.transport, 'publish') as publish:
                with self.assertRaisesRegex(ValueError, 'source endpoint'):
                    acquire.publish_authenticated(staging, root / 'destination', authenticated,
                                                  directory, b'{}', pins,
                                                  time.monotonic() + 10, self.stop, result)
                publish.assert_not_called()
            self.assertFalse(result['published']); self.assertFalse((staging / 'config.json').exists())

    def test_successful_authenticated_publication_keeps_exact_inode_bytes_and_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); staging = root / 'staging'; staging.mkdir()
            body = staging / 'model.safetensors'; body.write_bytes(b'abcd')
            authenticated = acquire.identity(body.stat())
            directory = (staging.stat().st_dev, staging.stat().st_ino)
            destination = root / 'destination'; result = dict(published=False)
            deadline = time.monotonic() + 10
            with patch.object(acquire.transport, 'DEADLINE', deadline):
                acquire.publish_authenticated(staging, destination, authenticated,
                                              directory, b'{"source":"pinned"}', {},
                                              deadline, self.stop, result)
            self.assertTrue(result['published']); self.assertFalse(staging.exists())
            self.assertEqual((destination / 'model.safetensors').read_bytes(), b'abcd')
            self.assertEqual(acquire.identity((destination / 'model.safetensors').stat()), authenticated)
            self.assertEqual((destination / 'config.json').read_bytes(), b'{"source":"pinned"}')

    def test_post_publication_failure_cannot_claim_nothing_published(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); staging = root / 'staging'; staging.mkdir()
            body = staging / 'model.safetensors'; body.write_bytes(b'abcd')
            authenticated = acquire.identity(body.stat())
            directory = (staging.stat().st_dev, staging.stat().st_ino)
            destination = root / 'destination'; result = dict(published=False)
            def replaced(source, target):
                source.rename(target)
                (target / 'model.safetensors').write_bytes(b'changed!')
            with patch.object(acquire.transport, 'publish', side_effect=replaced):
                with self.assertRaisesRegex(ValueError, 'published file identity'):
                    acquire.publish_authenticated(staging, destination, authenticated,
                                                  directory, b'{}', {},
                                                  time.monotonic() + 10, self.stop, result)
            self.assertTrue(result['published']); self.assertTrue(destination.is_dir())

    def test_receipt_failure_preserves_publication_and_original_cause(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'download.json').write_bytes(b'keep')
            result = dict(passed=True, published=True, primary_error='original cause', published_receipt_error=None)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr): acquire.write_receipt(root, result)
            self.assertFalse(result['passed']); self.assertTrue(result['published'])
            self.assertEqual(result['primary_error'], 'original cause')
            self.assertIsNotNone(result['published_receipt_error'])
            self.assertIn('"published": true', stderr.getvalue())
            self.assertEqual((root / 'download.json').read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
