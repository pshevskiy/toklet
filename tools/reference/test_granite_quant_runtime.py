"""Portable streaming-hash identity and failure controls, without MLX imports."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import granite_quant_runtime as runtime


class RuntimeFiles(unittest.TestCase):
    def pin(self, raw):
        return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def test_cold_runtime_import_keeps_numerical_packages_unloaded(self):
        self.assertNotIn('numpy', sys.modules)
        self.assertNotIn('mlx.core', sys.modules)

    def test_large_file_hashes_exactly_with_bounded_reads(self):
        raw = bytes(range(256)) * (3 * runtime.HASH_CHUNK // 256) + b'bounded-tail'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.bin'
            path.write_bytes(raw)
            requested = []
            read = os.read

            def observe(fd, size):
                requested.append(size)
                return read(fd, size)

            with patch.object(runtime.os, 'read', side_effect=observe):
                result = runtime.verify_file(path, self.pin(raw))
            self.assertEqual(result['bytes'], len(raw))
            self.assertEqual(result['sha256'], self.pin(raw)['sha256'])
            self.assertEqual(result['maximum_read_bytes'], runtime.HASH_CHUNK)
            self.assertLessEqual(max(requested), runtime.HASH_CHUNK)
            self.assertGreater(len(requested), 3)

    def test_empty_regular_file_is_still_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'empty'
            path.write_bytes(b'')
            result = runtime.verify_file(path, self.pin(b''))
            self.assertEqual(result['bytes'], 0)
            self.assertEqual(result['maximum_read_bytes'], 0)

    def test_pin_types_fields_hash_and_extents_are_strict(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.bin'
            path.write_bytes(b'abc')
            pin = self.pin(b'abc')
            for bad in [dict(pin, bytes=True), dict(pin, bytes=-1),
                        dict(pin, bytes=512 * 1024**2 + 1), dict(pin, extra=1),
                        dict(pin, sha256='A' * 64), dict(pin, sha256=None),
                        dict(pin, bytes=2), dict(pin, bytes=4), dict(pin, sha256='0' * 64)]:
                with self.assertRaises(ValueError):
                    runtime.verify_file(path, bad)

    def test_symlinks_and_directories_do_not_become_runtime_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.bin'
            path.write_bytes(b'abc')
            link = Path(folder) / 'link'
            link.symlink_to(path)
            with self.assertRaises((OSError, ValueError)):
                runtime.verify_file(link, self.pin(b'abc'))
            with self.assertRaises(ValueError):
                runtime.verify_file(Path(folder), self.pin(b''))

    def test_already_hashed_bytes_cannot_mutate_behind_the_descriptor(self):
        raw = b'a' * (runtime.HASH_CHUNK + 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.bin'
            path.write_bytes(raw)
            read = os.read
            changed = False

            def mutate(fd, size):
                nonlocal changed
                result = read(fd, size)
                if not changed:
                    changed = True
                    before = path.stat()
                    with path.open('r+b') as file:
                        file.write(b'b')
                    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
                return result

            with patch.object(runtime.os, 'read', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'changed during hashing'):
                    runtime.verify_file(path, self.pin(raw))

    def test_same_bytes_at_a_replaced_path_do_not_pass_identity(self):
        raw = b'a' * (runtime.HASH_CHUNK + 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.bin'
            path.write_bytes(raw)
            read = os.read
            replaced = False

            def replace(fd, size):
                nonlocal replaced
                result = read(fd, size)
                if not replaced:
                    replaced = True
                    path.rename(Path(folder) / 'old-inode')
                    path.write_bytes(raw)
                return result

            with patch.object(runtime.os, 'read', side_effect=replace):
                with self.assertRaisesRegex(ValueError, 'changed during hashing'):
                    runtime.verify_file(path, self.pin(raw))

    def test_growth_is_rejected_and_failed_reads_close_the_descriptor(self):
        raw = b'a' * (runtime.HASH_CHUNK + 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.bin'
            path.write_bytes(raw)
            read, opened = os.read, os.open
            descriptors = []
            grown = False

            def remember(*args, **kwargs):
                fd = opened(*args, **kwargs)
                descriptors.append(fd)
                return fd

            def grow(fd, size):
                nonlocal grown
                result = read(fd, size)
                if not grown:
                    grown = True
                    with path.open('ab') as file:
                        file.write(b'x')
                return result

            with patch.object(runtime.os, 'open', side_effect=remember), \
                    patch.object(runtime.os, 'read', side_effect=grow):
                with self.assertRaisesRegex(ValueError, 'grew past'):
                    runtime.verify_file(path, self.pin(raw))
            self.assertEqual(len(descriptors), 1)
            with self.assertRaises(OSError):
                os.fstat(descriptors[0])


if __name__ == '__main__':
    unittest.main()
