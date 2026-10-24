"""Stdlib admission tests; no MLX execution or numerical qualification."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import gated_delta_upstream as upstream


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class UpstreamAdmission(unittest.TestCase):
    def test_capture_requires_regular_bounded_authenticated_buffer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.py'
            source.write_bytes(b'not executed\n')
            self.assertEqual(upstream.captured(source, 128, sha(source.read_bytes())), b'not executed\n')
            for maximum, expected in [(1, sha(source.read_bytes())), (128, '0' * 64)]:
                with self.assertRaisesRegex(ValueError, 'identity'):
                    upstream.captured(source, maximum, expected)
            alias = root / 'alias.py'
            alias.symlink_to(source)
            with self.assertRaisesRegex(ValueError, 'regular'):
                upstream.captured(alias, 128, sha(source.read_bytes()))

    def test_output_and_device_reject_before_input_or_runtime_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'output'
            output.mkdir()
            with mock.patch.object(upstream.fixture, 'verify') as fixture, mock.patch.object(upstream, 'admit_runtime') as runtime:
                with self.assertRaisesRegex(ValueError, 'explicit CPU or GPU'):
                    upstream.compare(root, '0' * 64, '1' * 64, root / 'source', 'automatic', output, root / 'runtime', '2' * 64)
                (output / 'existing').write_bytes(b'owned')
                with self.assertRaisesRegex(ValueError, 'Empty existing'):
                    upstream.compare(root, '0' * 64, '1' * 64, root / 'source', 'cpu', output, root / 'runtime', '2' * 64)
                alias = root / 'alias'
                alias.symlink_to(output, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'Empty existing'):
                    upstream.compare(root, '0' * 64, '1' * 64, root / 'source', 'cpu', alias, root / 'runtime', '2' * 64)
                fixture.assert_not_called()
                runtime.assert_not_called()
                self.assertEqual((output / 'existing').read_bytes(), b'owned')

    def test_consumed_binary_and_source_admitted_before_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'output'
            output.mkdir()
            binary = b'owned bytes'
            (root / 'arrays.bin').write_bytes(binary)
            source = root / 'source'
            source.write_bytes(b'not the pinned upstream source')
            with mock.patch.object(upstream.fixture, 'verify', return_value={}) as verify, mock.patch.object(upstream, 'admit_runtime') as runtime:
                with self.assertRaisesRegex(ValueError, 'identity'):
                    upstream.compare(root, '0' * 64, '1' * 64, source, 'cpu', output, root / 'runtime', '2' * 64)
                with self.assertRaisesRegex(ValueError, 'identity'):
                    upstream.compare(root, '0' * 64, sha(binary), source, 'cpu', output, root / 'runtime', '2' * 64)
                self.assertEqual(verify.call_count, 2)
                runtime.assert_not_called()
                self.assertEqual(list(output.iterdir()), [])

    def test_runtime_manifest_identity_and_exact_inventory_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / 'runtime.json'
            value = dict(format='toklet-gated-delta-runtime-v1', versions=upstream.VERSIONS, files={})
            raw = json.dumps(value).encode()
            runtime.write_bytes(raw)
            with mock.patch.object(upstream, 'runtime_paths', return_value=[root / 'required']):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    upstream.admit_runtime(runtime, '0' * 64)
                with self.assertRaisesRegex(ValueError, 'exact|Exact'):
                    upstream.admit_runtime(runtime, sha(raw))
            raw = b'{"format":1,"format":2}'
            runtime.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                upstream.admit_runtime(runtime, sha(raw))

    def test_already_imported_runtime_rejects_without_importing_or_hashing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / 'runtime.json'
            value = dict(format='toklet-gated-delta-runtime-v1', versions=upstream.VERSIONS, files={})
            raw = json.dumps(value).encode()
            runtime.write_bytes(raw)
            with mock.patch.object(upstream, 'runtime_paths', return_value=[]), mock.patch.dict(sys.modules, {'mlx.__test_marker': object()}):
                with self.assertRaisesRegex(ValueError, 'already imported'):
                    upstream.admit_runtime(runtime, sha(raw))

    def test_runtime_file_length_hash_and_type_are_checked(self):
        # Isolate admission bookkeeping only. No numerical module is imported;
        # these synthetic bytes do not represent a usable reference runtime.
        visible = {name: value for name, value in sys.modules.items()
            if not (name == 'numpy' or name.startswith('numpy.') or name == 'mlx' or name.startswith('mlx.'))}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'runtime.bin'
            source.write_bytes(b'synthetic runtime bytes')
            runtime = root / 'runtime.json'
            pin = dict(bytes=source.stat().st_size, sha256=sha(source.read_bytes()))
            def write(current):
                raw = json.dumps(dict(format='toklet-gated-delta-runtime-v1',
                    versions=upstream.VERSIONS, files={str(source): current})).encode()
                runtime.write_bytes(raw)
                return sha(raw)
            with mock.patch.dict(sys.modules, visible, clear=True), mock.patch.object(upstream, 'runtime_paths', return_value=[source]), mock.patch.object(sys, 'pycache_prefix', str(root / 'cache')), mock.patch.object(upstream, 'selected_imports', return_value={}):
                admitted = upstream.admit_runtime(runtime, write(pin))
                self.assertEqual(admitted['files'], 1)
                for changed in [dict(pin, bytes=pin['bytes'] - 1), dict(pin, bytes=pin['bytes'] + 1),
                                dict(pin, bytes=True), dict(pin, sha256='0' * 64)]:
                    with self.assertRaises(ValueError):
                        upstream.admit_runtime(runtime, write(changed))
                expected = write(pin)
                source.write_bytes(b'x' * pin['bytes'])
                with self.assertRaisesRegex(ValueError, 'bytes differ'):
                    upstream.admit_runtime(runtime, expected)

    def test_shadowed_numpy_source_or_mlx_namespace_rejects_without_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, spec in [('numpy', None), ('numpy', mock.Mock(submodule_search_locations=[str(root / 'shadow')]))]:
                with mock.patch.object(upstream, 'find_spec', return_value=spec):
                    with self.assertRaisesRegex(ValueError, 'resolution|search path'):
                        upstream.selected_imports(root, [])
            numpy = mock.Mock(origin=str(root / 'numpy' / '__init__.py'),
                submodule_search_locations=[str(root / 'numpy')],
                loader=upstream.SourceFileLoader('numpy', str(root / 'numpy' / '__init__.py')))
            mlx = mock.Mock(origin=None, submodule_search_locations=[str(root / 'mlx'), str(root / 'shadow')])
            with mock.patch.object(upstream, 'find_spec', side_effect=[numpy, mlx]):
                with self.assertRaisesRegex(ValueError, 'search path'):
                    upstream.selected_imports(root, [root / 'numpy' / '__init__.py'])


if __name__ == '__main__':
    unittest.main()
