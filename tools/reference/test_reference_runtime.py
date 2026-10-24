"""Stdlib admission checks; synthetic files are not numerical qualification."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import reference_runtime as runtime


class ReferenceRuntime(unittest.TestCase):
    def test_buffer_bound_hash_regular_file_and_json_uniqueness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'file'
            raw = b'owned\0bytes'
            path.write_bytes(raw)
            pin = hashlib.sha256(raw).hexdigest()
            self.assertEqual(runtime.captured(path, len(raw), pin), raw)
            for maximum, expected in [(len(raw) - 1, pin), (len(raw), '0' * 64)]:
                with self.assertRaises(ValueError):
                    runtime.captured(path, maximum, expected)
            alias = root / 'alias'
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                runtime.captured(alias, len(raw), pin)
            with self.assertRaises(ValueError):
                json.loads('{"x":1,"x":2}', object_pairs_hook=runtime.json_unique)
            for text in ('NaN', 'Infinity', '1e999'):
                with self.assertRaises(ValueError):
                    json.loads(text, parse_constant=runtime.bad_json_number, parse_float=runtime.finite_json_number)

    def test_external_manifest_exact_inventory_and_synthetic_file_mutations(self):
        visible = {key: value for key, value in sys.modules.items()
            if not (key == 'numpy' or key.startswith('numpy.') or key == 'mlx' or key.startswith('mlx.'))}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / 'synthetic.bin'
            source.write_bytes(b'not a usable runtime')
            manifest_path = root / 'runtime.json'
            pin = dict(bytes=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            def write(current):
                raw = json.dumps(dict(format='toklet-mlx-reference-runtime-v1', versions=runtime.VERSIONS,
                    files={str(source): current})).encode()
                manifest_path.write_bytes(raw)
                return hashlib.sha256(raw).hexdigest()
            with mock.patch.dict(sys.modules, visible, clear=True), mock.patch.object(runtime, 'runtime_paths', return_value=[source]), mock.patch.object(runtime, 'selected_imports', return_value={}), mock.patch.object(runtime, 'version', side_effect=lambda name: runtime.VERSIONS[name]), mock.patch.object(sys, 'pycache_prefix', str(root / 'cache')):
                expected = write(pin)
                self.assertEqual(runtime.admit_runtime(manifest_path, expected, [source])['files'], 1)
                with self.assertRaises(ValueError):
                    runtime.admit_runtime(manifest_path, '0' * 64, [source])
                for changed in [dict(pin, bytes=True), dict(pin, bytes=pin['bytes'] - 1), dict(pin, bytes=pin['bytes'] + 1), dict(pin, sha256='0' * 64)]:
                    with self.assertRaises(ValueError):
                        runtime.admit_runtime(manifest_path, write(changed), [source])
                expected = write(pin)
                with mock.patch.object(runtime, 'runtime_paths', return_value=[]):
                    with self.assertRaises(ValueError):
                        runtime.admit_runtime(manifest_path, expected, [source])
                with mock.patch.dict(sys.modules, {'mlx.test_marker': object()}):
                    with self.assertRaises(ValueError):
                        runtime.admit_runtime(manifest_path, expected, [source])
                with mock.patch.object(runtime, 'version', return_value='wrong'):
                    with self.assertRaises(ValueError):
                        runtime.admit_runtime(manifest_path, expected, [source])

    def test_shadow_source_and_namespace_locations_reject_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(runtime, 'find_spec', return_value=None):
                with self.assertRaises(ValueError):
                    runtime.selected_imports(root, [])
            numpy_source = root / 'numpy/__init__.py'
            numpy = mock.Mock(origin=str(numpy_source), submodule_search_locations=[str(root / 'numpy')],
                loader=runtime.SourceFileLoader('numpy', str(numpy_source)))
            mlx = mock.Mock(origin=None, submodule_search_locations=[str(root / 'mlx'), str(root / 'shadow')])
            with mock.patch.object(runtime, 'find_spec', side_effect=[numpy, mlx]):
                with self.assertRaises(ValueError):
                    runtime.selected_imports(root, [numpy_source])


if __name__ == '__main__':
    unittest.main()
