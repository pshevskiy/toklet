"""Portable text-runtime admission failures; no model or third-party imports."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import chat_reference_runtime as runtime
import app_chat_qwen3 as producer


class RuntimeChecks(unittest.TestCase):
    def setUp(self):
        if 'TOKLET_CHAT_REFERENCE_OUTPUT' not in os.environ:
            raise unittest.SkipTest('Explicit owned text-runtime manifest required')
        self.output = Path(os.environ['TOKLET_CHAT_REFERENCE_OUTPUT']).resolve()
        self.manifest = self.output / 'runtime.json'
        self.raw = self.manifest.read_bytes()
        self.sha = hashlib.sha256(self.raw).hexdigest()
        drivers = [Path(n) for n in json.loads(self.raw)['files'] if Path(n).name in ('collect_qwen3_chat_reference.py', 'qwen3_chat_reference.py')]
        self.assertEqual(len(drivers), 1)
        self.sources = [Path(producer.__file__).resolve(), drivers[0]]
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.previous_cache = sys.pycache_prefix
        sys.pycache_prefix = str(self.folder / 'cache')
        runtime._admitted = None

    def tearDown(self):
        runtime._admitted = None
        sys.pycache_prefix = self.previous_cache
        self.temp.cleanup()
        self.assertFalse(any(name == p or name.startswith(p + '.') for name in sys.modules for p in runtime.PACKAGES))

    def admit(self, path=None, sha=None):
        return runtime.admit(path or self.manifest, sha or self.sha, self.sources)

    def changed(self, change):
        data = json.loads(self.raw)
        change(data)
        raw = json.dumps(data).encode()
        path = self.folder / 'mutated.json'
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_valid_admission_is_before_library_imports(self):
        result = self.admit()
        self.assertGreater(result['files'], 50)
        runtime.check()

    def test_capture_without_admission_cannot_import_libraries(self):
        with self.assertRaisesRegex(ValueError, 'must be admitted'):
            producer.capture(self.folder)

    def test_external_manifest_sha_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'buffer identity'):
            self.admit(sha='0' * 64)

    def test_exact_file_pin_mismatch(self):
        path, sha = self.changed(lambda data: data['files'][next(iter(data['files']))].update(sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'file bytes'):
            self.admit(path, sha)

    def test_inventory_member_missing(self):
        path, sha = self.changed(lambda data: data['files'].pop(next(iter(data['files']))))
        with self.assertRaisesRegex(ValueError, 'runtime closure'):
            self.admit(path, sha)

    def test_version_mismatch(self):
        path, sha = self.changed(lambda data: data['versions'].update(tokenizers='0.0.0'))
        with self.assertRaisesRegex(ValueError, 'runtime closure'):
            self.admit(path, sha)

    def test_preimported_library_rejects(self):
        with patch.dict(sys.modules, {'jinja2': types.ModuleType('jinja2')}):
            with self.assertRaisesRegex(ValueError, 'already imported'):
                self.admit()

    def test_preexisting_package_bytecode_rejects(self):
        manifest = json.loads(self.raw)
        source = next(Path(name) for name in manifest['files'] if name.endswith('/jinja2/__init__.py'))
        cache = Path(runtime.cache_from_source(str(source)))
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b'not executable bytecode')
        with self.assertRaisesRegex(ValueError, 'bytecode cache'):
            self.admit()

    def test_import_origin_change_rejects_after_admission(self):
        self.admit()
        wrong = types.ModuleType('jinja2')
        wrong.__file__ = '/unadmitted/jinja2.py'
        with patch.dict(sys.modules, {'jinja2': wrong}):
            with self.assertRaisesRegex(ValueError, 'Loaded text runtime'):
                runtime.check()

    def test_runtime_manifest_change_rejects_after_admission(self):
        self.admit()
        with patch.object(runtime, 'captured', side_effect=ValueError('changed manifest')):
            with self.assertRaisesRegex(ValueError, 'changed manifest'):
                runtime.check()

    def test_duplicate_manifest_json_rejects(self):
        raw = b'{"format":"x","format":"y"}'
        path = self.folder / 'duplicate.json'
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.admit(path, hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    unittest.main()
