"""Portable configuration, exact-read and publication-path controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import granite_convert as convert


def config():
    path = Path(__file__).resolve().parents[2] / 'tests/fixtures/granite-config-v1/granite_modified.json'
    return json.loads(path.read_bytes())


def matrices():
    return [dict(module='model.embed_tokens', bits=4, group_size=64, mode='affine'),
            dict(module='lm_head', bits=8, group_size=64, mode='affine')]


class Producer(unittest.TestCase):
    def test_configuration_adds_only_exact_rope_alias_and_explicit_recipe(self):
        original = config()
        before = copy.deepcopy(original)
        child = convert.converted_config(original, matrices())
        self.assertEqual(original, before)
        self.assertEqual(child['rope_theta'], original['rope_parameters']['rope_theta'])
        self.assertEqual(child['quantization'], dict(group_size=64, bits=4, mode='affine',
                                                   lm_head=dict(group_size=64, bits=8, mode='affine')))
        del child['rope_theta'], child['quantization']
        self.assertEqual(child, original)

    def test_ambiguous_or_non_bf16_source_semantics_reject(self):
        for change in [dict(model_type='llama'), dict(dtype='float16'), dict(torch_dtype='float32'),
                       dict(quantization={}), dict(rope_theta=1e7), dict(rope_theta=10000001),
                       dict(rope_scaling={'rope_type': 'yarn'}),
                       dict(rope_parameters={'rope_type': 'default', 'rope_theta': True}),
                       dict(rope_parameters={'rope_type': 'default', 'rope_theta': 1e7}),
                       dict(rope_parameters={'rope_type': 'default', 'rope_theta': 10000000, 'factor': 1})]:
            source = config()
            source.update(change)
            with self.assertRaises(ValueError):
                convert.converted_config(source, matrices())
        with self.assertRaises(ValueError):
            convert.converted_config(config(), matrices() + [matrices()[1]])

    def test_exact_reads_and_stream_hash_preserve_partial_read_work(self):
        raw = b'0123456789' * 1000
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'source').open('w+b') as file:
                file.write(raw)
                file.flush()
                pread = os.pread

                def partial(fd, count, offset):
                    return pread(fd, min(count, 317), offset)

                with patch.object(convert.os, 'pread', side_effect=partial):
                    self.assertEqual(convert.read_exact(file.fileno(), len(raw), 0), raw)
                    self.assertEqual(convert.hash_range(file.fileno(), len(raw)), hashlib.sha256(raw).hexdigest())
                for count, offset in [(True, 0), (convert.BLOCK + 1, 0), (1, -1), (1, False)]:
                    with self.assertRaises(ValueError):
                        convert.read_exact(file.fileno(), count, offset)
                with self.assertRaises(ValueError):
                    convert.hash_range(file.fileno(), len(raw) + 1)

    def test_named_source_replacement_does_not_pass_borrowed_descriptor(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source'
            path.write_bytes(b'original')
            with path.open('rb') as file:
                identity = convert.verify_named(file.fileno(), path)
                self.assertEqual(convert.verify_named(file.fileno(), path, identity), identity)
                path.rename(Path(folder) / 'old')
                path.write_bytes(b'original')
                with self.assertRaises(ValueError):
                    convert.verify_named(file.fileno(), path, identity)

    def test_existing_files_directories_and_dangling_links_reject_before_peer_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            occupied = root / 'occupied'
            occupied.write_bytes(b'keep')
            for existing in [occupied, root]:
                peer = root / 'new-peer'
                with self.assertRaises(ValueError):
                    convert.reserve_output(existing, peer)
                self.assertFalse(peer.exists())
                with self.assertRaises(ValueError):
                    convert.reserve_output(peer, existing)
                self.assertFalse(peer.exists())
            link = root / 'dangling'
            link.symlink_to(root / 'absent')
            with self.assertRaises(ValueError):
                convert.reserve_output(link, root / 'new-peer')
            with self.assertRaises(ValueError):
                convert.reserve_output(root / 'same', root / 'same')
            self.assertEqual(occupied.read_bytes(), b'keep')

    def test_cli_existing_output_rejects_before_opening_contract_or_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            occupied = root / 'occupied'
            occupied.write_bytes(b'keep')
            command = [sys.executable, str(Path(convert.__file__)),
                       '--environment-root', 'missing-env', '--runtime-manifest', 'missing-runtime',
                       '--runtime-sha256', '0' * 64, '--contract', 'missing-contract',
                       '--contract-sha256', '0' * 64, '--source-authentication', 'missing-auth',
                       '--fresh-output', str(occupied), '--fresh-evidence', str(root / 'new-evidence')]
            result = subprocess.run(command, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b'')
            self.assertIn(b'Never replace', result.stderr)
            self.assertNotIn(b'missing-contract', result.stderr)
            self.assertFalse((root / 'new-evidence').exists())
            self.assertEqual(occupied.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
