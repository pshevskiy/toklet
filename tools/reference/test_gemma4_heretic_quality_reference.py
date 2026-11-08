"""CPU controls for distinct parent pins and frozen quality-reference bounds."""
import hashlib
from pathlib import Path
import struct
import tempfile
import time
import unittest
import bf16_source
import gemma4_heretic_bf16_weights as parent
import gemma4_heretic_quality_reference as reference

ROOT = Path(__file__).resolve().parents[2]


class QualityReferenceTests(unittest.TestCase):
    def test_parent_metadata_matches_pinned_fixture_and_original_header(self):
        config = (ROOT/'tests/fixtures/gemma-heretic-bf16-metadata-v1/config.json').read_bytes()
        header = (ROOT/'tests/fixtures/gemma-unquantized-header-v1/header.json').read_bytes()
        prefix = struct.pack('<Q', parent.HEADER_BYTES)
        value = parent.header(config, prefix, header)
        self.assertEqual(len(value), 2012)
        self.assertEqual(value[parent.PREFIX+'embed_tokens_per_layer.weight']['shape'], [262144, 8960])
        self.assertEqual(hashlib.sha256(config).hexdigest(), parent.CONFIG_SHA256)

    def test_parent_metadata_rejects_original_config_or_changed_metadata(self):
        config = (ROOT/'tests/fixtures/gemma-heretic-bf16-metadata-v1/config.json').read_bytes()
        original = (ROOT/'tests/fixtures/gemma-unquantized-header-v1/config.json').read_bytes()
        header = (ROOT/'tests/fixtures/gemma-unquantized-header-v1/header.json').read_bytes()
        prefix = struct.pack('<Q', parent.HEADER_BYTES)
        for cfg, pre, body in [(original, prefix, header), (config+b' ', prefix, header),
                               (config, b'\0'*8, header), (config, prefix, header[:-1]),
                               (config, prefix, header[:-1]+bytes([header[-1]^1]))]:
            with self.subTest(config_bytes=len(cfg)), self.assertRaisesRegex(ValueError, 'Exact modified'):
                parent.header(cfg, pre, body)

    def test_parent_weight_binding_rejects_an_authenticated_other_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.bin'
            path.write_bytes(b'\0\0\x80\x3f')
            deadline = time.monotonic()+30
            with bf16_source.Source(path, 4, hashlib.sha256(path.read_bytes()).hexdigest(), deadline=deadline) as owner:
                with self.assertRaisesRegex(ValueError, 'Pinned authenticated modified'):
                    parent.weights(owner, b'{}', deadline)

    def test_quality_phases_preserve_frozen_prefix_and_fit_capture_bound(self):
        short, a, small = reference.plan('compat512')
        full, b, large = reference.plan('full1025')
        self.assertEqual(short, full[:512])
        self.assertEqual(a, [0, 31, 255, 511])
        self.assertEqual(b, [0, 31, 255, 511, 512, 767, 1023, 1024])
        self.assertEqual(len(full), 1025)
        self.assertEqual(large, (8*262144+2*1025*4608)*12)
        self.assertLess(small, large)
        self.assertLess(large, reference.MAX_CAPTURE)
        with self.assertRaisesRegex(ValueError, 'Named frozen'):
            reference.plan('full1026')


if __name__ == '__main__':
    unittest.main()
