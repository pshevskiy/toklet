import copy
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from lfm3_conv_bf16_fixture import conv_schema, head_gates, word_bytes


class FixtureAdmission(unittest.TestCase):
    def test_source_schema_covers_all_thirteen_banks(self):
        config = dict(hidden_size=2048, conv_L_cache=7, conv_bias=False,
            layer_types=['conv']*13 + ['full_attention']*6 + ['sliding_attention']*5)
        names = conv_schema(config)
        self.assertEqual(len(names), 13)
        self.assertEqual(set(names), {f'model.layers.{i}.conv.conv.weight' for i in range(13)})
        for key, value in [('hidden_size', 1024), ('conv_L_cache', 3), ('conv_bias', True),
                           ('layer_types', ['conv']*24)]:
            changed = copy.deepcopy(config)
            changed[key] = value
            with self.assertRaises(ValueError):
                conv_schema(changed)

    def test_word_packing_preserves_signed_zero_and_subnormals(self):
        words = [0, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x80, 0x8080, 0x3f80, 0xbf80]
        packed = word_bytes(words)
        self.assertEqual(len(packed), len(words)*2)
        self.assertEqual(list(struct.unpack('<10H', packed)), words)
        self.assertEqual(packed[:8], b'\x00\x00\x00\x80\x01\x00\x01\x80')

    def test_head_gate_inputs_cover_zero_one_and_subnormals(self):
        words = head_gates(17)
        self.assertEqual(len(words), 17*16)
        self.assertEqual(set(words), {0, 1, 128, 0x3c00, 0x3f00, 0x3f7f, 0x3f80})
        self.assertNotEqual(words[:16], words[16:32])

    def test_cli_rejects_existing_destinations_before_loading_runtime(self):
        producer = Path(__file__).with_name('lfm3_conv_bf16_fixture.py')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            existing = base/'raw'
            existing.mkdir()
            sentinel = existing/'sentinel'
            sentinel.write_text('unchanged')
            output = base/'fixture.json'
            command = [sys.executable, '-B', str(producer)]
            for key in ['contract', 'runtime', 'metadata', 'model']:
                command += ['--'+key, str(base/'missing')]
            command += ['--fresh-directory', str(existing), '--fresh-fixture', str(output)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Fresh fixture destinations required', result.stderr)
            self.assertEqual(sentinel.read_text(), 'unchanged')
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
