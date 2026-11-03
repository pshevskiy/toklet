import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import prepare_llama3_original_quality as subject


class QualityInputsTests(unittest.TestCase):
    def fixture(self):
        folder = subject.ROOT/'benchmarks/llama3-original-quality-v1'
        value = json.loads((folder/'manifest.json').read_bytes())
        raw = (folder/'tokens.u32le').read_bytes()
        return value, raw

    def test_frozen_targets_are_complete_and_long_target_is_held_out(self):
        value, raw = self.fixture()
        subject.validate(value)
        targets = [i for block in value['blocks'] for i in range(*block['global_target_indices'])]
        self.assertEqual(targets, list(range(1, 10001)))
        for block in value['blocks']:
            self.assertEqual(block['input_end']-block['input_start']-1, block['score_rows'][1])
            self.assertLessEqual(block['input_end']-block['input_start'], 1025)
        self.assertEqual(value['long_probe_prefix']['sha256'], subject.parent.digest(raw[:4104]))
        self.assertEqual(subject.select(list(struct.unpack('<10001I', raw))), raw)
        self.assertEqual(raw[:4100], (subject.ROOT/'benchmarks/llama3-loss1024-v1/tokens.u32le').read_bytes())

    def test_loader_rejects_corruption_wrong_prefix_and_policy(self):
        value, raw = self.fixture()
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            def check(document, tokens):
                manifest = (json.dumps(document)+'\n').encode()
                (folder/'manifest.json').write_bytes(manifest)
                (folder/'tokens.u32le').write_bytes(tokens)
                return subject.load(folder, subject.parent.digest(manifest))
            check(value, raw)
            with self.assertRaises(ValueError):
                check(value, raw[:-1]+bytes([raw[-1]^1]))
            wrong = copy.deepcopy(value); wrong['long_probe_prefix']['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                check(wrong, raw)
            wrong = copy.deepcopy(value); wrong['blocks'][-1]['global_target_indices'][1] -= 1
            with self.assertRaises(ValueError):
                check(wrong, raw)
            wrong = copy.deepcopy(value); wrong['policy']['mean_nll_increase_limit'] = 0.1
            with self.assertRaises(ValueError):
                check(wrong, raw)

    def test_invalid_token_stream_rejected(self):
        for ids in [[1]*10000, [1]*10000+[True], [1]*10000+[128256], [1]*10000+[-1]]:
            with self.assertRaises(ValueError):
                subject.select(ids)
