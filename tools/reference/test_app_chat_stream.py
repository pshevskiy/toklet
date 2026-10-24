"""Explicit text-oracle checks; no model/GPU/network dependency in this suite."""
from pathlib import Path
import json
import tempfile
import unittest

import app_chat_stream as producer
from tokenizers import Tokenizer

FIXTURE = Path(__file__).resolve().parents[2] / 'tests/fixtures/chat-stream-v1'


class StreamOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (FIXTURE / 'reference.json').read_bytes()
        cls.reference = json.loads(cls.raw)
        cls.tokenizers = {}
        for name, profile in cls.reference['profiles'].items():
            cls.tokenizers[name] = Tokenizer.from_file(str(FIXTURE / profile['compact_tokenizer']))

    def test_identity_and_compact_policy(self):
        reference = self.reference
        self.assertEqual(reference['format'], 'toklet-chat-stream-oracle-v1')
        self.assertFalse(reference['numerical_generation_qualified'])
        self.assertFalse(reference['append_only_stability_assumed'])
        self.assertEqual(reference['producer_sha256'], producer.digest(Path(producer.__file__).read_bytes()))
        self.assertEqual(set(reference['profiles']), {'smol', 'qwen', 'mistral'})
        for name, profile in reference['profiles'].items():
            _, repo, revision, pin = producer.PROFILES[name]
            self.assertEqual((profile['repo'], profile['revision'], profile['tokenizer_sha256']), (repo, revision, pin))
            raw = (FIXTURE / profile['compact_tokenizer']).read_bytes()
            self.assertEqual(producer.digest(raw), profile['compact_sha256'])
            compact = json.loads(raw)
            self.assertTrue(profile['compact_is_decode_only'])
            self.assertLess(len(raw), 8192)
            self.assertEqual(compact['model']['merges'], [])
            self.assertIsNone(compact['normalizer'])
            self.assertIsNone(compact['pre_tokenizer'])
            self.assertIsNone(compact['post_processor'])

    def test_all_complete_prefixes(self):
        count = 0
        for name, profile in self.reference['profiles'].items():
            names = set()
            for case in profile['cases']:
                self.assertNotIn(case['name'], names)
                names.add(case['name'])
                self.assertEqual(len(case['prefixes']), len(case['ids']) + 1)
                for end, text in enumerate(case['prefixes']):
                    self.assertEqual(self.tokenizers[name].decode(case['ids'][:end], skip_special_tokens=True), text)
                count += 1
        self.assertEqual(count, 33)

    def test_actual_131072_ids_not_word_count(self):
        count = 0
        for name, profile in self.reference['profiles'].items():
            for case in profile['long_cases']:
                self.assertTrue(case['repeat_ids'])
                ids = case['repeat_ids'] * case['repeat_count'] + case['tail_ids']
                self.assertEqual(len(ids), 131072)
                self.assertEqual(case['total_tokens'], 131072)
                raw = self.tokenizers[name].decode(ids, skip_special_tokens=True).encode()
                self.assertEqual(len(raw), case['utf8_bytes'])
                self.assertEqual(producer.digest(raw), case['text_sha256'])
                count += 1
        self.assertEqual(count, 8)

    def test_unicode_control_and_literal_witnesses(self):
        for profile in self.reference['profiles'].values():
            cases = {case['name']: case for case in profile['cases']}
            self.assertEqual(cases['split_globe']['prefixes'][-1], '🌍')
            self.assertEqual(cases['skipped_controls_between_bytes']['prefixes'][-1], '🌍')
            self.assertEqual(cases['literal_replacement']['prefixes'][-1], '�')
            self.assertTrue(all(text == '' for text in cases['controls_only']['prefixes']))
            self.assertNotEqual(cases['unfinished_globe']['prefixes'][-1], '🌍')
            self.assertIn('�', cases['unfinished_globe']['prefixes'][-1])

    def test_mistral_rewrites_entire_fallback_run(self):
        for name, profile in self.reference['profiles'].items():
            case = next(case for case in profile['cases'] if case['name'] == 'run_then_invalid')
            self.assertEqual(case['prefixes'][-2], 'A' * 129)
            self.assertEqual(case['prefixes'][-1], '�' * 130 if name == 'mistral' else 'A' * 129 + '�')
        # Append-only output cannot repair this sequence once the A run is sent.
        self.assertFalse(('�' * 130).startswith('A' * 129))

    def test_pinned_read_is_bounded_and_identity_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'tokenizer.json'
            path.write_bytes(b'valid')
            self.assertEqual(producer.admitted(path, producer.digest(b'valid')), b'valid')
            with self.assertRaises(ValueError):
                producer.admitted(path, producer.digest(b'other'))
            path.write_bytes(b' ' * (producer.MAX_ASSET_BYTES + 1))
            with self.assertRaises(ValueError):
                producer.admitted(path, producer.digest(path.read_bytes()))

    def test_exclusive_publication_and_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            outputs = {'a.json': b'{}\n', 'b.json': b'[]\n'}
            with self.assertRaises(ValueError):
                producer.publish({'a.json': b'{}', '../escape': b'x'}, path)
            self.assertEqual(list(path.iterdir()), [])
            producer.publish(outputs, path)
            producer.verify(outputs, path)
            with self.assertRaises(ValueError):
                producer.publish(outputs, path)
            (path / 'b.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                producer.verify(outputs, path)
            (path / 'unexpected').write_bytes(b'')
            with self.assertRaises(ValueError):
                producer.verify(outputs, path)

    def test_compact_policy_rejects_non_bpe_and_active_truncation(self):
        raw = json.loads((FIXTURE / 'qwen-tokenizer.json').read_bytes())
        raw['model']['type'] = 'WordPiece'
        with self.assertRaises(ValueError):
            producer.compact(producer.serialize(raw), set())
        raw['model']['type'] = 'BPE'
        raw['truncation'] = {'max_length': 1}
        with self.assertRaises(ValueError):
            producer.compact(producer.serialize(raw), set())


if __name__ == '__main__':
    unittest.main()
