"""Portable plain renderer/guard controls; no Jinja/tokenizer/model execution."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('phi3_text_fixture', Path(__file__).with_name('phi3_text_fixture.py'))
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class FakeTokenizer:
    def __init__(self):
        self.calls = []
        self.fail = False

    def id_to_token(self, token):
        return str(token) if 0 <= token < subject.MAPPED_SIZE else None

    def decode(self, ids, *, skip_special_tokens):
        self.calls.append((ids, skip_special_tokens))
        if self.fail:
            raise RuntimeError('injected decoder failure')
        return 'accepted'


class PlainText(unittest.TestCase):
    def test_literal_both_flags_optional_system_and_no_bos(self):
        messages = [{'role': 'user', 'content': ''}]
        self.assertEqual(subject.render(messages, True), '<|user|>\n<|end|>\n<|assistant|>\n')
        self.assertEqual(subject.render(messages, False), '<|user|>\n<|end|>\n<|end|>')
        messages = [{'role': 'system', 'content': 'Brief.'}, {'role': 'user', 'content': 'Hi'},
                    {'role': 'assistant', 'content': 'Hello'}]
        self.assertEqual(subject.render(messages, False),
            '<|system|>\nBrief.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\nHello<|end|>\n<|end|>')
        with self.assertRaises(ValueError):
            subject.render(messages, True)
        self.assertEqual(len(subject.CASES), 8)
        for case in subject.CASES.values():
            for generation in (False, True):
                result = subject.render(case, generation)
                self.assertNotIn('<s>', result)
                self.assertNotIn('<|endoftext|>', result)

    def test_preserves_whitespace_and_unicode_without_trim(self):
        content = ' \t café e\u0301\n\n👋\r\n '
        messages = [{'role': 'user', 'content': content}]
        before = json.dumps(messages, ensure_ascii=False)
        rendered = subject.render(messages, True)
        self.assertEqual(rendered, '<|user|>\n' + content + '<|end|>\n<|assistant|>\n')
        self.assertEqual(json.dumps(messages, ensure_ascii=False), before)
        controls = '\x1c\u00a0X\u2003\x1f'
        self.assertIn(controls, subject.render([{'role': 'user', 'content': controls}], False))

    def test_roles_fields_reserved_literals_and_bounds_reject(self):
        invalid = [[], [{'role': 'system', 'content': 'only'}], [{'role': 'assistant', 'content': 'x'}],
            [{'role': 'tool', 'content': 'x'}], [{'role': 'user', 'content': 'x', 'tools': []}],
            [{'role': 'user', 'content': 'x'}, {'role': 'user', 'content': 'again'}],
            [{'role': 'user', 'content': 'x'}, {'role': 'system', 'content': 'late'}],
            [{'role': 'user', 'content': None}], [{'role': 'user', 'content': '\ud800'}],
            [{'role': 'user', 'content': 'x' * 65536}], [{'role': 'user', 'content': ' ' * 65536}]]
        invalid.append([{'role': 'user' if i % 2 == 0 else 'assistant', 'content': ''} for i in range(65)])
        for messages in invalid:
            with self.subTest(messages=str(messages)[:80]):
                with self.assertRaises(ValueError):
                    subject.render(messages, True)
        for flag in (None, 1, 'true'):
            with self.assertRaises(ValueError):
                subject.render([{'role': 'user', 'content': ''}], flag)
        for literal in subject.ADDED:
            with self.assertRaises(ValueError):
                subject.render([{'role': 'user', 'content': 'safe' + literal + 'tail'}], True)
        with self.assertRaises(ValueError):
            subject.duplicate_checked('{"role":"user","role":"assistant"}')

    def test_decode_validates_whole_sequence_before_acceptance(self):
        tokenizer = FakeTokenizer()
        for bad in list(range(32011, 32064)) + [-1, 32064, True, 1.0, '1']:
            ids = [7, bad, 13]
            with self.assertRaises(ValueError):
                subject.decode_bounded(tokenizer, ids)
            self.assertEqual(ids, [7, bad, 13])
            self.assertEqual(tokenizer.calls, [])
        with self.assertRaises(ValueError):
            subject.decode_bounded(tokenizer, [7] * 513)
        self.assertEqual(subject.decode_bounded(tokenizer, [1, 2, 32000, 32007, 32010]), 'accepted')
        self.assertEqual(tokenizer.calls[-1], ([1, 2, 32000, 32007, 32010], False))
        tokenizer.fail = True
        ids = [7, 13]
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            subject.decode_bounded(tokenizer, ids)
        self.assertEqual(ids, [7, 13])
        self.assertEqual(subject.STOPS, (32000, 32007))
        self.assertTrue(all(token not in subject.STOPS for token in [1, 2]))

    def test_literal_utf8_byte_control_not_encode_roundtrip(self):
        vocabulary = {f'<0x{i:02X}>': i + 3 for i in range(256)}
        self.assertEqual(subject.byte_fallback_ids('é', vocabulary), [0xC3 + 3, 0xA9 + 3])
        self.assertEqual(subject.byte_fallback_ids('👋', vocabulary), [0xF0 + 3, 0x9F + 3, 0x91 + 3, 0x8B + 3])
        self.assertEqual(subject.direct_byte_expected(' x'), 'x')
        self.assertEqual(subject.direct_byte_expected('  x'), ' x')
        self.assertEqual(subject.direct_byte_expected('\tx'), '\tx')


if __name__ == '__main__':
    unittest.main()
