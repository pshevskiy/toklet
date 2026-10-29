"""Portable manual-codec controls; no tokenizer/template/model imports."""
import unittest

from lfm2_text_fixture import BOS, IM_START, IM_END, checked_ids, is_stop, render, parse_messages

RESERVED = [BOS, IM_START, IM_END, '<|endoftext|>', '<|file_end|>']


class Controls(unittest.TestCase):
    def test_literal_empty_and_both_prompt_flags(self):
        messages = [dict(role='user', content='')]
        expected = BOS + IM_START + 'user\n' + IM_END + '\n'
        self.assertEqual(render(messages, False, RESERVED), expected)
        self.assertEqual(render(messages, True, RESERVED), expected + IM_START + 'assistant\n')

    def test_system_alternation_and_whitespace_preserved(self):
        content = ' \t café é\n\n'
        messages = [dict(role='system', content='Be concise.'),
                    dict(role='user', content=content),
                    dict(role='assistant', content='Answer'),
                    dict(role='user', content='東京')]
        actual = render(messages, True, RESERVED)
        self.assertEqual(actual.count(BOS), 1)
        self.assertIn('user\n' + content + IM_END, actual)
        self.assertIn('assistant\nAnswer' + IM_END, actual)
        self.assertTrue(actual.endswith(IM_START + 'assistant\n'))
        # Ordinary added vocabulary entries are not reserved control literals.
        self.assertIn('Mathias python', render([dict(role='user', content='Mathias python')], False, RESERVED))

    def test_malformed_message_categories(self):
        invalid = [None, (), [], [None], [dict(role='system', content='x')],
                   [dict(role='assistant', content='x')], [dict(role='tool', content='x')],
                   [dict(role='user', content=1)], [dict(role=1, content='x')],
                   [dict(role='user')], [dict(role='user', content='x', extra='y')],
                   [dict(role='user', content='x'), dict(role='user', content='y')],
                   [dict(role='user', content='x'), dict(role='system', content='y')],
                   [dict(role='system', content='x'), dict(role='system', content='y')]]
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                render(messages, True, RESERVED)
        for flag in (0, 1, None, 'true'):
            with self.assertRaises(ValueError):
                render([dict(role='user', content='x')], flag, RESERVED)

    def test_reserved_literals_and_bounds(self):
        for literal in RESERVED:
            with self.assertRaises(ValueError):
                render([dict(role='user', content='before' + literal + 'after')], True, RESERVED)
        for content in ('x' * 65536, '東京' * 20000):
            with self.assertRaises(ValueError):
                render([dict(role='user', content=content)], True, RESERVED)
        with self.assertRaises(ValueError):
            render([dict(role='user' if i % 2 == 0 else 'assistant', content='')
                    for i in range(65)], True, RESERVED)

    def test_mapped_boundaries_and_unmapped_head_slots(self):
        self.assertEqual(checked_ids([0, 1, 64399]), [0, 1, 64399])
        for ids in ([-1], [True], [1.0], [64400], [65535], [65536], [0] * 513, (1,)):
            with self.assertRaises(ValueError):
                checked_ids(ids)
        self.assertEqual(checked_ids([]), [])
        self.assertEqual(len(checked_ids([0] * 512)), 512)

    def test_stop_is_exactly_seven(self):
        self.assertTrue(is_stop(7))
        for token in (0, 1, 2, 6, 64399):
            self.assertFalse(is_stop(token))
        with self.assertRaises(ValueError):
            is_stop(64400)

    def test_strict_json_before_schema_conversion(self):
        for raw in (b'[{"role":"user","role":"assistant","content":"x"}]',
                    b'[{"role":"user","content":NaN}]', b'[', b'x' * 65537):
            with self.assertRaises(ValueError):
                parse_messages(raw)
        messages = parse_messages(b'[{"role":"user","content":""}]')
        self.assertEqual(render(messages, False, RESERVED), BOS + IM_START + 'user\n' + IM_END + '\n')

    def test_assistant_ending_and_exact_message_count_boundary(self):
        messages = [dict(role='user' if i % 2 == 0 else 'assistant', content='') for i in range(64)]
        for generation in (False, True):
            actual = render(messages, generation, RESERVED)
            self.assertEqual(actual.count(IM_END), 64)
            self.assertEqual(actual.count(BOS), 1)
            self.assertTrue(actual.endswith(IM_START + 'assistant\n') if generation
                            else actual.endswith('assistant\n' + IM_END + '\n'))
        with self.assertRaises(ValueError):
            render(messages + [dict(role='user', content='')], False, RESERVED)


if __name__ == '__main__':
    unittest.main()
