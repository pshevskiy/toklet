"""Proposed stdlib renderer controls; actual tokenizer/Jinja export is explicit."""
import unittest
from llama3_text_fixture import render, SPECIAL, DATE


class PlainText(unittest.TestCase):
    def test_literal_empty_single_and_bos_owner(self):
        result=render([dict(role='user',content='')],SPECIAL)
        expected=('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
                  'Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>'
                  '<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>'
                  '<|start_header_id|>assistant<|end_header_id|>\n\n')
        self.assertEqual(result,expected);self.assertEqual(result.count('<|begin_of_text|>'),1)
        self.assertEqual(DATE,'26 Jul 2024')

    def test_roles_reserved_literals_and_bounds(self):
        for messages in [[],[dict(role='system',content='x')],[dict(role='tool',content='x')],
                         [dict(role='user',content='x'),dict(role='assistant',content='y')],
                         [dict(role='user',content='x',tools=[])],
                         [dict(role='user',content='x'*65536)],[dict(role='user',content=' '*65536)]]:
            with self.assertRaises(ValueError):render(messages,SPECIAL)
        for literal in SPECIAL:
            with self.assertRaises(ValueError):render([dict(role='user',content='safe'+literal)],SPECIAL)
        result=render([dict(role='user',content=' \t Привет 👋 \n')],SPECIAL)
        self.assertIn('\n\nПривет 👋<|eot_id|>',result)
        # Jinja/Python trim includes U+001C..001F, unlike Rust str::trim.
        self.assertIn('\n\nX<|eot_id|>',render([dict(role='user',content='\x1c\u00a0 X \u2003\x1f')],SPECIAL))


if __name__=='__main__':unittest.main()
