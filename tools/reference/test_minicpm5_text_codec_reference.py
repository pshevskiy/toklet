"""Download-free policy/capture controls; third-party text code stays unimported."""
import copy
import io
import json
import unittest

import minicpm5_text_codec_reference as reference


LITERALS = ['<s>', '</s>', '<think>', '</think>', '<|im_start|>', '<|im_end|>', '<unused_token_0>']
FLAGS = dict(enable_thinking=False, add_generation_prompt=True)


class Policy(unittest.TestCase):
    def test_empty_whitespace_unicode_and_initial_system_are_verbatim(self):
        for text in ['', '\t \r\n', 'e\u0301 é 世界🙂']:
            for initial in [[], [dict(role='system', content='S\n')]]:
                messages = initial + [dict(role='user', content=text)]
                before = copy.deepcopy(messages)
                for thinking in [False, True]:
                    for generation in [False, True]:
                        reference.validate(messages, dict(enable_thinking=thinking,
                                           add_generation_prompt=generation), LITERALS)
                self.assertEqual(messages, before)

    def test_final_assistant_is_only_valid_without_generation_prompt(self):
        messages = [dict(role='user', content=''), dict(role='assistant', content='')]
        reference.validate(messages, dict(FLAGS, add_generation_prompt=False), LITERALS)
        with self.assertRaises(ValueError):reference.validate(messages, FLAGS, LITERALS)

    def test_role_type_field_control_and_option_rejections(self):
        cases = [[], [dict(role='system', content='')], [dict(role='assistant', content='A')],
                 [dict(role='user', content=None)], [dict(role='user', content=[])],
                 [dict(role='user', content='x', tools=[])], [dict(role='tool', content='x')],
                 [dict(role='user', content='x\0')], [dict(role='user', content='<think>')],
                 [dict(role='user', content='<unused_token_0>')]]
        for messages in cases:
            with self.assertRaises(ValueError):reference.validate(messages, FLAGS, LITERALS)
        for value in [0, None, 'false']:
            with self.assertRaises(ValueError):
                reference.validate([dict(role='user', content='')], dict(FLAGS, enable_thinking=value), LITERALS)
        with self.assertRaises(ValueError):
            reference.validate([dict(role='user', content='')], dict(FLAGS, extra=False), LITERALS)

    def test_canonical_wire_bound_and_message_count(self):
        with self.assertRaises(ValueError):
            reference.validate([dict(role='user', content='x' * reference.MAX_TEXT)], FLAGS, LITERALS)
        messages = [dict(role='user' if i % 2 == 0 else 'assistant', content='') for i in range(63)]
        reference.validate(messages, FLAGS, LITERALS)
        with self.assertRaises(ValueError):
            reference.validate(messages + [dict(role='assistant', content='')], dict(FLAGS, add_generation_prompt=False), LITERALS)

    def test_capture_exact_utf8_and_bounded_short_write(self):
        stream = io.BytesIO();capture = reference.Capture(stream)
        capture.add(dict(text='é🙂'))
        self.assertEqual(json.loads(stream.getvalue()), dict(text='é🙂'))
        self.assertEqual(capture.bytes, len(stream.getvalue()))
        with self.assertRaises(ValueError):capture.add(dict(text='x' * (256 * 1024)))
        class Short:
            def write(self, raw):return len(raw) - 1
        with self.assertRaises(OSError):reference.Capture(Short()).add(dict(text='x'))

    def test_stop_slice_does_not_import_mlx_or_text_dependencies_at_import(self):
        import sys
        self.assertNotIn('mlx', sys.modules)
        self.assertNotIn('tokenizers', sys.modules)
        self.assertNotIn('jinja2', sys.modules)


if __name__ == '__main__':unittest.main()
