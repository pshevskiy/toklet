import io
import json
import unittest
import qwen35_text_codec_reference as ref


class Policy(unittest.TestCase):
    def setUp(self):self.options=dict(enable_thinking=False,add_generation_prompt=True);self.one=[dict(role='user',content='Hi')]
    def test_valid_text_unicode_and_alternation(self):
        for value in ['Hi','Привет 世界 🙂 e\u0301 é','\u00a0Hello\u3000']:
            ref.validate([dict(role='user',content=value)],self.options,[])
        ref.validate(self.one+[dict(role='assistant',content=' OK '),*self.one],self.options,[])
    def test_options_are_exact_and_typed(self):
        for value in [0,1,None,True]:
            options={**self.options,'enable_thinking':value}
            with self.subTest(value=value),self.assertRaises(ValueError):ref.validate(self.one,options,[])
        for options in [{},dict(self.options,tools=[]),dict(self.options,add_generation_prompt=1),dict(self.options,add_generation_prompt=False)]:
            with self.subTest(options=options),self.assertRaises(ValueError):ref.validate(self.one,options,[])
    def test_roles_fields_content_and_message_counts(self):
        cases=[[],self.one+[dict(role='assistant',content='Hi')],[dict(role='system',content='Hi')],
            [dict(role='user',content=[])],[dict(role='user',content=' ')],[dict(self.one[0],reasoning_content='x')]]
        for n in [64,65]:cases.append([dict(role='user' if i%2==0 else 'assistant',content='x') for i in range(n)])
        for messages in cases:
            with self.subTest(messages=messages[:2]),self.assertRaises(ValueError):ref.validate(messages,self.options,[])
        ref.validate([dict(role='user' if i%2==0 else 'assistant',content='x') for i in range(63)],self.options,[])
    def test_reserved_literals_escaped_forms_and_C0_controls(self):
        for text in ['<think>','<|im_end|>','\u001cHi','Hi\x00']:
            with self.subTest(text=text),self.assertRaises(ValueError):ref.validate([dict(role='user',content=text)],self.options,['<think>','<|im_end|>'])
        ref.validate([dict(role='user',content='a\x7fb\t\r\n')],self.options,[])
        decoded=json.loads('[{"role":"user","content":"\\u003cthink\\u003e"}]')
        with self.assertRaises(ValueError):ref.validate(decoded,self.options,['<think>'])
    def test_utf8_and_canonical_byte_limit(self):
        for value in ['é'*(ref.MAX_TEXT//2),'x'*(ref.MAX_TEXT-31)]:
            with self.subTest(chars=len(value)),self.assertRaises(ValueError):ref.validate([dict(role='user',content=value)],self.options,[])
    def test_duplicate_json_rejects(self):
        with self.assertRaises(ValueError):json.loads('{"role":"user","role":"assistant"}',object_pairs_hook=ref.runtime.common.json_unique)


class SourceAndCapture(unittest.TestCase):
    def test_unmodified_pinned_upstream_EOS_union(self):
        self.assertEqual(ref.eos_source_witness(248046,248044)['stop_ids'],[248044,248046])
        self.assertEqual(ref.eos_source_witness(248046,248046)['stop_ids'],[248046])
        self.assertEqual(ref.eos_source_witness(None,248044)['stop_ids'],[248044])
    def test_capture_bounds_before_partial_publication(self):
        stream=io.BytesIO();capture=ref.Capture(stream);capture.add(dict(ok=True));before=stream.getvalue()
        with self.assertRaises(ValueError):capture.add(dict(text='x'*ref.MAX_TEXT))
        self.assertEqual(stream.getvalue(),before)
        capture.count=128
        with self.assertRaises(ValueError):capture.add(dict(ok=True))
        self.assertEqual(stream.getvalue(),before)


if __name__=='__main__':unittest.main()
