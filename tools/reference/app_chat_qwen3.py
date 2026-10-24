"""Independent pinned Qwen3 Chat formatting, tokenization and decoding witness.

The original template runs only in this offline reference producer. Production
supports its alternating plain-text, non-thinking branch without remote Jinja.
"""
import argparse
import hashlib
import json
from pathlib import Path

import chat_reference_runtime as runtime

REPO = 'mlx-community/Qwen3-0.6B-8bit'
REVISION = '11de96878523501bcaa86104e3c186de07ff9068'
PINS = {
    'config.json': '9f387dcb1bf045cadb34a36e67569abd73c00e5322603df2b19d0c0f8cc9d240',
    'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json': '8e7dd4250db58b2a29792562d727b2715dc0fe79a94b446caa4c07db39eceba6',
}
BOUNDS = {'config.json': 16384, 'tokenizer.json': 16 * 1024 * 1024, 'tokenizer_config.json': 16384}
CASES = [
    [{'role': 'user', 'content': 'Hello!'}],
    [{'role': 'user', 'content': 'Hello, 🌍! Привет.'},
     {'role': 'assistant', 'content': 'Hello! How can I help?'},
     {'role': 'user', 'content': 'What is 2 + 2?'}],
    [{'role': 'user', 'content': 'Write a Python function add(a, b).'},
     {'role': 'assistant', 'content': '```python\ndef add(a, b):\n    return a + b\n```'},
     {'role': 'user', 'content': 'Explain the return statement.'}],
    [{'role': 'user', 'content': 'First question'},
     {'role': 'assistant', 'content': '\n\nFirst answer\n'},
     {'role': 'user', 'content': 'Second question'},
     {'role': 'assistant', 'content': 'Second answer'},
     {'role': 'user', 'content': 'Continue.'}],
    [{'role': 'user', 'content': '你好！ é 👨‍👩‍👧‍👦\nTabs\tand quotes: "hello".'}],
]


def capture(directory):
    runtime.check()
    from jinja2 import Environment, StrictUndefined
    from tokenizers import Tokenizer
    raw = {}
    for name, pin in PINS.items():
        with (directory / name).open('rb') as stream:
            data = stream.read(BOUNDS[name] + 1)
        if len(data) > BOUNDS[name] or hashlib.sha256(data).hexdigest() != pin:
            raise ValueError('Pinned reference asset differs: ' + name)
        raw[name] = data
    config = json.loads(raw['tokenizer_config.json'])
    model = json.loads(raw['config.json'])
    tokenizer = Tokenizer.from_str(raw['tokenizer.json'].decode())
    template = Environment(undefined=StrictUndefined).from_string(config['chat_template'])

    def render(messages, thinking):
        return template.render(messages=[dict(m, tool_calls=None, reasoning_content=None) for m in messages],
                               tools=None, add_generation_prompt=True, enable_thinking=thinking)

    cases = []
    for messages in CASES:
        rendered = render(messages, False)
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        cases.append(dict(messages=messages, rendered=rendered, ids=ids,
                          decoded=tokenizer.decode(ids, skip_special_tokens=True)))
    decode = []
    for text in ['Hello!', '你好，🌍! é', '```python\ndef add(a, b):\n    return a + b\n```', '<think>work</think> <tool_call>call</tool_call>']:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        decode.append(dict(text=text, ids=ids, decoded=tokenizer.decode(ids, skip_special_tokens=True),
                           prefixes=[tokenizer.decode(ids[:i], skip_special_tokens=True) for i in range(len(ids) + 1)]))
    declared = {int(key): (item['content'], item['special']) for key, item in config['added_tokens_decoder'].items()}
    actual = {key: (item.content, item.special) for key, item in tokenizer.get_added_tokens_decoder().items()}
    if len(declared) != 26 or declared != actual or any(tokenizer.token_to_id(text) != token for token, (text, _) in declared.items()):
        raise ValueError('Exact added-token identity and original special flags required')
    special = sorted((text, token, flag) for token, (text, flag) in declared.items())
    eos = tokenizer.token_to_id(config['eos_token'])
    if eos != model['eos_token_id'] or eos != 151645:
        raise ValueError('Independent EOS sources disagree')
    return dict(format='toklet-qwen3-chat-reference-v1', repository=REPO, revision=REVISION,
                asset_sha256=PINS, template_sha256=hashlib.sha256(config['chat_template'].encode()).hexdigest(),
                runtime=runtime.VERSIONS,
                mode='non-thinking plain alternating user/assistant, no system or tools', stop_ids=[eos],
                thinking_token_ids=[tokenizer.token_to_id('<think>'), tokenizer.token_to_id('</think>')],
                special_literals=[dict(text=text, id=token, special=flag) for text, token, flag in special], cases=cases,
                decode_cases=decode, thinking_counterexample=render(CASES[0], True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--check-sha256')
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--owner-source', type=Path, required=True)
    args = parser.parse_args()
    if args.output and args.check:
        parser.error('Choose output or check')
    admission = runtime.admit(args.runtime, args.runtime_sha256, [Path(__file__).resolve(), args.owner_source.resolve()])
    result = capture(args.directory)
    runtime.check()
    if args.check:
        if result != json.loads(runtime.captured(args.check, 1024 * 1024, args.check_sha256)):
            raise ValueError('Frozen independent reference differs')
        print('Independent Qwen3 Chat reference matches')
        print(json.dumps(admission, sort_keys=True))
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print('Independent Qwen3 Chat reference captured')
        print(json.dumps(admission, sort_keys=True))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
