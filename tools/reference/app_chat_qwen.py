"""Reproduce the frozen chat-only reference from an already acquired tokenizer.

No downloads or target-engine imports. Jinja renders the original pinned template;
the independent Python tokenizers binding supplies complete IDs and decoded text.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer


def export(directory):
    raw = (directory / 'tokenizer.json').read_bytes()
    config_raw = (directory / 'tokenizer_config.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf':
        raise ValueError('Wrong reference tokenizer')
    if hashlib.sha256(config_raw).hexdigest() != 'f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7':
        raise ValueError('Wrong reference chat configuration')
    config = json.loads(config_raw)
    tokenizer = Tokenizer.from_str(raw.decode())
    cases = []
    for messages in [
        [{'role': 'user', 'content': 'Hello!'}],
        [{'role': 'user', 'content': 'Hello, 🌍! Привет.'},
         {'role': 'assistant', 'content': 'Hello! How can I help?'},
         {'role': 'user', 'content': 'What is 2 + 2?'}],
    ]:
        text = Environment(undefined=StrictUndefined).from_string(config['chat_template']).render(
            messages=[dict(message, tool_calls=None) for message in messages],
            tools=None, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        cases.append(dict(messages=messages, rendered=text, ids=ids,
                          decoded=tokenizer.decode(ids, skip_special_tokens=True)))
    return dict(format='toklet-qwen-chat-reference-v1',
                repo='mlx-community/Qwen2.5-0.5B-Instruct-8bit',
                revision='6474d4c1a05bd7bd634a9c993560e4e7272eda3d',
                tokenizer_sha256=hashlib.sha256(raw).hexdigest(),
                config_sha256=hashlib.sha256(config_raw).hexdigest(),
                template_sha256=hashlib.sha256(config['chat_template'].encode()).hexdigest(),
                runtime={name: importlib.metadata.version(name) for name in ['jinja2', 'tokenizers']}, cases=cases)


if __name__ == '__main__':
    result = export(Path(sys.argv[1]))
    if len(sys.argv) == 3:
        if result != json.loads(Path(sys.argv[2]).read_bytes()):
            raise ValueError('Frozen independent reference differs')
        print('Independent chat reference matches')
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
