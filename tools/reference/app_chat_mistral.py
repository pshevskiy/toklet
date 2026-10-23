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
    if hashlib.sha256(raw).hexdigest() != 'e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49':
        raise ValueError('Wrong reference tokenizer')
    if hashlib.sha256(config_raw).hexdigest() != 'e8fbcfef34d855dba2c7f1d91a2c5470cf493d00ecfd06488ce21e45325bc8ae':
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
        text = Environment(undefined=StrictUndefined).from_string(next(item['template'] for item in config['chat_template'] if item['name'] == 'default')).render(
            messages=messages, bos_token=config["bos_token"], eos_token=config["eos_token"], add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        cases.append(dict(messages=messages, rendered=text, ids=ids,
                          decoded=tokenizer.decode(ids, skip_special_tokens=True)))
    return dict(format='toklet-mistral-chat-reference-v1',
                repo='mlx-community/Mistral-7B-Instruct-v0.3-4bit',
                revision='a4b8f870474b0eb527f466a03fbc187830d271f5',
                tokenizer_sha256=hashlib.sha256(raw).hexdigest(),
                config_sha256=hashlib.sha256(config_raw).hexdigest(),
                template_sha256=hashlib.sha256(next(item['template'] for item in config['chat_template'] if item['name'] == 'default').encode()).hexdigest(),
                runtime={name: importlib.metadata.version(name) for name in ['jinja2', 'tokenizers']}, cases=cases)


if __name__ == '__main__':
    result = export(Path(sys.argv[1]))
    if len(sys.argv) == 3:
        if result != json.loads(Path(sys.argv[2]).read_bytes()):
            raise ValueError('Frozen independent reference differs')
        print('Independent chat reference matches')
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
