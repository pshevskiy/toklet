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
    if hashlib.sha256(raw).hexdigest() != '7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7':
        raise ValueError('Wrong reference tokenizer')
    if hashlib.sha256(config_raw).hexdigest() != 'a27f638bd2831f5c3dea654a75838930f2b11fbe550c4d4e1d5d7bd07157b2ee':
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
            messages=messages, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        cases.append(dict(messages=messages, rendered=text, ids=ids,
                          decoded=tokenizer.decode(ids, skip_special_tokens=True)))
    return dict(format='toklet-smollm-chat-reference-v1',
                repo='mlx-community/SmolLM2-135M-Instruct-8bit',
                revision='0f0d9b8218915bc34d401e1a340b8c049d300d5e',
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
