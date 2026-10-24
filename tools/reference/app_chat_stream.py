"""Independent decode witnesses, with no target-engine imports or downloads.

Use already acquired, exact tokenizers. Complete-prefix and final text come from
Python tokenizers, not the target streaming algorithm. Compact assets retain
original token IDs/decoder semantics for decode-only offline Rust fixtures.
"""
from pathlib import Path
import copy
import hashlib
import importlib.metadata
import json
import sys

MAX_ASSET_BYTES = 8 * 1024 * 1024
LONG_TOKENS = 131072
PROFILES = {
    'smol': ('MLX-SmolLM2-135M-Instruct-8bit-0f0d9b8', 'mlx-community/SmolLM2-135M-Instruct-8bit',
             '0f0d9b8218915bc34d401e1a340b8c049d300d5e', '7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7'),
    'qwen': ('MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c', 'mlx-community/Qwen2.5-0.5B-Instruct-8bit',
             '6474d4c1a05bd7bd634a9c993560e4e7272eda3d', 'a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf'),
    'mistral': ('MLX-Mistral-7B-Instruct-v0.3-4bit-a4b8f87', 'mlx-community/Mistral-7B-Instruct-v0.3-4bit',
                'a4b8f870474b0eb527f466a03fbc187830d271f5', 'e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49'),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def serialize(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def admitted(path, expected):
    with path.open('rb') as stream:
        raw = stream.read(MAX_ASSET_BYTES + 1)
    if len(raw) > MAX_ASSET_BYTES or digest(raw) != expected:
        raise ValueError('Wrong or oversized pinned tokenizer: ' + str(path))
    return raw


def byte_alphabet():
    # Only selects byte-token IDs. It is never used to compute expected text.
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    values = direct + [byte for byte in range(256) if byte not in direct]
    chars = direct + list(range(256, 256 + 256 - len(direct)))
    return dict(zip(values, map(chr, chars)))


def compact(raw, used):
    source = json.loads(raw)
    if source['model']['type'] != 'BPE' or source['truncation'] or source['padding']:
        raise ValueError('Unexpected producer tokenizer policy')
    model = copy.deepcopy(source['model'])
    unknown = model.get('unk_token')
    model['vocab'] = {text: index for text, index in model['vocab'].items() if index in used or text == unknown}
    model['merges'] = []
    return serialize(dict(version='1.0', truncation=None, padding=None,
                          added_tokens=[item for item in source['added_tokens'] if item['id'] in used],
                          normalizer=None, pre_tokenizer=None, post_processor=None,
                          decoder=source['decoder'], model=model))


def export(models):
    from tokenizers import Tokenizer
    outputs = {}
    profiles = {}
    for key, (folder, repo, revision, pin) in PROFILES.items():
        raw = admitted(models / folder / 'tokenizer.json', pin)
        tokenizer = Tokenizer.from_str(raw.decode())
        def encode(text):
            return tokenizer.encode(text, add_special_tokens=False).ids
        def byte_id(byte):
            literal = f'<0x{byte:02X}>' if key == 'mistral' else byte_alphabet()[byte]
            index = tokenizer.token_to_id(literal)
            if index is None:
                raise ValueError('Missing byte token: ' + literal)
            return index
        specials = [item['id'] for item in json.loads(raw)['added_tokens'] if item['special']][:3]
        if len(specials) != 3:
            raise ValueError('Missing control witness')
        # Standalone continuation byte is invalid UTF-8 and exists in all three
        # vocabularies; SmolLM2 does not contain every impossible UTF-8 byte.
        a, invalid_byte = byte_id(65), byte_id(128)
        globe = [byte_id(byte) for byte in '🌍'.encode()]
        inputs = [
            ('empty', []), ('controls_only', specials),
            ('unicode', encode('Hello, 🌍! Привет. 中文 e\u0301\n')),
            ('spaces', encode('  A A\n\n\tB  ')),
            ('code', encode('```rust\nlet x = "🌍";\n```\n')),
            ('literal_replacement', encode('�')),
            ('split_globe', globe), ('unfinished_globe', globe[:-1]),
            ('invalid_bytes', [a, invalid_byte, byte_id(66), byte_id(226), byte_id(130)]),
            ('skipped_controls_between_bytes', [globe[0], specials[0], *globe[1:]]),
            ('run_then_invalid', [a] * 129 + [invalid_byte]),
        ]
        cases = []
        used = set()
        for name, ids in inputs:
            used.update(ids)
            cases.append(dict(name=name, ids=ids, prefixes=[tokenizer.decode(ids[:end], skip_special_tokens=True) for end in range(len(ids) + 1)]))
        word = tokenizer.token_to_id('▁A' if key == 'mistral' else 'A')
        if word is None:
            raise ValueError('Missing long word token')
        replacement = encode('�')
        if not replacement or len(replacement) > 16:
            raise ValueError('Unexpected literal replacement encoding')
        long_inputs = [('word', [word], []), ('literal_replacement', replacement, [])]
        if key == 'mistral':
            long_inputs.extend([('fallback_ascii', [a], []), ('fallback_invalid_terminal', [a], [invalid_byte])])
        long_cases = []
        for name, pattern, tail in long_inputs:
            count, remainder = divmod(LONG_TOKENS - len(tail), len(pattern))
            tail = [a] * remainder + tail
            ids = pattern * count + tail
            text = tokenizer.decode(ids, skip_special_tokens=True)
            used.update(ids)
            long_cases.append(dict(name=name, repeat_ids=pattern, repeat_count=count, tail_ids=tail,
                                   total_tokens=len(ids), utf8_bytes=len(text.encode()), text_sha256=digest(text.encode())))
        small_raw = compact(raw, used)
        small = Tokenizer.from_str(small_raw.decode())
        for case in cases:
            if [small.decode(case['ids'][:end], skip_special_tokens=True) for end in range(len(case['ids']) + 1)] != case['prefixes']:
                raise ValueError('Compact tokenizer differs: ' + key + '/' + case['name'])
        for case in long_cases:
            ids = case['repeat_ids'] * case['repeat_count'] + case['tail_ids']
            text = small.decode(ids, skip_special_tokens=True).encode()
            if len(text) != case['utf8_bytes'] or digest(text) != case['text_sha256']:
                raise ValueError('Compact long tokenizer differs')
        name = key + '-tokenizer.json'
        outputs[name] = small_raw
        profiles[key] = dict(repo=repo, revision=revision, tokenizer_sha256=pin,
                             compact_tokenizer=name, compact_sha256=digest(small_raw),
                             compact_is_decode_only=True, cases=cases, long_cases=long_cases)
    outputs['reference.json'] = serialize(dict(format='toklet-chat-stream-oracle-v1',
        producer_sha256=digest(Path(__file__).read_bytes()), runtime=dict(tokenizers=importlib.metadata.version('tokenizers')),
        numerical_generation_qualified=False, append_only_stability_assumed=False, profiles=profiles))
    return outputs


def publish(outputs, destination):
    # Preflight the full output set before writing any file. An interrupted
    # publication is incomplete and must use a fresh directory on retry.
    if not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('Empty existing output directory required')
    for name in outputs:
        if Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('Invalid fixture name')
    for name, raw in outputs.items():
        with (destination / name).open('xb') as stream:
            stream.write(raw)


def verify(outputs, destination):
    if {path.name for path in destination.iterdir()} != set(outputs):
        raise ValueError('Fixture inventory differs')
    for name, raw in outputs.items():
        if (destination / name).read_bytes() != raw:
            raise ValueError('Frozen fixture differs: ' + name)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in ('export', 'verify'):
        raise SystemExit('Use: app_chat_stream.py export|verify MODELS EMPTY_OUTPUT_OR_FIXTURE')
    outputs = export(Path(sys.argv[2]))
    (publish if sys.argv[1] == 'export' else verify)(outputs, Path(sys.argv[3]))
    print(json.dumps(dict(passed=True, profiles=3, short_cases=33, long_cases=8,
                          long_tokens=LONG_TOKENS, numerical_generation_qualified=False)))
