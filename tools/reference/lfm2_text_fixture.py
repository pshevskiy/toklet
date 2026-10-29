"""Bounded LFM2 codec reference; no model or numerical engine imports.

Manual rendering is independent of pinned Jinja. Runtime token IDs are exported
explicitly; parsing metadata alone never claims text or app support.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import os
import sysconfig

import text_reference_runtime

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'models/LFM2-350M-4bit-18dc72a'
REPOSITORY = 'mlx-community/LFM2-350M-4bit'
REVISION = '18dc72abf3b2337f9123cfd6eeeb58dfa7947066'
PINS = {
    'config.json': '95e6906d77ac647e8bf93c2d2bbd2f37eb9bf3fac8732a563653236ad81072af',
    'generation_config.json': '0f678f07a00ae4bdeb2b2bd5f37988a4b33a87686d9103fefe08de7a46522373',
    'chat_template.jinja': 'a805e50fed68938a076b07e2e602639611b50b1ced0e50f11eb92f1ba25be4dc',
    'tokenizer.json': '98cff83b4f6d7e9d8929bebc62b07e92cf1b3f99c80d16bafe8b84a75448f40b',
    'tokenizer_config.json': '36f511115e9d8952cbc9d15d9a20dfa7ce7d1444940e5c1dc42a762020c99bf5',
    'special_tokens_map.json': '742aefe2b7dec496e8caffdba03a75d0c1a9925d53bd3f3e0d388c96b591b6f4',
}
VERSIONS = {'jinja2': '3.1.6', 'markupsafe': '3.0.4', 'tokenizers': '0.22.2'}
BOS, IM_START, IM_END = '<|startoftext|>', '<|im_start|>', '<|im_end|>'
MAPPED, HEAD, MAX_BYTES = 64400, 65536, 65536
CASES = [
    ('empty_user', [{'role': 'user', 'content': ''}]),
    ('plain_user', [{'role': 'user', 'content': 'Say hello.'}]),
    ('system_user', [{'role': 'system', 'content': 'Be concise.'},
                     {'role': 'user', 'content': 'What is 2 + 2?'}]),
    ('assistant_history', [{'role': 'user', 'content': 'Name a color.'},
                           {'role': 'assistant', 'content': 'Blue.'},
                           {'role': 'user', 'content': 'Name another.'}]),
    ('unicode_user', [{'role': 'user', 'content': 'Café — 東京\nLine two.'}]),
]


def identity(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def checked_ids(ids):
    if (not isinstance(ids, list) or len(ids) > 512
            or any(type(token) is not int or not 0 <= token < MAPPED for token in ids)):
        raise ValueError('Unmapped or invalid codec token ID')
    return ids


def is_stop(token):
    checked_ids([token])
    return token == 7


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def parse_messages(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise ValueError('Message JSON exceeds64KiB')
    return strict_json(raw)


def render(messages, generation_prompt, reserved):
    if type(generation_prompt) is not bool:
        raise ValueError('Generation prompt flag must be Boolean')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        raise ValueError('Bounded nonempty message list required')
    parts, expected_role = [BOS], 'user'
    byte_count = len(BOS.encode('utf-8'))
    for index, message in enumerate(messages):
        if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                or type(message['role']) is not str or type(message['content']) is not str):
            raise ValueError('Message schema must be exactly string role/content')
        role, content = message['role'], message['content']
        if role == 'system' and index == 0:
            pass
        elif role == expected_role:
            expected_role = 'assistant' if role == 'user' else 'user'
        else:
            raise ValueError('Optional initial system then alternating user/assistant required')
        if len(content) > MAX_BYTES or any(literal in content for literal in reserved):
            raise ValueError('Reserved special-token literal or content bound')
        byte_count += (len(IM_START) + len(role) + 1 + len(content.encode('utf-8'))
                       + len(IM_END) + 1)
        if byte_count > MAX_BYTES:
            raise ValueError('Rendered conversation exceeds64KiB')
        parts.extend((IM_START, role, '\n', content, IM_END, '\n'))
    if messages[-1]['role'] == 'system':
        raise ValueError('Conversation must include a user message')
    if generation_prompt:
        parts.extend((IM_START, 'assistant\n'))
        byte_count += len(IM_START) + len('assistant\n')
    if byte_count > MAX_BYTES:
        raise ValueError('Rendered conversation exceeds64KiB')
    if len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) > MAX_BYTES:
        raise ValueError('Message JSON exceeds64KiB')
    text = ''.join(parts)
    if len(text.encode('utf-8')) != byte_count:
        raise ValueError('Rendered size accounting differs')
    return text


def runtime_closure():
    site = Path(sysconfig.get_path('purelib')).resolve()
    if not site.is_relative_to(ROOT):
        raise ValueError('Repository Python environment required')
    result = {}
    for package in ('jinja2', 'markupsafe', 'tokenizers'):
        folder = site / package
        if not folder.is_dir():
            raise ValueError('Missing codec package: ' + package)
        for path in sorted(folder.rglob('*')):
            if (path.is_file() and '__pycache__' not in path.parts
                    and path.suffix in ('.py', '.so', '.dylib')):
                result[str(path.relative_to(ROOT))] = identity(path)
    return result


class Capture:
    def __init__(self, output):
        self.path = output.with_suffix('.raw.jsonl')
        self.count = 0
        self.bytes = 0
        with self.path.open('xb'):
            pass

    def add(self, item):
        raw = (json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
        if self.count >= 600 or self.bytes + len(raw) > 8 * 1024**2:
            raise ValueError('Raw codec capture bound')
        with self.path.open('ab') as stream:
            if stream.write(raw) != len(raw):
                raise OSError('Short raw codec record')
        self.count += 1
        self.bytes += len(raw)


def export(output, admission, endpoint):
    if os.path.lexists(output):
        raise FileExistsError('Fresh output required')
    if any(importlib.metadata.version(name) != value for name, value in VERSIONS.items()):
        raise ValueError('Codec dependency version differs')
    assets = {}
    for name, digest in PINS.items():
        with (MODEL / name).open('rb') as stream:
            raw = stream.read(8 * 1024**2 + 1)
        if len(raw) > 8 * 1024**2 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Pinned codec asset differs: ' + name)
        assets[name] = raw
    before, producer = runtime_closure(), identity(Path(__file__))
    capture = Capture(output)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from tokenizers import Tokenizer
    from jinja2 import StrictUndefined
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    document = strict_json(assets['tokenizer.json'])
    config = strict_json(assets['tokenizer_config.json'])
    special = strict_json(assets['special_tokens_map.json'])
    model_config = strict_json(assets['config.json'])
    generation_config = strict_json(assets['generation_config.json'])
    if model_config['eos_token_id'] != 7 or generation_config['eos_token_id'] != 7:
        raise ValueError('Model/generation stop metadata differs')
    reserved = [token['content'] for token in document['added_tokens'] if token['special']]
    capture.add(dict(kind='structure', normalizer=document['normalizer'],
                     pre_tokenizer=document['pre_tokenizer'], post_processor=document['post_processor'],
                     decoder=document['decoder'], added_tokens=document['added_tokens'],
                     vocab_size=len(document['model']['vocab']), merges=len(document['model']['merges'])))
    if len(reserved) != 505 or len(set(reserved)) != 505:
        raise ValueError('Special-token inventory differs')
    tokenizer = Tokenizer.from_buffer(assets['tokenizer.json'])
    if (tokenizer.truncation is not None or tokenizer.padding is not None
            or set(tokenizer.get_vocab().values()) != set(range(MAPPED))):
        raise ValueError('Tokenizer mapping or modification differs')
    if (config['bos_token'] != BOS or config['eos_token'] != IM_END
            or config['add_bos_token'] is not True or config['add_eos_token'] is not False
            or special['bos_token']['content'] != BOS or special['eos_token']['content'] != IM_END):
        raise ValueError('BOS/stop metadata differs')
    for literal, token in ((BOS, 1), ('<|endoftext|>', 2), (IM_START, 6), (IM_END, 7)):
        if tokenizer.token_to_id(literal) != token:
            raise ValueError('Special-token mapping differs')
    ordinary = [token for token in document['added_tokens'] if not token['special']]
    if [(token['content'], token['id']) for token in ordinary] != [('Mathias', 64011), ('python', 64014)]:
        raise ValueError('Ordinary added-token metadata differs')
    for token in ordinary:
        text = render([dict(role='user', content=token['content'])], False, reserved)
        encoded = tokenizer.encode(token['content'], add_special_tokens=False).ids
        decoded = tokenizer.decode(checked_ids(encoded), skip_special_tokens=False)
        capture.add(dict(kind='ordinary_added', metadata=token, rendered=text,
                         ids=encoded, decoded=decoded))
        if encoded != [token['id']] or decoded != token['content']:
            raise ValueError('Ordinary added-token behavior differs')
    environment = ImmutableSandboxedEnvironment(undefined=StrictUndefined, autoescape=False,
                                                 loader=None, cache_size=0,
                                                 trim_blocks=True, lstrip_blocks=True)
    environment.globals.clear()
    template = environment.from_string(assets['chat_template.jinja'].decode())
    rows = []
    for name, messages in CASES:
        for generation in (False, True):
            manual = render(messages, generation, reserved)
            independent = template.render(messages=messages, bos_token=BOS,
                                          add_generation_prompt=generation)
            ids = tokenizer.encode(manual, add_special_tokens=False).ids
            doubled = tokenizer.encode(manual, add_special_tokens=True).ids
            capture.add(dict(kind='chat_operands', name=name, add_generation_prompt=generation,
                             manual=manual, template=independent, ids=ids,
                             postprocessor_ids=doubled))
            checked_ids(ids)
            row = dict(name=name, messages=messages, add_generation_prompt=generation,
                       rendered=manual, template_rendered=independent, ids=ids,
                       decoded=tokenizer.decode(ids, skip_special_tokens=False),
                       postprocessor_ids=doubled)
            capture.add(dict(kind='chat', **row))
            if manual != independent:
                raise ValueError('Manual/template bytes differ: ' + name)
            if not ids or ids[0] != 1 or ids.count(1) != 1 or doubled != [1] + ids:
                raise ValueError('BOS ownership differs')
            rows.append(row)
    failures = []
    for token in (-1, True, 64400, 65535, 65536):
        try:
            checked_ids([token])
        except ValueError:
            failures.append(dict(token=token, rejected_before_decode=True))
            capture.add(dict(kind='invalid_id', **failures[-1]))
        else:
            raise ValueError('Invalid token was admitted')
    for literal in reserved:
        rejected = False
        try:
            render([dict(role='user', content='prefix' + literal)], True, reserved)
        except ValueError:
            rejected = True
        capture.add(dict(kind='reserved_literal', literal=literal, rejected=rejected))
        if not rejected:
            raise ValueError('Special-token injection admitted')
    plain = []
    for text in ('', 'Hello, world!', ' a  b\n', 'Привет 👋 café é'):
        ids = checked_ids([1] + tokenizer.encode(text, add_special_tokens=False).ids)
        if ids.count(1) != 1:
            raise ValueError('Plain input BOS duplicated')
        plain.append(dict(text=text, ids=ids,
                          decoded=tokenizer.decode(ids, skip_special_tokens=False)))
        capture.add(dict(kind='plain', **plain[-1]))
    mapped_boundaries = checked_ids([0, 64399])
    boundary_decode = tokenizer.decode(mapped_boundaries, skip_special_tokens=False)
    stop_rows = [dict(token=token, is_stop=is_stop(token),
                      decoded=tokenizer.decode([token], skip_special_tokens=False))
                 for token in (2, 7)]
    capture.add(dict(kind='mapped_boundaries', ids=mapped_boundaries, decoded=boundary_decode))
    capture.add(dict(kind='stops', rows=stop_rows))
    if is_stop(2) or not is_stop(7):
        raise ValueError('Stop union differs')
    if runtime_closure() != before or identity(Path(__file__)) != producer:
        raise ValueError('Producer/runtime changed')
    if any(identity(MODEL / name)['sha256'] != digest for name, digest in PINS.items()):
        raise ValueError('Codec asset changed')
    stable_admission = {key: admission[key] for key in ('manifest_sha256', 'files', 'versions', 'imports')}
    endpoint()
    result = dict(format='toklet-lfm2-text-reference-v1', repository=REPOSITORY,
                  revision=REVISION, file_pins=PINS, producer=producer, versions=VERSIONS,
                  runtime=before, runtime_admission=stable_admission,
                  raw_capture=identity(capture.path), raw_records=capture.count,
                  cases=rows, plain_cases=plain, stop_rows=stop_rows,
                  mapped_boundary_decode=dict(ids=mapped_boundaries, decoded=boundary_decode),
                  rejected_ids=failures, rejected_reserved_literals=len(reserved),
                  stop_ids=[7], add_special_tokens=False,
                  decode_policy='tokenizers raw ByteLevel, skip_special_tokens=False; no Transformers cleanup',
                  clean_up_tokenization_spaces=False, transformers_executed=False,
                  model_executed=False, app_qualified=False)
    raw = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > 4 * 1024**2:
        raise ValueError('Codec fixture exceeds4MiB')
    with output.open('xb') as stream:
        stream.write(raw)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-sha256', required=True)
    args = parser.parse_args()
    admission, endpoint = text_reference_runtime.admit(
        args.runtime, args.runtime_sha256,
        [Path(__file__), Path(__file__).with_name('test_lfm2_text_fixture.py')])
    observation = dict(runtime_admission=admission, passed=False, error=None, endpoint_error=None)
    try:
        export(args.output, admission, endpoint)
        observation['passed'] = True
    except BaseException as error:
        observation['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        try:
            observation['endpoint'] = endpoint()
        except BaseException as error:
            observation['endpoint_error'] = type(error).__name__ + ': ' + str(error)
            observation['passed'] = False
            raise
        finally:
            raw = (json.dumps(observation, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
            if len(raw) > 4 * 1024**2:
                raise ValueError('Codec runtime observation exceeds4MiB')
            with args.output.with_suffix('.observation.json').open('xb') as stream:
                stream.write(raw)
