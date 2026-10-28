"""Pinned metadata-only Phi3 plain-text reference, no model or native execution.

The Jinja template is an offline reference only. This does not implement native
Chat or a streaming decoder. Its bounded decoder validates every ID before
calling tokenizers, so unknown head slots cannot silently disappear.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'models/MLX-Phi-3-mini-4k-instruct-4bit-5b3819e'
REPOSITORY = 'mlx-community/Phi-3-mini-4k-instruct-4bit'
REVISION = '5b3819ed6317784fb20eddeae9bed984f778d0d0'
PINS = {
    'config.json': (1030, '0e2e43bc4358b4cabbcc33c496f34e170fdfe04612a47428f1691d1e9ec5a568'),
    'tokenizer.json': (1844436, 'd0f067e1e15cd0a36ebef3668024882cb67a80b86fb4b7b4b128481f0d474db7'),
    'tokenizer_config.json': (3333, 'd6e13c85fbde9cf71f663da027cf558ab2bb9df80bd60c718be10dbba8d2a2be'),
    'special_tokens_map.json': (569, '474d699677e264edd3851559548cbbb61801ad8da6b249c81455007602313c34'),
}
VERSIONS = {'jinja2': '3.1.6', 'markupsafe': '3.0.4', 'tokenizers': '0.22.2'}
ADDED = {'<unk>': 0, '<s>': 1, '</s>': 2, '<|endoftext|>': 32000,
    '<|assistant|>': 32001, '<|placeholder1|>': 32002, '<|placeholder2|>': 32003,
    '<|placeholder3|>': 32004, '<|placeholder4|>': 32005, '<|system|>': 32006,
    '<|end|>': 32007, '<|placeholder5|>': 32008, '<|placeholder6|>': 32009,
    '<|user|>': 32010}
STOPS = (32000, 32007)
HEAD_SIZE = 32064
MAPPED_SIZE = 32011
MAX_TEXT_BYTES = 65536
MAX_TOKENS = 512
MAX_OUTPUT_BYTES = 2 * 1024**2
CASES = {
    'single': [{'role': 'user', 'content': 'Hello!'}],
    'multi': [{'role': 'user', 'content': 'Count to two.'},
              {'role': 'assistant', 'content': 'One, two.'},
              {'role': 'user', 'content': 'Now backwards.'}],
    'system': [{'role': 'system', 'content': 'Answer briefly.'},
               {'role': 'user', 'content': 'What is two plus two?'}],
    'empty_content': [{'role': 'user', 'content': ''}],
    'whitespace': [{'role': 'user', 'content': ' \t leading  and trailing \t '}],
    'newlines_tabs': [{'role': 'user', 'content': 'line one\n\n\tline two\r\n'}],
    'combining_unicode': [{'role': 'user', 'content': 'café e\u0301 — Привет'}],
    'emoji_bytefallback': [{'role': 'user', 'content': '👋 🧪 𐍈'}],
}
RAW_TEXTS = ('', 'Hello, world!', ' leading  trailing ', '\n\tline\r\n', 'café e\u0301', '👋🧪𐍈')
BYTE_TEXTS = ('é', 'e\u0301', '👋', '🧪', '𐍈', '\n\t', ' x')
BOUNDARY_LITERALS = ('<|user|>', '<|assistant|>', '<|system|>', '<|end|>', '</s>', '<|endoftext|>')


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def pin(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': Path(path).stat().st_size, 'sha256': digest}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def duplicate_checked(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def render(messages, add_generation_prompt):
    """Exact plain subset: optional system, then alternating user/assistant.

    Empty content is allowed; an empty conversation or system-only conversation
    is not. Non-generation rendering may end in user or assistant.
    """
    if type(add_generation_prompt) is not bool or type(messages) is not list or not 1 <= len(messages) <= 64:
        raise ValueError('Bounded nonempty plain conversation and boolean generation flag required')
    try:
        if len(json.dumps(messages, ensure_ascii=False, allow_nan=False).encode('utf-8')) > MAX_TEXT_BYTES:
            raise ValueError('Conversation exceeds64KiB')
    except (UnicodeEncodeError, TypeError) as error:
        raise ValueError('Conversation is not valid UTF8 plain metadata') from error
    output = []
    next_role = 'user'
    user_seen = False
    for index, message in enumerate(messages):
        if type(message) is not dict or set(message) != {'role', 'content'}:
            raise ValueError('Exact role/content fields required; tools and extra fields unsupported')
        role, content = message['role'], message['content']
        if type(content) is not str or any(literal in content for literal in ADDED):
            raise ValueError('Plain string content cannot contain reserved added-token literals')
        if index == 0 and role == 'system':
            pass
        elif role == next_role:
            user_seen |= role == 'user'
            next_role = 'assistant' if role == 'user' else 'user'
        else:
            raise ValueError('Role sequence must be optional system, user, assistant, user, ...')
        output.append('<|' + role + '|>\n' + content + '<|end|>\n')
    if not user_seen or (add_generation_prompt and messages[-1]['role'] != 'user'):
        raise ValueError('Generation ends in user; every conversation needs a user')
    output.append('<|assistant|>\n' if add_generation_prompt else '<|end|>')
    text = ''.join(output)
    if len(text.encode('utf-8')) > MAX_TEXT_BYTES:
        raise ValueError('Rendered conversation exceeds64KiB')
    return text


def decode_bounded(tokenizer, ids):
    """No mutation or output acceptance before complete ID validation/decode.

    Valid raw inference IDs in unmapped head slots deliberately reject here.
    This pure reference helper is not a native session rollback implementation.
    """
    if type(ids) not in (list, tuple) or len(ids) > MAX_TOKENS:
        raise ValueError('Bounded ID sequence required')
    for token in ids:
        if type(token) is not int or not 0 <= token < HEAD_SIZE:
            raise ValueError('ID outside numerical head')
        if token >= MAPPED_SIZE or tokenizer.id_to_token(token) is None:
            raise ValueError('Unmapped numerical ID cannot be accepted as text')
    result = tokenizer.decode(list(ids), skip_special_tokens=False)
    if type(result) is not str or len(result.encode('utf-8')) > MAX_TEXT_BYTES:
        raise ValueError('Decoded text exceeds bounded UTF8 contract')
    return result


def byte_fallback_ids(text, vocabulary):
    """Independent literal UTF8 byte sequence, not tokenizer.encode output."""
    return [vocabulary[f'<0x{byte:02X}>'] for byte in text.encode('utf-8')]


def direct_byte_expected(text):
    # ByteFallback -> Fuse -> Strip(' ',start=1,stop=0), on these direct IDs.
    # This narrow decoder control is not a general text encoding roundtrip.
    return text[1:] if text.startswith(' ') else text


def runtime_inventory():
    site = Path(sysconfig.get_path('purelib')).resolve()
    if not site.is_relative_to(ROOT):
        raise ValueError('Pinned repository virtual environment required')
    paths = {Path(sys.executable).resolve()}
    framework = Path(sys.base_prefix) / 'Python'
    if framework.is_file():
        paths.add(framework.resolve())
    for package, version in VERSIONS.items():
        distribution = importlib.metadata.distribution(package)
        if distribution.version != version:
            raise ValueError('Package version differs: ' + package)
        folder = site / package
        if not folder.is_dir():
            raise ValueError('Missing package directory: ' + package)
        paths.update(p.resolve() for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        metadata = [Path(distribution.locate_file(item)).resolve() for item in (distribution.files or [])
                    if any(part.endswith('.dist-info') for part in item.parts)]
        if not metadata:
            raise ValueError('Distribution metadata inventory absent: ' + package)
        if any(not path.is_relative_to(site) for path in metadata):
            raise ValueError('Distribution metadata escapes selected runtime')
        paths.update(metadata)
        for library in site.glob(package + '*.libs'):
            paths.update(p.resolve() for p in library.rglob('*') if p.is_file())
    # The owned controller additionally pins the complete Python stdlib before
    # starting this process and supplies a fresh pycache prefix with -I -B.
    return {str(p): pin(p) for p in sorted(paths)}


def assets():
    result = {}
    for name, (size, digest) in PINS.items():
        path = MODEL / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Metadata must be a regular non-symlink file: ' + name)
        with path.open('rb') as stream:
            raw = stream.read(size + 1)
        if identity(raw) != {'bytes': size, 'sha256': digest}:
            raise ValueError('Pinned consumed asset differs: ' + name)
        result[name] = duplicate_checked(raw)
    return result


def export(output):
    output = Path(output)
    progress_path = output.with_name(output.name + '.progress.jsonl')
    failure_path = output.with_name(output.name + '.failure.json')
    if any(os.path.lexists(p) for p in (output, progress_path, failure_path)):
        raise FileExistsError('Fresh output/progress/failure paths required')
    producer = pin(Path(__file__))
    try:
        before = runtime_inventory()
        data = assets()
        config, model, vocabulary = data['tokenizer_config.json'], data['config.json'], data['tokenizer.json']
        if (model['vocab_size'], model['eos_token_id'], config['eos_token'], config['bos_token']) != (32064, 32000, '<|end|>', '<s>'):
            raise ValueError('Head/BOS/EOS declarations differ')
        if config['add_bos_token'] is not False or config['add_eos_token'] is not False or config['clean_up_tokenization_spaces'] is not False:
            raise ValueError('Unexpected tokenizer modification settings')
        if data['special_tokens_map.json']['eos_token']['content'] != '<|endoftext|>':
            raise ValueError('Original special-map EOS declaration differs')
        processor = {'type': 'TemplateProcessing', 'single': [{'Sequence': {'id': 'A', 'type_id': 0}}],
                     'pair': [{'Sequence': {'id': 'A', 'type_id': 0}}, {'Sequence': {'id': 'B', 'type_id': 1}}], 'special_tokens': {}}
        if vocabulary['post_processor'] != processor or vocabulary['model']['type'] != 'BPE' or vocabulary['model']['byte_fallback'] is not True:
            raise ValueError('Pinned no-BOS postprocessor or byte fallback differs')
        normalizer = {'type': 'Sequence', 'normalizers': [{'type': 'Prepend', 'prepend': '▁'},
                       {'type': 'Replace', 'pattern': {'String': ' '}, 'content': '▁'}]}
        decoder = {'type': 'Sequence', 'decoders': [{'type': 'Replace', 'pattern': {'String': '▁'}, 'content': ' '},
                   {'type': 'ByteFallback'}, {'type': 'Fuse'}, {'type': 'Strip', 'content': ' ', 'start': 1, 'stop': 0}]}
        if vocabulary['normalizer'] != normalizer or vocabulary['decoder'] != decoder:
            raise ValueError('Pinned metaspace/decoder sequence differs')
        if {v['content']: int(k) for k, v in config['added_tokens_decoder'].items()} != ADDED:
            raise ValueError('Reserved added-token inventory differs')
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        from tokenizers import Tokenizer
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        tokenizer = Tokenizer.from_str(json.dumps(vocabulary, ensure_ascii=False))
        if tokenizer.padding is not None or tokenizer.truncation is not None:
            raise ValueError('Padding/truncation must remain disabled')
        full_vocab = tokenizer.get_vocab(with_added_tokens=True)
        if set(full_vocab.values()) != set(range(MAPPED_SIZE)) or tokenizer.get_vocab_size(with_added_tokens=True) != MAPPED_SIZE:
            raise ValueError('Mapped vocabulary coverage differs')
        env = ImmutableSandboxedEnvironment(undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
        env.globals.clear()
        template = env.from_string(config['chat_template'])
        rows, raw_rows, added_rows, byte_rows, boundaries, unmapped = [], [], [], [], [], []
        with progress_path.open('x', encoding='utf-8') as progress:
            def record(event, row):
                progress.write(json.dumps({'event': event, 'row': row}, ensure_ascii=False, allow_nan=False) + '\n')
                progress.flush()
            for name, messages in CASES.items():
                for generation in (False, True):
                    manual = render(messages, generation)
                    independent = template.render(messages=messages, add_generation_prompt=generation, eos_token=config['eos_token'])
                    if manual != independent:
                        raise ValueError('Independent template bytes differ: ' + name)
                    ids = tokenizer.encode(manual, add_special_tokens=False).ids
                    processed = tokenizer.encode(manual, add_special_tokens=True).ids
                    if not ids or ids != processed or 1 in ids:
                        raise ValueError('Automatic BOS/EOS or special-processing identity differs')
                    decoded = decode_bounded(tokenizer, ids)
                    row = dict(name=name, add_generation_prompt=generation, messages=messages,
                               rendered=manual, rendered_utf8=identity(manual.encode('utf-8')), ids=ids,
                               with_special_processing_ids=processed, decoded_with_specials=decoded,
                               decoded_utf8=identity(decoded.encode('utf-8')), rendered_equals_decoded=(manual == decoded))
                    rows.append(row); record('case', row)
            for text in RAW_TEXTS:
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if ids != tokenizer.encode(text, add_special_tokens=True).ids:
                    raise ValueError('Raw text postprocessing inserted tokens')
                row = dict(text=text, ids=ids, decoded_with_specials=decode_bounded(tokenizer, ids))
                raw_rows.append(row); record('raw', row)
            for literal, token in ADDED.items():
                if tokenizer.token_to_id(literal) != token or tokenizer.id_to_token(token) != literal:
                    raise ValueError('Added-token identity differs')
                if tokenizer.encode(literal, add_special_tokens=False).ids != [token] or tokenizer.encode(literal, add_special_tokens=True).ids != [token]:
                    raise ValueError('Literal special token ownership differs')
                decoded = decode_bounded(tokenizer, [token])
                if decoded != literal:
                    raise ValueError('Non-skipping literal decode differs')
                row = dict(id=token, literal=literal, stop=(token in STOPS), decoded_with_specials=decoded,
                           rstrip=config['added_tokens_decoder'][str(token)]['rstrip'])
                added_rows.append(row); record('added_token', row)
            for byte in range(256):
                literal = f'<0x{byte:02X}>'
                if vocabulary['model']['vocab'][literal] != byte + 3 or tokenizer.id_to_token(byte + 3) != literal:
                    raise ValueError('Literal byte-token mapping differs')
            for text in BYTE_TEXTS:
                ids = byte_fallback_ids(text, vocabulary['model']['vocab'])
                decoded = decode_bounded(tokenizer, ids)
                if decoded != direct_byte_expected(text):
                    raise ValueError('Independent UTF8 byte-fallback/Fuse/Strip control differs')
                row = dict(text=text, utf8=list(text.encode('utf-8')), ids=ids, decoded_with_specials=decoded,
                           strip_one_leading_space=text.startswith(' '))
                byte_rows.append(row); record('byte_fallback', row)
            for literal in BOUNDARY_LITERALS:
                token = ADDED[literal]
                content = literal + '\n\t  x'
                ids = tokenizer.encode(content, add_special_tokens=False).ids
                compact_ids = tokenizer.encode(literal + 'x', add_special_tokens=False).ids
                rstrip = config['added_tokens_decoder'][str(token)]['rstrip']
                if rstrip and ids != compact_ids:
                    raise ValueError('Added-token rstrip boundary differs')
                row = dict(text=content, literal=literal, id=token, rstrip=rstrip, ids=ids,
                           no_whitespace_ids=compact_ids, decoded_with_specials=decode_bounded(tokenizer, ids))
                boundaries.append(row); record('special_boundary', row)
            for token in range(MAPPED_SIZE, HEAD_SIZE):
                if tokenizer.id_to_token(token) is not None:
                    raise ValueError('Expected unmapped head slot is mapped')
                try:
                    decode_bounded(tokenizer, [token])
                except ValueError:
                    unmapped.append(token)
                else:
                    raise ValueError('Unmapped token was silently accepted')
            record('unmapped_rejected', unmapped)
        if len(rows) != 16 or len(unmapped) != 53 or set(STOPS) != {model['eos_token_id'], tokenizer.token_to_id(config['eos_token'])}:
            raise ValueError('Fixed case/stop/unmapped inventory differs')
        if runtime_inventory() != before or pin(Path(__file__)) != producer:
            raise ValueError('Producer/runtime endpoint changed')
        for name, (size, digest) in PINS.items():
            if pin(MODEL / name) != {'bytes': size, 'sha256': digest}:
                raise ValueError('Metadata endpoint changed: ' + name)
        result = dict(format='toklet-phi3-plain-text-v1', repository=REPOSITORY, revision=REVISION,
            file_pins={name: {'bytes': size, 'sha256': digest} for name, (size, digest) in PINS.items()},
            producer=producer, runtime_pins=before, versions=VERSIONS, cases=rows, raw_texts=raw_rows,
            added_token_controls=added_rows, direct_byte_fallback_controls=byte_rows, special_boundaries=boundaries,
            stop_ids=list(STOPS), not_stops=[1, 2], automatic_bos=False, add_special_tokens=False,
            skip_special_tokens=False, padding=False, truncation=False, cleanup_transform=False,
            mapped_ids=[0, MAPPED_SIZE - 1], unmapped_rejected_ids=unmapped, numerical_vocab=HEAD_SIZE,
            source_eos_declarations=dict(model=32000, tokenizer_config=32007, special_tokens_map=32000),
            rendered_decode_roundtrip_required=False,
            decode_policy='Explicit tokenizers non-skipping decode; preserve pinned metaspace/ByteFallback/Fuse/Strip behavior',
            model_executed=False, native_text_executed=False, chat_integration_qualified=False,
            native_session_prefix_preservation_qualified=False, full_quality_qualified=False)
        raw = encoded(result)
        if len(raw) > MAX_OUTPUT_BYTES:
            raise ValueError('Fixture exceeds2MiB output bound')
        with output.open('xb') as stream:
            stream.write(raw)
        return identity(raw)
    except BaseException as error:
        try:
            with failure_path.open('xb') as stream:
                stream.write(encoded(dict(complete=False, error=type(error).__name__ + ': ' + str(error))))
        except Exception as receipt_error:
            error.add_note('Failure receipt could not be written: ' + repr(receipt_error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output)))
