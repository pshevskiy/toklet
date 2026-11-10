"""Independent pinned MiniCPM5 basic-text witness; no numerical model/MLX import.

Run original immutable-sandbox Jinja and Tokenizers only after the existing
external text-runtime admission. The explicitly supplied environment root binds
the shared repository venv while implementation sources may be in a worktree.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import sysconfig
from types import SimpleNamespace

import text_reference_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]
MAX_TEXT = 65536
MAX_TOKENS = 65536
PROCESS_LIMIT = 1024 ** 3
REPO = 'bumblebuttpow/MiniCPM5-2B-heretic-abliterated-MLX-8bit'
REVISION = '50ab1b226834c290cdf68e460eb899e5cff7c61d'
PINS = {
    'config.json': (1056, 'ba1fda219aca07e7809d454db02512fe3fb29ff49e1eb2364c8e9487e35a6fe3'),
    'tokenizer.json': (9894424, 'ad92aaa8d3032c98a9158b8c5e8682bed10027ed6463e4fb1320fe5384210873'),
    'tokenizer_config.json': (539, '4d3748f79d8f82b68edf38b2a082b2df1f598e8e5a318dcb1f5c507d2f08b31b'),
    'generation_config.json': (227, '90a38054c1abd33cb068dd17286e29eb3827f8619eab86bfea50e1fdb1eb1beb'),
    'chat_template.jinja': (9060, 'cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c'),
}
UPSTREAM_SHA = 'ecb52655e86ff6d6db6a657601ece11f195e85971482eb2d0f40c20de17eb4c5'


def sources():
    upstream = Path(sysconfig.get_path('purelib')) / 'mlx_lm'
    return [Path(__file__).resolve(), ROOT / 'tools/reference/test_minicpm5_text_codec_reference.py',
            upstream / 'tokenizer_utils.py', upstream / '_version.py']


def bind_environment(root):
    root = root.resolve()
    if (Path(sys.prefix).resolve() != (root / '.venv').resolve()
            or not (root / 'Cargo.lock').is_file()):
        raise ValueError('Explicit shared repository text environment required')
    # Only the environment boundary moves. Every source/native/package identity
    # and actual loaded origin still passes the unchanged admission/endpoints.
    runtime.ROOT = root


def validate(messages, options, literals):
    if (type(options) is not dict or set(options) != {'enable_thinking', 'add_generation_prompt'}
            or any(type(value) is not bool for value in options.values())):
        raise ValueError('Explicit bool thinking/generation options required')
    if (type(messages) is not list or not 1 <= len(messages) <= 63
            or len(json.dumps(messages, ensure_ascii=False).encode()) > MAX_TEXT):
        raise ValueError('Bounded messages required')
    start = int(type(messages[0]) is dict and messages[0].get('role') == 'system')
    if start == len(messages):
        raise ValueError('At least one user required')
    for index, message in enumerate(messages):
        role = 'system' if start and index == 0 else ('user' if (index - start) % 2 == 0 else 'assistant')
        if (type(message) is not dict or set(message) != {'role', 'content'}
                or message['role'] != role or type(message['content']) is not str):
            raise ValueError('Basic alternating string conversation required')
        content = message['content']
        if any(x in content for x in literals) or any(ord(c) < 32 and c not in '\t\n\r' for c in content):
            raise ValueError('Reserved literals and unsupported C0 controls reject')
    if options['add_generation_prompt'] and messages[-1]['role'] != 'user':
        raise ValueError('Generation prompt requires final user')


def eos_witness(named, declared):
    path = Path(sysconfig.get_path('purelib')) / 'mlx_lm/tokenizer_utils.py'
    raw = runtime.common.captured(path, 23770, UPSTREAM_SHA)
    tree = ast.parse(raw)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TokenizerWrapper')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    nodes = []
    for node in init.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == '_eos_token_ids' for t in node.targets):
            nodes.append(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'tokenizer.eos_token_id is not None':
            nodes.append(node)
    text = '\n'.join(ast.unparse(node) for node in nodes)
    expected = 'self._eos_token_ids = set(eos_token_ids or [])\nif tokenizer.eos_token_id is not None:\n    self._eos_token_ids.add(tokenizer.eos_token_id)'
    if text != expected:
        raise ValueError('Pinned upstream stop-union source differs')
    owner = SimpleNamespace()
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'),
         dict(__builtins__={'set': set}, self=owner, eos_token_ids=declared,
              tokenizer=SimpleNamespace(eos_token_id=named)))
    return dict(stop_ids=sorted(owner._eos_token_ids), source_sha256=UPSTREAM_SHA,
                source_slice_sha256=hashlib.sha256(text.encode()).hexdigest(),
                scope='Exact selected upstream constructor source slice, not whole MLX-LM runtime')


def prefixes(tokenizer, ids):
    return {str(chunk): [dict(tokens=n, text=tokenizer.decode(ids[:n], skip_special_tokens=True))
                        for n in sorted({0, len(ids), *range(chunk, len(ids), chunk)})]
            for chunk in [1, 3, 128]}


class Capture:
    def __init__(self, stream):
        self.stream, self.count, self.bytes = stream, 0, 0

    def add(self, row):
        raw = (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
        if len(raw) > 256 * 1024 or self.bytes + len(raw) > 2 * 1024 ** 2 or self.count >= 256:
            raise ValueError('Text witness extent exceeded')
        if self.stream.write(raw) != len(raw):
            raise OSError('Short text witness write')
        self.stream.flush(); self.count += 1; self.bytes += len(raw)


def produce(capture, assets):
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    data = json.loads(assets['tokenizer.json'], object_pairs_hook=runtime.common.json_unique)
    tokenizer = Tokenizer.from_buffer(assets['tokenizer.json'])
    config = json.loads(assets['config.json'])
    settings = json.loads(assets['tokenizer_config.json'])
    generation = json.loads(assets['generation_config.json'])
    declared = generation['eos_token_id']
    if config['eos_token_id'] != declared or declared != [1, 130073] or tokenizer.token_to_id(settings['eos_token']) != 1:
        raise ValueError('Pinned EOS declarations differ')
    eos = eos_witness(1, declared)
    if eos['stop_ids'] != [1, 130073] or (tokenizer.get_vocab_size(False), tokenizer.get_vocab_size()) != (130072, 130560):
        raise ValueError('Text/stop vocabulary differs')
    capture.add(dict(kind='structure', base_vocab=130072, vocab=130560, bos=0,
                     added_tokens=data['added_tokens'], normalizer=data['normalizer'],
                     pre_tokenizer=data['pre_tokenizer'], post_processor=data['post_processor'],
                     decoder=data['decoder'], maximum_token_bytes=max(len(t.encode()) for t in tokenizer.get_vocab()), **eos))
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                               extensions=['jinja2.ext.loopcontrols'], loader=None, cache_size=0)
    original = environment.from_string(assets['chat_template.jinja'].decode())
    literals = [item['content'] for item in data['added_tokens']]
    chats = [
        ('hello', [dict(role='user', content='Hi')]),
        ('empty', [dict(role='user', content='')]),
        ('space', [dict(role='user', content='\t Hello \r\n')]),
        ('unicode', [dict(role='user', content='Привет 世界 🙂 e\u0301 é')]),
        ('code', [dict(role='user', content='def f(x):\n    return x + 1\n')]),
        ('multiturn', [dict(role='user', content='Hi'), dict(role='assistant', content='  Hello! \n'), dict(role='user', content='Next?')]),
        ('system', [dict(role='system', content='Be precise.\n'), dict(role='user', content='Hi')]),
        ('system_empty', [dict(role='system', content=''), dict(role='user', content='')]),
        ('system_multi', [dict(role='system', content='规则'), dict(role='user', content='Hello'), dict(role='assistant', content=''), dict(role='user', content='again')]),
        ('assistant_newlines', [dict(role='system', content='S'), dict(role='user', content='\nHi'), dict(role='assistant', content='\n\nA\nB\n'), dict(role='user', content='\nNext\n')]),
    ]
    for name, messages in chats:
        for thinking in [False, True]:
            for generation_prompt in [False, True]:
                options = dict(enable_thinking=thinking, add_generation_prompt=generation_prompt)
                validate(messages, options, literals)
                rendered = original.render(messages=messages, tools=None, bos_token='<s>', **options)
                if len(rendered.encode()) > MAX_TEXT:
                    raise ValueError('Rendered reference exceeds text bound')
                capture.add(dict(kind='chat', name=name, messages=messages, options=options, rendered=rendered,
                                 ids=tokenizer.encode(rendered, add_special_tokens=False).ids,
                                 add_special_tokens_ids=tokenizer.encode(rendered, add_special_tokens=True).ids))
    for name, messages in [('final_assistant', [dict(role='user', content='Hi'), dict(role='assistant', content='OK')]),
                           ('empty_assistant', [dict(role='system', content='S'), dict(role='user', content=''), dict(role='assistant', content='')]),
                           ('final_assistant_newlines', [dict(role='user', content='\n'), dict(role='assistant', content='\n\nA\nB\n')])]:
        for thinking in [False, True]:
            options = dict(enable_thinking=thinking, add_generation_prompt=False)
            validate(messages, options, literals)
            rendered = original.render(messages=messages, tools=None, bos_token='<s>', **options)
            capture.add(dict(kind='chat', name=name, messages=messages, options=options, rendered=rendered,
                             ids=tokenizer.encode(rendered, add_special_tokens=False).ids,
                             add_special_tokens_ids=tokenizer.encode(rendered, add_special_tokens=True).ids))
    texts = ['', 'hello', ' hello\n', 'Привет 世界 🙂', 'e\u0301 é', '<s></s><think></think>',
             '1234567890', 'a\tb\r\nc', '<unused_token_0>', '/think /no_think', 'def f():\n  return 1', 'a' * 257]
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        capture.add(dict(kind='text', input=text, ids=ids,
                         add_special_tokens_ids=tokenizer.encode(text, add_special_tokens=True).ids,
                         decoded=tokenizer.decode(ids, skip_special_tokens=True), prefixes=prefixes(tokenizer, ids)))
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs); extra = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte); cs.append(256 + extra); extra += 1
    alphabet = dict(zip(bs, map(chr, cs)))
    byte_id = lambda byte: tokenizer.token_to_id(alphabet[byte])
    sequences = [list('🙂'.encode()), list('€'.encode()), [0xf0, 0x9f], [0xe2, 0x28, 0xa1],
                 [0x80, 0xff, 0x61], [0xf0, 0x9f, 0xff, 0x82], [0xe2, 0x82, 0xac, 0x0a]]
    for index, values in enumerate(sequences):
        ids = [byte_id(v) for v in values]
        if any(v is None for v in ids): raise ValueError('Missing literal ByteLevel byte token')
        capture.add(dict(kind='decode', name='bytes_' + str(index), ids=ids,
                         decoded=tokenizer.decode(ids, skip_special_tokens=True),
                         raw_decoded=tokenizer.decode(ids, skip_special_tokens=False), prefixes=prefixes(tokenizer, ids)))
    for name, ids in [('special', [item['id'] for item in data['added_tokens'] if item['special']]),
                      ('thinking', [0, 8, 9, 1, 130073]), ('unused', [130082, 130559])]:
        capture.add(dict(kind='decode', name=name, ids=ids,
                         decoded=tokenizer.decode(ids, skip_special_tokens=True),
                         raw_decoded=tokenizer.decode(ids, skip_special_tokens=False), prefixes=prefixes(tokenizer, ids)))
    good = [dict(role='user', content='Hi')]; flags = dict(enable_thinking=False, add_generation_prompt=True)
    rejects = [('empty_messages', [], flags), ('system_only', [dict(role='system', content='S')], flags),
               ('assistant_first', [dict(role='assistant', content='A')], flags),
               ('duplicate_user', good + good, flags), ('tool', [dict(role='tool', content='x')], flags),
               ('array_content', [dict(role='user', content=[])], flags),
               ('null_content', [dict(role='user', content=None)], flags),
               ('reasoning_field', [dict(role='user', content='x', reasoning_content='x')], flags),
               ('tool_calls', [dict(role='user', content='x', tool_calls=[])], flags),
               ('system_mid', good + [dict(role='system', content='S')], flags),
               ('final_assistant_prompt', good + [dict(role='assistant', content='A')], flags),
               ('c0', [dict(role='user', content='x\0')], flags),
               ('too_long', [dict(role='user', content='x' * MAX_TEXT)], flags),
               ('too_many', [dict(role='user' if i % 2 == 0 else 'assistant', content='') for i in range(64)], flags)]
    for field in flags:
        for value in [None, 0, 'false']:
            rejects.append((field + '_' + repr(value), good, dict(flags, **{field: value})))
    for literal in ['<s>', '<|im_start|>', '<|im_end|>', '<think>', '</think>', '/think', '<unused_token_0>', '<unused_token_477>']:
        rejects.append(('reserved_' + literal, [dict(role='user', content=literal)], flags))
    for name, messages, options in rejects:
        try: validate(messages, options, literals)
        except ValueError: pass
        else: raise ValueError('Reference policy accepted negative: ' + name)
        capture.add(dict(kind='rejection', name=name, messages=messages, options=options))
    # Source BatchLongest/left is preserved. This API encodes one sequence at a
    # time; prove the equivalent one-item batch adds no padding, including empty.
    singles = []
    for text in ['', 'Hi', '世界🙂']:
        for special in [False, True]:
            one = tokenizer.encode(text, add_special_tokens=special).ids
            batch = tokenizer.encode_batch([text], add_special_tokens=special)[0].ids
            if one != batch: raise ValueError('Single-sequence source padding differs')
            singles.append(dict(text=text, add_special_tokens=special, ids=one))
    capture.add(dict(kind='padding', source_padding=data['padding'], source_truncation=data['truncation'],
                     runtime_padding=tokenizer.padding, runtime_truncation=tokenizer.truncation,
                     single_sequence_controls=singles))


def run(args, admission, endpoint):
    args.output.mkdir()
    assets = {name: runtime.common.captured(args.model / name, size, pin)
              for name, (size, pin) in PINS.items()}
    error, endpoint_errors, origins, capture = None, [], None, None
    try:
        with (args.output / 'records.jsonl').open('xb') as stream:
            capture = Capture(stream)
            produce(capture, assets)
    except BaseException as exception:
        error = type(exception).__name__ + ': ' + str(exception)
    finally:
        try: origins = endpoint()
        except BaseException as exception: endpoint_errors.append(str(exception))
        for name, (size, pin) in PINS.items():
            try: runtime.common.captured(args.model / name, size, pin)
            except BaseException as exception: endpoint_errors.append(name + ': ' + str(exception))
        path = args.output / 'records.jsonl'
        result = dict(format='toklet-minicpm5-text-reference-v1', passed=error is None and not endpoint_errors,
                      primary_error=error, endpoint_errors=endpoint_errors, runtime_admission=admission,
                      runtime_endpoint_origins=origins, repository=REPO, revision=REVISION, assets=PINS,
                      records=capture.count if capture else 0, bytes=path.stat().st_size if path.exists() else 0,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                      reference_process_limit_bytes=PROCESS_LIMIT,
                      model_inference_executed=False, native_codec_executed=False,
                      quality_qualified=False, managed_adoption_qualified=False)
        (args.output / 'observation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['passed', 'primary_error', 'endpoint_errors', 'records', 'bytes']}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'output', 'runtime-manifest', 'environment-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-runtime-sha256', required=True)
    args = parser.parse_args()
    bind_environment(args.environment_root)
    admission, endpoint = runtime.admit(args.runtime_manifest, args.expected_runtime_sha256, sources())
    sys.exit(run(args, admission, endpoint))
