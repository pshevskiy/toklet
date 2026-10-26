"""Inspect one pinned official BF16 header; never read weights or run a graph.

This is a text-reference layout, not a general Gemma importer. File offsets are
absolute, row-major and byte counted. Payload identity/fit remain separate gates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import stat

FILE_BYTES = 10246621918
FILE_SHA = '2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550'
HEADER_SHA = '12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e'
CONFIG_SHA = '1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330'
TEXT_CONFIG_SHA = '62f8428c49b31ebe5bb8186c7d352ce10498632148703db72568db3cd1a83257'
SOURCE_SHA = 'fd93159380556e46029260ae38e5ec53ffd750ecb9872d0e89f57b334a25ed4e'
PREFIX = 'model.language_model.'
MAX_HEADER = 8 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON field')
        result[key] = value
    return result


def document(raw):
    def nonfinite(_):
        raise ValueError('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def integer(value, minimum, maximum):
    require(type(value) is int and minimum <= value <= maximum, 'bounded integer')
    return value


def read(path, maximum):
    """Bound the same regular descriptor and both path/FD endpoint identities."""
    def identity(s):
        return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as file:
        before = os.fstat(file.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum,
                'regular bounded input')
        raw = file.read(maximum + 1)
        require(len(raw) == before.st_size and len(raw) <= maximum,
                'complete bounded input')
        require(identity(before) == identity(os.fstat(file.fileno()))
                == identity(path.lstat()), 'input identity changed')
    return raw


def parse_header(raw, file_bytes):
    """Validate every BF16 span, including scalar media tensors; no tensor load."""
    integer(file_bytes, 10, 12 * 1024**3)
    require(type(raw) is bytes and 2 <= len(raw) <= MAX_HEADER, 'bounded header')
    data = document(raw)
    require(type(data) is dict and data.get('__metadata__') == {'format': 'pt'},
            'PyTorch header metadata')
    require(1 < len(data) <= 4097, 'bounded tensor count')
    available = file_bytes - 8 - len(raw)
    require(available > 0, 'positive tensor payload')
    spans = []
    for name, item in data.items():
        if name == '__metadata__':
            continue
        require(type(name) is str and 0 < len(name.encode()) <= 256,
                'bounded tensor name')
        require(type(item) is dict and set(item) == {'dtype', 'shape', 'data_offsets'},
                'complete tensor descriptor')
        require(item['dtype'] == 'BF16', 'BF16 source tensor required')
        shape = item['shape']
        require(type(shape) is list and len(shape) <= 4, 'bounded tensor rank')
        for dimension in shape:
            integer(dimension, 1, 262144)
        extent = 2 * math.prod(shape)  # A rank-zero tensor contains one value.
        offsets = item['data_offsets']
        require(type(offsets) is list and len(offsets) == 2, 'tensor offset pair')
        start, end = [integer(value, 0, available) for value in offsets]
        require(start % 2 == 0 and end - start == extent and end > start,
                'BF16 tensor extent differs')
        spans.append((start, end, name))
    spans.sort()
    require(spans[0][0] == 0 and spans[-1][1] == available
            and all(a[1] == b[0] for a, b in zip(spans, spans[1:])),
            'complete nonoverlapping tensor coverage')
    return data


def text_shapes(config):
    """Resolve exact E2B geometry once, separating selected and unused tensors."""
    require(sha(encoded(config)) == TEXT_CONFIG_SHA, 'exact E2B text config')
    hidden, vocab = config['hidden_size'], config['vocab_size']
    ple, layers = config['hidden_size_per_layer_input'], config['num_hidden_layers']
    first_shared = layers - config['num_kv_shared_layers']
    selected = {
        'embed_tokens.weight': [vocab, hidden],
        'embed_tokens_per_layer.weight': [vocab, layers * ple],
        'per_layer_model_projection.weight': [layers * ple, hidden],
        'per_layer_projection_norm.weight': [ple],
        'norm.weight': [hidden],
    }
    unused = {}
    for index, kind in enumerate(config['layer_types']):
        base = f'layers.{index}.'
        dim = config['global_head_dim'] if kind == 'full_attention' else config['head_dim']
        width = config['intermediate_size'] * (2 if index >= first_shared else 1)
        for norm in ('input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm',
                     'post_feedforward_layernorm', 'post_per_layer_input_norm'):
            selected[base + norm + '.weight'] = [hidden]
        selected[base + 'layer_scalar'] = [1]
        selected[base + 'self_attn.q_norm.weight'] = [dim]
        selected[base + 'self_attn.q_proj.weight'] = [config['num_attention_heads'] * dim, hidden]
        selected[base + 'self_attn.o_proj.weight'] = [hidden, config['num_attention_heads'] * dim]
        selected[base + 'mlp.gate_proj.weight'] = [width, hidden]
        selected[base + 'mlp.up_proj.weight'] = [width, hidden]
        selected[base + 'mlp.down_proj.weight'] = [hidden, width]
        selected[base + 'per_layer_input_gate.weight'] = [ple, hidden]
        selected[base + 'per_layer_projection.weight'] = [hidden, ple]
        destination = selected if index < first_shared else unused
        destination[base + 'self_attn.k_norm.weight'] = [dim]
        for projection in ('k_proj', 'v_proj'):
            destination[base + 'self_attn.' + projection + '.weight'] = [config['num_key_value_heads'] * dim, hidden]
    require(len(selected) == 540 and len(unused) == 60, 'exact text selection count')
    return ({PREFIX + name: shape for name, shape in selected.items()},
            {PREFIX + name: shape for name, shape in unused.items()})


def inspect(header_raw, config_raw):
    require(sha(header_raw) == HEADER_SHA, 'pinned acquired header')
    require(sha(config_raw) == CONFIG_SHA, 'pinned official config')
    data = parse_header(header_raw, FILE_BYTES)
    require(len(data) == 2012, 'all2011 official tensors')
    selected, unused = text_shapes(document(config_raw)['text_config'])
    require({name for name in data if name.startswith(PREFIX)} == set(selected) | set(unused),
            'complete text tensor names')
    groups = {}
    for label, shapes in [('selected_text', selected), ('unused_shared_kv', unused)]:
        groups[label] = {}
        for name, shape in shapes.items():
            item = data[name]
            require(item['shape'] == shape, 'official text shape differs: ' + name)
            start, end = item['data_offsets']
            groups[label][name] = dict(shape=shape, dtype='BF16',
                                     file_offset=8 + len(header_raw) + start, bytes=end - start)
    for name in set(data) - {'__metadata__'} - set(selected) - set(unused):
        require(name.startswith(('model.audio_tower.', 'model.vision_tower.',
                                 'model.embed_audio.', 'model.embed_vision.')),
                'explicit media-only exclusions')
    selected_bytes = sum(value['bytes'] for value in groups['selected_text'].values())
    unused_bytes = sum(value['bytes'] for value in groups['unused_shared_kv'].values())
    matrices = sum(len(shape) == 2 for shape in selected.values())
    require((selected_bytes, unused_bytes, matrices) == (9257138758, 37761024, 278),
            'independent geometry byte/count totals')
    return dict(format='toklet-gemma-unquantized-layout-v1',
                repository='google/gemma-4-E2B-it', revision='70af34e20bd4b7a91f0de6b22675850c43922a03',
                advertised_container=dict(bytes=FILE_BYTES, sha256=FILE_SHA),
                acquired_header=dict(bytes=len(header_raw), sha256=HEADER_SHA),
                config_sha256=CONFIG_SHA, text_config_sha256=TEXT_CONFIG_SHA,
                counts=dict(all_tensors=2011, selected_matrices=278, selected_vectors=262,
                            unused_shared_kv=60, excluded_media=1411),
                bytes=dict(selected_text=selected_bytes, unused_shared_kv=unused_bytes,
                           excluded_media=FILE_BYTES - 8 - len(header_raw) - selected_bytes - unused_bytes),
                source_rule=dict(transformers_version='5.7.0', modeling_source_sha256=SOURCE_SHA,
                                 reviewed_inclusive_line_ranges=[[1158, 1195], [1215, 1237], [1592, 1601]],
                                 unused='shared consumers use producer13(sliding)/14(full) states; source ignores these unexpected K/V tensors on load'),
                **groups, full_payload_downloaded=False, full_payload_verified=False,
                weights_loaded=False, arithmetic_profile_qualified=False, model_fit_qualified=False,
                inference_executed=False, quality_scored=False, app_eligibility_changed=False,
                full_model_window_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--header', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(read(args.header, MAX_HEADER), read(args.config, 8192))
    with args.output.open('xb') as file:
        raw = encoded(result)
        require(file.write(raw) == len(raw), 'complete layout write')
    print(json.dumps(dict(passed=True, counts=result['counts'], bytes=result['bytes'],
                          manifest_sha256=sha(raw), full_payload_verified=False)))
