"""Data-only inspection of one pinned Qwen3 candidate; no model code executes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = 'mlx-community/Qwen3-0.6B-8bit'
REVISION = '11de96878523501bcaa86104e3c186de07ff9068'
DESTINATION = 'models/MLX-Qwen3-0.6B-8bit-11de968'
WEIGHT_SHA = '3ad5c96e0a476d4eee48e9f525d0ee0a3f1830b3e5b8e17b441de0661e921397'
CONFIG_SHA = '9f387dcb1bf045cadb34a36e67569abd73c00e5322603df2b19d0c0f8cc9d240'
WEIGHT_BYTES = 633442994
NAMES = {'README.md', 'added_tokens.json', 'config.json', 'merges.txt', 'model.safetensors',
         'model.safetensors.index.json', 'special_tokens_map.json', 'tokenizer.json',
         'tokenizer_config.json', 'vocab.json'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field: ' + key)
        result[key] = value
    return result


def document(path, maximum, expected):
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Bounded input identity differs: ' + str(path))
    return json.loads(raw, object_pairs_hook=unique)


def descriptor_sha(stream):
    position = stream.tell()
    try:
        stream.seek(0)
        return hashlib.file_digest(stream, 'sha256').hexdigest()
    finally:
        stream.seek(position)


def weight_header(path, expected_bytes, expected_sha):
    with path.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size != expected_bytes or descriptor_sha(stream) != expected_sha:
            raise ValueError('Consumed weight descriptor identity differs')
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('Missing safetensors length')
        length = struct.unpack('<Q', raw)[0]
        if not 0 < length < min(1024**2, expected_bytes - 8):
            raise ValueError('Bounded safetensors header required')
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError('Truncated safetensors header')
        return raw


def logical_shapes():
    shapes = {'model.embed_tokens.weight': [151936, 1024], 'model.norm.weight': [1024]}
    for layer in range(28):
        prefix = f'model.layers.{layer}'
        for norm in ['input_layernorm', 'post_attention_layernorm']:
            shapes[f'{prefix}.{norm}.weight'] = [1024]
        for norm in ['q_norm', 'k_norm']:
            shapes[f'{prefix}.self_attn.{norm}.weight'] = [128]
        for name, rows, columns in [('q', 2048, 1024), ('k', 1024, 1024), ('v', 1024, 1024),
                                    ('o', 1024, 2048), ('gate', 3072, 1024),
                                    ('up', 3072, 1024), ('down', 1024, 3072)]:
            block = 'self_attn' if name in ['q', 'k', 'v', 'o'] else 'mlp'
            shapes[f'{prefix}.{block}.{name}_proj.weight'] = [rows, columns]
    return shapes


def validate_header(header, index, payload_bytes):
    logical = logical_shapes()
    expected = {}
    for name, shape in logical.items():
        if len(shape) == 1:
            expected[name] = (None, shape)
        else:
            rows, columns = shape
            expected[name] = ('U32', [rows, columns // 4])
            for suffix in ['scales', 'biases']:
                expected[name.removesuffix('.weight') + '.' + suffix] = (None, [rows, columns // 64])
    if set(header) != set(expected) or set(index['weight_map']) != set(expected):
        raise ValueError('Exact Qwen3 tensor/index inventory required')
    spans = [];dtypes = {}
    for name, (required_dtype, shape) in expected.items():
        item = header[name]
        if (set(item) != {'dtype', 'shape', 'data_offsets'} or item['shape'] != shape
                or any(type(value) is not int or value <= 0 for value in item['shape'])
                or (item['dtype'] != required_dtype if required_dtype else item['dtype'] not in ['F16', 'BF16', 'F32'])
                or index['weight_map'][name] != 'model.safetensors'):
            raise ValueError('Qwen3 tensor shape/dtype/index differs: ' + name)
        begin, end = item['data_offsets']
        width = 4 if item['dtype'] in ['U32', 'F32'] else 2
        if (type(begin) is not int or type(end) is not int or begin < 0
                or end - begin != math.prod(shape) * width or end > payload_bytes):
            raise ValueError('Qwen3 tensor span differs: ' + name)
        spans.append((begin, end));dtypes[item['dtype']] = dtypes.get(item['dtype'], 0) + 1
        if name.endswith('.scales') and item['dtype'] != header[name.removesuffix('.scales') + '.biases']['dtype']:
            raise ValueError('Affine scale/bias dtypes differ')
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise ValueError('Tensor spans overlap or have gaps')
        cursor = end
    if (cursor != payload_bytes or type(index['metadata']['total_size']) is not int
            or index['metadata']['total_size'] != cursor):
        raise ValueError('Payload tail or index size differs')
    return dict(tensors=len(header), logical_tensors=len(logical), packed_matrices=197,
                floating_vectors=113, dtype_counts=dtypes, payload_bytes=cursor,
                projection_biases_absent=True, tied_output=True,
                query_width=2048, hidden_width=1024, kv_width=1024,
                qk_norm_vectors=56, qk_norm_width=128)


def tokenizer_inventory(tokenizer, settings, config):
    model = tokenizer['model'];vocab = model['vocab'];added = tokenizer['added_tokens']
    if model['type'] != 'BPE' or not isinstance(vocab, dict) or not isinstance(added, list):
        raise ValueError('Expected BPE tokenizer inventory')
    ids = list(vocab.values())
    if (any(type(token) is not int or not 0 <= token < 151936 for token in ids)
            or len(ids) != len(set(ids))):
        raise ValueError('Invalid base vocabulary IDs')
    base_by_id = {token: content for content, token in vocab.items()}
    controls = {};seen = set()
    for item in added:
        token = item['id'];content = item['content']
        if (type(token) is not int or not 0 <= token < 151936 or not isinstance(content, str)
                or token in seen or content in controls
                or (content in vocab and vocab[content] != token)
                or (token in base_by_id and base_by_id[token] != content)):
            raise ValueError('Invalid added token identity')
        controls[content] = token;seen.add(token)
    template = settings.get('chat_template')
    if not isinstance(template, str):
        raise ValueError('Expected textual template inventory')
    return dict(base_tokens=len(vocab), added_tokens=len(added), added_token_ids=controls,
                declared_bos_token_id=config['bos_token_id'], declared_eos_token_id=config['eos_token_id'],
                tokenizer_bos=settings.get('bos_token'), tokenizer_eos=settings.get('eos_token'),
                tokenizer_pad=settings.get('pad_token'), template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                template_mentions_enable_thinking='enable_thinking' in template,
                tokenizer_execution_qualified=False, chat_template_qualified=False)


def inspect(receipt, receipt_sha):
    download = document(receipt, 8 * 1024**2, receipt_sha)
    if (download['format'] != 'toklet-qwen3-range-acquisition-v1' or download['passed'] is not True
            or download['repository'] != REPOSITORY or download['revision'] != REVISION
            or download['destination'] != DESTINATION or set(download['files']) != NAMES
            or download['inference_executed'] is not False):
        raise ValueError('Exact successful candidate acquisition required')
    folder = ROOT / DESTINATION
    if folder.is_symlink() or {p.name for p in folder.iterdir()} != NAMES:
        raise ValueError('Exact published checkpoint directory required')
    def verify_files():
        for name, item in download['files'].items():
            path = folder / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('Published checkpoint changed: ' + name)
    verify_files()
    if (download['files']['config.json']['sha256'] != CONFIG_SHA
            or download['files']['model.safetensors'] != dict(bytes=WEIGHT_BYTES, sha256=WEIGHT_SHA)):
        raise ValueError('Pinned config/weight identity differs')
    config = document(folder / 'config.json', 65536, CONFIG_SHA)
    index = document(folder / 'model.safetensors.index.json', 1024**2, download['files']['model.safetensors.index.json']['sha256'])
    raw = weight_header(folder / 'model.safetensors', WEIGHT_BYTES, WEIGHT_SHA)
    length = len(raw)
    header = json.loads(raw, object_pairs_hook=unique);header.pop('__metadata__', None)
    result = validate_header(header, index, WEIGHT_BYTES - 8 - length)
    tokenizer = document(folder / 'tokenizer.json', 16 * 1024**2, download['files']['tokenizer.json']['sha256'])
    settings = document(folder / 'tokenizer_config.json', 65536, download['files']['tokenizer_config.json']['sha256'])
    result.update(format='toklet-qwen3-checkpoint-inspection-v1', passed=True,
                  repository=REPOSITORY, revision=REVISION, config=config,
                  header_bytes=length, header_sha256=hashlib.sha256(raw).hexdigest(),
                  files=download['files'], tokenizer=tokenizer_inventory(tokenizer, settings, config),
                  inference_executed=False, native_target_qualified=False, device_fit_qualified=False)
    verify_files()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if os.path.lexists(args.output) or not args.output.resolve().is_relative_to(ROOT / 'artifacts'):
        raise FileExistsError('Exclusive repository artifact output required')
    result = inspect(args.receipt, args.receipt_sha256)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps({key: result[key] for key in ['passed', 'tensors', 'packed_matrices', 'floating_vectors', 'dtype_counts', 'header_bytes']}))
