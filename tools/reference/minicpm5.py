"""Bounded independent MiniCPM5 MLX8 reference; never an adoption grant.

Ordinary split-half RoPE and the bias-free, separate-head Llama graph reuse the
unchanged NumPy FP32/FP64 arithmetic primitives. Packed rows use pread rather
than a full mapping/expanded model. BF16 metadata is reconstructed independently.
"""
from pathlib import Path
import hashlib
import json
import math
import os

REPOSITORY = 'bumblebuttpow/MiniCPM5-2B-heretic-abliterated-MLX-8bit'
REVISION = '50ab1b226834c290cdf68e460eb899e5cff7c61d'
WEIGHT_BYTES = 2674327346
PINS = {'config.json': 'ba1fda219aca07e7809d454db02512fe3fb29ff49e1eb2364c8e9487e35a6fe3',
        'model.safetensors.index.json': '50747e37090508f851cdea2f4d5a71c59c2a26a09c8e95c505371ac669e2ac36',
        'model.safetensors': '6fadca0fcdefe8b77fc144851ee2c2fd3c8949ccc0ec756b8ada27e3156c8910'}
GEOMETRY = dict(vocab_size=130560, hidden_size=2048, num_layers=42,
                num_query_heads=16, num_kv_heads=2, head_dim=128,
                intermediate_size=6144, rms_norm_eps=1e-6, rope_theta=5000000.0)
CASES = dict(single=dict(tokens=[0], position_offset=0),
             short=dict(tokens=[0, 7, 13, 3], position_offset=0),
             future_changed=dict(tokens=[0, 7, 13, 17], position_offset=0),
             offset5=dict(tokens=[7, 13, 3], position_offset=5),
             tile17=dict(tokens=[0, 7, 13, 3] * 4 + [0], position_offset=0),
             offset509=dict(tokens=[7, 13, 3], position_offset=509))
LAYERS = [0, 1, 41]
POLICY = dict(atol_floor=1e-4, envelope_multiplier=2, rtol=1e-3)


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON field: ' + key)
        result[key] = value
    return result


def bf16(raw, shape):
    import numpy as np
    need(type(raw) is bytes and len(raw) == 2 * math.prod(shape), 'BF16 byte count differs')
    values = (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).view('<f4').reshape(shape)
    need(np.isfinite(values).all(), 'Nonfinite BF16 source')
    return values


def decode(packed, scales, biases, columns):
    import numpy as np
    need(packed.dtype == np.dtype('<u4') and scales.dtype == np.float32
         and biases.dtype == np.float32, 'Affine8/BF16 input dtypes differ')
    need(type(columns) is int and columns > 0 and columns % 64 == 0
         and packed.ndim == 2 and len(packed) > 0
         and packed.shape[1] == columns // 4
         and scales.shape == biases.shape == (len(packed), columns // 64),
         'Affine8/group64 shapes differ')
    codes = np.stack([(packed >> shift) & 255 for shift in range(0, 32, 8)], axis=-1)
    codes = codes.reshape(len(packed), columns)
    values = (codes.astype(np.float64) * np.repeat(scales.astype(np.float64), 64, axis=-1)
              + np.repeat(biases.astype(np.float64), 64, axis=-1)).astype(np.float32)
    need(np.isfinite(values).all(), 'Nonfinite affine reconstruction')
    return values


def logical_shapes():
    result = {'model.embed_tokens.weight': [130560, 2048],
              'lm_head.weight': [130560, 2048], 'model.norm.weight': [2048]}
    for layer in range(42):
        prefix = f'model.layers.{layer}'
        for norm in ['input_layernorm', 'post_attention_layernorm']:
            result[f'{prefix}.{norm}.weight'] = [2048]
        for name, rows, cols in [('q', 2048, 2048), ('k', 256, 2048),
                                 ('v', 256, 2048), ('o', 2048, 2048),
                                 ('gate', 6144, 2048), ('up', 6144, 2048),
                                 ('down', 2048, 6144)]:
            block = 'self_attn' if name in ['q', 'k', 'v', 'o'] else 'mlp'
            result[f'{prefix}.{block}.{name}_proj.weight'] = [rows, cols]
    return result


class RowMatrix:
    """Lazy contiguous FP32 output rows; independent BLAS widens one tile."""
    def __init__(self, source, name):
        self.source, self.name = source, name
        self.rows, self.columns = source.logical[name]

    def __len__(self):
        return self.rows

    def __getitem__(self, key):
        need(isinstance(key, slice) and key.step in (None, 1), 'Matrix access requires contiguous rows')
        start, end, _ = key.indices(self.rows)
        need(start < end, 'Empty matrix row tile')
        module = self.name.removesuffix('.weight')
        return decode(self.source.raw(self.name, start, end),
                      self.source.raw(module + '.scales', start, end),
                      self.source.raw(module + '.biases', start, end), self.columns)


class Embedding:
    def __init__(self, source):
        self.matrix = RowMatrix(source, 'model.embed_tokens.weight')

    def __getitem__(self, tokens):
        import numpy as np
        need(type(tokens) is list and tokens and all(type(t) is int and 0 <= t < len(self.matrix)
                                                    for t in tokens), 'Invalid embedding token IDs')
        return np.concatenate([self.matrix[t:t + 1] for t in tokens], axis=0).astype(np.float16)


class Source:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.stream = None
        self.logical = logical_shapes()

    def __enter__(self):
        try:
            for name, pin in PINS.items():
                need(sha(self.folder / name) == pin, 'Pinned source differs: ' + name)
            config = json.loads((self.folder / 'config.json').read_text(), object_pairs_hook=unique)
            index = json.loads((self.folder / 'model.safetensors.index.json').read_text(), object_pairs_hook=unique)
            need(config['quantization'] == {'group_size': 64, 'bits': 8, 'mode': 'affine'}, 'Quantization declaration differs')
            self.stream = (self.folder / 'model.safetensors').open('rb')
            need(os.fstat(self.stream.fileno()).st_size == WEIGHT_BYTES, 'Source length differs')
            length = int.from_bytes(self.stream.read(8), 'little')
            need(0 < length <= 2 * 1024 ** 2, 'Source header exceeds bound')
            raw = self.stream.read(length)
            need(len(raw) == length and hashlib.sha256(raw).hexdigest() ==
                 'b6fdf59a1fce7d5fb3caf76725156c53c222d1e5f481624981cc3cb3abcacf07', 'Header differs')
            self.header = json.loads(raw, object_pairs_hook=unique)
            self.header.pop('__metadata__', None)
            self.origin = 8 + length
            expected = {}
            for name, shape in self.logical.items():
                if len(shape) == 1:
                    expected[name] = ('BF16', shape)
                else:
                    rows, cols = shape
                    expected[name] = ('U32', [rows, cols // 4])
                    for role in ['scales', 'biases']:
                        expected[name.removesuffix('.weight') + '.' + role] = ('BF16', [rows, cols // 64])
            need(set(self.header) == set(expected) == set(index['weight_map']), 'Physical tensor inventory differs')
            spans = []
            for name, (dtype, shape) in expected.items():
                item = self.header[name]
                begin, end = item['data_offsets']
                need(item['dtype'] == dtype and item['shape'] == shape
                     and type(begin) is int and type(end) is int and 0 <= begin < end
                     and end - begin == math.prod(shape) * (4 if dtype == 'U32' else 2)
                     and index['weight_map'][name] == 'model.safetensors', 'Tensor layout differs: ' + name)
                spans.append((begin, end))
            cursor = 0
            for begin, end in sorted(spans):
                need(begin == cursor, 'Noncontiguous tensor payload')
                cursor = end
            need(self.origin + cursor == WEIGHT_BYTES, 'Payload tail differs')
            return self
        except BaseException:
            if self.stream is not None:
                self.stream.close()
            raise

    def raw(self, name, start=0, end=None):
        import numpy as np
        item = self.header[name]
        shape = item['shape']
        if end is None:
            end = shape[0]
        need(type(start) is int and type(end) is int and 0 <= start < end <= shape[0], 'Invalid tensor row range')
        row_elements = math.prod(shape[1:])
        size = row_elements * (4 if item['dtype'] == 'U32' else 2)
        raw = os.pread(self.stream.fileno(), (end - start) * size,
                       self.origin + item['data_offsets'][0] + start * size)
        need(len(raw) == (end - start) * size, 'Truncated tensor row')
        selected = [end - start, *shape[1:]]
        return (np.frombuffer(raw, dtype='<u4').reshape(selected) if item['dtype'] == 'U32'
                else bf16(raw, selected))

    def __getitem__(self, key):
        import numpy as np
        if key == 'embed_tokens':
            return Embedding(self)
        if key == 'output':
            return RowMatrix(self, 'lm_head.weight')
        if key == 'norm':
            name = 'model.norm.weight'
        elif key.endswith(('input_layernorm', 'post_attention_layernorm')):
            name = 'model.' + key + '.weight'
        else:
            parts = key.split('.')
            block = 'self_attn' if parts[2] in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] else 'mlp'
            name = '.'.join(['model', *parts[:2], block, *parts[2:]])
        if len(self.logical[name]) == 2:
            return RowMatrix(self, name)
        value = self.raw(name).astype(np.float16)
        need(np.isfinite(value).all(), 'Nonfinite Half exclusion')
        return value

    def verify(self):
        need(self.stream is not None and not self.stream.closed, 'Source descriptor is closed')
        self.stream.seek(0)
        need(hashlib.file_digest(self.stream, 'sha256').hexdigest() == PINS['model.safetensors'],
             'Consumed source descriptor changed')
        need(all(sha(self.folder / name) == pin for name, pin in PINS.items()), 'Source path changed')

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.verify()
        finally:
            self.stream.close()


def forward(tokens, offset, weights, config, precision, layers):
    """Full ordinary Llama graph; selected traces, distinct output head."""
    import numpy as np
    from qwen_fp16 import linear, rmsnorm, rotary, attention
    if isinstance(weights,dict) and any(name.endswith('.bias') for name in weights):
        raise ValueError('Bias-free Llama reference rejects projection biases')
    need(type(tokens) is list and tokens and type(offset) is int and offset >= 0
         and offset + len(tokens) <= 512, 'Invalid bounded graph tokens/offset')
    n, dimension = len(tokens), config["head_dim"]
    x = weights["embed_tokens"][tokens].copy()
    values = {"embedding": x}
    for index in range(config["num_layers"]):
        prefix = f"layers.{index}"

        def save(name, tensor):
            if not np.isfinite(tensor).all():
                raise ValueError(f"Nonfinite FP16 reconstruction: {prefix}.{name}")
            if index in layers:
                values[f"{prefix}.{name}"] = tensor
            return tensor

        normalized = save("input_norm", rmsnorm(x, weights[f"{prefix}.input_layernorm"], config["rms_norm_eps"], precision))
        projected = {}
        for name, heads in (("q", config["num_query_heads"]), ("k", config["num_kv_heads"]), ("v", config["num_kv_heads"])):
            projected[name] = save(f"{name}_projection", linear(normalized, weights[f"{prefix}.{name}_proj.weight"],
                                     None, precision)).reshape(n, heads, dimension)
        query = save("q_rotary", rotary(projected["q"], offset, config["rope_theta"], precision))
        key = save("k_rotary", rotary(projected["k"], offset, config["rope_theta"], precision))
        scores, probabilities, context = attention(query, key, projected["v"], precision)
        save("attention_scores", scores)
        save("attention_probabilities", probabilities)
        save("attention_context", context)
        projected_attention = save("attention_projection", linear(context, weights[f"{prefix}.o_proj.weight"], None, precision))
        x = save("attention_residual", (x.astype(precision) + projected_attention).astype(np.float16))
        normalized = save("post_attention_norm", rmsnorm(x, weights[f"{prefix}.post_attention_layernorm"], config["rms_norm_eps"], precision))
        gate = save("gate_projection", linear(normalized, weights[f"{prefix}.gate_proj.weight"], None, precision))
        up = save("up_projection", linear(normalized, weights[f"{prefix}.up_proj.weight"], None, precision))
        wide = gate.astype(precision)
        exp = np.exp(-np.abs(wide))
        sigmoid = np.where(wide >= 0, precision(1) / (precision(1) + exp), exp / (precision(1) + exp))
        activated = save("gate_silu", (wide * sigmoid).astype(np.float16))
        gated = save("gated", (activated.astype(precision) * up).astype(np.float16))
        down = save("down_projection", linear(gated, weights[f"{prefix}.down_proj.weight"], None, precision))
        x = save("output", (x.astype(precision) + down).astype(np.float16))
    values["final_norm"] = rmsnorm(x, weights["norm"], config["rms_norm_eps"], precision)
    values["logits"] = linear(values["final_norm"], weights["output"], None, precision, np.float32)
    if not all(np.isfinite(array).all() for array in values.values()):
        raise ValueError("Nonfinite final FP16 reconstruction")
    return values
