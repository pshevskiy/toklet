"""Independent original-BF16 Llama3 graph with FP32 intermediates; no Metal/MLX.

Weights stay mapped as little-endian BF16. Widen one projection at a time and
project the tied table in vocabulary tiles; never expand a whole FP32 model.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import mmap
import os
import stat
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
import acquire_llama3_original as artifact


@dataclass(frozen=True)
class Shape:
    vocab: int = 128256
    hidden: int = 2048
    intermediate: int = 8192
    layers: int = 16
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 64


def expected_tensors(shape):
    result = {'model.embed_tokens.weight': [shape.vocab, shape.hidden], 'model.norm.weight': [shape.hidden]}
    for layer in range(shape.layers):
        base = f'model.layers.{layer}.'
        for name in ['input_layernorm', 'post_attention_layernorm']:
            result[base + name + '.weight'] = [shape.hidden]
        for name, rows in [('q_proj', shape.hidden), ('k_proj', shape.kv_heads * shape.head_dim),
                           ('v_proj', shape.kv_heads * shape.head_dim), ('o_proj', shape.hidden)]:
            result[base + 'self_attn.' + name + '.weight'] = [rows, shape.hidden]
        for name, dims in [('gate_proj', [shape.intermediate, shape.hidden]),
                           ('up_proj', [shape.intermediate, shape.hidden]),
                           ('down_proj', [shape.hidden, shape.intermediate])]:
            result[base + 'mlp.' + name + '.weight'] = dims
    return result


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def validate_header(header, payload_bytes, shape):
    header = dict(header)
    metadata = header.pop('__metadata__', {})
    if metadata != {'format': 'pt'} or set(header) != set(expected_tensors(shape)):
        raise ValueError('Original tensor inventory differs')
    intervals = []
    for name, dims in expected_tensors(shape).items():
        row = header[name]
        if (set(row) != {'dtype', 'shape', 'data_offsets'} or row['dtype'] != 'BF16'
                or row['shape'] != dims or any(type(n) is not int for n in row['shape'])):
            raise ValueError('Original tensor dtype/shape differs: ' + name)
        offsets = row['data_offsets']
        if (not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets)
                or offsets[0] < 0 or offsets[1] > payload_bytes
                or offsets[1] - offsets[0] != math.prod(dims) * 2):
            raise ValueError('Original tensor extent differs: ' + name)
        intervals.append(tuple(offsets))
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError('Original payload has overlap or gaps')
        end = stop
    if end != payload_bytes:
        raise ValueError('Original payload coverage differs')
    return header


def widen(words):
    if np.asarray(words).dtype != np.dtype('<u2'):
        raise ValueError('Expected little-endian BF16 words')
    values = np.asarray(words).astype(np.uint32)
    values <<= np.uint32(16)
    floats = values.view(np.float32)
    if not np.isfinite(floats).all():
        raise ValueError('Nonfinite original weight')
    return floats


class Reader:
    def __init__(self, directory):
        directory = Path(directory)
        acquisition = json.loads((directory / 'acquisition.json').read_bytes())
        if not acquisition['passed'] or acquisition['mirror']['revision'] != artifact.REVISION:
            raise ValueError('Original acquisition is incomplete')
        raw = (directory / 'config.json').read_bytes()
        if artifact.git_blob(raw) != artifact.BLOBS['config.json']:
            raise ValueError('Original configuration differs')
        config = json.loads(raw, object_pairs_hook=unique)
        checks = {'hidden_size': 2048, 'intermediate_size': 8192, 'num_hidden_layers': 16,
                  'num_attention_heads': 32, 'num_key_value_heads': 8, 'vocab_size': 128256,
                  'max_position_embeddings': 131072, 'rope_theta': 500000.0, 'rms_norm_eps': 1e-5,
                  'hidden_act': 'silu', 'tie_word_embeddings': True, 'attention_bias': False,
                  'head_dim': 64, 'model_type': 'llama', 'bos_token_id': 128000,
                  'eos_token_id': [128001, 128008, 128009], 'attention_dropout': 0.0,
                  'mlp_bias': False, 'torch_dtype': 'bfloat16', 'pretraining_tp': 1}
        if any(config.get(key) != value for key, value in checks.items()):
            raise ValueError('Original numerical configuration differs')
        if config['rope_scaling'] != {'factor': 32.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0,
                                      'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}:
            raise ValueError('Original rotary policy differs')
        self.shape = Shape()
        path = directory / 'model.safetensors'
        if path.is_symlink():
            raise ValueError('Original payload must be a regular file')
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        self.file = os.fdopen(descriptor, 'rb')
        try:
            if not stat.S_ISREG(os.fstat(self.file.fileno()).st_mode):
                raise ValueError('Original payload must be regular')
            self.identity = self.file_identity()
            if self.identity[2] != artifact.WEIGHT_BYTES or hashlib.file_digest(self.file, 'sha256').hexdigest() != artifact.WEIGHT_SHA:
                raise ValueError('Original payload identity differs')
            self.file.seek(0)
            length = struct.unpack('<Q', self.file.read(8))[0]
            if not 0 < length <= 1024**2:
                raise ValueError('Original header exceeds bound')
            header = json.loads(self.file.read(length), object_pairs_hook=unique)
            self.offset = 8 + length
            self.tensors = validate_header(header, artifact.WEIGHT_BYTES - self.offset, self.shape)
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self.file.close()
            raise

    def file_identity(self):
        s = os.fstat(self.file.fileno())
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def words(self, name):
        row = self.tensors[name]
        return np.ndarray(row['shape'], dtype='<u2', buffer=self.mapping,
                          offset=self.offset + row['data_offsets'][0])

    def weight(self, name, start=None, stop=None):
        words = self.words(name)
        return widen(words if start is None else words[start:stop])

    def embedding(self, ids):
        return widen(self.words('model.embed_tokens.weight')[ids])

    def close(self):
        unchanged = self.file_identity() == self.identity
        self.mapping.close()
        self.file.close()
        if not unchanged:
            raise ValueError('Original payload changed during scoring')


def rms(x, weight):
    mean = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x * (np.float32(1) / np.sqrt(mean + np.float32(1e-5)))) * weight


def rotary(x):
    # Original split-half convention and canonical Llama3 low/high wavelength mix.
    width = x.shape[-1]
    frequency = np.float32(1) / np.power(np.float32(500000), np.arange(0, width, 2, dtype=np.float32) / np.float32(width))
    wavelength = np.float32(2 * math.pi) / frequency
    smooth = (np.float32(8192) / wavelength - np.float32(1)) / np.float32(3)
    mixed = (np.float32(1) - smooth) * frequency / np.float32(32) + smooth * frequency
    frequency = np.where(wavelength > np.float32(8192), frequency / np.float32(32),
                         np.where(wavelength < np.float32(2048), frequency, mixed))
    angle = np.arange(x.shape[0], dtype=np.float32)[:, None] * frequency[None, :]
    c, s = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
    left, right = np.split(x, 2, axis=-1)
    return np.concatenate((left * c - right * s, right * c + left * s), axis=-1)


def attention(q, k, v):
    rows, heads, dim = q.shape
    result = np.empty_like(q)
    future = np.triu(np.ones((rows, rows), dtype=bool), 1)
    group = heads // k.shape[1]
    for head in range(heads):
        scores = (q[:, head] @ k[:, head // group].T) * np.float32(1 / math.sqrt(dim))
        scores[future] = -np.inf
        scores -= np.max(scores, axis=-1, keepdims=True)
        probability = np.exp(scores)
        probability /= np.sum(probability, axis=-1, keepdims=True, dtype=np.float32)
        result[:, head] = probability @ v[:, head // group]
    return result


def forward(reader, ids, tile=256):
    shape = reader.shape
    if not ids or len(ids) > 512 or any(type(i) is not int or not 0 <= i < shape.vocab for i in ids) or not 1 <= tile <= 4096:
        raise ValueError('Short reference input exceeds bounds')
    x = reader.embedding(ids)
    def linear(hidden, name):
        return hidden @ reader.weight(name).T
    for layer in range(shape.layers):
        base = f'model.layers.{layer}.'
        normalized = rms(x, reader.weight(base + 'input_layernorm.weight'))
        q = rotary(linear(normalized, base + 'self_attn.q_proj.weight').reshape(-1, shape.heads, shape.head_dim))
        k = rotary(linear(normalized, base + 'self_attn.k_proj.weight').reshape(-1, shape.kv_heads, shape.head_dim))
        v = linear(normalized, base + 'self_attn.v_proj.weight').reshape(-1, shape.kv_heads, shape.head_dim)
        x = x + linear(attention(q, k, v).reshape(-1, shape.hidden), base + 'self_attn.o_proj.weight')
        normalized = rms(x, reader.weight(base + 'post_attention_layernorm.weight'))
        gate = linear(normalized, base + 'mlp.gate_proj.weight')
        up = linear(normalized, base + 'mlp.up_proj.weight')
        activated = (gate / (np.float32(1) + np.exp(-gate))) * up
        x = x + linear(activated, base + 'mlp.down_proj.weight')
    hidden = rms(x, reader.weight('model.norm.weight'))
    logits = np.empty((len(ids), shape.vocab), dtype=np.float32)
    for start in range(0, shape.vocab, tile):
        stop = min(start + tile, shape.vocab)
        logits[:, start:stop] = hidden @ reader.weight('model.embed_tokens.weight', start, stop).T
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite original logits')
    return logits
