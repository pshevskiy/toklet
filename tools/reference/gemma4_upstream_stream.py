"""Bounded storage integration around the pinned upstream CPU BF16 forward.

This is an explicit reference harness, not an engine backend. Stock operators
retain all PLE, attention, rotary, normalization and softcap arithmetic.
"""
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4 import modeling_gemma4 as upstream

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = 'fd93159380556e46029260ae38e5ec53ffd750ecb9872d0e89f57b334a25ed4e'
EMBEDDING = 'model.embed_tokens.weight'
PLE = 'model.embed_tokens_per_layer.weight'


def require_runtime():
    if (importlib.metadata.version('transformers') != '5.7.0'
            or importlib.metadata.version('torch') != '2.9.1'
            or hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest() != SOURCE_SHA
            or sys.byteorder != 'little'):
        raise ValueError('Pinned little-endian upstream runtime required')


@contextmanager
def bf16_parameters():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        yield
    finally:
        torch.set_default_dtype(previous)


def tiny_config():
    parent = json.loads((ROOT/'tests/fixtures/gemma4-e2b-metadata-v1/config.json').read_text())['text_config']
    config = Gemma4TextConfig(vocab_size=17, vocab_size_per_layer_input=17,
        hidden_size=64, intermediate_size=64, hidden_size_per_layer_input=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        num_global_key_value_heads=1, head_dim=64, global_head_dim=128,
        sliding_window=4, max_position_embeddings=32, num_kv_shared_layers=2,
        layer_types=['sliding_attention','full_attention']*2,
        rope_parameters=copy.deepcopy(parent['rope_parameters']),
        use_double_wide_mlp=parent['use_double_wide_mlp'], rms_norm_eps=parent['rms_norm_eps'],
        final_logit_softcapping=30., tie_word_embeddings=True, attention_dropout=0.,
        use_cache=False, dtype='bfloat16')
    config._attn_implementation = 'eager'
    return config


def tensor_bytes(tensor):
    if tensor.dtype != torch.bfloat16 or tensor.device.type != 'cpu':
        raise ValueError('CPU BF16 tensor required')
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError('Finite BF16 tensor required')
    return tensor.detach().contiguous().view(torch.uint16).numpy().tobytes()


class TinyWeights:
    """Frozen tiny raw weights; large-checkpoint readers need separate admission."""
    def __init__(self, manifest, body):
        require_runtime()
        if (manifest['format'] != 'toklet-upstream-gemma-tiny-weights-v1'
                or len(body) > 32*1024**2 or hashlib.sha256(body).hexdigest() != manifest['sha256']):
            raise ValueError('Bounded tiny weight identity differs')
        self.manifest, self.body = manifest, body
        self.requests = []
        self.fail_layer = None
        self.fail_tensor = None
        offset = 0
        for item in manifest['tensors'].values():
            size = math.prod(item['shape'])*2
            raw = body[offset:offset+size]
            if (item['offset'] != offset or item['bytes'] != size or len(raw) != size
                    or hashlib.sha256(raw).hexdigest() != item['sha256']):
                raise ValueError('Tiny tensor inventory differs')
            if not bool(torch.isfinite(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)).all()):
                raise ValueError('Nonfinite tiny weights')
            offset += size
        if offset != len(body) or manifest['aliases'] != {'lm_head.weight': EMBEDDING}:
            raise ValueError('Tiny ownership inventory differs')

    def _item(self, name):
        name = self.manifest['aliases'].get(name, name)
        if name == self.fail_tensor or (self.fail_layer is not None and name.startswith(f'model.layers.{self.fail_layer}.')):
            raise RuntimeError('injected upstream weight read failure')
        return name, self.manifest['tensors'][name]

    def shape(self, name):
        return tuple(self._item(name)[1]['shape'])

    def tensor(self, name):
        name, item = self._item(name)
        if name in (EMBEDDING, PLE):
            raise ValueError('Whole embedding materialization forbidden')
        self.requests.append(('tensor', name, item['bytes']))
        raw = self.body[item['offset']:item['offset']+item['bytes']]
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(item['shape'])

    def gather(self, name, ids):
        name, item = self._item(name)
        if name not in (EMBEDDING, PLE) or not 1 <= len(ids) <= 32:
            raise ValueError('Bounded embedding gather required')
        if any(type(i) is not int or not 0 <= i < item['shape'][0] for i in ids):
            raise ValueError('Embedding index differs')
        stride = item['shape'][1]*2
        raw = b''.join(self.body[item['offset']+i*stride:item['offset']+(i+1)*stride] for i in ids)
        self.requests.append(('gather', name, len(raw)))
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(len(ids), item['shape'][1])

    def head(self, start, end):
        _, item = self._item(EMBEDDING)
        if not 0 <= start < end <= item['shape'][0] or end-start > 128:
            raise ValueError('Bounded tied-head rows required')
        stride = item['shape'][1]*2
        raw = self.body[item['offset']+start*stride:item['offset']+end*stride]
        self.requests.append(('head', EMBEDDING, len(raw)))
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(end-start, item['shape'][1])


def freeze(model):
    state = model.state_dict()
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError('Actual tied embedding ownership required')
    body = bytearray(); tensors = {}
    for name, value in sorted(state.items()):
        if name == 'lm_head.weight':
            continue
        raw = tensor_bytes(value)
        tensors[name] = dict(shape=list(value.shape), offset=len(body), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        body.extend(raw)
    return dict(format='toklet-upstream-gemma-tiny-weights-v1', tensors=tensors,
                aliases={'lm_head.weight':EMBEDDING}, sha256=hashlib.sha256(body).hexdigest()), bytes(body)


class LayerLease(torch.nn.Module):
    def __init__(self, config, index, reader, observer):
        super().__init__()
        self.config, self.index, self.reader, self.observer = config, index, reader, observer
        self.active = None

    def forward(self, *args, **kwargs):
        if self.observer['live'] != 0:
            raise RuntimeError('Multiple streamed layers materialized')
        self.observer['live'] += 1
        self.observer['peak'] = max(self.observer['peak'], self.observer['live'])
        layer = None
        values = {}
        try:
            with bf16_parameters(), torch.device('meta'):
                layer = upstream.Gemma4TextDecoderLayer(self.config, self.index)
            for name in layer.state_dict():
                values[name] = self.reader.tensor(f'model.layers.{self.index}.{name}')
            layer.load_state_dict(values, strict=True, assign=True)
            values.clear()
            self.active = layer.eval().requires_grad_(False)
            if any(p.device.type != 'cpu' or p.dtype != torch.bfloat16 for p in layer.parameters()):
                raise ValueError('Streamed layer parameter dtype/device differs')
            shared = kwargs['shared_kv_states']
            attention = layer.self_attn
            if attention.is_kv_shared_layer:
                producer = attention.kv_shared_layer_index
                actual = shared[producer]; saved = self.observer['producers'][producer]
                if any(a.data_ptr()!=b.data_ptr() for a,b in zip(actual,saved,strict=True)):
                    raise ValueError('Shared consumer changed producer tensors')
                if any(tensor_bytes(a)!=tensor_bytes(b) for a,b in zip(actual,self.observer['producer_copies'][producer],strict=True)):
                    raise ValueError('Shared producer values changed')
                self.observer['consumers'].append((self.index,producer))
            result = layer(*args, **kwargs)
            self.observer['outputs'][self.index] = result.detach().clone()
            if attention.store_full_length_kv:
                self.observer['producers'][self.index] = shared[self.index]
                self.observer['producer_copies'][self.index] = tuple(v.detach().clone() for v in shared[self.index])
            return result
        finally:
            self.active = None
            values.clear()
            if layer is not None:
                layer.to_empty(device='meta')
            del layer
            self.observer['live'] -= 1


class TiledHead(torch.nn.Module):
    def __init__(self, reader, vocab, tile):
        super().__init__()
        if not 1 <= tile <= 128:
            raise ValueError('Bounded head tile required')
        self.reader, self.vocab, self.tile = reader, vocab, tile

    def forward(self, hidden):
        return torch.cat([torch.nn.functional.linear(hidden, self.reader.head(start,min(start+self.tile,self.vocab)))
                          for start in range(0,self.vocab,self.tile)], dim=-1)


class Streamed:
    def __init__(self, config, reader, head_tile=4):
        require_runtime()
        self.config, self.reader = copy.deepcopy(config), reader
        if (reader.shape(EMBEDDING)!=(config.vocab_size,config.hidden_size)
                or reader.shape(PLE)!=(config.vocab_size_per_layer_input,config.num_hidden_layers*config.hidden_size_per_layer_input)):
            raise ValueError('Embedding geometry differs')
        self.observer = dict(live=0,peak=0,outputs={},producers={},producer_copies={},consumers=[])
        with bf16_parameters(), torch.device('meta'):
            self.model = upstream.Gemma4ForCausalLM(self.config)
        roots = {'model.norm.weight','model.per_layer_model_projection.weight','model.per_layer_projection_norm.weight'}
        state = self.model.state_dict()
        if {n for n in state if not n.startswith('model.layers.') and n not in (EMBEDDING,PLE,'lm_head.weight')} != roots:
            raise ValueError('Upstream root inventory differs')
        loaded = self.model.load_state_dict({n:reader.tensor(n) for n in roots}, strict=False, assign=True)
        if loaded.unexpected_keys or set(loaded.missing_keys) != set(state)-roots:
            raise ValueError('Unexpected root materialization')
        self.model.model.rotary_emb = upstream.Gemma4TextRotaryEmbedding(self.config, device='cpu')
        self.model.model.layers = torch.nn.ModuleList([LayerLease(self.config,i,reader,self.observer) for i in range(config.num_hidden_layers)])
        self.model.lm_head = TiledHead(reader,config.vocab_size,head_tile)
        self.model.eval().requires_grad_(False)
        self.scaled_embeddings = {}
        for name, width, scale in [(EMBEDDING,config.hidden_size,config.hidden_size**.5),
                                  (PLE,config.num_hidden_layers*config.hidden_size_per_layer_input,
                                   config.hidden_size_per_layer_input**.5)]:
            with torch.device('cpu'):
                compact = upstream.Gemma4TextScaledWordEmbedding(1,width,None,scale).to(dtype=torch.bfloat16)
            self.scaled_embeddings[name] = compact.eval().requires_grad_(False)

    def embeddings(self, ids, name):
        if ids.device.type != 'cpu' or ids.dtype != torch.int64 or ids.ndim != 2 or ids.numel()>32:
            raise ValueError('Bounded CPU token IDs required')
        unique, inverse = torch.unique(ids,sorted=True,return_inverse=True)
        values = self.reader.gather(name,unique.tolist())
        compact = self.scaled_embeddings[name]
        compact.weight = torch.nn.Parameter(values,requires_grad=False)
        return compact(inverse)

    def __call__(self, ids):
        self.observer.update(outputs={},producers={},producer_copies={},consumers=[])
        embedded = self.embeddings(ids,EMBEDDING)
        ple = self.embeddings(ids,PLE).reshape(
            *ids.shape,self.config.num_hidden_layers,self.config.hidden_size_per_layer_input)
        with torch.inference_mode():
            return self.model(inputs_embeds=embedded,per_layer_inputs=ple,
                              attention_mask=torch.ones_like(ids),use_cache=False).logits
