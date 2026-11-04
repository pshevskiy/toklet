"""Exact official text inventory and bounded raw BF16 upstream weight reader.

Reference-only single-caller adapter. Source owns authentication/lifetime; this
reader borrows it and never widens or materializes either embedding table.
"""
import hashlib
import json
import math
import torch
import bf16_source
import gemma4_upstream_stream as stream

FIXTURE = stream.ROOT/'tests/fixtures/gemma-unquantized-header-v1'
CONFIG_SHA = '1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330'
LAYOUT_SHA = '2419f70e7232fd75a6d361e0ef8e8acaf37f87f52124e63e1e04e814d1040840'


def official_name(name):
    if name == 'lm_head.weight':
        name = stream.EMBEDDING
    if not name.startswith('model.'):
        raise ValueError('Stock text state name required')
    return 'model.language_model.' + name[len('model.'):]


def admit():
    """Cold metadata admission: every persistent stock state has one source."""
    stream.require_runtime()
    config_raw = (FIXTURE/'config.json').read_bytes()
    layout_raw = (FIXTURE/'layout.json').read_bytes()
    if (hashlib.sha256(config_raw).hexdigest() != CONFIG_SHA
            or hashlib.sha256(layout_raw).hexdigest() != LAYOUT_SHA):
        raise ValueError('Official config/layout identity differs')
    document = json.loads(config_raw)
    config = stream.Gemma4TextConfig(**document['text_config'])
    config._attn_implementation = 'eager'
    config.use_cache = False
    with stream.bf16_parameters(), torch.device('meta'):
        model = stream.upstream.Gemma4ForCausalLM(config)
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError('Stock tied parameter ownership differs')
    state = model.state_dict()
    selected = json.loads(layout_raw)['selected_text']
    unused = json.loads(layout_raw)['unused_shared_kv']
    expected = {official_name(name): list(value.shape) for name,value in state.items()}
    if (len(state) != 541 or len(expected) != 540 or set(expected) != set(selected)
            or any(item['dtype'] != 'BF16' or item['shape'] != expected[name]
                   or item['bytes'] != 2*math.prod(expected[name]) for name,item in selected.items())):
        raise ValueError('Complete persistent stock/source inventory differs')
    excluded = {}
    for index,layer in enumerate(model.model.layers):
        if layer.self_attn.is_kv_shared_layer:
            dim = layer.self_attn.head_dim
            heads = config.num_attention_heads // layer.self_attn.num_key_value_groups
            for suffix,shape in [('k_proj.weight',[heads*dim,config.hidden_size]),
                                 ('v_proj.weight',[heads*dim,config.hidden_size]),
                                 ('k_norm.weight',[dim])]:
                excluded[official_name(f'model.layers.{index}.self_attn.{suffix}')] = shape
    if (len(excluded) != 60 or set(excluded) != set(unused) or set(unused)&set(selected)
            or any(item['dtype'] != 'BF16' or item['shape'] != excluded[name]
                   or item['bytes'] != 2*math.prod(excluded[name]) for name,item in unused.items())):
        raise ValueError('Explicit unused shared-consumer inventory differs')
    if sum(item['bytes'] for item in selected.values()) != 9257138758:
        raise ValueError('Selected BF16 byte accounting differs')
    roots = ('model.norm.weight','model.per_layer_model_projection.weight','model.per_layer_projection_norm.weight')
    ledger = dict(runtime_allowance=512*1024**2,
        roots=sum(selected[official_name(n)]['bytes'] for n in roots),
        largest_layer=max(sum(item['bytes'] for name,item in selected.items()
            if name.startswith(f'model.language_model.layers.{index}.')) for index in range(config.num_hidden_layers)),
        read_and_decode_tiles=2*bf16_source.MAX_READ_BYTES,
        captures_and_attention=64*1024**2)
    if sum(ledger.values()) > 6*1024**3:
        raise ValueError('Four-token reference memory allowance exceeds process policy')
    return config, layout_raw, dict(selected=540, excluded=60, persistent_layer_scalars=35,
        selected_bytes=9257138758, tied_head_source=official_name(stream.EMBEDDING),
        four_token_memory_ledger=ledger, memory_allowance_bytes=sum(ledger.values()),
        process_policy_bytes=6*1024**3)


class OfficialWeights:
    def __init__(self, source, *, deadline):
        self.config, self.layout_raw, self.inventory = admit()
        self.source, self.deadline = source, deadline
        selected = json.loads(self.layout_raw)['selected_text']
        self.entries = {name.replace('model.language_model.','model.',1):item for name,item in selected.items()}
        self.requests = []
        # Bind every view eagerly, before numerical execution or any payload read.
        self.rows = {name:bf16_source.official_rows(source,official_name(name),self.layout_raw,
                     deadline=deadline) for name in self.entries}

    def _name(self, name):
        name = stream.EMBEDDING if name == 'lm_head.weight' else name
        if name not in self.entries:
            raise ValueError('Selected persistent text tensor required')
        return name

    def shape(self, name):
        return tuple(self.entries[self._name(name)]['shape'])

    def _decode(self, raw, shape):
        value = torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(shape)
        if not bool(torch.isfinite(value).all()):
            self.source.poison('Nonfinite upstream BF16 weight')
            raise ValueError('Finite upstream BF16 weight required')
        return value

    def tensor(self, name):
        name = self._name(name)
        if name in (stream.EMBEDDING,stream.PLE):
            raise ValueError('Whole embedding materialization forbidden')
        row = self.rows[name]
        value = torch.empty((row.rows,row.columns),dtype=torch.bfloat16,device='cpu')
        self.requests.append(('tensor',name))
        for start in range(0,row.rows,128):
            end = min(start+128,row.rows)
            value[start:end].copy_(self._decode(row.read_raw(start,end),(end-start,row.columns)))
        return value.reshape(self.shape(name))

    def gather(self, name, ids):
        name = self._name(name)
        if name not in (stream.EMBEDDING,stream.PLE) or type(ids) is not list or not 1 <= len(ids) <= 32:
            raise ValueError('Bounded embedding gather required')
        row = self.rows[name]
        # Validate before recording a request that could otherwise be partial.
        for index in ids:
            bf16_source.integer(index,0,row.rows-1)
        self.requests.append(('gather',name,list(ids)))
        return self._decode(row.gather_raw(ids),(len(ids),row.columns))

    def head(self, start, end):
        row = self.rows[stream.EMBEDDING]
        bf16_source.integer(start,0,row.rows-1)
        bf16_source.integer(end,start+1,min(row.rows,start+128))
        self.requests.append(('head',stream.EMBEDDING,start,end))
        return self._decode(row.read_raw(start,end),(end-start,row.columns))


def request_spans(requests, selected):
    """Independent layout traversal for replaying the successful read chain.

    Deliberately does not call Rows or the reader's extent calculations.
    """
    for request in requests:
        operation,name,*args = request
        item = selected[official_name(name)]
        shape = item['shape']
        stride = 2*shape[-1]
        base = item['file_offset']
        if operation == 'tensor':
            count = shape[0] if len(shape) == 2 else 1
            for start in range(0,count,128):
                yield base+start*stride,min(128,count-start)*stride
        elif operation == 'gather':
            for index in args[0]:
                yield base+index*stride,stride
        elif operation == 'head':
            start,end = args
            yield base+start*stride,(end-start)*stride
        else:
            raise ValueError('Unknown logical read operation')


def planned_requests(config, tokens, head_tile=128):
    """Freeze complete root/layer/gather/head consumption before real outputs."""
    if (type(tokens) is not list or not 1 <= len(tokens) <= 32
            or any(type(i) is not int or not 0 <= i < min(config.vocab_size,config.vocab_size_per_layer_input) for i in tokens)
            or type(head_tile) is not int or not 1 <= head_tile <= 128):
        raise ValueError('Bounded four-token diagnostic plan required')
    roots = ['model.norm.weight','model.per_layer_model_projection.weight','model.per_layer_projection_norm.weight']
    result = [('tensor',name) for name in sorted(roots)]
    result.extend(('gather',name,sorted(set(tokens))) for name in (stream.EMBEDDING,stream.PLE))
    with stream.bf16_parameters(),torch.device('meta'):
        model = stream.upstream.Gemma4ForCausalLM(config)
    for index,layer in enumerate(model.model.layers):
        result.extend(('tensor',f'model.layers.{index}.{name}') for name in layer.state_dict())
    result.extend(('head',stream.EMBEDDING,start,min(start+head_tile,config.vocab_size))
                  for start in range(0,config.vocab_size,head_tile))
    return result
