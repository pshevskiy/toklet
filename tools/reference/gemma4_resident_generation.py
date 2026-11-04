"""Resident original BF16 weights and session-owned stock KV for task reference.

The PLE table stays on the authenticated descriptor. This is a CPU reference
harness, with a567-position task bound, not a Toklet backend or app admission.
"""
import copy
import math
import torch
from transformers.cache_utils import DynamicCache
import bf16_source
import gemma4_upstream_stream as stream

MAX_POSITIONS = 567


class Resident:
    def __init__(self, reader):
        stream.require_runtime()
        self.reader,self.config = reader,copy.deepcopy(reader.config)
        with stream.bf16_parameters(),torch.device('meta'):
            self.model = stream.upstream.Gemma4ForCausalLM(self.config)
        state = self.model.state_dict()
        canonical = [name for name in state if name not in (stream.PLE,'lm_head.weight')]
        weights_bytes = sum(2*math.prod(reader.shape(name)) for name in canonical)
        # Count physical producer caches, not query heads or shared consumers.
        kv_bytes=0
        for layer in self.model.model.layers:
            attention=layer.self_attn
            if not attention.is_kv_shared_layer:
                heads=self.config.num_attention_heads//attention.num_key_value_groups
                capacity=min(MAX_POSITIONS,self.config.sliding_window) if attention.is_sliding else MAX_POSITIONS
                kv_bytes+=4*heads*capacity*attention.head_dim
        self.ledger = dict(weights=weights_bytes,runtime=512*1024**2,
            kv=kv_bytes,scratch=256*1024**2,loading=3*bf16_source.MAX_READ_BYTES)
        if sum(self.ledger.values()) > 6*1024**3:
            raise ValueError('Resident reference admission exceeds6GiB')
        values={};table=None
        try:
            for name in canonical:
                if name==stream.EMBEDDING:
                    shape=reader.shape(name)
                    table=torch.empty(shape,dtype=torch.bfloat16,device='cpu')
                    for start in range(0,shape[0],128):
                        end=min(start+128,shape[0]);table[start:end].copy_(reader.head(start,end))
                    values[name]=table
                else:
                    values[name]=reader.tensor(name)
            values['lm_head.weight']=values[stream.EMBEDDING]
            loaded=self.model.load_state_dict(values,strict=False,assign=True)
            if loaded.unexpected_keys or loaded.missing_keys!=[stream.PLE]:
                raise ValueError('Resident persistent state inventory differs')
            self.model.lm_head.weight=self.model.model.embed_tokens.weight
            self.model.model.embed_tokens.embed_scale=torch.tensor(self.config.hidden_size**.5,dtype=torch.bfloat16,device='cpu')
            self.model.model.rotary_emb=stream.upstream.Gemma4TextRotaryEmbedding(self.config,device='cpu')
            if any(value.device.type!='cpu' or value.dtype!=torch.bfloat16
                   for name,value in self.model.state_dict().items() if name!=stream.PLE):
                raise ValueError('Resident BF16 CPU state differs')
            with torch.device('cpu'):
                self.ple=stream.upstream.Gemma4TextScaledWordEmbedding(1,
                    self.config.num_hidden_layers*self.config.hidden_size_per_layer_input,None,
                    self.config.hidden_size_per_layer_input**.5).to(dtype=torch.bfloat16)
            self.model.eval().requires_grad_(False);self.ple.eval().requires_grad_(False)
        except BaseException:
            self.model.to_empty(device='meta')
            table=None
            raise
        finally:
            values.clear()

    def inputs(self, ids):
        unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
        rows=torch.cat([self.reader.gather(stream.PLE,unique[start:start+32].tolist())
                        for start in range(0,unique.numel(),32)],dim=0)
        self.ple.weight=torch.nn.Parameter(rows,requires_grad=False)
        return dict(input_ids=ids,per_layer_inputs=self.ple(inverse).reshape(
            *ids.shape,self.config.num_hidden_layers,self.config.hidden_size_per_layer_input))

    def validate(self, tokens):
        if (type(tokens)is not list or not tokens or len(tokens)>MAX_POSITIONS
                or any(type(i)is not int or not 0<=i<min(self.config.vocab_size,self.config.vocab_size_per_layer_input)for i in tokens)):
            raise ValueError('Bounded task token IDs required')
        return torch.tensor([tokens],dtype=torch.int64,device='cpu')

    def stateless(self, tokens):
        ids=self.validate(tokens)
        with torch.inference_mode():
            return self.model(**self.inputs(ids),use_cache=False,logits_to_keep=1).logits


class Session:
    def __init__(self, resident):
        self.resident=resident
        self.reset()

    def reset(self):
        self.cache=DynamicCache(config=self.resident.config)
        self.tokens=[]
        self.replay_required=False

    def forward(self, tokens):
        ids=self.resident.validate(tokens)
        if self.replay_required:
            raise ValueError('Reference cache requires reset and replay')
        if len(self.tokens)+len(tokens)>MAX_POSITIONS:
            raise ValueError('Task context exceeds567positions')
        try:
            with torch.inference_mode():
                logits=self.resident.model(**self.resident.inputs(ids),past_key_values=self.cache,
                    use_cache=True,logits_to_keep=1).logits
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Finite reference logits required')
        except BaseException:
            self.replay_required=True
            raise
        self.tokens.extend(tokens)
        return logits
