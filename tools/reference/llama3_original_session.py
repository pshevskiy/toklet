"""Original FP32 reference with session-owned KV; no widened weight cache.

Keys are rotated and values unrotated, each contiguous [layer,kv_head,capacity,dim].
Only the valid prefix is read. Tentative layer writes stay beyond the committed
prefix, so a failed step can be retried without rolling back old cache rows.
"""
import math
import numpy as np
import llama3_original_fp32 as graph


def greedy(row):
    if row.ndim!=1 or not len(row) or not np.isfinite(row).all():
        raise ValueError('Finite reference distribution required')
    return int(np.argmax(row))  # First maximum is the lowest token ID.


def attention(q, k, v, offset):
    rows,heads,dim=q.shape
    result=np.empty_like(q)
    future=np.arange(k.shape[1])[None,:] > np.arange(offset,offset+rows)[:,None]
    group=heads//k.shape[0]
    for head in range(heads):
        scores=(q[:,head]@k[head//group].T)*np.float32(1/math.sqrt(dim))
        scores[future]=-np.inf
        scores-=np.max(scores,axis=-1,keepdims=True)
        probability=np.exp(scores)
        probability/=np.sum(probability,axis=-1,keepdims=True,dtype=np.float32)
        result[:,head]=probability@v[head//group]
    return result


class Session:
    def __init__(self, reader, capacity=566):
        if type(capacity) is not int or not 1<=capacity<=566:
            raise ValueError('Reference session capacity requires1..=566')
        self.reader=reader;self.capacity=capacity;s=reader.shape
        self.k=np.empty((s.layers,s.kv_heads,capacity,s.head_dim),dtype=np.float32)
        self.v=np.empty_like(self.k)
        self.position=0;self.history=[];self.logits=None

    @property
    def kv_bytes(self):
        return self.k.nbytes+self.v.nbytes

    def reset(self):
        self.position=0;self.history.clear();self.logits=None

    def forward(self, ids):
        s=self.reader.shape
        if (not isinstance(ids,list) or not ids or len(ids)+self.position>self.capacity
                or any(type(i) is not int or not 0<=i<s.vocab for i in ids)
                or (self.position>0 and len(ids)!=1)):
            raise ValueError('Reference requires admitted prefill or one decode token')
        next_history=self.history+ids  # Allocate before any state can be published.
        start=self.position;end=start+len(ids);reader=self.reader
        x=reader.embedding(ids)
        def linear(hidden,name):return hidden@reader.weight(name).T
        for layer in range(s.layers):
            base=f'model.layers.{layer}.'
            normalized=graph.rms(x,reader.weight(base+'input_layernorm.weight'))
            q=graph.rotary(linear(normalized,base+'self_attn.q_proj.weight').reshape(-1,s.heads,s.head_dim),start)
            k=graph.rotary(linear(normalized,base+'self_attn.k_proj.weight').reshape(-1,s.kv_heads,s.head_dim),start)
            v=linear(normalized,base+'self_attn.v_proj.weight').reshape(-1,s.kv_heads,s.head_dim)
            self.k[layer,:,start:end]=k.transpose(1,0,2);self.v[layer,:,start:end]=v.transpose(1,0,2)
            context=attention(q,self.k[layer,:,:end],self.v[layer,:,:end],start)
            x=x+linear(context.reshape(-1,s.hidden),base+'self_attn.o_proj.weight')
            normalized=graph.rms(x,reader.weight(base+'post_attention_layernorm.weight'))
            gate=linear(normalized,base+'mlp.gate_proj.weight')
            up=linear(normalized,base+'mlp.up_proj.weight')
            activated=(gate/(np.float32(1)+np.exp(-gate)))*up
            x=x+linear(activated,base+'mlp.down_proj.weight')
        hidden=graph.rms(x[-1:],reader.weight('model.norm.weight'))
        logits=np.empty(s.vocab,dtype=np.float32)
        for first in range(0,s.vocab,256):
            last=min(first+256,s.vocab)
            logits[first:last]=(hidden@reader.weight('model.embed_tokens.weight',first,last).T)[0]
        greedy(logits)  # Validate before publishing any new committed state.
        logits.flags.writeable=False
        self.history=next_history;self.logits=logits;self.position=end
        return logits
