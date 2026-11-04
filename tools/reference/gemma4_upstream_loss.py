"""Bounded teacher-forced scorer around the pinned stock BF16 text model.

One complete causal text forward, then full-vocabulary projection in at most32
position batches. This reference path is independent of Toklet kernels and is
not an application backend or a full-quality verdict.
"""
import torch
import bf16_source
import gemma4_upstream_stream as stream
import gemma4_upstream_weights as weights

MAX_POSITIONS = 1025
POSITION_BATCH = 32


def memory_ledger(config, positions):
    bf16_source.integer(positions,2,MAX_POSITIONS)
    # Conservative simultaneous bound: eager FP32 attention scores/probabilities,
    # PLE/gather copies, every observed layer output, full BF16 tied table,
    # one loaded layer, and FP64 logsumexp/reduction temporaries.
    return dict(runtime=512*1024**2,head=2*config.vocab_size*config.hidden_size,
        roots_and_layer=192*1024**2,
        eager_attention=4*4*config.num_attention_heads*positions*positions,
        ple_and_embeddings=4*2*positions*(config.num_hidden_layers*config.hidden_size_per_layer_input+config.hidden_size),
        layer_outputs=2*positions*config.hidden_size*config.num_hidden_layers,
        shared_kv_and_scratch=128*1024**2,
        logits_and_reductions=32*config.vocab_size*32)


class LossReference:
    def __init__(self, reader, *, positions):
        self.config,self.reader,self.positions = reader.config,reader,positions
        self.ledger = memory_ledger(self.config,positions)
        if sum(self.ledger.values()) > 6*1024**3:
            raise ValueError('Reference process memory admission exceeds6GiB')
        self.streamed = stream.Streamed(self.config,reader,128)
        self.head = None

    def _ids(self, ids):
        if (ids.device.type != 'cpu' or ids.dtype != torch.int64 or ids.ndim != 2
                or ids.shape[0] != 1 or not 2 <= ids.shape[1] <= self.positions):
            raise ValueError('One bounded CPU token prefix required')
        if bool(((ids<0)|(ids>=min(self.config.vocab_size,self.config.vocab_size_per_layer_input))).any()):
            raise ValueError('Token outside embedding vocabularies')

    def _embeddings(self, ids, name):
        unique,inverse = torch.unique(ids,sorted=True,return_inverse=True)
        rows = torch.cat([self.reader.gather(name,unique[start:start+32].tolist())
                          for start in range(0,unique.numel(),32)],dim=0)
        compact = self.streamed.scaled_embeddings[name]
        compact.weight = torch.nn.Parameter(rows,requires_grad=False)
        return compact(inverse)

    def _inputs(self, ids):
        self._ids(ids)
        self.streamed.observer.update(outputs={},producers={},producer_copies={},consumers=[])
        embedded = self._embeddings(ids,stream.EMBEDDING)
        ple = self._embeddings(ids,stream.PLE).reshape(
            *ids.shape,self.config.num_hidden_layers,self.config.hidden_size_per_layer_input)
        return dict(inputs_embeds=embedded,per_layer_inputs=ple,
                    attention_mask=torch.ones_like(ids),use_cache=False)

    def load_head(self):
        if self.head is None:
            shape = self.reader.shape(stream.EMBEDDING)
            table = torch.empty(shape,dtype=torch.bfloat16,device='cpu')
            for start in range(0,shape[0],128):
                end = min(start+128,shape[0])
                table[start:end].copy_(self.reader.head(start,end))
            with torch.device('meta'):
                head = torch.nn.Linear(shape[1],shape[0],bias=False,dtype=torch.bfloat16)
            head.weight = torch.nn.Parameter(table,requires_grad=False)
            self.head = head.eval()
        return self.head

    def _logits(self, hidden):
        logits = self.load_head()(hidden)
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = logits/cap
            logits = torch.tanh(logits)
            logits = logits*cap
        return logits

    def score(self, ids, *, sink=None):
        """Score exactly ids[1:] with rows0..N-2; never retain all logits."""
        with torch.inference_mode():
            hidden = self.streamed.model.model(**self._inputs(ids)).last_hidden_state
            nlls,greedy = [],[]
            for start in range(0,ids.shape[1]-1,POSITION_BATCH):
                end = min(start+POSITION_BATCH,ids.shape[1]-1)
                logits = self._logits(hidden[:,start:end])
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError('Finite reference logits required')
                if sink is not None:
                    sink(start,end,logits)
                values = logits.double()
                targets = ids[:,start+1:end+1]
                losses = values.logsumexp(-1)-values.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
                nlls.extend(losses[0].tolist())
                greedy.extend(logits.argmax(-1)[0].tolist())
                del values,losses,logits
            if len(nlls) != ids.shape[1]-1:
                raise ValueError('Exact next-token coverage differs')
            return dict(targets=len(nlls),nlls=nlls,greedy=greedy)

    def stock_logits(self, ids):
        """Pilot-only stock causal-LM comparison, at most32 scoring rows."""
        self._ids(ids)
        if ids.shape[1]>33:
            raise ValueError('Stock full-logit pilot exceeds32 targets')
        self.streamed.model.lm_head = self.load_head()
        with torch.inference_mode():
            return self.streamed.model(**self._inputs(ids),
                logits_to_keep=torch.arange(ids.shape[1]-1,dtype=torch.int64)).logits
