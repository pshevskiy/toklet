"""Exact BF16-source weights, temporary FP32 parameters, bounded eager oracle.

Only explicit reference export imports PyTorch. Original parameter data and tied
Parameter identity are restored even if a forward operation throws. No graph or
second complete FP32 model is retained. This is not an inference optimization.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/experiments"))
import native_usage

LIMIT=6*1024**3
ALLOWANCE=768*1024**2


class Parameters:
    def __init__(self,model,torch):
        self.model,self.torch=model,torch
        self.original={id(p):(p,p.data) for p in model.parameters()}
        if any(data.dtype!=torch.bfloat16 for _,data in self.original.values()):
            raise ValueError("Streamed source parameters must be exact BF16")
        self.resident=sum(data.numel()*data.element_size() for _,data in self.original.values())
        self.largest=max(data.numel() for _,data in self.original.values())
        self.admitted=self.resident+self.largest*8+ALLOWANCE
        if self.admitted>LIMIT:raise ValueError("Streamed oracle admission exceeds6GiB")
        self.active={};self.handles=[];self.samples=[]
        self.tied=model.lm_head.weight
        self.check_tied()
        if model.model.rotary_emb.inv_freq.dtype!=torch.float32:
            raise ValueError("Independent rotary frequencies must be constructed in FP32")
        for module in [model.model.embed_tokens,*model.model.layers,model.model.norm,model.lm_head]:
            self.handles.append(module.register_forward_pre_hook(self.enter))
            self.handles.append(module.register_forward_hook(self.leave,always_call=True))
        self.observe("resident")

    def check_tied(self):
        if self.model.lm_head.weight is not self.tied or self.model.model.embed_tokens.weight is not self.tied:
            raise ValueError("Streamed oracle lost tied Parameter identity")

    def observe(self,phase):
        value=native_usage.usage()
        if not 0<value["lifetime_max_phys_footprint"]<=LIMIT:
            raise ValueError("Streamed oracle native charge exceeds6GiB")
        self.samples.append({"phase":phase,"usage":value})

    def enter(self,module,inputs):
        if self.active:raise ValueError("Nested FP32 parameter conversion is not admitted")
        self.check_tied()
        parameters=list(module.parameters())
        extra=sum(p.numel()*4 for p in parameters)
        if self.resident+extra+ALLOWANCE>LIMIT:raise ValueError("FP32 scope exceeds oracle admission")
        try:
            for p in parameters:
                original=self.original[id(p)][1]
                if p.data.data_ptr()!=original.data_ptr():raise ValueError("Parameter was not restored before conversion")
                self.active[id(p)]=p
                p.data=original.float()
            self.check_tied();self.observe("fp32_scope")
        except BaseException:
            self.restore()
            raise

    def restore(self):
        for key,p in self.active.items():p.data=self.original[key][1]
        self.active.clear()
        self.check_tied()

    def leave(self,module,inputs,output):
        self.restore()

    def assert_restored(self):
        self.check_tied()
        if self.active or any(p.data.data_ptr()!=data.data_ptr() or p.dtype!=self.torch.bfloat16 for p,data in self.original.values()):
            raise ValueError("Oracle parameters still converted after forward")

    def report(self):
        self.assert_restored();self.observe("completed")
        return {"format":"toklet-streamed-reference-memory-v1","resident_bf16_bytes":self.resident,
                "largest_transient_fp64_bytes":self.largest*8,"application_framework_allowance_bytes":ALLOWANCE,
                "admitted_bytes":self.admitted,"samples":self.samples,
                "scope":"BF16 resident + one largest FP64 conversion + fixed allowance; FP32 scopes restored before functional FP64 oracle. Native charge may omit clean/shared pages."}

    def close(self):
        self.restore()
        for handle in self.handles:handle.remove()
        self.handles.clear()
