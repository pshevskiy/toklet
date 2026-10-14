"""Independent bounded group-8 Q4 v2 numerical-reference tensor reader.

No model-wide FP32/FP64 conversion or source mapping: source bytes stream through
bounded stripes. Only the tied Half table is cached. Metadata lookup is outside
the numerical engine and this module never imports Rust or Metal.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import numpy as np

STRIPE=1024
DECODE_ROWS=128


def sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda:stream.read(65536),b""):result.update(data)
    return result.hexdigest()


def descriptor_sha(stream):
    position = stream.tell()
    try:
        stream.seek(0)
        digest = hashlib.sha256()
        for data in iter(lambda: stream.read(65536), b""):
            digest.update(data)
        return digest.hexdigest()
    finally:
        stream.seek(position)


def bounded_bytes(path,limit,expected=None):
    # Hash exactly the bytes returned from this descriptor, before parsing.
    with Path(path).open("rb") as stream:
        data=stream.read(limit+1)
    if len(data)>limit:raise ValueError("Independent input exceeds bound")
    identity=hashlib.sha256(data).hexdigest()
    if expected is not None and identity!=expected:raise ValueError("Independent input identity differs")
    return data,identity


def decode_q4(data,rows,columns):
    if len(data)<64 or type(rows) is not int or type(columns) is not int or rows<=0 or columns<=0:
        raise ValueError("Independent Q4 shape/header differs")
    magic,version,group,r,c,stride,payload=struct.unpack("<8sIIQQQQ16x",data[:64])
    groups=(columns+7)//8
    expected_stride=((groups*8+63)//64)*64
    if (magic,version,group,r,c,stride,payload,len(data))!=(b"TOKQ4V2\0",2,8,rows,columns,expected_stride,rows*expected_stride,64+rows*expected_stride):
        raise ValueError("Independent Q4 header/layout differs")
    if any(data[48:64]):raise ValueError("Independent Q4 reserved bytes differ")
    if not 0<rows*columns<=151936*1536:raise ValueError("Independent Q4 tensor exceeds bounded shapes")
    result=np.empty((rows,columns),dtype=np.float32)
    record=np.dtype({"names":["scale","bias","codes"],"formats":["<f2","<f2",("u1",4)],"offsets":[0,2,4],"itemsize":8})
    row_bytes=np.ndarray((rows,stride),dtype=np.uint8,buffer=data,offset=64)
    if row_bytes[:,groups*8:].any():raise ValueError("Independent Q4 row padding differs")
    for start in range(0,rows,DECODE_ROWS):
        size=min(DECODE_ROWS,rows-start)
        values=np.ndarray((size,groups),dtype=record,buffer=data,offset=64+start*stride,strides=(stride,8))
        scale=values["scale"].astype(np.float32);bias=values["bias"].astype(np.float32)
        if not np.isfinite(scale).all() or not np.isfinite(bias).all() or (values["scale"].view("<u2")&0x8000).any():raise ValueError("Independent Q4 metadata differs")
        codes=np.empty((size,groups,8),dtype=np.uint8)
        codes[:,:,0::2]=values["codes"]&15
        codes[:,:,1::2]=values["codes"]>>4
        if codes[scale==0].any():raise ValueError("Independent Q4 constant codes differ")
        if columns%8 and codes[:,-1,columns%8:].any():raise ValueError("Independent Q4 column padding differs")
        expanded=codes.astype(np.float32)
        # Separate Float32 operations retain the format's reconstruction oracle.
        expanded*=scale[:,:,None];expanded+=bias[:,:,None]
        result[start:start+size]=expanded.reshape(size,groups*8)[:,:columns]
        # Release the owning stripe arrays before constructing the next stripe.
        # Keep the 9-byte/value admission conservative without two expanded copies.
        del expanded,codes,scale,bias,values
    return result


class Weights:
    def __init__(self,path,expected_sha,packed=None):
        self.path=Path(path);self.stream=self.path.open("rb");self.cached={};self.entries={}
        try:
            if descriptor_sha(self.stream)!=expected_sha:raise ValueError("Independent source descriptor hash differs")
            self.stream.seek(0)
            size=struct.unpack("<Q",self.stream.read(8))[0]
            if not 2<=size<=16*1024**2:raise ValueError("Independent source header exceeds bound")
            self.header=json.loads(self.stream.read(size));self.base=8+size
            position=0
            for item in sorted((v for k,v in self.header.items() if k!="__metadata__"),key=lambda v:v["data_offsets"][0]):
                shape=item["shape"]
                if not 1<=len(shape)<=2 or any(type(v) is not int or v<=0 for v in shape) or item["dtype"]!="BF16":raise ValueError("Independent source dtype/shape differs")
                begin,end=item["data_offsets"]
                if begin!=position or end-begin!=math.prod(shape)*2:raise ValueError("Independent source range differs")
                position=end
            if self.base+position!=os.fstat(self.stream.fileno()).st_size:raise ValueError("Independent source payload size differs")
            self.expected_sha=expected_sha;self.packed=Path(packed) if packed else None
            if self.packed:
                path=self.packed/"manifest.json"
                data,self.manifest_sha=bounded_bytes(path,1024**2)
                manifest=json.loads(data)
                _,self.config_sha=bounded_bytes(self.path.parent/"config.json",1024**2,manifest["config_sha256"])
                if manifest["format"]!="toklet-q4-model-v2" or manifest["quantizer"]!="toklet-q4-affine-f32-source-half-metadata-rne-g8-v2":raise ValueError("Independent Q4 quantizer differs")
                if manifest["excluded_dtype"]!="F16" or manifest["exclusions"]!=["norms","qkv_biases","tied_embedding_output"]:
                    raise ValueError("Independent v2 exclusions differ")
                if manifest["sources"]!=[{"file":self.path.name,"sha256":expected_sha}]:
                    raise ValueError("Independent Q4 source/config binding differs")
                self.entries={item["name"]:item for item in manifest["matrices"]}
                if len(self.entries)!=len(manifest["matrices"]):raise ValueError("Independent Q4 inventory repeats")
                expected={name for name in self.header if name.endswith(".weight") and name.split(".")[-2] in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")}
                if set(self.entries)!=expected:raise ValueError("Independent Q4 inventory differs")
                for name,item in self.entries.items():
                    if item["file"]!=name+".q4" or [item["rows"],item["columns"]]!=self.header[name]["shape"]:
                        raise ValueError("Independent Q4 matrix binding differs")
        except BaseException:
            self.stream.close();raise

    def __enter__(self):return self
    def __exit__(self,*exc):self.stream.close();self.cached.clear()

    @staticmethod
    def name(key):
        if key in ("embed_tokens","norm"):return "model."+key+".weight"
        parts=key.split(".")
        if parts[2] in ("input_layernorm","post_attention_layernorm"):return "model."+key+".weight"
        group="self_attn" if parts[2] in ("q_proj","k_proj","v_proj","o_proj") else "mlp"
        return ".".join(["model",*parts[:2],group,*parts[2:]])

    def __getitem__(self,key):
        if key in self.cached:return self.cached[key]
        name=self.name(key);item=self.header[name];shape=item["shape"]
        if name in self.entries:
            entry=self.entries[name];path=self.packed/entry["file"]
            if type(entry["packed_bytes"]) is not int or not 64<=entry["packed_bytes"]<=16*1024**2:raise ValueError("Independent packed tensor size differs")
            data,_=bounded_bytes(path,entry["packed_bytes"],entry["packed_sha256"])
            if len(data)!=entry["packed_bytes"]:raise ValueError("Independent packed tensor size differs")
            result=decode_q4(data,*shape)
        else:
            elements=math.prod(shape)
            if elements>151936*1536:raise ValueError("Independent source tensor exceeds bound")
            result=np.empty(elements,dtype=np.float16)
            step=STRIPE*(shape[-1] if len(shape)>1 else 1)
            self.stream.seek(self.base+item["data_offsets"][0])
            for start in range(0,elements,step):
                count=min(step,elements-start);data=self.stream.read(count*2)
                if len(data)!=count*2:raise ValueError("Independent source tensor shortened")
                words=np.frombuffer(data,dtype="<u2").astype(np.uint32)
                np.left_shift(words,16,out=words)
                result[start:start+count]=words.view(np.float32)
            result=result.reshape(shape)
            if not np.isfinite(result).all():raise ValueError("Independent Half source range overflow")
        if key=="embed_tokens":self.cached[key]=result
        return result

    def verify(self):
        if descriptor_sha(self.stream)!=self.expected_sha:raise ValueError("Independent source descriptor endpoint differs")
        if sha(self.path)!=self.expected_sha:raise ValueError("Independent source endpoint differs")
        if self.packed:
            if sha(self.path.parent/"config.json")!=self.config_sha:raise ValueError("Independent config endpoint differs")
            if sha(self.packed/"manifest.json")!=self.manifest_sha:raise ValueError("Independent packed inventory endpoint differs")
            for item in self.entries.values():
                if sha(self.packed/item["file"])!=item["packed_sha256"]:raise ValueError("Independent packed endpoint differs")
