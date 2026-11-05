"""Authored BF16 import oracle and immutable real metadata; no engine dependency."""
from pathlib import Path
import hashlib
import json
import struct
import sys
ROOT=Path(__file__).resolve().parents[2]
def sha(b):return hashlib.sha256(b).hexdigest()
def export():
    config=json.loads((ROOT/'tests/fixtures/granite-moe-source-v1/real-config.json').read_text())
    config.update(hidden_size=2,intermediate_size=3,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,vocab_size=3,num_local_experts=3,num_experts_per_tok=2,max_position_embeddings=16)
    h,i,e=2,3,3
    specs={'model.embed_tokens.weight':[3,h],'model.norm.weight':[h]}
    for name,shape in {'input_layernorm.weight':[h],'post_attention_layernorm.weight':[h],'self_attn.q_proj.weight':[h,h],'self_attn.k_proj.weight':[h,h],'self_attn.v_proj.weight':[h,h],'self_attn.o_proj.weight':[h,h],'block_sparse_moe.router.layer.weight':[e,h],'block_sparse_moe.input_linear.weight':[e,2*i,h],'block_sparse_moe.output_linear.weight':[e,h,i]}.items():specs['model.layers.0.'+name]=shape
    header={};payload=bytearray();expected=[]
    # Exactly representable source BF16 bit patterns and independent Half encodings.
    pairs=[(0x0000,0x0000),(0x8000,0x8000),(0x3f80,0x3c00),(0xbf80,0xbc00),(0x3380,0x0001),(0x0001,0x0000),(0x8001,0x8000),(0x3e80,0x3400)]
    cursor=0
    for name,shape in sorted(specs.items()):
        count=1
        for v in shape:count*=v
        start=len(payload);half=[]
        for _ in range(count):
            bf,hf=pairs[cursor%len(pairs)];cursor+=1;payload+=struct.pack('<H',bf);half.append(hf)
        header[name]=dict(dtype='BF16',shape=shape,data_offsets=[start,len(payload)])
        expected.append(dict(name=name,half_bits=half))
    config_raw=(json.dumps(config,indent=2)+'\n').encode()
    raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    container=struct.pack('<Q',len(raw))+raw+payload
    half_raw=b''.join(struct.pack('<H',x)for t in expected for x in t['half_bits'])
    return config_raw,container,dict(format='toklet-granite-moe-source-v1',producer_sha256=sha(Path(__file__).read_bytes()),config_sha256=sha(config_raw),model_sha256=sha(container),model_bytes=len(container),half_sha256=sha(half_raw),underflows=sum(x in [0x0001,0x8001]for x in [struct.unpack('<H',payload[j:j+2])[0]for j in range(0,len(payload),2)]),tensors=expected,
        real_metadata=dict(repository='ibm-granite/granite-3.1-1b-a400m-instruct',revision='0da7a48b0276d500ce5922fd2b33944091fc6c09',model_bytes=2669283096,expected_model_sha256='ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1',config_sha256=sha((ROOT/'tests/fixtures/granite-moe-source-v1/real-config.json').read_bytes()),header_sha256=sha((ROOT/'tests/fixtures/granite-moe-source-v1/real-header.json').read_bytes())))
if __name__=='__main__':
    directory=Path(sys.argv[1]);directory.mkdir(exist_ok=True)
    config,model,manifest=export()
    for name,raw in [('config.json',config),('model.safetensors',model),('manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())]:
        with (directory/name).open('xb')as file:file.write(raw)
