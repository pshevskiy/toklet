"""Independent tiny Granite scales; unchanged dense primitives and frozen gates.

Three declared profiles: Half FP32-ordered, Half FP64-wide, scalar FP32 storage.
No target values or execution are used. Old producers/fixtures remain unchanged.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import struct
import llama3_half_graph as dense
import smollm3_half_graph as ordinary
from llama3_half_graph import f32, half, flatten
ROOT=Path(__file__).resolve().parents[2]


def attention(q,k,v,precision,store,multiplier,cpu):
    n=len(q);scores=[[[0.]*n for _ in range(n)]for _ in range(4)]
    probabilities=[[[0.]*n for _ in range(n)]for _ in range(4)]
    output=[[0.]*32 for _ in range(n)]
    dot_precision=f32 if cpu else precision
    mass_precision=float if cpu else precision
    for head in range(4):
        qs,ks=head*8,(head//2)*8
        for token in range(n):
            for key in range(n):
                dot=dense.dot(q[token][qs:qs+8],k[key][ks:ks+8],dot_precision)
                scores[head][token][key]=f32(dot_precision(dot*multiplier))
            valid=scores[head][token][:token+1];maximum=max(valid)
            exponentials=[mass_precision(math.exp(mass_precision(x-maximum)))for x in valid]
            total=mass_precision(0)
            for x in exponentials:total=mass_precision(total+x)
            for i,x in enumerate(exponentials):probabilities[head][token][i]=f32(mass_precision(x/total))
            for feature in range(8):
                output[token][qs+feature]=store(dense.dot(probabilities[head][token][:token+1],
                    [v[i][ks+feature]for i in range(token+1)],dot_precision))
    return scores,probabilities,output


def forward(weights,config,tokens,offset,wide=False,cpu=False,omit=None):
    precision=float if wide or cpu else f32
    store=f32 if cpu else half
    captured={}
    def save(name,value):
        flat=flatten(value);assert all(math.isfinite(v)for v in flat)
        captured[name]=flat;return value
    def linear(x,w,output=None):return dense.linear(x,w,precision,output or store)
    def norm(x,w):
        return dense.source.norm(x,w,f32(config['rms_norm_eps'])) if cpu else dense.norm(x,w,f32(config['rms_norm_eps']),precision)
    def residual(x,delta,which):
        coefficient=store(1 if omit==which else config['residual_multiplier'])
        return [[store(precision(a+store(precision(b*coefficient))))for a,b in zip(row,branch,strict=True)]for row,branch in zip(x,delta,strict=True)]
    embedding=store(1 if omit=='embedding'else config['embedding_multiplier'])
    x=save('embedding',[[store(precision(store(v)*embedding))for v in weights['model.embed_tokens.weight'][t]]for t in tokens])
    for i in range(2):
        p=f'model.layers.{i}';s=f'layers.{i}.'
        normalized=save(s+'input_norm',norm(x,weights[p+'.input_layernorm.weight']))
        q=save(s+'q_projection',linear(normalized,weights[p+'.self_attn.q_proj.weight']))
        k=save(s+'k_projection',linear(normalized,weights[p+'.self_attn.k_proj.weight']))
        v=save(s+'v_projection',linear(normalized,weights[p+'.self_attn.v_proj.weight']))
        rotate=lambda x,heads: dense.source.rotary(x,heads,offset,False) if cpu else ordinary.rotary(x,heads,offset,wide)
        q=save(s+'q_rotary',rotate(q,4));k=save(s+'k_rotary',rotate(k,2))
        multiplier=f32(1/math.sqrt(8)if omit=='attention'else config['attention_multiplier'])
        scores,probs,context=attention(q,k,v,precision,store,multiplier,cpu)
        save(s+'attention_scores',scores);save(s+'attention_probabilities',probs);save(s+'attention_context',context)
        branch=save(s+'attention_projection',linear(context,weights[p+'.self_attn.o_proj.weight']))
        x=save(s+'attention_residual',residual(x,branch,'attention_residual'))
        normalized=save(s+'post_attention_norm',norm(x,weights[p+'.post_attention_layernorm.weight']))
        gate=save(s+'gate_projection',linear(normalized,weights[p+'.mlp.gate_proj.weight']))
        up=save(s+'up_projection',linear(normalized,weights[p+'.mlp.up_proj.weight']))
        active=[]
        for row in gate:
            values=[]
            for value in row:
                e=precision(math.exp(precision(-abs(value))));denominator=precision(1+e)
                sigmoid=precision(1/denominator)if value>=0 else precision(e/denominator)
                values.append(store(precision(value*sigmoid)))
            active.append(values)
        save(s+'gate_silu',active)
        gated=save(s+'gated',[[store(precision(a*b))for a,b in zip(row,other,strict=True)]for row,other in zip(active,up,strict=True)])
        down=save(s+'down_projection',linear(gated,weights[p+'.mlp.down_proj.weight']))
        x=save(s+'output',residual(x,down,'mlp_residual'))
    final=save('final_norm',norm(x,weights['model.norm.weight']))
    head=weights['model.embed_tokens.weight']if config['tie_word_embeddings']else weights['lm_head.weight']
    divisor=f32(1 if omit=='logits'else config['logits_scaling'])
    save('logits',[[f32(v/divisor)for v in row]for row in linear(final,head,f32)])
    return captured


def export():
    source=dense.source;manifest_path=source.SOURCE/'manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==source.SOURCE_SHA
    manifest=json.loads(manifest_path.read_text());inputs={str(manifest_path.relative_to(ROOT)):dense.pin(manifest_path)}
    templates,cases={},[]
    omissions=['embedding','attention','attention_residual','mlp_residual','logits']
    for variant in ['tied','untied']:
        for name in ['config.json','model.safetensors']:
            p=source.SOURCE/variant/name;assert dense.pin(p)==manifest['files'][variant+'/'+name]
            inputs[str(p.relative_to(ROOT))]=dense.pin(p)
        weights,config,_=dense.read_weights(variant,manifest);cpu_weights,_=source.matrices(variant,manifest)
        config=dict(config,model_type='granite',architectures=['GraniteForCausalLM'],max_position_embeddings=512,
            embedding_multiplier=12.,attention_multiplier=.015625,residual_multiplier=.22,logits_scaling=8.,rms_norm_eps=1e-5)
        templates[variant]=config
        for name,tokens,offset in dense.CASES:
            main=forward(weights,config,tokens,offset);wide=forward(weights,config,tokens,offset,True)
            cpu=forward(cpu_weights,config,tokens,offset,cpu=True)
            shapes=dense.contracts(len(tokens));assert set(main)==set(wide)==set(cpu)==set(shapes)
            arrays={}
            for stage,shape in shapes.items():
                assert len(main[stage])==len(wide[stage])==len(cpu[stage])==math.prod(shape)
                dtype='f32le'if stage.endswith(('attention_scores','attention_probabilities','logits'))else'f16le'
                envelope=max(abs(a-b)for a,b in zip(main[stage],wide[stage],strict=True))
                arrays[stage]=dict(shape=shape,dtype=dtype,main_bits=[dense.bits(v,dtype)for v in main[stage]],
                    wide_bits=[dense.bits(v,dtype)for v in wide[stage]],cpu_bits=[dense.bits(v,'f32le')for v in cpu[stage]],
                    precision_envelope=envelope,gate=dict(atol=max(1e-4,2*envelope),rtol=1e-3))
            cases.append(dict(variant=variant,name=name,tokens=tokens,position_offset=offset,initial_kv_tokens=0,arrays=arrays))
            if name=='offset5':
                for omitted in omissions:
                    wrong=forward(weights,config,tokens,offset,omit=omitted)
                    assert any(main[k]!=wrong[k]for k in ['embedding','layers.0.attention_scores','layers.0.attention_residual','layers.0.output','logits']),omitted
    for variant in ['tied','untied']:
        a=next(c for c in cases if c['variant']==variant and c['name']=='prompt')
        b=next(c for c in cases if c['variant']==variant and c['name']=='future_changed')
        for key in ['main_bits','wide_bits','cpu_bits']:assert a['arrays']['logits'][key][:192]==b['arrays']['logits'][key][:192]
    witness=dict(x=.125,r=.0013875961303710938,coefficient=half(.22),scaled=half(.0013875961303710938*half(.22)))
    witness['expected']=half(witness['x']+witness['scaled']);witness['wrong_fused']=half(witness['x']+witness['r']*witness['coefficient'])
    assert witness['expected']==.125244140625 and witness['wrong_fused']!=witness['expected']
    coefficient_witness=None
    for code in range(0x1400,0x6000):
        r=struct.unpack('<e',struct.pack('<H',code))[0]
        expected=half(f32(r*half(.22)));wrong=half(f32(r*f32(.22)))
        if expected!=wrong:
            coefficient_witness=dict(r=r,expected=expected,wrong_fp32_coefficient=wrong)
            break
    assert coefficient_witness is not None
    producers={str(p.relative_to(ROOT)):dense.pin(p)for p in [Path(__file__).resolve(),Path(dense.__file__),Path(source.__file__),Path(ordinary.__file__),Path(dense.positional.__file__)]}
    return dict(format='toklet-granite-half-graph-v1',policy=dense.POLICY,inputs=inputs,producers=producers,
        config_templates=templates,cases=cases,stage_count=312,scale_omissions_detected=omissions,residual_witness=witness,coefficient_witness=coefficient_witness,
        scope='Tiny scale boundaries only; no real model, quality, app or phone qualification')


if __name__=='__main__':
    with Path(sys.argv[1]).open('x')as file:json.dump(export(),file,indent=2,allow_nan=False);file.write('\n')
