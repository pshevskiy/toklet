"""Tiny mixed RoPE/NoPE Half graphs using unchanged independent dense primitives.

No native execution. Main FP32-ordered / wide FP64 share source bytes and Half
stores. The existing Half envelope is frozen before target collection.
"""
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import llama3_half_graph as dense
from llama3_half_graph import f32, half, flatten, norm, linear, attention

ROOT = Path(__file__).resolve().parents[2]


def rotary(x, heads, offset, wide):
    output = [list(row) for row in x]
    dim = 8
    with localcontext() as context:
        context.prec = 80
        for pair in range(dim//2):
            inverse = (Decimal(10000).ln()*Decimal(-2*pair)/dim).exp() if wide else f32(math.pow(10000, -2*pair/dim))
            for token, row in enumerate(x):
                angle = float(Decimal(offset+token)*inverse) if wide else f32(f32(offset+token)*inverse)
                sine, cosine = math.sin(angle), math.cos(angle)
                if not wide:
                    sine, cosine = f32(sine), f32(cosine)
                for head in range(heads):
                    a = head*dim+pair
                    b = a+dim//2
                    left = row[a]*cosine-row[b]*sine if wide else f32(f32(row[a]*cosine)-f32(row[b]*sine))
                    right = row[b]*cosine+row[a]*sine if wide else f32(f32(row[b]*cosine)+f32(row[a]*sine))
                    output[token][a], output[token][b] = half(left), half(right)
    return output


def forward(weights, config, tokens, offset, wide):
    precision = float if wide else f32
    captured = {}

    def save(name, values):
        values_flat = flatten(values)
        if not all(math.isfinite(v) for v in values_flat):
            raise ValueError('Nonfinite stage: ' + name)
        captured[name] = values_flat
        return values

    x = save('embedding', [[half(v) for v in weights['model.embed_tokens.weight'][token]] for token in tokens])
    epsilon = f32(config['rms_norm_eps'])
    for layer in range(2):
        p, stage = f'model.layers.{layer}', f'layers.{layer}.'
        normalized = save(stage + 'input_norm', norm(x, weights[p + '.input_layernorm.weight'], epsilon, precision))
        q = save(stage + 'q_projection', linear(normalized, weights[p + '.self_attn.q_proj.weight'], precision))
        k = save(stage + 'k_projection', linear(normalized, weights[p + '.self_attn.k_proj.weight'], precision))
        v = save(stage + 'v_projection', linear(normalized, weights[p + '.self_attn.v_proj.weight'], precision))
        q = save(stage + 'q_rotary', rotary(q, 4, offset, wide) if config['no_rope_layers'][layer] else q)
        k = save(stage + 'k_rotary', rotary(k, 2, offset, wide) if config['no_rope_layers'][layer] else k)
        scores, probabilities, value = attention(q, k, v, precision)
        save(stage + 'attention_scores', scores)
        save(stage + 'attention_probabilities', probabilities)
        value = save(stage + 'attention_context', value)
        projected = save(stage + 'attention_projection', linear(value, weights[p + '.self_attn.o_proj.weight'], precision))
        x = save(stage + 'attention_residual', [[half(precision(a + b)) for a, b in zip(row, delta, strict=True)]
                 for row, delta in zip(x, projected, strict=True)])
        normalized = save(stage + 'post_attention_norm', norm(x, weights[p + '.post_attention_layernorm.weight'], epsilon, precision))
        gate = save(stage + 'gate_projection', linear(normalized, weights[p + '.mlp.gate_proj.weight'], precision))
        up = save(stage + 'up_projection', linear(normalized, weights[p + '.mlp.up_proj.weight'], precision))
        active = []
        for row in gate:
            out = []
            for v in row:
                exponential = precision(math.exp(precision(-abs(v))))
                denominator = precision(precision(1) + exponential)
                sigmoid = precision(precision(1) / denominator) if v >= 0 else precision(exponential / denominator)
                out.append(half(precision(v * sigmoid)))
            active.append(out)
        save(stage + 'gate_silu', active)
        gated = save(stage + 'gated', [[half(precision(a * b)) for a, b in zip(row, other, strict=True)]
                     for row, other in zip(active, up, strict=True)])
        down = save(stage + 'down_projection', linear(gated, weights[p + '.mlp.down_proj.weight'], precision))
        x = save(stage + 'output', [[half(precision(a + b)) for a, b in zip(row, delta, strict=True)]
                 for row, delta in zip(x, down, strict=True)])
    final = save('final_norm', norm(x, weights['model.norm.weight'], epsilon, precision))
    head = weights['model.embed_tokens.weight'] if config['tie_word_embeddings'] else weights['lm_head.weight']
    save('logits', linear(final, head, precision, f32))
    return captured


def export():
    source = dense.source
    manifest = json.loads((source.SOURCE/'manifest.json').read_text())
    assert hashlib.sha256((source.SOURCE/'manifest.json').read_bytes()).hexdigest() == source.SOURCE_SHA
    inputs = {str((source.SOURCE/'manifest.json').relative_to(ROOT)): dense.pin(source.SOURCE/'manifest.json')}
    cases, templates = [], {}
    for variant in ['tied', 'untied']:
        for name in ['config.json', 'model.safetensors']:
            path = source.SOURCE/variant/name
            assert dense.pin(path) == manifest['files'][variant+'/'+name]
            inputs[str(path.relative_to(ROOT))] = dense.pin(path)
        weights, config, _ = dense.read_weights(variant, manifest)
        config = dict(config, model_type='smollm3', architectures=['SmolLM3ForCausalLM'],
            max_position_embeddings=512, no_rope_layers=[1, 0] if variant=='tied' else [0, 1],
            layer_types=['full_attention']*2, no_rope_layer_interval=4, pretraining_tp=2,
            max_window_layers=2, sliding_window=None, use_sliding_window=False)
        if variant == 'tied':
            config.pop('tie_word_embeddings')
        templates[variant] = config
        effective = dict(config, tie_word_embeddings=(variant=='tied'))
        for name, tokens, offset in dense.CASES:
            main = forward(weights, effective, tokens, offset, False)
            wide = forward(weights, effective, tokens, offset, True)
            shapes = dense.contracts(len(tokens))
            assert set(main)==set(wide)==set(shapes)
            arrays = {}
            for stage, shape in shapes.items():
                assert len(main[stage])==len(wide[stage])==math.prod(shape)
                dtype = 'f32le' if stage.endswith(('attention_scores', 'attention_probabilities', 'logits')) else 'f16le'
                envelope = max(abs(a-b) for a,b in zip(main[stage],wide[stage],strict=True))
                arrays[stage] = dict(shape=shape, dtype=dtype,
                    main_bits=[dense.bits(v,dtype) for v in main[stage]],
                    wide_bits=[dense.bits(v,dtype) for v in wide[stage]],
                    precision_envelope=envelope, gate=dict(atol=max(1e-4,2*envelope),rtol=1e-3))
            cases.append(dict(variant=variant,name=name,tokens=tokens,position_offset=offset,initial_kv_tokens=0,arrays=arrays))
    for variant in ['tied','untied']:
        normal = next(c for c in cases if c['variant']==variant and c['name']=='prompt')
        changed = next(c for c in cases if c['variant']==variant and c['name']=='future_changed')
        for profile in ['main_bits','wide_bits']:
            assert normal['arrays']['logits'][profile][:192]==changed['arrays']['logits'][profile][:192]
    producers = {str(p.relative_to(ROOT)): dense.pin(p) for p in [Path(__file__).resolve(),Path(dense.__file__),Path(source.__file__),Path(dense.positional.__file__)]}
    return dict(format='toklet-smollm3-half-graph-v1',policy=dense.POLICY,inputs=inputs,producers=producers,
        config_templates=templates,cases=cases,stage_count=312,
        scope='Tiny mixed rotary selection only; no real model, quality, app or phone qualification')


if __name__ == '__main__':
    destination = Path(sys.argv[1])
    with destination.open('x') as file:
        json.dump(export(),file,indent=2,allow_nan=False)
        file.write('\n')
