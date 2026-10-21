"""Export only the frozen tiny numerical weights to a local loader fixture.

No model download, tokenizer, changed arithmetic or random weights are involved.
Used by native C and Swift integration checks; output must be a new directory.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys


def export(root, destination):
    fixture = root / 'tests/fixtures/tiny-v1'
    manifest_bytes = (fixture / 'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != '1d40cec4f24ece5fdc7b727d3e96f41237621120908833ee6c7df83352acfdd4':
        raise ValueError('Frozen tiny manifest differs')
    manifest = json.loads(manifest_bytes)
    c = manifest['model']
    header, payload = {}, bytearray()
    for name, array in sorted(manifest['arrays'].items()):
        if not name.startswith('weights.'):
            continue
        raw = (fixture / array['file']).read_bytes()
        assert len(raw) == array['bytes']
        assert hashlib.sha256(raw).hexdigest() == array['sha256']
        assert array['dtype'] == 'f32le'
        if name == 'weights.embed_tokens':
            target = 'model.embed_tokens.weight'
        elif name == 'weights.norm':
            target = 'model.norm.weight'
        else:
            _, _, layer, op, *suffix = name.split('.')
            if op in ('input_layernorm', 'post_attention_layernorm'):
                target = f'model.layers.{layer}.{op}.weight'
            else:
                block = 'self_attn' if op in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'
                target = f'model.layers.{layer}.{block}.{op}.{suffix[0]}'
        assert target not in header
        header[target] = dict(dtype='F32', shape=array['shape'], data_offsets=[len(payload), len(payload)+len(raw)])
        payload.extend(raw)
    metadata = json.dumps(header, separators=(',', ':')).encode()
    metadata += b' ' * (-len(metadata) % 8)
    config = dict(model_type='qwen2', architectures=['Qwen2ForCausalLM'], hidden_act='silu',
                  hidden_size=c['hidden_size'], intermediate_size=c['intermediate_size'],
                  num_hidden_layers=c['num_layers'], num_attention_heads=c['num_query_heads'],
                  num_key_value_heads=c['num_kv_heads'], vocab_size=c['vocab_size'],
                  max_position_embeddings=c['capacity'], rms_norm_eps=c['rms_norm_eps'],
                  rope_theta=c['rope_theta'], tie_word_embeddings=True, torch_dtype='float32')
    expected = manifest['arrays']['greedy.tokens']
    raw = (fixture / expected['file']).read_bytes()
    assert expected['dtype'] == 'u32le' and expected['shape'] == [4]
    assert len(raw) == expected['bytes'] == 16
    assert hashlib.sha256(raw).hexdigest() == expected['sha256']
    expected_ids = list(struct.unpack('<4I', raw))
    assert expected_ids == [47, 47, 47, 47]
    destination.mkdir()
    (destination / 'config.json').write_text(json.dumps(config)+'\n')
    (destination / 'model.safetensors').write_bytes(struct.pack('<Q', len(metadata))+metadata+payload)
    print(json.dumps(dict(fixture=str(destination), expected_ids=expected_ids)))


if __name__ == '__main__':
    export(Path(__file__).resolve().parents[2], Path(sys.argv[1]))
