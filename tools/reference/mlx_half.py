"""Freeze independent MLX packed-weight/Half-activation graph gates before GPU wiring."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tests/fixtures/mlx-qwen-v1'
SOURCE_SHA = 'fdd390667091a78d69b2f09086a3919eb7499ee1f065c7944801f65a5e1953cd'
DEFAULT = ROOT / 'tests/fixtures/mlx-half-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['format'] != 'toklet-mlx-half-reference-v1' or manifest['source_manifest_sha256'] != SOURCE_SHA:
        raise ValueError('Unexpected MLX Half fixture identity')
    raw = (folder / 'arrays.bin').read_bytes()
    if sha(raw) != manifest['array_sha256'] or len(raw) != manifest['array_bytes']:
        raise ValueError('MLX Half payload identity differs')
    for name, entry in manifest['arrays'].items():
        data = raw[entry['offset']:entry['offset'] + entry['bytes']]
        if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
            raise ValueError('MLX Half array identity differs: ' + name)
    return manifest


def export(folder):
    if folder.exists():
        raise FileExistsError('Refusing to overwrite a frozen fixture')
    with (SOURCE / 'manifest.json').open('rb') as stream:
        source_bytes = stream.read(1024 * 1024 + 1)
    if len(source_bytes) > 1024 * 1024 or sha(source_bytes) != SOURCE_SHA:
        raise ValueError('Admitted MLX source fixture manifest differs')
    source = json.loads(source_bytes)
    files = {}
    for name, entry in source['files'].items():
        data = (SOURCE / name).read_bytes()
        if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
            raise ValueError('Admitted MLX source file differs: ' + name)
        files[name] = data
    producers = [Path(__file__).resolve(), ROOT / 'tools/reference/fp16.py', ROOT / 'tools/reference/tiny.py']
    producer_pins = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in producers}
    import numpy as np
    import fp16
    import tiny
    if np.__version__ != '2.3.2':
        raise ValueError('Pinned NumPy 2.3.2 required')
    checkpoint = files['model/model.safetensors']
    size = struct.unpack_from('<Q', checkpoint)[0]
    header = json.loads(checkpoint[8:8 + size])
    tensors = {}
    for name, entry in header.items():
        if name == '__metadata__':
            continue
        dtype = {'F32': '<f4', 'U32': '<u4'}[entry['dtype']]
        begin, end = entry['data_offsets']
        tensors[name] = np.frombuffer(checkpoint[8 + size + begin:8 + size + end], dtype=dtype).reshape(entry['shape'])
    canonical, weights = {}, {}
    embedding = None
    for name, tensor in tensors.items():
        if name.endswith(('.scales', '.biases')):
            continue
        key = name.removeprefix('model.').replace('.self_attn.', '.').replace('.mlp.', '.')
        if key in ('embed_tokens.weight', 'norm.weight') or key.endswith(('input_layernorm.weight', 'post_attention_layernorm.weight')):
            key = key.removesuffix('.weight')
        module = name.removesuffix('.weight')
        if module + '.scales' in tensors:
            config = source['model']['quantization']
            config = config.get(module, config)
            bits, group = config['bits'], config['group_size']
            codes = np.stack([(tensor >> s) & ((1 << bits) - 1) for s in range(0, 32, bits)], axis=-1).reshape(len(tensor), -1)
            scales = np.repeat(tensors[module + '.scales'].astype(np.float64), group, axis=-1)
            biases = np.repeat(tensors[module + '.biases'].astype(np.float64), group, axis=-1)
            value = (scales * codes.astype(np.float64) + biases).astype(np.float32)
            weights[key] = value if key != 'embed_tokens' else value.astype(np.float16)
            if key == 'embed_tokens':
                embedding = value
        else:
            value = tensor.copy()
            weights[key] = value.astype(np.float16)
        canonical[key] = value
    if embedding is None:
        raise ValueError('Fixture must exercise packed tied embeddings')
    arrays, raw = {}, bytearray()

    def forward(tokens, offset, precision):
        values = fp16.forward(tokens, offset, weights, tiny.MODEL, precision)
        # Input gather rounds to Half. Tied output uses the same original packed
        # embedding parameters reconstructed to FP32, not that rounded table.
        values['logits'] = fp16.linear(values['final_norm'], embedding, None, precision, np.float32)
        return values

    def save(name, value, wide, original):
        value = np.asarray(value)
        if not all(np.isfinite(v).all() for v in (value, wide, original)):
            raise ValueError('Nonfinite MLX Half oracle value')
        dtype = '<f2' if value.dtype == np.float16 else '<f4'
        data = value.astype(dtype).tobytes()
        envelope = float(np.max(np.abs(value.astype(np.float64) - wide.astype(np.float64))))
        storage = max(float(np.max(np.abs(v.astype(np.float64) - original.astype(np.float64)))) for v in (value, wide))
        arrays[name] = dict(offset=len(raw), bytes=len(data), sha256=sha(data), dtype=dtype, shape=list(value.shape),
                            precision_envelope=envelope, gate=dict(atol=max(1e-4, 2 * envelope), rtol=1e-3),
                            storage_envelope=storage, gate_against_source_fp32=dict(atol=2 * storage + 1e-5, rtol=1e-4))
        raw.extend(data)

    cases = {}
    for name, case in source['cases'].items():
        tokens, offset = case['tokens'], case['position_offset']
        main, wide = forward(tokens, offset, np.float32), forward(tokens, offset, np.float64)
        original = tiny.forward(tokens, offset, canonical, np.float32)
        for stage, value in main.items():
            save(f'cases.{name}.{stage}', value, wide[stage], original[stage][0])
        cases[name] = dict(tokens=tokens, position_offset=offset, greedy=np.argmax(main['logits'], axis=-1).tolist())
    history, greedy, source_greedy = [1, 7, 13, 3], [], []
    for step in range(4):
        main, wide = forward(history, 0, np.float32), forward(history, 0, np.float64)
        original = tiny.forward(history, 0, canonical, np.float32)['logits'][0][-1]
        save(f'greedy.step_{step}.logits', main['logits'][-1], wide['logits'][-1], original)
        source_token, token = int(np.argmax(original)), int(np.argmax(main['logits'][-1]))
        source_greedy.append(source_token)
        greedy.append(token)
        history.append(source_token)
    # Recheck admitted inputs and producers immediately before publication.
    if sha((SOURCE / 'manifest.json').read_bytes()) != SOURCE_SHA:
        raise ValueError('Source fixture changed during export')
    for name, data in files.items():
        if (SOURCE / name).read_bytes() != data:
            raise ValueError('Consumed source file changed: ' + name)
    for name, digest in producer_pins.items():
        if sha((ROOT / name).read_bytes()) != digest:
            raise ValueError('Producer changed during export: ' + name)
    folder.mkdir()
    (folder / 'arrays.bin').write_bytes(raw)
    manifest = dict(format='toklet-mlx-half-reference-v1', source_manifest_sha256=SOURCE_SHA, numpy=np.__version__,
                    producer_pins=producer_pins, model=source['model'], cases=cases, array_bytes=len(raw), array_sha256=sha(raw), arrays=arrays,
                    greedy=dict(prompt=[1, 7, 13, 3], history='Frozen canonical FP32 teacher-forced tokens', source_tokens=source_greedy, reconstructed_tokens=greedy),
                    policy=dict(reference='Packed FP32 reconstructed projections/tied output; Half input embedding, exclusions, activations and KV; FP32 scores/logits; ordered scalar FP32 with independent FP64 envelope.',
                                gate='max(1e-4, 2*independent_FP32_FP64_max_abs) + 1e-3*abs(reference)', source_gate='2*independent_Half_vs_canonical_FP32_max_abs + 1e-5 + 1e-4*abs(source)',
                                rationale='Existing tiny Q4/Half numerical policies, frozen before native MLX complete-model GPU execution; held-out quality gates are unchanged.'),
                    scope='Independent tiny numerical oracle only; no target complete-model GPU, real checkpoint, quality, performance or mobile qualification.')
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(folder)
    print(json.dumps(dict(arrays=len(arrays), bytes=len(raw), source_tokens=source_greedy, reconstructed_tokens=greedy)))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export', type=Path)
    mode.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.export:
        export(args.export)
    else:
        print(json.dumps(dict(verified=True, arrays=len(verify(args.check)['arrays']))))
