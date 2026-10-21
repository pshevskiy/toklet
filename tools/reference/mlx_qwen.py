"""Explicit tiny MLX-format Qwen exporter; stdlib-only offline hash checking.

Frozen tiny weights -> actual MLX quantization -> independently unpacked FP32
weights -> existing NumPy graph oracle. Actual mlx-lm CPU/GPU logits separately
validate supported zero-offset cases against unchanged tiny-v1 tolerances.
"""
import argparse
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'tests/fixtures/mlx-qwen-v1'
TINY_MANIFEST_SHA = '1d40cec4f24ece5fdc7b727d3e96f41237621120908833ee6c7df83352acfdd4'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['format'] != 'toklet-mlx-qwen-reference-v1':
        raise ValueError('Unknown MLX Qwen fixture format')
    if manifest['source_fixture_manifest_sha256'] != TINY_MANIFEST_SHA:
        raise ValueError('Unexpected source fixture identity')
    for name, identity in manifest['files'].items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Fixture path traversal')
        data = path.read_bytes()
        if len(data) != identity['bytes'] or sha(data) != identity['sha256']:
            raise ValueError('MLX Qwen fixture file identity differs: ' + name)
    raw = (folder / 'arrays.bin').read_bytes()
    for name, entry in manifest['arrays'].items():
        data = raw[entry['offset']:entry['offset'] + entry['bytes']]
        if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
            raise ValueError('MLX Qwen fixture array differs: ' + name)
    return manifest


def export(folder):
    if folder.exists():
        raise FileExistsError('Refusing to replace an immutable fixture')
    if version('mlx') != '0.32.3' or version('mlx-lm') != '0.32.0':
        raise ValueError('Pinned MLX 0.32.3 / mlx-lm 0.32.0 required')
    base_folder = ROOT / 'tests/fixtures/tiny-v1'
    manifest_path = base_folder / 'manifest.json'
    with manifest_path.open('rb') as source:
        manifest_bytes = source.read(1024 * 1024 + 1)
    if len(manifest_bytes) > 1024 * 1024 or sha(manifest_bytes) != TINY_MANIFEST_SHA:
        raise ValueError('Frozen tiny-v1 manifest identity differs')
    base_manifest = json.loads(manifest_bytes)
    # Resolve selected producer files through package metadata without importing
    # numerical modules. Capture these identities before imports/execution.
    mlx_distribution = distribution('mlx')
    core_files = [p for p in mlx_distribution.files if str(p).startswith('mlx/core.') and str(p).endswith('.so')]
    if len(core_files) != 1:
        raise ValueError('Expected one identified MLX core extension')
    producer_files = [Path(__file__).resolve(), ROOT / 'tools/reference/tiny.py',
                      Path(distribution('mlx-lm').locate_file('mlx_lm/models/qwen2.py')).resolve(),
                      Path(mlx_distribution.locate_file(core_files[0])).resolve()]
    producer_pins = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in producer_files}
    versions = {name: version(name) for name in ['mlx', 'mlx-lm', 'numpy']}
    consumed_source_pins = {}
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen2 import Model, ModelArgs
    import tiny

    config = dict(model_type='qwen2', architectures=['Qwen2ForCausalLM'], hidden_act='silu',
                  hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                  max_position_embeddings=32, rms_norm_eps=1e-6, rope_theta=1e6,
                  tie_word_embeddings=True, torch_dtype='float32')
    excluded = 'model.layers.0.self_attn.o_proj'
    overridden = 'model.layers.1.mlp.down_proj'
    quantization = dict(bits=4, group_size=32)
    quantization[excluded] = False
    quantization[overridden] = dict(bits=8, group_size=64)
    config['quantization'] = quantization
    payloads, canonical = {}, {}
    mx.set_default_device(mx.cpu)
    for key, entry in base_manifest['arrays'].items():
        if not key.startswith('weights.'):
            continue
        source_name = key.removeprefix('weights.')
        data = (base_folder / entry['file']).read_bytes()
        if sha(data) != entry['sha256']:
            raise ValueError('Frozen source array differs: ' + key)
        consumed_source_pins[entry['file']] = sha(data)
        original = np.frombuffer(data, dtype='<f4').reshape(entry['shape']).copy()
        name = 'model.' + source_name
        if source_name.startswith('layers.'):
            _, index, leaf, *tail = source_name.split('.')
            if leaf in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                name = f'model.layers.{index}.self_attn.{leaf}.' + '.'.join(tail)
            elif leaf in ('gate_proj', 'up_proj', 'down_proj'):
                name = f'model.layers.{index}.mlp.{leaf}.' + '.'.join(tail)
        if source_name in ('embed_tokens', 'norm') or source_name.endswith(('input_layernorm', 'post_attention_layernorm')):
            name += '.weight'
        module = name.removesuffix('.weight')
        if original.ndim == 2 and module != excluded:
            bits, group = (8, 64) if module == overridden else (4, 32)
            w, s, b = mx.quantize(mx.array(original), group_size=group, bits=bits)
            mx.eval(w, s, b)
            payloads[name] = w
            payloads[module + '.scales'] = s
            payloads[module + '.biases'] = b
            # Independent fixed-width unpacking for the fixture's 4/8-bit cases.
            words = np.asarray(w, dtype=np.uint32)
            codes = np.stack([(words >> shift) & ((1 << bits) - 1) for shift in range(0, 32, bits)], axis=-1).reshape(original.shape)
            scales = np.repeat(np.asarray(s, dtype=np.float64), group, axis=-1)
            biases = np.repeat(np.asarray(b, dtype=np.float64), group, axis=-1)
            canonical[source_name] = (scales * codes.astype(np.float64) + biases).astype(np.float32)
        else:
            payloads[name] = mx.array(original)
            canonical[source_name] = original
    model = Model(ModelArgs.from_dict(config))

    def predicate(path, module):
        if path == excluded or not hasattr(module, 'to_quantized'):
            return False
        return dict(bits=8, group_size=64) if path == overridden else True

    nn.quantize(model, group_size=32, bits=4, class_predicate=predicate)
    model.load_weights(list(payloads.items()), strict=True)
    mx.eval(model.parameters())
    arrays, raw, native_counts = {}, bytearray(), dict(CPU=0, GPU=0)

    def save(name, values, operation):
        value = np.asarray(values, dtype='<f4')
        data = value.tobytes()
        if not np.all(np.isfinite(value)):
            raise ValueError('Nonfinite oracle output: ' + name)
        arrays[name] = dict(offset=len(raw), bytes=len(data), sha256=sha(data), shape=list(value.shape), operation=operation)
        raw.extend(data)

    for name, case in tiny.CASES.items():
        references = tiny.forward(case['tokens'], case['position_offset'], canonical, np.float32)
        for stage, (array, operation) in references.items():
            save(f'cases.{name}.{stage}', array, operation)
        if case['position_offset'] == 0:
            for label, device in [('CPU', mx.cpu), ('GPU', mx.gpu)]:
                with mx.stream(device):
                    logits = model(mx.array([case['tokens']], dtype=mx.uint32))
                    mx.eval(logits)
                    actual = np.asarray(logits)[0]
                np.testing.assert_allclose(actual, references['logits'][0], **tiny.TOLERANCES['logits'])
                save(f'native.{label}.{name}.logits', actual, 'logits')
                native_counts[label] += 1
    # Publish only while all selected code and consumed input paths still match
    # their admitted bytes. This identifies selected producers, not a complete
    # shared-library or Python dependency closure.
    if sha(manifest_path.read_bytes()) != TINY_MANIFEST_SHA:
        raise ValueError('Source manifest changed during export')
    for name, digest in consumed_source_pins.items():
        if sha((base_folder / name).read_bytes()) != digest:
            raise ValueError('Consumed source array changed during export: ' + name)
    for name, digest in producer_pins.items():
        if sha((ROOT / name).read_bytes()) != digest:
            raise ValueError('Selected producer changed during export: ' + name)
    if versions != {name: version(name) for name in versions}:
        raise ValueError('Producer package version changed during export')
    folder.mkdir()
    (folder / 'model').mkdir()
    mx.save_safetensors(str(folder / 'model/model.safetensors'), payloads)
    (folder / 'model/config.json').write_text(json.dumps(config, indent=2) + '\n')
    (folder / 'arrays.bin').write_bytes(raw)
    files = {}
    for name in ['model/config.json', 'model/model.safetensors', 'arrays.bin']:
        data = (folder / name).read_bytes()
        files[name] = dict(bytes=len(data), sha256=sha(data))
    manifest = dict(format='toklet-mlx-qwen-reference-v1', model=config, cases=tiny.CASES,
                    policy=dict(tolerances=tiny.TOLERANCES, reconstruction='Independent unsigned 4/8-bit codes, FP64 affine multiply/add, FP32 weights; existing NumPy FP32 graph.', native='Actual mlx-lm quantized Qwen2 FP32 CPU/GPU logits for zero-offset cases only; not identical arithmetic or quality qualification.'),
                    source_fixture_manifest_sha256=TINY_MANIFEST_SHA, consumed_source_array_pins=consumed_source_pins,
                    producer=dict(mlx=versions['mlx'], mlx_lm=versions['mlx-lm'], numpy=versions['numpy'], selected_file_pins=producer_pins,
                                  endpoint_inputs_unchanged=True, complete_dependency_closure_claimed=False),
                    native_case_counts=native_counts, files=files, arrays=arrays)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(folder)
    print(json.dumps(dict(arrays=len(arrays), bytes=len(raw), native_cases=native_counts)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export', type=Path)
    mode.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.export:
        export(args.export)
    else:
        manifest = verify(args.check)
        print(json.dumps(dict(verified=True, arrays=len(manifest['arrays']))))
