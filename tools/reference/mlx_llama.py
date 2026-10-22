"""Independent bias-free Llama fixtures, frozen before native decoder changes.

Export explicitly uses pinned MLX for quantization/upstream checks and NumPy
scalar reconstruction. Check mode is stdlib-only and never imports MLX/NumPy.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import struct
import os
import re
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tests/fixtures/tiny-v1'
SOURCE_SHA = '1d40cec4f24ece5fdc7b727d3e96f41237621120908833ee6c7df83352acfdd4'
DEFAULT = ROOT / 'tests/fixtures/mlx-llama-v1'
VERSIONS = {'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'numpy': '2.3.2'}
SCOPE = 'Independent tiny bias-free full-causal ordinary-RoPE Llama tied/untied oracle only; no native target, real checkpoint, performance or device qualification'
POLICY = dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3,
              source_atol=1e-5, source_rtol=1e-4)



CASES = dict(prompt=dict(tokens=[1, 7, 13, 3], position_offset=0),
    future_changed=dict(tokens=[1, 7, 13, 17], position_offset=0),
    offset=dict(tokens=[7, 13, 3], position_offset=5), single=dict(tokens=[11], position_offset=0),
    capacity=dict(tokens=[(7 * i + 1) % 64 for i in range(32)], position_offset=0))
GEOMETRY = dict(vocab_size=64, num_layers=2, hidden_size=32, num_query_heads=4, num_kv_heads=2, head_dim=8,
    intermediate_size=64, capacity=32, rms_norm_eps=1e-6, rope_theta=10000.0, rope_pairing='split_half',
    tie_word_embeddings=True, projection_biases=[])


def configuration(tied):
    result = dict(model_type='llama', architectures=['LlamaForCausalLM'], hidden_act='silu', hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=64, max_position_embeddings=32, rms_norm_eps=1e-6, rope_theta=10000.0,
        attention_bias=False, mlp_bias=False, tie_word_embeddings=tied, torch_dtype='float32')
    quantization = dict(bits=4, group_size=32)
    quantization['model.layers.0.self_attn.o_proj'] = False
    quantization['model.layers.1.mlp.down_proj'] = dict(bits=8, group_size=64)
    if not tied: quantization['lm_head'] = dict(bits=8, group_size=32)
    result['quantization'] = quantization
    return result


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def captured(path, expected=None, limit=16 * 1024**2):
    with path.open('rb') as stream: raw = stream.read(limit + 1)
    if len(raw) > limit or expected is not None and sha(raw) != expected:
        raise ValueError('Bounded admitted bytes differ: ' + str(path))
    return raw


def forward(tokens, offset, weights, config, precision, storage):
    """Ordered independent reductions; no bias terms, separate explicit head."""
    import numpy as np
    import fp16
    if any(name.endswith('.bias') for name in weights):
        raise ValueError('Bias-free reference rejects explicit projection biases')
    x = weights['embed_tokens'][tokens].astype(storage)
    values = {'embedding': x}
    for layer in range(config['num_layers']):
        prefix = f'layers.{layer}'
        def save(name, value):
            values[prefix + '.' + name] = value
            return value
        normalized = save('input_norm', fp16.rmsnorm(x, weights[prefix + '.input_layernorm'], config['rms_norm_eps'], precision))
        projected = {}
        for name, heads in [('q', config['num_query_heads']), ('k', config['num_kv_heads']), ('v', config['num_kv_heads'])]:
            projected[name] = save(name + '_projection', fp16.linear(normalized, weights[prefix + '.' + name + '_proj.weight'],
                None, precision, storage).reshape(-1, heads, config['head_dim']))
        q = save('q_rotary', fp16.rotary(projected['q'], offset, config['rope_theta'], precision))
        k = save('k_rotary', fp16.rotary(projected['k'], offset, config['rope_theta'], precision))
        n = len(tokens);heads = config['num_query_heads'];dim = config['head_dim'];group = heads // config['num_kv_heads']
        scores = np.empty((heads, n, n), dtype=np.float32)
        probabilities = np.zeros_like(scores)
        context = np.empty((n, heads * dim), dtype=storage)
        for head in range(heads):
            for token in range(n):
                for position in range(n):
                    scores[head, token, position] = precision(fp16.dot(q[token, head], k[position, head // group], precision) / np.sqrt(precision(dim)))
                valid = scores[head, token, :token + 1].astype(precision)
                exponential = np.exp(valid - max(valid));denominator = precision(0)
                for item in exponential: denominator = precision(denominator + item)
                probabilities[head, token, :token + 1] = exponential / denominator
                for feature in range(dim):
                    context[token, head * dim + feature] = fp16.dot(probabilities[head, token, :token + 1],
                        projected['v'][:token + 1, head // group, feature], precision)
        save('attention_scores', scores);save('attention_probabilities', probabilities);save('attention_context', context)
        attention = save('attention_projection', fp16.linear(context, weights[prefix + '.o_proj.weight'], None, precision, storage))
        x = save('attention_residual', (x.astype(precision) + attention).astype(storage))
        normalized = save('post_attention_norm', fp16.rmsnorm(x, weights[prefix + '.post_attention_layernorm'], config['rms_norm_eps'], precision))
        gate = save('gate_projection', fp16.linear(normalized, weights[prefix + '.gate_proj.weight'], None, precision, storage))
        up = save('up_projection', fp16.linear(normalized, weights[prefix + '.up_proj.weight'], None, precision, storage))
        wide = gate.astype(precision);exponential = np.exp(-np.abs(wide))
        sigmoid = np.where(wide >= 0, precision(1) / (precision(1) + exponential), exponential / (precision(1) + exponential))
        active = save('gate_silu', (wide * sigmoid).astype(storage))
        gated = save('gated', (active.astype(precision) * up).astype(storage))
        down = save('down_projection', fp16.linear(gated, weights[prefix + '.down_proj.weight'], None, precision, storage))
        x = save('output', (x.astype(precision) + down).astype(storage))
    values['final_norm'] = fp16.rmsnorm(x, weights['norm'], config['rms_norm_eps'], precision)
    values['logits'] = fp16.linear(values['final_norm'], weights['output'], None, precision, np.float32)
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError('Nonfinite independent Llama graph')
    return values



def array_contract():
    result = {}
    for variant in ['tied', 'untied']:
        for case, description in CASES.items():
            n = len(description['tokens'])
            stages = dict(embedding=[n, 32], final_norm=[n, 32], logits=[n, 64])
            for layer in range(2):
                shapes = dict(input_norm=[n, 32], q_projection=[n, 4, 8], k_projection=[n, 2, 8], v_projection=[n, 2, 8],
                    q_rotary=[n, 4, 8], k_rotary=[n, 2, 8], attention_scores=[4, n, n], attention_probabilities=[4, n, n],
                    attention_context=[n, 32], attention_projection=[n, 32], attention_residual=[n, 32], post_attention_norm=[n, 32],
                    gate_projection=[n, 64], up_projection=[n, 64], gate_silu=[n, 64], gated=[n, 64], down_projection=[n, 32], output=[n, 32])
                stages.update({f'layers.{layer}.{key}': shape for key, shape in shapes.items()})
            for precision in ['fp32', 'half']:
                for stage, shape in stages.items():
                    dtype = '<f2' if precision == 'half' and not stage.endswith(('attention_scores', 'attention_probabilities', 'logits')) else '<f4'
                    result[f'{variant}.{precision}.cases.{case}.{stage}'] = (shape, dtype, 'half_rounding_envelope' if precision == 'half' else 'fp32_reference')
            if description['position_offset'] == 0:
                for backend in ['CPU', 'GPU']:result[f'{variant}.native.{backend}.{case}.logits'] = ([n, 64], '<f4', None)
        for backend in ['CPU', 'GPU']:result[f'{variant}.native.{backend}.cached.logits'] = ([4, 64], '<f4', None)
        for step in range(4):result[f'{variant}.greedy.step_{step}.logits'] = ([64], '<f4', 'half_rounding_envelope')
    result['untied.witness.tied_substitution.logits'] = ([4, 64], '<f4', None)
    return result



def producer_paths():
    site = Path(sysconfig.get_path('purelib')).resolve()
    if not site.is_relative_to(ROOT): raise ValueError('Repository virtual environment required')
    paths = {Path(__file__).resolve(), ROOT / 'tools/reference/fp16.py'}
    for package in ['mlx', 'mlx_lm', 'numpy']:
        folder = site / package
        if not folder.is_dir(): raise ValueError('Reference producer absent')
        paths.update(path for path in folder.rglob('*') if path.is_file() and '__pycache__' not in path.parts
            and (package == 'mlx' or path.suffix in ['.py', '.so', '.dylib']))
    for folder in site.glob('numpy*.libs'):
        paths.update(path for path in folder.rglob('*') if path.is_file() and path.suffix in ['.so', '.dylib'])
    return sorted(paths)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify(folder, expected_manifest_sha=None):
    manifest = json.loads(captured(folder / 'manifest.json', expected_manifest_sha))
    if (manifest['format'] != 'toklet-mlx-llama-reference-v1' or manifest['source_manifest_sha256'] != SOURCE_SHA
        or encoded(manifest['policy']) != encoded(POLICY) or set(manifest['variants']) != {'tied', 'untied'}):
        raise ValueError('Llama fixture identity/policy differs')
    if (set(manifest) != {'format', 'source_manifest_sha256', 'producer_pins', 'consumed_source_pins', 'versions', 'policy', 'variants', 'arrays', 'files', 'scope'}
        or manifest['versions'] != VERSIONS or manifest['scope'] != SCOPE):
        raise ValueError('Exact reference metadata required')
    pins = manifest['producer_pins']
    for name, pin in {**pins, **manifest['consumed_source_pins']}.items():
        if (not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
            or not isinstance(pin, str) or re.fullmatch('[0-9a-f]{64}', pin) is None):
            raise ValueError('Relative producer/input SHA256 identities required')
    for suffix in ['/mlx/lib/mlx.metallib', '/mlx/lib/libmlx.dylib', '/mlx_lm/models/llama.py']:
        if not any(name.endswith(suffix) for name in pins):raise ValueError('Required reference runtime closure absent')
    if (not any('/mlx/core.' in name and name.endswith('.so') for name in pins)
        or not any('/numpy/' in name and name.endswith('.so') for name in pins)
        or any(name not in pins for name in ['tools/reference/mlx_llama.py', 'tools/reference/fp16.py'])):
        raise ValueError('Required selected producers absent')
    source = json.loads(captured(SOURCE / 'manifest.json', SOURCE_SHA))
    expected_inputs = {str((SOURCE / row['file']).relative_to(ROOT)): row['sha256'] for name, row in source['arrays'].items() if name.startswith('weights.') and not name.endswith('.bias')}
    if manifest['consumed_source_pins'] != expected_inputs: raise ValueError('Exact consumed source weights required')
    for variant, tied in [('tied', True), ('untied', False)]:
        data = manifest['variants'][variant]
        if set(data) != {'config', 'geometry', 'cases', 'greedy', 'untied_witness'}:raise ValueError('Exact variant metadata required')
        if any(set(row) != {'tokens', 'position_offset', 'greedy'} for row in data['cases'].values()):raise ValueError('Exact case metadata required')
        if (encoded(data['config']) != encoded(configuration(tied))
            or encoded(data['geometry']) != encoded(dict(GEOMETRY, tie_word_embeddings=tied))
            or encoded({key: {field: row[field] for field in ['tokens', 'position_offset']} for key, row in data['cases'].items()}) != encoded(CASES)):
            raise ValueError('Exact Llama fixture configuration/cases required')
        model = json.loads(captured(folder / variant / 'config.json'))
        if encoded(model) != encoded(configuration(tied)): raise ValueError('Published Llama config differs')
    if set(manifest['files']) != {'arrays.bin', 'tied/config.json', 'tied/model.safetensors', 'untied/config.json', 'untied/model.safetensors'}:
        raise ValueError('Exact fixture file inventory required')
    for name, item in manifest['files'].items():
        path = folder / name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Confined regular fixture file required')
        if len(captured(path, item['sha256'])) != item['bytes']:
            raise ValueError('Llama fixture byte count differs')
    raw = captured(folder / 'arrays.bin', manifest['files']['arrays.bin']['sha256'])
    contract = array_contract()
    if set(manifest['arrays']) != set(contract): raise ValueError('Exact Llama stage/native/greedy coverage required')
    end = 0
    for name, item in sorted(manifest['arrays'].items(), key=lambda pair: pair[1]['offset']):
        if (item['shape'], item['dtype'], item.get('gate_kind')) != contract[name]:
            raise ValueError('Exact Llama array shape/dtype/gate kind required')
        required = {'offset', 'bytes', 'sha256', 'dtype', 'shape'}
        if contract[name][2] is not None: required.update(['gate', 'gate_kind', 'precision_envelope'])
        if set(item) != required: raise ValueError('Exact required array gate/envelope fields missing')
        size = 2 if item['dtype'] == '<f2' else 4 if item['dtype'] == '<f4' else 0
        count = 1
        for dimension in item['shape']:
            if type(dimension) is not int or dimension <= 0: raise ValueError('Positive array dimensions required')
            count *= dimension
        begin, length = item['offset'], item['bytes']
        if type(begin) is not int or begin != end or length != count * size or not size or begin + length > len(raw):
            raise ValueError('Contiguous exact fixture array layout required')
        data = raw[begin:begin + length]
        if not all(math.isfinite(value) for (value,) in struct.iter_unpack('<e' if size == 2 else '<f', data)):
            raise ValueError('Nonfinite frozen fixture values')
        if sha(data) != item['sha256']:
            raise ValueError('Llama fixture array differs: ' + name)
        if 'gate' in item:
            if not number(item['precision_envelope']) or item['precision_envelope'] < 0: raise ValueError('Finite nonnegative envelope required')
            expected = dict(atol=max(POLICY['atol_floor'], POLICY['envelope_factor'] * item['precision_envelope']), rtol=POLICY['rtol']) if item['gate_kind'] == 'half_rounding_envelope' else dict(atol=POLICY['source_atol'], rtol=POLICY['source_rtol'])
            if (set(item['gate']) != {'atol', 'rtol'} or any(not number(value) for value in [*item['gate'].values(), *expected.values()])
                or item['gate'] != expected): raise ValueError('Frozen finite envelope policy differs')
        end += length
    if end != len(raw): raise ValueError('Uninventoried fixture array bytes')
    def values(name):
        item = manifest['arrays'][name]
        return [value for (value,) in struct.iter_unpack('<f', raw[item['offset']:item['offset'] + item['bytes']])]
    def argmax(row): return max(range(len(row)), key=lambda index: row[index])
    def tokens(value, count):
        return isinstance(value, list) and len(value) == count and all(type(token) is int and 0 <= token < 64 for token in value)
    for variant in ['tied', 'untied']:
        data = manifest['variants'][variant]
        for case, description in CASES.items():
            rows = values(f'{variant}.half.cases.{case}.logits')
            expected = [argmax(rows[start:start + 64]) for start in range(0, len(rows), 64)]
            if not tokens(data['cases'][case]['greedy'], len(description['tokens'])) or data['cases'][case]['greedy'] != expected:
                raise ValueError('Exact case greedy IDs required')
        generated = data['greedy']
        if (set(generated) != {'prompt', 'tokens'} or not tokens(generated['prompt'], 4) or generated['prompt'] != CASES['prompt']['tokens']
            or not tokens(generated['tokens'], 4) or generated['tokens'] != [argmax(values(f'{variant}.greedy.step_{step}.logits')) for step in range(4)]):
            raise ValueError('Exact frozen generation IDs required')
        main = values(f'{variant}.half.cases.prompt.logits')
        wrong = main if variant == 'tied' else values('untied.witness.tied_substitution.logits')
        witness = dict(expected_last_id=argmax(main[-64:]), tied_substitution_last_id=argmax(wrong[-64:]),
            maximum_difference=max(abs(a - b) for a, b in zip(main, wrong, strict=True)))
        actual = data['untied_witness']
        if (set(actual) != set(witness) or any(type(actual[key]) is not int for key in ['expected_last_id', 'tied_substitution_last_id'])
            or not number(actual['maximum_difference']) or actual != witness):
            raise ValueError('Exact untied substitution witness required')
        if variant == 'untied' and witness['expected_last_id'] == witness['tied_substitution_last_id']:
            raise ValueError('Untied head must change selected ID')
    return manifest


def export(folder):
    if os.path.lexists(folder): raise FileExistsError('Never overwrite a reference fixture')
    versions = {name: version(name) for name in ['mlx', 'mlx-lm', 'numpy']}
    if versions != VERSIONS:
        raise ValueError('Pinned reference package versions required')
    base = json.loads(captured(SOURCE / 'manifest.json', SOURCE_SHA))
    pins = {str(path.relative_to(ROOT)): sha(captured(path, limit=256 * 1024**2)) for path in producer_paths()}
    inputs = {}
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.models.cache import KVCache
    mx.set_default_device(mx.cpu)
    original = {}
    for name, item in base['arrays'].items():
        if not name.startswith('weights.') or name.endswith('.bias'): continue
        raw = captured(SOURCE / item['file'], item['sha256']);inputs[str((SOURCE / item['file']).relative_to(ROOT))] = item['sha256']
        original[name.removeprefix('weights.')] = np.frombuffer(raw, dtype='<f4').reshape(item['shape']).copy()
    geometry = dict(base['model'], rope_theta=10000.0, projection_biases=[])
    if geometry != GEOMETRY or base['cases'] != CASES: raise ValueError('Exact frozen base geometry/cases required')
    arrays = {};raw_arrays = bytearray();variants = {};payload_sets = {}
    def save(name, value, wide=None, mixed=False):
        value = np.asarray(value);dtype = '<f2' if value.dtype == np.float16 else '<f4'
        raw = value.astype(dtype).tobytes()
        if not np.isfinite(value).all() or wide is not None and not np.isfinite(wide).all(): raise ValueError('Nonfinite fixture array')
        item = dict(offset=len(raw_arrays), bytes=len(raw), sha256=sha(raw), dtype=dtype, shape=list(value.shape))
        if wide is not None:
            envelope = float(np.max(np.abs(value.astype(np.float64) - wide.astype(np.float64))))
            item.update(precision_envelope=envelope, gate_kind='half_rounding_envelope' if mixed else 'fp32_reference',
                gate=dict(atol=max(POLICY['atol_floor'], 2 * envelope), rtol=POLICY['rtol']) if mixed else dict(atol=POLICY['source_atol'], rtol=POLICY['source_rtol']))
        arrays[name] = item;raw_arrays.extend(raw)
    for variant, tied in [('tied', True), ('untied', False)]:
        config = configuration(tied);quantization = config['quantization']
        excluded = 'model.layers.0.self_attn.o_proj';overridden = 'model.layers.1.mlp.down_proj'
        source = dict(original)
        if not tied:
            # Deliberately different token rows/scales: tying this head changes
            # distributions and selected IDs, not merely allocation accounting.
            source['lm_head'] = (np.roll(original['embed_tokens'], 9, axis=0) * np.float32(-1.25)).copy()
        payloads = {};canonical = {};half_weights = {}
        for key, value in source.items():
            if key == 'lm_head': name = 'lm_head.weight'
            elif key.startswith('layers.'):
                _, index, leaf, *tail = key.split('.')
                block = 'self_attn' if leaf in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] else 'mlp' if leaf in ['gate_proj', 'up_proj', 'down_proj'] else None
                name = f'model.layers.{index}.' + (block + '.' if block else '') + leaf + ('.' + '.'.join(tail) if tail else '.weight')
            else: name = 'model.' + key + '.weight'
            module = name.removesuffix('.weight')
            if value.ndim == 2 and module != excluded:
                override = quantization.get(module, quantization);bits, group = override['bits'], override['group_size']
                packed, scales, biases = mx.quantize(mx.array(value), bits=bits, group_size=group);mx.eval(packed, scales, biases)
                payloads[name] = packed;payloads[module + '.scales'] = scales;payloads[module + '.biases'] = biases
                words = np.asarray(packed, dtype=np.uint32)
                codes = np.stack([(words >> shift) & ((1 << bits) - 1) for shift in range(0, 32, bits)], axis=-1).reshape(value.shape)
                reconstructed = (np.repeat(np.asarray(scales, dtype=np.float64), group, axis=1) * codes
                    + np.repeat(np.asarray(biases, dtype=np.float64), group, axis=1)).astype(np.float32)
                half_weights[key] = reconstructed if key != 'embed_tokens' else reconstructed.astype(np.float16)
            else:
                payloads[name] = mx.array(value);reconstructed = value;half_weights[key] = value.astype(np.float16)
            canonical[key] = reconstructed
        canonical['output'] = canonical['embed_tokens'] if tied else canonical.pop('lm_head')
        half_weights['output'] = canonical['output']
        model = Model(ModelArgs.from_dict(config))
        def predicate(path, module):
            if path == excluded or not hasattr(module, 'to_quantized'): return False
            return quantization.get(path, True)
        nn.quantize(model, bits=4, group_size=32, class_predicate=predicate)
        model.load_weights(list(payloads.items()), strict=True);mx.eval(model.parameters())
        cases = {}
        for case, description in base['cases'].items():
            tokens, offset = description['tokens'], description['position_offset']
            full = forward(tokens, offset, canonical, geometry, np.float32, np.float32)
            wide = forward(tokens, offset, canonical, geometry, np.float64, np.float32)
            half = forward(tokens, offset, half_weights, geometry, np.float32, np.float16)
            half_wide = forward(tokens, offset, half_weights, geometry, np.float64, np.float16)
            for stage in full:
                save(f'{variant}.fp32.cases.{case}.{stage}', full[stage], wide[stage])
                save(f'{variant}.half.cases.{case}.{stage}', half[stage], half_wide[stage], mixed=True)
            cases[case] = dict(tokens=tokens, position_offset=offset, greedy=np.argmax(half['logits'], axis=-1).tolist())
            if offset == 0:
                for label, device in [('CPU', mx.cpu), ('GPU', mx.gpu)]:
                    with mx.stream(device):
                        logits = model(mx.array([tokens], dtype=mx.uint32));mx.eval(logits);actual = np.asarray(logits)[0]
                    np.testing.assert_allclose(actual, full['logits'], atol=POLICY['source_atol'], rtol=POLICY['source_rtol'])
                    save(f'{variant}.native.{label}.{case}.logits', actual)
        prompt = cases['prompt']['tokens'];expected = forward(prompt, 0, canonical, geometry, np.float32, np.float32)['logits']
        for label, device in [('CPU', mx.cpu), ('GPU', mx.gpu)]:
            with mx.stream(device):
                cache = [KVCache() for _ in range(2)]
                first = model(mx.array([prompt[:2]], dtype=mx.uint32), cache=cache);mx.eval(first)
                second = model(mx.array([prompt[2:]], dtype=mx.uint32), cache=cache);mx.eval(second)
                actual = np.concatenate([np.asarray(first)[0], np.asarray(second)[0]])
            np.testing.assert_allclose(actual, expected, atol=POLICY['source_atol'], rtol=POLICY['source_rtol'])
            save(f'{variant}.native.{label}.cached.logits', actual)
        history = list(prompt);tokens = []
        for step in range(4):
            main = forward(history, 0, half_weights, geometry, np.float32, np.float16)
            wide = forward(history, 0, half_weights, geometry, np.float64, np.float16)
            save(f'{variant}.greedy.step_{step}.logits', main['logits'][-1], wide['logits'][-1], mixed=True)
            token = int(np.argmax(main['logits'][-1]));tokens.append(token);history.append(token)
        witness = forward(prompt, 0, half_weights, geometry, np.float32, np.float16)
        substituted = dict(half_weights, output=canonical['embed_tokens'])
        wrong = forward(prompt, 0, substituted, geometry, np.float32, np.float16)
        if not tied and (np.array_equal(witness['logits'], wrong['logits']) or np.argmax(witness['logits'][-1]) == np.argmax(wrong['logits'][-1])):
            raise ValueError('Untied-head fixture must distinguish silent tying')
        if not tied: save('untied.witness.tied_substitution.logits', wrong['logits'])
        variants[variant] = dict(config=config, geometry=dict(geometry, tie_word_embeddings=tied), cases=cases,
            greedy=dict(prompt=prompt, tokens=tokens), untied_witness=dict(expected_last_id=int(np.argmax(witness['logits'][-1])),
                tied_substitution_last_id=int(np.argmax(wrong['logits'][-1])), maximum_difference=float(np.max(np.abs(witness['logits'].astype(np.float64) - wrong['logits'].astype(np.float64))))))
        payload_sets[variant] = payloads
    for name, pin in {**pins, **inputs, str((SOURCE / 'manifest.json').relative_to(ROOT)): SOURCE_SHA}.items():
        captured(ROOT / name, pin, 256 * 1024**2)
    if {name: version(name) for name in versions} != versions: raise ValueError('Reference package changed')
    folder.mkdir();files = {}
    for variant, payloads in payload_sets.items():
        model_folder = folder / variant;model_folder.mkdir()
        mx.save_safetensors(str(model_folder / 'model.safetensors'), payloads)
        (model_folder / 'config.json').write_text(json.dumps(variants[variant]['config'], indent=2) + '\n')
    (folder / 'arrays.bin').write_bytes(raw_arrays)
    for path in sorted(folder.rglob('*')):
        if path.is_file(): files[str(path.relative_to(folder))] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    manifest = dict(format='toklet-mlx-llama-reference-v1', source_manifest_sha256=SOURCE_SHA, producer_pins=pins,
        consumed_source_pins=inputs, versions=versions, policy=POLICY, variants=variants, arrays=arrays, files=files,
        scope=SCOPE)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    verify(folder, sha(captured(folder / 'manifest.json')))
    for name, pin in {**pins, **inputs, str((SOURCE / 'manifest.json').relative_to(ROOT)): SOURCE_SHA}.items():
        captured(ROOT / name, pin, 256 * 1024**2)
    if {str(path.relative_to(ROOT)) for path in producer_paths()} != set(pins):raise ValueError('Producer file closure changed')
    print(json.dumps(dict(arrays=len(arrays), bytes=len(raw_arrays), variants={key: value['greedy'] for key, value in variants.items()})))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser();group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--export', type=Path);group.add_argument('--check', type=Path)
    parser.add_argument('--manifest-sha256');args = parser.parse_args()
    if args.check and not args.manifest_sha256:parser.error('--check requires an externally pinned --manifest-sha256')
    if args.export: export(args.export)
    else: print(json.dumps(dict(verified=True, arrays=len(verify(args.check, args.manifest_sha256)['arrays']))))
