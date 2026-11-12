"""Explicit admitted CPU reference controls/captures, never target execution."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import granite_convert_layout as layout
import granite_reference_graph as graph
from granite_reference_weights import Child
import minicpm5_context_runtime as runtime

need = graph.need
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests/fixtures/granite-half-graph-v1/fixture.json'
FIXTURE_SHA = '852711a8bfffd68f389f5db8a987ff6e4f56dd5e5d747bcbe89cccb089b58dd9'
FIXTURE_LIMIT = 4 * 1024**2
FLOOR = 2 * 1024**3


def decoded(path, maximum, expected):
    return json.loads(runtime.common.captured(Path(path), maximum, runtime.common.sha(expected)),
                      object_pairs_hook=runtime.common.json_unique,
                      parse_constant=runtime.common.bad_json_number)


def sources():
    names = ['granite_reference_probe.py', 'granite_reference_graph.py',
             'test_granite_reference_graph.py', 'granite_reference_weights.py',
             'test_granite_reference_weights.py', 'granite_convert_layout.py',
             'granite_quantize.py', 'minicpm5_quantize.py', 'minicpm5_parent.py',
             'minicpm5_context_runtime.py', 'reference_runtime.py', 'text_reference_runtime.py']
    fixture = decoded(FIXTURE, FIXTURE_LIMIT, FIXTURE_SHA)
    selected = {ROOT / 'tools/reference' / name for name in names} | {FIXTURE}
    for category in ('inputs', 'producers'):
        for name, pin in fixture[category].items():
            path = ROOT / name
            need(path.resolve().is_relative_to(ROOT) and not path.is_symlink(),
                 'Exact local fixture source required')
            raw = runtime.common.captured(path, 2 * 1024**2, pin['sha256'])
            need(len(raw) == pin['bytes'], 'Fixture source extent differs')
            selected.add(path)
    return sorted(selected)


def publish(path, value):
    with path.open('x') as file:
        json.dump(value, file, indent=2)
        file.write('\n')


def write_array(path, value, dtype, np):
    """Write/hash/read back <=1 MiB views, avoiding a full-array bytes copy."""
    contiguous = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    raw = memoryview(contiguous).cast('B')
    digest = hashlib.sha256()
    with path.open('xb') as file:
        cursor = 0
        while cursor < len(raw):
            part = raw[cursor:cursor + 1024**2]
            written = file.write(part)
            need(type(written) is int and 0 < written <= len(part), 'Array write failed to advance')
            digest.update(part[:written])
            cursor += written
        file.flush()
    with path.open('rb') as file:
        stored = hashlib.file_digest(file, 'sha256').hexdigest()
    need(path.stat().st_size == len(raw) and stored == digest.hexdigest(),
         'Complete saved reference array differs')
    return dict(bytes=len(raw), sha256=stored, shape=list(value.shape), dtype=dtype)


def controls(output, np):
    fixture = decoded(FIXTURE, FIXTURE_LIMIT, FIXTURE_SHA)
    import llama3_half_graph as tiny
    source = json.loads((tiny.source.SOURCE / 'manifest.json').read_bytes())

    class ArrayWeights:
        def __init__(self, values):
            self.values = values

        def shape(self, name):
            return np.asarray(self.values[name]).shape

        def rows(self, name, first, count, _):
            return np.asarray(self.values[name], dtype=np.float32)[first:first + count]

        def norm(self, name, _):
            return np.asarray(self.values[name], dtype=np.float16)

        def gather(self, tokens, _):
            return np.asarray(self.values['model.embed_tokens.weight'], dtype=np.float32)[tokens].astype(np.float16)

    renames = dict(q='q_projection', k='k_projection', v='v_projection',
                   gate='gate_projection', up='up_projection', silu='gate_silu', down='down_projection')
    records, comparisons, mismatches = [], 0, 0
    for case in fixture['cases']:
        original = fixture['config_templates'][case['variant']]
        config = dict(hidden=original['hidden_size'], intermediate=original['intermediate_size'],
                      layers=original['num_hidden_layers'], vocabulary=original['vocab_size'],
                      query_heads=original['num_attention_heads'], kv_heads=original['num_key_value_heads'],
                      head_dim=original['head_dim'], theta=original['rope_theta'], rms_eps=original['rms_norm_eps'],
                      attention_scale=original['attention_multiplier'], embedding_scale=original['embedding_multiplier'],
                      residual_scale=original['residual_multiplier'], logits_divisor=original['logits_scaling'],
                      tied_head=original['tie_word_embeddings'])
        values, _, _ = tiny.read_weights(case['variant'], source)
        for wide in (False, True):
            profile = 'wide' if wide else 'main'

            def emit(name, value):
                nonlocal comparisons, mismatches
                parts = name.split('.')
                parts[-1] = renames.get(parts[-1], parts[-1])
                spec = case['arrays']['.'.join(parts)]
                dtype, word = ('<f2', '<u2') if spec['dtype'] == 'f16le' else ('<f4', '<u4')
                outcomes = {}
                for reference in ('main', 'wide'):
                    expected = np.asarray(spec[reference + '_bits'], dtype=word).view(dtype).reshape(value.shape)
                    error = np.abs(value.astype(np.float64) - expected.astype(np.float64))
                    limits = spec['gate']['atol'] + spec['gate']['rtol'] * np.abs(expected.astype(np.float64))
                    bad = error > limits
                    count = int(np.count_nonzero(bad))
                    comparisons += 1
                    mismatches += count
                    outcomes[reference] = dict(mismatching_values=count, maximum_absolute_error=float(error.max()),
                                               gate=spec['gate'])
                records.append(dict(variant=case['variant'], case=case['name'], profile=profile, stage=name,
                                    values=int(value.size), dtype=dtype,
                                    actual_sha256=hashlib.sha256(value.astype(dtype, copy=False).tobytes()).hexdigest(),
                                    references=outcomes))
                if any(row['mismatching_values'] for row in outcomes.values()):
                    failure = output / f"{case['variant']}-{case['name']}-{profile}-{name}.bin"
                    write_array(failure, value, dtype, np)

            graph.forward(ArrayWeights(values), config, case['tokens'], case['position_offset'], wide, np, emit)
    need(len(records) == 560 and comparisons == 1120, 'Complete frozen tiny graph control coverage required')
    with (output / 'records.jsonl').open('x') as file:
        for row in records:
            file.write(json.dumps(row, separators=(',', ':')) + '\n')
    return dict(passed=mismatches == 0, fixture_sha256=FIXTURE_SHA, profiles=2, source_cases=8,
                arrays=len(records), reference_comparisons=comparisons, mismatching_values=mismatches,
                frozen_gates_unchanged=True, real_model_consumed=False)


def capture(args, output, np):
    protocol = decoded(args.protocol, 2 * 1024**2, args.protocol_sha256)
    contract = decoded(args.contract, 2 * 1024**2, args.contract_sha256)
    manifest = decoded(args.conversion_manifest, 2 * 1024**2,
                       protocol['model']['conversion_manifest_sha256'])
    raw = runtime.common.captured(Path(args.conversion_journal), 16 * 1024**2,
                                  protocol['model']['conversion_journal_sha256'])
    tiles = [row for line in raw.splitlines()
             if (row := json.loads(line, object_pairs_hook=runtime.common.json_unique))['event'] == 'matrix_tile']
    need(protocol['format'] == 'toklet-granite-full-graph-prospective-v1'
         and protocol['numerical_policy']['atol_floor'] == 1e-4
         and protocol['numerical_policy']['envelope_factor'] == 2
         and protocol['numerical_policy']['rtol'] == 1e-3,
         'Frozen independent numerical protocol required')
    need(manifest['passed'] is True and manifest['matrix_count'] == 282 and manifest['norm_count'] == 81
         and manifest['contract_sha256'] == args.contract_sha256
         and manifest['child']['sha256'] == protocol['model']['data_sha256']
         and manifest['child']['config_sha256'] == protocol['model']['config_sha256'],
         'Exact authenticated mixed child conversion required')
    model = Path(args.model)
    config = decoded(model / 'config.json', 1024**2, protocol['model']['config_sha256'])
    need(config['model_type'] == 'granite' and config['architectures'] == ['GraniteForCausalLM']
         and config['rope_parameters'] == {'rope_type': 'default', 'rope_theta': config['rope_theta']},
         'Exact default-RoPE Granite source configuration required')
    declared = dict(hidden=config['hidden_size'], intermediate=config['intermediate_size'],
                    layers=config['num_hidden_layers'], vocabulary=config['vocab_size'],
                    query_heads=config['num_attention_heads'], kv_heads=config['num_key_value_heads'],
                    head_dim=config['hidden_size'] // config['num_attention_heads'],
                    theta=config['rope_theta'], rms_eps=config['rms_norm_eps'],
                    attention_scale=config['attention_multiplier'], embedding_scale=config['embedding_multiplier'],
                    residual_scale=config['residual_multiplier'], logits_divisor=config['logits_scaling'],
                    tied_head=config['tie_word_embeddings'])
    need(protocol['geometry'] == declared, 'Reference geometry/scalars differ from pinned model config')
    planned = layout.output_layout(contract['recipe']['matrices'], contract['recipe']['norms'])
    need(len(planned['matrices']) == 282 and len(planned['logical']) == 363
         and hashlib.sha256(planned['prefix']).hexdigest() == manifest['child']['header_sha256'],
         'Complete frozen child layout required')
    expected_file = {k: manifest['child'][k] for k in ('bytes', 'sha256')}
    need(len(protocol['cases']) == 10 and all(re.fullmatch('[a-z][a-z0-9_]*', c['name'])
                                            for c in protocol['cases'])
         and len({c['name'] for c in protocol['cases']}) == 10, 'Ten unique fixed graph cases required')
    for case in protocol['cases']:
        graph.geometry(protocol['geometry'], case['tokens'], case['offset'])
    per_token = sum(width * (4 if name == 'logits' else 2)
                    for name, (_, width) in graph.contracts(protocol['geometry'], 1).items())
    one_raw = per_token * sum(len(c['tokens']) for c in protocol['cases'])
    need(one_raw == protocol['memory_and_disk']['raw_bytes_per_reference_profile_or_lossless_native_capture'],
         'Complete raw array budget differs')
    # Reserve the two independent profiles and the subsequent native raw capture.
    need(shutil.disk_usage(output).free >= 3 * one_raw + FLOOR, 'Full raw capture reservation exceeds disk floor')
    arrays, cases, published = {}, [], 0
    with Child(model / 'model.safetensors', planned, expected_file, tiles, manifest['exact_norms']) as child:
        owner = graph.PackedWeights(child)
        for case in protocol['cases']:
            for wide in (False, True):
                profile = 'wide' if wide else 'main'
                folder = output / case['name'] / profile
                folder.mkdir(parents=True)
                started = time.monotonic_ns()
                seen = 0

                def emit(name, value):
                    nonlocal published, seen
                    need(shutil.disk_usage(output).free >= 3 * one_raw - published + FLOOR,
                         'Remaining complete capture reservation exceeds disk floor')
                    key = case['name'] + '.' + profile + '.' + name
                    need(key not in arrays, 'Unique captured graph stage required')
                    path = folder / (name + '.bin')
                    entry = write_array(path, value, '<f4' if name == 'logits' else '<f2', np)
                    entry['file'] = str(path.relative_to(output))
                    arrays[key] = entry
                    published += entry['bytes']
                    seen += 1
                    if name.endswith('.output'):
                        print(json.dumps(dict(event='reference_layer', case=case['name'], profile=profile,
                                              stage=name, captured_arrays=seen)), flush=True)

                last = graph.forward(owner, protocol['geometry'], case['tokens'], case['offset'], wide, np, emit)
                need(seen == 643, 'Every full-model native-visible stage required')
                top = np.lexsort((np.arange(len(last)), -last))[:2]
                cases.append(dict(name=case['name'], profile=profile, arrays=seen,
                                  last_logits_sha256=hashlib.sha256(last.astype('<f4', copy=False).tobytes()).hexdigest(),
                                  top_two_ids=top.tolist(), margin=float(last[top[0]]) - float(last[top[1]]),
                                  diagnostic_graph_and_publication_ns=time.monotonic_ns() - started))
                del last
        consumed = child.endpoint()
    need(len(arrays) == 12860 and published == 2 * one_raw, 'Complete independent profile publication required')
    decoded(model / 'config.json', 1024**2, protocol['model']['config_sha256'])
    return dict(passed=True, protocol_sha256=args.protocol_sha256, contract_sha256=args.contract_sha256,
                conversion_manifest_sha256=protocol['model']['conversion_manifest_sha256'],
                numerical_policy=protocol['numerical_policy'], geometry=protocol['geometry'],
                cases=cases, arrays=arrays, published_bytes=published, source_consumption=consumed,
                native_compared=False, graph_qualified=False, quality_qualified=False,
                performance_qualified=False, managed_adoption_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('controls', 'capture'), required=True)
    for name in ('environment-root', 'runtime-manifest', 'runtime-sha256', 'fresh-output'):
        parser.add_argument('--' + name, required=True)
    for name in ('protocol', 'protocol-sha256', 'contract', 'contract-sha256',
                 'conversion-manifest', 'conversion-journal', 'model'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.mode == 'capture':
        need(all(getattr(args, name.replace('-', '_')) for name in
                 ('protocol', 'protocol-sha256', 'contract', 'contract-sha256',
                  'conversion-manifest', 'conversion-journal', 'model')),
             'Complete externally pinned full-model capture arguments required')
    output = Path(args.fresh_output)
    need(not output.exists() and not output.is_symlink(), 'Fresh reference output required')
    if args.model:
        need(not output.resolve().is_relative_to(Path(args.model).resolve())
             and not Path(args.model).resolve().is_relative_to(output.resolve()),
             'Reference output must not overlap model assets')
    need(os.environ.get('MallocLargeCache') == os.environ.get('MallocMediumZone') == '0',
         'Bounded CPU reference requires launch-time MallocLargeCache=0 and MallocMediumZone=0')
    output.mkdir()
    runtime.bind_environment(args.environment_root)
    admitted, endpoint = runtime.admit(Path(args.runtime_manifest), args.runtime_sha256, sources())
    import numpy as np
    result = controls(output, np) if args.mode == 'controls' else capture(args, output, np)
    result.update(format='toklet-granite-independent-reference-v1', mode=args.mode,
                  admitted_runtime=admitted, endpoint_origins=endpoint(),
                  allocator_policy=dict(launch_environment={'MallocLargeCache': '0', 'MallocMediumZone': '0'},
                                        scope='CPU reference subprocess only; engine default unchanged',
                                        admission_counters_reset_or_discounted=False),
                  whole_goal_complete=False)
    publish(output / 'manifest.json', result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('arrays', 'admitted_runtime', 'endpoint_origins', 'cases', 'geometry')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
