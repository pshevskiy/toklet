"""Independent header/layout replay via previously frozen packed-model geometry.

Never imports the producer. Does not read weight payloads or infer media support.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import check_gemma_hybrid_tiny as safe

ROOT = Path(__file__).resolve().parents[2]


def verify(header_path, config_path, layout_path, layout_sha):
    header_raw = safe.read(header_path, 8 * 1024**2)
    safe.same(hashlib.sha256(header_raw).hexdigest(), '12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e')
    config_raw = safe.read(config_path, 8192)
    safe.same(hashlib.sha256(config_raw).hexdigest(), '1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330')
    raw = safe.read(layout_path, 2 * 1024**2)
    safe.same(hashlib.sha256(raw).hexdigest(), layout_sha)
    actual = safe.document(raw)
    container_bytes = 10246621918
    header = safe.document(header_raw)
    safe.same(header['__metadata__'], {'format': 'pt'})
    safe.same(len(header), 2012)
    intervals = []
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        safe.same(set(tensor), {'dtype', 'shape', 'data_offsets'})
        safe.same(tensor['dtype'], 'BF16')
        shape = tensor['shape']
        safe.require(type(shape) is list and len(shape) <= 4
                     and all(type(n) is int and 1 <= n <= 262144 for n in shape), 'source tensor shape')
        pair = tensor['data_offsets']
        safe.require(type(pair) is list and len(pair) == 2
                     and all(type(n) is int and n >= 0 and n % 2 == 0 for n in pair), 'source tensor offsets')
        start, end = pair
        safe.same(end - start, math.prod(shape) * 2)
        intervals.append((start, end))
    intervals.sort()
    safe.same(intervals[0][0], 0)
    cursor = 0
    for start, end in intervals:
        safe.same(start, cursor)
        cursor = end
    safe.same(cursor, container_bytes - 8 - len(header_raw))

    # This independent route maps the prior278 matrix/262 vector shapes rather
    # than repeating the producer's per-layer shape construction.
    inventory_raw = safe.read(ROOT / 'tests/fixtures/gemma-resident-storage-reference-v1/inventory.json', 2 * 1024**2)
    safe.same(hashlib.sha256(inventory_raw).hexdigest(), 'a3dff3b9d825fcf4fa6f391262732c5f429345682d683d6a39bc551ddacd68e2')
    inventory = safe.document(inventory_raw)
    selected = {}
    for item in inventory['matrices'] + inventory['vectors']:
        safe.require(item['name'].startswith('language_model.model.'), 'prior text name')
        name = 'model.language_model.' + item['name'][len('language_model.model.'):]
        shape = [item['rows'], item['columns']] if item['kind'] == 'matrix' else [item['count']]
        safe.same(header[name]['shape'], shape)
        start, end = header[name]['data_offsets']
        selected[name] = dict(shape=shape, dtype='BF16', file_offset=8 + len(header_raw) + start, bytes=end - start)
    safe.same(len(selected), 540)
    unused = {}
    expected_unused = {f'model.language_model.layers.{i}.self_attn.{suffix}'
                       for i in range(15, 35) for suffix in ('k_norm.weight', 'k_proj.weight', 'v_proj.weight')}
    for name in expected_unused:
        layer = int(re.fullmatch(r'model\.language_model\.layers\.(\d+)\.self_attn\..+', name)[1])
        dim = 512 if layer % 5 == 4 else 256
        shape = [dim] if name.endswith('k_norm.weight') else [dim, 1536]
        safe.same(header[name]['shape'], shape)
        start, end = header[name]['data_offsets']
        unused[name] = dict(shape=shape, dtype='BF16', file_offset=8 + len(header_raw) + start, bytes=end - start)
    text = {name for name in header if name.startswith('model.language_model.')}
    safe.same(text, set(selected) | set(unused))
    media = set(header) - {'__metadata__'} - text
    safe.same(len(media), 1411)
    safe.require(all(name.startswith(('model.audio_tower.', 'model.vision_tower.',
                                     'model.embed_audio.', 'model.embed_vision.')) for name in media), 'media exclusions')
    selected_bytes = sum(item['bytes'] for item in selected.values())
    unused_bytes = sum(item['bytes'] for item in unused.values())
    safe.same(selected_bytes, 9257138758)
    safe.same(unused_bytes, 37761024)
    expected = dict(format='toklet-gemma-unquantized-layout-v1',
                    repository='google/gemma-4-E2B-it', revision='70af34e20bd4b7a91f0de6b22675850c43922a03',
                    advertised_container=dict(bytes=container_bytes, sha256='2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550'),
                    acquired_header=dict(bytes=len(header_raw), sha256='12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e'),
                    config_sha256='1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330',
                    text_config_sha256='62f8428c49b31ebe5bb8186c7d352ce10498632148703db72568db3cd1a83257',
                    counts=dict(all_tensors=2011, selected_matrices=278, selected_vectors=262, unused_shared_kv=60, excluded_media=1411),
                    bytes=dict(selected_text=selected_bytes, unused_shared_kv=unused_bytes, excluded_media=cursor - selected_bytes - unused_bytes),
                    source_rule=dict(transformers_version='5.7.0', modeling_source_sha256='fd93159380556e46029260ae38e5ec53ffd750ecb9872d0e89f57b334a25ed4e',
                                     reviewed_inclusive_line_ranges=[[1158, 1195], [1215, 1237], [1592, 1601]],
                                     unused='shared consumers use producer13(sliding)/14(full) states; source ignores these unexpected K/V tensors on load'),
                    selected_text=selected, unused_shared_kv=unused,
                    full_payload_downloaded=False, full_payload_verified=False, weights_loaded=False,
                    arithmetic_profile_qualified=False, model_fit_qualified=False, inference_executed=False,
                    quality_scored=False, app_eligibility_changed=False, full_model_window_qualified=False)
    safe.same(actual, expected)
    return dict(passed=True, all_header_extents_compared=2011, selected_tensors_compared=540,
                unused_shared_tensors_compared=60, byte_totals=expected['bytes'],
                full_payload_verified=False, inference_executed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--header', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--layout-sha', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.header, args.config, args.layout, args.layout_sha)))
