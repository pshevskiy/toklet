"""Offline validation of independently frozen MiniCPM5 CPU reference bits."""
from pathlib import Path
import json
import hashlib
import math
import re

import minicpm5 as model
import minicpm5_reference as producer

REPO, REVISION = model.REPOSITORY, model.REVISION
CASES, POLICY = model.CASES, model.POLICY
expected_arrays, unique = producer.expected_arrays, model.unique
need = model.need


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def same(actual, expected):
    """JSON identity includes scalar types; bool never substitutes for0/1."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(same(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def producer_pins_match(recorded, current, source):
    if same(recorded, current):
        return True
    # One reviewed, non-executable EOF cleanup only. All five producer pins
    # remain exact; the old model bytes are precisely the clean bytes plus LF.
    name = 'tools/reference/minicpm5.py'
    old = '74557564fe62dd9836eb511f7db053bd598cd3d1e93d100450236a819e4167b9'
    clean = '38befa2c53a6f23bb2a8767235b1f78f59042f4db15fc1282a6609fb794c969e'
    expected = dict(current)
    expected[name] = old
    return (type(source) is bytes and len(source) <= 64 * 1024
            and current.get(name) == clean and hashlib.sha256(source).hexdigest() == clean
            and hashlib.sha256(source + b'\n').hexdigest() == old and same(recorded, expected))


def read(folder, item):
    import numpy as np
    with (Path(folder) / item['file']).open('rb') as stream:
        stream.seek(item['offset'])
        raw = stream.read(item['bytes'])
    need(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'],
         'Reference array bytes differ')
    return np.frombuffer(raw, dtype=item['dtype']).reshape(item['shape'])


def verify(folder, external_sha):
    import numpy as np
    folder = Path(folder)
    need(digest(external_sha), 'Explicit external reference identity required')
    with (folder / 'manifest.json').open('rb') as stream:
        raw = stream.read(2 * 1024 ** 2 + 1)
    need(len(raw) <= 2 * 1024 ** 2, 'Oversize reference manifest')
    need(hashlib.sha256(raw).hexdigest() == external_sha, 'Explicit external reference identity differs')
    value = json.loads(raw, object_pairs_hook=unique)
    need(type(value) is dict and set(value) == {'format', 'model', 'geometry', 'cases', 'layers',
         'policy', 'contract_sha256', 'producer_pins', 'package_pins', 'python', 'thread_environment',
         'memory_admission', 'observed_memory', 'arrays', 'gates', 'greedy', 'files', 'elapsed_ns',
         'native_outcomes_seen', 'parent_lineage_authenticated', 'model_quality_qualified',
         'model_adoption_qualified', 'gate_scope'}, 'Reference fields differ')
    need(value['format'] == 'toklet-minicpm5-independent-short-v1'
         and same(value['model'], dict(repository=REPO, revision=REVISION, files=model.PINS))
         and same(value['geometry'], model.GEOMETRY) and same(value['cases'], CASES)
         and same(value['layers'], model.LAYERS) and same(value['policy'], POLICY),
         'Fixed reference identity/geometry/policy differs')
    for field in ['native_outcomes_seen', 'parent_lineage_authenticated',
                  'model_quality_qualified', 'model_adoption_qualified']:
        need(value[field] is False, 'Reference relabels qualification')
    with (producer.ROOT / 'tools/reference/minicpm5.py').open('rb') as stream:
        model_source = stream.read(64 * 1024 + 1)
    need(producer_pins_match(value['producer_pins'], producer.producer_pins(), model_source)
         and value['package_pins'] == producer.package_pins(), 'Producer/package pins differ')
    need(digest(value['contract_sha256']) and same(value['thread_environment'],
         {key: '1' for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                              'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']})
         and value['gate_scope'] == 'Six authored cases only; teacher/scoring logits use the same frozen cases-only pooledlogit gate, never enlarge it',
         'Reference prospective contract/thread/gate scope differs')
    need(same(value['memory_admission'], producer.admission()), 'Reference admission ledger differs')
    observed = value['observed_memory']
    need(type(observed) is dict and set(observed) == {'peak_rss_bytes', 'resident_size',
         'phys_footprint', 'lifetime_max_phys_footprint'} and all(type(v) is int and 0 < v <= 6 * 1024 ** 3
         for v in observed.values()), 'Reference observed memory is invalid')
    need(type(value['elapsed_ns']) is int and 0 < value['elapsed_ns'] <= 1800 * 10 ** 9,
         'Reference elapsed clock invalid')
    inventory = expected_arrays()
    need(set(value['arrays']) == {p + '.' + name for p in ['fp32', 'fp64'] for name in inventory},
         'Reference array coverage differs')
    need(set(value['files']) == {'fp32.bin', 'fp64.bin'}, 'Reference file inventory differs')
    loaded, envelope = {}, {}
    for profile in ['fp32', 'fp64']:
        spans = []
        for name, (shape, dtype) in inventory.items():
            item = value['arrays'][profile + '.' + name]
            need(type(item) is dict and set(item) == {'file', 'offset', 'bytes', 'shape', 'dtype', 'sha256', 'gate_group'},
                 'Reference descriptor fields differ')
            group = name.split('.', 2)[2] if name.startswith('cases.') else 'logits'
            need(item['shape'] == shape and all(type(n) is int for n in item['shape'])
                 and item['dtype'] == dtype and item['file'] == profile + '.bin'
                 and type(item['offset']) is int and item['offset'] >= 0
                 and type(item['bytes']) is int and item['bytes'] == math.prod(shape) * np.dtype(dtype).itemsize
                 and digest(item['sha256']) and item['gate_group'] == group, 'Reference descriptor layout differs')
            spans.append((item['offset'], item['offset'] + item['bytes']))
            array = read(folder, item)
            need(np.isfinite(array).all(), 'Reference contains nonfinite values')
            loaded[profile + '.' + name] = array
        cursor = 0
        for start, end in sorted(spans):
            need(start == cursor, 'Reference spans overlap or contain gaps')
            cursor = end
        file = folder / (profile + '.bin')
        need(value['files'][file.name] == dict(bytes=cursor, sha256=model.sha(file))
             and cursor == file.stat().st_size and cursor <= 128 * 1024 ** 2, 'Reference raw file differs')
    for name in inventory:
        if name.startswith('cases.'):
            group = name.split('.', 2)[2]
            delta = float(np.max(np.abs(loaded['fp32.' + name].astype(np.float64)
                                       - loaded['fp64.' + name].astype(np.float64))))
            envelope[group] = max(envelope.get(group, 0.), delta)
    need(set(value['gates']) == set(envelope), 'Gate coverage differs')
    for group, delta in envelope.items():
        gate = value['gates'][group]
        need(type(gate) is dict and set(gate) == {'atol', 'rtol', 'pooled_cpu_profile_max_abs'}
             and all(number(v) for v in gate.values())
             and gate == dict(atol=max(1e-4, 2 * delta), rtol=1e-3,
                              pooled_cpu_profile_max_abs=delta), 'Frozen independent gate differs')
    prefix = CASES['short']['tokens'].copy()
    need(type(value['greedy']) is list and len(value['greedy']) == 4, 'Teacher history coverage differs')
    for step, row in enumerate(value['greedy']):
        need(type(row) is dict and set(row) == {'prefix', 'fp32_id', 'fp64_id'}
             and row['prefix'] == prefix and all(type(t) is int for t in row['prefix'])
             and type(row['fp32_id']) is int and type(row['fp64_id']) is int
             and row['fp32_id'] == int(np.argmax(loaded[f'fp32.greedy.step_{step}.logits']))
             and row['fp64_id'] == int(np.argmax(loaded[f'fp64.greedy.step_{step}.logits'])),
             'Teacher selection/prefix differs from reference bits')
        prefix.append(row['fp64_id'])
    need(model.sha(folder / 'manifest.json') == external_sha, 'Reference identity changed during validation')
    return value
