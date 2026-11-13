"""Frozen full-coordinate Qwen3.5-2B graph diagnostic contract.

No native outputs, numerical runtime, tokenizer, or target engine is imported.
The short phase is an initial full-model diagnostic, not context/quality proof.
"""
import hashlib
from functools import lru_cache
import math
from pathlib import Path
import re
import struct

FORMAT = 'toklet-qwen35-2b-graph-capture-v1'
GEOMETRY = dict(hidden=2048, intermediate=6144, vocab=248320, layers=24,
    query_heads=8, kv_heads=2, head_dim=256, rotated_dims=64,
    key_heads=16, value_heads=16, key_dim=128, value_dim=128,
    max_positions=262144, rope_theta=10000000.0, rms_epsilon=1e-6)
GATE = dict(atol=1e-5, rtol=1e-4)
QUALIFICATIONS = dict(full_model_graph=False, representative_context=False,
    quality=False, performance=False, adoption=False)
CASES = (
    ('single0', (1,), 0),
    ('prefix4', (1, 7, 13, 3), 0),
    ('future_changed4', (1, 7, 13, 5), 0),
    ('offset3', (2, 11, 9), 37),
)
PROFILES = dict(native=('native_fp32',), reference=('ideal_fp64', 'staged_fp32'))
STATE_STAGES = ('delta_state', 'delta_history', 'attention_k_cache', 'attention_v_cache')
PIN_FIELDS = ('config_bytes', 'config_sha256', 'header_bytes', 'header_sha256',
              'container_bytes', 'container_sha256')
MAX_JSON = 4 * 1024**2
MAX_JOURNAL = 1024**3
MAX_FRAME_JSON = 2048
MAX_FRAME_BYTES = 2 * 1024**2
PHASES = ('initial_short', 'context17', 'fixed255', 'context512')
TOKENS17 = (1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10, 19, 8, 21, 23, 29)
DISK_FLOOR = 2 * 1024**3
# External private-lifetime monitoring remains mandatory. This reservation is a
# preflight hypothesis; only actual complete-process accounting can qualify it.
REFERENCE_LEDGER = dict(python_state_and_temporaries=224 * 1024**2,
    interpreter_and_numerical_runtime=128 * 1024**2,
    row_tiles_and_source_buffers=64 * 1024**2,
    streamed_capture_buffers=16 * 1024**2, observer_helper=32 * 1024**2)


def need(value, message):
    if not value:
        raise ValueError(message)


def exact(left, right):
    return type(left) is type(right) and (
        set(left) == set(right) and all(exact(left[k], right[k]) for k in left)
        if type(left) is dict else
        len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right, strict=True))
        if type(left) in (list, tuple) else left == right)


def sha(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
         'Lowercase full SHA256 required')
    return value


def protocol_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def cases(phase):
    need(type(phase) is str and phase in PHASES, 'Frozen admitted graph phase required')
    selected = CASES
    if phase == 'context17':
        selected = (('context17', TOKENS17, 0), ('offset17', TOKENS17, 37),
                    ('future_changed17', TOKENS17[:-1] + (31,), 0))
    elif phase in ('fixed255', 'context512'):
        length = 255 if phase == 'fixed255' else 512
        selected = ((phase, tuple(TOKENS17[k % len(TOKENS17)] for k in range(length)), 0),)
    return [dict(name=n, tokens=list(t), position_offset=o) for n, t, o in selected]


@lru_cache(maxsize=1)
def shapes(committed):
    """All flattened observation coordinates, with semantic row-major shapes."""
    need(type(committed) is int and 1 <= committed <= 512, 'Admitted graph committed count required')
    g = GEOMETRY
    h, i, qh, kh, dim = (g[k] for k in ('hidden', 'intermediate', 'query_heads', 'kv_heads', 'head_dim'))
    hv, hk, dv, dk = (g[k] for k in ('value_heads', 'key_heads', 'value_dim', 'key_dim'))
    channels = 2 * hk * dk + hv * dv
    result = {(None, 'embedding', None): [h], (None, 'final_norm', None): [h],
              (None, 'logits', None): [g['vocab']]}
    common = dict(input_norm=[h], operator_output=[h], operator_residual=[h],
        post_norm=[h], mlp_gate=[i], mlp_up=[i], mlp_activated=[i], mlp_output=[h], layer_output=[h])
    delta = dict(delta_qkv=[channels], delta_z=[hv, dv], delta_a=[hv], delta_b=[hv],
        delta_convolved=[channels], delta_q=[hk, dk], delta_k=[hk, dk], delta_v=[hv, dv],
        delta_decay=[hv], delta_beta=[hv], delta_state=[hv, dv, dk],
        delta_history=[3, channels], delta_output=[hv, dv], delta_norm=[hv, dv], delta_gated=[hv, dv])
    full = dict(attention_q_gate=[qh, 2, dim], attention_q=[qh, dim], attention_k=[kh, dim],
        attention_v=[kh, dim], attention_q_norm=[qh, dim], attention_k_norm=[kh, dim],
        attention_q_rot=[qh, dim], attention_k_rot=[kh, dim],
        attention_context=[qh, dim], attention_gated=[qh, dim])
    for layer in range(g['layers']):
        for stage, shape in {**common, **(full if layer % 4 == 3 else delta)}.items():
            result[layer, stage, None] = shape
        if layer % 4 == 3:
            for head in range(kh):
                for stage in ('attention_k_cache', 'attention_v_cache'):
                    result[layer, stage, head] = [committed, dim]
            for head in range(qh):
                for stage in ('attention_scores', 'attention_probabilities'):
                    result[layer, stage, head] = [committed]
    need(len(result) == 669, 'Complete 2B graph stage inventory differs')
    return result


def inventory(phase='initial_short'):
    return {(case['name'], token, layer, stage, head): shape
        for case in cases(phase) for token in range(len(case['tokens']))
        for (layer, stage, head), shape in shapes(token + 1).items()}


def journal_limit(phase, role):
    need(role in PROFILES, 'Known capture role required')
    return min(MAX_JOURNAL, max(32 * 1024**2,
        sum(len(c['tokens']) for c in cases(phase)) * 669 * len(PROFILES[role]) * 1024))


def plan(phase='initial_short'):
    selected = cases(phase)
    values = sum(sum(sum(math.prod(shape) for shape in shapes(t + 1).values())
                     for t in range(len(case['tokens']))) for case in selected)
    arrays = 669 * sum(len(case['tokens']) for case in selected)
    journals = journal_limit(phase, 'native') + journal_limit(phase, 'reference')
    return dict(phase=phase, cases=selected, arrays_per_profile=arrays,
        values_per_profile=values, reference_rows_bytes=values * 12,
        native_rows_bytes=values * 4, journal_reservation_bytes=journals,
        combined_capture_reservation_bytes=values * 16 + journals,
        streamed_capture_reservation_bytes=2 * journals + 32 * 1024**2,
        disk_floor_bytes=DISK_FLOOR, reference_ledger=REFERENCE_LEDGER,
        reference_reserved_bytes=sum(REFERENCE_LEDGER.values()),
        representative_context_qualified=False)


def source_pins(pins):
    need(type(pins) is dict and set(pins) == set(PIN_FIELDS), 'Exact actual ArtifactPins required')
    for field in PIN_FIELDS:
        if field.endswith('_sha256'):
            need(sha(pins[field]) != '0' * 64, 'Actual nonzero ArtifactPins SHA256 required')
        else:
            need(type(pins[field]) is int and 0 < pins[field] <= 6 * 1024**3,
                 'Bounded ArtifactPins integer required')
    need(pins['header_bytes'] < 1024**2 and pins['config_bytes'] < 1024**2
         and pins['container_bytes'] > pins['header_bytes'] + 8, 'Bounded source metadata required')
    return pins


def observation(row, role, phase='initial_short'):
    fields = {'profile', 'case', 'token', 'position', 'layer', 'stage', 'head',
              'shape', 'dtype', 'count', 'offset', 'bytes', 'sha256'}
    need(type(row) is dict and set(row) == fields, 'Exact complete observation descriptor required')
    need(type(role) is str and role in PROFILES and row['profile'] in PROFILES[role], 'Explicit permitted profile required')
    named = {case['name']: case for case in cases(phase)}
    need(type(row['case']) is str and row['case'] in named, 'Frozen observation case required')
    case = named[row['case']]
    need(type(row['token']) is int and 0 <= row['token'] < len(case['tokens'])
         and type(row['position']) is int and row['position'] == case['position_offset'] + row['token'],
         'Exact observation token and absolute position required')
    need(row['layer'] is None or type(row['layer']) is int, 'Strict optional layer required')
    need(row['head'] is None or type(row['head']) is int, 'Strict optional head required')
    need(type(row['stage']) is str, 'Exact stage spelling required')
    key = row['layer'], row['stage'], row['head']
    expected = shapes(row['token'] + 1)
    need(key in expected and exact(row['shape'], expected[key]), 'Exact stage shape/head required')
    dtype = 'F64' if row['profile'] == 'ideal_fp64' else 'F32'
    need(type(row['dtype']) is str and row['dtype'] == dtype, 'Exact observation dtype required')
    count = math.prod(expected[key])
    need(type(row['count']) is int and row['count'] == count
         and type(row['offset']) is int and row['offset'] >= 0
         and type(row['bytes']) is int and row['bytes'] == count * (8 if dtype == 'F64' else 4),
         'Complete contiguous array extent required')
    sha(row['sha256'])
    return row['profile'], row['case'], row['token'], *key


def split_stage(stage):
    need(type(stage) is str, 'Graph stage string required')
    if '.' not in stage:
        return stage, None
    name, suffix = stage.rsplit('.', 1)
    need(name in ('attention_k_cache', 'attention_v_cache', 'attention_scores', 'attention_probabilities')
         and re.fullmatch('0|[1-9][0-9]*', suffix) is not None, 'Canonical split-head stage required')
    return name, int(suffix)


def finite(raw, dtype):
    need(type(raw) is bytes and dtype in ('F32', 'F64')
         and len(raw) > 0 and len(raw) % (4 if dtype == 'F32' else 8) == 0, 'Exact floating bytes required')
    need(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f' if dtype == 'F32' else '<d', raw)),
         'Nonfinite observation rejected')


def compare(actual, reference, dtype):
    """Every coordinate, inclusive unchanged 4B/9B gate; no fitted envelope."""
    finite(actual, 'F32')
    finite(reference, dtype)
    need(len(actual) // 4 == len(reference) // (4 if dtype == 'F32' else 8), 'Equal complete arrays required')
    count, failed, maximum = 0, 0, 0.0
    for (a,), (b,) in zip(struct.iter_unpack('<f', actual),
                         struct.iter_unpack('<f' if dtype == 'F32' else '<d', reference), strict=True):
        delta = abs(a - b)
        count += 1
        failed += delta > GATE['atol'] + GATE['rtol'] * abs(b)
        maximum = max(maximum, delta)
    return dict(values=count, failed_values=failed, max_abs=maximum)
