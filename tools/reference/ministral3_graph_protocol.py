"""Cold explicit Ministral graph geometry and unchanged dual-profile gates.

This is a reference protocol, not ordinary model or device admission. Numerical
packages are never imported here. Hidden, query and KV widths are independent.
"""
import math
import struct

POLICY = dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3)
STAGES = ('input_norm', 'q', 'k', 'v', 'q_rotary', 'k_rotary',
          'attention_context', 'attention_projection', 'attention_residual',
          'post_attention_norm', 'gate', 'silu', 'up', 'gated', 'down', 'output')
TINY = dict(hidden=64, intermediate=128, layers=1, vocabulary=64,
            query_heads=2, kv_heads=1, head_dim=128, theta=1000000.0,
            rms_eps=1e-5, tied_head=True)
REAL = dict(hidden=3072, intermediate=9216, layers=26, vocabulary=131072,
            query_heads=32, kv_heads=8, head_dim=128, theta=1000000.0,
            rms_eps=1e-5, tied_head=True)
TINY_CASES = (
    ('single0', [1], 0),
    ('prompt', [1, 7, 13, 3], 0),
    ('future_changed', [1, 7, 13, 5], 0),
    ('offset5', [2, 11, 9], 5),
    ('boundary509', [1, 7, 13], 509),
    # An absolute-position composition fixture, not an admitted context window.
    ('diagnostic_gain16384', [2, 11], 16384),
)


def need(value, message):
    if not value:
        raise ValueError(message)


def f32(value):
    need(type(value) in (int, float) and math.isfinite(value), 'Finite FP32 value required')
    try:
        result = struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError('FP32 overflow') from error
    need(math.isfinite(result), 'Finite FP32 value required')
    return result


def geometry(config, tokens, offset, diagnostic_offset=False):
    need(type(config) is dict and set(config) == set(REAL), 'Complete explicit Ministral geometry required')
    for key in ('hidden', 'intermediate', 'layers', 'vocabulary', 'query_heads', 'kv_heads'):
        need(type(config[key]) is int and 1 <= config[key] <= REAL[key], 'Bounded geometry required: ' + key)
    need(type(config['head_dim']) is int and config['head_dim'] == 128
         and config['hidden'] % 64 == config['intermediate'] % 64 == 0
         and config['query_heads'] % config['kv_heads'] == 0
         and type(config['tied_head']) is bool and config['tied_head'],
         'Explicit D128 grouped-query, group64 and tied-head geometry required')
    need(type(config['theta']) is float and config['theta'] == 1000000.0
         and type(config['rms_eps']) is float and config['rms_eps'] == 1e-5,
         'Selected source rotary and normalization scalars required')
    need(type(tokens) is list and 1 <= len(tokens) <= 512
         and all(type(token) is int and 0 <= token < config['vocabulary'] for token in tokens),
         'Exact bounded nonempty token IDs required')
    need(type(diagnostic_offset) is bool and type(offset) is int and offset >= 0
         and offset + len(tokens) <= (262144 if diagnostic_offset else 512),
         'Explicit bounded absolute-position span required')


def contracts(config, count):
    geometry(config, [0] * count if type(count) is int and 1 <= count <= 512 else [], 0)
    h, q, kv = (config['hidden'], config['query_heads'] * 128, config['kv_heads'] * 128)
    widths = dict.fromkeys(STAGES, h)
    widths.update(q=q, q_rotary=q, attention_context=q, k=kv, v=kv, k_rotary=kv)
    widths.update({name: config['intermediate'] for name in ('gate', 'silu', 'up', 'gated')})
    result = {name: [count, width] for name, width in
              [('embedding', h), ('final_norm', h), ('logits', config['vocabulary'])]}
    for layer in range(config['layers']):
        result.update({f'layers.{layer}.{name}': [count, width] for name, width in widths.items()})
    return result


def gate(main, wide):
    need(type(main) is list and type(wide) is list and len(main) == len(wide) > 0
         and all(type(v) in (int, float) and math.isfinite(v) for v in main + wide),
         'Complete finite independent profiles required')
    envelope = max(abs(a - b) for a, b in zip(main, wide, strict=True))
    return envelope, dict(atol=max(POLICY['atol_floor'], POLICY['envelope_factor'] * envelope),
                          rtol=POLICY['rtol'])


def member(actual, reference, tolerance):
    need(type(actual) in (int, float) and math.isfinite(actual)
         and type(reference) in (int, float) and math.isfinite(reference), 'Finite comparison required')
    need(type(tolerance) is dict and set(tolerance) == {'atol', 'rtol'}
         and type(tolerance['atol']) is float and math.isfinite(tolerance['atol'])
         and tolerance['atol'] >= POLICY['atol_floor']
         and type(tolerance['rtol']) is float and tolerance['rtol'] == POLICY['rtol'],
         'Frozen per-array tolerance required')
    return abs(actual - reference) <= tolerance['atol'] + tolerance['rtol'] * abs(reference)
