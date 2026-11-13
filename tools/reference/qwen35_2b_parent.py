"""Bounded prospective text equivalence with a whole authenticated GGUF stream.

The historical publisher's converter is unknown. Exact matrix identity and
specified small-parameter narrowing are separate from functional SSM-rate parity.
No parent body is retained. Numerical packages are supplied only after admission.
"""
import hashlib
import math
from pathlib import Path
import struct

import qwen35_2b_layout as layout

need = layout.need
BLOCK = 1024**2
MAX_CA_BYTES = 1024**2
PARENT = dict(repository='HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive',
              revision='cafbddcb9940f38618e2b2ba84149584949379f9',
              file='Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-BF16.gguf', bytes=3775708640,
              sha256='d98cff5f907e2186ac8243ac5676a47dc24e1822cd6ed674c9c92c7503915a1b')
DIRECTORY_START = 10942273
DIRECTORY_END = 10961110
DATA_START = 10961120
DIRECTORY_SHA256 = 'df6a5067fd5e2233936b85bff66f6dfafc1e479f447a967ccbcb31d6345467e2'
RULE_REVISION = '5676dfbe01e7ebe0387913680eee0ed39bf4a363'
RULE_SOURCES = {
    'convert_hf_to_gguf.py': '297502d115010e97b996e20dc9cdcfc90ee293df9b616259e5a646b64f410630',
    'conversion/__init__.py': '0cf965b31162ab3a479a48bfc0dd4f8f0e5dbb5c14d7f2238855a74a8b8d407c',
    'conversion/qwen.py': '2f62b62fb1527c7693a8e178ca39fae6f04578dd089fb609725e1fce99b5c0cf',
    'conversion/base.py': '648fdc0fadb1a605c642c6de9e90e03ea8eede0622595a975e5e3b934ba26330',
    'src/models/qwen35.cpp': '923e830962205b214ff624328520296c91262d0d82f75aee2cb592c8d4dab2af',
    'src/llama-graph.cpp': '77becc9efb8b8eca7ae706dd499d137f694523725c4b849f50cd964bc32b25f8',
    'gguf-py/gguf/tensor_mapping.py': '065375aa115b9a5a6940f2a3e1d23cc5cc6a640a7281f0b677af023ed47f7584',
    'gguf-py/gguf/quants.py': 'f896726209ba8c91afd083dd87031c2d39dc4ef6a78e8a6c12763eeb8ed6ff84',
    'gguf-py/gguf/gguf_writer.py': '1b2ada9355750eb9dc8f1eda619850278a5f3823aab957672436fd8b50324542',
}
SUFFIXES = {
    'input_layernorm.weight': 'attn_norm.weight',
    'post_attention_layernorm.weight': 'post_attention_norm.weight',
    'linear_attn.A_log': 'ssm_a', 'linear_attn.conv1d.weight': 'ssm_conv1d.weight',
    'linear_attn.dt_bias': 'ssm_dt.bias', 'linear_attn.in_proj_a.weight': 'ssm_alpha.weight',
    'linear_attn.in_proj_b.weight': 'ssm_beta.weight', 'linear_attn.in_proj_qkv.weight': 'attn_qkv.weight',
    'linear_attn.in_proj_z.weight': 'attn_gate.weight', 'linear_attn.norm.weight': 'ssm_norm.weight',
    'linear_attn.out_proj.weight': 'ssm_out.weight', 'mlp.down_proj.weight': 'ffn_down.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
    'self_attn.k_norm.weight': 'attn_k_norm.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.o_proj.weight': 'attn_output.weight', 'self_attn.q_norm.weight': 'attn_q_norm.weight',
    'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.v_proj.weight': 'attn_v.weight',
}


def protocol():
    return dict(format='toklet-qwen35-2b-parent-lineage-protocol-v1', source=dict(layout.SOURCE),
                parent=dict(PARENT), directory=dict(start=DIRECTORY_START, end=DIRECTORY_END,
                    data_start=DATA_START, sha256=DIRECTORY_SHA256, alignment=32),
                rules=dict(repository='ggml-org/llama.cpp', revision=RULE_REVISION,
                           source_sha256=dict(RULE_SOURCES),
                           GGML_format=dict(repository='ggml-org/ggml', revision='cd6d2b75f5d3e4b2f3fe8e8fbc6c8b84ac80e74a',
                               source_sha256={'docs/gguf.md':'64caa0360e093f1ff66bfa67016d24879b8d23acaf11095c0410ab778a2cb2a9',
                                   'include/ggml.h':'1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
                                   'include/gguf.h':'e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4'}),
                           matrices='exact_BF16_bytes_ratio1_no_permutation',
                           small='finite_F32_RNE_to_exact_selected_BF16_no_additional_norm_offset',
                           A_log='original_selected_F32_preserved_parent_rate_vs_negative_exp',
                           profiles=['staged_fp32', 'ideal_fp64'], atol=1e-5, rtol=1e-4),
                expected=dict(logical_tensors=320, matrix_tensors=187, matrix_values=1881276416,
                    narrowed_BF16_tensors=115, narrowed_BF16_values=548384,
                    original_F32_A_log_tensors=18, original_F32_A_log_values=288),
                maximum_read_bytes=BLOCK, audit_limit_bytes=512*1024**2,
                helper_reserve_bytes=32*1024**2, engine_limit_bytes=6*1024**3,
                historical_publisher_lineage_proved=False,
                model_numerics_quality_fit_performance_adoption_qualified=False)


def source_mapping():
    result = {}
    for name, item in layout.selected_layout()['logical'].items():
        short = name.removeprefix(layout.PREFIX)
        if short == 'embed_tokens.weight':
            target = 'token_embd.weight'
        elif short == 'norm.weight':
            target = 'output_norm.weight'
        else:
            parts = short.split('.', 2)
            target = 'blk.' + parts[1] + '.' + SUFFIXES[parts[2]]
        shape = item['shape']
        kind = 'matrix' if len(shape) == 2 else 'A_log' if name.endswith('.A_log') else 'narrowed_BF16'
        dimensions = [4, 6144] if len(shape) == 3 else shape[::-1]
        need(target not in result, 'Duplicate mapped parent tensor')
        result[target] = dict(name=name, shape=shape, dtype=item['dtype'], kind=kind,
                              dimensions=dimensions, type=30 if kind == 'matrix' else 0)
    need(len(result) == 320, 'Complete text mapping required')
    return result


def directory(raw):
    need(type(raw) is bytes and len(raw) == DIRECTORY_END-DIRECTORY_START
         and hashlib.sha256(raw).hexdigest() == DIRECTORY_SHA256,
         'Exact metadata-only frozen parent directory required')
    expected = source_mapping()
    rows, cursor, seen = [], 0, set()
    def number(code):
        nonlocal cursor
        size = struct.calcsize('<'+code)
        need(cursor+size <= len(raw), 'Truncated parent directory')
        value = struct.unpack_from('<'+code, raw, cursor)[0]
        cursor += size
        return value
    for _ in range(320):
        size = number('Q')
        need(0 < size <= 1024 and cursor+size <= len(raw), 'Bounded parent tensor name required')
        name = raw[cursor:cursor+size].decode('utf-8'); cursor += size
        rank = number('I')
        need(rank in (1, 2), 'Selected parent tensor rank differs')
        dimensions = [number('Q') for _ in range(rank)]
        kind, offset = number('I'), number('Q')
        need(name in expected and name not in seen, 'Unknown or duplicate parent tensor')
        spec = expected[name]
        need(dimensions == spec['dimensions'] and kind == spec['type'], 'Parent tensor representation differs')
        seen.add(name)
        count = math.prod(dimensions)
        rows.append(dict(parent_name=name, parent_offset=offset, parent_bytes=count*(2 if kind == 30 else 4),
                         **spec))
    need(cursor == len(raw) and len(seen) == len(expected), 'Complete exact parent directory coverage required')
    rows.sort(key=lambda row: row['parent_offset'])
    need(rows[0]['parent_offset'] == 0 and all(a['parent_offset']+a['parent_bytes'] == b['parent_offset']
         for a, b in zip(rows, rows[1:])) and rows[-1]['parent_offset']+rows[-1]['parent_bytes'] == PARENT['bytes']-DATA_START,
         'Complete tight parent payload coverage required')
    return rows


def narrow_f32(raw):
    """gguf/quants.py BF16 RNE, rejecting nonfinite input/output first."""
    need(type(raw) is bytes and len(raw) % 4 == 0 and len(raw) <= BLOCK, 'Bounded complete F32 bytes required')
    out = bytearray(len(raw)//2)
    for index, (word,) in enumerate(struct.iter_unpack('<I', raw)):
        need(word & 0x7f800000 != 0x7f800000, 'Nonfinite parent small parameter')
        narrowed = (word + 0x7fff + ((word >> 16) & 1)) >> 16
        need(narrowed & 0x7f80 != 0x7f80, 'Nonfinite narrowed parent parameter')
        struct.pack_into('<H', out, index*2, narrowed)
    return bytes(out)


def numerical_callbacks(np):
    """The caller admits the complete runtime before supplying NumPy."""
    def finite(raw):
        words = np.frombuffer(raw, dtype='<u2')
        return not np.any((words & 0x7f80) == 0x7f80)
    def rates(raw):
        need(type(raw) is bytes and len(raw) == 64, 'Original sixteen F32 A_log values required')
        values = np.frombuffer(raw, dtype='<f4')
        need(np.isfinite(values).all(), 'Nonfinite selected A_log')
        main = np.negative(np.exp(values, dtype=np.float32), dtype=np.float32)
        wide = [-math.exp(value) for (value,) in struct.iter_unpack('<f', raw)]
        need(np.isfinite(main).all() and all(math.isfinite(value) and value < 0 for value in wide),
             'Nonfinite or underflowed selected SSM rate')
        return dict(staged_fp32=main.astype('<f4', copy=False).tobytes(), ideal_fp64=wide)
    return finite, rates


def rate_parity(parent_raw, source_raw, rates):
    need(type(parent_raw) is bytes and type(source_raw) is bytes and len(parent_raw) == len(source_raw)
         and len(parent_raw) % 4 == 0 and 0 < len(parent_raw) <= 64, 'Matching bounded F32 rate vectors required')
    actual = [value for (value,) in struct.iter_unpack('<f', parent_raw)]
    need(all(math.isfinite(value) and value < 0 for value in actual), 'Parent SSM rates must be negative finite')
    need(all(math.isfinite(value) for (value,) in struct.iter_unpack('<f', source_raw)), 'Original A_log must remain finite F32')
    profiles = rates(source_raw)
    need(type(profiles) is dict and set(profiles) == {'staged_fp32', 'ideal_fp64'}, 'Both independent rate profiles required')
    main = profiles['staged_fp32']
    need(type(main) is bytes and len(main) == len(parent_raw), 'Complete staged F32 rate oracle required')
    profiles = dict(staged_fp32=[v for (v,) in struct.iter_unpack('<f', main)], ideal_fp64=profiles['ideal_fp64'])
    result = {}
    for profile, values in profiles.items():
        need(type(values) is list and len(values) == len(actual)
             and all(type(v) in (int, float) and math.isfinite(v) and v < 0 for v in values),
             'Complete finite negative rate profile required')
        failed = [i for i, (a, b) in enumerate(zip(actual, values)) if abs(a-b) > 1e-5 + 1e-4*abs(b)]
        result[profile] = dict(passed=not failed, values=len(values), failed_coordinates=len(failed),
                               first_failed_coordinates=failed[:8], atol=1e-5, rtol=1e-4)
    return result


def compare_payload(read, entries, selected_read, source_spans, finite, rates, event):
    """Compare one ordered text body; the owner authenticates prefix/EOF/SHA.

    The stream is read once and discarded. A failed numerical comparison is
    recorded while the remainder is consumed, so whole-stream authentication
    may close independently of a scientific rejection.
    """
    reports = []
    for entry in entries:
        span = source_spans[entry['name']]
        parent_hash, source_hash = hashlib.sha256(), hashlib.sha256()
        passed = True
        profiles = None
        for first in range(0, entry['parent_bytes'], BLOCK):
            size = min(BLOCK, entry['parent_bytes']-first)
            raw = read(size)
            need(type(raw) is bytes and len(raw) == size, 'Truncated parent tensor body')
            parent_hash.update(raw)
            source_size = size if entry['kind'] in ('matrix', 'A_log') else size//2
            source_offset = first if entry['kind'] in ('matrix', 'A_log') else first//2
            original = selected_read(source_size, span['start']+source_offset)
            need(type(original) is bytes and len(original) == source_size, 'Selected source tensor read differs')
            source_hash.update(original)
            if entry['kind'] == 'matrix':
                need(finite(raw) and finite(original), 'Nonfinite BF16 matrix')
                equal = raw == original
            elif entry['kind'] == 'narrowed_BF16':
                equal = narrow_f32(raw) == original
            else:
                profiles = rate_parity(raw, original, rates)
                equal = all(value['passed'] for value in profiles.values())
            passed = passed and equal
            event(dict(event='parent_comparison_chunk', parent_name=entry['parent_name'], name=entry['name'],
                       kind=entry['kind'], first_parent_byte=first, parent_bytes=size,
                       source_bytes=source_size, parent_sha256=hashlib.sha256(raw).hexdigest(),
                       source_sha256=hashlib.sha256(original).hexdigest(), passed=equal,
                       functional_rate_profiles=profiles, whole_parent_authenticated=False))
        report = dict(name=entry['name'], parent_name=entry['parent_name'], kind=entry['kind'],
                      parent_bytes=entry['parent_bytes'], source_bytes=span['end']-span['start'],
                      source_sha256=source_hash.hexdigest(), parent_sha256=parent_hash.hexdigest(),
                      passed=passed, functional_rate_profiles=profiles)
        reports.append(report)
        event(dict(event='parent_tensor_compared', whole_parent_authenticated=False, **report))
    return reports


class HashedStream:
    """A forward-only bounded reader; no full parent owner or retained body."""
    def __init__(self, stream):
        self.stream, self.position = stream, 0
        self.digest = hashlib.sha256()

    def read(self, count):
        need(type(count) is int and 0 < count <= BLOCK, 'Bounded positive stream read required')
        result = bytearray()
        while len(result) < count:
            raw = self.stream.read(count-len(result))
            need(type(raw) is bytes and 0 < len(raw) <= count-len(result), 'Truncated or invalid parent stream')
            result.extend(raw)
        raw = bytes(result)
        self.digest.update(raw)
        self.position += len(raw)
        return raw

    def skip_to(self, end):
        need(type(end) is int and self.position <= end <= PARENT['bytes'], 'Forward bounded stream position required')
        while self.position < end:
            self.read(min(BLOCK, end-self.position))


def authenticate_stream(stream, selected_read, source_spans, finite, rates, event):
    owner = HashedStream(stream)
    need(owner.read(24) == b'GGUF' + struct.pack('<IQQ', 3, 320, 34), 'Exact parent GGUF prefix differs')
    owner.skip_to(DIRECTORY_START)
    entries = directory(owner.read(DIRECTORY_END-DIRECTORY_START))
    owner.skip_to(DATA_START)
    reports = compare_payload(owner.read, entries, selected_read, source_spans, finite, rates, event)
    need(owner.position == PARENT['bytes'] and stream.read(1) == b'', 'Whole parent extent/EOF differs')
    need(owner.digest.hexdigest() == PARENT['sha256'], 'Whole streamed parent SHA256 differs')
    return reports, dict(PARENT, whole_SHA256_authenticated=True, EOF_checked=True, retained_payload_bytes=0)


def validate_certificate(value, protocol_sha256):
    need(type(value) is dict and value.get('format') == 'toklet-qwen35-2b-parent-lineage-v1'
         and value.get('passed') is True and value.get('protocol_sha256') == layout.identifier(protocol_sha256),
         'Externally pinned completed selected-parent lineage certificate required')
    layout.saved.same(value.get('protocol'), protocol(), 'Frozen parent operation/source protocol differs')
    transport = value.get('transport')
    need(type(transport) is dict and set(transport) == {'https_only', 'certificate_verification',
         'hostname_verification', 'TLS_verified_for_completed_stream', 'ca'}, 'Explicit verified TLS transport receipt required')
    for key, expected in dict(https_only=True, certificate_verification='CERT_REQUIRED',
                             hostname_verification=True, TLS_verified_for_completed_stream=True).items():
        layout.saved.same(transport[key], expected, 'Verified certificate/hostname transport required')
    ca = transport['ca']
    need(type(ca) is dict and set(ca) == {'path', 'bytes', 'sha256', 'before_identity', 'after_identity'}
         and type(ca['path']) is str and Path(ca['path']).is_absolute()
         and type(ca['bytes']) is int and 0 < ca['bytes'] <= MAX_CA_BYTES, 'Bounded external CA receipt required')
    layout.identifier(ca['sha256'])
    layout.saved.pin_identity(ca['before_identity']); layout.saved.pin_identity(ca['after_identity'])
    layout.saved.same(ca['before_identity'], ca['after_identity'], 'External CA endpoints differ')
    layout.saved.same(ca['before_identity'][2], ca['bytes'], 'External CA extent differs')
    parent = value.get('parent')
    need(type(parent) is dict, 'Whole parent receipt required')
    for key, expected in PARENT.items():
        layout.saved.same(parent.get(key), expected, 'Whole parent identity differs')
    need(parent.get('whole_SHA256_authenticated') is True and parent.get('EOF_checked') is True
         and type(parent.get('retained_payload_bytes')) is int and parent['retained_payload_bytes'] == 0,
         'Complete streamed/discarded parent authentication required')
    selected = value.get('selected_source')
    need(type(selected) is dict, 'Authenticated selected source endpoint required')
    for key, expected in layout.SOURCE.items():
        layout.saved.same(selected.get(key), expected, 'Selected source identity differs')
    need(selected.get('whole_SHA256_authenticated_before') is True
         and selected.get('whole_SHA256_authenticated_after') is True
         and selected.get('before_identity') == selected.get('after_identity'),
         'Selected source byte and descriptor endpoints must close')
    layout.saved.pin_identity(selected.get('before_identity'))
    layout.saved.same(selected['before_identity'][2], layout.SOURCE_BYTES, 'Selected source extent differs')
    need(value.get('historical_publisher_lineage_proved') is False
         and value.get('A_log_bitwise_reconstruction_claimed') is False
         and value.get('model_numerics_quality_fit_performance_adoption_qualified') is False,
         'Prospective text equivalence must not imply model or historical qualification')
    reports = value.get('tensors')
    expected = source_mapping()
    need(type(reports) is list and len(reports) == len(expected) and all(type(row) is dict for row in reports)
         and {row.get('parent_name') for row in reports} == set(expected), 'Complete unique text lineage reports required')
    for row in reports:
        spec = expected[row['parent_name']]
        need(row.get('name') == spec['name'] and row.get('kind') == spec['kind'] and row.get('passed') is True,
             'Every selected text comparison must pass')
        layout.identifier(row.get('source_sha256')); layout.identifier(row.get('parent_sha256'))
        count = math.prod(spec['shape'])
        layout.saved.same(row.get('source_bytes'), count*(4 if spec['dtype'] == 'F32' else 2),
                          'Complete selected tensor comparison extent required')
        layout.saved.same(row.get('parent_bytes'), count*(2 if spec['kind'] == 'matrix' else 4),
                          'Complete parent tensor comparison extent required')
        if spec['kind'] == 'A_log':
            profiles = row.get('functional_rate_profiles')
            need(type(profiles) is dict and set(profiles) == {'staged_fp32', 'ideal_fp64'}, 'Both rate profiles required')
            for item in profiles.values():
                layout.saved.same(item, dict(passed=True, values=16, failed_coordinates=0,
                                             first_failed_coordinates=[], atol=1e-5, rtol=1e-4),
                                  'Unchanged complete functional rate gates required')
    return value
