"""Independent, source-specific9B row binding with an explicit separate head.

No target engine or MLX graph is imported. The old4B entry point and its complete
inventory remain strict. This binding consumes the additional physical head.
"""
import hashlib
import json
import math
import struct
import affine4_large_rows as large
import affine4_tiles as tiles
import bf16_source
import check_qwen35_9b_payload as pin
import qwen35_cpu_graph as graph
import qwen35_real_weights as base

need, unique, component = base.need, base.unique, base.component
REPOSITORY = 'lukey03/Qwen3.5-9B-abliterated-MLX-4bit'
REVISION = '8ff61b6a3af49b4f3973a5ed8ff0937de812c0d3'
HEADER_BYTES = 120881
CONFIG_BYTES = 2431
ORIGIN = HEADER_BYTES + 8
HEAD = 'language_model.lm_head.weight'
EMBED = graph.P+'embed_tokens.weight'
GEOMETRY = dict(hidden=4096, intermediate=12288, vocab=248320, layers=32,
    query_heads=16, kv_heads=4, head_dim=256, rotated_dims=64,
    key_heads=16, value_heads=32, key_dim=128, value_dim=128,
    max_positions=262144, rope_theta=10000000.0, rms_epsilon=1e-6)


def bind(owner, table, origin, geometry, deadline):
    """Tiny/test or authenticated source binding; public eligibility is weights()."""
    import numpy as np
    cursor = base.Cursor(owner, deadline)
    matrices, vectors, consumed = {}, {}, set()
    specs = graph.specs(geometry)
    specs[HEAD] = ('matrix', [geometry['vocab'], geometry['hidden']])
    for name, (kind, shape) in specs.items():
        if kind == 'matrix':
            rows, columns = shape
            stem = name.removesuffix('.weight')
            offsets = []
            for key, dtype, sizes in [(name, 'U32', [rows, columns//8]),
                (stem+'.scales', 'BF16', [rows, columns//64]),
                (stem+'.biases', 'BF16', [rows, columns//64])]:
                offsets.append(component(table, origin, owner.size, key, dtype, sizes))
                consumed.add(key)
            matrices[name] = base.Matrix(large.LargeMatrixRows(cursor, owner.size, rows, columns, *offsets))
        else:
            offset = component(table, origin, owner.size, name, 'BF16', shape)
            count = math.prod(shape)
            need(count <= 32768, 'Small unquantized owner exceeds bound')
            values = []
            for start in range(0, count, 8192):
                n = min(8192, count-start)
                raw = owner.read(offset+2*start, 2*n, deadline=deadline)
                need(len(raw) == 2*n, 'Truncated unaligned BF16 component')
                values.extend(tiles.bf16(np.frombuffer(raw, dtype='<u2')).tolist())
            vectors[name] = values
            consumed.add(name)
    need(consumed == set(table), 'Unconsumed or missing9B physical components')
    need(matrices[EMBED] is not matrices[HEAD], 'Distinct9B table owners required')
    return matrices, vectors


def weights(owner, config, deadline):
    need(type(owner) is bf16_source.Source and owner.size == pin.FILE_BYTES
         and owner.sha256 == pin.FILE_SHA, 'Authenticated pinned9B source required')
    need(type(config) is bytes and len(config) == CONFIG_BYTES
         and hashlib.sha256(config).hexdigest() == pin.CONFIG_SHA, 'Pinned9B config differs')
    need(owner.read(0, 8, deadline=deadline) == struct.pack('<Q', HEADER_BYTES), '9B header prefix differs')
    raw = owner.read(8, HEADER_BYTES, deadline=deadline)
    need(hashlib.sha256(raw).hexdigest() == pin.HEADER_SHA, 'Pinned9B header differs')
    table = json.loads(raw, object_pairs_hook=unique)
    table.pop('__metadata__')
    need(len(table) == 927, 'Physical9B inventory differs')
    spans = sorted(item['data_offsets'] for item in table.values())
    need(spans[0][0] == 0 and spans[-1][1] == owner.size-ORIGIN
         and all(a[1] == b[0] for a, b in zip(spans, spans[1:])), 'Complete9B payload coverage differs')
    matrices, vectors = bind(owner, table, ORIGIN, GEOMETRY, deadline)
    need(len(matrices) == 250 and len(vectors) == 177, 'Logical9B inventory differs')
    return matrices, vectors


def math_type(matrices):
    """Bind the final operation without changing direct input-table indexing."""
    embedding, head = matrices[EMBED], matrices[HEAD]
    need(embedding is not head, 'Separate output table required')
    class OutputMath(base.Math):
        def linear(self, values, matrix):
            return super().linear(values, head if matrix is embedding else matrix)
    return OutputMath


def expected_native():
    result = {(None, 'embedding', None): 4096, (None, 'final_norm', None): 4096,
              (None, 'logits', None): 248320}
    common = dict(input_norm=4096, operator_output=4096, operator_residual=4096,
        post_norm=4096, mlp_gate=12288, mlp_up=12288, mlp_activated=12288,
        mlp_output=4096, layer_output=4096)
    delta = dict(delta_qkv=8192, delta_z=4096, delta_a=32, delta_b=32,
        delta_convolved=8192, delta_q=2048, delta_k=2048, delta_v=4096,
        delta_decay=32, delta_beta=32, delta_state=524288, delta_history=24576,
        delta_output=4096, delta_norm=4096, delta_gated=4096)
    attention = dict(attention_q_gate=8192, attention_q=4096, attention_k=1024,
        attention_v=1024, attention_q_norm=4096, attention_k_norm=1024,
        attention_q_rot=4096, attention_k_rot=1024, attention_context=4096, attention_gated=4096)
    for layer in range(32):
        for name, count in {**common, **(attention if layer%4 == 3 else delta)}.items():
            result[layer, name, None] = count
        if layer%4 == 3:
            for head in range(4):
                for name in ['attention_k_cache', 'attention_v_cache']: result[layer, name, head] = 256
            for head in range(16):
                for name in ['attention_scores', 'attention_probabilities']: result[layer, name, head] = 1
    return result
