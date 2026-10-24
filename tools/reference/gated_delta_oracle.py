"""Independent FP64 gated-delta mathematical witnesses.

No Toklet, MLX, or downloaded Python source is imported. The pinned upstream
source is bound as inert bytes. Explicit upstream numerical parity is a later
required check, not a claim made by this mathematical reference producer.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from gated_delta_fixture import CASE_SPECS, MAX_ARRAY_BYTES, UPSTREAM_REVISION, UPSTREAM_SHA256


def _numpy():
    import numpy as np
    return np


def _float(value):
    np = _numpy()
    value = np.asarray(value)
    if value.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Only real FP32/FP64 reference inputs are admitted')
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite reference input')
    return value


def _geometry(q, k, v, state):
    if q.ndim != 3 or q.shape != k.shape or v.ndim != 3 or state.ndim != 3:
        raise ValueError('Expected Q/K[T,Hk,Dk], V[T,Hv,Dv], state[Hv,Dv,Dk]')
    tokens, key_heads, key_dim = q.shape
    other_tokens, value_heads, value_dim = v.shape
    if (tokens != other_tokens or not 0 <= tokens <= 8
            or not 1 <= key_heads <= 4 or not 1 <= value_heads <= 8
            or value_heads % key_heads or not 1 <= key_dim <= 128
            or not 1 <= value_dim <= 128
            or state.shape != (value_heads, value_dim, key_dim)):
        raise ValueError('Reference dimensions or grouped-head mapping are invalid')
    return tokens, key_heads, value_heads, key_dim, value_dim


def normalize_qk(q, k, epsilon):
    """L2 epsilon is added to the sum; one Dk**-0.5 scale is folded into Q."""
    np = _numpy()
    q, k = _float(q), _float(k)
    if q.ndim != 3 or q.shape != k.shape or not 1 <= q.shape[-1] <= 128:
        raise ValueError('Invalid normalization dimensions')
    if isinstance(epsilon, (bool, np.bool_)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Invalid L2 normalization epsilon')
    q, k = q.astype(np.float64), k.astype(np.float64)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + epsilon)
            k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + epsilon)
    except FloatingPointError as error:
        raise ValueError('Normalization intermediate exceeds FP64 reference bounds') from error
    return q / math.sqrt(q.shape[-1]), k


def gates(a, b, a_log, dt_bias):
    """Stable scalar decay and [0,1] update gate; no vector/negative-eigen gate."""
    np = _numpy()
    a, b, a_log, dt_bias = [_float(x) for x in (a, b, a_log, dt_bias)]
    if (a.ndim != 2 or a.shape != b.shape or a_log.ndim != 1
            or a_log.shape != dt_bias.shape or a.shape[1:] != a_log.shape):
        raise ValueError('Expected A/B[T,Hv] and A_log/dt_bias[Hv]')
    a, b, a_log, dt_bias = [x.astype(np.float64) for x in (a, b, a_log, dt_bias)]
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        decay = np.exp(-np.exp(a_log)[None, :] * np.logaddexp(0.0, a + dt_bias))
        beta = np.exp(-np.logaddexp(0.0, -b))
    if not np.isfinite(decay).all() or not np.isfinite(beta).all():
        raise ValueError('Nonfinite reference gate result')
    return decay, beta


def run_normalized(q, k, v, decay, beta, initial_state):
    """FP64 recurrence with explicit grouped heads and value-major state rows.

    Decay precedes prediction, then a rank-one correction, then Q readout.
    This unmasked reference intentionally has no masked-output equivalence claim.
    The input state is immutable; step states are retained only for tiny fixtures.
    """
    np = _numpy()
    q, k, v, decay, beta, initial_state = [_float(x) for x in (q, k, v, decay, beta, initial_state)]
    tokens, key_heads, value_heads, key_dim, value_dim = _geometry(q, k, v, initial_state)
    if decay.shape != (tokens, value_heads) or beta.shape != decay.shape:
        raise ValueError('Gate shape differs from value-head count')
    if ((decay < 0).any() or (decay > 1).any()
            or (beta < 0).any() or (beta > 1).any()):
        raise ValueError('Only scalar [0,1] decay and update gates are admitted')
    q, k, v, decay, beta = [x.astype(np.float64) for x in (q, k, v, decay, beta)]
    current = initial_state.astype(np.float64, copy=True)
    output = np.empty((tokens, value_heads, value_dim), dtype=np.float64)
    states = np.empty((tokens, value_heads, value_dim, key_dim), dtype=np.float64)
    key_indices = np.arange(value_heads) // (value_heads // key_heads)
    for token in range(tokens):
        key = k[token, key_indices]
        query = q[token, key_indices]
        decayed = current * decay[token, :, None, None]
        prediction = np.einsum('hvk,hk->hv', decayed, key)
        correction = (v[token] - prediction) * beta[token, :, None]
        current = decayed + correction[:, :, None] * key[:, None, :]
        output[token] = np.einsum('hvk,hk->hv', current, query)
        states[token] = current
    if not np.isfinite(output).all() or not np.isfinite(states).all():
        raise ValueError('Nonfinite recurrence result')
    return output, states, current


def case(name, tokens, key_heads, value_heads, key_dim, value_dim):
    """Deterministic FP32 inputs with zero/epsilon-dominated normalization lanes."""
    np = _numpy()
    def values(shape, multiplier, offset, scale):
        count = math.prod(shape)
        grid = ((np.arange(count, dtype=np.int64) * multiplier + offset) % 37) - 18
        return (grid.astype(np.float64) * scale).astype(np.float32).reshape(shape)
    q = values((tokens, key_heads, key_dim), 7, 3, 0.03125)
    k = values(q.shape, 11, 2, 0.015625)
    v = values((tokens, value_heads, value_dim), 5, 8, 0.0625)
    if tokens:
        q[0, 0] = 0
        k[0, -1] = np.float32(1e-8)
    a = values((tokens, value_heads), 3, 11, 0.25)
    b = values(a.shape, 13, 6, 0.125)
    if tokens and value_heads > 1:
        a[0, :2] = [-24, 24]
        b[0, :2] = [-24, 24]
    a_log = np.log(np.linspace(0.25, 3.0, value_heads)).astype(np.float32)
    dt_bias = values((value_heads,), 2, 3, 0.03125)
    initial = values((value_heads, value_dim, key_dim), 17, 4, 0.0078125)
    _geometry(q, k, v, initial)
    q_normalized, k_normalized = normalize_qk(q, k, 1e-6)
    decay, beta = gates(a, b, a_log, dt_bias)
    output, states, final = run_normalized(q_normalized, k_normalized, v, decay, beta, initial)
    return dict(name=name, geometry=dict(tokens=tokens, key_heads=key_heads, value_heads=value_heads,
                    key_dim=key_dim, value_dim=value_dim, epsilon=1e-6),
                arrays=dict(raw_q=q, raw_k=k, v=v, a=a, b=b, a_log=a_log, dt_bias=dt_bias,
                    initial_state=initial, normalized_q=q_normalized, normalized_k=k_normalized,
                    decay=decay, beta=beta, output=output, states=states, final_state=final))


def export(upstream_source):
    """Return a fresh binary/JSON mathematical reference, without executing it upstream."""
    np = _numpy()
    with upstream_source.open('rb') as stream:
        upstream = stream.read(256 * 1024 + 1)
    if len(upstream) > 256 * 1024 or hashlib.sha256(upstream).hexdigest() != UPSTREAM_SHA256:
        raise ValueError('Pinned upstream source bytes differ')
    cases = [case(*spec) for spec in CASE_SPECS]
    blob = bytearray()
    rows = []
    for item in cases:
        descriptors = {}
        for name, value in item['arrays'].items():
            dtype = '<f4' if value.dtype == np.dtype('float32') else '<f8'
            raw = np.ascontiguousarray(value, dtype=dtype).tobytes()
            padding = (-len(blob)) % 8
            if len(blob) + padding + len(raw) > MAX_ARRAY_BYTES:
                raise ValueError('Reference binary exceeds its bound')
            blob.extend(b'\0' * padding)
            descriptors[name] = dict(offset=len(blob), bytes=len(raw), dtype=dtype,
                                      shape=list(value.shape), sha256=hashlib.sha256(raw).hexdigest())
            blob.extend(raw)
        rows.append(dict(name=item['name'], geometry=item['geometry'], arrays=descriptors))
    manifest = dict(format='toklet-gated-delta-math-reference-v1',
        status='mathematical reference; upstream numerical parity and fixture acceptance required',
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(numpy=np.__version__), upstream_revision=UPSTREAM_REVISION,
        upstream_source_sha256=UPSTREAM_SHA256, upstream_source_executed=False,
        upstream_numerical_parity_checked=False, masked_outputs_supported=False,
        precision='FP32 stored inputs, FP64 normalization/gates/recurrence/output/state',
        normalization='sum-of-squares epsilon; Dk**-0.5 folded into Q exactly once',
        head_mapping='key_head = value_head // (value_heads // key_heads)',
        state_layout='[value_heads,value_dim,key_dim] contiguous key_dim',
        binary=dict(path='arrays.bin', bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(), alignment=8),
        cases=rows, native_graph_implemented=False, numerical_device_qualified=False)
    encoded = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    return {'manifest.json': encoded, 'arrays.bin': bytes(blob)}


def publish(outputs, destination):
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('Empty existing output directory required')
    if set(outputs) != {'manifest.json', 'arrays.bin'}:
        raise ValueError('Unexpected reference output inventory')
    for name, raw in outputs.items():
        with (destination / name).open('xb') as stream:
            stream.write(raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    publish(export(options.upstream_source), options.output)
    print('Mathematical reference exported; upstream parity and acceptance remain required.')
