"""Bounded NumPy reference candidate; FP64 BLAS is a distinct reduction profile.

Batch one and a single KV head only. Inputs are already rotated/normalized.
This does not implement a Toklet kernel, cache transaction or model admission.
"""
import math

PROFILE = 'fp64-blas-qk-pv-scalar-fsum-softmax-explicit-stage-boundaries-v1'
QUERY_TILE = 8
MAX_POSITIONS = 1025
MAX_ARRAY_BYTES = 128 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def attention(q, k, v, *, staged, window=None, query_start=0, key_start=0,
              diagnostics=False):
    """Return context, or (context, all scores, masked probabilities) for Q<=9.

    K/V describe a contiguous suffix ending at the last query. They must cover
    every visible key, including the complete prefix for full attention. The
    default context-only path keeps scores/probabilities in eight-query tiles.
    A diagnostic call also reports masked-column scores for the frozen witness.
    """
    import numpy as np

    require(type(staged) is bool and type(diagnostics) is bool,
            'Explicit reference precision and diagnostic mode required')
    require(all(type(x) is np.ndarray and x.ndim == 4 for x in (q, k, v)),
            'Reference attention arrays must have rank four')
    dtype = np.dtype('float32' if staged else 'float64')
    require(all(x.dtype == dtype for x in (q, k, v)),
            'Reference attention input dtype differs')
    require(q.shape[0] == 1 and k.shape[:2] == (1, 1) and v.shape == k.shape,
            'Reference attention requires batch one and one KV head')
    _, heads, queries, dims = q.shape
    keys = k.shape[2]
    require(1 <= heads <= 8 and 1 <= queries <= MAX_POSITIONS and
            1 <= keys <= MAX_POSITIONS and 1 <= dims <= 512 and k.shape[3] == dims,
            'Bounded reference attention geometry required')
    require(type(query_start) is int and type(key_start) is int and
            0 <= key_start <= query_start and
            query_start + queries == key_start + keys <= MAX_POSITIONS,
            'Reference attention absolute bounds differ')
    require(window is None or type(window) is int and 1 <= window <= 512,
            'Reference attention window differs')
    required_start = 0 if window is None else max(0, query_start - window + 1)
    require(key_start <= required_start,
            'Reference attention keys omit visible prefix')
    require(not diagnostics or queries <= 9, 'Reference attention diagnostic query bound')

    # Conservative logical-array reservation, not a process-memory admission.
    # Includes inputs/output, conversion overlap, finite-check temporaries,
    # old/new tile overlap, optional diagnostics and bounded range metadata.
    elements = q.size + k.size + v.size
    arrays = (elements * dtype.itemsize + q.size * dtype.itemsize + elements +
              2 * keys * dims * 8 + 4 * QUERY_TILE * dims * 8 +
              8 * QUERY_TILE * keys * 8 + 1024**2)
    if diagnostics:
        arrays += 2 * heads * queries * keys * dtype.itemsize
    require(arrays <= MAX_ARRAY_BYTES, 'Reference attention array reservation exceeded')
    require(all(np.isfinite(x).all() for x in (q, k, v)),
            'Finite reference attention inputs required')

    ranges = []
    for row in range(queries):
        absolute = query_start + row
        start = 0 if window is None else max(0, absolute - window + 1 - key_start)
        end = absolute - key_start + 1
        require(0 <= start < end <= keys, 'Reference attention has no visible keys')
        ranges.append((start, end))

    keys64 = k[0, 0].astype(np.float64, copy=False)
    values64 = v[0, 0].astype(np.float64, copy=False)
    output = np.empty(q.shape, dtype=dtype)
    all_scores = np.empty((heads, queries, keys), dtype=dtype) if diagnostics else None
    all_probabilities = np.empty_like(all_scores) if diagnostics else None
    for head in range(heads):
        for before in range(0, queries, QUERY_TILE):
            after = min(before + QUERY_TILE, queries)
            end_key = keys if diagnostics else ranges[after - 1][1]
            query64 = q[0, head, before:after].astype(np.float64, copy=False)
            with np.errstate(over='ignore', invalid='ignore'):
                dot = query64 @ keys64[:end_key].T
                scores = dot.astype(dtype, copy=False)
            require(np.isfinite(dot).all() and np.isfinite(scores).all(),
                    'Nonfinite reference attention scores')
            probabilities = np.zeros(scores.shape, dtype=dtype)
            for row, (start, end) in enumerate(ranges[before:after]):
                active = scores[row, start:end]
                maximum = float(max(active))
                shifted = [float(value) - maximum for value in active]
                require(all(math.isfinite(value) for value in shifted),
                        'Nonfinite reference attention softmax shift')
                exponent = [math.exp(value) for value in shifted]
                total = math.fsum(exponent)
                probabilities[row, start:end] = np.array(
                    [value / total for value in exponent], dtype=dtype)
            require(np.isfinite(probabilities).all(),
                    'Nonfinite reference attention probabilities')
            with np.errstate(over='ignore', invalid='ignore'):
                dot = probabilities.astype(np.float64, copy=False) @ values64[:end_key]
                rounded = dot.astype(dtype, copy=False)
            require(np.isfinite(dot).all() and np.isfinite(rounded).all(),
                    'Nonfinite reference attention context')
            output[0, head, before:after] = rounded
            if diagnostics:
                all_scores[head, before:after] = scores
                all_probabilities[head, before:after] = probabilities
    return (output, all_scores, all_probabilities) if diagnostics else output
