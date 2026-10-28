"""Independent seeded Half old-ring/new-chunk attention and copy references.

Stdlib only; no model, target state implementation or GPU is imported. Physical
retentionW and minimal old numerical viewW-1 are deliberately distinct. Main
emulates prescribed serial FP32 arithmetic; wide is a separate FP64 reference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

FORMAT = 'toklet-gemma3-staged-attention-v1'
PROTOCOL = '7b4508605e7432c39fd70ce409ad4dd658541054361b2e02937acfa02055e30f'
POLICY = dict(fatal_reference='main', atol_floor=1e-4, envelope_factor=2, rtol=1e-3)
HEADS, MAX_CHUNK, GUARD, PAD = 4, 32, 7, 0x7e55
MAX_BINARY, MAX_JSON = 16 * 1024**2, 256 * 1024
CASES = tuple(dict(name=prefix + '_' + kind, kind=kind, committed=c, tokens=n,
    dim=d, capacity=cap, window=w if kind == 'local' else None, pattern=pattern)
    for prefix, c, n, d, cap, w, pattern in (
        ('tiny', 5, 7, 32, 12, 4, 'dyadic'),
        ('boundary513', 512, 1, 256, 544, 512, 'uniform'),
        ('boundary514', 513, 1, 256, 544, 512, 'uniform'),
        ('chunk32', 500, 32, 256, 544, 512, 'dyadic'))
    for kind in ('local', 'full'))


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def half_word(x):
    return struct.unpack('<H', struct.pack('<e', x))[0]


def half_value(x):
    return struct.unpack('<e', struct.pack('<H', x))[0]


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def word_identity(words):
    return identity(b''.join(struct.pack('<H', x) for x in words))


def shape(c):
    for key in ('committed', 'tokens', 'dim', 'capacity'):
        if type(c[key]) is not int:
            raise ValueError('Integer shape required')
    start, n, d, cap, w = c['committed'], c['tokens'], c['dim'], c['capacity'], c['window']
    if start < 0 or not 1 <= n <= MAX_CHUNK or start + n > cap:
        raise ValueError('Bounded absolute positions')
    if (d, cap) not in ((32, 12), (256, 544)):
        raise ValueError('Fixed dimensional profile')
    if w is not None and (type(w) is not int or w != (4 if d == 32 else 512)):
        raise ValueError('Fixed window profile')
    if c['kind'] != ('local' if w else 'full') or c['pattern'] not in ('dyadic', 'uniform'):
        raise ValueError('Kind/pattern profile')
    return start, n, d, cap, w


def view(c):
    committed, n, d, cap, window = shape(c)
    first = max(0, committed - (window - 1)) if window else 0
    retention = window or cap
    old = committed - first
    slot = first % retention
    first_count = min(old, retention - slot)
    spans = []
    if first_count:
        spans.append(dict(absolute_start=first, slot=slot, rows=first_count))
    if old > first_count:
        spans.append(dict(absolute_start=first + first_count, slot=0, rows=old - first_count))
    return dict(start=first, end=committed + n, old_spans=spans,
                new_start=committed, new_rows=n, retention=retention)


def copy_plan(c):
    committed, n, d, cap, window = shape(c)
    skip = max(0, n - window) if window else 0
    keep = n - skip
    slot = (committed + skip) % window if window else committed
    retention = window or cap
    first = min(keep, retention - slot)
    runs = [dict(source_row=skip, destination_row=slot, rows=first)]
    if keep > first:
        runs.append(dict(source_row=skip + first, destination_row=0, rows=keep - first))
    return runs


def row(position, dim, kind, pattern):
    if kind == 'v' and pattern == 'uniform':
        values = [0.0] * dim
        for feature, key in enumerate((0, 1, 512, 513)):
            if position == key:
                values[feature] = 512.0
        return [half_word(x) for x in values]
    modulus, divisor = (17, 32) if kind == 'k' else (19, 16)
    return [half_word(((position * 5 + feature * 3 + (1 if kind == 'k' else 7)) % modulus
                      - modulus // 2) / divisor) for feature in range(dim)]


def physical(committed, retention, dim, kind, pattern, local):
    words = [PAD] * (retention * dim + GUARD)
    first = max(0, committed - retention) if local else 0
    for position in range(first, committed):
        slot = position % retention if local else position
        words[slot * dim:(slot + 1) * dim] = row(position, dim, kind, pattern)
    return words


def operands(c):
    committed, n, d, cap, window = shape(c)
    retention, pattern = window or cap, c['pattern']
    old = {kind: physical(committed, retention, d, kind, pattern, window is not None) for kind in ('k', 'v')}
    staged = {kind: [PAD] * (MAX_CHUNK * d + GUARD) for kind in ('k', 'v')}
    q = [PAD] * (MAX_CHUNK * HEADS * d + GUARD)
    for token in range(n):
        for kind in ('k', 'v'):
            staged[kind][token * d:(token + 1) * d] = row(committed + token, d, kind, pattern)
        for head in range(HEADS):
            for feature in range(d):
                value = 0.0 if pattern == 'uniform' else ((7 * (committed + token) + 5 * head + 3 * feature) % 13 - 6) / 32
                q[(token * HEADS + head) * d + feature] = half_word(value)
    return old, staged, q


def decoded_view(c, old, staged, q):
    committed, n, d, cap, window = shape(c)
    description = view(c)
    retention = window or cap
    if len(q) != MAX_CHUNK * HEADS * d + GUARD:
        raise ValueError('Q physical extent')
    for kind in ('k', 'v'):
        if len(old[kind]) != retention * d + GUARD or len(staged[kind]) != MAX_CHUNK * d + GUARD:
            raise ValueError('KV physical extent')
    selected = {'k': [], 'v': []}
    for position in range(description['start'], description['end']):
        for kind in ('k', 'v'):
            if position < committed:
                index = (position % retention if window else position) * d
                values = old[kind][index:index + d]
            else:
                index = (position - committed) * d
                values = staged[kind][index:index + d]
            selected[kind].append([half_value(x) for x in values])
    queries = [[half_value(x) for x in q[token * HEADS * d:(token + 1) * HEADS * d]] for token in range(n)]
    if any(not math.isfinite(x) for rows in [selected['k'], selected['v'], queries] for values in rows for x in values):
        raise ValueError('Nonfinite active numerical input')
    return description, queries, selected['k'], selected['v']


def attention(c, old, staged, q, profile):
    if profile not in ('main', 'wide'):
        raise ValueError('Unknown precision profile')
    committed, n, d, _, window = shape(c)
    description, queries, keys, values = decoded_view(c, old, staged, q)
    first, count = description['start'], len(keys)
    scores, probs = [], []
    context = [0] * (n * HEADS * d)
    root = f32(math.sqrt(d)) if profile == 'main' else math.sqrt(d)
    for head in range(HEADS):
        for token, query in enumerate(queries):
            position = committed + token
            first_visible = max(first, position - (window - 1)) if window else first
            included = range(first_visible - first, position - first + 1)
            row_scores = []
            for key in keys:
                if profile == 'main':
                    total = 0.0
                    for feature in range(d):
                        total = f32(total + f32(query[head * d + feature] * key[feature]))
                else:
                    total = math.fsum(query[head * d + feature] * key[feature] for feature in range(d))
                row_scores.append(f32(total / root))
            maximum = max(row_scores[i] for i in included)
            exponential = [0.0] * count
            mass = 0.0
            for i in range(count):
                if i in included:
                    difference = f32(row_scores[i] - maximum) if profile == 'main' else row_scores[i] - maximum
                    e = math.exp(difference)
                    exponential[i] = f32(e) if profile == 'main' else e
                mass = f32(mass + exponential[i]) if profile == 'main' else mass + exponential[i]
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError('Softmax mass')
            row_probs = [f32(e / mass) for e in exponential]
            for feature in range(d):
                if profile == 'main':
                    total = 0.0
                    for i in included:
                        total = f32(total + f32(row_probs[i] * values[i][feature]))
                else:
                    total = math.fsum(row_probs[i] * values[i][feature] for i in included)
                context[(token * HEADS + head) * d + feature] = half_word(total)
            scores.extend(row_scores)
            probs.extend(row_probs)
    if any(not math.isfinite(x) for x in scores + probs) or any(not math.isfinite(half_value(x)) for x in context):
        raise ValueError('Nonfinite attention output')
    return dict(scores=scores, probabilities=probs, context=context)


def commit_reference(c, old, staged):
    """Apply copies to fresh arrays; never mutates the input old cache/staging."""
    _, _, dim, _, _ = shape(c)
    result = {kind: words[:] for kind, words in old.items()}
    for run in copy_plan(c):
        src, dst, length = run['source_row'] * dim, run['destination_row'] * dim, run['rows'] * dim
        for kind in ('k', 'v'):
            result[kind][dst:dst + length] = staged[kind][src:src + length]
    return result


def validate_frames(owners, frames, math_ok=True, poisoned=False):
    """Small independent state predicate, not an implementation of GPU ownership."""
    if poisoned or not math_ok or not owners or len(owners) != len(frames):
        raise ValueError('Uncommittable set')
    ids = [item['owner'] for item in owners]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate owner')
    for owner, frame in zip(owners, frames, strict=True):
        if any(type(owner[key]) is not int or owner[key] < 0 for key in ('owner', 'generation', 'committed', 'capacity')):
            raise ValueError('Owner metadata type/range')
        if owner['owner'] == 0 or owner['generation'] >= 2**64 - 1:
            raise ValueError('Owner identity/generation')
        if any(type(frame[key]) is not int for key in ('owner', 'generation', 'committed', 'tokens')):
            raise ValueError('Frame metadata type')
        if any(owner[key] != frame[key] for key in ('owner', 'generation', 'committed')):
            raise ValueError('Foreign/stale frame')
        if type(frame['tokens']) is not int or not 1 <= frame['tokens'] <= MAX_CHUNK:
            raise ValueError('Invalid frame count')
        if owner['committed'] + frame['tokens'] > owner['capacity']:
            raise ValueError('Commit overflow')
    if len({(f['committed'], f['tokens']) for f in frames}) != 1:
        raise ValueError('Unmatched batch prefix')


def export(directory):
    before = identity(Path(__file__).read_bytes())
    directory.mkdir(parents=False, exist_ok=False)
    cases, previous = [], {}
    try:
        with (directory / 'arrays.bin').open('xb') as binary, (directory / 'progress.jsonl').open('x') as progress:
            def save(values, kind, dimensions):
                raw = b''.join(struct.pack('<' + kind, x) for x in values)
                offset = binary.tell()
                if offset + len(raw) > MAX_BINARY:
                    raise ValueError('Binary bound')
                binary.write(raw)
                return dict(file='arrays.bin', dtype='u16le' if kind == 'H' else 'f32le', offset=offset, shape=dimensions, **identity(raw))
            for c in CASES:
                progress.write(json.dumps(dict(case=c['name'], started=True)) + '\n')
                progress.flush()
                old, staged, q = operands(c)
                pins = {name: word_identity(values) for name, values in [('q', q), *[(f'old_{k}', v) for k, v in old.items()], *[(f'new_{k}', v) for k, v in staged.items()]]}
                if c['name'].startswith('boundary514') and old != previous[c['kind']]:
                    raise ValueError('Sequential boundary cache differs')
                main = attention(c, old, staged, q, 'main')
                wide = attention(c, old, staged, q, 'wide')
                after = commit_reference(c, old, staged)
                committed, n, dim, cap, window = shape(c)
                expected = {k: physical(committed + n, window or cap, dim, k, c['pattern'], window is not None) for k in ('k', 'v')}
                if after != expected:
                    raise ValueError('Copy plan differs from independent absolute reconstruction')
                if c['name'].startswith('boundary513'):
                    previous[c['kind']] = after
                for name, values in [('q', q), *[(f'old_{k}', v) for k, v in old.items()], *[(f'new_{k}', v) for k, v in staged.items()]]:
                    if word_identity(values) != pins[name]:
                        raise ValueError('Source attention/copy mutated input')
                desc = view(c)
                exact_context = None
                if c['pattern'] == 'uniform' and window is not None:
                    exact_context = [0, 0x3c00, 0x3c00, 0] if committed == 512 else [0, 0, 0x3c00, 0x3c00]
                    for profile in (main, wide):
                        for head in range(HEADS):
                            if profile['context'][head * dim:head * dim + 4] != exact_context:
                                raise ValueError('Uniform local visibility differs')
                visible = [[max(0, committed + row - (window - 1)) if window else 0, committed + row + 1] for row in range(n)]
                for profile in (main, wide):
                    keys = desc['end'] - desc['start']
                    for head in range(HEADS):
                        for token, (start, end) in enumerate(visible):
                            for i in range(keys):
                                value = profile['probabilities'][(head * n + token) * keys + i]
                                if not start <= desc['start'] + i < end and value != 0:
                                    raise ValueError('Nonzero masked probability')
                inputs = {'q': save(q, 'H', [len(q)])}
                for kind in ('k', 'v'):
                    inputs['old_' + kind] = save(old[kind], 'H', [len(old[kind])])
                    inputs['new_' + kind] = save(staged[kind], 'H', [len(staged[kind])])
                    inputs['expected_after_' + kind] = save(after[kind], 'H', [len(after[kind])])
                arrays = {}
                for stage in ('scores', 'probabilities', 'context'):
                    is_context = stage == 'context'
                    a = [half_value(x) for x in main[stage]] if is_context else main[stage]
                    b = [half_value(x) for x in wide[stage]] if is_context else wide[stage]
                    envelope = max(abs(x - y) for x, y in zip(a, b, strict=True))
                    dimensions = [n, HEADS, dim] if is_context else [HEADS, n, desc['end'] - desc['start']]
                    kind = 'H' if is_context else 'f'
                    arrays[stage] = dict(main=save(main[stage], kind, dimensions), wide=save(wide[stage], kind, dimensions),
                        envelope=envelope, gate=dict(atol=max(1e-4, 2 * envelope), rtol=1e-3))
                cases.append(dict(**c, view=desc, visible_ranges=visible, copy_runs=copy_plan(c),
                    query_heads=HEADS, kv_heads=1, chunk_capacity=MAX_CHUNK, score_key_capacity=cap,
                    inputs=inputs, arrays=arrays, exact_context_first_four_per_head=exact_context,
                    seeded_state_only=True, source_input_immutability=pins))
                binary.flush()
                progress.write(json.dumps(dict(case=c['name'], complete=True, arrays_end=binary.tell())) + '\n')
                progress.flush()
        files = {'arrays.bin': identity((directory / 'arrays.bin').read_bytes())}
        if identity(Path(__file__).read_bytes()) != before or identity((directory / 'arrays.bin').read_bytes()) != files['arrays.bin']:
            raise ValueError('Endpoint changed')
        manifest = dict(format=FORMAT, protocol_sha256=PROTOCOL, producer=before, files=files, cases=cases,
            policy=POLICY, guard=GUARD, input_padding_half_word=PAD, source_reference_executed=True,
            target_gpu_executed=False, full_graph_executed=False,
            numerical_profile_arrays=48, physical_state_arrays=56, stored_arrays=104,
            context_values_per_profile=71424, score_values_per_profile=145012, probability_values_per_profile=145012,
            runtime=dict(python=sys.version, executable=str(Path(sys.executable).resolve())),
            scope='Seeded primitive numerical/copy expectations; source controls do not certify GPU ownership, completion, poisoning or reset')
        raw = encoded(manifest)
        if len(raw) > MAX_JSON:
            raise ValueError('Manifest bound')
        (directory / 'manifest.json').write_bytes(raw)
        return identity(raw)
    except BaseException as error:
        (directory / 'failure.json').write_bytes(encoded(dict(complete=False, error=type(error).__name__ + ': ' + str(error))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir)))
