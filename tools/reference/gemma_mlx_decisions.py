"""Pure planning/statistics for the bounded same-checkpoint MLX witness."""
import array
import hashlib
import math
import struct
import sys
import gemma4_text_math as geometry


def samples(record, role):
    if role == 'original':
        return [value['token_id'] for value in record['generation']['samples']]
    result = list(record['visible_token_ids'])
    if record['stop_reason']['kind'] == 'stop_token':
        result.append(record['stop_reason']['token_id'])
    return result


def select(cases, outcomes, original, native):
    if not all(len(value) == 100 for value in [cases, original, native, outcomes['baseline'], outcomes['native']]):
        raise ValueError('Complete admitted paired inventory required')
    selected = []
    for category in ['code', 'prose', 'continuation']:
        sides = {'original': [], 'native': []}
        for index, case in enumerate(cases):
            for values in [original, native, outcomes['baseline'], outcomes['native']]:
                if values[index]['id'] != case['id'] or values[index]['category'] != case['category']:
                    raise ValueError('Frozen case/outcome order differs')
            a, b = outcomes['baseline'][index]['passed'], outcomes['native'][index]['passed']
            if type(a) is not bool or type(b) is not bool:
                raise ValueError('Resolved binary outcomes required')
            if case['category'] == category and a != b:
                sides['original' if a else 'native'].append(index)
        if not any(sides.values()):
            continue
        if not all(sides.values()):
            raise ValueError('Fixed bidirectional witness rule unavailable in affected category')
        for side in ['original', 'native']:
            index = sides[side][0]
            case = cases[index]
            a, b = samples(original[index], 'original'), samples(native[index], 'native')
            different = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if different is None:
                raise ValueError('Different grading outcomes lack a sampled-ID disagreement')
            prefix = case['prompt_tokens']+a[:different]
            if not 1 <= len(prefix) <= 567 or any(type(i) is not int or not 0 <= i < 262144 for i in prefix+[a[different], b[different]]):
                raise ValueError('Bounded identical input prefix required')
            selected.append(dict(index=index, id=case['id'], category=category, only_winner=side,
                prompt_ids=case['prompt_tokens'], agreed_ids=a[:different], decision_index=different,
                original_id=a[different], native_id=b[different], prefix_tokens=len(prefix),
                prefix_sha256=hashlib.sha256(struct.pack('<'+'I'*len(prefix), *prefix)).hexdigest()))
    return selected


def inventory():
    result = {}
    for name, shape in geometry.shapes(geometry.E2B).items():
        key = 'language_model.model.'+name
        if len(shape) == 2:
            rows, columns = shape
            result[key] = dict(shape=[rows, columns//8], dtype='U32')
            for suffix in ['scales', 'biases']:
                result[key.removesuffix('weight')+suffix] = dict(shape=[rows, columns//64], dtype='BF16')
        else:
            result[key] = dict(shape=list(shape), dtype='BF16')
    return result


def statistics(raw, original_id, native_id, greedy, vocab=262144):
    if sys.byteorder != 'little' or type(raw) is not bytes or len(raw) != vocab*4 or not 2 <= vocab <= 262144:
        raise ValueError('Complete bounded little-endian FP32 row required')
    if original_id == native_id or any(type(i) is not int or not 0 <= i < vocab for i in [original_id, native_id, greedy]):
        raise ValueError('Distinct candidate IDs and valid stock greedy required')
    row = array.array('f')
    row.frombytes(raw)
    best = []
    for index, value in enumerate(row):
        if not math.isfinite(value):
            raise ValueError('Finite complete stock row required')
        best.append((value, -index))
        best.sort(reverse=True)
        del best[2:]
    if greedy != -best[0][1]:
        raise ValueError('Stock argmax differs from complete-row lowest-ID replay')
    return dict(greedy_id=greedy, top_two_ids=[-value[1] for value in best],
                top_two_margin=float(best[0][0])-float(best[1][0]),
                original_candidate_logit=float(row[original_id]), native_candidate_logit=float(row[native_id]),
                agrees_with='native' if greedy == native_id else 'original' if greedy == original_id else 'neither',
                row_sha256=hashlib.sha256(raw).hexdigest(), row_bytes=len(raw),
                row_encoding='little-endian FP32 exact widening of reported stock dtype')
