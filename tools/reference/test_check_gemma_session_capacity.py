"""Saved-output controls for context ownership and independent KV/numerical replay."""
from pathlib import Path
import argparse
import json
import check_gemma_session_capacity as checker


def controls(raw, mode):
    checker.verify(raw, mode)
    cases = {
        'kv': [('producer', 'independent logical KV view differs'),
               ('head_order', 'independent logical KV view differs'),
               ('kv_word', 'independent logical KV view differs'),
               ('position', 'independent logical KV view differs'),
               ('segment_stride', 'independent logical KV view differs'),
               ('visible', 'independent logical KV view differs')],
        'raw': [('value', 'independent capacity numerical gate failed'),
                ('scratch', 'independent context scratch differs'),
                ('history', 'record value differs'),
                ('cache', 'record value differs')],
        'generation': [('sample', 'record value differs'),
                       ('pending', 'record value differs'),
                       ('history', 'record value differs')],
        'e2b': [('cache', 'record value differs'),
                ('qualification', 'record value differs')],
    }[mode] + [('wrapper', 'exact successful session-capacity wrapper required'),
              ('prefix', 'exact session-capacity test required')]
    results = []
    for mutation, error in cases:
        lines = raw.decode().splitlines()
        changed = False
        for index, line in enumerate(lines):
            if mutation == 'wrapper' and line.startswith('test result: ok.'):
                lines[index] = line.replace('1 passed', '2 passed')
                changed = True
                break
            if 'GEMMA_SESSION_CAPACITY_' not in line:
                continue
            before, body = line.split('GEMMA_SESSION_CAPACITY_', 1)
            label, body = body.split(' ', 1)
            item = json.loads(body)
            if mutation == 'prefix':
                before = 'test unrelated::passing_test ... '
            elif mode == 'kv' and label == 'VIEW':
                if mutation == 'producer': item['producer'] += 1
                elif mutation == 'position': item['end'] += 1
                elif mutation == 'kv_word': item['k_bits'][0] ^= 1
                elif mutation == 'segment_stride': item['segments'][2]['head_stride'] += 1
                elif mutation == 'visible': item['visible_ranges'][0][0] += 1
                elif mutation == 'head_order':
                    half = len(item['k_bits']) // 2
                    item['k_bits'] = item['k_bits'][half:] + item['k_bits'][:half]
                else: continue
            elif mode == 'raw' and label == 'ROWS' and mutation == 'value':
                item['values'][0] += 1.
            elif mode == 'raw' and label == 'MEMORY':
                if mutation == 'scratch': item['scratch_elements'][0] += 1
                elif mutation == 'history': item['history_bytes'] += 4
                elif mutation == 'cache': item['cache_payload_bytes'] += 4
                else: continue
                for key in ('planned_session_bytes', 'actual_session_bytes', 'budget'):
                    item[key] += 4
            elif mode == 'generation' and label == 'GENERATION':
                if mutation == 'sample' and item['capacity'] > 1:
                    item['sampled_token'] = (item['sampled_token'] + 1) % 17
                    item['pending_history'][-1] = item['sampled_token']
                elif mutation == 'pending':
                    item['evaluated_before_retry'] += 1
                    item['pending_before_retry'] -= 1
                elif mutation == 'history':
                    item['memory']['history_bytes'] += 4
                    item['memory']['admitted_bytes'] += 4
                else: continue
            elif mode == 'e2b' and label == 'E2B_ALLOCATION':
                if mutation == 'cache': item['cache_payload_bytes'] += 4
                elif mutation == 'qualification': item['whole_model_executed'] = True
                else: continue
            else:
                continue
            lines[index] = before + 'GEMMA_SESSION_CAPACITY_' + label + ' ' + json.dumps(item, allow_nan=False)
            changed = True
            break
        checker.packed.require(changed, 'coherent session-capacity mutation required')
        try:
            checker.verify(('\n'.join(lines) + '\n').encode(), mode)
        except ValueError as actual:
            checker.packed.same(str(actual), error)
            results.append(dict(control=mutation, rejected=True, intended_error=error))
        else:
            raise ValueError('session-capacity mutation accepted: ' + mutation)
    return dict(passed=True, controls=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=checker.TESTS)
    parser.add_argument('capture', type=Path)
    args = parser.parse_args()
    print(json.dumps(controls(checker.packed.read(args.capture, 16*1024**2), args.mode)))
