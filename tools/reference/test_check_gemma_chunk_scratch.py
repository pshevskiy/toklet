"""Explicit semantic corruption controls for a completed tiny CPU transcript."""
from pathlib import Path
import argparse
import json
import check_gemma_chunk_scratch as checker


def controls(raw):
    checker.verify(raw); results = []
    for mode, error in [('value', 'independent chunk numerical gate failed'),
                        ('scratch', 'independent chunk scratch shape differs'),
                        ('missing_recovery', 'complete ordered chunk records required'),
                        ('recovery', 'record value differs'),
                        ('wrapper', 'exact successful chunk wrapper required'),
                        ('prefix', 'exact chunk test prefix required')]:
        lines = raw.decode().splitlines(); changed = False
        for index, line in enumerate(lines):
            if mode == 'wrapper' and line.startswith('test result: ok.'):
                lines[index] = line.replace('1 passed', '2 passed'); changed = True; break
            if 'GEMMA_CHUNK_SCRATCH_' not in line: continue
            before, body = line.split('GEMMA_CHUNK_SCRATCH_', 1); label, body = body.split(' ', 1); value = json.loads(body)
            if mode == 'prefix': before = 'test unrelated::passing_test ... '
            elif mode == 'value' and label == 'ROWS': value['values'][0] += 1.
            elif mode == 'scratch' and label == 'MEMORY':
                value['scratch_elements'][0] += 1
                for key in ('planned_session_bytes', 'actual_session_bytes', 'budget'): value[key] += 4
            elif mode == 'missing_recovery' and label == 'RECOVERY': lines.pop(index); changed = True; break
            elif mode == 'recovery' and label == 'RECOVERY': value['prior_prefix_preserved'] = False
            else: continue
            lines[index] = before + 'GEMMA_CHUNK_SCRATCH_' + label + ' ' + json.dumps(value, allow_nan=False)
            changed = True; break
        checker.fixture.require(changed, 'coherent chunk control mutation required')
        mutated = ('\n'.join(lines) + '\n').encode()
        try: checker.verify(mutated)
        except ValueError as actual:
            checker.fixture.same(str(actual), error)
            results.append(dict(control=mode, rejected=True, intended_error=error))
        else: raise ValueError('coherent chunk control accepted: ' + mode)
    return dict(passed=True, controls=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('capture', type=Path); args = parser.parse_args()
    print(json.dumps(controls(checker.fixture.read(args.capture, 1024**2))))
