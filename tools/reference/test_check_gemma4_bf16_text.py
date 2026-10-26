"""Explicit coherent graph-capture controls, without changing source weights."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import shutil
import signal
import struct
import tempfile
import check_gemma4_bf16_text as checker


def controls(capture, plan, report, failures):
    original = checker.source.safe.document(checker.source.safe.read(capture / 'observation.json', 4 * 1024**2))
    cases = [('order', 'ordered graph state differs'), ('shape', 'ordered graph state differs'),
             ('audit', 'complete graph read audit differs'), ('prediction', 'graph predictions differ'),
             ('nonfinite', 'nonfinite saved graph state'), ('prefix', 'future graph changed causal prefix'),
             ('boundary', 'independent graph boundary differs: original/math_fp64/embedding')]
    for mode, message in cases:
        result = copy.deepcopy(original)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); output = root / 'capture'; output.mkdir()
            for name in checker.TOKENS: shutil.copyfile(capture / (name + '.bin'), output / (name + '.bin'))
            if mode == 'order': result['arrays'][0], result['arrays'][1] = result['arrays'][1], result['arrays'][0]
            elif mode == 'shape': result['arrays'][0]['shape'] = [1, 2, 3072]
            elif mode == 'audit': result['source_audit']['reads'] += 1
            elif mode == 'prediction': result['predictions']['original/math_fp64'][0]['token'] += 1
            else:
                sequences = ['original', 'future'] if mode == 'boundary' else ['future' if mode == 'prefix' else 'original']
                for sequence in sequences:
                    spec = next(s for s in result['arrays'] if s['sequence'] == sequence and s['contract'] == 'math_fp64' and s['name'] == 'embedding')
                    path = output / (sequence + '.bin')
                    with path.open('r+b') as file:
                        file.seek(spec['offset']); value = struct.unpack('<d', file.read(8))[0]
                        file.seek(spec['offset']); file.write(struct.pack('<d', float('inf') if mode == 'nonfinite' else value + 1.))
                        file.seek(spec['offset']); spec['sha256'] = hashlib.sha256(file.read(spec['bytes'])).hexdigest()
                    with path.open('rb') as file: result['files'][sequence]['sha256'] = hashlib.file_digest(file, 'sha256').hexdigest()
            body = checker.encoded(result); (output / 'observation.json').write_bytes(body)
            attempt = dict(control=mode, expected_error=message, actual_error=None, rejected=False,
                           coherent_manifest_sha256=checker.source.digest(body), coherent_files=result['files'])
            report['controls'].append(attempt)
            partial = dict(prefix=[], cross_precision=[], boundaries=[], boundary_sources=[])
            try: checker.replay(output, plan, attempt['coherent_manifest_sha256'], partial)
            except ValueError as error:
                attempt['actual_error'] = str(error)
                if str(error) == message:
                    attempt['rejected'] = True
                    continue
                primary = error
            except BaseException as error: primary = error
            else: primary = ValueError('coherent graph control accepted: ' + mode)
            attempt['partial_check'] = partial
            try: failures.mkdir(exist_ok=True); shutil.copytree(root, failures / mode)
            except BaseException as error: primary.add_note('graph control capture: ' + str(error))
            raise primary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    report = dict(passed=False, primary_error=None, controls=[])
    def stop(number, _):
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(str(number))
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        controls(args.capture, checker.load_plan(args.plan, args.plan_sha), report, args.output.parent / (args.output.stem + '-failures'))
        report['passed'] = True
    except BaseException as error:
        report['primary_error'] = type(error).__name__ + ': ' + str(error); report['cleanup_notes'] = getattr(error, '__notes__', [])
    finally:
        for sig in old: signal.signal(sig, signal.SIG_IGN)
        try:
            with args.output.open('x') as file: file.write(checker.encoded(report).decode())
        finally:
            for sig, handler in old.items(): signal.signal(sig, handler)
    print(json.dumps(dict(passed=report['passed'],primary_error=report['primary_error'],controls=len(report['controls']))))
    raise SystemExit(0 if report['passed'] else 1)
