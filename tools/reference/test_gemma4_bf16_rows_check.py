"""Explicit saved-target controls; no source checkpoint access.

Use --target only after a completed real cold worker. The real source files
are never opened or changed by these saved-output controls.
"""
from pathlib import Path
import argparse
import copy
import json
import shutil
import struct
import tempfile
import signal
import check_gemma4_bf16_rows as checker


def controls(oracle, plan, plan_sha, target, report, failure_root):
    raw = checker.oracle.safe.read(target / 'arrays.bin', 64 * 1024**2)
    original = checker.oracle.safe.document(checker.oracle.safe.read(target.parent / (target.name + '-result.json'), 4 * 1024**2))
    for mode, expected_error in [('row', 'target row bits differ'), ('dtype', 'target array metadata differs'),
                                 ('audit', 'ordered target read audit differs'), ('projection', 'saved target projection differs'),
                                 ('nonfinite', 'nonfinite target projection')]:
        result = copy.deepcopy(original); changed = bytearray(raw)
        if mode == 'row':
            capture = result['observation']['captures'][0]; changed[capture['offset']] ^= 1
        elif mode == 'dtype': result['observation']['captures'][0]['dtype'] = '<f8'
        elif mode == 'audit': result['observation']['source_audit']['reads'] += 1
        else:
            capture = next(c for c in result['observation']['captures'] if 'comparisons' in c)
            offset = capture['offset']; value = struct.unpack_from('<f', changed, offset)[0]
            struct.pack_into('<f', changed, offset, float('nan') if mode == 'nonfinite' else value + 1.)
        if mode in ('row', 'projection', 'nonfinite'):
            start = capture['offset']; capture['sha256'] = checker.oracle.digest(changed[start:start + capture['bytes']])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); output = root / 'target'; output.mkdir()
            (output / 'arrays.bin').write_bytes(changed)
            body = checker.encoded(result); (root / 'target-result.json').write_bytes(body)
            attempt = dict(control=mode, expected_error=expected_error, actual_error=None, rejected=False,
                           coherent_result_sha256=checker.oracle.digest(body), coherent_arrays_sha256=checker.oracle.digest(changed))
            report['controls'].append(attempt)
            try:
                checker.replay(oracle, plan, plan_sha, output)
            except ValueError as error:
                attempt['actual_error'] = str(error)
                if str(error) == expected_error:
                    attempt['rejected'] = True
                    continue
                primary = error
            except BaseException as error:
                primary = error
            else:
                primary = ValueError('saved-target corruption accepted: ' + mode)
            try:
                failure_root.mkdir(exist_ok=True); shutil.copytree(root, failure_root / mode)
            except BaseException as error: primary.add_note('failed control capture: ' + str(error))
            raise primary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--target', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = dict(passed=False, primary_error=None, controls=[])
    def stop(number, _):
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(str(number))
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        controls(args.oracle, args.plan, args.plan_sha, args.target, result, args.output.parent / (args.output.stem + '-failures'))
        result['passed'] = True
    except BaseException as error:
        result['primary_error'] = type(error).__name__ + ': ' + str(error)
        result['cleanup_notes'] = getattr(error, '__notes__', [])
    finally:
        for sig in old: signal.signal(sig, signal.SIG_IGN)
        try:
            with args.output.open('x') as file: json.dump(result, file, indent=2, allow_nan=False); file.write('\n')
        finally:
            for sig, handler in old.items(): signal.signal(sig, handler)
    print(json.dumps(result)); raise SystemExit(0 if result['passed'] else 1)
