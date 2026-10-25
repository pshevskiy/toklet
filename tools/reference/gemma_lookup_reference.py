"""Extract complete real embedding/PLE rows before native table qualification."""
from pathlib import Path
import hashlib
import json
import os
import sys
from gemma_head_reference import authenticate, fingerprint, load, publish

SELECTED = [
    dict(role='embedding', name='embedding', shape=[4, 1536], source_offset=68935680, bytes=24576,
         sha256='0c7d6fa084231e7c3a80d742dbe6a0814bd6b129a48464b0ce9763c88c2fb811'),
    dict(role='ple_lookup', name='ple_lookup', shape=[4, 8960], source_offset=68984832, bytes=143360,
         sha256='016b59789c1ad4c2d728c89d47b061c09da0cde3ae5685129b8bd71c20eb8056'),
]

def capture(root, output):
    output.mkdir()
    report = dict(format='toklet-gemma-lookup-reference-v1', passed=False, contract='staged_fp32',
                  precision='Original independent staged-FP32 embedding and per-layer lookup rows from reconstructed-FP32 packed weights',
                  tokens=[2, 100, 257, 100], positions=[0, 1, 2, 3], arrays=[], binary_bytes=0,
                  binary_cap_bytes=256 * 1024, sources={}, array_attempt=None, primary_error=None,
                  native_activation_parity=False, engine_executed=False, gpu_executed=False,
                  actual_full_tables_qualified=False, source_or_precision_failures_waived=False)
    primary = None
    try:
        report['sources'] = authenticate(root)
        source_report = load(root / 'artifacts/m8-gemma4-text-real-v2/observation/observation.json', 4 * 1024**2)
        if source_report['passed'] is not True or source_report['tokens'] != report['tokens'] or source_report['cross_precision_failed_arrays'] != 5 or source_report['cross_precision_failed_values'] != 57:
            raise ValueError('acknowledged complete mathematical trace required')
        with (root / 'artifacts/m8-gemma4-text-real-v2/observation/arrays.bin').open('rb') as source, (output / 'arrays.bin').open('xb') as destination:
            for selected in SELECTED:
                matches = [a for a in source_report['arrays'] if a['contract'] == 'staged_fp32' and a['name'] == selected['name']]
                if len(matches) != 1:
                    raise ValueError('unique selected source required')
                entry = matches[0]
                wanted = dict(shape=[1] + selected['shape'], dtype='<f4', offset=selected['source_offset'], bytes=selected['bytes'], sha256=selected['sha256'], completed=True)
                if any(type(entry[key]) is not type(value) or entry[key] != value for key, value in wanted.items()):
                    raise ValueError('selected source descriptor differs')
                item = dict(selected, dtype='F32', offset=destination.tell(), completed=False)
                report['arrays'].append(item)
                if item['offset'] + item['bytes'] > report['binary_cap_bytes']:
                    raise ValueError('lookup binary cap exceeded')
                report['array_attempt'] = dict(role=item['role'], offset=item['offset'], planned_bytes=item['bytes'], written_bytes=0)
                source.seek(item['source_offset'])
                digest = hashlib.sha256()
                remaining = item['bytes']
                while remaining:
                    raw = source.read(min(65536, remaining))
                    if not raw:
                        raise ValueError('selected source is short')
                    if destination.write(raw) != len(raw):
                        raise OSError('short lookup array write')
                    destination.flush()
                    digest.update(raw)
                    remaining -= len(raw)
                    report['binary_bytes'] = destination.tell()
                    report['array_attempt']['written_bytes'] += len(raw)
                if digest.hexdigest() != item['sha256']:
                    raise ValueError('copied source differs')
                item['completed'] = True
                report['array_attempt'] = None
        if report['binary_bytes'] != 167936 or authenticate(root) != report['sources']:
            raise ValueError('endpoint or complete lookup inventory differs')
        report['passed'] = True
    except BaseException as error:
        primary = error
        report['passed'] = False
        report['primary_error'] = type(error).__name__ + ': ' + str(error)
    finally:
        try:
            publish(output / 'observation.json', report)
        except BaseException as error:
            if primary is None:
                primary = error
                report['passed'] = False
                report['primary_error'] = type(error).__name__ + ': ' + str(error)
            else:
                primary.add_note('Report publication failed: ' + str(error))
        manifest = dict(format='toklet-gemma-lookup-reference-manifest-v1', passed=report['passed'], files={}, file_errors=[])
        for name in ('observation.json', 'arrays.bin'):
            if not os.path.lexists(output / name):
                continue
            try:
                manifest['files'][name] = fingerprint(output / name)
            except BaseException as error:
                manifest['file_errors'].append(name + ': ' + str(error))
                manifest['passed'] = False
                if primary is None:
                    primary = error
                else:
                    primary.add_note('Partial file retention failed: ' + str(error))
        try:
            publish(output / 'manifest.json', manifest)
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note('Manifest publication failed: ' + str(error))
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    return dict(passed=True, arrays=2, binary_bytes=167936, engine_executed=False, gpu_executed=False)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('repository root and new output directory')
    print(json.dumps(capture(Path(sys.argv[1]), Path(sys.argv[2]))))
