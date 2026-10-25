"""Extract the complete head witness from the accepted independent math trace."""
from pathlib import Path
import hashlib
import json
import os
import stat
import sys

SOURCES = {
    'benchmarks/results/2026-10-25-gemma4-text-math-v1.json': (None, 'c8691409336f5242076ad8859784dde7df8c1a5cdcc493f9b3c0bdcc8e974579'),
    'artifacts/m8-gemma4-text-real-v2/observation/observation.json': (1954149, 'ca86bd9836c88d5084f770b9ae84ecdb6e1e5a91d775e7cda315702450e918e9'),
    'artifacts/m8-gemma4-text-real-v2/observation/arrays.bin': (103403520, '352385699238b422b430b6f535c2eac99b6e90543becb4c46caa0c59f930ed93'),
    'tests/fixtures/gemma4-text-math-v1/math.json': (800566, 'f60c9f355b9008cfba6acd5d39c89c108de2d1ed5a3c8013a3a595576d8f910b'),
    'tools/reference/gemma4_text_math.py': (None, '6c04a81b93c95edfad4a4de646da3731b71a6dd175163fb856d2009bd103b096'),
}
SELECTED = [
    dict(role='input', name='final_norm', shape=[4, 1536], source_offset=94990336, bytes=24576,
         sha256='9c4e07465e765d1df255936e7f5a8e0f71b22b9ba1612d1fa6414501b731bfa9'),
    dict(role='expected', name='head', shape=[4, 262144], source_offset=95014912, bytes=4194304,
         sha256='dcd99568f15eb6c8a4c60728eef4aaa1873f4de37e3b8e9c3c7cb580911fd548'),
]

def identity(s):
    return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns

def fingerprint(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= 128 * 1024**2:
            raise ValueError('bounded regular source required')
        digest = hashlib.sha256()
        count = 0
        while True:
            raw = stream.read(min(65536, before.st_size - count + 1))
            if not raw:
                break
            count += len(raw)
            if count > before.st_size:
                raise ValueError('source grew')
            digest.update(raw)
        if count != before.st_size or identity(os.fstat(stream.fileno())) != identity(before) or identity(path.lstat()) != identity(before):
            raise ValueError('source changed or shortened')
        return dict(bytes=count, sha256=digest.hexdigest())

def load(path, cap):
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap:
        raise ValueError('bounded document required')
    return json.loads(raw)

def authenticate(root):
    result = {}
    for name, (size, wanted) in SOURCES.items():
        entry = fingerprint(root / name)
        if entry['sha256'] != wanted or size is not None and entry['bytes'] != size:
            raise ValueError('authenticated source differs: ' + name)
        result[name] = entry
    return result

def publish(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def capture(root, output):
    output.mkdir()
    report = dict(format='toklet-gemma-head-reference-v1', passed=False, contract='staged_fp32',
                  precision='Original independently captured reconstructed-FP32 weights, FP32 input and rounded FP32 head output',
                  tokens=[2, 100, 257, 100], positions=[0, 1, 2, 3], arrays=[], binary_bytes=0,
                  binary_cap_bytes=8 * 1024**2, sources={}, array_attempt=None, primary_error=None,
                  native_activation_parity=False, engine_executed=False, gpu_executed=False,
                  actual_full_head_qualified=False, source_or_precision_failures_waived=False)
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
                    raise ValueError('head binary cap exceeded')
                report['array_attempt'] = dict(role=item['role'], offset=item['offset'], planned_bytes=item['bytes'], written_bytes=0)
                source.seek(item['source_offset'])
                digest = hashlib.sha256()
                remaining = item['bytes']
                while remaining:
                    raw = source.read(min(65536, remaining))
                    if not raw:
                        raise ValueError('selected source is short')
                    if destination.write(raw) != len(raw):
                        raise OSError('short head array write')
                    destination.flush()
                    digest.update(raw)
                    remaining -= len(raw)
                    report['binary_bytes'] = destination.tell()
                    report['array_attempt']['written_bytes'] += len(raw)
                if digest.hexdigest() != item['sha256']:
                    raise ValueError('copied source differs')
                item['completed'] = True
                report['array_attempt'] = None
        if report['binary_bytes'] != 4218880 or authenticate(root) != report['sources']:
            raise ValueError('endpoint or complete head inventory differs')
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
        manifest = dict(format='toklet-gemma-head-reference-manifest-v1', passed=report['passed'], files={}, file_errors=[])
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
    return dict(passed=True, arrays=2, binary_bytes=4218880, engine_executed=False, gpu_executed=False)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('repository root and new output directory')
    print(json.dumps(capture(Path(sys.argv[1]), Path(sys.argv[2]))))
