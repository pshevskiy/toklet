"""Independent complete-byte replay of the pre-native full-head extraction."""
from pathlib import Path
import hashlib
import json
import math
import os
import stat
import struct
import sys


def same(actual, expected):
    if type(actual) is not type(expected):
        raise ValueError('typed field differs')
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError('record fields differ')
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('record length differs')
        for a, b in zip(actual, expected, strict=True):
            same(a, b)
    elif actual != expected:
        raise ValueError('record value differs')


def read(path, cap):
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap:
        raise ValueError('bounded file required')
    return raw


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('finite JSON required')))


def identity(path):
    def state(s):
        return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        original = os.fstat(stream.fileno())
        if not stat.S_ISREG(original.st_mode) or not 0 <= original.st_size <= 128 * 1024**2:
            raise ValueError('bounded regular source required')
        total = 0
        digest = hashlib.sha256()
        while total <= original.st_size:
            block = stream.read(min(65536, original.st_size + 1 - total))
            if not block:
                break
            total += len(block)
            digest.update(block)
        if total != original.st_size or state(original) != state(os.fstat(stream.fileno())) or state(original) != state(path.lstat()):
            raise ValueError('complete stable source required')
        return dict(bytes=total, sha256=digest.hexdigest())


def sources(root):
    pins = [
        ('benchmarks/results/2026-10-25-gemma4-text-math-v1.json', None, 'c8691409336f5242076ad8859784dde7df8c1a5cdcc493f9b3c0bdcc8e974579'),
        ('artifacts/m8-gemma4-text-real-v2/observation/observation.json', 1954149, 'ca86bd9836c88d5084f770b9ae84ecdb6e1e5a91d775e7cda315702450e918e9'),
        ('artifacts/m8-gemma4-text-real-v2/observation/arrays.bin', 103403520, '352385699238b422b430b6f535c2eac99b6e90543becb4c46caa0c59f930ed93'),
        ('tests/fixtures/gemma4-text-math-v1/math.json', 800566, 'f60c9f355b9008cfba6acd5d39c89c108de2d1ed5a3c8013a3a595576d8f910b'),
        ('tools/reference/gemma4_text_math.py', None, '6c04a81b93c95edfad4a4de646da3731b71a6dd175163fb856d2009bd103b096'),
    ]
    result = {}
    for name, size, sha in pins:
        actual = identity(root / name)
        if actual['sha256'] != sha or size is not None and actual['bytes'] != size:
            raise ValueError('original source identity differs')
        result[name] = actual
    return result


def verify(root, directory):
    original = sources(root)
    report_raw = read(directory / 'observation.json', 65536)
    binary = read(directory / 'arrays.bin', 8 * 1024**2)
    report = parse(report_raw)
    manifest = parse(read(directory / 'manifest.json', 65536))
    same(manifest, dict(format='toklet-gemma-head-reference-manifest-v1', passed=True,
                        files={name: dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest()) for name, body in
                               [('observation.json', report_raw), ('arrays.bin', binary)]}, file_errors=[]))
    descriptors = []
    offset = 0
    trace = parse(read(root / 'artifacts/m8-gemma4-text-real-v2/observation/observation.json', 4 * 1024**2))
    compact = parse(read(root / 'tests/fixtures/gemma4-text-math-v1/math.json', 2 * 1024**2))
    with (root / 'artifacts/m8-gemma4-text-real-v2/observation/arrays.bin').open('rb') as stream:
        for role, name, width, start, count, sha in [
            ('input', 'final_norm', 1536, 94990336, 24576, '9c4e07465e765d1df255936e7f5a8e0f71b22b9ba1612d1fa6414501b731bfa9'),
            ('expected', 'head', 262144, 95014912, 4194304, 'dcd99568f15eb6c8a4c60728eef4aaa1873f4de37e3b8e9c3c7cb580911fd548')]:
            candidates = [v for v in trace['arrays'] if v['name'] == name and v['contract'] == 'staged_fp32']
            witnesses = [v for v in compact['arrays'] if v['name'] == name and v['contract'] == 'staged_fp32']
            if len(candidates) != 1 or len(witnesses) != 1:
                raise ValueError('unique original and compact array required')
            entry, witness = candidates[0], witnesses[0]
            for key, value in dict(shape=[1, 4, width], dtype='<f4', offset=start, bytes=count, sha256=sha, completed=True).items():
                same(entry[key], value)
            same(witness['source_shape'], [1, 4, width])
            same(witness['source_array_sha256'], sha)
            stream.seek(start)
            remaining = count
            position = offset
            digest = hashlib.sha256()
            while remaining:
                block = stream.read(min(65536, remaining))
                if not block or binary[position:position + len(block)] != block:
                    raise ValueError('complete copied array differs')
                if any(not math.isfinite(v) for (v,) in struct.iter_unpack('<f', block)):
                    raise ValueError('finite complete F32 array required')
                digest.update(block)
                position += len(block)
                remaining -= len(block)
            if digest.hexdigest() != sha:
                raise ValueError('original selected array digest differs')
            descriptors.append(dict(role=role, name=name, shape=[4, width], source_offset=start, bytes=count,
                                    sha256=sha, dtype='F32', offset=offset, completed=True))
            offset += count
    same(report, dict(format='toklet-gemma-head-reference-v1', passed=True, contract='staged_fp32',
                      precision='Original independently captured reconstructed-FP32 weights, FP32 input and rounded FP32 head output',
                      tokens=[2, 100, 257, 100], positions=[0, 1, 2, 3], arrays=descriptors, binary_bytes=4218880,
                      binary_cap_bytes=8 * 1024**2, sources=original, array_attempt=None, primary_error=None,
                      native_activation_parity=False, engine_executed=False, gpu_executed=False,
                      actual_full_head_qualified=False, source_or_precision_failures_waived=False))
    if len(binary) != offset or sources(root) != original:
        raise ValueError('endpoint or full binary length differs')
    return dict(passed=True, arrays=2, input_values=6144, output_values=1048576, binary_bytes=4218880,
                numerical_reference_reexecuted=False, independent_extraction_replay=True,
                native_activation_parity=False, engine_executed=False, gpu_executed=False)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('repository root and frozen head directory')
    print(json.dumps(verify(Path(sys.argv[1]), Path(sys.argv[2]))))
