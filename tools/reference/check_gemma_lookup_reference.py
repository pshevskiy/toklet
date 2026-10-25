"""Independent complete-byte replay of original embedding/PLE row extraction."""
from pathlib import Path
import hashlib
import json
import math
import struct
import sys
from check_gemma_head_reference import same, read, parse, sources, identity

def verify(root, directory):
    original = sources(root)
    report_raw = read(directory / 'observation.json', 65536)
    binary = read(directory / 'arrays.bin', 256 * 1024)
    report = parse(report_raw)
    manifest = parse(read(directory / 'manifest.json', 65536))
    same(manifest, dict(format='toklet-gemma-lookup-reference-manifest-v1', passed=True,
                        files={name: dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest()) for name, body in
                               [('observation.json', report_raw), ('arrays.bin', binary)]}, file_errors=[]))
    descriptors = []
    offset = 0
    trace = parse(read(root / 'artifacts/m8-gemma4-text-real-v2/observation/observation.json', 4 * 1024**2))
    compact = parse(read(root / 'tests/fixtures/gemma4-text-math-v1/math.json', 2 * 1024**2))
    with (root / 'artifacts/m8-gemma4-text-real-v2/observation/arrays.bin').open('rb') as stream:
        for role, name, width, start, count, sha in [
            ('embedding', 'embedding', 1536, 68935680, 24576, '0c7d6fa084231e7c3a80d742dbe6a0814bd6b129a48464b0ce9763c88c2fb811'),
            ('ple_lookup', 'ple_lookup', 8960, 68984832, 143360, '016b59789c1ad4c2d728c89d47b061c09da0cde3ae5685129b8bd71c20eb8056')]:
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
    same(report, dict(format='toklet-gemma-lookup-reference-v1', passed=True, contract='staged_fp32',
                      precision='Original independent staged-FP32 embedding and per-layer lookup rows from reconstructed-FP32 packed weights',
                      tokens=[2, 100, 257, 100], positions=[0, 1, 2, 3], arrays=descriptors, binary_bytes=167936,
                      binary_cap_bytes=256 * 1024, sources=original, array_attempt=None, primary_error=None,
                      native_activation_parity=False, engine_executed=False, gpu_executed=False,
                      actual_full_tables_qualified=False, source_or_precision_failures_waived=False))
    if len(binary) != offset or sources(root) != original:
        raise ValueError('endpoint or full binary length differs')
    return dict(passed=True, arrays=2, embedding_values=6144, per_layer_values=35840, binary_bytes=167936,
                numerical_reference_reexecuted=False, independent_extraction_replay=True,
                native_activation_parity=False, engine_executed=False, gpu_executed=False)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('repository root and frozen lookup directory')
    print(json.dumps(verify(Path(sys.argv[1]), Path(sys.argv[2]))))
