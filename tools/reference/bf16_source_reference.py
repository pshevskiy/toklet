"""Cold exact-bit BF16 row-reader witness; no model graph or quality scoring."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import signal
import sys
import threading
import time
from unittest.mock import patch
import affine4_tiles
import bf16_source
import check_gemma_hybrid_tiny as safe
import reference_runtime as runtime
import text_reference_runtime as text_runtime

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests/fixtures/bf16-source-v1'


def sources():
    # Preserve existing runtime contracts. The union adds the complete Python
    # closure to the NumPy/MLX inventory; MLX/text packages need not execute.
    return text_runtime.paths([Path(__file__).resolve(), Path(bf16_source.__file__).resolve(),
                               Path(affine4_tiles.__file__).resolve(), Path(safe.__file__).resolve()])


def endpoint(manifest, selected):
    safe.same(runtime.runtime_paths(sources()), selected)
    for path in selected:
        pin = manifest['files'][str(path)]
        raw = safe.read(path, pin['bytes'])
        safe.same(len(raw), pin['bytes'])
        safe.same(hashlib.sha256(raw).hexdigest(), pin['sha256'])
    origins = {}
    for name, module in list(sys.modules.items()):
        origin = getattr(module, '__file__', None)
        if origin:
            path = str(Path(origin).resolve())
            safe.require(path in manifest['files'], 'unadmitted actual module: ' + name)
            origins[name] = path
    library = ctypes.CDLL(None)
    count = library._dyld_image_count; count.argtypes = []; count.restype = ctypes.c_uint32
    image = library._dyld_get_image_name; image.argtypes = [ctypes.c_uint32]; image.restype = ctypes.c_char_p
    non_system = []
    for index in range(count()):
        path = str(Path(image(index).decode()).resolve())
        if not path.startswith(('/System/', '/usr/lib/')):
            safe.require(path in manifest['files'], 'unadmitted actual native image: ' + path)
            non_system.append(path)
    return dict(modules=origins, non_system_native_images=sorted(non_system))


def tiny(output, observation):
    import numpy as np
    safe.same(np.__version__, '2.3.2')
    raw = safe.read(FIXTURE / 'source.bin', 40)
    expected = safe.read(FIXTURE / 'finite.f32le', 48)
    safe.same(hashlib.sha256(raw).hexdigest(), 'cd1f0d6dd1061d41d1d6f53109fb850962398bacc97332e00ae8cad91c1f6fc3')
    safe.same(hashlib.sha256(expected).hexdigest(), '8313445c8a666035c464caebf7e25d22fd66e8f52654fc7d583cbf04872a30a3')
    deadline = time.monotonic() + 60
    captures = observation['captures']
    def capture(name, values, oracle):
        body = values.tobytes(order='C')
        record = dict(name=name, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                      expected_sha256=hashlib.sha256(oracle).hexdigest(),
                      shape=list(values.shape), dtype=values.dtype.str, captured=False, compared=False)
        captures.append(record)
        with (output / (name + '.f32le')).open('xb') as file:
            safe.same(file.write(body), len(body)); file.flush()
        record['captured'] = True
        safe.same(record['dtype'], '<f4')
        safe.same(record['shape'], {'full': [4, 3], 'first': [2, 3], 'second': [2, 3],
                                    'gather': [4, 3], 'scalar': [1, 1]}[name])
        safe.same(body, oracle); record['compared'] = True
        observation['compared_values'] += len(body) // 4
    with bf16_source.Source(FIXTURE / 'source.bin', 40, hashlib.sha256(raw).hexdigest(), deadline=deadline) as owner:
        matrix = bf16_source.Rows(owner, 4, 3, 8, deadline)
        capture('full', matrix.read_rows(0, 4), expected)
        capture('first', matrix.read_rows(0, 2), expected[:24])
        capture('second', matrix.read_rows(2, 4), expected[24:])
        capture('gather', matrix.gather([3, 0, 3, 1]), expected[36:48] + expected[:12] + expected[36:48] + expected[12:24])
        capture('scalar', bf16_source.Rows(owner, 1, 1, 32, deadline).read_rows(0, 1), b'\0\0\x80\x3f')
    audit = owner.audit()
    observation['nonfinite_controls'] = []
    for index, word in enumerate([0x7f80, 0xff80, 0x7fc1]):
        control = dict(word_bits=word, rejected=False, poisoned=False)
        observation['nonfinite_controls'].append(control)
        with bf16_source.Source(FIXTURE / 'source.bin', 40, hashlib.sha256(raw).hexdigest(), deadline=deadline) as owner:
            bad = bf16_source.Rows(owner, 1, 1, 34 + index * 2, deadline)
            try:
                bad.read_rows(0, 1)
            except ValueError as error:
                safe.same(str(error), 'Nonfinite BF16 metadata')
                safe.require(owner.poisoned, 'nonfinite source must poison')
                control.update(rejected=True, poisoned=True)
            else:
                raise ValueError('nonfinite BF16 accepted')
    stop = threading.Event()
    with bf16_source.Source(FIXTURE / 'source.bin', 40, hashlib.sha256(raw).hexdigest(), deadline=deadline) as owner:
        rows = bf16_source.Rows(owner, 4, 3, 8, deadline, stop)
        original = affine4_tiles.bf16
        def cancelled(bits):
            result = original(bits); stop.set(); return result
        with patch.object(affine4_tiles, 'bf16', side_effect=cancelled):
            try:
                rows.read_rows(0, 1)
            except InterruptedError:
                safe.require(not owner.poisoned, 'request cancellation must not poison immutable weights')
            else:
                raise ValueError('conversion ignored cancellation')
    safe.same(observation['compared_values'], 37)
    observation.update(passed=True, raw_read_audit=audit, nonfinite_rejected=True,
                       conversion_cancellation_rejected=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--runtime-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selected = runtime.runtime_paths(sources())
    admission = runtime.admit_runtime(args.runtime_manifest, args.runtime_sha, sources())
    manifest = safe.document(safe.read(args.runtime_manifest, 4 * 1024**2))
    result = dict(passed=False, primary_error=None, endpoint_errors=[], admission=admission)
    def stop(number, _):
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(str(number))
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        args.output.mkdir()
        result['observation'] = dict(passed=False, captures=[], compared_values=0,
                                     official_checkpoint=False, mlx_executed=False,
                                     inference_executed=False, quality_scored=False, model_fit_qualified=False)
        tiny(args.output, result['observation']); result['passed'] = True
    except BaseException as error:
        result['primary_error'] = type(error).__name__ + ': ' + str(error)
    finally:
        for sig in old: signal.signal(sig, signal.SIG_IGN)
        try: result['runtime_endpoint'] = endpoint(manifest, selected)
        except BaseException as error:
            result['endpoint_errors'].append(type(error).__name__ + ': ' + str(error))
            result['passed'] = False
        try:
            with (args.output.parent / (args.output.name + '-result.json')).open('x') as file:
                json.dump(result, file, indent=2); file.write('\n')
        finally:
            for sig, handler in old.items(): signal.signal(sig, handler)
    print(json.dumps({key: result.get(key) for key in ['passed', 'primary_error', 'endpoint_errors', 'observation']}))
    sys.exit(0 if result['passed'] and not result['endpoint_errors'] else 1)
