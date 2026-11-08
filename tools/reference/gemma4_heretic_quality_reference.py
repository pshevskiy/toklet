"""Independent selected logits and absolute producer K/V for quality-only wrap.

No engine, KV-ring API or quality scores. Each process authenticates exactly one
named original-representation source and runs both unchanged math contracts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import time
import bf16_source
import gemma4_heretic_bf16_weights as parent
import gemma4_heretic_reference as short
import gemma4_heretic_weights as mixed

ROOT = Path(__file__).resolve().parents[2]
MAX_CAPTURE = 256 * 1024**2
SOURCES = {'modified-bf16': parent, 'quantized-mlx': mixed}


def extras():
    return [*short.extras(), Path(__file__).resolve(), Path(parent.__file__).resolve()]


def plan(phase):
    phases = {'compat512': (512, [0, 31, 255, 511]),
              'full1025': (1025, [0, 31, 255, 511, 512, 767, 1023, 1024])}
    if phase not in phases:
        raise ValueError('Named frozen wrap phase required')
    raw = short.runtime_owner.runtime.captured(ROOT / 'benchmarks/gemma-e2b-wrap-v1/tokens.u32le',
        4100, '20e4789c70ba16a364591f621b274e3532f09ccb1859a51b227cc5b9c14b2503')
    corpus = short.runtime_owner.runtime.captured(ROOT / 'benchmarks/gemma-quality-v1/held_out_loss.u32le',
        40004, '877cbe18574b47e4b6a8d1b58fb79621c37e0531bd1f9504e50ce63d186f8c46')
    if raw != corpus[:4100]:
        raise ValueError('Frozen wrap/quality corpus prefix differs')
    count, positions = phases[phase]
    ids = list(struct.unpack('<1025I', raw))[:count]
    expected = (len(positions) * 262144 + 2 * len(ids) * 4608) * 12
    if expected > MAX_CAPTURE:
        raise ValueError('Quality reference capture exceeds fixed bound')
    return ids, positions, expected


def collect(output, directory, source, phase, admission, endpoint):
    import numpy as np
    if np.__version__ != '2.3.2' or source not in SOURCES:
        raise ValueError('Pinned numerical runtime and named source required')
    adapter = SOURCES[source]
    ids, positions, planned = plan(phase)
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Fresh artifact output required')
    output.mkdir()
    record = dict(format='toklet-heretic-quality-reference-v1', completed=False, source_kind=source,
        repository=adapter.REPOSITORY, revision=adapter.REVISION, source_sha256=adapter.FILE_SHA256,
        phase=phase, tokens=ids, selected_positions=positions, planned_rows_bytes=planned,
        arrays=[], runtime_admission=admission, primary_error=None, endpoint_errors=[],
        gate=dict(atol=1e-5, rtol=1e-4), source=None, process_limit_bytes=6*1024**3,
        ledger_mib=short.context_math.LEDGER_MIB, contract_times={}, quality_scored=False,
        managed_adoption_qualified=False, engine_executed=False)
    owner = None
    try:
        if hashlib.sha256(Path(short.graph.__file__).read_bytes()).hexdigest() != short.context_math.ORIGINAL_SOURCE_SHA256:
            raise ValueError('Frozen mathematical implementation changed')
        if sum(record['ledger_mib'].values()) * 1024**2 + MAX_CAPTURE > record['process_limit_bytes']:
            raise ValueError('Reference memory ledger exceeds6GiB')
        config_bytes = parent.CONFIG_BYTES if source == 'modified-bf16' else 6035
        config = short.runtime_owner.runtime.captured(directory / 'config.json', config_bytes, adapter.CONFIG_SHA256)
        deadline = time.monotonic() + 10800
        with bf16_source.Source(directory / 'model.safetensors', adapter.FILE_BYTES,
                                adapter.FILE_SHA256, deadline=deadline) as owner:
            weights = adapter.weights(owner, config, deadline)
            with (output / 'rows.bin').open('xb') as binary:
                for contract, staged in [('math_fp64', False), ('staged_fp32', True)]:
                    started = time.monotonic()
                    seen = set()
                    print(json.dumps(dict(event='contract-start', contract=contract, source=source)), flush=True)

                    def write(value, **metadata):
                        if value.dtype != np.dtype('<f4' if staged else '<f8') or not np.isfinite(value).all():
                            raise ValueError('Reference state representation differs')
                        raw = value.tobytes(order='C')
                        if binary.tell() + len(raw) > planned:
                            raise ValueError('Reference capture exceeds plan')
                        record['arrays'].append(dict(contract=contract, **metadata, offset=binary.tell(),
                            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
                        if binary.write(raw) != len(raw):
                            raise OSError('Short reference capture write')

                    def capture(name, value):
                        if name.startswith('layer.') and name.endswith('.output'):
                            print(json.dumps(dict(event='layer', contract=contract,
                                layer=int(name.split('.')[1]), elapsed_seconds=time.monotonic()-started)), flush=True)
                        if name == 'logits':
                            if name in seen or value.shape != (1, len(positions), 262144):
                                raise ValueError('Selected reference logits differ')
                            for row, position in enumerate(positions):
                                write(value[0, row], kind='logits', position=position)
                            seen.add(name)
                            return
                        match = re.fullmatch(r'layer\.(\d+)\.(k_rotary|self_attn\.v_norm)', name)
                        if match is None:
                            return
                        index = int(match[1])
                        kind = 'k' if match[2] == 'k_rotary' else 'v'
                        key = (index, kind)
                        if not 0 <= index < 15 or key in seen:
                            raise ValueError('Reference producer inventory differs')
                        dim = 512 if index % 5 == 4 else 256
                        shape = (1, 1, len(ids), dim) if kind == 'k' else (1, len(ids), 1, dim)
                        if value.shape != shape:
                            raise ValueError('Reference producer shape differs')
                        write(value.reshape(len(ids), dim), kind=kind, owner=index, tokens=len(ids),
                              head_dim=dim, retention='global' if index % 5 == 4 else 'local512')
                        seen.add(key)

                    short.context_math.evaluate(weights, ids, staged, selected_positions=positions, sink=capture)
                    if seen != {'logits'} | {(i, k) for i in range(15) for k in ('k', 'v')}:
                        raise ValueError('Reference capture incomplete')
                    record['contract_times'][contract] = time.monotonic() - started
                if binary.tell() != planned:
                    raise ValueError('Reference capture byte count differs')
            digest = hashlib.sha256()
            for offset in range(0, owner.size, bf16_source.HASH_BLOCK):
                digest.update(owner.read(offset, min(bf16_source.HASH_BLOCK, owner.size-offset), deadline=deadline))
            if digest.hexdigest() != adapter.FILE_SHA256:
                raise ValueError('Source endpoint bytes differ')
            record['source_endpoint_sha256'] = digest.hexdigest()
        with (output / 'rows.bin').open('rb') as file:
            record['rows_sha256'] = hashlib.file_digest(file, 'sha256').hexdigest()
        record['completed'] = True
    except BaseException as error:
        record['primary_error'] = type(error).__name__ + ': ' + str(error)[:2048]
    finally:
        if owner is not None:
            record['source'] = owner.audit()
        try:
            record['runtime_endpoint'] = endpoint()
        except BaseException as error:
            record['endpoint_errors'].append(type(error).__name__ + ': ' + str(error)[:2048])
        record['completed'] = record['completed'] and record['primary_error'] is None and not record['endpoint_errors']
        (output / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-directory', type=Path, required=True)
    parser.add_argument('--source', choices=SOURCES, required=True)
    parser.add_argument('--phase', choices=['compat512', 'full1025'], required=True)
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--native-manifest', type=Path, required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    args = parser.parse_args()
    admission, guard = short.runtime_owner.runtime_context(args.runtime_manifest, args.expected_runtime_sha256,
        args.native_manifest, args.expected_native_sha256, extra_sources=extras())
    result = collect(args.output, args.model_directory, args.source, args.phase, admission, guard)
    print(json.dumps(dict(completed=result['completed'], error=result['primary_error'])))
    raise SystemExit(0 if result['completed'] else 1)
