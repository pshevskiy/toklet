"""Independent full-sequence E2B oracle for real local-KV wrap qualification."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import time
import gemma4_e2b_context_reference as prior

ROOT = prior.ROOT
CONTRACT = ROOT / 'benchmarks/gemma-e2b-wrap-v1'
graph = prior.graph
source_graph = prior.source_graph


def inputs(phase):
    manifest = json.loads((CONTRACT / 'manifest.json').read_text())
    raw = (CONTRACT / 'tokens.u32le').read_bytes()
    parent = manifest['parent_tokens']
    body = (ROOT / parent['path']).read_bytes()
    if (len(raw) != 1025 * 4 or raw != body[:len(raw)]
            or hashlib.sha256(raw).hexdigest() != manifest['tokens']['sha256']
            or hashlib.sha256(body).hexdigest() != parent['sha256']):
        raise ValueError('Frozen wrap corpus prefix differs')
    spec = manifest['phases'][phase]
    expected = {'compat512': (512, [0, 31, 255, 511]),
                'full1025': (1025, [0, 31, 255, 511, 512, 767, 1023, 1024])}[phase]
    if (spec['tokens'], spec['positions']) != expected:
        raise ValueError('Frozen wrap phase differs')
    return list(struct.unpack('<1025I', raw))[:spec['tokens']], spec['positions']


def collect(output, phase, admission, guard):
    import numpy as np
    ids, positions = inputs(phase)
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Fresh artifact output required')
    output.mkdir()
    arrays = []
    # All absolute K/V rows once per physical producer; no ring/session API.
    planned = (len(positions) * 262144 + 2 * len(ids) * 4608) * 12
    record = dict(completed=False, phase=phase, tokens=ids,
                  selected_positions=positions, planned_rows_bytes=planned,
                  source_pins=source_graph.source.PINS, runtime_admission=admission,
                  ledger_mib=graph.LEDGER_MIB,
                  frozen_math_source_sha256=graph.ORIGINAL_SOURCE_SHA256,
                  input_contract_sha256=source_graph.source.sha(CONTRACT / 'manifest.json'),
                  contract_times={}, prior_precision_failures_waived=False)
    try:
        if source_graph.source.sha(Path(graph.frozen.__file__)) != graph.ORIGINAL_SOURCE_SHA256:
            raise ValueError('Historical mathematical source differs')
        if sum(graph.LEDGER_MIB.values()) * 1024**2 > 6 * 1024**3:
            raise ValueError('Reference allowance exceeds process policy')
        for name, sha in source_graph.source.PINS.items():
            if source_graph.source.sha(source_graph.source.MODEL / name) != sha:
                raise ValueError('Original source identity differs')
        with (source_graph.source.MODEL / 'model.safetensors').open('rb') as stream, (output / 'rows.bin').open('xb') as binary:
            weights = source_graph.weights(stream)
            for contract, staged in [('math_fp64', False), ('staged_fp32', True)]:
                start = time.perf_counter_ns()
                seen = set()
                print(json.dumps(dict(event='contract-start', phase=phase,
                                      contract=contract, tokens=len(ids))), flush=True)

                def write(value, **metadata):
                    if not np.isfinite(value).all():
                        raise ValueError('Nonfinite selected reference state')
                    raw = np.asarray(value, dtype='<f4' if staged else '<f8').tobytes()
                    arrays.append(dict(contract=contract, **metadata, offset=binary.tell(),
                                       bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
                    binary.write(raw)

                def capture(name, value):
                    if name == 'logits':
                        if value.shape != (1, len(positions), 262144) or 'logits' in seen:
                            raise ValueError('Selected reference logits differ')
                        for row, position in enumerate(positions):
                            write(value[0, row], kind='logits', position=position)
                        seen.add('logits')
                        return
                    match = re.fullmatch(r'layer\.(\d+)\.(k_rotary|self_attn\.v_norm)', name)
                    if match is None:
                        return
                    owner = int(match[1])
                    kind = 'k' if match[2] == 'k_rotary' else 'v'
                    key = (owner, kind)
                    if not 0 <= owner < 15 or key in seen:
                        raise ValueError('Physical producer inventory differs')
                    dim = 512 if owner % 5 == 4 else 256
                    shape = (1, 1, len(ids), dim) if kind == 'k' else (1, len(ids), 1, dim)
                    if value.shape != shape:
                        raise ValueError('Physical producer shape differs')
                    write(value.reshape(len(ids), dim), kind=kind, owner=owner,
                          tokens=len(ids), head_dim=dim,
                          retention='global' if owner % 5 == 4 else 'local512')
                    seen.add(key)

                graph.evaluate(weights, ids, staged, selected_positions=positions, sink=capture)
                if seen != {'logits'} | {(owner, kind) for owner in range(15) for kind in ['k', 'v']}:
                    raise ValueError('Missing physical producer or selected logits')
                record['contract_times'][contract] = (time.perf_counter_ns() - start) / 1e9
                print(json.dumps(dict(event='contract-complete', phase=phase,
                                      contract=contract, seconds=record['contract_times'][contract])), flush=True)
            if binary.tell() != planned:
                raise ValueError('Selected reference output size differs')
        record.update(completed=True, arrays=arrays,
                      rows_sha256=source_graph.source.sha(output / 'rows.bin'))
    except BaseException as error:
        record['failure'] = dict(type=type(error).__name__, message=str(error)[:2048])
    finally:
        try:
            record['runtime_endpoint'] = guard()
        except BaseException as error:
            record.update(completed=False, endpoint_failure=str(error)[:2048])
        (output / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['compat512', 'full1025'], required=True)
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--native-manifest', type=Path, required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    args = parser.parse_args()
    admission, guard = source_graph.source.runtime_context(
        args.runtime_manifest, args.expected_runtime_sha256,
        args.native_manifest, args.expected_native_sha256,
        extra_sources=[Path(__file__).resolve(), Path(prior.__file__).resolve(),
                       Path(graph.__file__).resolve(), Path(source_graph.__file__).resolve(),
                       Path(source_graph.graph.__file__).resolve()])
    result = collect(args.output, args.phase, admission, guard)
    print(json.dumps(dict(completed=result['completed'], failure=result.get('failure'))))
    raise SystemExit(0 if result['completed'] else 1)
