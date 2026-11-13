"""Independently stream all short-phase Qwen3.5-2B graph coordinates.

The prequalified scalar/vector graph arithmetic is unchanged. A caller must
admit and monitor the actual numerical runtime before invoking collect().
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time

import bf16_source
import build_qwen35_vector_graph as builder
import qwen35_vector_graph as graph
import qwen35_2b_graph_protocol as p
import qwen35_2b_weights as weights

need = p.need


def extras():
    import qwen35_recurrent_vector
    return [Path(v.__file__).resolve() for v in (builder, graph, qwen35_recurrent_vector,
        p, weights, weights.base, weights.base.pin, weights.tiles, weights.graph, bf16_source)] + [
        Path(__file__).resolve(), Path(__file__).resolve().parents[2] /
        'tests/fixtures/qwen35-2b-source-v1/source-config.json']


def summary(role, phase, manifest_sha256, pins):
    plan = p.plan(phase)
    return dict(format=p.FORMAT, role=role, phase=phase, capture_mode='raw', protocol_sha256=p.protocol_sha256(),
        source_manifest_sha256=p.sha(manifest_sha256), artifact_pins=pins, cases=p.cases(phase),
        geometry=p.GEOMETRY, gate=p.GATE, profiles=list(p.PROFILES[role]),
        arrays=plan['arrays_per_profile'] * len(p.PROFILES[role]),
        values_per_profile=plan['values_per_profile'], rows_bytes=plan[role + '_rows_bytes'],
        rows_sha256=None, journal_bytes=0, journal_sha256=None, completed=False,
        qualifications=p.QUALIFICATIONS)


def endpoint(owner, deadline):
    digest = hashlib.sha256()
    for offset in range(0, owner.size, bf16_source.HASH_BLOCK):
        digest.update(owner.read(offset, min(bf16_source.HASH_BLOCK, owner.size - offset), deadline=deadline))
    need(digest.hexdigest() == owner.sha256, 'Whole child source endpoint differs')
    return digest.hexdigest()


def collect(output, manifest_path, manifest_sha256, admission, runtime_endpoint, phase='initial_short'):
    """Heavy entry point; never invoked by portable tests or cold planning."""
    import numpy as np
    need(np.__version__ == '2.3.2', 'Pinned NumPy 2.3.2 required')
    provenance = builder.validate()
    need(phase == 'initial_short', 'Deeper phases require bounded streamed comparison')
    source = weights.artifact_manifest(manifest_path, manifest_sha256)
    pins, directory = source['artifact_pins'], Path(manifest_path).resolve().parent
    plan = p.plan(phase)
    output = Path(output).absolute()
    need(output.parent.is_dir() and not output.exists(), 'Fresh exclusive capture output required')
    need(shutil.disk_usage(output.parent).free >= plan['combined_capture_reservation_bytes'] + p.DISK_FLOOR,
         'Fresh complete-phase disk admission required')
    need(plan['reference_reserved_bytes'] <= 512 * 1024**2, 'Reference reservation exceeds setup cap')
    output.mkdir()
    result = summary('reference', phase, manifest_sha256, pins)
    audit = dict(graph_provenance=provenance, resource_plan=plan, runtime_admission=admission,
        profile_ns={}, source=None, source_endpoint_sha256=None, primary_error=None,
        endpoint_errors=[], runtime_endpoint=None, engine_executed=False)
    owner = None
    started = time.monotonic_ns()
    try:
        config = weights.captured(directory / 'config.json', pins['config_bytes'], pins['config_sha256'])
        deadline = time.monotonic() + 24 * 3600
        with bf16_source.Source(directory / 'model.safetensors', pins['container_bytes'],
                                 pins['container_sha256'], deadline=deadline) as owner:
            matrices, vectors = weights.weights(owner, config, pins, deadline)
            rows_digest, journal_digest = hashlib.sha256(), hashlib.sha256()
            with (output / 'rows.bin').open('xb') as binary, (output / 'arrays.jsonl').open('xb') as journal:
                seen, totals = set(), {}
                for profile, staged in [('ideal_fp64', False), ('staged_fp32', True)]:
                    profile_started = time.monotonic_ns()
                    totals[profile] = 0
                    for case in p.cases(phase):
                        def save(row):
                            stage, head = p.split_stage(row['stage'])
                            shape = p.shapes(row['token'] + 1)[row['layer'], stage, head]
                            values = row['values']
                            raw = struct.pack('<' + ('f' if staged else 'd') * len(values), *values)
                            descriptor = dict(profile=profile, case=case['name'], token=row['token'],
                                position=case['position_offset'] + row['token'], layer=row['layer'],
                                stage=stage, head=head, shape=shape, dtype='F32' if staged else 'F64',
                                count=len(values), offset=binary.tell(), bytes=len(raw),
                                sha256=hashlib.sha256(raw).hexdigest())
                            key = p.observation(descriptor, 'reference', phase)
                            need(key not in seen, 'Duplicate independent observation')
                            p.finite(raw, descriptor['dtype'])
                            need(binary.tell() + len(raw) <= result['rows_bytes'], 'Reference capture exceeds frozen size')
                            line = (json.dumps(descriptor, separators=(',', ':'), allow_nan=False) + '\n').encode()
                            need(len(line) <= 4096 and journal.tell() + len(line) <= p.MAX_JOURNAL,
                                 'Reference journal exceeds frozen bound')
                            need(binary.write(raw) == len(raw) and journal.write(line) == len(line), 'Short reference output write')
                            rows_digest.update(raw)
                            journal_digest.update(line)
                            seen.add(key)
                            totals[profile] += len(values)
                        trace = graph.forward(matrices, vectors, case['tokens'], case['position_offset'], staged,
                            geometry=p.GEOMETRY, math_type=weights.Math, sink=save)
                        need(trace == [], 'Independent graph must stream without retained traces')
                        print(json.dumps(dict(event='case-complete', profile=profile, case=case['name'])), flush=True)
                    audit['profile_ns'][profile] = time.monotonic_ns() - profile_started
                need(len(seen) == result['arrays'] and binary.tell() == result['rows_bytes']
                     and all(n == result['values_per_profile'] for n in totals.values()),
                     'Complete independent coordinates/profiles required')
                result.update(rows_sha256=rows_digest.hexdigest(), journal_bytes=journal.tell(),
                              journal_sha256=journal_digest.hexdigest())
            audit['source_endpoint_sha256'] = endpoint(owner, deadline)
        audit['source'] = owner.audit()
        audit['runtime_endpoint'] = runtime_endpoint()
        result['completed'] = True
    except BaseException as error:
        audit['primary_error'] = dict(type=type(error).__name__, message=str(error)[:4096])
    finally:
        if owner is not None:
            if not owner.closed:
                try:
                    owner.close()
                except BaseException as error:
                    audit['endpoint_errors'].append(str(error)[:4096])
            audit['source'] = owner.audit()
        if audit['runtime_endpoint'] is None:
            try:
                audit['runtime_endpoint'] = runtime_endpoint()
            except BaseException as error:
                audit['endpoint_errors'].append(str(error)[:4096])
        if audit['endpoint_errors']:
            result['completed'] = False
        audit['collection_ns'] = time.monotonic_ns() - started
        (output / 'runtime.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def collect_stream(output, manifest_path, manifest_sha256, admission, runtime_endpoint,
                   phase, profile, stream, native_folder):
    """One profile and native pipe. The root owns process launch/accounting."""
    import numpy as np
    from qwen35_2b_graph_check import StreamComparison
    need(np.__version__ == '2.3.2', 'Pinned NumPy 2.3.2 required')
    need(profile in ('ideal_fp64', 'staged_fp32'), 'One explicit independent stream profile required')
    source = weights.artifact_manifest(manifest_path, manifest_sha256)
    pins, directory = source['artifact_pins'], Path(manifest_path).resolve().parent
    plan = p.plan(phase)
    output = Path(output).absolute()
    need(output.parent.is_dir() and not output.exists(), 'Fresh exclusive stream output required')
    need(shutil.disk_usage(output.parent).free >= plan['streamed_capture_reservation_bytes'] + p.DISK_FLOOR,
         'Fresh complete streamed-phase disk admission required')
    need(plan['reference_reserved_bytes'] <= 512 * 1024**2, 'Stream reference exceeds setup reservation')
    comparison = StreamComparison(stream, output, profile, phase, manifest_sha256, pins)
    audit = dict(graph_provenance=builder.validate(), resource_plan=plan, runtime_admission=admission,
        source=None, source_endpoint_sha256=None, runtime_endpoint=None, primary_error=None,
        endpoint_errors=[], engine_executed=False)
    owner, result = None, None
    started = time.monotonic_ns()
    try:
        config = weights.captured(directory / 'config.json', pins['config_bytes'], pins['config_sha256'])
        deadline = time.monotonic() + 24 * 3600
        with bf16_source.Source(directory / 'model.safetensors', pins['container_bytes'],
                                 pins['container_sha256'], deadline=deadline) as owner:
            matrices, vectors = weights.weights(owner, config, pins, deadline)
            for case in p.cases(phase):
                def save(row):
                    comparison.save(case, row)
                trace = graph.forward(matrices, vectors, case['tokens'], case['position_offset'],
                    profile == 'staged_fp32', geometry=p.GEOMETRY, math_type=weights.Math, sink=save)
                need(trace == [], 'Reference must stream every coordinate')
            result = comparison.finish(native_folder)
            audit['source_endpoint_sha256'] = endpoint(owner, deadline)
        audit['source'] = owner.audit()
        audit['runtime_endpoint'] = runtime_endpoint()
    except BaseException as error:
        audit['primary_error'] = dict(type=type(error).__name__, message=str(error)[:4096])
    finally:
        comparison.close()
        if owner is not None:
            if not owner.closed:
                try:
                    owner.close()
                except BaseException as error:
                    audit['endpoint_errors'].append(str(error)[:4096])
            audit['source'] = owner.audit()
        if audit['runtime_endpoint'] is None:
            try:
                audit['runtime_endpoint'] = runtime_endpoint()
            except BaseException as error:
                audit['endpoint_errors'].append(str(error)[:4096])
        audit['collection_ns'] = time.monotonic_ns() - started
        if audit['primary_error'] is not None or audit['endpoint_errors']:
            result = dict(format='toklet-qwen35-2b-stream-failure-v1', completed=False,
                          phase=phase, profile=profile, qualifications=p.QUALIFICATIONS)
        (output / 'runtime.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child-manifest', type=Path, required=True)
    parser.add_argument('--expected-child-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--native-manifest', type=Path, required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    parser.add_argument('--phase', choices=p.PHASES, default='initial_short')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--profile', choices=['ideal_fp64', 'staged_fp32'])
    parser.add_argument('--native-capture', type=Path)
    args = parser.parse_args()
    need(not args.stream or args.profile is not None and args.native_capture is not None,
         'Stream profile and owned native capture folder required')
    import gemma4_affine_reference as runtime_owner
    admission, guard = runtime_owner.runtime_context(args.runtime_manifest, args.expected_runtime_sha256,
        args.native_manifest, args.expected_native_sha256,
        extra_sources=extras() + [Path(__file__).with_name('qwen35_2b_graph_check.py')])
    if args.stream:
        result = collect_stream(args.output, args.child_manifest, args.expected_child_manifest_sha256,
            admission, guard, args.phase, args.profile, sys.stdin.buffer, args.native_capture)
    else:
        result = collect(args.output, args.child_manifest, args.expected_child_manifest_sha256,
            admission, guard, args.phase)
    print(json.dumps(dict(completed=result['completed'], passed=result.get('passed'))))
    raise SystemExit(0 if result['completed'] and result.get('passed', True) else 1)
