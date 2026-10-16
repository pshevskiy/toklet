"""Bounded paired replay to investigate original changed winner distributions.

Run with every other native measurement/test/build queue idle. An explicitly
reviewed isolated build and an exact original completed quality record are
required. The original quality worker, collectors and runtime stay unchanged.
The diagnostic adds disk writes; none of its durations qualify performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import m6_context as base
import m6_context_audit as audit
import m6_winner_analysis as analysis
import m6_winner_capture_build as build

ROOT = base.ROOT
FORMAT = 'toklet-m6-winner-capture-v1'
TIMEOUT = 43200
LIFETIME = 86400


def selected_indices(original, requested):
    changed = [r['target_index'] for r in original['worker']['phases'][0]['events'][1:-1]
               if r['reference_greedy'] != r['online_greedy']]
    indices = changed if requested is None else requested
    if (type(indices) is not list or not 1 <= len(indices) <= 8
            or any(type(i) is not int or i not in changed for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError('Require 1..8 sorted unique original changed target indices')
    return indices


def validate_events(events, original, indices, ids):
    captures, comparisons = events[:len(indices)], events[len(indices):]
    expected = original['worker']['phases'][0]['events']
    if comparisons != expected:
        raise ValueError('Diagnostic distributions/metrics/state do not exactly reproduce original rows')
    for capture, index in zip(captures, indices, strict=True):
        row = expected[index+1]
        prefix = ids[:len(ids)-128+index]
        wanted = dict(event='capture', target_index=index, prefix_tokens=len(prefix),
            prefix_sha256=base.shared.history_sha(prefix), vocabulary_values=151936,
            reference_file=f'reference-target{index:04}.fp32le', online_file=f'online-target{index:04}.fp32le',
            reference_logits_sha256=row['reference_logits_sha256'], online_logits_sha256=row['online_logits_sha256'])
        if set(capture) != set(wanted):
            raise ValueError('Original full-distribution capture metadata fields differ')
        for key, value in wanted.items():
            base.exact(capture[key], value, f'original full-distribution capture metadata {key}')
    return captures


def capture_bindings(directory, captures, original):
    if directory.is_symlink() or directory.resolve() != directory or not directory.is_dir():
        raise ValueError('Capture directory is not a regular admitted path')
    expected = {c[f'{mode}_file'] for c in captures for mode in base.MODES}
    if set(p.name for p in directory.iterdir()) != expected:
        raise ValueError('Capture file inventory differs')
    pins = {}
    rows = original['worker']['phases'][0]['events']
    for capture in captures:
        row = rows[capture['target_index']+1]
        for mode in base.MODES:
            path = build.relative(str((directory/capture[f'{mode}_file']).relative_to(ROOT)))
            if not path.is_file():
                raise ValueError('Capture payload is not a regular file')
            with path.open('rb') as stream:
                data = stream.read(151936*4+1)
            sha = row[f'{mode}_logits_sha256']
            if len(data) != 151936*4 or hashlib.sha256(data).hexdigest() != sha:
                raise ValueError('Capture endpoint differs from original full-distribution identity')
            pins[path] = sha
    return pins


def verify_files(directory, captures, original):
    pins = capture_bindings(directory, captures, original)
    results = []
    rows = original['worker']['phases'][0]['events']
    for capture in captures:
        files = [directory/capture[f'{mode}_file'] for mode in base.MODES]
        result = analysis.analyze(*files, rows[capture['target_index']+1])
        result['prefix_sha256'] = capture['prefix_sha256']
        results.append(result)
    return results, pins


class Worker(base.Worker):
    def __init__(self, *args, capture_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.command.append(str(capture_dir))
        self.record.update(command=self.command, event_read_timeout_seconds=TIMEOUT, lifetime_timeout_seconds=LIFETIME)

    def __enter__(self):
        try:
            self.check()  # All large hashes complete before native work starts.
            self.child = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, start_new_session=True)
            self.record['worker_pid'] = self.child.pid
            self.transport = base.shared.metal_engine.Transport(self.child, self.raw, self.stderr,
                stdout_limit=16*1024**2, event_limit=139, request_timeout=TIMEOUT, lifetime_timeout=LIFETIME)
            event = self.transport.read()
            self.admission = event; self.record['ready'] = event
            base.exact(event.get('format'), FORMAT, 'diagnostic ready format')
            base.ready({**event, 'format': 'toklet-m6-context-v1'}, self.model, self.capacity, 128, False)
            self.usage('ready')
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise


def admission(build_path, expected_sha):
    record, sha = audit.read_json(build_path)
    base.exact(sha, expected_sha, 'explicitly reviewed diagnostic build SHA')
    pins = build.verify_record(record)
    pins[build_path] = sha
    return ROOT/record['binary']['path'], pins


def anchor_admission(path, expected_sha):
    original, sha = audit.read_json(path)
    base.exact(sha, expected_sha, 'explicit original quality record SHA')
    proof = audit.audit([path])
    base.exact(proof['bindings'].get(str(path.relative_to(ROOT))), sha, 'parsed versus replayed original SHA')
    if original['plan'] != 'quality':
        raise ValueError('Original completed quality collection required')
    return original, {ROOT/p: s for p, s in proof['bindings'].items()}


def run(args):
    output = build.relative(args.output)
    directory = build.relative(args.capture_dir)
    paths = [output, output.with_suffix('.raw.jsonl'), output.with_suffix('.stderr'), directory]
    if len(set(paths)) != 4 or any(p.exists() for p in paths) or any(p.parent != ROOT/'artifacts' for p in paths):
        raise ValueError('Require distinct fresh diagnostic destinations directly under artifacts')
    binary, pins = admission(build.relative(args.build), args.build_sha256)
    original, anchor_pins = anchor_admission(build.relative(args.anchor), args.anchor_sha256)
    for path, sha in anchor_pins.items():
        if path in pins and pins[path] != sha:
            raise ValueError('Diagnostic/original provenance conflict')
        pins[path] = sha
    model, depth = original['model'], original['depth']
    _, arrays, source, packed, fixed = base.inputs(model)
    for path, sha in fixed.items():
        if path in pins and pins[path] != sha:
            raise ValueError('Diagnostic frozen input conflict')
        pins[path] = sha
    modules = (base, audit, analysis, build, base.shared, base.shared.metal_engine,
               base.shared.native_usage, base.shared.metal_continue, base.loss)
    files = set(pins) | {Path(__file__).resolve(), *(Path(m.__file__).resolve() for m in modules),
                         ROOT/'tools/experiments/m6_fp32_context_audit.py', ROOT/'tools/experiments/m6_fp32_values.py',
                         ROOT/'tools/experiments/m6_typed_quality.py', ROOT/'tools/experiments/m6_typed_quality_audit.py',
                         ROOT/'tools/experiments/m6_long_quality_audit.py',
                         ROOT/'tools/experiments/m6_context_scope_audit.py'}
    identities = base.capture_identities(files, pins)
    indices = selected_indices(original, args.indices)
    ids = arrays[depth]
    request = dict(original['worker']['phases'][0]['request'], captures=indices)
    def check():
        build.verify_record(audit.read_json(build.relative(args.build))[0])
        if any(base.shared.metal_engine.digest(ROOT/p) != sha for p, sha in identities.items()):
            raise ValueError('Diagnostic consumed inputs changed before launch')
    result = dict(format='toklet-m6-winner-capture-collection-v1', passed=False, collection_completed=False,
        investigation_complete=False, model=model, depth=depth, indices=indices,
        original_quality_record=args.anchor, original_quality_sha256=args.anchor_sha256,
        build_record=args.build, build_sha256=args.build_sha256,
        identities=identities, endpoint_errors=[], capture_directory=args.capture_dir,
        power_before=base.shared.metal_continue.endpoints(), m6_complete=False,
        scope='Exact original full-distribution replay and independent bounded verification only; no latency, internal-cause or quality acceptance')
    with output.open('x') as report, paths[1].open('x') as raw, paths[2].open('xb') as stderr:
        worker = Worker(binary, source, packed, depth, 128, False, model, raw, stderr, check, capture_dir=directory)
        result['worker'] = worker.record
        try:
            with worker:
                events = worker.call(request, 'comparison_done')
                captures = validate_events(events, original, indices, ids)
            # File reads/hashes, independent math and endpoint checks occur only
            # after natural worker exit, EOF and owned cleanup.
            result['analyses'], result_pins = verify_files(directory, captures, original)
            result['capture_bindings'] = {str(p.relative_to(ROOT)): sha for p, sha in result_pins.items()}
            result['post_exit_analysis_usage'] = base.shared.native_usage.usage()
            peak = result['post_exit_analysis_usage'].get('lifetime_max_phys_footprint')
            if type(peak) is not int or not 0 < peak <= base.LIMIT:
                raise ValueError('Post-exit analysis native charge outside6GiB')
            result['analysis_memory_scope'] = 'Driver lifetime peak sampled after full-vector analysis; numerical worker already exited, not a simultaneous worker allocation'
            result.update(passed=worker.record['passed'], collection_completed=worker.record['passed'])
        except BaseException as error:
            result.update(passed=False, collection_completed=False, error=f'{type(error).__name__}: {error}')
        finally:
            try:
                check()
                if result['passed']:
                    final_pins = capture_bindings(directory, captures, original)
                    if {str(p.relative_to(ROOT)): sha for p, sha in final_pins.items()} != result['capture_bindings']:
                        raise ValueError('Final capture endpoint identity map differs')
                    result['capture_endpoints_unchanged'] = True
                result['identities_unchanged'] = True
            except BaseException as error:
                result['identities_unchanged'] = False; result['endpoint_errors'].append(str(error))
            try:
                result['power_after'] = base.shared.metal_continue.endpoints()
            except BaseException as error:
                result['endpoint_errors'].append(str(error))
            if result['endpoint_errors']:
                result.update(passed=False, collection_completed=False)
            json.dump(result, report, indent=2, allow_nan=False); report.write('\n')
    if not result['passed']:
        raise ValueError('Diagnostic failed; immutable attempted evidence retained')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True)
    parser.add_argument('--build-sha256', required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--anchor-sha256', required=True)
    parser.add_argument('--capture-dir', required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
