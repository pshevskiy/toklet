"""Isolated synthetic I8 KV collection; never an engine adoption decision.

The parent acknowledges each completed GPU event only after sampling both owned
native charges. No hashes, builds or other workers run inside the timing loop.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

import metal_engine
import native_usage

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = 'benchmarks/m6-kv-component-v1/manifest.json'
LIMIT = 6 * 1024**3
GEOMETRIES = {'small': (24, 14, 64), 'larger': (28, 12, 128)}
FORMATS = ('fp16', 'i8')


def load(path, limit=2*1024**2):
    with path.open('rb') as stream:
        data = stream.read(limit+1)
    if not data or len(data) > limit:
        raise ValueError('Empty or oversized component input')
    value = json.loads(data, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    return value, hashlib.sha256(data).hexdigest()


def observe_environment():
    def observe(command):
        return subprocess.check_output(command, text=True, timeout=10).strip()
    return dict(platform=observe(['/usr/bin/sw_vers']),
                hardware=observe(['/usr/sbin/sysctl', '-n', 'hw.model']),
                chip=observe(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string']),
                battery=observe(['/usr/bin/pmset', '-g', 'batt']),
                policy=observe(['/usr/bin/pmset', '-g', 'custom']),
                thermal=observe(['/usr/bin/pmset', '-g', 'therm']))


def exact(value, expected, label):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f'Component {label} differs')


def finite(value, label, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError(f'Invalid component {label}')
    return value


def admitted(model):
    layers, heads, d = GEOMETRIES[model]
    q, cache = heads*d, 2*32768*d
    words = q+7 + 2*(cache+7) + 2*(cache//2+7) + 2*(cache//32+7)
    return words*2*layers + (q+7)*2+32 + 576*1024**2 + 32768*8 + q*16 + 1024**2


def ready(event, model, collection):
    layers, heads, d = GEOMETRIES[model]
    fields = dict(event='ready', format='toklet-m6-kv-component-v1', model_shape=model,
                  collection=collection, layers=layers, query_heads=heads, kv_heads=2,
                  head_dim=d, capacity=32768, debug_assertions=False, m6_complete=False,
                  admitted_bytes=admitted(model), framework_allowance_bytes=512*1024**2,
                  driver_allowance_bytes=64*1024**2, scratch_allowance_bytes=1024**2,
                  oracle_score_bytes=32768*8, oracle_output_bytes=heads*d*16)
    for key, value in fields.items():
        exact(event.get(key), value, key)
    if not isinstance(event.get('device'), str) or not event['device'] or event['admitted_bytes'] > LIMIT:
        raise ValueError('Component device/admission unavailable')
    width = event.get('execution_width')
    if type(width) is not int or width < 32:
        raise ValueError('Component execution width unavailable')
    hashes = event.get('input_hashes')
    if not isinstance(hashes, list) or len(hashes) != layers or any(
            not isinstance(row, list) or len(row) != 7 or any(
                not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in row) for row in hashes):
        raise ValueError('Component input hash coverage differs')
    if len({tuple(row) for row in hashes}) != layers:
        raise ValueError('Component layers reuse input content')
    errors = event.get('quantization')
    if not isinstance(errors, list) or len(errors) != layers:
        raise ValueError('Component quantization coverage differs')
    for index, row in enumerate(errors):
        exact(row.get('layer'), index, 'quantization layer')
        for name in ('k', 'v'):
            item = row[name]
            exact(item.get('values'), 2*32768*d, 'quantized values')
            for key in ('max_reconstruction_error', 'rms_reconstruction_error'):
                finite(item.get(key), key)
            count = item.get('clamped_values')
            if type(count) is not int or not 0 <= count <= item['values']:
                raise ValueError('Component clamp count invalid')


def correctness(event, model):
    exact(event.get('event'), 'correctness', 'correctness event')
    value = event['result']
    layers, heads, d = GEOMETRIES[model]
    for key, expected in dict(compared_outputs=3*layers*heads*d, expected_outputs=3*layers*heads*d,
                              every_layer_checked=True, rejected_invalid_cases=7,
                              exact_word_values_checked=256, valid_attention_outputs_checked=6*heads*d,
                              quantization_error_is_not_arithmetic_error=True).items():
        exact(value.get(key), expected, key)
    for key in ('max_arithmetic_error', 'quantization_attention_max_error', 'quantization_attention_rms_error'):
        finite(value.get(key), key)


def sample(event, model, pair, ordinal):
    exact(event.get('event'), 'sample', 'sample event')
    exact(event.get('pair'), pair, 'pair')
    exact(event.get('phase'), 'warmup' if pair < 3 else 'sample', 'phase')
    value = event['sample']
    order = FORMATS if pair % 2 == 0 else FORMATS[::-1]
    for key, expected in dict(format=order[ordinal], operator_count=GEOMETRIES[model][0],
                              command_count=1, status_zero=True, output_guard_unchanged=True).items():
        exact(value.get(key), expected, key)
    for key in ('completed_gpu_seconds', 'completed_host_seconds', 'host_encode_seconds'):
        finite(value.get(key), key, positive=True)
    if value['host_encode_seconds'] > value['completed_host_seconds']:
        raise ValueError('Component host timing order invalid')


def done(event):
    for key, expected in dict(event='done', samples=26, inputs_unchanged=True,
                              every_layer_checked=True, cache_write_cost_included=False, m6_complete=False).items():
        exact(event.get(key), expected, key)


def summaries(events):
    result = {}
    for fmt in FORMATS:
        result[fmt] = {}
        for key in ('completed_gpu_seconds', 'completed_host_seconds', 'host_encode_seconds'):
            values = [e['sample'][key] for e in events if e.get('event') == 'sample'
                      and e['phase'] == 'sample' and e['sample']['format'] == fmt]
            if len(values) != 10:
                raise ValueError('Component measured coverage differs')
            median = statistics.median(values)
            result[fmt][key] = dict(count=10, median=median,
                mad=statistics.median(abs(v-median) for v in values), minimum=min(values), maximum=max(values))
    a, b = (result[f]['completed_gpu_seconds'] for f in FORMATS)
    result['gpu_median_reduction_fraction'] = 1-b['median']/a['median']
    result['gpu_median_gain_at_least_five_percent'] = result['gpu_median_reduction_fraction'] >= .05
    result['gpu_observed_ranges_separate'] = b['maximum'] < a['minimum']
    result['adoption_qualified'] = False
    return result


def build_admission(path, expected_sha):
    """Require the exact reviewed build record, not an inferred current binary."""
    record, sha = load(path, 8*1024**2)
    exact(sha, expected_sha, 'reviewed build record SHA')
    exact(record.get('format'), 'toklet-m6-kv-component-build-v1', 'build format')
    exact(record.get('passed'), True, 'build success')
    from m6_kv_component_build import verify_record
    pins = verify_record(record)
    pins[path] = sha
    pins.update({p: metal_engine.digest(p) for p in (
        Path(__file__).resolve(), Path(metal_engine.__file__).resolve(), Path(native_usage.__file__).resolve())})
    def check():
        verify_record(record)
        if any(metal_engine.digest(p) != digest for p, digest in pins.items()):
            raise ValueError('Component source/build identity changed')
    check()
    return ROOT/record['binary']['path'], pins, check


def run(args):
    output = args.output.resolve()
    paths = [output, output.with_suffix('.raw.jsonl'), output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):
        raise ValueError('Component evidence destination exists')
    binary, pins, check = build_admission(args.build.resolve(), args.build_sha256)
    contract, sha = load(ROOT/CONTRACT)
    exact(contract.get('format'), 'toklet-m6-i8-kv-component-contract-v1', 'contract')
    # Contract bytes were included before building; no post-hoc contract changes.
    exact(sha, pins[ROOT/CONTRACT], 'frozen contract')
    result = dict(format='toklet-m6-kv-component-collection-v1', model_shape=args.model,
                  collection=args.collection, passed=False, m6_complete=False, adoption_qualified=False,
                  events=[], usage=[], cleanup_errors=[], environment_errors=[], build_sha256=args.build_sha256,
                  provenance={str(p.relative_to(ROOT)): s for p, s in pins.items()})
    result['command'] = [str(binary), args.model, args.collection]
    child = transport = None
    primary = None
    with paths[0].open('x') as report, paths[1].open('x') as raw, paths[2].open('xb') as stderr:
        try:
            result['environment_before'] = observe_environment()
            check()
            child = subprocess.Popen(result['command'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, start_new_session=True)
            transport = metal_engine.Transport(child, raw, stderr, event_limit=29,
                                               request_timeout=7200, lifetime_timeout=14400)
            def read(validate):
                event = transport.read()
                try:
                    json.dumps(event, allow_nan=False)
                except ValueError as error:
                    result['rejected_event'] = dict(ordinal=len(result['events'])+1,
                                                   reason='Nonfinite JSON; see preserved raw log')
                    raise ValueError('Nonfinite component event; see preserved raw log') from error
                result['events'].append(event)
                validate(event)
                own, driver = native_usage.usage(child.pid), native_usage.usage()
                peaks = [v.get('lifetime_max_phys_footprint') for v in (own, driver)]
                if any(type(v) is not int or v <= 0 for v in peaks) or sum(peaks) > LIMIT:
                    raise ValueError('Component joint owned native peak outside6GiB')
                result['usage'].append(dict(event=len(result['events']), worker=own, driver=driver,
                                           joint_native_lifetime_peak_bytes=sum(peaks)))
                child.stdin.write(b'continue\n')
                child.stdin.flush()
            read(lambda e: ready(e, args.model, args.collection))
            read(lambda e: correctness(e, args.model))
            for pair in range(13):
                for ordinal in range(2):
                    read(lambda e: sample(e, args.model, pair, ordinal))
            read(done)
            child.stdin.close()
            result['exit_code'] = transport.finish()
            if result['exit_code'] != 0 or transport.pending or not transport.stdout_closed or transport.events != 29:
                raise ValueError('Component natural EOF/event coverage differs')
            result['summaries'] = summaries(result['events'])
            result['passed'] = True
        except BaseException as error:
            primary = error
            result['error'] = f'{type(error).__name__}: {error}'
            result['passed'] = False
        finally:
            actions = []
            if child:
                actions += [lambda: metal_engine.kill_owned_group(child), lambda: child.wait(timeout=5)]
                actions += [s.close for s in (child.stdin, child.stdout, child.stderr) if not s.closed]
            if transport:
                actions.append(transport.close)
            for action in actions:
                try:
                    action()
                except BaseException as error:
                    result['cleanup_errors'].append(str(error))
            try:
                check()
                result['identities_unchanged'] = True
            except BaseException as error:
                result['identities_unchanged'] = False
                result['endpoint_error'] = str(error)
            try:
                result['environment_after'] = observe_environment()
            except BaseException as error:
                result['environment_errors'].append(f'{type(error).__name__}: {error}')
            result['passed'] &= not result['cleanup_errors'] and result['identities_unchanged'] and not result['environment_errors']
            result['memory_caveat'] = 'Native libproc charge can omit clean/shared pages; numerical admission is separate.'
            report.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if primary:
        raise primary
    if not result['passed']:
        raise ValueError('Component cleanup/endpoint checks failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=GEOMETRIES, required=True)
    parser.add_argument('--collection', choices=('primary', 'confirmation'), required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--build-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
