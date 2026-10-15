"""Paired M6 context qualification against the frozen pre-collection contract.

Ordinary timing, teacher-forced quality and counter profiles use separate
workers. Long timings are descriptive; the repeated short primary is separate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys

import metal_controlled as shared
import metal_quality_v2 as loss

ROOT = shared.ROOT
LIMIT = shared.LIMIT
RECEIPT = 'benchmarks/results/2026-10-15-m6-context-worker-qualification-v1.json'
RECEIPT_SHA = '3d34b4516a7794c6599ed7c5f2e566434c5910fd07355c0719ee7dac206cd02c'
CONTRACT = 'benchmarks/m6-context-v1/manifest.json'
CONTRACT_SHA = '6a704eb37b681efacbdce88e58c6c86930e65a5ec1507a9e1b18551cc4a9dcaa'
MODES = ('reference', 'online')


def bounded(path, limit, digest):
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Frozen bounded input differs: {path}')
    return data


def compiled_admission():
    receipt = json.loads(bounded(ROOT / RECEIPT, 1024**2, RECEIPT_SHA))
    if receipt.get('passed') is not True or receipt.get('review', {}).get('accepted') is not True:
        raise ValueError('Reviewed paired worker required')
    binary = ROOT / receipt['binary']['path']
    snapshot = ROOT / receipt['source_snapshot']
    actual = {str(p.relative_to(ROOT)) for p in [ROOT/'Cargo.toml', ROOT/'Cargo.lock',
              ROOT/'examples/metal_context.rs', *(ROOT/'src').rglob('*.rs'), *(ROOT/'src').rglob('*.metal')]}
    if set(receipt['source_pins']) != actual:
        raise ValueError('Compiled source closure differs')
    pins = {ROOT/RECEIPT: RECEIPT_SHA, binary: receipt['binary']['sha256']}
    pins.update({ROOT/p: s for p, s in receipt['source_pins'].items()})
    pins.update({snapshot/p: s for p, s in receipt['source_pins'].items()})
    def check():
        if any(shared.metal_engine.digest(p) != s for p, s in pins.items()):
            raise ValueError('Reviewed compiled source/executable changed before launch')
    check()
    return binary, pins, check


def inputs(model):
    manifest = json.loads(bounded(ROOT/CONTRACT, 1024**2, CONTRACT_SHA))
    arrays = {}
    pins = {ROOT/CONTRACT: CONTRACT_SHA}
    for row in manifest['rows']:
        count = row['total_tokens']
        path = (ROOT/CONTRACT).parent / row['file']
        data = bounded(path, count*4, row['sha256'])
        if len(data) != count*4:
            raise ValueError('Long token count differs')
        ids = list(struct.unpack(f'<{count}I', data))
        if any(i >= 151936 for i in ids) or shared.history_sha(ids[:-128]) != row['prompt_sha256'] or shared.history_sha(ids[-128:]) != row['continuation_sha256']:
            raise ValueError('Long token identity differs')
        arrays[count] = ids
        pins[path] = row['sha256']
    spec = loss.SPECS[model]
    source = ROOT/'models'/spec['name']
    pins.update({source/'model.safetensors': spec['source'], source/'config.json': spec['config']})
    packed = ROOT/'artifacts/m5-larger-q4-v2' if model == 'larger' else None
    if packed:
        geometry = json.loads(bounded(ROOT/loss.GEOMETRY_REL, 2*1024**2, loss.GEOMETRY_SHA))
        if geometry.get('passed') is not True or geometry.get('review', {}).get('accepted') is not True:
            raise ValueError('Reviewed packed geometry required')
        artifact = {ROOT/p: v['sha256'] for p, v in geometry['bindings'].items() if p.startswith('artifacts/m5-larger-q4-v2/')}
        if artifact.get(packed/'manifest.json') != spec['packed'] or set(artifact) != set(packed.glob('*.q4')) | {packed/'manifest.json'}:
            raise ValueError('Frozen packed inventory differs')
        pins.update(artifact)
        pins[ROOT/loss.GEOMETRY_REL] = loss.GEOMETRY_SHA
    return manifest, arrays, source, packed, pins


def controlled_pins(manifest):
    """Bind all bytes consumed by prepare.verify, not their later observed values."""
    pins = {shared.prepare.CONTRACT/'manifest.json': shared.CONTRACT_SHA,
            ROOT/'benchmarks/sources.json': manifest['sources_sha256'],
            ROOT/'tools/bench/requirements.lock': manifest['python_dependencies_sha256']}
    pins.update({shared.prepare.CONTRACT/v['file']: v['sha256'] for v in manifest['arrays'].values()})
    pins.update({shared.prepare.CONTRACT/p: s for p, s in manifest['authored_text_sha256'].items()})
    return pins


def capture_identities(files, pins):
    identities = {str(p.relative_to(ROOT)): shared.metal_engine.digest(p) for p in sorted(files)}
    if any(identities[str(p.relative_to(ROOT))] != s for p, s in pins.items()):
        raise ValueError('Consumed frozen bytes differ')
    return identities


def finite(value, label, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError(f'Invalid {label}')
    return value


def exact(value, expected, label):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f'Actual {label} differs')


def ready(event, model, capacity, chunk, profiling):
    large = model == 'larger'
    hidden, layers, heads, dimension, weights = ((1536, 28, 12, 128, 1777247428) if large else (896, 24, 14, 64, 988069596))
    expected = dict(event='ready', format='toklet-m6-context-v1', debug_assertions=False,
                    context_capacity=capacity, chunk_size=chunk, profiling=profiling,
                    hidden_size=hidden, num_layers=layers, vocab_size=151936,
                    projection_mode='Q4CompensatedTiled' if large else 'HalfCompensated',
                    normalization_mode='Compensated' if large else 'Original', readback_mode='FullLogits',
                    kv_dtype='FP16', weight_dtype='Q4 v2 + FP16 exclusions' if large else 'FP16',
                    packed_format_version=2 if large else None, group_size=8 if large else None,
                    quantizer='toklet-q4-affine-f32-source-half-metadata-rne-g8-v2' if large else None,
                    shared_weights_bytes=weights, application_bytes=64*1024**2)
    for key, value in expected.items():
        exact(event.get(key), value, key)
    if type(event.get('device')) is not str or not event['device']:
        raise ValueError('Missing device')
    finite(event.get('load_seconds'), 'load duration')
    for key in ('half_execution_widths', 'q4_execution_widths'):
        widths = event.get(key)
        if type(widths) is not list or len(widths) != 2 or any(type(w) is not int or not 0 < w <= 0xffffffff for w in widths) or widths[0] != widths[1]:
            raise ValueError('Invalid queried execution widths')
    fields = {'weights_bytes', 'kv_bytes', 'scratch_bytes', 'history_bytes', 'logits_bytes', 'metadata_bytes', 'metadata_allowance_bytes', 'admitted_bytes'}
    for name in MODES:
        memory = event.get(f'{name}_memory')
        if type(memory) is not dict or set(memory) != fields or any(type(v) is not int or v < 0 for v in memory.values()):
            raise ValueError('Invalid memory inventory')
        expected_memory = dict(weights_bytes=weights, kv_bytes=(2*capacity*dimension+7)*4*layers,
                               history_bytes=capacity*4, logits_bytes=151936*8,
                               metadata_allowance_bytes=(128 if profiling else 64)*1024**2,
                               admitted_bytes=sum(v for k, v in memory.items() if k != 'admitted_bytes'))
        if any(memory[k] != v for k, v in expected_memory.items()):
            raise ValueError('Complete memory inventory differs')
    a, b = event['reference_memory'], event['online_memory']
    if a['scratch_bytes']-b['scratch_bytes'] != heads*chunk*capacity*8-8:
        raise ValueError('Depth-sized score storage not removed')
    unique = a['admitted_bytes']+b['admitted_bytes']-weights+64*1024**2
    exact(event.get('unique_memory_admitted_bytes'), unique, 'unique admission')
    if not 0 < unique <= LIMIT:
        raise ValueError('Joint admission exceeds6GiB')


def timing(value):
    exact(value.get('command_count'), 1, 'completed command count')
    for key in ('command_creation_seconds', 'clock_calibration_seconds', 'commit_seconds', 'host_encode_seconds', 'wait_seconds', 'gpu_start_seconds', 'gpu_end_seconds', 'readback_seconds'):
        finite(value.get(key), key)
    return finite(value['gpu_end_seconds']-value['gpu_start_seconds'], 'completed GPU duration', positive=True)


def sample(event, mode, memory, prefix, continuation, reused=0, prefill=None):
    exact(event.get('event'), 'sample', 'sample event')
    exact(event.get('mode'), mode, 'sample mode')
    if event.get('memory') != memory:
        raise ValueError('Persistent memory changed')
    shared.state(event, prefix+continuation, len(prefix)+len(continuation)-1)
    value = event['sample']
    for key, expected in dict(reused_tokens=reused, prefill_tokens=len(prefix) if prefill is None else prefill,
                              emitted_tokens=len(continuation), evaluated_tokens=len(prefix)+len(continuation)-1, pending_tokens=1).items():
        exact(value.get(key), expected, key)
    timestamps = value.get('token_event_seconds')
    if type(timestamps) is not list or len(timestamps) != len(continuation):
        raise ValueError('Token timing count differs')
    for t in timestamps:
        finite(t, 'token timestamp')
    if any(b < a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('Token clock order differs')
    for key in ('prefill_seconds', 'ttft_seconds', 'decode_seconds', 'total_seconds'):
        finite(value.get(key), key)
    if not value['prefill_seconds'] <= value['ttft_seconds'] == timestamps[0] <= timestamps[-1] <= value['total_seconds'] or value['decode_seconds'] != timestamps[-1]-timestamps[0]:
        raise ValueError('Timing boundary differs')
    timing(event['last_completed_timing'])
    return value


def comparison(events, ids, memories):
    prefix, continuation = ids[:-128], ids[-128:]
    exact(events[0], dict(event='comparison_begin', scored_rows=128, initial_prefix_tokens=len(prefix)), 'comparison begin')
    if len(events) != 130:
        raise ValueError('Incomplete paired quality rows')
    changed = []
    for i, row in enumerate(events[1:-1]):
        for key, expected in dict(event='comparison_row', target_index=i, target_token=continuation[i], prefix_tokens=len(prefix)+i, vocabulary_values_compared=151936).items():
            exact(row.get(key), expected, key)
        for key in ('reference_nll', 'online_nll', 'reference_margin', 'max_absolute_logit_delta', 'rms_logit_delta'):
            finite(row.get(key), key)
        if row['rms_logit_delta'] > row['max_absolute_logit_delta']:
            raise ValueError('Invalid paired error metrics')
        for key in ('reference_greedy', 'online_greedy'):
            if type(row.get(key)) is not int or not 0 <= row[key] < 151936:
                raise ValueError('Invalid paired winner')
        for key in ('reference_logits_sha256', 'online_logits_sha256'):
            sha = row.get(key)
            if type(sha) is not str or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                raise ValueError('Invalid paired vocabulary hash')
        for key in ('reference_winner_logits', 'online_winner_logits'):
            values = row.get(key)
            if type(values) is not list or len(values) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError('Invalid winner logits')
            if abs(values[1]-values[0]) > row['max_absolute_logit_delta']+1e-12:
                raise ValueError('Winner exceeds vocabulary error')
        ar, br = row['reference_winner_logits'], row['online_winner_logits']
        if ar[0] < br[0] or br[1] < ar[1]:
            raise ValueError('Recorded winners do not win')
        if row['reference_greedy'] != row['online_greedy']:
            changed.append({**row, 'within_error_envelope': row['reference_margin'] <= 2*row['max_absolute_logit_delta']})
    done = events[-1]
    exact(done.get('event'), 'comparison_done', 'comparison end')
    exact(done.get('scored_rows'), 128, 'scored rows')
    if done.get('memories') != memories or type(done.get('states')) is not list or len(done['states']) != 2:
        raise ValueError('Paired final inventory differs')
    for state in done['states']:
        shared.state({'state': state}, ids, len(ids)-1)
    delta = math.fsum(row['online_nll']-row['reference_nll'] for row in events[1:-1])/128
    return dict(scored_targets=128, mean_nll_increase=delta, gate_passed=delta <= .05, argmax_changes=changed,
                max_absolute_logit_delta=max(r['max_absolute_logit_delta'] for r in events[1:-1]))


def union(intervals):
    total, end = 0., -math.inf
    for start, finish in sorted(intervals):
        if finish < start:
            raise ValueError('Reversed interval')
        total += max(0., finish-max(start, end))
        end = max(end, finish)
    return total


def profile(events, mode, memory, model, ids, repeat):
    begin, done = events[0], events[-1]
    layers = 28 if model == 'larger' else 24
    count = layers*17+(6 if model == 'larger' else 4)
    if len(events) != count+2:
        raise ValueError('Profile coverage differs')
    for e, name in ((begin, 'profile_begin'), (done, 'profile_done')):
        exact(e.get('event'), name, name)
        exact(e.get('mode'), mode, 'profile mode')
    exact(begin.get('reused_tokens'), len(ids)-2 if repeat else 0, 'profile reused')
    exact(begin.get('prefill_tokens'), 1 if repeat else len(ids)-1, 'profile prefill')
    exact(done.get('operator_count'), count, 'operator count')
    if done.get('memory') != memory:
        raise ValueError('Profile memory changed')
    shared.state(done, ids, len(ids))
    duration = timing(done['timing'])
    start, end = begin['start'], begin['end']
    for pair in (start, end):
        for key in ('cpu_nanoseconds', 'gpu_ticks'):
            if type(pair.get(key)) is not int or not 0 < pair[key] < 2**64:
                raise ValueError('Invalid paired clock')
    cpu, gpu = end['cpu_nanoseconds']-start['cpu_nanoseconds'], end['gpu_ticks']-start['gpu_ticks']
    if cpu <= 0 or gpu <= 0:
        raise ValueError('Nonpositive clock calibration')
    finite(begin.get('host_clock_sample_bracket_seconds'), 'host clock bracket')
    intervals = []
    attention = 'attention' if mode == 'reference' else 'attention_decode_online'
    for i, wrapper in enumerate(events[1:-1]):
        exact(wrapper.get('event'), 'profile_operator', 'operator event')
        exact(wrapper.get('index'), i, 'operator index')
        p = wrapper['profile']
        if type(p.get('name')) is not str or not p['name']:
            raise ValueError('Missing operator name')
        host0 = finite(p.get('host_begin_seconds'), 'host begin')
        host1 = finite(p.get('host_end_seconds'), 'host end')
        if host1 < host0:
            raise ValueError('Host encoder ordering differs')
        for key in ('begin', 'end'):
            tick = p.get(f'gpu_{key}_ticks')
            if type(tick) is not int or not start['gpu_ticks'] <= tick <= end['gpu_ticks']:
                raise ValueError('Counter outside clock calibration')
            seconds = finite(p.get(f'gpu_{key}_seconds'), 'calibrated counter')
            expected = (tick-start['gpu_ticks'])/gpu*cpu/1e9
            if not math.isclose(seconds, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Counter conversion differs')
        if p['gpu_end_ticks'] < p['gpu_begin_ticks']:
            raise ValueError('GPU encoder ordering differs')
        if p['name'] == attention:
            intervals.append((p['gpu_begin_seconds'], p['gpu_end_seconds']))
    if len(intervals) != layers:
        raise ValueError('Attention operator coverage differs')
    attention_seconds = union(intervals)
    fraction = attention_seconds/duration
    if not 0 <= fraction <= 1:
        raise ValueError('Attention fraction outside matching completed command')
    return dict(attention_seconds=attention_seconds, completed_gpu_seconds=duration, attention_fraction=fraction)


class Worker:
    def __init__(self, binary, source, packed, capacity, chunk, profiling, model, raw, stderr, check):
        self.child = self.transport = None
        self.record = dict(passed=False, phases=[], usage=[], cleanup_errors=[])
        self.command = [str(binary), str(source), str(packed) if packed else '-', str(capacity), str(chunk)]+(['profile'] if profiling else [])
        self.record['command'] = self.command
        self.check, self.model, self.capacity, self.chunk, self.profiling = check, model, capacity, chunk, profiling
        self.raw, self.stderr = raw, stderr

    def __enter__(self):
        try:
            self.check()  # Immediately before Popen; never hash during measured requests.
            self.child = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            self.transport = shared.metal_engine.Transport(self.child, self.raw, self.stderr, stdout_limit=16*1024**2, event_limit=32768, request_timeout=7200, lifetime_timeout=86400)
            self.admission = self.transport.read()
            self.record['ready'] = self.admission
            ready(self.admission, self.model, self.capacity, self.chunk, self.profiling)
            self.usage('ready')
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def usage(self, label):
        own, driver = shared.native_usage.usage(self.child.pid), shared.native_usage.usage()
        values = [p.get('lifetime_max_phys_footprint') for p in (own, driver)]
        if any(type(v) is not int or v <= 0 for v in values) or sum(values) > LIMIT:
            raise ValueError('Joint owned native charge outside6GiB')
        self.record['usage'].append(dict(phase=label, worker=own, driver=driver, joint_native_lifetime_peak_bytes=sum(values)))

    def call(self, request, final):
        line = (json.dumps(request)+'\n').encode()
        if len(line) > 1024**2:
            raise ValueError('Request exceeds1MiB')
        self.child.stdin.write(line); self.child.stdin.flush()
        events = []
        while True:
            event = self.transport.read(); events.append(event)
            if len(events) > 1026:
                raise ValueError('Request event bound exceeded')
            if event.get('event') in (final, 'error'):
                break
        self.record['phases'].append(dict(request=request, events=events))
        self.usage(f'phase_{len(self.record["phases"])}')
        return events

    def __exit__(self, kind, value, traceback):
        primary = value if kind is not None else None
        try:
            if kind is None:
                self.child.stdin.close()
                self.record['exit_code'] = self.transport.finish()
                if self.record['exit_code'] != 0 or self.transport.pending or not self.transport.stdout_closed or self.transport.events != 1+sum(len(p['events']) for p in self.record['phases']):
                    raise ValueError('Natural EOF/event coverage differs')
                self.record['passed'] = True
        except BaseException as error:
            primary = error
            raise
        finally:
            if primary is not None:
                self.record['error'] = f'{type(primary).__name__}: {primary}'
            actions = []
            if self.child:
                actions += [lambda: shared.metal_engine.kill_owned_group(self.child), lambda: self.child.wait(timeout=5)]
                actions += [s.close for s in (self.child.stdin, self.child.stdout, self.child.stderr)]
            if self.transport:
                actions.append(self.transport.close)
            for action in actions:
                try:
                    action()
                except BaseException as error:
                    self.record['cleanup_errors'].append(str(error))
            if self.record['cleanup_errors']:
                self.record['passed'] = False
                if primary is None:
                    raise ValueError('Owned worker cleanup failed')


def summaries(rows):
    result = {}
    for mode in MODES:
        selected = [r['sample'] for r in rows if r['phase'] == 'sample' and r['mode'] == mode]
        result[mode] = {}
        for key in ('prefill_seconds', 'ttft_seconds', 'decode_seconds', 'total_seconds'):
            values = [r[key] for r in selected]
            median = statistics.median(values)
            result[mode][key] = dict(median=median, mad=statistics.median(abs(v-median) for v in values), minimum=min(values), maximum=max(values), count=len(values))
    return result


def measurement_plan(plan):
    if plan == 'matrix':
        return [(name, 3 if name == 'medium' else 1, 10 if name == 'medium' else 3) for name in shared.frozen.WORKLOADS]
    if plan in ('primary', 'confirmation'):
        return [('medium', 3, 10)]  # Frozen M6 primary is512 prompt/128 outputs.
    return [('boundary', 1, 2)] if plan == 'long' else [('boundary', 0, 1)]


def collect(worker, plan, ids, manifest, arrays, result):
    memories = [worker.admission[f'{m}_memory'] for m in MODES]
    if plan == 'quality':
        events = worker.call(dict(action='compare', tokens=ids[:-128], continuation=ids[-128:]), 'comparison_done')
        result['quality'] = comparison(events, ids, memories)
        if not result['quality']['gate_passed']:
            raise ValueError('Frozen per-depth paired NLL gate rejected')
    elif plan == 'profile':
        result['profiles'] = []
        for mode, memory in zip(MODES, memories):
            for i in range(3):
                events = worker.call(dict(action='profile', mode=mode, tokens=ids[:-1], continuation=ids[-1:]), 'profile_done')
                result['profiles'].append(dict(mode=mode, index=i, **profile(events, mode, memory, worker.model, ids, i > 0)))
        result['profile_medians'] = {m: statistics.median(p['attention_fraction'] for p in result['profiles'] if p['mode'] == m) for m in MODES}
    else:
        result['workloads'] = []
        for name, warm, measured in measurement_plan(plan):
            short = plan in ('primary', 'confirmation', 'matrix')
            rows = []; result['workloads'].append(dict(name=name, warmups=warm, measured_pairs=measured, samples=rows))
            for i in range(warm+measured):
                for mode in MODES if i % 2 == 0 else MODES[::-1]:
                    memory = worker.admission[f'{mode}_memory']
                    if short:
                        setup, request, _ = shared.requests(name, arrays, manifest)
                        for prep in setup:
                            e = worker.call({**prep, 'mode': mode}, 'prepared')[-1]
                            expected_ids = arrays['prompt_512']+([manifest['continuation']['pending_sample']['pending_token']] if prep['action'] == 'stage' else [])
                            if e.get('event') != 'prepared' or e.get('memory') != memory or e.get('mode') != mode:
                                raise ValueError('Cache setup failed')
                            shared.state(e, expected_ids, 512)
                        event = worker.call({**request, 'mode': mode}, 'sample')[-1]
                        value = shared.sample(event, name, {'memory': memory}, arrays, manifest)
                        exact(event.get('mode'), mode, 'sample mode')
                    else:
                        prompt, continuation = (ids[:4], ids[4:6]) if plan == 'probe' else (ids[:-128], ids[-128:])
                        event = worker.call(dict(action='reset', mode=mode, tokens=prompt, continuation=continuation), 'sample')[-1]
                        value = sample(event, mode, memory, prompt, continuation)
                    rows.append(dict(index=i, mode=mode, phase='warmup' if i < warm else 'sample', sample=value))
                    print(f'{worker.model}/{plan}/{len(ids)}/{name}/{i+1}/{mode}: {value["total_seconds"]:.6f}s', flush=True)
            result['workloads'][-1]['summary'] = summaries(rows)
        if plan in ('long', 'long-confirmation'):
            for mode, memory in zip(MODES, memories):
                e = worker.call(dict(action='prepare', mode=mode, tokens=ids), 'prepared')[-1]
                if e.get('event') != 'prepared' or e.get('memory') != memory:
                    raise ValueError('Final capacity not committed')
                shared.state(e, ids, len(ids))
                rejected = worker.call(dict(action='stage', mode=mode, tokens=[1]), 'prepared')[-1]
                if rejected.get('event') != 'error' or rejected.get('memories') != memories:
                    raise ValueError('Beyond-capacity mutation accepted')
                shared.state({'state': rejected['states'][MODES.index(mode)]}, ids, len(ids))


def run(output, model, plan, depth):
    paths = [output, output.with_suffix('.raw.jsonl'), output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):
        raise ValueError('Exclusive context evidence exists')
    binary, pins, check = compiled_admission()
    contract, long_arrays, source, packed, fixed = inputs(model)
    manifest, arrays = shared.contract()
    pins.update(fixed)
    pins.update(controlled_pins(manifest))
    modules = (shared, loss, loss.shared, loss.large, loss.large.quality_rows, shared.frozen, shared.common,
               shared.prepare, shared.metal_engine, shared.native_usage, shared.metal_prefill, shared.metal_continue)
    files = set(pins) | {Path(__file__), *(Path(m.__file__) for m in modules)}
    files.update(shared.prepare.CONTRACT/v['file'] for v in manifest['arrays'].values())
    files.add(shared.prepare.CONTRACT/'manifest.json')
    identities = capture_identities(files, pins)
    capacity = 2048 if plan in ('primary', 'confirmation', 'matrix', 'probe') else depth
    ids = long_arrays[depth]
    result = dict(format='toklet-m6-context-collection-v1', passed=False, collection_completed=False,
                  model=model, plan=plan, depth=depth, capacity=capacity, profiling=plan == 'profile',
                  scope='Paired controlled inference; long timings descriptive; no natural long-document ability or task gate',
                  identities=identities, revision=shared.metal_continue.observation(['git', 'rev-parse', 'HEAD']),
                  compiler=shared.metal_continue.observation(['rustc', '-Vv']),
                  platform=shared.metal_continue.observation(['/usr/bin/sw_vers']),
                  power_before=shared.metal_continue.endpoints(), energy_joules=None, continuous_memory_pressure=None)
    def before_launch():
        check()
        if any(shared.metal_engine.digest(ROOT/p) != s for p, s in identities.items()):
            raise ValueError('Consumed collection inputs changed before launch')
    with paths[0].open('x') as report, paths[1].open('x') as raw, paths[2].open('xb') as stderr:
        worker = Worker(binary, source, packed, capacity, 128, plan == 'profile', model, raw, stderr, before_launch)
        result['worker'] = worker.record
        try:
            with worker:
                collect(worker, plan, ids, manifest, arrays, result)
            result.update(passed=worker.record['passed'], collection_completed=worker.record['passed'])
        except BaseException as error:
            result.update(passed=False, error=f'{type(error).__name__}: {error}')
        finally:
            result['endpoint_errors'] = []
            try:
                result['identities_unchanged'] = all(shared.metal_engine.digest(ROOT/p) == s for p, s in identities.items())
                check()
            except BaseException as error:
                result['identities_unchanged'] = False; result['endpoint_errors'].append(str(error))
            try:
                result['power_after'] = shared.metal_continue.endpoints()
            except BaseException as error:
                result['endpoint_errors'].append(str(error))
            if not result['identities_unchanged'] or result['endpoint_errors']:
                result.update(passed=False, collection_completed=False)
            json.dump(result, report, indent=2, allow_nan=False); report.write('\n')
    if not result['passed']:
        raise ValueError('M6 context collection failed; see retained evidence')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=('small', 'larger'), required=True)
    p.add_argument('--plan', choices=('probe', 'primary', 'confirmation', 'matrix', 'long', 'long-confirmation', 'quality', 'profile'), required=True)
    p.add_argument('--depth', type=int, choices=(4096, 8192, 16384, 32768), default=4096)
    a = p.parse_args(); run(a.output.resolve(), a.model, a.plan, a.depth)
