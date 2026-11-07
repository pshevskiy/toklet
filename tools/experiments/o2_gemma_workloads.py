#!/usr/bin/env python3
"""Pinned native Gemma numerical workloads; no downloads or UI claims.

Two 3-warmup/10-measured batches of new processes and matched continuation
requests, explicit state controls, and >=600 seconds of sustained reset work.
Raw captures use a separate 64 MiB diagnostic disk bound, not the summary ring.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from m7_swift_sources import ROOT, worker_sources

sys.path.insert(0, str(ROOT / 'tools/bench'))
from request_cost_report import parse, require, number  # noqa: E402

CAPTURE_LIMIT = 64 * 1024 * 1024
NAMES = ('reset', 'append', 'fully_cached', 'pending', 'edited_prefix')


def validate_row(row):
    require(row['kind'] == 'request' and row['mode'] in ('legacy', 'observed'), 'Invalid request mode')
    start = number(row['submittedUptimeNanoseconds'], 'start', True)
    end = number(row['completedUptimeNanoseconds'], 'end', True)
    require(end > start, 'Invalid request interval')
    ids, callbacks = row['ids'], row['callbackUptimeNanoseconds']
    require(type(row['status']) is int and row['status'] in (0,1), 'Unexpected benchmark status')
    require(type(row['reason']) is int and row['reason'] in range(4), 'Invalid reason')
    require(type(row['trial']) is int and 0 <= row['trial'] <= 2048 and type(row['warmup']) is bool, 'Invalid trial')
    require(type(row['maximum']) is int and 0 <= row['maximum'] <= 128, 'Invalid output capacity')
    require(type(row['operation']) is int and row['operation'] in range(4), 'Invalid operation')
    require(len(row['input']) <= 512 and all(type(t) is int and 0 <= t < 262144 for t in row['input']), 'Invalid input IDs')
    require(len(ids) == len(callbacks) <= row['maximum'] <= 128, 'Invalid emitted work')
    require(all(type(token) is int and 0 <= token < 262144 for token in ids), 'Invalid token ID')
    require(all(type(t) is int and start <= t <= end for t in callbacks)
            and all(a <= b for a, b in zip(callbacks, callbacks[1:])), 'Invalid callback ordering')
    for state in ('before', 'after'):
        item = row[state]
        for key in ('history', 'committed', 'pending'):
            number(item[key], state + key, True)
        require(item['history'] == item['committed'] + item['pending'] <= 512, 'Invalid state accounting')
    footprint = row['footprint']
    current = number(footprint['physicalBytes'], 'footprint', True)
    peak = number(footprint['lifetimePeakPhysicalBytes'], 'peak', True)
    require(current <= peak <= 6 * 1024**3, 'Memory ceiling exceeded')
    require(type(row['thermalState']) is int and row['thermalState'] in range(4), 'Invalid thermal state')
    native = row.get('native')
    if row['mode'] == 'legacy' or row['status'] != 0:
        require(native is None, 'Unexpected native phases')
        return
    require(native is not None, 'Missing completed native report')
    for key in ('prefillSeconds', 'decodeSeconds', 'samplingSeconds', 'totalSeconds'):
        number(native[key], key)
    require(sum(native[k] for k in ('prefillSeconds', 'decodeSeconds', 'samplingSeconds')) <= native['totalSeconds'] + 1e-9,
            'Native phases exceed total')
    require(native['totalSeconds'] <= (end-start)/1e9 + 1e-6, 'Native total exceeds caller')
    for key in ('reusedTokens', 'prefillTokens', 'sampledTokens', 'emittedTokens', 'sessionInstance'):
        number(native[key], key, True)
    require(native['sessionInstance'] > 0 and native['emittedTokens'] == len(ids), 'Native owner/emission mismatch')
    require(len(ids) <= native['sampledTokens'] <= min(len(ids)+1,row['maximum']), 'Invalid sampled work')
    first = native.get('firstTokenSeconds')
    require((first is not None) == bool(ids), 'First-token availability mismatch')
    if first is not None:
        require(number(first, 'native TTFT') <= native['totalSeconds'], 'Invalid native TTFT')
    rate = native.get('prefillTokensPerSecond')
    expected = native['prefillTokens']/native['prefillSeconds'] if native['prefillTokens'] and native['prefillSeconds'] else None
    require((rate is None) == (expected is None), 'Invalid zero-work prefill rate')
    if expected is not None:
        require(math.isclose(number(rate, 'prefill rate'), expected, rel_tol=1e-12), 'Invalid prefill rate')


def latency(row):
    start = row['submittedUptimeNanoseconds']
    callbacks = row['callbackUptimeNanoseconds']
    return dict(total_ms=(row['completedUptimeNanoseconds']-start)/1e6,
                ttft_ms=(callbacks[0]-start)/1e6 if callbacks else None,
                decode_tokens_per_second=(len(callbacks)-1)/((callbacks[-1]-callbacks[0])/1e9)
                if len(callbacks) > 1 and callbacks[-1] > callbacks[0] else None)


def paired_report(rows):
    for row in rows:
        validate_row(row)
    groups = {name: [] for name in NAMES}
    require(len(rows) == 132, 'Expected two references plus 13 pairs for five workloads')
    require(rows[0]['workload'] == 'reference' and rows[1]['workload'] == 'edited_reference'
            and rows[0]['ids'] == [818,5279,529,7001], 'Invalid numerical references')
    prompt = rows[0]['input']
    require(len(prompt) == 15 and prompt[-1] != prompt[2], 'Invalid pinned input')
    edited = prompt[:-1] + [prompt[2]]
    require(rows[1]['input'] == edited, 'Invalid edited reference input')
    for name in NAMES:
        for trial in range(13):
            pair = [r for r in rows if r['workload'] == name and r['trial'] == trial]
            require(len(pair) == 2 and {r['mode'] for r in pair} == {'legacy', 'observed'}, 'Incomplete pair')
            off, on = next(r for r in pair if r['mode'] == 'legacy'), next(r for r in pair if r['mode'] == 'observed')
            require(all(r['warmup'] is (trial < 3) for r in pair), 'Invalid warmup label')
            require(all(off[k] == on[k] for k in ('input', 'operation', 'maximum', 'before', 'after', 'ids', 'reason', 'status')),
                    'Paired work/state differs')
            require(on['status'] == 0 and on['reason'] == 1 and on['after'] == dict(history=19, committed=18, pending=1),
                    'Unexpected fixed-token result')
            expected_request = {'reset':(3,prompt,4),'append':(1,prompt[-1:],4),'fully_cached':(0,prompt,4),
                                'pending':(2,[],3),'edited_prefix':(0,edited,4)}[name]
            require((on['operation'],on['input'],on['maximum']) == expected_request, 'Request input/operation changed')
            # These are the frozen setup states, not a general claim that an
            # arbitrary cancelled prefill can have at most one pending ID.
            before = {'reset':(19,18,1),'append':(14,14,0),'fully_cached':(15,15,0),
                      'pending':(16,15,1),'edited_prefix':(19,18,1)}[name]
            require(on['before'] == dict(zip(('history','committed','pending'),before)), 'Setup state differs from frozen work')
            expected_ids = rows[1]['ids'] if name == 'edited_prefix' else rows[0]['ids'][1:] if name == 'pending' else rows[0]['ids']
            require(on['ids'] == expected_ids, 'Reference output differs')
            # Gemma's ring/shared-state owner cannot rewind an edited committed
            # prefix. GenerationSession::set_tokens marks it for full replay;
            # unchanged cached/pending/append routes retain their valid state.
            expected_counts = {'reset': (0,15), 'append': (14,1), 'fully_cached': (15,0), 'pending': (15,1), 'edited_prefix': (0,15)}[name]
            native = on['native']
            require((native['reusedTokens'], native['prefillTokens']) == expected_counts, 'Reuse/prefill mismatch')
            require(native['sampledTokens'] == on['maximum'] == len(on['ids']), 'Fixed output changed')
            if trial >= 3:
                groups[name].append((off,on))
    result = {}
    for name, pairs in groups.items():
        metric = lambda mode, key: [latency(pair[mode])[key] for pair in pairs]
        phase = {key: statistics.median([pair[1]['native'][key] for pair in pairs])
                 for key in ('prefillSeconds', 'decodeSeconds', 'samplingSeconds', 'totalSeconds')}
        result[name] = dict(measured_per_mode=len(pairs), native_phase_median_seconds=phase,
            caller_median={mode: {key: statistics.median(metric(index,key)) for key in ('total_ms','ttft_ms','decode_tokens_per_second')}
                           for index,mode in enumerate(('legacy','observed'))},
            paired_overhead_median_pct={key: statistics.median([100*(latency(on)[key]/latency(off)[key]-1) for off,on in pairs])
                                        for key in ('total_ms','ttft_ms')})
    return result


def sustained_report(rows):
    require(len(rows) >= 4 and len(rows) <= 2048, 'Invalid sustained capture length')
    for index,row in enumerate(rows):
        validate_row(row)
        require(row['workload'] == 'sustained' and row['trial'] == index and row['warmup'] is (index < 3)
                and row['status'] == 0 and row['reason'] == 1 and row['mode'] == 'observed'
                and row['operation'] == 3 and len(row['input']) == 15 and row['maximum'] == 4
                and row['ids'] == [818,5279,529,7001]
                and row['after'] == dict(history=19,committed=18,pending=1)
                and (row['native']['reusedTokens'],row['native']['prefillTokens']) == (0,15), 'Sustained work changed')
        require(index == 0 or rows[index-1]['completedUptimeNanoseconds'] <= row['submittedUptimeNanoseconds'],
                'Sustained requests overlap or reorder')
        require(row['before'] == (dict(history=0,committed=0,pending=0) if index == 0 else rows[index-1]['after']),
                'Sustained state continuity differs')
    measured = rows[3:]
    wall = (measured[-1]['completedUptimeNanoseconds']-measured[0]['submittedUptimeNanoseconds'])/1e9
    active = sum(latency(r)['total_ms']/1000 for r in measured)
    require(wall >= 600 and active/wall >= .95, 'Insufficient sustained duration/duty cycle')
    totals = [latency(r)['total_ms'] for r in measured]
    return dict(measured_requests=len(measured), wall_seconds=wall, active_native_caller_seconds=active,
                duty_cycle=active/wall, total_ms_median=statistics.median(totals), total_ms_range=[min(totals),max(totals)],
                thermal_states=sorted({r['thermalState'] for r in measured}),
                process_lifetime_peak_bytes=max(r['footprint']['lifetimePeakPhysicalBytes'] for r in rows),
                tail_percentiles_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    sources = [*worker_sources(), ROOT/'apple/tests/swift/GemmaWorkloadMain.swift']
    inputs = [*sources,ROOT/'apple/include/toklet.h',Path(__file__).resolve(),ROOT/'tools/bench/request_cost_report.py',args.archive.resolve()]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    pins = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    commands, launch_samples = [], []
    binary = out/'gemma-workload'
    with (out/'compile.log').open('w') as log:
        command = ['xcrun','swiftc','-swift-version','6','-strict-concurrency=complete','-warnings-as-errors','-O',
                   '-I',ROOT/'apple/include','-module-cache-path',out/'ModuleCache',*sources,args.archive,
                   '-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation',
                   '-framework','Security','-lobjc','-liconv','-lresolv','-lc++','-o',binary]
        result = subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,timeout=300)
    require(result.returncode == 0,'Compilation failed; see compile.log')

    def run(label, mode):
        capture = out/(label+'.jsonl')
        command = list(map(str,[binary,args.model.resolve(),mode]))
        start = time.monotonic_ns(); received = None; total = 0; rows = []
        with capture.open('xb') as sink, (out/(label+'.stderr')).open('wb') as log:
            process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
            try:
                # A timeout also applies to a stalled/no-newline subprocess.
                import selectors
                selector = selectors.DefaultSelector(); selector.register(process.stdout,selectors.EVENT_READ)
                pending = b''; deadline = time.monotonic()+1800
                while True:
                    require(time.monotonic() < deadline,'Subprocess deadline exceeded')
                    if not selector.select(timeout=min(30,max(.01,deadline-time.monotonic()))):
                        continue
                    data = os.read(process.stdout.fileno(),65536)
                    if not data:
                        require(not pending,'Incomplete JSON line'); break
                    total += len(data); require(total <= CAPTURE_LIMIT,'Diagnostic capture limit exceeded')
                    sink.write(data); pending += data
                    while b'\n' in pending:
                        line,pending = pending.split(b'\n',1)
                        require(len(line) <= 65536,'Oversized diagnostic record')
                        row = parse(line); rows.append(row)
                        require(len(rows) <= 2049,'Diagnostic row limit exceeded')
                        if row['kind'] == 'ready':
                            require(received is None and len(rows) == 1,'Unexpected ready record')
                            received = time.monotonic_ns()
                    require(len(pending) <= 65536,'Oversized unfinished diagnostic record')
                status = process.wait(timeout=30)
                require(status == 0,'Workload failed: '+label)
            finally:
                selector.close(); process.stdout.close()
                if process.poll() is None:
                    process.kill(); process.wait(timeout=30)
        commands.append(dict(label=label,command=command,exit_code=status,start_monotonic_ns=start,end_monotonic_ns=time.monotonic_ns()))
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        require(received is not None,'Missing readiness acknowledgment')
        ready = rows[0]
        clocks = [number(ready[k],k,True) for k in ('entry','loadStart','loadEnd','sessionEnd','ready','prepared')]
        require(clocks == sorted(clocks),'Invalid load clock ordering')
        require(ready['admittedBytes'] <= 6*1024**3 and ready['after']['lifetimePeakPhysicalBytes'] <= 6*1024**3,'Load memory ceiling')
        launch_samples.append(dict(label=label,launch_to_ready_receipt_ms=(received-start)/1e6,
            entry_to_prepared_ms=(ready['prepared']-ready['entry'])/1e6,
            model_load_ms=(ready['loadEnd']-ready['loadStart'])/1e6,
            session_ms=(ready['sessionEnd']-ready['loadEnd'])/1e6,
            codec_cancel_ms=(ready['ready']-ready['sessionEnd'])/1e6,
            thermal_state=ready['thermalState'],process_lifetime_peak_bytes=ready['after']['lifetimePeakPhysicalBytes']))
        (out/'launch-samples.json').write_text(json.dumps(launch_samples,indent=2)+'\n')
        for row in rows[1:]: validate_row(row)
        return rows[1:]

    metadata = dict(format='toklet-o2-gemma-workloads-v1',source_sha256=pins,
        engine_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
        os=subprocess.check_output(['sw_vers'],text=True).strip(),
        power_before=subprocess.check_output(['pmset','-g','batt'],text=True).strip(),
        profile='google/gemma-4-E2B-it',revision='70af34e20bd4b7a91f0de6b22675850c43922a03',
        manifest_sha256='654dc6dfd66fd893a24f0faa39355b71d5e7ffc3f53c2dac394dfcfd9ed95298',
        capacity=512,chunk=32,precision='official BF16/G2, FP32 KV, FullLogits',
        cache_temperature='Uncontrolled filesystem/driver caches; each cold trial creates a new process; no cold-storage claim',
        scope='Raw numerical fixed-token native C API, separate from managed Chat/UI; observed report copies existing native timers',
        unavailable=['Conversion versus compilation within audited load','UI/render/final-answer timing','Measured DRAM/energy','p95/p99 qualification'],
        warmups=3,repetitions=10,launch_samples=launch_samples)
    (out/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n')
    cohorts = []
    controls = run('controls','controls')
    require(len(controls) == 7,'Missing controls')
    for batch in (1,2):
        for trial in range(13):
            cold = run(f'cold-{batch}-{trial:02d}','cold')
            require(len(cold) == 1 and cold[0]['ids'] == [818,5279,529,7001],'Cold output differs')
        paired = paired_report(run(f'paired-{batch}','paired'))
        cold_samples = [r for r in launch_samples if r['label'].startswith(f'cold-{batch}-') and int(r['label'].split('-')[-1]) >= 3]
        cohorts.append(dict(batch=batch,paired=paired,cold_medians_ms={key:statistics.median([r[key] for r in cold_samples])
            for key in ('launch_to_ready_receipt_ms','entry_to_prepared_ms','model_load_ms','session_ms','codec_cancel_ms')}))
        (out/'progress.json').write_text(json.dumps(dict(cohorts=cohorts,launch_samples=launch_samples),indent=2)+'\n')
    sustained = sustained_report(run('sustained','sustained'))
    require(all(sha(ROOT/path) == pin for path,pin in pins.items()),'Source/archive changed during measurements')
    metadata.update(cohorts=cohorts,sustained=sustained,passed=True,
                    power_after=subprocess.check_output(['pmset','-g','batt'],text=True).strip(),
                    default_summary_mode='disabled',overhead_qualified=False)
    (out/'result.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,result=str(out/'result.json'),sustained_seconds=sustained['wall_seconds'])))


if __name__ == '__main__':
    main()
