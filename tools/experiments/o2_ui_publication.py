#!/usr/bin/env python3
"""Explicit Gemma controller publication checks and paired cost; no screen capture."""
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
from o1_gemma_summary import comparison

sys.path.insert(0,str(ROOT/'tools/bench'))
from request_cost_report import parse, read_bounded, require, number, summaries, digest

TEXT_SHA256 = hashlib.sha256(b'The capital of France is **Paris**.').hexdigest()
TOKENS = [818,5279,529,7001,563,5213,50429,84750]


def clock(value):
    require(type(value) is int and 0 <= value <= 2**64-1,'Invalid UI host clock')
    return value


def milliseconds(value, expected):
    require(math.isclose(number(value,'UI duration'),expected,rel_tol=1e-12,abs_tol=1e-6),'UI duration inconsistent')


def analyse_ui_batch(folder, build_id):
    require(digest(build_id,64),'Invalid UI build identity')
    raw = parse(read_bounded(folder/'samples.json',1024*1024))
    require(type(raw) is list and len(raw) == 26,'Expected 13 paired UI trials')
    manifest,records = summaries(folder/'summaries.jsonl',build_id)
    require(number(manifest['uiUpdateErrors'],'UI update errors',True) == 0 and len(records) == 13,'UI summary coverage incomplete')
    require(manifest['profile'] == 'google/gemma-4-E2B-it BF16 managed Mac G2'
            and manifest['revision'] == '70af34e20bd4b7a91f0de6b22675850c43922a03'
            and manifest['modelAndCodecManifestSHA256'] == '654dc6dfd66fd893a24f0faa39355b71d5e7ffc3f53c2dac394dfcfd9ed95298',
            'Wrong UI model identity')
    seen,all_ids,measured = set(),set(),[]
    for sample in raw:
        identity = sample['request_id']
        require(isinstance(identity,str) and identity and identity not in all_ids,'Duplicate UI request')
        all_ids.add(identity)
        require(type(sample['trial']) is int and 0 <= sample['trial'] < 13
                and sample['warmup'] is (sample['trial'] < 3),'Invalid UI warmup labels')
        require(sample['mode'] in ('disabled','summary'),'Invalid UI collection mode')
        start,end = clock(sample['submitted_uptime_ns']),clock(sample['completed_uptime_ns'])
        require(end > start,'Invalid UI caller interval')
        milliseconds(sample['total_ms'],(end-start)/1e6)
        require(0 < number(sample['ttft_ms'],'UI caller TTFT') <= sample['total_ms'],'Invalid caller TTFT')
        for key in ('history','committed','pending'):
            number(sample[key],'UI state '+key,True)
        require(sample['tokens'] == TOKENS and (sample['history'],sample['committed'],sample['pending']) == (24,23,1)
                and sample['text_sha256'] == TEXT_SHA256,'UI work/text/state differs')
        current = number(sample['footprint_current_bytes'],'UI footprint',True)
        peak = number(sample['process_lifetime_peak_bytes'],'UI lifetime peak',True)
        require(0 < current <= peak <= 6*1024**3,'UI component memory ceiling exceeded')
        require(type(sample['thermal_state']) is int and sample['thermal_state'] in range(4),'Invalid UI thermal state')
        if sample['mode'] != 'summary':continue
        require(identity in records and identity not in seen,'Missing UI correlation')
        seen.add(identity);row = records[identity];ui = row['ui']
        for key in ('history','committed','pending'):
            number(row['after'][key],'UI summary state '+key,True)
            require(row['after'][key] == sample[key],'UI summary/raw state differs')
        native = row['nativeMetrics']
        work = [number(row[key],'UI '+key,True) for key in ('promptTokens','callbackTokens','nativeEmittedTokens')]
        work += [number(native[key],'UI native '+key,True)
                 for key in ('prefillTokens','reusedTokens','sampledTokens','emittedTokens')]
        require(work == [15,8,8,15,0,9,8],'UI fixed prompt/native work differs')
        require((row['capacity'],row['chunk'],row['requestedOutputTokens']) == (512,32,32),'UI workload configuration changed')
        begin = clock(ui['submittedUptimeNanoseconds']);handled = clock(ui['handledUptimeNanoseconds'])
        published = clock(ui['publicationUptimeNanoseconds'])
        worker_start,worker_end = row['submittedUptimeNanoseconds'],row['completedUptimeNanoseconds']
        require(start <= begin <= worker_start <= worker_end <= handled == published <= end,'UI clock nesting invalid')
        require(ui['outcome'] == 'published' and ui['selectedConversation'] is True and ui['nonemptyText'] is True,
                'UI did not publish selected nonempty text')
        expected = dict(preparationMilliseconds=(worker_start-begin)/1e6,
            callbackTTFTMilliseconds=(row['firstCallbackNanoseconds']-begin)/1e6,
            terminalHandlingMilliseconds=(handled-begin)/1e6,
            workerToHandlingMilliseconds=(handled-worker_end)/1e6,
            publicationMilliseconds=(published-begin)/1e6,
            firstNonemptyPublicationMilliseconds=(published-begin)/1e6,
            firstAnswerPublicationMilliseconds=(published-begin)/1e6)
        for key,value in expected.items():milliseconds(ui[key],value)
        milliseconds(sample['ttft_ms'],(row['firstCallbackNanoseconds']-start)/1e6)
        if not sample['warmup']:measured.append(ui)
    require(seen == set(records) and len(measured) == 10,'Unmatched UI records')
    return dict(comparison=comparison(raw),measured_per_mode=10,
        publication_median_ms={key:statistics.median([row[key] for row in measured]) for key in expected},
        process_lifetime_peak_bytes=max(row['process_lifetime_peak_bytes'] for row in raw),
        thermal_states=sorted({row['thermal_state'] for row in raw}),render_measured=False,
        scope='Send handler to controller publication/100ms poll acknowledgment; no window/render timing or whole-app memory qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    parser.add_argument('archive',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args();out = args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    harness = sorted((ROOT/'apple/Harness').glob('*.swift'))
    harness = [p for p in harness if p.name != 'HarnessApp.swift']
    entries = [ROOT/'apple/tests/swift/UIPublicationMain.swift',ROOT/'apple/tests/swift/UIPublicationLoadedMain.swift']
    inputs = [*harness,*entries,ROOT/'apple/include/toklet.h',ROOT/'apple/include/module.modulemap',
              Path(__file__).resolve(),ROOT/'tools/experiments/o1_gemma_summary.py',ROOT/'tools/bench/request_cost_report.py',args.archive.resolve()]
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    pins = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    build_id = hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()
    env = dict(os.environ,TOKLET_BUILD_ID=build_id)
    commands = []
    def run(label,command,timeout=600):
        start = time.monotonic_ns()
        with (out/(label+'.log')).open('w') as log:
            result = subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        commands.append(dict(label=label,command=list(map(str,command)),exit_code=result.returncode,
                             start_monotonic_ns=start,end_monotonic_ns=time.monotonic_ns()))
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        require(result.returncode == 0,label+' failed; see log')
    flags = ['xcrun','swiftc','-swift-version','6','-strict-concurrency=complete','-warnings-as-errors','-parse-as-library','-O',
             '-I',ROOT/'apple/include','-module-cache-path',out/'ModuleCache']
    links = [args.archive,'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation',
             '-framework','Security','-lobjc','-liconv','-lresolv','-lc++']
    run('compile-unit',[*flags,*worker_sources(),entries[0],*links,'-o',out/'unit'])
    run('unit',[out/'unit'])
    run('compile-loaded',[*flags,*harness,entries[1],*links,'-o',out/'loaded'])
    metadata = dict(format='toklet-o2-ui-publication-v1',source_sha256=pins,build_id=build_id,
        archive_sha256=sha(args.archive),base_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
        os=subprocess.check_output(['sw_vers'],text=True).strip(),
        power_before=subprocess.check_output(['pmset','-g','batt'],text=True).strip())
    (out/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n')
    run('controls',[out/'loaded',args.model.resolve(),out/'controls','controls'])
    cohorts = []
    for repeat in (1,2):
        folder = out/f'paired-{repeat}'
        run(f'paired-{repeat}',[out/'loaded',args.model.resolve(),folder,'paired'])
        cohorts.append(analyse_ui_batch(folder,build_id))
    require(all(sha(ROOT/name) == pin for name,pin in pins.items()),'UI measurement inputs changed')
    qualified = all(c['comparison'][metric]['median_ratio_overhead_pct'] <= 1 and
                    c['comparison'][metric]['paired_median_bootstrap_95_pct'][1] <= 1
                    for c in cohorts for metric in ('ttft_ms','total_ms'))
    metadata.update(cohorts=cohorts,component_overhead_qualified=qualified,summary_overhead_qualified=False,
                    default_mode='disabled',render_measured=False,passed=True,
                    power_after=subprocess.check_output(['pmset','-g','batt'],text=True).strip(),
                    limitations=['Controller fixture creates no window; publication is not rendered visibility',
                                 '100ms poll acknowledgment includes persistence after publication',
                                 'Native worker/headless overhead and actual rendering remain separate qualifications',
                                 'Fresh isolated test history only; no physical iPhone or screen capture'])
    (out/'result.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,component_overhead_qualified=qualified,render_measured=False)))


if __name__ == '__main__':main()
