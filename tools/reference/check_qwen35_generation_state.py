"""Closed bounded Qwen generation-state and forward phase-baseline evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import check_qwen35_metal_benchmark as prior

need, exact, document, read = prior.need, prior.exact, prior.document, prior.read
LIMIT = 6 * 1024**3
CONTRACT = '693262545477dca892a7a7a4111d3d948c5051f102c66da4c97da584ae198e3a'
WITNESS = '65a949bdd0d5b46861a7eb2b4713e808522716497cba791a0c4e4a0c035c298c'
COMPILED = '7e106b91d54c091b565f37b30906317967a78e475b14769cb643e174e2844a5d'
TESTS = dict(
    tiny='model::qwen35::tests::metal::generation::prefix_retry_and_repair_preserve_pending_history_and_wrapper_budget',
    fidelity='model::qwen35::tests::metal::real::generation::actual_source_generation_state',
    baseline='model::qwen35::tests::metal::real::generation::matched_generation_step_baseline')
TAGS = dict(tiny='QWEN35_GENERATION_TINY', fidelity='QWEN35_GENERATION_STATE', baseline='QWEN35_GENERATION_BASELINE')


def digest(path):
    return hashlib.sha256(read(path, 128*1024**2)).hexdigest()


def labels(log, tag):
    prefixes = [tag+' ']
    prefixes += ['test '+TESTS[kind]+' ... '+tag+' ' for kind in TAGS if TAGS[kind] == tag]
    return [prior.context.prior.decode(line[len(prefix):]) for line in log.splitlines()
            for prefix in prefixes if line.startswith(prefix)]


def witness(root):
    need(digest(root/'acceptance-contract.json') == CONTRACT and digest(root/'generation-witness-contract.json') == WITNESS,
         'Frozen generation contracts differ')
    value = document(root/'generation-witness-contract.json')
    for field, pin in [('prior_source_receipt', 'prior_source_receipt_sha256'), ('prior_fidelity', 'prior_fidelity_sha256')]:
        need(digest(Path(value[field])) == value[pin], 'Frozen prior numerical evidence differs')
    evidence = document(Path(value['prior_fidelity']))
    need(evidence['coupled_projection_fidelity_passed'] is True and exact(evidence['failed_values'], 0),
         'Prior independent numerical prerequisite required')
    return value


def memory(report, expected):
    numerical = expected['numerical_memory']
    wrapper = dict(numerical=numerical, pending_history_bytes=expected['pending_history_bytes'],
                   wrapper_metadata_bytes=expected['wrapper_metadata_bytes'],
                   admitted_bytes=numerical['admitted_bytes']+expected['pending_history_bytes']+expected['wrapper_metadata_bytes'])
    need(exact(report['memory'], wrapper), 'Distinct numerical and wrapper owners must reconcile')
    combined = numerical['admitted_bytes']+wrapper['admitted_bytes']-numerical['weights_and_model_bytes']+128*1024**2
    need(exact(report['combined_admitted_bytes'], combined) and combined <= LIMIT
         and exact(report['diagnostic_allowance_bytes'], 128*1024**2), 'Shared model must count once in simultaneous admission')
    return combined


def closed(root, folder, kind):
    need(kind in TESTS, 'Named generation phase required')
    expected = witness(root)
    receipt = document(folder/'checks.json')
    need(receipt['passed'] is True and exact(receipt['exit_code'], 0) and receipt['error'] is None
         and receipt['group_absent'] is True and receipt['changed_inputs'] == [] and receipt['kind'] == kind,
         'Closed successful owned generation process required')
    peak = receipt['lifetime_peak_physical_bytes']
    need(type(peak) is int and 0 < peak <= LIMIT, 'Measured lifetime peak must fit6GiB')
    inventory = {'console.log', 'time-and-errors.log', 'inputs.json', 'environment.json', 'process.json'}
    need(set(receipt['files_sha256']) == inventory, 'Exact sealed native evidence inventory required')
    for name in inventory:
        need(digest(folder/name) == receipt['files_sha256'][name], 'Native evidence seal differs')
    raw = read(folder/'time-and-errors.log', 2*1024**2).decode()
    peaks = re.findall(r'^\s*(\d+)\s+peak memory footprint\s*$', raw, re.M)
    need(len(peaks) == 1 and int(peaks[0]) == peak, 'Raw instrument/receipt peak differs')
    inputs = document(folder/'inputs.json')
    need(inputs['format'] == 'toklet-qwen35-generation-native-v1' and inputs['kind'] == kind
         and inputs['contract_sha256'] == CONTRACT and inputs['witness_contract_sha256'] == WITNESS,
         'Frozen generation scope differs')
    need(inputs['build_file'] == 'native-build-3/checks.json'
         and inputs['build_sha256'] == digest(root/inputs['build_file']), 'Named native build binding differs')
    build = document(root/inputs['build_file'])
    need(digest(root/'compiled-profile-contract.json') == COMPILED, 'Frozen compiled profile differs')
    compiled = document(root/'compiled-profile-contract.json')
    need(compiled['build_sha256'] == inputs['build_sha256'] and compiled['binary_sha256'] == inputs['binary_sha256']
         and compiled['compiler_sha256'] == digest(root/'native-build-3/compiler.jsonl')
         and exact(compiled['profile'],dict(opt_level='3',debuginfo=0,debug_assertions=False,overflow_checks=False,test=True))
         and exact(compiled['features'],['default','metal']) and compiled['rustflags'] == [], 'Qualified release build required')
    need(build['passed'] is True and exact(build['source_pins'], inputs['source_pins'])
         and exact(receipt['source_pin_count'], len(build['source_pins']))
         and inputs['binary_sha256'] == receipt['binary_sha256'] == build['binary_sha256'] == digest(Path(build['binary'])),
         'Compiled source and retained native binary differ')
    need(exact(inputs['command'], ['/usr/bin/time', '-l', build['binary'], TESTS[kind], '--exact', '--ignored', '--nocapture', '--test-threads=1']),
         'Exact owned native test command required')
    need(exact(inputs['process_limit_bytes'], LIMIT) and exact(inputs['external_environment_sample_seconds'], 30),
         'Unchanged process cap and external sampling required')
    need(inputs['driver_sha256'] == digest(root/'run-native-2.py'), 'Native driver differs')
    process = document(folder/'process.json')
    need(exact(process, receipt['process']) and set(process) == {'pid', 'group', 'parent_pid'}
         and all(type(v) is int and v > 0 for v in process.values())
         and process['pid'] == process['group'] and process['pid'] != process['parent_pid'], 'Owned process identity differs')
    need(exact(inputs['allocator_environment'], expected['allocator_environment']), 'Declared owned-child allocator profile differs')
    inherited = inputs['inherited_allocator_environment']
    need(type(inherited) is dict and set(inherited) == set(expected['allocator_environment'])
         and all(v is None or type(v) is str for v in inherited.values()), 'Inherited allocator values must remain explicit')
    duration = receipt['elapsed_seconds']
    need(type(duration) in [int, float] and 0 < duration < 330, 'Bounded owned process duration required')
    log = read(folder/'console.log', 2*1024**2).decode()
    reports = labels(log, TAGS[kind])
    need(len(reports) == 1 and exact(reports[0], receipt['report']) and '1 passed; 0 failed' in log,
         'One matching footer and successful explicit test required')
    report = reports[0]
    common = {'source_sha256','tokens','position_offset','capacity','chunk','memory','combined_admitted_bytes',
              'diagnostic_allowance_bytes','allocator_environment','instrumentation_overhead_qualified',
              'quality_qualified','managed_adoption_qualified'}
    if kind == 'fidelity':
        need(set(report) == common|{'load_report','full_state','failed_state','retry_state','snapshot','changed_snapshot',
                                    'full_logits_bits_exact','warmed_rust_allocations','prefix_failure_repair_retry_exact',
                                    'reset_discard_exact','deeper_context_qualified','request_metrics_qualified'},
             'Exact generation state footer schema required')
    elif kind == 'baseline':
        need(set(report) == common|{'direct_memory','cohort','warmups','measurements','scope','request_ttft_qualified'},
             'Exact generation baseline footer schema required')
    if kind != 'baseline':
        need(inputs['cohort'] is None and receipt['cohort'] is None, 'Untimed control has no cohort')
    if kind != 'tiny':
        need(exact(report['allocator_environment'], expected['allocator_environment']), 'Actual child allocator settings differ')
        for field, value in dict(source_sha256=prior.context.producer.pin.FILE_SHA, tokens=expected['inputs'],
                                 position_offset=37, capacity=512, chunk=32, instrumentation_overhead_qualified=False,
                                 quality_qualified=False, managed_adoption_qualified=False).items():
            need(exact(report[field], value), 'Frozen numerical/generation scope differs: '+field)
        memory(report, expected)
    return receipt, report, inputs, log, expected


def state(history, committed, action, work):
    return dict(history_tokens=history, committed_tokens=committed, pending_tokens=history-committed,
                prefix_action=action, last_numerical_work=work)


def tiny(root, folder):
    receipt, report, _, _, _ = closed(root, folder, 'tiny')
    need(set(report) == {'prefix_retry_bits_exact','pending_history_preserved','repair_overlap_budget_rejected',
                         'warmed_rust_allocations','memory'}, 'Exact tiny control scope required')
    for field in ['prefix_retry_bits_exact','pending_history_preserved','repair_overlap_budget_rejected']:
        need(report[field] is True, 'Tiny generation recovery control failed')
    need(exact(report['warmed_rust_allocations'],0), 'Zero warmed Rust allocations required')
    owner=report['memory']; numerical=owner['numerical']
    need(all(type(v) is int and v>0 for v in numerical.values())
         and sum(v for k,v in numerical.items() if k!='admitted_bytes') == numerical['admitted_bytes'],
         'Tiny numerical owners must reconcile')
    need(exact(owner['pending_history_bytes'],68) and exact(owner['wrapper_metadata_bytes'],208)
         and exact(owner['admitted_bytes'],numerical['admitted_bytes']+276), 'Tiny wrapper owners must reconcile')
    return dict(passed=True,peak_bytes=receipt['lifetime_peak_physical_bytes'],prefix_retry_bits_exact=True,
                repair_overlap_budget_rejected=True,warmed_rust_allocations=0)


def fidelity(root, folder):
    receipt, report, _, _, expected = closed(root, folder, 'fidelity')
    need(exact(report['load_report'], expected['load_report']), 'Authenticated consuming load ledger differs')
    fields = dict(full_state=state(17, 17, 'FullyCached', prior.context.work(17)),
                  failed_state=state(17, 5, 'ShortenedReplay', {**prior.context.work(0, 'Error'), 'gathers_completed': 1}),
                  retry_state=state(17, 17, 'ShortenedReplay', prior.context.work(12)),
                  snapshot=expected['state_sha256'], full_logits_bits_exact=True, warmed_rust_allocations=0,
                  prefix_failure_repair_retry_exact=True, reset_discard_exact=True,
                  deeper_context_qualified=False, request_metrics_qualified=False)
    for key, value in fields.items():
        need(exact(report[key], value), 'Complete generation state/recovery witness differs: '+key)
    need(type(report['changed_snapshot']) is str and re.fullmatch('[a-f0-9]{64}', report['changed_snapshot'])
         and report['changed_snapshot'] != report['snapshot'], 'Changed prompt needs distinct full state witness')
    return dict(passed=True,source_inputs=17,position_offset=37,state_sha256=report['snapshot'],
                native_peak_bytes=receipt['lifetime_peak_physical_bytes'],generation_state_qualified=True,
                deeper_context_qualified=False,request_metrics_qualified=False,instrumentation_overhead_qualified=False,
                quality_qualified=False,managed_adoption_qualified=False)


def samples(rows, cohort):
    need(type(rows) is list and len(rows) == 13, 'Three warmups and ten measured pairs required')
    result = {key: [] for key in ['numerical_ns', 'generation_ns']}
    for i, row in enumerate(rows):
        fields = dict(iteration=i,warmup=i<3,cohort=cohort,first_role='generation' if (i+cohort)%2 else 'numerical',
                      inputs=17,position_offset=37,warmed_rust_allocations=0,all_state_logits_bits_exact=True)
        need(set(row) == set(fields)|set(result), 'Exact paired sample schema required')
        for key, value in fields.items():
            need(exact(row[key], value), 'Frozen sample work/order differs: '+key)
        for key in result:
            need(type(row[key]) is int and row[key] > 0, 'Positive completed phase nanoseconds required')
            if i >= 3:
                result[key].append(row[key])
    return {key:dict(samples_ns=v,median_ns=statistics.median(v),mad_ns=statistics.median(abs(x-statistics.median(v)) for x in v),
                     minimum_ns=min(v),maximum_ns=max(v)) for key,v in result.items()}


def baseline(root, first, second):
    need(first != second, 'Distinct confirmation folders required')
    need(document(root/'fidelity-final.json')['generation_state_qualified'] is True, 'State prerequisite must qualify first')
    runs=[]; identities=[]; environments=[]
    for cohort, folder in enumerate([first, second]):
        receipt, report, inputs, log, expected = closed(root, folder, 'baseline')
        need(exact(report['cohort'],cohort) and exact(inputs['cohort'],cohort) and exact(receipt['cohort'],cohort)
             and exact(report['warmups'],3) and exact(report['measurements'],10), 'Paired cohort/repetition identity differs')
        need(exact(report['direct_memory'],expected['numerical_memory']) and report['request_ttft_qualified'] is False,
             'Direct owner/scope differs')
        need(report['scope'] == 'Completed17-input forward with resident weights; resets outside timer, wrapper timer includes set_tokens and pending-drain loop; diagnostic compare/output outside; no request or instrumentation-overhead qualification', 'Exact timed scope required')
        need(inputs['fidelity_sha256'] == digest(root/'fidelity-final.json'), 'Pre-baseline fidelity binding differs')
        rows=labels(log,'QWEN35_GENERATION_SAMPLE'); phases=samples(rows,cohort)
        need(sum(r['numerical_ns']+r['generation_ns'] for r in rows) < receipt['elapsed_seconds']*1e9, 'Timers exceed owned process duration')
        environment=document(folder/'environment.json')
        need(prior.conditions(environment,receipt['elapsed_seconds']), 'Controlled power/thermal/pressure timeline required')
        runs.append(dict(cohort=cohort,phases=phases,peak_bytes=receipt['lifetime_peak_physical_bytes'],environment_qualified=True))
        identities.append((receipt['process']['pid'],inputs['binary_sha256'],inputs['source_pins']))
        environments.append(environment)
    need(identities[0][0] != identities[1][0] and identities[0][1:] == identities[1][1:], 'Confirmation needs distinct process with same native inputs')
    need(all(exact(environments[0]['start'][key],environments[1]['start'][key]) for key in ['hardware','os','configuration']), 'Confirmation hardware/OS/configuration differs')
    return dict(passed=True,runs=runs,warmups_per_role_total=6,measurements_per_role_total=20,
                scope='Resident17-input numerical and generation-step phase baselines only; explicit owned-process allocator profile',
                speedup_qualified=False,request_ttft_qualified=False,instrumentation_overhead_qualified=False,
                sustained_qualified=False,quality_qualified=False,managed_adoption_qualified=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('first',type=Path)
    parser.add_argument('--second',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=baseline(args.root,args.first,args.second) if args.second else fidelity(args.root,args.first)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result))
