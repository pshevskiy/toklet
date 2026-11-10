"""Strict completed text-request/parity reader; source quality stays separate."""
import argparse
import hashlib
import json
from pathlib import Path

import minicpm5_performance as common
from minicpm5_profile import timestamp_grain_ns

need, exact = common.need, common.exact


def integer(value, lower=0):
    return type(value) is int and lower <= value < 2 ** 63


def owner(value):
    elapsed = common.interval(value)
    need(integer(value.get('final_wait4_peak_rss_bytes'), 1)
         and value['final_wait4_peak_rss_bytes'] <= 6 * 1024 ** 3
         and integer(value.get('sampled_lifetime_physical_peak_bytes'), 1)
         and value['sampled_lifetime_physical_peak_bytes'] <= 6 * 1024 ** 3
         and integer(value.get('minimum_observed_free_bytes'))
         and value['minimum_observed_free_bytes'] >= 2 * 1024 ** 3
         and value.get('final_physical_peak_available') is False,
         'Text owner resource/physical scope differs')
    return elapsed


def read(path, owned):
    with Path(path).open('rb') as stream:
        raw = stream.read(16 * 1024 ** 2 + 1)
    need(len(raw) <= 16 * 1024 ** 2 and hashlib.sha256(raw).hexdigest()
         == owned.get('text_report_sha256'), 'Text report differs from completed owned receipt')
    return json.loads(raw, object_pairs_hook=common.unique)


def memory(actual, expected):
    need(exact(actual, expected) and all(integer(v) for v in actual.values())
         and actual['admitted_bytes'] == sum(v for k, v in actual.items() if k != 'admitted_bytes'),
         'Text session owner ledger differs')


def external(contract):
    values = [contract[key] for key in ['codec_allowance_bytes', 'stream_allowance_bytes', 'application_allowance_bytes']]
    need(exact(values, [536870912, 4194304, 33554432]), 'Text external allowance differs')
    return sum(values)


def validate_parity(report, owned, contract, workload):
    owner(owned);work = contract['workloads'][workload];reserve = external(contract)
    need(type(report) is dict and set(report) == {'format', 'profile', 'capacity', 'prefill_chunk',
         'position_offset', 'stop_ids', 'repository', 'revision', 'prompt_ids', 'sampled_ids', 'text',
         'maximum_new_tokens', 'combined_admitted_bytes', 'shared_model_bytes', 'serial_memory',
         'parallel_memory', 'codec_allowance_bytes', 'stream_allowance_bytes',
         'application_allowance_bytes', 'steps', 'pending_after', 'history_after', 'committed_after',
         'model_quality_adoption_qualified'}, 'Text parity fields differ')
    need(report['format'] == 'toklet-minicpm5-text-parity-v1'
         and report['repository'] == common.REPOSITORY and report['revision'] == common.REVISION
         and report['profile'] == common.PROFILE and report['model_quality_adoption_qualified'] is False
         and exact(report['prompt_ids'], work['prompt_ids']) and exact(report['stop_ids'], [1, 130073]),
         'Text parity identity/quality differs')
    for key, expected in [('capacity', 512), ('prefill_chunk', 32), ('position_offset', 0),
                          ('maximum_new_tokens', work['maximum_new_tokens']), ('shared_model_bytes', 2674218150),
                          ('codec_allowance_bytes', 536870912), ('stream_allowance_bytes', 4194304),
                          ('application_allowance_bytes', 33554432)]:
        need(exact(report[key], expected), 'Text parity geometry/owner differs')
    memory(report['serial_memory'], contract['expected_session_memory']['serial'])
    memory(report['parallel_memory'], contract['expected_session_memory']['parallel-store'])
    combined = report['serial_memory']['admitted_bytes'] + report['parallel_memory']['admitted_bytes'] - 2674218150 + reserve
    need(exact(report['combined_admitted_bytes'], combined) and combined <= 6 * 1024 ** 3,
         'Text parity shared-model accounting differs')
    sampled = report['sampled_ids'];maximum = work['maximum_new_tokens'];prompt = len(work['prompt_ids'])
    need(type(sampled) is list and 1 <= len(sampled) <= maximum
         and all(integer(v) and v < 130560 for v in sampled)
         and not any(v in [1, 130073] for v in sampled[:-1])
         and (sampled[-1] in [1, 130073] or len(sampled) == maximum)
         and type(report['text']) is str and len(report['text'].encode()) <= 65536,
         'Text parity sampled/stop/decoded sequence differs')
    need(type(report['steps']) is list and len(report['steps']) == len(sampled), 'Text parity steps missing')
    for i, row in enumerate(report['steps']):
        need(type(row) is dict and set(row) == {'step', 'logits_values', 'logits_sha256', 'all_logits_exact',
             'selected', 'committed_after', 'pending_after'} and row['all_logits_exact'] is True
             and type(row['logits_sha256']) is str and len(row['logits_sha256']) == 64
             and all(c in '0123456789abcdef' for c in row['logits_sha256']), 'Text complete-logit parity witness differs')
        for key, expected in [('step', i), ('logits_values', 130560), ('selected', sampled[i]),
                              ('committed_after', prompt + i), ('pending_after', 1)]:
            need(exact(row[key], expected), 'Text parity committed/pending history differs')
    for key, expected in [('pending_after', 1), ('history_after', prompt + len(sampled)),
                          ('committed_after', prompt + len(sampled) - 1)]:
        need(exact(report[key], expected), 'Text parity terminal state differs')
    return dict(sampled_ids=sampled, text=report['text'], source_numerical_qualification_extended=False)


def validate(report, owned, contract, workload, variant, expected):
    elapsed = owner(owned);need(variant in ['serial', 'parallel-store'], 'Unknown text implementation')
    work = contract['workloads'][workload];reserve = external(contract)
    need(type(expected) is dict and set(expected) == {'sampled_ids', 'text', 'source_numerical_qualification_extended'}
         and expected['source_numerical_qualification_extended'] is False
         and type(expected['sampled_ids']) is list and 1 <= len(expected['sampled_ids']) <= work['maximum_new_tokens']
         and all(integer(v) and v < 130560 for v in expected['sampled_ids'])
         and not any(v in [1, 130073] for v in expected['sampled_ids'][:-1])
         and (expected['sampled_ids'][-1] in [1, 130073] or len(expected['sampled_ids']) == work['maximum_new_tokens'])
         and type(expected['text']) is str and len(expected['text'].encode()) <= 65536,
         'Expected native text sequence/scope invalid')
    fields = {'format', 'repository', 'revision', 'device', 'profile', 'normalization_implementation',
              'tokenizer_sha256', 'options', 'messages_sha256', 'options_sha256', 'prompt_ids', 'stop_ids',
              'maximum_new_tokens', 'capacity', 'prefill_chunk', 'position_offset', 'warmups', 'measured_samples',
              'cache_state', 'load_timing', 'codec_load_ns', 'preflight_text_ns', 'session_setup_ns',
              'session_memory', 'codec_allowance_bytes', 'stream_allowance_bytes', 'application_allowance_bytes',
              'maximum_admitted_bytes', 'samples', 'request_clock_scope', 'first_text_scope',
              'diagnostics_enabled', 'operator_counts_available', 'allocation_count', 'console_publication_ns',
              'user_visible_ttft_ns', 'model_quality_qualified', 'managed_adoption_qualified',
              'ordinary_telemetry_overhead_qualified'}
    implementation = 'serial' if variant == 'serial' else 'serial-reduction-parallel-store'
    need(type(report) is dict and set(report) == fields, 'Text request report fields differ')
    need(report['format'] == 'toklet-minicpm5-text-request-v1' and report['repository'] == common.REPOSITORY
         and report['revision'] == common.REVISION and report['profile'] == common.PROFILE
         and report['device'] == 'Apple M1 Pro' and report['normalization_implementation'] == implementation
         and report['tokenizer_sha256'] == contract['tokenizer_sha256']
         and exact(report['options'], work['options']) and exact(report['prompt_ids'], work['prompt_ids'])
         and report['messages_sha256'] == work['messages_sha256'] and report['options_sha256'] == work['options_sha256']
         and exact(report['stop_ids'], [1, 130073]), 'Text identity/input/implementation differs')
    for key, value in [('maximum_new_tokens', work['maximum_new_tokens']), ('capacity', 512), ('prefill_chunk', 32),
                        ('position_offset', 0), ('warmups', 3), ('measured_samples', 10),
                        ('codec_allowance_bytes', 536870912), ('stream_allowance_bytes', 4194304),
                        ('application_allowance_bytes', 33554432)]:
        need(exact(report[key], value), 'Text geometry/resource contract differs')
    for key in ['diagnostics_enabled', 'operator_counts_available', 'model_quality_qualified',
                'managed_adoption_qualified', 'ordinary_telemetry_overhead_qualified']:
        need(report[key] is False, 'Unqualified text diagnostic/adoption claim')
    for key in ['allocation_count', 'console_publication_ns', 'user_visible_ttft_ns']:
        need(report[key] is None, 'Unavailable text metric fabricated')
    need(report['cache_state'] == 'one loaded model/codec/session;reset-empty-history-per-request;warm-filesystem'
         and report['request_clock_scope'] == contract['request_clock_scope']
         and report['first_text_scope'] == contract['first_text_scope'], 'Text clock/cache scope differs')
    memory(report['session_memory'], contract['expected_session_memory'][variant])
    need(exact(report['maximum_admitted_bytes'], report['session_memory']['admitted_bytes'] + reserve)
         and report['maximum_admitted_bytes'] <= 6 * 1024 ** 3, 'Text aggregate admission differs')
    setup = 0
    for key in ['codec_load_ns', 'preflight_text_ns', 'session_setup_ns']:
        need(integer(report[key]) and report[key] <= elapsed, 'Text setup clock invalid');setup += report[key]
    sampled = expected['sampled_ids'];prompt = len(work['prompt_ids']);count = len(sampled)
    need(1 <= count <= work['maximum_new_tokens'], 'Expected native text sequence invalid')
    need(type(report['samples']) is list and len(report['samples']) == 13, 'Text request coverage incomplete')
    derived = [];total = 0;validation = 0;first_gpu = None;previous_gpu = 0.
    for i, sample in enumerate(report['samples']):
        need(type(sample) is dict and set(sample) == {'sample', 'warmup', 'request_ns', 'render_ns', 'encode_ns',
             'stream_setup_ns', 'stream_finish_ns', 'finish_bytes', 'first_decoded_text_ns', 'full_decode_validation_ns',
             'input_tokens', 'sampled_ids', 'text', 'chunks', 'decisions', 'committed_after', 'pending_after',
             'history_after', 'stop_reason'}, 'Text sample fields differ')
        need(exact(sample['sample'], i) and sample['warmup'] is (i < 3)
             and exact(sample['sampled_ids'], sampled) and sample['text'] == expected['text']
             and sample['stop_reason'] == ('eos' if sampled[-1] in [1, 130073] else 'limit'),
             'Text sample order/history/output/stop differs')
        for key, value in [('input_tokens', prompt), ('committed_after', prompt + count - 1),
                            ('pending_after', 1), ('history_after', prompt + count)]:
            need(exact(sample[key], value), 'Text final committed/pending state differs')
        request = sample['request_ns'];need(integer(request, 1) and request <= elapsed, 'Text request clock invalid')
        for key in ['render_ns', 'encode_ns', 'stream_setup_ns', 'stream_finish_ns', 'full_decode_validation_ns']:
            need(integer(sample[key]) and sample[key] <= elapsed, 'Text phase clock invalid')
        prefix = sample['render_ns'] + sample['encode_ns'] + sample['stream_setup_ns']
        need(type(sample['chunks']) is list and len(sample['chunks']) == (prompt + 31) // 32 + count - 1
             and type(sample['decisions']) is list and len(sample['decisions']) == count,
             'Text completed chunk/decision coverage differs')
        cursor = 0;committed = 0;previous = 0;previous_decode = 0;text_bytes = 0;walls = 0;selects = 0;decodes = 0;gpu = 0.;commands = 0;prefill = 0;request_first_gpu = None;first_text_step = None
        for step, decision in enumerate(sample['decisions']):
            need(type(decision) is dict and set(decision) == {'step', 'selected', 'decision_ns', 'selection_ns',
                 'decode_ns', 'delta_bytes', 'text_bytes', 'first_chunk', 'chunks', 'committed_after', 'pending_after'},
                 'Text decision fields differ')
            expected_rows = []
            if step == 0:
                left = prompt
                while left:
                    n = min(32, left);expected_rows.append(n);left -= n
            else: expected_rows = [1]
            for key, value in [('step', step), ('selected', sampled[step]), ('first_chunk', cursor),
                                ('chunks', len(expected_rows)), ('committed_after', prompt + step), ('pending_after', 1)]:
                need(exact(decision[key], value), 'Text decision work/state/selection differs')
            decision_walls = 0
            for rows in expected_rows:
                chunk = sample['chunks'][cursor];committed += rows
                need(type(chunk) is dict and set(chunk) == {'decision', 'rows', 'committed_after', 'pending_after', 'wall_ns', 'completed_timing'}, 'Text chunk fields differ')
                for key, value in [('decision', step), ('rows', rows), ('committed_after', committed),
                                    ('pending_after', max(0, prompt - committed) if step == 0 else 0)]:
                    need(exact(chunk[key], value), 'Text completed chunk work/prefix differs')
                wall = chunk['wall_ns'];timing = chunk['completed_timing']
                need(integer(wall, 1) and type(timing) is dict
                     and set(timing) == set(common.TIMING_FIELDS) | {'command_count'}
                     and all(common.finite(timing[k]) and timing[k] >= 0 for k in common.TIMING_FIELDS)
                     and exact(timing['command_count'], 1), 'Text completed timing invalid')
                host = sum(common.duration_ns(timing[key]) for key in common.TIMING_FIELDS if not key.startswith('gpu_'))
                begin, end = timing['gpu_start_seconds'], timing['gpu_end_seconds']
                need(host <= wall and previous_gpu <= begin < end and (end - begin) * 10 ** 9 <= wall,
                     'Text host sum/GPU coverage/serial order invalid')
                timestamp_grain_ns(begin, end)
                first_gpu = begin if first_gpu is None else first_gpu;request_first_gpu = begin if request_first_gpu is None else request_first_gpu;previous_gpu = end
                walls += wall;decision_walls += wall;gpu += end - begin;commands += 1;cursor += 1
            if step == 0: prefill = decision_walls
            event, selection, decode = decision['decision_ns'], decision['selection_ns'], decision['decode_ns']
            need(integer(event, 1) and integer(selection) and integer(decode)
                 and previous < event <= request
                 and previous + (prefix if step == 0 else previous_decode) + decision_walls + selection <= event,
                 'Text disjoint decision/phase clocks exceed event window')
            need(integer(decision['delta_bytes']) and integer(decision['text_bytes'])
                 and decision['text_bytes'] == text_bytes + decision['delta_bytes'], 'Text incremental byte counters differ')
            if decision['delta_bytes'] and first_text_step is None:first_text_step = step
            text_bytes = decision['text_bytes'];previous = event;previous_decode = decode;selects += selection;decodes += decode
        need(integer(sample['finish_bytes']) and len(sample['text'].encode()) == text_bytes + sample['finish_bytes']
             and text_bytes + sample['finish_bytes'] <= 65536, 'Text final decoded bytes differ')
        first_text = sample['first_decoded_text_ns']
        if text_bytes + sample['finish_bytes'] == 0:
            need(first_text is None, 'Empty decoded response has fabricated first text')
        else:
            low = sample['decisions'][first_text_step]['decision_ns'] if first_text_step is not None else previous + previous_decode
            if first_text_step is not None and first_text_step + 1 < count:
                following = sample['decisions'][first_text_step + 1]
                following_chunks = sample['chunks'][following['first_chunk']:following['first_chunk'] + following['chunks']]
                # First text from this push precedes all completed work and
                # selection that lead to the following decision.
                high = following['decision_ns'] - sum(chunk['wall_ns'] for chunk in following_chunks) - following['selection_ns']
            elif first_text_step is not None:
                high = request - sample['stream_finish_ns']
            else:
                # A finish-only delta is emitted within finish itself.
                high = request
            need(integer(first_text, 1) and low <= first_text <= high, 'First decoded text event differs')
        need(previous + previous_decode + sample['stream_finish_ns'] <= request
             and prefix + walls + selects + decodes + sample['stream_finish_ns'] <= request,
             'Text disjoint request phase sum exceeds request wall')
        need(prefix + common.duration_ns(previous_gpu - request_first_gpu) <= request + timestamp_grain_ns(request_first_gpu, previous_gpu),
             'Text input plus whole GPU request series exceeds request wall')
        events = [x['decision_ns'] for x in sample['decisions']]
        derived.append(dict(sample=i, warmup=i < 3, request_ns=request, decision_ttft_ns=events[0],
                            first_decoded_text_ns=first_text, render_ns=sample['render_ns'], encode_ns=sample['encode_ns'],
                            decode_interval_ns=events[-1] - events[0] if count > 1 else None,
                            decode_tokens_per_second=(count - 1) * 10 ** 9 / (events[-1] - events[0]) if count > 1 else None,
                            prefill_tokens_per_second=prompt * 10 ** 9 / prefill, completed_phase_ns=walls,
                            selection_ns=selects, incremental_decode_ns=decodes + sample['stream_finish_ns'],
                            gpu_seconds=gpu, command_count=commands, sampled_tokens=count, committed_tokens=prompt + count - 1))
        total += request;validation += sample['full_decode_validation_ns']
    common.validate_load_timing(report['load_timing'], elapsed, setup + total + validation)
    load = common.duration_ns(report['load_timing']['load_seconds'])
    need(load + setup + common.duration_ns(previous_gpu - first_gpu) <= elapsed + timestamp_grain_ns(first_gpu, previous_gpu),
         'Text disjoint load/setup/whole GPU series exceeds owned lifetime')
    return dict(all_samples=derived, measured=[x for x in derived if not x['warmup']], completed_requests=13,
                measured_requests=10, warmups=3, sampled_tokens=13 * count,
                total_request_ns=total, owned_elapsed_ns=elapsed, normalization_implementation=variant,
                ordinary_telemetry_overhead_qualified=False, model_quality_adoption_qualified=False,
                deeper_source_numerical_qualification=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['report', 'owned', 'contract', 'parity', 'parity_owned', 'fresh_output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('workload');parser.add_argument('variant', choices=['serial', 'parallel-store'])
    args = parser.parse_args()
    def metadata(path):
        with path.open('rb') as stream:raw = stream.read(2 * 1024 ** 2 + 1)
        need(len(raw) <= 2 * 1024 ** 2, 'Oversize text metadata')
        return json.loads(raw, object_pairs_hook=common.unique)
    owned = metadata(args.owned);contract = metadata(args.contract);parity_owned = metadata(args.parity_owned)
    expected = validate_parity(read(args.parity, parity_owned), parity_owned, contract, args.workload)
    result = validate(read(args.report, owned), owned, contract, args.workload, args.variant, expected)
    result['environment_evaluated'] = False
    with args.fresh_output.open('x') as stream:stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(validated_requests=13, environment_evaluated=False, model_adoption_qualified=False)))
