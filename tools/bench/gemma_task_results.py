"""Gemma inference-journal admission and literal adapter to qualified grading.

Never runs generated code. Canonical grading cases are used only after their
literal scoring fields match the separate authoritative Gemma input contract.
"""
import hashlib
import importlib.metadata
import math
import os
import task_cases


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def collection(cases, records, role):
    require(role in ['original', 'native'] and len(cases) == len(records) == 100,
            'Complete paired role/case inventory required')
    normalized = []
    for case, record in zip(cases, records, strict=True):
        require(record['id'] == case['id'] and record['category'] == case['category']
                and record['prompt_tokens'] == len(case['prompt_tokens'])
                and record['output_reserve'] == case['max_output_tokens'],
                'Gemma inference identity/reserve differs')
        generation = record['generation']
        visible = generation['visible_token_ids'] if role == 'original' else record['visible_token_ids']
        reason = generation['stop_reason'] if role == 'original' else record['stop_reason']
        cap = case['max_output_tokens']
        require(type(visible) is list and len(visible) <= cap
                and all(type(i) is int and 0 <= i < 262144 and i not in case['stop_ids'] for i in visible),
                'Gemma visible IDs or hidden-stop exclusion differ')
        if reason['kind'] == 'stop_token':
            require(type(reason['token_id']) is int and reason['token_id'] in case['stop_ids'] and len(visible) < cap,
                    'Hidden stop exceeds the sample reserve')
            expected = visible + [reason['token_id']]
        elif reason['kind'] == 'output_limit' and len(visible) == cap:
            expected = list(visible)
        else:
            raise ValueError('Incomplete task termination')
        history = len(case['prompt_tokens']) + len(expected)
        require(type(generation['history_tokens']) is int and generation['history_tokens'] == history
                and type(generation['pending_tokens']) is int and generation['pending_tokens'] == 1,
                'Sampled task history differs')
        if role == 'native':
            counts = dict(reused_tokens=0, prefill_tokens=len(case['prompt_tokens']),
                          sampled_tokens=len(expected), emitted_tokens=len(visible), evaluated_tokens=history-1)
            require(generation['reason'] == reason
                    and all(type(generation[k]) is int and generation[k] == v for k, v in counts.items()),
                    'Native committed/sample counters differ')
        else:
            require(generation['committed_tokens'] == history-1, 'Original committed prefix differs')
            samples = generation['samples']
            replay = record['trajectory']
            require(len(samples) == len(expected) and replay['verified'] is True and not replay['mismatches']
                    and replay['visible_predictions'] == len(visible)
                    and replay['terminal_stop_checked'] == (reason['kind'] == 'stop_token')
                    and len(replay['rows']) == len(expected), 'Original trajectory remains unresolved')
            for index, (token, sample, row) in enumerate(zip(expected, samples, replay['rows'], strict=True)):
                require(sample['index'] == index and sample['token_id'] == token and finite(sample['margin'])
                        and digest(sample['row_sha256']) and row['index'] == index
                        and row['absolute_row'] == len(case['prompt_tokens'])-1+index
                        and row['expected_id'] == row['stateless_id'] == token
                        and row['incremental_margin'] == sample['margin'] and finite(row['stateless_margin'])
                        and digest(row['stateless_row_sha256']), 'Original sampled/verified decision differs')
        normalized.append(dict(id=case['id'], category=case['category'], visible_token_ids=visible, stop_reason=reason))
    return normalized


def canonical_grading_cases(gemma_cases, canonical=None):
    canonical = task_cases.load_cases() if canonical is None else canonical
    require(len(gemma_cases) == len(canonical) == 100, 'All literal grading cases required')
    for actual, grading in zip(gemma_cases, canonical, strict=True):
        keys = ['id', 'category'] + (['code_prefix', 'tests', 'entry_point'] if grading['category'] == 'code' else ['expected'])
        require(all(type(actual.get(k)) is str and actual[k].encode('utf-8') == grading[k].encode('utf-8') for k in keys),
                'Literal grading case order/content differs')
        task_cases.validate_case(grading)
    return canonical


def decode(cases, records, tokenizer_bytes, tokenizer_spec, factory=None):
    require(len(cases) == len(records) == 100 and len(tokenizer_bytes) == tokenizer_spec['bytes']
            and hashlib.sha256(tokenizer_bytes).hexdigest() == tokenizer_spec['sha256'],
            'Exact authenticated tokenizer and complete responses required')
    if factory is None:
        require(importlib.metadata.version('tokenizers') == '0.22.2', 'Pinned tokenizer runtime required')
        from tokenizers import Tokenizer
        factory = Tokenizer.from_buffer
    tokenizer = factory(tokenizer_bytes)
    decoded = []
    for case, record in zip(cases, records, strict=True):
        require(case['id'] == record['id'] and case['category'] == record['category'], 'Decoded case order differs')
        text = tokenizer.decode(record['visible_token_ids'], skip_special_tokens=True)
        require(type(text) is str and len(text.encode('utf-8')) <= 65536, 'Decoded completion exceeds bound')
        decoded.append(dict(id=case['id'], category=case['category'], visible_token_ids=record['visible_token_ids'],
                            visible_text=text, visible_text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest()))
    return decoded


def require_idle(workers):
    require(len(workers) == 2, 'Both owned model workers required')
    pids = set()
    for worker in workers:
        command = worker['command']
        pid = command['pid']
        require(worker['passed'] is True and worker['cases'] == 100 and type(pid) is int and pid > 1
                and pid not in pids and command['exit_code'] == 0 and command['primary_error'] is None
                and command['cleanup_errors'] == [], 'Complete clean model-worker exits required')
        for target in [pid, -pid]:
            try:
                os.kill(target, 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('Model process/group remains live')
        pids.add(pid)
    return sorted(pids)


def decision(cases, outcomes):
    baseline = task_cases.aggregate(cases, outcomes['baseline'])
    native = task_cases.aggregate(cases, outcomes['native'])
    return dict(complete=True, baseline_tasks=baseline, native_tasks=native,
                task_pass_rate_loss_percentage_points=baseline['passed']-native['passed'],
                task_limit_percentage_points=2, task_gate_passed=native['passed']+2 >= baseline['passed'],
                full_quality_qualified=False, app_discovery_qualified=False, device_qualified=False)
