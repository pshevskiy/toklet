"""Independent full-logit replay plus preserved qualified generation/codec states.

State schedules are compared to pinned previously accepted scalar captures.
Numerical values come directly from the older independent full math witnesses.
Neither same-engine state refinement nor argmax substitutes for numerical gates.
"""
from pathlib import Path
import hashlib
import json
import math
import struct
import sys
import check_gemma_hybrid_tiny as checked

require, same = checked.require, checked.same
ASSETS = {
    'raw': ('m8-gemma-generate-real-v1', 'f1d614ce5d5655eed160478b5b073c40548719e21a4612198d41cc2f7c37c339', 'm8-gemma-greedy-real-v1', '5e2e6b45121b76edd0e81db3ad80a237f15b691d059bc022d65862d97ab1df1c', '5779e6366517cedef11e9b892d596d7087a3cc5b5c32da8c10d546562fc17cff'),
    'basic-hello': ('m8-gemma-chat-real-v1', '5d9e55e32efffc15f763afd5f3737d7c03c2ea0c4766b62197700257c46fd370', 'm8-gemma-basic-chat-real-v1', 'df15179c3f72129eaddc8450ca537bbacf71ff98678f52ed95d51046f8fee28a', '3c3b83440205950db3bd885feb4918351fbfbf04ab3e9204dcdf6f6caa187a06'),
    'basic-hello-stream': ('m8-gemma-chat-stream-real-v1', '8d0fc8f4fe719041c4270c8298890158f222a8f37d47db3792436dd2fc01fcfd', 'm8-gemma-basic-chat-real-v1', 'df15179c3f72129eaddc8450ca537bbacf71ff98678f52ed95d51046f8fee28a', '3c3b83440205950db3bd885feb4918351fbfbf04ab3e9204dcdf6f6caa187a06'),
}
TIMERS = {'first_token_seconds', 'prefill_seconds', 'decode_seconds', 'sampling_seconds', 'total_seconds'}
CATEGORIES = ['weights_bytes', 'kv_bytes', 'scratch_bytes', 'history_bytes', 'logits_bytes', 'metadata_bytes', 'metadata_allowance_bytes']


def pinned(path, digest, limit):
    raw = checked.read(path, limit)
    require(hashlib.sha256(raw).hexdigest() == digest, 'previously frozen identity: '+str(path))
    return raw


def references(root, profile):
    baseline, baseline_sha, oracle, oracle_sha, binary_sha = ASSETS[profile]
    state = checked.document(pinned(root/baseline/'observation/observation.json', baseline_sha, 1024**2))
    witness = checked.document(pinned(root/oracle/'observation/observation.json', oracle_sha, 1024**2))
    binary = pinned(root/oracle/'observation/arrays.bin', binary_sha, 128*1024**2)
    expected = {}
    for item in witness['arrays']:
        if item['name'] != 'logits': continue
        key = (item['step'], item['contract'])
        require(key not in expected and item['completed'] is True, 'complete unique independent logits')
        same(item['shape'], [262144]); same(item['dtype'], '<f4' if item['contract'] == 'staged_fp32' else '<f8')
        body = memoryview(binary)[item['offset']:item['offset']+item['bytes']]
        require(len(body) == 262144*(4 if item['dtype'] == '<f4' else 8) and hashlib.sha256(body).hexdigest() == item['sha256'], 'complete independent distribution')
        expected[key] = (item, body)
    same(set(expected), {(i,c) for i in range(4 if profile == 'raw' else 11) for c in ['math_fp64','staged_fp32']})
    return state, witness, expected


def state_without_timers(value):
    if isinstance(value, dict): return {k: state_without_timers(v) for k,v in value.items() if k not in TIMERS}
    if isinstance(value, list): return [state_without_timers(v) for v in value]
    return value


def replay(profile, observation, raw, baseline, witness, expected):
    same(observation['format'], baseline['format'])
    for key in ['passed', 'prompt', 'max_executable_positions', 'local_cache_slots', 'global_cache_slots', 'memory_limit_bytes', 'diagnostic_byte_limit', 'completed_array_bytes']:
        same(observation[key], baseline[key])
    for key in ['primary_error', 'capture_error']: same(observation[key], None)
    same(observation['endpoint_errors'], [])
    same(observation['execution_backend'], 'hybrid'); same(observation['construction_context_dropped'], True)
    for flag in ['numerical_comparison_complete','original_bf16_qualified','tokenizer_chat_or_model_stop_policy_qualified','metal_qualified','performance_qualified','quality_qualified','device_fit_qualified','app_eligibility_changed']:
        same(observation[flag], False)
    memory = observation['memory']
    same(set(memory), set(CATEGORIES)|{'admitted_bytes'})
    require(all(type(v) is int and v > 0 for v in memory.values()), 'positive exact memory counts')
    same(sum(memory[k] for k in CATEGORIES), memory['admitted_bytes'])
    require(type(observation['model_bytes_including_store']) is int and 2604665212 < observation['model_bytes_including_store'] < memory['admitted_bytes'], 'resident model and session ownership')
    require(type(observation['hybrid_norm_copy_bytes']) is int and observation['hybrid_norm_copy_bytes'] >= 1143948, 'count copied vector ownership')
    probe = observation['generation_admission_probe']
    same(probe, dict(required=memory['admitted_bytes'], limit=memory['admitted_bytes']-1,
        minus_one_expected_budget_error=True, error=f"MemoryBudget {{ required: {memory['admitted_bytes']}, limit: {memory['admitted_bytes']-1} }}", exact_memory_unchanged=True))
    reserve = 16*1024**2
    if profile != 'raw':
        for key in ['profile','rendered_prompt','stop_ids','formatted_fixture_sha256','tokenizer_allowance_bytes','diagnostic_allowance_bytes','text_streaming_qualified']:
            same(observation[key], baseline[key])
        reserve += 1024**3
        if profile == 'basic-hello-stream':
            same(observation['stream_allowance_bytes'], 4*1024**2); reserve += 4*1024**2
    same(observation['diagnostic_allowance_bytes'], 16*1024**2)
    same(observation['numerical_budget_bytes'], 6*1024**3-reserve)
    same(observation['combined_admitted_bytes'], memory['admitted_bytes']+reserve)
    require(memory['admitted_bytes'] <= observation['numerical_budget_bytes'] and observation['combined_admitted_bytes'] <= 6*1024**3, 'whole diagnostic admission')
    same(state_without_timers(observation['scenarios']), state_without_timers(baseline['scenarios']))
    for scenario in observation['scenarios']:
        for report in scenario['reports']:
            for key in TIMERS-{'first_token_seconds'}:
                require(type(report[key]) in (int,float) and math.isfinite(report[key]) and report[key] >= 0, 'finite nonnegative generation timing')
            first = report['first_token_seconds']
            if report['emitted_tokens']:
                require(type(first) in (int,float) and math.isfinite(first) and 0 <= first <= report['total_seconds'], 'first visible token timing')
            else: same(first, None)
    same(len(observation['arrays']), len(baseline['arrays']))
    offset = 0; rows = {}; comparisons = []
    for item, old in zip(observation['arrays'], baseline['arrays'], strict=True):
        same({k:v for k,v in item.items() if k != 'sha256'}, {k:v for k,v in old.items() if k != 'sha256'})
        same(item['offset'], offset); same(item['bytes'], 1048576); same(item['shape'], [262144]); same(item['dtype'], '<f4'); same(item['completed'], True)
        step = item['step']; same(item['prefix'], witness['steps'][step]['prefix'])
        block = memoryview(raw)[offset:offset+item['bytes']]
        require(len(block) == item['bytes'] and hashlib.sha256(block).hexdigest() == item['sha256'], 'complete native generation row')
        maximum = -math.inf; token = None
        for index,(value,) in enumerate(struct.iter_unpack('<f',block)):
            require(math.isfinite(value), 'finite native distribution')
            if value > maximum: maximum,token = value,index
        same(token,witness['sampled_ids'][step])
        rows[item['scenario'],step] = block
        for contract in ['math_fp64','staged_fp32']:
            spec,body = expected[step,contract]; failed = 0; worst = 0.
            for (a,), (b,) in zip(struct.iter_unpack('<f', block), struct.iter_unpack('<f' if spec['dtype'] == '<f4' else '<d', body), strict=True):
                require(math.isfinite(b), 'finite reference distribution')
                ratio = abs(a-b)/(1e-5+1e-4*abs(b)); worst = max(worst,ratio); failed += ratio > 1.
            comparisons.append(dict(scenario=item['scenario'],step=step,contract=contract,values=262144,failed_values=failed,maximum_gate_ratio=worst))
        offset += item['bytes']
    require(offset == len(raw) == observation['completed_array_bytes'], 'complete generation binary')
    bitwise = all(body == rows[0,step] for (scenario,step),body in rows.items() if scenario != 0)
    require(bitwise, 'same-prefix schedule distributions differ')
    staged = all(v['failed_values'] == 0 for v in comparisons if v['contract'] == 'staged_fp32')
    ideal = all(v['failed_values'] == 0 for v in comparisons if v['contract'] == 'math_fp64')
    return dict(passed=staged,profile=profile,staged_fp32_passed=staged,ideal_fp64_passed=ideal,comparisons=comparisons,
        rows=len(observation['arrays']),schedules_bitwise_identical=bitwise,whole_e2b_qualified=False,full_model_window_qualified=False,
        performance_qualified=False,phone_qualified=False,mathematical_failures_waived=False,
        state_scope='Refinement of pinned previously qualified scalar schedules/codec states, distinct from independent full-logit numerical comparison.')


if __name__ == '__main__':
    profile,directory,artifacts = sys.argv[1:]
    state,witness,expected = references(Path(artifacts),profile)
    root=Path(directory)
    result=replay(profile,checked.document(checked.read(root/'observation.json',1024**2)),checked.read(root/'arrays.bin',64*1024**2),state,witness,expected)
    print(json.dumps(result));sys.exit(0 if result['passed'] else 1)
