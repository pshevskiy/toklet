"""Thin complete-collection adapter for existing literal task graders.

No model or generated code executes here. The enclosing window authenticates
natural model exits, exact journals, source/model identity and Linux isolation.
"""
import hashlib
import math
import struct

import prepare_qwen3_quality as contract
import task_cases

FORMATS = {'native': 'toklet-qwen3-quality-diagnostic-events-v1',
           'baseline': 'toklet-qwen3-bf16-quality-events-v1'}


def require(condition, message):
    if not condition:raise ValueError(message)


def integer(value, expected):
    return type(value) is int and value == expected


def events(path, expected):
    body = contract.checked(path, expected, 32 * 1024**2)
    require(body.endswith(b'\n'), 'complete newline-terminated collection required')
    return [contract.safe.document(line) for line in body.splitlines()]


def ready(rows, profile, mode, manifest):
    require(profile in FORMATS and rows, 'explicit complete collection profile required')
    value = rows[0]
    require(value.get('event') == 'ready' and value.get('format') == FORMATS[profile]
        and value.get('mode') == mode and value.get('full_quality_workload') is True,
        'full-workload readiness required; probes cannot qualify')
    require(integer(value.get('targets'), 10000 if mode == 'loss' else 0)
        and integer(value.get('tasks'), 100 if mode == 'tasks' else 0), 'complete workload readiness counts')
    if profile == 'native':
        file = 'held_out_loss.u32le' if mode == 'loss' else 'execution.jsonl'
        require(value.get('input_sha256') == manifest['files'][file]['sha256']
            and value.get('context_capacity') == 1025 and value.get('prefill_chunk_size') == 128
            and value.get('debug_assertions') is False
            and value.get('c512_numerical_gate_status') == 'rejected_unchanged'
            and value.get('projection') == 'mlx_affine' and value.get('normalization') == 'compensated'
            and value.get('attention') == 'reference', 'frozen native diagnostic profile differs')
    else:
        require(value.get('input_manifest_sha256') == contract.sha(contract.encoded(manifest)),
                'frozen baseline input manifest differs')


def same_sum(actual, expected):
    return type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,abs_tol=1e-10,rel_tol=1e-13)


def loss(rows, profile, manifest, ids):
    require(len(ids) == 10001, 'all10000 frozen targets required')
    ready(rows, profile, 'loss', manifest)
    losses = [];block_values = [];block = 0;start = 0;evaluated = 0;finished = False
    for index, value in enumerate(rows[1:], 1):
        require(not finished, 'events after loss completion')
        kind = value.get('event');end = min(start+1024,10000)
        if kind == 'loss':
            target_index = len(losses)
            require(target_index < end and integer(value.get('target_index'),target_index)
                and integer(value.get('block'),block) and integer(value.get('position'),target_index-start)
                and integer(value.get('target_token'),ids[target_index+1]), 'loss target identity/order differs')
            nll = value.get('negative_log_likelihood');bits = value.get('target_logit_f32_bits')
            digest = value.get('logits_sha256')
            require(type(nll) in (int,float) and math.isfinite(nll) and nll >= 0
                and type(bits) is int and 0 <= bits < 2**32
                and math.isfinite(struct.unpack('<f',struct.pack('<I',bits))[0])
                and isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
                'finite authenticated distribution statistics required')
            losses.append(nll);block_values.append(nll)
        elif kind == 'chunk_done':
            require(profile == 'native' and block < 10, 'unexpected chunk protocol or extra completed block')
            count = min(128,end-start+1-evaluated)
            require(count > 0 and integer(value.get('block'),block) and integer(value.get('rows'),count)
                and integer(value.get('evaluated_tokens'),evaluated+count) and integer(value.get('pending_tokens'),0)
                and len(block_values)==min(evaluated+count,end-start), 'native complete chunk commitment differs')
            evaluated += count
        elif kind == 'block_done':
            require(block < 10 and len(block_values)==end-start and len(losses)==end
                and integer(value.get('block'),block) and integer(value.get('target_start'),start)
                and integer(value.get('target_end'),end) and integer(value.get('targets'),end-start)
                and integer(value.get('evaluated_tokens'),end-start+1) and integer(value.get('pending_tokens'),0)
                and same_sum(value.get('total_negative_log_likelihood'),math.fsum(block_values))
                and (profile != 'native' or evaluated==end-start+1), 'complete loss block coverage/reduction differs')
            block += 1;start = end;block_values = [];evaluated = 0
        elif kind == 'done':
            require(index==len(rows)-1 and block==10 and len(losses)==10000 and not block_values
                and integer(value.get('targets'),10000) and integer(value.get('tasks'),0)
                and value.get('mode')=='loss' and value.get('full_quality_workload') is True
                and same_sum(value.get('total_negative_log_likelihood'),math.fsum(losses))
                and same_sum(value.get('mean_negative_log_likelihood'),math.fsum(losses)/10000),
                'complete loss terminal record required')
            finished = True
        else:raise ValueError('unknown loss event')
    require(finished, 'loss collection is incomplete')
    return dict(targets=10000,blocks=10,total_negative_log_likelihood=math.fsum(losses),
        mean_negative_log_likelihood=math.fsum(losses)/10000)


def tasks(rows, profile, manifest, cases):
    require(len(cases)==100, 'all100 source cases required')
    ready(rows, profile, 'tasks', manifest)
    results = [];visible = [];sampled = 0;started = False;hidden_stop = False;finished = False
    for index,value in enumerate(rows[1:],1):
        require(not finished, 'events after task completion')
        kind = value.get('event')
        if kind == 'done':
            require(index==len(rows)-1 and len(results)==100 and not visible and not started
                and sampled==0 and integer(value.get('targets'),0) and integer(value.get('tasks'),100),
                'complete task terminal record required')
            require(profile!='baseline' or value.get('mode')=='tasks' and value.get('full_quality_workload') is True,
                    'baseline task terminal workload must remain complete')
            finished = True;continue
        require(len(results)<100, 'extra quality case')
        case = cases[len(results)]
        require(value.get('id')==case['id'], 'task identity/order differs')
        if kind == 'task_started':
            require(profile=='baseline' and not started and sampled==0, 'duplicate or unexpected task start')
            started = True
        elif kind == 'token':
            require(profile!='baseline' or started, 'baseline token before case start')
            token = value.get('token_id')
            require(type(token) is int and 0<=token<151936 and not hidden_stop, 'token after hidden EOS or invalid vocabulary')
            require(integer(value.get('index'),sampled), 'sampled event order differs')
            if profile=='baseline':
                require(value.get('visible') is (token!=151645), 'source hidden stop visibility differs')
            else:require(token!=151645, 'native EOS must remain hidden')
            sampled += 1
            require(sampled<=case['max_output_tokens'], 'extra sampled output token')
            if token==151645:hidden_stop = True
            else:visible.append(token)
        elif kind == 'task_done':
            require(profile!='baseline' or started, 'baseline completion before case start')
            report = value.get('report',{});n = report.get('sampled_tokens');reason = report.get('reason')
            require(type(n) is int and 1<=n<=case['max_output_tokens'] and value.get('category')==case['category']
                and integer(value.get('prompt_tokens'),len(case['prompt_tokens']))
                and integer(value.get('requested_output_tokens'),case['max_output_tokens'])
                and value.get('visible_token_ids')==visible and integer(report.get('emitted_tokens'),len(visible))
                and integer(report.get('evaluated_tokens'),len(case['prompt_tokens'])+n-1)
                and integer(report.get('pending_tokens'),1)
                and integer(report.get('history_tokens'),len(case['prompt_tokens'])+n), 'task output/pending accounting differs')
            if reason == dict(kind='output_limit'):
                require(n==case['max_output_tokens'] and n==len(visible) and not hidden_stop, 'output cap completion differs')
            elif reason == dict(kind='stop_token',token_id=151645):
                require(n==len(visible)+1 and (profile!='baseline' or hidden_stop), 'hidden stop sampled slot differs')
            else:raise ValueError('cancelled/unresolved task cannot be scored')
            require(sampled==(n if profile=='baseline' else len(visible)), 'token events and completion disagree')
            results.append(dict(id=case['id'],category=case['category'],visible_token_ids=list(visible),report=report))
            visible = [];sampled = 0;started = False;hidden_stop = False
        else:raise ValueError('unknown task event')
    require(finished, 'task collection is incomplete')
    return results


def legacy_case(case):
    category = case['category'];cap = {'code':256,'prose':32,'continuation':16}.get(category)
    require(cap==case['max_output_tokens'] and 0<len(case['prompt_tokens'])+cap<=1025,
            'frozen Qwen3 shared budget differs')
    result = dict(case,stop_policy=f'eos_or_{cap}',scoring=task_cases.SCORERS[category])
    task_cases.validate_case(result)
    return result


def decode(cases, responses):
    from tokenizers import Tokenizer
    body = contract.checked(contract.MODEL/'tokenizer.json',contract.PINS['tokenizer.json'][1],16*1024**2)
    tokenizer = Tokenizer.from_buffer(body)
    require(len(cases)==len(responses)==100, 'complete task decode required')
    values = []
    for case,response in zip(cases,responses,strict=True):
        require(case['id']==response['id'], 'decoder case order differs')
        text = tokenizer.decode(response['visible_token_ids'],skip_special_tokens=True)
        require(len(text.encode())<=65536, 'bounded whole-response decode required')
        adapted = legacy_case(case)
        value = dict(response,visible_text=text,visible_text_sha256=hashlib.sha256(text.encode()).hexdigest())
        if case['category']=='code':
            program = task_cases.human_eval_program(adapted,text)
            value['program_payload'] = dict(case_id=case['id'],program=program,
                program_sha256=hashlib.sha256(program.encode()).hexdigest())
        else:
            passed = task_cases.grade_text(adapted,text)
            value['outcome'] = dict(id=case['id'],category=case['category'],passed=passed,status='model_pass'if passed else'model_fail')
        values.append(value)
    return values


def decision(candidate_loss, baseline_loss, cases, candidate_outcomes, baseline_outcomes):
    require(integer(candidate_loss['targets'],10000) and integer(baseline_loss['targets'],10000)
        and all(type(v['mean_negative_log_likelihood']) in (int,float)
                and math.isfinite(v['mean_negative_log_likelihood']) for v in (candidate_loss,baseline_loss)),
        'complete finite loss comparison required')
    native = task_cases.aggregate([legacy_case(c)for c in cases],candidate_outcomes)
    base = task_cases.aggregate([legacy_case(c)for c in cases],baseline_outcomes)
    delta = candidate_loss['mean_negative_log_likelihood']-baseline_loss['mean_negative_log_likelihood']
    loss_pass = delta<=0.05;task_pass = native['passed']+2>=base['passed']
    return dict(complete=True,loss_increase_nats_per_token=delta,loss_limit=0.05,loss_passed=loss_pass,
        task_pass_rate_loss_percentage_points=base['passed']-native['passed'],task_limit_percentage_points=2,
        tasks_passed=task_pass,candidate_tasks=native,baseline_tasks=base,
        decision='retain_experimental_candidate'if loss_pass and task_pass else'reject_precision_candidate',
        numerical_qualified=False,app_window_changed=False,full_model_window_qualified=False)
