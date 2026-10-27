"""Model-independent full-workload policy; historical Gemma producer stays frozen."""

SCORING={
 'sampling':'greedy_lowest_token_id_on_ties',
 'stop_policy':'stop_before_visible_decode; hidden_stop_consumes_one_sampled_output_slot',
 'visible_decode':'skip_special_tokens=True; decode_final_visible_token_sequence',
 'code':'append_visible_completion_exactly_to_original_prefix; no_fence_removal_or_repair; all_source_tests_in_isolated_grader',
 'prose':'Python_str_strip_equals_expected_uppercase_letter; no_explanation_extraction',
 'continuation':'first_Python_whitespace_delimited_visible_word_equals_expected_case_and_punctuation; empty_is_failure',
 'reset':'independent_task_reset_to_position_zero; no_cache_across_cases',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)

def blocks(targets=10000):
    require(type(targets)is int and targets==10000,'complete held-out target policy')
    return [dict(block=i,start=start,end=min(start+1024,targets),targets=min(1024,targets-start),
                 input_start=start,input_end=min(start+1024,targets)+1,reset_position=0)
            for i,start in enumerate(range(0,targets,1024))]

def make_tasks(code,prose,continuation):
    require(len(code)>=40 and len(prose)>=30 and len(continuation)>=30,'all first-source task rows required')
    result=[]
    for index,case in enumerate(code[:40]):
        require(case['task_id']==f'HumanEval/{index}','first HumanEval source identities')
        text='Complete this Python function. Return only the indented completion, without repeating the function signature or using Markdown fences.\n\n'+case['prompt']
        result.append(dict(id=case['task_id'],category='code',source_row=index,user_content=text,max_output_tokens=256,
            code_prefix=case['prompt'],tests=case['test'],entry_point=case['entry_point']))
    for index,case in enumerate(prose[:30]):
        options=[case['Best Answer']]
        for wrong in case['Incorrect Answers'].split(';'):
            wrong=wrong.strip()
            if wrong and wrong not in options:options.append(wrong)
            if len(options)==4:break
        require(len(options)>=2,'TruthfulQA distractor coverage')
        rotation=index%len(options);options=options[rotation:]+options[:rotation]
        text='Choose the true answer. Reply with exactly one uppercase letter.\n'+case['Question']+'\n'+'\n'.join(f'{chr(65+i)}. {answer}' for i,answer in enumerate(options))
        result.append(dict(id=f'truthfulqa/row-{index}',category='prose',source_row=index,user_content=text,max_output_tokens=32,
            expected=chr(65+(-rotation)%len(options)),option_rotation=rotation,options=options))
    for index,case in enumerate(continuation[:30]):
        prompt,space,expected=case['text'].rstrip().rpartition(' ');require(bool(space and expected),'LAMBADA original last word')
        result.append(dict(id=f'lambada/row-{index}',category='continuation',source_row=index,raw_prompt=prompt,max_output_tokens=16,expected=expected))
    require(len(result)==100 and len({v['id'] for v in result})==100,'complete unique task IDs')
    return result
