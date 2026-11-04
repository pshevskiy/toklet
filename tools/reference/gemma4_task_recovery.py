"""Deterministic proposal repair; the original stateless gate stays unchanged.

At the first divergent sampled ID, discard the entire suffix, take the exact
stateless choice, and generate a new suffix from the corrected prefix. Every
completed proposal must pass whole-trajectory verification before acceptance.
"""
import copy
import hashlib
import json
import gemma4_original_tasks as tasks
import gemma4_resident_generation as reference


def sampled_ids(proposal):
    ids=list(proposal['visible_token_ids'])
    if proposal['stop_reason']['kind']=='stop_token':
        ids.append(proposal['stop_reason']['token_id'])
    return ids


def repair(model, case, *, initial=None, max_repairs=8, progress=None):
    if type(max_repairs)is not int or not 0<=max_repairs<=8:
        raise ValueError('Bounded deterministic repair count required')
    prompt=case['prompt_tokens'];maximum=case['max_output_tokens'];stops=case['stop_ids']
    if not prompt or len(prompt)+maximum>reference.MAX_POSITIONS:
        raise ValueError('Full original task reserve required')
    session=reference.Session(model)
    proposal=copy.deepcopy(initial)if initial is not None else tasks.generate(session,prompt,maximum,stops,progress=progress)
    attempts=[];seen=set()
    for iteration in range(max_repairs+1):
        ids=sampled_ids(proposal)
        identity=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if identity in seen:
            return dict(verified=False,reason='proposal_cycle',generation=proposal,attempts=attempts)
        seen.add(identity)
        replay=tasks.trajectory(model,prompt,proposal,stops)
        if replay['verified']:
            return dict(verified=True,generation=proposal,trajectory=replay,attempts=attempts)
        if iteration==max_repairs:
            return dict(verified=False,reason='repair_bound',generation=proposal,trajectory=replay,attempts=attempts)
        first=replay['mismatches'][0];index=first['index'];selected=first['stateless_id']
        if not 0<=index<len(ids) or len(ids)>maximum:
            raise ValueError('First divergent sample outside complete proposal')
        prefix=ids[:index]
        if any(token in stops for token in prefix):
            raise ValueError('Hidden stop inside accepted prefix')
        samples=copy.deepcopy(proposal['samples'][:index])
        samples.append(dict(index=index,token_id=selected,margin=first['stateless_margin'],
            row_sha256=first['stateless_row_sha256'],origin='stateless_correction'))
        attempts.append(dict(iteration=iteration,rejected_proposal_sha256=identity,
            first_mismatch=first,discarded_sampled_suffix=len(ids)-index))
        session.reset()
        if selected in stops or index+1==maximum:
            visible=prefix if selected in stops else prefix+[selected]
            reason=dict(kind='stop_token',token_id=selected)if selected in stops else dict(kind='output_limit')
            history=prompt+visible+([selected]if selected in stops else[])
            session.forward(history[:-1])
            proposal=dict(visible_token_ids=visible,stop_reason=reason,samples=samples,
                committed_tokens=len(session.tokens),history_tokens=len(history),pending_tokens=1)
        else:
            corrected=prefix+[selected]
            def tail_progress(count):
                if progress is not None:progress(len(corrected)+count)
            tail=tasks.generate(session,prompt+corrected,maximum-len(corrected),stops,progress=tail_progress)
            for sample in tail['samples']:
                sample['index']+=len(corrected)
            proposal=dict(tail,visible_token_ids=corrected+tail['visible_token_ids'],samples=samples+tail['samples'])
        history=prompt+sampled_ids(proposal)
        if session.tokens!=history[:-1] or proposal['history_tokens']!=len(history) or proposal['pending_tokens']!=1:
            raise ValueError('Repaired sampled/committed state differs')
    raise AssertionError('bounded repair loop exhausted unexpectedly')
