"""Greedy task controller with mandatory stateless sampled-trajectory replay.

Numerical reference only. Decoding/grading happens after model workers exit.
Hidden stops consume a sample slot but are neither visible nor committed to KV.
"""
import hashlib
import torch
import gemma4_resident_generation as reference
import gemma4_upstream_stream as stream


def inspect(row):
    row=row.reshape(-1)
    raw=stream.tensor_bytes(row)
    top=row.float().topk(2).values
    return dict(token_id=int(row.argmax()),margin=float(top[0]-top[1]),
                row_sha256=hashlib.sha256(raw).hexdigest())


def generate(session, prompt, maximum, stops, *, progress=None):
    if (type(maximum)is not int or maximum<=0 or not prompt
            or len(prompt)+maximum>reference.MAX_POSITIONS or session.tokens
            or session.replay_required or session.cache.get_seq_length()!=0):
        raise ValueError('Fresh admitted task session and full reserve required')
    row=session.forward(prompt);visible=[];samples=[];reason=None
    for index in range(maximum):
        sample=inspect(row);sample['index']=index;samples.append(sample)
        selected=sample['token_id']
        if progress is not None and ((index+1)%32==0 or selected in stops or index+1==maximum):
            progress(index+1)
        if selected in stops:
            reason=dict(kind='stop_token',token_id=selected)
            break
        visible.append(selected)
        if len(visible)==maximum:
            reason=dict(kind='output_limit')
            break
        row=session.forward([selected])
    history=prompt+visible+([reason['token_id']]if reason['kind']=='stop_token'else[])
    if session.tokens!=history[:-1] or len(history)!=len(session.tokens)+1:
        raise ValueError('Sampled and committed task state differs')
    return dict(visible_token_ids=visible,stop_reason=reason,samples=samples,
                committed_tokens=len(session.tokens),history_tokens=len(history),pending_tokens=1)


def teacher_rows(model, tokens, positions):
    """One stock text forward; project one sampling row at a time like decode."""
    ids=model.validate(tokens)
    if any(type(i)is not int or not 0<=i<len(tokens)for i in positions):
        raise ValueError('Teacher-forced row outside complete trajectory')
    with torch.inference_mode():
        hidden=model.model.model(**model.inputs(ids),use_cache=False).last_hidden_state
        for position in positions:
            logits=model.model.lm_head(hidden[:,position:position+1])
            cap=model.config.final_logit_softcapping
            if cap is not None:
                logits=logits/cap
                logits=torch.tanh(logits)
                logits=logits*cap
            yield logits


def trajectory(model, prompt, result, stops):
    visible=result['visible_token_ids'];reason=result['stop_reason']
    if any(type(i)is not int or not 0<=i<model.config.vocab_size or i in stops for i in visible):
        raise ValueError('Visible task tokens differ')
    expected=list(visible)
    if reason['kind']=='stop_token':
        if reason['token_id']not in stops:
            raise ValueError('Unknown hidden stop')
        expected.append(reason['token_id'])
    elif reason['kind']!='output_limit':
        raise ValueError('Incomplete task termination')
    if not expected or len(expected)!=len(result['samples']):
        raise ValueError('Complete sampled trajectory required')
    positions=list(range(len(prompt)-1,len(prompt)-1+len(expected)))
    rows=[];mismatches=[]
    for index,(wanted,row)in enumerate(zip(expected,teacher_rows(model,prompt+visible,positions),strict=True)):
        sample=result['samples'][index]
        if sample['index']!=index or sample['token_id']!=wanted:
            raise ValueError('Sampled ID/order differs')
        observed=inspect(row)
        item=dict(index=index,absolute_row=positions[index],expected_id=wanted,
                  stateless_id=observed['token_id'],incremental_margin=sample['margin'],
                  stateless_margin=observed['margin'],stateless_row_sha256=observed['row_sha256'])
        rows.append(item)
        if wanted!=observed['token_id']:
            mismatches.append(item)
    return dict(verified=not mismatches,rows=rows,mismatches=mismatches,
                visible_predictions=len(visible),terminal_stop_checked=reason['kind']=='stop_token')
