"""Bounded structural replay of task journals; decoding never runs model code."""
import hashlib
import json
import math
from pathlib import Path
import prepare_llama3_tasks as inputs


def document(path, limit=8*1024**2):
    with Path(path).open('rb')as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise ValueError('Oversized task evidence')
    return json.loads(raw),raw


def digest(value):return isinstance(value,str)and len(value)==64 and all(c in '0123456789abcdef'for c in value)


def finite(value):return type(value)in(int,float)and math.isfinite(value)and value>=0


def collection(folder, input_folder, manifest_sha, role, probe=False):
    if role not in ['original','native']:raise ValueError('Unknown task role')
    manifest,cases=inputs.load(input_folder,manifest_sha);folder=Path(folder)
    if probe:cases=[dict(id='authored-probe',category='probe',prompt_tokens=[128000,1,7,13],max_output_tokens=4)]
    report,_=document(folder/'result.json')
    if (report['format']!=f'toklet-llama3-quality-tasks-{role}-v1' or report['completed']is not True
            or report['mode']!=('authored-probe'if probe else'full-tasks')or report['manifest_sha256']!=manifest_sha
            or report['cases']!=len(cases)or report['full_quality_qualified']is not False):
        raise ValueError('Incomplete task collection')
    with(folder/'tasks.jsonl').open('rb')as stream:raw=stream.read(8*1024**2+1)
    if len(raw)>8*1024**2 or hashlib.sha256(raw).hexdigest()!=report['tasks_sha256']:raise ValueError('Task journal changed')
    records=[json.loads(line)for line in raw.splitlines()]
    if len(records)!=len(cases):raise ValueError('Missing or duplicate task records')
    for case,record in zip(cases,records):
        ids=record['visible_token_ids'];reason=record['stop_reason'];generation=record['generation'];cap=case['max_output_tokens'];prompt=len(case['prompt_tokens'])
        if (record['id']!=case['id']or record['category']!=case['category']or record['prompt_tokens']!=prompt
                or record['output_reserve']!=cap or not isinstance(ids,list)or len(ids)>cap
                or any(type(i)is not int or not 0<=i<128256 or i in inputs.STOPS for i in ids)):
            raise ValueError('Task identity, admission or visible IDs differ')
        if reason['kind']=='stop_token':
            if type(reason['token_id'])is not int or reason['token_id']not in inputs.STOPS or len(ids)>=cap:raise ValueError('Hidden stop differs')
            expected=ids+[reason['token_id']]
        elif reason['kind']=='output_limit'and len(ids)==cap:expected=list(ids)
        else:raise ValueError('Incomplete task stop reason')
        history=prompt+len(expected);committed=history-1
        if (generation['history_tokens']!=history or generation['pending_tokens']!=1
                or type(generation['history_tokens'])is not int or type(generation['pending_tokens'])is not int):
            raise ValueError('Sampled task history differs')
        if role=='native':
            if (generation['reason']!=reason or generation['reused_tokens']!=0 or generation['prefill_tokens']!=prompt
                    or generation['sampled_tokens']!=len(expected)or generation['emitted_tokens']!=len(ids)
                    or generation['evaluated_tokens']!=committed):raise ValueError('Native committed/sample counters differ')
        else:
            if generation['visible_token_ids']!=ids or generation['stop_reason']!=reason or generation['committed_tokens']!=committed:
                raise ValueError('Original committed/sample counters differ')
            samples=generation['samples'];replay=record['trajectory']
            if (len(samples)!=len(expected)or replay['verified']is not True or replay['mismatches']
                    or replay['visible_predictions']!=len(ids)or replay['terminal_stop_checked']!=(reason['kind']=='stop_token')
                    or len(replay['rows'])!=len(expected)):
                raise ValueError('Original trajectory is unresolved')
            for index,(token,sample,row)in enumerate(zip(expected,samples,replay['rows'])):
                if (sample['index']!=index or sample['token_id']!=token or not finite(sample['margin'])or not digest(sample['row_sha256'])
                        or row['index']!=index or row['absolute_row']!=prompt-1+index or row['expected_id']!=token
                        or row['stateless_id']!=token or row['incremental_margin']!=sample['margin']or not finite(row['stateless_margin'])):
                    raise ValueError('Original prediction/margin record differs')
    if role=='original':
        if report['model']!=manifest['models']['original']or report['all_original_trajectories_verified']is not True or not 0<report['peak_rss_bytes']<=6*1024**3:
            raise ValueError('Original model/replay/memory evidence differs')
    elif (report['repository']!=manifest['models']['native']['repository']or report['revision']!=manifest['models']['native']['revision']
            or not 0<report['admitted_bytes_including_app']<=6*1024**3
            or report['profile']!=dict(projection='MlxAffine',normalization='Compensated',attention='Reference',readback='FullLogits',chunk=32,capacity=566)):
        raise ValueError('Native model/profile/memory evidence differs')
    return report,records
