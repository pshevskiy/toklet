"""Exact case/stop/cap/pending/reset checks for the native generation path."""
import math
import check_gemma_loss_block as safe


def check(transcript, cases, *, pilot):
    safe.require(b'test result: ok.'in transcript,'Native controller did not finish')
    rows=safe.labels(transcript,b'GEMMA_HYBRID_REAL_TASK_RESULT ')
    summary=safe.labels(transcript,b'GEMMA_HYBRID_REAL_TASK_SUMMARY ')
    selected=[i for i in range(100)if ((i in [0,40,70])==pilot)]
    safe.require(len(cases)==100 and len(rows)==len(selected) and len(summary)==1,'Complete controller case coverage required')
    for index,row in zip(selected,rows,strict=True):
        case=cases[index]
        safe.require(row['index']==index and type(row['index'])is int and row['id']==case['id']
            and row['category']==case['category'] and row['prompt_tokens']==len(case['prompt_tokens'])
            and row['output_reserve']==case['max_output_tokens'] and row['full_quality_qualified']is False,'Frozen task input/reserve differs')
        visible=row['visible_token_ids'];reason=row['stop_reason'];report=row['generation']
        safe.require(type(visible)is list and len(visible)<=case['max_output_tokens']
            and all(type(i)is int and 0<=i<262144 and i not in case['stop_ids']for i in visible),'Visible IDs include hidden stop or invalid vocabulary')
        if reason['kind']=='stop_token':
            safe.require(type(reason['token_id'])is int and reason['token_id']in case['stop_ids'] and len(visible)<case['max_output_tokens'],'Unknown hidden stop')
            sampled=len(visible)+1
        elif reason['kind']=='output_limit':
            safe.require(len(visible)==case['max_output_tokens'],'Output reserve truncated')
            sampled=len(visible)
        else:
            raise ValueError('Incomplete or cancelled task')
        history=len(case['prompt_tokens'])+sampled
        expected=dict(reused_tokens=0,prefill_tokens=len(case['prompt_tokens']),sampled_tokens=sampled,
            emitted_tokens=len(visible),history_tokens=history,evaluated_tokens=history-1,pending_tokens=1)
        safe.require(report['reason']==reason and all(type(report[k])is int and report[k]==v for k,v in expected.items()),'Sampled/committed history accounting differs')
        for key in ['first_token_seconds','prefill_seconds','decode_seconds','sampling_seconds','total_seconds']:
            safe.require(type(report[key])in(int,float)and math.isfinite(report[key])and report[key]>=0,'Invalid controller time')
        safe.require(row['memory']['admitted_bytes']==rows[0]['memory']['admitted_bytes'],'Controller allocation changed between cases')
        safe.require(type(row['memory']['admitted_bytes'])is int and 0<row['memory']['admitted_bytes']<=6*1024**3-128*1024**2,'Native controller admission differs')
    item=summary[0]
    safe.require(item['cases']==len(selected)and item['pilot']is pilot and item['capacity']==567
        and item['chunk_capacity']==32 and item['reset_checked']is True and item['quality_qualified']is False
        and item['admitted_bytes_including_diagnostic']==rows[0]['memory']['admitted_bytes']+128*1024**2
        and 0<item['admitted_bytes_including_diagnostic']<=6*1024**3,'Controller reset/summary differs')
    return dict(passed=True,cases=len(rows),pilot=pilot,indices=selected,rows=rows,quality_qualified=False)
