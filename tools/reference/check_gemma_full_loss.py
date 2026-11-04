"""Exact frozen 10,000-target join and weighted original-relative loss gate."""
import math
import check_gemma_loss_block as block


def collect(records, ids, first=1, last=10000):
    block.require(len(ids)==10001 and len(records)==last-first+1,'Complete loss target inventory required')
    values=[]
    for index,item in enumerate(records,first):
        number,local=divmod(index-1,1024)
        values.append(block.statistics(item,block=number,local_row=local,
            global_index=index,target_id=ids[index]))
    return values


def remaining_native(transcript, ids):
    block.require(b'test result: ok.'in transcript,'Native run did not finish')
    records=block.labels(transcript,b'GEMMA_HYBRID_REAL_LOSS_TARGET ')
    starts=block.labels(transcript,b'GEMMA_HYBRID_REAL_LOSS_BLOCK_START ')
    unscored=block.labels(transcript,b'GEMMA_HYBRID_REAL_LOSS_UNSCORED ')
    summary=block.labels(transcript,b'GEMMA_HYBRID_REAL_LOSS_SUMMARY ')
    footer=block.labels(transcript,b'GEMMA_HYBRID_CAPTURE ')
    block.require(len(starts)==len(unscored)==9 and len(summary)==len(footer)==1,'Block/reset inventory differs')
    for number,(start,end)in enumerate(zip(starts,unscored,strict=True),1):
        count=min(1024,10000-number*1024)
        block.require(start==dict(block=number,input_start=number*1024,targets=count,positions=count+1,history_empty=True),'Position reset or block bounds differ')
        block.require(end['block']==number and end['position']==count and block.digest(end['row_sha256']),'Unscored block tail differs')
    values=collect(records,ids,1025,10000)
    item=summary[0]
    block.require(item['targets']==8976 and item['capacity']==1025 and item['chunk_capacity']==32
        and item['session_allocated_bytes']==271117548 and item['quality_qualified']is False
        and block.close(item['mean_nll'],math.fsum(values)/8976),'Remaining native summary differs')
    block.require(not block.labels(transcript,b'GEMMA_HYBRID_ARRAY ')
        and footer[0]['bytes']==footer[0]['records']==0
        and footer[0]['sha256']=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855','Unexpected full-row capture')
    return records


def check(original, native, ids):
    source_values=collect(original,ids);native_values=collect(native,ids)
    means=[math.fsum(values)/10000 for values in [source_values,native_values]]
    delta=means[1]-means[0]
    return dict(completed=True,targets=10000,original_mean_nll=means[0],native_mean_nll=means[1],
        native_minus_original=delta,mean_nll_increase_limit=.05,loss_gate_passed=delta<=.05,
        paired_mean_delta=math.fsum(b-a for a,b in zip(source_values,native_values,strict=True))/10000,
        task_gate_pending=True,full_quality_qualified=False,earlier_fidelity_failures_waived=False)
