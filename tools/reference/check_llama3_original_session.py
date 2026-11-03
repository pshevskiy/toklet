"""Stdlib replay of incremental reference rows, greedy choices and state facts."""
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import struct
import check_llama3_original_short as rows
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import prepare_llama3_original_quality as inputs


def check(folder, contract_file, contract_sha, storage_profile='mapped-bf16'):
    folder=Path(folder);raw=Path(contract_file).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=contract_sha:raise ValueError('Frozen contract differs')
    policy=json.loads(raw,object_pairs_hook=inputs.parent.unique)
    _,ids=inputs.load(inputs.ROOT/'benchmarks/llama3-original-quality-v1',policy['quality_manifest_sha256'])
    report=json.loads((folder/'source.json').read_bytes())
    if storage_profile not in ['mapped-bf16','streamed-fp32'] or report.get('storage_profile','mapped-bf16')!=storage_profile:
        raise ValueError('Reference storage identity differs')
    counts=[n for c in policy['cases']for n in range(c['prefill_tokens'],c['continuation_end']+1)]
    if (report['format']!='toklet-llama3-original-incremental-source-v1' or report['passed']is not True
            or report['contract_sha256']!=contract_sha or report['full_quality_qualified']is not False
            or [c['prefix_tokens']for c in report['cases']]!=counts or report['kv_bytes']!=37093376
            or not 0<report['peak_rss_bytes']<=6*1024**3 or report['overflow_preserves_prefix']is not True
            or report['reset_matches_fresh']is not True or report['all_tied_lowest_id']!=0):
        raise ValueError('Incomplete incremental report')
    rows.replay_rows(folder,report['cases'],ids)
    for case in report['cases']:
        if (case['greedy_matches']is not True or case['consistency']['batch']['greedy_lowest_id']!=case['consistency']['prefix']['greedy_lowest_id']
                or not math.isfinite(case['consistency']['prefix_greedy_margin'])or case['consistency']['prefix_greedy_margin']<0):
            raise ValueError('Incremental greedy conformance differs')
        raw=(folder/case['arrays']['prefix']['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=case['arrays']['prefix']['sha256']:
            raise ValueError('Saved margin row differs')
        top=heapq.nlargest(2,struct.unpack('<128256f',raw))
        margin=struct.unpack('<f',struct.pack('<f',top[0]-top[1]))[0]
        if case['consistency']['prefix_greedy_margin']!=margin:
            raise ValueError('Reported greedy margin does not replay')
    probe=report['decode_probe'];expected=[(r,n)for r in range(5)for n in range(17,21)]
    if (probe['warmups']!=3 or probe['repetitions']!=5 or probe['weight_cache']is not (storage_profile=='streamed-fp32')
            or probe['head_tile']!=256 or probe['publish_throughput_claim']is not False
            or [(s['repetition'],s['context_before'])for s in probe['samples']]!=expected
            or any(type(s['elapsed_ns'])is not int or s['elapsed_ns']<=0 for s in probe['samples'])):
        raise ValueError('Decode probe sample inventory differs')
    if storage_profile=='streamed-fp32':
        plan=report['cache_plan'];tensors=report['cache_tensors'];shape=report['maximum_task_replay_shape']
        if (plan['weight_bytes']!=4943257600 or plan['kv_bytes']!=37093376
                or plan['capacity']!=566 or plan['chunk_elements']!=1048576
                or plan['runtime_scratch_reserve_bytes']!=512*1024**2 or plan['conversion_reserve_bytes']!=64*1024**2
                or plan['admitted_bytes']!=sum(plan[k]for k in ['weight_bytes','kv_bytes','runtime_scratch_reserve_bytes','conversion_reserve_bytes'])
                or plan['admitted_bytes']>plan['memory_limit_bytes'] or plan['memory_limit_bytes']!=6*1024**3
                or report['bf16_source_closed_before_execution']is not True or len(tensors)!=146
                or sum(t['bytes']for t in tensors.values())!=4943257600
                or any(t['bytes']!=math.prod(t['shape'])*4 or t['bytes']<=0
                       or len(t['fp32_sha256'])!=64 or any(c not in '0123456789abcdef'for c in t['fp32_sha256'])
                       for t in tensors.values())
                or shape['input_tokens']!=566 or shape['prompt_tokens']!=310 or shape['visible_tokens']!=256
                or shape['score_rows']!=[309,565] or shape['logits_bytes']!=131334144
                or shape['task_outcomes_observed']is not False):
            raise ValueError('FP32 cache admission or maximum replay facts differ')
    return dict(passed=True,rows=len(counts)*2,greedy_checks=len(counts),decode_probe_samples=20,
                full_quality_qualified=False,task_gate_pending=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path);parser.add_argument('contract',type=Path);parser.add_argument('contract_sha')
    parser.add_argument('--storage-profile',choices=['mapped-bf16','streamed-fp32'],default='mapped-bf16')
    args=parser.parse_args();print(json.dumps(check(args.folder,args.contract,args.contract_sha,args.storage_profile),indent=2))
