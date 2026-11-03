"""Validate paired 10000-target statistics and apply the frozen loss gate.

Per-target statistics and row hashes are authenticated producer outputs, not
saved full distributions. NLL formula/coverage/means replay independently.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import check_llama3_loss1024 as common
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import prepare_llama3_original_quality as inputs
import acquire_llama3_original as original


def collect(records, ids, blocks):
    expected={i:(b['block'],i-b['input_start']-1) for b in blocks for i in range(*b['global_target_indices'])}
    values={}
    for record in records:
        index=record.get('global_target_index')
        if type(index) is not int or index not in expected or index in values:
            raise ValueError('Duplicate or invalid loss target index')
        if (record.get('block'),record.get('local_row'))!=expected[index]:
            raise ValueError('Loss position/reset differs')
        values[index]=common.statistics(record,ids[index])
    if set(values)!=set(range(1,10001)):raise ValueError('Incomplete loss targets')
    return [values[i] for i in range(1,10001)]


def read_journal(folder, report):
    path=folder/'targets.jsonl'
    with path.open('rb') as stream:raw=stream.read(8*1024**2+1)
    if len(raw)>8*1024**2 or hashlib.sha256(raw).hexdigest()!=report['targets_sha256']:
        raise ValueError('Target journal differs')
    return [json.loads(line,object_pairs_hook=inputs.parent.unique) for line in raw.splitlines()]


def alignment_crosscheck(records, source_folder, source_sha, native_folder, native_sha):
    previous=[]
    for folder, expected in [(source_folder,source_sha),(native_folder,native_sha)]:
        filename='source.json' if len(previous)==0 else 'result.json'
        report,raw=common.bounded_json(Path(folder)/filename)
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError('Pinned alignment report differs')
        previous.append(report)
    source,native=previous
    if (source['format']!='toklet-llama3-original-long-source-v1' or not source['passed']
            or native['format']!='toklet-llama3-quality-context-v1' or not native['ok']):
        raise ValueError('Prior alignment did not pass')
    prior_source={c['prefix_tokens']:c['consistency']['prefix']['nll'] for c in source['cases']}
    prior_native={c['prefix']:c['numeric_sha256'] for c in native['observations'] if c['kind']=='independent'}
    by_index=[{r['global_target_index']:r for r in rows}for rows in records]
    for index in [512,513,1024]:
        if abs(by_index[0][index]['nll']-prior_source[index])>1e-4:
            raise ValueError('Original full-loss alignment NLL differs')
        if by_index[1][index].get('numeric_row_sha256')!=prior_native[index]:
            raise ValueError('Native full-loss numerical row differs')
    return [512,513,1024]


def check(input_folder, manifest_sha, source_folder, source_sha, native_folder, native_sha, alignment=None):
    manifest,ids=inputs.load(input_folder,manifest_sha)
    folders=[Path(source_folder),Path(native_folder)];reports=[]
    for folder,sha in zip(folders,[source_sha,native_sha]):
        report,raw=common.bounded_json(folder/'result.json')
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Pinned loss report differs')
        if (report['completed'] is not True or report['full_quality_qualified'] is not False
                or report['manifest_sha256']!=manifest_sha or report['targets']!=10000):
            raise ValueError('Incomplete loss report')
        reports.append(report)
    source,native=reports
    if (source['format']!='toklet-llama3-quality-loss-original-v1'
            or source['model']!=dict(repository=original.META,revision=original.META_REVISION,weight_sha256=original.WEIGHT_SHA)
            or not 0<source['peak_rss_bytes']<=6*1024**3
            or source['blocks']!=[dict(block=b['block'],targets=b['score_rows'][1],input_positions=b['input_end']-b['input_start'])for b in manifest['blocks']]):
        raise ValueError('Original loss identity differs')
    if (native['format']!='toklet-llama3-quality-loss-native-v1' or native['repository']!=inputs.parent.REPOSITORY
            or native['revision']!=inputs.parent.REVISION or not 0<native['admitted_bytes_including_app']<=6*1024**3
            or native['profile']!=dict(projection='MlxAffine',normalization='Compensated',attention='Reference',chunk=32,capacity=1025)):
        raise ValueError('Native loss identity/profile differs')
    records=[read_journal(folder,report)for folder,report in zip(folders,reports)]
    values=[collect(rows,ids,manifest['blocks'])for rows in records]
    aligned=alignment_crosscheck(records,*alignment) if alignment is not None else []
    means=[math.fsum(v)/10000 for v in values]
    if any(not common.close(report['mean_nll'],mean)for report,mean in zip(reports,means)):
        raise ValueError('Recorded loss mean differs')
    delta=means[1]-means[0]
    return dict(completed=True,targets=10000,original_mean_nll=means[0],native_mean_nll=means[1],
                native_minus_original=delta,mean_nll_increase_limit=0.05,loss_gate_passed=delta<=0.05,
                full_quality_qualified=False,task_gate_pending=True,alignment_crosscheck_indices=aligned)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','manifest_sha','source','source_sha','native','native_sha']:parser.add_argument(name)
    for name in ['alignment-source','alignment-source-sha256','alignment-native','alignment-native-sha256']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    print(json.dumps(check(args.inputs,args.manifest_sha,args.source,args.source_sha,args.native,args.native_sha,(args.alignment_source,args.alignment_source_sha256,args.alignment_native,args.alignment_native_sha256)),indent=2))
