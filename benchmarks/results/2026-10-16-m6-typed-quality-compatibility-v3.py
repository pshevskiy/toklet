"""Seven original successes plus strict rejection of the failed 32K attempt.

No native launch and no full eight-record coverage/M6 acceptance.
"""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'Cargo.toml').is_file() and (p/'tools/experiments').is_dir())
sys.path.insert(0,str(ROOT/'tools/experiments'))
import m6_fp32_context_audit as typed


def run():
    source=Path(__file__).resolve(); source_sha=typed.audit.read_bytes(source)[1]
    modules=(typed,typed.values,typed.typed_audit,typed.typed_audit.collector,typed.quality,typed.scope)
    initial={str(Path(m.__file__).resolve().relative_to(ROOT)):typed.audit.read_bytes(Path(m.__file__).resolve())[1] for m in modules}
    paths=[ROOT/'artifacts'/f'm6-context-{model}-quality-{depth}-v{2 if model=="small" and depth==16384 else 1}.json'
           for model in typed.quality.MODELS for depth in typed.quality.DEPTHS
           if (model,depth)!=('larger',32768)]
    loaded=[(p,*typed.audit.read_json(p)) for p in paths]
    proof=typed.replay(paths,loaded)
    for path,_,sha in loaded:
        if proof['bindings'].get(str(path.relative_to(ROOT)))!=sha:raise ValueError('Parsed compatibility input differs')
    restored,precision=typed.restore([r for _,r,_ in loaded])
    reversals=[];count=0
    for record in restored:
        for row in record['worker']['phases'][0]['events'][1:-1]:
            count+=1
            change=typed.quality.winner_accounting(row)
            if change:reversals.append(dict(model=record['model'],depth=record['depth'],**change))
    if count!=896:raise ValueError('Seven-record target coverage differs')
    failed_path=ROOT/'artifacts/m6-context-larger-quality-32768-v1.json'
    failed,failed_sha=typed.audit.read_json(failed_path)
    raw_path=failed_path.with_suffix('.raw.jsonl');raw,raw_sha=typed.audit.read_json(raw_path,lines=True)
    if failed.get('passed') is not False or failed.get('collection_completed') is not False or failed['worker'].get('passed') is not False or 'exit_code' in failed['worker']:
        raise ValueError('Original failure/lack of natural exit must remain explicit')
    manifest,arrays=typed.audit.base.shared.contract();_,long_arrays,_,_,_=typed.audit.base.inputs('small')
    try:typed.audit.replay(failed,raw,manifest,arrays,long_arrays[32768])
    except ValueError as error:
        if str(error)!='Collection not complete: passed':raise
    else:raise ValueError('Failed original was admitted')
    events=failed['worker']['phases'][0]['events']
    if len(events)!=130 or len(raw)!=131:raise ValueError('Original failed raw coverage differs')
    # Diagnostic-only arithmetic: never promotes the failed record to success.
    nll=math.fsum(row['online_nll']-row['reference_nll'] for row in events[1:-1])/128
    pins=dict(proof['bindings'])
    for p,sha in ((source,source_sha),(failed_path,failed_sha),(raw_path,raw_sha)):
        pins[str(p.relative_to(ROOT))]=sha
    for key,sha in initial.items():
        if key in pins and pins[key]!=sha:raise ValueError('Conflicting pre-replay module identity')
        pins[key]=sha
    stderr=failed_path.with_suffix('.stderr');pins[str(stderr.relative_to(ROOT))]=typed.audit.read_bytes(stderr)[1]
    typed.audit.verify_bindings(pins)
    return dict(format='toklet-m6-typed-quality-compatibility-v1',passed=True,
        records=7,scored_targets=count,raw_events=sum(row['raw_events'] for row in proof['records']),
        complete_depth_target_coverage=False,m6_complete=False,failed_original_rejected=True,
        failed_attempt_diagnostic_nll_only=nll,fp32_reconstruction=precision,
        changed_predictions=reversals,replay=proof,bindings=pins,
        scope='Seven original successes replayed; failed32K stays rejected; typed retry not yet qualified; no full quality/M6 acceptance')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Exclusive compatibility output exists')
    result=run()
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('passed','records','scored_targets','raw_events','failed_original_rejected','complete_depth_target_coverage')}))
