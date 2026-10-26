"""Independent numerical/sample replay for completed CPU reference timings."""
from array import array
from pathlib import Path
import argparse
import json
import math
import statistics
import sys
import check_gemma_reference_attention as checker


def verify(directory,sha,report):
    raw=checker.safe.read(directory/'observation.json',1024**2)
    checker.safe.require(checker.digest(raw)==sha,'external timing capture differs');capture=checker.safe.document(raw)
    checker.safe.same(set(capture),{'format','passed','primary_error','endpoint_errors','runtime_admission','runtime_endpoint',
        'order','warmups','repetitions','scalar_profile','candidate_profile','memory_allowance_bytes','trials','arrays','cases',
        'files','engine_performance_qualified','phone_qualified'})
    checker.safe.same(capture['format'],'toklet-reference-attention-timing-v1')
    checker.safe.same(capture['passed'],True);checker.safe.same(capture['primary_error'],None);checker.safe.same(capture['endpoint_errors'],[])
    checker.safe.same(capture['scalar_profile'],'scalar-fsum');checker.safe.same(capture['candidate_profile'],checker.PROFILE)
    checker.safe.same((capture['warmups'],capture['repetitions'],capture['memory_allowance_bytes']),(3,10,512*1024**2))
    checker.safe.same(capture['engine_performance_qualified'],False);checker.safe.same(capture['phone_qualified'],False)
    checker.safe.require(capture['order']in('scalar-first','fast-first'),'timing order differs')
    base=['scalar','fast']if capture['order']=='scalar-first'else['fast','scalar']
    meta,expected_raw=checker.scalar_fixture();expected={(v['case'],v['contract'],v['name']):v for v in meta['arrays']}
    signatures=[]
    for case in('global1025_tail','trimmed_local512'):
        for phase,count in [('warmup',3),('sample',10)]:
            for index in range(count):
                for algorithm in(base if index%2==0 else base[::-1]):
                    signatures.append(dict(case=case,phase=phase,index=index,algorithm=algorithm))
    checker.safe.same(len(capture['trials']),len(signatures));checker.safe.same(len(capture['arrays']),156)
    checker.safe.same(set(capture['files']),{'arrays'})
    identity=checker.tiny.identity(directory/'arrays.bin',64*1024**2);checker.safe.same(identity,capture['files']['arrays'])
    report.update(comparisons=[],cases=[],raw_samples={});cursor=0;position=0
    for trial,wanted in zip(capture['trials'],signatures,strict=True):
        checker.safe.same(set(trial),set(wanted)|{'elapsed_seconds','numerical_passed'})
        checker.safe.same({key:trial[key]for key in wanted},wanted);checker.safe.same(trial['numerical_passed'],True)
        checker.safe.require(type(trial['elapsed_seconds'])is float and 0<trial['elapsed_seconds']<600,'bounded completed timing required')
        if trial['phase']=='sample':report['raw_samples'].setdefault(trial['case'],{}).setdefault(trial['algorithm'],[]).append(trial['elapsed_seconds'])
        for name in('context','scores','probabilities'):
            item=capture['arrays'][position];position+=1;prior=expected[(trial['case'],'staged_fp32',name)]
            header=dict(**wanted,name=name,shape=prior['shape'],dtype='<f4',offset=cursor,bytes=prior['bytes'],captured=True)
            checker.safe.same(set(item),set(header)|{'sha256'});checker.safe.same({key:item[key]for key in header},header)
            values=checker.load(directory,item,'arrays.bin');body=expected_raw[prior['offset']:prior['offset']+prior['bytes']]
            checker.safe.require(checker.digest(body)==prior['sha256'],'frozen timing values differ')
            target=array('f');target.frombytes(body)
            if sys.byteorder!='little':target.byteswap()
            result=dict(**wanted,name=name,**checker.compare(values,target));report['comparisons'].append(result)
            checker.safe.require(result['passed'],'timed reference numerical gate failed');cursor+=prior['bytes']
    checker.safe.same(cursor,identity['bytes'])
    checker.safe.same(len(capture['cases']),2)
    for summary in capture['cases']:
        checker.safe.same(set(summary),{'name','contract','geometry','algorithms'})
        case=next(v for v in meta['cases']if v['name']==summary['name'])
        checker.safe.same(summary['geometry'],case);checker.safe.same(summary['contract'],'staged_fp32')
        algorithms={}
        for algorithm in('scalar','fast'):
            values=report['raw_samples'][case['name']][algorithm];checker.safe.same(len(values),10)
            algorithms[algorithm]=dict(raw_seconds=values,median_seconds=statistics.median(values),min_seconds=min(values),max_seconds=max(values))
        checker.safe.same(summary['algorithms'],algorithms)
        ratio=algorithms['fast']['median_seconds']/algorithms['scalar']['median_seconds']
        report['cases'].append(dict(name=case['name'],algorithms=algorithms,fast_to_scalar_ratio=ratio))
        checker.safe.require(ratio<=0.9,'reference attention median acceptance failed')
        checker.safe.require(algorithms['fast']['max_seconds']<=0.95*algorithms['scalar']['min_seconds'],'reference attention observed variation acceptance failed')
    checker.safe.same({v['name']for v in report['cases']},{'global1025_tail','trimmed_local512'})
    checker.safe.same(checker.tiny.identity(directory/'arrays.bin',64*1024**2),identity)
    checker.safe.require(checker.digest(checker.safe.read(directory/'observation.json',1024**2))==sha,'timing manifest endpoint differs')
    report.update(passed=True,warmups=3,repetitions=10,completed_trials=52,arrays_checked=156,
                  compared_values=sum(v['values']for v in report['comparisons']),
                  engine_performance_qualified=False,phone_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--manifest-sha',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=dict(passed=False,primary_error=None)
    try:verify(args.capture,args.manifest_sha,report)
    except BaseException as error:report.update(passed=False,primary_error=type(error).__name__+': '+str(error))
    finally:
        with args.output.open('xb')as file:file.write(checker.encoded(report))
    print(json.dumps({key:report[key]for key in('passed','primary_error')}));sys.exit(0 if report['passed']else 1)
