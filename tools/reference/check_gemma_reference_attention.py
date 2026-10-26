"""Stdlib candidate replay against pre-target scalar arrays; no candidate import."""
from array import array
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import check_gemma_hybrid_tiny as safe
import check_gemma4_long_witness as tiny
import check_gemma4_bf16_text as real

ROOT=Path(__file__).resolve().parents[2]
PROFILE='fp64-blas-qk-pv-scalar-fsum-softmax-explicit-stage-boundaries-v1'
FALSE=('engine_executed','quality_scored','model_fit_qualified','app_eligibility_changed',
       'phone_qualified','full_model_window_qualified')


def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def load(directory,item,file):
    fd=os.open(directory/file,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb')as stream:
        stream.seek(item['offset']);body=stream.read(item['bytes'])
    safe.require(len(body)==item['bytes']and digest(body)==item['sha256'],'candidate saved array differs')
    values=array('d'if item['dtype']=='<f8'else'f');values.frombytes(body)
    if sys.byteorder!='little':values.byteswap()
    safe.require(all(math.isfinite(v)for v in values),'finite candidate array required')
    return values


def compare(actual,expected):
    safe.same(len(actual),len(expected));maximum=0.;failed=0;examples=[]
    for index,(a,b)in enumerate(zip(actual,expected,strict=True)):
        fraction=abs(a-b)/(1e-5+1e-4*abs(b));maximum=max(maximum,fraction)
        if fraction>1.:
            failed+=1
            if len(examples)<16:examples.append(index)
    return dict(values=len(actual),failed_values=failed,failed_examples=examples,
                maximum_gate_fraction=maximum,passed=failed==0)


def contrast(staged,ideal):
    safe.same(len(staged),len(ideal));maximum=0.;indices=[]
    for index,(a,b)in enumerate(zip(staged,ideal,strict=True)):
        fraction=abs(a-b)/(1e-5+1e-4*abs(b));maximum=max(maximum,fraction)
        if fraction>1.:
            safe.require(len(indices)<1024,'bounded cross-precision discrepancy list required')
            indices.append(index)
    return dict(values=len(ideal),failed_values=len(indices),failed_indices=indices,maximum_gate_fraction=maximum)


def predictions(values,width):
    result=[]
    for before in range(0,len(values),width):
        row=values[before:before+width];winner=max(range(width),key=lambda i:row[i])
        result.append(winner)
    return result


def scalar_fixture():
    folder=ROOT/'tests/fixtures/gemma-reference-attention-v1'
    body=safe.read(folder/'manifest.json',64*1024)
    safe.require(digest(body)=='36365f416c5f9572bfad7d6ff42c51760e43085025857648905350f86924edd8','frozen scalar manifest differs')
    meta=safe.document(body);raw=safe.read(folder/'arrays.bin',8*1024**2)
    safe.same(meta['binary'],dict(bytes=len(raw),sha256=digest(raw)))
    safe.require(digest(raw)=='f7a13f39536c816d566a5caf167b6f439f11b59b34ae206561b93d5ac28a1b7f','frozen scalar arrays differ')
    return meta,raw


def verify(args,report):
    raw=safe.read(args.capture/'observation.json',4*1024**2)
    safe.require(digest(raw)==args.manifest_sha,'external candidate capture differs')
    capture=safe.document(raw)
    fields={'format','mode','reduction_profile','passed','primary_error','endpoint_errors','arrays','inputs','files',
            'source_audit','tokens','runtime_admission','runtime_endpoint','memory_allowance_bytes',*FALSE}
    if args.mode=='standalone':fields.add('cases')
    safe.same(set(capture),fields);safe.same(capture['format'],'toklet-reference-attention-candidate-v1')
    safe.same(capture['mode'],args.mode);safe.require(capture['reduction_profile']==PROFILE,'reference reduction profile differs')
    safe.same(capture['passed'],True);safe.same(capture['primary_error'],None);safe.same(capture['endpoint_errors'],[])
    safe.same(capture['memory_allowance_bytes'],512*1024**2)
    safe.require(type(capture['runtime_endpoint'])is dict and bool(capture['runtime_endpoint']),'runtime endpoint required')
    for key in FALSE:safe.same(capture[key],False)
    report.update(comparisons=[],cross_precision=[],greedy=[],masked_probability_values=0)
    if args.mode=='standalone':
        meta,expected_raw=scalar_fixture()
        safe.same(capture['cases'],meta['cases']);safe.same(capture['tokens'],None);safe.same(capture['source_audit'],None)
        safe.same(capture['inputs'],[{key:value for key,value in v.items()if key!='source_offset'}for v in meta['inputs']])
        safe.same(len(capture['arrays']),48)
        safe.same(set(capture['files']),{'arrays'})
        observed=tiny.identity(args.capture/'arrays.bin',meta['binary']['bytes'])
        safe.same(observed,capture['files']['arrays']);safe.same(observed['bytes'],meta['binary']['bytes'])
        for actual,wanted in zip(capture['arrays'],meta['arrays'],strict=True):
            safe.same(set(actual),{'case','contract','name','shape','dtype','offset','bytes','sha256','captured'})
            header={key:wanted[key]for key in('case','contract','name','shape','dtype','offset','bytes')}
            safe.same({key:actual[key]for key in header},header);safe.same(actual['captured'],True)
            values=load(args.capture,actual,'arrays.bin');old=array('d'if wanted['dtype']=='<f8'else'f')
            body=expected_raw[wanted['offset']:wanted['offset']+wanted['bytes']]
            safe.require(digest(body)==wanted['sha256'],'scalar array digest differs');old.frombytes(body)
            if sys.byteorder!='little':old.byteswap()
            result=dict(case=actual['case'],contract=actual['contract'],name=actual['name'],**compare(values,old))
            report['comparisons'].append(result);safe.require(result['passed'],'candidate standalone numerical gate failed')
            if actual['name']=='probabilities':
                case=next(c for c in meta['cases']if c['name']==actual['case'])
                for head in range(case['heads']):
                    for query in range(case['queries']):
                        position=case['query_start']+query
                        start=0 if case['window']is None else max(0,position-case['window']+1-case['key_start'])
                        end=position-case['key_start']+1;base=(head*case['queries']+query)*case['keys']
                        for key in list(range(start))+list(range(end,case['keys'])):
                            safe.require(values[base+key]==0.,'nonzero masked reference probability')
                            report['masked_probability_values']+=1
        safe.same(tiny.identity(args.capture/'arrays.bin',meta['binary']['bytes']),observed)
    else:
        specification=(tiny.load_plan if args.mode=='tiny'else real.load_plan)(args.plan,args.plan_sha)
        old_body=safe.read(args.expected/'observation.json',4*1024**2)
        safe.require(digest(old_body)==args.expected_manifest_sha,'frozen full graph capture differs')
        old=safe.document(old_body);safe.same(old['passed'],True)
        safe.same(old['tokens'],specification['tokens']);safe.same(capture['tokens'],specification['tokens'])
        safe.same(capture['inputs'],[])
        if args.mode=='real':safe.same(capture['source_audit'],specification['source_audit'])
        else:safe.same(capture['source_audit'],None)
        safe.same(set(capture['files']),set(specification['tokens']))
        for sequence in specification['tokens']:
            safe.same(tiny.identity(args.expected/(sequence+'.bin'),specification['file_bytes']),old['files'][sequence])
            safe.same(tiny.identity(args.capture/(sequence+'.bin'),specification['file_bytes']),capture['files'][sequence])
            safe.same(capture['files'][sequence]['bytes'],specification['file_bytes'])
        safe.same(len(old['arrays']),len(specification['captures']));safe.same(len(capture['arrays']),len(specification['captures']))
        current={};frozen={}
        for actual,prior,wanted in zip(capture['arrays'],old['arrays'],specification['captures'],strict=True):
            for entry in(actual,prior):
                safe.same(set(entry),set(wanted)|{'sha256','captured'})
                safe.same({key:entry[key]for key in wanted},wanted);safe.same(entry['captured'],True)
            key=tuple(actual[name]for name in('sequence','contract','name'));current[key]=actual;frozen[key]=prior
            values=load(args.capture,actual,actual['sequence']+'.bin');expected=load(args.expected,prior,prior['sequence']+'.bin')
            result=dict(sequence=actual['sequence'],contract=actual['contract'],name=actual['name'],**compare(values,expected))
            report['comparisons'].append(result);safe.require(result['passed'],'candidate frozen graph numerical gate failed')
            if actual['name']=='logits':
                width=actual['shape'][-1];a=predictions(values,width);b=predictions(expected,width)
                report['greedy'].append(dict(sequence=actual['sequence'],contract=actual['contract'],positions=len(a),passed=a==b))
                safe.require(a==b,'candidate frozen greedy predictions differ')
        ideal='math'if args.mode=='tiny'else'math_fp64'
        for sequence in specification['tokens']:
            names=[item['name']for item in specification['captures']if(item['sequence'],item['contract'])==(sequence,ideal)]
            for name in names:
                previous=contrast(load(args.expected,frozen[(sequence,'staged_fp32',name)],sequence+'.bin'),load(args.expected,frozen[(sequence,ideal,name)],sequence+'.bin'))
                changed=contrast(load(args.capture,current[(sequence,'staged_fp32',name)],sequence+'.bin'),load(args.capture,current[(sequence,ideal,name)],sequence+'.bin'))
                report['cross_precision'].append(dict(sequence=sequence,name=name,prior=previous,candidate=changed))
                safe.require(previous['failed_indices']==changed['failed_indices'],'cross-precision discrepancy set changed')
        for sequence in specification['tokens']:
            safe.same(tiny.identity(args.expected/(sequence+'.bin'),specification['file_bytes']),old['files'][sequence])
            safe.same(tiny.identity(args.capture/(sequence+'.bin'),specification['file_bytes']),capture['files'][sequence])
        safe.require(digest(safe.read(args.expected/'observation.json',4*1024**2))==args.expected_manifest_sha,'expected graph endpoint differs')
    safe.require(digest(safe.read(args.capture/'observation.json',4*1024**2))==args.manifest_sha,'candidate capture endpoint differs')
    report.update(passed=True,mode=args.mode,reduction_profile=PROFILE,
                  compared_values=sum(v['values']for v in report['comparisons']),
                  retained_prior_cross_precision_failures=sum(v['prior']['failed_values']for v in report['cross_precision']),
                  candidate_cross_precision_failures=sum(v['candidate']['failed_values']for v in report['cross_precision']),
                  **{key:False for key in FALSE})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['standalone','tiny','real'],required=True)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--plan',type=Path);parser.add_argument('--plan-sha');parser.add_argument('--expected',type=Path)
    parser.add_argument('--expected-manifest-sha');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=dict(passed=False,primary_error=None)
    try:verify(args,report)
    except BaseException as error:report.update(passed=False,primary_error=type(error).__name__+': '+str(error))
    finally:
        with args.output.open('xb')as file:file.write(encoded(report))
    print(json.dumps({key:report[key]for key in('passed','primary_error')}));sys.exit(0 if report['passed']else 1)
