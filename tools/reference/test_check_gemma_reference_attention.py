"""Coherent candidate/timing reader controls with retained exact word patches."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import copy
import json
import shutil
import struct
import tempfile
import check_gemma_reference_attention as checker
import check_gemma_reference_attention_timing as timing


def mutate(folder,manifest,item,index,value,patches):
    size=8 if item['dtype']=='<f8'else 4;code='<d'if size==8 else'<f'
    filename='arrays.bin'if'sequence'not in item else item['sequence']+'.bin'
    with(folder/filename).open('r+b')as file:
        offset=item['offset']+index*size;file.seek(offset);before=file.read(size)
        if value is None:value=struct.unpack(code,before)[0]+1.
        after=struct.pack(code,value);file.seek(offset);checker.safe.same(file.write(after),size)
        file.seek(item['offset']);body=file.read(item['bytes']);item['sha256']=checker.digest(body)
    name='arrays'if filename=='arrays.bin'else item['sequence']
    manifest['files'][name]=checker.tiny.identity(folder/filename,128*1024**2)
    patches.append(dict(file=filename,name=item['name'],offset=offset,before_hex=before.hex(),after_hex=after.hex()))


def cross_patch(args,folder,manifest,patches):
    old=checker.safe.document(checker.safe.read(args.expected/'observation.json',4*1024**2))
    lookup={(v['sequence'],v['contract'],v['name']):v for v in old['arrays']}
    for item in old['arrays']:
        if(item['sequence'],item['contract'])!=('original','math_fp64'):continue
        ideal=checker.load(args.expected,item,'original.bin')
        stage=lookup[('original','staged_fp32',item['name'])]
        reference=checker.load(args.expected,stage,'original.bin')
        for index,(a,b)in enumerate(zip(reference,ideal,strict=True)):
            bound=1e-5+1e-4*abs(b);fraction=abs(a-b)/bound
            if not 0.2<fraction<0.9:continue
            altered=b+(1.1*bound if a>=b else-1.1*bound)
            altered=struct.unpack('<f',struct.pack('<f',altered))[0]
            if abs(altered-a)/(1e-5+1e-4*abs(a))>=0.95 or abs(altered-b)/bound<=1.:continue
            current=next(v for v in manifest['arrays']if(v.get('sequence'),v['contract'],v['name'])==('original','staged_fp32',item['name']))
            mutate(folder,manifest,current,index,altered,patches)
            return dict(name=item['name'],index=index,prior_cross_fraction=fraction,
                        altered_cross_fraction=abs(altered-b)/bound,
                        matching_contract_fraction=abs(altered-a)/(1e-5+1e-4*abs(a)))
    raise ValueError('coherent retained-discrepancy control unavailable')


def controls(args,result):
    result.update(passed=False,controls=[])
    cases=[('standalone','profile','reference reduction profile differs'),
           ('standalone','value','candidate standalone numerical gate failed'),
           ('standalone','mask','nonzero masked reference probability'),
           ('tiny','value','candidate frozen graph numerical gate failed'),
           ('real','value','candidate frozen graph numerical gate failed'),
           ('real','audit','record value differs'),
           ('real','cross','cross-precision discrepancy set changed'),
           ('timing','median','reference attention median acceptance failed'),
           ('timing','missing','record value differs'),
           ('timing','value','timed reference numerical gate failed')]
    for mode,mutation,error in cases:
        source=args.root/('benchmark-a'if mode=='timing'else'tiny-a'if mode=='tiny'else mode)
        captured=checker.safe.document(checker.safe.read(source/'observation.json',4*1024**2))
        entry=dict(mode=mode,mutation=mutation,intended_error=error,actual_error=None,rejected=False,patches=[])
        result['controls'].append(entry)
        with tempfile.TemporaryDirectory(prefix='candidate-control-',dir=args.output.parent)as temp:
            folder=Path(temp);manifest=copy.deepcopy(captured)
            for name in manifest['files']:
                filename='arrays.bin'if name=='arrays'else name+'.bin'
                shutil.copyfile(source/filename,folder/filename)
            if mode=='timing':
                if mutation=='missing':manifest['trials'].pop()
                elif mutation=='median':
                    for trial in manifest['trials']:
                        if(trial['phase'],trial['algorithm'])==('sample','scalar'):trial['elapsed_seconds']=1e-6
                    for case in manifest['cases']:
                        case['algorithms']['scalar']=dict(raw_seconds=[1e-6]*10,median_seconds=1e-6,min_seconds=1e-6,max_seconds=1e-6)
                else:mutate(folder,manifest,manifest['arrays'][0],0,None,entry['patches'])
            elif mutation=='profile':manifest['reduction_profile']='unqualified-reduction'
            elif mutation=='audit':manifest['source_audit']['read_bytes']+=1
            elif mutation=='cross':
                setting=SimpleNamespace(expected=args.real_expected)
                entry['cross_boundary']=cross_patch(setting,folder,manifest,entry['patches'])
            elif mutation=='mask':
                item=next(v for v in manifest['arrays']if(v['case'],v['contract'],v['name'])==('tiny_local','math','probabilities'))
                mutate(folder,manifest,item,0,1e-6,entry['patches'])
            else:mutate(folder,manifest,manifest['arrays'][0],0,None,entry['patches'])
            body=checker.encoded(manifest);(folder/'observation.json').write_bytes(body)
            sha=checker.digest(body);entry['manifest_sha256']=sha;entry['files']=manifest['files'];partial={}
            setting=SimpleNamespace(capture=folder,manifest_sha=sha,mode=mode,
                                    plan=args.tiny_plan if mode=='tiny'else args.real_plan,
                                    plan_sha=args.tiny_plan_sha if mode=='tiny'else args.real_plan_sha,
                                    expected=args.tiny_expected if mode=='tiny'else args.real_expected,
                                    expected_manifest_sha=args.tiny_expected_sha if mode=='tiny'else args.real_expected_sha)
            try:
                if mode=='timing':timing.verify(folder,sha,partial)
                else:checker.verify(setting,partial)
            except ValueError as actual:
                entry['actual_error']=str(actual);checker.safe.same(str(actual),error);entry['rejected']=True
            else:raise ValueError('candidate reader control accepted: '+mode+'/'+mutation)
            entry['completed_comparisons_before_rejection']=len(partial.get('comparisons',[]))
            entry['completed_cross_checks_before_rejection']=len(partial.get('cross_precision',[]))
    result['passed']=all(v['rejected']for v in result['controls'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    for prefix in('tiny','real'):
        parser.add_argument('--'+prefix+'-plan',type=Path,required=True);parser.add_argument('--'+prefix+'-plan-sha',required=True)
        parser.add_argument('--'+prefix+'-expected',type=Path,required=True);parser.add_argument('--'+prefix+'-expected-sha',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=dict(passed=False,primary_error=None)
    try:controls(args,result)
    except BaseException as error:result.update(passed=False,primary_error=type(error).__name__+': '+str(error))
    finally:
        with args.output.open('xb')as file:file.write(checker.encoded(result))
    print(json.dumps(dict(passed=result['passed'],primary_error=result['primary_error'],controls=len(result.get('controls',[])))))
    raise SystemExit(0 if result['passed']else 1)
