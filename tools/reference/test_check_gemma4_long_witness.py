"""Coherent saved-capture controls; source captures remain immutable."""
from pathlib import Path
import argparse
import copy
import json
import shutil
import struct
import tempfile
import check_gemma4_long_witness as checker


def controls(directory, plan, manifest_sha, output, result):
    baseline={};checker.verify(directory,plan,manifest_sha,baseline)
    captured=checker.safe.document(checker.safe.read(directory/'observation.json',1024**2))
    result.update(passed=False,controls=[])
    cases=[('shape','record value differs'),('missing','record value differs'),
           ('scope','record value differs'),
           ('short','frozen short long-witness prefix differs'),
           ('future','future token changed long causal prefix'),
           ('rotary','independent long boundary differs: original/math/layer.2.late_rotary'),
           ('attention','independent long boundary differs: original/math/layer.3.late_attention'),
           ('softcap','independent long boundary differs: original/math/softcap')]
    for mode,error in cases:
        record=dict(control=mode,intended_error=error,rejected=False,actual_error=None,patches=[])
        result['controls'].append(record)
        with tempfile.TemporaryDirectory(prefix='long-control-',dir=output.parent)as temporary:
            destination=Path(temporary);manifest=copy.deepcopy(captured)
            for sequence in plan['tokens']:
                shutil.copyfile(directory/(sequence+'.bin'),destination/(sequence+'.bin'))
            if mode=='shape':manifest['arrays'][0]['shape'][-1]+=1
            elif mode=='missing':manifest['arrays'].pop()
            elif mode=='scope':manifest['native_engine_executed']=True
            else:
                name={'short':'embedding','future':'layer.3.context','rotary':'layer.2.q_rotary',
                      'attention':'layer.3.context','softcap':'logits'}[mode]
                sequences=['original']if mode=='short'else['future']if mode=='future'else['original','future']
                position=0 if mode=='short'else 512 if mode=='future'else 1024
                for sequence in sequences:
                    entry=next(v for v in manifest['arrays']if(v['sequence'],v['contract'],v['name'])==(sequence,'math',name))
                    axis=2 if name.endswith(('rotary','context'))else 1
                    stride=1
                    for n in entry['shape'][axis+1:]:stride*=n
                    offset=entry['offset']+position*stride*8
                    path=destination/(sequence+'.bin')
                    with path.open('r+b')as file:
                        file.seek(offset);before=file.read(8);value=struct.unpack('<d',before)[0]
                        after=struct.pack('<d',value+1.);file.seek(offset);checker.safe.same(file.write(after),8)
                        file.seek(entry['offset']);body=file.read(entry['bytes']);entry['sha256']=checker.digest(body)
                    record['patches'].append(dict(sequence=sequence,name=name,offset=offset,before_hex=before.hex(),after_hex=after.hex()))
                    manifest['files'][sequence]=checker.identity(path,plan['file_bytes'])
            body=checker.encoded(manifest);(destination/'observation.json').write_bytes(body)
            record['mutated_manifest_sha256']=checker.digest(body)
            partial={}
            try:checker.verify(destination,plan,checker.digest(body),partial)
            except ValueError as actual:
                record['actual_error']=str(actual);checker.safe.same(str(actual),error);record['rejected']=True
            else:raise ValueError('coherent long control accepted: '+mode)
            record['completed_comparisons_before_rejection']={key:len(partial.get(key,[]))for key in('short_prefix','future_prefix','cross_precision','boundaries')}
            # Store the exact mutations and identities, sufficient to reconstruct
            # the private controls from the pinned originals without retaining
            # another218MiB for each intended failed comparison.
            record['mutated_files']=manifest['files']
    result['passed']=all(v['rejected']for v in result['controls'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--manifest-sha',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=dict(passed=False,primary_error=None)
    try:controls(args.capture,checker.load_plan(args.plan,args.plan_sha),args.manifest_sha,args.output,result)
    except BaseException as error:result.update(passed=False,primary_error=type(error).__name__+': '+str(error))
    finally:
        with args.output.open('xb')as file:file.write(checker.encoded(result))
    print(json.dumps(dict(passed=result['passed'],primary_error=result['primary_error'],controls=len(result.get('controls',[])))))
    raise SystemExit(0 if result['passed']else 1)
