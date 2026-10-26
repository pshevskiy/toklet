"""Coherent scalar-oracle controls, before the optimized reference exists."""
from pathlib import Path
import argparse
import copy
import json
import shutil
import struct
import tempfile
import check_gemma_reference_attention_oracle as checker


def controls(directory,sha,output,result):
    checker.verify(directory,sha,{})
    original=checker.safe.document(checker.safe.read(directory/'manifest.json',256*1024))
    result.update(controls=[],passed=False)
    cases=[('q','independent attention input recipe differs'),
           ('scores','independent attention single/math/scores differs'),
           ('probabilities','independent attention single/math/probabilities differs'),
           ('context','independent attention single/math/context differs'),
           ('origin','record value differs'),('scope','record value differs'),('missing','record value differs')]
    for name,error in cases:
        entry=dict(control=name,intended_error=error,actual_error=None,rejected=False,patch=None)
        result['controls'].append(entry)
        with tempfile.TemporaryDirectory(prefix='attention-control-',dir=output.parent)as temporary:
            folder=Path(temporary);manifest=copy.deepcopy(original)
            shutil.copyfile(directory/'arrays.bin',folder/'arrays.bin')
            if name=='origin':manifest['cases'][0]['key_start']+=1
            elif name=='scope':manifest['new_fast_reference_implemented']=True
            elif name=='missing':manifest['arrays'].pop()
            else:
                item=next(v for v in manifest['arrays']if(v['case'],v['contract'],v['name'])==('single','math',name))
                with(folder/'arrays.bin').open('r+b')as file:
                    file.seek(item['offset']);before=file.read(8);value=struct.unpack('<d',before)[0]
                    after=struct.pack('<d',value+1.);file.seek(item['offset']);checker.safe.same(file.write(after),8)
                    file.seek(item['offset']);body=file.read(item['bytes']);item['sha256']=checker.digest(body)
                entry['patch']=dict(offset=item['offset'],before_hex=before.hex(),after_hex=after.hex())
                body=checker.safe.read(folder/'arrays.bin',64*1024**2)
                manifest['binary']=dict(bytes=len(body),sha256=checker.digest(body));del body
            body=checker.encoded(manifest);(folder/'manifest.json').write_bytes(body)
            entry['manifest_sha256']=checker.digest(body);entry['binary']=manifest['binary'];partial={}
            try:checker.verify(folder,checker.digest(body),partial)
            except ValueError as actual:
                entry['actual_error']=str(actual);checker.safe.same(str(actual),error);entry['rejected']=True
            else:raise ValueError('attention control accepted: '+name)
            entry['completed_comparisons_before_rejection']=len(partial.get('comparisons',[]))
    result['passed']=all(v['rejected']for v in result['controls'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--manifest-sha',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=dict(passed=False,primary_error=None)
    try:controls(args.capture,args.manifest_sha,args.output,result)
    except BaseException as error:result.update(passed=False,primary_error=type(error).__name__+': '+str(error))
    finally:
        with args.output.open('xb')as file:file.write(checker.encoded(result))
    print(json.dumps(dict(passed=result['passed'],primary_error=result['primary_error'],controls=len(result.get('controls',[])))))
    raise SystemExit(0 if result['passed']else 1)
