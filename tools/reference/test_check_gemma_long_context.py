"""Coherent observed-record mutations reach the intended semantic/numeric gate."""
from pathlib import Path
import argparse
import json
import check_gemma_long_context as reader


def controls(text,mode):
    definitions={
        'native':[
            ('GEMMA_LONG_NATIVE_STATE','value','native frozen numerical gate failed'),
            ('GEMMA_LONG_NATIVE_MEMORY','memory','native memory admission differs'),
            ('GEMMA_LONG_NATIVE_MEMORY','memory_coherent','native memory admission differs'),
            ('GEMMA_LONG_NATIVE_MEMORY','chunk','native memory admission differs'),
            ('GEMMA_LONG_NATIVE_COMPLETE','remove','incomplete native cases')],
        'kv':[
            ('GEMMA_LONG_KV_VIEW','negative_layer','KV layer out of range'),
            ('GEMMA_LONG_KV_COMMIT','negative_producer','KV producer out of range'),
            ('GEMMA_LONG_KV_VIEW','logical','KV k_sha256 differs'),
            ('GEMMA_LONG_KV_VIEW','stride','KV segments differs'),
            ('GEMMA_LONG_KV_VIEW','visible','KV visible differs'),
            ('GEMMA_LONG_KV_COMMIT','commit','KV committed slots differ'),
            ('GEMMA_LONG_KV_COMPLETE','remove','incomplete KV coverage')],
        'generation':[
            ('GEMMA_LONG_NATIVE_STATE','value','native frozen numerical gate failed'),
            ('GEMMA_LONG_GENERATION_COMPLETE','pending','generation history semantics differ'),
            ('GEMMA_LONG_GENERATION_COMPLETE','memory_sum','generation memory sum differs'),
            ('GEMMA_LONG_GENERATION_COMPLETE','memory_geometry','generation memory geometry differs'),
            ('GEMMA_LONG_GENERATION_COMPLETE','remove','incomplete generation checks')],
    }
    results=[]
    for prefix,mutation,intended in definitions[mode]:
        index=text.find(prefix+' ')
        if index<0:raise ValueError('control target absent')
        start=text.rfind('\n',0,index)+1;end=text.find('\n',index)
        if end<0:end=len(text)
        before=text[start:index];r=json.loads(text[index+len(prefix)+1:end])
        if mutation=='remove':replacement=before
        else:
            if mutation=='value':r['values'][0]+=1
            elif mutation=='memory':r['actual']+=1
            elif mutation=='memory_coherent':r['actual']+=4;r['planned']+=4
            elif mutation=='logical':r['k_sha256']='0'*64
            elif mutation=='stride':r['segments'][0]['head_stride']+=1
            elif mutation=='visible':r['visible'][0][0]+=1
            elif mutation=='commit':r['v_sha256']='0'*64
            elif mutation=='pending':r['pending_before_retry']=0
            elif mutation=='chunk':r['chunk']=1
            elif mutation=='negative_layer':r['layer']=-35
            elif mutation=='negative_producer':r['producer']=-15
            elif mutation=='memory_sum':r['memory']['admitted_bytes']+=1
            elif mutation=='memory_geometry':r['memory']['kv_bytes']+=4;r['memory']['admitted_bytes']+=4
            replacement=before+prefix+' '+json.dumps(r,separators=(',',':'))+'\n'
        changed=text[:start]+replacement+text[end+1:]
        try:reader.replay(changed,mode)
        except ValueError as ex:
            actual=str(ex)
            if actual!=intended:raise ValueError('wrong rejection: '+actual+' expected '+intended)from ex
        else:raise ValueError('coherent control accepted')
        results.append(dict(mutation=mutation,intended_error=intended,actual_error=actual,rejected=True))
        del changed
    return dict(passed=True,mode=mode,controls=results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['native','generation','kv']);parser.add_argument('capture',type=Path);args=parser.parse_args()
    print(json.dumps(controls(args.capture.read_text(),args.mode)))


if __name__=='__main__':main()
