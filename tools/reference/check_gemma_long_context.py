"""Independent stdlib replay of native long scalar/KV records against frozen bytes."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import struct

ROOT=Path(__file__).resolve().parents[2]
CASES={f'c{c}-chunk{chunk}-uneven{str(u).lower()}':c for c,chunk,u in [(511,32,True),(512,32,True),(513,32,True),(1024,32,True),(1025,1,False),(1025,3,False),(1025,32,False),(1025,32,True)]}

TESTS=dict(native='model::gemma_packed::tests::long_context::frozen_long_logits_kv_boundaries_and_failure_retry',kv='model::shared_kv::tests::long_context::frozen_two_head_w512_segments_routes_and_wrap_commits',generation='model::gemma_packed::generation::tests::long_context::pending_last_position_failure_and_sampled_history_replay')


def counts(capacity, chunk):
    return ([chunk*64]*4+[chunk*2*128]*3+[chunk*128]*2+[chunk*4*64]*4
            +[chunk*64]+[chunk*64]*2+[chunk*128]*3+[2*chunk*capacity]*2+[chunk*17]*2)


def integer(value, low, high, message):
    if type(value)is not int or not low<=value<=high:raise ValueError(message)
    return value


def load(path,pin):
    body=path.read_bytes()
    if hashlib.sha256(body).hexdigest()!=pin:raise ValueError('frozen fixture identity differs')
    return body


def lines(text):
    """Visit one line at a time without retaining another full split transcript."""
    start=0
    while start<len(text):
        end=text.find('\n',start)
        if end<0:end=len(text)
        yield text[start:end]
        start=end+1


def replay(text,mode):
    if not text.isascii()or len(text)>192*1024**2:raise ValueError('capture exceeds bound')
    rows=[];wrapper=[]
    for line in lines(text):
        if 'GEMMA_LONG_'in line:
            before,body=line.split('GEMMA_LONG_',1)
            if before not in ('','test '+TESTS[mode]+' ... '):raise ValueError('wrong native capture wrapper')
            prefix,body=('GEMMA_LONG_'+body).split(' ',1);rows.append((prefix,json.loads(body)))
        elif line.strip():wrapper.append(line.strip())
    if len(wrapper)!=3 or wrapper[:2]!=['running 1 test','ok']or re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s',wrapper[2])is None:raise ValueError('unsuccessful native test wrapper')
    if not rows:raise ValueError('missing long context records')
    compared=0;greedy=0;maximum=0.;coverage={};completed=set();memory=set();views=set();commits=set();summary=False
    if mode in ('native','generation'):
        manifest=json.loads(load(ROOT/'tests/fixtures/gemma-long-witness-v1/manifest.json','bee93a03eb8f3ff6fe1e46919621d8702c1f4eb1a7ada850edb6a987fc750465'))
        binary=load(ROOT/'tests/fixtures/gemma-long-witness-v1/arrays.bin','2940891fd6a32d22c7fa574aad633d8adc3681a1281c23722a97e1111e938ec3')
        specs={(x['contract'],x['name']):x for x in manifest['arrays']if x['sequence']=='original'}
    else:
        fixture=json.loads(load(ROOT/'tests/fixtures/gemma-long-capacity-v1/expected.json','ffeb7e3c375f061de09cc4f47b232dbadf98bcb168e37176a5107f411eaa8f29'))
        steps={(ci,s['before']):s for ci,c in enumerate(fixture['cases'])for s in c['steps']}
    for prefix,r in rows:
        if prefix=='GEMMA_LONG_NATIVE_STATE':
            if mode not in ('native','generation'):raise ValueError('wrong record mode')
            case=r['case'];positions=r['positions'];name=r['name'];values=r['values']
            if type(positions)is not list or not positions or positions!=sorted(set(positions))or any(type(p)is not int or not 0<=p<1025 for p in positions):raise ValueError('invalid captured positions')
            if mode=='native'and(case not in CASES or any(p>=CASES[case]for p in positions)):raise ValueError('native case capacity differs')
            if mode=='generation'and(case,name,positions)not in [('generation-prompt','logits',[1023]),('generation-restored','logits',[1024])]:raise ValueError('generation frozen positions differ')
            for contract in ('math','staged_fp32'):
                spec=specs[contract,name];shape=spec['source_shape'];axis=2 if name.endswith(('rotary','context'))else 1
                heads=math.prod(shape[:axis]);stride=math.prod(shape[axis+1:]);available=spec['positions'];width=8 if spec['dtype']=='<f8'else 4
                if(r['heads'],r['stride'],len(values))!=(heads,stride,heads*len(positions)*stride):raise ValueError('captured state shape differs')
                if any(p not in available for p in positions):raise ValueError('unfrozen captured position')
                for head in range(heads):
                    for i,p in enumerate(positions):
                        origin=spec['offset']+((head*len(available)+available.index(p))*stride)*width
                        target=struct.unpack_from('<'+('d'if width==8 else 'f')*stride,binary,origin)
                        for feature,w in enumerate(target):
                            a=values[(head*len(positions)+i)*stride+feature]
                            if type(a)not in (float,int)or not math.isfinite(a):raise ValueError('nonfinite captured value')
                            ratio=abs(a-w)/(1e-5+1e-4*abs(w));maximum=max(maximum,ratio);compared+=1
                            if ratio>1:raise ValueError('native frozen numerical gate failed')
                        if name=='logits':
                            observed=values[(head*len(positions)+i)*stride:(head*len(positions)+i+1)*stride]
                            if max(range(stride),key=lambda x:observed[x])!=max(range(stride),key=lambda x:target[x]):raise ValueError('native frozen raw argmax differs')
                            greedy+=1
            coverage.setdefault((case,name),set()).update(positions)
        elif prefix=='GEMMA_LONG_NATIVE_MEMORY':
            case=r['case']
            if case not in CASES or r['capacity']!=CASES[case]or r['chunk']!=int(case.split('-chunk')[1].split('-')[0]):raise ValueError('native memory admission differs')
            capacity=r['capacity'];chunk=r['chunk'];expected=counts(capacity,chunk)
            cache=min(4,capacity)*512+capacity*1024;stage=chunk*1536
            if r['scratch_elements']!=expected or r['cache_payload_bytes']!=cache or r['staging_payload_bytes']!=stage or r['history_bytes']!=capacity*4 or(r['cache_owner_capacity'],r['staging_owner_capacity'])!=(2,2):raise ValueError('native memory geometry differs')
            for k in ('session_header_bytes','route_heap_bytes','cache_owner_bytes','staging_owner_bytes','model_bytes'):integer(r[k],1,2*1024**2,'invalid native memory metadata')
            total=r['session_header_bytes']+r['route_heap_bytes']+capacity*4+2*r['cache_owner_bytes']+2*r['staging_owner_bytes']+cache+stage+4*sum(expected)
            if r['planned']!=total or r['actual']!=total:raise ValueError('native memory admission differs')
            if case in memory:raise ValueError('duplicate native memory record')
            memory.add(case)
        elif prefix=='GEMMA_LONG_NATIVE_COMPLETE':
            case=r['case']
            if case not in CASES or r['capacity']!=CASES[case]or any(r[k]is not True for k in ('zero_forward_allocations','failure_retry','overflow_atomic','reset_replay','pointers_stable')):raise ValueError('native completion differs')
            if case in completed:raise ValueError('duplicate native completion')
            completed.add(case)
        elif prefix=='GEMMA_LONG_GENERATION_COMPLETE':
            if mode!='generation'or completed:raise ValueError('generation completion mode/duplicate')
            if(r['capacity'],r['chunk'],r['frozen_prompt_position'],r['pending_before_retry'],r['committed_before_retry'],r['final_committed'])!=(1025,32,1023,1,1024,1025):raise ValueError('generation history semantics differ')
            if any(r[k]is not True for k in ('failure_atomic','sampled_history_replay_bitwise','frozen_prompt_restored','pointers_stable')):raise ValueError('generation completion differs')
            spec=specs['staged_fp32','logits'];w=struct.unpack_from('<17f',binary,spec['offset']+1023*17*4)
            if r['sampled_token']!=max(range(17),key=lambda i:w[i]):raise ValueError('generation frozen prompt argmax differs')
            m=r['memory'];categories=('weights_bytes','kv_bytes','scratch_bytes','history_bytes','logits_bytes','metadata_bytes','metadata_allowance_bytes')
            if set(m)!=set(categories)|{'admitted_bytes'}or any(type(x)is not int or x<0 for x in m.values()):raise ValueError('generation memory ownership differs')
            if sum(m[k]for k in categories)!=m['admitted_bytes']or not 0<m['admitted_bytes']<512*1024**2:raise ValueError('generation memory sum differs')
            if m['history_bytes']!=8200 or m['logits_bytes']!=2*32*17*4 or m['kv_bytes']!=4*512+1025*1024+32*1536 or m['scratch_bytes']!=4*sum(counts(1025,32))-m['logits_bytes']:raise ValueError('generation memory geometry differs')
            completed.add('generation')
        elif prefix=='GEMMA_LONG_KV_VIEW':
            if mode!='kv':raise ValueError('wrong KV mode')
            case=integer(r['case'],0,5,'KV case out of range');before=integer(r['before'],0,1024,'KV offset out of range');layer=integer(r['layer'],0,34,'KV layer out of range')
            integer(r['producer'],0,14,'KV producer out of range');step=steps[case,before];producer=fixture['routes'][layer];expected=step['producers'][producer]
            if r['producer']!=producer or r['count']!=step['count']:raise ValueError('KV route/count differs')
            for k in ('start','end','k_sha256','v_sha256','segments','visible'):
                if r[k]!=expected[k]:raise ValueError('KV '+k+' differs')
            key=(case,before,layer)
            if key in views:raise ValueError('duplicate KV view')
            views.add(key)
        elif prefix=='GEMMA_LONG_KV_COMMIT':
            if mode!='kv':raise ValueError('wrong KV mode')
            integer(r['case'],0,5,'KV case out of range');integer(r['before'],0,1024,'KV offset out of range');integer(r['producer'],0,14,'KV producer out of range')
            step=steps[r['case'],r['before']];expected=step['producers'][r['producer']]
            if r['k_sha256']!=expected['committed_k_sha256']or r['v_sha256']!=expected['committed_v_sha256']or r['offset']!=step['before']+step['count']:raise ValueError('KV committed slots differ')
            key=(r['case'],r['before'],r['producer'])
            if key in commits:raise ValueError('duplicate KV commit')
            commits.add(key)
        elif prefix=='GEMMA_LONG_KV_COMPLETE':
            case=r['case'];expected=fixture['cases'][case]
            if r['capacity']!=expected['capacity']or r['steps']!=len(expected['steps'])or r['pointers_stable']is not True or r['overflow_atomic']is not True:raise ValueError('KV completion differs')
            if case in completed:raise ValueError('duplicate KV completion')
            completed.add(case)
        elif prefix=='GEMMA_LONG_KV_SUMMARY':
            if summary or r!=dict(views=9870,straddling_appends=2):raise ValueError('KV summary differs')
            summary=True
        else:raise ValueError('unexpected long context record')
    if mode=='native':
        if completed!=set(CASES)or memory!=set(CASES):raise ValueError('incomplete native cases')
        for case,capacity in CASES.items():
            for name in ('logits','layer.0.k_rotary','layer.0.self_attn.v_norm','layer.1.k_rotary','layer.1.self_attn.v_norm'):
                if coverage.get((case,name))!=set(range(capacity)):raise ValueError('incomplete native state positions')
        for layer in range(4):
            for name in ('output','self_attn.o_proj','self_attn.q_norm','post_per_layer_input_norm','per_layer_projection','per_layer_input_gate','mlp.up_proj','q_rotary','context'):
                if coverage.get(('c1025-chunk32-uneventrue',f'layer.{layer}.'+name))!=set(manifest['selected_positions']):raise ValueError('incomplete selected late intermediate states')
    elif mode=='generation':
        if completed!={'generation'}or set(coverage)!={('generation-prompt','logits'),('generation-restored','logits')}:raise ValueError('incomplete generation checks')
    elif mode=='kv':
        expected_views={(ci,before,layer)for ci,before in steps for layer in range(35)}
        expected_commits={(ci,before,producer)for ci,before in steps for producer in range(15)}
        if completed!=set(range(6))or views!=expected_views or commits!=expected_commits or not summary:raise ValueError('incomplete KV coverage')
    else:raise ValueError('unknown replay mode')
    return dict(passed=True,mode=mode,records=len(rows),compared_values=compared,raw_argmax_rows=greedy,maximum_gate_fraction=maximum,completed_cases=len(completed),views=len(views),commits=len(commits),native_real_e2b_context_raised=False,phone_qualified=False,full_model_window_qualified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['native','generation','kv']);parser.add_argument('capture',type=Path);args=parser.parse_args()
    print(json.dumps(replay(args.capture.read_text(),args.mode)))


if __name__=='__main__':main()
