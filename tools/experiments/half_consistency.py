"""Bounded all-layer Half consistency capture; no arithmetic or quality adoption."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import time

import metal_engine
import native_usage

ROOT=Path(__file__).resolve().parents[2]
BASELINE_SHA='301c6a9a245153d93e401dae93208facd263e68e4e1a446beacb11c93c6d6767'
MODEL_SHA='dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
CONFIG_SHA='98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670'
STAGES=('input_norm','q','k','v','q_rotary','k_rotary','attention_context','attention_projection','attention_residual','post_attention_norm','gate','silu','up','gated','down','output')
LIMIT=6*1024**3

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def baseline_record():
    with (ROOT/'benchmarks/results/2026-10-14-larger-half-held-out-loss-v1.json').open('rb') as f:data=f.read(2*1024**2+1)
    if len(data)>2*1024**2 or hashlib.sha256(data).hexdigest()!=BASELINE_SHA:raise ValueError('Exact bounded baseline bytes differ')
    return json.loads(data)

def projection_geometry(first):
    stage=first['stage']
    choices={'q':('input_norm',1536,1536,True),'k':('input_norm',256,1536,True),'v':('input_norm',256,1536,True),
             'attention_projection':('attention_context',1536,1536,False),'gate':('post_attention_norm',8960,1536,False),
             'up':('post_attention_norm',8960,1536,False),'down':('gated',1536,8960,False),'logits':('final_norm',151936,1536,False)}
    if stage not in choices or (first['layer'] is None)!=(stage=='logits'):raise ValueError('First differing operation is not a projection')
    return choices[stage]

def half_value(bits):
    e=(bits>>10)&31;m=bits&1023
    if e==31:raise ValueError('Exact oracle requires finite Half')
    value=math.ldexp(m if e==0 else 1024+m,-24 if e==0 else e-25)
    return -value if bits&0x8000 else value

def exact_half(numerator,denominator_power):
    """Integer round-to-nearest/ties-even of an exact dyadic rational."""
    sign=0x8000 if numerator<0 else 0;n=abs(numerator);den=1<<denominator_power
    if n>=65520*den:return sign|0x7c00
    if n>=65504*den:return sign|0x7bff
    # Use integer cross products, including when the input denominator is small.
    def compare(bits):
        num,d=half_value(bits).as_integer_ratio();return n*d-num*den
    lo,hi=0,0x7bff
    while lo<hi:
        middle=(lo+hi+1)//2
        if compare(middle)>=0:lo=middle
        else:hi=middle-1
    if compare(lo)==0:return sign|lo
    lower,ld=half_value(lo).as_integer_ratio();upper,ud=half_value(lo+1).as_integer_ratio()
    difference=2*n*ld*ud-(lower*ud+upper*ld)*den
    rounded=lo if difference<0 or (difference==0 and lo%2==0) else lo+1
    return sign|rounded

def dyadic_dot(x,w,bias):
    terms=[]
    for a,b in zip(x,w,strict=True):
        an,ad=float(a).as_integer_ratio();bn,bd=float(b).as_integer_ratio();terms.append((an*bn,(ad*bd).bit_length()-1))
    n,d=float(bias).as_integer_ratio();terms.append((n,d.bit_length()-1));power=max(p for _,p in terms)
    numerator=sum(n<<(power-p) for n,p in terms)
    return exact_half(numerator,power),numerator,power

def inventory(all_layers=True):
    result=[(None,'embedding',1536)]
    if all_layers:
        for layer in range(28):
            for stage in STAGES:
                width=256 if stage in ('k','v','k_rotary') else 8960 if stage in ('gate','silu','up','gated') else 1536
                result.append((layer,stage,width))
    return result+[(None,'final_norm',1536),(None,'logits',151936)]

def array(directory,item):
    import numpy as np
    name=item['file'];shape=item['shape'];dtype=item['dtype']
    if (type(name) is not str or Path(name).name!=name or type(shape) is not list or not shape
            or any(type(v) is not int or not 0<v<=151936 for v in shape) or dtype not in ('f32le','f16le')):raise ValueError('Array path/shape/dtype invalid')
    size=math.prod(shape)*(4 if dtype=='f32le' else 2)
    if type(item['bytes']) is not int or item['bytes']!=size or not 0<size<=10*1024**2:raise ValueError('Array byte bound differs')
    path=directory/name
    with path.open('rb') as f:data=f.read(size+1)
    if len(data)!=size or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Array frozen bytes differ')
    return np.frombuffer(data,dtype='<u4' if dtype=='f32le' else '<u2').reshape(shape)

def validate_inventory(record,all_layers):
    expected=inventory(all_layers)
    if len(record['stages'])!=len(expected):raise ValueError('Complete stage coverage differs')
    for item,(layer,stage,width) in zip(record['stages'],expected,strict=True):
        if (item.get('layer'),item.get('stage'),item.get('shape'),item.get('half_storage'),item.get('dtype'))!=(layer,stage,[4,width],stage!='logits','f32le'):raise ValueError('Stage identity/shape/dtype differs')
    if all_layers:
        if len(record['cache'])!=56:raise ValueError('Cache coverage differs')
        for item,(layer,kind) in zip(record['cache'],((l,k) for l in range(28) for k in ('k','v')),strict=True):
            if (item['layer'],item['kind'],item['shape'],item['dtype'])!=(layer,kind,[2,4,128],'f16le'):raise ValueError('Cache identity/shape differs')
    if type(record['admitted_bytes_including_extra']) is not int or not 0<record['admitted_bytes_including_extra']<=LIMIT:raise ValueError('Capture admission differs')

def analyze(directory):
    import numpy as np
    path=directory/'manifest.json'
    if path.stat().st_size>2*1024**2:raise ValueError('Capture manifest exceeds bound')
    m=json.loads(path.read_text())
    if (m['format']!='toklet-half-consistency-capture-v1' or m['tokens']!=[9707,11,358,498] or m['capacity']!=1025 or m['chunks']!=[1,128]
            or m['projection_mode']!='Original' or m['normalization_mode']!='Original' or m['baseline_sha256']!=BASELINE_SHA
            or m['extra_admitted_bytes']!=256*1024**2 or not 0<m['retained_host_words_bytes']<128*1024**2):raise ValueError('Capture scope/configuration differs')
    for name in ('single_base','batch_base','single','batch'):validate_inventory(m[name],name in ('single','batch'))
    baseline=baseline_record();baseline_matches=[]
    for index,mode in enumerate(('single_base','batch_base')):
        words=array(directory,m[mode]['stages'][2]);values=words.view('<f4')
        if not np.isfinite(values).all():raise ValueError('Nonfinite baseline logits')
        for row in range(3):
            saved=baseline['runs'][index]['losses'][row];target=m['tokens'][row+1];v=values[row].astype(np.float64);maximum=float(np.max(v))
            nll=maximum-float(v[target])+math.log(math.fsum(math.exp(float(x)-maximum) for x in v))
            if (saved['target_index']!=row or saved['target_token']!=target or saved['target_logit_f32_bits']!=int(words[row,target])
                    or saved['logits_sha256']!=hashlib.sha256(words[row].tobytes()).hexdigest() or abs(nll-saved['negative_log_likelihood'])>1e-12):raise ValueError('Captured baseline differs from pinned failed scorer')
            baseline_matches.append({'mode':mode,'row':row,'logits_sha256':saved['logits_sha256'],'independent_negative_log_likelihood':nll})
    stats=[];max_admission=max(m[k]['admitted_bytes_including_extra'] for k in ('single_base','batch_base','single','batch'))
    for order,(a,b) in enumerate(zip(m['single']['stages'],m['batch']['stages'],strict=True)):
        aw,bw=array(directory,a),array(directory,b);af,bf=aw.view('<f4'),bw.view('<f4')
        if not np.isfinite(af).all() or not np.isfinite(bf).all():raise ValueError('Captured nonfinite stage')
        mask=aw!=bw;count=int(np.count_nonzero(mask));first=np.argwhere(mask)[0].tolist() if count else None
        maximum=float(np.max(np.abs(af.astype(np.float64)-bf.astype(np.float64))))
        ulp=None
        if a['half_storage']:
            ah,bh=af.astype(np.float16).view(np.uint16).astype(np.int32),bf.astype(np.float16).view(np.uint16).astype(np.int32)
            ak=np.where(ah&0x8000,0x8000-(ah&0x7fff),0x8000+ah);bk=np.where(bh&0x8000,0x8000-(bh&0x7fff),0x8000+bh);ulp=int(np.max(np.abs(ak-bk)))
        stats.append({'order':order,'layer':a['layer'],'stage':a['stage'],'unequal_values':count,'first_coordinate':first,'maximum_absolute_difference':maximum,'maximum_half_ulp_distance':ulp})
    if stats!=m['stage_differences']:raise ValueError('Native stage-difference analysis differs')
    perturbation=[]
    for base,full in [('single_base','single'),('batch_base','batch')]:
        items={(v['layer'],v['stage']):v for v in m[full]['stages']}
        perturbation.append(all(np.array_equal(array(directory,v),array(directory,items[(v['layer'],v['stage'])])) for v in m[base]['stages']))
    if perturbation!=m['diagnostics_preserve_scorer_stage_bits']:raise ValueError('Diagnostic perturbation record differs')
    cache_equal=[]
    for index,(a,b) in enumerate(zip(m['single']['cache'],m['batch']['cache'],strict=True)):
        aw,bw=array(directory,a),array(directory,b);cache_equal.append(bool(np.array_equal(aw,bw)))
        layer=index//2;stage='k_rotary' if index%2==0 else 'v'
        for mode,words in [('single',aw),('batch',bw)]:
            item=next(v for v in m[mode]['stages'] if v['layer']==layer and v['stage']==stage)
            reference=array(directory,item).view('<f4').astype(np.float16).view(np.uint16).reshape(4,2,128).transpose(1,0,2)
            if not np.array_equal(words,reference):raise ValueError('Committed cache does not match captured complete prefix')
    replay=m['projection_replay'];oracle=None
    if 'skipped' not in replay:
        first=next((s for s in stats if s['unequal_values']),None)
        if replay['first']!=first:raise ValueError('Replay first divergence differs')
        predecessor,out,in_width,has_bias=projection_geometry(first)
        if (replay['complete_inputs_equal'] is not True or replay['full_weight_shape']!=[out,in_width] or replay['float_output']!=(first['stage']=='logits')
                or replay['bias_is_zero']!= (not has_bias) or (replay['resident_logical_half_bias_sha256'] is None)!= (not has_bias)):raise ValueError('Replay projection metadata differs')
        input_words=array(directory,replay['input'])
        for mode in ('single','batch'):
            item=next(v for v in m[mode]['stages'] if v['layer']==first['layer'] and v['stage']==predecessor)
            if not np.array_equal(input_words,array(directory,item)):raise ValueError('Replay complete inputs differ from captured predecessor')
        inputs=array(directory,replay['input']).view('<f4').astype(np.float64);weights=array(directory,replay['weights']).view('<f2').astype(np.float64);bias=array(directory,replay['bias']).view('<f2').astype(np.float64)
        indices=replay['weight_output_indices']
        if (indices!=sorted(set(indices)) or not 1<=len(indices)<=64 or any(type(i) is not int or not 0<=i<out for i in indices)
                or inputs.shape!=(4,in_width) or weights.shape!=(len(indices),in_width) or bias.shape!=(len(indices),)):raise ValueError('Replay selected matrix geometry differs')
        if not np.isfinite(inputs).all() or not np.isfinite(weights).all() or not np.isfinite(bias).all() or (not has_bias and np.any(bias!=0)):raise ValueError('Replay nonfinite/zero-bias operands differ')
        actual_words=array(directory,m['single']['stages'][first['order']]);other_words=array(directory,m['batch']['stages'][first['order']])
        expected_indices=np.flatnonzero(np.any(actual_words!=other_words,axis=0))[:64].tolist()
        if indices!=expected_indices:raise ValueError('Replay selected divergence indices differ')
        want=inputs@weights.T+bias;raw={};stored={}
        for mode,chunk,n in [('single',1,1),('batch',128,4)]:
            v=replay[mode]
            if (v['chunk_capacity']!=chunk or v['active_rows_per_call']!=n or v['input_stride_half_elements']!=in_width or v['result_stride_f32_elements']!=151936
                    or v['weight_stride_half_elements']!=in_width or v['stored_stride_half_elements']!=(None if replay['float_output'] else out)
                    or v['alpha']!=1 or v['beta']!=1 or v['transpose_left'] is not False or v['transpose_right'] is not True):raise ValueError('MPS replay descriptor/construction differs')
            raw[mode]=array(directory,v['raw']).view('<f4')
            if raw[mode].shape!=(4,out):raise ValueError('Replay output coverage differs')
            stored[mode]=raw[mode] if replay['float_output'] else array(directory,v['stored']).view('<f4')
            actual_item=m[mode]['stages'][replay['first']['order']];actual=array(directory,actual_item).view('<f4')
            if bool(np.array_equal(stored[mode].view('<u4'),actual.view('<u4')))!=v['matches_graph_output_bits']:raise ValueError('Replay graph-match flag differs')
            if not replay['float_output'] and not np.array_equal(raw[mode].astype(np.float16).astype(np.float32).view('<u4'),stored[mode].view('<u4')):raise ValueError('Replay stored Half does not match raw FP32 rounding')
        witnesses=[]
        if not replay['float_output']:
            differing=stored['single'][:,indices].view('<u4')!=stored['batch'][:,indices].view('<u4')
            for row,col in np.argwhere(differing)[:32]:
                row,col=int(row),int(col);exact_bits,num,power=dyadic_dot(inputs[row],weights[col],bias[col]);out_index=indices[col]
                witnesses.append({'row':row,'output':out_index,'exact_half_bits':exact_bits,'exact_numerator':str(num),'exact_denominator_power':power,
                    'fp64_reference':float(want[row,col]),'single_f32_bits':int(raw['single'].view('<u4')[row,out_index]),'batch_f32_bits':int(raw['batch'].view('<u4')[row,out_index]),
                    'single_half_bits':int(stored['single'][row,out_index].astype(np.float16).view(np.uint16)),'batch_half_bits':int(stored['batch'][row,out_index].astype(np.float16).view(np.uint16))})
        oracle={'selected_output_rows':indices,'single_maximum_fp64_absolute_error':float(np.max(np.abs(raw['single'][:,indices].astype(np.float64)-want))),
                'batch_maximum_fp64_absolute_error':float(np.max(np.abs(raw['batch'][:,indices].astype(np.float64)-want))),
                'single_matches_graph':replay['single']['matches_graph_output_bits'],'batch_matches_graph':replay['batch']['matches_graph_output_bits'],'exact_dyadic_half_witnesses':witnesses}
    return {'format':'toklet-half-consistency-analysis-v1','complete_stage_pairs':451,'baseline_scorer_matches':baseline_matches,'diagnostics_preserve_scorer_stage_bits':perturbation,'maximum_admitted_bytes':max_admission,
            'first_differing_stage':next((s for s in stats if s['unequal_values']),None),'first_numerically_differing_stage':next((s for s in stats if s['maximum_absolute_difference']>0),None),
            'complete_cache_prefix_pairs_equal':cache_equal,'projection_oracle':oracle,'stage_differences':stats,'scope':'Localization only; no production arithmetic/full-loss acceptance'}

def collect(binary,directory,output):
    paths=[output,output.with_suffix('.stdout'),output.with_suffix('.stderr')]
    if directory.exists() or any(p.exists() for p in paths):raise ValueError('Exclusive capture/evidence destination exists')
    files=[Path(__file__),Path(metal_engine.__file__),Path(native_usage.__file__),binary,ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'src/metal/half_consistency.rs',ROOT/'examples/support/loss.rs',ROOT/'benchmarks/results/2026-10-14-larger-half-held-out-loss-v1.json']+list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))
    identities={str(p):digest(p) for p in files};baseline=ROOT/'benchmarks/results/2026-10-14-larger-half-held-out-loss-v1.json'
    for name,expected in [('model.safetensors',MODEL_SHA),('config.json',CONFIG_SHA)]:
        path=ROOT/'models/Qwen2.5-1.5B-Instruct'/name;identities[str(path)]=digest(path)
        if identities[str(path)]!=expected:raise ValueError('Pinned source/config differs')
    if identities[str(baseline)]!=BASELINE_SHA:raise ValueError('Frozen failed probe record differs')
    result={'format':'toklet-half-consistency-collection-v1','passed':False,'identities':identities,'cleanup_errors':[]};worker=None;selector=None;start=time.monotonic();buffers={'stdout':bytearray(),'stderr':bytearray()};seen=False
    with paths[0].open('x') as report,paths[1].open('xb') as stdout,paths[2].open('xb') as stderr:
        try:
            worker=subprocess.Popen([str(binary),'metal::half_consistency::larger_half_probe_capture','--exact','--ignored','--nocapture','--test-threads=1'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env=dict(os.environ,TOKLET_HALF_CAPTURE_DIR=str(directory)))
            selector=selectors.DefaultSelector()
            for name,stream in [('stdout',worker.stdout),('stderr',worker.stderr)]:os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,name)
            while selector.get_map():
                if time.monotonic()-start>600:raise TimeoutError('Bounded native capture timeout')
                for key,_ in selector.select(.25):
                    data=os.read(key.fileobj.fileno(),65536)
                    if not data:selector.unregister(key.fileobj);continue
                    name=key.data;buffers[name].extend(data)
                    if len(buffers[name])>1024**2:raise ValueError('Native capture output exceeds bound')
                    (stdout if name=='stdout' else stderr).write(data)
                    if name=='stdout' and not seen and b'\nTOKLET_HALF_CAPTURE_DONE\n' in buffers[name]:
                        seen=True;left=native_usage.usage(worker.pid);right=native_usage.usage();charge=left['lifetime_max_phys_footprint']+right['lifetime_max_phys_footprint']
                        if not 0<charge<=LIMIT:raise ValueError('Capture joint native charge exceeds6GiB')
                        result['idle_usage']={'worker':left,'driver':right,'joint_native_lifetime_peak_bytes':charge};worker.stdin.write(b'continue\n');worker.stdin.flush()
            exit_code=worker.wait(timeout=5);result['exit_code']=exit_code
            if exit_code!=0 or not seen or b'test result: ok. 1 passed' not in buffers['stdout']:raise ValueError('Native capture did not execute one passing test')
            # Analysis and large endpoint hashes run only after GPU worker completion.
            result['analysis']=analyze(directory);result['passed']=True
        except BaseException as error:result['error']=f'{type(error).__name__}: {error}'
        finally:
            def cleanup(label,action):
                try:action()
                except BaseException as error:result['cleanup_errors'].append(f'{label}: {type(error).__name__}: {error}')
            if worker:
                cleanup('kill owned group',lambda:metal_engine.kill_owned_group(worker));cleanup('reap child',lambda:worker.wait(timeout=5))
                for stream in (worker.stdin,worker.stdout,worker.stderr):cleanup('close pipe',stream.close)
            if selector:cleanup('close selector',selector.close)
            try:result['identities_unchanged']=all(digest(Path(p))==v for p,v in identities.items())
            except BaseException as error:result['identities_unchanged']=False;result['identity_error']=str(error)
            if result['cleanup_errors'] or not result['identities_unchanged']:result['passed']=False
            json.dump(result,report,indent=2,allow_nan=False);report.write('\n')
    if not result['passed']:raise ValueError('Half localization failed; see retained record')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.binary.resolve(),a.capture.resolve(),a.output.resolve())
