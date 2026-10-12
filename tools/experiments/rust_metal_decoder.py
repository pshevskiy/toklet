"""Qualify a resident two-layer tiny GPU decoder and four frozen greedy steps."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import rust_metal_layer as layer
rms=layer.rms


def unpack(bits,kind):
    return struct.unpack(f'<{len(bits)}{"e" if kind=="H" else "f"}',struct.pack(f'<{len(bits)}{kind}',*bits))


def cases():
    h=rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA);s=rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
    layers,la,overflow=layer.cases();by_id={c['id']:c for c in layers}
    def bits(key,dtype):
        data,item=rms.payload(rms.FP16,h,key,dtype)
        return list(struct.unpack(f'<{math.prod(item["shape"])}{"H" if dtype=="f16le" else "I"}',data)),item
    table,ti=bits('weights.embed_tokens','f16le');norm,ni=bits('weights.norm','f16le')
    if ti['shape']!=[64,32] or ni['shape']!=[32] or h['model']!=s['model'] or not h['model']['tie_word_embeddings']:raise ValueError('Decoder tied topology differs')
    weights=[]
    for index in range(2):
        item=copy.deepcopy(by_id[f'cases.prompt.layers.{index}.layer'])
        for key in ('format','id','input_bits','rows','position'):del item[key]
        weights.append(item)
    def anchor(key,dtype):
        b,item=bits(key,dtype);source,ref=rms.payload(rms.FP32,s,key,'f32le')
        if item['shape']!=ref['shape']:raise ValueError('Decoder anchor shape differs')
        return {'bits':b,'source':list(struct.unpack(f'<{math.prod(ref["shape"])}f',source)),'gate':item['gate_against_source_fp32'],'oracle_sha256':item['sha256'],'source_sha256':ref['sha256']}
    inputs=[];anchors={}
    for name,c in h['cases'].items():
        prefix=f'cases.{name}';tokens=c['tokens']
        item={'format':'toklet-metal-decoder-input-v1','id':prefix,'rows':len(tokens),'position':c['position_offset'],'tokens':tokens.copy(),'embedding_weight_bits':table,'final_norm_weight_bits':norm,'layers':copy.deepcopy(weights)}
        inputs.append(item);anchors[prefix]={'layers':[copy.deepcopy(la[f'{prefix}.layers.{i}.layer']) for i in range(2)],'embedding':anchor(prefix+'.embedding','f16le'),'norm':anchor(prefix+'.final_norm','f16le'),'logits':anchor(prefix+'.logits','f32le')}
    history=s['greedy']['prompt_tokens'].copy()
    if h['greedy']['source_generated_tokens']!=s['greedy']['generated_tokens'] or h['greedy']['stop_policy']!='fixed_4_steps_no_eos':raise ValueError('Decoder greedy history differs')
    for step,token in enumerate(s['greedy']['generated_tokens']):
        prefix=f'greedy.step_{step}'
        inputs.append({'format':'toklet-metal-decoder-input-v1','id':prefix,'rows':len(history),'position':0,'tokens':history.copy(),'embedding_weight_bits':table,'final_norm_weight_bits':norm,'layers':copy.deepcopy(weights)})
        anchors[prefix]={'logits':anchor(prefix+'.logits','f32le'),'expected_token':token,'history':'source FP32 frozen teacher-forced history'}
        history.append(token)
    overflow=copy.deepcopy(inputs[-1]);overflow['id']='manual.overflow';overflow['rows']=1;overflow['tokens']=[0];overflow['embedding_weight_bits']=[0]*2048;overflow['embedding_weight_bits'][0]=0x3c00;overflow['layers'][0]['input_norm_weight_bits']=[0x7bff]+[0]*31
    return inputs,anchors,overflow


def qualify(bits,expected,kind):
    if any(type(b)is not int or not 0<=b<(65536 if kind=='H' else 2**32) for b in bits) or len(bits)!=len(expected['bits']):raise ValueError('Decoder output bit shape/type differs')
    values=unpack(bits,kind)
    if not all(map(math.isfinite,values)):raise ValueError('Decoder nonfinite output')
    gate=expected['gate'];fraction=max(abs(a-b)/(gate['atol']+gate['rtol']*abs(b)) for a,b in zip(values,expected['source']))
    if fraction>1:raise ValueError('Decoder frozen source gate failed')
    return {'values':len(bits),'equal_frozen_bits':sum(a==b for a,b in zip(bits,expected['bits'])),'source_fp32_max_gate_fraction':fraction,'output_sha256':hashlib.sha256(struct.pack(f'<{len(bits)}{kind}',*bits)).hexdigest(),'oracle_sha256':expected['oracle_sha256'],'source_sha256':expected['source_sha256'],'gate':gate}


def validate(event,item,anchor):
    if event.get('event')!='idle_result' or event.get('passed')is not True:raise ValueError('Decoder native failure')
    r=event['result'];n=item['rows'];width=r['threadgroup_width'];m=r['metadata']
    if (r['operation']!='resident_tiny_decoder_half_boundaries' or r['id']!=item['id'] or r['rows_count']!=n or r['position']!=item['position'] or r['allocations_reused']is not True or r['tied_embedding_allocation']is not True):raise ValueError('Decoder topology differs')
    if m['mps_device_supported']is not True or m['unified_memory']is not True or m['math_mode']!='safe' or m['math_functions']!='precise' or m['language_version']!='2.3' or m['minimum_macos_major']!=15:raise ValueError('Decoder math policy differs')
    caps=m['embedding_pipeline_caps']
    if not 0<width<=min(caps['execution_width'],caps['max_threads'],m['execution_width'],m['pipeline_max_threads'],m['device_max_threads'][0],*[v for c in m['graph_pipeline_caps'] for v in (c['execution_width'],c['max_threads'])]):raise ValueError('Decoder launch bounds differ')
    wl,hl,fl,outer=(r[k] for k in ('layer_weight_layouts','layer_half_layouts','layer_float_layouts','outer_layouts'))
    if [len(wl),len(hl),len(fl),len(outer)]!=[13,18,9,6] or r['token_buffer_bytes']!=(n+7)*4:raise ValueError('Decoder allocation inventory differs')
    for l,shape,scalar in zip(outer,[(64,32),(1,32),(1,64),(n,32),(n,32),(n,64)],[2,2,2,2,2,4]):
        if (l['rows'],l['columns'])!=shape or l['stride']<shape[1] or l['bytes']!=scalar*l['elements'] or l['elements']<(shape[0]*l['stride']+7):raise ValueError('Decoder outer layout differs')
    if outer[0]['stride']!=32 or outer[3]!=hl[0] or outer[4]!=hl[0] or outer[5]['elements']!=n*outer[5]['stride']+7:raise ValueError('Decoder outer storage differs')
    shared=2*sum(l['bytes'] for group in (wl,hl,fl) for l in group)+sum(l['bytes'] for l in outer)+r['token_buffer_bytes'];total=3*shared+rms.probe.ALLOWANCE
    if r['shared_buffer_bytes']!=shared or r['host_snapshot_readback_bytes']!=2*shared or r['admitted_bytes']!=total or m['maximum_admitted_bytes']!=total or total>rms.probe.LIMIT:raise ValueError('Decoder combined admission differs')
    embedded=r['first_embedding_bits'];normalized=r['first_normalized_bits'];logits=r['first_logits_bits'];frames=r['first_layers'];reps=r['rows']
    expected_embedding=[v for token in item['tokens'] for v in item['embedding_weight_bits'][token*32:(token+1)*32]]
    if embedded!=expected_embedding or len(normalized)!=n*32 or len(logits)!=n*64 or len(frames)!=2 or len(reps)!=13:raise ValueError('Decoder embedding/output shapes differ')
    def finite_array(bits,count,kind):
        if len(bits)!=count or any(type(b)is not int or not 0<=b<(65536 if kind=='H' else 2**32) for b in bits) or not all(map(math.isfinite,unpack(bits,kind))):raise ValueError('Decoder array shape/type/finite differs')
    for bits,count,kind in [(embedded,n*32,'H'),(normalized,n*32,'H'),(logits,n*64,'I')]:finite_array(bits,count,kind)
    for frame in frames:
        if len(frame['half'])!=18 or len(frame['single'])!=9:raise ValueError('Decoder layer frame inventory differs')
        for bits,count in zip(frame['half'],[n*32,n*16,n*16,n*32,n*16,512,512,n*32,n*32]+[n*32]*3+[n*64]*4+[n*32]*2):finite_array(bits,count,'H')
        for bits,count in zip(frame['single'],[n*32,n*16,n*16,4*n*n,4*n*n,n*32,n*64,n*64,n*32]):finite_array(bits,count,'I')
    qualifications=[];incoming=embedded
    # Reuse the standalone layer validator as a diagnostic view, with its
    # logical incoming allocation. No view submits work or adds a wait; native
    # combined admission and single command evidence are checked above/below.
    for index,frame in enumerate(frames):
        if len(frame['half'])!=18 or len(frame['single'])!=9:raise ValueError('Decoder layer frame inventory differs')
        if 'layers' in anchor:
            il={'rows':n,'columns':32,'stride':32,'elements':n*32+7,'bytes':(n*32+7)*2};view_shared=il['bytes']+sum(l['bytes'] for group in (wl,hl,fl) for l in group)
            view={'operation':'resident_tiny_layer_half_boundaries','id':f'{item["id"]}.layers.{index}.layer','rows_count':n,'position':item['position'],'threadgroup_width':width,'metadata':{**m,'maximum_admitted_bytes':3*view_shared+rms.probe.ALLOWANCE},'input_layouts':[il]+wl,'half_layouts':hl,'float_layouts':fl,'shared_buffer_bytes':view_shared,'host_snapshot_readback_bytes':2*view_shared,'admitted_bytes':3*view_shared+rms.probe.ALLOWANCE,'allocations_reused':True,'first_half_intermediates':copy.deepcopy(frame['half']),'first_fp32_intermediates':copy.deepcopy(frame['single']),'rows':[{**row,'output_bits':frame['half'][17]} for row in reps]}
            layer.validate({'event':'idle_result','passed':True,'result':view},{**item,'id':view['id'],'input_bits':incoming},anchor['layers'][index]);qualifications.append(view['qualification'])
        else:
            # Greedy fixture freezes only last-row logits. These intermediate
            # values have repeat/finite/guard checks but no invented gates.
            if any(not all(map(math.isfinite,unpack(a,'H'))) for a in frame['half']) or any(not all(map(math.isfinite,unpack(a,'I'))) for a in frame['single']):raise ValueError('Decoder greedy nonfinite intermediate')
        incoming=frame['half'][17]
    q={'layers':qualifications}
    if 'embedding'in anchor:q['embedding']=qualify(embedded,anchor['embedding'],'H');q['norm']=qualify(normalized,anchor['norm'],'H')
    last=logits[-64:];q['logits']=qualify(logits if 'layers'in anchor else last,anchor['logits'],'I')
    values=unpack(last,'I');token=max(range(64),key=lambda i:values[i])
    if 'expected_token'in anchor and token!=anchor['expected_token']:raise ValueError('Decoder frozen greedy token differs')
    for i,row in enumerate(reps):
        if row['phase']!=('warmup'if i<3 else 'sample') or row['repetition']!=(i+1 if i<3 else i-2) or row['command_completed']is not True or row['commit_wait_pairs']!=1 or row['inputs_guards_and_intermediates_unchanged']is not True or row['last_logits_bits']!=last or type(row['token_id'])is not int or row['token_id']!=token or not math.isfinite(row['encode_commit_wait_seconds']) or row['encode_commit_wait_seconds']<=0:raise ValueError('Decoder repetition/greedy policy differs')
        row['last_logits_sha256']=hashlib.sha256(struct.pack('<64I',*last)).hexdigest();del row['last_logits_bits']
    r['qualification']=q;r['history_policy']=anchor.get('history','frozen prefix')
    for key in ('first_layers','first_embedding_bits','first_normalized_bits','first_logits_bits'):del r[key]


def run(output,build,directory):
    rms.probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+'.stderr.txt'),output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():raise ValueError('Refusing to overwrite decoder inputs')
    for path in (build,*evidence):
        if directory.resolve()==path.resolve() or directory.resolve()in path.resolve().parents or path.resolve()in directory.resolve().parents:raise ValueError('Decoder paths must be disjoint')
    if any(root.resolve()==path.resolve() or root.resolve()in path.resolve().parents for root in (rms.FP16,rms.FP32) for path in (build,directory,*evidence)):raise ValueError('Decoder outputs must be outside fixtures')
    inputs,anchors,overflow=cases();li,la,lo=layer.cases();directory.mkdir(parents=True);jobs=[];identities={}
    payloads=[(f'decoder_{i+1}',item,'--decoder-input',lambda e,item=item:validate(e,item,anchors[item['id']])) for i,item in enumerate(inputs)]
    payloads.append(('decoder_overflow',overflow,'--decoder-input',layer.overflow_rejected))
    payloads.extend((f'layer_{i+1}',item,'--layer-input',lambda e,item=item:layer.validate(e,item,la[item['id']])) for i,item in enumerate(li));payloads.append(('layer_overflow',lo,'--layer-input',layer.overflow_rejected))
    invalid=copy.deepcopy(inputs[0]);invalid['tokens'][0]=64
    def rejected(e):
        if e.get('passed')is not False or e.get('error',{}).get('stage')!='fixture' or e['error']['message']!='decoder token outside vocabulary':raise ValueError('Decoder invalid-token failure differs')
        return False
    payloads.append(('decoder_invalid_token',invalid,'--decoder-input',rejected))
    for label,item,option,validator in payloads:
        path=directory/f'{label}.json'
        with path.open('x')as stream:stream.write(json.dumps(item,separators=(',',':'))+'\n')
        identities[str(path.resolve())]=rms.probe.common.digest(path);jobs.append((label,[option,str(path.resolve())],validator))
    modules=(layer,layer.branch,layer.cached,layer.attention,layer.attention.cache,layer.attention.rotary,layer.attention.rotary.composed,rms)
    drivers={str(Path(m.__file__).resolve()):rms.probe.common.digest(Path(m.__file__)) for m in modules};drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(e):
        result=previous(e)
        if cases()!=(inputs,anchors,overflow):raise ValueError('Decoder frozen inputs changed')
        for path,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(path))!=sha:raise ValueError('Decoder input/driver identity changed')
        return result
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context={'input_sha256':identities,'qualification_driver_sha256':drivers,'fp16_manifest_sha256':rms.FP16_MANIFEST_SHA,'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,'scope':'Five complete tiny GPU decoder prefixes, four frozen teacher-forced greedy steps, ten standalone layer regressions, finite-input overflow and invalid-token rejection. Embedding -> two resident layers -> final RMS -> tied FP32 logits in one command/wait. No saved layer inputs, root backend/session/Qwen/performance/M4 closure.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True);a=p.parse_args();run(a.output,a.build_dir,a.input_dir)
if __name__=='__main__':main()
