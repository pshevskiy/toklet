"""Strict actual-source CPU capture accounting and first-input invariants.

This is not independent coupled-model fidelity. Native intermediate operands
are used for local state/cache checks. Source embedding/sample head rows are
independently reconstructed; all other projections need a later closed oracle.
"""
import argparse
import hashlib
import json
import math
import os
import stat
import struct
from pathlib import Path
import check_qwen35_payload as payload


def need(ok,message):
    if not ok:raise ValueError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def expected():
    result={(None,'embedding',None):2560,(None,'final_norm',None):2560,(None,'logits',None):248320}
    common=dict(input_norm=2560,operator_output=2560,operator_residual=2560,post_norm=2560,
                mlp_gate=9216,mlp_up=9216,mlp_activated=9216,mlp_output=2560,layer_output=2560)
    delta=dict(delta_qkv=8192,delta_z=4096,delta_a=32,delta_b=32,delta_convolved=8192,
               delta_q=2048,delta_k=2048,delta_v=4096,delta_decay=32,delta_beta=32,
               delta_state=524288,delta_history=24576,delta_output=4096,delta_norm=4096,delta_gated=4096)
    attention=dict(attention_q_gate=8192,attention_q=4096,attention_k=1024,attention_v=1024,
                   attention_q_norm=4096,attention_k_norm=1024,attention_q_rot=4096,
                   attention_k_rot=1024,attention_context=4096,attention_gated=4096)
    for layer in range(32):
        for name,count in {**common,**(attention if layer%4==3 else delta)}.items():result[layer,name,None]=count
        if layer%4==3:
            for head in range(4):
                for name in ['attention_k_cache','attention_v_cache']:result[layer,name,head]=256
            for head in range(16):
                for name in ['attention_scores','attention_probabilities']:result[layer,name,head]=1
    return result


def key(row):
    for field in ['token','position','count','offset','bytes']:
        need(type(row[field]) is int and row[field]>=0,'Strict integer array fields required')
    need(row['token']==row['position']==0,'Frozen first-input position differs')
    for field in ['layer','head']:need(row[field] is None or type(row[field]) is int,'Strict owner/head required')
    need(type(row['stage']) is str,'Stage name required')
    return row['layer'],row['stage'],row['head']


def check(folder,model):
    receipt=json.loads((folder/'checks.json').read_text());need(receipt['passed'] is True and type(receipt['exit_code']) is int and receipt['exit_code']==0 and receipt['group_absent'] is True and receipt['changed_inputs']==[],'Completed owned probe required')
    peak=receipt['lifetime_peak_physical_bytes'];need(type(peak) is int and 0<peak<=6*1024**3,'Measured physical limit required')
    need(set(receipt['files_sha256'])=={'console.log','time-and-errors.log','inputs.json','capture/rows.f32le'},'Sealed file inventory differs')
    for name,sha in receipt['files_sha256'].items():
        need((folder/name).stat().st_size<=128*1024**2,'Sealed file bound differs')
        need(digest((folder/name).read_bytes())==sha,'Sealed capture input differs')
    log=(folder/'console.log').read_text();labels=lambda tag:[json.loads(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith(tag+' ')]
    report,=labels('QWEN35_GRAPH_REPORT');need(report==receipt['report'],'Sealed report differs')
    need(report['source_sha256']==payload.FILE_SHA and report['container_bytes']==payload.FILE_BYTES and report['config_sha256']==payload.CONFIG_SHA and report['header_sha256']==payload.HEADER_SHA,'Pinned source differs')
    need(report['tokens']==[1] and report['position_offset']==0 and report['capacity']==512 and report['chunk']==32,'Frozen CPU input differs')
    need(all(type(v) is int for v in report['tokens']) and all(type(report[k]) is int for k in ['position_offset','capacity','chunk','arrays','capture_bytes','greedy_lowest_id']),'Strict integer input/report fields required')
    need(report['first_token_executed'] is True and report['independent_real_source_fidelity_qualified'] is False and report['quality_qualified'] is False and report['managed_adoption_qualified'] is False,'Qualification scope differs')
    memory=report['memory'];need(all(type(v) is int and v>0 for v in memory.values()),'Strict positive memory counts required')
    for name,count in dict(recurrent_bank_bytes=150994944,convolution_bank_bytes=7077888,kv_bytes=33554432,scratch_bytes=504320,logits_bytes=63569920,history_bytes=2048,metadata_allowance_bytes=64*1024**2).items():need(memory[name]==count,'Memory geometry differs: '+name)
    need(2369040384<memory['weights_and_model_bytes']<2369040384+1024**2,'Source/model owner bytes differ')
    need(sum(v for k,v in memory.items() if k!='admitted_bytes')==memory['admitted_bytes']<=6*1024**3-128*1024**2,'Admission sum/diagnostic reserve differs')
    need(report['work']==dict(outcome='Success',tokens_completed=1,tokens_committed=1,projections_completed=249,gathers_completed=1,convolution_steps_completed=24,recurrence_steps_completed=24,attention_heads_completed=128),'Completed work counters differ')
    need(all(type(v) is int for k,v in report['work'].items() if k!='outcome'),'Strict integer work counters required')
    need(type(report['warmed_rust_allocations']) is int and report['warmed_rust_allocations']==0,'No-allocation witness missing')
    for name in ['load_ns','bind_ns','observed_forward_ns','plain_forward_ns']:need(type(report[name]) is int and report[name]>0,'Completed CPU phase time required')
    body_path=folder/'capture/rows.f32le';need(body_path.stat().st_size==report['capture_bytes']<=128*1024**2,'Capture bound differs');body=body_path.read_bytes()
    wanted=expected();arrays=labels('QWEN35_GRAPH_ARRAY');need(len(arrays)==len(wanted)==report['arrays']==1051,'Array coverage differs')
    offset=0;rows={}
    for row in arrays:
        k=key(row);need(k in wanted and k not in rows,'Array owner/stage uniqueness differs')
        need(row['count']==wanted[k] and row['offset']==offset and row['bytes']==wanted[k]*4,'Array extent differs')
        raw=memoryview(body)[offset:offset+row['bytes']];need(len(raw)==row['bytes'] and digest(raw)==row['sha256'],'Array body digest differs')
        need(all(math.isfinite(v) for v, in struct.iter_unpack('<f',raw)),'Nonfinite array');rows[k]=raw;offset+=len(raw)
    need(offset==len(body),'Trailing capture data')
    unpack=lambda k:struct.unpack('<'+'f'*wanted[k],rows[k])
    for layer in range(3,32,4):
        qgate=unpack((layer,'attention_q_gate',None));q=unpack((layer,'attention_q',None));need(q==tuple(v for h in range(16) for v in qgate[512*h:512*h+256]),'Per-head gate split differs')
        # Active zero-angle arithmetic may change a signed zero; inactive tails
        # retain bits in the producer. This local identity is numerical equality.
        need(unpack((layer,'attention_q_norm',None))==unpack((layer,'attention_q_rot',None)) and unpack((layer,'attention_k_norm',None))==unpack((layer,'attention_k_rot',None)),'Zero-position rotary identity differs')
        v=unpack((layer,'attention_v',None));context=unpack((layer,'attention_context',None));need(context==tuple(v[(h//4)*256+d] for h in range(16) for d in range(256)),'First-input grouped attention differs')
        for h in range(16):need(unpack((layer,'attention_probabilities',h))==(1.0,),'First-input softmax differs')
    for layer in range(32):
        if layer%4==3:continue
        history=rows[layer,'delta_history',None];need(not any(history[:2*8192*4]) and history[2*8192*4:]==rows[layer,'delta_qkv',None],'Raw four-tap history differs')
    file=model/'model.safetensors';need((model/'config.json').stat().st_size==3024,'Source config size differs');config=(model/'config.json').read_bytes();need(digest(config)==payload.CONFIG_SHA,'Source config differs')
    with file.open('rb') as f:
        before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode) and before.st_size==payload.FILE_BYTES and payload.hash_file(f)==payload.FILE_SHA,'Complete source identity differs');f.seek(0)
        need(f.read(8)==struct.pack('<Q',119829),'Source prefix differs');header=f.read(119829);need(digest(header)==payload.HEADER_SHA,'Source header differs');header=json.loads(header)
        def affine(name,row):
            columns=header[name]['shape'][1]*8;base=name.removesuffix('.weight');parts=[]
            for part,stride in [(name,columns//2),(base+'.scales',columns//64*2),(base+'.biases',columns//64*2)]:
                f.seek(119837+header[part]['data_offsets'][0]+row*stride);raw=f.read(stride);need(len(raw)==stride,'Truncated sampled affine row');parts.append(raw)
            return payload.affine_row(*parts,columns)
        embedding='language_model.model.embed_tokens.weight';need(affine(embedding,1)==rows[None,'embedding',None],'Raw source embedding row differs')
        final=unpack((None,'final_norm',None));logits=unpack((None,'logits',None));greedy=max(range(len(logits)),key=logits.__getitem__);need(greedy==report['greedy_lowest_id'],'Greedy tie policy differs')
        sample=sorted({0,124160,248319,greedy})
        for row in sample:
            weights=struct.unpack('<2560f',affine(embedding,row));want=math.fsum(a*b for a,b in zip(final,weights,strict=True));need(abs(logits[row]-want)<=1e-5+1e-4*abs(want),'Sampled tied-head projection differs')
        need(payload.hash_file(f)==payload.FILE_SHA,'Source endpoint digest differs');after=os.fstat(f.fileno())
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Source descriptor changed')
    return dict(passed=True,arrays=1051,values=len(body)//4,capture_sha256=digest(body),memory=memory,work=report['work'],lifetime_peak_physical_bytes=peak,source_embedding_row_bit_exact=True,sampled_tied_head_rows=sample,first_input_cache_history_invariants=True,independent_real_source_fidelity_qualified=False,quality_qualified=False,managed_adoption_qualified=False,scope='Sealed actual-source one-input CPU capture/accounting, embedding/sample-head reconstruction and local first-input invariants only. Native intermediate operands are not an independent coupled-model oracle.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('model',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=check(args.folder,args.model)
    with args.output.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps(result))
