"""Replay native first-block statistics and independently score eight anchors.

Journal statistics authenticate producer reductions; only the selected complete
FP32 rows support independent distribution replay. Aggregate quality stays open.
"""
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
POSITIONS = [0,31,255,511,512,767,1023,1024]
VOCAB = 262144


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a,b):
    return type(a) in (int,float) and math.isfinite(a) and abs(a-b)<=8*math.ulp(max(1.,abs(b)))


def digest(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def bounded(path, limit):
    with Path(path).open('rb') as file:
        raw=file.read(limit+1)
    require(len(raw)<=limit,'Oversized input')
    return raw


def labels(raw, prefix):
    return [json.loads(line[len(prefix):]) for line in raw.splitlines() if line.startswith(prefix)]


def prior_anchors():
    evidence=json.loads(bounded(ROOT/'benchmarks/results/2026-11-04-gemma-hybrid1025-v1.json',65536))
    require(evidence['state_passed'] and evidence['astra_review'].startswith('approved'),'Prior native state not reviewed')
    name='artifacts/gemma-hybrid1025-2026-11-04/native.stdout'
    raw=bounded(ROOT/name,2*1024**2)
    require(hashlib.sha256(raw).hexdigest()==evidence['receipts_sha256'][name],'Prior capture identity differs')
    result={}
    for item in labels(raw,b'GEMMA_HYBRID_ARRAY '):
        if item['name']=='wrap.logits':
            position=item['before']
            require(position in POSITIONS,'Unexpected prior position')
            if position in result:
                require(result[position]==item['sha256'],'Prior schedules differ')
            result[position]=item['sha256']
    require(set(result)==set(POSITIONS),'Incomplete prior anchors')
    return result


def replay(transcript, body, ids, previous):
    require(len(ids)==1025 and b'test result: ok.' in transcript,'Incomplete native run')
    records=labels(transcript,b'GEMMA_HYBRID_REAL_LOSS_TARGET ')
    summary=labels(transcript,b'GEMMA_HYBRID_REAL_LOSS_SUMMARY ')
    unscored=labels(transcript,b'GEMMA_HYBRID_REAL_LOSS_UNSCORED ')
    arrays=labels(transcript,b'GEMMA_HYBRID_ARRAY ')
    footer=labels(transcript,b'GEMMA_HYBRID_CAPTURE ')
    require(len(records)==1024 and len(summary)==len(unscored)==len(footer)==1,'Target inventory differs')
    values=[]
    for position,item in enumerate(records):
        fields=['block','local_row','global_target_index','target','target_fp32_bits','vocab','greedy_lowest_id']
        require(all(type(item.get(k)) is int for k in fields),'Integer target fields required')
        require((item['block'],item['local_row'],item['global_target_index'],item['target'],item['vocab'])==
                (0,position,position+1,ids[position+1],VOCAB),'Target order/alignment differs')
        require(0<=item['target_fp32_bits']<2**32 and 0<=item['greedy_lowest_id']<VOCAB and digest(item['row_sha256']),'Invalid row metadata')
        maximum,mass,nll=(item[k] for k in ['maximum','exp_mass','nll'])
        target=struct.unpack('<f',struct.pack('<I',item['target_fp32_bits']))[0]
        require(all(type(v) in (int,float) and math.isfinite(v) for v in [maximum,mass,nll,target]),'Nonfinite statistics')
        require(maximum>=target and 1<=mass<=VOCAB and nll>=0,'Invalid distribution statistics')
        require(struct.unpack('<f',struct.pack('<f',maximum))[0]==maximum,'Maximum is not FP32')
        require(close(nll,(maximum-target)+math.log(mass)),'Stable NLL formula differs')
        require(target!=maximum or item['greedy_lowest_id']<=item['target'],'Greedy tie contradicts target')
        values.append(nll)
    require(unscored[0]['position']==1024 and unscored[0]['row_sha256']==previous[1024],'Unscored witness differs')
    require(len(arrays)==8 and len(body)==8*VOCAB*4,'Anchor capture extent differs')
    offset=0;anchors=[]
    for index,(position,item) in enumerate(zip(POSITIONS,arrays,strict=True)):
        raw=body[offset:offset+VOCAB*4]
        sha=hashlib.sha256(raw).hexdigest()
        require((item['record'],item['name'],item['offset'],item['bytes'],item['before'],item['count'],item['values'])==
                (index,'loss.anchor',offset,VOCAB*4,position,1,VOCAB),'Anchor layout differs')
        require(item['sha256']==sha==previous[position],'Prior native row hash differs')
        row=array('f');row.frombytes(raw)
        if sys.byteorder!='little':row.byteswap()
        require(all(math.isfinite(v) for v in row),'Nonfinite anchor')
        if position<1024:
            record=records[position]
            maximum=max(row);mass=math.fsum(math.exp(v-maximum) for v in row)
            target=row[ids[position+1]];nll=(maximum-target)+math.log(mass)
            require(record['row_sha256']==sha and record['target_fp32_bits']==struct.unpack('<I',struct.pack('<f',target))[0], 'Anchor target/hash differs')
            require(record['greedy_lowest_id']==row.index(maximum) and close(record['maximum'],maximum)
                    and close(record['exp_mass'],mass) and close(record['nll'],nll),'Independent anchor statistics differ')
            anchors.append(position)
        offset+=len(raw)
    require(footer[0]['bytes']==len(body) and footer[0]['records']==8 and footer[0]['sha256']==hashlib.sha256(body).hexdigest(),'Capture footer differs')
    mean=math.fsum(values)/1024
    require(summary[0]['targets']==1024 and summary[0]['capacity']==1025 and summary[0]['chunk_capacity']==32
            and summary[0]['session_allocated_bytes']==271117548 and summary[0]['quality_qualified'] is False
            and close(summary[0]['mean_nll'],mean),'Native summary differs')
    return dict(passed=True,targets=1024,mean_nll=mean,prior_row_hashes=8,independent_scored_anchors=anchors,
                unscored_position=1024,quality_qualified=False,nlls=values)


if __name__=='__main__':
    transcript,body=map(Path,sys.argv[1:3])
    raw=bounded(ROOT/'benchmarks/gemma-quality-v1/held_out_loss.u32le',40004)
    require(hashlib.sha256(raw).hexdigest()=='877cbe18574b47e4b6a8d1b58fb79621c37e0531bd1f9504e50ce63d186f8c46','Frozen loss IDs differ')
    ids=list(struct.unpack('<10001I',raw)[:1025])
    print(json.dumps(replay(bounded(transcript,2*1024**2),bounded(body,8*VOCAB*4),ids,prior_anchors()),indent=2))
