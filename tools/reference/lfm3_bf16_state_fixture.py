"""Independent integer/BF16 state oracle, without importing model or tensor code."""
from array import array
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

CONTRACT_SHA='ad7329ea7d02b63334b7229a78c414f16b5095076d96c62a528dfdd5e08e4953'
RUNTIME_SHA='1ca684c819f499a2ede6074c364b64c64c55b3fbd7fc29ccd23ff25a937202ed'


def need(value,reason):
    if not value:raise ValueError(reason)


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_pins(pins,label):
    for path,pin in pins.items():need(digest(Path(path))==pin,label+' changed')


def word(owner,field,head,position,lane):
    index=(owner*8191+field*4099+head*2053+position*37+lane*17)%65280
    return index if index<32640 else index+128


def packed(values):
    result=array('H',values)
    if sys.byteorder!='little':result.byteswap()
    return result.tobytes()


def rows(owner,field,heads,dims,positions):
    return packed(0 if p is None else word(owner,field,h,p,lane)
                  for h in range(heads) for p in positions for lane in range(dims))


def row_digest(owner,field,heads,dims,positions):
    return hashlib.sha256(rows(owner,field,heads,dims,positions)).hexdigest()


def visibility(offset,tokens,window):
    return [(0 if window is None else max(0,q-window+1),q+1)
            for q in range(offset,offset+tokens)]


def ranges_digest(ranges):
    h=hashlib.sha256()
    for start,end in ranges:h.update(struct.pack('<QQ',start,end))
    return h.hexdigest()


def mask_digest(offset,tokens,start,window):
    h=hashlib.sha256()
    for first,last in visibility(offset,tokens,window):
        h.update(bytes(first<=key<last for key in range(start,offset+tokens)))
    return h.hexdigest()


def produce(policy):
    observed=set(policy['observe_positions'])
    history=[];kv=[]
    for name,chunks in policy['schedules'].items():
        offset=0;records=[]
        for step,tokens in enumerate(chunks):
            after=offset+tokens
            if after in observed or step==len(chunks)-1 or len(chunks)<=32:
                previous=[None]*max(0,6-offset)+list(range(max(0,offset-6),offset))
                final=[None]*max(0,6-after)+list(range(max(0,after-6),after))
                records.append(dict(step=step,offset=offset,tokens=tokens,after=after,
                    previous_sha256=row_digest(0,2,1,2048,previous),
                    candidate_sha256=row_digest(0,2,1,2048,final)))
            offset=after
        history.append(dict(name=name,owner=0,channels=2048,chunks=chunks,observations=records))
        for window,owner in [(None,2),(256,4)]:
            capacity=2048 if window is None else window
            physical=[None]*capacity;offset=0;records=[]
            for step,tokens in enumerate(chunks):
                after=offset+tokens
                view_start=0 if window is None else max(0,offset-window+1)
                for position in range(offset,after):
                    physical[position if window is None else position%capacity]=position
                if after in observed or step==len(chunks)-1 or len(chunks)<=32:
                    positions=list(range(view_start,after))
                    ranges=visibility(offset,tokens,window)
                    records.append(dict(step=step,offset=offset,tokens=tokens,after=after,
                        view_start=view_start,view_end=after,
                        view_key_sha256=row_digest(owner,0,4,128,positions),
                        view_value_sha256=row_digest(owner,1,4,128,positions),
                        visible_ranges_sha256=ranges_digest(ranges),
                        mask_sha256=mask_digest(offset,tokens,view_start,window),
                        physical_key_sha256=row_digest(owner,0,4,128,physical),
                        physical_value_sha256=row_digest(owner,1,4,128,physical),
                        positions=physical.copy()))
                offset=after
            kv.append(dict(name=name+('-full' if window is None else '-sliding256'),
                owner=owner,heads=4,dims=128,capacity=capacity,max_positions=2048,
                window=window,chunks=chunks,observations=records))
    finite=[word for word in range(65536) if word&0x7f80!=0x7f80]
    need(len(finite)==65280,'Finite BF16 grid differs')
    return dict(format='toklet-lfm3-bf16-state-fixture-v1',fixture_completed=True,
        contract_sha256=CONTRACT_SHA,runtime_sha256=RUNTIME_SHA,
        producer_sha256=digest(Path(__file__)),
        scope='Independent exact integer/BF16 storage/causal ranges; source runtime probes and native numerical/state checks are separate',
        finite_bf16=dict(words=65280,sha256=hashlib.sha256(packed(finite)).hexdigest()),
        history=history,kv=kv,model_execution_qualified=False,quality_adoption_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['contract','runtime','fresh_fixture']:
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args()
    need(not args.fresh_fixture.exists() and args.fresh_fixture.parent.is_dir(),'Fresh fixture destination required')
    need(digest(args.contract)==CONTRACT_SHA,'Frozen BF16 state contract differs')
    need(digest(args.runtime)==RUNTIME_SHA,'Pinned BF16 state runtime differs')
    policy=json.loads(args.contract.read_text());runtime=json.loads(args.runtime.read_text())
    verify_pins(policy['source_sha256'],'Source metadata')
    verify_pins(runtime['files'],'Reference runtime')
    result=produce(policy)
    verify_pins(runtime['files'],'Reference runtime')
    with args.fresh_fixture.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(fixture_completed=True,history_cases=len(result['history']),
        history_observations=sum(len(c['observations']) for c in result['history']),
        kv_cases=len(result['kv']),kv_observations=sum(len(c['observations']) for c in result['kv']),
        fixture_sha256=digest(args.fresh_fixture))))


if __name__=='__main__':main()
