"""Replay P2 loss against the original pinned Mac target stream."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

run = Path(sys.argv[1])
archive = Path('artifacts/m8-mlx8-full-loss-driver-v1/mac-quality-full.stdout')
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='75968fa2c69d54aea070a504f87c3ae68a241e351aa2d62ee14ad84f450f4499'
prefix='TOKLET_QUALITY_SAMPLE_JSON='
old = [json.loads(line[len(prefix):])['target'] for line in archive.read_text().splitlines()
       if line.startswith(prefix)]
assert len(old)==10000
report={}
digests={}
for mode, chunk in [('legacy',1),('paired',128)]:
    records=[json.loads(line) for line in (run/f'{mode}.jsonl').read_text().splitlines()]
    ready, = [r for r in records if r['event']=='ready']
    assert (ready['mode'],ready['chunk'],ready['capacity'],ready['block_targets'])==(mode,chunk,1025,1024)
    assert ready['combined_admitted_bytes']<=1536*1024**2
    targets=[r for r in records if r['event']=='target']
    assert len(targets)==10000
    digests[mode]=[row['row_sha256'] for row in targets]
    for i,(a,b) in enumerate(zip(targets,old,strict=True)):
        assert (a['index'],a['block'],a['local_position'],a['target'])==(i,i//1024,i%1024,b['score']['target'])
        assert math.isfinite(a['nll']) and a['nll']>=0
        assert len(a['row_sha256'])==64
        assert a['nll']==b['score']['negativeLogLikelihood']
        assert a['greedy']==b['score']['greedy']
        bits,=struct.unpack('<I',struct.pack('<f',b['score']['targetLogit']))
        assert a['target_bits']==bits
    mean=math.fsum(row['nll'] for row in targets)/10000
    done=records[-1]
    assert done['event']=='done' and done['passed']
    assert done['targets']==10000 and abs(mean-done['mean_nll'])<=1e-12
    assert mean-2.883725149929484<=0.05
    if mode=='paired':
        dispatch,=[row for row in records if row['event']=='dispatch']
        assert dispatch==dict(event='dispatch',rows=128,tile4_stages=168)
        assert done['paired_complete_rows']==10000
    report[mode]=dict(mean_nll=mean,delta=mean-2.883725149929484,
                      largest_delta_from_archived_target=max(abs(a['nll']-b['score']['negativeLogLikelihood'])
                         for a,b in zip(targets,old,strict=True)),complete_paired_rows=done['paired_complete_rows'])
report['cross_schedule_exact_rows']=sum(a==b for a,b in zip(digests['legacy'],digests['paired'],strict=True))
assert report['cross_schedule_exact_rows']==10000
print(json.dumps(report,indent=2,allow_nan=False))
