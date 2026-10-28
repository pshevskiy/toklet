"""Prospective fixed same-artifact1024-target loss inputs; no model execution.

Explicit preparation only. Ordinary verifier uses stdlib and reads frozen IDs;
prepare/replay additionally need the pinned PyArrow/tokenizers runtime.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[2]
FORMAT='toklet-llama3-loss1024-inputs-v1'
REPOSITORY='mlx-community/Llama-3.2-1B-Instruct-8bit'
REVISION='d48cdf0a4ea22d893b7c63a99d6a693e24822795'
DATASET={'path':'artifacts/datasets/wikitext-2-raw-test.parquet','bytes':732610,
         'sha256':'5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91',
         'revision':'b08601e04326c79dfdd32d625aee71d232d685c3','split':'test','configuration':'wikitext-2-raw-v1'}
TOKENIZER={'path':'models/MLX-Llama-3.2-1B-Instruct-8bit-d48cdf0/tokenizer.json','bytes':9085657,
           'sha256':'79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4'}
CORPUS={'utf8_bytes':1290546,'sha256':'430983e5a711fbc4c6576edffecacf47ff23d58f62f3923d2386d798e0a794c9',
        'recipe':'text column, original row order, newline join rows if truthy; no stripping'}
PARENTS={
 'benchmarks/sources.json':{'bytes':5858,'sha256':'dc2f82fb50b3f1f763c4757a992429b9122f3628c049709bdd7cb2252c69ea28'},
 'benchmarks/contract-v1/manifest.json':{'bytes':6490,'sha256':'3f09dcf70f6d9620ce5172dcbaa6428f3487a3edf412f2bfce57937faab72c66'},
 'benchmarks/qwen3-context4096-v1/manifest.json':{'bytes':2143,'sha256':'710054f18fe9a983399c230d7c31c7e99ddcff24f0748498dc3ce4a8b917c7c9'},
}
BLOCKS=[{'block':i,'input_start':start,'input_end':start+257,'score_rows':[0,256],
         'global_target_indices':[start+1,start+257],'targets':256,'reset_position':0}
        for i,start in enumerate([0,256,512,768])]
POLICY={'targets':1024,'capacity':512,'block_tokens':257,'reset_each_block':True,
        'add_special_tokens':False,'chat_template':False,'stop_on_eos':False,
        'mean_nll_increase_limit':0.05,'reference':'same packed artifact canonical FP64 arithmetic/Half storage',
        'cross_profile_greedy_gate':False,'unquantized_quality_qualified':False,'full_task_quality_qualified':False}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate field: '+key)
        result[key]=value
    return result


def read_pinned(path,pin):
    with Path(path).open('rb') as stream:raw=stream.read(pin['bytes']+1)
    if len(raw)!=pin['bytes'] or digest(raw)!=pin['sha256']:raise ValueError('Consumed identity differs: '+str(path))
    return raw


def select(ids):
    if len(ids)<1025:raise ValueError('Insufficient corpus tokens')
    selected=ids[:1025]
    if any(type(i) is not int or not 0<=i<128256 for i in selected):raise ValueError('Token outside vocabulary')
    return struct.pack('<1025I',*selected)


def derive(replay=False):
    for name,pin in PARENTS.items():read_pinned(ROOT/name,pin)
    source=json.loads(read_pinned(ROOT/'benchmarks/sources.json',PARENTS['benchmarks/sources.json']))
    assets=[a for a in source['assets'] if a['id']=='wikitext.test']
    if len(assets)!=1 or any(assets[0][k]!=DATASET[k] for k in ['bytes','sha256','revision','path']):
        raise ValueError('Original held-out source differs')
    for name,version in [('pyarrow','20.0.0'),('tokenizers','0.22.2')]:
        if importlib.metadata.version(name)!=version:raise ValueError('Preparation runtime differs')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    raw=read_pinned(ROOT/DATASET['path'],DATASET)
    tokenraw=read_pinned(ROOT/TOKENIZER['path'],TOKENIZER)
    parquet=pq.ParquetFile(pa.BufferReader(raw))
    if replay:
        rows=[]
        for batch in parquet.iter_batches(batch_size=257,columns=['text'],use_threads=False):rows.extend(batch.column(0).to_pylist())
        tokenizer=Tokenizer.from_str(tokenraw.decode('utf-8'))
    else:
        rows=parquet.read(columns=['text'],use_threads=False).column(0).to_pylist()
        tokenizer=Tokenizer.from_buffer(tokenraw)
    if any(type(row) is not str for row in rows):raise ValueError('String corpus rows required')
    corpus='\n'.join(row for row in rows if row);encoded=corpus.encode('utf-8')
    if len(encoded)!=CORPUS['utf8_bytes'] or digest(encoded)!=CORPUS['sha256']:raise ValueError('Original raw corpus differs')
    if tokenizer.padding is not None or tokenizer.truncation is not None:raise ValueError('Padding/truncation forbidden')
    ids=tokenizer.encode(corpus,add_special_tokens=False).ids
    selected=select(ids)
    for name,pin in PARENTS.items():read_pinned(ROOT/name,pin)
    read_pinned(ROOT/DATASET['path'],DATASET);read_pinned(ROOT/TOKENIZER['path'],TOKENIZER)
    return selected,len(ids)


def validate(manifest):
    expected={'format':FORMAT,'repository':REPOSITORY,'revision':REVISION,'dataset':DATASET,
              'tokenizer':TOKENIZER,'corpus':CORPUS,'parents':PARENTS,'blocks':BLOCKS,'policy':POLICY}
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen input contract differs')
    if (type(manifest['complete_corpus_tokens']) is not int or manifest['complete_corpus_tokens']<1025
            or manifest['model_executed'] is not False):raise ValueError('Input provenance differs')
    pin=manifest['tokens']
    if set(pin)!={'file','dtype','count','bytes','sha256'} or pin['file']!='tokens.u32le' or pin['dtype']!='u32le' or pin['count']!=1025 or pin['bytes']!=4100:
        raise ValueError('Frozen token layout differs')
    if not isinstance(pin['sha256'],str) or len(pin['sha256'])!=64 or any(c not in '0123456789abcdef' for c in pin['sha256']):
        raise ValueError('Frozen token digest differs')


def load(folder,manifest_sha):
    folder=Path(folder)
    with (folder/'manifest.json').open('rb') as stream:raw=stream.read(65537)
    if len(raw)>65536 or digest(raw)!=manifest_sha:raise ValueError('Input manifest identity differs')
    manifest=json.loads(raw,object_pairs_hook=unique);validate(manifest)
    raw=read_pinned(folder/'tokens.u32le',manifest['tokens'])
    ids=list(struct.unpack('<1025I',raw))
    if select(ids)!=raw:raise ValueError('Input token reconstruction differs')
    return manifest,ids


def run(folder,replay=False,manifest_sha=None):
    folder=Path(folder)
    if not replay and os.path.lexists(folder):raise FileExistsError('Fresh input folder required')
    if replay and manifest_sha is None:raise ValueError('Replay requires external manifest identity')
    raw,count=derive(replay)
    if replay:
        manifest,ids=load(folder,manifest_sha)
        if select(ids)!=raw or manifest['complete_corpus_tokens']!=count:raise ValueError('Independent input replay differs')
        return dict(replayed=True,model_executed=False,manifest_sha256=manifest_sha,tokens_sha256=digest(raw))
    manifest=dict(format=FORMAT,repository=REPOSITORY,revision=REVISION,dataset=DATASET,tokenizer=TOKENIZER,
                  corpus=CORPUS,parents=PARENTS,blocks=BLOCKS,policy=POLICY,complete_corpus_tokens=count,
                  tokens=dict(file='tokens.u32le',dtype='u32le',count=1025,bytes=4100,sha256=digest(raw)),model_executed=False)
    validate(manifest);folder.mkdir()
    (folder/'tokens.u32le').write_bytes(raw)
    pending=folder/'.manifest.pending';pending.write_text(json.dumps(manifest,indent=2)+'\n');pending.rename(folder/'manifest.json')
    load(folder,digest((folder/'manifest.json').read_bytes()))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',type=Path);group.add_argument('--replay',type=Path);group.add_argument('--verify',type=Path)
    parser.add_argument('--manifest-sha256');args=parser.parse_args()
    if (args.replay or args.verify) and not args.manifest_sha256:parser.error('External manifest SHA required')
    if args.verify:load(args.verify,args.manifest_sha256);print(json.dumps(dict(verified=True,model_executed=False)))
    else:print(json.dumps(run(args.prepare or args.replay,bool(args.replay),args.manifest_sha256),allow_nan=False))
