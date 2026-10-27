"""Freeze/replay the prospective4096 inputs; no model or quality execution."""
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/reference'))
from qwen3_context_precision import Owner

NUMERICAL = '21d54e567025f70fdda01d7f1bede1a71f730c6b7c10aec62f5aaffec52a7f08'
QUALITY = '908e037de5b399d9e9a8dd09aa9fcc568f67abb714b06508672deac6bb8ac529'
SOURCES = 'dc2f82fb50b3f1f763c4757a992429b9122f3628c049709bdd7cb2252c69ea28'
REQUIRED = dict(format='toklet-qwen3-context4096-inputs-v1', capacity=4096,
    numerical_parent_manifest_sha256=NUMERICAL, quality_parent_manifest_sha256=QUALITY,
    quality_global_token_start=10001, score_rows=[3071,4095],
    score_target_indices=[3072,4096], targets=1024, quality_mean_nll_increase_limit=0.05)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def choose(original, complete_ids, previous):
    if original != [9707,11,1879,0]*128 or len(previous) != 10001:
        raise ValueError('Frozen numerical/old quality inputs differ')
    if len(complete_ids) < 14097 or complete_ids[:10001] != previous:
        raise ValueError('Complete unclipped corpus must preserve the old prefix')
    if any(type(v) is not int or not 0 <= v < 151936 for v in complete_ids):
        raise ValueError('Integer in-vocabulary raw token IDs required')
    a = original*8
    b = a[:-1]+[(a[-1]+1)%151936]
    q = complete_ids[10001:14097]
    return {'A.u32le':a, 'B.u32le':b, 'Q.u32le':q}


def validate(manifest):
    for key, value in REQUIRED.items():
        if manifest.get(key) != value or type(manifest.get(key)) is not type(value):
            raise ValueError('Frozen input/quality policy differs: '+key)
    if set(manifest['files']) != {'A.u32le','B.u32le','Q.u32le'}:
        raise ValueError('All three exact input arrays required')
    for item in manifest['files'].values():
        if item['bytes'] != 16384 or len(item['sha256']) != 64:
            raise ValueError('Complete4096 input identity required')


def derive(replay, report):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from importlib.metadata import version
    if version('pyarrow') != '20.0.0' or version('tokenizers') != '0.22.2':
        raise ValueError('Pinned input-preparation runtime required')
    with ExitStack() as stack:
        def read(path, expected, maximum):
            path = Path(path)
            if isinstance(expected,str):
                expected = dict(bytes=path.stat().st_size,sha256=expected)
            if expected['bytes'] > maximum:
                raise ValueError('Parent exceeds input bound')
            owner = Owner(path,expected);report['parents'].append(owner.audit)
            stack.enter_context(owner);owner.file.seek(0)
            raw=owner.file.read(expected['bytes'])
            if len(raw)!=expected['bytes'] or sha(raw)!=expected['sha256']:
                raise ValueError('Actually consumed parent differs')
            return raw
        sources=json.loads(read(ROOT/'benchmarks/sources.json',SOURCES,1024**2))
        numerical=json.loads(read(ROOT/'tests/fixtures/qwen3-context512-v1/manifest.json',NUMERICAL,1024**2))
        quality=json.loads(read(ROOT/'benchmarks/qwen3-quality-v1/manifest.json',QUALITY,1024**2))
        assets=[v for v in sources['assets'] if v['id']=='wikitext.test']
        if len(assets)!=1:raise ValueError('Unique pinned WikiText test source required')
        asset=assets[0]
        raw=read(ROOT/asset['path'],{k:asset[k] for k in ('bytes','sha256')},4*1024**2)
        table=pq.ParquetFile(pa.BufferReader(raw))
        if replay:
            rows=[]
            for batch in table.iter_batches(batch_size=257,columns=['text'],use_threads=False):
                rows.extend(batch.column(0).to_pylist())
        else:
            rows=table.read(columns=['text'],use_threads=False).column(0).to_pylist()
        if not all(type(row) is str for row in rows):raise ValueError('String corpus rows required')
        corpus='\n'.join(row for row in rows if row)
        if len(corpus.encode())>16*1024**2:raise ValueError('Corpus exceeds frozen bound')
        tokenpin=quality['tokenizer_assets']['tokenizer.json']
        tokenfile=ROOT/'models/MLX-Qwen3-0.6B-8bit-11de968/tokenizer.json'
        tokenbytes=read(tokenfile,tokenpin,16*1024**2)
        tokenizer=Tokenizer.from_file(str(tokenfile)) if replay else Tokenizer.from_buffer(tokenbytes)
        if tokenizer.truncation is not None or tokenizer.padding is not None:
            raise ValueError('No clipping or padding permitted')
        ids=tokenizer.encode(corpus,add_special_tokens=False).ids
        previous=read(ROOT/'benchmarks/qwen3-quality-v1/held_out_loss.u32le',quality['files']['held_out_loss.u32le'],40004)
        selected=choose(numerical['plan']['cases']['original'],ids,list(struct.unpack('<10001I',previous)))
        report.update(dataset=asset,tokenizer=tokenpin,corpus_utf8_bytes=len(corpus.encode()),
                      corpus_sha256=sha(corpus.encode()),complete_corpus_tokens=len(ids),
                      old_prefix_preserved=True,new_span_global_half_open=[10001,14097],
                      add_special_tokens=False,join_policy='newline_join_nonempty_original_rows')
    return {name:struct.pack('<4096I',*values) for name,values in selected.items()}


def run(output, replay=False):
    output=Path(output);report=dict(passed=False,model_executed=False,parents=[],error=None,replay=replay)
    try:
        arrays=derive(replay,report)
        identities={name:dict(bytes=len(raw),sha256=sha(raw)) for name,raw in arrays.items()}
        if replay:
            manifest=json.loads((output/'manifest.json').read_bytes());validate(manifest)
            if manifest['files']!=identities:raise ValueError('Independent input reconstruction differs')
            for name,raw in arrays.items():
                if (output/name).read_bytes()!=raw:raise ValueError('Saved input differs: '+name)
        else:
            output.mkdir()
            for name,raw in arrays.items():
                with(output/name).open('xb') as stream:stream.write(raw)
            manifest=dict(REQUIRED,files=identities,provenance={k:v for k,v in report.items() if k not in ('passed','parents','error','replay')},
                native_recipe=dict(B='A with last token incremented by1 modulo151936',source='eight original512 repeats; structural witness, not quality'),
                numerical_or_quality_executed=False,ordinary_app_capacity_changed=False)
            validate(manifest)
            with(output/'manifest.json').open('x') as stream:
                json.dump(manifest,stream,indent=2,allow_nan=False);stream.write('\n')
        report['files']=identities;report['manifest_sha256']=sha((output/'manifest.json').read_bytes());report['passed']=True
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
    return report


if __name__=='__main__':
    result=run(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='replay')
    print(json.dumps(result,indent=2,allow_nan=False))
    sys.exit(0 if result['passed'] else 1)
