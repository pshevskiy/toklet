"""Freeze complete original MiniCPM5 quality inputs before model scoring."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/reference'))
import minicpm5_text_codec_reference as codec
import gemma_quality_runtime as parsers
import quality_contract_policy as policy

SOURCES_SHA = 'dc2f82fb50b3f1f763c4757a992429b9122f3628c049709bdd7cb2252c69ea28'
OLD_SHA = '3f09dcf70f6d9620ce5172dcbaa6428f3487a3edf412f2bfce57937faab72c66'
STOCK = ('openbmb/MiniCPM5-2B', '42433e90dad95123ab87729fd7011e7110629556')
STOCK_WEIGHT = ('model-00000-of-00001.safetensors', 5033557096,
                '14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c')
PARSER_LIMIT = 1024**3
need = policy.require


def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False)+'\n').encode()
def document(raw):
    return json.loads(raw, object_pairs_hook=codec.runtime.common.json_unique,
                      parse_constant=codec.runtime.common.bad_json_number)
def same(a,b):return type(a) is type(b) and encoded(a)==encoded(b)
def captured(path, pin, maximum):return codec.runtime.common.captured(path, maximum, pin)


def sources():
    folder=Path(__file__).resolve().parent
    return parsers.sources([Path(__file__).resolve(), folder/'check_minicpm5_quality_inputs.py',
        folder/'test_minicpm5_quality_inputs.py', Path(policy.__file__).resolve(),
        Path(codec.__file__).resolve(), *codec.sources()])


def contract(path, pin):
    raw=captured(path,pin,2*1024**2);value=document(raw)
    need(value['format']=='toklet-minicpm5-quality-inputs-contract-v1'
         and value['prospective_before_loss_task_or_native_quality_outcomes'] is True,
         'Prospective full quality contract required')
    need(same(value['quality'],dict(mean_nll_increase_limit_nats_per_token=0.05,
        task_pass_rate_loss_limit_percentage_points=2.0,loss_targets=10000,tasks=100,
        categories=dict(code=40,prose=30,continuation=30),
        output_caps=dict(code=256,prose=32,continuation=16))), 'Original full quality gates differ')
    need(value['corpus_source_manifest_sha256']==SOURCES_SHA
         and value['historical_quality_contract_sha256']==OLD_SHA
         and value['candidate']['repository']==codec.REPO
         and value['candidate']['revision']==codec.REVISION
         and (value['original_stock']['repository'],value['original_stock']['revision'])==STOCK,
         'Pinned corpus/candidate/stock identity differs')
    for key,want in [('source_parser_allowance_bytes',PARSER_LIMIT),('codec_allowance_bytes',512*1024**2),
                     ('engine_ceiling_bytes',6*1024**3),('disk_floor_bytes',2*1024**3),
                     ('current_engine_capacity',512),('current_prefill_chunk',32)]:
        need(type(value[key]) is int and value[key]==want, 'Quality preparation bounds differ')
    return value


def datasets(environment):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from importlib.metadata import version
    need(np.__version__=='2.3.2' and version('pyarrow')=='20.0.0', 'Exact data-parser versions required')
    source=document(captured(environment/'benchmarks/sources.json',SOURCES_SHA,1024**2))
    old=document(captured(environment/'benchmarks/contract-v1/manifest.json',OLD_SHA,1024**2))
    need(old['quality']['mean_nll_increase_limit_nats_per_token']==0.05
         and old['quality']['task_pass_rate_loss_limit_percentage_points']==2.0,
         'Historical quality gates changed')
    pins={row['id']:row for row in source['assets'] if row['id'] in
          ['wikitext.test','humaneval.test','truthfulqa.test','lambada.test']}
    need(len(pins)==4,'Four original datasets required')
    raw={}
    for key,row in pins.items():
        raw[key]=captured(environment/row['path'],row['sha256'],4*1024**2)
        need(len(raw[key])==row['bytes'],'Original dataset extent differs')
    rows=pq.ParquetFile(pa.BufferReader(raw['wikitext.test'])).read(columns=['text'],use_threads=False).column(0).to_pylist()
    need(all(type(t) is str for t in rows),'WikiText string rows required')
    text='\n'.join(t for t in rows if t)
    need(65536<=len(text.encode())<=16*1024**2,'Complete corpus size differs')
    with gzip.GzipFile(fileobj=io.BytesIO(raw['humaneval.test'])) as stream:code=stream.read(2*1024**2+1)
    need(len(code)<=2*1024**2,'HumanEval decompression exceeds bound')
    code=[document(line) for line in code.splitlines()]
    prose=list(csv.DictReader(io.StringIO(raw['truthfulqa.test'].decode(),newline='')))
    continuation=[document(line) for line in raw['lambada.test'].splitlines()]
    return pins,text,code,prose,continuation


def assets(directory):
    result={name:captured(directory/name,pin,count) for name,(count,pin) in codec.PINS.items()}
    need(all(len(result[name])==count for name,(count,pin) in codec.PINS.items()),'Exact text asset extent differs')
    return result


def stock_assets(directory, frozen, candidate):
    result={}
    for name,pin in frozen['original_stock']['metadata_pins'].items():
        result[name]=captured(directory/name,pin['sha256'],16*1024**2)
        need(len(result[name])==pin['bytes'],'Stock metadata extent differs')
    api=document(result['model-api.json']);need((api['id'],api['sha'])==STOCK,'Stock API identity differs')
    entries={row['rfilename']:row for row in api['siblings']}
    name,size,sha=STOCK_WEIGHT
    need(entries[name]['size']==size and entries[name]['lfs']['sha256']==sha
         and same(frozen['original_stock']['weight_objects'],[entries[name]]),'Original stock weight identity differs')
    for name,raw in result.items():
        if name=='model-api.json':continue
        item=entries[name];need(item['size']==len(raw),'Stock API/file extent differs')
        pin=item['lfs']['sha256'] if 'lfs' in item else item['blobId']
        actual=digest(raw) if 'lfs' in item else hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        need(actual==pin,'Stock API/file digest differs')
    original=document(result['tokenizer.json']);converted=document(candidate['tokenizer.json'])
    need(set(original)==set(converted) and all(same(original[k],converted[k])
         for k in original if k!='padding'),'Original tokenizer semantics differ')
    need(original['padding'] is None and converted['padding']==dict(strategy='BatchLongest',direction='Left',
         pad_to_multiple_of=None,pad_id=1,pad_type_id=0,pad_token='</s>'), 'Explicit source padding difference changed')
    for raw in [candidate['config.json'],result['config.json']]:
        cfg=document(raw)
        for key,want in [('vocab_size',130560),('hidden_size',2048),('intermediate_size',6144),
                         ('num_hidden_layers',42),('num_attention_heads',16),('num_key_value_heads',2),
                         ('head_dim',128),('bos_token_id',0),('eos_token_id',[1,130073]),
                         ('tie_word_embeddings',False),('rms_norm_eps',1e-6),('rope_theta',5000000)]:
            need(same(cfg[key],want),'Shared input/geometry declaration differs: '+key)
    return result


def prefix_bytes(text):
    raw=text.encode()[:65536]
    while True:
        try:raw.decode();return raw
        except UnicodeDecodeError as e:
            need(e.end==len(raw),'Only final UTF8 boundary may be incomplete');raw=raw[:e.start]


def prepare(environment, candidate_directory, stock_directory, frozen, contract_sha):
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    source_pins,text,code,prose,continuation=datasets(environment)
    candidate=assets(candidate_directory);stock=stock_assets(stock_directory,frozen,candidate)
    tokenizer=Tokenizer.from_buffer(candidate['tokenizer.json']);original=Tokenizer.from_buffer(stock['tokenizer.json'])
    need(tokenizer.get_vocab_size()==original.get_vocab_size()==130560,'Original/candidate vocabulary differs')
    literals=[v.content for v in tokenizer.get_added_tokens_decoder().values()]
    need(len(literals)==510,'Exact reserved-token inventory differs')
    environment_jinja=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,
        extensions=['jinja2.ext.loopcontrols'],loader=None,cache_size=0)
    templates=[environment_jinja.from_string(a['chat_template.jinja'].decode()) for a in [candidate,stock]]
    all_ids=tokenizer.encode(text,add_special_tokens=False).ids
    need(len(all_ids)>10000 and all_ids==original.encode(text,add_special_tokens=False).ids,
         'Complete corpus original/candidate token IDs differ')
    loss=all_ids[:10001];prefix=prefix_bytes(text)
    need(tokenizer.encode(prefix.decode(),add_special_tokens=False).ids[:10001]==loss,
         'Verification prefix cannot replace complete loss coverage')
    tasks=[]
    for source in policy.make_tasks(code,prose,continuation):
        row=dict(source);messages=None
        if row['category']=='continuation':rendered=row.pop('raw_prompt');rendered_stock=rendered
        else:
            messages=[dict(role='user',content=row['user_content'])]
            options=dict(enable_thinking=False,add_generation_prompt=True)
            codec.validate(messages,options,literals)
            rendered,rendered_stock=[template.render(messages=messages,tools=None,bos_token='<s>',**options)
                                     for template in templates]
        need(rendered==rendered_stock and len(rendered.encode())<=65536,
             'Full task original/candidate render differs: '+row['id'])
        ids=tokenizer.encode(rendered,add_special_tokens=False).ids
        need(ids and ids==original.encode(rendered_stock,add_special_tokens=False).ids
             and len(ids)<=65536,'Full task original/candidate token IDs differ')
        row.update(messages=messages,prompt=rendered,prompt_tokens=ids,
                   required_capacity=len(ids)+row['max_output_tokens'])
        tasks.append(row)
    eos=codec.eos_witness(1,[1,130073]);need(eos['stop_ids']==[1,130073],'Original stop union differs')
    payloads={'held_out_loss.u32le':struct.pack('<10001I',*loss),
              'tasks.jsonl':b''.join(encoded(row) for row in tasks),
              'loss_verification_prefix.txt':prefix}
    need(sum(map(len,payloads.values()))<=2*1024**2,'Full quality inputs exceed bound')
    record=dict(format='toklet-minicpm5-complete-quality-inputs-v1',contract_sha256=contract_sha,
        candidate=frozen['candidate'],modified_parent=frozen['modified_parent'],
        original_stock=frozen['original_stock'],comparison_order=frozen['comparisons'],
        datasets=source_pins,quality=frozen['quality'],scoring=policy.SCORING,blocks=policy.blocks(),
        complete_corpus_tokens=len(all_ids),loss_targets=10000,task_cases=100,
        cases=[dict(id=row['id'],category=row['category'],prompt_tokens=len(row['prompt_tokens']),
                    max_output_tokens=row['max_output_tokens'],required_capacity=row['required_capacity'],
                    payload_sha256=digest(encoded(row))) for row in tasks],
        assets={name:dict(bytes=len(raw),sha256=digest(raw)) for name,raw in candidate.items()},
        files={name:dict(bytes=len(raw),sha256=digest(raw)) for name,raw in payloads.items()},
        capacity=dict(loss_context_required=1025,maximum_task_required=max(row['required_capacity'] for row in tasks),
            full_suite_required=max(1025,max(row['required_capacity'] for row in tasks)),
            current_engine_capacity=512,current_engine_satisfies_contract=False,no_workload_truncation=True),
        input_equivalence=dict(complete_corpus_token_ids=True,all_100_task_rendered_strings=True,
            all_100_task_token_ids=True,tokenizer_core_structure_equal=True,
            padding_difference='stock None vs candidate BatchLongest-left; single sequence retains identical IDs',
            wrapper_defaults_used=False),stops=eos,
        model_executed=False,quality_scored=False,generated_code_executed=False,
        parent_weights_downloaded=False,managed_adoption_qualified=False,performance_or_overhead_qualified=False)
    payloads['manifest.json']=encoded(record)
    return payloads,record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['environment_root','candidate_assets','stock_metadata','contract','contract_sha256',
                 'runtime_manifest','runtime_sha256','fresh_output']:
        parser.add_argument('--'+name.replace('_','-'),required=True)
    args=parser.parse_args();root=Path(args.environment_root).resolve();codec.bind_environment(root)
    frozen=contract(Path(args.contract),args.contract_sha256)
    selected=sources();admitted,endpoint=parsers.admit(Path(args.runtime_manifest),args.runtime_sha256,selected)
    result=dict(passed=False,admitted_runtime=admitted,primary_error=None,endpoint_errors=[])
    try:
        payloads,record=prepare(root,Path(args.candidate_assets),Path(args.stock_metadata),frozen,args.contract_sha256)
        output=Path(args.fresh_output);output.mkdir()
        for name,raw in payloads.items():
            with (output/name).open('xb') as stream:need(stream.write(raw)==len(raw),'Short quality input write')
        result.update(passed=True,manifest_sha256=digest(payloads['manifest.json']),capacity=record['capacity'],
            loss_targets=10000,task_cases=100,quality_scored=False,model_executed=False)
    except BaseException as e:result['primary_error']=type(e).__name__+': '+str(e)
    finally:
        try:
            need(sources()==selected,'Quality input/runtime source inventory changed')
            result['runtime_endpoint']=endpoint()
        except BaseException as e:result['endpoint_errors'].append(type(e).__name__+': '+str(e))
    output=Path(args.fresh_output)
    with output.with_name(output.name+'-result.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['admitted_runtime','runtime_endpoint']}))
    return 0 if result['passed'] and not result['endpoint_errors'] else 1


if __name__=='__main__':sys.exit(main())
