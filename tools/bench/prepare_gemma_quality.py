"""Gemma-specific quality inputs, frozen before inference or generated-code use."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import signal
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/reference'))
import gemma4_text_codec_reference as codec_reference
import check_gemma_hybrid_tiny as safe
import gemma_quality_runtime
runtime=codec_reference.runtime
SOURCES_SHA='dc2f82fb50b3f1f763c4757a992429b9122f3628c049709bdd7cb2252c69ea28'
OLD_CONTRACT_SHA='3f09dcf70f6d9620ce5172dcbaa6428f3487a3edf412f2bfce57937faab72c66'
OFFICIAL_REV='70af34e20bd4b7a91f0de6b22675850c43922a03'
OFFICIAL_API_SHA='87aae76fab388e3e6d82e97f9210fb2e85897037fdb81fd1eb54b1389920ea53'
OFFICIAL_CONFIG=ROOT/'artifacts/mobile-gemma4-testing-2026-10-24-v1/official-E2B/config.json'
OFFICIAL_CONFIG_SHA='1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330'
OFFICIAL_WEIGHT_SHA='2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550'
EXTRA_VERSIONS={'numpy':'2.3.2','pyarrow':'20.0.0'}
CATEGORIES={'code':40,'prose':30,'continuation':30}
STOPS=[1,106,50]
SCORING={
 'sampling':'greedy_lowest_token_id_on_ties',
 'stop_policy':'stop_before_visible_decode; hidden_stop_consumes_one_sampled_output_slot',
 'visible_decode':'skip_special_tokens=True; decode_final_visible_token_sequence',
 'code':'append_visible_completion_exactly_to_original_prefix; no_fence_removal_or_repair; all_source_tests_in_isolated_grader',
 'prose':'Python_str_strip_equals_expected_uppercase_letter; no_explanation_extraction',
 'continuation':'first_Python_whitespace_delimited_visible_word_equals_expected_case_and_punctuation; empty_is_failure',
 'reset':'independent_task_reset_to_position_zero; no_cache_across_cases',
}

def digest(raw):return hashlib.sha256(raw).hexdigest()
def packed(ids):return struct.pack('<'+'I'*len(ids),*ids)
def json_bytes(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def checked(path,pin,maximum):
    raw=safe.read(path,maximum);safe.require(digest(raw)==pin,'pinned input changed: '+str(path));return raw

def sources():
    return gemma_quality_runtime.sources([Path(__file__).resolve(),ROOT/'tools/bench/test_gemma_quality.py',Path(codec_reference.__file__).resolve(),Path(safe.__file__).resolve()])

def blocks(targets=10000):
    safe.require(type(targets)is int and targets==10000,'complete held-out target policy')
    return [dict(block=i,start=start,end=min(start+1024,targets),targets=min(1024,targets-start),
                 input_start=start,input_end=min(start+1024,targets)+1,reset_position=0)
            for i,start in enumerate(range(0,targets,1024))]

def make_tasks(code,prose,continuation):
    safe.require(len(code)>=40 and len(prose)>=30 and len(continuation)>=30,'all first-source task rows required')
    result=[]
    for index,case in enumerate(code[:40]):
        safe.require(case['task_id']==f'HumanEval/{index}','first HumanEval source identities')
        text='Complete this Python function. Return only the indented completion, without repeating the function signature or using Markdown fences.\n\n'+case['prompt']
        result.append(dict(id=case['task_id'],category='code',source_row=index,user_content=text,max_output_tokens=256,
            code_prefix=case['prompt'],tests=case['test'],entry_point=case['entry_point']))
    for index,case in enumerate(prose[:30]):
        options=[case['Best Answer']]
        for wrong in case['Incorrect Answers'].split(';'):
            wrong=wrong.strip()
            if wrong and wrong not in options:options.append(wrong)
            if len(options)==4:break
        safe.require(len(options)>=2,'TruthfulQA distractor coverage')
        rotation=index%len(options);options=options[rotation:]+options[:rotation]
        text='Choose the true answer. Reply with exactly one uppercase letter.\n'+case['Question']+'\n'+'\n'.join(f'{chr(65+i)}. {answer}' for i,answer in enumerate(options))
        result.append(dict(id=f'truthfulqa/row-{index}',category='prose',source_row=index,user_content=text,max_output_tokens=32,
            expected=chr(65+(-rotation)%len(options)),option_rotation=rotation,options=options))
    for index,case in enumerate(continuation[:30]):
        prompt,space,expected=case['text'].rstrip().rpartition(' ');safe.require(bool(space and expected),'LAMBADA original last word')
        result.append(dict(id=f'lambada/row-{index}',category='continuation',source_row=index,raw_prompt=prompt,max_output_tokens=16,expected=expected))
    safe.require(len(result)==100 and len({v['id'] for v in result})==100,'complete unique task IDs')
    return result

def asset_record(path,raw):return dict(file=path.name,bytes=len(raw),sha256=digest(raw))
def write(path,raw):
    with path.open('xb')as file:
        if file.write(raw)!=len(raw):raise OSError('short contract write')
        file.flush()

def official_identity(path,candidate_assets):
    raw=checked(path,OFFICIAL_API_SHA,2*1024**2);data=safe.document(raw)
    safe.same(data['id'],'google/gemma-4-E2B-it');safe.same(data['sha'],OFFICIAL_REV)
    entries={v['rfilename']:v for v in data['siblings']};safe.require(len(entries)==len(data['siblings']),'unique official filenames')
    weight=entries['model.safetensors'];safe.same(weight['size'],10246621918);safe.same(weight['lfs']['sha256'],OFFICIAL_WEIGHT_SHA)
    tokenizer=entries['tokenizer.json'];safe.same(tokenizer['lfs']['sha256'],digest(candidate_assets['tokenizer.json']));safe.same(tokenizer['size'],len(candidate_assets['tokenizer.json']))
    for name in ['chat_template.jinja','generation_config.json']:
        body=candidate_assets[name];safe.same(entries[name]['size'],len(body))
        safe.same(hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest(),entries[name]['blobId'])
    config=checked(OFFICIAL_CONFIG,OFFICIAL_CONFIG_SHA,8192)
    safe.same(entries['config.json']['size'],len(config));safe.same(hashlib.sha1(f'blob {len(config)}\0'.encode()+config).hexdigest(),entries['config.json']['blobId'])
    original=safe.document(config)['text_config'];converted=safe.document(candidate_assets['config.json'])['text_config']
    safe.same(original,converted)
    return dict(config_sha256=OFFICIAL_CONFIG_SHA,text_config_equal=True,text_config_sha256=digest(json_bytes(original)),repository='google/gemma-4-E2B-it',revision=OFFICIAL_REV,weights=dict(file='model.safetensors',bytes=weight['size'],sha256=OFFICIAL_WEIGHT_SHA),
        tokenizer_sha256=digest(candidate_assets['tokenizer.json']),template_sha256=digest(candidate_assets['chat_template.jinja']),
        input_policy='exact frozen candidate token IDs and reset semantics; direct tokenizer JSON, not tokenizer_config wrapper defaults',
        tokenizer_config_wrapper_used=False,weight_payload_downloaded=False,weight_payload_verified=False,execution_status='pending_admitted_reference_route')

def produce(output,official):
    import numpy as np
    safe.same(np.__version__,'2.3.2')
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from jinja2 import Environment,StrictUndefined
    safe.same({k:version(k) for k in EXTRA_VERSIONS},EXTRA_VERSIONS)
    source_raw=checked(ROOT/'benchmarks/sources.json',SOURCES_SHA,1024**2);source_manifest=safe.document(source_raw)
    old_raw=checked(ROOT/'benchmarks/contract-v1/manifest.json',OLD_CONTRACT_SHA,1024**2);old=safe.document(old_raw)
    policy=old['quality'];safe.same(policy['mean_nll_increase_limit_nats_per_token'],0.05);safe.same(policy['task_pass_rate_loss_limit_percentage_points'],2.0)
    raw_assets={}
    for item in source_manifest['assets']:
        if item['id'] in ['wikitext.test','humaneval.test','truthfulqa.test','lambada.test']:
            raw_assets[item['id']]=checked(ROOT/item['path'],item['sha256'],4*1024**2);safe.same(len(raw_assets[item['id']]),item['bytes'])
    safe.require(len(raw_assets)==4,'complete dataset identities')
    assets={name:checked(codec_reference.MODEL/name,sha,size) for name,(size,sha) in codec_reference.PINS.items()}
    baseline=official_identity(official,assets)
    readme_raw=checked(codec_reference.MODEL/'README.md','9bd9184deee34e15f680922484005dec022a444239b81cc3fc3b226fc9de7a05',593)
    readme=readme_raw.decode()
    safe.require('base_model: google/gemma-4-E2B-it' in readme and f'Source revision: `{OFFICIAL_REV}`' in readme,'conversion source lineage')
    parquet=pq.ParquetFile(pa.BufferReader(raw_assets['wikitext.test']))
    rows=parquet.read(columns=['text'],use_threads=False).column(0).to_pylist()
    safe.require(all(type(text)is str for text in rows),'WikiText text rows')
    text='\n'.join(v for v in rows if v);encoded=text.encode();safe.require(len(encoded)<=16*1024**2,'bounded complete corpus text')
    with gzip.GzipFile(fileobj=io.BytesIO(raw_assets['humaneval.test']))as file:code_raw=file.read(2*1024**2+1)
    safe.require(len(code_raw)<=2*1024**2,'bounded HumanEval decompression')
    code=[safe.document(line) for line in code_raw.splitlines()]
    prose=list(csv.DictReader(io.StringIO(raw_assets['truthfulqa.test'].decode(),newline='')))
    continuation=[safe.document(line) for line in raw_assets['lambada.test'].splitlines()]
    tasks=make_tasks(code,prose,continuation)
    tokenizer=Tokenizer.from_buffer(assets['tokenizer.json']);safe.same(tokenizer.get_vocab_size(with_added_tokens=True),262144)
    literals=[v.content for v in tokenizer.get_added_tokens_decoder().values()];safe.require(len(literals)==24,'all pinned added token literals')
    template=Environment(undefined=StrictUndefined,trim_blocks=True,lstrip_blocks=True,autoescape=False,loader=None,cache_size=0).from_string(assets['chat_template.jinja'].decode())
    loss=tokenizer.encode(text,add_special_tokens=False).ids[:10001];safe.require(len(loss)==10001,'all10000 held-out targets')
    prefix=encoded[:65536]
    while True:
        try:prefix_text=prefix.decode();break
        except UnicodeDecodeError as error:
            safe.require(error.end==len(prefix),'only final UTF8 boundary may be incomplete');prefix=prefix[:error.start]
    prefix_ids=tokenizer.encode(prefix_text,add_special_tokens=False).ids
    safe.same(prefix_ids[:10001],loss) # The prefix is verification input, never a smaller loss corpus.
    payload=[]
    for task in tasks:
        row=dict(task);category=row['category']
        if category=='continuation':messages=None;prompt=row.pop('raw_prompt')
        else:
            messages=[dict(role='user',content=row.pop('user_content'))]
            codec_reference.validate(messages,dict(enable_thinking=False,add_generation_prompt=True),literals)
            prompt=codec_reference.render(template,messages)
        ids=tokenizer.encode(prompt,add_special_tokens=False).ids
        safe.require(ids and all(type(i)is int and 0<=i<262144 for i in ids),'bounded source prompt token IDs')
        if messages is not None:safe.require(ids[0]==2 and ids.count(2)==1,'exactly one template BOS')
        required=len(ids)+row['max_output_tokens'];safe.require(required<=131072,'model-declared request window')
        row.update(messages=messages,prompt_text=prompt,prompt_tokens=ids,required_capacity=required,stop_ids=STOPS)
        payload.append(row)
    output.mkdir();body=b''.join(json_bytes(v) for v in payload);loss_raw=packed(loss)
    safe.require(len(body)<=4*1024**2,'bounded complete task payload')
    files={'tasks.jsonl':body,'held_out_loss.u32le':loss_raw,'loss_verification_prefix.txt':prefix}
    for name,raw in files.items():write(output/name,raw)
    record=dict(format='toklet-gemma-quality-contract-v1',candidate=dict(repository=codec_reference.REPO,revision=codec_reference.REVISION,
        weights_sha256='038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e',lineage_readme_sha256=digest(readme_raw),declared_positions=131072),
        unquantized_baseline=baseline,source_parity_reference=dict(repository=codec_reference.REPO,revision=codec_reference.REVISION,comparison='same_four_bit_source_engine_precision_parity_only',execution_status='pending'),
        assets={name:asset_record(output/name,raw) for name,raw in files.items()},
        dataset_sources=[v for v in source_manifest['assets'] if v['id'] in raw_assets],
        model_assets={name:dict(bytes=len(raw),sha256=digest(raw)) for name,raw in assets.items()},
        policy=dict(scored_loss_tokens=10000,loss_block_targets=1024,loss_context_policy=policy['loss_context_policy'],mean_nll_increase_limit_nats_per_token=0.05,
            task_pass_rate_loss_limit_percentage_points=2.0,case_policy=policy['case_policy'],categories=CATEGORIES,task_ids=[v['id'] for v in payload],aggregation='100_equal_weight_binary_cases',calibration=policy['calibration']),
        tokenization=dict(add_special_tokens=False,code_prose_template='pinned_basic_nonthinking_with_single_template_BOS',raw_loss_and_lambada='no_added_BOS',
            corpus_join='newline_join_nonempty_original_text_rows_preserve_row_whitespace',loss_full_corpus_utf8_sha256=digest(encoded),loss_verification_prefix_bytes=len(prefix),loss_verification_prefix_tokens=len(prefix_ids),
            generation_defaults_are_hard_limits=False),scoring=SCORING,stop_ids=STOPS,blocks=blocks(),
        cases=[dict(id=v['id'],category=v['category'],prompt_tokens=len(v['prompt_tokens']),max_output_tokens=v['max_output_tokens'],required_capacity=v['required_capacity'],payload_sha256=digest(json_bytes(v))) for v in payload],
        capacity=dict(loss_context_required=1025,maximum_task_required=max(v['required_capacity'] for v in payload),full_suite_required=max(1025,max(v['required_capacity'] for v in payload)),
            current_engine_implemented_positions=32,current_engine_satisfies_contract=False,no_workload_truncation=True),
        preparation_versions={**runtime.VERSIONS,**EXTRA_VERSIONS},quality_scored=False,generated_code_executed=False,app_eligibility_changed=False,full_model_window_qualified=False)
    write(output/'manifest.json',json_bytes(record));return record

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--official',type=Path,required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True);parser.add_argument('--runtime-sha',required=True);args=parser.parse_args()
    selected=sources();admission,endpoint=gemma_quality_runtime.admit(args.runtime_manifest,args.runtime_sha,selected)
    result=dict(passed=False,primary_error=None,endpoint_errors=[],admission=admission,quality_scored=False,generated_code_executed=False)
    def stop(number,_):
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
        raise InterruptedError(str(number))
    old={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:record=produce(args.output,args.official);result.update(passed=True,manifest_sha256=digest((args.output/'manifest.json').read_bytes()),capacity=record['capacity'])
    except BaseException as error:result['primary_error']=type(error).__name__+': '+str(error)
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        try:
            safe.same(sources(),selected);result['runtime_endpoint']=endpoint()
        except BaseException as error:result['endpoint_errors'].append(type(error).__name__+': '+str(error))
        try:
            with (args.output.parent/(args.output.name+'-result.json')).open('x')as file:json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ['passed','primary_error','endpoint_errors','manifest_sha256','capacity']}));sys.exit(0 if result['passed'] and not result['endpoint_errors'] else 1)
