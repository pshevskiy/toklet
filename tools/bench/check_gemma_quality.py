"""Workload integrity plus independent dataset/tokenizer preparation replay.

Shared runtime admission contains no case-construction or tokenization logic.
No candidate inference, source inference, grading or generated code executes.
"""
from pathlib import Path
import argparse
import csv
import copy
import gc
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import signal
import struct
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/reference'))
import gemma4_text_codec_reference as seed
import check_gemma_hybrid_tiny as safe
import gemma_quality_runtime
runtime=seed.runtime

SCORING={
 'sampling':'greedy_lowest_token_id_on_ties',
 'stop_policy':'stop_before_visible_decode; hidden_stop_consumes_one_sampled_output_slot',
 'visible_decode':'skip_special_tokens=True; decode_final_visible_token_sequence',
 'code':'append_visible_completion_exactly_to_original_prefix; no_fence_removal_or_repair; all_source_tests_in_isolated_grader',
 'prose':'Python_str_strip_equals_expected_uppercase_letter; no_explanation_extraction',
 'continuation':'first_Python_whitespace_delimited_visible_word_equals_expected_case_and_punctuation; empty_is_failure',
 'reset':'independent_task_reset_to_position_zero; no_cache_across_cases',
}
BASELINE={'config_sha256': '1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330', 'text_config_equal': True, 'text_config_sha256': '62f8428c49b31ebe5bb8186c7d352ce10498632148703db72568db3cd1a83257', 'repository': 'google/gemma-4-E2B-it', 'revision': '70af34e20bd4b7a91f0de6b22675850c43922a03', 'weights': {'file': 'model.safetensors', 'bytes': 10246621918, 'sha256': '2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550'}, 'tokenizer_sha256': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f', 'template_sha256': '2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082', 'input_policy': 'exact frozen candidate token IDs and reset semantics; direct tokenizer JSON, not tokenizer_config wrapper defaults', 'tokenizer_config_wrapper_used': False, 'weight_payload_downloaded': False, 'weight_payload_verified': False, 'execution_status': 'pending_admitted_reference_route'}
def sources():return gemma_quality_runtime.sources([Path(__file__).resolve(),ROOT/'tools/bench/test_gemma_quality.py',Path(seed.__file__).resolve(),Path(safe.__file__).resolve()])
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def structure(directory,manifest_sha):
    body=safe.read(directory/'manifest.json',2*1024**2);safe.require(sha(body)==manifest_sha,'externally frozen workload manifest')
    m=safe.document(body)
    safe.same(set(m),{'format','candidate','unquantized_baseline','source_parity_reference','assets','dataset_sources','model_assets','policy','tokenization','scoring','stop_ids','blocks','cases','capacity','preparation_versions','quality_scored','generated_code_executed','app_eligibility_changed','full_model_window_qualified'})
    safe.same(m['format'],'toklet-gemma-quality-contract-v1')
    for flag in ['quality_scored','generated_code_executed','app_eligibility_changed','full_model_window_qualified']:safe.same(m[flag],False)
    safe.same(m['candidate'],dict(repository='mlx-community/gemma-4-e2b-it-4bit',revision='238767527555cb75a05732a84dff5d6ba0dd6809',weights_sha256='038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e',lineage_readme_sha256='9bd9184deee34e15f680922484005dec022a444239b81cc3fc3b226fc9de7a05',declared_positions=131072))
    safe.same(set(m['policy']),{'scored_loss_tokens','loss_block_targets','loss_context_policy','mean_nll_increase_limit_nats_per_token','task_pass_rate_loss_limit_percentage_points','case_policy','categories','task_ids','aggregation','calibration'})
    safe.same(m['scoring'],SCORING)
    safe.same(m['policy']['case_policy'],'first_source_rows_in_order_no_model_based_selection')
    for key,value in [('add_special_tokens',False),('code_prose_template','pinned_basic_nonthinking_with_single_template_BOS'),('raw_loss_and_lambada','no_added_BOS'),('corpus_join','newline_join_nonempty_original_text_rows_preserve_row_whitespace'),('generation_defaults_are_hard_limits',False)]:safe.same(m['tokenization'][key],value)
    safe.same(m['policy']['loss_context_policy'],'independent_blocks_of_up_to_1025_tokens_one_token_overlap_score_every_target_once')
    safe.same(m['policy']['calibration'],'held_out_sources_and_all_selected_cases_forbidden_for_quantizer_tuning')
    safe.same(m['stop_ids'],[1,106,50]);safe.same(m['policy']['scored_loss_tokens'],10000);safe.same(m['policy']['loss_block_targets'],1024)
    safe.same(m['policy']['mean_nll_increase_limit_nats_per_token'],0.05);safe.same(m['policy']['task_pass_rate_loss_limit_percentage_points'],2.)
    safe.same(m['policy']['categories'],dict(code=40,prose=30,continuation=30));safe.same(m['policy']['aggregation'],'100_equal_weight_binary_cases')
    expected_ids=[f'HumanEval/{i}' for i in range(40)]+[f'truthfulqa/row-{i}' for i in range(30)]+[f'lambada/row-{i}' for i in range(30)]
    safe.same(m['policy']['task_ids'],expected_ids)
    assets={}
    safe.same(set(m['assets']),{'tasks.jsonl','held_out_loss.u32le','loss_verification_prefix.txt'})
    for name,spec in m['assets'].items():
        safe.same(set(spec),{'file','bytes','sha256'});safe.same(spec['file'],name)
        raw=safe.read(directory/name,4*1024**2)
        safe.require(type(spec['bytes'])is int and spec['bytes']==len(raw) and spec['sha256']==sha(raw),'complete contract asset')
        assets[name]=raw
    safe.same(len(assets['held_out_loss.u32le']),40004)
    tokens=[v for (v,) in struct.iter_unpack('<I',assets['held_out_loss.u32le'])]
    safe.require(all(i<262144 for i in tokens),'Gemma loss vocabulary')
    expected_blocks=[]
    for i,start in enumerate(range(0,10000,1024)):
        end=min(start+1024,10000);expected_blocks.append(dict(block=i,start=start,end=end,targets=end-start,input_start=start,input_end=end+1,reset_position=0))
    safe.same(m['blocks'],expected_blocks)
    payload=[safe.document(line) for line in assets['tasks.jsonl'].splitlines()];safe.same([v['id'] for v in payload],expected_ids);safe.same(len(m['cases']),100)
    for index,(row,summary) in enumerate(zip(payload,m['cases'],strict=True)):
        category=row['category'];safe.same(category,'code' if index<40 else 'prose' if index<70 else 'continuation')
        common={'id','category','source_row','max_output_tokens','messages','prompt_text','prompt_tokens','required_capacity','stop_ids'}
        extra={'code_prefix','tests','entry_point'} if category=='code' else {'expected','option_rotation','options'} if category=='prose' else {'expected'}
        safe.same(set(row),common|extra)
        cap={'code':256,'prose':32,'continuation':16}[category];safe.same(row['max_output_tokens'],cap);safe.same(row['stop_ids'],[1,106,50])
        ids=row['prompt_tokens'];safe.require(type(ids)is list and ids and all(type(i)is int and 0<=i<262144 for i in ids),'complete prompt IDs')
        safe.same(row['required_capacity'],len(ids)+cap)
        safe.same(summary,dict(id=row['id'],category=category,prompt_tokens=len(ids),max_output_tokens=cap,required_capacity=len(ids)+cap,payload_sha256=sha(encoded(row))))
        if category!='continuation':safe.require(ids[0]==2 and ids.count(2)==1,'one template BOS')
        else:safe.same(row['messages'],None)
    maximum=max(v['required_capacity'] for v in payload)
    safe.same(m['capacity'],dict(loss_context_required=1025,maximum_task_required=maximum,full_suite_required=max(1025,maximum),current_engine_implemented_positions=32,current_engine_satisfies_contract=False,no_workload_truncation=True))
    safe.same(m['unquantized_baseline'],BASELINE)
    safe.same(m['source_parity_reference'],dict(repository='mlx-community/gemma-4-e2b-it-4bit',revision='238767527555cb75a05732a84dff5d6ba0dd6809',comparison='same_four_bit_source_engine_precision_parity_only',execution_status='pending'))
    return m,payload,tokens,assets

def equal(actual,expected,label):
    try:safe.same(actual,expected)
    except ValueError as error:raise ValueError(label) from error

def source_replay(directory,manifest_sha,official):
    import numpy as np
    safe.same(np.__version__,'2.3.2')
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    m,payload,tokens,assets=structure(directory,manifest_sha)
    safe.same(m['preparation_versions'],{**runtime.VERSIONS,'numpy':'2.3.2','pyarrow':'20.0.0'})
    safe.same({n:version(n) for n in ('numpy','pyarrow')},{'numpy':'2.3.2','pyarrow':'20.0.0'})
    source_raw=safe.read(ROOT/'benchmarks/sources.json',1024**2);safe.require(sha(source_raw)=='dc2f82fb50b3f1f763c4757a992429b9122f3628c049709bdd7cb2252c69ea28','original dataset source manifest')
    source=safe.document(source_raw);expected_sources=[v for v in source['assets'] if v['id'] in {'wikitext.test','humaneval.test','truthfulqa.test','lambada.test'}];safe.same(m['dataset_sources'],expected_sources)
    raw={}
    for item in expected_sources:
        value=safe.read(ROOT/item['path'],4*1024**2);safe.require(len(value)==item['bytes'] and sha(value)==item['sha256'],'original source bytes');raw[item['id']]=value
    safe.same(set(m['model_assets']),set(seed.PINS))
    for name,(size,pin) in seed.PINS.items():
        safe.same(m['model_assets'][name],dict(bytes=size,sha256=pin))
        actual=safe.read(seed.MODEL/name,size);safe.require(len(actual)==size and sha(actual)==pin,'source model metadata')
    api=safe.read(official,2*1024**2);safe.require(sha(api)=='87aae76fab388e3e6d82e97f9210fb2e85897037fdb81fd1eb54b1389920ea53','official source inventory identity')
    inventory=safe.document(api);safe.same(inventory['id'],BASELINE['repository']);safe.same(inventory['sha'],BASELINE['revision'])
    entries={v['rfilename']:v for v in inventory['siblings']};safe.require(len(entries)==len(inventory['siblings']),'official unique asset inventory')
    for name in ['model.safetensors','tokenizer.json']:
        item=entries[name];expected=BASELINE['weights'] if name=='model.safetensors' else dict(bytes=32169626,sha256=BASELINE['tokenizer_sha256'])
        safe.same(item['size'],expected['bytes']);safe.same(item['lfs']['sha256'],expected['sha256'])
    for name in ['chat_template.jinja','generation_config.json']:
        body=safe.read(seed.MODEL/name,65536);safe.same(entries[name]['size'],len(body));safe.same(entries[name]['blobId'],hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest())
    config=safe.read(ROOT/'artifacts/mobile-gemma4-testing-2026-10-24-v1/official-E2B/config.json',8192);safe.same(sha(config),BASELINE['config_sha256'])
    safe.same(entries['config.json']['blobId'],hashlib.sha1(f'blob {len(config)}\0'.encode()+config).hexdigest())
    text_config=safe.document(config)['text_config'];converted=safe.document(safe.read(seed.MODEL/'config.json',8192))['text_config'];safe.same(text_config,converted);safe.same(sha(encoded(text_config)),BASELINE['text_config_sha256'])
    readme=safe.read(seed.MODEL/'README.md',593);safe.same(sha(readme),'9bd9184deee34e15f680922484005dec022a444239b81cc3fc3b226fc9de7a05')
    lines=readme.decode().splitlines();safe.require('base_model: google/gemma-4-E2B-it' in lines and '- Source revision: `'+BASELINE['revision']+'`' in lines,'conversion lineage')
    tokenizer_raw=safe.read(seed.MODEL/'tokenizer.json',33*1024**2);safe.require(sha(tokenizer_raw)=='cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f','tokenizer source identity')
    tokenizer=Tokenizer.from_buffer(tokenizer_raw)
    parquet=pq.ParquetFile(pa.BufferReader(raw['wikitext.test']));texts=[]
    for batch in parquet.iter_batches(batch_size=73,columns=['text'],use_threads=False):
        for text in batch.column(0).to_pylist():
            safe.require(type(text)is str,'WikiText strings')
            if text:texts.append(text)
    corpus='\n'.join(texts);safe.same(sha(corpus.encode()),m['tokenization']['loss_full_corpus_utf8_sha256'])
    equal(tokenizer.encode(corpus,add_special_tokens=False).ids[:10001],tokens,'source loss token IDs differ')
    prefix=assets['loss_verification_prefix.txt'];safe.require(len(prefix)<=65536 and corpus.encode().startswith(prefix),'original corpus verification prefix')
    prefix_ids=tokenizer.encode(prefix.decode(),add_special_tokens=False).ids;safe.same(prefix_ids[:10001],tokens)
    safe.same(m['tokenization']['loss_verification_prefix_tokens'],len(prefix_ids));safe.same(m['tokenization']['loss_verification_prefix_bytes'],len(prefix))
    safe.same(m['tokenization'],dict(add_special_tokens=False,code_prose_template='pinned_basic_nonthinking_with_single_template_BOS',raw_loss_and_lambada='no_added_BOS',corpus_join='newline_join_nonempty_original_text_rows_preserve_row_whitespace',loss_full_corpus_utf8_sha256=sha(corpus.encode()),loss_verification_prefix_bytes=len(prefix),loss_verification_prefix_tokens=len(prefix_ids),generation_defaults_are_hard_limits=False))
    with gzip.GzipFile(fileobj=io.BytesIO(raw['humaneval.test']))as file:code_data=file.read(2*1024**2+1)
    safe.require(len(code_data)<=2*1024**2,'HumanEval decompression bound');code=[safe.document(v) for v in code_data.splitlines()]
    prose=list(csv.DictReader(io.StringIO(raw['truthfulqa.test'].decode(),newline='')));continuation=[safe.document(v) for v in raw['lambada.test'].splitlines()]
    for index,row in enumerate(payload):
        category=row['category'];local=index if index<40 else index-40 if index<70 else index-70;safe.same(row['source_row'],local)
        if category=='code':
            original=code[local];safe.same(row['id'],original['task_id'])
            for key,target in [('code_prefix','prompt'),('tests','test'),('entry_point','entry_point')]:equal(row[key],original[target],'source HumanEval '+key+' differs')
            content='Complete this Python function. Return only the indented completion, without repeating the function signature or using Markdown fences.\n\n'+original['prompt']
        elif category=='prose':
            original=prose[local];choices=[original['Best Answer']]
            for value in original['Incorrect Answers'].split(';'):
                item=value.strip()
                if item and item not in choices:choices.append(item)
                if len(choices)==4:break
            safe.require(len(choices)>=2,'prose distractors');shift=local%len(choices);choices=choices[shift:]+choices[:shift]
            safe.same(row['options'],choices);safe.same(row['option_rotation'],shift);equal(row['expected'],chr(65+(-shift)%len(choices)),'source TruthfulQA answer differs')
            content='Choose the true answer. Reply with exactly one uppercase letter.\n'+original['Question']+'\n'+'\n'.join(chr(65+i)+'. '+v for i,v in enumerate(choices))
        else:
            original=continuation[local]['text'].rstrip().rsplit(' ',1);safe.require(len(original)==2,'continuation suffix');content=original[0];equal(row['expected'],original[1],'source LAMBADA answer differs')
        if category=='continuation':expected_prompt=content
        else:
            equal(row['messages'],[dict(role='user',content=content)],'source prompt messages differ')
            expected_prompt='<bos><|turn>user\n'+content.strip()+'<turn|>\n<|turn>model\n'
        equal(row['prompt_text'],expected_prompt,'source rendered prompt differs');equal(row['prompt_tokens'],tokenizer.encode(expected_prompt,add_special_tokens=False).ids,'source prompt token IDs differ')
    return dict(passed=True,tasks=100,loss_targets=10000,blocks=10,capacity=m['capacity'],source_rows_and_tokenization_verified=True,quality_scored=False,generated_code_executed=False,full_model_window_qualified=False)

def source_corruption_controls(directory,manifest_sha,official):
    original,rows,_,assets=structure(directory,manifest_sha)
    controls=[('prompt_messages','source prompt messages differ'),('truthful_answer','source TruthfulQA answer differs'),
        ('lambada_answer','source LAMBADA answer differs'),('prompt_ids','source prompt token IDs differ'),
        ('loss_ids','source loss token IDs differ'),('code_tests','source HumanEval tests differs')]
    completed=[]
    for label,expected_error in controls:
        m=copy.deepcopy(original);payload=copy.deepcopy(rows);data=dict(assets)
        if label=='prompt_messages':payload[0]['messages'][0]['content']+=' synthetic corruption'
        elif label=='truthful_answer':payload[40]['expected']='B' if payload[40]['expected']=='A' else 'A'
        elif label=='lambada_answer':payload[70]['expected']+='Corrupted'
        elif label=='prompt_ids':payload[0]['prompt_tokens'][1]=(payload[0]['prompt_tokens'][1]+1)%262144
        elif label=='loss_ids':
            changed=bytearray(data['held_out_loss.u32le']);value=struct.unpack_from('<I',changed,4)[0];struct.pack_into('<I',changed,4,(value+1)%262144);data['held_out_loss.u32le']=bytes(changed)
        else:payload[0]['tests']+='\n# synthetic verifier corruption'
        data['tasks.jsonl']=b''.join(encoded(v) for v in payload)
        for name,raw in data.items():m['assets'][name]=dict(file=name,bytes=len(raw),sha256=sha(raw))
        m['cases']=[dict(id=v['id'],category=v['category'],prompt_tokens=len(v['prompt_tokens']),max_output_tokens=v['max_output_tokens'],required_capacity=v['required_capacity'],payload_sha256=sha(encoded(v))) for v in payload]
        with tempfile.TemporaryDirectory(prefix='quality-source-controls-',dir=directory.parent)as folder:
            root=Path(folder)
            for name,raw in data.items():
                with(root/name).open('xb')as file:file.write(raw)
            manifest=encoded(m)
            with(root/'manifest.json').open('xb')as file:file.write(manifest)
            changed_sha=sha(manifest);structure(root,changed_sha) # A hash/shape failure cannot count as source replay evidence.
            try:source_replay(root,changed_sha,official)
            except ValueError as error:
                safe.same(str(error),expected_error);completed.append(dict(control=label,coherent_manifest_sha256=changed_sha,expected_error=expected_error,rejected=True))
            else:raise ValueError('coherent source corruption accepted: '+label)
        gc.collect()
    return completed

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--official',type=Path,required=True);parser.add_argument('--exercise-controls',action='store_true')
    parser.add_argument('--runtime-manifest',type=Path,required=True);parser.add_argument('--runtime-sha',required=True);args=parser.parse_args()
    selected=sources();admission,endpoint=gemma_quality_runtime.admit(args.runtime_manifest,args.runtime_sha,selected)
    result=dict(passed=False,primary_error=None,endpoint_errors=[],admission=admission)
    def stop(number,_):
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
        raise InterruptedError(str(number))
    old={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        result['replay']=source_replay(args.contract,args.manifest_sha,args.official)
        if args.exercise_controls:result['source_controls']=source_corruption_controls(args.contract,args.manifest_sha,args.official)
        result['passed']=True
    except BaseException as error:result['primary_error']=type(error).__name__+': '+str(error)
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        try:safe.same(sources(),selected);result['runtime_endpoint']=endpoint()
        except BaseException as error:result['endpoint_errors'].append(type(error).__name__+': '+str(error))
        try:
            report_name='source-controls-replay.json' if args.exercise_controls else 'independent-replay.json'
            with(args.contract.parent/report_name).open('x')as file:json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ['passed','primary_error','endpoint_errors','replay']}));sys.exit(0 if result['passed'] and not result['endpoint_errors'] else 1)
