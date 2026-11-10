"""Validate full MiniCPM5 inputs and replay their frozen source recipe."""
import argparse
import json
from pathlib import Path
import re
import struct
import sys

import prepare_minicpm5_quality as producer

need,same,digest,encoded,document = producer.need,producer.same,producer.digest,producer.encoded,producer.document
FILES = {'held_out_loss.u32le','tasks.jsonl','loss_verification_prefix.txt'}
FIELDS = {'format','contract_sha256','candidate','modified_parent','original_stock','comparison_order',
    'datasets','quality','scoring','blocks','complete_corpus_tokens','loss_targets','task_cases','cases',
    'assets','files','capacity','input_equivalence','stops','model_executed','quality_scored',
    'generated_code_executed','parent_weights_downloaded','managed_adoption_qualified',
    'performance_or_overhead_qualified'}


def integer(value,minimum=0):return type(value) is int and value>=minimum
def sha(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def validate(blobs,frozen,contract_sha):
    need(type(blobs) is dict and set(blobs)==FILES|{'manifest.json'},'Complete quality file inventory differs')
    need(all(type(raw) is bytes and len(raw)<=2*1024**2 for raw in blobs.values()),'Quality payload bound differs')
    manifest=document(blobs['manifest.json'])
    need(type(manifest) is dict and set(manifest)==FIELDS
         and manifest['format']=='toklet-minicpm5-complete-quality-inputs-v1'
         and manifest['contract_sha256']==contract_sha,'Full quality manifest schema/contract differs')
    for key,source_key in [('candidate','candidate'),('modified_parent','modified_parent'),
        ('original_stock','original_stock'),('quality','quality'),('comparison_order','comparisons')]:
        need(same(manifest[key],frozen[source_key]),'Quality source identity/order/gates differ')
    need(same(manifest['scoring'],producer.policy.SCORING)
         and same(manifest['blocks'],producer.policy.blocks()),'Original scoring or complete loss blocks differ')
    need(integer(manifest['complete_corpus_tokens'],10001)
         and type(manifest['loss_targets']) is int and manifest['loss_targets']==10000
         and type(manifest['task_cases']) is int and manifest['task_cases']==100,
         'Complete corpus/loss/task coverage differs')
    for key in ['model_executed','quality_scored','generated_code_executed','parent_weights_downloaded',
                'managed_adoption_qualified','performance_or_overhead_qualified']:
        need(manifest[key] is False,'Input preparation cannot qualify execution/adoption')
    need(type(manifest['files']) is dict and set(manifest['files'])==FILES,'Complete input descriptors differ')
    for name,pin in manifest['files'].items():
        need(type(pin) is dict and set(pin)=={'bytes','sha256'} and integer(pin['bytes'],1)
             and pin['bytes']==len(blobs[name]) and sha(pin['sha256'])
             and pin['sha256']==digest(blobs[name]),'Input payload hash/extent differs')
    need(len(blobs['held_out_loss.u32le'])==40004,'All 10001 loss token IDs required')
    ids=struct.unpack('<10001I',blobs['held_out_loss.u32le'])
    need(all(token<130560 for token in ids),'Loss token outside vocabulary')
    prefix=blobs['loss_verification_prefix.txt'];prefix.decode()
    need(65532<=len(prefix)<=65536,'Complete verification-prefix extent differs')
    need(blobs['tasks.jsonl'].endswith(b'\n'),'Incomplete task journal')
    tasks=[document(line) for line in blobs['tasks.jsonl'].splitlines()]
    need(len(tasks)==100 and type(manifest['cases']) is list and len(manifest['cases'])==100,
         'All 100 original task records required')
    expected_ids=[f'HumanEval/{i}' for i in range(40)]+[f'truthfulqa/row-{i}' for i in range(30)]+[f'lambada/row-{i}' for i in range(30)]
    for index,(row,case,name) in enumerate(zip(tasks,manifest['cases'],expected_ids)):
        category='code' if index<40 else 'prose' if index<70 else 'continuation'
        cap={'code':256,'prose':32,'continuation':16}[category]
        source_row=index if index<40 else index-40 if index<70 else index-70
        need(type(row) is dict and row.get('id')==name and row.get('category')==category
             and type(row.get('source_row')) is int and row['source_row']==source_row
             and type(row.get('max_output_tokens')) is int and row['max_output_tokens']==cap,
             'Original task identities/category/caps differ')
        prompt=row.get('prompt_tokens');text=row.get('prompt')
        need(type(text) is str and 0<len(text.encode())<=65536
             and type(prompt) is list and 0<len(prompt)<=65536
             and all(integer(token) and token<130560 for token in prompt)
             and type(row.get('required_capacity')) is int and row['required_capacity']==len(prompt)+cap,
             'Complete task prompt/capacity differs')
        need(type(case) is dict and same(case,dict(id=name,category=category,prompt_tokens=len(prompt),
            max_output_tokens=cap,required_capacity=len(prompt)+cap,payload_sha256=digest(encoded(row)))),
            'Task summary/payload hash differs')
    capacity=dict(loss_context_required=1025,maximum_task_required=max(row['required_capacity'] for row in tasks),
        full_suite_required=max(1025,max(row['required_capacity'] for row in tasks)),
        current_engine_capacity=512,current_engine_satisfies_contract=False,no_workload_truncation=True)
    need(same(manifest['capacity'],capacity),'Complete capacity cannot be shortened or marked qualified')
    need(same(manifest['input_equivalence'],dict(complete_corpus_token_ids=True,
        all_100_task_rendered_strings=True,all_100_task_token_ids=True,tokenizer_core_structure_equal=True,
        padding_difference='stock None vs candidate BatchLongest-left; single sequence retains identical IDs',
        wrapper_defaults_used=False)),'Original input-equivalence scope differs')
    need(type(manifest['stops']) is dict and same(manifest['stops']['stop_ids'],[1,130073]),
         'Exact stop union differs')
    return manifest


def load(capture,expected):
    raw=producer.captured(capture/'manifest.json',expected,2*1024**2)
    manifest=document(raw);need(set(manifest.get('files',{}))==FILES,'Fixed complete input filenames required')
    blobs={'manifest.json':raw}
    for name in FILES:
        pin=manifest['files'][name]
        need(type(pin) is dict and sha(pin.get('sha256')),'Typed input digest required')
        blobs[name]=producer.captured(capture/name,pin['sha256'],2*1024**2)
    return blobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['environment_root','candidate_assets','stock_metadata','contract','contract_sha256',
                 'runtime_manifest','runtime_sha256','capture','manifest_sha256','fresh_seal']:
        parser.add_argument('--'+name.replace('_','-'),required=True)
    args=parser.parse_args();root=Path(args.environment_root).resolve();producer.codec.bind_environment(root)
    frozen=producer.contract(Path(args.contract),args.contract_sha256)
    selected=producer.sources()
    admitted,endpoint=producer.parsers.admit(Path(args.runtime_manifest),args.runtime_sha256,selected)
    blobs=load(Path(args.capture),args.manifest_sha256);manifest=validate(blobs,frozen,args.contract_sha256)
    reconstructed,again=producer.prepare(root,Path(args.candidate_assets),Path(args.stock_metadata),frozen,args.contract_sha256)
    need(blobs==reconstructed and same(manifest,again),'Complete source-recipe byte replay differs')
    need(producer.sources()==selected,'Replay source/runtime inventory changed')
    final=endpoint()
    result=dict(format='toklet-minicpm5-quality-input-replay-v1',passed=True,
        manifest_sha256=args.manifest_sha256,loss_targets=10000,task_cases=100,capacity=manifest['capacity'],
        payload_sha256={name:digest(raw) for name,raw in blobs.items()},
        admitted_runtime=admitted,runtime_endpoint=final,complete_source_recipe_replayed=True,
        independent_numerical_or_native_codec_executed=False,model_executed=False,quality_scored=False,
        generated_code_executed=False,managed_adoption_qualified=False)
    with Path(args.fresh_seal).open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['admitted_runtime','runtime_endpoint','payload_sha256']}))


if __name__=='__main__':main()
