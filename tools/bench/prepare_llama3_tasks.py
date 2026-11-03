"""Freeze shared Llama task IDs/reserves; never execute a model or generated code."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import task_cases
import acquire_llama3_original as original_pins

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/reference'))
import llama3_text_fixture as codec_policy
CAPACITY=566
STOPS=[128001,128008,128009]
MODELS=dict(original=dict(repository=original_pins.META,revision=original_pins.META_REVISION,weight_sha256=original_pins.WEIGHT_SHA),
            native=dict(repository=codec_policy.REPO,revision=codec_policy.REVISION,weight_sha256='cd3582a78766654e3e2107e3b4d51d9cb18e6754b659bfb44a34328ef9d7fa52'))
PREFIX='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def prompt(case,reserved):
    text=case['prompt_text']
    if case['category']=='continuation':return text
    if not text.startswith(PREFIX)or not text.endswith(SUFFIX):
        raise ValueError('Canonical task wrapper differs')
    return codec_policy.render([dict(role='user',content=text[len(PREFIX):-len(SUFFIX)])],reserved)


def validate_cases(cases):
    expected=[f'HumanEval/{i}'for i in range(40)]+[f'truthfulqa/row-{i}'for i in range(30)]+[f'lambada/row-{i}'for i in range(30)]
    if len(cases)!=100 or [c['id']for c in cases]!=expected or Counter(c['category']for c in cases)!=task_cases.CATEGORIES:
        raise ValueError('All100 frozen cases in source order required')
    for index,case in enumerate(cases):
        category='code'if index<40 else 'prose'if index<70 else 'continuation'
        ids=case['prompt_tokens'];cap=task_cases.CAPS[category]
        if (set(case)!={'id','category','prompt_tokens','max_output_tokens'}or case['category']!=category
                or type(case['max_output_tokens'])is not int or case['max_output_tokens']!=cap
                or not isinstance(ids,list)or not ids or len(ids)+cap>CAPACITY
                or any(type(i)is not int or not 0<=i<128256 for i in ids)):
            raise ValueError('Task IDs, reserves or capacity differ')


def derive(replay=False):
    if any(importlib.metadata.version(k)!=v for k,v in codec_policy.VERSIONS.items()):raise ValueError('Text runtime differs')
    before=codec_policy.closure()
    assets={}
    for name,digest in codec_policy.PINS.items():
        with(codec_policy.MODEL/name).open('rb')as f:raw=f.read(16*1024**2+1)
        if len(raw)>16*1024**2 or sha(raw)!=digest:raise ValueError('Pinned codec asset differs')
        assets[name]=raw
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_str(assets['tokenizer.json'].decode())if replay else Tokenizer.from_buffer(assets['tokenizer.json'])
    if tokenizer.padding is not None or tokenizer.truncation is not None:raise ValueError('Token modification forbidden')
    reserved=[v.content for v in tokenizer.get_added_tokens_decoder().values()]
    if len(reserved)!=256 or len(set(reserved))!=256:raise ValueError('Pinned reserved inventory differs')
    template=None
    if replay:
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        env=ImmutableSandboxedEnvironment(undefined=StrictUndefined,trim_blocks=True,lstrip_blocks=True)
        env.globals.clear();template=env.from_string(json.loads(assets['tokenizer_config.json'])['chat_template'])
    cases=[]
    for source in task_cases.load_cases():
        text=prompt(source,reserved)
        if replay and source['category']!='continuation':
            user=source['prompt_text'][len(PREFIX):-len(SUFFIX)]
            independent=template.render(messages=[dict(role='user',content=user)],bos_token='<|begin_of_text|>',date_string=codec_policy.DATE,tools=None,tools_in_user_message=True,add_generation_prompt=True)
            if independent!=text:raise ValueError('Independent pinned Jinja rendered bytes differ')
            if tokenizer.encode(independent,add_special_tokens=False).ids!=tokenizer.encode(text,add_special_tokens=False).ids:
                raise ValueError('Independent rendered token IDs differ')
            text=independent
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        if source['category']!='continuation'and (ids[0]!=128000 or ids.count(128000)!=1):raise ValueError('BOS ownership differs')
        cases.append(dict(id=source['id'],category=source['category'],prompt_tokens=ids,max_output_tokens=source['max_output_tokens']))
    validate_cases(cases)
    if len(cases[32]['prompt_tokens'])+cases[32]['max_output_tokens']!=566:raise ValueError('HumanEval/32 full reserve differs')
    if codec_policy.closure()!=before:raise ValueError('Text provider changed')
    for name,digest in codec_policy.PINS.items():
        if sha((codec_policy.MODEL/name).read_bytes())!=digest:raise ValueError('Codec asset changed')
    return cases


def validate_manifest(value):
    if (value['format']!='toklet-llama3-quality-tasks-v1' or value['source_tasks']['sha256']!=task_cases.TASK_SHA
            or value['models']!=MODELS or value['categories']!=task_cases.CATEGORIES or value['output_caps']!=task_cases.CAPS
            or value['capacity']!=566 or value['vocabulary']!=128256 or value['stop_ids']!=STOPS
            or value['tokenizer_pins']!=codec_policy.PINS or value['date_string']!=codec_policy.DATE
            or value['add_special_tokens']is not False or value['decode_policy']!='raw ByteLevel; skip special tokens; no Transformers cleanup'
            or value['sampling']!='greedy_lowest_token_id_on_ties'
            or value['pass_rate_loss_limit_percentage_points']!=2.0 or value['aggregation']!='100_equal_weight_binary_cases'
            or value['original_trajectory_policy']!='exact stateless visible/terminal-stop greedy replay before grading'
            or value['model_executed']is not False or value['tasks']['file']!='execution.jsonl'):
        raise ValueError('Frozen task policy differs')


def load(folder, expected_sha):
    folder=Path(folder)
    with(folder/'manifest.json').open('rb')as f:raw=f.read(65537)
    if len(raw)>65536 or sha(raw)!=expected_sha:raise ValueError('Task manifest differs')
    value=json.loads(raw);validate_manifest(value)
    with(folder/'execution.jsonl').open('rb')as f:body=f.read(4*1024**2+1)
    if len(body)>4*1024**2 or len(body)!=value['tasks']['bytes']or sha(body)!=value['tasks']['sha256']:
        raise ValueError('Frozen execution inputs differ')
    cases=[json.loads(line)for line in body.splitlines()];validate_cases(cases)
    if value['task_ids']!=[c['id']for c in cases]:raise ValueError('Frozen task order differs')
    return value,cases


def run(folder,replay=False,expected_sha=None):
    folder=Path(folder)
    if not replay and os.path.lexists(folder):raise FileExistsError('Fresh task input folder required')
    cases=derive(replay)
    body=''.join(json.dumps(c,separators=(',',':'))+'\n'for c in cases).encode()
    if replay:
        _,existing=load(folder,expected_sha)
        if existing!=cases:raise ValueError('Independent task token replay differs')
        return dict(replayed=True,model_executed=False,cases=100)
    source_path=ROOT/'artifacts/quality-v1/tasks.jsonl';source=source_path.read_bytes()
    if sha(source)!=task_cases.TASK_SHA:raise ValueError('Grading source differs')
    value=dict(format='toklet-llama3-quality-tasks-v1',models=MODELS,source_tasks=dict(path=str(source_path.relative_to(ROOT)),bytes=len(source),sha256=task_cases.TASK_SHA),
               categories=task_cases.CATEGORIES,output_caps=task_cases.CAPS,capacity=566,vocabulary=128256,stop_ids=STOPS,
               tokenizer_pins=codec_policy.PINS,date_string=codec_policy.DATE,add_special_tokens=False,
               decode_policy='raw ByteLevel; skip special tokens; no Transformers cleanup',
               sampling='greedy_lowest_token_id_on_ties',pass_rate_loss_limit_percentage_points=2.0,aggregation='100_equal_weight_binary_cases',
               preparation_versions=codec_policy.VERSIONS,preparation_provider_sha256=codec_policy.closure(),
               preparation_source_sha256={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in [Path(__file__),Path(task_cases.__file__),Path(codec_policy.__file__),ROOT/'tools/bench/prepare.py']},
               original_trajectory_policy='exact stateless visible/terminal-stop greedy replay before grading',
               task_ids=[c['id']for c in cases],tasks=dict(file='execution.jsonl',bytes=len(body),sha256=sha(body)),model_executed=False)
    validate_manifest(value);folder.mkdir();(folder/'execution.jsonl').write_bytes(body)
    (folder/'manifest.json').write_text(json.dumps(value,indent=2)+'\n')
    load(folder,sha((folder/'manifest.json').read_bytes()))
    return dict(prepared=True,model_executed=False,cases=100,maximum_required=max(len(c['prompt_tokens'])+c['max_output_tokens']for c in cases),manifest_sha256=sha((folder/'manifest.json').read_bytes()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',type=Path);group.add_argument('--replay',type=Path)
    parser.add_argument('--manifest-sha256');args=parser.parse_args()
    if args.replay and not args.manifest_sha256:parser.error('Replay requires external manifest SHA')
    print(json.dumps(run(args.prepare or args.replay,bool(args.replay),args.manifest_sha256),indent=2))
