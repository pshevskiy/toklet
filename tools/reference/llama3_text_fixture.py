"""Prospective metadata-only Llama3 plain-text fixture producer; no model load.

Pinned original Jinja is used only as an independent offline reference. The app
must use its own bounded renderer and never execute remote template code.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sysconfig

ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/'models/MLX-Llama-3.2-1B-Instruct-8bit-d48cdf0'
REPO='mlx-community/Llama-3.2-1B-Instruct-8bit'
REVISION='d48cdf0a4ea22d893b7c63a99d6a693e24822795'
PINS={'tokenizer.json':'79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4',
      'tokenizer_config.json':'9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e',
      'special_tokens_map.json':'6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec'}
VERSIONS={'jinja2':'3.1.6','tokenizers':'0.22.2'}
DATE='26 Jul 2024'
SPECIAL={'<|begin_of_text|>':128000,'<|end_of_text|>':128001,
         '<|start_header_id|>':128006,'<|end_header_id|>':128007,
         '<|eom_id|>':128008,'<|eot_id|>':128009}
STOPS=[128001,128008,128009]
CASES={
 'single':[{'role':'user','content':'Hello!'}],
 'multi':[{'role':'user','content':'Count to two.'},{'role':'assistant','content':'One, two.'},
          {'role':'user','content':'Теперь по-русски 👋'}],
 'trim':[{'role':'user','content':' \t café é\n\n'}],
 'empty':[{'role':'user','content':''}],
 'python_trim':[{'role':'user','content':'\x1c\u00a0 X \u2003\x1f'}],
}
RAW_TEXTS=['Hello, world!',' a  b\n','Привет 👋 café é','\t x \n']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def render(messages,reserved):
    if not isinstance(messages,list) or not messages or len(messages)>63 or len(messages)%2!=1:
        raise ValueError('Conversation must end in user')
    if len(json.dumps(messages,ensure_ascii=False).encode('utf-8'))>65536:
        raise ValueError('Conversation JSON exceeds64KiB')
    text='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
    text+='Cutting Knowledge Date: December 2023\nToday Date: '+DATE+'\n\n<|eot_id|>'
    for i,message in enumerate(messages):
        role='user' if i%2==0 else 'assistant'
        if (not isinstance(message,dict) or set(message)!={'role','content'} or message['role']!=role
                or not isinstance(message['content'],str) or any(x in message['content'] for x in reserved)):
            raise ValueError('Unsupported message or reserved literal')
        text+='<|start_header_id|>'+role+'<|end_header_id|>\n\n'+message['content'].strip()+'<|eot_id|>'
    text+='<|start_header_id|>assistant<|end_header_id|>\n\n'
    if len(text.encode('utf-8'))>65536:raise ValueError('Rendered conversation exceeds64KiB')
    return text


def closure():
    site=Path(sysconfig.get_path('purelib')).resolve()
    if not site.is_relative_to(ROOT):raise ValueError('Repository runtime required')
    paths=[]
    for name in ['jinja2','markupsafe','tokenizers']:
        folder=site/name
        if not folder.is_dir():raise ValueError('Package missing: '+name)
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.so','.dylib'])
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def export(output):
    if os.path.lexists(output):raise FileExistsError('Fresh output required')
    if any(importlib.metadata.version(k)!=v for k,v in VERSIONS.items()):raise ValueError('Runtime versions differ')
    assets={}
    for name,digest in PINS.items():
        with (MODEL/name).open('rb') as stream:raw=stream.read(16*1024**2+1)
        if len(raw)>16*1024**2 or hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Pinned asset differs: '+name)
        assets[name]=json.loads(raw)
    before=closure();producer=sha(Path(__file__))
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from tokenizers import Tokenizer
    from jinja2 import StrictUndefined
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    tokenizer=Tokenizer.from_str(json.dumps(assets['tokenizer.json']))
    config=assets['tokenizer_config.json']
    if config['bos_token']!='<|begin_of_text|>' or config['eos_token']!='<|eot_id|>' or config['clean_up_tokenization_spaces'] is not True:
        raise ValueError('Pinned text settings differ')
    if tokenizer.truncation is not None or tokenizer.padding is not None:raise ValueError('Unexpected token modification')
    for literal,token in SPECIAL.items():
        if tokenizer.token_to_id(literal)!=token:raise ValueError('Special mapping differs')
    reserved=[v['content'] for v in config['added_tokens_decoder'].values()]
    env=ImmutableSandboxedEnvironment(undefined=StrictUndefined,trim_blocks=True,lstrip_blocks=True)
    env.globals.clear()
    template=env.from_string(config['chat_template'])
    rows=[]
    for name,messages in CASES.items():
        expected=template.render(messages=messages,bos_token='<|begin_of_text|>',date_string=DATE,
                                 tools=None,tools_in_user_message=True,add_generation_prompt=True)
        manual=render(messages,reserved)
        if expected!=manual:raise ValueError('Independent template bytes differ: '+name)
        ids=tokenizer.encode(expected,add_special_tokens=False).ids
        duplicate=tokenizer.encode(expected,add_special_tokens=True).ids
        if not ids or ids[0]!=128000 or ids.count(128000)!=1 or duplicate[:2]!=[128000,128000] or duplicate.count(128000)!=2:
            raise ValueError('BOS ownership differs')
        if len(ids)>512 or any(not 0<=i<128256 for i in ids):raise ValueError('Text fixture exceeds admitted tokens')
        decoded=tokenizer.decode(ids,skip_special_tokens=False)
        if decoded!=expected:raise ValueError('Rendered byte-level decode differs')
        rows.append(dict(name=name,messages=messages,rendered=expected,rendered_sha256=hashlib.sha256(expected.encode()).hexdigest(),
                         ids=ids,decoded_with_specials=decoded,bos_with_special_processing=duplicate[:2]))
    raw_rows=[]
    for text in RAW_TEXTS:
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        decoded=tokenizer.decode(ids,skip_special_tokens=False)
        if decoded!=text:raise ValueError('Raw byte-level roundtrip differs')
        raw_rows.append(dict(text=text,ids=ids,decoded=decoded))
    stop_rows=[]
    for token in STOPS:
        literal=next(k for k,v in SPECIAL.items() if v==token)
        if tokenizer.encode(literal,add_special_tokens=False).ids!=[token]:raise ValueError('Stop encoding differs')
        decoded=tokenizer.decode([token],skip_special_tokens=True)
        if decoded!='':raise ValueError('Hidden stop decode differs')
        stop_rows.append(dict(id=token,literal=literal,decoded_skip_specials=decoded))
    if closure()!=before or sha(Path(__file__))!=producer:raise ValueError('Producer runtime changed')
    for name,digest in PINS.items():
        if sha(MODEL/name)!=digest:raise ValueError('Text asset changed')
    result=dict(format='toklet-llama3-plain-text-v1',repository=REPO,revision=REVISION,file_pins=PINS,
                producer_sha256=producer,package_pins=before,versions=VERSIONS,date_string=DATE,
                add_special_tokens=False,decode_policy='raw tokenizers ByteLevel; no Transformers cleanup transform',
                stop_ids=STOPS,cases=rows,raw_roundtrips=raw_rows,stop_decodes=stop_rows,
                model_executed=False,chat_integration_qualified=False,full_quality_qualified=False)
    with output.open('x') as stream:stream.write(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();export(args.output)
