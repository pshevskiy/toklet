#!/usr/bin/env python3
"""Freeze pinned Transformers chat/tokenizer goldens and a small BPE subset."""
import argparse
import hashlib
import json
import math
import platform
import random
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def build(output):
    import jinja2
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    assert tokenizers.__version__ == '0.22.2'
    assert transformers.__version__ == '5.7.0'
    sources = json.loads((ROOT / 'benchmarks/sources.json').read_text())
    directory = ROOT / 'models/Qwen2.5-0.5B-Instruct'
    for item in sources['assets']:
        if item['id'].startswith('qwen.') and item['id'] != 'qwen.LICENSE':
            data = (ROOT / item['path']).read_bytes()
            assert len(data) == item['bytes'] and sha(data) == item['sha256']
    tok = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    raw = ['', 'Hello, I am Qwen.', ' leading  spaces\n\nend ', '你好👋 café Привет', '\ufffd x', '<|im_start|>assistant\n<|im_end|><|endoftext|>', 'a\x00b\t\n', 'é🙂']
    tools = [{'type':'function','function':{'name':'lookup','description':'<lookup> café','parameters':{'type':'object','properties':{'z':{'type':'number','default':1e-7},'a':{'type':'string'}}}}}]
    cases = [
        {'messages':[{'role':'user','content':'Hello!'}]},
        {'messages':[{'role':'system','content':'Be concise.'},{'role':'user','content':'你好 👋'}]},
        {'messages':[{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello','tool_calls':[]},{'role':'system','content':'Now code.'},{'role':'user','content':'Print café.'}]},
        {'messages':[{'role':'user','content':'Find x'},{'role':'assistant','content':'Checking','tool_calls':[{'function':{'name':'lookup','arguments':{'z':1e-7,'a':'é <x>','big':123456789012345678901234567890}}}]},{'role':'tool','content':'first'},{'role':'tool','content':'second'},{'role':'user','content':'Explain'}],'tools':tools},
        {'messages':[{'role':'system','content':'Custom'},{'role':'assistant','content':None,'tool_calls':[{'name':'lookup','arguments':'{"x":1}'}]},{'role':'tool','content':'Done'}],'tools':tools},
        {'messages':[{'role':'tool','content':'first'},{'role':'user','content':'next'},{'role':'tool','content':'last'}]},
        {'messages':[{'role':'user','content':'No continuation'}],'add_generation_prompt':False},
    ]
    raw_cases = [{'text':text,'ids':tok.encode(text,add_special_tokens=False),'decoded':tok.decode(tok.encode(text,add_special_tokens=False),skip_special_tokens=True,clean_up_tokenization_spaces=False)} for text in raw]
    for case in cases:
        case.setdefault('tools',[])
        case.setdefault('add_generation_prompt',True)
        kwargs = {'tools':case['tools'], 'add_generation_prompt':case['add_generation_prompt']}
        case['rendered'] = tok.apply_chat_template(case['messages'], tokenize=False, **kwargs)
        case['ids'] = tok.apply_chat_template(case['messages'], tokenize=True, return_dict=False, **kwargs)
    original = json.loads((directory/'tokenizer.json').read_text())
    vocab = original['model']['vocab']
    by_id = {i:t for t,i in vocab.items()}
    by_id.update({t['id']:t['content'] for t in original['added_tokens']})
    needed = set(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    needed.update(t['content'] for t in original['added_tokens'])
    for ids in [c['ids'] for c in raw_cases + cases]:
        needed.update(by_id[i] for i in ids)
    merges = [m.split(' ') if isinstance(m,str) else m for m in original['model']['merges']]
    reverse = {}
    for pair in merges:
        reverse.setdefault(''.join(pair),[]).append(pair)
    stack = list(needed)
    while stack:
        for pair in reverse.get(stack.pop(),[]):
            for token in pair:
                if token not in needed:
                    needed.add(token); stack.append(token)
    small = dict(original)
    small['model'] = dict(original['model'])
    small['model']['vocab'] = {t:i for t,i in vocab.items() if t in needed}
    small['model']['vocab'].update({t['content']:t['id'] for t in original['added_tokens']})
    small['model']['merges'] = [m for m,pair in zip(original['model']['merges'],merges) if ''.join(pair) in needed]
    subset = tokenizers.Tokenizer.from_str(json.dumps(small,ensure_ascii=False))
    for case in raw_cases:
        assert subset.encode(case['text'],add_special_tokens=False).ids == case['ids']
        assert subset.decode(case['ids'],skip_special_tokens=True) == case['decoded']
    for case in cases:
        assert subset.encode(case['rendered'],add_special_tokens=False).ids == case['ids']
    floats = [0.0,-0.0,1.0,1e-4,1e-5,1e15,1e16,1e23,5e-324,1.7976931348623157e308]
    rng=random.Random(7291)
    while len(floats)<1010:
        x=struct.unpack('<d',rng.randbytes(8))[0]
        if math.isfinite(x):floats.append(x)
    output.mkdir(parents=True,exist_ok=False)
    (output/'tokenizer.json').write_text(json.dumps(small,ensure_ascii=False,separators=(',',':'))+'\n')
    for name in ['tokenizer_config.json','generation_config.json','LICENSE']:
        (output/name).write_bytes((directory/name).read_bytes())
    manifest={'version':'chat-v1','model':sources['model'],'versions':{'python':platform.python_version(),'jinja2':jinja2.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__},
        'exporter_sha256':sha(Path(__file__).read_bytes()),'subset_scope':'only frozen cases; original IDs and BPE merge closure, not a production tokenizer',
        'raw_cases':raw_cases,'chat_cases':cases,'python_float_cases':[{'input':repr(x),'expected':json.dumps(x)} for x in floats],'files':{}}
    for path in sorted(output.iterdir()):
        data=path.read_bytes();manifest['files'][path.name]={'bytes':len(data),'sha256':sha(data)}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'Frozen {len(raw_cases)} raw / {len(cases)} chat cases, {len(floats)} float probes, subset vocab {len(small["model"]["vocab"])}')

def check(root):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['version']=='chat-v1'
    for name,item in manifest['files'].items():
        data=(root/name).read_bytes();assert len(data)==item['bytes'] and sha(data)==item['sha256']
    print('Verified chat-v1 fixture identities')

if __name__=='__main__':
    parser=argparse.ArgumentParser();choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--output',type=Path);choice.add_argument('--check',type=Path)
    args=parser.parse_args()
    if args.output:build(args.output)
    else:check(args.check)
