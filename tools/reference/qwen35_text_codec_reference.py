"""Independent pinned Qwen3.5 bounded text witness; no model inference.

Original Jinja/Tokenizers run only after external text-runtime admission.
The EOS convention executes two unmodified pinned MLX-LM source statements;
that source-slice proof is separate from whole-upstream runtime qualification.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import signal
import sys
import sysconfig
from types import SimpleNamespace
import text_reference_runtime as runtime

ROOT=Path(__file__).resolve().parents[2]
REPO='avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit'
REVISION='fb98fbc59c71a567f3e23435386c9e93951261db'
MAX_TEXT=65536
MAX_TOKENS=65536
MAX_CAPTURE=1024*1024
PROCESS_LIMIT=1024**3
PINS={'tokenizer_config.json': {'bytes': 9432, 'sha256': '8fd4316758163961892a98841fe13c6354373a9080e13eaa86b1bff257f6ff9d'}, 'config.json': {'bytes': 3024, 'sha256': 'ac6f8df4f4c1fd9d2583996bdd7070ea801b703e8a2b36f6d4b673a94ea419b5'}, 'tokenizer.json': {'bytes': 19989343, 'sha256': '87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4'}, 'generation_config.json': {'bytes': 115, 'sha256': '0c35bb39fbaed1ac0656baabc4f4e9bda20214e12336d0e4e8755aac1f487c2e'}, 'chat_template.jinja': {'bytes': 7967, 'sha256': '19b43ed4a381c5ff680a33611685df18c9af5f64c381878f0614b7f8f2cc5755'}}
UPSTREAM_SHA='ecb52655e86ff6d6db6a657601ece11f195e85971482eb2d0f40c20de17eb4c5'


def sources():
    upstream=Path(sysconfig.get_path('purelib'))/'mlx_lm'
    return [Path(__file__).resolve(),ROOT/'tools/reference/test_qwen35_text_codec_reference.py',upstream/'tokenizer_utils.py',upstream/'utils.py',upstream/'_version.py']


def validate(messages,options,literals):
    if (type(options) is not dict or set(options)!={'enable_thinking','add_generation_prompt'}
            or type(options['enable_thinking']) is not bool or options['enable_thinking']
            or type(options['add_generation_prompt']) is not bool or not options['add_generation_prompt']):
        raise ValueError('Only explicit bounded non-thinking generation supported')
    if (type(messages) is not list or not 1<=len(messages)<=63 or len(messages)%2!=1
            or len(json.dumps(messages,ensure_ascii=False).encode())>MAX_TEXT):
        raise ValueError('Bounded alternating conversation required')
    for i,message in enumerate(messages):
        if (type(message) is not dict or set(message)!={'role','content'}
                or message['role']!=('user' if i%2==0 else 'assistant')
                or type(message['content']) is not str or not message['content'].strip()):
            raise ValueError('Only alternating nonempty text messages supported')
        if any(v in message['content'] for v in literals) or any(ord(c)<32 and c not in '\t\n\r' for c in message['content']):
            raise ValueError('Reserved literals and unsupported C0 controls rejected')


def eos_source_witness(named,declared):
    path=Path(sysconfig.get_path('purelib'))/'mlx_lm/tokenizer_utils.py'
    raw=runtime.common.captured(path,23770,UPSTREAM_SHA)
    tree=ast.parse(raw)
    cls=next(v for v in tree.body if isinstance(v,ast.ClassDef) and v.name=='TokenizerWrapper')
    init=next(v for v in cls.body if isinstance(v,ast.FunctionDef) and v.name=='__init__')
    selected=[]
    for item in init.body:
        if isinstance(item,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='_eos_token_ids' for t in item.targets):selected.append(item)
        if isinstance(item,ast.If) and ast.unparse(item.test)=='tokenizer.eos_token_id is not None':selected.append(item)
    text='\n'.join(ast.unparse(v) for v in selected)
    expected='self._eos_token_ids = set(eos_token_ids or [])\nif tokenizer.eos_token_id is not None:\n    self._eos_token_ids.add(tokenizer.eos_token_id)'
    if text!=expected:raise ValueError('Pinned upstream EOS source slice differs')
    # The selected caller normalizes an integer to a list before construction.
    # Execute the exact two constructor statements, without importing MLX or
    # executing its model/runtime modules in this closed text-only producer.
    owner=SimpleNamespace()
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),
         dict(__builtins__={'set':set},self=owner,eos_token_ids=[declared],tokenizer=SimpleNamespace(eos_token_id=named)))
    return dict(stop_ids=sorted(owner._eos_token_ids),source_sha256=UPSTREAM_SHA,
        source_slice_sha256=hashlib.sha256(text.encode()).hexdigest(),scope='Unmodified source-slice EOS union, not whole upstream runtime qualification')


class Capture:
    def __init__(self,stream):self.stream=stream;self.count=0;self.bytes=0
    def add(self,value):
        raw=(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
        if len(raw)>MAX_TEXT or self.bytes+len(raw)>MAX_CAPTURE or self.count>=128:raise ValueError('Bounded text witness exceeded')
        if self.stream.write(raw)!=len(raw):raise OSError('Short witness write')
        self.stream.flush();self.count+=1;self.bytes+=len(raw)


def prefixes(tokenizer,ids):
    return {str(chunk):[dict(tokens=i,text=tokenizer.decode(ids[:i],skip_special_tokens=True))
        for i in sorted({0,len(ids),*range(chunk,len(ids),chunk)})] for chunk in [1,3,128]}


def produce(capture,assets):
    from tokenizers import Tokenizer
    from jinja2 import Environment,StrictUndefined
    data=json.loads(assets['tokenizer.json'],object_pairs_hook=runtime.common.json_unique)
    cfg=json.loads(assets['tokenizer_config.json']);model=json.loads(assets['config.json']);generation=json.loads(assets['generation_config.json'])
    template=assets['chat_template.jinja'].decode()
    if cfg['chat_template']!=template:raise ValueError('Embedded/standalone template must match')
    tokenizer=Tokenizer.from_buffer(assets['tokenizer.json'])
    named=tokenizer.token_to_id(cfg['eos_token']);declared=generation['eos_token_id']
    if model['eos_token_id']!=declared or model['text_config']['eos_token_id']!=declared:raise ValueError('Model/generation EOS declarations differ')
    stops=eos_source_witness(named,declared)
    if stops['stop_ids']!=[248044,248046] or (tokenizer.get_vocab_size(False),tokenizer.get_vocab_size())!=(248044,248077):raise ValueError('Exact tokenizer/stop geometry required')
    added=data['added_tokens'];literals=[v['content'] for v in added]
    capture.add(dict(kind='structure',base_vocab=248044,vocab=248077,model_padded_vocab=248320,merges=len(data['model']['merges']),added_tokens=added,
        normalizer=data['normalizer'],pre_tokenizer=data['pre_tokenizer'],post_processor=data['post_processor'],decoder=data['decoder'],
        declared_eos=declared,named_eos=named,**stops,
        maximum_token_bytes=max(len(token.encode()) for token in tokenizer.get_vocab()),
        maximum_decode_expansion_bytes=max(3*len(token) for token in tokenizer.get_vocab())))
    environment=Environment(undefined=StrictUndefined,trim_blocks=True,lstrip_blocks=True,loader=None,cache_size=0,extensions=['jinja2.ext.loopcontrols'])
    def raise_exception(message):raise ValueError(message)
    environment.globals['raise_exception']=raise_exception
    original=environment.from_string(template)
    options=dict(enable_thinking=False,add_generation_prompt=True)
    chats=[('hello',[dict(role='user',content='Hi')]),('trim',[dict(role='user',content='\t Hello \r\n')]),
        ('unicode',[dict(role='user',content='Привет 世界 🙂 e\u0301 é')]),
        ('multiturn',[dict(role='user',content='Hi'),dict(role='assistant',content='  Hello! \n'),dict(role='user',content='Next?')]),
        ('unicode_trim',[dict(role='user',content='\u00a0Hello\u3000'),dict(role='assistant',content='\u2003OK\u202f'),dict(role='user',content='Again')]),
        ('code',[dict(role='user',content='Write add(a,b).'),dict(role='assistant',content='```python\ndef add(a,b):\n    return a+b\n```'),dict(role='user',content='Explain.')])]
    for name,messages in chats:
        validate(messages,options,literals)
        rendered=original.render(messages=[dict(m,reasoning_content=None,tool_calls=None) for m in messages],tools=None,**options)
        if len(rendered.encode())>MAX_TEXT:raise ValueError('Rendered byte bound')
        ids=tokenizer.encode(rendered,add_special_tokens=False).ids
        capture.add(dict(kind='chat',name=name,messages=messages,options=options,rendered=rendered,ids=ids,
            add_special_tokens_ids=tokenizer.encode(rendered,add_special_tokens=True).ids,
            decode_keep=tokenizer.decode(ids,skip_special_tokens=False),decode_skip=tokenizer.decode(ids,skip_special_tokens=True),prefix_decodes=prefixes(tokenizer,ids)))
    for text in ['', ' ','  leading','trailing  ','a  b','a\tb\nc\r\nd','é e\u0301','Привет 世界 🙂','\ufffd','<think>work</think> <tool_call>call</tool_call>']:
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        capture.add(dict(kind='text',input=text,ids=ids,add_special_tokens_ids=tokenizer.encode(text,add_special_tokens=True).ids,
            decode_keep=tokenizer.decode(ids,skip_special_tokens=False),decode_skip=tokenizer.decode(ids,skip_special_tokens=True),prefix_decodes=prefixes(tokenizer,ids)))
    for item in added:
        ids=tokenizer.encode(item['content'],add_special_tokens=False).ids
        capture.add(dict(kind='control',metadata=item,ids=ids,resolved_id=tokenizer.token_to_id(item['content']),
            decode_keep=tokenizer.decode(ids,skip_special_tokens=False),decode_skip=tokenizer.decode(ids,skip_special_tokens=True)))
    # ByteLevel has one base-vocabulary spelling for every byte. Original
    # tokenizer decoding is the oracle, including its invalid/incomplete runs.
    values=list(range(ord('!'),ord('~')+1))+list(range(0xa1,0xad))+list(range(0xae,0x100));mapping={v:chr(v) for v in values};next_char=256
    for value in range(256):
        if value not in mapping:mapping[value]=chr(next_char);next_char+=1
    for name,values in [('two',[0xc3,0xa9]),('three',[0xe2,0x82,0xac]),('four',[0xf0,0x9f,0x99,0x82]),('incomplete',[0xe2,0x82]),('invalid',[0xff]),('valid_then_invalid',[0xe2,0x82,0xac,0xff])]:
        ids=[tokenizer.token_to_id(mapping[value]) for value in values]
        if any(v is None for v in ids):raise ValueError('Byte alphabet missing')
        capture.add(dict(kind='bytes',name=name,ids=ids,decode_keep=tokenizer.decode(ids,skip_special_tokens=False),decode_skip=tokenizer.decode(ids,skip_special_tokens=True),prefix_decodes=prefixes(tokenizer,ids)))
    one=[dict(role='user',content='Hi')]
    cases=[('empty',[],options),('system',[dict(role='system',content='Hi')],options),('assistant_end',one+[dict(role='assistant',content='Hi')],options),
        ('array',[dict(role='user',content=[])],options),('reasoning',[dict(one[0],reasoning_content='secret')],options),
        ('tool',[dict(role='tool',content='x')],options),('C0',[dict(role='user',content='\u001cHi')],options),
        ('thinking',one,dict(options,enable_thinking=True)),('no_generation_prompt',one,dict(options,add_generation_prompt=False)),('tools',one,dict(options,tools=[]))]
    cases.extend(('literal_'+str(v['id']),[dict(role='user',content=v['content'])],options) for v in added)
    for name,messages,settings in cases:
        try:validate(messages,settings,literals)
        except ValueError:error=True
        else:raise ValueError('Invalid bounded-profile case accepted')
        capture.add(dict(kind='rejection',name=name,messages=messages,options=settings,rejected=error))
    capture.add(dict(kind='padding',ids=[248077,248319],upstream_decode=tokenizer.decode([248077,248319],skip_special_tokens=True),native_policy='Explicit invalid-ID rejection before decode/stream; no numerical logit masking'))


def run(model,output,admission,endpoint):
    output.mkdir();capture=None;error=None;end_errors=[]
    assets={}
    try:
        assets={n:runtime.common.captured(model/n,v['bytes'],v['sha256']) for n,v in PINS.items()}
        with (output/'records.jsonl').open('xb') as stream:
            capture=Capture(stream);produce(capture,assets)
    except BaseException as failure:error=type(failure).__name__+': '+str(failure)
    finally:
        try:origins=endpoint()
        except BaseException as failure:origins=None;end_errors.append(str(failure))
        for n,v in PINS.items():
            try:runtime.common.captured(model/n,v['bytes'],v['sha256'])
            except BaseException as failure:end_errors.append(n+': '+str(failure))
        raw=output/'records.jsonl'
        report=dict(format='toklet-qwen35-text-reference-v1',passed=error is None and not end_errors,
            primary_error=error,endpoint_errors=end_errors,runtime_admission=admission,runtime_endpoint_origins=origins,
            repository=REPO,revision=REVISION,assets=PINS,records=capture.count if capture else 0,
            bytes=raw.stat().st_size if raw.exists() else 0,sha256=hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else None,
            process_limit_bytes=PROCESS_LIMIT,model_inference_executed=False,codec_native_executed=False,streaming_qualified=False,quality_qualified=False,managed_adoption_qualified=False)
        (output/'observation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','primary_error','endpoint_errors','records','bytes']}))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True);parser.add_argument('--expected-runtime-sha256',required=True)
    args=parser.parse_args();admission,endpoint=runtime.admit(args.runtime_manifest,args.expected_runtime_sha256,sources())
    sys.exit(run(args.model,args.output,admission,endpoint))
