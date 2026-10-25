"""Pinned upstream text-runtime witness before a native Gemma text profile.

Only the authenticated local Jinja text executes, with no template loader or
repository Python. Model inference, app admission and streaming are separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

import text_reference_runtime as runtime

ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/'models/MLX-Gemma4-E2B-4bit-2387675'
REPO='mlx-community/gemma-4-e2b-it-4bit'
REVISION='238767527555cb75a05732a84dff5d6ba0dd6809'
PINS={
    'chat_template.jinja':(17336,'2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082'),
    'config.json':(6395,'6397cb6eca41b911d1dcab74e17941351057bd759284052a2331918ff6f9246c'),
    'generation_config.json':(208,'d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de'),
    'tokenizer.json':(32169626,'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'),
    'tokenizer_config.json':(2740,'080d9e1aff284e2f6043889cd05367966f7c7b80e025fbc0b06745e218158656')}
MAX_TEXT=64*1024
MAX_CAPTURE=1024*1024
MAX_RECORDS=128
PROCESS_ALLOWANCE=1024**3


def sources():
    return [Path(__file__).resolve(),ROOT/'tools/reference/test_gemma4_text_codec_reference.py']


def validate(messages, options, literals):
    if options != {'enable_thinking':False,'add_generation_prompt':True} or any(type(v) is not bool for v in options.values()):
        raise ValueError('Only basic non-thinking generation prompts supported')
    if (type(messages) is not list or not 1<=len(messages)<=63 or len(messages)%2!=1
            or len(json.dumps(messages,ensure_ascii=False).encode())>MAX_TEXT):
        raise ValueError('Bounded nonempty alternating conversation required')
    for i,message in enumerate(messages):
        if (type(message) is not dict or set(message)!={'role','content'}
                or message['role']!=('user' if i%2==0 else 'assistant')
                or type(message['content']) is not str or not message['content'].strip()):
            raise ValueError('Alternating nonempty string messages required')
        content=message['content']
        if any(t in content for t in literals) or any(ord(c)<32 and c not in '\t\n\r' for c in content):
            raise ValueError('Control literals/unsupported control characters rejected')


class Capture:
    def __init__(self,stream):self.stream=stream;self.bytes=0;self.records=[]
    def add(self,value):
        raw=(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
        if len(raw)>MAX_TEXT or self.bytes+len(raw)>MAX_CAPTURE or len(self.records)>=MAX_RECORDS:
            raise ValueError('Text witness capture exceeds fixed bounds')
        item=dict(offset=self.bytes,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),completed=False)
        self.records.append(item)
        if self.stream.write(raw)!=len(raw):raise OSError('Short text witness write')
        self.stream.flush();self.bytes+=len(raw);item['completed']=True


def render(template,messages):
    # Only the fixed, already validated subset reaches the pinned template.
    # Empty tool_calls is supplied explicitly for its StrictUndefined branch.
    return template.render(messages=[dict(m,tool_calls=[]) for m in messages],tools=[],
        enable_thinking=False,add_generation_prompt=True,bos_token='<bos>')


def prefix_decodes(tokenizer,ids):
    return {str(chunk):[dict(tokens=i,text=tokenizer.decode(ids[:i],skip_special_tokens=True))
        for i in [*range(chunk,len(ids),chunk),len(ids)]] for chunk in (1,3,128)}


def produce(capture,assets):
    from tokenizers import Tokenizer
    from jinja2 import Environment, StrictUndefined
    metadata=json.loads(assets['tokenizer.json'],object_pairs_hook=runtime.common.json_unique)
    shape=dict(vocab=len(metadata['model']['vocab']),merges=len(metadata['model']['merges']),
        normalizer=metadata['normalizer'],pre_tokenizer=metadata['pre_tokenizer'],
        post_processor=metadata['post_processor'],decoder=metadata['decoder'])
    added=metadata['added_tokens'];del metadata
    capture.add(dict(kind='structure',**shape,added_tokens=added))
    if shape['vocab']!=262144 or shape['merges']!=514906 or len(added)!=24:
        raise ValueError('Pinned tokenizer inventory differs')
    tokenizer=Tokenizer.from_buffer(assets['tokenizer.json'])
    environment=Environment(undefined=StrictUndefined,trim_blocks=True,lstrip_blocks=True,
                            autoescape=False,loader=None,cache_size=0)
    template=environment.from_string(assets['chat_template.jinja'].decode())
    literals=[t['content'] for t in added]
    chats=[
        ('hello',[{'role':'user','content':'Hi'}]),
        ('trim',[{'role':'user','content':'\t Hello \r\n'}]),
        ('unicode',[{'role':'user','content':'Привет 世界 🙂 e\u0301 é'}]),
        ('literal_underline',[{'role':'user','content':'▁Hello'}]),
        ('multiturn',[{'role':'user','content':'Hi'},{'role':'assistant','content':'  Hello! \n'},
                      {'role':'user','content':'Next?'}]),
        ('unicode_trim',[{'role':'user','content':'\u00a0Hello\u3000'},{'role':'assistant','content':'\u2003OK\u202f'},
                         {'role':'user','content':'Again'}])]
    options=dict(enable_thinking=False,add_generation_prompt=True)
    for name,messages in chats:
        validate(messages,options,literals);text=render(template,messages)
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        capture.add(dict(kind='chat',name=name,messages=messages,options=options,
            rendered=text,rendered_utf8_sha256=hashlib.sha256(text.encode()).hexdigest(),ids=ids,
            add_special_tokens_ids=tokenizer.encode(text,add_special_tokens=True).ids,
            decode_keep_special=tokenizer.decode(ids,skip_special_tokens=False),
            decode_skip_special=tokenizer.decode(ids,skip_special_tokens=True),
            prefix_decodes=prefix_decodes(tokenizer,ids)))
    one=[{'role':'user','content':'Hi'}]
    rejected=[('system',[{'role':'system','content':'Hi'}],options),
        ('content_array',[{'role':'user','content':[]}],options),
        ('assistant_end',[*one,{'role':'assistant','content':'Hi'}],options),
        ('reasoning',[dict(one[0],reasoning='secret')],options),
        ('tool_turn',[{'role':'tool','content':'Hi'}],options),
        ('control_character',[{'role':'user','content':'\u001cHi'}],options),
        ('thinking',one,dict(options,enable_thinking=True)),
        ('no_generation_prompt',one,dict(options,add_generation_prompt=False)),
        ('tools',one,dict(options,tools=[]))]
    rejected.extend(('literal_'+str(t['id']),[{'role':'user','content':t['content']}],options) for t in added)
    for name,messages,settings in rejected:
        error=None
        try:validate(messages,settings,literals)
        except ValueError as failure:error=str(failure)
        capture.add(dict(kind='rejection',name=name,messages=messages,options=settings,rejected=error is not None,error=error))
        if error is None:raise ValueError('Unsupported text request was accepted')
    for text in ('',' ','  leading','trailing  ','a  b','a\tb\nc\r\nd','▁','a▁b','é e\u0301','Привет 世界 🙂','\ufffd'):
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        capture.add(dict(kind='text',input=text,normalized=tokenizer.normalizer.normalize_str(text),
            ids=ids,add_special_tokens_ids=tokenizer.encode(text,add_special_tokens=True).ids,
            decode=tokenizer.decode(ids,skip_special_tokens=True),prefix_decodes=prefix_decodes(tokenizer,ids)))
    for token in added:
        literal=token['content'];ids=tokenizer.encode(literal,add_special_tokens=False).ids
        capture.add(dict(kind='control',metadata=token,resolved_id=tokenizer.token_to_id(literal),ids=ids,
            decode_keep=tokenizer.decode(ids,skip_special_tokens=False),decode_skip=tokenizer.decode(ids,skip_special_tokens=True)))
    byte_cases=[('two',[0xc3,0xa9]),('three',[0xe2,0x82,0xac]),('four',[0xf0,0x9f,0x99,0x82]),
        ('incomplete',[0xe2,0x82]),('invalid',[0xff]),('valid_then_invalid',[0xe2,0x82,0xac,0xff])]
    for name,values in byte_cases:
        ids=[tokenizer.token_to_id('<0x'+format(v,'02X')+'>') for v in values]
        if any(v is None for v in ids):raise ValueError('Byte fallback inventory absent')
        variants=[('raw',ids),('special_between',ids[:1]+[1]+ids[1:]),
                  ('normal_after',ids+tokenizer.encode('x',add_special_tokens=False).ids)]
        for suffix,sequence in variants:
            capture.add(dict(kind='fallback',name=name+'_'+suffix,byte_values=values,ids=sequence,
                tokens=[tokenizer.id_to_token(v) for v in sequence],
                decode=tokenizer.decode(sequence,skip_special_tokens=True),prefix_decodes=prefix_decodes(tokenizer,sequence)))
    generation=json.loads(assets['generation_config.json']);config=json.loads(assets['config.json'])
    stops=generation['eos_token_id']
    capture.add(dict(kind='stop_policy',selected_artifact_stops=stops,
        wrapper_stops=config['eos_token_id'],nested_text_stop=config['text_config']['eos_token_id'],
        identities=[dict(id=v,token=tokenizer.id_to_token(v),skip_decode=tokenizer.decode([v],skip_special_tokens=True)) for v in stops],
        generation_defaults_are_sampling=generation['do_sample'],generation_defaults_qualified=False))
    if stops!=[1,106,50]:raise ValueError('Pinned selected-artifact stops differ')
    module=template.make_module(dict(messages=[dict(one[0],tool_calls=[])],tools=[],enable_thinking=False,
                                     add_generation_prompt=True,bos_token='<bos>'))
    for value in (' \tvisible\n','\u001cvisible\u001f','a<|channel>thought\nhidden<channel|>b'):
        capture.add(dict(kind='upstream_trim_diagnostic',input=value,
            strip_thinking=str(module.strip_thinking(value)),
            admitted_basic_subset=not any(t in value for t in literals) and not any(ord(c)<32 and c not in '\t\n\r' for c in value)))


def run(output,admission,endpoint):
    output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('New artifact output required')
    output.mkdir();stream=None;capture=None
    record=dict(format='toklet-gemma4-text-codec-reference-v1',passed=False,repository=REPO,revision=REVISION,
        runtime_admission=admission,primary_error=None,endpoint_errors=[],process_allowance_bytes=PROCESS_ALLOWANCE,
        native_text_profile_executed=False,native_model_executed=False,append_only_streaming_qualified=False,
        quality_qualified=False,device_fit_qualified=False,app_eligibility_changed=False)
    def interrupted(number,_):
        for v in (signal.SIGTERM,signal.SIGINT):signal.signal(v,signal.SIG_IGN)
        raise InterruptedError('Received signal '+str(number))
    old={v:signal.signal(v,interrupted) for v in (signal.SIGTERM,signal.SIGINT)}
    try:
        assets={name:runtime.common.captured(MODEL/name,size,sha) for name,(size,sha) in PINS.items()}
        stream=(output/'records.jsonl').open('xb');capture=Capture(stream)
        produce(capture,assets)
    except BaseException as error:record['primary_error']=type(error).__name__+': '+str(error)
    finally:
        for v in old:signal.signal(v,signal.SIG_IGN)
        try:record['runtime_endpoint_origins']=endpoint()
        except BaseException as error:record['endpoint_errors'].append('runtime: '+type(error).__name__+': '+str(error))
        for name,(size,sha) in PINS.items():
            try:runtime.common.captured(MODEL/name,size,sha)
            except BaseException as error:record['endpoint_errors'].append(name+': '+type(error).__name__+': '+str(error))
        try:
            if stream is not None:stream.close()
        except BaseException as error:record['endpoint_errors'].append('close: '+type(error).__name__+': '+str(error))
        record['records']=capture.records if capture else []
        path=output/'records.jsonl'
        try:
            if path.exists():
                with path.open('rb') as source:raw=source.read(MAX_CAPTURE+1)
                if len(raw)>MAX_CAPTURE:raise ValueError('Partial text capture exceeds bound')
                record['raw']=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        except BaseException as error:record['endpoint_errors'].append('raw capture: '+type(error).__name__+': '+str(error))
        record['passed']=record['primary_error'] is None and not record['endpoint_errors']
        try:
            with (output/'observation.json').open('x') as file:json.dump(record,file,indent=2,allow_nan=False);file.write('\n')
        finally:
            for v,handler in old.items():signal.signal(v,handler)
    print(json.dumps(dict(passed=record['passed'],records=len(record['records']),primary_error=record['primary_error'],endpoint_errors=record['endpoint_errors'])))
    return 0 if record['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True);parser.add_argument('--expected-runtime-sha256',required=True)
    args=parser.parse_args()
    admission,endpoint=runtime.admit(args.runtime_manifest,args.expected_runtime_sha256,sources())
    sys.exit(run(args.output,admission,endpoint))
