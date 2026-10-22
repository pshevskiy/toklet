"""Stage immutable raw-ID MLX diagnostic jobs for Mac and iOS; no inference/downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = 'tests/fixtures/mlx-qwen-real-v1/manifest.json'
REFERENCE_SHA = 'fe4fbeeb2866e99ac3b19d838491879a70b016ffcad071c0ff093db38d025d9b'
REFERENCE8 = 'tests/fixtures/mlx-qwen-real8-v1/manifest.json'
REFERENCE8_SHA = '1f98482bceba6c5627767d50cd2c4bda461d97174ee6e29346c5223215270531'
CONTRACT = 'benchmarks/contract-v1/manifest.json'
CONTRACT_SHA = '3f09dcf70f6d9620ce5172dcbaa6428f3487a3edf412f2bfce57937faab72c66'
CORPUS = 'benchmarks/contract-v1/held_out_loss.u32le'
CORPUS_SHA = 'f53c5b0c446959164c4d2265166040123f95be5b6d9eec784895fb45a8215beb'
TASKS = 'artifacts/quality-v1/tasks.jsonl'
TASKS_SHA = 'b096b8eb40960f38ae0e9f3268484eed10bf2a6bebbc981a45a061ee461aab7c'
TINY = {'config.json':'0c60668f247d6584a76f06c8a536cf3e8c4be0ff17c352e58c7d122f1c18ce5e',
        'model.safetensors':'ff728944657c95bc20f31e929663b4a728e13a056312e072962bb2867c6fb4be'}
WORKLOADS = ['measurement-short','measurement-128','measurement-512','measurement-1024','measurement-sustained',
             'quality-short','quality-full','tasks-short','tasks-full','state-component','state-512']


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def captured(name, expected, maximum):
    with (ROOT/name).open('rb') as stream:data=stream.read(maximum+1)
    if len(data)>maximum or hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Frozen input differs: '+name)
    return data


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def frozen_prompt(count,pins):
    contract=json.loads(captured(CONTRACT,CONTRACT_SHA,256*1024));item=contract['arrays']['prompt_'+str(count)]
    if item['count']!=count or item['dtype']!='u32le' or item['file']!='prompt_'+str(count)+'.u32le':
        raise ValueError('Frozen prompt schema differs')
    name='benchmarks/contract-v1/'+item['file'];raw=captured(name,item['sha256'],count*4)
    if len(raw)!=count*4:raise ValueError('Exact prompt size differs')
    pins.update({CONTRACT:CONTRACT_SHA,name:item['sha256']})
    return list(struct.unpack('<'+str(count)+'I',raw))


def job(model, workload):
    if model not in ['tiny','qwen','qwen8'] or workload not in WORKLOADS:raise ValueError('Unsupported explicit MLX job')
    if model == 'tiny' and workload not in ['measurement-short','quality-short','tasks-short','state-component']:
        raise ValueError('Tiny is component scope only')
    reference_name,reference_sha=(REFERENCE8,REFERENCE8_SHA) if model=='qwen8' else (REFERENCE,REFERENCE_SHA)
    reference=json.loads(captured(reference_name,reference_sha,256*1024));pins={reference_name:reference_sha}
    source={'tiny':'tests/fixtures/mlx-qwen-v1/model',
            'qwen':'models/MLX-Qwen2.5-0.5B-Instruct-4bit-a5339a4',
            'qwen8':'models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c'}[model]
    identity=TINY if model=='tiny' else reference['model']['file_pins']
    folder='M8Inputs/'+model+'/Model'
    host={folder+'/'+name:source+'/'+name for name in identity}
    files={folder+'/'+name:pin for name,pin in identity.items()}
    pins.update({host[name]:pin for name,pin in files.items()})
    config=dict(version=2,mlx=True,modelFolder=folder,packedFolder='',files=files,metal=True,
                numericalBudget=(256 if model=='tiny' else 1536)*1024**2,
                processBudget=(512 if model=='tiny' else 2560)*1024**2)
    prompt=[1,7,13,3] if model=='tiny' else reference['greedy']['prompt']
    payloads={}
    if workload.startswith('measurement'):
        mode='measurement';short=workload=='measurement-short'
        if not short:
            count=512 if workload=='measurement-sustained' else int(workload.rsplit('-',1)[1])
            prompt=frozen_prompt(count,pins)
        config.update(capacity=32 if short else 1152,chunk=3 if short else 128,prompt=prompt,maximum=4 if short else 128,
                      expectedPrefix=([47]*4 if model=='tiny' else reference['greedy']['fp64_tokens']) if short else [],
                      warmups=3,repetitions=10,minimumNativeSeconds=600 if workload=='measurement-sustained' else 0)
    elif workload.startswith('quality'):
        mode='quality';short=workload=='quality-short'
        if model=='tiny':raw=struct.pack('<4I',*prompt)
        else:
            raw=captured(CORPUS,CORPUS_SHA,40004)
            if len(raw)!=40004:raise ValueError('Full corpus size differs')
            pins[CORPUS]=CORPUS_SHA
            if short:raw=raw[:16]
        digest=hashlib.sha256(raw).hexdigest();name='M8Corpora/'+digest+'/tokens.u32le';payloads[name]=raw
        block=3 if short else 1024
        config.update(capacity=block+1,chunk=1,tokensFile=name,tokensSHA256=digest,targets=len(raw)//4-1,blockSize=block)
    elif workload.startswith('tasks'):
        mode='tasks';short=workload=='tasks-short'
        if model=='tiny':
            cases=[dict(id='first',promptTokens=[1,7,13],maximum=3),dict(id='second',promptTokens=prompt,maximum=2)]
        else:
            raw=captured(TASKS,TASKS_SHA,1024**2);original=[json.loads(line) for line in raw.decode().splitlines()]
            if len(original)!=100 or len({item['id'] for item in original})!=100:raise ValueError('Frozen100-task inventory differs')
            cases=[dict(id=item['id'],promptTokens=item['prompt_tokens'],maximum=min(8,item['max_output_tokens']) if short else item['max_output_tokens'])
                   for item in (original[:1] if short else original)]
            pins[TASKS]=TASKS_SHA
        data=dict(version=1,vocabulary=64 if model=='tiny' else 151936,stops=[62,63] if model=='tiny' else [151645,151643],cases=cases)
        raw=encoded(data);digest=hashlib.sha256(raw).hexdigest();name='M8Tasks/'+digest+'/tasks.json';payloads[name]=raw
        config.update(capacity=32 if model=='tiny' else 562,chunk=3 if model=='tiny' else 128,tasksFile=name,tasksSHA256=digest,caseCount=len(cases))
    elif workload=='state-512':
        mode='state'
        config.update(capacity=1152,chunk=128,prompt=frozen_prompt(512,pins),maximum=128,vocabulary=151936)
    else:
        mode='state-component'
        config.update(capacity=32,chunk=3,prompt=prompt,maximum=25,vocabulary=64 if model=='tiny' else 151936)
    return mode,config,host,pins,payloads


def stage(model, workload, output):
    output=Path(output).absolute()
    if os.path.lexists(output):raise FileExistsError('Never overwrite an existing trial')
    if not output.resolve().is_relative_to(ROOT/'artifacts'):raise ValueError('Repository artifacts required')
    mode,config,host,pins,payloads=job(model,workload)
    producers=[Path(__file__).resolve(),ROOT/'apple/tests/swift/MLXRealMain.swift']
    pins.update({str(path.relative_to(ROOT)):sha(path) for path in producers})
    for target,source in host.items():
        path=ROOT/source
        if path.is_symlink() or sha(path)!=config['files'][target]:raise ValueError('Selected artifact differs: '+source)
    output.mkdir();base=output/'transfer';base.mkdir()
    for target,source in host.items():
        path=base/target;path.parent.mkdir(parents=True,exist_ok=True);os.link(ROOT/source,path)
        if sha(path)!=config['files'][target]:raise ValueError('Staged hard link differs')
    for name,raw in payloads.items():
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
    raw=encoded(config);digest=hashlib.sha256(raw).hexdigest();name='M7Jobs/'+digest+'.json'
    path=base/name;path.parent.mkdir()
    with path.open('xb') as stream:stream.write(raw)
    for source,pin in pins.items():
        if sha(ROOT/source)!=pin:raise ValueError('Input endpoint changed: '+source)
    record=dict(format='toklet-m8-mlx-staged-job-v1',passed=True,model=model,workload=workload,mode=mode,config=config,
                bindings=pins,host=host,job=dict(relative_path=name,sha256=digest),
                staged_files={str(p.relative_to(base)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(base.rglob('*')) if p.is_file()},
                full_quality_qualified=False,physical_device_qualified=False,performance_qualified=False)
    with (output/'staged.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['tiny','qwen','qwen8'],required=True)
    parser.add_argument('--workload',choices=WORKLOADS,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();record=stage(args.model,args.workload,args.output)
    print(json.dumps(dict(passed=record['passed'],mode=record['mode'],job=record['job'])))
