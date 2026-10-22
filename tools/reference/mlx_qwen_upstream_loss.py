"""Explicit diagnostic upstream-runtime loss for one immutable public snapshot.

This is not a Toklet execution backend or a same-precision performance comparison.
No tokenizer, remote Python module, model download or generated program is used.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import sysconfig

import mlx_qwen_real as reference

ROOT=reference.ROOT
sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_apple_package as package
import m8_mlx_jobs as jobs
import native_usage

NATIVE='benchmarks/results/2026-10-22-native-mlx-full-loss-v1.json'
NATIVE_SHA='8a96442eada56cb1a4bf93274c91a682559a90011066df8d8dd8ad59b38c194f'
BASELINE='benchmarks/results/2026-10-11-held-out-scalar-loss-v1.json'
BASELINE_SHA='cae67d941141ee0be68ef358186afb37f29d84c9eac7f23b8eeaf252089e7cc3'
VERSIONS={'mlx':'0.32.3','mlx-lm':'0.32.0','numpy':'2.3.2'}
PROTECTED={'target/release/toklet':'5250694e814ea112148a52cfdeaf5bc6286ff2804b3acfd8f89e80bc73122a59',
           'target/release/examples/cpu_bench':'e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea'}


def loss(logits,target):
    """Independent stable FP64 log-sum-exp of exactly widened FP32 logits."""
    import numpy as np
    if (logits.dtype!=np.dtype('float32') or logits.ndim!=1 or logits.size==0
        or type(target) is not int or not 0<=target<len(logits) or not np.isfinite(logits).all()):
        raise ValueError('Finite one-dimensional FP32 logits and valid target required')
    wide=logits.astype(np.float64);greedy=int(np.argmax(wide));maximum=float(wide[greedy])
    # fsum of the positive exponentials avoids an order-dependent aggregate.
    total=math.fsum(np.exp(wide-maximum));value=maximum+math.log(total)-float(wide[target])
    if not math.isfinite(value) or value<0:raise ValueError('Invalid loss')
    return dict(target=target,greedy=greedy,target_logit=float(logits[target]),greedy_logit=float(logits[greedy]),negative_log_likelihood=value)


def blocks(targets=10000,size=1024):
    if type(targets) is not int or targets!=10000 or type(size) is not int or size!=1024:
        raise ValueError('Frozen full-corpus policy required')
    return [(start,min(start+size,targets)) for start in range(0,targets,size)]


def model_inventory(folder):
    matches=sorted(folder.glob('model*.safetensors'))
    if (folder.is_symlink() or not folder.is_dir() or matches!=[folder/'model.safetensors']
        or matches[0].is_symlink() or not matches[0].is_file()):
        raise ValueError('Exact upstream model safetensors inventory required')


def retained_journal(path):
    """Bind actual retained bytes, independently of attempted row writes."""
    if path.is_symlink():raise ValueError('Unexpected journal symlink')
    if not path.exists():return None
    digest=hashlib.sha256();size=0;lines=0;last=None
    with path.open('rb') as stream:
        while data:=stream.read(256*1024):
            size+=len(data)
            if size>32*1024**2:raise ValueError('Retained journal exceeds bound')
            digest.update(data);lines+=data.count(b'\n');last=data[-1]
    return dict(bytes=size,sha256=digest.hexdigest(),complete_lines=lines,trailing_bytes=bool(size and last!=10))


def producers():
    """Selected MLX/MLX-LM/NumPy source/native binaries, not all Python/OS code."""
    site=Path(sysconfig.get_path('purelib')).resolve()
    if not site.is_relative_to(ROOT):raise ValueError('Repository virtual environment required')
    files={Path(__file__).resolve(),Path(reference.__file__).resolve(),Path(package.__file__).resolve(),Path(jobs.__file__).resolve(),Path(native_usage.__file__).resolve()}
    for name in ['mlx','mlx_lm','numpy']:
        folder=site/name
        if not folder.is_dir():raise ValueError('Selected producer package absent')
        files.update(path for path in folder.rglob('*') if path.is_file() and '__pycache__' not in path.parts
                     and (name=='mlx' or path.suffix in ['.py','.so','.dylib']))
    # NumPy wheels may keep their linked libraries beside the Python package.
    for folder in site.glob('numpy*.libs'):
        files.update(path for path in folder.rglob('*') if path.is_file() and path.suffix in ['.so','.dylib'])
    return {str(path.relative_to(ROOT)):reference.sha(path) for path in sorted(files)}


def run(output):
    if os.path.lexists(output):raise FileExistsError('Never overwrite an upstream trial')
    output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Repository output required')
    output.mkdir();pins=producers();pins.update(PROTECTED);primary=None;errors=[];journal_errors=[];retained=None;samples=[];summary=None;comparison=None;dtypes=set();count=0;raw_digest=hashlib.sha256();losses=[];differences=[];target_differences=[];greedy_differences=0
    def merge(extra):
        for name,pin in extra.items():
            if name in pins and pins[name]!=pin:raise ValueError('Conflicting admission: '+name)
            pins[name]=pin
    def guard():
        model_inventory(reference.MODEL)
        for name,pin in pins.items():
            if reference.sha(ROOT/name)!=pin:raise ValueError('Endpoint differs: '+name)
    def captured(name,pin,maximum=32*1024**2):
        with (ROOT/name).open('rb') as stream:data=stream.read(maximum+1)
        if len(data)>maximum or hashlib.sha256(data).hexdigest()!=pin:raise ValueError('Consumed bytes differ: '+name)
        merge({name:pin});return data
    def metadata(name,pin):return json.loads(captured(name,pin))
    previous=signal.signal(signal.SIGTERM,package.interrupted)
    try:
        model_inventory(reference.MODEL)
        for name,expected in VERSIONS.items():
            if importlib.metadata.version(name)!=expected:raise ValueError('Selected runtime version differs: '+name)
        for name,pin in reference.PINS.items():merge({str((reference.MODEL/name).relative_to(ROOT)):pin})
        config=metadata(str((reference.MODEL/'config.json').relative_to(ROOT)),reference.PINS['config.json'])
        if config['model_type']!='qwen2' or config['quantization']!={'bits':4,'group_size':64} or 'model_file' in config:raise ValueError('Selected static Qwen2 source required')
        corpus=captured(jobs.CORPUS,jobs.CORPUS_SHA,40004)
        if len(corpus)!=40004:raise ValueError('Exact full corpus required')
        tokens=list(struct.unpack('<10001I',corpus))
        contract=metadata(jobs.CONTRACT,jobs.CONTRACT_SHA);limit=contract['quality']['mean_nll_increase_limit_nats_per_token']
        baseline=metadata(BASELINE,BASELINE_SHA);baseline_mean=baseline['runs'][0]['mean_negative_log_likelihood']
        if limit!=0.05 or baseline_mean!=2.883725149929484:raise ValueError('Frozen comparison policy differs')
        native=metadata(NATIVE,NATIVE_SHA)
        if not native['passed'] or native['execution']['loss_gates']['mac']['passed'] is not False:raise ValueError('Completed rejected native experiment required')
        raw_name='artifacts/m8-full-loss-driver-v1/mac-quality-full.stdout';native_raw=captured(raw_name,native['execution']['source_pins'][raw_name],64*1024**2)
        native_rows=[json.loads(line.split('=',1)[1])['target'] for line in native_raw.decode().splitlines() if line.startswith('TOKLET_QUALITY_SAMPLE_JSON=')]
        if len(native_rows)!=10000:raise ValueError('Complete native targets required')
        for i,row in enumerate(native_rows):
            if (row['targetIndex'],row['block'],row['position'],row['score']['target'])!=(i,i//1024,i%1024+1,tokens[i+1]):raise ValueError('Native target order differs')
        guard()
        import mlx.core as mx
        import numpy as np
        from mlx_lm.utils import load_model
        from mlx_lm.models.qwen2 import Model,ModelArgs
        from mlx_lm.models.cache import make_prompt_cache
        mx.set_default_device(mx.gpu)
        samples.append(dict(phase='before-model',usage=native_usage.usage()))
        # Local-only static classes; trust_remote_code stays false.
        model,loaded=load_model(reference.MODEL,lazy=False,strict=True,get_model_classes=lambda config:(Model,ModelArgs),trust_remote_code=False)
        if loaded!=config or model.args.vocab_size!=151936:raise ValueError('Loaded source differs')
        loaded_usage=native_usage.usage();samples.append(dict(phase='loaded',usage=loaded_usage))
        if not 0<loaded_usage['lifetime_max_phys_footprint']<=6*1024**3:raise ValueError('Loaded upstream reference exceeds Mac budget')
        guard()
        with (output/'targets.jsonl').open('xb') as stream:
            for start,end in blocks():
                cache=make_prompt_cache(model)
                for index in range(start,end):
                    gpu=model(mx.array([[tokens[index]]],dtype=mx.uint32),cache=cache)
                    if tuple(gpu.shape)!=(1,1,151936):raise ValueError('Upstream distribution shape differs')
                    dtypes.add(str(gpu.dtype));mx.eval(gpu)
                    logits=np.array(gpu[0,-1,:].astype(mx.float32),copy=True);score=loss(logits,tokens[index+1]);losses.append(score['negative_log_likelihood'])
                    old=native_rows[index]['score'];differences.append(abs(score['negative_log_likelihood']-old['negativeLogLikelihood']));target_differences.append(abs(score['target_logit']-old['targetLogit']));greedy_differences+=score['greedy']!=old['greedy']
                    row=dict(target_index=index,block=start//1024,position=index-start+1,score=score)
                    data=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();stream.write(data);raw_digest.update(data);count+=1
                stream.flush();mx.eval([entry.state for entry in cache]);usage=native_usage.usage();samples.append(dict(phase='block-'+str(start//1024),usage=usage))
                if not 0<usage['lifetime_max_phys_footprint']<=6*1024**3:raise ValueError('Upstream reference exceeds Mac budget')
                del cache;mx.clear_cache();guard();print(json.dumps(dict(completed_targets=count,block=start//1024)),flush=True)
        if count!=10000:raise ValueError('Incomplete target coverage')
        total=math.fsum(losses);mean=total/count;delta=mean-baseline_mean
        summary=dict(targets=count,blocks=10,block_size=1024,policy=quality_policy(),sum_negative_log_likelihood=total,mean_negative_log_likelihood=mean,baseline_mean=baseline_mean,increase=delta,limit=limit,loss_gate_passed=delta<=limit)
        comparison=dict(targets=count,native_mean=native['execution']['loss_gates']['mac']['mean_negative_log_likelihood'],mean_difference=mean-native['execution']['loss_gates']['mac']['mean_negative_log_likelihood'],maximum_loss_difference=max(differences),maximum_target_logit_difference=max(target_differences),greedy_disagreements=greedy_differences,scope='Descriptive same-artifact comparison; upstream output precision and kernels differ. No new acceptance tolerance.')
        del model;mx.clear_cache();samples.append(dict(phase='closed',usage=native_usage.usage()))
        if reference.sha(output/'targets.jsonl')!=raw_digest.hexdigest():raise ValueError('Completed raw stream differs')
        merge({str((output/'targets.jsonl').relative_to(ROOT)):raw_digest.hexdigest()});guard()
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        signal.signal(signal.SIGTERM,previous)
        try:
            retained=retained_journal(output/'targets.jsonl')
            if retained:merge({str((output/'targets.jsonl').relative_to(ROOT)):retained['sha256']})
        except BaseException as error:journal_errors.append(type(error).__name__+': '+str(error))
        try:guard()
        except BaseException as error:errors.append(type(error).__name__+': '+str(error))
    journal_complete=retained and retained['sha256']==raw_digest.hexdigest() and retained['complete_lines']==10000 and not retained['trailing_bytes']
    record=dict(format='toklet-mlx-upstream-full-loss-v1',passed=primary is None and not errors and not journal_errors and count==10000 and bool(journal_complete),versions=VERSIONS,selected_producer_and_input_pins=pins,protected_binary_pins=PROTECTED,repository=reference.REPO,revision=reference.REVISION,logit_dtypes=sorted(dtypes),native_usage=samples,completed_targets=count,raw_sha256=retained and retained['sha256'],attempted_journal_sha256=raw_digest.hexdigest(),retained_journal=retained,journal_finalization_errors=journal_errors,summary=summary,native_comparison=comparison,primary_error=primary,endpoint_errors=errors,product_backend_added=False,performance_qualified=False,checkpoint_adopted=False)
    data=(json.dumps(record,indent=2,allow_nan=False)+'\n').encode()
    with (output/'result.json').open('xb') as stream:stream.write(data)
    guard()
    if reference.sha(output/'result.json')!=hashlib.sha256(data).hexdigest():raise ValueError('Published result differs')
    print(json.dumps(dict(passed=record['passed'],summary=summary,native_comparison=comparison,error=primary,endpoint_errors=errors)),flush=True)
    return record


def quality_policy():return 'independent_blocks_of_up_to_1025_tokens_one_token_overlap_score_every_target_once'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    sys.exit(0 if run(parser.parse_args().output)['passed'] else 1)
