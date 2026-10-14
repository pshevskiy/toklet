"""One frozen Q4 v2 candidate: unchanged guards before 10,000-target loss.

Reuse only qualified transport/row/policy helpers; v1 collectors and defaults
remain unchanged. This records loss, never task quality or latency acceptance.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile

import metal_quality as shared
import metal_quality_larger as large

ROOT=shared.ROOT
GEOMETRY_REL='benchmarks/results/2026-10-14-q4-v2-real-geometry-qualification-v1.json'
GEOMETRY_SHA='b63ef25e1331fe92898f1e7dbf11d05d074517859b72e8fd8dc873675d21029d'
MODE='q4-v2-compensated-tiled'
SPECS={
    'small':dict(name='Qwen2.5-0.5B-Instruct',source=shared.SOURCE_SHA,config=shared.CONFIG_SHA,
                 packed='56b3ed180328892cb83489b4f5d252dc68b3584729e8994e4cf16419242f49f7',
                 weights=630251436,hidden=896,layers=24,baseline='benchmarks/results/2026-10-11-held-out-scalar-loss-analysis-v1.json',
                 baseline_sha=shared.BASELINE_SHA,mean=2.883725149929484),
    'larger':dict(name='Qwen2.5-1.5B-Instruct',source=large.SOURCE_SHA,config=large.CONFIG_SHA,
                  packed='55fd3877289139a74a815ea3e65d5f7ee6a39ef837b5b441f74e764e9113abdb',
                  weights=1777247428,hidden=1536,layers=28,baseline=large.BASELINE_REL,
                  baseline_sha=large.BASELINE_SHA,mean=large.BASELINE_MEAN),
}


def ready(event,tokens,sha,quantized,chunk,*,label):
    spec=SPECS[label]
    expected={'event':'ready','format':'toklet-metal-held-out-loss-events-v1','backend':'metal',
              'quantized':True,'debug_assertions':False,'target_tokens':len(tokens)-1,
              'corpus_sha256':sha,'context_policy':shared.POLICY,'block_targets':1024,'context_capacity':1025,
              'prefill_chunk_size':chunk,'logit_positions':'all_causal_rows','weight_dtype':'Q4 v2 + FP16 exclusions',
              'kv_dtype':'FP16','q4_format':'V2','q4_quantizer':'toklet-q4-affine-f32-source-half-metadata-rne-g8-v2','hidden_size':spec['hidden'],'num_layers':spec['layers'],
              'vocab_size':151936,'projection_mode':'Q4CompensatedTiled','normalization_mode':'Compensated'}
    if quantized is not True or any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in expected.items()):
        raise ValueError('Frozen Q4 v2 loss configuration differs')
    widths=event.get('q4_compensated_execution_widths')
    if (type(widths) is not list or len(widths)!=2 or any(type(v) is not int or v<=0 for v in widths)
            or widths[0]!=widths[1]):raise ValueError('Q4 v2 execution widths differ')
    shared.continuation.memory(event['memory'])
    if (event['memory']['weights_bytes']!=spec['weights']
            or type(event.get('application_allowance_bytes')) is not int or event['application_allowance_bytes']!=64*1024**2
            or type(event.get('memory_admitted_bytes')) is not int
            or event['memory_admitted_bytes']!=event['memory']['admitted_bytes']+64*1024**2
            or not 0<event['memory_admitted_bytes']<=shared.LIMIT):raise ValueError('Q4 v2 residency/admission differs')


def probe_specs():
    """Admit expected names/order/token bytes without a path hash/read split."""
    path=ROOT/'tests/fixtures/qwen-v2/manifest.json'
    manifest=json.loads(shared.frozen_bytes(path,1024**2,shared.quality.PROBE_SHA))
    result=[]
    for name in ('future_changed','short'):
        case=manifest['cases'][name];tokens=case['tokens']
        if case['offset']!=0 or len(tokens)!=4 or any(type(t) is not int or not 0<=t<151936 for t in tokens):
            raise ValueError('Frozen short probe shape differs')
        sha=hashlib.sha256(struct.pack('<4I',*tokens)).hexdigest()
        result.append((name,tokens,sha))
    return result


def collect(binary,source,packed,corpus,tokens,baseline,label,raw,stderr,full,*,result=None):
    """Guard ordering is shared by native collection and portable regression tests."""
    if result is None:result={}
    result.update(passed=False,probes_completed=False,collection_completed=False,quality_completed=False,runs=[])
    admission=partial(ready,label=label)
    with tempfile.TemporaryDirectory(prefix='toklet-q4-v2-loss-') as directory:
        expected=probe_specs()
        probes=shared.quality.probe_inputs(Path(directory));result['probes']=probes
        if [(p['name'],p['tokens'],p['corpus_sha256']) for p in probes]!=expected:
            raise ValueError('Returned short probe metadata differs from frozen admission')
        if label=='larger' and [(p['name'],p['tokens'],p['corpus_sha256']) for p in probes]!=[(p['name'],p['tokens'],p['corpus_sha256']) for p in baseline['probes']]:
            raise ValueError('Matched larger probes differ')
        for i,probe in enumerate(probes):
            source_nll=probe['source_nll'] if label=='small' else [r['negative_log_likelihood'] for r in baseline['runs'][2*i+1]['losses']]
            if label=='larger':
                for key in ('source_nll','source_logit_path','source_logit_sha256'):del probe[key]
            if len(source_nll)!=len(probe['tokens'])-1:raise ValueError('Matched probe coverage differs')
            probe['matched_fp32_nll']=source_nll
            paired=[]
            for chunk in (1,128):
                record=shared.child(binary,source,packed,Path(probe['path']),probe['tokens'],True,chunk,raw,stderr,
                                    admission=admission,extra_args=(label,MODE))
                result['runs'].append(record)
                if not record['passed']:return result
                paired.append(record['losses'])
            probe['batch_minus_single_nll']=shared.quality.check_probe(paired[1],[r['negative_log_likelihood'] for r in paired[0]])
            probe['batch_minus_matched_fp32_nll']=[r['negative_log_likelihood']-n for r,n in zip(paired[1],source_nll)]
    result.update(probes_completed=True,passed=True)
    if full:
        record=shared.child(binary,source,packed,corpus,tokens,True,128,raw,stderr,admission=admission,extra_args=(label,MODE))
        result['runs'].append(record)
        if not record['passed']:
            result['passed']=False;return result
        mean=record['mean_negative_log_likelihood'];delta=mean-SPECS[label]['mean']
        result.update(collection_completed=True,quality_completed=True,scored_targets=10000,context_policy=shared.POLICY,
                      mean_negative_log_likelihood=mean,baseline_mean_negative_log_likelihood=SPECS[label]['mean'],
                      increase_nats_per_token=delta,quality_gate={'maximum_increase_nats_per_token':.05,'passed':delta<=.05},passed=delta<=.05)
    return result


def run(binary,output,label,full):
    paths=[output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):raise ValueError('Q4 v2 evidence destination exists')
    spec=SPECS[label];source=ROOT/'models'/spec['name'];packed=ROOT/f'artifacts/m5-{label}-q4-v2'
    corpus=ROOT/'benchmarks/contract-v1/held_out_loss.u32le';baseline_path=ROOT/spec['baseline']
    tokens=list(struct.unpack('<10001I',shared.frozen_bytes(corpus,10001*4,shared.CORPUS_SHA)))
    baseline=json.loads(shared.frozen_bytes(baseline_path,8*1024**2,spec['baseline_sha']))
    if label=='larger':large.reference(baseline,tokens)
    elif baseline['scored_targets']!=10000 or baseline['context_policy']!=shared.POLICY or baseline['mean_negative_log_likelihood']!=spec['mean']:
        raise ValueError('Frozen small FP32 baseline differs')
    contract=json.loads(shared.frozen_bytes(corpus.parent/'manifest.json',2*1024**2,shared.CONTRACT_SHA))
    if contract['quality']['mean_nll_increase_limit_nats_per_token']!=.05:raise ValueError('Frozen loss gate differs')
    geometry_path=ROOT/GEOMETRY_REL
    geometry=json.loads(shared.frozen_bytes(geometry_path,1024**2,GEOMETRY_SHA))
    if geometry['passed'] is not True or geometry['review'].get('accepted') is not True:raise ValueError('Reviewed GPU geometry required')
    # Recheck all frozen consumed model/reference bytes, not just the manifest.
    artifact_pins={p:v['sha256'] for p,v in geometry['bindings'].items()
                   if p.startswith((f'artifacts/m5-{label}-q4-v2',f'tests/fixtures/qwen-{label}-q4-v2'))}
    if artifact_pins.get(str(packed.relative_to(ROOT)/'manifest.json'))!=spec['packed']:raise ValueError('Qualified packed inventory differs')
    files=[Path(__file__),Path(shared.__file__),Path(large.__file__),Path(large.quality_rows.__file__),
           Path(shared.metal_engine.__file__),Path(shared.native_usage.__file__),Path(shared.continuation.__file__),
           Path(shared.quality.__file__),Path(shared.quality.prepare.__file__),binary,geometry_path,corpus,
           corpus.parent/'manifest.json',baseline_path,source/'model.safetensors',source/'config.json',
           ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'examples/metal_quality.rs',ROOT/'examples/support/loss.rs']
    # The shared short-input generator consumes these exact unquantized source
    # bytes even for larger token inputs. Bind their endpoints as well.
    probe_path=ROOT/'tests/fixtures/qwen-v2/manifest.json'
    full_path=ROOT/'artifacts/qwen-forward-v2/manifest.json'
    fp32=json.loads(shared.frozen_bytes(full_path,1024**2,shared.quality.FULL_SHA))
    shared.frozen_bytes(probe_path,1024**2,shared.quality.PROBE_SHA)
    probe_pins={probe_path:shared.quality.PROBE_SHA,full_path:shared.quality.FULL_SHA}
    for name in ('future_changed','short'):
        item=fp32['arrays'][f'cases.{name}.logits']
        path=full_path.parent/item['file']
        if not path.resolve().is_relative_to(full_path.parent.resolve()):raise ValueError('FP32 probe path escapes reference')
        probe_pins[path]=item['sha256']
    files+=list(probe_pins)+list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))+[ROOT/p for p in artifact_pins]
    identities={str(p):shared.digest(p) for p in files}
    initial={source/'model.safetensors':spec['source'],source/'config.json':spec['config'],corpus:shared.CORPUS_SHA,
             corpus.parent/'manifest.json':shared.CONTRACT_SHA,baseline_path:spec['baseline_sha'],geometry_path:GEOMETRY_SHA}
    initial.update({ROOT/p:sha for p,sha in artifact_pins.items()});initial.update(probe_pins)
    if any(identities[str(p)]!=sha for p,sha in initial.items()):raise ValueError('Initial frozen Q4 v2 identities changed')
    result={'format':'toklet-q4-v2-held-out-loss-v1','model':label,'passed':False,'collection_completed':False,'quality_completed':False,
            'probes_completed':False,'full_corpus_requested':full,'quantized':True,'projection_mode':'Q4CompensatedTiled',
            'normalization_mode':'Compensated','weight_dtype':'Q4 v2 + FP16 exclusions',
            'scope':'One frozen candidate loss only; task quality and latency separate','identities':identities,
            'engine_revision':shared.continuation.observation(['git','rev-parse','HEAD']),
            'compiler':shared.continuation.observation(['rustc','-Vv']),'platform':shared.continuation.observation(['/usr/bin/sw_vers']),
            'build':'CARGO_TARGET_DIR=artifacts/m5-checks cargo build --release --features metal --example metal_quality; default release flags',
            'python':sys.version,'power_before':shared.continuation.endpoints(),
            'memory_scope':'Explicit shared allocation admission and idle owned-worker + driver native lifetime charge under6GiB.'}
    with paths[0].open('x') as summary,paths[1].open('x') as raw,paths[2].open('xb') as stderr:
        try:collect(binary,source,packed,corpus,tokens,baseline,label,raw,stderr,full,result=result)
        except BaseException as error:result['passed']=False;result['error']=f'{type(error).__name__}: {error}'
        finally:
            result['endpoint_errors']=[]
            try:result['identities_unchanged']=all(shared.digest(Path(p))==sha for p,sha in identities.items())
            except BaseException as error:result['identities_unchanged']=False;result['endpoint_errors'].append(str(error))
            try:result['power_after']=shared.continuation.endpoints()
            except BaseException as error:result['endpoint_errors'].append(str(error))
            if not result['identities_unchanged'] or result['endpoint_errors']:
                result.update(passed=False,collection_completed=False,quality_completed=False)
            json.dump(result,summary,indent=2,allow_nan=False);summary.write('\n')
    if not result['probes_completed'] or full and not result['collection_completed'] or not result['identities_unchanged'] or result['endpoint_errors']:
        raise ValueError('Q4 v2 loss collection failed; see retained evidence')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',choices=SPECS,required=True);parser.add_argument('--full',action='store_true')
    args=parser.parse_args();result=run(args.binary.resolve(),args.output.resolve(),args.model,args.full)
    if not result['passed']:sys.exit(2)
