"""Frozen M6 Online attention loss checks for the two selected M5 models.

Original 10,000-target policy and FP32 thresholds remain unchanged. Short
single/batch guards precede the full corpus; no task or latency acceptance.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import struct
import sys
import tempfile

import metal_quality_v2 as v2

shared = v2.shared
ROOT = shared.ROOT


def runtime_paths():
    return sorted([*ROOT.joinpath('src').rglob('*.rs'),
                   *ROOT.joinpath('src').rglob('*.metal'),
                   *(ROOT / p for p in ('Cargo.toml', 'Cargo.lock',
                                        'examples/metal_quality.rs', 'examples/support/loss.rs'))])


def build_admission(path, binary):
    """Bind the exact reviewed build snapshot and reassert just before Popen."""
    def manifest_bytes():
        with path.open('rb') as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError('Build manifest exceeds1MiB')
        return raw
    raw = manifest_bytes()
    value = json.loads(raw)
    if value.get('format') != 'toklet-m6-quality-build-v1':
        raise ValueError('Unknown compiled source manifest')
    expected = {str(p.relative_to(ROOT)) for p in runtime_paths()}
    pins = value['sources']
    if type(pins) is not dict or set(pins) != expected:
        raise ValueError('Compiled source closure differs')
    snapshot = ROOT / value['snapshot']
    if not snapshot.resolve().is_relative_to((ROOT / 'artifacts').resolve()):
        raise ValueError('Build snapshot outside artifacts')
    if binary.resolve() != (ROOT / value['binary']['path']).resolve():
        raise ValueError('Build executable differs')
    def check():
        if manifest_bytes() != raw or shared.digest(binary) != value['binary']['sha256']:
            raise ValueError('Build manifest/executable changed')
        for name, digest in pins.items():
            if shared.digest(ROOT / name) != digest or shared.digest(snapshot / name) != digest:
                raise ValueError(f'Compiled source changed: {name}')
    check()
    return value, check


def ready(event, tokens, digest, quantized, chunk, *, model):
    if event.get('attention_mode') != 'Online':
        raise ValueError('Actual Online attention required')
    if model == 'small':
        shared.small_half_admission(event, tokens, digest, quantized, chunk)
    else:
        v2.ready(event, tokens, digest, quantized, chunk, label=model)


def collect(binary, source, packed, corpus, tokens, baseline, model, full,
            raw, stderr, before_launch, result):
    result.update(passed=False, probes_completed=False, collection_completed=False,
                  quality_completed=False, runs=[])
    quantized = model == 'larger'
    selection = v2.MODE if quantized else 'half-compensated'
    admission = partial(ready, model=model)
    def child(path, ids, chunk):
        return shared.child(binary, source, packed, path, ids, quantized, chunk,
                            raw, stderr, admission=admission,
                            extra_args=(model, selection, 'online'), before_launch=before_launch)
    with tempfile.TemporaryDirectory(prefix='toklet-m6-loss-') as directory:
        probes = shared.quality.probe_inputs(Path(directory))
        if [(p['name'], p['tokens'], p['corpus_sha256']) for p in probes] != v2.probe_specs():
            raise ValueError('Frozen guard inputs differ')
        result['probes'] = probes
        for i, probe in enumerate(probes):
            source_nll = (probe['source_nll'] if model == 'small' else
                          [r['negative_log_likelihood'] for r in baseline['runs'][2*i+1]['losses']])
            if model == 'larger':
                for name in ('source_nll', 'source_logit_path', 'source_logit_sha256'):
                    del probe[name]
            probe['matched_fp32_nll'] = source_nll
            pairs = []
            for chunk in (1, 128):
                run = child(Path(probe['path']), probe['tokens'], chunk)
                result['runs'].append(run)
                if not run['passed']:
                    return
                pairs.append(run['losses'])
            probe['batch_minus_single_nll'] = shared.quality.check_probe(
                pairs[1], [r['negative_log_likelihood'] for r in pairs[0]])
            probe['batch_minus_matched_fp32_nll'] = [r['negative_log_likelihood']-n
                                                    for r,n in zip(pairs[1],source_nll,strict=True)]
    result.update(probes_completed=True, passed=True)
    if full:
        run = child(corpus, tokens, 128)
        result['runs'].append(run)
        result['passed'] = False
        if not run['passed']:
            return
        mean = run['mean_negative_log_likelihood']
        delta = mean - v2.SPECS[model]['mean']
        result.update(collection_completed=True, quality_completed=True, scored_targets=10000,
                      context_policy=shared.POLICY, mean_negative_log_likelihood=mean,
                      baseline_mean_negative_log_likelihood=v2.SPECS[model]['mean'],
                      increase_nats_per_token=delta,
                      quality_gate={'maximum_increase_nats_per_token':.05, 'passed':delta<=.05},
                      passed=delta<=.05)


def run(binary, manifest_path, output, model, full):
    paths = [output, output.with_suffix('.raw.jsonl'), output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):
        raise ValueError('Refusing to overwrite M6 loss evidence')
    build, check_build = build_admission(manifest_path, binary)
    spec = v2.SPECS[model]
    source = ROOT / 'models' / spec['name']
    packed = ROOT / 'artifacts/m5-larger-q4-v2' if model == 'larger' else None
    corpus = ROOT / 'benchmarks/contract-v1/held_out_loss.u32le'
    tokens = list(struct.unpack('<10001I', shared.frozen_bytes(corpus, 40004, shared.CORPUS_SHA)))
    baseline_path = ROOT / spec['baseline']
    baseline = json.loads(shared.frozen_bytes(baseline_path, 8*1024**2, spec['baseline_sha']))
    if model == 'larger':
        v2.large.reference(baseline, tokens)
    elif (baseline['scored_targets'], baseline['context_policy'], baseline['mean_negative_log_likelihood']) != (10000,shared.POLICY,spec['mean']):
        raise ValueError('Frozen small FP32 baseline differs')
    contract_path = corpus.parent / 'manifest.json'
    contract = json.loads(shared.frozen_bytes(contract_path, 2*1024**2, shared.CONTRACT_SHA))
    if contract['quality']['mean_nll_increase_limit_nats_per_token'] != .05:
        raise ValueError('Frozen loss gate differs')
    # Bind every payload read by the shared short-probe helper, including its
    # small unquantized fixture source when checking the larger geometry.
    probe_manifest = ROOT / 'tests/fixtures/qwen-v2/manifest.json'
    source_manifest = ROOT / 'artifacts/qwen-forward-v2/manifest.json'
    fp32 = json.loads(shared.frozen_bytes(source_manifest, 1024**2, shared.quality.FULL_SHA))
    shared.frozen_bytes(probe_manifest, 1024**2, shared.quality.PROBE_SHA)
    fixed = {source/'model.safetensors':spec['source'], source/'config.json':spec['config'],
             corpus:shared.CORPUS_SHA, contract_path:shared.CONTRACT_SHA,
             baseline_path:spec['baseline_sha'], probe_manifest:shared.quality.PROBE_SHA,
             source_manifest:shared.quality.FULL_SHA}
    for name in ('future_changed','short'):
        item = fp32['arrays'][f'cases.{name}.logits']
        path = source_manifest.parent / item['file']
        if not path.resolve().is_relative_to(source_manifest.parent.resolve()):
            raise ValueError('Short probe source escapes fixture')
        fixed[path] = item['sha256']
    if packed is not None:
        geometry_path = ROOT / v2.GEOMETRY_REL
        geometry = json.loads(shared.frozen_bytes(geometry_path, 2*1024**2, v2.GEOMETRY_SHA))
        if geometry.get('passed') is not True or geometry.get('review',{}).get('accepted') is not True:
            raise ValueError('Reviewed v2 geometry required')
        fixed[geometry_path] = v2.GEOMETRY_SHA
        prefix = 'artifacts/m5-larger-q4-v2/'
        artifact = {ROOT/p:v['sha256'] for p,v in geometry['bindings'].items() if p.startswith(prefix)}
        if artifact.get(packed/'manifest.json') != spec['packed']:
            raise ValueError('Frozen selected Q4 artifact differs')
        if set(artifact) != set(packed.glob('*.q4')) | {packed/'manifest.json'}:
            raise ValueError('Packed payload inventory differs')
        fixed.update(artifact)
    modules = (v2,shared,v2.large,v2.large.quality_rows,shared.metal_engine,
               shared.native_usage,shared.continuation,shared.quality,shared.quality.prepare)
    files = set(fixed) | set(runtime_paths()) | {binary,manifest_path,Path(__file__)}
    files.update(Path(m.__file__) for m in modules)
    identities = {str(p.relative_to(ROOT)):shared.digest(p) for p in sorted(files)}
    if any(identities[str(p.relative_to(ROOT))] != digest for p,digest in fixed.items()):
        raise ValueError('Frozen consumed bytes differ')
    check_build()
    result = {'format':'toklet-m6-online-loss-v1','model':model,'attention_mode':'Online',
              'full_corpus_requested':full,'passed':False,'collection_completed':False,'quality_completed':False,
              'scope':'Unchanged short guards and10000-target loss only; no task or latency acceptance',
              'compiled_build':build,'identities':identities,
              'revision':shared.continuation.observation(['git','rev-parse','HEAD']),
              'compiler':shared.continuation.observation(['rustc','-Vv']),
              'platform':shared.continuation.observation(['/usr/bin/sw_vers']),
              'power_before':shared.continuation.endpoints()}
    with paths[0].open('x') as summary, paths[1].open('x') as raw, paths[2].open('xb') as stderr:
        try:
            collect(binary,source,packed,corpus,tokens,baseline,model,full,raw,stderr,check_build,result)
        except BaseException as error:
            result.update(passed=False,error=f'{type(error).__name__}: {error}')
        finally:
            result['endpoint_errors'] = []
            try:
                result['identities_unchanged'] = all(shared.digest(ROOT/p)==digest for p,digest in identities.items())
                check_build()
            except BaseException as error:
                result['identities_unchanged'] = False
                result['endpoint_errors'].append(str(error))
            try:
                result['power_after'] = shared.continuation.endpoints()
            except BaseException as error:
                result['endpoint_errors'].append(str(error))
            if not result['identities_unchanged'] or result['endpoint_errors']:
                result.update(passed=False,collection_completed=False,quality_completed=False)
            json.dump(result,summary,indent=2,allow_nan=False);summary.write('\n')
    if not result['passed']:
        raise ValueError('M6 Online loss collection or gate failed; see retained evidence')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--build-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',choices=('small','larger'),required=True)
    parser.add_argument('--full',action='store_true')
    a = parser.parse_args()
    run(a.binary.resolve(),a.build_manifest.resolve(),a.output.resolve(),a.model,a.full)
