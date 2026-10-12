"""Qualify separate tiny attention projections through the Rust MPS boundary."""
import argparse
import json
import math
from pathlib import Path
import struct

import rust_metal_projection as projection
import rust_metal_mlp as composed
rms = projection.rms


def batches():
    half = rms.manifest(rms.FP16, rms.FP16_MANIFEST_SHA)
    single = rms.manifest(rms.FP32, rms.FP32_MANIFEST_SHA)
    groups, anchors = [], {}
    for stage in ('k', 'v', 'o'):
        width, outputs = 32, (32 if stage == 'o' else 16)
        for layer in range(2):
            cases = []
            weight, weight_item = rms.payload(rms.FP16, half, f'weights.layers.{layer}.{stage}_proj.weight', 'f16le')
            if weight_item['shape'] != [outputs, width]:
                raise ValueError('attention projection weight shape differs')
            weight_bits = list(struct.unpack(f'<{outputs*width}H', weight))
            bias_key = f'weights.layers.{layer}.{stage}_proj.bias'
            if stage == 'o':
                if bias_key in half['arrays'] or bias_key in single['arrays']:
                    raise ValueError('Output attention bias policy differs')
                bias_bits, bias_item = [0]*outputs, None
            else:
                bias, bias_item = rms.payload(rms.FP16, half, bias_key, 'f16le')
                if bias_item['shape'] != [outputs]:
                    raise ValueError('Attention projection bias shape differs')
                bias_bits = list(struct.unpack(f'<{outputs}H', bias))
            for name, item in sorted(half['cases'].items()):
                rows = len(item['tokens'])
                prefix = f'cases.{name}.layers.{layer}'
                key = f"{prefix}.{'attention' if stage == 'o' else stage}_projection"
                input_key = f"{prefix}.{'attention_context' if stage == 'o' else 'input_norm'}"
                input_data, input_item = rms.payload(rms.FP16, half, input_key, 'f16le')
                oracle, oracle_item = rms.payload(rms.FP16, half, key, 'f16le')
                source, source_item = rms.payload(rms.FP32, single, key, 'f32le')
                if (input_item['shape'] != [rows, width] or oracle_item['shape'] != ([rows, outputs] if stage == 'o' else [rows, 2, 8])
                        or source_item['shape'] != ([rows, outputs] if stage == 'o' else [rows, 2, 8])):
                    raise ValueError('attention projection frozen/source shape differs')
                a = list(struct.unpack(f'<{rows*width}H', input_data))
                bits = list(struct.unpack(f'<{rows*outputs}H', oracle))
                refs = list(struct.unpack(f'<{rows*outputs}f', source))
                if any(value & 0x7c00 == 0x7c00 for values in (a, weight_bits, bias_bits, bits) for value in values) or not all(map(math.isfinite, refs)):
                    raise ValueError('attention projection source is nonfinite')
                cases.append({'id': key, 'rows': rows, 'inputs': width, 'outputs': outputs,
                              'input_bits': a, 'weight_bits': weight_bits.copy(), 'bias_bits': bias_bits.copy()})
                anchors[key] = {'source': refs, 'oracle_bits': bits, 'gate': oracle_item['gate_against_source_fp32'],
                                'input_sha256': input_item['sha256'], 'weight_sha256': weight_item['sha256'],
                                'oracle_sha256': oracle_item['sha256'], 'source_sha256': source_item['sha256'],
                                'bias_sha256': None if bias_item is None else bias_item['sha256'],
                                'bias': 'Absent; resident positive-zero Half vector' if bias_item is None else 'Frozen Half bias'}
            # Split by layer to keep each completed protocol line below one MiB
            # at the existing bound, rather than enlarging its transport cap.
            inventory = sum(c['rows']*c['inputs'] + c['outputs']*c['inputs'] + c['outputs'] + c['rows']*c['outputs'] for c in cases)
            if len(cases) != 5 or inventory > 32768:
                raise ValueError('attention projection batch inventory differs')
            groups.append((f'{stage}_layer_{layer}', cases))
    if len(anchors) != 30 or sum(c['rows']*c['outputs'] for _,cs in groups for c in cs) != 5632:
        raise ValueError('attention projection complete inventory differs')
    return groups, anchors


def run(output, build, directory):
    rms.probe.check_paths(output, build)
    evidence = (output, output.with_suffix(output.suffix+'.stderr.txt'), output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():
        raise ValueError('Refusing to overwrite attention projection inputs')
    for path in (build,*evidence):
        if directory.resolve() == path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:
            raise ValueError('attention projection inputs/build/evidence must be disjoint')
    for source in (rms.FP16,rms.FP32):
        if any(source.resolve() == path.resolve() or source.resolve() in path.resolve().parents for path in (build,directory,*evidence)):
            raise ValueError('attention projection outputs must be outside frozen fixtures')
    groups, anchors = batches()
    query, query_anchors, overflow = projection.cases()
    tiny, tiny_anchors, gating_overflow = projection.gating.cases()
    tails, tail_anchors = projection.gating.tail_cases()
    directory.mkdir(parents=True)
    jobs, identities = [], {}
    mlps, mlp_anchors, mlp_overflow = composed.cases()
    for index, case in enumerate(mlps + [mlp_overflow]):
        label = f'mlp_{index+1}' if index < len(mlps) else 'mlp_overflow'
        path = directory/f'{label}.json'
        with path.open('x') as stream:
            stream.write(json.dumps(case,separators=(',',':'))+'\n')
        identities[str(path.resolve())] = rms.probe.common.digest(path)
        validator = (lambda e,case=case: composed.validate(e,case,mlp_anchors[case['id']])) if index < len(mlps) else composed.overflow_rejected
        jobs.append((label,['--mlp-input',str(path.resolve())],validator))
    payloads = [(label,cases,'--projection-input',lambda event,cases=cases:projection.validate(event,cases,anchors)) for label,cases in groups]
    stage_groups, stage_anchors = composed.stages.batches()
    payloads += [(label,cases,'--projection-input',lambda e,cases=cases:projection.validate(e,cases,stage_anchors)) for label,cases in stage_groups]
    payloads += [
        ('query_manual_regression',query,'--projection-input',lambda e:projection.validate(e,query,query_anchors)),
        ('projection_overflow_regression',[overflow],'--projection-input',projection.overflow_rejected),
        ('gating_regression',tiny,'--gating-input',lambda e:projection.gating.validate(e,tiny,tiny_anchors)),
        ('gating_tails_regression',tails,'--gating-input',lambda e:projection.gating.validate_tails(e,tails,tail_anchors)),
        ('gating_overflow_regression',[gating_overflow],'--gating-input',projection.gating.overflow_rejected),
    ]
    for label,cases,option,validator in payloads:
        path=directory/f'{label}.json'
        with path.open('x') as stream:
            stream.write(json.dumps({'format':'toklet-metal-projection-input-v1' if option=='--projection-input' else 'toklet-metal-gating-input-v1','cases':cases},separators=(',',':'))+'\n')
        identities[str(path.resolve())]=rms.probe.common.digest(path)
        jobs.append((label,[option,str(path.resolve())],validator))
    modules=(composed,composed.stages,projection,projection.gating,rms,projection.gating.silu,projection.gating.residual,projection.gating.residual.embedding)
    drivers={str(Path(m.__file__).resolve()):rms.probe.common.digest(Path(m.__file__)) for m in modules}
    drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        result=previous(event)
        if batches() != (groups,anchors) or composed.cases() != (mlps,mlp_anchors,mlp_overflow):raise ValueError('attention projection frozen inputs changed')
        for name,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(name)) != sha:raise ValueError('attention projection input/driver identity changed')
        return result
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context={
        'input_sha256':identities,'qualification_driver_sha256':drivers,
        'fp16_manifest_sha256':rms.FP16_MANIFEST_SHA,'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,
        'scope':'30 separate frozen tiny K/V/O projections:32->16 K/V with saved Half bias,32->32 O with zero bias. Saved input_norm or attention_context; contiguous token/head/dim views flatten without transpose. Original source gates and Half store. No propagated attention/KV, Qwen, full decoder, quality, latency or M4 completion.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
    args=p.parse_args();run(args.output,args.build_dir,args.input_dir)

if __name__=='__main__':main()
