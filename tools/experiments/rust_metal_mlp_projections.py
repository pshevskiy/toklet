"""Qualify separate tiny MLP projections through the Rust MPS boundary."""
import argparse
import json
import math
from pathlib import Path
import struct

import rust_metal_projection as projection
rms = projection.rms


def batches():
    half = rms.manifest(rms.FP16, rms.FP16_MANIFEST_SHA)
    single = rms.manifest(rms.FP32, rms.FP32_MANIFEST_SHA)
    groups, anchors = [], {}
    for stage in ('gate', 'up', 'down'):
        width, outputs = (64, 32) if stage == 'down' else (32, 64)
        for layer in range(2):
            cases = []
            weight, weight_item = rms.payload(rms.FP16, half, f'weights.layers.{layer}.{stage}_proj.weight', 'f16le')
            if weight_item['shape'] != [outputs, width]:
                raise ValueError('MLP projection weight shape differs')
            weight_bits = list(struct.unpack(f'<{outputs*width}H', weight))
            for name, item in sorted(half['cases'].items()):
                rows = len(item['tokens'])
                prefix = f'cases.{name}.layers.{layer}'
                key = f'{prefix}.{stage}_projection'
                input_key = f"{prefix}.{'gated' if stage == 'down' else 'post_attention_norm'}"
                input_data, input_item = rms.payload(rms.FP16, half, input_key, 'f16le')
                oracle, oracle_item = rms.payload(rms.FP16, half, key, 'f16le')
                source, source_item = rms.payload(rms.FP32, single, key, 'f32le')
                if (input_item['shape'] != [rows, width] or oracle_item['shape'] != [rows, outputs]
                        or source_item['shape'] != [rows, outputs]):
                    raise ValueError('MLP projection frozen/source shape differs')
                a = list(struct.unpack(f'<{rows*width}H', input_data))
                bits = list(struct.unpack(f'<{rows*outputs}H', oracle))
                refs = list(struct.unpack(f'<{rows*outputs}f', source))
                if any(value & 0x7c00 == 0x7c00 for values in (a, weight_bits, bits) for value in values) or not all(map(math.isfinite, refs)):
                    raise ValueError('MLP projection source is nonfinite')
                cases.append({'id': key, 'rows': rows, 'inputs': width, 'outputs': outputs,
                              'input_bits': a, 'weight_bits': weight_bits.copy(), 'bias_bits': [0]*outputs})
                anchors[key] = {'source': refs, 'oracle_bits': bits, 'gate': oracle_item['gate_against_source_fp32'],
                                'input_sha256': input_item['sha256'], 'weight_sha256': weight_item['sha256'],
                                'oracle_sha256': oracle_item['sha256'], 'source_sha256': source_item['sha256'],
                                'bias': 'Absent in frozen model; resident positive-zero Half vector for existing beta1 interface'}
            # Split by layer to keep each completed protocol line below one MiB
            # at the existing bound, rather than enlarging its transport cap.
            inventory = sum(c['rows']*c['inputs'] + c['outputs']*c['inputs'] + c['outputs'] + c['rows']*c['outputs'] for c in cases)
            if len(cases) != 5 or inventory > 32768:
                raise ValueError('MLP projection batch inventory differs')
            groups.append((f'{stage}_layer_{layer}', cases))
    if len(anchors) != 30 or sum(c['rows']*c['outputs'] for _,cs in groups for c in cs) != 14080:
        raise ValueError('MLP projection complete inventory differs')
    return groups, anchors


def run(output, build, directory):
    rms.probe.check_paths(output, build)
    evidence = (output, output.with_suffix(output.suffix+'.stderr.txt'), output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():
        raise ValueError('Refusing to overwrite MLP projection inputs')
    for path in (build,*evidence):
        if directory.resolve() == path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:
            raise ValueError('MLP projection inputs/build/evidence must be disjoint')
    for source in (rms.FP16,rms.FP32):
        if any(source.resolve() == path.resolve() or source.resolve() in path.resolve().parents for path in (build,directory,*evidence)):
            raise ValueError('MLP projection outputs must be outside frozen fixtures')
    groups, anchors = batches()
    query, query_anchors, overflow = projection.cases()
    tiny, tiny_anchors, gating_overflow = projection.gating.cases()
    tails, tail_anchors = projection.gating.tail_cases()
    directory.mkdir(parents=True)
    jobs, identities = [], {}
    payloads = [(label,cases,'--projection-input',lambda event,cases=cases:projection.validate(event,cases,anchors)) for label,cases in groups]
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
    modules=(projection,projection.gating,rms,projection.gating.silu,projection.gating.residual,projection.gating.residual.embedding)
    drivers={str(Path(m.__file__).resolve()):rms.probe.common.digest(Path(m.__file__)) for m in modules}
    drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        result=previous(event)
        if batches() != (groups,anchors):raise ValueError('MLP projection frozen inputs changed')
        for name,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(name)) != sha:raise ValueError('MLP projection input/driver identity changed')
        return result
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context={
        'input_sha256':identities,'qualification_driver_sha256':drivers,
        'fp16_manifest_sha256':rms.FP16_MANIFEST_SHA,'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,
        'scope':'30 separate frozen tiny gate/up/down projection stages at actual32->64 and64->32 shapes. All layer/case inputs are saved references; no propagation/composed MLP. Absent biases represented as positive-zero Half vectors in existing beta1 interface. Unchanged source gates and final Half store; same-binary query/manual/gating/tails/overflow regression. No Qwen, full decoder, quality, latency or M4 completion.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
    args=p.parse_args();run(args.output,args.build_dir,args.input_dir)

if __name__=='__main__':main()
