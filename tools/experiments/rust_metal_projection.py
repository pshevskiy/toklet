"""Qualify a bounded Rust MPS Half projection with FP32 bias/result staging."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import rust_metal_gating as gating
rms = gating.rms


def half_bits(values):
    return list(struct.unpack(f'<{len(values)}H', struct.pack(f'<{len(values)}e', *values)))


def cases():
    half = rms.manifest(rms.FP16, rms.FP16_MANIFEST_SHA)
    single = rms.manifest(rms.FP32, rms.FP32_MANIFEST_SHA)
    inputs, anchors = [], {}
    for name, item in sorted(half['cases'].items()):
        rows = len(item['tokens'])
        for layer in range(2):
            prefix = f'cases.{name}.layers.{layer}'
            arrays = [rms.payload(rms.FP16, half, key, 'f16le') for key in
                      (f'{prefix}.input_norm', f'weights.layers.{layer}.q_proj.weight', f'weights.layers.{layer}.q_proj.bias', f'{prefix}.q_projection')]
            shapes = [[rows, 32], [32, 32], [32], [rows, 4, 8]]
            if [item['shape'] for _, item in arrays] != shapes:
                raise ValueError('Projection frozen shape differs')
            key = f'{prefix}.q_projection'
            reference, ref = rms.payload(rms.FP32, single, key, 'f32le')
            if ref['shape'] != [rows, 4, 8]:
                raise ValueError('Projection source shape differs')
            bits = [list(struct.unpack(f'<{len(payload)//2}H', payload)) for payload, _ in arrays]
            refs = list(struct.unpack(f'<{rows*32}f', reference))
            if any(value & 0x7c00 == 0x7c00 for array in bits for value in array) or not all(map(math.isfinite, refs)):
                raise ValueError('Projection oracle is nonfinite')
            inputs.append({'id': key, 'rows': rows, 'inputs': 32, 'outputs': 32,
                           'input_bits': bits[0], 'weight_bits': bits[1], 'bias_bits': bits[2]})
            anchors[key] = {'source': refs, 'oracle_bits': bits[3], 'gate': arrays[3][1]['gate_against_source_fp32'],
                            'input_sha256': arrays[0][1]['sha256'], 'weight_sha256': arrays[1][1]['sha256'],
                            'bias_sha256': arrays[2][1]['sha256'], 'oracle_sha256': arrays[3][1]['sha256'],
                            'source_sha256': ref['sha256']}
    for key, rows, width, outputs, a, b, bias, expected in [
        ('manual.bias_rounding', 1, 2, 1, [1, 1], [1, 1/2048], [1/2048], [1+1/1024]),
        ('manual.irregular', 2, 3, 2, [2,-1,4,-1,0,2], [1,2,-1,.5,-2,3], [.25,-.5], [-3.75,14.5,-2.75,5]),
        ('manual.finite_cancellation', 1, 2, 1, [65504,65504], [2,-1], [0], [65504]),
    ]:
        # Independently reconstruct exact dyadic manual dots plus bias before a
        # single Half store; the bias tie must not first round the dot to Half.
        actual = [sum(a[row*width+i]*b[col*width+i] for i in range(width))+bias[col]
                  for row in range(rows) for col in range(outputs)]
        if actual != expected:
            raise ValueError('Manual independent projection differs')
        inputs.append({'id':key, 'rows':rows, 'inputs':width, 'outputs':outputs,
                       'input_bits':half_bits(a), 'weight_bits':half_bits(b), 'bias_bits':half_bits(bias)})
        anchors[key] = {'source':expected, 'oracle_bits':half_bits(expected), 'gate':None}
    if len(inputs) != 13:
        raise ValueError('Projection inventory differs')
    overflow = {'id':'manual.overflow', 'rows':1, 'inputs':1, 'outputs':1,
                'input_bits':[0x7bff], 'weight_bits':[0x4000], 'bias_bits':[0]}
    return inputs, anchors, overflow


def validate(event, inputs, anchors):
    if event.get('passed') is not True or event.get('event') != 'idle_result':
        raise ValueError('Native projection failed')
    result = event['result']
    meta = result['metadata']
    if (result['operation'] != 'mps_fp16_inputs_fp32_bias_and_result_fp16_store'
            or meta['unified_memory'] is not True or meta['language_version'] != '2.3'
            or meta['math_mode'] != 'safe' or meta['math_functions'] != 'precise'
            or meta['minimum_macos_major'] != 15 or meta.get('mps_device_supported') is not True
            or [c['id'] for c in result['cases']] != [c['id'] for c in inputs]):
        raise ValueError('Projection policy/inventory differs')
    for case, item in zip(result['cases'], inputs):
        rows, width, outputs = item['rows'], item['inputs'], item['outputs']
        count = rows * outputs
        launch_width = case['threadgroup_width']
        limits = [meta['execution_width'], meta['pipeline_max_threads'], meta['bias_execution_width'],
                  meta['bias_max_threads'], meta['device_max_threads'][0]]
        if type(launch_width) is not int or not 0 < launch_width <= min(limits):
            raise ValueError('Projection dispatch differs')
        groups = (count + launch_width - 1) // launch_width
        storage_rows = (groups * launch_width + outputs - 1) // outputs
        strides = [case[k] for k in ('input_stride','weight_stride','fp32_stride','output_stride')]
        if any(type(n) is not int or n <= 0 for n in strides) or strides[0] < width or strides[1] < width or min(strides[2:]) < outputs:
            raise ValueError('Projection strides differ')
        a = (rows*strides[0]+7)*2
        b = (outputs*strides[1]+7)*2
        bias = (outputs+7)*2
        c = (storage_rows*strides[2]+7)*4
        out = (storage_rows*strides[3]+7)*2
        shared = a+b+bias+c+out
        host = 2*(a+b+bias)+c+out
        if (case['rows_count'] != rows or case['inputs'] != width or case['outputs'] != outputs
                or case['threadgroups'] != groups or case['storage_rows'] != storage_rows
                or case['shared_buffer_bytes'] != shared or case['host_baseline_readback_bytes'] != host
                or case['admitted_bytes'] != shared+host+rms.probe.ALLOWANCE
                or case['admitted_bytes'] > rms.probe.LIMIT or len(case['rows']) != 13):
            raise ValueError('Projection shape/storage/admission differs')
        anchor = anchors[item['id']]
        for index, row in enumerate(case['rows']):
            bits, fp32 = row['output_bits'], row['fp32_bits']
            if (len(bits) != count or len(fp32) != count
                    or any(type(n) is not int or not 0 <= n <= 65535 for n in bits)
                    or any(type(n) is not int or not 0 <= n < 2**32 for n in fp32)
                    or row['phase'] != ('warmup' if index < 3 else 'sample')
                    or row['repetition'] != (index+1 if index < 3 else index-2)
                    or row['command_completed'] is not True or row['inputs_and_guards_unchanged'] is not True
                    or not math.isfinite(row['encode_commit_wait_seconds']) or row['encode_commit_wait_seconds'] <= 0):
                raise ValueError('Projection completed-work policy differs')
            values = struct.unpack(f'<{count}e',struct.pack(f'<{count}H',*bits))
            floats = struct.unpack(f'<{count}f',struct.pack(f'<{count}I',*fp32))
            if not all(map(math.isfinite, (*values, *floats))) or bits != half_bits(floats):
                raise ValueError('Projection must store one finite Half rounding of FP32 result')
            errors = [abs(value-ref) for value, ref in zip(values, anchor['source'])]
            gate = anchor['gate']
            fraction = max(error/(gate['atol']+gate['rtol']*abs(ref)) for error,ref in zip(errors,anchor['source'])) if gate else 0
            if fraction > 1 or gate is None and (bits != anchor['oracle_bits'] or max(errors) != 0):
                raise ValueError('Frozen projection gate/manual bits failed')
            row['qualification'] = {'source_fp32_max_gate_fraction':fraction, 'source_fp32_max_abs_error':max(errors),
                                    'equal_frozen_half_values':sum(a==b for a,b in zip(bits,anchor['oracle_bits'])),
                                    'single_half_store':True,
                                    'output_sha256':hashlib.sha256(struct.pack(f'<{count}H',*bits)).hexdigest()}
            del row['output_bits']; del row['fp32_bits']
        case['anchors'] = {k:v for k,v in anchor.items() if k not in ('source','oracle_bits')}
    if meta['maximum_admitted_bytes'] != max(c['admitted_bytes'] for c in result['cases']):
        raise ValueError('Projection maximum admission differs')


def overflow_rejected(event):
    if (event.get('passed') is not False or event.get('error',{}).get('stage') != 'validation'
            or event['error']['message'] != 'completed projection nonfinite; result discarded'):
        raise ValueError('Completed projection overflow rejection differs')
    return False


def run(output, build, directory):
    rms.probe.check_paths(output, build)
    evidence = (output, output.with_suffix(output.suffix+'.stderr.txt'), output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():
        raise ValueError('Refusing to overwrite projection inputs')
    protected = (rms.FP16, rms.FP32)
    for path in (build, *evidence):
        if directory.resolve() == path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:
            raise ValueError('Projection inputs/build/evidence must be disjoint')
    for source in protected:
        if any(source.resolve() == path.resolve() or source.resolve() in path.resolve().parents for path in (build,directory,*evidence)):
            raise ValueError('Projection outputs must be outside frozen fixtures')
    inputs, anchors, overflow = cases()
    tiny, tiny_anchors, gating_overflow = gating.cases()
    tails, tail_anchors = gating.tail_cases()
    directory.mkdir(parents=True)
    jobs, identities = [], {}
    for label, values, option, validator in [
        ('projection_frozen_manual',inputs,'--projection-input',lambda e:validate(e,inputs,anchors)),
        ('projection_overflow',[overflow],'--projection-input',overflow_rejected),
        ('gating_regression',tiny,'--gating-input',lambda e:gating.validate(e,tiny,tiny_anchors)),
        ('gating_tails_regression',tails,'--gating-input',lambda e:gating.validate_tails(e,tails,tail_anchors)),
        ('gating_overflow_regression',[gating_overflow],'--gating-input',gating.overflow_rejected),
    ]:
        path = directory/f'{label}.json'
        with path.open('x') as stream:
            stream.write(json.dumps({'format':'toklet-metal-projection-input-v1' if option=='--projection-input' else 'toklet-metal-gating-input-v1','cases':values},separators=(',',':'))+'\n')
        identities[str(path.resolve())] = rms.probe.common.digest(path)
        jobs.append((label,[option,str(path.resolve())],validator))
    drivers = {str(Path(module.__file__).resolve()):rms.probe.common.digest(Path(module.__file__)) for module in (gating,rms,gating.silu,gating.residual,gating.residual.embedding)}
    drivers[str(Path(__file__).resolve())] = rms.probe.common.digest(Path(__file__))
    previous = jobs[-1][2]
    def final(event):
        result = previous(event)
        if cases() != (inputs,anchors,overflow):
            raise ValueError('Frozen projection references changed')
        for name,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(name)) != sha:
                raise ValueError('Projection input/driver identity changed')
        return result
    jobs[-1] = (jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context={
        'qualification_driver_sha256':drivers,'input_sha256':identities,
        'fp16_manifest_sha256':rms.FP16_MANIFEST_SHA,'source_fp32_manifest_sha256':rms.FP32_MANIFEST_SHA,
        'scope':'10 frozen tiny query projections and3 independent manual bias/stride/cancellation cases. Resident Half operands, GPU bias -> MPS FP32 result -> one Half store in one command. Frozen source gates and exact manual outputs. Same-binary add/gating/tails/overflow regressions. No Qwen projection, full decoder, performance or M4 completion.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
    args=p.parse_args();run(args.output,args.build_dir,args.input_dir)

if __name__=='__main__':
    main()
